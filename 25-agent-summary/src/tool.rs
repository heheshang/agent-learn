//! **工具层** —— `Tool` trait、注册表、真实工具实现。
//!
//! # 这个模块补上了 25 之前的**最大缺口**
//!
//! 25 原本有 `tooling.rs`（Schema 校验 + SQL 护栏 + 错误分类），
//! 但**没有 `Tool` trait、没有注册表、没有工具真的被执行** ——
//! `check_tool_call` 只是把 SQL 拼出来扔给护栏检查，
//! **从不执行**，所以「护栏保护了什么」没有可执行的证据。
//!
//! # 三层职责
//!
//! | 层 | 位置 | 职责 |
//! |---|---|---|
//! | **定义** | [`ToolSpec`] | 名字、描述、Schema |
//! | **执行** | [`Tool`] trait | 一个具体的可调用工具 |
//! | **派发** | [`ToolRegistry`] | 找工具 → 填默认值 → 校验 → 执行 → 失败转观察 |
//!
//! # 三条硬约定
//!
//! ## ① 工具失败**不是**请求失败
//!
//! 工具报错时把错误信息当成一条 `Observation` 喂回模型，
//! 模型读到「参数 `scope` 必须是字符串」就能自己纠正。
//!
//! ## ② 参数错了**别发请求**
//!
//! Schema 本地校验，错误信息必须**可执行**
//! （指出哪个参数、期望什么、实际是什么）。
//!
//! ## ③ 只有 Transient 该重试
//!
//! 参数填错时重试是纯浪费 —— 参数不会因为重试变对。

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::tooling::{ErrorCode, ToolError, check_sql, fill_defaults, validate};

/// 工具执行返回给调用方的 future。
///
/// ## ⚠️ 为什么是 `Pin<Box<dyn Future>>`
///
/// `async fn` 在 trait 里**不是 dyn 兼容**的，
/// 而工具注册表需要 `Box<dyn Tool>`。
///
/// 这和 07 的 `LlmFuture`、14 的 `ToolFuture`、16 的 `ToolFuture` 同理。
pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolOutcome, ToolError>> + Send + 'a>>;

/// 工具执行的结果。
#[derive(Debug, Clone)]
pub struct ToolOutcome {
    /// 给模型看的内容（会成为 `tool` 消息的正文）。
    pub observation: String,

    /// 是否成功。
    pub ok: bool,

    /// 工具名。
    pub tool: String,

    /// 结构化结果（可选）。
    pub data: Option<Value>,
}

impl ToolOutcome {
    /// 成功（纯文本）。
    pub fn ok(tool: impl Into<String>, observation: impl Into<String>) -> Self {
        Self {
            observation: observation.into(),
            ok: true,
            tool: tool.into(),
            data: None,
        }
    }

    /// 成功（带结构化数据）。
    pub fn ok_with_data(
        tool: impl Into<String>,
        observation: impl Into<String>,
        data: Value,
    ) -> Self {
        Self {
            observation: observation.into(),
            ok: true,
            tool: tool.into(),
            data: Some(data),
        }
    }

    /// 失败（**不是错误** —— 错误文本会喂回模型让它自我纠正）。
    pub fn failed(tool: impl Into<String>, observation: impl Into<String>) -> Self {
        Self {
            observation: observation.into(),
            ok: false,
            tool: tool.into(),
            data: None,
        }
    }
}

/// 工具运行时上下文。
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    /// 用户标识（可用于审计 / 限流）。
    pub user_id: Option<String>,

    /// 时间窗口（工具可能需要按窗口聚合）。
    pub window: Option<crate::model::Window>,
}

impl ToolContext {
    /// 空上下文。
    pub fn empty() -> Self {
        Self::default()
    }

    /// 带窗口的上下文。
    pub fn with_window(window: crate::model::Window) -> Self {
        Self {
            user_id: None,
            window: Some(window),
        }
    }
}

/// 工具定义（发给模型的那一份）。
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    /// 工具名（`snake_case`）。
    pub name: String,

    /// 工具描述。
    ///
    /// ⚠️ **描述是模型行为的一部分**，不是文档。
    /// OpenAI「最佳实践」第一条：
    /// 「明确描述函数的目的和每个参数（以及它的格式），以及输出代表什么」。
    pub description: String,

    /// 参数的 JSON Schema。
    pub schema: Value,

    /// 是否开启 strict 模式。
    ///
    /// ## ⚠️ 为什么默认**不开**
    ///
    /// DeepSeek **接受** `strict` 但**不强制**（实测）。
    /// 而 strict 要求「所有字段都进 `required`」——
    /// 对「参数多且有默认值」的工具反而是负担。
    ///
    /// 所以本地 Schema 校验是**必需品**，strict 只能讨好 OpenAI。
    pub strict: bool,
}

impl ToolSpec {
    /// 构造一个工具定义（**不开** strict）。
    pub fn new(name: impl Into<String>, description: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            strict: false,
        }
    }

    /// 构造一个工具定义并**开启** strict 模式。
    pub fn strict(name: impl Into<String>, description: impl Into<String>, schema: Value) -> Self {
        Self {
            strict: true,
            ..Self::new(name, description, schema)
        }
    }

    /// 切换 strict 模式。
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// 翻译成 `genai::chat::Tool`。
    ///
    /// ## ⚠️★ Schema 的 sanitization **完全交给 genai 适配器**
    ///
    /// 本项目**不自己**改写 Schema —— 改写发生在适配器发请求之前：
    ///
    /// - `OpenAiStrictTool`：所有 `properties` 都进 `required`、
    ///   递归补 `additionalProperties: false`、按需内联 `$ref`
    /// - `AnthropicStructured`：只补 `additionalProperties: false`
    ///
    /// ## ★ 为什么**不自己**改写
    ///
    /// **自己改写 = 与 provider 的校验规则漂移。**
    ///
    /// ⚠️ 本项目**曾经**手写过一版，只改了**根层**的
    /// `required` / `additionalProperties`，**嵌套对象完全没递归** ——
    /// 而 OpenAI strict 方言要求**每一层**都满足。
    /// 症状是 provider 报一个看不懂的 400。
    ///
    /// ⚠️ 而且**改写两遍**更糟：第一遍按 OpenAI 方言，
    /// 第二遍适配器又按 provider 方言 ——
    /// 对 OpenAI 幂等，对 Anthropic 就是**方言冲突**。
    pub fn to_genai_tool(&self) -> genai::chat::Tool {
        // ⚠️ **原样透传** —— 消毒由适配器按 provider 方言做。
        genai::chat::Tool::new(self.name.clone())
            .with_description(self.description.clone())
            .with_schema(self.schema.clone())
            .with_strict(self.strict)
    }
}

/// 一个「可被模型调用」的函数。
///
/// 三个必须做对的地方：
///
/// 1. **描述** —— 说清楚「什么时候该调我」
/// 2. **Schema** —— 参数名用模型的词表
/// 3. **错误处理** —— 失败时返回 [`ToolError`]
pub trait Tool: Send + Sync {
    /// 工具定义。
    fn spec(&self) -> ToolSpec;

    /// 执行工具。
    ///
    /// `Err` 表示「这次调用失败了，原因在这里」——
    /// 它**不会**中断 Agent 循环，注册表会把它变成一条 `tool` 消息喂回模型。
    fn call<'a>(&'a self, args: &'a Value, ctx: &'a ToolContext) -> ToolFuture<'a>;
}

/// 工具注册表。
///
/// 负责：按名字分发 → 填默认值 → 校验 → 执行 → 失败转观察。
#[derive(Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field(
                "tools",
                &self.tools.iter().map(|t| t.spec().name).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ToolRegistry {
    /// 创建注册表。
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self { tools }
    }

    /// 工具数量。
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// 所有工具名。
    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.spec().name).collect()
    }

    /// 所有工具定义。
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|t| t.spec()).collect()
    }

    /// 按名字取工具。
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.spec().name == name)
    }

    /// 按名字白名单筛选（**动态工具选择**）。
    ///
    /// OpenAI 最佳实践：「**保持初始可用函数的数量足够少**」，
    /// 建议一开始少于 20 个函数。
    pub fn allow_names(&self, names: &[&str]) -> Self {
        Self {
            tools: self
                .tools
                .iter()
                .filter(|t| names.contains(&t.spec().name.as_str()))
                .cloned()
                .collect(),
        }
    }

    /// 调用工具 —— **失败转观察**。
    ///
    /// 这是 Agent 循环**唯一**应该调用的入口。
    ///
    /// 四件事：
    ///
    /// 1. 找工具（找不到 → 生成「可用工具列表」的错误观察）
    /// 2. 填默认值
    /// 3. 校验参数（不通过 → 生成可执行的错误观察）
    /// 4. 执行（失败 → 生成错误观察）
    pub async fn call(&self, name: &str, args: &Value, ctx: &ToolContext) -> ToolOutcome {
        match self.call_inner(name, args, ctx).await {
            Ok(outcome) => outcome,
            Err(err) => {
                agent_logging::warn!(
                    tool = %name,
                    code = %err.code.id(),
                    kind = %err.kind.name(),
                    "工具调用失败（错误将作为 tool 消息喂回模型）"
                );
                ToolOutcome::failed(name, err.render())
            }
        }
    }

    /// 调用工具 —— 错误**向上传播**。
    ///
    /// 给「用户直接调工具」的 HTTP 端点用：
    /// 那时用户扮演的是「模型」的角色，他应该看到真正的错误。
    pub async fn call_strict(
        &self,
        name: &str,
        args: &Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        self.call_inner(name, args, ctx).await
    }

    /// 内部实现：查找 → 填默认值 → 校验 → 执行。
    async fn call_inner(
        &self,
        name: &str,
        args: &Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let Some(tool) = self.get(name) else {
            agent_logging::warn!(tool = %name, "未知工具");
            return Err(ToolError::new(
                ErrorCode::UnknownTool,
                name,
                format!("没有名为 {name} 的工具"),
            )
            .with_hint(format!("可用工具：{}", self.names().join(", "))));
        };

        let spec = tool.spec();

        // ── 参数准备 ────────────────────────────────────────────
        let mut object = match args {
            Value::Object(map) => map.clone(),
            Value::Null => serde_json::Map::new(),
            other => {
                return Err(ToolError::new(
                    ErrorCode::InvalidArguments,
                    name,
                    format!("参数必须是 JSON 对象，实际是 {}", json_type_name(other)),
                )
                .with_hint("把所有参数放进一个 JSON 对象里。"));
            }
        };

        // 填默认值 → 再校验（顺序不能反）。
        fill_defaults(&spec.schema, &mut object);

        let validation = validate(&spec.schema, &Value::Object(object.clone()));
        if !validation.is_ok() {
            agent_logging::warn!(
                tool = %name,
                issues = validation.issues.len(),
                "参数校验不通过 —— 不执行工具"
            );
            return Err(
                ToolError::new(ErrorCode::InvalidArguments, name, validation.message())
                    .with_hint("按上面的提示修正参数后重新调用。"),
            );
        }

        // ── 执行 ────────────────────────────────────────────────
        agent_logging::info!(tool = %name, "开始执行工具");
        tool.call(&Value::Object(object), ctx).await
    }
}

/// JSON 值的类型名。
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ─────────────────────────────────────────────────────────────────────
// 真实工具实现
// ─────────────────────────────────────────────────────────────────────

/// 「按模块统计」工具 —— ★ **真的去聚合**。
///
/// ## ⚠️ 这是 25 之前缺的那一环
///
/// 之前的 `check_tool_call` 把 SQL 拼出来扔给护栏检查，**从不执行**。
/// 现在这个工具**真的**跑一遍聚合，并把结果给模型。
///
/// ⚠️ **它必须真的查数据** —— 否则模型拿到的数字是编的，
/// 这正是 15-agent-sql 实测踩过的最隐蔽的坑
/// （「答案看起来很专业但数字是编的」）。
pub struct ModuleStatsTool {
    /// 活动记录。
    activities: Arc<Vec<crate::model::Activity>>,
}

impl std::fmt::Debug for ModuleStatsTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleStatsTool")
            .field("activities", &self.activities.len())
            .finish()
    }
}

impl ModuleStatsTool {
    /// 创建工具。
    pub fn new(activities: Arc<Vec<crate::model::Activity>>) -> Self {
        Self { activities }
    }
}

impl Tool for ModuleStatsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "module_stats",
            "统计某个模块的活动记录数量、类型分布与风险项。回答「上周 cache 模块做了哪些事、有没有风险」这类问题时使用。",
            json!({
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "模块名，例如 cache、rag、agent、async、batch、vllm。",
                        "minLength": 1
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "返回的亮点条数上限。"
                    },
                    "include_risks": {
                        "type": "boolean",
                        "default": true,
                        "description": "是否包含风险项。"
                    }
                },
                "required": ["scope"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let scope = args["scope"].as_str().unwrap_or_default();
            let limit = args["limit"].as_u64().unwrap_or(10).max(1) as usize;
            let include_risks = args["include_risks"].as_bool().unwrap_or(true);

            // ⚠️ **没有窗口时也要能算** —— 用全量数据。
            //
            // ⚠️ `ctx.window` 是 `Copy` —— 先取出原值，
            //    后面的观察文本还要用（要说清时间范围）。
            let requested_window = ctx.window;
            let window = requested_window.unwrap_or_else(|| {
                let min = self.activities.iter().map(|a| a.ts).min().unwrap_or(0);
                let max = self.activities.iter().map(|a| a.ts).max().unwrap_or(0);
                crate::model::Window::new(min, max + 1)
            });

            let matched: Vec<&crate::model::Activity> = self
                .activities
                .iter()
                .filter(|a| a.scope == scope && window.contains(a.ts))
                .collect();

            if matched.is_empty() {
                // ⚠️ 「没有数据」是**正常结果**，不是错误 ——
                //    但必须**如实告诉模型**，否则它会编。
                let available: Vec<&str> = {
                    let mut scopes: Vec<&str> =
                        self.activities.iter().map(|a| a.scope.as_str()).collect();
                    scopes.sort_unstable();
                    scopes.dedup();
                    scopes
                };
                return Ok(ToolOutcome::ok_with_data(
                    "module_stats",
                    format!(
                        "模块 {scope} 在该时间窗口内没有活动记录。可用模块：{}",
                        available.join(", ")
                    ),
                    json!({ "scope": scope, "total": 0, "hits": [] }),
                ));
            }

            // 统计。
            let mut by_kind: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            let mut risks: Vec<&crate::model::Activity> = Vec::new();
            for a in &matched {
                *by_kind.entry(a.kind.as_str()).or_insert(0) += 1;
                if a.risk {
                    risks.push(a);
                }
            }

            // 亮点：权重最高的若干条。
            let mut highlights: Vec<&crate::model::Activity> = matched.clone();
            highlights.sort_by(|a, b| b.weight.cmp(&a.weight).then(a.id.cmp(&b.id)));
            highlights.truncate(limit);

            let kind_summary = by_kind
                .iter()
                .map(|(k, v)| format!("{k}×{v}"))
                .collect::<Vec<_>>()
                .join("、");

            // ★ 观察里必须说清**时间范围** ——
            //    模型问「上周做了什么」时，如果不知道窗口是哪天到哪天，
            //    它只能猜（实测：模型因此说「module_stats 不区分时间范围」）。
            let window_note = match requested_window {
                Some(w) => format!("（时间范围：{} ~ {}）", w.start, w.end),
                None => "（时间范围：全部）".to_string(),
            };

            let mut lines = vec![format!(
                "模块 {scope}：共 {} 条活动（{kind_summary}）{window_note}",
                matched.len()
            )];

            for h in &highlights {
                lines.push(format!("- [{}] {}（权重 {}）", h.kind, h.title, h.weight));
            }

            if include_risks && !risks.is_empty() {
                lines.push(format!("风险项 {} 条：", risks.len()));
                for r in risks.iter().take(limit) {
                    lines.push(format!("- 【风险】{}（{}）", r.title, r.kind));
                }
            }

            // ★ 结构化数据放在**最后一行**（`@@DATA:{...}` 约定）——
            //   这样收集证据时可以解回来，**发给模型前剥掉**
            //   （那坨 JSON 占 token 还干扰模型）。
            let data = json!({
                "scope": scope,
                "total": matched.len(),
                "by_kind": by_kind,
                "risks": risks.len(),
                "highlights": highlights.iter().map(|h| json!({
                    "id": h.id,
                    "kind": h.kind,
                    "title": h.title,
                    "author": h.author,
                    "weight": h.weight,
                    "risk": h.risk,
                })).collect::<Vec<_>>(),
            });

            let mut observation = lines.join("\n");
            observation.push_str(&format!("\n@@DATA:{}", data));

            Ok(ToolOutcome::ok_with_data("module_stats", observation, data))
        })
    }
}

/// 「检索活动记录」工具 —— 走混合检索。
pub struct SearchActivitiesTool {
    /// 检索器。
    retriever: Arc<crate::retrieve::HybridRetriever>,
}

impl std::fmt::Debug for SearchActivitiesTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchActivitiesTool").finish()
    }
}

impl SearchActivitiesTool {
    /// 创建工具。
    pub fn new(retriever: Arc<crate::retrieve::HybridRetriever>) -> Self {
        Self { retriever }
    }
}

impl Tool for SearchActivitiesTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "search_activities",
            "在活动记录里做关键词与语义混合检索。回答「关于缓存的那些工作是什么」这类**开放式**问题时使用；已知模块名时优先用 module_stats。",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索词。",
                        "minLength": 1
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "返回条数。"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let query = args["query"].as_str().unwrap_or_default();
            let top_k = args["top_k"].as_u64().unwrap_or(5).max(1) as usize;

            let hits = self.retriever.fuse(query, top_k);
            if hits.is_empty() {
                return Ok(ToolOutcome::ok_with_data(
                    "search_activities",
                    format!("没有检索到与「{query}」相关的活动记录。"),
                    json!({ "query": query, "hits": [] }),
                ));
            }

            let mut lines = vec![format!("检索「{query}」命中 {} 条：", hits.len())];
            for h in &hits {
                lines.push(format!("- [{}] {}", h.source, h.text.replace('\n', " ")));
            }

            let data = json!({
                "query": query,
                "hits": hits.iter().map(|h| json!({
                    "id": h.id,
                    "source": h.source,
                    "rank": h.rank,
                    "route": h.route,
                    "text": h.text,
                })).collect::<Vec<_>>(),
            });

            let mut observation = lines.join("\n");
            observation.push_str(&format!("\n@@DATA:{}", data));

            Ok(ToolOutcome::ok_with_data(
                "search_activities",
                observation,
                data,
            ))
        })
    }
}

/// 「算术」工具 —— ★ 统计数字必须**来自计算**，不能来自模型的心算。
pub struct CalculateTool;

impl std::fmt::Debug for CalculateTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalculateTool").finish()
    }
}

impl Tool for CalculateTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "calculate",
            "计算一个算术表达式（支持 + - * / 与括号）。**任何从统计里推导出的数字都必须用这个工具算**，不要心算。",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "算术表达式，例如 (12 + 8) / 4。",
                        "minLength": 1
                    }
                },
                "required": ["expression"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let expression = args["expression"].as_str().unwrap_or_default();

            match crate::agent::calculate(expression) {
                Ok(value) => Ok(ToolOutcome::ok_with_data(
                    "calculate",
                    format!("{expression} = {value}"),
                    json!({ "expression": expression, "value": value }),
                )),
                Err(e) => Ok(ToolOutcome::failed(
                    "calculate",
                    ToolError::new(ErrorCode::InvalidArguments, "calculate", e)
                        .with_hint("检查括号与运算符，只用 + - * / 和数字。")
                        .render(),
                )),
            }
        })
    }
}

/// 「查询统计」工具 —— 走 **SQL 护栏** 的那条路。
///
/// ## ⚠️ 为什么这个工具值得存在
///
/// 25 原本有 SQL 护栏，但**没有任何东西真的用 SQL**。
/// 这个工具让护栏有真实的守门对象：
/// 模型想查库 → 生成 SQL → 护栏检查 → 通过才执行。
///
/// ## ⚠️ 只读
///
/// 护栏只放行 `SELECT` / `WITH` / `VALUES` / `EXPLAIN`，
/// 写操作一律被拦下（`ToolError` 的 `RuleViolation`）。
pub struct QueryTool {
    /// 活动记录（只读）。
    activities: Arc<Vec<crate::model::Activity>>,
}

impl std::fmt::Debug for QueryTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryTool")
            .field("activities", &self.activities.len())
            .finish()
    }
}

impl QueryTool {
    /// 创建工具。
    pub fn new(activities: Arc<Vec<crate::model::Activity>>) -> Self {
        Self { activities }
    }
}

impl Tool for QueryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "query_stats",
            "对活动记录执行**只读** SQL 查询（只支持 SELECT / WITH）。用于按条件筛选和统计，例如统计某个作者在某段时间的任务数。",
            json!({
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "只读 SQL。表名固定为 activity，列为 id、ts、kind、author、scope、title、weight、risk。",
                        "minLength": 1
                    }
                },
                "required": ["sql"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let sql = args["sql"].as_str().unwrap_or_default();

            // ★ 护栏先行 —— **参数对了不代表 SQL 安全**。
            //
            // ⚠️ 这里必须**真的调用** `check_sql`，
            //    不能只「拼出来检查一下就扔掉」。
            //    护栏的价值在于**拦得住**，不在于「检查过」。
            if let Err(guard) = check_sql(sql) {
                agent_logging::warn!(
                    sql = %sql,
                    reason = %guard,
                    "SQL 被护栏拦下"
                );
                return Ok(ToolOutcome::failed(
                    "query_stats",
                    ToolError::new(
                        ErrorCode::RuleViolation,
                        "query_stats",
                        format!("SQL 未通过安全检查: {guard}"),
                    )
                    .with_hint("改成只读的 SELECT 查询，不要包含写操作或多条语句。")
                    .render(),
                ));
            }

            // ⚠️ 本项目不引入 SQLite —— 查询用**内存过滤**实现。
            //    护栏保证「不会写、不会多语句」，剩下的语义由下面的过滤实现。
            //
            // ★ 窗口要生效 —— 与 `module_stats` **同一口径**。
            let window = ctx.window;
            let filtered = match self.filter_by_scope(sql, window) {
                Ok(rows) => rows,
                Err(reason) => {
                    // ⚠️ **明确报错，不假装执行** ——
                    //    静默忽略会让模型拿着错的数据继续推理。
                    agent_logging::warn!(
                        sql = %sql,
                        reason = %reason,
                        "查询包含不支持的构造 —— 如实报错"
                    );
                    return Ok(ToolOutcome::failed(
                        "query_stats",
                        ToolError::new(ErrorCode::InvalidArguments, "query_stats", reason)
                            .with_hint(
                                "本工具只支持 `SELECT * FROM activity WHERE scope = '...' \
                                 或 `WHERE author = '...'`。",
                            )
                            .render(),
                    ));
                }
            };

            // ★ 观察里必须说清**时间范围** ——
            //    否则模型只能猜「上周」到底是哪天到哪天。
            let window_note = match window {
                Some(w) => format!("（时间范围：{} ~ {}）", w.start, w.end),
                None => "（时间范围：全部）".to_string(),
            };

            let mut lines = vec![format!("查询命中 {} 条记录{window_note}。", filtered.len())];
            for a in filtered.iter().take(20) {
                lines.push(format!("- [{}] {}（{}）", a.scope, a.title, a.kind));
            }
            if filtered.len() > 20 {
                lines.push(format!("... 还有 {} 条未显示。", filtered.len() - 20));
            }

            let data = json!({
                "sql": sql,
                "total": filtered.len(),
                "window": window.map(|w| json!({"start": w.start, "end": w.end})),
                "rows": filtered.iter().take(20).map(|a| json!({
                    "id": a.id,
                    "ts": a.ts,
                    "kind": a.kind,
                    "author": a.author,
                    "scope": a.scope,
                    "title": a.title,
                    "weight": a.weight,
                    "risk": a.risk,
                })).collect::<Vec<_>>(),
            });

            let mut observation = lines.join("\n");
            observation.push_str(&format!("\n@@DATA:{}", data));

            Ok(ToolOutcome::ok_with_data("query_stats", observation, data))
        })
    }
}

impl QueryTool {
    /// ★ 极简 SQL 解释器 —— 只支持 `SELECT * FROM activity WHERE scope = '...' [AND author = '...']`。
    ///
    /// ## ⚠️ 为什么不用真的 SQLite
    ///
    /// 25 的定位是**压测台** —— 引入 SQLite 会带来一个
    /// 「二进制依赖 + 换机器要重编译」的成本，
    /// 而 25 的核心产出（归因方法论）**不需要真正的 SQL 引擎**。
    ///
    /// 但护栏必须**真的守门** —— 所以这里保留了完整的护栏检查，
    /// 只是把「执行」简化成了过滤。
    ///
    /// ## ⚠️⚠️ **不支持的构造必须报错，不能假装执行**
    ///
    /// 这是实测踩出来的**最严重的一个 bug**。
    ///
    /// 第一版只提取 `scope = '...'` 和 `author = '...'`，
    /// 遇到 `GROUP BY` / `COUNT(*)` 就**默默忽略**，
    /// 返回一份原始行列表 —— **看起来完全成功**。
    ///
    /// 真实模型（DeepSeek）的实际行为：
    ///
    /// ```text
    /// 第 2 步：query_stats → 「查询命中 834 条记录。」
    /// 第 3 步：query_stats → 「查询命中 834 条记录。」
    /// 第 4 步：query_stats → 「查询命中 834 条记录。」
    /// ```
    ///
    /// 模型连试三种聚合 SQL（`MIN/MAX(ts)`、`GROUP BY kind`、`COUNT(*)`），
    /// 拿到的都是**同一份 834 行原始数据** ——
    /// 它最后在答案里写：
    ///
    /// > 「`query_stats` 工具没有按我的 SQL 执行」
    ///
    /// ⚠️ **模型比人更早察觉工具在骗它** —— 但代价是
    /// **3 次无效工具调用 + 一整轮拉锯战**。
    ///
    /// ## ★ 教训
    ///
    /// **「假装成功」比「明确报错」危险得多** ——
    /// 报错时模型会立刻换写法，
    /// 假装成功时模型会**拿着错的数据继续推理**，
    /// 而且**无法区分**「数据就是这样」和「工具没执行我的查询」。
    ///
    /// 所以：不支持的构造 → **明确报错 + 告诉它支持的语法**。
    fn filter_by_scope(
        &self,
        sql: &str,
        window: Option<crate::model::Window>,
    ) -> Result<Vec<&crate::model::Activity>, String> {
        let lowered = sql.to_lowercase();

        // ⚠️ **先检查有没有不支持的构造** ——
        //    顺序不能反：先提取条件再检查的话，
        //    `SELECT scope, COUNT(*) FROM activity GROUP BY scope`
        //    会因为「有 scope 关键字」而被误判成支持的查询。
        if let Some(unsupported) = find_unsupported_construct(&lowered) {
            return Err(unsupported);
        }

        let scope = extract_bare_condition(&lowered, "scope");
        let author = extract_bare_condition(&lowered, "author");
        let risk = extract_bare_condition(&lowered, "risk");

        // ★★ **每一个 WHERE 条件都必须被识别** ★★
        //
        // ⚠️ 这是实测踩出来的**同类 bug 的第二个实例** ——
        //
        // 第一版只提取 `scope` 和 `author`，模型传了 `risk = 1` 时
        // **条件被静默忽略**，返回全部 239 条 ——
        // 模型在答案里写道：
        //
        // > 「我传 `risk = 1` 的查询返回了全部 239 条，
        // > 说明该字段的过滤没有生效」
        //
        // 它察觉到了，但代价是**多花工具调用去调和**。
        //
        // ⚠️ **这和「静默忽略 GROUP BY」是同一个病的两种症状** ——
        // **静默忽略任何条件都会让模型拿到与查询无关的数据却以为那就是结果**。
        //
        // ## ★ 为什么用「列白名单」而不是「数条件个数」
        //
        // 第一版数 `列 = 值` 的个数，但模型还会写 `ts > 1700000000` ——
        // **比较运算符根本不是 `=`，压根没被数到**，
        // 于是「0 个条件 == 0 个已识别」直接放行，
        // 返回全量数据。**这是同一个漏洞的第三处**。
        //
        // 改成「扫出 WHERE 之后的所有列名，与白名单比对」——
        // 与用什么运算符**无关**，漏不掉。
        let mentioned = extract_where_columns(&lowered);
        let unsupported_columns: Vec<String> = mentioned
            .iter()
            .filter(|c| !SUPPORTED_COLUMNS.contains(&c.as_str()))
            .cloned()
            .collect();

        if !unsupported_columns.is_empty() {
            let mut hint = format!(
                "本工具只支持 {} 三列的过滤；当前查询用到了不支持的列：{}。",
                SUPPORTED_COLUMNS.join("、"),
                unsupported_columns.join("、")
            );
            hint.push_str(
                "这些条件被**拒绝执行**（而不是被忽略）——\
                 因为忽略它们会让你拿到与查询无关的数据却以为那就是结果。",
            );
            return Err(hint);
        }

        if scope.is_none() && author.is_none() && risk.is_none() {
            return Err(
                "本工具只支持 `WHERE scope = '...'`、`WHERE author = '...'` \
                 或 `WHERE risk = '1'` 形式的等值条件；\
                 当前查询没有可识别的条件。"
                    .to_string(),
            );
        }

        Ok(self
            .activities
            .iter()
            .filter(|a| {
                scope
                    .as_ref()
                    .is_none_or(|s| a.scope.eq_ignore_ascii_case(s))
                    && author
                        .as_ref()
                        .is_none_or(|s| a.author.eq_ignore_ascii_case(s))
                    && risk.as_ref().is_none_or(|r| {
                        // ⚠️ `risk` 列是布尔语义 ——
                        //    模型可能写 `risk = 1` / `risk = true` / `risk = '1'`。
                        let want = matches!(r.as_str(), "1" | "true" | "yes" | "t");
                        a.risk == want
                    })
                    // ★ **窗口必须生效** —— 与 `module_stats` 保持同一口径。
                    //
                    // ⚠️ 第一版这里**没有**窗口过滤，
                    //    导致同一份数据有两个口径：
                    //    `module_stats` 返回 239 条（窗口内），
                    //    `query_stats` 返回 834 条（全量）。
                    //    模型两边对比就发现「对不上」，白白多花工具调用去调和。
                    && window.is_none_or(|w| w.contains(a.ts))
            })
            .collect())
    }
}

/// ★ 本工具支持的列（**白名单**）。
///
/// ⚠️ 白名单而不是黑名单 ——
/// 黑名单会漏（模型能写出你想不到的条件形式）。
const SUPPORTED_COLUMNS: &[&str] = &["scope", "author", "risk"];

/// ★ 扫出 `WHERE` 之后提到的所有列名。
///
/// ## ⚠️ 为什么用「扫列名」而不是「数条件个数」
///
/// 因为模型会写**各种运算符**：`=`、`>`、`<`、`>=`、
/// `BETWEEN`、`LIKE`、`IN`…
///
/// 数「`列 = 值`」的模式时，`ts > 1700000000` 压根不匹配 ——
/// 于是「0 个条件 == 0 个已识别」直接放行，返回全量数据。
///
/// **这是同一个漏洞的第三处**（前两处：`GROUP BY` 静默忽略、
/// `risk = 1` 静默忽略）。
///
/// 扫列名与运算符**无关**，漏不掉。
fn extract_where_columns(lowered_sql: &str) -> Vec<String> {
    let Some(idx) = lowered_sql.find(" where ") else {
        return Vec::new();
    };

    let rest = &lowered_sql[idx + " where ".len()..];

    // 表里的所有列名（用来在 WHERE 之后找它们）。
    const ALL_COLUMNS: &[&str] = &[
        "scope", "author", "risk", "ts", "id", "kind", "title", "weight",
    ];

    let mut found: Vec<String> = Vec::new();
    for column in ALL_COLUMNS {
        // 出现在 WHERE 之后就算提到了。
        // ⚠️ 用「词边界」判断 —— `ts` 出现在 `scope` 里不算。
        let mut search_from = 0usize;
        while let Some(pos) = rest[search_from..].find(column) {
            let abs = search_from + pos;
            let before = rest[..abs].chars().next_back();
            let after = rest[abs + column.len()..].chars().next();

            let is_word = before.is_none_or(|c| !c.is_alphanumeric() && c != '_')
                && after.is_none_or(|c| !c.is_alphanumeric() && c != '_');

            if is_word {
                found.push((*column).to_string());
                break;
            }

            search_from = abs + column.len();
        }
    }

    found
}

/// ★ 数一数 SQL 的 `WHERE` 里有几个条件。
///
/// ## ⚠️ 判据
///
/// 数 `WHERE` 之后出现的**列名 = 值** 的个数。
/// 找不到 `WHERE` 就是 0 个条件。
///
/// ## ⚠️⚠️ **这个函数是「反面教材」，只保留给测试**
///
/// 它**会漏** —— 模型写 `ts > 1700000000` 时压根不匹配，
/// 于是「0 个条件 == 0 个已识别」直接放行，返回全量数据。
///
/// 真正的判定用 [`extract_where_columns`]（列白名单），
/// 因为它与用什么运算符**无关**，漏不掉。
///
/// ⚠️ 保留它是为了让这个教训**可执行验证** ——
/// 回归测试 `counting_where_conditions_misses_comparison_operators`
/// 明确锁住「它会漏」这个事实，防止有人后来又改回用它。
#[cfg(test)]
fn count_where_conditions(lowered_sql: &str) -> usize {
    let Some(idx) = lowered_sql.find(" where ") else {
        return 0;
    };

    let rest = &lowered_sql[idx + " where ".len()..];

    // 数 `列 = 值` 的模式：`标识符` 紧跟 `=`。
    let mut count = 0usize;
    let bytes: Vec<char> = rest.chars().collect();
    let mut i = 0usize;

    while i < bytes.len() {
        // 找标识符的起点。
        if !bytes[i].is_ascii_alphabetic() && bytes[i] != '_' {
            i += 1;
            continue;
        }

        // 扫标识符。
        let start = i;
        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == '_') {
            i += 1;
        }

        // 跳过空白。
        let mut j = i;
        while j < bytes.len() && bytes[j].is_whitespace() {
            j += 1;
        }

        // 后面是 `=` 才算一个条件。
        if j < bytes.len() && bytes[j] == '=' {
            count += 1;
            i = j;
        } else {
            i = start + 1;
        }
    }

    count
}

/// ★ 找出 SQL 里**不支持**的构造。
///
/// ## ⚠️ 为什么要显式列出来
///
/// 因为「静默忽略」会让模型拿到**与查询无关的数据**却以为那就是结果。
/// 明确报错 + 告诉它支持的语法，模型一步就能改对。
fn find_unsupported_construct(lowered_sql: &str) -> Option<String> {
    // ⚠️ 聚合函数与分组 —— 本工具只做**过滤**，不做聚合。
    //
    //    聚合应该走 `module_stats`（它真的在聚合）。
    const UNSUPPORTED: &[(&str, &str)] = &[
        (
            "group by",
            "聚合与分组请改用 module_stats 工具（它真的在聚合）",
        ),
        (
            "count(",
            "聚合与分组请改用 module_stats 工具（它真的在聚合）",
        ),
        ("sum(", "聚合与分组请改用 module_stats 工具（它真的在聚合）"),
        ("avg(", "聚合与分组请改用 module_stats 工具（它真的在聚合）"),
        ("min(", "聚合与分组请改用 module_stats 工具（它真的在聚合）"),
        ("max(", "聚合与分组请改用 module_stats 工具（它真的在聚合）"),
        ("join ", "本工具不支持 JOIN，请只查 activity 一张表"),
        (
            "order by",
            "本工具不支持 ORDER BY，请改用 module_stats 的排序输出",
        ),
        ("having", "本工具不支持 HAVING，请改用 module_stats 工具"),
        ("insert", "只支持只读查询"),
        ("update", "只支持只读查询"),
        ("delete", "只支持只读查询"),
        ("drop", "只支持只读查询"),
        ("alter", "只支持只读查询"),
        ("create", "只支持只读查询"),
    ];

    for (needle, hint) in UNSUPPORTED {
        if lowered_sql.contains(needle) {
            return Some(format!("查询包含不支持的构造「{needle}」—— {hint}"));
        }
    }

    None
}

/// 从 SQL 里抽出一个 `列 = '值'` 形式的条件。
fn extract_string_condition(lowered_sql: &str, column: &str) -> Option<String> {
    let needle = format!("{column} = '");
    let start = lowered_sql.find(&needle)? + needle.len();
    let rest = &lowered_sql[start..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

/// ★ 从 SQL 里抽出一个 `列 = 值` 形式的条件（**值可以不带引号**）。
///
/// ## ⚠️ 为什么需要这个
///
/// 模型写 `risk = 1` 而不是 `risk = '1'` ——
/// `risk` 是布尔列，**裸数字**才是自然写法。
///
/// 第一版只认引号形式，于是 `risk = 1` 的条件
/// **既没被识别、也没被拒绝** —— 直接被忽略。
fn extract_bare_condition(lowered_sql: &str, column: &str) -> Option<String> {
    // 先试引号形式。
    if let Some(v) = extract_string_condition(lowered_sql, column) {
        return Some(v);
    }

    // 再试裸值形式：`列 = 词`（词不能是空格或逗号）。
    let needle = format!("{column} = ");
    let start = lowered_sql.find(&needle)? + needle.len();
    let rest = &lowered_sql[start..];

    let end = rest
        .find(|c: char| c.is_whitespace() || c == ',' || c == ')')
        .unwrap_or(rest.len());

    if end == 0 {
        return None;
    }

    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Activity;
    use crate::tooling::ErrorKind;

    fn activities(n: usize) -> Arc<Vec<Activity>> {
        Arc::new(
            (0..n)
                .map(|i| Activity {
                    id: format!("a-{i:04}"),
                    ts: 1_700_000_000 + (i as i64 % 7) * 86_400,
                    kind: ["feature", "fix", "incident", "docs"][i % 4].to_string(),
                    author: ["alice", "bob", "carol"][i % 3].to_string(),
                    scope: ["cache", "rag", "agent"][i % 3].to_string(),
                    title: format!("任务 {i}"),
                    weight: (i % 10) as i32,
                    risk: i % 20 == 0,
                })
                .collect(),
        )
    }

    fn registry() -> ToolRegistry {
        ToolRegistry::new(vec![
            Arc::new(ModuleStatsTool::new(activities(30))),
            Arc::new(CalculateTool),
        ])
    }

    #[tokio::test]
    async fn calls_a_real_tool_and_returns_data() {
        let reg = registry();
        let outcome = reg
            .call(
                "module_stats",
                &json!({"scope": "cache"}),
                &ToolContext::empty(),
            )
            .await;

        assert!(outcome.ok);
        assert!(outcome.observation.contains("模块 cache"));
        // ★ 结构化数据必须真的有 —— 「工具真的执行了」的证据。
        assert!(outcome.data.is_some());
        assert!(outcome.data.unwrap()["total"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn unknown_tool_becomes_an_actionable_observation() {
        let reg = registry();
        let outcome = reg.call("nope", &json!({}), &ToolContext::empty()).await;

        assert!(!outcome.ok);
        // ★ 错误里必须列出**可用工具** —— 模型据此纠正。
        assert!(outcome.observation.contains("module_stats"));
        assert!(outcome.observation.contains("E_UNKNOWN_TOOL"));
    }

    #[tokio::test]
    async fn invalid_arguments_become_an_observation_not_an_error() {
        let reg = registry();
        // 缺必填参数 `scope`。
        let outcome = reg
            .call("module_stats", &json!({}), &ToolContext::empty())
            .await;

        assert!(!outcome.ok);
        assert!(outcome.observation.contains("E_INVALID_ARGUMENTS"));
        assert!(outcome.observation.contains("scope"));
    }

    #[tokio::test]
    async fn defaults_are_filled_before_validation() {
        let reg = registry();
        // 只给 scope，limit / include_risks 走默认值。
        let outcome = reg
            .call(
                "module_stats",
                &json!({"scope": "rag"}),
                &ToolContext::empty(),
            )
            .await;
        assert!(outcome.ok);
    }

    #[tokio::test]
    async fn calculate_tool_computes() {
        let reg = registry();
        let outcome = reg
            .call(
                "calculate",
                &json!({"expression": "(12 + 8) / 4"}),
                &ToolContext::empty(),
            )
            .await;
        assert!(outcome.ok);
        assert!(outcome.observation.contains('5'));
    }

    #[tokio::test]
    async fn calculate_failure_is_an_observation() {
        let reg = registry();
        let outcome = reg
            .call(
                "calculate",
                &json!({"expression": "1 +"}),
                &ToolContext::empty(),
            )
            .await;
        assert!(!outcome.ok);
        assert!(outcome.observation.contains("E_INVALID_ARGUMENTS"));
    }

    #[tokio::test]
    async fn sql_guard_actually_blocks_writes() {
        let reg = ToolRegistry::new(vec![Arc::new(QueryTool::new(activities(30)))]);
        let outcome = reg
            .call(
                "query_stats",
                &json!({"sql": "DELETE FROM activity"}),
                &ToolContext::empty(),
            )
            .await;

        // ★ 护栏必须**真的拦住** —— 这就是它存在的意义。
        assert!(!outcome.ok);
        assert!(outcome.observation.contains("E_RULE_VIOLATION"));
    }

    #[tokio::test]
    async fn sql_guard_allows_readonly_queries() {
        let reg = ToolRegistry::new(vec![Arc::new(QueryTool::new(activities(30)))]);
        let outcome = reg
            .call(
                "query_stats",
                &json!({"sql": "SELECT * FROM activity WHERE scope = 'cache'"}),
                &ToolContext::empty(),
            )
            .await;

        assert!(outcome.ok);
        assert!(outcome.data.unwrap()["total"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn unrecognized_where_conditions_are_rejected_not_ignored() {
        // ⚠️ **实测踩出来的「静默忽略」的第二个实例** ——
        //
        // 第一版只提取 `scope` 和 `author`，
        // 模型传 `risk = 1` 时条件被**静默忽略**，返回全部 239 条。
        // 模型自己发现了：「我传 `risk = 1` 的查询返回了全部 239 条，
        // 说明该字段的过滤没有生效」。
        let reg = ToolRegistry::new(vec![Arc::new(QueryTool::new(activities(30)))]);

        let outcome = reg
            .call(
                "query_stats",
                &json!({"sql": "SELECT * FROM activity WHERE ts > 1700000000"}),
                &ToolContext::empty(),
            )
            .await;

        // ★ 未识别的条件必须**明确拒绝**。
        assert!(!outcome.ok);
        assert!(outcome.observation.contains("E_INVALID_ARGUMENTS"));
        assert!(outcome.observation.contains("ts"));
    }

    #[tokio::test]
    async fn risk_column_filter_actually_filters() {
        // ⚠️ 修好后 `risk = 1` 必须**真的过滤**，不能返回全部。
        let acts = activities(30);
        let reg = ToolRegistry::new(vec![Arc::new(QueryTool::new(acts))]);

        let outcome = reg
            .call(
                "query_stats",
                &json!({"sql": "SELECT * FROM activity WHERE risk = 1"}),
                &ToolContext::empty(),
            )
            .await;

        assert!(outcome.ok);
        let total = outcome.data.unwrap()["total"].as_u64().unwrap();

        // 30 条里 `i % 25 == 0` 的只有 2 条。
        assert_eq!(total, 2, "risk = 1 必须真的过滤，不能返回全部");
    }

    #[tokio::test]
    async fn counts_where_conditions() {
        assert_eq!(count_where_conditions("select * from activity"), 0);
        assert_eq!(
            count_where_conditions("select * from activity where scope = 'cache'"),
            1
        );
        assert_eq!(
            count_where_conditions(
                "select * from activity where scope = 'cache' and author = 'alice'"
            ),
            2
        );
        assert_eq!(
            count_where_conditions("select * from activity where risk = 1"),
            1
        );
    }

    #[test]
    fn counting_where_conditions_misses_comparison_operators() {
        // ⚠️ **这个反面教材必须锁住「它会漏」这个事实** ——
        //    防止有人后来又改回用它做判定。
        //
        // `ts > 1700000000` 用的是 `>` 不是 `=`，
        // 数条件个数**数不到** → 「0 == 0」→ 直接放行 → 返回全量数据。
        assert_eq!(
            count_where_conditions("select * from activity where ts > 1700000000"),
            0,
            "数条件个数会漏掉比较运算符 —— 所以判定必须用列白名单"
        );

        // ★ 列白名单**不会**漏。
        let mentioned = extract_where_columns("select * from activity where ts > 1700000000");
        assert!(mentioned.iter().any(|c| c == "ts"));
    }

    #[tokio::test]
    async fn extracts_bare_conditions() {
        let sql = "select * from activity where risk = 1";
        assert_eq!(
            extract_bare_condition(&sql.to_lowercase(), "risk"),
            Some("1".to_string())
        );

        let sql = "select * from activity where scope = 'cache'";
        assert_eq!(
            extract_bare_condition(&sql.to_lowercase(), "scope"),
            Some("cache".to_string())
        );
    }

    #[test]
    fn extracts_string_conditions() {
        let sql = "select * from activity where scope = 'cache' and author = 'alice'";
        assert_eq!(
            extract_string_condition(&sql.to_lowercase(), "scope"),
            Some("cache".to_string())
        );
        assert_eq!(
            extract_string_condition(&sql.to_lowercase(), "author"),
            Some("alice".to_string())
        );
        assert_eq!(extract_string_condition(&sql.to_lowercase(), "kind"), None);
    }

    // ── P0 回归测试：真实模型跑出来的三个 bug ────────────────

    #[tokio::test]
    async fn unsupported_sql_constructs_fail_instead_of_silently_ignoring() {
        // ⚠️ **实测踩出来的最严重的 bug** ——
        //
        // 第一版遇到 `GROUP BY` / `COUNT(*)` 就**默默忽略**，
        // 返回一份原始行列表 —— **看起来完全成功**。
        //
        // 真实模型（DeepSeek）因此连试三种聚合 SQL，拿到同一份数据，
        // 最后在答案里写「query_stats 工具没有按我的 SQL 执行」。
        let reg = ToolRegistry::new(vec![Arc::new(QueryTool::new(activities(30)))]);

        for sql in [
            "SELECT scope, COUNT(*) FROM activity GROUP BY scope",
            "SELECT COUNT(*) FROM activity",
            "SELECT MIN(ts), MAX(ts) FROM activity",
        ] {
            let outcome = reg
                .call("query_stats", &json!({"sql": sql}), &ToolContext::empty())
                .await;

            // ★ 必须**明确失败**，不能假装成功。
            assert!(!outcome.ok, "「{sql}」应该明确报错");
            assert!(
                outcome.observation.contains("E_INVALID_ARGUMENTS"),
                "「{sql}」的错误应该是参数类错误"
            );
            // ★ 错误信息必须**告诉它该怎么改**。
            assert!(
                outcome.observation.contains("module_stats"),
                "错误提示应该指向能做聚合的工具"
            );
        }
    }

    #[tokio::test]
    async fn query_stats_and_module_stats_use_the_same_window() {
        // ⚠️ **实测踩出来的第二个 bug** ——
        //
        // 第一版 `query_stats` **不按窗口过滤**，
        // 导致同一份数据两个口径：
        // `module_stats` 返回窗口内（239 条），`query_stats` 返回全量（834 条）。
        // 模型两边对比就发现「对不上」，白白多花工具调用去调和。
        let acts = activities(30);
        let reg = ToolRegistry::new(vec![
            Arc::new(ModuleStatsTool::new(acts.clone())),
            Arc::new(QueryTool::new(acts)),
        ]);

        // 只覆盖前 10 天。
        let window = crate::model::Window::new(1_700_000_000, 1_700_000_000 + 10 * 86_400);
        let ctx = ToolContext::with_window(window);

        let stats = reg
            .call("module_stats", &json!({"scope": "cache"}), &ctx)
            .await;
        let query = reg
            .call(
                "query_stats",
                &json!({"sql": "SELECT * FROM activity WHERE scope = 'cache'"}),
                &ctx,
            )
            .await;

        assert!(stats.ok);
        assert!(query.ok);

        let stats_total = stats.data.unwrap()["total"].as_u64().unwrap();
        let query_total = query.data.unwrap()["total"].as_u64().unwrap();

        // ★ 两个工具对同一份数据、同一窗口必须给出**同一个数**。
        assert_eq!(
            stats_total, query_total,
            "两个工具的时间窗口口径必须一致，否则模型会以为数据对不上"
        );
    }

    #[tokio::test]
    async fn observations_state_the_time_window() {
        // ⚠️ **实测踩出来的第三个 bug** ——
        //
        // 观察里没说清时间范围，模型只能猜「上周」是哪天到哪天，
        // 于是它说「module_stats 不区分时间范围」。
        let acts = activities(30);
        let reg = ToolRegistry::new(vec![
            Arc::new(ModuleStatsTool::new(acts.clone())),
            Arc::new(QueryTool::new(acts)),
        ]);

        let window = crate::model::Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400);
        let ctx = ToolContext::with_window(window);

        let stats = reg
            .call("module_stats", &json!({"scope": "cache"}), &ctx)
            .await;
        assert!(stats.observation.contains("时间范围"));

        let query = reg
            .call(
                "query_stats",
                &json!({"sql": "SELECT * FROM activity WHERE scope = 'cache'"}),
                &ctx,
            )
            .await;
        assert!(query.observation.contains("时间范围"));
    }

    #[test]
    fn allow_names_filters_the_registry() {
        let reg = registry();
        let filtered = reg.allow_names(&["calculate"]);
        assert_eq!(filtered.names(), vec!["calculate".to_string()]);
    }

    /// ★ strict 的 Schema 消毒**交给 genai 适配器** ——
    /// 本地**不预改写**。
    #[test]
    fn strict_tool_does_not_presanitize_the_schema() {
        let spec = ToolSpec::strict(
            "t",
            "d",
            json!({
                "type": "object",
                "properties": { "a": {"type": "string"}, "b": {"type": "string"} },
                "required": ["a"]
            }),
        );

        let tool = spec.to_genai_tool();

        // ⚠️★ `b` 仍在 `required` 之外 —— 说明本地**没有**预改写。
        //
        // ⚠️ 早期版本这里是「本地 sanitize 一遍」，断言 `["a", "b"]`。
        //    那样做等于**改写两遍**：第一遍按 OpenAI 方言，
        //    第二遍适配器又按 provider 方言 ——
        //    对 OpenAI 幂等，对 Anthropic 就是**方言冲突**。
        //
        //    现在本地只**透传**，消毒由适配器按 provider 方言做。
        assert_eq!(tool.schema.clone().unwrap()["required"], json!(["a"]));

        // strict 标记必须透传 —— 适配器据此决定要不要 sanitize。
        assert_eq!(tool.strict, Some(true));
    }

    /// ★ 非 strict 时 Schema **原样**透传。
    #[test]
    fn non_strict_tool_keeps_the_schema_untouched() {
        let spec = ToolSpec::new(
            "t",
            "d",
            json!({
                "type": "object",
                "properties": { "a": {"type": "string"}, "b": {"type": "string"} },
                "required": ["a"]
            }),
        );

        let tool = spec.to_genai_tool();

        // ⚠️ 非 strict **不能**改写 ——
        //    带 `default` 的可选参数会被强制要求填写。
        assert_eq!(tool.schema.unwrap()["required"], json!(["a"]));
        assert_eq!(tool.strict, Some(false));
    }

    /// ★ genai 的 `sanitize_json_schema` 会**递归**处理嵌套对象。
    ///
    /// 这是本项目原来手写版最大的缺陷：只改了根层。
    #[test]
    fn genai_sanitizer_recurses_into_nested_objects() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "nested": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x"]
                }
            },
            "required": ["a"]
        });

        let sanitized = genai::chat::sanitize_json_schema(
            &schema,
            genai::chat::JsonSchemaDialect::OpenAiStrictTool,
        );

        // 根层：全部字段必填 + 禁止额外字段。
        let root_required = sanitized["required"].as_array().unwrap();
        assert!(root_required.contains(&json!("a")));
        assert!(root_required.contains(&json!("nested")));
        assert_eq!(sanitized["additionalProperties"], json!(false));

        // ★ 嵌套层也要处理 —— 手写版漏了这一步。
        let nested_required = sanitized["properties"]["nested"]["required"]
            .as_array()
            .unwrap();
        assert!(nested_required.contains(&json!("x")));
        assert!(nested_required.contains(&json!("y")));
        assert_eq!(
            sanitized["properties"]["nested"]["additionalProperties"],
            json!(false)
        );
    }

    #[test]
    fn error_kinds_still_drive_retry_and_degrade() {
        let business = ToolError::new(ErrorCode::InvalidArguments, "t", "x");
        assert!(!business.retryable());
        assert!(business.degradable());

        let transient = ToolError::new(ErrorCode::Timeout, "t", "x");
        assert!(transient.retryable());
        assert!(transient.degradable());

        let fatal = ToolError::new(ErrorCode::Unauthorized, "t", "x");
        assert!(!fatal.retryable());
        // ⚠️ Fatal **不能**降级 —— 否则模型以为「工具坏了」转而编答案。
        assert!(!fatal.degradable());
        assert_eq!(fatal.kind, ErrorKind::Fatal);
    }
}
