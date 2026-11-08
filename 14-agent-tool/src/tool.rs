//! 工具抽象层 —— 对应 LangChain 的 `@tool` 装饰器与 `Tool` 类。
//!
//! ## LangChain 的工具是怎么定义的
//!
//! ```python
//! from langchain.tools import tool
//!
//! @tool
//! def get_weather(city: str) -> str:
//!     """Get weather for a city."""
//!     return f"It's always sunny in {city}!"
//! ```
//!
//! 三个要素：**名字**、**描述**、**参数 Schema**。
//!
//! 教程里那句「**工具的描述写得不好时，模型会反复调同一个工具**」
//! 直接说明了这三要素的性质 —— 它们不是文档，是**模型行为的一部分**。
//! 所以本项目把「描述」写进 [`ToolSpec`]，并在测试里断言它非空。
//!
//! ## 本项目与 LangChain 的对应关系
//!
//! | LangChain | 本项目 |
//! |---|---|
//! | `@tool` 装饰器 | 实现 [`Tool`] trait |
//! | Pydantic `args_schema` | [`crate::schema`] 的 JSON Schema 子集校验器 |
//! | `ToolRuntime`（state / context / store） | [`ToolContext`]（只读上下文） |
//! | `ToolMessage` | [`ToolOutcome`] 的 `observation` |
//! | `handle_tool_errors` 中间件 | [`ToolPolicy`] + [`ToolRegistry::call`] |
//! | `MCPAdapter` | [`crate::mcp`] 的 MCP 客户端 |
//! | `return_direct=True` | [`ToolSpec::return_direct`] |
//!
//! ## 为什么工具必须是 **async**
//!
//! 13-agent-core 的工具是**同步**的（BM25 检索 + 手写算术），所以
//! `fn run(&self, args: &Value) -> ToolOutcome` 就够了。
//!
//! 但本项目的核心工具要**访问外部 HTTP 服务**（天气 API / MCP 服务端），
//! 同步的 `fn` 无法表达 I/O 等待 —— 只能同步阻塞（会卡死 tokio 线程）
//! 或者干脆做不到。
//!
//! 所以这里用显式装箱的 future：
//!
//! ```text
//! async fn      →  fn f() -> Pin<Box<dyn Future<Output = T> + Send + '_>>
//! ```
//!
//! `async fn` 在 trait 里**不是 dyn 兼容**的 —— 而工具注册表需要
//! `Box<dyn Tool>`，所以必须显式装箱。
//! （和 [`crate::llm`] 里的 `LlmFuture` 是同一个道理。）

use agent_logging::{debug, info, warn};
use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::ToolError;
use crate::schema;

/// 工具执行返回给调用方的 future。
pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolOutcome, ToolError>> + Send + 'a>>;

/// 工具定义（名字 + 描述 + JSON Schema）。
///
/// 这三项是**工具的「契约」**：
///
/// - **名字** —— 模型用它来调用，必须是 `snake_case`
///   （LangChain 文档明确警告：带空格或特殊字符的名字会被部分厂商拒绝）
/// - **描述** —— 模型靠它判断「什么时候该调这个工具」
/// - **Schema** —— 既告诉模型参数怎么填，也用于**本地校验**
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// 工具名（`snake_case`）。
    pub name: String,

    /// 工具描述。
    pub description: String,

    /// JSON Schema（参数定义）。
    pub schema: Value,

    /// 是否「直接返回」。
    ///
    /// ## 对应 LangChain 的 `return_direct=True`
    ///
    /// 工具的输出**不需要再过一遍模型** —— 直接作为最终答案返回。
    ///
    /// ⚠️ 只有当「工具输出本身就是完整答案」时才该开：
    /// 需要模型再总结、再推理的场景开了会丢掉推理能力。
    ///
    /// ⚠️ LangChain 文档里还有一个细节：
    /// **同一批并行调用里，只有全部工具都是 `return_direct` 才短路**；
    /// 混着调用时不会提前退出。
    /// 本项目串行执行工具，所以只看当前这一个工具。
    pub return_direct: bool,
}

impl ToolSpec {
    /// 构造一个工具定义（不带 `return_direct`）。
    pub fn new(name: impl Into<String>, description: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            return_direct: false,
        }
    }

    /// 打开 `return_direct`。
    pub fn with_return_direct(mut self) -> Self {
        self.return_direct = true;
        self
    }

    /// 把 Schema 渲染成给模型看的参数说明。
    pub fn render_parameters(&self) -> String {
        schema::render_schema(&self.schema)
    }
}

/// 工具执行的结果。
#[derive(Debug, Clone)]
pub struct ToolOutcome {
    /// 给模型看的文本（会变成 `Observation:`）。
    pub observation: String,

    /// 是否成功（失败时 `observation` 是错误描述）。
    pub ok: bool,

    /// 工具名（用于日志）。
    pub tool: String,

    /// 结构化结果（可选）。
    ///
    /// ## 对应 LangChain 的「返回 object」
    ///
    /// LangChain 教程里说：返回 `str` 时「模型读到这段文本并决定下一步」；
    /// 返回 `dict` 时「模型能读具体字段并推理」。
    ///
    /// 本项目两者都支持 —— 天气工具返回结构化数据（温度、风力等字段），
    /// 同时把同样的信息渲染成可读文本作为 `observation`。
    /// 结构化数据进 API 响应，文本进模型上下文。
    pub data: Option<Value>,
}

impl ToolOutcome {
    /// 成功的结果（纯文本）。
    pub fn ok(tool: impl Into<String>, observation: impl Into<String>) -> Self {
        Self {
            observation: observation.into(),
            ok: true,
            tool: tool.into(),
            data: None,
        }
    }

    /// 成功的结果（带结构化数据）。
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

    /// 失败的结果（**不是错误** —— 错误文本会喂回模型让它自我纠正）。
    pub fn failed(tool: impl Into<String>, observation: impl Into<String>) -> Self {
        Self {
            observation: observation.into(),
            ok: false,
            tool: tool.into(),
            data: None,
        }
    }
}

/// 工具运行时可读的上下文。
///
/// ## 对应 LangChain 的 `ToolRuntime`（简化版）
///
/// LangChain 的 `ToolRuntime` 能访问 state（短期记忆）、context（不可变配置）、
/// store（长期记忆）、stream_writer（流式输出）等。
///
/// 本项目只实现**真正有用的那一部分**：不可变的调用上下文
/// （用户身份、超时预算、是否允许联网）。
///
/// ### 为什么不给工具「写状态」的能力
///
/// LangChain 里工具返回 `Command` 就能更新 agent state。
/// 本项目刻意**不做**这件事 —— 因为：
///
/// 1. ReAct 循环的「短期记忆」已经由**轨迹**承担了（`Thought/Action/Observation`）
/// 2. 工具写状态会让工具从「纯函数」变成「有副作用」，
///    那样同一个工具就没法被安全地重试了
/// 3. 本项目的核心教学点是**工具契约**，不是状态管理
///
/// 需要长期记忆的话，08-rag 的 BM25 索引已经扮演了这个角色。
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    /// 用户标识（可用于按用户做限流 / 审计）。
    pub user_id: Option<String>,

    /// 单次工具调用的超时（毫秒）。
    ///
    /// 默认 0 = 不额外限制（用 HTTP 客户端自己的超时）。
    pub timeout_ms: u64,
}

impl ToolContext {
    /// 空上下文。
    pub fn empty() -> Self {
        Self::default()
    }

    /// 带用户标识的上下文。
    pub fn for_user(user_id: impl Into<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            timeout_ms: 0,
        }
    }
}

/// 工具 trait。
///
/// 实现它就等于在 LangChain 里写一个 `@tool` 函数。
///
/// ## 三个必须做对的地方
///
/// 1. **描述** —— 说清楚「什么时候该调我」，不只是「我是什么」
/// 2. **Schema** —— 参数名用模型的词表（`location` 而不是 `q`）
/// 3. **错误处理** —— 失败时返回 [`ToolError`]，由调用方转成 `Observation`
pub trait Tool: Send + Sync {
    /// 工具定义。
    fn spec(&self) -> ToolSpec;

    /// 执行工具。
    ///
    /// ## 返回 `Err` 的语义
    ///
    /// `Err(ToolError)` 表示「这次调用失败了，原因在这里」。
    /// 它**不会**中断整个 Agent 循环 —— 调用方（[`ToolRegistry::call`]）
    /// 会把它渲染成一条 `Observation` 喂回模型。
    fn call<'a>(&'a self, args: &'a Value, ctx: &'a ToolContext) -> ToolFuture<'a>;
}

/// 工具失败时的处理策略。
///
/// 对应 LangChain 的 `handle_tool_errors` 中间件。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPolicy {
    /// 把错误转成 `Observation` 喂回模型（**默认**）。
    ///
    /// ## 为什么这是默认值
    ///
    /// 模型读到「参数 `location` 必须是字符串」就能自己纠正。
    /// 而如果整个请求直接失败，用户看到的是一个 500 ——
    /// **模型连自我纠正的机会都没有**。
    ///
    /// 这正是 LangChain 教程里 `handle_tool_errors` 的用意：
    /// ```python
    /// return ToolMessage(content=f"Tool error: ...", tool_call_id=...)
    /// ```
    ConvertToObservation,

    /// 直接向上传播错误（整个调用失败）。
    ///
    /// 适用于「工具结果不可信，模型拿到错误信息反而会被误导」的场景。
    Propagate,
}

/// 工具注册表。
///
/// 负责：按名字分发、参数校验、错误转观察。
///
/// ## 为什么需要这一层
///
/// Agent 循环不该关心「工具失败怎么办」「参数怎么校验」——
/// 这些是**工具基础设施**，不是循环逻辑。
///
/// 而且这一层让「动态工具选择」变得很自然：
/// 造一个新的注册表，筛选出本次允许使用的工具就行。
#[derive(Clone)]
pub struct ToolRegistry {
    tools: Vec<std::sync::Arc<dyn Tool>>,
    policy: ToolPolicy,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field(
                "tools",
                &self.tools.iter().map(|t| t.spec().name).collect::<Vec<_>>(),
            )
            .field("policy", &self.policy)
            .finish()
    }
}

impl ToolRegistry {
    /// 创建注册表（默认策略：转观察）。
    pub fn new(tools: Vec<std::sync::Arc<dyn Tool>>) -> Self {
        Self {
            tools,
            policy: ToolPolicy::ConvertToObservation,
        }
    }

    /// 更换错误策略。
    pub fn with_policy(mut self, policy: ToolPolicy) -> Self {
        self.policy = policy;
        self
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
    pub fn get(&self, name: &str) -> Option<&std::sync::Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.spec().name == name)
    }

    /// 筛选出子集（**动态工具选择**）。
    ///
    /// ## 对应 LangChain 的动态工具选择
    ///
    /// LangChain 教程里说：「工具太多会让模型过载（overload context）
    /// 并增加错误；太少又限制了能力」。
    ///
    /// 本项目提供三种筛选维度：
    ///
    /// - 按名字白名单 —— 权限 / 功能开关
    /// - 按名字前缀 —— 「只暴露公开工具」
    /// - 按 `return_direct` —— 「只暴露直通工具」
    ///
    /// 克隆开销很小（`Arc` 共享），所以「筛选」比「重建」自然得多。
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&ToolSpec) -> bool,
    {
        Self {
            tools: self
                .tools
                .iter()
                .filter(|t| predicate(&t.spec()))
                .cloned()
                .collect(),
            policy: self.policy,
        }
    }

    /// 按名字白名单筛选。
    pub fn allow_names(&self, names: &[&str]) -> Self {
        self.filter(|spec| names.contains(&spec.name.as_str()))
    }

    /// 按名字前缀筛选。
    pub fn allow_prefix(&self, prefix: &str) -> Self {
        self.filter(|spec| spec.name.starts_with(prefix))
    }

    /// 只保留「直通」工具。
    pub fn allow_return_direct(&self) -> Self {
        self.filter(|spec| spec.return_direct)
    }

    /// 合并另一个注册表的工具（同名时后者覆盖前者）。
    ///
    /// 用来把 MCP 工具和本地工具拼成一份清单。
    pub fn merge(&self, other: &ToolRegistry) -> Self {
        let mut tools = self.tools.clone();
        for tool in &other.tools {
            tools.retain(|t| t.spec().name != tool.spec().name);
            tools.push(tool.clone());
        }
        Self {
            tools,
            policy: self.policy,
        }
    }

    /// 调用工具 —— **失败转观察**。
    ///
    /// ## 这是 Agent 循环唯一应该调用的入口
    ///
    /// 它负责四件事：
    ///
    /// 1. 找工具（找不到 → 生成「可用工具列表」的错误观察）
    /// 2. 填默认值（LangChain 的 `Field(default=...)` 语义）
    /// 3. 校验参数（Schema 不通过 → 生成可执行的错误观察）
    /// 4. 执行工具（失败 → 生成错误观察）
    ///
    /// 全部「失败」都变成 `ok = false` 的 [`ToolOutcome`]，
    /// **永远不返回 `Err`**（除非策略是 `Propagate`）。
    pub async fn call(&self, name: &str, args: &Value, ctx: &ToolContext) -> ToolOutcome {
        let outcome = match self.call_inner(name, args, ctx).await {
            Ok(outcome) => outcome,
            Err(err) => {
                warn!(tool = %name, error = %err, policy = ?self.policy, "工具调用失败");
                match self.policy {
                    ToolPolicy::ConvertToObservation => ToolOutcome::failed(name, err.to_string()),
                    ToolPolicy::Propagate => ToolOutcome::failed(
                        name,
                        format!("错误（策略为 Propagate，未中断流程）: {err}"),
                    ),
                }
            }
        };

        if outcome.ok {
            debug!(tool = %name, "工具调用成功");
        }

        outcome
    }

    /// 调用工具 —— 错误**向上传播**。
    ///
    /// 给「用户直接调工具」的 HTTP 端点用：那时用户就是模型，
    /// 他应该看到真正的错误，而不是被喂回去的观察。
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
            warn!(tool = %name, "未知工具");
            return Err(ToolError::unknown(name, self.names().join(", ")));
        };

        let spec = tool.spec();

        // ── 参数准备 ────────────────────────────────────────────
        let mut object = match args {
            Value::Object(map) => map.clone(),
            Value::Null => serde_json::Map::new(),
            other => {
                return Err(ToolError::invalid_args(vec![format!(
                    "必须是 JSON 对象，实际是 {}",
                    match other {
                        Value::String(_) => "string",
                        Value::Number(_) => "number",
                        Value::Bool(_) => "boolean",
                        Value::Array(_) => "array",
                        _ => "unknown",
                    }
                )]));
            }
        };

        // 填默认值 → 再校验。
        schema::fill_defaults(&spec.schema, &mut object);

        let validation = schema::validate(&spec.schema, &Value::Object(object.clone()));
        if !validation.is_ok() {
            warn!(
                tool = %name,
                issues = validation.issues.len(),
                "参数校验不通过 —— 不发请求"
            );
            return Err(ToolError::invalid_args(
                validation.issues.iter().map(|i| i.to_string()).collect(),
            ));
        }

        // ── 执行 ────────────────────────────────────────────────
        info!(tool = %name, "开始执行工具");
        tool.call(&Value::Object(object), ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 一个确定性的假工具：把 `echo` 参数原样返回。
    struct EchoTool;

    impl Tool for EchoTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(
                "echo",
                "回显传入的文本。用于测试。",
                json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "要回显的文本" }
                    },
                    "required": ["text"]
                }),
            )
        }

        fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
            let text = args
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Box::pin(async move { Ok(ToolOutcome::ok("echo", format!("回显: {text}"))) })
        }
    }

    /// 一个总是失败的工具。
    struct FailingTool;

    impl Tool for FailingTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new("boom", "总是失败", json!({ "type": "object" }))
        }

        fn call<'a>(&'a self, _args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
            Box::pin(async move { Err(ToolError::execution("boom", "炸了")) })
        }
    }

    fn registry() -> ToolRegistry {
        ToolRegistry::new(vec![
            std::sync::Arc::new(EchoTool),
            std::sync::Arc::new(FailingTool),
        ])
    }

    #[tokio::test]
    async fn calls_a_tool_successfully() {
        let reg = registry();
        let outcome = reg
            .call("echo", &json!({ "text": "你好" }), &ToolContext::empty())
            .await;

        assert!(outcome.ok);
        assert!(outcome.observation.contains("你好"));
    }

    #[tokio::test]
    async fn unknown_tool_becomes_observation_listing_available_tools() {
        let reg = registry();
        let outcome = reg.call("nope", &json!({}), &ToolContext::empty()).await;

        assert!(!outcome.ok);
        assert!(outcome.observation.contains("未知工具"));
        // 必须告诉模型有哪些工具可用 —— 它要靠这个自我纠正。
        assert!(outcome.observation.contains("echo"));
        assert!(outcome.observation.contains("boom"));
    }

    #[tokio::test]
    async fn invalid_args_become_actionable_observation() {
        let reg = registry();
        let outcome = reg.call("echo", &json!({}), &ToolContext::empty()).await;

        assert!(!outcome.ok);
        assert!(outcome.observation.contains("text"));
        assert!(outcome.observation.contains("必填"));
    }

    #[tokio::test]
    async fn tool_failure_becomes_observation_not_error() {
        let reg = registry();
        let outcome = reg.call("boom", &json!({}), &ToolContext::empty()).await;

        // 关键：调用**不返回 Err** —— 失败被转成了观察。
        assert!(!outcome.ok);
        assert!(outcome.observation.contains("炸了"));
    }

    #[tokio::test]
    async fn strict_call_propagates_errors() {
        let reg = registry();
        let err = reg
            .call_strict("boom", &json!({}), &ToolContext::empty())
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::Execution { .. }));
    }

    #[tokio::test]
    async fn defaults_are_filled_before_validation() {
        struct DefaultTool;

        impl Tool for DefaultTool {
            fn spec(&self) -> ToolSpec {
                ToolSpec::new(
                    "with_default",
                    "有默认参数的工具",
                    json!({
                        "type": "object",
                        "properties": {
                            "city": { "type": "string", "default": "北京" }
                        }
                    }),
                )
            }

            fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
                let city = args
                    .get("city")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Box::pin(async move { Ok(ToolOutcome::ok("with_default", city)) })
            }
        }

        let reg = ToolRegistry::new(vec![std::sync::Arc::new(DefaultTool)]);
        let outcome = reg
            .call("with_default", &json!({}), &ToolContext::empty())
            .await;

        assert!(outcome.ok);
        assert_eq!(outcome.observation, "北京");
    }

    #[test]
    fn filter_by_name_whitelist() {
        let reg = registry();
        let filtered = reg.allow_names(&["echo"]);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.names(), vec!["echo".to_string()]);
    }

    #[test]
    fn filter_by_prefix() {
        let reg = registry();
        assert_eq!(reg.allow_prefix("e").len(), 1);
        assert_eq!(reg.allow_prefix("zzz").len(), 0);
    }

    #[test]
    fn merge_replaces_same_named_tools() {
        let reg = registry();
        let merged = reg.merge(&ToolRegistry::new(vec![std::sync::Arc::new(EchoTool)]));

        // echo 只有一个（被覆盖，不是重复）。
        assert_eq!(merged.names().iter().filter(|n| *n == "echo").count(), 1);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn spec_renders_parameters_for_the_prompt() {
        let spec = EchoTool.spec();
        let rendered = spec.render_parameters();

        assert!(rendered.contains("text (string, 必填)"));
        assert!(rendered.contains("要回显的文本"));
    }
}
