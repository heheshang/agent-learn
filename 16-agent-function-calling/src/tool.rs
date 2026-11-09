//! 工具 trait 与注册表 —— 「模型说调哪个工具、参数是什么」之后的全部机制。
//!
//! ## 与 14-agent-tool 的关系
//!
//! 14 的 `ToolRegistry` 已经把三件事做对了：
//!
//! 1. 按名字分发
//! 2. **本地 Schema 校验**（模型填错就别发请求）
//! 3. **失败转 Observation**（错误喂回模型让它自我纠正）
//!
//! 本项目直接**复用**它的校验器（`agent_14::schema`）和错误类型
//! （`agent_14::error::ToolError`），但自己写了一个 trait 和注册表 ——
//! 因为 14 的 `ToolSpec` **没有 `strict` 字段**，而 strict 是本项目的核心之一。
//!
//! ## 为什么「本地校验」在本项目更重要
//!
//! OpenAI 文档说 strict 模式能「确保函数调用可靠地遵守 schema」。
//! **但 DeepSeek 接受 strict 却并不强制执行**（实测）。
//!
//! 所以本项目的立场很明确：
//!
//! > **参数正确性不能外包给模型。**
//!
//! 本地 Schema 校验是**必需品**，不是「优化」。
//!
//! ## 「失败转 Observation」在结构化协议下更自然
//!
//! 13/14/15 的文本协议里，工具失败要渲染成 `Observation: 错误...` 塞进提示词。
//! 本项目不用渲染 —— 工具结果本来就是**一条 `tool` 角色的消息**，
//! 内容就是错误文本。模型看到它同样能自我纠正。
//!
//! ```text
//! assistant: tool_calls=[get_weather(city="Atlantis")]
//! tool:      错误：找不到城市 `Atlantis`，请换一个城市名。
//! ```
//!
//! 实测（deepseek-flash）：模型读到这条错误后，没有编造答案，
//! 而是如实转述了错误并主动问「需要我查一下 Atlanta 的天气吗？」。

use agent_logging::{debug, info, warn};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::error::ToolError;
use crate::tools::ToolSpec;

/// 工具执行返回给调用方的 future。
///
/// ## 为什么是 `Pin<Box<dyn Future>>`
///
/// 和 14 的 `ToolFuture`、07 的 `LlmFuture` 同理：
/// `async fn` 在 trait 里**不是 dyn 兼容**的，
/// 而工具注册表需要 `Box<dyn FunctionTool>`。
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
}

impl ToolContext {
    /// 空上下文。
    pub fn empty() -> Self {
        Self::default()
    }
}

/// 一个「可被模型调用」的函数。
///
/// 实现它就等于在 OpenAI 里定义一个 function。
///
/// 三个必须做对的地方（和 14 一致）：
///
/// 1. **描述** —— 说清楚「什么时候该调我」
/// 2. **Schema** —— 参数名用模型的词表
/// 3. **错误处理** —— 失败时返回 [`ToolError`]
pub trait FunctionTool: Send + Sync {
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
    tools: Vec<Arc<dyn FunctionTool>>,
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
    pub fn new(tools: Vec<Arc<dyn FunctionTool>>) -> Self {
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
    pub fn get(&self, name: &str) -> Option<&Arc<dyn FunctionTool>> {
        self.tools.iter().find(|t| t.spec().name == name)
    }

    /// 按名字白名单筛选（**动态工具选择**）。
    ///
    /// OpenAI 最佳实践里有一条：
    /// 「**保持初始可用函数的数量足够少**，以获得更高的准确率
    /// —— 建议一开始少于 20 个函数。」
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
            Ok(outcome) => {
                debug!(tool = %name, ok = outcome.ok, "工具调用完成");
                outcome
            }
            Err(err) => {
                warn!(tool = %name, error = %err, "工具调用失败（错误将作为 tool 消息喂回模型）");
                ToolOutcome::failed(name, err.to_string())
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

        // 填默认值 → 再校验（顺序不能反）。
        agent_14::schema::fill_defaults(&spec.schema, &mut object);

        let validation = agent_14::schema::validate(&spec.schema, &Value::Object(object.clone()));
        if !validation.is_ok() {
            warn!(tool = %name, issues = validation.issues.len(), "参数校验不通过 —— 不执行工具");
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

    /// 一个确定性的假工具：把 `text` 原样返回。
    struct EchoTool;

    impl FunctionTool for EchoTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(
                "echo",
                "回显文本",
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
            Box::pin(async move {
                let text = args
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                Ok(ToolOutcome::ok("echo", text))
            })
        }
    }

    /// 一个总是失败的工具。
    struct FailingTool;

    impl FunctionTool for FailingTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(
                "boom",
                "总是失败",
                json!({ "type": "object", "properties": {} }),
            )
        }

        fn call<'a>(&'a self, _args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
            Box::pin(async move { Err(ToolError::execution("boom", "炸了")) })
        }
    }

    fn registry() -> ToolRegistry {
        ToolRegistry::new(vec![Arc::new(EchoTool), Arc::new(FailingTool)])
    }

    #[tokio::test]
    async fn calls_a_tool_successfully() {
        let outcome = registry()
            .call("echo", &json!({ "text": "你好" }), &ToolContext::empty())
            .await;

        assert!(outcome.ok);
        assert_eq!(outcome.observation, "你好");
    }

    #[tokio::test]
    async fn unknown_tool_becomes_an_observation_not_an_error() {
        let outcome = registry()
            .call("nope", &json!({}), &ToolContext::empty())
            .await;

        assert!(!outcome.ok);
        // 错误信息里要列出可用工具 —— 模型才知道能调什么。
        assert!(outcome.observation.contains("echo"));
        assert!(outcome.observation.contains("boom"));
    }

    #[tokio::test]
    async fn invalid_args_become_an_observation() {
        let outcome = registry()
            .call("echo", &json!({}), &ToolContext::empty())
            .await;

        assert!(!outcome.ok);
        assert!(outcome.observation.contains("text"));
    }

    #[tokio::test]
    async fn tool_failure_becomes_an_observation() {
        let outcome = registry()
            .call("boom", &json!({}), &ToolContext::empty())
            .await;

        assert!(!outcome.ok);
        assert!(outcome.observation.contains("炸了"));
    }

    #[tokio::test]
    async fn strict_call_propagates_the_error() {
        let err = registry()
            .call_strict("nope", &json!({}), &ToolContext::empty())
            .await
            .unwrap_err();

        assert!(err.to_string().contains("nope"));
    }

    #[tokio::test]
    async fn rejects_non_object_arguments() {
        let outcome = registry()
            .call("echo", &json!("不是对象"), &ToolContext::empty())
            .await;

        assert!(!outcome.ok);
        assert!(outcome.observation.contains("JSON 对象"));
    }

    #[tokio::test]
    async fn fills_defaults_before_validation() {
        struct DefaultTool;

        impl FunctionTool for DefaultTool {
            fn spec(&self) -> ToolSpec {
                ToolSpec::new(
                    "with_default",
                    "有默认参数",
                    json!({
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "days": { "type": "integer", "default": 3 }
                        },
                        "required": ["city"]
                    }),
                )
            }

            fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
                Box::pin(async move { Ok(ToolOutcome::ok("with_default", args.to_string())) })
            }
        }

        let reg = ToolRegistry::new(vec![Arc::new(DefaultTool)]);
        let outcome = reg
            .call(
                "with_default",
                &json!({ "city": "北京" }),
                &ToolContext::empty(),
            )
            .await;

        assert!(outcome.ok);
        assert!(outcome.observation.contains("3"));
    }

    #[test]
    fn filters_by_allowed_names() {
        let reg = registry().allow_names(&["echo"]);
        assert_eq!(reg.names(), vec!["echo".to_string()]);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn registry_is_debuggable() {
        let text = format!("{:?}", registry());
        assert!(text.contains("echo"));
    }
}
