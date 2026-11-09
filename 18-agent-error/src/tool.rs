//! 工具层 —— 定义「工具失败」到底意味着什么。
//!
//! ## 三个工具，三种失败语义
//!
//! | 工具 | 典型失败 | 类别 | 该怎么办 |
//! |---|---|---|---|
//! | [`FlakyTool`] | 前 N 次失败、之后成功 | **Transient** | 重试 → 成功 |
//! | [`StrictTool`] | 参数不合法 | **Business** | 不重试 → 降级给模型改参数 |
//! | [`GuardedTool`] | 鉴权失败 | **Fatal** | 不重试 → 终止 |
//!
//! ## 为什么工具自己就要声明失败类别
//!
//! 因为**只有工具知道**失败是什么性质：
//!
//! - 「数据库文件被锁」→ Transient（等一会就好）
//! - 「字段 `age` 不存在」→ Business（模型改参数才行）
//! - 「API key 无效」→ Fatal（人要改配置）
//!
//! 把这个判断放在**工具内部**，中间件才有一个统一的入口
//! 来做「重试 / 降级 / 终止」的决策。
//!
//! ## 与 16 的关系
//!
//! 16 的 `FunctionTool` 返回 `Result<ToolOutcome, ToolError>`，
//! 但它的 `ToolError` 只有四种（未知工具 / 参数不合法 / 执行失败 / 上游失败），
//! **没有 Transient / Business / Fatal 的区分** —— 所以本项目自己写一套 trait。
//!
//! 但**Schema 校验器直接复用 14 的**（`agent_14::schema`）——
//! 校验规则没必要重写第二遍。

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_logging::{debug, warn};
use serde_json::{Value, json};

use crate::error::{ErrorCode, ErrorKind, ToolError};
use crate::tools::ToolSpec;

/// 工具执行返回给调用方的 future。
///
/// ## 为什么是 `Pin<Box<dyn Future>>`
///
/// 和 14 的 `ToolFuture`、16 的 `ToolFuture` 同理：
/// `async fn` 在 trait 里**不是 dyn 兼容**的，
/// 而工具注册表需要 `Box<dyn ErrorAwareTool>`。
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
}

/// 工具运行时上下文。
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    /// 用户标识（可用于审计 / 限流）。
    pub user_id: Option<String>,

    /// 当前是第几次尝试（从 1 开始）。
    ///
    /// ## 为什么工具需要知道尝试次数
    ///
    /// 有些工具「前几次失败、之后成功」（比如有本地缓存的 flaky 上游）。
    /// 工具需要这个信息才能决定**自己**要不要再试一次。
    pub attempt: usize,
}

impl ToolContext {
    /// 空上下文。
    pub fn empty() -> Self {
        Self::default()
    }

    /// 指定尝试次数的上下文。
    pub fn with_attempt(attempt: usize) -> Self {
        Self {
            user_id: None,
            attempt: attempt.max(1),
        }
    }
}

/// 一个**会声明失败类别**的工具。
///
/// ## 与 16 的 `FunctionTool` 的区别
///
/// 16 的 trait 返回 `Result<ToolOutcome, agent_16::error::ToolError>`，
/// 错误类型的分类太粗。
///
/// 本项目的 trait 返回 `Result<ToolOutcome, crate::error::ToolError>`，
/// 后者带 [`ErrorKind`] —— **中间件据此决定重试还是降级**。
pub trait ErrorAwareTool: Send + Sync {
    /// 工具定义。
    fn spec(&self) -> ToolSpec;

    /// 执行工具。
    ///
    /// `Err` 表示「这次调用失败了」——
    /// 它**不会**中断 Agent 循环，中间件会按 `kind` 处置：
    /// 重试、降级成观察，或终止。
    fn call<'a>(&'a self, args: &'a Value, ctx: &'a ToolContext) -> ToolFuture<'a>;
}

// ─────────────────────────────────────────────────────────────────────
// ① FlakyTool —— 可重试错误的代表
// ─────────────────────────────────────────────────────────────────────

/// `flaky_fetch` 工具名。
pub const FLAKY_FETCH_TOOL_NAME: &str = "flaky_fetch";

/// 一个**前 N 次失败、之后成功**的工具。
///
/// ## 为什么要有它
///
/// 它是「可重试错误」的可执行样本：
/// 如果中间件的重试逻辑写对了，这个工具最终会成功；
/// 如果写错了（不重试），Agent 会拿到一个失败的观察。
///
/// ## 失败模式
///
/// - `fail_times = 2` → 第 1、2 次失败（Transient），第 3 次成功
/// - `fail_times = 99` → 一直失败 → 重试耗尽后降级
pub struct FlakyTool {
    fail_times: usize,
    calls: AtomicUsize,
}

impl std::fmt::Debug for FlakyTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlakyTool")
            .field("fail_times", &self.fail_times)
            .field("calls", &self.calls.load(Ordering::Relaxed))
            .finish()
    }
}

impl FlakyTool {
    /// 创建工具（前 `fail_times` 次失败）。
    pub fn new(fail_times: usize) -> Self {
        Self {
            fail_times,
            calls: AtomicUsize::new(0),
        }
    }

    /// 总共被调用了几次。
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl ErrorAwareTool for FlakyTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            FLAKY_FETCH_TOOL_NAME,
            "从某个不稳定的上游拉取数据。前几次调用会失败，\
             这属于**临时故障**，重试后可能成功。\
             如果重试仍然失败，请如实告知用户服务暂时不可用。",
            json!({
                "type": "object",
                "properties": {
                    "resource": {
                        "type": "string",
                        "description": "要拉取的资源标识，例如 \"report/2026\"。"
                    }
                },
                "required": ["resource"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;

        Box::pin(async move {
            let resource = args
                .get("resource")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();

            if n <= self.fail_times {
                // ⚠️ 这是**可重试**错误 —— 上游临时不可用。
                debug!(tool = FLAKY_FETCH_TOOL_NAME, attempt = n, "上游临时故障");
                return Err(ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    FLAKY_FETCH_TOOL_NAME,
                    format!("上游暂时不可用（第 {n} 次尝试）"),
                )
                .with_hint("这是临时故障，稍后重试可能成功。"));
            }

            Ok(ToolOutcome::ok_with_data(
                FLAKY_FETCH_TOOL_NAME,
                format!("拉取成功：{resource}"),
                json!({ "resource": resource, "attempts": n }),
            ))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ② StrictTool —— 业务失败的代表
// ─────────────────────────────────────────────────────────────────────

/// `strict_lookup` 工具名。
pub const STRICT_LOOKUP_TOOL_NAME: &str = "strict_lookup";

/// 一个**参数不合法就失败**的工具。
///
/// ## 失败模式
///
/// - 缺少必填参数 → Business（InvalidArguments）
/// - 参数类型不对 → Business（InvalidArguments）
/// - 查无此记录 → Business（NotFound）
///
/// ⚠️ 全部都是 **Business**：重试**不可能**成功。
pub struct StrictTool {
    known_ids: Vec<String>,
}

impl std::fmt::Debug for StrictTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StrictTool")
            .field("known_ids", &self.known_ids)
            .finish()
    }
}

impl StrictTool {
    /// 创建工具（`known_ids` 是「存在」的记录 id）。
    pub fn new(known_ids: Vec<String>) -> Self {
        Self { known_ids }
    }
}

impl ErrorAwareTool for StrictTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            STRICT_LOOKUP_TOOL_NAME,
            "按 id 查询一条记录。id 必须是字符串。\
             如果 id 不存在或格式不对，工具会明确告诉你哪里错了，\
             请修正后重试，**不要**凭空编造记录内容。",
            json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "记录 id，例如 \"rec-1\"。"
                    }
                },
                "required": ["id"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            // ⚠️ 类型不对 → Business。
            if let Some(id) = args.get("id")
                && !id.is_string()
            {
                return Err(ToolError::invalid_args(
                    STRICT_LOOKUP_TOOL_NAME,
                    format!("`id` 必须是字符串，实际是 {}", json_type_name(id)),
                )
                .with_hint("把 `id` 改成字符串，例如 \"rec-1\"。"));
            }

            let id = args
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();

            if id.is_empty() {
                return Err(
                    ToolError::invalid_args(STRICT_LOOKUP_TOOL_NAME, "`id` 不能为空")
                        .with_hint("传入一个非空的记录 id，例如 \"rec-1\"。"),
                );
            }

            if !self.known_ids.iter().any(|k| k == &id) {
                // ⚠️ 查无此记录也是 **Business** ——
                //    重试同一个不存在的 id 永远不会成功。
                return Err(ToolError::not_found(
                    STRICT_LOOKUP_TOOL_NAME,
                    format!("记录 `{id}` 不存在"),
                )
                .with_hint(format!(
                    "可用的 id 有：{}。请从列表里选一个，或向用户确认。",
                    self.known_ids.join(", ")
                )));
            }

            Ok(ToolOutcome::ok_with_data(
                STRICT_LOOKUP_TOOL_NAME,
                format!("找到记录 {id}"),
                json!({ "id": id }),
            ))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ③ GuardedTool —— 致命错误的代表
// ─────────────────────────────────────────────────────────────────────

/// `guarded_pay` 工具名。
pub const GUARDED_PAY_TOOL_NAME: &str = "guarded_pay";

/// 一个**鉴权失败就报错**的工具。
///
/// ## 失败模式
///
/// - 缺少 `api_key` → Fatal（Unauthorized）
///
/// ⚠️ Fatal **不能**降级成观察 ——
/// 降级后模型会以为「工具坏了」，转而编一个「支付失败」的答案，
/// 把配置问题掩盖成幻觉。
pub struct GuardedTool {
    api_key: Option<String>,
}

impl std::fmt::Debug for GuardedTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ⚠️ 不打印 api_key —— 密钥不能进日志。
        f.debug_struct("GuardedTool")
            .field("has_key", &self.api_key.is_some())
            .finish()
    }
}

impl GuardedTool {
    /// 创建工具（`api_key` 为 `None` 表示没配密钥）。
    pub fn new(api_key: Option<String>) -> Self {
        Self { api_key }
    }
}

impl ErrorAwareTool for GuardedTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            GUARDED_PAY_TOOL_NAME,
            "发起一次支付。需要 API key 才能调用。",
            json!({
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "支付金额（元）。"
                    }
                },
                "required": ["amount"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            if self.api_key.is_none() {
                // ⚠️ Fatal —— 重试没用，降级会掩盖问题。
                warn!(tool = GUARDED_PAY_TOOL_NAME, "鉴权失败：未配置 API key");
                return Err(ToolError::unauthorized(
                    GUARDED_PAY_TOOL_NAME,
                    "未配置支付服务的 API key",
                ));
            }

            let amount = args.get("amount").and_then(Value::as_f64).unwrap_or(0.0);

            Ok(ToolOutcome::ok_with_data(
                GUARDED_PAY_TOOL_NAME,
                format!("支付成功：{amount} 元"),
                json!({ "amount": amount }),
            ))
        })
    }
}

/// JSON 类型名（用于错误文案）。
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

/// 工具注册表。
///
/// ## 与 16 的 `ToolRegistry` 的区别
///
/// 16 的注册表在调用失败时**直接把错误变成观察**（`ToolOutcome::failed`）——
/// 它**没有重试的概念**。
///
/// 本项目的注册表**只负责分发与校验**，
/// 「失败之后怎么办」交给中间件（[`crate::middleware`]）——
/// 因为那是**策略**，不该固化在注册表里。
#[derive(Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn ErrorAwareTool>>,
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
    pub fn new(tools: Vec<Arc<dyn ErrorAwareTool>>) -> Self {
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
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ErrorAwareTool>> {
        self.tools.iter().find(|t| t.spec().name == name)
    }

    /// 调用工具 —— **错误向上传播**。
    ///
    /// ## 与 16 的 `call` 的区别
    ///
    /// 16 的 `call` 失败时返回 `ToolOutcome::failed`（错误被吞掉）。
    /// 本项目的 `call` **保留错误** —— 因为重试/降级策略在中间件里，
    /// 中间件需要看到错误的**类别**才能决策。
    pub async fn call(
        &self,
        name: &str,
        args: &Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let Some(tool) = self.get(name) else {
            return Err(ToolError::unknown_tool(name, self.names().join(", ")));
        };

        let spec = tool.spec();

        // ── 参数准备 ────────────────────────────────────────────
        let mut object = match args {
            Value::Object(map) => map.clone(),
            Value::Null => serde_json::Map::new(),
            other => {
                return Err(ToolError::invalid_args(
                    name,
                    format!("参数必须是 JSON 对象，实际是 {}", json_type_name(other)),
                )
                .with_hint("把参数改成 JSON 对象，例如 {\"key\": \"value\"}。"));
            }
        };

        // 填默认值 → 再校验（顺序不能反）。
        agent_14::schema::fill_defaults(&spec.schema, &mut object);

        let validation = agent_14::schema::validate(&spec.schema, &Value::Object(object.clone()));
        if !validation.is_ok() {
            let issues: Vec<String> = validation.issues.iter().map(|i| i.to_string()).collect();
            return Err(ToolError::invalid_args(name, issues.join("；"))
                .with_hint("按上面的提示修正参数后重新调用。"));
        }

        debug!(tool = %name, "开始执行工具");
        tool.call(&Value::Object(object), ctx).await
    }
}

/// 构建默认的演示注册表。
///
/// ## 三个工具分别代表三类失败
///
/// - `flaky_fetch` —— Transient（重试能好）
/// - `strict_lookup` —— Business（改参数才行）
/// - `guarded_pay` —— Fatal（人改配置才行）
pub fn demo_registry() -> ToolRegistry {
    ToolRegistry::new(vec![
        Arc::new(FlakyTool::new(2)),
        Arc::new(StrictTool::new(vec![
            "rec-1".to_string(),
            "rec-2".to_string(),
        ])),
        Arc::new(GuardedTool::new(None)),
    ])
}

/// 让 `ErrorKind` 的导入不被判定为未使用（它在文档里被引用）。
#[allow(dead_code)]
fn _kind_marker(kind: ErrorKind) -> ErrorKind {
    kind
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn flaky_tool_fails_transiently_then_succeeds() {
        let tool = FlakyTool::new(2);

        let err = tool
            .call(&json!({ "resource": "r" }), &ToolContext::empty())
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Transient);
        assert!(err.retryable());

        let err = tool
            .call(&json!({ "resource": "r" }), &ToolContext::empty())
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Transient);

        // 第 3 次成功。
        let outcome = tool
            .call(&json!({ "resource": "r" }), &ToolContext::empty())
            .await
            .unwrap();
        assert!(outcome.observation.contains("拉取成功"));
        assert_eq!(tool.call_count(), 3);
    }

    #[tokio::test]
    async fn strict_tool_reports_business_failures() {
        let tool = StrictTool::new(vec!["rec-1".to_string()]);

        // 类型不对。
        let err = tool
            .call(&json!({ "id": 42 }), &ToolContext::empty())
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Business);
        assert!(!err.retryable());
        assert!(err.to_string().contains("E_INVALID_ARGUMENTS"));

        // 查无此记录 —— 也是 Business。
        let err = tool
            .call(&json!({ "id": "nope" }), &ToolContext::empty())
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Business);
        assert_eq!(err.code, ErrorCode::NotFound);
        // hint 里要列出可用的 id —— 模型才知道能选什么。
        assert!(err.to_string().contains("rec-1"));
    }

    #[tokio::test]
    async fn guarded_tool_reports_fatal_failure() {
        let tool = GuardedTool::new(None);

        let err = tool
            .call(&json!({ "amount": 10 }), &ToolContext::empty())
            .await
            .unwrap_err();

        assert_eq!(err.kind, ErrorKind::Fatal);
        assert!(!err.retryable());
        assert!(!err.degradable());
    }

    #[tokio::test]
    async fn guarded_tool_succeeds_with_a_key() {
        let tool = GuardedTool::new(Some("k".to_string()));
        let outcome = tool
            .call(&json!({ "amount": 10 }), &ToolContext::empty())
            .await
            .unwrap();
        assert!(outcome.ok);
    }

    #[tokio::test]
    async fn registry_validates_arguments_before_calling() {
        // ⚠️ Schema 校验失败属于 Business —— 不是 Transient。
        let registry = demo_registry();
        let err = registry
            .call(
                FLAKY_FETCH_TOOL_NAME,
                &json!({ "resource": 42 }),
                &ToolContext::empty(),
            )
            .await
            .unwrap_err();

        assert_eq!(err.kind, ErrorKind::Business);
        assert_eq!(err.code, ErrorCode::InvalidArguments);
    }

    #[tokio::test]
    async fn registry_reports_unknown_tool() {
        let registry = demo_registry();
        let err = registry
            .call("nope", &json!({}), &ToolContext::empty())
            .await
            .unwrap_err();

        assert_eq!(err.kind, ErrorKind::Business);
        assert!(err.to_string().contains(FLAKY_FETCH_TOOL_NAME));
    }

    #[tokio::test]
    async fn registry_rejects_non_object_arguments() {
        let registry = demo_registry();
        let err = registry
            .call(
                FLAKY_FETCH_TOOL_NAME,
                &json!("不是对象"),
                &ToolContext::empty(),
            )
            .await
            .unwrap_err();

        assert_eq!(err.kind, ErrorKind::Business);
        assert!(err.to_string().contains("JSON 对象"));
    }

    #[test]
    fn demo_registry_has_three_tools() {
        let registry = demo_registry();
        assert_eq!(registry.len(), 3);
        assert!(
            registry
                .names()
                .contains(&FLAKY_FETCH_TOOL_NAME.to_string())
        );
        assert!(
            registry
                .names()
                .contains(&STRICT_LOOKUP_TOOL_NAME.to_string())
        );
        assert!(
            registry
                .names()
                .contains(&GUARDED_PAY_TOOL_NAME.to_string())
        );
    }
}
