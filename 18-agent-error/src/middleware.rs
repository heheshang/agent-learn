//! 中间件 —— 对齐 LangChain 的 **`wrap_tool_call`**。
//!
//! # 教程原文的口径
//!
//! LangChain 的[自定义中间件](https://docs.langchain.com/oss/python/langchain/middleware/custom)文档里，
//! 「通过 middleware 返回可诊断的工具失败」这一节的要点是：
//!
//! ```text
//! wrap_tool_call(request, handler) -> ToolMessage | Command
//! ```
//!
//! 中间件包住每次工具调用，可以：
//!
//! - 观察这次调用（打日志、统计）
//! - 决定**要不要**真的调用 handler（短路）
//! - 决定**调用几次** handler（重试）
//! - 把异常转成 `ToolMessage` 回灌给模型
//!
//! # 本项目的实现
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  ErrorHandlingMiddleware::wrap_tool_call                  │
//! │                                                             │
//! │   for attempt in 1..=max_retries:                          │
//! │     result = handler(request)                             │
//! │     match result:                                          │
//! │       Ok  → 直接返回（成功不需要任何加工）                 │
//! │       Err(e) →                                            │
//! │         e.kind == Transient && 还有次数 → 退避后重试      │
//! │         e.kind == Transient && 次数耗尽 → 降级成观察       │
//! │         e.kind == Business            → 降级成观察         │
//! │         e.kind == Fatal               → 向上传播（终止）  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # 三个关键设计
//!
//! ## ① 「降级」的产物是一段**可诊断**的观察文本
//!
//! 不是 `"工具执行失败"`，而是
//! `[E_INVALID_ARGUMENTS] 参数不合法：...。建议：...`。
//!
//! 模型读到它能**自己纠正**；运维读到它能**定位问题**。
//!
//! ## ② 降级**不等于**吞掉错误
//!
//! 降级后的观察里带着错误码和类别，
//! 并且中间件会**记一条 `warn!` 日志** ——
//! 「降级必须打 warn」是全仓库的硬约定（见 00-logging）。
//!
//! ## ③ Fatal 不降级
//!
//! 鉴权失败时降级 = 让模型编一个「支付失败」的答案。
//! 所以 Fatal **向上传播**，由调用方决定怎么处理。
//!
//! # ⚠️ 但 Fatal 仍然要产出回执
//!
//! 16 实测踩出来的坑：每个 `tool_call` 都必须有对应的 `tool` 消息，
//! **少一条整个请求就 400**。
//!
//! 所以本项目区分两件事：
//!
//! - **中间件**返回 `Err`（Fatal 向上传播）
//! - **Agent 循环**捕获这个 `Err`，**照样生成一条 `tool` 消息**（内容是错误文本），
//!   然后**终止循环**并把错误上报
//!
//! 这样既不会「少一条回执导致 400」，也不会「把配置问题伪装成模型答案」。

use std::time::Duration;

use agent_logging::{debug, info, warn};

use crate::error::{ErrorKind, ToolError};
use crate::tool::{ToolContext, ToolFuture, ToolOutcome, ToolRegistry};

/// 一次工具调用的**请求**。
///
/// ## 对应 LangChain 的 `ToolCallRequest`
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    /// 工具名。
    pub name: String,

    /// 参数。
    pub arguments: serde_json::Value,

    /// 调用 ID（回执时用它对账）。
    ///
    /// ⚠️ 这是 16 实测踩出来的坑 ——
    /// 每个 `call_id` 都必须有回执，少一条就 400。
    pub call_id: String,
}

/// 中间件的**处置结果**。
///
/// ## 为什么不让中间件直接返回 `ToolOutcome`
///
/// 因为有两件事需要区分：
///
/// - **给模型看的**（观察文本）—— 成功和降级失败都有
/// - **给调用方看的**（错误）—— 只有 Fatal 有
///
/// 所以返回类型是「结果 + 可选的致命错误」。
#[derive(Debug, Clone)]
pub struct MiddlewareOutcome {
    /// 给模型的观察文本。
    pub observation: String,

    /// 工具是否**成功**。
    pub ok: bool,

    /// 实际执行的次数（1 = 一次就成）。
    pub attempts: usize,

    /// 最终的错误（降级成功时是 `Some`，致命时是 `Some` 且 `fatal=true`）。
    pub error: Option<ToolError>,

    /// 这个错误是不是**致命**的（没被降级）。
    pub fatal: bool,
}

impl MiddlewareOutcome {
    /// 成功的结果。
    pub fn success(observation: String, attempts: usize) -> Self {
        Self {
            observation,
            ok: true,
            attempts,
            error: None,
            fatal: false,
        }
    }

    /// 降级后的结果（错误被转成观察）。
    pub fn degraded(error: ToolError, attempts: usize) -> Self {
        Self {
            observation: error.to_string(),
            ok: false,
            attempts,
            error: Some(error),
            fatal: false,
        }
    }

    /// 致命结果（错误向上传播）。
    pub fn fatal(error: ToolError, attempts: usize) -> Self {
        Self {
            observation: error.to_string(),
            ok: false,
            attempts,
            error: Some(error),
            fatal: true,
        }
    }
}

/// 错误处理中间件。
///
/// ## 配置项
///
/// | 配置 | 默认 | 含义 |
/// |---|---|---|
/// | `max_retries` | 2 | **额外**的重试次数（总次数 = 1 + max_retries） |
/// | `base_delay` | 50ms | 指数退避的基准延迟 |
/// | `max_delay` | 2s | 退避上限 |
///
/// ## ⚠️ 只对 Transient 重试
///
/// 这是本项目最核心的一条规则：
///
/// - **Transient** → 重试（重试可能成功）
/// - **Business** → 不重试（重试**一定**浪费时间）
/// - **Fatal** → 不重试（重试是掩盖问题）
///
/// 有回归测试 `business_errors_are_never_retried` 锁住。
pub struct ErrorHandlingMiddleware {
    registry: ToolRegistry,
    max_retries: usize,
    base_delay: Duration,
    max_delay: Duration,
}

impl std::fmt::Debug for ErrorHandlingMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorHandlingMiddleware")
            .field("tools", &self.registry.names())
            .field("max_retries", &self.max_retries)
            .field("base_delay_ms", &self.base_delay.as_millis())
            .finish()
    }
}

impl ErrorHandlingMiddleware {
    /// 创建中间件。
    pub fn new(registry: ToolRegistry) -> Self {
        Self {
            registry,
            max_retries: 2,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(2),
        }
    }

    /// 设置额外重试次数。
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries.min(10);
        self
    }

    /// 设置退避基准延迟。
    pub fn with_base_delay(mut self, base_delay: Duration) -> Self {
        self.base_delay = base_delay;
        self
    }

    /// 设置退避上限。
    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
        self
    }

    /// 工具注册表。
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// 包住一次工具调用 —— **核心逻辑**。
    ///
    /// ## 决策表
    ///
    /// | 错误类别 | 还有重试次数？ | 动作 |
    /// |---|---|---|
    /// | Transient | 是 | 退避后重试 |
    /// | Transient | 否 | **降级**成观察 |
    /// | Business | — | **降级**成观察（不重试） |
    /// | Fatal | — | **向上传播**（终止） |
    ///
    /// ## 为什么 Business 不重试
    ///
    /// 参数填错了、id 不存在 —— 重试**同样的**输入，
    /// 结果**一定**还是失败。这是纯浪费，而且会推迟模型的自我纠正。
    pub async fn wrap_tool_call(&self, request: &ToolCallRequest) -> MiddlewareOutcome {
        let total_attempts = self.max_retries + 1;
        let mut attempt = 0usize;

        loop {
            attempt += 1;

            let ctx = ToolContext::with_attempt(attempt);
            let result = self
                .registry
                .call(&request.name, &request.arguments, &ctx)
                .await;

            let outcome = match result {
                Ok(outcome) => {
                    debug!(
                        tool = %request.name,
                        attempt,
                        "工具调用成功"
                    );
                    return MiddlewareOutcome::success(outcome.observation, attempt);
                }
                Err(err) => err,
            };

            // ── 决策 ────────────────────────────────────────────
            let kind = outcome.kind;

            if kind == ErrorKind::Fatal {
                // ⚠️ 致命错误：不重试、不降级，直接向上传播。
                //
                // 降级成观察会让模型以为「工具坏了」，
                // 转而编一个答案，把配置问题掩盖成幻觉。
                warn!(
                    tool = %request.name,
                    code = outcome.diagnostic_code(),
                    kind = kind.name(),
                    attempt,
                    "致命错误 —— 不重试、不降级，直接终止"
                );
                return MiddlewareOutcome::fatal(outcome, attempt);
            }

            if kind == ErrorKind::Transient && attempt < total_attempts {
                // 可重试错误 + 还有次数 → 退避后重试。
                let delay = self.backoff(attempt);
                info!(
                    tool = %request.name,
                    code = outcome.diagnostic_code(),
                    attempt,
                    next_attempt = attempt + 1,
                    delay_ms = delay.as_millis() as u64,
                    "可重试错误 —— 退避后重试"
                );
                tokio::time::sleep(delay).await;
                continue;
            }

            // ── 降级 ────────────────────────────────────────────
            //
            // 走到这里只有两种情况：
            // 1. Business —— 重试也没用，直接降级
            // 2. Transient 且重试耗尽 —— 最后一次机会用掉了，降级
            //
            // ⚠️ 降级**必须**打 warn —— 静默降级等于埋雷。
            warn!(
                tool = %request.name,
                code = outcome.diagnostic_code(),
                kind = kind.name(),
                attempt,
                reason = if kind == ErrorKind::Business {
                    "业务失败（重试无效）"
                } else {
                    "重试次数耗尽"
                },
                "错误已降级为观察喂回模型"
            );

            return MiddlewareOutcome::degraded(outcome, attempt);
        }
    }

    /// 指数退避 —— 第 n 次失败后等多久。
    ///
    /// ## 公式
    ///
    /// ```text
    /// delay = min(base_delay × 2^(n-1), max_delay)
    /// ```
    ///
    /// ⚠️ `base_delay` 为 0 时（测试用）直接返回 0，
    /// 不做 `2^n` 的幂运算 —— 避免溢出。
    pub fn backoff(&self, attempt: usize) -> Duration {
        if self.base_delay.is_zero() {
            return Duration::ZERO;
        }

        let shift = attempt.saturating_sub(1).min(16);
        let factor = 1u32 << shift;
        let delay = self.base_delay.saturating_mul(factor);

        delay.min(self.max_delay)
    }
}

/// 一个「handler」的替身 —— 用于测试中间件的决策逻辑。
///
/// ## 为什么需要它
///
/// 中间件的决策（重试几次、降不降级）必须能**确定性**测试。
/// 真实工具的失败次数是随机的（网络抖动），
/// 所以这里抽一个 trait，让测试能精确控制「第几次失败」。
pub trait ToolHandler: Send + Sync {
    /// 执行一次调用。
    fn handle<'a>(&'a self, request: &'a ToolCallRequest) -> ToolFuture<'a>;
}

/// 直接走注册表的 handler。
pub struct RegistryHandler {
    registry: ToolRegistry,
}

impl std::fmt::Debug for RegistryHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryHandler")
            .field("tools", &self.registry.names())
            .finish()
    }
}

impl RegistryHandler {
    /// 创建 handler。
    pub fn new(registry: ToolRegistry) -> Self {
        Self { registry }
    }
}

impl ToolHandler for RegistryHandler {
    fn handle<'a>(&'a self, request: &'a ToolCallRequest) -> ToolFuture<'a> {
        Box::pin(async move {
            self.registry
                .call(
                    &request.name,
                    &request.arguments,
                    &ToolContext::with_attempt(1),
                )
                .await
        })
    }
}

/// 让 `ToolOutcome` 的导入不被判定为未使用（它在类型签名里出现）。
#[allow(dead_code)]
fn _outcome_marker(outcome: ToolOutcome) -> ToolOutcome {
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{
        FLAKY_FETCH_TOOL_NAME, FlakyTool, GUARDED_PAY_TOOL_NAME, GuardedTool,
        STRICT_LOOKUP_TOOL_NAME, StrictTool, demo_registry,
    };
    use serde_json::json;
    use std::sync::Arc;

    fn request(name: &str, args: serde_json::Value) -> ToolCallRequest {
        ToolCallRequest {
            name: name.to_string(),
            arguments: args,
            call_id: "c1".to_string(),
        }
    }

    #[tokio::test]
    async fn transient_errors_are_retried_until_success() {
        // FlakyTool 前 2 次失败 —— 正好用完 max_retries=2。
        let tool = Arc::new(FlakyTool::new(2));
        let registry = ToolRegistry::new(vec![tool.clone()]);
        let middleware = ErrorHandlingMiddleware::new(registry).with_base_delay(Duration::ZERO);

        let outcome = middleware
            .wrap_tool_call(&request(FLAKY_FETCH_TOOL_NAME, json!({ "resource": "r" })))
            .await;

        assert!(outcome.ok);
        assert_eq!(outcome.attempts, 3);
        assert_eq!(tool.call_count(), 3);
    }

    #[tokio::test]
    async fn transient_errors_are_degraded_after_retries_exhausted() {
        // 一直失败 → 重试耗尽 → 降级。
        let tool = Arc::new(FlakyTool::new(99));
        let registry = ToolRegistry::new(vec![tool.clone()]);
        let middleware = ErrorHandlingMiddleware::new(registry).with_base_delay(Duration::ZERO);

        let outcome = middleware
            .wrap_tool_call(&request(FLAKY_FETCH_TOOL_NAME, json!({ "resource": "r" })))
            .await;

        assert!(!outcome.ok);
        assert!(!outcome.fatal);
        assert_eq!(outcome.attempts, 3);
        // 降级后的观察必须**带错误码** —— 这才是「可诊断」。
        assert!(outcome.observation.contains("E_UPSTREAM_UNAVAILABLE"));
        assert!(outcome.error.is_some());
    }

    #[tokio::test]
    async fn business_errors_are_never_retried() {
        // ⚠️ 核心断言：Business 错误**一次都不重试**。
        let tool = Arc::new(StrictTool::new(vec!["rec-1".to_string()]));
        let registry = ToolRegistry::new(vec![tool.clone()]);
        let middleware = ErrorHandlingMiddleware::new(registry).with_base_delay(Duration::ZERO);

        let outcome = middleware
            .wrap_tool_call(&request(STRICT_LOOKUP_TOOL_NAME, json!({ "id": "nope" })))
            .await;

        assert!(!outcome.ok);
        assert!(!outcome.fatal);
        // ⚠️ 只调了 1 次 —— 没有重试。
        assert_eq!(outcome.attempts, 1);
        assert!(outcome.observation.contains("E_NOT_FOUND"));
        // 观察里要有可执行的建议。
        assert!(outcome.observation.contains("建议"));
        assert!(outcome.observation.contains("rec-1"));
    }

    #[tokio::test]
    async fn fatal_errors_propagate_without_retry_or_degradation() {
        // ⚠️ 致命错误：不重试、不降级。
        let tool = Arc::new(GuardedTool::new(None));
        let registry = ToolRegistry::new(vec![tool.clone()]);
        let middleware = ErrorHandlingMiddleware::new(registry).with_base_delay(Duration::ZERO);

        let outcome = middleware
            .wrap_tool_call(&request(GUARDED_PAY_TOOL_NAME, json!({ "amount": 10 })))
            .await;

        assert!(!outcome.ok);
        assert!(outcome.fatal);
        assert_eq!(outcome.attempts, 1);
        assert_eq!(
            outcome.error.as_ref().unwrap().code,
            crate::error::ErrorCode::Unauthorized
        );
    }

    #[tokio::test]
    async fn schema_violations_are_business_errors() {
        let registry = demo_registry();
        let middleware = ErrorHandlingMiddleware::new(registry).with_base_delay(Duration::ZERO);

        // `resource` 必须是字符串。
        let outcome = middleware
            .wrap_tool_call(&request(FLAKY_FETCH_TOOL_NAME, json!({ "resource": 42 })))
            .await;

        assert!(!outcome.ok);
        assert!(!outcome.fatal);
        assert_eq!(outcome.attempts, 1);
        assert!(outcome.observation.contains("E_INVALID_ARGUMENTS"));
    }

    #[tokio::test]
    async fn unknown_tools_are_business_errors() {
        let registry = demo_registry();
        let middleware = ErrorHandlingMiddleware::new(registry).with_base_delay(Duration::ZERO);

        let outcome = middleware.wrap_tool_call(&request("nope", json!({}))).await;

        assert!(!outcome.ok);
        assert!(!outcome.fatal);
        assert_eq!(outcome.attempts, 1);
        assert!(outcome.observation.contains("E_UNKNOWN_TOOL"));
    }

    #[test]
    fn backoff_grows_exponentially_and_is_capped() {
        let middleware = ErrorHandlingMiddleware::new(demo_registry())
            .with_base_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_millis(500));

        assert_eq!(middleware.backoff(1), Duration::from_millis(100));
        assert_eq!(middleware.backoff(2), Duration::from_millis(200));
        assert_eq!(middleware.backoff(3), Duration::from_millis(400));
        // 第 4 次：800 > 500 → 被截断。
        assert_eq!(middleware.backoff(4), Duration::from_millis(500));
        // 更大的 attempt 也不会溢出。
        assert_eq!(middleware.backoff(100), Duration::from_millis(500));
    }

    #[test]
    fn zero_base_delay_means_no_waiting() {
        let middleware =
            ErrorHandlingMiddleware::new(demo_registry()).with_base_delay(Duration::ZERO);
        assert_eq!(middleware.backoff(1), Duration::ZERO);
        assert_eq!(middleware.backoff(50), Duration::ZERO);
    }

    #[test]
    fn max_retries_is_capped() {
        let middleware = ErrorHandlingMiddleware::new(demo_registry()).with_max_retries(999);
        assert_eq!(middleware.max_retries, 10);
    }
}
