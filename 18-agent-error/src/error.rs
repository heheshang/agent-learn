//! 错误分类 —— 本项目的地基。
//!
//! # 为什么要自己写一套分类
//!
//! LangChain 的教程里，`wrap_tool_call` 中间件把工具异常转成 `ToolMessage` 回灌给模型。
//! 但教程只说了「**怎么把错误变成观察**」，没有回答一个更关键的问题：
//!
//! > **哪些错误该重试，哪些错误重试也没用？**
//!
//! 如果对所有错误一律重试 3 次：
//!
//! - 「参数填错了」这种**业务失败**会白白重试 3 次 —— 参数不会因为重试变对
//! - 「上游 503」这种**可重试错误**只试 1 次就放弃 —— 白白浪费了恢复机会
//!
//! 所以本项目把错误分成三类，并给出**确定性的处置策略**：
//!
//! | 类别 | 含义 | 重试？ | 降级成观察？ | 例子 |
//! |---|---|---|---|---|
//! | [`ErrorKind::Transient`] | **可重试**：外部条件临时不满足 | ✅ 是（指数退避） | ✅ 是 | 上游超时 / 429 限流 / 5xx |
//! | [`ErrorKind::Business`] | **业务失败**：输入或业务规则不满足 | ❌ 否 | ✅ 是 | 参数非法 / 未知工具 / 查无此记录 |
//! | [`ErrorKind::Fatal`] | **致命错误**：环境配置或权限问题 | ❌ 否 | ❌ 否（终止） | 鉴权失败 / 协议被破坏 |
//!
//! ## 三个关键判断
//!
//! **① 「重试」和「降级」是两个正交的维度。**
//!
//! - 重试 = 再调一次**同样的**工具
//! - 降级 = 把错误变成**观察**喂回模型，让模型换条路
//!
//! 一个 Transient 错误两者都该做（先重试，重试耗尽再降级）；
//! 一个 Business 错误只该降级（模型改参数才是出路）；
//! 一个 Fatal 错误两者都不该做（重试是浪费，降级会掩盖配置问题）。
//!
//! **② Fatal 错误不能降级。**
//!
//! 因为降级后的观察是**给模型看的** —— 模型会以为「这个工具坏了」，
//! 于是转而编一个答案。鉴权失败时正确的行为是**明确告诉调用方**。
//!
//! **③ 无论哪一类，最终都必须产出一条工具回执。**
//!
//! 这是 16 实测踩出来的坑：`tool_calls` 里每个 `call_id` 都必须有对应的
//! `tool` 消息，**少一条整个请求就 400**。所以 Fatal 错误虽然要终止，
//! 但终止前仍然要为它生成回执。
//!
//! ## 错误码（稳定契约）
//!
//! [`ErrorCode`] 是**给人和模型看的稳定标识**，
//! 不要用错误文案做判断 —— 文案会变，码不会。

use std::fmt;

use serde::{Deserialize, Serialize};

/// 错误的**处置类别**。
///
/// ## 三个类别的判定依据
///
/// | | Transient | Business | Fatal |
/// |---|---|---|---|
/// | 成因 | 外部服务的**临时**状态 | **调用方**的输入 / 业务规则 | **环境配置** / 权限 |
/// | 同样的输入重试会变好吗 | **会**（等服务恢复） | **不会** | **不会** |
/// | 谁能修 | 等一等 | 改参数 | 改配置 |
/// | 该不该重试 | 该 | 不该 | 不该 |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// **可重试错误** —— 外部条件的临时问题。
    ///
    /// 重试（配合指数退避）**可能**成功。
    Transient,

    /// **业务失败** —— 请求本身不合法或业务规则不满足。
    ///
    /// 重试**不可能**成功，但**可以**把错误降级成观察，
    /// 让模型自己改正参数后重来一次。
    Business,

    /// **致命错误** —— 环境或权限问题，任何自动重试都是徒劳。
    ///
    /// 不能降级：必须终止并让调用方知道。
    Fatal,
}

impl ErrorKind {
    /// 类别名（用于日志与诊断文本）。
    pub fn name(&self) -> &'static str {
        match self {
            ErrorKind::Transient => "transient",
            ErrorKind::Business => "business",
            ErrorKind::Fatal => "fatal",
        }
    }

    /// 该不该**重试**。
    ///
    /// ⚠️ 只有 [`ErrorKind::Transient`] 为 `true`。
    ///
    /// Business 重试是**纯浪费**（参数不会因为重试变对）；
    /// Fatal 重试是**掩盖问题**（配置错了不重试就永远错）。
    pub fn retryable(&self) -> bool {
        matches!(self, ErrorKind::Transient)
    }

    /// 该不该**降级成观察**喂回模型。
    ///
    /// Transient 和 Business 都可以降级：
    /// 前者模型可以「换个工具/稍后再问」，后者模型可以「改参数再试」。
    ///
    /// Fatal **不可以** —— 降级等于让模型以为「工具坏了」，
    /// 它会转而编造答案，把配置问题掩盖成幻觉。
    pub fn degradable(&self) -> bool {
        matches!(self, ErrorKind::Transient | ErrorKind::Business)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// 稳定的错误码。
///
/// ## 为什么要有码
///
/// 错误文案会变（「超时」→「请求超时，请稍后重试」），
/// 但**分类决策**必须稳定 —— 所以判断依据是码，不是文案。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    // ── Transient ────────────────────────────────────────────
    /// 上游服务超时。
    UpstreamTimeout,
    /// 上游服务不可用（连接失败 / 5xx）。
    UpstreamUnavailable,
    /// 被限流（429）。
    RateLimited,

    // ── Business ─────────────────────────────────────────────
    /// 参数不合法（Schema 校验失败）。
    InvalidArguments,
    /// 调用了不存在的工具。
    UnknownTool,
    /// 目标资源不存在（查无此记录）。
    NotFound,
    /// 业务规则不满足（如金额超出限额）。
    RuleViolation,

    // ── Fatal ────────────────────────────────────────────────
    /// 鉴权失败（API key 无效 / 无权限）。
    Unauthorized,
    /// 内部缺陷（不变量被破坏）。
    InternalFault,
}

impl ErrorCode {
    /// 默认的错误类别。
    ///
    /// ⚠️ 这是**默认**映射 —— 某些具体场景可以覆盖
    /// （见 [`ToolError::with_kind`]）。
    ///
    /// 比如「本地数据库文件被锁」默认是 Transient（重试能好），
    /// 但「配置文件缺失」是 Fatal。
    pub fn default_kind(&self) -> ErrorKind {
        match self {
            ErrorCode::UpstreamTimeout
            | ErrorCode::UpstreamUnavailable
            | ErrorCode::RateLimited => ErrorKind::Transient,

            ErrorCode::InvalidArguments
            | ErrorCode::UnknownTool
            | ErrorCode::NotFound
            | ErrorCode::RuleViolation => ErrorKind::Business,

            ErrorCode::Unauthorized | ErrorCode::InternalFault => ErrorKind::Fatal,
        }
    }

    /// 错误码的稳定标识（`E_UPSTREAM_TIMEOUT` 这种）。
    pub fn id(&self) -> &'static str {
        match self {
            ErrorCode::UpstreamTimeout => "E_UPSTREAM_TIMEOUT",
            ErrorCode::UpstreamUnavailable => "E_UPSTREAM_UNAVAILABLE",
            ErrorCode::RateLimited => "E_RATE_LIMITED",
            ErrorCode::InvalidArguments => "E_INVALID_ARGUMENTS",
            ErrorCode::UnknownTool => "E_UNKNOWN_TOOL",
            ErrorCode::NotFound => "E_NOT_FOUND",
            ErrorCode::RuleViolation => "E_RULE_VIOLATION",
            ErrorCode::Unauthorized => "E_UNAUTHORIZED",
            ErrorCode::InternalFault => "E_INTERNAL_FAULT",
        }
    }

    /// 人类可读的简短说明（用于诊断文本）。
    pub fn summary(&self) -> &'static str {
        match self {
            ErrorCode::UpstreamTimeout => "上游服务响应超时",
            ErrorCode::UpstreamUnavailable => "上游服务不可用",
            ErrorCode::RateLimited => "请求被限流",
            ErrorCode::InvalidArguments => "参数不合法",
            ErrorCode::UnknownTool => "工具不存在",
            ErrorCode::NotFound => "目标资源不存在",
            ErrorCode::RuleViolation => "违反业务规则",
            ErrorCode::Unauthorized => "鉴权失败",
            ErrorCode::InternalFault => "内部错误",
        }
    }

    /// 该错误码**默认**能不能重试。
    pub fn retryable(&self) -> bool {
        self.default_kind().retryable()
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// 工具层的错误 —— **可诊断**。
///
/// ## 「可诊断」是什么意思
///
/// 一个好的工具错误要让三样东西都能读懂：
///
/// 1. **人** —— 看日志就知道发生了什么、严重吗
/// 2. **模型** —— 看观察就知道下一步该干什么
/// 3. **机器** —— 能按 `kind` 决定重试 / 降级 / 终止
///
/// 所以它有三个字段各司其职：
///
/// - [`code`](ToolError::code) —— 稳定的机器可读标识
/// - [`kind`](ToolError::kind) —— 决定处置策略
/// - [`hint`](ToolError::hint) —— 给模型的**可执行**下一步
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolError {
    /// 稳定错误码。
    pub code: ErrorCode,

    /// 处置类别。
    pub kind: ErrorKind,

    /// 哪个工具出的错。
    pub tool: String,

    /// 原始原因（给人看）。
    pub message: String,

    /// 给模型的可执行建议（可选）。
    ///
    /// ## 为什么它必须是「可执行」的
    ///
    /// 「参数错误」模型无从下手；
    /// 「`location` 需要城市名（如 北京），不能是省名」模型立刻知道该改什么。
    ///
    /// ⚠️ 本项目的约定：**没有 hint 的 Business 错误视为缺陷**——
    /// 因为 Business 错误的价值就在于「模型能据此纠正」。
    #[serde(default)]
    pub hint: Option<String>,
}

impl ToolError {
    /// 构造一个错误（类别取错误码的默认值）。
    pub fn new(code: ErrorCode, tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            kind: code.default_kind(),
            tool: tool.into(),
            message: message.into(),
            hint: None,
        }
    }

    /// 覆盖处置类别。
    ///
    /// 用于「同一个错误码在不同上下文里处置不同」的场景 ——
    /// 比如「参数非法」在「用户直接调工具」时是 BadRequest，
    /// 在「Agent 循环里」是 Business（降级成观察）。
    pub fn with_kind(mut self, kind: ErrorKind) -> Self {
        self.kind = kind;
        self
    }

    /// 附加给模型的可执行建议。
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// 该不该重试。
    pub fn retryable(&self) -> bool {
        self.kind.retryable()
    }

    /// 该不该降级成观察。
    pub fn degradable(&self) -> bool {
        self.kind.degradable()
    }

    /// 诊断码（如 `E_INVALID_ARGUMENTS`）。
    pub fn diagnostic_code(&self) -> &'static str {
        self.code.id()
    }
}

impl fmt::Display for ToolError {
    /// 渲染成**给模型看的观察文本**。
    ///
    /// ## 格式
    ///
    /// ```text
    /// [E_INVALID_ARGUMENTS] 参数不合法：location 必须是字符串，实际是数字。
    /// 建议：`location` 需要城市名（如 北京），不能是省名。
    /// ```
    ///
    /// 三段式：**码 + 原因 + 建议**。
    ///
    /// ⚠️ 建议段是**可选**的 —— Transient / Fatal 通常没有建议
    /// （前者重试就行，后者要人改配置）。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}：{}",
            self.diagnostic_code(),
            self.code.summary(),
            self.message
        )?;

        if let Some(hint) = &self.hint {
            write!(f, "。建议：{hint}")?;
        }

        Ok(())
    }
}

impl std::error::Error for ToolError {}

/// 构造几个常用错误的便捷函数。
impl ToolError {
    /// 参数不合法。
    pub fn invalid_args(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArguments, tool, message)
    }

    /// 未知工具。
    pub fn unknown_tool(tool: impl Into<String>, available: impl Into<String>) -> Self {
        let tool = tool.into();
        let available = available.into();
        Self::new(
            ErrorCode::UnknownTool,
            tool.clone(),
            format!("未知工具 `{tool}`，可用工具：{available}"),
        )
        .with_hint(format!("请从可用工具中选择：{available}"))
    }

    /// 上游超时。
    pub fn timeout(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::UpstreamTimeout, tool, message)
    }

    /// 上游不可用。
    pub fn upstream(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::UpstreamUnavailable, tool, message)
    }

    /// 限流。
    pub fn rate_limited(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::RateLimited, tool, message)
    }

    /// 资源不存在。
    pub fn not_found(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, tool, message)
    }

    /// 业务规则不满足。
    pub fn rule_violation(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::RuleViolation, tool, message)
    }

    /// 鉴权失败。
    pub fn unauthorized(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Unauthorized, tool, message)
    }

    /// 内部错误。
    pub fn internal(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InternalFault, tool, message)
    }
}

/// 应用层错误（HTTP 边界用）。
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    /// 请求参数不合法。
    #[error("请求参数不合法: {0}")]
    BadRequest(String),

    /// 模型调用失败（模型本身挂了 / 协议被破坏）。
    #[error("模型调用失败: {0}")]
    Llm(String),

    /// 上游服务暂时不可用。
    ///
    /// ## ⚠️ 必须是 503 而不是 500
    ///
    /// 500 会让调用方以为「重试也没用」，
    /// 而 503 的标准语义就是「**稍后重试可能成功**」。
    ///
    /// 这正是「可重试错误」在 HTTP 层的落地。
    #[error("上游服务暂时不可用: {0}")]
    Upstream(String),

    /// 工具执行失败（且**不可**降级）。
    ///
    /// ⚠️ 只有 Fatal 类工具错误会走到这里 —— Transient / Business
    /// 都在中间件里被消化掉了（重试或降级成观察）。
    #[error("工具执行失败: {0}")]
    Tool(String),

    /// 内部错误。
    #[error("内部错误: {0}")]
    Internal(String),
}

impl AppError {
    /// 对应的 HTTP 状态码。
    pub fn status_code(&self) -> u16 {
        match self {
            AppError::BadRequest(_) => 400,
            AppError::Llm(_) => 502,
            AppError::Upstream(_) => 503,
            AppError::Tool(_) => 500,
            AppError::Internal(_) => 500,
        }
    }

    /// 工具错误 → 应用错误。
    ///
    /// ## 状态码的选择
    ///
    /// | 类别 | 状态码 | 理由 |
    /// |---|---|---|
    /// | Business | 400 | 是**调用方**的参数错了 |
    /// | Transient | 503 | 是**上游**暂时不可用，值得重试 |
    /// | Fatal | 500 | 服务端自己的问题（鉴权/缺陷） |
    ///
    /// ⚠️ Transient 必须是 503 而不是 500 ——
    /// 500 会让调用方以为「重试也没用」。
    pub fn from_tool_error(err: &ToolError) -> Self {
        match err.kind {
            ErrorKind::Business => AppError::BadRequest(err.to_string()),
            ErrorKind::Transient => AppError::Upstream(err.to_string()),
            ErrorKind::Fatal => AppError::Tool(err.to_string()),
        }
    }
}

/// 本项目的 `Result` 别名。
pub type AppResult<T> = Result<T, AppError>;

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Internal(e.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::Internal(e.to_string())
    }
}

impl From<agent_16::error::ToolError> for AppError {
    fn from(e: agent_16::error::ToolError) -> Self {
        // ⚠️ 16 的 `ToolError` 种类比本项目粗（没有 Transient/Fatal 的区分）。
        // 这里保守映射：
        // - Upstream（外部服务）→ Transient（可重试）
        // - UnknownTool / InvalidArgs → Business
        // - Execution（工具内部失败）→ Fatal
        //
        // 保守的代价：Execution 映射成 Fatal 会让「工具内部异常」不再降级。
        // 这是**故意的** —— 本项目的 ToolError 才是精细分类的权威来源，
        // 16 的错误只作为兼容入口。
        match e {
            agent_16::error::ToolError::UnknownTool { .. }
            | agent_16::error::ToolError::InvalidArgs { .. } => AppError::BadRequest(e.to_string()),
            agent_16::error::ToolError::Upstream { message } => AppError::Upstream(message),
            agent_16::error::ToolError::Execution { .. } => AppError::Tool(e.to_string()),
        }
    }
}

impl From<agent_16::error::AppError> for AppError {
    fn from(e: agent_16::error::AppError) -> Self {
        match e {
            agent_16::error::AppError::BadRequest(msg) => AppError::BadRequest(msg),
            agent_16::error::AppError::Llm(msg) => AppError::Llm(msg),
            agent_16::error::AppError::Tool(msg) => AppError::Tool(msg),
            other => AppError::Internal(other.to_string()),
        }
    }
}

/// 让 [`AppError`] 本身就能作为 axum 的错误响应。
impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        match &self {
            AppError::BadRequest(_) => {
                agent_logging::warn!(status = status.as_u16(), error = %self, "请求参数不合法");
            }
            AppError::Llm(_) => {
                agent_logging::error!(status = status.as_u16(), error = %self, "上游调用失败");
            }
            AppError::Upstream(_) => {
                agent_logging::warn!(status = status.as_u16(), error = %self, "上游暂时不可用");
            }
            AppError::Tool(_) | AppError::Internal(_) => {
                agent_logging::error!(status = status.as_u16(), error = %self, "请求处理失败");
            }
        }

        let body = crate::api::models::ErrorResponse {
            error: status.canonical_reason().unwrap_or("Error").to_string(),
            message: self.to_string(),
        };
        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_errors_are_retryable_and_degradable() {
        let e = ToolError::timeout("get_weather", "连接超时");
        assert_eq!(e.kind, ErrorKind::Transient);
        assert!(e.retryable());
        assert!(e.degradable());
    }

    #[test]
    fn business_errors_are_not_retryable_but_degradable() {
        // ⚠️ 业务失败重试是**纯浪费** —— 参数不会因为重试变对。
        let e = ToolError::invalid_args("get_weather", "location 必须是字符串");
        assert_eq!(e.kind, ErrorKind::Business);
        assert!(!e.retryable());
        assert!(e.degradable());
    }

    #[test]
    fn fatal_errors_are_neither_retryable_nor_degradable() {
        // ⚠️ Fatal **不能**降级 —— 降级会让模型以为「工具坏了」，
        //    转而编造答案，把配置问题掩盖成幻觉。
        let e = ToolError::unauthorized("get_weather", "API key 无效");
        assert_eq!(e.kind, ErrorKind::Fatal);
        assert!(!e.retryable());
        assert!(!e.degradable());
    }

    #[test]
    fn error_code_mapping_is_stable() {
        assert_eq!(
            ErrorCode::UpstreamTimeout.default_kind(),
            ErrorKind::Transient
        );
        assert_eq!(
            ErrorCode::InvalidArguments.default_kind(),
            ErrorKind::Business
        );
        assert_eq!(ErrorCode::Unauthorized.default_kind(), ErrorKind::Fatal);
        assert_eq!(ErrorCode::NotFound.default_kind(), ErrorKind::Business);
        assert_eq!(ErrorCode::RateLimited.default_kind(), ErrorKind::Transient);
    }

    #[test]
    fn kind_can_be_overridden() {
        // 「参数非法」在「用户直接调工具」时是 BadRequest，
        // 在「Agent 循环里」是 Business（降级成观察）。
        let e = ToolError::invalid_args("t", "x").with_kind(ErrorKind::Fatal);
        assert!(!e.degradable());
    }

    #[test]
    fn diagnostic_text_has_code_reason_and_hint() {
        let e = ToolError::invalid_args("get_weather", "location 必须是字符串")
            .with_hint("`location` 需要城市名（如 北京）");
        let text = e.to_string();

        assert!(text.contains("E_INVALID_ARGUMENTS"));
        assert!(text.contains("参数不合法"));
        assert!(text.contains("location 必须是字符串"));
        assert!(text.contains("建议"));
        assert!(text.contains("北京"));
    }

    #[test]
    fn diagnostic_text_without_hint_is_still_valid() {
        let e = ToolError::timeout("get_weather", "连接超时");
        let text = e.to_string();
        assert!(text.contains("E_UPSTREAM_TIMEOUT"));
        assert!(!text.contains("建议"));
    }

    #[test]
    fn tool_errors_map_to_the_right_http_codes() {
        // ⚠️ Transient 必须是 503 —— 500 会让调用方以为「重试也没用」。
        let transient: AppError = AppError::from_tool_error(&ToolError::timeout("t", "x"));
        assert_eq!(transient.status_code(), 503);

        let business: AppError = AppError::from_tool_error(&ToolError::invalid_args("t", "x"));
        assert_eq!(business.status_code(), 400);

        let fatal: AppError = AppError::from_tool_error(&ToolError::unauthorized("t", "x"));
        assert_eq!(fatal.status_code(), 500);
    }

    #[test]
    fn upstream_maps_to_503() {
        // 16 的 `ToolError::Upstream` 是外部服务失败 → 可重试 → 503。
        let err: AppError = agent_16::error::ToolError::upstream("连不上").into();
        assert_eq!(err.status_code(), 503);
    }

    #[test]
    fn unknown_tool_error_lists_available_tools() {
        let e = ToolError::unknown_tool("foo", "search, calculate");
        let text = e.to_string();
        assert!(text.contains("foo"));
        assert!(text.contains("search"));
        assert!(text.contains("calculate"));
    }
}
