//! 统一错误类型。
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`AppError::BadRequest`] | 400 | 请求参数不合法（问题为空、TTL 非法等） |
//! | [`AppError::Llm`] | 502 | 模型调用失败 |
//! | [`AppError::Tool`] | 500 | 工具执行失败 |
//! | [`AppError::Upstream`] | 502 | 外部服务失败（天气 API） |
//! | [`AppError::Cache`] | 503 | **缓存操作失败且不允许降级** |
//! | [`AppError::Internal`] | 500 | 内部错误 |
//!
//! ## ⚠️ 缓存错误为什么单独一类
//!
//! 缓存**通常**不会产生 HTTP 错误 —— 它降级（读失败当 miss）。
//!
//! 但有一种情况例外：`CACHE_REDIS_REQUIRED=1` 且 Redis 连不上，
//! 此时**启动就失败**（不提供服务），而不是降级。
//!
//! 运行时如果缓存操作失败并要求严格模式，返回 **503** 而不是 500 ——
//! 503 的标准语义是「稍后重试可能成功」，而缓存不可用正是这种情况。

use agent_logging::{error, warn};
use thiserror::Error;

pub use agent_16::error::ToolError;

/// 应用层错误。
#[derive(Debug, Error)]
pub enum AppError {
    /// 请求参数不合法。
    #[error("请求参数不合法: {0}")]
    BadRequest(String),

    /// 模型调用失败。
    #[error("模型调用失败: {0}")]
    Llm(String),

    /// 工具执行失败。
    #[error("工具执行失败: {0}")]
    Tool(String),

    /// 外部服务失败。
    #[error("外部服务失败: {0}")]
    Upstream(String),

    /// 缓存操作失败。
    #[error("缓存操作失败: {0}")]
    Cache(String),

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
            AppError::Tool(_) => 500,
            AppError::Upstream(_) => 502,
            AppError::Cache(_) => 503,
            AppError::Internal(_) => 500,
        }
    }

    /// 这个错误**能不能降级**。
    ///
    /// ## 只有工具错误能降级
    ///
    /// 缓存错误**不降级**（它已经在缓存层内部降级过了，
    /// 能冒到这里说明是「严格模式」下的显式失败）。
    pub fn is_degradable(&self) -> bool {
        matches!(self, AppError::Tool(_))
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

impl From<ToolError> for AppError {
    fn from(e: ToolError) -> Self {
        match e {
            ToolError::InvalidArgs { .. } | ToolError::UnknownTool { .. } => {
                AppError::BadRequest(e.to_string())
            }
            ToolError::Upstream { message } => AppError::Upstream(message),
            ToolError::Execution { .. } => AppError::Tool(e.to_string()),
        }
    }
}

impl From<agent_13::error::AgentError> for AppError {
    fn from(e: agent_13::error::AgentError) -> Self {
        match e {
            agent_13::error::AgentError::BadRequest(msg) => AppError::BadRequest(msg),
            agent_13::error::AgentError::Tool(msg) => AppError::Tool(msg),
            agent_13::error::AgentError::Llm(msg) => AppError::Llm(msg),
            other => AppError::Internal(other.to_string()),
        }
    }
}

/// 从 16 的 [`agent_16::AppError`] 转换。
///
/// ## 为什么需要这个 `From`
///
/// 本项目复用 16 的 `FunctionLlm` trait，而它的错误类型是 `agent_16::AppError`。
/// 如果不转换，Agent 循环里每一个 `?` 都要写 `.map_err(...)` —— 噪音。
impl From<agent_16::AppError> for AppError {
    fn from(e: agent_16::AppError) -> Self {
        match e {
            agent_16::AppError::BadRequest(msg) => AppError::BadRequest(msg),
            agent_16::AppError::Llm(msg) => AppError::Llm(msg),
            agent_16::AppError::Tool(msg) => AppError::Tool(msg),
            agent_16::AppError::Parse(msg) => AppError::Internal(msg),
            agent_16::AppError::Upstream(msg) => AppError::Upstream(msg),
            agent_16::AppError::Internal(msg) => AppError::Internal(msg),
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
                warn!(status = status.as_u16(), error = %self, "请求参数不合法");
            }
            AppError::Cache(_) => {
                warn!(status = status.as_u16(), error = %self, "缓存操作失败");
            }
            AppError::Upstream(_) | AppError::Llm(_) => {
                error!(status = status.as_u16(), error = %self, "上游调用失败");
            }
            AppError::Tool(_) | AppError::Internal(_) => {
                error!(status = status.as_u16(), error = %self, "请求处理失败");
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
    fn maps_errors_to_http_codes() {
        assert_eq!(AppError::BadRequest("x".into()).status_code(), 400);
        assert_eq!(AppError::Llm("x".into()).status_code(), 502);
        assert_eq!(AppError::Tool("x".into()).status_code(), 500);
        assert_eq!(AppError::Upstream("x".into()).status_code(), 502);
        // ⚠️ 缓存不可用 → 503（稍后重试可能成功），不是 500。
        assert_eq!(AppError::Cache("x".into()).status_code(), 503);
        assert_eq!(AppError::Internal("x".into()).status_code(), 500);
    }

    #[test]
    fn only_tool_error_is_degradable() {
        assert!(AppError::Tool("x".into()).is_degradable());
        assert!(!AppError::Llm("x".into()).is_degradable());
        assert!(!AppError::Cache("x".into()).is_degradable());
    }

    #[test]
    fn argument_errors_map_to_bad_request() {
        let err: AppError = ToolError::invalid_args(vec!["x".to_string()]).into();
        assert_eq!(err.status_code(), 400);
    }
}
