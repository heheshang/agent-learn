//! 统一错误类型。
//!
//! ## 与 18 的关系
//!
//! **工具层**的错误直接复用 18 的 [`ErrorKind`] / [`ErrorCode`] /
//! [`ToolError`] —— 因为本项目直接复用 18 的
//! [`ErrorHandlingMiddleware`]，中间件的「重试 / 降级 / 终止」决策
//! 直接吃这几个类型。自己再定义一套只会让两边的重试判定对不上。
//!
//! 但**应用层**的错误自己定义 —— 因为本项目多了一种 18 没有的错误：
//! **超时**（`AppError::Timeout`）。
//!
//! ## ⚠️ 为什么超时必须是 504 而不是 500
//!
//! 超时的语义是「**上游在约定时间内没返回**」。
//! - 502 = 上游返回了错的东西（坏网关）
//! - 503 = 上游暂时不可用（稍后重试可能成功）
//! - **504 = 上游没在时限内响应（网关超时）**
//!
//! 调用方看到 504 才知道「这是时间问题，不是服务坏了」。

use agent_logging::{error, warn};
use thiserror::Error;

/// 复用 18 的工具错误体系（Transient / Business / Fatal 三分类）。
pub use agent_18::error::{ErrorCode, ErrorKind, ToolError};

/// 应用层错误（HTTP 边界用）。
#[derive(Debug, Error)]
pub enum AppError {
    /// 请求参数不合法。
    #[error("请求参数不合法: {0}")]
    BadRequest(String),

    /// 超时（上游没在时限内响应）。
    ///
    /// ⚠️ 对应 HTTP **504**（Gateway Timeout）。
    #[error("请求超时: {0}")]
    Timeout(String),

    /// 模型调用失败（模型挂了 / 协议被破坏）。
    ///
    /// 对应 HTTP **502** —— 「没有模型就没有答案」，但这是**上游**的问题，
    /// 不是本服务内部的问题。
    #[error("模型调用失败: {0}")]
    Llm(String),

    /// 工具执行失败（且**不可**降级）。
    #[error("工具执行失败: {0}")]
    Tool(String),

    /// 上游服务暂时不可用。
    ///
    /// 对应 HTTP **503**。
    #[error("上游服务暂时不可用: {0}")]
    Upstream(String),

    /// 内部错误。
    #[error("内部错误: {0}")]
    Internal(String),
}

impl AppError {
    /// 对应的 HTTP 状态码。
    pub fn status_code(&self) -> u16 {
        match self {
            AppError::BadRequest(_) => 400,
            AppError::Timeout(_) => 504,
            AppError::Llm(_) => 502,
            AppError::Tool(_) => 500,
            AppError::Upstream(_) => 503,
            AppError::Internal(_) => 500,
        }
    }

    /// 工具错误 → 应用错误。
    ///
    /// | 类别 | 状态码 | 理由 |
    /// |---|---|---|
    /// | Business | 400 | 是**调用方**的参数错了 |
    /// | Transient | 503 | 是**上游**暂时不可用，值得重试 |
    /// | Fatal | 500 | 服务端自己的问题 |
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

impl From<agent_18::error::AppError> for AppError {
    fn from(e: agent_18::error::AppError) -> Self {
        match e {
            agent_18::error::AppError::BadRequest(msg) => AppError::BadRequest(msg),
            agent_18::error::AppError::Llm(msg) => AppError::Llm(msg),
            agent_18::error::AppError::Upstream(msg) => AppError::Upstream(msg),
            agent_18::error::AppError::Tool(msg) => AppError::Tool(msg),
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
            AppError::Timeout(_) => {
                agent_logging::warn!(status = status.as_u16(), error = %self, "请求超时");
            }
            AppError::Upstream(_) | AppError::Llm(_) => {
                agent_logging::error!(status = status.as_u16(), error = %self, "上游调用失败");
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

/// 让 `warn` / `error` 的导入不被判定为未使用（它们在宏里被引用）。
#[allow(dead_code)]
fn _marker(err: &AppError) {
    warn!("{}", err);
    error!("{}", err);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_errors_to_http_codes() {
        assert_eq!(AppError::BadRequest("x".into()).status_code(), 400);
        // ⚠️ 超时必须是 504，不是 500 ——
        //    「上游没按时回」和「服务端坏了」是两回事。
        assert_eq!(AppError::Timeout("x".into()).status_code(), 504);
        assert_eq!(AppError::Llm("x".into()).status_code(), 502);
        assert_eq!(AppError::Upstream("x".into()).status_code(), 503);
        assert_eq!(AppError::Tool("x".into()).status_code(), 500);
        assert_eq!(AppError::Internal("x".into()).status_code(), 500);
    }

    #[test]
    fn tool_error_maps_by_kind() {
        let business = ToolError::invalid_args("t", "参数不对");
        assert_eq!(AppError::from_tool_error(&business).status_code(), 400);

        let transient = ToolError::timeout("t", "超时");
        assert_eq!(AppError::from_tool_error(&transient).status_code(), 503);

        let fatal = ToolError::unauthorized("t", "没配 key");
        assert_eq!(AppError::from_tool_error(&fatal).status_code(), 500);
    }

    #[test]
    fn converts_agent_16_errors() {
        let e: AppError = agent_16::error::AppError::Llm("挂了".into()).into();
        assert_eq!(e.status_code(), 502);
    }
}
