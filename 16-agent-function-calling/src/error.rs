//! 统一错误类型。
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`AppError::BadRequest`] | 400 | 请求参数不合法（问题为空、工具名非法等） |
//! | [`AppError::Llm`] | 502 | 模型调用失败（网络、鉴权、上游错误） |
//! | [`AppError::Tool`] | 500 | 工具执行失败（未被降级吸收时） |
//! | [`AppError::Parse`] | 502 | 模型输出解析失败 |
//! | [`AppError::Upstream`] | 502 | 外部服务失败（天气 API） |
//! | [`AppError::Internal`] | 500 | 内部错误（IO、序列化等） |
//!
//! ## 与 14 的 `ToolError` 的关系
//!
//! 本项目**直接复用** `agent_14::error::ToolError` ——
//! 它的四种错误（未知工具 / 参数不合法 / 执行失败 / 上游失败）
//! 已经把「错误信息必须让模型能自我纠正」这件事做对了。
//!
//! 复用意味着：14 修过的坑这里自动继承，不用再踩一遍。

use agent_logging::{error, warn};
use thiserror::Error;

pub use agent_14::error::ToolError;

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

    /// 模型输出解析失败。
    #[error("模型输出解析失败: {0}")]
    Parse(String),

    /// 外部服务失败。
    #[error("外部服务失败: {0}")]
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
            AppError::Llm(_) => 502,
            AppError::Tool(_) => 500,
            AppError::Parse(_) => 502,
            AppError::Upstream(_) => 502,
            AppError::Internal(_) => 500,
        }
    }

    /// 这个错误**能不能降级**。
    ///
    /// ## 只有工具错误能降级
    ///
    /// 工具失败时，错误信息本身就是有用的 `tool` 消息 ——
    /// 模型能读到「没有这一列：xxx」然后换个参数重试。
    ///
    /// 模型失败 / 解析失败不能降级：前者没有推理能力，
    /// 后者说明输出契约被破坏了，继续跑只会产出垃圾。
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
            // 参数不合法 / 未知工具 —— 是**调用方**的问题，不是服务端的。
            //
            // ⚠️ 在 Agent 循环里它们会被降级成 tool 消息；
            //    但在「用户直接调工具」的端点里，用户扮演的是模型角色，
            //    所以应该得到 400 而不是 500 ——
            //    500 会让调用方以为「服务端坏了，重试也没用」。
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
            AppError::Upstream(_) | AppError::Llm(_) | AppError::Parse(_) => {
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
        assert_eq!(AppError::Parse("x".into()).status_code(), 502);
        assert_eq!(AppError::Upstream("x".into()).status_code(), 502);
        assert_eq!(AppError::Internal("x".into()).status_code(), 500);
    }

    #[test]
    fn only_tool_error_is_degradable() {
        assert!(AppError::Tool("x".into()).is_degradable());
        assert!(!AppError::Llm("x".into()).is_degradable());
    }

    #[test]
    fn argument_errors_map_to_bad_request() {
        // ⚠️ 参数填错和未知工具是**调用方**的问题 → 400。
        let err: AppError = ToolError::invalid_args(vec!["x".to_string()]).into();
        assert_eq!(err.status_code(), 400);

        let err: AppError = ToolError::unknown("foo", "bar").into();
        assert_eq!(err.status_code(), 400);
    }

    #[test]
    fn upstream_errors_map_to_502() {
        let err: AppError = ToolError::upstream("连不上").into();
        assert_eq!(err.status_code(), 502);
    }
}
