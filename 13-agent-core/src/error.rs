//! 统一错误类型。
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`AgentError::BadRequest`] | 400 | 请求参数不合法（任务为空、模式名非法） |
//! | [`AgentError::Llm`] | 502 | 模型调用失败（网络、鉴权、上游错误） |
//! | [`AgentError::Tool`] | 500 | 工具执行失败（参数非法、底层 IO 错误） |
//! | [`AgentError::Parse`] | 502 | 模型输出解析失败（格式不对、缺字段） |
//! | [`AgentError::Internal`] | 500 | 内部错误（IO、序列化等） |
//!
//! ## 降级语义
//!
//! 这个项目里只有**工具执行失败**是可降级的 —— 因为工具失败时，
//! 错误信息本身可以当作一条 `Observation` 喂回模型，模型能读到并自我纠正
//! （这正是 `06-rag` 里 `ToolError` 的设计思路）。
//!
//! 而模型调用失败、解析失败都是**硬失败**：前者没有推理能力，
//! 后者说明模型的输出契约被破坏了，继续跑只会产出垃圾。

use agent_logging::{error, warn};
use thiserror::Error;

/// Agent 系统的错误。
#[derive(Debug, Error)]
pub enum AgentError {
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

    /// 内部错误。
    #[error("内部错误: {0}")]
    Internal(String),
}

impl AgentError {
    /// 对应的 HTTP 状态码。
    pub fn status_code(&self) -> u16 {
        match self {
            AgentError::BadRequest(_) => 400,
            AgentError::Llm(_) => 502,
            AgentError::Tool(_) => 500,
            AgentError::Parse(_) => 502,
            AgentError::Internal(_) => 500,
        }
    }

    /// 这个错误**能不能降级**。
    ///
    /// - 工具失败 → **能降级**（把错误当观察喂回模型，让它换参数重试）
    /// - 模型失败 → 不能降级（没有模型就没有推理）
    /// - 解析失败 → 不能降级（契约被破坏，继续跑只会产出垃圾）
    pub fn is_degradable(&self) -> bool {
        matches!(self, AgentError::Tool(_))
    }
}

/// 本项目的 `Result` 别名。
pub type AgentResult<T> = Result<T, AgentError>;

impl From<std::io::Error> for AgentError {
    fn from(e: std::io::Error) -> Self {
        AgentError::Internal(e.to_string())
    }
}

impl From<serde_json::Error> for AgentError {
    fn from(e: serde_json::Error) -> Self {
        AgentError::Internal(e.to_string())
    }
}

/// 把 07-query 的 `QueryError` 转成本项目的错误。
///
/// ## 为什么需要这个转换
///
/// 复用 07 的 `Llm` trait 意味着 LLM 层返回的是 `QueryError`。
/// 而 Agent 层的错误语义更丰富（多了 `Parse` 和 `Tool`）——
/// 所以要在边界处做一次**语义映射**，而不是直接 `Internal`。
///
/// 映射规则：
///
/// - `QueryError::Transformation` / `Generation` → [`AgentError::Llm`]
/// - `QueryError::BadRequest` → [`AgentError::BadRequest`]
/// - 其余 → [`AgentError::Internal`]
impl From<rag_07::error::QueryError> for AgentError {
    fn from(e: rag_07::error::QueryError) -> Self {
        match e {
            rag_07::error::QueryError::BadRequest(msg) => AgentError::BadRequest(msg),
            rag_07::error::QueryError::Transformation(msg)
            | rag_07::error::QueryError::Generation(msg) => AgentError::Llm(msg),
            other => AgentError::Internal(other.to_string()),
        }
    }
}

/// 让 `AgentError` 本身就能作为 axum 的错误响应。
///
/// 实现它之后，「请求体解析失败」和「业务校验失败」会走到
/// **完全相同**的响应格式上（见 [`crate::api::extract::JsonBody`]）。
impl axum::response::IntoResponse for AgentError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        match &self {
            AgentError::BadRequest(_) => {
                warn!(status = status.as_u16(), error = %self, "请求参数不合法");
            }
            AgentError::Llm(_) | AgentError::Parse(_) => {
                error!(status = status.as_u16(), error = %self, "模型链路失败");
            }
            AgentError::Tool(_) | AgentError::Internal(_) => {
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
        assert_eq!(AgentError::BadRequest("x".into()).status_code(), 400);
        assert_eq!(AgentError::Llm("x".into()).status_code(), 502);
        assert_eq!(AgentError::Tool("x".into()).status_code(), 500);
        assert_eq!(AgentError::Parse("x".into()).status_code(), 502);
        assert_eq!(AgentError::Internal("x".into()).status_code(), 500);
    }

    #[test]
    fn only_tool_error_is_degradable() {
        assert!(AgentError::Tool("x".into()).is_degradable());
        assert!(!AgentError::Llm("x".into()).is_degradable());
        assert!(!AgentError::Parse("x".into()).is_degradable());
    }
}
