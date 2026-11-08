//! 统一错误类型。
//!
//! 错误分四类，映射到 HTTP 状态码：
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`RagError::BadRequest`] | 400 | 请求参数不合法（空问题、top_k 超范围） |
//! | [`RagError::NotReady`] | 503 | 索引还没建好 |
//! | [`RagError::Retrieval`] | 500 | 检索失败（索引为空、维度不匹配） |
//! | [`RagError::Generation`] | 502 | LLM 调用失败（网络、鉴权、模型报错） |
//!
//! 分开的意义：调用方能区分「我请求写错了」（改请求）和
//! 「你服务坏了」（重试/报警），而不用去猜错误消息里的中文。

use agent_logging::{error, warn};
use thiserror::Error;

/// RAG 系统的错误。
#[derive(Debug, Error)]
pub enum RagError {
    /// 请求参数不合法。
    #[error("请求参数不合法: {0}")]
    BadRequest(String),

    /// 索引还没建好 —— 服务已启动但还没有知识库。
    #[error("索引未就绪: {0}")]
    NotReady(String),

    /// 检索失败。
    #[error("检索失败: {0}")]
    Retrieval(String),

    /// 生成失败（LLM 调用出错）。
    #[error("生成失败: {0}")]
    Generation(String),

    /// 内部错误（IO、解析等）。
    #[error("内部错误: {0}")]
    Internal(String),
}

impl RagError {
    /// 对应的 HTTP 状态码。
    pub fn status_code(&self) -> u16 {
        match self {
            RagError::BadRequest(_) => 400,
            RagError::NotReady(_) => 503,
            RagError::Retrieval(_) => 500,
            RagError::Generation(_) => 502,
            RagError::Internal(_) => 500,
        }
    }
}

/// 本项目的 `Result` 别名。
pub type RagResult<T> = Result<T, RagError>;

impl From<std::io::Error> for RagError {
    fn from(e: std::io::Error) -> Self {
        RagError::Internal(e.to_string())
    }
}

impl From<serde_json::Error> for RagError {
    fn from(e: serde_json::Error) -> Self {
        RagError::Internal(e.to_string())
    }
}

/// 让 `RagError` 本身就能作为 axum 的错误响应。
///
/// 实现它之后，`RagError` 就可以直接当作 extractor 的 `Rejection` 类型 ——
/// 见 [`crate::api::extract::JsonBody`]。这样「请求体解析失败」和
/// 「业务校验失败」会走到**完全相同**的响应格式上。
impl axum::response::IntoResponse for RagError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // 错误分级：4xx 是调用方的问题，5xx 才是服务端的问题。
        // 混在一个级别里，告警规则没法区分「用户传错了」和「服务坏了」。
        match &self {
            RagError::BadRequest(_) => {
                warn!(status = status.as_u16(), error = %self, "请求参数不合法");
            }
            RagError::NotReady(_) => {
                warn!(status = status.as_u16(), error = %self, "索引未就绪");
            }
            _ => {
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
        assert_eq!(RagError::BadRequest("x".into()).status_code(), 400);
        assert_eq!(RagError::NotReady("x".into()).status_code(), 503);
        assert_eq!(RagError::Retrieval("x".into()).status_code(), 500);
        assert_eq!(RagError::Generation("x".into()).status_code(), 502);
        assert_eq!(RagError::Internal("x".into()).status_code(), 500);
    }
}
