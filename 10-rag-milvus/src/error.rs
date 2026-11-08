//! 统一错误类型。
//!
//! 错误分四类，映射到 HTTP 状态码：
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`RagError::BadRequest`] | 400 | 请求参数不合法 |
//! | [`RagError::NotReady`] | 503 | Milvus 没起来 / collection 没加载 |
//! | [`RagError::Storage`] | 502 | 向量库操作失败（连接、写入、检索） |
//! | [`RagError::Internal`] | 500 | 内部错误（IO、解析等） |
//!
//! ## 为什么单独一个 `Storage` 类
//!
//! 因为「Milvus 挂了」和「参数写错了」是**两种完全不同的事故**：
//!
//! - 参数写错了 → 4xx，是调用方的 bug，重试没用
//! - Milvus 挂了 → 502，是基础设施故障，**应该重试 / 降级**
//!
//! 把它们混在一个 `Internal` 里，运维就分不清「代码 bug」和「服务故障」。

use agent_logging::{error, warn};
use thiserror::Error;

/// 向量检索系统的错误。
#[derive(Debug, Error)]
pub enum RagError {
    /// 请求参数不合法。
    #[error("请求参数不合法: {0}")]
    BadRequest(String),

    /// 服务未就绪（向量库没起来 / collection 未加载）。
    #[error("服务未就绪: {0}")]
    NotReady(String),

    /// 向量库操作失败。
    #[error("向量库操作失败: {0}")]
    Storage(String),

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
            RagError::Storage(_) => 502,
            RagError::Internal(_) => 500,
        }
    }

    /// 这个错误是否**值得重试**。
    ///
    /// 只有基础设施故障（向量库不可达）才值得重试 ——
    /// 参数错误重试一万次也是错的。
    pub fn is_retryable(&self) -> bool {
        matches!(self, RagError::Storage(_) | RagError::NotReady(_))
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

impl From<rag_05::store::StoreError> for RagError {
    fn from(e: rag_05::store::StoreError) -> Self {
        RagError::Storage(e.to_string())
    }
}

impl From<rag_05::embed::EmbedError> for RagError {
    fn from(e: rag_05::embed::EmbedError) -> Self {
        RagError::Internal(e.to_string())
    }
}

/// 让 `RagError` 本身就能作为 axum 的错误响应。
impl axum::response::IntoResponse for RagError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // 4xx 是调用方的 bug，5xx 才是服务端 / 基础设施故障。
        match &self {
            RagError::BadRequest(_) => {
                warn!(status = status.as_u16(), error = %self, "请求参数不合法");
            }
            RagError::NotReady(_) => {
                warn!(status = status.as_u16(), error = %self, "向量库未就绪");
            }
            RagError::Storage(_) => {
                error!(status = status.as_u16(), error = %self, "向量库操作失败");
            }
            RagError::Internal(_) => {
                error!(status = status.as_u16(), error = %self, "内部错误");
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
        assert_eq!(RagError::Storage("x".into()).status_code(), 502);
        assert_eq!(RagError::Internal("x".into()).status_code(), 500);
    }

    #[test]
    fn only_infra_errors_are_retryable() {
        assert!(RagError::Storage("x".into()).is_retryable());
        assert!(RagError::NotReady("x".into()).is_retryable());
        assert!(!RagError::BadRequest("x".into()).is_retryable());
        assert!(!RagError::Internal("x".into()).is_retryable());
    }
}
