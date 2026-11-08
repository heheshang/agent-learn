//! 统一错误类型。
//!
//! 错误分四类，映射到 HTTP 状态码：
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`AdvanceError::BadRequest`] | 400 | 请求参数不合法 |
//! | [`AdvanceError::NotFound`] | 404 | 文件 / 资源不存在 |
//! | [`AdvanceError::Parse`] | 422 | PDF 解析失败（结构损坏、加密等） |
//! | [`AdvanceError::Internal`] | 500 | 内部错误（IO、编码等） |
//!
//! ## 为什么把「解析失败」单列一类
//!
//! 解析失败**不是**内部错误，也不是「参数错了」。
//!
//! - 参数错了 → 用户改一下就行（400）
//! - 内部错误 → 我们的 bug（500）
//! - 解析失败 → **文件本身有问题**，用户换一份文件或者我们加一个解析器才行（422）
//!
//! 混在一起的话，运维看到的告警全是 500，以为是服务挂了 —— 其实是有人在
//! 上传损坏的 PDF。

use agent_logging::{error, warn};
use thiserror::Error;

/// 高级数据处理的错误。
#[derive(Debug, Error)]
pub enum AdvanceError {
    /// 请求参数不合法。
    #[error("请求参数不合法: {0}")]
    BadRequest(String),

    /// 资源不存在。
    #[error("资源不存在: {0}")]
    NotFound(String),

    /// 文档解析失败。
    #[error("文档解析失败: {0}")]
    Parse(String),

    /// 内部错误。
    #[error("内部错误: {0}")]
    Internal(String),
}

impl AdvanceError {
    /// 对应的 HTTP 状态码。
    pub fn status_code(&self) -> u16 {
        match self {
            AdvanceError::BadRequest(_) => 400,
            AdvanceError::NotFound(_) => 404,
            AdvanceError::Parse(_) => 422,
            AdvanceError::Internal(_) => 500,
        }
    }
}

/// 本项目的 `Result` 别名。
pub type AdvanceResult<T> = Result<T, AdvanceError>;

impl From<std::io::Error> for AdvanceError {
    fn from(e: std::io::Error) -> Self {
        AdvanceError::Internal(e.to_string())
    }
}

impl From<serde_json::Error> for AdvanceError {
    fn from(e: serde_json::Error) -> Self {
        AdvanceError::Internal(e.to_string())
    }
}

impl axum::response::IntoResponse for AdvanceError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // 4xx 是调用方的输入问题，5xx 才是我们的 bug。
        match &self {
            AdvanceError::BadRequest(_) | AdvanceError::NotFound(_) => {
                warn!(status = status.as_u16(), error = %self, "请求被拒绝");
            }
            AdvanceError::Parse(_) => {
                error!(status = status.as_u16(), error = %self, "文档解析失败");
            }
            AdvanceError::Internal(_) => {
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
        assert_eq!(AdvanceError::BadRequest("x".into()).status_code(), 400);
        assert_eq!(AdvanceError::NotFound("x".into()).status_code(), 404);
        assert_eq!(AdvanceError::Parse("x".into()).status_code(), 422);
        assert_eq!(AdvanceError::Internal("x".into()).status_code(), 500);
    }

    #[test]
    fn io_error_becomes_internal() {
        let err: AdvanceError = std::io::Error::other("boom").into();
        assert_eq!(err.status_code(), 500);
    }
}
