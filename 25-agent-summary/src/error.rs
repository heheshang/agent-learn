//! 统一错误类型 —— HTTP 状态码映射。
//!
//! ## 状态码映射（与 18 项目一致）
//!
//! | 类别 | 状态码 | 为什么 |
//! |---|---|---|
//! | Business | **400** | 调用方的问题 |
//! | Transient | **503** | 503 的标准语义是「稍后重试可能成功」—— 用 500 会让调用方以为重试没用 |
//! | Fatal | **500** | 服务端的问题 |
//! | 模型调用失败 | **502** | 「没有模型就没有答案」 |
//!
//! ## ⚠️ `AppError` 自己实现 `IntoResponse`
//!
//! 写成 `Result<_, Response>` 会让 `Err` 变体至少 128 字节，
//! 触发 clippy 的 `result_large_err`（`-D warnings` 门槛）。

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// 应用错误。
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    /// 参数不合法（400）。
    #[error("参数不合法：{0}")]
    BadRequest(String),

    /// 上游暂时不可用（503）。
    #[error("上游暂时不可用：{0}")]
    Unavailable(String),

    /// 内部错误（500）。
    #[error("内部错误：{0}")]
    Internal(String),

    /// 资源不存在（404）。
    #[error("资源不存在：{0}")]
    NotFound(String),
}

/// 错误响应体。
#[derive(Debug, Serialize)]
pub struct ErrorBody {
    /// 错误码。
    pub code: &'static str,

    /// 错误消息。
    pub message: String,
}

impl AppError {
    /// 状态码。
    pub fn status_code(&self) -> StatusCode {
        match self {
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
        }
    }

    /// 错误码。
    pub fn code(&self) -> &'static str {
        match self {
            AppError::BadRequest(_) => "E_BAD_REQUEST",
            AppError::Unavailable(_) => "E_UNAVAILABLE",
            AppError::Internal(_) => "E_INTERNAL",
            AppError::NotFound(_) => "E_NOT_FOUND",
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let code = self.code();
        let message = self.to_string();

        // ⚠️ 4xx 用 `warn!`、5xx 用 `error!` ——
        //    告警不能被 4xx 淹没。
        if status.is_client_error() {
            agent_logging::warn!("请求失败 {} {}：{}", code, status.as_u16(), message);
        } else {
            agent_logging::error!("请求失败 {} {}：{}", code, status.as_u16(), message);
        }

        (status, Json(ErrorBody { code, message })).into_response()
    }
}

/// 便捷类型别名。
pub type AppResult<T> = Result<T, AppError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_follow_the_matrix() {
        assert_eq!(
            AppError::BadRequest("x".into()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            AppError::Unavailable("x".into()).status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            AppError::Internal("x".into()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            AppError::NotFound("x".into()).status_code(),
            StatusCode::NOT_FOUND
        );
    }

    #[test]
    fn error_body_carries_a_stable_code() {
        let e = AppError::BadRequest("参数错了".into());
        assert_eq!(e.code(), "E_BAD_REQUEST");
        assert!(e.to_string().contains("参数错了"));
    }
}
