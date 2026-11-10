//! 应用错误类型。
//!
//! ## 三类错误，对应三种 HTTP 状态码
//!
//! | 类别 | 状态码 | 含义 |
//! |---|---|---|
//! | `BadRequest` | 400 | 调用方的问题（参数错、批次为空） |
//! | `Upstream` | 502/503 | 外部依赖的问题（远程 embedding 服务挂了） |
//! | `Internal` | 500 | 本项目的缺陷 |
//!
//! ⚠️ **不要把上游超时映射成 500** ——
//! 500 会让调用方以为「服务端坏了」，而 503 的标准语义是
//! 「稍后重试可能成功」（18 项目的教训）。

use thiserror::Error;

/// 应用级错误。
#[derive(Debug, Error)]
pub enum AppError {
    /// 调用方的问题 → 400。
    #[error("请求参数错误: {0}")]
    BadRequest(String),

    /// 上游（远程模型服务）的问题 → 503。
    #[error("上游服务错误: {0}")]
    Upstream(String),

    /// 本项目的缺陷 → 500。
    #[error("内部错误: {0}")]
    Internal(String),

    /// 模型推理失败 → 500。
    #[error("推理失败: {0}")]
    Inference(String),
}

impl AppError {
    /// HTTP 状态码。
    pub fn status_code(&self) -> u16 {
        match self {
            AppError::BadRequest(_) => 400,
            AppError::Upstream(_) => 503,
            AppError::Internal(_) | AppError::Inference(_) => 500,
        }
    }

    /// 稳定的错误码（响应体里返回）。
    pub fn code(&self) -> &'static str {
        match self {
            AppError::BadRequest(_) => "E_BAD_REQUEST",
            AppError::Upstream(_) => "E_UPSTREAM",
            AppError::Internal(_) => "E_INTERNAL",
            AppError::Inference(_) => "E_INFERENCE",
        }
    }
}

/// 应用级 Result。
pub type AppResult<T> = Result<T, AppError>;

/// 把 [`AppError`] 直接转成 HTTP 响应。
///
/// ## ⚠️ 为什么不写成 `Result<_, Response>`
///
/// 那样 `Err` 变体会是 `axum::response::Response`（至少 128 字节），
/// clippy 的 `result_large_err` 会报错（`-D warnings` 门槛）。
///
/// 正确做法是让**错误类型自己**实现 `IntoResponse` ——
/// 于是处理器的签名就是 `Result<impl IntoResponse, AppError>`，
/// `Err` 变体只有 `AppError` 那么小。
impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // ⚠️ 4xx 用 `warn!`、5xx 用 `error!` ——
        //    告警不能被 4xx 淹没（全仓库日志约定）。
        match &self {
            AppError::BadRequest(_) => {
                agent_logging::warn!(status = status.as_u16(), error = %self, "请求参数不合法");
            }
            AppError::Upstream(_) => {
                agent_logging::error!(status = status.as_u16(), error = %self, "上游调用失败");
            }
            AppError::Internal(_) | AppError::Inference(_) => {
                agent_logging::error!(status = status.as_u16(), error = %self, "请求处理失败");
            }
        }

        let body = crate::api::models::ErrorResponse {
            code: self.code().to_string(),
            message: self.to_string(),
        };

        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_request_maps_to_400() {
        assert_eq!(AppError::BadRequest("x".into()).status_code(), 400);
    }

    #[test]
    fn upstream_maps_to_503_not_500() {
        // ⚠️ 503 的语义是「稍后重试可能成功」——
        //    500 会让调用方以为重试没用。
        assert_eq!(AppError::Upstream("timeout".into()).status_code(), 503);
    }

    #[test]
    fn internal_maps_to_500() {
        assert_eq!(AppError::Internal("bug".into()).status_code(), 500);
    }

    #[test]
    fn error_codes_are_distinct() {
        let codes = [
            AppError::BadRequest("a".into()).code(),
            AppError::Upstream("a".into()).code(),
            AppError::Internal("a".into()).code(),
            AppError::Inference("a".into()).code(),
        ];
        let unique: std::collections::HashSet<_> = codes.iter().collect();
        assert_eq!(unique.len(), codes.len());
    }
}
