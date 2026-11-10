//! 应用错误类型。
//!
//! ## 三类错误，对应三种 HTTP 状态码
//!
//! | 类别 | 状态码 | 含义 |
//! |---|---|---|
//! | `BadRequest` | 400 | 调用方的问题（参数错、prompt 为空） |
//! | `Upstream` | 502/503 | 上游（vLLM 服务）的问题 |
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

    /// 上游（vLLM 服务）的问题 → 503。
    #[error("上游服务错误: {0}")]
    Upstream(String),

    /// 本项目缺陷 → 500。
    #[error("内部错误: {0}")]
    Internal(String),
}

impl AppError {
    /// HTTP 状态码。
    pub fn status_code(&self) -> u16 {
        match self {
            AppError::BadRequest(_) => 400,
            AppError::Upstream(_) => 503,
            AppError::Internal(_) => 500,
        }
    }

    /// 稳定的错误码（响应体里返回）。
    pub fn code(&self) -> &'static str {
        match self {
            AppError::BadRequest(_) => "E_BAD_REQUEST",
            AppError::Upstream(_) => "E_UPSTREAM",
            AppError::Internal(_) => "E_INTERNAL",
        }
    }
}

/// 应用级 Result。
pub type AppResult<T> = Result<T, AppError>;

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
        ];
        let unique: std::collections::HashSet<_> = codes.iter().collect();
        assert_eq!(unique.len(), codes.len());
    }
}
