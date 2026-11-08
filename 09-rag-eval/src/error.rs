//! 统一错误类型。
//!
//! 错误分六类，映射到 HTTP 状态码：
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`EvalError::BadRequest`] | 400 | 请求参数不合法（空问题、非法指标名） |
//! | [`EvalError::NotReady`] | 503 | 索引或评估集还没就绪 |
//! | [`EvalError::Dataset`] | 500 | 评估集格式错误 / 加载失败 |
//! | [`EvalError::Retrieval`] | 500 | 检索失败 |
//! | [`EvalError::Judging`] | 502 | LLM-as-a-judge 调用失败或解析失败 |
//! | [`EvalError::Internal`] | 500 | 内部错误（IO、解析等） |
//!
//! ## 为什么把「评估」的错误和「被评估系统」的错误分开
//!
//! 08-rag 的 [`rag_08::error::RagError`] 描述的是**被评估系统**的故障：
//! 检索挂了、重排挂了、生成挂了。
//!
//! 本项目的 [`EvalError`] 描述的是**评估器自身**的故障：
//! 评估集格式不对、judge 调不通、指标算不出来。
//!
//! 两者混在一起会丢掉一个关键信息：**「系统跑得好不好」和「评估器跑得好不好」
//! 是两回事**。
//!
//! 比如 judge 返回了无法解析的内容 —— 这不是系统的错，是评估的错。
//! 如果都塞进一个错误类型里，报告里就会出现「检索失败」这种误导性的归因。

use agent_logging::{error, warn};
use thiserror::Error;

/// 评估系统的错误。
#[derive(Debug, Error)]
pub enum EvalError {
    /// 请求参数不合法。
    #[error("请求参数不合法: {0}")]
    BadRequest(String),

    /// 未就绪 —— 索引或评估集还没准备好。
    #[error("评估未就绪: {0}")]
    NotReady(String),

    /// 评估集相关错误（文件缺失、格式错误、字段缺失）。
    #[error("评估集错误: {0}")]
    Dataset(String),

    /// 被评估系统的检索失败。
    #[error("检索失败: {0}")]
    Retrieval(String),

    /// judge 调用失败 / 输出解析失败。
    #[error("评估判定失败: {0}")]
    Judging(String),

    /// 内部错误（IO、解析等）。
    #[error("内部错误: {0}")]
    Internal(String),
}

impl EvalError {
    /// 对应的 HTTP 状态码。
    pub fn status_code(&self) -> u16 {
        match self {
            EvalError::BadRequest(_) => 400,
            EvalError::NotReady(_) => 503,
            EvalError::Dataset(_) => 500,
            EvalError::Retrieval(_) => 500,
            EvalError::Judging(_) => 502,
            EvalError::Internal(_) => 500,
        }
    }

    /// 这个错误**能不能降级**。
    ///
    /// - judge 失败 → **能降级**（退回确定性兜底算法，指标偏保守但仍然有数）
    /// - 生成失败 → 不能降级（没有答案就没什么可评估的）
    /// - 评估集错误 → 不能降级（数据错了，后面全是错的）
    ///
    /// 和 07/08 一样的取舍：**增强环节失败不该让整条链路崩掉**。
    pub fn is_degradable(&self) -> bool {
        matches!(self, EvalError::Judging(_))
    }
}

/// 本项目的 `Result` 别名。
pub type EvalResult<T> = Result<T, EvalError>;

impl From<std::io::Error> for EvalError {
    fn from(e: std::io::Error) -> Self {
        EvalError::Internal(e.to_string())
    }
}

impl From<serde_json::Error> for EvalError {
    fn from(e: serde_json::Error) -> Self {
        EvalError::Internal(e.to_string())
    }
}

impl From<rag_08::error::RagError> for EvalError {
    fn from(e: rag_08::error::RagError) -> Self {
        match e {
            rag_08::error::RagError::BadRequest(msg) => EvalError::BadRequest(msg),
            rag_08::error::RagError::NotReady(msg) => EvalError::NotReady(msg),
            other => EvalError::Retrieval(other.to_string()),
        }
    }
}

impl axum::response::IntoResponse for EvalError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // 错误分级：4xx 是调用方的问题，5xx 才是服务端的问题。
        match &self {
            EvalError::BadRequest(_) => {
                warn!(status = status.as_u16(), error = %self, "请求参数不合法");
            }
            EvalError::NotReady(_) => {
                warn!(status = status.as_u16(), error = %self, "评估未就绪");
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
        assert_eq!(EvalError::BadRequest("x".into()).status_code(), 400);
        assert_eq!(EvalError::NotReady("x".into()).status_code(), 503);
        assert_eq!(EvalError::Dataset("x".into()).status_code(), 500);
        assert_eq!(EvalError::Retrieval("x".into()).status_code(), 500);
        assert_eq!(EvalError::Judging("x".into()).status_code(), 502);
        assert_eq!(EvalError::Internal("x".into()).status_code(), 500);
    }

    #[test]
    fn only_judging_is_degradable() {
        assert!(EvalError::Judging("x".into()).is_degradable());
        assert!(!EvalError::Dataset("x".into()).is_degradable());
        assert!(!EvalError::Retrieval("x".into()).is_degradable());
    }

    #[test]
    fn rag_error_conversion_keeps_categories() {
        let bad = EvalError::from(rag_08::error::RagError::BadRequest("空".to_string()));
        assert_eq!(bad.status_code(), 400);

        let retrieval = EvalError::from(rag_08::error::RagError::Retrieval("挂了".to_string()));
        assert_eq!(retrieval.status_code(), 500);
    }
}
