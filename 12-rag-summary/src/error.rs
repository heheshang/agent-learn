//! 统一错误类型。
//!
//! 错误分六类，映射到 HTTP 状态码：
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`RagError::BadRequest`] | 400 | 请求参数不合法（空问题、非法模式名） |
//! | [`RagError::NotReady`] | 503 | 索引还没建好 / Milvus 不可达 |
//! | [`RagError::Retrieval`] | 500 | 检索失败（索引为空、维度不匹配） |
//! | [`RagError::Rerank`] | 502 | 重排失败（cross-encoder 推理出错） |
//! | [`RagError::Generation`] | 502 | LLM 调用失败 |
//! | [`RagError::Internal`] | 500 | 内部错误（IO、解析等） |
//!
//! ## 降级语义
//!
//! 本项目有三个**可降级**环节，它们失败时都不应该让整个请求失败：
//!
//! | 环节 | 失败后的行为 |
//! |---|---|
//! | 查询改写（多问法） | 退回原问题继续检索 |
//! | 重排（cross-encoder） | 退回融合后的原始顺序 |
//! | 上下文压缩 | 退回未压缩的完整片段（超预算就砍掉最后几块） |
//!
//! 只有「检索彻底失败」和「生成失败」是不可降级的 —— 前者没数据，
//! 后者没答案。
//!
//! 所以 [`RagError::is_degradable`] 对这三个环节返回 `true`，
//! 让「哪些环节挂了服务还能不能服务」变成一个**可验证的行为**，
//! 而不是散落在各处 `if let Ok(...)` 里的隐式逻辑。

use thiserror::Error;

/// 摘要式 RAG 系统的错误。
#[derive(Debug, Error)]
pub enum RagError {
    /// 请求参数不合法。
    #[error("请求参数不合法: {0}")]
    BadRequest(String),

    /// 索引未就绪 —— 服务已启动但还没有知识库，或 Milvus 不可达。
    #[error("索引未就绪: {0}")]
    NotReady(String),

    /// 检索失败。
    #[error("检索失败: {0}")]
    Retrieval(String),

    /// 重排失败（cross-encoder 推理出错、rerank API 调用失败）。
    #[error("重排失败: {0}")]
    Rerank(String),

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
            RagError::Rerank(_) => 502,
            RagError::Generation(_) => 502,
            RagError::Internal(_) => 500,
        }
    }

    /// 这个错误**能不能降级**。
    ///
    /// - 重排失败 → 能降级（退回融合后的原始顺序）
    /// - 检索失败 → 不能降级（没数据就没有答案）
    /// - 生成失败 → 不能降级（没有模型就没有答案）
    pub fn is_degradable(&self) -> bool {
        matches!(self, RagError::Rerank(_))
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
/// 实现它之后，「请求体解析失败」和「业务校验失败」会走到
/// **完全相同**的响应格式上（见 [`crate::api::extract::JsonBody`]）。
impl axum::response::IntoResponse for RagError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
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
        assert_eq!(RagError::Rerank("x".into()).status_code(), 502);
        assert_eq!(RagError::Generation("x".into()).status_code(), 502);
        assert_eq!(RagError::Internal("x".into()).status_code(), 500);
    }

    #[test]
    fn only_rerank_is_degradable() {
        assert!(RagError::Rerank("x".into()).is_degradable());
        assert!(!RagError::Generation("x".into()).is_degradable());
        assert!(!RagError::Retrieval("x".into()).is_degradable());
    }
}
