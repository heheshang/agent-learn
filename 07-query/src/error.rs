//! 统一错误类型。
//!
//! 错误分五类，映射到 HTTP 状态码：
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`QueryError::BadRequest`] | 400 | 请求参数不合法（空问题、top_k 超范围） |
//! | [`QueryError::NotReady`] | 503 | 索引还没建好 |
//! | [`QueryError::Retrieval`] | 500 | 检索失败（索引为空、维度不匹配） |
//! | [`QueryError::Transformation`] | 502 | 查询改写失败（LLM 调用出错） |
//! | [`QueryError::Generation`] | 502 | 生成失败（LLM 调用出错） |
//!
//! 分开的意义：调用方能区分「我请求写错了」（改请求）和
//! 「你服务坏了」（重试/报警），而不用去猜错误消息里的中文。
//!
//! ## 为什么 `Transformation` 和 `Generation` 都是 502
//!
//! 它们都是「调外部 LLM 失败」，语义上对调用方是一样的 ——
//! 不是我的请求有问题，是上游模型挂了。
//! 但在**代码里**必须分开：改写失败要能降级（退回原问题继续检索），
//! 生成失败没有降级路径（没有上下文就没答案）。
//! 把它们合成一个 `Llm` 错误，就没法区分「可以降级」和「必须报错」。

use thiserror::Error;

/// 查询改写系统的错误。
#[derive(Debug, Error)]
pub enum QueryError {
    /// 请求参数不合法。
    #[error("请求参数不合法: {0}")]
    BadRequest(String),

    /// 索引还没建好 —— 服务已启动但还没有知识库。
    #[error("索引未就绪: {0}")]
    NotReady(String),

    /// 检索失败。
    #[error("检索失败: {0}")]
    Retrieval(String),

    /// 查询改写失败（LLM 调用出错）。
    #[error("查询改写失败: {0}")]
    Transformation(String),

    /// 生成失败（LLM 调用出错）。
    #[error("生成失败: {0}")]
    Generation(String),

    /// 内部错误（IO、解析等）。
    #[error("内部错误: {0}")]
    Internal(String),
}

impl QueryError {
    /// 对应的 HTTP 状态码。
    pub fn status_code(&self) -> u16 {
        match self {
            QueryError::BadRequest(_) => 400,
            QueryError::NotReady(_) => 503,
            QueryError::Retrieval(_) => 500,
            QueryError::Transformation(_) => 502,
            QueryError::Generation(_) => 502,
            QueryError::Internal(_) => 500,
        }
    }

    /// 这个错误**能不能降级**。
    ///
    /// 降级 = 「用原始问题继续跑，结果仍然可用」。
    ///
    /// - 改写失败 → **能降级**（LLM 挂了就用原问题检索，召回差一点但能用）
    /// - 生成失败 → **不能降级**（没有模型就没有答案）
    /// - 检索失败 → 不能降级（没数据）
    ///
    /// 这个判断让「LLM 挂掉时服务还能不能提供基础能力」变成一个显式的决定，
    /// 而不是各处 `if let Ok(...)` 里散落的隐式行为。
    pub fn is_degradable(&self) -> bool {
        matches!(self, QueryError::Transformation(_))
    }
}

/// 本项目的 `Result` 别名。
pub type QueryResult<T> = Result<T, QueryError>;

impl From<std::io::Error> for QueryError {
    fn from(e: std::io::Error) -> Self {
        QueryError::Internal(e.to_string())
    }
}

impl From<serde_json::Error> for QueryError {
    fn from(e: serde_json::Error) -> Self {
        QueryError::Internal(e.to_string())
    }
}

/// 让 `QueryError` 本身就能作为 axum 的错误响应。
///
/// 实现它之后，`QueryError` 就可以直接当作 extractor 的 `Rejection` 类型 ——
/// 见 [`crate::api::extract::JsonBody`]。这样「请求体解析失败」和
/// 「业务校验失败」会走到**完全相同**的响应格式上。
impl axum::response::IntoResponse for QueryError {
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
        assert_eq!(QueryError::BadRequest("x".into()).status_code(), 400);
        assert_eq!(QueryError::NotReady("x".into()).status_code(), 503);
        assert_eq!(QueryError::Retrieval("x".into()).status_code(), 500);
        assert_eq!(QueryError::Transformation("x".into()).status_code(), 502);
        assert_eq!(QueryError::Generation("x".into()).status_code(), 502);
        assert_eq!(QueryError::Internal("x".into()).status_code(), 500);
    }

    #[test]
    fn only_transformation_is_degradable() {
        assert!(QueryError::Transformation("x".into()).is_degradable());
        assert!(!QueryError::Generation("x".into()).is_degradable());
        assert!(!QueryError::Retrieval("x".into()).is_degradable());
    }
}
