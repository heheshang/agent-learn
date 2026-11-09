//! 统一错误类型。
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`AppError::BadRequest`] | 400 | 请求参数不合法（问题为空、SQL 被护栏拦截等） |
//! | [`AppError::Llm`] | 502 | 模型调用失败（网络、鉴权、上游错误） |
//! | [`AppError::Tool`] | 500 | 工具执行失败（未被降级吸收时） |
//! | [`AppError::Parse`] | 502 | 模型输出解析失败 |
//! | [`AppError::Upstream`] | 502 | 外部服务失败 |
//! | [`AppError::Internal`] | 500 | 内部错误（IO、序列化、SQLite 错误等） |
//!
//! ## 三条硬约定
//!
//! 1. **SQL 被护栏拦截是 400，不是 500** ——
//!    那不是「服务端坏了」，是「这条 SQL 不该被执行」。
//! 2. **工具错误永远是「可观察的失败」** ——
//!    它会被渲染成一条 `Observation` 喂回模型，模型读到
//!    「SQL 执行失败：no such column: xxx」就能自己改对。
//! 3. **只有用户直接调工具时**，工具错误才变成 HTTP 响应 ——
//!    那时用户扮演了「模型」的角色。

use agent_logging::{error, warn};
use thiserror::Error;

/// SQL 护栏拦截时抛出的错误。
///
/// ## 为什么护栏错误要单独一个类型
///
/// 它不是「SQL 写错了」（那是模型能自我纠正的**执行错误**），
/// 而是「这条 SQL **根本不该被执行**」（危险操作）。
///
/// 两种错误的处理方式完全不同：
///
/// - **执行错误** → 喂回模型，让它改 SQL 再试（LangChain 的核心机制）
/// - **护栏拦截** → 同样喂回模型，但要明确告诉它「这条路被堵死了」，
///   否则模型会反复尝试同一条危险 SQL
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GuardError {
    /// 不是只读语句（不是 SELECT / WITH / VALUES / EXPLAIN 开头）。
    #[error("只允许 SELECT / WITH / VALUES / EXPLAIN 开头的只读查询，实际以 `{0}` 开头")]
    NotReadOnly(String),

    /// 包含危险关键字。
    #[error("SQL 里包含被禁止的关键字 `{keyword}`（只允许只读查询）")]
    ForbiddenKeyword {
        /// 命中的关键字。
        keyword: String,
    },

    /// 多语句（分号后还有内容）。
    #[error("只允许单条语句，但检测到多个语句（分号后仍有内容）")]
    MultipleStatements,

    /// 语句为空。
    #[error("SQL 不能为空")]
    Empty,

    /// 语句太长。
    #[error("SQL 太长（{len} 字符，上限 {limit}）")]
    TooLong {
        /// 实际长度。
        len: usize,
        /// 上限。
        limit: usize,
    },
}

impl GuardError {
    /// 拼成一句给模型看的话。
    ///
    /// ## 为什么文案要「可执行」
    ///
    /// 它会被喂回模型 —— 所以必须说清楚**下一步该怎么做**，
    /// 而不是只说「不合法」。
    pub fn message(&self) -> String {
        match self {
            GuardError::NotReadOnly(kw) => format!(
                "错误：数据库是**只读**的，这条 SQL 被安全护栏拦截 —— \
                 只允许 SELECT / WITH / VALUES / EXPLAIN 开头的只读查询，\
                 而它以 `{kw}` 开头。请改写成只读查询（例如把 UPDATE 换成 SELECT）。"
            ),
            GuardError::ForbiddenKeyword { keyword } => format!(
                "错误：这条 SQL 被安全护栏拦截 —— 含有被禁止的关键字 `{keyword}`。\
                 数据库是**只读**的，禁止 INSERT / UPDATE / DELETE / DROP / ALTER / \
                 CREATE / ATTACH / PRAGMA 等任何写操作。请只写 SELECT 查询。"
            ),
            GuardError::MultipleStatements => "错误：这条 SQL 被安全护栏拦截 —— \
                 只允许**单条**语句（分号后不能再有内容）。\
                 请把多条语句拆成多次查询。"
                .to_string(),
            GuardError::Empty => "错误：SQL 为空。请给出一条只读查询。".to_string(),
            GuardError::TooLong { len, limit } => {
                format!("错误：这条 SQL 太长（{len} 字符，上限 {limit}）。请简化查询。")
            }
        }
    }
}

/// 应用层错误。
#[derive(Debug, Error)]
pub enum AppError {
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

    /// 外部服务失败。
    #[error("外部服务失败: {0}")]
    Upstream(String),

    /// 内部错误。
    #[error("内部错误: {0}")]
    Internal(String),
}

impl AppError {
    /// 对应的 HTTP 状态码。
    pub fn status_code(&self) -> u16 {
        match self {
            AppError::BadRequest(_) => 400,
            AppError::Llm(_) => 502,
            AppError::Tool(_) => 500,
            AppError::Parse(_) => 502,
            AppError::Upstream(_) => 502,
            AppError::Internal(_) => 500,
        }
    }

    /// 这个错误**能不能降级**。
    ///
    /// ## 只有工具错误能降级
    ///
    /// 工具失败时，错误信息本身就是有用的 `Observation` ——
    /// 模型能读到「没有这一列：xxx」然后换一条 SQL 重试。
    ///
    /// 模型失败 / 解析失败不能降级：前者没有推理能力，
    /// 后者说明输出契约被破坏了，继续跑只会产出垃圾。
    pub fn is_degradable(&self) -> bool {
        matches!(self, AppError::Tool(_))
    }
}

/// 本项目的 `Result` 别名。
pub type AppResult<T> = Result<T, AppError>;

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Internal(e.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::Internal(e.to_string())
    }
}

impl From<rusqlite::Error> for AppError {
    fn from(e: rusqlite::Error) -> Self {
        AppError::Internal(format!("SQLite 错误: {e}"))
    }
}

impl From<agent_14::error::ToolError> for AppError {
    fn from(e: agent_14::error::ToolError) -> Self {
        match e {
            // 参数不合法 / 未知工具 —— 是**调用方**的问题，不是服务端的。
            agent_14::error::ToolError::InvalidArgs { .. }
            | agent_14::error::ToolError::UnknownTool { .. } => AppError::BadRequest(e.to_string()),
            agent_14::error::ToolError::Upstream { message } => AppError::Upstream(message),
            agent_14::error::ToolError::Execution { .. } => AppError::Tool(e.to_string()),
        }
    }
}

impl From<rag_07::error::QueryError> for AppError {
    fn from(e: rag_07::error::QueryError) -> Self {
        match e {
            rag_07::error::QueryError::BadRequest(msg) => AppError::BadRequest(msg),
            rag_07::error::QueryError::Transformation(msg)
            | rag_07::error::QueryError::Generation(msg) => AppError::Llm(msg),
            other => AppError::Internal(other.to_string()),
        }
    }
}

/// 让 [`AppError`] 本身就能作为 axum 的错误响应。
impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        match &self {
            AppError::BadRequest(_) => {
                warn!(status = status.as_u16(), error = %self, "请求参数不合法");
            }
            AppError::Upstream(_) | AppError::Llm(_) | AppError::Parse(_) => {
                error!(status = status.as_u16(), error = %self, "上游调用失败");
            }
            AppError::Tool(_) | AppError::Internal(_) => {
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
        assert_eq!(AppError::BadRequest("x".into()).status_code(), 400);
        assert_eq!(AppError::Llm("x".into()).status_code(), 502);
        assert_eq!(AppError::Tool("x".into()).status_code(), 500);
        assert_eq!(AppError::Parse("x".into()).status_code(), 502);
        assert_eq!(AppError::Upstream("x".into()).status_code(), 502);
        assert_eq!(AppError::Internal("x".into()).status_code(), 500);
    }

    #[test]
    fn only_tool_error_is_degradable() {
        assert!(AppError::Tool("x".into()).is_degradable());
        assert!(!AppError::Llm("x".into()).is_degradable());
        assert!(!AppError::Upstream("x".into()).is_degradable());
    }

    #[test]
    fn guard_error_messages_are_actionable() {
        // 护栏错误会被喂回模型 —— 文案必须说清楚下一步该做什么。
        let err = GuardError::NotReadOnly("UPDATE".to_string());
        let msg = err.message();
        assert!(msg.contains("只读"));
        assert!(msg.contains("SELECT"));

        let err = GuardError::ForbiddenKeyword {
            keyword: "DROP".to_string(),
        };
        assert!(err.message().contains("DROP"));

        let err = GuardError::MultipleStatements;
        assert!(err.message().contains("单条"));
    }
}
