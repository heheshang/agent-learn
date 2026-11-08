//! 统一错误类型。
//!
//! | 错误 | HTTP | 含义 |
//! |---|---|---|
//! | [`AppError::BadRequest`] | 400 | 请求参数不合法（问题为空、作用域名非法） |
//! | [`AppError::Llm`] | 502 | 模型调用失败（网络、鉴权、上游错误） |
//! | [`AppError::Tool`] | 500 | 工具执行失败（未被降级吸收时） |
//! | [`AppError::Parse`] | 502 | 模型输出解析失败 |
//! | [`AppError::Upstream`] | 502 | 外部服务失败（天气 API、MCP 服务端） |
//! | [`AppError::Internal`] | 500 | 内部错误（IO、序列化等） |//!
//! ## 为什么工具错误要单独一个类型
//!
//! LangChain 教程里用 `handle_tool_errors` 中间件把工具异常
//! 转成 `ToolMessage` 回灌给模型 —— 本项目用
//! [`ToolError`] 表达同样的语义。
//!
//! **工具错误永远是「可观察的失败」，不是请求失败**：
//! 它会被渲染成一条 `Observation` 喂回模型，模型读到
//! 「参数 `location` 必须是字符串」就能自己纠正。
//!
//! 只有当**用户直接调工具**（`POST /api/tools/call`）时，
//! 工具错误才会变成 HTTP 响应 —— 那时用户就扮演了「模型」的角色。

use agent_logging::{error, warn};
use thiserror::Error;

/// 工具层错误。
///
/// ## 每一种错误都带「模型能据此纠正」的信息
///
/// - [`ToolError::UnknownTool`] —— 列出可用工具
/// - [`ToolError::InvalidArgs`] —— 指出哪个参数、期望什么类型
/// - [`ToolError::Execution`] —— 工具内部逻辑失败
/// - [`ToolError::Upstream`] —— 外部服务失败（天气 API / MCP）
#[derive(Debug, Error, Clone)]
pub enum ToolError {
    /// 调用了不存在的工具。
    #[error("错误：未知工具 `{name}`。可用工具：{available}")]
    UnknownTool {
        /// 模型给出的工具名。
        name: String,
        /// 可用工具列表（逗号分隔）。
        available: String,
    },

    /// 参数不合法（Schema 校验失败）。
    #[error("错误：参数不合法 —— {issues}")]
    InvalidArgs {
        /// 所有问题（分号分隔）。
        issues: String,
    },

    /// 工具执行失败。
    #[error("错误：工具 `{tool}` 执行失败 —— {message}")]
    Execution {
        /// 工具名。
        tool: String,
        /// 失败原因。
        message: String,
    },

    /// 外部服务不可用。
    #[error("错误：外部服务不可用 —— {message}")]
    Upstream {
        /// 失败原因。
        message: String,
    },
}

impl ToolError {
    /// 构造「未知工具」错误。
    pub fn unknown(name: impl Into<String>, available: impl Into<String>) -> Self {
        ToolError::UnknownTool {
            name: name.into(),
            available: available.into(),
        }
    }

    /// 构造「参数不合法」错误。
    pub fn invalid_args(issues: Vec<String>) -> Self {
        ToolError::InvalidArgs {
            issues: issues.join("；"),
        }
    }

    /// 构造「工具执行失败」错误。
    pub fn execution(tool: impl Into<String>, message: impl Into<String>) -> Self {
        ToolError::Execution {
            tool: tool.into(),
            message: message.into(),
        }
    }

    /// 构造「上游服务失败」错误。
    pub fn upstream(message: impl Into<String>) -> Self {
        ToolError::Upstream {
            message: message.into(),
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
    /// 模型能读到「缺少必填参数 `location`」然后换个参数重试。
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

impl From<ToolError> for AppError {
    fn from(e: ToolError) -> Self {
        match e {
            // 参数不合法 / 未知工具 —— 是**调用方**的问题，不是服务端的。
            //
            // ⚠️ 在 Agent 循环里它们会被降级成 Observation；
            //    但在「用户直接调工具」的端点里，用户扮演的是模型角色，
            //    所以应该得到一个 400 而不是 500 ——
            //    500 会让调用方以为「服务端坏了，重试也没用」。
            ToolError::InvalidArgs { .. } | ToolError::UnknownTool { .. } => {
                AppError::BadRequest(e.to_string())
            }
            ToolError::Upstream { message } => AppError::Upstream(message),
            ToolError::Execution { .. } => AppError::Tool(e.to_string()),
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
    fn tool_errors_carry_actionable_text() {
        // 工具错误的文案必须让模型能自我纠正 —— 所以要带具体信息。
        let err = ToolError::unknown("foo", "search, get_weather");
        assert!(err.to_string().contains("foo"));
        assert!(err.to_string().contains("search"));

        let err = ToolError::invalid_args(vec!["location 必须是字符串".to_string()]);
        assert!(err.to_string().contains("location"));
    }

    #[test]
    fn argument_errors_map_to_bad_request() {
        // ⚠️ 「参数填错」和「未知工具」是**调用方**的问题 → 400。
        //    500 会让调用方以为「服务端坏了，重试也没用」。
        let err: AppError = ToolError::invalid_args(vec!["x".to_string()]).into();
        assert_eq!(err.status_code(), 400);

        let err: AppError = ToolError::unknown("foo", "bar").into();
        assert_eq!(err.status_code(), 400);

        // 上游服务失败 → 502。
        let err: AppError = ToolError::upstream("连不上").into();
        assert_eq!(err.status_code(), 502);

        // 工具内部逻辑失败 → 500。
        let err: AppError = ToolError::execution("tool", "炸了").into();
        assert_eq!(err.status_code(), 500);
    }
}
