//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见 ——
//! 想知道服务能接什么、返回什么，看这一个文件就够了。

use serde::{Deserialize, Serialize};

/// 问答请求体。
///
/// ```json
/// { "question": "什么是 RAG？", "top_k": 4 }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct AskRequest {
    /// 用户的问题。
    pub question: String,

    /// 召回的块数（可选，默认用配置里的值）。
    #[serde(default)]
    pub top_k: Option<usize>,

    /// 是否同时返回检索结果明细（可选，默认 false）。
    ///
    /// 调试时打开，能看到「模型到底看到了什么」。
    #[serde(default)]
    pub include_context: bool,
}

/// 问答响应体。
#[derive(Debug, Clone, Serialize)]
pub struct AskResponse {
    /// 最终答案。
    pub answer: String,

    /// 引用到的块。
    pub citations: Vec<crate::generation::Citation>,

    /// 检索到的块（`include_context=true` 时才有内容）。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub retrieved: Vec<crate::retrieval::RetrievedChunk>,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,

    /// token 用量。
    pub usage: crate::generation::TokenUsage,

    /// 使用的模型。
    pub model: String,
}

/// Agent 问答响应体。
#[derive(Debug, Clone, Serialize)]
pub struct AgentResponse {
    /// 最终答案。
    pub answer: String,

    /// 工具调用记录 —— Agent 调了什么、传了什么参数、拿回什么。
    pub tool_calls: Vec<crate::agent::ToolCallRecord>,

    /// 迭代轮数。
    pub iterations: usize,

    /// 是否触达迭代上限。
    pub hit_max_iterations: bool,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,

    /// token 用量。
    pub usage: crate::generation::TokenUsage,

    /// 使用的模型。
    pub model: String,
}

/// 纯检索请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    /// 查询词。
    pub query: String,

    /// 召回数量（可选）。
    #[serde(default)]
    pub top_k: Option<usize>,
}

/// 纯检索响应体。
#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    /// 查询词。
    pub query: String,

    /// 命中块。
    pub chunks: Vec<crate::retrieval::RetrievedChunk>,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 重建索引响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ReindexResponse {
    /// 是否成功。
    pub ok: bool,

    /// 索引统计。
    pub stats: crate::index::IndexStats,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 错误响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// 错误类别（HTTP 状态码的规范名）。
    pub error: String,

    /// 错误详情。
    pub message: String,
}
