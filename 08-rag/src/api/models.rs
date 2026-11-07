//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见 ——
//! 想知道服务能接什么、返回什么，看这一个文件就够了。

use serde::{Deserialize, Serialize};

use crate::hybrid::{HybridHit, SearchMode};
use crate::pipeline::Citation;

/// 问答请求体。
///
/// ```json
/// { "question": "什么是 RAG？", "mode": "hybrid" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct AskRequest {
    /// 用户的问题。
    pub question: String,

    /// 检索模式（可选，默认用配置里的）。
    #[serde(default)]
    pub mode: Option<String>,

    /// 是否返回检索明细（可选，默认 false）。
    #[serde(default)]
    pub include_context: bool,
}

/// 问答响应体。
#[derive(Debug, Clone, Serialize)]
pub struct AskResponse {
    /// 最终答案。
    pub answer: String,

    /// 引用到的块。
    pub citations: Vec<Citation>,

    /// 最终命中的块（`include_context=true` 时才有内容）。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hits: Vec<HybridHit>,

    /// 检索模式。
    pub mode: SearchMode,

    /// 是否执行了重排。
    pub reranked: bool,

    /// 降级原因（如果有）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,

    /// 检索耗时（毫秒）。
    pub retrieve_ms: u64,

    /// 重排耗时（毫秒）。
    pub rerank_ms: u64,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 纯检索请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    /// 问题。
    pub question: String,

    /// 检索模式（可选，默认用配置里的）。
    #[serde(default)]
    pub mode: Option<String>,

    /// 是否返回融合后（重排前）的结果（可选，默认 false）。
    #[serde(default)]
    pub include_fused: bool,
}

/// 纯检索响应体。
#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    /// 原始问题。
    pub question: String,

    /// 检索模式。
    pub mode: SearchMode,

    /// 最终命中（重排后）。
    pub hits: Vec<HybridHit>,

    /// 融合后的命中（重排前；`include_fused=true` 时才有内容）。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub fused_hits: Vec<HybridHit>,

    /// 是否执行了重排。
    pub reranked: bool,

    /// 降级原因（如果有）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,

    /// 检索耗时（毫秒）。
    pub retrieve_ms: u64,

    /// 重排耗时（毫秒）。
    pub rerank_ms: u64,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 多模式对比请求体。
///
/// ```json
/// {
///   "question": "BM25 是什么",
///   "modes": ["vector", "bm25", "hybrid"],
///   "expected": ["倒排索引"]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CompareRequest {
    /// 问题。
    pub question: String,

    /// 要对比的模式列表（可选，默认全部）。
    #[serde(default)]
    pub modes: Option<Vec<String>>,

    /// 期望命中的关键词（可选）。
    ///
    /// 给了的话，响应里每个模式会带一个 `hit` 布尔值：
    /// 该模式的召回里是否出现了含这些关键词的块。
    ///
    /// 这是本项目最实用的「量化收益」方式 ——
    /// 不用人工判读，直接看哪个模式把正确块捞上来了。
    #[serde(default)]
    pub expected: Option<Vec<String>>,

    /// 是否关闭重排（可选，默认 false）——
    /// 用来对比「重排前 vs 重排后」。
    #[serde(default)]
    pub no_rerank: bool,
}

/// 多模式对比响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CompareResponse {
    /// 原始问题。
    pub question: String,

    /// 每个模式的结果。
    pub results: Vec<ModeResult>,

    /// 模式之间的召回重叠度矩阵。
    ///
    /// `overlap[i][j]` = 模式 i 和模式 j 共同召回的块数。
    ///
    /// 这个矩阵能直观看出：
    ///
    /// - 对角线 = 该模式自己召回的块数
    /// - 非对角线 = 两个模式的「共识部分」
    /// - 全矩阵都高 → 混合检索没带来新信息
    /// - 某行明显偏低 → 那个模式的视角独特，有价值
    pub overlap: Vec<Vec<usize>>,

    /// 命中的模式（`expected` 给了才有意义）。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub successful_modes: Vec<String>,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 单个模式的对比结果。
#[derive(Debug, Clone, Serialize)]
pub struct ModeResult {
    /// 模式名。
    pub mode: SearchMode,

    /// 最终命中（重排后）。
    pub hits: Vec<HybridHit>,

    /// 命中的块 ID 列表（方便做重叠度计算和人工核对）。
    pub hit_ids: Vec<String>,

    /// 是否命中了 `expected` 里的关键词（没给 `expected` 时为 `None`）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hit: Option<bool>,

    /// 是否执行了重排。
    pub reranked: bool,

    /// 降级原因（如果有）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,

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

    /// BM25 索引里的文档数。
    pub bm25_documents: usize,

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
