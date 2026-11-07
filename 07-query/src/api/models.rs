//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见 ——
//! 想知道服务能接什么、返回什么，看这一个文件就够了。

use serde::{Deserialize, Serialize};

use crate::fusion::{FusedHit, RankedRun};
use crate::transform::{Strategy, TransformedQuery};

/// 问答请求体。
///
/// ```json
/// { "question": "什么是 RAG？", "strategy": "multi_query" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct AskRequest {
    /// 用户的问题。
    pub question: String,

    /// 使用的改写策略（可选，默认用配置里的）。
    #[serde(default)]
    pub strategy: Option<String>,

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
    pub citations: Vec<crate::pipeline::Citation>,

    /// 改写后的查询（看模型把问题改成了什么）。
    pub transformed_queries: Vec<TransformedQuery>,

    /// 融合后的命中（`include_context=true` 时才有内容）。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hits: Vec<FusedHit>,

    /// 是否降级（改写失败退回原问题）。
    pub degraded: bool,

    /// 降级原因。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,

    /// 改写耗时（毫秒）。
    pub transform_ms: u64,

    /// 检索耗时（毫秒）。
    pub retrieve_ms: u64,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 纯检索请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    /// 问题。
    pub question: String,

    /// 使用的改写策略（可选，默认用配置里的）。
    #[serde(default)]
    pub strategy: Option<String>,

    /// 融合方法（可选，默认用配置里的）。
    #[serde(default)]
    pub fusion: Option<String>,

    /// 是否返回每路的原始结果（可选，默认 false）。
    #[serde(default)]
    pub include_runs: bool,
}

/// 纯检索响应体。
#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    /// 原始问题。
    pub question: String,

    /// 使用的策略。
    pub strategy: Strategy,

    /// 改写后的查询。
    pub transformed_queries: Vec<TransformedQuery>,

    /// 融合后的命中。
    pub hits: Vec<FusedHit>,

    /// 每路的原始结果（`include_runs=true` 时才有内容）。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub runs: Vec<RankedRun>,

    /// 是否降级。
    pub degraded: bool,

    /// 降级原因。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 只改写的请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct TransformRequest {
    /// 问题。
    pub question: String,

    /// 使用的策略（可选，默认用配置里的）。
    #[serde(default)]
    pub strategy: Option<String>,
}

/// 只改写的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct TransformResponse {
    /// 原始问题。
    pub question: String,

    /// 使用的策略。
    pub strategy: Strategy,

    /// 改写后的查询。
    pub transformed_queries: Vec<TransformedQuery>,

    /// 是否降级。
    pub degraded: bool,

    /// 降级原因。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 多策略对比请求体。
///
/// ```json
/// {
///   "question": "模型的知识有截止日期怎么办",
///   "strategies": ["none", "hyde", "multi_query", "decompose"],
///   "expected": ["检索增强生成"]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CompareRequest {
    /// 问题。
    pub question: String,

    /// 要对比的策略列表（可选，默认全部）。
    #[serde(default)]
    pub strategies: Option<Vec<String>>,

    /// 期望命中的关键词（可选）——
    ///
    /// 给了的话，响应里每个策略会带一个 `hit` 布尔值：
    /// 该策略的召回里是否出现了含这些关键词的块。
    ///
    /// 这是本项目最实用的「量化收益」方式 ——
    /// 不用人工判读，直接看哪个策略把正确块捞上来了。
    #[serde(default)]
    pub expected: Option<Vec<String>>,

    /// 融合方法（可选，默认用配置里的）。
    #[serde(default)]
    pub fusion: Option<String>,

    /// 是否返回每路的原始结果（可选，默认 false）。
    #[serde(default)]
    pub include_runs: bool,
}

/// 多策略对比响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CompareResponse {
    /// 原始问题。
    pub question: String,

    /// 每个策略的结果。
    pub results: Vec<StrategyResult>,

    /// 策略之间的召回重叠度矩阵。
    ///
    /// `overlap[i][j]` = 策略 i 和策略 j 共同召回的块数。
    ///
    /// 这个矩阵能直观看出：
    ///
    /// - 对角线 = 该策略自己召回的块数
    /// - 非对角线 = 两个策略的「共识部分」
    /// - 全矩阵都高 → 改写没带来新信息（白花 LLM 钱）
    /// - 某行明显偏低 → 那个策略的视角独特，有价值
    pub overlap: Vec<Vec<usize>>,

    /// 命中的策略（`expected` 给了才有意义）。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub successful_strategies: Vec<String>,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 单个策略的对比结果。
#[derive(Debug, Clone, Serialize)]
pub struct StrategyResult {
    /// 策略名。
    pub strategy: Strategy,

    /// 改写后的查询。
    pub transformed_queries: Vec<TransformedQuery>,

    /// 融合后的命中。
    pub hits: Vec<FusedHit>,

    /// 命中的块 ID 列表（方便做重叠度计算和人工核对）。
    pub hit_ids: Vec<String>,

    /// 是否命中了 `expected` 里的关键词（没给 `expected` 时为 `None`）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hit: Option<bool>,

    /// 是否降级。
    pub degraded: bool,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 每路的原始结果（`include_runs=true` 时才有内容）。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub runs: Vec<RankedRun>,
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
