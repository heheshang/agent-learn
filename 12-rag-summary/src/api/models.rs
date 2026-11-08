//! API 的请求/响应模型。
//!
//! ## 为什么要单独一个模块
//!
//! 处理器（handler）只做「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `retriever` / `pipeline` 里。
//! 所以这些类型是**纯数据结构**，不含任何业务判断。
//!
//! ## 统一错误响应
//!
//! 所有错误（参数错误、解析错误、后端故障）都是同一种格式：
//!
//! ```json
//! { "error": "Bad Request", "message": "请求参数不合法: 问题不能为空" }
//! ```

use serde::{Deserialize, Serialize};

use crate::config::SearchMode;
use crate::retriever::Hit;

/// 统一错误响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// 错误类别（HTTP 状态码的规范名称）。
    pub error: String,

    /// 错误详情。
    pub message: String,
}

/// 检索请求。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SearchRequest {
    /// 查询文本。
    pub question: String,

    /// 检索模式（缺省用配置里的默认模式）。
    pub mode: Option<String>,

    /// 返回条数（缺省用配置里的 `top_k`）。
    pub top_k: Option<usize>,
}

/// 检索响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// 原问题。
    pub question: String,

    /// 检索模式。
    pub mode: SearchMode,

    /// 实际走的检索路径。
    pub path: String,

    /// 命中的块。
    pub hits: Vec<Hit>,

    /// 检索耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 问答请求。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AskRequest {
    /// 问题。
    pub question: String,

    /// 检索模式（缺省用配置里的默认模式）。
    pub mode: Option<String>,
}

/// 问答响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskResponse {
    /// 答案。
    pub answer: String,

    /// 引用到的块。
    pub citations: Vec<crate::pipeline::Citation>,

    /// 进入上下文的片段（压缩后）。
    pub passages: Vec<Hit>,

    /// 上下文文本。
    pub context: String,

    /// 压缩统计。
    pub compression: crate::pipeline::CompressOutcome,

    /// 检索过程。
    pub retrieval: crate::pipeline::RetrievalOutcome,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 跑评估集的请求。
///
/// ## ⚠️ 和 `CompareRequest` 分开
///
/// `CompareRequest` 要求 `question`（同题对照用同一个问题跑各档）。
/// 但评估是**整个数据集**跑一遍，不需要问题 ——
/// 硬复用会逼调用方传一个无意义的 `question`，
/// 而且字段名会误导人以为「评估一个问题」。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct EvalRequest {
    /// 要评估的模式（缺省跑完整阶梯）。
    pub modes: Option<Vec<String>>,

    /// 每个变体最多跑多少个样本（缺省跑全量）。
    ///
    /// ⚠️ 全量是「22 样本 × 5 变体 × (1 次生成 + 8 个指标)」——
    /// 全是网络往返。调试时设个小数字能快速确认链路通不通。
    pub limit: Option<usize>,
}

/// 同题对照请求 —— 跑完整消融阶梯。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CompareRequest {
    /// 问题。
    pub question: String,

    /// 要对比的模式（缺省跑完整阶梯）。
    pub modes: Option<Vec<String>>,

    /// 每个变体最多跑多少个样本（缺省跑全量）。
    ///
    /// ⚠️ 全量是「22 样本 × 5 变体 × (1 次生成 + 8 个指标)」——
    /// 全是网络往返。调试时设个小数字能快速确认链路通不通。
    pub limit: Option<usize>,
}

/// 某个模式的对照结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareVariant {
    /// 模式名。
    pub mode: String,

    /// 是否成功。
    pub ok: bool,

    /// 命中的块。
    pub hits: Vec<Hit>,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 失败原因。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// 同题对照响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareResponse {
    /// 问题。
    pub question: String,

    /// 各模式的结果。
    pub variants: Vec<CompareVariant>,

    /// 各模式之间的**块 ID 重合度**。
    ///
    /// 这是「同一个问题，不同检索策略召回了多少相同内容」的量化指标。
    /// 重合度高 = 策略差异小；重合度低 = 某个策略有独特贡献。
    pub overlap: f32,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 重建索引响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReindexResponse {
    /// collection 名。
    pub collection: String,

    /// 灌进去多少条。
    pub chunks: usize,

    /// 服务端确认的条数。
    pub server_count: usize,

    /// 客户端 BM25 索引的文档数。
    pub bm25_docs: usize,

    /// 是否新建了 collection。
    pub created: bool,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}
