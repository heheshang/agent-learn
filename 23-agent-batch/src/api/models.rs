//! API 的请求 / 响应模型。

use serde::{Deserialize, Serialize};

use crate::batching::BatchStrategy;

/// 批量嵌入请求。
#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    /// 要嵌入的文本列表。
    pub texts: Vec<String>,

    /// 批策略（默认 `fixed`）。
    #[serde(default)]
    pub strategy: Option<String>,
}

/// 批量嵌入响应。
#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    /// 向量列表（顺序与输入一致）。
    pub vectors: Vec<Vec<f32>>,

    /// 条数。
    pub count: usize,

    /// 向量维度。
    pub dimension: usize,

    /// 度量。
    pub stats: crate::metrics::BatchStats,

    /// 使用的策略。
    pub strategy: BatchStrategy,
}

/// 批量重排请求。
#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    /// 查询。
    pub query: String,

    /// 候选文档。
    pub documents: Vec<RerankDocRequest>,

    /// 保留多少条（默认 3）。
    #[serde(default)]
    pub top_n: Option<usize>,

    /// 批策略（默认 `fixed`）。
    #[serde(default)]
    pub strategy: Option<String>,
}

/// 请求里的一条候选文档。
#[derive(Debug, Deserialize)]
pub struct RerankDocRequest {
    /// 文档 ID。
    pub id: String,

    /// 来源文件名。
    #[serde(default)]
    pub source: String,

    /// 标题路径。
    #[serde(default)]
    pub heading_path: Vec<String>,

    /// 正文。
    pub text: String,
}

impl RerankDocRequest {
    /// 转成 08 的候选块。
    pub fn to_rerank_document(&self) -> rag_08::reranker::RerankDocument {
        rag_08::reranker::RerankDocument::new(
            self.id.clone(),
            self.source.clone(),
            self.heading_path.clone(),
            self.text.clone(),
        )
    }
}

/// 批量重排响应。
#[derive(Debug, Serialize)]
pub struct RerankResponse {
    /// 重排后的命中。
    pub hits: Vec<rag_08::reranker::RerankedHit>,

    /// 候选数。
    pub candidate_count: usize,

    /// 度量。
    pub stats: crate::metrics::BatchStats,

    /// 使用的策略。
    pub strategy: BatchStrategy,
}

/// 嵌入对照实验请求。
#[derive(Debug, Deserialize)]
pub struct BenchEmbedRequest {
    /// 样本条数（默认取配置）。
    #[serde(default)]
    pub sample_size: Option<usize>,

    /// 轮数（默认取配置）。
    #[serde(default)]
    pub rounds: Option<usize>,
}

/// 重排对照实验请求。
#[derive(Debug, Deserialize)]
pub struct BenchRerankRequest {
    /// 查询。
    #[serde(default = "default_query")]
    pub query: String,

    /// 样本条数。
    #[serde(default)]
    pub sample_size: Option<usize>,

    /// 保留多少条。
    #[serde(default)]
    pub top_n: Option<usize>,

    /// 轮数。
    #[serde(default)]
    pub rounds: Option<usize>,
}

/// 默认查询。
fn default_query() -> String {
    "批处理如何提升吞吐量".to_string()
}

/// 批大小扫描请求。
#[derive(Debug, Deserialize)]
pub struct BenchSweepRequest {
    /// 样本条数。
    #[serde(default)]
    pub sample_size: Option<usize>,

    /// 要扫描的批大小档位。
    #[serde(default)]
    pub batch_sizes: Option<Vec<usize>>,

    /// 轮数。
    #[serde(default)]
    pub rounds: Option<usize>,
}

/// 远程对照实验请求。
#[derive(Debug, Deserialize)]
pub struct BenchRemoteRequest {
    /// 样本条数。
    #[serde(default)]
    pub sample_size: Option<usize>,

    /// 批大小。
    #[serde(default)]
    pub batch_size: Option<usize>,
}

/// 批量建索引请求。
#[derive(Debug, Deserialize)]
pub struct IndexRequest {
    /// 文档列表（id / text / heading_path）。
    pub documents: Vec<IndexDocRequest>,

    /// 批策略。
    #[serde(default)]
    pub strategy: Option<String>,
}

/// 一条待索引的文档。
#[derive(Debug, Deserialize)]
pub struct IndexDocRequest {
    /// 文档 ID。
    pub id: String,

    /// 正文。
    pub text: String,

    /// 标题路径。
    #[serde(default)]
    pub heading_path: Vec<String>,
}

/// 建索引响应。
#[derive(Debug, Serialize)]
pub struct IndexResponse {
    /// 文档数。
    pub doc_count: usize,

    /// 向量维度。
    pub dimension: usize,

    /// 报告。
    pub report: crate::pipeline::IndexReport,
}

/// 批量检索请求。
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// 查询。
    pub query: String,

    /// 召回数。
    #[serde(default)]
    pub top_k: Option<usize>,

    /// 精排保留数。
    #[serde(default)]
    pub top_n: Option<usize>,

    /// 批策略。
    #[serde(default)]
    pub strategy: Option<String>,
}

/// 统一错误响应。
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// 错误码。
    pub code: String,

    /// 错误消息。
    pub message: String,
}
