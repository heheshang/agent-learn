//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见 ——
//! 想知道服务能接什么、返回什么，看这一个文件就够了。

use serde::{Deserialize, Serialize};

/// 检索模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// 纯稠密向量检索。
    Dense,
    /// 服务端混合检索（稠密 + BM25，RRF 融合）。
    Hybrid,
    /// 纯关键词检索（Milvus 的 BM25 稀疏向量）。
    Keyword,
}

impl SearchMode {
    /// 字符串形式。
    pub fn as_str(&self) -> &'static str {
        match self {
            SearchMode::Dense => "dense",
            SearchMode::Hybrid => "hybrid",
            SearchMode::Keyword => "keyword",
        }
    }

    /// 从字符串解析。
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "dense" | "vector" => Some(SearchMode::Dense),
            "hybrid" | "mixed" => Some(SearchMode::Hybrid),
            "keyword" | "bm25" | "text" => Some(SearchMode::Keyword),
            _ => None,
        }
    }
}

/// 检索请求体。
///
/// ```json
/// { "question": "什么是 RAG？", "mode": "dense", "top_k": 5 }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    /// 查询串。
    pub question: String,

    /// 检索模式（可选，默认 dense）。
    #[serde(default)]
    pub mode: Option<String>,

    /// 返回条数（可选，默认用配置里的 top_k）。
    #[serde(default)]
    pub top_k: Option<usize>,

    /// 服务端过滤表达式（可选），例如 `source == "kb.md"`。
    #[serde(default)]
    pub filter: Option<String>,
}

/// 一条命中。
#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    /// 块 ID（`{文件名}#{序号}`）。
    pub id: String,
    /// 相似度分数（越大越相似）。
    pub score: f32,
    /// 来源文件。
    pub source: String,
    /// 标题路径。
    pub heading_path: Vec<String>,
    /// 块正文。
    pub text: String,
}

impl From<rag_05::store::SearchHit> for Hit {
    fn from(hit: rag_05::store::SearchHit) -> Self {
        Self {
            id: hit.id,
            score: hit.score,
            source: hit.source,
            heading_path: hit.heading_path,
            text: hit.text,
        }
    }
}

/// 检索响应体。
#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    /// 原问题。
    pub question: String,
    /// 检索模式。
    pub mode: SearchMode,
    /// 后端。
    pub backend: String,
    /// 命中列表。
    pub hits: Vec<Hit>,
    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 单个后端的结果（对照用）。
#[derive(Debug, Clone, Serialize)]
pub struct BackendResult {
    /// 后端名。
    pub backend: String,
    /// 是否成功。
    pub ok: bool,
    /// 命中数。
    pub count: usize,
    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
    /// 命中列表。
    pub hits: Vec<Hit>,
    /// 失败原因。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// 对照请求体。
///
/// ```json
/// { "question": "混合检索怎么融合两路结果", "top_k": 5 }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CompareRequest {
    /// 查询串。
    pub question: String,

    /// 返回条数。
    #[serde(default)]
    pub top_k: Option<usize>,
}

/// 对照响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CompareResponse {
    /// 原问题。
    pub question: String,
    /// 各自的检索结果。
    pub results: Vec<BackendResult>,
    /// 两个后端命中的重合度（0~1）。
    pub overlap: f32,
    /// 重合的块 ID。
    pub shared_ids: Vec<String>,
    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 重建索引响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ReindexResponse {
    /// 后端。
    pub backend: String,
    /// 灌入条数。
    pub chunks: usize,
    /// 服务端条数。
    pub server_count: usize,
    /// collection 名。
    pub collection: String,
    /// 是否新建。
    pub created: bool,
    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 统一错误响应。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// HTTP 状态的规范名。
    pub error: String,
    /// 详细说明。
    pub message: String,
}
