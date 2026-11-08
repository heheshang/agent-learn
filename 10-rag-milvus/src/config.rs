//! 全局配置。
//!
//! 所有「可调参数」集中在这里，来源有两档（后者覆盖前者）：
//!
//! ```text
//! 默认值  <  环境变量
//! ```
//!
//! 环境变量名带 `MILVUS_` 前缀，避免和其它项目撞名
//! （06 用 `RAG_`，07 用 `QUERY_`，08 用 `HYBRID_`）。
//!
//! ## 三组配置
//!
//! | 组 | 作用 | 关键项 |
//! |---|---|---|
//! | [`MilvusConfig`] | Milvus 连接与 collection 形态 | `uri` / `token` / `collection` / `index_type` |
//! | [`QdrantConfig`] | Qdrant 连接与 collection 形态 | `url` / `api_key` / `collection` |
//! | [`AppConfig`] | 应用级（数据目录、嵌入模型、监听地址） | `data_dir` / `embedding_model` / `bind_addr` |

use std::path::PathBuf;

use rag_05::embed::Model;

/// Milvus 索引类型。
///
/// Milvus 服务端的索引构建器不是随便选的 —— 不同索引对应
/// 不同的「数据规模 / 延迟 / 召回」取舍。
///
/// | 类型 | 原理 | 适用 |
/// |---|---|---|
/// | [`IndexKind::Flat`] | 暴力遍历，无索引 | 数据量小（<1万），要 100% 精确 |
/// | [`IndexKind::Hnsw`] | 多层可导航小世界图 | **默认**。大规模、高召回、低延迟 |
/// | [`IndexKind::IvfFlat`] | k-means 聚类剪枝 | 十万级，内存吃紧 |
/// | [`IndexKind::DiskAnn`] | 磁盘图索引 | 十亿级，内存放不下 |
/// | [`IndexKind::Scann`] | 量化 + 重排 | 极高召回要求 |
/// | [`IndexKind::AutoIndex`] | 让服务端自己选 | 不想操心时 |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// 暴力精确检索。
    Flat,
    /// 分层图索引。
    Hnsw,
    /// 倒排文件索引。
    IvfFlat,
    /// 磁盘图索引。
    DiskAnn,
    /// 量化扫描索引。
    Scann,
    /// 服务端自动选择。
    AutoIndex,
}

impl IndexKind {
    /// Milvus `IndexType` 的字符串形式（用于日志与配置展示）。
    pub fn as_str(&self) -> &'static str {
        match self {
            IndexKind::Flat => "FLAT",
            IndexKind::Hnsw => "HNSW",
            IndexKind::IvfFlat => "IVF_FLAT",
            IndexKind::DiskAnn => "DISKANN",
            IndexKind::Scann => "SCANN",
            IndexKind::AutoIndex => "AUTOINDEX",
        }
    }

    /// 从字符串解析。
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.to_ascii_uppercase().as_str() {
            "FLAT" => Some(IndexKind::Flat),
            "HNSW" => Some(IndexKind::Hnsw),
            "IVF_FLAT" | "IVFFLAT" => Some(IndexKind::IvfFlat),
            "DISKANN" => Some(IndexKind::DiskAnn),
            "SCANN" => Some(IndexKind::Scann),
            "AUTOINDEX" | "AUTO" => Some(IndexKind::AutoIndex),
            _ => None,
        }
    }

    /// 全部可选值（用于错误提示）。
    pub fn all() -> &'static [IndexKind] {
        &[
            IndexKind::Flat,
            IndexKind::Hnsw,
            IndexKind::IvfFlat,
            IndexKind::DiskAnn,
            IndexKind::Scann,
            IndexKind::AutoIndex,
        ]
    }
}

/// Milvus 连接与 collection 配置。
#[derive(Debug, Clone)]
pub struct MilvusConfig {
    /// gRPC 地址。
    pub uri: String,

    /// 认证 token（`用户名:密码`）。默认 `root:Milvus`。
    pub token: String,

    /// 数据库名（Milvus 2.4+ 支持多数据库）。
    pub database: String,

    /// collection 名。
    pub collection: String,

    /// 向量字段名。
    pub vector_field: String,

    /// 主键字段名。
    pub primary_field: String,

    /// 标量字段名（存 chunk id）。
    pub id_field: String,

    /// 原文字段名。
    pub text_field: String,

    /// 标题路径字段名（JSON 数组）。
    pub heading_field: String,

    /// 索引类型。
    pub index_type: IndexKind,

    /// 距离度量。
    ///
    /// 我们的向量已经 L2 归一化，所以 `COSINE` 得到的分数就是余弦相似度
    /// —— 和 05-rag 的 FAISS `InnerProduct` 语义完全一致。
    pub metric: MetricKind,

    /// 是否启用服务端 BM25（混合检索）。
    pub enable_bm25: bool,

    /// BM25 稀疏向量字段名。
    pub sparse_field: String,
}

/// 距离度量。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// 余弦相似度（越大越相似）。
    Cosine,
    /// 内积（越大越相似）。
    Ip,
    /// L2 距离（越小越相似）。
    L2,
}

impl MetricKind {
    /// Milvus `MetricType` 的字符串形式。
    pub fn as_str(&self) -> &'static str {
        match self {
            MetricKind::Cosine => "COSINE",
            MetricKind::Ip => "IP",
            MetricKind::L2 => "L2",
        }
    }

    /// 从字符串解析。
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.to_ascii_uppercase().as_str() {
            "COSINE" => Some(MetricKind::Cosine),
            "IP" => Some(MetricKind::Ip),
            "L2" => Some(MetricKind::L2),
            _ => None,
        }
    }
}

impl Default for MilvusConfig {
    fn default() -> Self {
        Self {
            uri: "http://127.0.0.1:19530".to_string(),
            token: "root:Milvus".to_string(),
            database: "default".to_string(),
            collection: "rag_chunks".to_string(),
            vector_field: "vector".to_string(),
            primary_field: "pk".to_string(),
            id_field: "chunk_id".to_string(),
            text_field: "text".to_string(),
            heading_field: "heading_path".to_string(),
            index_type: IndexKind::Hnsw,
            metric: MetricKind::Cosine,
            enable_bm25: true,
            sparse_field: "sparse".to_string(),
        }
    }
}

/// Qdrant 连接与 collection 配置。
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// gRPC 地址。
    pub url: String,

    /// API key（Qdrant Cloud 需要；本地留空）。
    pub api_key: Option<String>,

    /// collection 名。
    pub collection: String,

    /// 向量字段名（Qdrant 叫 named vectors）。
    pub vector_field: String,

    /// 距离度量。
    pub metric: MetricKind,

    /// HNSW 的每层连接数。
    pub hnsw_m: u32,

    /// HNSW 的构建期候选数。
    pub hnsw_ef_construct: u32,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:6334".to_string(),
            api_key: None,
            collection: "rag_chunks".to_string(),
            vector_field: "vector".to_string(),
            metric: MetricKind::Cosine,
            hnsw_m: 16,
            hnsw_ef_construct: 100,
        }
    }
}

/// 用哪个向量库。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Milvus。
    Milvus,
    /// Qdrant。
    Qdrant,
}

impl BackendKind {
    /// 字符串形式。
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Milvus => "milvus",
            BackendKind::Qdrant => "qdrant",
        }
    }

    /// 从字符串解析。
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "milvus" => Some(BackendKind::Milvus),
            "qdrant" => Some(BackendKind::Qdrant),
            _ => None,
        }
    }

    /// 全部可选值。
    pub fn all() -> &'static [BackendKind] {
        &[BackendKind::Milvus, BackendKind::Qdrant]
    }
}

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// 知识库文档目录。
    pub data_dir: PathBuf,

    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 嵌入模型。
    pub embedding_model: Model,

    /// 使用的向量库。
    pub backend: BackendKind,

    /// Milvus 配置。
    pub milvus: MilvusConfig,

    /// Qdrant 配置。
    pub qdrant: QdrantConfig,

    /// 检索返回条数。
    pub top_k: usize,

    /// 每路召回条数（混合检索时）。
    pub per_path_k: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            data_dir: manifest.join("data"),
            bind_addr: "127.0.0.1:3030".to_string(),
            embedding_model: Model::MultilingualE5Small,
            backend: BackendKind::Milvus,
            milvus: MilvusConfig::default(),
            qdrant: QdrantConfig::default(),
            top_k: 5,
            per_path_k: 20,
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置（未设置的项用默认值）。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `MILVUS_DATA_DIR` | 知识库目录 | `10-rag-milvus/data` |
    /// | `MILVUS_BIND_ADDR` | 监听地址 | `127.0.0.1:3030` |
    /// | `MILVUS_BACKEND` | `milvus` / `qdrant` | `milvus` |
    /// | `MILVUS_URI` | Milvus gRPC 地址 | `http://127.0.0.1:19530` |
    /// | `MILVUS_TOKEN` | Milvus 认证 token | `root:Milvus` |
    /// | `MILVUS_DATABASE` | Milvus 数据库 | `default` |
    /// | `MILVUS_COLLECTION` | collection 名 | `rag_chunks` |
    /// | `MILVUS_INDEX_TYPE` | `flat`/`hnsw`/`ivf_flat`/`diskann`/`scann`/`auto` | `hnsw` |
    /// | `MILVUS_METRIC` | `cosine`/`ip`/`l2` | `cosine` |
    /// | `MILVUS_ENABLE_BM25` | 是否启用服务端 BM25 | `true` |
    /// | `QDRANT_URL` | Qdrant gRPC 地址 | `http://127.0.0.1:6334` |
    /// | `QDRANT_API_KEY` | Qdrant API key | 空 |
    /// | `QDRANT_COLLECTION` | collection 名 | `rag_chunks` |
    /// | `MILVUS_EMBEDDING_MODEL` | `e5` / `bge` / `minilm` | `e5` |
    /// | `MILVUS_TOP_K` | 检索返回条数 | `5` |
    /// | `MILVUS_PER_PATH_K` | 每路召回数 | `20` |
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Some(dir) = env_string("MILVUS_DATA_DIR") {
            config.data_dir = PathBuf::from(dir);
        }
        if let Some(addr) = env_string("MILVUS_BIND_ADDR") {
            config.bind_addr = addr;
        }
        if let Some(backend) = env_string("MILVUS_BACKEND")
            && let Some(kind) = BackendKind::parse(&backend)
        {
            config.backend = kind;
        }
        if let Some(uri) = env_string("MILVUS_URI") {
            config.milvus.uri = uri;
        }
        if let Some(token) = env_string("MILVUS_TOKEN") {
            config.milvus.token = token;
        }
        if let Some(db) = env_string("MILVUS_DATABASE") {
            config.milvus.database = db;
        }
        if let Some(collection) = env_string("MILVUS_COLLECTION") {
            config.milvus.collection = collection.clone();
            config.qdrant.collection = collection;
        }
        if let Some(index) = env_string("MILVUS_INDEX_TYPE")
            && let Some(kind) = IndexKind::parse(&index)
        {
            config.milvus.index_type = kind;
        }
        if let Some(metric) = env_string("MILVUS_METRIC")
            && let Some(kind) = MetricKind::parse(&metric)
        {
            config.milvus.metric = kind;
            config.qdrant.metric = kind;
        }
        if let Some(enabled) = env_string("MILVUS_ENABLE_BM25") {
            config.milvus.enable_bm25 = enabled != "0" && enabled.to_lowercase() != "false";
        }
        if let Some(url) = env_string("QDRANT_URL") {
            config.qdrant.url = url;
        }
        if let Some(key) = env_string("QDRANT_API_KEY") {
            config.qdrant.api_key = Some(key);
        }
        if let Some(model) = env_string("MILVUS_EMBEDDING_MODEL") {
            config.embedding_model = match model.to_lowercase().as_str() {
                "bge" => Model::BGESmallZH,
                "minilm" => Model::MiniLM,
                _ => Model::MultilingualE5Small,
            };
        }
        if let Some(top_k) = env_usize("MILVUS_TOP_K") {
            config.top_k = top_k.max(1);
        }
        if let Some(k) = env_usize("MILVUS_PER_PATH_K") {
            config.per_path_k = k.max(1);
        }

        config
    }

    /// 配置摘要（启动时打印）。
    pub fn summary(&self) -> String {
        format!(
            "  后端        : {}\n  语料目录    : {}\n  监听地址    : {}\n  嵌入模型    : {} (dim={})\n  Milvus      : {} @ {}\n  Collection : {}\n  索引        : {} / {}\n  服务端 BM25 : {}\n  Qdrant      : {} @ {}",
            self.backend.as_str(),
            self.data_dir.display(),
            self.bind_addr,
            model_name(self.embedding_model),
            model_dimension(self.embedding_model),
            self.milvus.collection,
            self.milvus.uri,
            self.milvus.collection,
            self.milvus.index_type.as_str(),
            self.milvus.metric.as_str(),
            if self.milvus.enable_bm25 {
                "启用"
            } else {
                "关闭"
            },
            self.qdrant.collection,
            self.qdrant.url,
        )
    }
}

/// 读一个非空的环境变量。
fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

/// 读一个 usize 环境变量（解析失败就忽略）。
fn env_usize(key: &str) -> Option<usize> {
    env_string(key).and_then(|v| v.parse().ok())
}

/// 嵌入模型的显示名。
pub fn model_name(model: Model) -> &'static str {
    match model {
        Model::MiniLM => "sentence-transformers/all-MiniLM-L6-v2",
        Model::MultilingualE5Small => "intfloat/multilingual-e5-small",
        Model::BGESmallZH => "BAAI/bge-small-zh-v1.5",
    }
}

/// 嵌入模型的维度。
pub fn model_dimension(model: Model) -> usize {
    match model {
        Model::MiniLM => 384,
        Model::MultilingualE5Small => 384,
        Model::BGESmallZH => 512,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_kind_round_trips() {
        for kind in IndexKind::all() {
            assert_eq!(IndexKind::parse(kind.as_str()), Some(*kind));
        }
    }

    #[test]
    fn unknown_index_kind_is_rejected() {
        assert_eq!(IndexKind::parse("nope"), None);
    }

    #[test]
    fn backend_kind_round_trips() {
        for kind in BackendKind::all() {
            assert_eq!(BackendKind::parse(kind.as_str()), Some(*kind));
        }
    }

    #[test]
    fn metric_kind_round_trips() {
        assert_eq!(MetricKind::parse("COSINE"), Some(MetricKind::Cosine));
        assert_eq!(MetricKind::parse("ip"), Some(MetricKind::Ip));
        assert_eq!(MetricKind::parse("L2"), Some(MetricKind::L2));
        assert_eq!(MetricKind::parse("hamming"), None);
    }
}
