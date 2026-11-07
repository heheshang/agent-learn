//! 全局配置。
//!
//! 所有「可调参数」集中在这里，来源有三档（后者覆盖前者）：
//!
//! ```text
//! 默认值  <  环境变量  <  配置文件
//! ```
//!
//! 环境变量名带 `RAG_` 前缀，避免和其它程序的变量撞名。
//!
//! ## 为什么要把参数集中
//!
//! RAG 系统里最容易被「悄悄改坏」的参数是 `top_k` 和 `chunk_size`：
//! 改一个数字，检索结果就变了，答案质量跟着变，但没有任何报错。
//! 把它们收拢到一处 + 启动时打印出来，至少能保证「出问题时知道当前配置是什么」。

use std::path::PathBuf;

use rag_04::splitter::SplitterConfig;

/// 检索参数。
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// 召回的块数（top-k）。
    ///
    /// 太小 → 上下文不够，模型答不全；太大 → 引入噪声块，
    /// 反而干扰生成，还烧 token。经验值 3~5。
    pub top_k: usize,

    /// 相似度下限。低于这个分数的命中块被丢弃。
    ///
    /// ⚠️ Naive RAG 默认**关闭**（`None`）—— 这正是「朴素」的含义之一：
    /// 检索到什么就用什么，不做「都不相关就拒答」的判断。
    /// 设成 `Some(0.5)` 就能体验「带阈值的检索」。
    pub min_score: Option<f32>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            min_score: None,
        }
    }
}

/// 生成参数。
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// 生成用的模型名（`genai` 按前缀选适配器，`deepseek-*` → DeepSeek）。
    pub model: String,

    /// 采样温度。RAG 问答建议低一点（0.0~0.3），保证答案稳定可复现。
    pub temperature: f64,

    /// 单次生成的最大 token 数。
    ///
    /// ⚠️ DeepSeek 的推理模型会把「思考」也算进 completion tokens，
    /// 所以这个值不能太小 —— 实测给到 2000 才够它先思考再回答。
    pub max_tokens: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            model: "deepseek-flash".to_string(),
            temperature: 0.1,
            max_tokens: 2000,
        }
    }
}

/// 嵌入模型选择。
///
/// 复用 05-rag 的 `Model` 枚举 —— 换模型 = 换索引，必须重建。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModel {
    /// `intfloat/multilingual-e5-small`（384 维，多语言）。
    MultilingualE5Small,
    /// `BAAI/bge-small-zh-v1.5`（512 维，中文）。
    BgeSmallZh,
}

impl EmbeddingModel {
    /// 转换成 05-rag 的模型枚举。
    pub fn to_model(self) -> rag_05::Model {
        match self {
            EmbeddingModel::MultilingualE5Small => rag_05::Model::MultilingualE5Small,
            EmbeddingModel::BgeSmallZh => rag_05::Model::BGESmallZH,
        }
    }

    /// 从环境变量解析：`e5` / `bge`。
    pub fn from_env_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "e5" | "multilingual-e5-small" | "e5-small" => {
                Some(EmbeddingModel::MultilingualE5Small)
            }
            "bge" | "bge-small-zh" | "bge-zh" => Some(EmbeddingModel::BgeSmallZh),
            _ => None,
        }
    }
}

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// 知识库文档目录。
    pub data_dir: PathBuf,

    /// 索引文件路径（FAISS）。
    pub index_path: PathBuf,

    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 嵌入模型。
    pub embedding_model: EmbeddingModel,

    /// 分块配置。
    pub splitter: SplitterConfig,

    /// 检索配置。
    pub retrieval: RetrievalConfig,

    /// 生成配置。
    pub generation: GenerationConfig,

    /// 启动时是否自动建索引（索引不存在时）。
    pub auto_index: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            data_dir: manifest.join("../04-rag/data"),
            index_path: manifest.join("index/index.faiss"),
            bind_addr: "127.0.0.1:3000".to_string(),
            embedding_model: EmbeddingModel::MultilingualE5Small,
            splitter: SplitterConfig::default(),
            retrieval: RetrievalConfig::default(),
            generation: GenerationConfig::default(),
            auto_index: true,
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置（未设置的项用默认值）。
    ///
    /// 支持的环境变量：
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `RAG_DATA_DIR` | 知识库目录 | `../04-rag/data` |
    /// | `RAG_INDEX_PATH` | 索引文件路径 | `index/index.faiss` |
    /// | `RAG_BIND_ADDR` | 监听地址 | `127.0.0.1:3000` |
    /// | `RAG_EMBEDDING_MODEL` | `e5` / `bge` | `e5` |
    /// | `RAG_TOP_K` | 召回块数 | `4` |
    /// | `RAG_MIN_SCORE` | 相似度下限（可空） | 空 |
    /// | `RAG_MODEL` | 生成模型 | `deepseek-flash` |
    /// | `RAG_TEMPERATURE` | 采样温度 | `0.1` |
    /// | `RAG_MAX_TOKENS` | 最大生成 token | `2000` |
    /// | `RAG_CHUNK_SIZE` | 块大小（token） | `300` |
    /// | `RAG_CHUNK_OVERLAP` | 块重叠（token） | `30` |
    /// | `RAG_AUTO_INDEX` | 启动时自动建索引 | `true` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("RAG_DATA_DIR") {
            config.data_dir = PathBuf::from(v);
        }
        if let Some(v) = env_str("RAG_INDEX_PATH") {
            config.index_path = PathBuf::from(v);
        }
        if let Some(v) = env_str("RAG_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("RAG_EMBEDDING_MODEL")
            && let Some(m) = EmbeddingModel::from_env_value(&v)
        {
            config.embedding_model = m;
        }
        if let Some(v) = env_usize("RAG_TOP_K") {
            config.retrieval.top_k = v;
        }
        if let Some(v) = env_f32("RAG_MIN_SCORE") {
            config.retrieval.min_score = Some(v);
        }
        if let Some(v) = env_str("RAG_MODEL") {
            config.generation.model = v;
        }
        if let Some(v) = env_f64("RAG_TEMPERATURE") {
            config.generation.temperature = v;
        }
        if let Some(v) = env_u32("RAG_MAX_TOKENS") {
            config.generation.max_tokens = v;
        }
        if let Some(v) = env_usize("RAG_CHUNK_SIZE") {
            config.splitter.chunk_size = v;
        }
        if let Some(v) = env_usize("RAG_CHUNK_OVERLAP") {
            config.splitter.chunk_overlap = v;
        }
        if let Some(v) = env_bool("RAG_AUTO_INDEX") {
            config.auto_index = v;
        }

        config
    }

    /// 打印当前配置（启动时调用，方便复现问题）。
    pub fn summary(&self) -> String {
        format!(
            "数据目录: {}\n索引路径: {}\n监听地址: {}\n嵌入模型: {:?}\n\
             分块: {} token / {} 重叠\n检索: top_k={} min_score={:?}\n生成: {} temp={} max_tokens={}",
            self.data_dir.display(),
            self.index_path.display(),
            self.bind_addr,
            self.embedding_model,
            self.splitter.chunk_size,
            self.splitter.chunk_overlap,
            self.retrieval.top_k,
            self.retrieval.min_score,
            self.generation.model,
            self.generation.temperature,
            self.generation.max_tokens,
        )
    }
}

/// 读一个非空环境变量。
fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// 读一个 `usize` 环境变量（解析失败就忽略，保留默认值）。
fn env_usize(key: &str) -> Option<usize> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `u32` 环境变量。
fn env_u32(key: &str) -> Option<u32> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `f32` 环境变量。
fn env_f32(key: &str) -> Option<f32> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `f64` 环境变量。
fn env_f64(key: &str) -> Option<f64> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个布尔环境变量（`1/true/yes/on` 为真）。
fn env_bool(key: &str) -> Option<bool> {
    env_str(key).map(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_embedding_model_names() {
        assert_eq!(
            EmbeddingModel::from_env_value("e5"),
            Some(EmbeddingModel::MultilingualE5Small)
        );
        assert_eq!(
            EmbeddingModel::from_env_value(" BGE "),
            Some(EmbeddingModel::BgeSmallZh)
        );
        assert_eq!(EmbeddingModel::from_env_value("unknown"), None);
    }

    #[test]
    fn default_top_k_is_sane() {
        assert!((1..=10).contains(&RetrievalConfig::default().top_k));
    }
}
