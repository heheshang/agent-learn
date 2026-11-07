//! 全局配置。
//!
//! 所有「可调参数」集中在这里，来源有两档（后者覆盖前者）：
//!
//! ```text
//! 默认值  <  环境变量
//! ```
//!
//! 环境变量名带 `HYBRID_` 前缀，避免和其它项目的变量撞名
//! （06 用 `RAG_`，07 用 `QUERY_`）。
//!
//! ## 为什么把参数集中
//!
//! 混合检索里最容易被「悄悄改坏」的是 `top_k`、`per_path_k` 和
//! `rerank_candidate_k` —— 改一个数字，检索结果就变了，
//! 答案质量跟着变，但没有任何报错。
//!
//! 把它们收拢到一处 + 启动时打印出来，至少能保证「出问题时知道当前配置是什么」。
//!
//! ## 复用 06 的类型
//!
//! `EmbeddingModel` 和 `GenerationConfig` 直接用 `rag_06::config` 里的 ——
//! 没必要抄一份，改一处就生效两处。

use std::path::PathBuf;

use rag_06::config::{EmbeddingModel, GenerationConfig};

use crate::hybrid::SearchMode;
use crate::reranker::{RerankBackend, RerankerModel};

/// 检索参数。
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// 最终保留的块数（top-k）。
    ///
    /// 太小 → 上下文不够，模型答不全；太大 → 引入噪声块，
    /// 反而干扰生成，还烧 token。经验值 3~5。
    pub top_k: usize,

    /// 每路召回多少条（融合前的候选池大小）。
    ///
    /// 默认 `None` = 自动取 `top_k * 2`（最低 8，上限 50）。
    pub per_path_k: Option<usize>,

    /// 检索模式。
    pub mode: SearchMode,

    /// 相似度下限。低于这个分数的命中块被丢弃（只对向量路生效）。
    pub min_score: Option<f32>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            per_path_k: None,
            mode: SearchMode::Hybrid,
            min_score: None,
        }
    }
}

impl RetrievalConfig {
    /// 实际使用的「每路召回数量」。
    ///
    /// 自动模式 = `top_k * 2`，最低 8，上限 50。
    ///
    /// 为什么是 2 倍：两路检索 + 融合后，真正有用的块可能来自任何一路。
    /// 每路只取 `top_k` 条的话，融合后可能挤满重复内容。
    pub fn effective_per_path_k(&self) -> usize {
        match self.per_path_k {
            Some(k) => k.clamp(1, 50),
            None => (self.top_k * 2).clamp(8, 50),
        }
    }
}

/// 重排参数。
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// 重排后端。
    pub backend: RerankBackend,

    /// 本地 reranker 的模型选择。
    pub model: RerankerModel,

    /// Cohere 的模型名（`rerank-multilingual-v3.0` / `rerank-english-v3.0`）。
    pub cohere_model: String,

    /// Cohere API key。
    pub cohere_api_key: Option<String>,

    /// 送进 reranker 的候选数（**召回阶段的上限**）。
    ///
    /// ⚠️ 这个值决定重排的**天花板** —— reranker 只能重排它拿到的东西。
    ///
    /// 教程里的例子是「召回 10 条 → 重排成 2 条」，这里默认 20：
    /// 候选越多，重排越有机会把真正相关的捞上来，但耗时也线性增长。
    ///
    /// 经验值 10~50。
    pub candidate_k: usize,

    /// 重排后保留多少条。
    ///
    /// 通常等于 `top_k`。但也可以更小（比如教程里的 `top_n=2`）——
    /// 让模型只看最精的少量片段。
    pub top_n: usize,

    /// 是否启用重排。
    ///
    /// 设为 `false` 就是**对照组**：只看融合后的原始顺序。
    pub enabled: bool,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            backend: RerankBackend::Local,
            model: RerankerModel::BgeRerankerBase,
            cohere_model: "rerank-multilingual-v3.0".to_string(),
            cohere_api_key: None,
            candidate_k: 20,
            top_n: 4,
            enabled: true,
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
    pub splitter: rag_04::splitter::SplitterConfig,

    /// 检索配置。
    pub retrieval: RetrievalConfig,

    /// 重排配置。
    pub rerank: RerankConfig,

    /// 生成配置。
    pub generation: GenerationConfig,

    /// 启动时是否自动建索引（索引不存在时）。
    pub auto_index: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            data_dir: manifest.join("data"),
            index_path: manifest.join("index/index.faiss"),
            bind_addr: "127.0.0.1:3020".to_string(),
            embedding_model: EmbeddingModel::MultilingualE5Small,
            splitter: rag_04::splitter::SplitterConfig::default(),
            retrieval: RetrievalConfig::default(),
            rerank: RerankConfig::default(),
            generation: GenerationConfig::default(),
            auto_index: true,
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置（未设置的项用默认值）。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `HYBRID_DATA_DIR` | 知识库目录 | `08-rag/data` |
    /// | `HYBRID_INDEX_PATH` | 索引文件路径 | `index/index.faiss` |
    /// | `HYBRID_BIND_ADDR` | 监听地址 | `127.0.0.1:3020` |
    /// | `HYBRID_EMBEDDING_MODEL` | `e5` / `bge` | `e5` |
    /// | `HYBRID_TOP_K` | 最终块数 | `4` |
    /// | `HYBRID_PER_PATH_K` | 每路召回数 | 自动（top_k × 2） |
    /// | `HYBRID_MODE` | `vector` / `bm25` / `hybrid` | `hybrid` |
    /// | `HYBRID_MIN_SCORE` | 相似度下限（可空） | 空 |
    /// | `HYBRID_RERANK_ENABLED` | 是否启用重排 | `true` |
    /// | `HYBRID_RERANK_BACKEND` | `local` / `cohere` | `local` |
    /// | `HYBRID_RERANK_MODEL` | `bge` / `m3` / `jina` | `bge` |
    /// | `COHERE_API_KEY` | Cohere API key | 空 |
    /// | `HYBRID_COHERE_MODEL` | Cohere 模型名 | `rerank-multilingual-v3.0` |
    /// | `HYBRID_RERANK_CANDIDATE_K` | 送进重排的候选数 | `20` |
    /// | `HYBRID_RERANK_TOP_N` | 重排后保留条数 | `4` |
    /// | `HYBRID_MODEL` | 生成模型 | `deepseek-flash` |
    /// | `HYBRID_TEMPERATURE` | 生成温度 | `0.1` |
    /// | `HYBRID_MAX_TOKENS` | 生成 max tokens | `2000` |
    /// | `HYBRID_AUTO_INDEX` | 启动时自动建索引 | `true` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("HYBRID_DATA_DIR") {
            config.data_dir = PathBuf::from(v);
        }
        if let Some(v) = env_str("HYBRID_INDEX_PATH") {
            config.index_path = PathBuf::from(v);
        }
        if let Some(v) = env_str("HYBRID_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("HYBRID_EMBEDDING_MODEL")
            && let Some(m) = EmbeddingModel::from_env_value(&v)
        {
            config.embedding_model = m;
        }
        if let Some(v) = env_usize("HYBRID_TOP_K") {
            config.retrieval.top_k = v;
        }
        if let Some(v) = env_usize("HYBRID_PER_PATH_K") {
            config.retrieval.per_path_k = Some(v);
        }
        if let Some(v) = env_str("HYBRID_MODE")
            && let Some(m) = SearchMode::parse(&v)
        {
            config.retrieval.mode = m;
        }
        if let Some(v) = env_f32("HYBRID_MIN_SCORE") {
            config.retrieval.min_score = Some(v);
        }
        if let Some(v) = env_bool("HYBRID_RERANK_ENABLED") {
            config.rerank.enabled = v;
        }
        if let Some(v) = env_str("HYBRID_RERANK_BACKEND")
            && let Some(b) = RerankBackend::from_env_value(&v)
        {
            config.rerank.backend = b;
        }
        if let Some(v) = env_str("HYBRID_RERANK_MODEL")
            && let Some(m) = RerankerModel::from_env_value(&v)
        {
            config.rerank.model = m;
        }
        if let Some(v) = env_str("COHERE_API_KEY") {
            config.rerank.cohere_api_key = Some(v);
        }
        if let Some(v) = env_str("HYBRID_COHERE_MODEL") {
            config.rerank.cohere_model = v;
        }
        if let Some(v) = env_usize("HYBRID_RERANK_CANDIDATE_K") {
            config.rerank.candidate_k = v.clamp(1, 200);
        }
        if let Some(v) = env_usize("HYBRID_RERANK_TOP_N") {
            config.rerank.top_n = v.clamp(1, 50);
        }
        if let Some(v) = env_str("HYBRID_MODEL") {
            config.generation.model = v;
        }
        if let Some(v) = env_f64("HYBRID_TEMPERATURE") {
            config.generation.temperature = v;
        }
        if let Some(v) = env_u32("HYBRID_MAX_TOKENS") {
            config.generation.max_tokens = v;
        }
        if let Some(v) = env_bool("HYBRID_AUTO_INDEX") {
            config.auto_index = v;
        }

        config
    }

    /// 打印当前配置（启动时调用，方便复现问题）。
    pub fn summary(&self) -> String {
        format!(
            "数据目录: {}\n索引路径: {}\n监听地址: {}\n嵌入模型: {:?}\n\
             分块: {} token / {} 重叠\n\
             检索: 模式={} top_k={} 每路召回={} min_score={:?}\n\
             重排: 启用={} 后端={} 模型={} 候选={} top_n={}\n\
             生成: {} temp={} max_tokens={}",
            self.data_dir.display(),
            self.index_path.display(),
            self.bind_addr,
            self.embedding_model,
            self.splitter.chunk_size,
            self.splitter.chunk_overlap,
            self.retrieval.mode.name(),
            self.retrieval.top_k,
            self.retrieval.effective_per_path_k(),
            self.retrieval.min_score,
            self.rerank.enabled,
            self.rerank.backend.name(),
            self.rerank.model.display_name(),
            self.rerank.candidate_k,
            self.rerank.top_n,
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
    fn per_path_k_defaults_to_double_top_k() {
        let config = RetrievalConfig {
            top_k: 4,
            per_path_k: None,
            ..Default::default()
        };
        assert_eq!(config.effective_per_path_k(), 8);
    }

    #[test]
    fn per_path_k_has_a_floor() {
        let config = RetrievalConfig {
            top_k: 1,
            per_path_k: None,
            ..Default::default()
        };
        assert_eq!(config.effective_per_path_k(), 8);
    }

    #[test]
    fn per_path_k_has_a_ceiling() {
        let config = RetrievalConfig {
            top_k: 100,
            per_path_k: Some(500),
            ..Default::default()
        };
        assert_eq!(config.effective_per_path_k(), 50);
    }

    #[test]
    fn default_top_k_is_sane() {
        assert!((1..=10).contains(&RetrievalConfig::default().top_k));
    }

    #[test]
    fn default_mode_is_hybrid() {
        assert_eq!(RetrievalConfig::default().mode, SearchMode::Hybrid);
    }

    #[test]
    fn rerank_candidate_k_is_at_least_top_n() {
        // 候选数小于 top_n 是没有意义的 —— 会导致结果被截断。
        let config = RerankConfig::default();
        assert!(config.candidate_k >= config.top_n);
    }

    #[test]
    fn default_rerank_is_local() {
        // 本地后端不需要 API key —— 保证 `cargo run` 开箱可用。
        assert_eq!(RerankConfig::default().backend, RerankBackend::Local);
    }
}
