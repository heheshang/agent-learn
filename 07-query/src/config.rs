//! 全局配置。
//!
//! 所有「可调参数」集中在这里，来源有三档（后者覆盖前者）：
//!
//! ```text
//! 默认值  <  环境变量
//! ```
//!
//! 环境变量名带 `QUERY_` 前缀，避免和其它程序的变量撞名。
//!
//! ## 为什么要把参数集中
//!
//! RAG 系统里最容易被「悄悄改坏」的参数是 `top_k` 和 `per_query_k`：
//! 改一个数字，检索结果就变了，答案质量跟着变，但没有任何报错。
//! 把它们收拢到一处 + 启动时打印出来，至少能保证「出问题时知道当前配置是什么」。
//!
//! ## 与 06-rag 的差异
//!
//! 06 的配置是「一个默认策略」，07 是「**一组可对比的策略**」——
//! 因为本项目的核心产出就是那张「各策略召回差异」的表。
//!
//! ## 复用 06-rag 的类型
//!
//! `EmbeddingModel` 和 `GenerationConfig` 直接用 `rag_06::config` 里的 ——
//! 没必要抄一份，改一处就生效两处。

use std::path::PathBuf;

use rag_06::config::{EmbeddingModel, GenerationConfig};

use crate::fusion::FusionMethod;
use crate::transform::Strategy;

/// 改写配置。
#[derive(Debug, Clone)]
pub struct TransformConfig {
    /// 默认使用的改写策略。
    pub default_strategy: Strategy,

    /// Multi-Query 生成的问法数量。
    pub num_queries: usize,

    /// Decompose 拆出的子问题数量上限。
    pub max_sub_questions: usize,

    /// 改写用的模型（与生成模型可以不同 —— 改写更适合用快模型）。
    pub transform_model: String,

    /// 改写温度。改写是「换个说法」，不需要多样性，低温度更稳。
    pub transform_temperature: f64,

    /// 改写的最大 token 数。HyDE 要写一整段，需要给够。
    pub transform_max_tokens: u32,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            default_strategy: Strategy::MultiQuery,
            num_queries: 3,
            max_sub_questions: 4,
            transform_model: "deepseek-flash".to_string(),
            transform_temperature: 0.0,
            transform_max_tokens: 512,
        }
    }
}

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
    pub per_query_k: Option<usize>,

    /// 融合方法。
    pub fusion: FusionMethod,

    /// 相似度下限。低于这个分数的命中块被丢弃。
    pub min_score: Option<f32>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            per_query_k: None,
            fusion: FusionMethod::Rrf,
            min_score: None,
        }
    }
}

impl RetrievalConfig {
    /// 实际使用的「每路召回数量」。
    ///
    /// 自动模式 = `top_k * 2`，最低 8，上限 50。
    ///
    /// 为什么是 2 倍：多路检索 + 融合后，真正有用的块可能来自任何一路。
    /// 每路只取 `top_k` 条的话，融合后可能挤满重复内容。
    pub fn effective_per_query_k(&self) -> usize {
        match self.per_query_k {
            Some(k) => k.clamp(1, 50),
            None => (self.top_k * 2).clamp(8, 50),
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

    /// 改写配置。
    pub transform: TransformConfig,

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
            data_dir: manifest.join("data"),
            index_path: manifest.join("index/index.faiss"),
            bind_addr: "127.0.0.1:3010".to_string(),
            embedding_model: EmbeddingModel::MultilingualE5Small,
            transform: TransformConfig::default(),
            retrieval: RetrievalConfig::default(),
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
    /// | `QUERY_DATA_DIR` | 知识库目录 | `07-query/data` |
    /// | `QUERY_INDEX_PATH` | 索引文件路径 | `index/index.faiss` |
    /// | `QUERY_BIND_ADDR` | 监听地址 | `127.0.0.1:3010` |
    /// | `QUERY_EMBEDDING_MODEL` | `e5` / `bge` | `e5` |
    /// | `QUERY_STRATEGY` | 默认改写策略 | `multi_query` |
    /// | `QUERY_NUM_QUERIES` | Multi-Query 问法数 | `3` |
    /// | `QUERY_MAX_SUB_QUESTIONS` | Decompose 子问题上限 | `4` |
    /// | `QUERY_TRANSFORM_MODEL` | 改写模型 | `deepseek-flash` |
    /// | `QUERY_TRANSFORM_TEMPERATURE` | 改写温度 | `0.0` |
    /// | `QUERY_TRANSFORM_MAX_TOKENS` | 改写 max tokens | `512` |
    /// | `QUERY_TOP_K` | 最终块数 | `4` |
    /// | `QUERY_PER_QUERY_K` | 每路召回数 | 自动（top_k × 2） |
    /// | `QUERY_FUSION` | 融合方法 | `rrf` |
    /// | `QUERY_MIN_SCORE` | 相似度下限（可空） | 空 |
    /// | `QUERY_MODEL` | 生成模型 | `deepseek-flash` |
    /// | `QUERY_TEMPERATURE` | 生成温度 | `0.1` |
    /// | `QUERY_MAX_TOKENS` | 生成 max tokens | `2000` |
    /// | `QUERY_AUTO_INDEX` | 启动时自动建索引 | `true` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("QUERY_DATA_DIR") {
            config.data_dir = PathBuf::from(v);
        }
        if let Some(v) = env_str("QUERY_INDEX_PATH") {
            config.index_path = PathBuf::from(v);
        }
        if let Some(v) = env_str("QUERY_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("QUERY_EMBEDDING_MODEL")
            && let Some(m) = EmbeddingModel::from_env_value(&v)
        {
            config.embedding_model = m;
        }
        if let Some(v) = env_str("QUERY_STRATEGY")
            && let Some(s) = Strategy::parse(&v)
        {
            config.transform.default_strategy = s;
        }
        if let Some(v) = env_usize("QUERY_NUM_QUERIES") {
            config.transform.num_queries = v.max(1);
        }
        if let Some(v) = env_usize("QUERY_MAX_SUB_QUESTIONS") {
            config.transform.max_sub_questions = v.max(1);
        }
        if let Some(v) = env_str("QUERY_TRANSFORM_MODEL") {
            config.transform.transform_model = v;
        }
        if let Some(v) = env_f64("QUERY_TRANSFORM_TEMPERATURE") {
            config.transform.transform_temperature = v;
        }
        if let Some(v) = env_u32("QUERY_TRANSFORM_MAX_TOKENS") {
            config.transform.transform_max_tokens = v;
        }
        if let Some(v) = env_usize("QUERY_TOP_K") {
            config.retrieval.top_k = v;
        }
        if let Some(v) = env_usize("QUERY_PER_QUERY_K") {
            config.retrieval.per_query_k = Some(v);
        }
        if let Some(v) = env_str("QUERY_FUSION")
            && let Some(f) = FusionMethod::parse(&v)
        {
            config.retrieval.fusion = f;
        }
        if let Some(v) = env_f32("QUERY_MIN_SCORE") {
            config.retrieval.min_score = Some(v);
        }
        if let Some(v) = env_str("QUERY_MODEL") {
            config.generation.model = v;
        }
        if let Some(v) = env_f64("QUERY_TEMPERATURE") {
            config.generation.temperature = v;
        }
        if let Some(v) = env_u32("QUERY_MAX_TOKENS") {
            config.generation.max_tokens = v;
        }
        if let Some(v) = env_bool("QUERY_AUTO_INDEX") {
            config.auto_index = v;
        }

        config
    }

    /// 打印当前配置（启动时调用，方便复现问题）。
    pub fn summary(&self) -> String {
        format!(
            "数据目录: {}\n索引路径: {}\n监听地址: {}\n嵌入模型: {:?}\n\
             改写: 默认策略={} 问法数={} 子问题上限={} 模型={} temp={}\n\
             检索: top_k={} 每路召回={} 融合={} min_score={:?}\n\
             生成: {} temp={} max_tokens={}",
            self.data_dir.display(),
            self.index_path.display(),
            self.bind_addr,
            self.embedding_model,
            self.transform.default_strategy.name(),
            self.transform.num_queries,
            self.transform.max_sub_questions,
            self.transform.transform_model,
            self.transform.transform_temperature,
            self.retrieval.top_k,
            self.retrieval.effective_per_query_k(),
            self.retrieval.fusion.name(),
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
    fn per_query_k_defaults_to_double_top_k() {
        let config = RetrievalConfig {
            top_k: 4,
            per_query_k: None,
            ..Default::default()
        };
        assert_eq!(config.effective_per_query_k(), 8);
    }

    #[test]
    fn per_query_k_has_a_floor() {
        let config = RetrievalConfig {
            top_k: 1,
            per_query_k: None,
            ..Default::default()
        };
        assert_eq!(config.effective_per_query_k(), 8);
    }

    #[test]
    fn per_query_k_has_a_ceiling() {
        let config = RetrievalConfig {
            top_k: 100,
            per_query_k: Some(500),
            ..Default::default()
        };
        assert_eq!(config.effective_per_query_k(), 50);
    }

    #[test]
    fn default_top_k_is_sane() {
        assert!((1..=10).contains(&RetrievalConfig::default().top_k));
    }
}
