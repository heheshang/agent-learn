/// 全局配置。
///
/// 所有「可调参数」集中在这里，来源有两档（后者覆盖前者）：
///
/// ```text
/// 默认值  <  环境变量
/// ```
///
/// 环境变量名带 `SUMMARY_` 前缀，避免和其它项目撞名
/// （06 用 `RAG_`，07 用 `QUERY_`，08 用 `HYBRID_`，10 用 `MILVUS_`）。
///
/// ## 本项目特有的三组配置
///
/// | 组 | 作用 | 关键项 |
/// |---|---|---|
/// | [`RetrievalConfig`] | 检索的「层数」与每层参数 | `mode` / `per_path_k` / `candidate_k` |
/// | [`RerankConfig`] | 重排后端与候选池 | `backend` / `candidate_k` / `top_n` |
/// | [`CompressConfig`] | 上下文压缩预算 | `budget_tokens` / `min_tokens` |
///
/// ## 为什么要复用 MilvusConfig 而不是抄一份
///
/// 10-rag-milvus 已经把 Milvus 的连接参数、collection 形态、
/// 索引类型、度量方式全都定义清楚了。抄一份的话，
/// 「Milvus 升级了字段名」这种改动要改两个地方 —— 而且很容易漏。
///
/// 所以这里直接 `pub use rag_10::config::MilvusConfig`，
/// 只在它之上覆盖「本项目特有的默认值」（collection 名不同）。
use std::path::PathBuf;

use rag_08::reranker::{RerankBackend, RerankerModel};
use rag_10::config::{MetricKind, MilvusConfig};

pub use rag_10::config::model_dimension;
pub use rag_10::config::model_name;

/// 检索模式 —— 本项目的**消融阶梯**。
///
/// 每一档都是「上一档 + 一个增强」，所以差值是**可归因**的：
///
/// | 模式 | 做了什么 | 用来回答什么问题 |
/// |---|---|---|
/// | [`SearchMode::Dense`] | 只做向量检索 | 基线：语义检索能到多少分？ |
/// | [`SearchMode::Keyword`] | 只做 BM25 关键词检索 | 关键词路单独能到多少分？ |
/// | [`SearchMode::Hybrid`] | 服务端混合检索（dense + BM25 + RRF） | 混合比纯向量好多少？ |
/// | [`SearchMode::HybridRerank`] | 混合 + 本地 cross-encoder 重排 | 重排又贡献了多少？ |
/// | [`SearchMode::Full`] | 多问法改写 + 混合 + 重排 + 压缩 | 全量优化总共值多少？ |
///
/// ## ⚠️ 为什么「稠密」用 Milvus，「关键词」用客户端 BM25
///
/// Milvus 的服务端 BM25 走的是**英文优先的 `standard` 分析器**，
/// 对中文的分词效果有限（整句被当成一个 token）。
/// 所以本项目的关键词路**不用**服务端 BM25，而是复用 08-rag
/// 那套「CJK 单字 + 双字」的客户端 BM25 —— 它对中文效果好得多。
///
/// 两条路都保留，是为了让「服务端 vs 客户端」这件事也能被对比出来。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// 只做稠密向量检索（**基线**）。
    Dense,

    /// 只做客户端 BM25 关键词检索（**对照组**）。
    Keyword,

    /// 服务端混合检索：Milvus dense + BM25，RRF 融合。
    Hybrid,

    /// 服务端混合检索 + 本地 cross-encoder 重排。
    HybridRerank,

    /// 全量：多问法改写 → 混合检索 → 重排 → 上下文压缩。
    Full,
}

impl SearchMode {
    /// 模式名。
    pub fn name(self) -> &'static str {
        match self {
            SearchMode::Dense => "dense",
            SearchMode::Keyword => "keyword",
            SearchMode::Hybrid => "hybrid",
            SearchMode::HybridRerank => "hybrid_rerank",
            SearchMode::Full => "full",
        }
    }

    /// 从字符串解析（不区分大小写，支持连字符）。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "dense" | "vector" | "semantic" => Some(SearchMode::Dense),
            "keyword" | "bm25" | "sparse" => Some(SearchMode::Keyword),
            "hybrid" | "mixed" => Some(SearchMode::Hybrid),
            "hybrid_rerank" | "rerank" => Some(SearchMode::HybridRerank),
            "full" | "complete" => Some(SearchMode::Full),
            _ => None,
        }
    }

    /// 所有模式（**按阶梯顺序** —— 从基线到全量）。
    pub fn all() -> Vec<Self> {
        vec![
            SearchMode::Dense,
            SearchMode::Keyword,
            SearchMode::Hybrid,
            SearchMode::HybridRerank,
            SearchMode::Full,
        ]
    }

    /// 是否启用查询改写（多问法）。
    pub fn uses_query_rewrite(self) -> bool {
        self == SearchMode::Full
    }

    /// 是否启用重排。
    pub fn uses_rerank(self) -> bool {
        matches!(self, SearchMode::HybridRerank | SearchMode::Full)
    }

    /// 是否启用上下文压缩。
    pub fn uses_compression(self) -> bool {
        self == SearchMode::Full
    }

    /// 是否走服务端混合检索。
    ///
    /// `Keyword` 不走 —— 它走客户端 BM25。
    pub fn uses_server_hybrid(self) -> bool {
        matches!(
            self,
            SearchMode::Hybrid | SearchMode::HybridRerank | SearchMode::Full
        )
    }
}

/// 检索参数。
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// 默认的检索模式。
    pub mode: SearchMode,

    /// 最终保留的块数（top-k）。
    ///
    /// 这是**压缩前**的数量 —— 压缩阶段会在这个基础上按 token 预算
    /// 再砍掉几块。
    pub top_k: usize,

    /// 每路召回条数（Milvus 的 `per_path_k`）。
    ///
    /// ⚠️ 必须 ≥ `candidate_k` 的一半，否则两路合并去重后
    /// 候选池凑不满，重排器的工作范围被卡死（和 08-rag 同一个坑）。
    pub per_path_k: usize,

    /// 相似度下限。低于这个分数的稠密命中块被丢弃。
    ///
    /// ## ⚠️ 只对「能拿到稠密分数」的路径生效
    ///
    /// - `Dense` / `ClientFusion` 模式：**生效**
    /// - `Hybrid` / `HybridRerank` / `Full` 模式（服务端混合）：**不生效**
    ///
    /// 原因：服务端返回的是**融合后**的 RRF 分数，两路的原始分数
    /// 已经被服务端合成为一个值 —— 客户端拿不到「这个块的稠密相似度」。
    ///
    /// 强行拿 RRF 分数和余弦相似度阈值比较是**没有意义的**（两种量纲）。
    /// 所以这里诚实地「不生效」，而不是给出一个看起来能用的错误过滤。
    pub min_score: Option<f32>,

    /// 是否启用客户端 BM25 回退（Milvus 不可达时）。
    pub enable_local_fallback: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            mode: SearchMode::Full,
            top_k: 5,
            per_path_k: 20,
            min_score: None,
            enable_local_fallback: true,
        }
    }
}

impl RetrievalConfig {
    /// 实际使用的「每路召回数量」。
    ///
    /// 永远 ≥ `candidate_k` —— 候选池是重排的天花板，
    /// 召回不够的话重排就无从谈起。
    pub fn effective_per_path_k(&self, candidate_k: usize) -> usize {
        self.per_path_k.max(candidate_k).max(1)
    }
}

/// 重排参数。
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// 重排后端。
    pub backend: RerankBackend,

    /// 本地 reranker 的模型选择。
    pub model: RerankerModel,

    /// Cohere 的模型名。
    pub cohere_model: String,

    /// Cohere API key。
    pub cohere_api_key: Option<String>,

    /// 送进 reranker 的候选数（**召回阶段的上限**）。
    ///
    /// ⚠️ 这个值决定重排的**天花板** —— reranker 只能重排它拿到的东西。
    /// 经验值 10~50。
    pub candidate_k: usize,

    /// 重排后保留多少条。
    pub top_n: usize,

    /// 是否启用重排（`false` = 对照组）。
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
            top_n: 5,
            enabled: true,
        }
    }
}

/// 上下文压缩参数。
///
/// ## 为什么需要压缩
///
/// 检索给回 5 个块，每个块 300 token —— 加上标题和引用标记，
/// 光上下文就 1500+ token。而**摘要类任务**的输出通常只有几百字，
/// 上下文里塞太多无关内容反而会：
///
/// 1. 稀释关键信息 —— 「中间迷失」（lost in the middle）
/// 2. 拉高成本和延迟
/// 3. 增加幻觉面 —— 模型会去引用那些其实用不上的片段
///
/// ## 压缩策略
///
/// 本项目用**抽取式压缩**（extractive），不用生成式：
///
/// - 生成式压缩（让 LLM 总结）会**丢信息**，而且引入新的失败点
/// - 抽取式压缩只做「挑选 + 截断」，原始句子还在，可溯源
///
/// 具体做法：按相关度顺序逐块累加 token 预算，
/// 超预算的块**只保留能装下的部分**（按句子边界切）。
#[derive(Debug, Clone)]
pub struct CompressConfig {
    /// 上下文的 token 预算。
    pub budget_tokens: usize,

    /// 单块最少保留的 token —— 防止第一个块就把预算吃光。
    pub min_tokens: usize,

    /// 是否启用压缩。
    pub enabled: bool,
}

impl Default for CompressConfig {
    fn default() -> Self {
        Self {
            budget_tokens: 1200,
            min_tokens: 80,
            enabled: true,
        }
    }
}

/// 查询改写参数。
#[derive(Debug, Clone)]
pub struct RewriteConfig {
    /// 生成的问法数量。
    pub num_queries: usize,

    /// 是否启用改写。
    pub enabled: bool,
}

impl Default for RewriteConfig {
    fn default() -> Self {
        Self {
            num_queries: 3,
            enabled: true,
        }
    }
}

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// 知识库文档目录。
    pub data_dir: PathBuf,

    /// 评估集路径。
    pub eval_dataset_path: PathBuf,

    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 嵌入模型。
    pub embedding_model: rag_05::embed::Model,

    /// Milvus 配置（复用 10-rag-milvus 的类型）。
    pub milvus: MilvusConfig,

    /// 分块配置。
    pub splitter: rag_04::splitter::SplitterConfig,

    /// 检索配置。
    pub retrieval: RetrievalConfig,

    /// 重排配置。
    pub rerank: RerankConfig,

    /// 压缩配置。
    pub compress: CompressConfig,

    /// 改写配置。
    pub rewrite: RewriteConfig,

    /// 生成模型。
    pub generation_model: String,

    /// 生成温度。
    pub generation_temperature: f64,

    /// 生成 max tokens。
    pub generation_max_tokens: u32,

    /// 启动时是否自动建索引。
    pub auto_index: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // ⚠️ collection 名和 10-rag-milvus 不同 —— 否则两个项目的索引
        // 会互相覆盖（而且 embedding 模型一样，撞起来更难发现）。
        let milvus = MilvusConfig {
            collection: "rag_summary_chunks".to_string(),
            ..Default::default()
        };

        Self {
            data_dir: manifest.join("data"),
            eval_dataset_path: manifest.join("../09-rag-eval/data/eval_cases.json"),
            bind_addr: "127.0.0.1:3050".to_string(),
            embedding_model: rag_05::embed::Model::MultilingualE5Small,
            milvus,
            splitter: rag_04::splitter::SplitterConfig::default(),
            retrieval: RetrievalConfig::default(),
            rerank: RerankConfig::default(),
            compress: CompressConfig::default(),
            rewrite: RewriteConfig::default(),
            generation_model: "deepseek-flash".to_string(),
            generation_temperature: 0.1,
            generation_max_tokens: 2000,
            auto_index: true,
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置（未设置的项用默认值）。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `SUMMARY_DATA_DIR` | 知识库目录 | `12-rag-summary/data` |
    /// | `SUMMARY_EVAL_DATASET` | 评估集路径 | `09-rag-eval/data/eval_cases.json` |
    /// | `SUMMARY_BIND_ADDR` | 监听地址 | `127.0.0.1:3050` |
    /// | `SUMMARY_MODE` | `dense`/`keyword`/`hybrid`/`hybrid_rerank`/`full` | `full` |
    /// | `SUMMARY_TOP_K` | 最终块数 | `5` |
    /// | `SUMMARY_PER_PATH_K` | 每路召回数 | `20` |
    /// | `SUMMARY_MIN_SCORE` | 相似度下限（可空） | 空 |
    /// | `SUMMARY_RERANK_ENABLED` | 是否启用重排 | `true` |
    /// | `SUMMARY_RERANK_BACKEND` | `local` / `cohere` | `local` |
    /// | `SUMMARY_RERANK_MODEL` | `bge` / `m3` / `jina` | `bge` |
    /// | `SUMMARY_RERANK_CANDIDATE_K` | 送进重排的候选数 | `20` |
    /// | `SUMMARY_RERANK_TOP_N` | 重排后保留条数 | `5` |
    /// | `COHERE_API_KEY` | Cohere API key | 空 |
    /// | `SUMMARY_COMPRESS_ENABLED` | 是否启用上下文压缩 | `true` |
    /// | `SUMMARY_COMPRESS_BUDGET` | 上下文 token 预算 | `1200` |
    /// | `SUMMARY_REWRITE_ENABLED` | 是否启用多问法改写 | `true` |
    /// | `SUMMARY_REWRITE_NUM_QUERIES` | 问法数量 | `3` |
    /// | `SUMMARY_MODEL` | 生成模型 | `deepseek-flash` |
    /// | `SUMMARY_TEMPERATURE` | 生成温度 | `0.1` |
    /// | `SUMMARY_MAX_TOKENS` | 生成 max tokens | `2000` |
    /// | `SUMMARY_AUTO_INDEX` | 启动时自动建索引 | `true` |
    /// | `SUMMARY_MILVUS_URI` | Milvus gRPC 地址 | `http://127.0.0.1:19530` |
    /// | `SUMMARY_MILVUS_TOKEN` | Milvus 认证 token | `root:Milvus` |
    /// | `SUMMARY_MILVUS_COLLECTION` | collection 名 | `rag_summary_chunks` |
    /// | `SUMMARY_MILVUS_ENABLE_BM25` | 是否启用服务端 BM25 | `true` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("SUMMARY_DATA_DIR") {
            config.data_dir = PathBuf::from(v);
        }
        if let Some(v) = env_str("SUMMARY_EVAL_DATASET") {
            config.eval_dataset_path = PathBuf::from(v);
        }
        if let Some(v) = env_str("SUMMARY_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("SUMMARY_MODE")
            && let Some(m) = SearchMode::parse(&v)
        {
            config.retrieval.mode = m;
        }
        if let Some(v) = env_usize("SUMMARY_TOP_K") {
            config.retrieval.top_k = v.max(1);
        }
        if let Some(v) = env_usize("SUMMARY_PER_PATH_K") {
            config.retrieval.per_path_k = v.max(1);
        }
        if let Some(v) = env_f32("SUMMARY_MIN_SCORE") {
            config.retrieval.min_score = Some(v);
        }
        if let Some(v) = env_bool("SUMMARY_RERANK_ENABLED") {
            config.rerank.enabled = v;
        }
        if let Some(v) = env_str("SUMMARY_RERANK_BACKEND")
            && let Some(b) = RerankBackend::from_env_value(&v)
        {
            config.rerank.backend = b;
        }
        if let Some(v) = env_str("SUMMARY_RERANK_MODEL")
            && let Some(m) = RerankerModel::from_env_value(&v)
        {
            config.rerank.model = m;
        }
        if let Some(v) = env_str("COHERE_API_KEY") {
            config.rerank.cohere_api_key = Some(v);
        }
        if let Some(v) = env_usize("SUMMARY_RERANK_CANDIDATE_K") {
            config.rerank.candidate_k = v.clamp(1, 200);
        }
        if let Some(v) = env_usize("SUMMARY_RERANK_TOP_N") {
            config.rerank.top_n = v.clamp(1, 50);
        }
        if let Some(v) = env_bool("SUMMARY_COMPRESS_ENABLED") {
            config.compress.enabled = v;
        }
        if let Some(v) = env_usize("SUMMARY_COMPRESS_BUDGET") {
            config.compress.budget_tokens = v.max(100);
        }
        if let Some(v) = env_bool("SUMMARY_REWRITE_ENABLED") {
            config.rewrite.enabled = v;
        }
        if let Some(v) = env_usize("SUMMARY_REWRITE_NUM_QUERIES") {
            config.rewrite.num_queries = v.clamp(1, 5);
        }
        if let Some(v) = env_str("SUMMARY_MODEL") {
            config.generation_model = v;
        }
        if let Some(v) = env_f64("SUMMARY_TEMPERATURE") {
            config.generation_temperature = v;
        }
        if let Some(v) = env_u32("SUMMARY_MAX_TOKENS") {
            config.generation_max_tokens = v;
        }
        if let Some(v) = env_bool("SUMMARY_AUTO_INDEX") {
            config.auto_index = v;
        }
        if let Some(v) = env_str("SUMMARY_MILVUS_URI") {
            config.milvus.uri = v;
        }
        if let Some(v) = env_str("SUMMARY_MILVUS_TOKEN") {
            config.milvus.token = v;
        }
        if let Some(v) = env_str("SUMMARY_MILVUS_DATABASE") {
            config.milvus.database = v;
        }
        if let Some(v) = env_str("SUMMARY_MILVUS_COLLECTION") {
            config.milvus.collection = v;
        }
        if let Some(v) = env_bool("SUMMARY_MILVUS_ENABLE_BM25") {
            config.milvus.enable_bm25 = v;
        }
        if let Some(v) = env_str("SUMMARY_MILVUS_METRIC")
            && let Some(m) = MetricKind::parse(&v)
        {
            config.milvus.metric = m;
        }

        config
    }

    /// 配置摘要（启动时打印）。
    pub fn summary(&self) -> String {
        format!(
            "语料目录    : {}\n\
             评估集      : {}\n\
             监听地址    : {}\n\
             嵌入模型    : {} (dim={})\n\
             Milvus      : {}\n\
             Collection : {}\n\
             服务端 BM25 : {}\n\
             检索        : 模式={} top_k={} 每路召回={} min_score={:?}\n\
             重排        : 启用={} 后端={} 模型={} 候选={} top_n={}\n\
             压缩        : 启用={} 预算={} token\n\
             改写        : 启用={} 问法数={}\n\
             生成        : {} temp={} max_tokens={}",
            self.data_dir.display(),
            self.eval_dataset_path.display(),
            self.bind_addr,
            model_name(self.embedding_model),
            model_dimension(self.embedding_model),
            self.milvus.uri,
            self.milvus.collection,
            if self.milvus.enable_bm25 {
                "启用"
            } else {
                "关闭"
            },
            self.retrieval.mode.name(),
            self.retrieval.top_k,
            self.retrieval.effective_per_path_k(self.rerank.candidate_k),
            self.retrieval.min_score,
            self.rerank.enabled,
            self.rerank.backend.name(),
            self.rerank.model.display_name(),
            self.rerank.candidate_k,
            self.rerank.top_n,
            self.compress.enabled,
            self.compress.budget_tokens,
            self.rewrite.enabled,
            self.rewrite.num_queries,
            self.generation_model,
            self.generation_temperature,
            self.generation_max_tokens,
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
    fn search_mode_round_trips() {
        for mode in SearchMode::all() {
            assert_eq!(SearchMode::parse(mode.name()), Some(mode));
        }
    }

    #[test]
    fn parses_lenient_mode_names() {
        assert_eq!(SearchMode::parse("Dense"), Some(SearchMode::Dense));
        assert_eq!(SearchMode::parse("bm25"), Some(SearchMode::Keyword));
        assert_eq!(
            SearchMode::parse("HYBRID-RERANK"),
            Some(SearchMode::HybridRerank)
        );
        assert_eq!(SearchMode::parse("unknown"), None);
    }

    #[test]
    fn ladder_is_ordered_from_baseline_to_full() {
        let modes = SearchMode::all();
        assert_eq!(modes.first(), Some(&SearchMode::Dense));
        assert_eq!(modes.last(), Some(&SearchMode::Full));
    }

    #[test]
    fn only_full_mode_uses_rewrite_and_compression() {
        for mode in SearchMode::all() {
            assert_eq!(mode.uses_query_rewrite(), mode == SearchMode::Full);
            assert_eq!(mode.uses_compression(), mode == SearchMode::Full);
        }
    }

    #[test]
    fn keyword_mode_does_not_use_server_hybrid() {
        assert!(!SearchMode::Keyword.uses_server_hybrid());
        assert!(SearchMode::Hybrid.uses_server_hybrid());
        assert!(SearchMode::Full.uses_server_hybrid());
    }

    #[test]
    fn per_path_k_never_below_candidate_k() {
        let config = RetrievalConfig {
            per_path_k: 2,
            ..Default::default()
        };
        assert_eq!(config.effective_per_path_k(20), 20);
    }

    #[test]
    fn default_collection_differs_from_milvus_project() {
        // 两个项目共用 Milvus 实例 —— collection 名必须不同，
        // 否则索引会互相覆盖。
        let config = AppConfig::default();
        assert_eq!(config.milvus.collection, "rag_summary_chunks");
        assert_ne!(
            config.milvus.collection,
            rag_10::config::MilvusConfig::default().collection
        );
    }

    #[test]
    fn default_port_is_3050() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3050");
    }

    #[test]
    fn rerank_candidate_k_is_at_least_top_n() {
        let config = RerankConfig::default();
        assert!(config.candidate_k >= config.top_n);
    }
}
