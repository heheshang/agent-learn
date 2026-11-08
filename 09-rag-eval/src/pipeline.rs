//! 评估流水线 —— 把「系统」和「评估集」接在一起。
//!
//! ## 链路
//!
//! ```text
//! 评估集 ──> 逐样本 ──> 被评估系统 ──> 检索结果 + 答案
//!                          │
//!                    ┌─────┴─────┐
//!                    │  评估上下文 │  （标准答案 + 实际输出）
//!                    └─────┬─────┘
//!                          ▼
//!                     指标计算 ──> 报告
//! ```
//!
//! ## 为什么「被评估系统」是一个 trait
//!
//! 因为要对比的**不是同一个系统的不同参数**，而是**不同的检索策略**：
//!
//! | 版本 | 实现 |
//! |---|---|
//! | baseline（只走向量） | [`SystemUnderEval`] 的 vector 模式 |
//! | optimized（混合 + 重排） | [`SystemUnderEval`] 的 hybrid 模式 |
//!
//! 它们共用同一个索引、同一套 Prompt、同一个 LLM ——
//! **唯一的差别是检索策略**。这样对比出来的差值才干净。
//!
//! ## 评估是「只读」的
//!
//! 评估过程**不修改**被评估系统的任何状态（索引、配置）。
//! 它只是「拿一批问题去问，然后打分」。
//!
//! 所以同一个索引可以被反复评估不同策略 —— 这是对比实验的前提。

use agent_logging::{debug, info, warn};
use std::sync::Arc;
use std::time::Instant;

use rag_04::splitter::Splitter;
use rag_08::config::AppConfig;
use rag_08::hybrid::SearchMode;
use rag_08::index::{self, IndexStats};
use rag_08::pipeline::HybridPipeline;

use crate::dataset::{ChunkIndex, EvalDataset};
use crate::error::{EvalError, EvalResult};
use crate::metrics::types::EvalContext;
use crate::report::EvalReport;

/// 默认评估并发度。
///
/// ## 为什么是 4
///
/// 每个样本要跑 1 次生成 + 4~5 次 judge 调用。
/// 并发 4 大约能把耗时压到 1/4，同时不会明显触发上游限流。
pub const DEFAULT_CONCURRENCY: usize = 4;

/// 被评估系统的检索策略。
///
/// ## 为什么要枚举而不是直接传 `SearchMode`
///
/// 因为 `SearchMode` 是 08-rag 的类型，而本项目要评估的「策略」可能更多：
/// 将来加「查询改写 + 混合检索」时，只需要往这个枚举里加一个变体，
/// 而不用改所有调用点。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalVariant {
    /// 只走向量检索（**对照组** —— 06-rag 的水平）。
    Vector,

    /// 只走 BM25（**对照组**）。
    Bm25,

    /// 混合检索 + 重排（**优化组** —— 08-rag 的水平）。
    Hybrid,

    /// 不做检索，直接让模型回答（**绝对基线**）。
    ///
    /// 这个变体极其重要 —— 它回答的问题是：
    /// **「RAG 到底比直接问模型强多少？」**
    ///
    /// 没有这个基线，所有 RAG 的提升都缺少参照系。
    NoRetrieval,
}

impl EvalVariant {
    /// 变体名。
    pub fn name(self) -> &'static str {
        match self {
            EvalVariant::Vector => "vector",
            EvalVariant::Bm25 => "bm25",
            EvalVariant::Hybrid => "hybrid",
            EvalVariant::NoRetrieval => "no_retrieval",
        }
    }

    /// 从字符串解析。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "vector" | "dense" => Some(EvalVariant::Vector),
            "bm25" | "keyword" => Some(EvalVariant::Bm25),
            "hybrid" | "mixed" => Some(EvalVariant::Hybrid),
            "no_retrieval" | "none" | "baseline" => Some(EvalVariant::NoRetrieval),
            _ => None,
        }
    }

    /// 所有变体（`NoRetrieval` 在最前 —— 它是最基础的基线）。
    pub fn all() -> Vec<Self> {
        vec![
            EvalVariant::NoRetrieval,
            EvalVariant::Vector,
            EvalVariant::Bm25,
            EvalVariant::Hybrid,
        ]
    }

    /// 对应的 08-rag 检索模式（`NoRetrieval` 没有对应模式）。
    pub fn search_mode(self) -> Option<SearchMode> {
        match self {
            EvalVariant::Vector => Some(SearchMode::Vector),
            EvalVariant::Bm25 => Some(SearchMode::Bm25),
            EvalVariant::Hybrid => Some(SearchMode::Hybrid),
            EvalVariant::NoRetrieval => None,
        }
    }
}

/// 被评估的系统 —— 检索 + 生成。
///
/// ## 降级语义
///
/// | 环节失败 | 行为 |
/// |---|---|
/// | 检索失败 | 记为「检索失败」，用空上下文继续生成（评估结果标记为不可信） |
/// | 生成失败 | 跳过该样本，记入失败列表 |
///
/// 检索失败后**继续生成**是有意的：这样报告里还能看到「系统检索挂了」
/// 这个事实，而不是整次评估中断。
pub struct SystemUnderEval {
    pipeline: Arc<HybridPipeline>,
}

impl std::fmt::Debug for SystemUnderEval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemUnderEval").finish()
    }
}

impl SystemUnderEval {
    /// 包装一个 08-rag 的流水线。
    pub fn new(pipeline: Arc<HybridPipeline>) -> Self {
        Self { pipeline }
    }

    /// 底层流水线（供调用方直接使用）。
    pub fn pipeline(&self) -> &Arc<HybridPipeline> {
        &self.pipeline
    }

    /// 跑一个问题，返回（答案，检索到的上下文，检索到的块 ID）。
    pub async fn run(&self, question: &str, variant: EvalVariant) -> EvalResult<RunOutput> {
        match variant {
            EvalVariant::NoRetrieval => {
                // 不检索 —— 直接用「无上下文」提示词问模型。
                //
                // ⚠️ 走的是 08-rag 的 `assemble_no_context` ——
                // 它明确告诉模型「没有资料，请说无法回答」。
                //
                // 那这样测出来的分会不会恒为 0？
                //
                // **不会** —— 因为 `assemble_no_context` 里的
                // 「请直接说明无法回答」是**祈使句**，不是**限制**。
                // 实际模型仍然会尝试回答（尤其是模型本身的先验知识里
                // 有答案时，比如「TCP 三次握手是为什么」）。
                //
                // 这正是我们要测的：**模型靠先验知识能答对多少**。
                // 如果它真的一点都答不出来，这个基线就是 0 分，
                // 说明知识库里的内容确实是模型不知道的 —— 反而更有说服力。
                let prompt = rag_06::prompt::assemble_no_context(question);
                let answer = self
                    .pipeline
                    .llm()
                    .chat(&prompt.system, &prompt.user)
                    .await
                    .map_err(|e| EvalError::Retrieval(format!("生成失败: {e}")))?;

                Ok(RunOutput {
                    answer,
                    contexts: Vec::new(),
                    retrieved_ids: Vec::new(),
                    retrieval_failed: false,
                })
            }
            _ => {
                let mode = variant.search_mode().expect("非 NoRetrieval 都有检索模式");
                self.run_with_mode(question, mode).await
            }
        }
    }

    /// 跑一个问题（显式指定检索模式）。
    ///
    /// 和 [`SystemUnderEval::run`] 的区别：这里不经过 `EvalVariant`，
    /// 直接给检索模式 —— 「快速评估」端点用这个。
    pub async fn run_with_mode(&self, question: &str, mode: SearchMode) -> EvalResult<RunOutput> {
        let retrieval = match self.pipeline.retrieve_with(question, mode).await {
            Ok(r) => r,
            Err(e) => {
                // 检索失败 —— 记下来，但继续（用空上下文生成）。
                warn!(question = %question, error = %e, "检索失败，降级为空上下文");
                return self.run_without_context(question, &e.to_string()).await;
            }
        };

        let contexts: Vec<String> = retrieval
            .hits
            .iter()
            .map(|h| {
                if h.heading_path.is_empty() {
                    h.text.clone()
                } else {
                    format!("# {}\n{}", h.heading_path.join(" > "), h.text)
                }
            })
            .collect();

        let retrieved_ids: Vec<String> = retrieval.hit_ids();

        let prompt = rag_06::prompt::assemble_no_context(question);
        let user_prompt = format!(
            "【参考资料】\n{}\n\n【问题】\n{}",
            if contexts.is_empty() {
                "（本次检索没有召回任何资料片段）".to_string()
            } else {
                contexts.join("\n\n")
            },
            question
        );

        let answer = self
            .pipeline
            .llm()
            .chat(&prompt.system, &user_prompt)
            .await
            .map_err(|e| {
                warn!(question = %question, error = %e, "生成失败");
                EvalError::Retrieval(format!("生成失败: {e}"))
            })?;

        Ok(RunOutput {
            answer,
            contexts,
            retrieved_ids,
            retrieval_failed: false,
        })
    }

    /// 检索失败后的降级路径：空上下文 + 生成。
    async fn run_without_context(&self, question: &str, reason: &str) -> EvalResult<RunOutput> {
        let prompt = rag_06::prompt::assemble_no_context(question);
        let answer = self
            .pipeline
            .llm()
            .chat(&prompt.system, &prompt.user)
            .await
            .map_err(|e| EvalError::Retrieval(format!("生成失败: {e}")))?;

        Ok(RunOutput {
            answer,
            contexts: Vec::new(),
            retrieved_ids: Vec::new(),
            retrieval_failed: true,
        })
        .map(|mut o| {
            o.answer = format!("[检索失败: {reason}] {}", o.answer);
            o
        })
    }
}

/// 一次运行的输出。
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// 生成的答案。
    pub answer: String,

    /// 检索到的上下文。
    pub contexts: Vec<String>,

    /// 检索到的块 ID。
    pub retrieved_ids: Vec<String>,

    /// 检索是否失败（失败时 `contexts` 为空）。
    pub retrieval_failed: bool,
}

/// 评估流水线。
pub struct EvalPipeline {
    system: Arc<SystemUnderEval>,
    chunk_index: Arc<ChunkIndex>,
    index_stats: IndexStats,
    /// 评估并发度。
    ///
    /// ## 为什么需要它
    ///
    /// 每个样本要跑 1 次生成 + 4~5 次 judge 调用 —— 全是网络往返。
    /// 22 个样本串行跑要几分钟。
    ///
    /// 但并发**不是越高越好** —— 打爆上游限流会拿到一堆 429，
    /// 失败率上升反而更慢。所以默认 4。
    concurrency: usize,
}

impl std::fmt::Debug for EvalPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvalPipeline")
            .field("index", &self.index_stats)
            .field("concurrency", &self.concurrency)
            .finish()
    }
}

impl EvalPipeline {
    /// 创建流水线。
    pub fn new(
        system: Arc<SystemUnderEval>,
        chunk_index: Arc<ChunkIndex>,
        index_stats: IndexStats,
    ) -> Self {
        Self {
            system,
            chunk_index,
            index_stats,
            concurrency: DEFAULT_CONCURRENCY,
        }
    }

    /// 设置并发度。
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// 并发度。
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// 索引统计。
    pub fn index_stats(&self) -> &IndexStats {
        &self.index_stats
    }

    /// 块索引。
    pub fn chunk_index(&self) -> &Arc<ChunkIndex> {
        &self.chunk_index
    }

    /// 跑单个问题（不指定变体，用给定的检索模式）。
    ///
    /// 「快速评估」端点用这个 —— 它只需要一次检索 + 一次生成。
    pub async fn run_question(
        &self,
        question: &str,
        mode: SearchMode,
    ) -> EvalResult<crate::pipeline::RunOutput> {
        self.system.run_with_mode(question, mode).await
    }

    /// 跑一个变体，产出报告。
    ///
    /// ## 失败的处理
    ///
    /// - **生成失败** → 该样本被跳过（不进报告），记入失败计数
    /// - **指标失败** → 该指标在这个样本上记为失败，但样本仍然在报告里
    ///
    /// 第一条很重要：生成失败意味着「没有答案」，
    /// 硬凑一个空答案会让指标看起来像 0 分 —— 但实际上是「没跑出来」。
    ///
    /// ## ⚠️ 并发度要受限
    ///
    /// 每个样本要跑 1 次生成 + 4~5 次 judge 调用，全是网络请求。
    /// 22 个样本串行跑要好几分钟 —— 但无限并发会打爆上游的限流。
    ///
    /// 所以用 [`Semaphore`] 限流（默认 4）——
    /// 比串行快 4 倍，又不会触发 429。
    pub async fn run_variant(
        &self,
        dataset: &EvalDataset,
        variant: EvalVariant,
        evaluator: &crate::report::Evaluator,
    ) -> EvalResult<(EvalReport, usize)> {
        let start = Instant::now();

        let (contexts, failed) = self.run_cases(dataset, variant).await?;

        let title = format!("09-rag-eval · {variant:?} 评估");
        info!(
            variant = variant.name(),
            case_count = contexts.len(),
            failed,
            "变体评估完成"
        );
        let report = evaluator
            .evaluate(
                dataset,
                &contexts,
                &title,
                variant.name(),
                start.elapsed().as_millis() as u64,
            )
            .await;

        Ok((report, failed))
    }

    /// 并发跑所有样本，返回（成功的结果，失败数）。
    ///
    /// ## 为什么用 `JoinSet` 而不是 `futures::future::join_all`
    ///
    /// `join_all` 会等**全部**完成才返回 —— 一个样本卡住，整个评估就卡住。
    ///
    /// `JoinSet` 可以**逐个收割**已完成的任务（`join_next`），
    /// 所以慢的样本不会拖住快的样本的收割。
    ///
    /// ## ⚠️ 结果顺序
    ///
    /// 并发完成顺序是**不确定的** —— 所以最后按样本 ID 排序，
    /// 保证报告里的样本顺序稳定（否则两次运行的报告无法 diff）。
    async fn run_cases<'a>(
        &'a self,
        dataset: &'a EvalDataset,
        variant: EvalVariant,
    ) -> EvalResult<(Vec<EvalContext<'a>>, usize)> {
        use tokio::sync::Semaphore;

        let semaphore = Arc::new(Semaphore::new(self.concurrency));
        let mut set: tokio::task::JoinSet<EvalResult<(usize, RunOutput)>> =
            tokio::task::JoinSet::new();

        for (position, case) in dataset.cases.iter().enumerate() {
            let semaphore = Arc::clone(&semaphore);
            let system = Arc::clone(&self.system);
            // 任务必须 `'static` —— 所以把问题**克隆**进去，而不是借用。
            //
            // 克隆的只是一行字符串，成本可以忽略；
            // 而借用会导致「数据逃逸出方法」的编译错误。
            let question = case.question.clone();

            set.spawn(async move {
                // 获取许可 —— 满了就等。
                let _permit = semaphore
                    .acquire()
                    .await
                    .map_err(|e| EvalError::Internal(format!("并发控制失败: {e}")))?;

                let output = system.run(&question, variant).await?;
                Ok((position, output))
            });
        }

        let mut results: Vec<(usize, RunOutput)> = Vec::new();
        let mut failed = 0usize;

        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok((position, output))) => results.push((position, output)),
                Ok(Err(e)) => {
                    // 生成失败 —— 跳过这个样本。
                    failed += 1;
                    warn!(error = %e, "评估样本失败，已跳过");
                }
                Err(e) => {
                    // 任务本身 panic 了 —— 也算失败。
                    failed += 1;
                    warn!(error = %e, "评估任务失败，已跳过");
                }
            }
        }

        debug!(sample_count = results.len(), failed, "评估样本收集完成");

        // 按评估集顺序排序，保证报告稳定（否则两次运行的报告无法 diff）。
        results.sort_by_key(|(position, _)| *position);

        let contexts = results
            .into_iter()
            .map(|(position, output)| EvalContext {
                case: &dataset.cases[position],
                answer: output.answer,
                contexts: output.contexts,
                retrieved_ids: output.retrieved_ids,
                chunk_index: &self.chunk_index,
            })
            .collect();

        Ok((contexts, failed))
    }
}

/// 装配评估流水线（建索引 + 建块索引 + 装流水线）。
///
/// ## 为什么块索引要单独建
///
/// 因为评估集里标注的是**章节标题**，需要在运行时翻译成**块 ID**。
/// 而块 ID 依赖当前的分块配置 —— 所以必须用**和索引完全相同**的分块配置
/// 重新切一遍文档。
///
/// 代价是几毫秒（纯字符串操作，不需要模型）。换来的是：
/// **改 `chunk_size` 不会让评估集失效**。
pub async fn bootstrap(config: &AppConfig) -> EvalResult<EvalPipeline> {
    let (store, bm25, stats, _created) = index::ensure_index(config)
        .await
        .map_err(|e| EvalError::Retrieval(format!("准备索引失败: {e}")))?;

    // 用与索引相同的分块配置重建「章节 → 块 ID」映射。
    let chunk_index = build_chunk_index(&config.data_dir, &config.splitter)?;

    let embedder = index::make_embedder(config.embedding_model.to_model())
        .map_err(|e| EvalError::Internal(format!("创建嵌入器失败: {e}")))?;

    let retriever = Arc::new(rag_08::pipeline::make_retriever(
        store,
        bm25,
        Box::new(embedder),
        config,
    ));

    let reranker: Option<Arc<dyn rag_08::reranker::Reranker>> = if config.rerank.enabled {
        let r = rag_08::reranker::make_reranker(
            config.rerank.backend,
            config.rerank.model,
            Some(index::cache_dir()),
            config.rerank.cohere_api_key.as_deref(),
            &config.rerank.cohere_model,
        )
        .map_err(|e| EvalError::Internal(format!("创建重排器失败: {e}")))?;
        Some(r)
    } else {
        None
    };

    let llm = Arc::new(rag_08::llm::GenaiLlm::new(
        &config.generation.model,
        rag_08::llm::LlmOptions {
            temperature: config.generation.temperature,
            max_tokens: config.generation.max_tokens,
            // ⚠️ 生成侧**保留思考** —— 问答质量需要推理。
            // 这里给足 max_tokens（默认 2000）让思考 + 正文都放得下。
            disable_reasoning: false,
        },
    )) as Arc<dyn rag_08::llm::Llm>;

    let pipeline = Arc::new(HybridPipeline::new(
        retriever,
        reranker,
        llm,
        config.retrieval.clone(),
        config.rerank.clone(),
    ));

    Ok(EvalPipeline::new(
        Arc::new(SystemUnderEval::new(pipeline)),
        Arc::new(chunk_index),
        stats,
    ))
}

/// 创建嵌入器（供指标注册表使用）。
///
/// ## 为什么单独暴露
///
/// 语义相关度指标（answer relevancy / context relevancy）需要嵌入模型
/// 来算「原问题 ↔ 反推问题」的余弦相似度 —— 这和被评估系统用的是**同一个**
/// 嵌入模型（因为要保证向量空间一致）。
pub fn bootstrap_embedder(
    config: &rag_08::config::AppConfig,
) -> EvalResult<rag_05::embed::LocalEmbedder> {
    index::make_embedder(config.embedding_model.to_model())
        .map_err(|e| EvalError::Internal(format!("创建嵌入器失败: {e}")))
}

/// 建「章节 → 块 ID」的映射。
///
/// ## ⚠️ 必须和建索引用**完全相同**的分块配置
///
/// 否则块 ID 对不上 —— 评估集标注的相关块永远匹配不上，
/// 召回率恒为 0，而且**不会报任何错**。
///
/// 这是本项目最隐蔽的一个坑，所以在这里写清楚。
pub fn build_chunk_index(
    data_dir: &std::path::Path,
    splitter_config: &rag_04::splitter::SplitterConfig,
) -> EvalResult<ChunkIndex> {
    let counter = rag_04::splitter::TokenCounter::new()
        .map_err(|e| EvalError::Internal(format!("创建 token 计数器失败: {e}")))?;
    let splitter = rag_04::splitter::StructureAwareSplitter::new(splitter_config.clone());

    let documents = rag_04::loader::load_directory(data_dir)
        .map_err(|e| EvalError::Dataset(format!("加载文档失败: {e}")))?;

    let mut chunks: Vec<(String, Vec<String>)> = Vec::new();
    for doc in &documents {
        for chunk in splitter.split(doc, &counter) {
            chunks.push((
                rag_05::embed::chunk_id(&chunk.source, chunk.index),
                chunk.heading_path.clone(),
            ));
        }
    }

    Ok(ChunkIndex::build(&chunks))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_variant_names() {
        assert_eq!(EvalVariant::parse("hybrid"), Some(EvalVariant::Hybrid));
        assert_eq!(EvalVariant::parse("VECTOR"), Some(EvalVariant::Vector));
        assert_eq!(
            EvalVariant::parse("no_retrieval"),
            Some(EvalVariant::NoRetrieval)
        );
        assert_eq!(EvalVariant::parse("unknown"), None);
    }

    #[test]
    fn all_variants_have_search_mode_except_no_retrieval() {
        for variant in EvalVariant::all() {
            if variant == EvalVariant::NoRetrieval {
                assert!(variant.search_mode().is_none());
            } else {
                assert!(variant.search_mode().is_some());
            }
        }
    }

    #[test]
    fn chunk_index_matches_knowledge_base_sections() {
        // ⚠️ 知识库在 08-rag/data —— 09 自己的 data/ 放的是评估集（json）。
        let data_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../08-rag/data");
        let index =
            build_chunk_index(&data_dir, &rag_04::splitter::SplitterConfig::default()).unwrap();

        assert!(index.chunk_count() > 0);
        assert!(
            !index.ids_by_section("BM25 打分算法").is_empty(),
            "知识库里应该能找到「BM25 打分算法」章节"
        );
    }
}
