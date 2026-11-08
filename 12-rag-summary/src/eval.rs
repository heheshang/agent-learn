//! 评估 —— 消融实验矩阵。
//!
//! ## 本项目要回答的问题
//!
//! 前 11 个项目各自证明了一件事：
//!
//! - 08-rag：混合检索 + 重排 比纯向量好
//! - 09-rag-eval：怎么**量化**「好多少」
//! - 10-rag-milvus：换成生产向量库后效果如何
//!
//! 本项目要把它们**叠在一起**，然后问一个更具体的问题：
//!
//! > **从「纯向量检索」到「全量优化」，每一步各贡献了多少？**
//!
//! ## 消融阶梯
//!
//! ```text
//! dense ──> keyword ──> hybrid ──> hybrid_rerank ──> full
//!  ▲          ▲           ▲            ▲              ▲
//!  │          │           │            │              │
//!  基线      对照组    +混合融合    +交叉编码器    +改写+压缩
//! ```
//!
//! 每一档都是「上一档 + 一个增强」，所以相邻两档的差值是**可归因**的 ——
//! 这正是「消融实验」的意义。
//!
//! ## 指标复用 09-rag-eval
//!
//! 评估集、指标注册表、报告结构、对比报告**全部复用** `rag_09`：
//!
//! - 确定性指标：`recall_at_k`、`precision_at_k`、`ndcg_at_k`、`mrr`、
//!   `context_precision_by_id`、`context_recall_by_id`、`token_f1`、`exact_match`
//! - judge 指标：`faithfulness`、`answer_correctness`、`answer_relevancy`
//!
//! 所以 12 的报告可以和 09 的报告**直接对比** —— 同一套尺子。

use std::sync::Arc;
use std::time::Instant;

use rag_09::dataset::{ChunkIndex, EvalDataset};
use rag_09::metrics::{EvalContext, MetricRegistry};
use rag_09::report::{ComparisonReport, EvalReport, Evaluator, Verdict};

use crate::config::SearchMode;
use rag_04::splitter::Splitter;

use crate::error::{RagError, RagResult};
use crate::pipeline::SummaryPipeline;

/// 默认评估并发度。
///
/// 每个样本要跑 1 次生成 + 4~5 次 judge 调用 —— 全是网络往返。
pub const DEFAULT_CONCURRENCY: usize = 4;

/// 一次评估运行的输出。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VariantResult {
    /// 变体名。
    pub variant: String,

    /// 评估报告。
    pub report: EvalReport,

    /// 失败的样本数（生成失败 → 跳过，不进报告）。
    pub failed: usize,
}

/// 消融实验结果 —— 多个变体的报告 + 相邻档位的差值。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AblationResult {
    /// 各变体的报告（按阶梯顺序）。
    pub reports: Vec<VariantResult>,

    /// 相邻档位的对比（`reports[i]` vs `reports[i+1]`）。
    pub steps: Vec<ComparisonReport>,
}

impl AblationResult {
    /// 总耗时（毫秒）。
    pub fn elapsed_ms(&self) -> u64 {
        self.reports.iter().map(|r| r.report.elapsed_ms).sum()
    }

    /// 渲染成对比表格。
    ///
    /// 每一档一行，列出关键指标的平均分。
    pub fn to_table(&self) -> String {
        let mut table = crate::table::Table::new(&[
            "变体", "recall@5", "ndcg@5", "mrr", "ctx_rec", "ctx_prec", "token_f1", "耗时",
        ])
        .right_align(&[1, 2, 3, 4, 5, 6, 7]);

        for result in &self.reports {
            let report = &result.report;

            table.row(&[
                &result.variant,
                &format!("{:.3}", report.average("recall_at_k").unwrap_or(0.0)),
                &format!("{:.3}", report.average("ndcg_at_k").unwrap_or(0.0)),
                &format!("{:.3}", report.average("mrr").unwrap_or(0.0)),
                &format!(
                    "{:.3}",
                    report.average("context_recall_by_id").unwrap_or(0.0)
                ),
                &format!(
                    "{:.3}",
                    report.average("context_precision_by_id").unwrap_or(0.0)
                ),
                &format!("{:.3}", report.average("token_f1").unwrap_or(0.0)),
                &format!("{}s", report.elapsed_ms / 1000),
            ]);
        }

        table.render()
    }

    /// 渲染成「每一步贡献了多少」的表格。
    ///
    /// ## ⚠️ 这里只列**检索侧**的指标
    ///
    /// `token_f1` / `exact_match` 这类**生成侧**指标也会列出来，
    /// 但要小心解读：它们衡量的是「答案和标准答案的字面重合度」，
    /// 而摘要式任务的答案**本来就不该**和标准答案逐字相同。
    ///
    /// 所以看这两项时，只该看**相对变化**（这一档比上一档好了还是差了），
    /// 不该看绝对值。
    pub fn to_step_table(&self) -> String {
        let mut out = String::new();

        for (index, step) in self.steps.iter().enumerate() {
            let from = &self.reports[index].variant;
            let to = &self.reports[index + 1].variant;

            out.push_str(&format!("\n{from} → {to}\n"));

            let mut table = crate::table::Table::new(&["指标", "基线", "优化后", "差值", ""])
                .right_align(&[1, 2, 3]);

            for delta in &step.deltas {
                let mark = match delta.verdict {
                    Verdict::Improved => "↑ 改善",
                    Verdict::Regressed => "↓ 退步",
                    Verdict::Unchanged => "= 持平",
                };

                table.row(&[
                    &delta.name,
                    &format!("{:.4}", delta.baseline),
                    &format!("{:.4}", delta.optimized),
                    &format!("{:+.4}", delta.delta),
                    mark,
                ]);
            }

            out.push_str(&table.render());
        }

        out
    }
}

/// 评估器 —— 跑消融实验矩阵。
pub struct AblationRunner {
    pipeline: Arc<SummaryPipeline>,
    chunk_index: Arc<ChunkIndex>,
    evaluator: Evaluator,
    concurrency: usize,
}

impl std::fmt::Debug for AblationRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AblationRunner")
            .field("concurrency", &self.concurrency)
            .finish()
    }
}

impl AblationRunner {
    /// 创建评估器。
    pub fn new(
        pipeline: Arc<SummaryPipeline>,
        chunk_index: Arc<ChunkIndex>,
        registry: MetricRegistry,
    ) -> Self {
        Self {
            pipeline,
            chunk_index,
            evaluator: Evaluator::new(registry.metrics().to_vec()),
            concurrency: DEFAULT_CONCURRENCY,
        }
    }

    /// 设置并发度。
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// 跑完整消融阶梯。
    ///
    /// 返回所有变体的报告 + 相邻档位的差值。
    pub async fn run_ladder(
        &self,
        dataset: &EvalDataset,
        modes: &[SearchMode],
    ) -> RagResult<AblationResult> {
        self.run_ladder_limited(dataset, modes, None).await
    }

    /// 跑消融阶梯（可限制每个变体的样本数）。
    pub async fn run_ladder_limited(
        &self,
        dataset: &EvalDataset,
        modes: &[SearchMode],
        limit: Option<usize>,
    ) -> RagResult<AblationResult> {
        let mut reports = Vec::with_capacity(modes.len());

        for mode in modes {
            let result = self.run_variant_limited(dataset, *mode, limit).await?;
            reports.push(result);
        }

        let steps = build_steps(&reports);

        Ok(AblationResult { reports, steps })
    }

    /// 跑单个变体（可限制样本数 —— 快速冒烟用）。
    ///
    /// ## ⚠️ 为什么需要限制样本数
    ///
    /// 全量评估是「22 样本 × 5 变体 × (1 次生成 + N 次指标)」——
    /// 每次都是网络往返，串行跑要好几分钟。
    ///
    /// 调试时只想确认「链路通不通」，跑 3 个样本就够了。
    /// 所以这里支持 `limit`：**取评估集的前 N 条**（保持顺序，
    /// 这样结果可复现）。
    pub async fn run_variant_limited(
        &self,
        dataset: &EvalDataset,
        mode: SearchMode,
        limit: Option<usize>,
    ) -> RagResult<VariantResult> {
        let start = Instant::now();

        let (contexts, failed) = self.run_cases(dataset, mode, limit).await?;

        let title = format!("12-rag-summary · {} 评估", mode.name());
        let report = self
            .evaluator
            .evaluate(
                dataset,
                &contexts,
                &title,
                mode.name(),
                start.elapsed().as_millis() as u64,
            )
            .await;

        Ok(VariantResult {
            variant: mode.name().to_string(),
            report,
            failed,
        })
    }

    /// 跑单个变体（全量）。
    pub async fn run_variant(
        &self,
        dataset: &EvalDataset,
        mode: SearchMode,
    ) -> RagResult<VariantResult> {
        self.run_variant_limited(dataset, mode, None).await
    }

    /// 并发跑所有样本。
    ///
    /// ## 失败的处理
    ///
    /// - **生成失败** → 该样本被跳过（不进报告），记入失败计数
    /// - **指标失败** → 该指标在这个样本上记为失败，但样本仍然在报告里
    ///
    /// 第一条很重要：生成失败意味着「没有答案」，
    /// 硬凑一个空答案会让指标看起来像 0 分 —— 但实际上是「没跑出来」。
    async fn run_cases<'a>(
        &'a self,
        dataset: &'a EvalDataset,
        mode: SearchMode,
        limit: Option<usize>,
    ) -> RagResult<(Vec<EvalContext<'a>>, usize)> {
        use tokio::sync::Semaphore;

        let semaphore = Arc::new(Semaphore::new(self.concurrency));
        let mut set: tokio::task::JoinSet<RagResult<(usize, RunOutput)>> =
            tokio::task::JoinSet::new();

        // ⚠️ 限制样本数时取**前 N 条**（保持顺序）——
        // 这样两次运行的结果可以直接 diff。
        let take = limit
            .unwrap_or(dataset.cases.len())
            .min(dataset.cases.len());

        for (position, case) in dataset.cases.iter().take(take).enumerate() {
            let semaphore = Arc::clone(&semaphore);
            let pipeline = Arc::clone(&self.pipeline);
            // 任务必须 `'static` —— 所以把问题**克隆**进去。
            let question = case.question.clone();

            set.spawn(async move {
                let _permit = semaphore
                    .acquire()
                    .await
                    .map_err(|e| RagError::Internal(format!("并发控制失败: {e}")))?;

                let output = run_one(&pipeline, &question, mode).await?;
                Ok((position, output))
            });
        }

        let mut results: Vec<(usize, RunOutput)> = Vec::new();
        let mut failed = 0usize;

        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok((position, output))) => results.push((position, output)),
                Ok(Err(_)) => failed += 1,
                Err(e) => {
                    failed += 1;
                    eprintln!("  ⚠ 评估任务失败: {e}");
                }
            }
        }

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

/// 一次样本运行的输出。
struct RunOutput {
    answer: String,
    contexts: Vec<String>,
    retrieved_ids: Vec<String>,
}

/// 跑一个样本。
async fn run_one(
    pipeline: &SummaryPipeline,
    question: &str,
    mode: SearchMode,
) -> RagResult<RunOutput> {
    let outcome = pipeline.ask_with(question, mode).await?;

    Ok(RunOutput {
        answer: outcome.answer,
        contexts: outcome
            .compression
            .stats
            .passages
            .iter()
            .map(|p| p.text.clone())
            .collect(),
        retrieved_ids: outcome
            .compression
            .stats
            .passages
            .iter()
            .map(|p| p.id.clone())
            .collect(),
    })
}

/// 构建相邻档位的对比报告。
fn build_steps(reports: &[VariantResult]) -> Vec<ComparisonReport> {
    let mut steps = Vec::new();

    for index in 0..reports.len().saturating_sub(1) {
        let baseline = &reports[index].report;
        let optimized = &reports[index + 1].report;

        steps.push(ComparisonReport::compare(baseline, optimized));
    }

    steps
}

/// 建「章节标题 → 块 ID」的映射。
///
/// ## ⚠️ 必须和建索引用**完全相同**的分块配置
///
/// 否则块 ID 对不上 —— 评估集标注的相关块永远匹配不上，
/// 召回率恒为 0，而且**不会报任何错**。
pub fn build_chunk_index(
    data_dir: &std::path::Path,
    splitter_config: &rag_04::splitter::SplitterConfig,
) -> RagResult<ChunkIndex> {
    let counter = rag_04::splitter::TokenCounter::new()
        .map_err(|e| RagError::Internal(format!("创建 token 计数器失败: {e}")))?;
    let splitter = rag_04::splitter::StructureAwareSplitter::new(splitter_config.clone());

    let documents = rag_04::loader::load_directory(data_dir)
        .map_err(|e| RagError::Internal(format!("加载文档失败: {e}")))?;

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
    fn chunk_index_matches_knowledge_base_sections() {
        let data_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let index =
            build_chunk_index(&data_dir, &rag_04::splitter::SplitterConfig::default()).unwrap();

        assert!(index.chunk_count() > 0);
        assert!(
            !index.ids_by_section("BM25 打分算法").is_empty(),
            "知识库里应该能找到「BM25 打分算法」章节"
        );
    }

    #[test]
    fn steps_compare_adjacent_variants() {
        // 构造两个空报告，验证 step 数量 = 报告数 - 1。
        let reports: Vec<VariantResult> = (0..3)
            .map(|i| VariantResult {
                variant: format!("v{i}"),
                report: EvalReport {
                    title: format!("v{i}"),
                    variant: format!("v{i}"),
                    dataset: "d".to_string(),
                    case_count: 0,
                    summary: Vec::new(),
                    cases: Vec::new(),
                    elapsed_ms: 0,
                },
                failed: 0,
            })
            .collect();

        let steps = build_steps(&reports);
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn single_variant_produces_no_steps() {
        let reports = vec![VariantResult {
            variant: "only".to_string(),
            report: EvalReport {
                title: "only".to_string(),
                variant: "only".to_string(),
                dataset: "d".to_string(),
                case_count: 0,
                summary: Vec::new(),
                cases: Vec::new(),
                elapsed_ms: 0,
            },
            failed: 0,
        }];

        assert!(build_steps(&reports).is_empty());
    }
}
