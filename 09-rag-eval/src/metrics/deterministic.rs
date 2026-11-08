//! 确定性指标 —— 不需要 LLM 的那一半。
//!
//! ## 为什么要有这一组
//!
//! 评估指标分两大类：
//!
//! | 类型 | 例子 | 特点 |
//! |---|---|---|
//! | **确定性** | token F1、recall@k、nDCG、MRR | 纯计算，秒级，无 API 费用，**完全可复现** |
//! | **LLM-as-a-judge** | faithfulness、answer relevancy | 要调模型，慢、贵、有随机性 |
//!
//! 如果只有 judge 指标，那每次评估的分数都会有波动 ——
//! 「优化后 faithfulness 从 0.62 涨到 0.68」这个结论**不可信**，
//! 因为 judge 本身的抖动就有 ±0.05。
//!
//! 所以：**用确定性指标锚定「检索准不准」，用 judge 指标衡量「答案好不好」**。
//! 前者是地基，后者是补充。
//!
//! ## 这一组覆盖的工具谱系
//!
//! | 指标 | 来自 |
//! |---|---|
//! | `recall_at_k` / `precision_at_k` / `ndcg` / `mrr` | FlashRAG 的 `retrieval_doc_*` 系列 |
//! | `token_f1` / `exact_match` | FlashRAG 的 EM / F1 系列 |
//! | `context_precision_by_id` / `context_recall_by_id` | RAGAs 的 context 指标（ID 版） |

use std::collections::HashSet;

use crate::metrics::types::{EvalContext, Metric, MetricFuture, MetricScore};

/// Token 级 F1 —— 答案和标准答案的 token 交集。
///
/// ## 公式
///
/// ```text
///            2 · |答案tokens ∩ 标准tokens|
///  F1 = ─────────────────────────────
///         |答案tokens| + |标准tokens|
/// ```
///
/// ## 为什么用 token F1 而不是 exact match
///
/// exact match 要求答案**逐字相同** —— 对 RAG 来说几乎不可能达标：
/// 同一个问题，模型每次措辞都不同，但意思是对的。
///
/// token F1 允许「说法不同、词用对了」也能拿到分。
///
/// ## 分词
///
/// 复用 08-rag 的 [`rag_08::tokenize::tokenize`]（CJK unigram + bigram，
/// 拉丁整词小写）—— 和 BM25 用的是同一套分词，保证全项目口径一致。
pub struct TokenF1;

impl TokenF1 {
    /// 创建。
    pub fn new() -> Self {
        Self
    }
}

impl Default for TokenF1 {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for TokenF1 {
    fn name(&self) -> &str {
        "token_f1"
    }

    fn description(&self) -> &str {
        "答案与标准答案的 token 级 F1（FlashRAG 的 F1 指标）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            let reference = ctx.case.reference_answer.trim();
            if reference.is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "评估集没有标准答案，无法计算 token F1",
                ));
            }

            let answer_tokens = rag_08::tokenize::tokenize(&ctx.answer);
            let reference_tokens = rag_08::tokenize::tokenize(reference);

            if answer_tokens.is_empty() || reference_tokens.is_empty() {
                return Ok(MetricScore::with_reason(0.0, "答案或标准答案分词后为空"));
            }

            let answer_set: HashSet<&str> = answer_tokens.iter().map(|s| s.as_str()).collect();
            let reference_set: HashSet<&str> =
                reference_tokens.iter().map(|s| s.as_str()).collect();

            let overlap = answer_set.intersection(&reference_set).count();

            // ⚠️ 分母用**去重后的**集合大小 —— 否则答案里重复同一个词
            // 会虚增交集、压低 precision。
            let precision = overlap as f32 / answer_set.len() as f32;
            let recall = overlap as f32 / reference_set.len() as f32;

            let f1 = if precision + recall <= f32::EPSILON {
                0.0
            } else {
                2.0 * precision * recall / (precision + recall)
            };

            Ok(MetricScore::with_reason(
                f1,
                format!("precision={precision:.3} recall={recall:.3} 共同 token={overlap}"),
            ))
        })
    }

    fn threshold(&self) -> f32 {
        0.5
    }
}

/// Exact Match —— 归一化后答案是否与标准答案完全一致。
///
/// ## 为什么几乎没有 RAG 系统能拿高分
///
/// 这个指标在开放域问答里近乎无用 —— 措辞不同就 0 分。
/// 但它有一个不可替代的用途：**作为下界参照**。
///
/// 如果 token F1 是 0.7 而 EM 是 0，说明答案「意思对但说法不同」——
/// 这是健康的。如果 EM 反而比 token F1 的某个子集还高，说明归一化逻辑有问题。
pub struct ExactMatch;

impl Metric for ExactMatch {
    fn name(&self) -> &str {
        "exact_match"
    }

    fn description(&self) -> &str {
        "答案与标准答案归一化后是否完全一致（FlashRAG 的 EM 指标）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            let reference = ctx.case.reference_answer.trim();
            if reference.is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "评估集没有标准答案，无法计算 exact match",
                ));
            }

            let answer = normalize_answer(&ctx.answer);
            let reference = normalize_answer(reference);

            let matched = answer == reference && !answer.is_empty();

            Ok(MetricScore::with_reason(
                if matched { 1.0 } else { 0.0 },
                if matched {
                    "完全一致".to_string()
                } else {
                    "存在差异".to_string()
                },
            ))
        })
    }
}

/// 归一化答案 —— 去掉空白和标点，只留下内容词。
fn normalize_answer(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Recall@k —— 前 k 条检索结果里命中了多少相关块。
///
/// ## 公式
///
/// ```text
///         |检索前 k 条 ∩ 相关块|
///  R@k = ────────────────────────
///              |相关块|
/// ```
///
/// ## 这是最核心的检索指标
///
/// 因为它直接回答：**正确答案在不在候选集里？**
///
/// ⚠️ 不在候选集里，后面的一切（重排、生成）都救不回来 ——
/// 这就是「召回是天花板」的意思。
pub struct RecallAtK {
    k: usize,
}

impl RecallAtK {
    /// 创建（`k` 默认 5）。
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Metric for RecallAtK {
    fn name(&self) -> &str {
        "recall_at_k"
    }

    fn description(&self) -> &str {
        "前 k 条检索结果的召回率（FlashRAG 的 retrieval_doc_recall）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            let relevant = ctx.case.resolve_relevant_ids(ctx.chunk_index);
            if relevant.is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "评估集没有标注相关块，无法计算召回率",
                ));
            }

            let relevant_set: HashSet<&str> = relevant.iter().map(|s| s.as_str()).collect();
            let retrieved: Vec<&str> = ctx
                .retrieved_ids
                .iter()
                .take(self.k)
                .map(|s| s.as_str())
                .collect();

            let hits = retrieved
                .iter()
                .filter(|id| relevant_set.contains(*id))
                .count();

            let recall = hits as f32 / relevant_set.len() as f32;

            Ok(MetricScore::with_reason(
                recall,
                format!(
                    "前 {} 条命中 {}/{} 个相关块",
                    self.k,
                    hits,
                    relevant_set.len()
                ),
            ))
        })
    }

    fn threshold(&self) -> f32 {
        0.8
    }
}

/// Precision@k —— 前 k 条检索结果里有多少是相关的。
///
/// 与 recall 的区别：recall 看「有没有捞到」，precision 看「捞进来的是不是都对」。
///
/// 两者要一起看：
///
/// | recall | precision | 含义 |
/// |---|---|---|
/// | 高 | 高 | 完美 |
/// | 低 | 高 | 相关块没被召回 —— 候选池太小 / 排序太靠后 |
/// | 高 | 低 | 召回了但混进大量噪声 —— 需要更严的阈值 |
/// | 低 | 低 | 检索基本失效 |
pub struct PrecisionAtK {
    k: usize,
}

impl PrecisionAtK {
    /// 创建（`k` 默认 5）。
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Metric for PrecisionAtK {
    fn name(&self) -> &str {
        "precision_at_k"
    }

    fn description(&self) -> &str {
        "前 k 条检索结果的精确率（FlashRAG 的 retrieval_doc_precision）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            let relevant = ctx.case.resolve_relevant_ids(ctx.chunk_index);
            let retrieved: Vec<&str> = ctx
                .retrieved_ids
                .iter()
                .take(self.k)
                .map(|s| s.as_str())
                .collect();

            if retrieved.is_empty() {
                return Ok(MetricScore::with_reason(0.0, "没有检索到任何内容"));
            }

            if relevant.is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "评估集没有标注相关块，无法计算精确率",
                ));
            }

            let relevant_set: HashSet<&str> = relevant.iter().map(|s| s.as_str()).collect();
            let hits = retrieved
                .iter()
                .filter(|id| relevant_set.contains(*id))
                .count();

            Ok(MetricScore::with_reason(
                hits as f32 / retrieved.len() as f32,
                format!("前 {} 条中 {hits} 条相关", self.k),
            ))
        })
    }
}

/// nDCG@k —— 归一化折损累积增益。
///
/// ## 公式
///
/// ```text
///                       Σ (rel_i · 2^-i)          ← 实际排序
///  nDCG@k = ─────────────────────────────────
///                       Σ rel_i · 2^-i            ← 理想排序
///         i=1..k                                 i=1..k
/// ```
///
/// ## 为什么用 nDCG 而不是 recall
///
/// recall 只看「有没有」，**完全忽略位置**。
///
/// 假设正确答案排在第 1 位和排在第 5 位 —— recall 一样。
/// 但前者几乎必然被用到（模型先看到最相关的），后者很可能被噪声挤掉。
///
/// nDCG 用 `2^-i` 给靠后的位置打折，正好刻画了「位置很重要」这件事。
///
/// ## ⚠️ 这里的增益是二值的
///
/// 相关块算 1，不相关算 0 —— 因为我们的标注是「相关 / 不相关」，
/// 没有分级相关性。用分级增益需要标注 3 个等级，成本高很多。
pub struct NdcgAtK {
    k: usize,
}

impl NdcgAtK {
    /// 创建（`k` 默认 5）。
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Metric for NdcgAtK {
    fn name(&self) -> &str {
        "ndcg_at_k"
    }

    fn description(&self) -> &str {
        "前 k 条检索结果的归一化折损累积增益（位置加权的排序质量）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            let relevant = ctx.case.resolve_relevant_ids(ctx.chunk_index);
            if relevant.is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "评估集没有标注相关块，无法计算 nDCG",
                ));
            }

            let relevant_set: HashSet<&str> = relevant.iter().map(|s| s.as_str()).collect();
            let retrieved: Vec<&str> = ctx
                .retrieved_ids
                .iter()
                .take(self.k)
                .map(|s| s.as_str())
                .collect();

            if retrieved.is_empty() {
                return Ok(MetricScore::with_reason(0.0, "没有检索到任何内容"));
            }

            // 实际 DCG。
            let dcg: f32 = retrieved
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    let rel = if relevant_set.contains(*id) { 1.0 } else { 0.0 };
                    rel * 2f32.powf(-(i as f32 + 1.0))
                })
                .sum();

            // 理想 DCG：相关块全部排在最前面。
            let relevant_count = relevant_set.len().min(self.k);
            let idcg: f32 = (0..relevant_count)
                .map(|i| 2f32.powf(-(i as f32 + 1.0)))
                .sum();

            if idcg <= f32::EPSILON {
                return Ok(MetricScore::with_reason(0.0, "理想 DCG 为零"));
            }

            Ok(MetricScore::with_reason(
                dcg / idcg,
                format!("DCG={dcg:.4} IDCG={idcg:.4}"),
            ))
        })
    }

    fn threshold(&self) -> f32 {
        0.8
    }
}

/// MRR —— 平均倒数排名（Mean Reciprocal Rank）。
///
/// ## 公式
///
/// ```text
///  MRR = (1/N) · Σ 1 / rank_i
/// ```
///
/// `rank_i` 是第 i 个查询的**第一个**相关结果的位置（1 开始）。
///
/// ## 和 nDCG 的区别
///
/// MRR **只看第一个正确结果在哪**，完全忽略后面的。
/// nDCG 看整个排序。
///
/// 什么时候用哪个：
///
/// - **MRR** —— 用户问一个问题，答案只需要「有且有一个对的」，
///   第一个正确结果出现得越早越好（典型的问答场景）
/// - **nDCG** —— 需要多个相关信息都排上来（比如「分别解释 A 和 B」）
pub struct Mrr;

impl Metric for Mrr {
    fn name(&self) -> &str {
        "mrr"
    }

    fn description(&self) -> &str {
        "第一个相关结果排名的倒数（第一个正确结果出现得越早分越高）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            let relevant = ctx.case.resolve_relevant_ids(ctx.chunk_index);
            if relevant.is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "评估集没有标注相关块，无法计算 MRR",
                ));
            }

            let relevant_set: HashSet<&str> = relevant.iter().map(|s| s.as_str()).collect();

            let reciprocal = ctx
                .retrieved_ids
                .iter()
                .position(|id| relevant_set.contains(id.as_str()))
                .map(|i| 1.0 / (i as f32 + 1.0))
                .unwrap_or(0.0);

            Ok(MetricScore::with_reason(
                reciprocal,
                if reciprocal > 0.0 {
                    format!(
                        "第一个相关结果排在第 {} 位",
                        (1.0 / reciprocal).round() as usize
                    )
                } else {
                    "没有检索到相关块".to_string()
                },
            ))
        })
    }

    fn threshold(&self) -> f32 {
        0.8
    }
}

/// Context Precision（ID 版）—— 检索结果里「相关块」的比例。
///
/// ## 和 RAGAs 官方版本的区别
///
/// RAGAs 的 `context_precision` 是**用 LLM 逐句判断**「这句话和问题相关吗」，
/// 再按位置加权平均。
///
/// 这个版本用**标注的相关块 ID** 来算 —— 精确、免费、可复现，
/// 但前提是评估集标注了相关块。
///
/// 两个版本**互补**：ID 版能告诉你「检索有没有找对东西」，
/// LLM 版能告诉你「即便找对了块，块里的内容有多少是废话」。
///
/// ## ⚠️ 这个指标在小知识库上会「天然偏低」
///
/// 分母是**实际送进上下文的块数**。如果知识库只有 11 个块、
/// `top_k=4`，那么每个问题最多只能召回 4 个块 —— 其中只有 1 个是相关的，
/// 剩下 3 个**必然是噪声**。
///
/// ```text
/// context_precision_by_id = 1 / 4 = 0.25
/// ```
///
/// 这不是检索质量的 bug，而是**评估集规模**和**知识库规模**不匹配。
/// 真实的知识库有几千个块时，这个指标才有意义。
///
/// 所以读这个数字时要配合看 `context_recall_by_id`：
///
/// - recall 高 + precision 低 → 召回没问题，但候选池里噪声多（知识库太小）
/// - recall 低 + precision 低 → 检索确实有问题
pub struct ContextPrecisionById;

impl Metric for ContextPrecisionById {
    fn name(&self) -> &str {
        "context_precision_by_id"
    }

    fn description(&self) -> &str {
        "检索结果中相关块的比例（ID 版，无需 LLM）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            let relevant = ctx.case.resolve_relevant_ids(ctx.chunk_index);
            if relevant.is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "评估集没有标注相关块，无法计算上下文精确率",
                ));
            }

            if ctx.retrieved_ids.is_empty() {
                return Ok(MetricScore::with_reason(0.0, "没有检索到任何内容"));
            }

            let relevant_set: HashSet<&str> = relevant.iter().map(|s| s.as_str()).collect();
            let hits = ctx
                .retrieved_ids
                .iter()
                .filter(|id| relevant_set.contains(id.as_str()))
                .count();

            Ok(MetricScore::with_reason(
                hits as f32 / ctx.retrieved_ids.len() as f32,
                format!("{} 条检索结果中 {hits} 条是相关块", ctx.retrieved_ids.len()),
            ))
        })
    }
}

/// Context Recall（ID 版）—— 相关块里有多少被检索到了。
///
/// 这是**检索侧的底线指标**：
/// 相关块没被召回 → 模型不可能基于它作答 → 答案再好也是编的。
///
/// 所以它和 `faithfulness`（答案是否有据可依）是**因果关系**：
/// 上下文召回低 → 忠实度必然低。
pub struct ContextRecallById;

impl Metric for ContextRecallById {
    fn name(&self) -> &str {
        "context_recall_by_id"
    }

    fn description(&self) -> &str {
        "相关块被检索到的比例（ID 版，无需 LLM）"
    }

    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a> {
        Box::pin(async move {
            let relevant = ctx.case.resolve_relevant_ids(ctx.chunk_index);
            if relevant.is_empty() {
                return Ok(MetricScore::with_reason(
                    0.0,
                    "评估集没有标注相关块，无法计算上下文召回率",
                ));
            }

            if ctx.retrieved_ids.is_empty() {
                return Ok(MetricScore::with_reason(0.0, "没有检索到任何内容"));
            }

            let retrieved_set: HashSet<&str> =
                ctx.retrieved_ids.iter().map(|s| s.as_str()).collect();
            let hits = relevant
                .iter()
                .filter(|id| retrieved_set.contains(id.as_str()))
                .count();

            Ok(MetricScore::with_reason(
                hits as f32 / relevant.len() as f32,
                format!("{hits}/{} 个相关块被召回", relevant.len()),
            ))
        })
    }

    fn threshold(&self) -> f32 {
        0.9
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{ChunkIndex, EvalCase};

    fn case() -> EvalCase {
        EvalCase::new("c1", "BM25 打分公式是什么")
            .with_answer("BM25 由 IDF 和词频饱和组成")
            .with_relevant_ids(vec!["kb.md#0".to_string(), "kb.md#1".to_string()])
    }

    fn chunk_index() -> ChunkIndex {
        ChunkIndex::build(&[
            ("kb.md#0".to_string(), vec!["BM25 打分算法".to_string()]),
            ("kb.md#1".to_string(), vec!["BM25 打分算法".to_string()]),
        ])
    }

    /// 测试夹具 —— 把「样本 + 块索引」打包，避免每个用例都要传两个参数。
    struct Fixture {
        case: EvalCase,
        index: ChunkIndex,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                case: case(),
                index: chunk_index(),
            }
        }

        fn ctx(&self, answer: &str, retrieved: &[&str]) -> EvalContext<'_> {
            EvalContext {
                case: &self.case,
                answer: answer.to_string(),
                contexts: retrieved.iter().map(|s| s.to_string()).collect(),
                retrieved_ids: retrieved.iter().map(|s| s.to_string()).collect(),
                chunk_index: &self.index,
            }
        }
    }

    #[tokio::test]
    async fn token_f1_counts_overlap() {
        let f = Fixture::new();
        let ctx = f.ctx("BM25 由 IDF 和词频组成", &[]);

        let score = TokenF1::new().score(&ctx).await.unwrap();
        assert!(
            score.value > 0.5,
            "有共同 token，分数应该不低：{}",
            score.value
        );
        assert!(score.value < 1.0, "不完全一致，分数不应为 1");
    }

    #[tokio::test]
    async fn token_f1_is_zero_for_unrelated_answer() {
        let f = Fixture::new();
        let ctx = f.ctx("光合作用发生在类囊体膜", &[]);

        let score = TokenF1::new().score(&ctx).await.unwrap();
        assert_eq!(score.value, 0.0);
    }

    #[tokio::test]
    async fn token_f1_requires_reference_answer() {
        let f = Fixture {
            case: EvalCase::new("c1", "问题"),
            index: chunk_index(),
        };
        let ctx = f.ctx("答案", &[]);

        let score = TokenF1::new().score(&ctx).await.unwrap();
        assert_eq!(score.value, 0.0);
        assert!(score.reason.unwrap().contains("标准答案"));
    }

    #[tokio::test]
    async fn exact_match_normalizes_punctuation() {
        let f = Fixture::new();
        let ctx = f.ctx("BM25由IDF和词频饱和组成", &[]);

        let score = ExactMatch.score(&ctx).await.unwrap();
        assert_eq!(score.value, 1.0);
    }

    #[tokio::test]
    async fn recall_at_k_counts_hits() {
        let f = Fixture::new();
        let ctx = f.ctx("答案", &["kb.md#0", "kb.md#9"]);

        // 1/2 个相关块被召回。
        let score = RecallAtK::new(5).score(&ctx).await.unwrap();
        assert_eq!(score.value, 0.5);
    }

    #[tokio::test]
    async fn recall_at_k_respects_k_cutoff() {
        let f = Fixture::new();
        // 相关块在第 3 位，k=2 时不该被算进去。
        let ctx = f.ctx("答案", &["kb.md#9", "kb.md#8", "kb.md#0"]);

        let score = RecallAtK::new(2).score(&ctx).await.unwrap();
        assert_eq!(score.value, 0.0);
    }

    #[tokio::test]
    async fn precision_at_k_counts_ratio() {
        let f = Fixture::new();
        let ctx = f.ctx("答案", &["kb.md#0", "kb.md#1", "kb.md#9"]);

        let score = PrecisionAtK::new(3).score(&ctx).await.unwrap();
        assert!((score.value - 2.0 / 3.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn ndcg_rewards_early_positions() {
        let f = Fixture::new();
        let early = f.ctx("答案", &["kb.md#0", "kb.md#1"]);
        let late = f.ctx("答案", &["kb.md#9", "kb.md#0"]);

        let early_score = NdcgAtK::new(5).score(&early).await.unwrap();
        let late_score = NdcgAtK::new(5).score(&late).await.unwrap();

        assert!(
            early_score.value > late_score.value,
            "相关块排前面应该分更高：{} vs {}",
            early_score.value,
            late_score.value
        );
    }

    #[tokio::test]
    async fn ndcg_is_one_when_perfect() {
        let f = Fixture::new();
        let ctx = f.ctx("答案", &["kb.md#0", "kb.md#1"]);

        let score = NdcgAtK::new(5).score(&ctx).await.unwrap();
        assert!((score.value - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn mrr_rewards_first_hit() {
        let f = Fixture::new();
        let first = f.ctx("答案", &["kb.md#0", "kb.md#1"]);
        let second = f.ctx("答案", &["kb.md#9", "kb.md#0"]);

        assert_eq!(Mrr.score(&first).await.unwrap().value, 1.0);
        assert_eq!(Mrr.score(&second).await.unwrap().value, 0.5);
    }

    #[tokio::test]
    async fn mrr_is_zero_when_nothing_relevant() {
        let f = Fixture::new();
        let ctx = f.ctx("答案", &["kb.md#7", "kb.md#8"]);

        assert_eq!(Mrr.score(&ctx).await.unwrap().value, 0.0);
    }

    #[tokio::test]
    async fn context_precision_and_recall_by_id() {
        let f = Fixture::new();
        let ctx = f.ctx("答案", &["kb.md#0", "kb.md#9"]);

        let precision = ContextPrecisionById.score(&ctx).await.unwrap();
        let recall = ContextRecallById.score(&ctx).await.unwrap();

        assert!((precision.value - 0.5).abs() < 1e-6);
        assert!((recall.value - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn metrics_without_relevant_ids_report_zero_with_reason() {
        let f = Fixture {
            case: EvalCase::new("c1", "问题"),
            index: chunk_index(),
        };
        let ctx = f.ctx("答案", &["kb.md#0"]);

        let score = RecallAtK::new(5).score(&ctx).await.unwrap();
        assert_eq!(score.value, 0.0);
        assert!(score.reason.unwrap().contains("相关块"));
    }
}
