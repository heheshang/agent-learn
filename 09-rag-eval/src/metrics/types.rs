//! 指标的类型定义 —— 所有指标的公共契约。
//!
//! ## 三个核心类型
//!
//! | 类型 | 作用 |
//! |---|---|
//! | [`EvalCase`] | 一条评估样本（问题 + 标准答案 + 标准上下文 + 相关块 ID） |
//! | [`EvalContext`] | 一次评估的输入（评估样本 + 系统的实际输出） |
//! | [`Metric`] | 指标 trait —— 「给一次评估的输入，返回一个分数」 |
//!
//! ## 为什么 `EvalContext` 和 `EvalCase` 要分开
//!
//! 因为**评估集是静态的，系统输出是动态的**。
//!
//! ```text
//! EvalCase   （数据集里写的）        EvalContext  （跑完系统后拼出来的）
//! ├─ question                       ├─ case:  &EvalCase
//! ├─ reference_answer               ├─ answer: 系统生成的答案
//! ├─ reference_contexts             ├─ contexts: 系统检索到的上下文
//! └─ relevant_chunk_ids             └─ retrieved_ids: 检索到的块 ID
//! ```
//!
//! 一个 `EvalCase` 会被跑出**多个** `EvalContext`（优化前一个、优化后一个），
//! 这正是「优化前后对比」的实现基础 —— **评估集不变，只有输出变**。

use serde::{Deserialize, Serialize};

use crate::dataset::{ChunkIndex, EvalCase};
use crate::error::EvalResult;

/// 单个指标的得分。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricScore {
    /// 指标值。
    ///
    /// ⚠️ 绝大多数指标在 `[0, 1]` 区间 —— 因为它们都要能聚合成一个总分。
    ///
    /// 例外：`ndcg` 和 `mrr` 也是 `[0, 1]`，但它们是**排名质量**指标，
    /// 和「比例类」指标（precision / recall / faithfulness）语义不同，
    /// 不应该混在一个加权平均里。
    pub value: f32,

    /// 判定理由。
    ///
    /// LLM-as-a-judge 指标必须给出理由 —— 否则分数不可解释，
    /// 你只知道「faithfulness = 0.6」，不知道**哪句话是幻觉**。
    ///
    /// 这个字段是本项目「可解释评估」的核心：报告里每条低分都附带原因。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl MetricScore {
    /// 只给数值。
    pub fn new(value: f32) -> Self {
        Self {
            value,
            reason: None,
        }
    }

    /// 数值 + 理由。
    pub fn with_reason(value: f32, reason: impl Into<String>) -> Self {
        Self {
            value,
            reason: Some(reason.into()),
        }
    }

    /// 是否达到阈值。
    pub fn passed(&self, threshold: f32) -> bool {
        self.value >= threshold
    }
}

/// 一次评估的输入。
///
/// 包含「标准答案」和「系统实际输出」两侧 —— 指标据此计算差异。
#[derive(Debug, Clone)]
pub struct EvalContext<'a> {
    /// 评估样本（标准答案侧）。
    pub case: &'a EvalCase,

    /// 系统生成的答案。
    pub answer: String,

    /// 系统检索到的上下文（按相关度降序）。
    ///
    /// 每个元素是一段文本 —— 对应 RAGAs 里 `retrieved_contexts` 的位置。
    pub contexts: Vec<String>,

    /// 检索到的块 ID（顺序同 `contexts`）。
    ///
    /// 检索指标（recall@k / nDCG / MRR）用它和 `case.relevant_chunk_ids` 比对。
    pub retrieved_ids: Vec<String>,

    /// 块索引 —— 「章节标题 → 块 ID」的映射。
    ///
    /// 用来把评估集里标注的**章节标题**翻译成**块 ID**。
    /// 见 [`EvalCase::resolve_relevant_ids`]。
    pub chunk_index: &'a ChunkIndex,
}

impl<'a> EvalContext<'a> {
    /// 构造评估上下文。
    pub fn new(
        case: &'a EvalCase,
        answer: impl Into<String>,
        contexts: Vec<String>,
        retrieved_ids: Vec<String>,
        chunk_index: &'a ChunkIndex,
    ) -> Self {
        Self {
            case,
            answer: answer.into(),
            contexts,
            retrieved_ids,
            chunk_index,
        }
    }

    /// 检索到的上下文拼成一段文本（judge 用）。
    pub fn context_text(&self) -> String {
        self.contexts
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// 指标调用的异步 future。
///
/// ## 为什么是 `Pin<Box<dyn Future>>` 而不是 `async fn`
///
/// 因为本模块需要 `Box<dyn Metric>` —— **trait 必须 dyn 兼容**。
///
/// 而 Rust 的两种「返回 future」的写法都**不 dyn 兼容**：
///
/// - `async fn` → 编译期展开成 `-> impl Future`，不是 dyn 兼容
/// - `fn f() -> impl Future` → RPITIT 同样不是 dyn 兼容
///
/// 所以只能显式装箱。
///
/// 这和 07 的 `TransformFuture`、08 的 `Reranker::rerank` 是同一个套路。
pub type MetricFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = EvalResult<MetricScore>> + Send + 'a>>;

/// 指标 trait。
///
/// ## 为什么是 trait 而不是枚举
///
/// 因为指标的**依赖不同**：
///
/// - 确定性指标（token F1、recall@k）什么都不需要
/// - LLM judge 指标需要一个 `Llm`
/// - 语义相关度指标需要一个 `Embedder`
///
/// 用枚举的话，`match` 分支里到处要判空。用 trait 的话，
/// 依赖在**构造时**就注入好了 —— 指标自己不管依赖从哪来。
///
/// 和 07 的 `QueryTransformer` / 08 的 `Reranker` 是同一个套路：
/// **抽象成 trait，是为了能塞进 `Box<dyn ...>` 并在运行时替换**。
pub trait Metric: Send + Sync {
    /// 指标名（用于报告表格的列名）。
    fn name(&self) -> &str;

    /// 指标说明（用于报告表头下方的解释）。
    fn description(&self) -> &str;

    /// 计算指标。
    ///
    /// 返回 `Err` 表示**这次评估没做出来**（judge 调用失败、解析失败）。
    /// 返回 `Ok` 但分数很低表示**评估做出来了，而且分数低** ——
    /// 两者语义完全不同，报告里必须区分。
    fn score<'a>(&'a self, ctx: &'a EvalContext<'a>) -> MetricFuture<'a>;

    /// 指标的默认通过阈值。
    ///
    /// 用在报告的「通过率」列 —— 每个指标的阈值不一样
    /// （faithfulness 要求 0.9，context_precision 0.7 …）。
    fn threshold(&self) -> f32 {
        0.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case() -> EvalCase {
        EvalCase {
            id: "c1".to_string(),
            question: "问题".to_string(),
            reference_answer: "标准答案".to_string(),
            reference_contexts: vec!["标准上下文".to_string()],
            relevant_chunk_ids: vec!["kb.md#0".to_string()],
            relevant_sections: Vec::new(),
        }
    }

    #[test]
    fn metric_score_defaults_pass_threshold() {
        let score = MetricScore::new(0.8);
        assert!(score.passed(0.7));
        assert!(!score.passed(0.9));
        assert!(score.reason.is_none());
    }

    #[test]
    fn metric_score_keeps_reason() {
        let score = MetricScore::with_reason(0.5, "因为……");
        assert_eq!(score.reason.as_deref(), Some("因为……"));
    }

    #[test]
    fn context_text_numbers_each_context() {
        let c = case();
        let index = ChunkIndex::build(&[("kb.md#0".to_string(), vec!["标题".to_string()])]);
        let ctx = EvalContext::new(
            &c,
            "答案",
            vec!["第一段".to_string(), "第二段".to_string()],
            vec!["a".to_string(), "b".to_string()],
            &index,
        );

        let text = ctx.context_text();
        assert!(text.contains("[1] 第一段"));
        assert!(text.contains("[2] 第二段"));
    }

    #[test]
    fn empty_contexts_produce_empty_text() {
        let c = case();
        let index = ChunkIndex::build(&[]);
        let ctx = EvalContext::new(&c, "答案", Vec::new(), Vec::new(), &index);
        assert!(ctx.context_text().is_empty());
    }
}
