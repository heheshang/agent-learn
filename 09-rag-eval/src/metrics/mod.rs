//! 指标集合 —— 评估的「量表」。
//!
//! ## 指标的三个来源
//!
//! 本项目的指标不是凭空发明的，而是从三个成熟框架里提炼的：
//!
//! | 框架 | 它贡献的指标 | 特点 |
//! |---|---|---|
//! | [RAGAs](https://docs.ragas.io/en/latest/index.html) | faithfulness / answer_relevancy / context_relevancy / answer_correctness | LLM-as-a-judge，语义层面 |
//! | [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | recall@k / precision@k / nDCG / EM / F1 | 检索与答案的量化对比，benchmark 导向 |
//! | [DeepEval](https://github.com/confident-ai/deepeval) | RAGAS 聚合指标、context precision/recall 口径 | 与 RAGAs 同源的另一套实现，可交叉验证 |
//!
//! ## 为什么指标要用 trait 而不是枚举
//!
//! 因为指标的**依赖不同**：
//!
//! - 确定性指标（token F1、recall@k）什么都不需要
//! - LLM judge 指标需要一个 `Llm`
//! - 语义相关度指标还需要一个 `Embedder`
//!
//! 用 trait，依赖在**构造时**注入 —— 指标自己不管依赖从哪来。
//!
//! ## 指标集的三层结构
//!
//! ```text
//! 检索侧（确定性）    答案侧（确定性）      答案侧（judge）
//! ─────────────    ──────────────      ──────────────
//! recall_at_k        token_f1             faithfulness
//! precision_at_k     exact_match          answer_relevancy
//! ndcg_at_k          context_precision_   context_relevancy
//! mrr                by_id                answer_correctness
//!                    context_recall_by_id
//! ```

pub mod deterministic;
pub mod judge;
pub mod types;

use std::sync::Arc;

use rag_05::embed::Embedder;
use rag_07::llm::Llm;

pub use deterministic::{
    ContextPrecisionById, ContextRecallById, ExactMatch, Mrr, NdcgAtK, PrecisionAtK, RecallAtK,
    TokenF1,
};
pub use judge::{AnswerCorrectness, AnswerRelevancy, ContextRelevancy, Faithfulness};
pub use types::{EvalContext, Metric, MetricScore};

/// 指标注册表 —— 按名字索引指标。
///
/// ## 为什么需要「按名字找指标」
///
/// 因为 API 层要支持这样的请求：
///
/// ```json
/// { "question": "…", "metrics": ["faithfulness", "recall_at_k"] }
/// ```
///
/// 用户传的是**字符串**，不是 trait 对象。所以要有一张名字 → 指标的表。
///
/// ## ⚠️ 但 LLM judge 指标依赖运行时注入的 LLM
///
/// 所以注册表是**在运行时构造**的 —— 拿到 `Llm` 和 `Embedder` 之后
/// 才能把 judge 指标塞进去。
///
/// 这就是为什么 [`MetricRegistry::new`] 需要那么多参数。
pub struct MetricRegistry {
    metrics: Vec<Arc<dyn Metric>>,
}

impl std::fmt::Debug for MetricRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricRegistry")
            .field(
                "metrics",
                &self.metrics.iter().map(|m| m.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl MetricRegistry {
    /// 构造完整指标集。
    ///
    /// `llm` 为 `None` 时**不注册 judge 指标** —— 这样在没配 API key 的环境里
    /// 依然能跑确定性指标的评估（而不是启动就报错）。
    ///
    /// `embedder` 为 `None` 时同样不注册需要嵌入的指标。
    pub fn new(llm: Option<Arc<dyn Llm>>, embedder: Option<Arc<dyn Embedder>>) -> Self {
        let mut metrics: Vec<Arc<dyn Metric>> = vec![
            Arc::new(TokenF1::new()),
            Arc::new(ExactMatch),
            Arc::new(RecallAtK::new(5)),
            Arc::new(PrecisionAtK::new(5)),
            Arc::new(NdcgAtK::new(5)),
            Arc::new(Mrr),
            Arc::new(ContextPrecisionById),
            Arc::new(ContextRecallById),
        ];

        if let Some(ref llm) = llm {
            metrics.push(Arc::new(Faithfulness::new(Arc::clone(llm))));
            metrics.push(Arc::new(AnswerCorrectness::new(Arc::clone(llm))));
        }

        if let (Some(llm), Some(embedder)) = (llm, embedder) {
            metrics.push(Arc::new(AnswerRelevancy::new(
                Arc::clone(&llm),
                Arc::clone(&embedder),
            )));
            metrics.push(Arc::new(ContextRelevancy::new(
                Arc::clone(&llm),
                Arc::clone(&embedder),
            )));
        }

        Self { metrics }
    }

    /// 只注册确定性指标（不调 LLM，不需要 API key）。
    pub fn deterministic_only() -> Self {
        Self::new(None, None)
    }

    /// 注册表的全部指标。
    pub fn metrics(&self) -> &[Arc<dyn Metric>] {
        &self.metrics
    }

    /// 所有指标的引用（按注册顺序）。
    pub fn all_refs(&self) -> Vec<&dyn Metric> {
        self.metrics.iter().map(|m| m.as_ref()).collect()
    }

    /// 按名字找指标。
    pub fn get(&self, name: &str) -> Option<&dyn Metric> {
        self.metrics
            .iter()
            .find(|m| m.name() == name)
            .map(|m| m.as_ref())
    }

    /// 所有指标名。
    pub fn names(&self) -> Vec<&str> {
        self.metrics.iter().map(|m| m.name()).collect()
    }

    /// 按名字列表筛选指标（顺序按请求顺序）。
    ///
    /// 找不到的名字 → 返回错误列表，不静默跳过。
    ///
    /// ⚠️ 静默跳过是很常见的坑：用户请求了 `faithfulness`，
    /// 但因为没配 key 它没注册 —— 如果静默跳过，报告里就没这一列，
    /// 用户会以为「这个指标是 0 分」，而不是「这个指标没跑」。
    pub fn select(&self, names: &[String]) -> Result<Vec<Arc<dyn Metric>>, Vec<String>> {
        let mut selected = Vec::with_capacity(names.len());
        let mut missing = Vec::new();

        for name in names {
            match self.metrics.iter().find(|m| m.name() == name) {
                Some(metric) => selected.push(Arc::clone(metric)),
                None => missing.push(name.clone()),
            }
        }

        if missing.is_empty() {
            Ok(selected)
        } else {
            Err(missing)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_registry_has_no_judge_metrics() {
        let registry = MetricRegistry::deterministic_only();
        let names = registry.names();

        assert!(names.contains(&"token_f1"));
        assert!(names.contains(&"recall_at_k"));
        assert!(!names.contains(&"faithfulness"));
    }

    #[test]
    fn registry_lookup_by_name() {
        let registry = MetricRegistry::deterministic_only();
        assert_eq!(registry.get("token_f1").unwrap().name(), "token_f1");
        assert!(registry.get("不存在的指标").is_none());
    }

    #[test]
    fn select_returns_missing_names() {
        let registry = MetricRegistry::deterministic_only();

        let ok = registry
            .select(&["token_f1".to_string(), "mrr".to_string()])
            .unwrap();
        assert_eq!(ok.len(), 2);

        let missing = registry.select(&["token_f1".to_string(), "不存在".to_string()]);
        assert!(missing.is_err());
        // `unwrap_err` 要求 Ok 类型实现 Debug —— `Arc<dyn Metric>` 不实现，
        // 所以这里手动取 Err 分支。
        match missing {
            Err(names) => assert_eq!(names, vec!["不存在".to_string()]),
            Ok(_) => panic!("应该报缺失指标"),
        }
    }
}
