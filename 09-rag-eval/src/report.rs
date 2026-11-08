//! 评估报告 —— 评估的最终产出。
//!
//! ## 报告要回答三个问题
//!
//! 1. **系统现在有多好？** → 聚合分数（每个指标的平均分 + 通过率）
//! 2. **哪里不好？** → 逐样本明细 + 每个低分样本的**判定理由**
//! 3. **优化有没有用？** → 对比报告（baseline vs optimized 的差值）
//!
//! 第 3 个才是本项目的核心目的 —— **收益不靠嘴说，靠差值证明**。
//!
//! ## 为什么一定要给「理由」
//!
//! 一个孤零零的 `faithfulness = 0.62` 没法指导任何行动。
//! 而 `faithfulness = 0.62（第 3 句「XXX」无依据）` 直接告诉你该修什么。
//!
//! 所以 [`MetricScore::reason`] 不是可选的装饰 —— 它是评估结论的**载体**。

use agent_logging::{info, warn};
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::metrics::types::{EvalContext, Metric, MetricScore};

/// 单个样本的评估结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    /// 样本 ID。
    pub id: String,

    /// 问题。
    pub question: String,

    /// 系统生成的答案。
    pub answer: String,

    /// 检索到的块 ID（按相关度降序）。
    pub retrieved_ids: Vec<String>,

    /// 每个指标的得分（指标名 → 得分）。
    ///
    /// ⚠️ 失败的指标**不在这里** —— 它们被记到 [`CaseResult::errors`]。
    /// 「指标算出来是 0 分」和「指标没算出来」是两回事。
    pub scores: BTreeMap<String, MetricScore>,

    /// 计算失败的指标（指标名 → 错误信息）。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub errors: BTreeMap<String, String>,
}

impl CaseResult {
    /// 某个指标的得分。
    pub fn score(&self, metric_name: &str) -> Option<&MetricScore> {
        self.scores.get(metric_name)
    }

    /// 计算成功的指标数。
    pub fn evaluated_count(&self) -> usize {
        self.scores.len()
    }
}

/// 某个指标的聚合结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    /// 指标名。
    pub name: String,

    /// 指标说明。
    pub description: String,

    /// 平均分。
    pub average: f32,

    /// 最低分。
    pub min: f32,

    /// 最高分。
    pub max: f32,

    /// 通过率（分数达到阈值的样本比例）。
    pub pass_rate: f32,

    /// 通过阈值。
    pub threshold: f32,

    /// 参与计算的样本数。
    pub evaluated: usize,

    /// 计算失败的样本数。
    pub failed: usize,

    /// 最低分的样本 ID（定位问题用）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_case_id: Option<String>,
}

/// 一份评估报告。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// 报告标题（通常是「系统名 + 模式」）。
    pub title: String,

    /// 被评估的模式名（`vector` / `bm25` / `hybrid` 等）。
    pub variant: String,

    /// 评估集名称。
    pub dataset: String,

    /// 样本数。
    pub case_count: usize,

    /// 每个指标的聚合结果。
    pub summary: Vec<MetricSummary>,

    /// 逐样本明细。
    pub cases: Vec<CaseResult>,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

impl EvalReport {
    /// 找某个指标的聚合结果。
    pub fn metric(&self, name: &str) -> Option<&MetricSummary> {
        self.summary.iter().find(|s| s.name == name)
    }

    /// 某个指标的平均分。
    pub fn average(&self, name: &str) -> Option<f32> {
        self.metric(name).map(|s| s.average)
    }

    /// 渲染成对齐的文本表格。
    ///
    /// ## 为什么要有文本输出
    ///
    /// 因为报告要能**贴在 PR 里**、**贴进 issue**、**贴进聊天窗口**。
    /// JSON 适合机器读，表格适合人读 —— 两种都要。
    pub fn to_table(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("=== {} ===\n", self.title));
        out.push_str(&format!(
            "模式: {} | 评估集: {} | 样本数: {} | 耗时: {}ms\n\n",
            self.variant, self.dataset, self.case_count, self.elapsed_ms
        ));

        // 指标汇总表。
        let name_width = self
            .summary
            .iter()
            .map(|s| s.name.len())
            .max()
            .unwrap_or(10)
            .max(10);

        out.push_str(&format!(
            "{:<name_width$}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}\n",
            "指标",
            "平均分",
            "最低",
            "最高",
            "通过率",
            "样本",
            name_width = name_width
        ));
        out.push_str(&"-".repeat(name_width + 48));
        out.push('\n');

        for summary in &self.summary {
            out.push_str(&format!(
                "{:<name_width$}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.2}  {:>6}\n",
                summary.name,
                summary.average,
                summary.min,
                summary.max,
                summary.pass_rate,
                summary.evaluated,
                name_width = name_width
            ));
        }

        // 最差的几个样本 —— 报告的价值在这里。
        let mut worst: Vec<&CaseResult> = self.cases.iter().collect();
        worst.sort_by(|a, b| {
            let a_avg = average_of(a);
            let b_avg = average_of(b);
            a_avg
                .partial_cmp(&b_avg)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(worst_case) = worst.first()
            && average_of(worst_case) < 0.8
        {
            out.push_str(&format!(
                "\n最差样本（平均分 {:.4}）:\n  ID: {}\n  问: {}\n  答: {}\n",
                average_of(worst_case),
                worst_case.id,
                worst_case.question,
                worst_case.answer.chars().take(100).collect::<String>()
            ));

            for (name, score) in &worst_case.scores {
                if score.value < 0.8 {
                    out.push_str(&format!(
                        "    [{}] {:.4}{}\n",
                        name,
                        score.value,
                        score
                            .reason
                            .as_ref()
                            .map(|r| format!(" —— {r}"))
                            .unwrap_or_default()
                    ));
                }
            }
        }

        out
    }
}

/// 一个样本的平均分（所有成功指标的算术平均）。
fn average_of(case: &CaseResult) -> f32 {
    if case.scores.is_empty() {
        return 0.0;
    }
    let sum: f32 = case.scores.values().map(|s| s.value).sum();
    sum / case.scores.len() as f32
}

/// 评估执行器 —— 把「样本 + 指标」跑成「报告」。
///
/// ## 为什么是一个独立的结构
///
/// 因为「跑评估」这件事有三件容易出错的事：
///
/// 1. **失败要能降级** —— 某个指标的 judge 挂了，不该让整次评估失败
/// 2. **失败和 0 分要分开** —— 前者是「没测出来」，后者是「测出来是 0」
/// 3. **要能部分成功** —— 10 个样本跑了 9 个，也要出报告
///
/// 这三点集中在一个地方处理，而不是散落在调用方。
pub struct Evaluator {
    metrics: Vec<std::sync::Arc<dyn Metric>>,
}

impl std::fmt::Debug for Evaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Evaluator")
            .field(
                "metrics",
                &self.metrics.iter().map(|m| m.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Evaluator {
    /// 创建执行器。
    pub fn new(metrics: Vec<std::sync::Arc<dyn Metric>>) -> Self {
        Self { metrics }
    }

    /// 使用的指标。
    pub fn metrics(&self) -> &[std::sync::Arc<dyn Metric>] {
        &self.metrics
    }

    /// 评估一个样本。
    ///
    /// **单个指标的失败不会中断整个评估** —— 它被记到 `errors` 里。
    pub async fn evaluate_case(&self, ctx: &EvalContext<'_>) -> CaseResult {
        let mut scores: BTreeMap<String, MetricScore> = BTreeMap::new();
        let mut errors: BTreeMap<String, String> = BTreeMap::new();

        for metric in &self.metrics {
            match metric.score(ctx).await {
                Ok(score) => {
                    scores.insert(metric.name().to_string(), score);
                }
                Err(e) => {
                    warn!(metric = %metric.name(), case_id = %ctx.case.id, error = %e, "指标计算失败");
                    errors.insert(metric.name().to_string(), e.to_string());
                }
            }
        }

        CaseResult {
            id: ctx.case.id.clone(),
            question: ctx.case.question.clone(),
            answer: ctx.answer.clone(),
            retrieved_ids: ctx.retrieved_ids.clone(),
            scores,
            errors,
        }
    }

    /// 评估整个评估集。
    ///
    /// `title` / `variant` 用来标记「这份报告是谁的」——
    /// 对比报告里需要靠它区分 baseline 和 optimized。
    pub async fn evaluate(
        &self,
        dataset: &crate::dataset::EvalDataset,
        contexts: &[EvalContext<'_>],
        title: &str,
        variant: &str,
        elapsed_ms: u64,
    ) -> EvalReport {
        let mut cases: Vec<CaseResult> = Vec::with_capacity(contexts.len());
        for ctx in contexts {
            cases.push(self.evaluate_case(ctx).await);
        }

        let summary = self.summarize(&cases);

        info!(
            title = %title,
            variant = %variant,
            case_count = cases.len(),
            elapsed_ms,
            "评估报告生成完成"
        );

        EvalReport {
            title: title.to_string(),
            variant: variant.to_string(),
            dataset: dataset.name.clone(),
            case_count: cases.len(),
            summary,
            cases,
            elapsed_ms,
        }
    }

    /// 聚合每个指标的统计量。
    fn summarize(&self, cases: &[CaseResult]) -> Vec<MetricSummary> {
        self.metrics
            .iter()
            .map(|metric| {
                let values: Vec<f32> = cases
                    .iter()
                    .filter_map(|c| c.scores.get(metric.name()).map(|s| s.value))
                    .collect();

                let failed = cases.len() - values.len();

                if values.is_empty() {
                    return MetricSummary {
                        name: metric.name().to_string(),
                        description: metric.description().to_string(),
                        average: 0.0,
                        min: 0.0,
                        max: 0.0,
                        pass_rate: 0.0,
                        threshold: metric.threshold(),
                        evaluated: 0,
                        failed,
                        worst_case_id: None,
                    };
                }

                let sum: f32 = values.iter().sum();
                let average = sum / values.len() as f32;
                let min = values.iter().copied().fold(f32::MAX, f32::min);
                let max = values.iter().copied().fold(f32::MIN, f32::max);

                let passed = values.iter().filter(|v| **v >= metric.threshold()).count();

                // 找最低分的样本。
                let worst_case_id = cases
                    .iter()
                    .filter_map(|c| c.scores.get(metric.name()).map(|s| (s.value, c.id.clone())))
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(_, id)| id);

                MetricSummary {
                    name: metric.name().to_string(),
                    description: metric.description().to_string(),
                    average,
                    min,
                    max,
                    pass_rate: passed as f32 / values.len() as f32,
                    threshold: metric.threshold(),
                    evaluated: values.len(),
                    failed,
                    worst_case_id,
                }
            })
            .collect()
    }
}

/// 对比报告 —— 两个版本的差值。
///
/// ## 为什么这是本项目的核心产出
///
/// 单个报告只能说「现在是这样」。对比报告才能说「**改成这样之后好了多少**」。
///
/// 而「优化后涨了 0.05」这种结论必须有一个前提：**这个差值大于噪声**。
///
/// 所以对比报告里除了差值，还有：
/// - `improved` / `regressed` —— 哪些指标变好了、哪些变差了
/// - 逐样本的变化 —— 哪些样本的答案被改好了
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// 基线报告标题。
    pub baseline_title: String,

    /// 优化版报告标题。
    pub optimized_title: String,

    /// 每个指标的差值。
    pub deltas: Vec<MetricDelta>,
}

/// 单个指标的差值。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDelta {
    /// 指标名。
    pub name: String,

    /// 基线平均分。
    pub baseline: f32,

    /// 优化版平均分。
    pub optimized: f32,

    /// 差值（优化版 - 基线）。
    pub delta: f32,

    /// 变好了 / 变差了 / 没变。
    pub verdict: Verdict,
}

/// 变化结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// 明显变好（差值 ≥ 0.05）。
    Improved,

    /// 明显变差（差值 ≤ -0.05）。
    Regressed,

    /// 基本没变（差值在 ±0.05 之间）。
    ///
    /// ⚠️ 这个「噪声带」是**刻意**的 —— LLM judge 本身有抖动，
    /// 小于 5% 的差值**不能**当成真实收益。
    Unchanged,
}

impl Verdict {
    /// 根据差值判断结论。
    pub fn from_delta(delta: f32) -> Self {
        if delta >= 0.05 {
            Verdict::Improved
        } else if delta <= -0.05 {
            Verdict::Regressed
        } else {
            Verdict::Unchanged
        }
    }
}

impl ComparisonReport {
    /// 对比两份报告。
    pub fn compare(baseline: &EvalReport, optimized: &EvalReport) -> Self {
        let mut deltas = Vec::new();

        for base in &baseline.summary {
            let opt = optimized.metric(&base.name);

            let (optimized_avg, delta) = match opt {
                Some(o) => (o.average, o.average - base.average),
                None => (0.0, -base.average),
            };

            deltas.push(MetricDelta {
                name: base.name.clone(),
                baseline: base.average,
                optimized: optimized_avg,
                delta,
                verdict: Verdict::from_delta(delta),
            });
        }

        Self {
            baseline_title: baseline.title.clone(),
            optimized_title: optimized.title.clone(),
            deltas,
        }
    }

    /// 渲染成文本表格。
    pub fn to_table(&self) -> String {
        let mut out = String::new();
        out.push_str("=== 对比报告 ===\n");
        out.push_str(&format!(
            "基线: {}\n优化: {}\n\n",
            self.baseline_title, self.optimized_title
        ));

        let name_width = self
            .deltas
            .iter()
            .map(|d| d.name.len())
            .max()
            .unwrap_or(10)
            .max(10);

        out.push_str(&format!(
            "{:<name_width$}  {:>9}  {:>9}  {:>9}  {}\n",
            "指标",
            "基线",
            "优化后",
            "差值",
            "结论",
            name_width = name_width
        ));
        out.push_str(&"-".repeat(name_width + 46));
        out.push('\n');

        for delta in &self.deltas {
            let verdict = match delta.verdict {
                Verdict::Improved => "↑ 变好",
                Verdict::Regressed => "↓ 变差",
                Verdict::Unchanged => "→ 持平",
            };

            out.push_str(&format!(
                "{:<name_width$}  {:>9.4}  {:>9.4}  {:>+9.4}  {}\n",
                delta.name,
                delta.baseline,
                delta.optimized,
                delta.delta,
                verdict,
                name_width = name_width
            ));
        }

        out
    }

    /// 明显变好的指标。
    pub fn improved(&self) -> Vec<&MetricDelta> {
        self.deltas
            .iter()
            .filter(|d| d.verdict == Verdict::Improved)
            .collect()
    }

    /// 明显变差的指标。
    pub fn regressed(&self) -> Vec<&MetricDelta> {
        self.deltas
            .iter()
            .filter(|d| d.verdict == Verdict::Regressed)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::EvalCase;

    fn score(value: f32) -> MetricScore {
        MetricScore::with_reason(value, "理由")
    }

    fn case_result(id: &str, values: &[(&str, f32)]) -> CaseResult {
        let mut scores = BTreeMap::new();
        for (name, value) in values {
            scores.insert(name.to_string(), score(*value));
        }
        CaseResult {
            id: id.to_string(),
            question: format!("问题 {id}"),
            answer: "答案".to_string(),
            retrieved_ids: Vec::new(),
            scores,
            errors: BTreeMap::new(),
        }
    }

    struct StubMetric {
        name: &'static str,
        threshold: f32,
        value: f32,
    }

    impl Metric for StubMetric {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            "测试用指标"
        }
        fn score<'a>(
            &'a self,
            _ctx: &'a EvalContext<'a>,
        ) -> crate::metrics::types::MetricFuture<'a> {
            Box::pin(async move { Ok(score(self.value)) })
        }
        fn threshold(&self) -> f32 {
            self.threshold
        }
    }

    fn dataset() -> crate::dataset::EvalDataset {
        crate::dataset::EvalDataset::new("test", vec![EvalCase::new("c1", "问题")])
    }

    #[tokio::test]
    async fn evaluator_runs_all_metrics() {
        let evaluator = Evaluator::new(vec![
            std::sync::Arc::new(StubMetric {
                name: "m1",
                threshold: 0.5,
                value: 0.8,
            }),
            std::sync::Arc::new(StubMetric {
                name: "m2",
                threshold: 0.5,
                value: 0.3,
            }),
        ]);

        let case = EvalCase::new("c1", "问题");
        let index = crate::dataset::ChunkIndex::build(&[]);
        let ctx = EvalContext::new(&case, "答案", Vec::new(), Vec::new(), &index);

        let result = evaluator.evaluate_case(&ctx).await;

        assert_eq!(result.evaluated_count(), 2);
        assert_eq!(result.score("m1").unwrap().value, 0.8);
        assert_eq!(result.score("m2").unwrap().value, 0.3);
    }

    #[tokio::test]
    async fn evaluator_records_failures_separately() {
        struct Failing;
        impl Metric for Failing {
            fn name(&self) -> &str {
                "failing"
            }
            fn description(&self) -> &str {
                "总是失败"
            }
            fn score<'a>(
                &'a self,
                _ctx: &'a EvalContext<'a>,
            ) -> crate::metrics::types::MetricFuture<'a> {
                Box::pin(async move {
                    Err(crate::error::EvalError::Judging("judge 挂了".to_string()))
                })
            }
        }

        let evaluator = Evaluator::new(vec![std::sync::Arc::new(Failing)]);
        let case = EvalCase::new("c1", "问题");
        let index = crate::dataset::ChunkIndex::build(&[]);
        let ctx = EvalContext::new(&case, "答案", Vec::new(), Vec::new(), &index);

        let result = evaluator.evaluate_case(&ctx).await;

        assert_eq!(result.evaluated_count(), 0);
        assert!(result.errors.contains_key("failing"));
    }

    #[tokio::test]
    async fn summary_computes_average_and_pass_rate() {
        let evaluator = Evaluator::new(vec![std::sync::Arc::new(StubMetric {
            name: "m1",
            threshold: 0.5,
            value: 0.8,
        })]);

        let case = EvalCase::new("c1", "问题");
        let index = crate::dataset::ChunkIndex::build(&[]);
        let ctx = EvalContext::new(&case, "答案", Vec::new(), Vec::new(), &index);

        let report = evaluator
            .evaluate(&dataset(), &[ctx], "测试", "baseline", 10)
            .await;

        let summary = report.metric("m1").unwrap();
        assert_eq!(summary.average, 0.8);
        assert_eq!(summary.pass_rate, 1.0);
        assert_eq!(summary.evaluated, 1);
    }

    #[test]
    fn comparison_detects_improvement_and_regression() {
        let baseline = EvalReport {
            title: "基线".to_string(),
            variant: "baseline".to_string(),
            dataset: "test".to_string(),
            case_count: 1,
            summary: vec![
                MetricSummary {
                    name: "m1".to_string(),
                    description: "d".to_string(),
                    average: 0.5,
                    min: 0.5,
                    max: 0.5,
                    pass_rate: 0.5,
                    threshold: 0.7,
                    evaluated: 1,
                    failed: 0,
                    worst_case_id: None,
                },
                MetricSummary {
                    name: "m2".to_string(),
                    description: "d".to_string(),
                    average: 0.8,
                    min: 0.8,
                    max: 0.8,
                    pass_rate: 1.0,
                    threshold: 0.7,
                    evaluated: 1,
                    failed: 0,
                    worst_case_id: None,
                },
            ],
            cases: Vec::new(),
            elapsed_ms: 0,
        };

        let mut optimized = baseline.clone();
        optimized.title = "优化".to_string();
        optimized.summary[0].average = 0.7;
        optimized.summary[1].average = 0.6;

        let comparison = ComparisonReport::compare(&baseline, &optimized);

        assert_eq!(comparison.improved().len(), 1);
        assert_eq!(comparison.improved()[0].name, "m1");
        assert_eq!(comparison.regressed().len(), 1);
        assert_eq!(comparison.regressed()[0].name, "m2");
    }

    #[test]
    fn small_deltas_count_as_unchanged() {
        assert_eq!(Verdict::from_delta(0.01), Verdict::Unchanged);
        assert_eq!(Verdict::from_delta(0.05), Verdict::Improved);
        assert_eq!(Verdict::from_delta(-0.05), Verdict::Regressed);
    }

    #[test]
    fn report_table_mentions_metric_names() {
        let report = EvalReport {
            title: "测试".to_string(),
            variant: "baseline".to_string(),
            dataset: "test".to_string(),
            case_count: 1,
            summary: vec![MetricSummary {
                name: "token_f1".to_string(),
                description: "d".to_string(),
                average: 0.5,
                min: 0.5,
                max: 0.5,
                pass_rate: 0.5,
                threshold: 0.7,
                evaluated: 1,
                failed: 0,
                worst_case_id: None,
            }],
            cases: vec![case_result("c1", &[("token_f1", 0.5)])],
            elapsed_ms: 0,
        };

        let table = report.to_table();
        assert!(table.contains("token_f1"));
        assert!(table.contains("测试"));
    }
}
