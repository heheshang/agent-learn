//! vLLM 服务的**度量** —— 对齐 `vllm bench serve` 的指标口径。
//!
//! # 为什么不能只看「总吞吐」
//!
//! 一份只报「总请求数 / 总耗时」的报告**没法用来做决策** ——
//! 因为「吞吐高」和「延迟可接受」是两个独立维度。
//!
//! vLLM 官方 `vllm bench serve` 报的指标分三类：
//!
//! | 类别 | 指标 | 含义 |
//! |---|---|---|
//! | **吞吐** | Request throughput | 每秒完成的请求数 |
//! | | Output token throughput | 每秒产出的 token 数（**更接近真实算力负载**） |
//! | **延迟** | TTFT (Time To First Token) | 到达 → 第一个 token |
//! | | TPOT (Time Per Output Token) | 后续每个 token 的平均间隔 |
//! | | E2EL (End-to-End Latency) | 到达 → 全部生成完 |
//! | **质量** | GOODPUT | 在 SLO 约束下「达标」的请求数/秒 |
//!
//! # ⚠️ GOODPUT 是最有用的那个指标
//!
//! 朴素吞吐会把「快但烂」的服务和「慢但稳」的服务混为一谈：
//!
//! ```text
//! 服务 A：吞吐 100 req/s，但 TTFT p99 = 30 秒（用户早走了）
//! 服务 B：吞吐  40 req/s，但 TTFT p99 = 200ms（用户满意）
//! ```
//!
//! GOODPUT 加上 SLO 约束（`ttft < 500ms 且 tpot < 50ms`）后，
//! B 的 goodput 远高于 A —— 这才是**该报的数字**。
//!
//! # ⚠️ 百分位用最近秩法，不用插值法
//!
//! 与 23 项目一致：小样本下插值会产生「样本里根本不存在的值」
//! （「p95 = 37.5ms」但从没测出 37.5ms）。
//! 最近秩法保证结果**一定是某个真实测到的值**。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::scheduler::Outcome;

/// 一组服务度量。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServeMetrics {
    /// 请求总数。
    pub total_requests: usize,

    /// 总耗时（毫秒）。
    pub wall_ms: f64,

    /// ★ 请求吞吐（请求/秒）。
    pub request_throughput_per_sec: f64,

    /// ★ 输出 token 吞吐（token/秒）。
    ///
    /// ⚠️ 这个指标比「请求/秒」更能反映**真实的算力负载** ——
    /// 一条请求生成 500 个 token 和生成 5 个 token，
    /// 对 GPU 的压力差两个数量级，但「请求/秒」完全看不出来。
    pub output_token_throughput_per_sec: f64,

    /// TTFT 的 p50（毫秒）。
    pub ttft_p50_ms: f64,

    /// TTFT 的 p95（毫秒）。
    pub ttft_p95_ms: f64,

    /// TTFT 的 p99（毫秒）。
    pub ttft_p99_ms: f64,

    /// TPOT 的 p50（毫秒）。
    pub tpot_p50_ms: f64,

    /// TPOT 的 p95（毫秒）。
    pub tpot_p95_ms: f64,

    /// 端到端时延的 p50（毫秒）。
    pub e2e_p50_ms: f64,

    /// 端到端时延的 p95（毫秒）。
    pub e2e_p95_ms: f64,

    /// 平均排队等待（毫秒）—— 到达 → 被接纳。
    pub avg_queue_wait_ms: f64,

    /// 平均输出 token 数。
    pub avg_output_tokens: f64,
}

impl ServeMetrics {
    /// 从一批请求结果算出度量。
    pub fn from_outcomes(outcomes: &[Outcome], wall_ms: f64) -> Self {
        if outcomes.is_empty() {
            return Self::empty();
        }

        let ttfts: Vec<f64> = outcomes.iter().map(|o| o.ttft_ms).collect();
        let tpots: Vec<f64> = outcomes.iter().map(|o| o.tpot_ms()).collect();
        let e2es: Vec<f64> = outcomes.iter().map(|o| o.e2e_ms).collect();
        let waits: Vec<f64> = outcomes.iter().map(|o| o.queue_wait_ms()).collect();

        let total_output_tokens: usize = outcomes.iter().map(|o| o.output_tokens).sum();
        let secs = wall_ms / 1000.0;

        Self {
            total_requests: outcomes.len(),
            wall_ms,
            request_throughput_per_sec: if secs > 0.0 {
                outcomes.len() as f64 / secs
            } else {
                0.0
            },
            output_token_throughput_per_sec: if secs > 0.0 {
                total_output_tokens as f64 / secs
            } else {
                0.0
            },
            ttft_p50_ms: percentile(&ttfts, 50),
            ttft_p95_ms: percentile(&ttfts, 95),
            ttft_p99_ms: percentile(&ttfts, 99),
            tpot_p50_ms: percentile(&tpots, 50),
            tpot_p95_ms: percentile(&tpots, 95),
            e2e_p50_ms: percentile(&e2es, 50),
            e2e_p95_ms: percentile(&e2es, 95),
            avg_queue_wait_ms: mean(&waits),
            avg_output_tokens: total_output_tokens as f64 / outcomes.len() as f64,
        }
    }

    /// 空结果。
    pub fn empty() -> Self {
        Self {
            total_requests: 0,
            wall_ms: 0.0,
            request_throughput_per_sec: 0.0,
            output_token_throughput_per_sec: 0.0,
            ttft_p50_ms: 0.0,
            ttft_p95_ms: 0.0,
            ttft_p99_ms: 0.0,
            tpot_p50_ms: 0.0,
            tpot_p95_ms: 0.0,
            e2e_p50_ms: 0.0,
            e2e_p95_ms: 0.0,
            avg_queue_wait_ms: 0.0,
            avg_output_tokens: 0.0,
        }
    }
}

/// **GOODPUT** 的 SLO 约束。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Slo {
    /// TTFT 上限（毫秒）。
    pub ttft_ms: f64,

    /// TPOT 上限（毫秒）。
    pub tpot_ms: f64,
}

impl Default for Slo {
    fn default() -> Self {
        Self {
            ttft_ms: 500.0,
            tpot_ms: 50.0,
        }
    }
}

impl Slo {
    /// 构造一份 SLO。
    pub fn new(ttft_ms: f64, tpot_ms: f64) -> Self {
        Self { ttft_ms, tpot_ms }
    }

    /// 判断一个结果是否达标。
    pub fn satisfied_by(&self, outcome: &Outcome) -> bool {
        outcome.ttft_ms <= self.ttft_ms && outcome.tpot_ms() <= self.tpot_ms
    }
}

/// GOODPUT 度量。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Goodput {
    /// 用的 SLO。
    pub slo: Slo,

    /// 达标的请求数。
    pub satisfied_requests: usize,

    /// 总请求数。
    pub total_requests: usize,

    /// ★ 达标请求吞吐（请求/秒）。
    pub goodput_per_sec: f64,

    /// ★ 达标率（0.0 ~ 1.0）。
    pub satisfaction_rate: f64,
}

impl Goodput {
    /// 从结果算出 goodput。
    pub fn from_outcomes(outcomes: &[Outcome], wall_ms: f64, slo: Slo) -> Self {
        if outcomes.is_empty() {
            return Self {
                slo,
                satisfied_requests: 0,
                total_requests: 0,
                goodput_per_sec: 0.0,
                satisfaction_rate: 0.0,
            };
        }

        let satisfied_requests = outcomes.iter().filter(|o| slo.satisfied_by(o)).count();
        let secs = wall_ms / 1000.0;

        Self {
            slo,
            satisfied_requests,
            total_requests: outcomes.len(),
            goodput_per_sec: if secs > 0.0 {
                satisfied_requests as f64 / secs
            } else {
                0.0
            },
            satisfaction_rate: satisfied_requests as f64 / outcomes.len() as f64,
        }
    }
}

/// **多档 SLO 的扫描** —— 找出「这条曲线」上的取舍点。
///
/// 对同一批结果，在多个 SLO 严格度下各算一次 goodput。
/// 画出来就是 vLLM 官方的「goodput vs SLO」权衡曲线。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SloSweep {
    /// 每档的结果。
    pub points: Vec<Goodput>,
}

impl SloSweep {
    /// 对一组 SLO 逐档计算。
    pub fn sweep(outcomes: &[Outcome], wall_ms: f64, slos: &[Slo]) -> Self {
        Self {
            points: slos
                .iter()
                .map(|s| Goodput::from_outcomes(outcomes, wall_ms, *s))
                .collect(),
        }
    }
}

/// 百分位（最近秩法）。
///
/// n = 0 → 0；n = 1 → 唯一那个值。
pub fn percentile(values: &[f64], p: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // 最近秩：第 ceil(p/100 × n) 个（1-based）。
    // ⚠️ 用整数运算避免浮点误差导致 off-by-one。
    let rank = (p * sorted.len()).div_ceil(100).max(1);
    sorted[(rank - 1).min(sorted.len() - 1)]
}

/// 均值。
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// 标准差（总体标准差）。
///
/// 用途：报告「加速比在几次运行间的波动」。
/// 本仓库的经验是**加速比本身会浮动**，
/// 可复现的结论是「相对关系」而不是绝对值。
pub fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let var = values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64;
    var.sqrt()
}

/// 一组**参数扫描**的结果（每个参数一档）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepPoint {
    /// 参数名。
    pub param: String,

    /// 参数值。
    pub value: String,

    /// 该档的度量。
    pub metrics: ServeMetrics,

    /// 该档的 goodput（用默认 SLO）。
    pub goodput: Goodput,
}

/// 把多档结果整理成「参数 → 度量」的表。
pub fn to_sweep_points(
    param: &str,
    entries: &[(String, ServeMetrics, Goodput)],
) -> Vec<SweepPoint> {
    entries
        .iter()
        .map(|(value, metrics, goodput)| SweepPoint {
            param: param.to_string(),
            value: value.clone(),
            metrics: metrics.clone(),
            goodput: goodput.clone(),
        })
        .collect()
}

/// 汇总一组结果的「相对基线倍数」—— 用于并排对比。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    /// 基线的名字。
    pub baseline_name: String,

    /// 对照的名字。
    pub variant_name: String,

    /// 请求吞吐倍数。
    pub request_throughput_ratio: f64,

    /// token 吞吐倍数。
    pub token_throughput_ratio: f64,

    /// TTFT p95 的倍数（**大于 1 表示变差**）。
    pub ttft_p95_ratio: f64,

    /// goodput 倍数。
    pub goodput_ratio: f64,
}

impl Comparison {
    /// 对比两组度量。
    pub fn new(
        baseline_name: &str,
        baseline: &ServeMetrics,
        baseline_goodput: &Goodput,
        variant_name: &str,
        variant: &ServeMetrics,
        variant_goodput: &Goodput,
    ) -> Self {
        Self {
            baseline_name: baseline_name.to_string(),
            variant_name: variant_name.to_string(),
            request_throughput_ratio: ratio(
                variant.request_throughput_per_sec,
                baseline.request_throughput_per_sec,
            ),
            token_throughput_ratio: ratio(
                variant.output_token_throughput_per_sec,
                baseline.output_token_throughput_per_sec,
            ),
            ttft_p95_ratio: ratio(variant.ttft_p95_ms, baseline.ttft_p95_ms),
            goodput_ratio: ratio(
                variant_goodput.goodput_per_sec,
                baseline_goodput.goodput_per_sec,
            ),
        }
    }
}

/// 两个值的比值（分母为 0 时返回 1.0，避免除零）。
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() < f64::EPSILON {
        return 1.0;
    }
    numerator / denominator
}

/// 把分组的结果按 key 聚合成 map（用于多次运行的合并）。
pub fn group_by_id(outcomes: &[Outcome]) -> HashMap<usize, &Outcome> {
    outcomes.iter().map(|o| (o.id, o)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{Arrival, BatchingMode, RequestSpec};

    fn outcome(
        id: usize,
        arrived: f64,
        admitted: f64,
        ttft: f64,
        tpot: f64,
        tokens: usize,
    ) -> Outcome {
        let mut token_times = vec![arrived + ttft];
        for i in 1..tokens {
            token_times.push(arrived + ttft + tpot * i as f64);
        }
        Outcome {
            id,
            arrived_ms: arrived,
            admitted_ms: admitted,
            ttft_ms: ttft,
            e2e_ms: ttft + tpot * (tokens.saturating_sub(1)) as f64,
            output_tokens: tokens,
            token_times_ms: token_times,
        }
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        assert!((percentile(&values, 50) - 50.0).abs() < 1e-9);
        assert!((percentile(&values, 95) - 100.0).abs() < 1e-9);
        assert!((percentile(&values, 100) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_of_empty_is_zero() {
        assert_eq!(percentile(&[], 50), 0.0);
    }

    #[test]
    fn percentile_never_exceeds_the_max_observed() {
        let values = vec![7.0, 13.0, 29.0];
        for p in [0, 25, 50, 75, 99, 100] {
            let v = percentile(&values, p);
            assert!((7.0..=29.0).contains(&v));
            assert!(values.contains(&v), "p={p} 的 {v} 不是样本里的值");
        }
    }

    #[test]
    fn token_throughput_reflects_output_length_not_request_count() {
        // 两条请求：一条 1 token，一条 100 token。
        let outcomes = vec![
            outcome(0, 0.0, 0.0, 10.0, 0.0, 1),
            outcome(1, 0.0, 0.0, 10.0, 1.0, 100),
        ];
        let m = ServeMetrics::from_outcomes(&outcomes, 1000.0);

        // 1 秒内 2 个请求 → 2 req/s
        assert!((m.request_throughput_per_sec - 2.0).abs() < 1e-9);
        // 101 个 token → 101 token/s
        assert!((m.output_token_throughput_per_sec - 101.0).abs() < 1e-9);
    }

    #[test]
    fn goodput_only_counts_requests_meeting_the_slo() {
        let slo = Slo::new(100.0, 10.0);
        let outcomes = vec![
            outcome(0, 0.0, 0.0, 50.0, 5.0, 10),   // 达标
            outcome(1, 0.0, 0.0, 500.0, 5.0, 10),  // TTFT 超标
            outcome(2, 0.0, 0.0, 50.0, 100.0, 10), // TPOT 超标
        ];
        let g = Goodput::from_outcomes(&outcomes, 1000.0, slo);

        assert_eq!(g.satisfied_requests, 1);
        assert_eq!(g.total_requests, 3);
        assert!((g.satisfaction_rate - 1.0 / 3.0).abs() < 1e-9);
        assert!((g.goodput_per_sec - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_tighter_slo_never_yields_more_goodput() {
        let outcomes: Vec<Outcome> = (0..20)
            .map(|i| outcome(i, 0.0, 0.0, 10.0 + i as f64 * 20.0, 5.0, 10))
            .collect();

        let loose = Goodput::from_outcomes(&outcomes, 1000.0, Slo::new(1000.0, 100.0));
        let tight = Goodput::from_outcomes(&outcomes, 1000.0, Slo::new(100.0, 10.0));

        assert!(tight.satisfied_requests <= loose.satisfied_requests);
        assert!(tight.goodput_per_sec <= loose.goodput_per_sec);
    }

    #[test]
    fn empty_outcomes_produce_zero_metrics() {
        let m = ServeMetrics::from_outcomes(&[], 0.0);
        assert_eq!(m.total_requests, 0);
        assert_eq!(m.request_throughput_per_sec, 0.0);
        assert_eq!(m.ttft_p50_ms, 0.0);
    }

    #[test]
    fn comparison_reports_ratios_in_both_directions() {
        let baseline =
            ServeMetrics::from_outcomes(&[outcome(0, 0.0, 0.0, 100.0, 10.0, 10)], 1000.0);
        let variant = ServeMetrics::from_outcomes(&[outcome(0, 0.0, 0.0, 50.0, 5.0, 10)], 500.0);
        let slo = Slo::default();
        let bg = Goodput::from_outcomes(&[], 0.0, slo);
        let vg = Goodput::from_outcomes(&[], 0.0, slo);

        let c = Comparison::new("基线", &baseline, &bg, "对照", &variant, &vg);
        assert!((c.request_throughput_ratio - 2.0).abs() < 1e-9);
        // ⚠️ TTFT 变好 → 倍数 < 1。
        assert!((c.ttft_p95_ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn comparison_does_not_divide_by_zero() {
        let empty = ServeMetrics::empty();
        let g = Goodput::from_outcomes(&[], 0.0, Slo::default());
        let c = Comparison::new("a", &empty, &g, "b", &empty, &g);
        assert_eq!(c.request_throughput_ratio, 1.0);
        assert_eq!(c.goodput_ratio, 1.0);
    }

    #[test]
    fn stddev_of_identical_values_is_zero() {
        assert!(stddev(&[2.0, 2.0, 2.0]).abs() < 1e-9);
        assert!((stddev(&[1.0, 3.0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sweep_produces_one_point_per_slo() {
        let outcomes: Vec<Outcome> = (0..5).map(|i| outcome(i, 0.0, 0.0, 10.0, 5.0, 4)).collect();
        let slos = vec![
            Slo::new(1000.0, 100.0),
            Slo::new(50.0, 10.0),
            Slo::new(10.0, 1.0),
        ];
        let sweep = SloSweep::sweep(&outcomes, 1000.0, &slos);
        assert_eq!(sweep.points.len(), 3);
    }

    #[test]
    fn simulate_integration_continuous_is_faster_than_static() {
        // 集成验证：模拟器 + 度量模块串起来。
        let arrivals: Vec<Arrival> = (0..40)
            .map(|i| Arrival {
                at_ms: i as f64 * 5.0,
                spec: RequestSpec {
                    id: i,
                    prompt_tokens: 64,
                    output_tokens: 32,
                },
            })
            .collect();

        let config = crate::scheduler::EngineConfig::default();
        let s = crate::scheduler::simulate(&config, BatchingMode::Static, &arrivals);
        let c = crate::scheduler::simulate(&config, BatchingMode::Continuous, &arrivals);

        let sm = s.metrics();
        let cm = c.metrics();

        assert_eq!(sm.total_requests, 40);
        assert_eq!(cm.total_requests, 40);
        // ⚠️ 连续批处理的 token 吞吐更高。
        assert!(
            cm.output_token_throughput_per_sec > sm.output_token_throughput_per_sec,
            "连续批处理 token 吞吐应更高：{} vs {}",
            cm.output_token_throughput_per_sec,
            sm.output_token_throughput_per_sec
        );
    }
}
