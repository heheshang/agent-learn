//! ★ 对照实验 —— 把「批处理优化到底带来了什么」量化出来。
//!
//! # 三组对照实验
//!
//! | 实验 | 唯一变量 | 回答的问题 |
//! |---|---|---|
//! | **实验 A** | 批处理模式（静态 vs 连续） | 连续批处理比静态快多少？ |
//! | **实验 B** | `max_num_seqs`（并发上限） | 批开多大最划算？ |
//! | **实验 C** | chunked prefill（开/关） | 长 prompt 会不会拖垮其它请求？ |
//!
//! # ⚠️ 变量单一原则
//!
//! 每组实验**只改一个变量**，其它全部固定 ——
//! 否则差值归因不到变量上（20/21/22 项目的反复教训）。
//!
//! # ⚠️ 每一组都跑**多轮取中位数**
//!
//! 单轮的数字会被抖动污染。本项目默认跑 3 轮，
//! 报告**中位数**并同时给出**标准差** ——
//! 「加速比在 1.8x~2.3x 之间浮动」是常态，
//! 可复现的结论是**相对关系**而不是绝对值。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::metrics::{Comparison, Goodput, ServeMetrics, Slo};
use crate::scheduler::{BatchingMode, EngineConfig, simulate};
use crate::workload::{self, WorkloadConfig};

/// 一轮实验的原始结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundResult {
    /// 第几轮（从 0 开始）。
    pub round: usize,

    /// 该轮的度量。
    pub metrics: ServeMetrics,

    /// 该轮的 goodput。
    pub goodput: Goodput,
}

/// 一组对照实验的完整结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// 实验名字。
    pub name: String,

    /// 被扫描的参数名。
    pub param: String,

    /// 每个参数档位的结果（中位数聚合后）。
    pub variants: Vec<VariantResult>,
}

/// 一个参数档位的结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantResult {
    /// 参数值（字符串，便于展示）。
    pub value: String,

    /// 中位数聚合后的度量。
    pub metrics: ServeMetrics,

    /// 中位数聚合后的 goodput。
    pub goodput: Goodput,

    /// 各轮的度量（保留原始数据，便于看抖动）。
    pub rounds: Vec<ServeMetrics>,

    /// 请求吞吐在各轮间的**标准差**（相对抖动）。
    pub request_throughput_stddev: f64,
}

impl VariantResult {
    /// 吞吐的**相对抖动**（标准差 / 均值）。
    ///
    /// ⚠️ 这个值比标准差本身更有意义 ——
    /// 吞吐是 1000 req/s 时抖 50 是 5%，吞吐 10 req/s 时抖 5 是 50%。
    pub fn throughput_cv(&self) -> f64 {
        let mean = crate::metrics::mean(
            &self
                .rounds
                .iter()
                .map(|m| m.request_throughput_per_sec)
                .collect::<Vec<_>>(),
        );
        if mean.abs() < f64::EPSILON {
            return 0.0;
        }
        self.request_throughput_stddev / mean
    }
}

/// ★ 跑一组对照实验。
///
/// ## 参数
///
/// - `name` —— 实验名字
/// - `param` —— 被扫描的参数名
/// - `variants` —— 每个档位的 `(标签, 引擎配置)`
/// - `workload` —— 负载配置（**所有档位共用** → 变量单一）
/// - `rounds` —— 每档跑几轮
/// - `slo` —— goodput 的 SLO 约束
pub fn run_experiment(
    name: &str,
    param: &str,
    variants: &[(&str, EngineConfig)],
    workload: &WorkloadConfig,
    rounds: usize,
    slo: Slo,
) -> ExperimentResult {
    let arrivals = workload::generate(workload);
    let mut results = Vec::with_capacity(variants.len());

    for (label, engine) in variants {
        let mut round_results: Vec<RoundResult> = Vec::new();

        for round in 0..rounds.max(1) {
            let sim = simulate(engine, BatchingMode::Continuous, &arrivals);
            let metrics = sim.metrics();
            let goodput = Goodput::from_outcomes(&sim.outcomes, sim.wall_ms, slo);
            round_results.push(RoundResult {
                round,
                metrics,
                goodput,
            });
        }

        // ⚠️ 取**中位数**而不是平均值 ——
        //    平均值会被一次极端抖动拉偏，而中位数不会。
        let metrics = median_metrics(&round_results);
        let goodput = median_goodput(&round_results);
        let throughputs: Vec<f64> = round_results
            .iter()
            .map(|r| r.metrics.request_throughput_per_sec)
            .collect();

        results.push(VariantResult {
            value: (*label).to_string(),
            metrics,
            goodput,
            rounds: round_results.into_iter().map(|r| r.metrics).collect(),
            request_throughput_stddev: crate::metrics::stddev(&throughputs),
        });
    }

    ExperimentResult {
        name: name.to_string(),
        param: param.to_string(),
        variants: results,
    }
}

/// 取中位数聚合后的度量。
fn median_metrics(rounds: &[RoundResult]) -> ServeMetrics {
    if rounds.is_empty() {
        return ServeMetrics::empty();
    }

    let median_of = |f: fn(&ServeMetrics) -> f64| -> f64 {
        let mut values: Vec<f64> = rounds.iter().map(|r| f(&r.metrics)).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values[values.len() / 2]
    };

    ServeMetrics {
        total_requests: rounds[0].metrics.total_requests,
        wall_ms: median_of(|m| m.wall_ms),
        request_throughput_per_sec: median_of(|m| m.request_throughput_per_sec),
        output_token_throughput_per_sec: median_of(|m| m.output_token_throughput_per_sec),
        ttft_p50_ms: median_of(|m| m.ttft_p50_ms),
        ttft_p95_ms: median_of(|m| m.ttft_p95_ms),
        ttft_p99_ms: median_of(|m| m.ttft_p99_ms),
        tpot_p50_ms: median_of(|m| m.tpot_p50_ms),
        tpot_p95_ms: median_of(|m| m.tpot_p95_ms),
        e2e_p50_ms: median_of(|m| m.e2e_p50_ms),
        e2e_p95_ms: median_of(|m| m.e2e_p95_ms),
        avg_queue_wait_ms: median_of(|m| m.avg_queue_wait_ms),
        avg_output_tokens: median_of(|m| m.avg_output_tokens),
    }
}

/// 取中位数聚合后的 goodput。
fn median_goodput(rounds: &[RoundResult]) -> Goodput {
    if rounds.is_empty() {
        return Goodput::from_outcomes(&[], 0.0, Slo::default());
    }

    let mut points = rounds.iter().map(|r| r.goodput.clone()).collect::<Vec<_>>();
    points.sort_by(|a, b| {
        a.goodput_per_sec
            .partial_cmp(&b.goodput_per_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut picked = points[points.len() / 2].clone();
    // ⚠️ 达标数取中位数（而不是用中位数那轮的）——
    //    两者可能不一致（不同轮的达标情况不同）。
    let mut satisfied: Vec<usize> = rounds
        .iter()
        .map(|r| r.goodput.satisfied_requests)
        .collect();
    satisfied.sort_unstable();
    picked.satisfied_requests = satisfied[satisfied.len() / 2];
    picked
}

/// ★ 实验 A：静态批处理 vs 连续批处理。
///
/// ## 预期
///
/// 连续批处理的**吞吐更高**、**排队等待更短**。
///
/// ## ⚠️ 但 TTFT 可能**变差**
///
/// 连续批处理下，请求要等「位置」而不是等「整批」——
/// 听起来更好，但一旦批被塞满，新请求就得等更久。
/// 到底是变好还是变坏，**取决于负载强度** —— 所以必须实测。
pub fn experiment_batching_mode(
    workload: &WorkloadConfig,
    rounds: usize,
    slo: Slo,
) -> ExperimentResult {
    let engine = EngineConfig::default();
    let arrivals = workload::generate(workload);

    let mut variants = Vec::new();

    for (label, mode) in [
        ("静态批处理", BatchingMode::Static),
        ("连续批处理", BatchingMode::Continuous),
    ] {
        let mut round_results = Vec::new();
        for round in 0..rounds.max(1) {
            let sim = simulate(&engine, mode, &arrivals);
            let metrics = sim.metrics();
            let goodput = Goodput::from_outcomes(&sim.outcomes, sim.wall_ms, slo);
            round_results.push(RoundResult {
                round,
                metrics,
                goodput,
            });
        }

        let metrics = median_metrics(&round_results);
        let goodput = median_goodput(&round_results);
        let throughputs: Vec<f64> = round_results
            .iter()
            .map(|r| r.metrics.request_throughput_per_sec)
            .collect();

        variants.push(VariantResult {
            value: label.to_string(),
            metrics,
            goodput,
            rounds: round_results.into_iter().map(|r| r.metrics).collect(),
            request_throughput_stddev: crate::metrics::stddev(&throughputs),
        });
    }

    ExperimentResult {
        name: "实验 A：批处理模式对照".to_string(),
        param: "批处理模式".to_string(),
        variants,
    }
}

/// ★ 实验 B：扫描 `max_num_seqs`（服务端同时处理的序列数上限）。
///
/// ## 预期
///
/// - 批**小** → 每批的固定开销分摊不足 → 吞吐低
/// - 批**大** → 排队变长 → TTFT 变差
/// - 存在一个**甜点**
pub fn experiment_max_num_seqs(
    workload: &WorkloadConfig,
    rounds: usize,
    slo: Slo,
) -> ExperimentResult {
    let variants: Vec<(&str, EngineConfig)> = [1usize, 4, 16, 64, 256]
        .iter()
        .map(|&n| {
            (
                Box::leak(n.to_string().into_boxed_str()) as &str,
                EngineConfig {
                    max_num_seqs: n,
                    ..EngineConfig::default()
                },
            )
        })
        .collect();

    run_experiment(
        "实验 B：服务端并发上限扫描",
        "max_num_seqs",
        &variants,
        workload,
        rounds,
        slo,
    )
}

/// ★ 实验 C：chunked prefill 开/关。
///
/// ## 预期
///
/// 关闭时，一条长 prompt 会**霸占**整个 forward，
/// 其它正在 decode 的请求全部被阻塞（head-of-line blocking），
/// 它们的 TTFT 会明显变差。
pub fn experiment_chunked_prefill(
    workload: &WorkloadConfig,
    rounds: usize,
    slo: Slo,
) -> ExperimentResult {
    let variants: Vec<(&str, EngineConfig)> = vec![
        (
            "关闭",
            EngineConfig {
                enable_chunked_prefill: false,
                ..EngineConfig::default()
            },
        ),
        (
            "开启",
            EngineConfig {
                enable_chunked_prefill: true,
                ..EngineConfig::default()
            },
        ),
    ];

    run_experiment(
        "实验 C：chunked prefill 对照",
        "chunked prefill",
        &variants,
        workload,
        rounds,
        slo,
    )
}

/// 把多组实验的结果整理成一个可序列化的总览。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentSuite {
    /// 每组实验。
    pub experiments: Vec<ExperimentResult>,

    /// 用的 SLO。
    pub slo: Slo,
}

impl ExperimentSuite {
    /// 构造一个总览。
    pub fn new(experiments: Vec<ExperimentResult>, slo: Slo) -> Self {
        Self { experiments, slo }
    }

    /// 找某个实验。
    pub fn find(&self, name: &str) -> Option<&ExperimentResult> {
        self.experiments.iter().find(|e| e.name.contains(name))
    }

    /// 把某个实验的第一个档位当基线，与后续档位对比。
    pub fn compare_with_first(&self, experiment: &ExperimentResult) -> Vec<Comparison> {
        let Some(base) = experiment.variants.first() else {
            return Vec::new();
        };
        experiment
            .variants
            .iter()
            .skip(1)
            .map(|v| {
                Comparison::new(
                    &base.value,
                    &base.metrics,
                    &base.goodput,
                    &v.value,
                    &v.metrics,
                    &v.goodput,
                )
            })
            .collect()
    }
}

/// 各档位吞吐的汇总（用于画曲线）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThroughputCurve {
    /// 参数名。
    pub param: String,

    /// 每个档位的「标签 → 吞吐」。
    pub points: BTreeMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::ArrivalModel;

    fn workload() -> WorkloadConfig {
        WorkloadConfig {
            num_requests: 60,
            arrival_model: ArrivalModel::Burst,
            request_rate: 20.0,
            // ⚠️ ★ 必须是**参差不齐**的长度 ——
            //    长度一致时静态批处理与连续批处理**完全等价**，
            //    测不出差异是**正确**的，不是 bug。
            //    真实场景里请求长度天然参差。
            length_distribution: crate::workload::LengthDistribution::Uniform,
            min_input_tokens: 32,
            max_input_tokens: 512,
            min_output_tokens: 16,
            max_output_tokens: 128,
            seed: 42,
        }
    }

    #[test]
    fn continuous_batching_beats_static_in_throughput() {
        let result = experiment_batching_mode(&workload(), 1, Slo::default());
        assert_eq!(result.variants.len(), 2);

        let statik = &result.variants[0];
        let cont = &result.variants[1];
        assert!(
            cont.metrics.request_throughput_per_sec > statik.metrics.request_throughput_per_sec,
            "连续批处理吞吐应更高：{} vs {}",
            cont.metrics.request_throughput_per_sec,
            statik.metrics.request_throughput_per_sec
        );
    }

    #[test]
    fn continuous_batching_has_shorter_queue_wait() {
        let result = experiment_batching_mode(&workload(), 1, Slo::default());
        let statik = &result.variants[0];
        let cont = &result.variants[1];
        assert!(
            cont.metrics.avg_queue_wait_ms < statik.metrics.avg_queue_wait_ms,
            "连续批处理排队等待应更短：{} vs {}",
            cont.metrics.avg_queue_wait_ms,
            statik.metrics.avg_queue_wait_ms
        );
    }

    #[test]
    fn max_num_seqs_scan_has_a_sweet_spot() {
        let result = experiment_max_num_seqs(&workload(), 1, Slo::default());
        assert_eq!(result.variants.len(), 5);

        // 单序列（max_num_seqs=1）必然最慢 —— 批不起来。
        let single = &result.variants[0];
        let best = result
            .variants
            .iter()
            .max_by(|a, b| {
                a.metrics
                    .request_throughput_per_sec
                    .partial_cmp(&b.metrics.request_throughput_per_sec)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        assert!(
            single.metrics.request_throughput_per_sec < best.metrics.request_throughput_per_sec
        );
    }

    #[test]
    fn chunked_prefill_improves_tail_latency_under_mixed_lengths() {
        // ⚠️ 关键前提：负载里**必须有一条长 prompt**，
        //    否则两种模式没有区别（测不出差异）。
        let mut wl = workload();
        wl.length_distribution = crate::workload::LengthDistribution::Uniform;
        wl.min_input_tokens = 16;
        wl.max_input_tokens = 4096;
        wl.num_requests = 40;

        let result = experiment_chunked_prefill(&wl, 1, Slo::default());
        let off = &result.variants[0];
        let on = &result.variants[1];

        assert!(
            on.metrics.ttft_p95_ms < off.metrics.ttft_p95_ms,
            "chunked prefill 应降低 TTFT p95：{} vs {}",
            on.metrics.ttft_p95_ms,
            off.metrics.ttft_p95_ms
        );
    }

    #[test]
    fn multiple_rounds_report_stddev() {
        let result = experiment_batching_mode(&workload(), 3, Slo::default());
        let v = &result.variants[0];
        assert_eq!(v.rounds.len(), 3);
        assert!(v.request_throughput_stddev >= 0.0);
    }

    #[test]
    fn suite_can_find_and_compare_experiments() {
        let a = experiment_batching_mode(&workload(), 1, Slo::default());
        let suite = ExperimentSuite::new(vec![a], Slo::default());

        let found = suite.find("实验 A").unwrap();
        let comparisons = suite.compare_with_first(found);
        assert_eq!(comparisons.len(), 1);
        // 连续批处理相对静态批处理的吞吐倍数应 > 1。
        assert!(comparisons[0].request_throughput_ratio > 1.0);
    }
}
