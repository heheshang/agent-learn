//! 度量层 —— QPS / P99 等压测指标。
//!
//! # ⚠️ 本项目**不用**第三方压测框架做指标口径
//!
//! Locust / JMeter 有自己的口径（它们的 P99 算法、QPS 定义各不相同）。
//! 本项目**自己算**，是为了让「Rust 内置压测」与「Locust 压测」**可比** ——
//! 两者对同一服务跑出来的数字必须能对照。
//!
//! # 三类指标
//!
//! | 类别 | 指标 | 含义 |
//! |---|---|---|
//! | **吞吐** | QPS | 每秒完成的请求数 |
//! | | 成功率 | 成功请求 / 总请求 |
//! | **延迟** | p50 / p90 / p95 / **p99** | 端到端延迟的百分位 |
//! | | max | 最慢的那一次 |
//!
//! # ⚠️ P99 用**最近秩法**，不用插值
//!
//! 与 23 / 24 保持一致。小样本下插值会产生「样本里根本不存在的值」
//! （「p99 = 37.5ms」但从没测出 37.5ms）。
//! 最近秩法保证结果**一定是某个真实测到的值**。
//!
//! # ⚠️ P99 需要**足够多的样本**
//!
//! 100 个请求的 P99 就是「第 99 个请求」—— 几乎没有统计意义。
//! 所以本项目的压测默认请求数是 **500**，且报告里会**同时打印样本数**，
//! 提醒读者「P99 的可信度取决于样本数」。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// 一次压测的结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadReport {
    /// 变体名（`naive` / `optimized` / …）。
    pub variant: String,

    /// 总请求数。
    pub total_requests: usize,

    /// 成功数。
    pub succeeded: usize,

    /// 失败数。
    pub failed: usize,

    /// 总墙钟时间（毫秒）。
    pub wall_ms: f64,

    /// ★ 每秒请求数（QPS）。
    pub qps: f64,

    /// ★ 成功率（0.0 ~ 1.0）。
    pub success_rate: f64,

    /// 延迟 p50（毫秒）。
    pub p50_ms: f64,

    /// 延迟 p90（毫秒）。
    pub p90_ms: f64,

    /// 延迟 p95（毫秒）。
    pub p95_ms: f64,

    /// ★ 延迟 p99（毫秒）。
    pub p99_ms: f64,

    /// 最慢的一次（毫秒）。
    pub max_ms: f64,

    /// 平均延迟（毫秒）。
    pub avg_ms: f64,

    /// 并发度。
    pub concurrency: usize,

    /// 状态码分布（`200` → 次数）。
    pub status_counts: HashMap<String, usize>,

    /// 缓存命中数（如果被测服务暴露这个字段）。
    pub cache_hits: Option<u64>,
}

impl LoadReport {
    /// 从原始采样算出报告。
    pub fn from_samples(
        variant: &str,
        samples: &[Sample],
        wall_ms: f64,
        concurrency: usize,
    ) -> Self {
        let succeeded = samples.iter().filter(|s| s.ok).count();
        let failed = samples.len() - succeeded;

        let mut latencies: Vec<f64> = samples.iter().map(|s| s.elapsed_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut status_counts: HashMap<String, usize> = HashMap::new();
        for s in samples {
            *status_counts
                .entry(
                    s.status
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "err".into()),
                )
                .or_insert(0) += 1;
        }

        let cache_hits = if samples.iter().all(|s| s.cache_hit.is_some()) {
            Some(samples.iter().filter(|s| s.cache_hit == Some(true)).count() as u64)
        } else {
            None
        };

        let secs = wall_ms / 1000.0;

        Self {
            variant: variant.to_string(),
            total_requests: samples.len(),
            succeeded,
            failed,
            wall_ms,
            qps: if secs > 0.0 {
                samples.len() as f64 / secs
            } else {
                0.0
            },
            success_rate: if samples.is_empty() {
                0.0
            } else {
                succeeded as f64 / samples.len() as f64
            },
            p50_ms: percentile(&latencies, 50),
            p90_ms: percentile(&latencies, 90),
            p95_ms: percentile(&latencies, 95),
            p99_ms: percentile(&latencies, 99),
            max_ms: latencies.last().copied().unwrap_or(0.0),
            avg_ms: mean(&latencies),
            concurrency,
            status_counts,
            cache_hits,
        }
    }

    /// 空报告。
    pub fn empty(variant: &str) -> Self {
        Self {
            variant: variant.to_string(),
            total_requests: 0,
            succeeded: 0,
            failed: 0,
            wall_ms: 0.0,
            qps: 0.0,
            success_rate: 0.0,
            p50_ms: 0.0,
            p90_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            max_ms: 0.0,
            avg_ms: 0.0,
            concurrency: 0,
            status_counts: HashMap::new(),
            cache_hits: None,
        }
    }
}

/// 一次请求的采样。
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// 是否成功。
    pub ok: bool,

    /// 端到端耗时（毫秒）。
    pub elapsed_ms: f64,

    /// HTTP 状态码（非 HTTP 场景为 `None`）。
    pub status: Option<u16>,

    /// 该请求是否命中缓存。
    pub cache_hit: Option<bool>,
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

/// 两个报告的**对比**。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    /// 基线变体名。
    pub baseline: String,

    /// 对照变体名。
    pub variant: String,

    /// QPS 倍数（> 1 表示更快）。
    pub qps_ratio: f64,

    /// p50 倍数（> 1 表示更慢）。
    pub p50_ratio: f64,

    /// p99 倍数（> 1 表示更慢）。
    pub p99_ratio: f64,

    /// 吞吐提升百分比。
    pub qps_gain_pct: f64,

    /// p99 下降百分比。
    pub p99_drop_pct: f64,
}

impl Comparison {
    /// 对比两份报告。
    pub fn new(baseline: &LoadReport, variant: &LoadReport) -> Self {
        let qps_ratio = safe_ratio(variant.qps, baseline.qps);
        let p50_ratio = safe_ratio(variant.p50_ms, baseline.p50_ms);
        let p99_ratio = safe_ratio(variant.p99_ms, baseline.p99_ms);

        Self {
            baseline: baseline.variant.clone(),
            variant: variant.variant.clone(),
            qps_ratio,
            p50_ratio,
            p99_ratio,
            qps_gain_pct: (qps_ratio - 1.0) * 100.0,
            p99_drop_pct: (1.0 - p99_ratio) * 100.0,
        }
    }
}

/// 安全除法（分母为 0 时返回 1.0）。
fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() < f64::EPSILON {
        1.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ok: bool, ms: f64) -> Sample {
        Sample {
            ok,
            elapsed_ms: ms,
            status: Some(if ok { 200 } else { 500 }),
            cache_hit: None,
        }
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(percentile(&values, 50), 50.0);
        assert_eq!(percentile(&values, 99), 99.0);
        assert_eq!(percentile(&values, 100), 100.0);
    }

    #[test]
    fn percentile_never_invents_a_value() {
        let values = vec![7.0, 13.0, 29.0];
        for p in [0, 25, 50, 75, 99, 100] {
            assert!(values.contains(&percentile(&values, p)));
        }
    }

    #[test]
    fn percentile_of_empty_is_zero() {
        assert_eq!(percentile(&[], 99), 0.0);
    }

    #[test]
    fn qps_is_requests_over_wall_time() {
        let samples: Vec<Sample> = (0..100).map(|_| sample(true, 10.0)).collect();
        let report = LoadReport::from_samples("x", &samples, 1000.0, 10);

        assert_eq!(report.total_requests, 100);
        assert_eq!(report.succeeded, 100);
        assert!((report.qps - 100.0).abs() < 1e-9);
        assert!((report.success_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn failures_are_counted_and_lower_success_rate() {
        let mut samples: Vec<Sample> = (0..90).map(|_| sample(true, 10.0)).collect();
        samples.extend((0..10).map(|_| sample(false, 20.0)));

        let report = LoadReport::from_samples("x", &samples, 1000.0, 10);
        assert_eq!(report.failed, 10);
        assert!((report.success_rate - 0.9).abs() < 1e-9);
        assert_eq!(report.status_counts.get("500"), Some(&10));
    }

    #[test]
    fn cache_hits_are_summed_when_all_samples_report_it() {
        let samples: Vec<Sample> = vec![
            Sample {
                ok: true,
                elapsed_ms: 1.0,
                status: Some(200),
                cache_hit: Some(true),
            },
            Sample {
                ok: true,
                elapsed_ms: 2.0,
                status: Some(200),
                cache_hit: Some(false),
            },
        ];

        let report = LoadReport::from_samples("x", &samples, 1000.0, 1);
        assert_eq!(report.cache_hits, Some(1));
    }

    #[test]
    fn cache_hits_are_none_when_samples_are_inconsistent() {
        let mut samples: Vec<Sample> = vec![sample(true, 1.0)];
        samples[0].cache_hit = Some(true);
        samples.push(Sample {
            ok: true,
            elapsed_ms: 1.0,
            status: Some(200),
            cache_hit: None,
        });

        let report = LoadReport::from_samples("x", &samples, 1000.0, 1);
        assert_eq!(report.cache_hits, None);
    }

    #[test]
    fn comparison_reports_both_directions() {
        let baseline = LoadReport::from_samples("naive", &[sample(true, 100.0)], 1000.0, 1);
        let variant = LoadReport::from_samples("optimized", &[sample(true, 50.0)], 1000.0, 1);

        let c = Comparison::new(&baseline, &variant);
        // 同样的墙钟、同样的请求数 → QPS 相同。
        assert!((c.qps_ratio - 1.0).abs() < 1e-9);
        // p50 减半 → 倍数 0.5（越小越好）。
        assert!((c.p50_ratio - 0.5).abs() < 1e-9);
        assert!((c.p99_drop_pct - 50.0).abs() < 1e-9);
    }

    #[test]
    fn comparison_does_not_divide_by_zero() {
        let empty = LoadReport::empty("a");
        let c = Comparison::new(&empty, &empty);
        assert_eq!(c.qps_ratio, 1.0);
        assert_eq!(c.p99_ratio, 1.0);
    }
}
