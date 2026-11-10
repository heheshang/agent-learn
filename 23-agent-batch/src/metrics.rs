//! 批处理的**度量** —— 把「快了多少」量化出来。
//!
//! ## 为什么单独抽一个模块
//!
//! 因为批处理的收益**不是**只看总耗时。
//!
//! 只看总耗时会得出错误结论：
//!
//! ```text
//! 逐条 32 条：总耗时 3200 ms，单条 p50 = 100 ms
//! 批处理 32 条：总耗时 800 ms，单条 p50 = 800 ms
//! ```
//!
//! 总耗时快了 4x，但**单条延迟从 100ms 涨到 800ms** ——
//! 如果用户是**逐条等结果**（而不是攒一批一起要），
//! 体验反而**变差**了。
//!
//! 所以批处理必须**同时**报告两个维度：
//!
//! | 维度 | 指标 | 谁关心 |
//! |---|---|---|
//! | **吞吐** | 条/秒、总耗时 | 离线批量灌库、索引构建 |
//! | **延迟** | 单条 p50 / p95 | 在线单条查询 |
//!
//! ## 百分位的算法
//!
//! 用**最近秩法**（nearest-rank）：
//! 第 p 百分位 = 排序后第 `ceil(p/100 × n)` 个元素（1-based）。
//!
//! ⚠️ 不用「插值法」是因为小样本下插值会产生
//! 样本里**根本不存在**的值（「p95 = 37.5ms」但从没测出 37.5ms），
//! 而本项目的样本量往往就是十几到几十条。
//!
//! ## ⚠️ 一个必须讲清的坑：总耗时 ≠ 单条延迟之和
//!
//! 批处理里 N 条数据**共享**一次前向的时间，
//! 所以「单条延迟之和」远大于「总耗时」。
//!
//! 本模块的 `sum_of_item_latencies_ms` 与 `elapsed_ms` 分开报，
//! 就是为了让这个差值**显式可见** ——
//! 否则会有人拿「单条延迟」当吞吐指标，得出完全错误的结论。

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// 从原始度量**派生**出来的指标（序列化时平铺进 JSON）。
///
/// ## 为什么要单独一个结构体
///
/// [`BatchStats`] 存的是「原始数据」（各批耗时、各条延迟），
/// 而吞吐 / 百分位 / 共享倍数都是**从原始数据算出来的**。
///
/// 把它们做成 `#[serde(flatten)]` 的辅助结构体：
///
/// - 内存里**不冗余存储**（不会出现「存的值」和「算的值」不一致）
/// - 序列化时**自动带上**（API 消费者直接就能拿到 `throughput_per_sec`）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedMetrics {
    /// 吞吐量（条/秒）。
    pub throughput_per_sec: f64,

    /// 每批平均条数。
    pub avg_batch_fill: f64,

    /// 单条延迟 p50（毫秒）。
    pub p50: u64,

    /// 单条延迟 p95（毫秒）。
    pub p95: u64,

    /// 单条延迟最大值（毫秒）。
    pub max_latency: u64,

    /// 单条延迟之和（毫秒）。
    pub sum_of_item_latencies_ms: u64,

    /// 「共享时间」的倍数。
    pub sharing_factor: f64,
}

/// 一批任务的度量结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    /// 总条数。
    pub total_items: usize,

    /// 总批次数。
    pub total_batches: usize,

    /// 端到端耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 每条数据的**完成延迟**（毫秒，从批次开始算到该条完成）。
    pub item_latencies_ms: Vec<u64>,

    /// 每个**批次**的耗时（毫秒）。
    pub batch_latencies_ms: Vec<u64>,

    /// 批大小。
    pub batch_size: usize,

    /// 是否按长度分桶。
    pub bucketed: bool,

    /// 派生指标（序列化时平铺）。
    #[serde(flatten)]
    pub derived: DerivedMetrics,
}

impl BatchStats {
    /// 构造一份度量。
    pub fn new(
        total_items: usize,
        total_batches: usize,
        elapsed: Duration,
        item_latencies_ms: Vec<u64>,
        batch_latencies_ms: Vec<u64>,
        batch_size: usize,
        bucketed: bool,
    ) -> Self {
        let mut stats = Self {
            total_items,
            total_batches,
            elapsed_ms: elapsed.as_millis() as u64,
            item_latencies_ms,
            batch_latencies_ms,
            batch_size,
            bucketed,
            derived: DerivedMetrics {
                throughput_per_sec: 0.0,
                avg_batch_fill: 0.0,
                p50: 0,
                p95: 0,
                max_latency: 0,
                sum_of_item_latencies_ms: 0,
                sharing_factor: 0.0,
            },
        };
        stats.refresh_derived();
        stats
    }

    /// 重新计算派生指标。
    ///
    /// ⚠️ 必须在**改动原始字段之后**调用，
    ///    否则派生值会和原始值对不上。
    pub fn refresh_derived(&mut self) {
        self.derived = DerivedMetrics {
            throughput_per_sec: self.throughput_per_sec(),
            avg_batch_fill: self.avg_batch_fill(),
            p50: self.p50(),
            p95: self.p95(),
            max_latency: self.max_latency(),
            sum_of_item_latencies_ms: self.sum_of_item_latencies_ms(),
            sharing_factor: self.sharing_factor(),
        };
    }

    /// 吞吐量（条/秒）。
    ///
    /// ⚠️ 用 `total_items / elapsed`，**不是** `total_items / sum(item_latencies)`。
    ///
    /// 后者是「单条延迟之和」，批处理下它会**严重高估**耗时，
    /// 从而低估吞吐（这是 22/23 项目反复出现的度量错误）。
    pub fn throughput_per_sec(&self) -> f64 {
        if self.elapsed_ms == 0 {
            return 0.0;
        }
        self.total_items as f64 / (self.elapsed_ms as f64 / 1000.0)
    }

    /// 每批平均条数。
    pub fn avg_batch_fill(&self) -> f64 {
        if self.total_batches == 0 {
            return 0.0;
        }
        self.total_items as f64 / self.total_batches as f64
    }

    /// 单条延迟的 p50（毫秒）。
    pub fn p50(&self) -> u64 {
        percentile(&self.item_latencies_ms, 50)
    }

    /// 单条延迟的 p95（毫秒）。
    pub fn p95(&self) -> u64 {
        percentile(&self.item_latencies_ms, 95)
    }

    /// 单条延迟的最大值（毫秒）。
    pub fn max_latency(&self) -> u64 {
        self.item_latencies_ms.iter().copied().max().unwrap_or(0)
    }

    /// 单条延迟之和（毫秒）。
    ///
    /// ⚠️ 批处理下这个值**远大于** `elapsed_ms` ——
    /// 因为 N 条数据共享同一次前向的时间。
    /// 报出来是为了让这个差值显式可见。
    pub fn sum_of_item_latencies_ms(&self) -> u64 {
        self.item_latencies_ms.iter().sum()
    }

    /// 「共享时间」的倍数 = 单条延迟之和 / 端到端耗时。
    ///
    /// 这个值 ≈ 有效批大小（理想情况下）。
    /// 它量化了「一次前向的服务了几条数据」。
    pub fn sharing_factor(&self) -> f64 {
        if self.elapsed_ms == 0 {
            return 0.0;
        }
        self.sum_of_item_latencies_ms() as f64 / self.elapsed_ms as f64
    }

    /// 相对基线的加速比。
    ///
    /// 基线耗时为 0 时返回 1.0（避免除零）。
    pub fn speedup_vs(&self, baseline_ms: u64) -> f64 {
        if baseline_ms == 0 {
            return 1.0;
        }
        baseline_ms as f64 / self.elapsed_ms.max(1) as f64
    }

    /// 相对基线的吞吐倍数。
    pub fn throughput_ratio_vs(&self, baseline: &BatchStats) -> f64 {
        let b = baseline.throughput_per_sec();
        if b <= f64::EPSILON {
            return 1.0;
        }
        self.throughput_per_sec() / b
    }

    /// 一行摘要（给终端表格用）。
    pub fn summary_line(&self) -> String {
        format!(
            "条数 {}　批次 {}　批大小 {}　分桶 {}　总耗时 {} ms　吞吐 {:.1} 条/秒　p50 {} ms　p95 {} ms",
            self.total_items,
            self.total_batches,
            self.batch_size,
            if self.bucketed { "是" } else { "否" },
            self.elapsed_ms,
            self.throughput_per_sec(),
            self.p50(),
            self.p95(),
        )
    }
}

/// 百分位（最近秩法）。
///
/// n = 0 → 0；n = 1 → 唯一那个值。
pub fn percentile(values: &[u64], p: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();

    // 最近秩：第 ceil(p/100 × n) 个（1-based）。
    // ⚠️ 用整数运算避免浮点误差导致 off-by-one。
    let rank = (p * sorted.len()).div_ceil(100).max(1);
    sorted[(rank - 1).min(sorted.len() - 1)]
}

/// 两组数据的**均值**。
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// 两组数据的**标准差**（总体标准差）。
///
/// 用途：报告「加速比在几次运行间的波动」——
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_nearest_rank() {
        let values = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

        // p50 → ceil(0.5*10) = 5 → 第 5 个 = 50
        assert_eq!(percentile(&values, 50), 50);
        // p95 → ceil(0.95*10) = 10 → 第 10 个 = 100
        assert_eq!(percentile(&values, 95), 100);
        // p100 → 最后一个
        assert_eq!(percentile(&values, 100), 100);
    }

    #[test]
    fn percentile_of_single_value_is_that_value() {
        assert_eq!(percentile(&[42], 50), 42);
        assert_eq!(percentile(&[42], 95), 42);
    }

    #[test]
    fn percentile_of_empty_is_zero() {
        assert_eq!(percentile(&[], 50), 0);
    }

    #[test]
    fn percentile_never_exceeds_the_max_observed() {
        // ⚠️ 插值法会产生「样本里不存在的值」，
        //    最近秩法保证结果一定是某个真实测到的值。
        let values = vec![7u64, 13, 29];
        for p in [0, 25, 50, 75, 99, 100] {
            let v = percentile(&values, p);
            assert!((7..=29).contains(&v), "p={p} 得到 {v}，越界");
            assert!(values.contains(&v), "p={p} 的 {v} 不是样本里的值");
        }
    }

    #[test]
    fn throughput_uses_elapsed_not_sum_of_item_latencies() {
        let stats = BatchStats::new(
            4,
            1,
            Duration::from_millis(100),
            vec![100, 100, 100, 100],
            vec![100],
            4,
            false,
        );

        // 4 条数据共享一次 100ms 的前向 → 吞吐 40 条/秒。
        assert!((stats.throughput_per_sec() - 40.0).abs() < 1e-6);

        // ⚠️ 单条延迟之和 = 400ms，但端到端只有 100ms。
        assert_eq!(stats.sum_of_item_latencies_ms(), 400);
        assert_eq!(stats.elapsed_ms, 100);
        assert!((stats.sharing_factor() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn speedup_and_ratio_are_consistent() {
        let baseline = BatchStats::new(
            4,
            4,
            Duration::from_millis(400),
            vec![100, 100, 100, 100],
            vec![100, 100, 100, 100],
            1,
            false,
        );
        let batched = BatchStats::new(
            4,
            1,
            Duration::from_millis(100),
            vec![100, 100, 100, 100],
            vec![100],
            4,
            false,
        );

        assert!((batched.speedup_vs(baseline.elapsed_ms) - 4.0).abs() < 1e-6);
        assert!((batched.throughput_ratio_vs(&baseline) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn zero_elapsed_does_not_divide_by_zero() {
        let stats = BatchStats::new(1, 1, Duration::from_millis(0), vec![0], vec![0], 1, false);
        assert_eq!(stats.throughput_per_sec(), 0.0);
        assert_eq!(stats.sharing_factor(), 0.0);
        assert_eq!(stats.speedup_vs(0), 1.0);
    }

    #[test]
    fn stddev_of_identical_values_is_zero() {
        assert!(stddev(&[2.0, 2.0, 2.0]).abs() < 1e-9);
    }

    #[test]
    fn stddev_of_two_values() {
        // [1, 3] → 均值 2 → 方差 ((1-2)² + (3-2)²)/2 = 1 → 标准差 1
        assert!((stddev(&[1.0, 3.0]) - 1.0).abs() < 1e-9);
    }
}
