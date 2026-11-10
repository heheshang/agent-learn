//! ★ **padding 浪费**的量化 —— 分桶策略收益的**根因**。
//!
//! # 为什么这个指标是核心
//!
//! 批处理省下的是「固定开销」。
//! 但批处理**同时引入**了 padding 浪费。
//!
//! ```text
//! 神经网络要求一个 tensor 的形状规整，
//! 所以批里的短文本必须被 padding 到该批最长的那条。
//!
//! 批里三条，长度 12 / 300 / 8 token
//!                  ↓ padding 到 300
//! 实际算的是 12×300 + 300×300 + 8×300
//! ```
//!
//! 短的那两条**白算了**。
//! 所以批越大、长度差异越大，浪费越多。
//!
//! ## 两个指标
//!
//! | 指标 | 定义 | 意义 |
//! |---|---|---|
//! | **padding 系数** | `Σ(批内最长 × 批内条数) / Σ(实际长度)` | 1.0 = 零浪费；2.0 = 算了两倍 |
//! | **浪费占比** | `1 - 1/系数` | 浪费掉的比例 |
//!
//! ## 为什么必须**同时**报「批大小」和「padding 系数」
//!
//! 只看 padding 系数会得出错误结论 ——
//! 把批大小调到 1，系数就是 1.0（零浪费），但吞吐也归零。
//!
//! 所以真正的权衡是：
//!
//! ```text
//! 批大小 ↑  → 固定开销分摊更多（吞吐 ↑）但 padding 系数可能 ↑
//! 批大小 ↓  → padding 系数 ↓（浪费少）但固定开销分摊少（吞吐 ↓）
//! ```
//!
//! **长度分桶的价值正是在这个权衡里取胜**：
//! 它在**同样的批大小**下把 padding 系数压到更低。
//! 本模块的 `compare_padding` 就是用来证明这一点的。

use serde::{Deserialize, Serialize};

/// 一次批处理的 padding 统计。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddingReport {
    /// 批大小。
    pub batch_size: usize,

    /// 总批次数。
    pub batch_count: usize,

    /// 实际 token 数之和（估算）。
    pub actual_tokens: u64,

    /// 「补齐后」实际计算的 token 数之和（估算）。
    ///
    /// = Σ(每批最长 × 该批条数)
    pub padded_tokens: u64,

    /// padding 系数（≥ 1.0）。
    pub padding_factor: f64,

    /// 浪费占比（0.0 ~ 1.0）。
    pub waste_ratio: f64,
}

impl PaddingReport {
    /// 构造一份报告。
    pub fn new(batch_size: usize, batch_count: usize, actual: u64, padded: u64) -> Self {
        let padding_factor = if actual == 0 {
            1.0
        } else {
            padded as f64 / actual as f64
        };

        Self {
            batch_size,
            batch_count,
            actual_tokens: actual,
            padded_tokens: padded,
            padding_factor,
            waste_ratio: 1.0 - (1.0 / padding_factor.max(1.0)),
        }
    }

    /// 一行摘要。
    pub fn summary_line(&self) -> String {
        format!(
            "批大小 {:>4}　批次 {:>4}　实际 {} token　补齐后 {} token　padding 系数 {:.3}　浪费 {:.1}%",
            self.batch_size,
            self.batch_count,
            self.actual_tokens,
            self.padded_tokens,
            self.padding_factor,
            self.waste_ratio * 100.0,
        )
    }
}

/// 计算一批「按固定批大小切分」时的 padding 浪费。
///
/// ## ⚠️ 这里算的是**理论**浪费
///
/// 假设推理耗时与「补齐后的 token 数」成正比。
/// 真实耗时还包含固定开销，所以实测加速比会**小于**
/// padding 系数给出的上限 —— 这是正常的。
///
/// 但**组间对比**（固定批 vs 分桶）依然有效，
/// 因为两组的固定开销是一样的。
pub fn padding_of_chunks(chunks: &[Vec<usize>]) -> PaddingReport {
    let mut actual = 0u64;
    let mut padded = 0u64;

    for chunk in chunks {
        if chunk.is_empty() {
            continue;
        }
        let max = *chunk.iter().max().unwrap();
        actual += chunk.iter().map(|l| *l as u64).sum::<u64>();
        padded += max as u64 * chunk.len() as u64;
    }

    PaddingReport::new(0, chunks.len(), actual, padded)
}

/// 计算一批「按长度排序后切分」时的 padding 浪费。
///
/// ⚠️ 与 [`padding_of_chunks`] 的**唯一区别**是切分前的排序 ——
/// 这正是「分桶」的全部收益来源。
pub fn padding_of_sorted(lengths: &[usize], batch_size: usize) -> PaddingReport {
    if lengths.is_empty() || batch_size == 0 {
        return PaddingReport::new(batch_size, 0, 0, 0);
    }

    let mut sorted = lengths.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));

    let chunks: Vec<Vec<usize>> = sorted.chunks(batch_size).map(|c| c.to_vec()).collect();
    let mut report = padding_of_chunks(&chunks);
    report.batch_size = batch_size;
    report
}

/// 对比「固定批」与「长度分桶」的 padding 浪费。
///
/// ## ⚠️ 这个对比是分桶收益的**直接证据**
///
/// 两组的**批大小相同**、**输入相同**，
/// 唯一区别是「切分前是否按长度排序」。
/// 所以 padding 系数的差值**可以直接归因**到排序。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddingComparison {
    /// 固定批（不排序）的 padding 统计。
    pub fixed: PaddingReport,

    /// 长度分桶（先排序）的 padding 统计。
    pub bucketed: PaddingReport,

    /// padding 系数的下降比例（0.0 ~ 1.0）。
    pub improvement: f64,
}

/// 对比两种切分方式的 padding 浪费。
pub fn compare_padding(lengths: &[usize], batch_size: usize) -> PaddingComparison {
    let batch_size = batch_size.max(1);

    // 固定批：按原始顺序切。
    let fixed_chunks: Vec<Vec<usize>> = lengths.chunks(batch_size).map(|c| c.to_vec()).collect();
    let mut fixed = padding_of_chunks(&fixed_chunks);
    fixed.batch_size = batch_size;

    // 分桶：先按长度排序再切。
    let bucketed = padding_of_sorted(lengths, batch_size);

    let improvement = if fixed.padding_factor > f64::EPSILON {
        (fixed.padding_factor - bucketed.padding_factor) / fixed.padding_factor
    } else {
        0.0
    };

    PaddingComparison {
        fixed,
        bucketed,
        improvement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_lengths_have_no_padding_waste() {
        let lengths = vec![10; 8];
        let report = padding_of_sorted(&lengths, 4);

        assert!((report.padding_factor - 1.0).abs() < 1e-9);
        assert!((report.waste_ratio - 0.0).abs() < 1e-9);
    }

    #[test]
    fn sorted_batches_have_less_waste_than_unsorted() {
        // 交替的长短 → 不排序时每批都混着长短。
        let lengths: Vec<usize> = (0..8).map(|i| if i % 2 == 0 { 100 } else { 10 }).collect();

        let comparison = compare_padding(&lengths, 4);

        assert!(
            comparison.bucketed.padding_factor < comparison.fixed.padding_factor,
            "分桶后 padding 系数应更低: {} vs {}",
            comparison.bucketed.padding_factor,
            comparison.fixed.padding_factor
        );
        assert!(comparison.improvement > 0.0);
    }

    #[test]
    fn padding_factor_is_never_below_one() {
        let lengths = vec![1, 100, 2, 200, 3];
        let report = padding_of_sorted(&lengths, 5);
        assert!(report.padding_factor >= 1.0);
    }

    #[test]
    fn single_item_batch_has_no_waste() {
        let lengths = vec![100, 10, 50];
        let report = padding_of_sorted(&lengths, 1);
        assert!((report.padding_factor - 1.0).abs() < 1e-9);
    }

    #[test]
    fn empty_input_produces_zero_report() {
        let report = padding_of_sorted(&[], 8);
        assert_eq!(report.actual_tokens, 0);
        assert_eq!(report.batch_count, 0);
        assert!((report.padding_factor - 1.0).abs() < 1e-9);
    }

    #[test]
    fn comparison_of_uniform_lengths_shows_no_improvement() {
        let lengths = vec![10; 8];
        let comparison = compare_padding(&lengths, 4);

        assert!((comparison.improvement - 0.0).abs() < 1e-9);
    }
}
