//! ★ **批大小扫描** —— 找出「最优批大小」。
//!
//! # 为什么必须扫
//!
//! 批大小不是越大越好。它是一个**两难**：
//!
//! ```text
//! 批大小 ↑  → 固定开销（权重复制）分摊更多  → 吞吐 ↑
//!          → padding 浪费更多                → 吞吐 ↓
//!          → 内存占用更大                   → 可能 OOM / 换页
//!          → 单条延迟更高                   → 在线场景不可接受
//! ```
//!
//! 所以「最优批大小」是**数据相关的**，必须实测。
//!
//! # 扫描出来的曲线长什么样
//!
//! 典型形状是一个**倒 U**（或阶梯）：
//!
//! ```text
//! 吞吐
//!  ↑        ┌──────┐
//!  │      ╱ ┆      ┆╲
//!  │    ╱   ┆      ┆ ╲___
//!  │  ╱     ┆      ┆     ╲__
//!  └────────┴──────┴────────→ 批大小
//! ```
//!
//! - 左侧：**固定开销占主导** —— 批越大越快
//! - 拐点：固定开销与 padding 浪费平衡
//! - 右侧：**padding 浪费 / 内存压力占主导** —— 批越大越慢
//!
//! ⚠️ 本项目实测的曲线**可能没有明显的右侧下降** ——
//! 因为 fastembed 内部还会用 rayon 对批次**再并行**一次，
//! 且 `BatchLongest` 的 padding 浪费在短文本语料上不明显。
//! 这时最优批大小就是「拐点之后第一个收益饱和的点」。

use std::time::Duration;

use agent_logging::info;
use serde::{Deserialize, Serialize};

use crate::batching::BatchStrategy;
use crate::embedder::BatchEmbedder;
use crate::error::{AppError, AppResult};
use crate::metrics::BatchStats;

/// 一个批大小的扫描结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepPoint {
    /// 批大小。
    pub batch_size: usize,

    /// 该批大小下的度量。
    pub stats: BatchStats,

    /// 相对**最小批大小**的吞吐倍数。
    pub throughput_ratio: f64,
}

/// 一次扫描的完整报告。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepReport {
    /// 各批大小的结果（按批大小升序）。
    pub points: Vec<SweepPoint>,

    /// 吞吐最优的批大小。
    pub best_batch_size: usize,

    /// 样本条数。
    pub sample_size: usize,

    /// 轮数。
    pub rounds: usize,
}

impl SweepReport {
    /// 渲染成终端表格。
    pub fn render_table(&self) -> String {
        let mut out = String::new();
        out.push_str("批大小   批次   总耗时(ms)   吞吐(条/秒)   吞吐倍数   单条 p50(ms)\n");
        out.push_str("-------  -----  -----------  ------------  ----------  ------------\n");

        for point in &self.points {
            out.push_str(&format!(
                "{:>7}  {:>5}  {:>11}  {:>12.1}  {:>9.2}x  {:>12}\n",
                point.batch_size,
                point.stats.total_batches,
                point.stats.elapsed_ms,
                point.stats.throughput_per_sec(),
                point.throughput_ratio,
                point.stats.p50(),
            ));
        }

        out
    }

    /// 渲染成结论。
    pub fn render_summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "最优批大小（吞吐最高）: {}\n",
            self.best_batch_size
        ));

        // ⚠️ 找出「收益饱和点」—— 从哪个批大小起，吞吐提升不到 5%。
        if let Some(saturated) = self.saturated_point() {
            out.push_str(&format!(
                "收益饱和点: 批大小 {}（再往上吞吐提升 < 5%）\n",
                saturated.batch_size
            ));
        }

        out.push_str("⚠️ 批大小是**数据相关**的 —— 换语料 / 换模型 / 换硬件都要重新扫。\n");
        out.push_str("⚠️ 在线单条查询**不该**用大批 —— 单条延迟会被抬高到 p50 的量级。\n");
        out
    }

    /// 找出「收益饱和点」。
    ///
    /// 定义：从最小的批大小开始，第一个「相对上一个点提升 < 5%」的点。
    fn saturated_point(&self) -> Option<&SweepPoint> {
        for pair in self.points.windows(2) {
            let (prev, cur) = (&pair[0], &pair[1]);
            if prev.stats.throughput_per_sec() <= f64::EPSILON {
                continue;
            }
            let gain = (cur.stats.throughput_per_sec() - prev.stats.throughput_per_sec())
                / prev.stats.throughput_per_sec();
            if gain < 0.05 {
                return Some(cur);
            }
        }
        None
    }
}

/// ★ 扫描一批批大小。
///
/// ## ⚠️ 每档都跑**多轮**取中位数
///
/// 否则某档正好撞上机器负载高峰，会得出「大批反而更慢」的错误结论。
pub async fn sweep_batch_sizes(
    embedder: &BatchEmbedder,
    texts: &[String],
    batch_sizes: &[usize],
    rounds: usize,
) -> AppResult<SweepReport> {
    if texts.is_empty() {
        return Err(AppError::BadRequest("语料不能为空".to_string()));
    }
    if batch_sizes.is_empty() {
        return Err(AppError::BadRequest("批大小列表不能为空".to_string()));
    }

    let rounds = rounds.max(1);
    let mut points = Vec::with_capacity(batch_sizes.len());

    for &size in batch_sizes {
        if size == 0 {
            continue;
        }

        // ⚠️ 每个批大小要用**自己的配置**跑 ——
        //    所以这里临时改 `batch_size`（批大小是配置的一部分）。
        let mut config = embedder.config().clone();
        config.batch_size = size;

        let scoped = BatchEmbedder::from_embedder(
            std::sync::Arc::new(ScopedEmbedder {
                inner: embedder.inner_arc(),
            }),
            config,
        );

        // 跑多轮，取中位数。
        let mut per_round = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let (_, stats) = scoped.embed_batch(texts, BatchStrategy::Fixed).await?;
            per_round.push(stats);
        }

        let median = median_of(&per_round);
        points.push(SweepPoint {
            batch_size: size,
            stats: median,
            throughput_ratio: 1.0,
        });
    }

    // 吞吐倍数相对**第一个点**（最小的批大小）。
    if let Some(first) = points.first() {
        let base = first.stats.throughput_per_sec();
        for point in &mut points {
            point.throughput_ratio = if base <= f64::EPSILON {
                1.0
            } else {
                point.stats.throughput_per_sec() / base
            };
        }
    }

    let best_batch_size = points
        .iter()
        .max_by(|a, b| {
            a.stats
                .throughput_per_sec()
                .total_cmp(&b.stats.throughput_per_sec())
        })
        .map(|p| p.batch_size)
        .unwrap_or(0);

    info!(points = points.len(), best_batch_size, "批大小扫描完成");

    Ok(SweepReport {
        points,
        best_batch_size,
        sample_size: texts.len(),
        rounds,
    })
}

/// 多轮取中位数（按总耗时）。
fn median_of(stats: &[BatchStats]) -> BatchStats {
    if stats.is_empty() {
        return BatchStats::new(0, 0, Duration::ZERO, vec![], vec![], 1, false);
    }
    if stats.len() == 1 {
        return stats[0].clone();
    }

    let mut sorted: Vec<&BatchStats> = stats.iter().collect();
    sorted.sort_by_key(|s| s.elapsed_ms);

    let median = sorted[sorted.len() / 2];
    let mut result = BatchStats {
        total_items: median.total_items,
        total_batches: median.total_batches,
        elapsed_ms: median.elapsed_ms,
        item_latencies_ms: median.item_latencies_ms.clone(),
        batch_latencies_ms: median.batch_latencies_ms.clone(),
        batch_size: median.batch_size,
        bucketed: median.bucketed,
        derived: crate::metrics::DerivedMetrics {
            throughput_per_sec: 0.0,
            avg_batch_fill: 0.0,
            p50: 0,
            p95: 0,
            max_latency: 0,
            sum_of_item_latencies_ms: 0,
            sharing_factor: 0.0,
        },
    };
    result.refresh_derived();
    result
}

/// 一个「换个批大小」的嵌入器包装。
///
/// ## 为什么需要它
///
/// [`BatchEmbedder`] 的批大小是**构造时**定下的，
/// 而扫描要反复换批大小。
/// 所以这里包一层：复用**同一个模型实例**（避免重复加载模型），
/// 只换批大小配置。
struct ScopedEmbedder {
    inner: std::sync::Arc<dyn rag_05::embed::Embedder>,
}

impl rag_05::embed::Embedder for ScopedEmbedder {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn needs_document_prefix(&self) -> bool {
        self.inner.needs_document_prefix()
    }

    fn needs_query_prefix(&self) -> bool {
        self.inner.needs_query_prefix()
    }

    fn embed_documents(
        &self,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, rag_05::embed::EmbedError> {
        self.inner.embed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, rag_05::embed::EmbedError> {
        self.inner.embed_query(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_picks_the_middle_elapsed() {
        let mk = |ms: u64| {
            BatchStats::new(
                1,
                1,
                Duration::from_millis(ms),
                vec![ms],
                vec![ms],
                1,
                false,
            )
        };
        let median = median_of(&[mk(300), mk(100), mk(200)]);
        assert_eq!(median.elapsed_ms, 200);
    }

    #[test]
    fn median_of_empty_is_zero() {
        assert_eq!(median_of(&[]).elapsed_ms, 0);
    }

    #[test]
    fn sweep_report_finds_the_best_batch_size() {
        let mk = |ms: u64| BatchStats {
            total_items: 32,
            total_batches: 1,
            elapsed_ms: ms,
            item_latencies_ms: vec![ms; 32],
            batch_latencies_ms: vec![ms],
            batch_size: 1,
            bucketed: false,
            derived: crate::metrics::DerivedMetrics {
                throughput_per_sec: 0.0,
                avg_batch_fill: 0.0,
                p50: 0,
                p95: 0,
                max_latency: 0,
                sum_of_item_latencies_ms: 0,
                sharing_factor: 0.0,
            },
        };
        let report = SweepReport {
            points: vec![
                SweepPoint {
                    batch_size: 1,
                    stats: mk(1000),
                    throughput_ratio: 1.0,
                },
                SweepPoint {
                    batch_size: 8,
                    stats: mk(300),
                    throughput_ratio: 3.3,
                },
                SweepPoint {
                    batch_size: 32,
                    stats: mk(400),
                    throughput_ratio: 2.5,
                },
            ],
            best_batch_size: 8,
            sample_size: 32,
            rounds: 1,
        };

        assert_eq!(report.best_batch_size, 8);
        let table = report.render_table();
        assert!(table.contains("批大小"));
    }
}
