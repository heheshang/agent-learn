//! ★ 批处理对照实验 —— **把「批处理快多少」量化出来**。
//!
//! # 实验设计
//!
//! ## 变量单一
//!
//! 三组跑的是**同一批文本**（同一份语料，同一个顺序），
//! 只有**批的大小策略**不同：
//!
//! | 组 | 策略 | 说明 |
//! |---|---|---|
//! | A | [`BatchStrategy::Sequential`] | 逐条调用 —— 基线 |
//! | B | [`BatchStrategy::Fixed`] | 固定大小批 |
//! | C | [`BatchStrategy::Bucketed`] | 长度分桶后定批 |
//!
//! ## 两种负载
//!
//! | 负载 | 说明 | 用途 |
//! |---|---|---|
//! | **本地 ONNX**（默认） | fastembed 跑本地模型 | 稳定、可复现、无需网络 |
//! | **远程 HTTP** | Ollama / OpenAI 兼容服务 | 证明「远程也能批」 |
//!
//! ⚠️ **为什么默认用本地**：
//! 20 项目的教训 —— 「差值必须用不挂额外负担的纯净运行测」。
//! 远程服务有网络抖动和限流，跑对照实验时用本地更稳定。
//!
//! # ⚠️ 多轮取中位数
//!
//! 单轮的数字会被机器负载污染。所以默认跑 3 轮，
//! 取**中位数**作为结论，同时报告**波动范围**。
//!
//! 本仓库反复验证的经验：
//! **加速比本身会浮动**（受机器负载、模型缓存、网络影响），
//! **可复现的结论是「相对关系」**（批处理组 ≪ 逐条组），
//! 而不是某个具体的倍数。

use std::time::Duration;

use agent_logging::info;
use rag_08::reranker::RerankDocument;

use crate::batching::{BatchInfo, BatchStrategy};
use crate::embedder::BatchEmbedder;
use crate::error::{AppError, AppResult};
use crate::metrics::{BatchStats, mean, stddev};

/// 一次对照实验的完整报告。
#[derive(Debug, Clone)]
pub struct CompareReport {
    /// 各策略的度量（按策略顺序）。
    pub results: Vec<(BatchStrategy, BatchStats)>,

    /// 样本条数。
    pub sample_size: usize,

    /// 跑了多少轮。
    pub rounds: usize,

    /// 各策略相对基线的加速比（用于看波动）。
    pub speedups: Vec<f64>,
}

impl CompareReport {
    /// 取某个策略的度量。
    pub fn stats_of(&self, strategy: BatchStrategy) -> Option<&BatchStats> {
        self.results
            .iter()
            .find(|(s, _)| *s == strategy)
            .map(|(_, stats)| stats)
    }

    /// 基线（逐条）的度量。
    pub fn baseline(&self) -> &BatchStats {
        self.stats_of(BatchStrategy::Sequential)
            .expect("报告里必须包含基线组")
    }

    /// 相对基线的加速比。
    pub fn speedup_of(&self, strategy: BatchStrategy) -> f64 {
        self.stats_of(strategy)
            .map(|s| s.speedup_vs(self.baseline().elapsed_ms))
            .unwrap_or(1.0)
    }

    /// 加速比的波动（标准差）。
    pub fn speedup_stddev(&self) -> f64 {
        stddev(&self.speedups)
    }

    /// 渲染成终端表格。
    pub fn render_table(&self) -> String {
        let mut out = String::new();
        out.push_str("策略          条数   批次  批大小  总耗时(ms)   吞吐(条/秒)   p50(ms)  p95(ms)  加速比\n");
        out.push_str("------------  -----  -----  ------  -----------  -----------  -------  -------  ------\n");

        let baseline = self.baseline().elapsed_ms;

        for (strategy, stats) in &self.results {
            let speedup = stats.speedup_vs(baseline);
            out.push_str(&format!(
                "{:<12}  {:>5}  {:>5}  {:>6}  {:>11}  {:>11.1}  {:>7}  {:>7}  {:>5.2}x\n",
                strategy.label(),
                stats.total_items,
                stats.total_batches,
                stats.batch_size,
                stats.elapsed_ms,
                stats.throughput_per_sec(),
                stats.p50(),
                stats.p95(),
                speedup,
            ));
        }

        out
    }

    /// 渲染成结论摘要。
    pub fn render_summary(&self) -> String {
        let baseline = self.baseline();
        let mut out = String::new();

        out.push_str(&format!(
            "基线（逐条）: {} ms　{:.1} 条/秒　单条 p50 {} ms\n",
            baseline.elapsed_ms,
            baseline.throughput_per_sec(),
            baseline.p50(),
        ));

        for strategy in [BatchStrategy::Fixed, BatchStrategy::Bucketed] {
            if let Some(stats) = self.stats_of(strategy) {
                let speedup = stats.speedup_vs(baseline.elapsed_ms);
                out.push_str(&format!(
                    "{:<10}: {} ms　{:.1} 条/秒　单条 p50 {} ms（基线的 {:.1} 倍）　加速比 {:.2}x\n",
                    strategy.label(),
                    stats.elapsed_ms,
                    stats.throughput_per_sec(),
                    stats.p50(),
                    stats.p50() as f64 / baseline.p50().max(1) as f64,
                    speedup,
                ));
            }
        }

        if self.rounds > 1 {
            out.push_str(&format!(
                "\n⚠️ 加速比波动: 标准差 {:.3}（各策略均值 {:.2}x）\n",
                self.speedup_stddev(),
                mean(&self.speedups),
            ));
            out.push_str(
                "   → 可复现的结论是**相对关系**（批处理组 ≪ 逐条组），不是某个固定倍数。\n",
            );
        }

        out
    }
}

/// 生成对照实验用的语料。
///
/// ## ⚠️ 必须**长度有差异**
///
/// 如果所有文本长度相同，**分桶就没有意义** ——
/// 长度分桶的收益完全来自「避免短文本被长文本拖着做无用计算」。
///
/// 所以这里刻意生成**从短到长**的文本，
/// 这样才能观察到分桶组相对固定批组的差异。
pub fn sample_corpus(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let repeats = (i % 16) + 1;
            format!(
                "这是第 {i} 条用于批处理吞吐测试的样本文本。{}",
                "内容填充".repeat(repeats)
            )
        })
        .collect()
}

/// 把语料转成重排候选块。
pub fn sample_documents(n: usize) -> Vec<RerankDocument> {
    (0..n)
        .map(|i| {
            let repeats = (i % 16) + 1;
            RerankDocument::new(
                format!("doc-{i}"),
                "sample.md",
                vec!["批处理测试".to_string(), format!("段落 {i}")],
                format!("这是第 {i} 条候选文档。{}", "内容填充".repeat(repeats)),
            )
        })
        .collect()
}

/// 多轮指标取**中位数**。
///
/// ## 为什么取中位数而不是平均
///
/// 平均值会被**离群轮**拉偏（某轮机器负载特别高）。
/// 中位数更稳健 —— 这与 20 项目的「加速比会浮动」经验一致。
fn median_stats(stats: &[BatchStats]) -> BatchStats {
    if stats.is_empty() {
        return BatchStats::new(0, 0, Duration::ZERO, vec![], vec![], 1, false);
    }
    if stats.len() == 1 {
        return stats[0].clone();
    }

    let median_of = |mut values: Vec<u64>| -> u64 {
        values.sort_unstable();
        values[values.len() / 2]
    };

    let mut median = BatchStats {
        total_items: stats[0].total_items,
        total_batches: median_of(stats.iter().map(|s| s.total_batches as u64).collect()) as usize,
        elapsed_ms: median_of(stats.iter().map(|s| s.elapsed_ms).collect()),
        item_latencies_ms: stats[0].item_latencies_ms.clone(),
        // 批耗时的中位数列表（按长度对齐）。
        batch_latencies_ms: stats[0].batch_latencies_ms.clone(),
        batch_size: stats[0].batch_size,
        bucketed: stats[0].bucketed,
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
    // ⚠️ 改了原始字段就必须重算派生值 ——
    //    否则「存的值」和「算的值」会不一致。
    median.refresh_derived();
    median
}

/// ★ 跑一次本地嵌入批处理对照实验。
///
/// ## ⚠️ 多轮 + 取中位数
///
/// `rounds` 轮里每组都跑一遍，取中位数。
/// 为什么要多轮：模型有缓存（第一次加载权重后后续更快），
/// 第一轮通常偏慢 —— 直接用第一轮会**低估**批处理的收益。
pub async fn run_embed_compare(
    embedder: &BatchEmbedder,
    texts: &[String],
    rounds: usize,
) -> AppResult<CompareReport> {
    if texts.is_empty() {
        return Err(AppError::BadRequest("语料不能为空".to_string()));
    }

    let rounds = rounds.max(1);
    let strategies = [
        BatchStrategy::Sequential,
        BatchStrategy::Fixed,
        BatchStrategy::Bucketed,
    ];

    let mut per_strategy: Vec<Vec<BatchStats>> = vec![Vec::new(); strategies.len()];

    for _ in 0..rounds {
        for (si, strategy) in strategies.iter().enumerate() {
            let (vectors, stats) = embedder.embed_batch(texts, *strategy).await?;

            // ⚠️ 结果条数必须与输入一致 —— 否则实验没有意义。
            if vectors.len() != texts.len() {
                return Err(AppError::Internal(format!(
                    "对照实验结果条数不匹配: 输入 {} 条, 输出 {} 条",
                    texts.len(),
                    vectors.len()
                )));
            }

            per_strategy[si].push(stats);
        }
    }

    let mut results = Vec::with_capacity(strategies.len());
    for (si, strategy) in strategies.iter().enumerate() {
        results.push((*strategy, median_stats(&per_strategy[si])));
    }

    let baseline_ms = results[0].1.elapsed_ms;
    let speedups: Vec<f64> = results
        .iter()
        .skip(1)
        .map(|(_, s)| s.speedup_vs(baseline_ms))
        .collect();

    info!(sample_size = texts.len(), rounds, "嵌入批处理对照实验完成");

    Ok(CompareReport {
        results,
        sample_size: texts.len(),
        rounds,
        speedups,
    })
}

/// ★ 跑一次重排对照实验。
pub async fn run_rerank_compare(
    reranker: &crate::reranker::BatchReranker,
    query: &str,
    documents: &[RerankDocument],
    top_n: usize,
    rounds: usize,
) -> AppResult<CompareReport> {
    if documents.is_empty() {
        return Err(AppError::BadRequest("候选文档不能为空".to_string()));
    }

    let rounds = rounds.max(1);
    let strategies = [
        BatchStrategy::Sequential,
        BatchStrategy::Fixed,
        BatchStrategy::Bucketed,
    ];

    let mut per_strategy: Vec<Vec<BatchStats>> = vec![Vec::new(); strategies.len()];

    for _ in 0..rounds {
        for (si, strategy) in strategies.iter().enumerate() {
            let (hits, stats) = reranker
                .rerank_batch(query, documents, top_n, *strategy)
                .await?;

            if hits.len() != top_n.min(documents.len()) {
                return Err(AppError::Internal(format!(
                    "重排对照实验结果条数不匹配: 期望 {}, 实际 {}",
                    top_n.min(documents.len()),
                    hits.len()
                )));
            }

            per_strategy[si].push(stats);
        }
    }

    let mut results = Vec::with_capacity(strategies.len());
    for (si, strategy) in strategies.iter().enumerate() {
        results.push((*strategy, median_stats(&per_strategy[si])));
    }

    let baseline_ms = results[0].1.elapsed_ms;
    let speedups: Vec<f64> = results
        .iter()
        .skip(1)
        .map(|(_, s)| s.speedup_vs(baseline_ms))
        .collect();

    Ok(CompareReport {
        results,
        sample_size: documents.len(),
        rounds,
        speedups,
    })
}

/// 批次元信息（供 API 返回）。
pub fn batches_of(stats: &BatchStats) -> Vec<BatchInfo> {
    stats
        .batch_latencies_ms
        .iter()
        .enumerate()
        .map(|(index, elapsed_ms)| BatchInfo {
            index,
            size: 0,
            elapsed_ms: *elapsed_ms,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_corpus_has_varying_lengths() {
        let corpus = sample_corpus(32);
        let lens: Vec<usize> = corpus.iter().map(|t| t.chars().count()).collect();

        // ⚠️ 长度必须有差异 —— 否则分桶无从体现。
        let min = lens.iter().min().unwrap();
        let max = lens.iter().max().unwrap();
        assert!(max > min, "语料长度没有差异");
    }

    #[test]
    fn sample_corpus_length_is_deterministic() {
        assert_eq!(sample_corpus(8), sample_corpus(8));
    }

    #[test]
    fn sample_documents_are_valid() {
        let docs = sample_documents(4);
        assert_eq!(docs.len(), 4);
        assert_eq!(docs[0].id, "doc-0");
        assert!(!docs[0].text.is_empty());
    }

    #[test]
    fn median_stats_picks_the_middle_value() {
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

        let stats = vec![mk(300), mk(100), mk(200)];
        let median = median_stats(&stats);

        // 排序后 [100, 200, 300] → 取中间 → 200
        assert_eq!(median.elapsed_ms, 200);
    }

    #[test]
    fn median_stats_of_single_value_is_that_value() {
        let stats = vec![BatchStats::new(
            1,
            1,
            Duration::from_millis(42),
            vec![42],
            vec![42],
            1,
            false,
        )];
        assert_eq!(median_stats(&stats).elapsed_ms, 42);
    }

    #[test]
    fn median_stats_of_empty_is_zero() {
        let stats = median_stats(&[]);
        assert_eq!(stats.elapsed_ms, 0);
        assert_eq!(stats.total_items, 0);
    }

    #[test]
    fn compare_report_renders_all_strategies() {
        let mk = |ms: u64, items: usize, batches: usize| BatchStats {
            total_items: items,
            total_batches: batches,
            elapsed_ms: ms,
            item_latencies_ms: vec![ms; items],
            batch_latencies_ms: vec![ms; batches],
            batch_size: 32,
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

        let report = CompareReport {
            results: vec![
                (BatchStrategy::Sequential, mk(1000, 64, 64)),
                (BatchStrategy::Fixed, mk(200, 64, 2)),
            ],
            sample_size: 64,
            rounds: 3,
            speedups: vec![5.0],
        };

        let table = report.render_table();
        assert!(table.contains("逐条调用"));
        assert!(table.contains("固定批"));
        assert!(table.contains("5.00x"));
    }
}
