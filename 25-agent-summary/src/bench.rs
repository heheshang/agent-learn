//! **对照实验** —— 三档策略 + 并发扫描。
//!
//! # ★ 实验设计的三个硬要求
//!
//! ## ① 变量单一
//!
//! 同一批数据、同样的顺序、同样的请求数，只变**一个**变量。
//!
//! ## ② 两档必须产出**完全一致**的结果
//!
//! 否则测的是「不同的功能」而不是「优化」。
//! 由 `naive_and_preagg_produce_identical_digests` 锁住。
//!
//! ## ③ 先 warm-up 再计时
//!
//! 优化是**摊还**的：第一次调用要建索引 + 填缓存，比 naive 更慢。
//! 不 warm-up 就会得出「优化反而更慢」的错误结论。

use std::time::Instant;

use crate::aggregate::{Aggregator, Strategy};
use crate::loadtest::{BenchFn, LoadConfig, run_inprocess, run_inprocess_named};
use crate::metrics::{Comparison, LoadReport};
use crate::model::{Activity, Window};

/// 一次实验的结果。
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    /// 实验名。
    pub name: String,

    /// 变体名。
    pub variant: String,

    /// 压测报告。
    pub report: LoadReport,
}

/// 三档对照实验。
///
/// ## ⚠️ 必须**每档独立建聚合器**
///
/// 因为缓存是**跨调用**的 —— 复用同一个聚合器会让第二档直接命中第一档的缓存，
/// 差值就失真了。
pub async fn run_three_tier(
    activities: &[Activity],
    window: &Window,
    config: &LoadConfig,
) -> Vec<ExperimentResult> {
    run_three_tier_with_windows(activities, &[*window], config).await
}

/// 三档对照实验（**多个窗口**）。
///
/// ## ⚠️ 必须用**多个不同的窗口**
///
/// 如果 500 次请求全用**同一个窗口**，缓存档会 100% 命中，
/// 缓存收益被放大成几百倍 —— 那个数字**漂亮但不可信**。
///
/// 真实场景里查询窗口是**发散的**（不同的周、不同的部门）。
/// 所以这里让请求在**若干个窗口**间轮转，
/// 缓存命中率回到一个**现实的**比例。
pub async fn run_three_tier_with_windows(
    activities: &[Activity],
    windows: &[Window],
    config: &LoadConfig,
) -> Vec<ExperimentResult> {
    let mut results = Vec::new();

    for strategy in Strategy::all() {
        // ★ 每档独立建聚合器。
        let mut aggregator = Aggregator::new(activities.to_vec(), *strategy);

        // ★ warm-up —— 优化是摊还的，不 warm-up 会低估。
        for w in windows {
            let _ = aggregator.aggregate(w);
        }

        let agg = std::sync::Arc::new(std::sync::Mutex::new(aggregator));
        let wins: Vec<Window> = windows.to_vec();
        let f: BenchFn = std::sync::Arc::new(move |i| {
            let mut guard = agg.lock().expect("聚合器锁");
            // ★ 请求在窗口间轮转 —— 模拟真实负载。
            let w = wins[i % wins.len()];
            let _ = guard.aggregate(&w);
            true
        });

        let report = run_inprocess_named(strategy.name(), f, config).await;
        results.push(ExperimentResult {
            name: "三档策略对照".to_string(),
            variant: strategy.name().to_string(),
            report,
        });
    }

    results
}

/// ★ 缓存收益**完全由命中率决定** —— 扫不同「不同键的数量」。
///
/// ## ⚠️ 这才是缓存收益的**正确归因方式**
///
/// 一个容易犯的错：拿「同一个窗口请求 500 次」去测缓存，
/// 得出「缓存快 258 倍」—— 那个数字**漂亮但不可信**，
/// 因为命中率是 100%（warm-up 把所有键都填了）。
///
/// 缓存收益的公式其实很简单：
///
/// ```text
/// 加速比 ≈ 1 / (未命中比例 × 单次计算成本 + 命中比例 × 缓存查找成本)
/// ```
///
/// 所以**命中率是唯一的自变量**。本实验把它显式扫出来：
///
/// | 不同键数 | 命中率 | 加速比 |
/// |---|---|---|
/// | 1 | ~100% | 最大 |
/// | 16 | ~97% | 略小 |
/// | 256 | ~37% | 显著变小 |
/// | 5000 | ~0% | ≈ 1（缓存无用）|
///
/// **结论**：缓存不是「万能加速」——
/// 它的收益**完全由工作负载的键分布决定**。
/// 键分布发散的系统上，缓存带来的可能只有个位数百分比。
pub async fn run_cache_hit_sweep(
    activities: &[Activity],
    config: &LoadConfig,
    distinct_keys: &[usize],
) -> Vec<ExperimentResult> {
    let mut results = Vec::new();
    let base_ts = 1_700_000_000i64;

    for &keys in distinct_keys {
        // 造 `keys` 个不同的窗口。
        // ⚠️ 窗口必须**真的互不相同** —— 否则 `keys` 再大也只有 28 个键。
        //    用「起点不同 + 跨度不同」保证唯一。
        let windows: Vec<Window> = (0..keys)
            .map(|i| {
                let start = base_ts + (i as i64) * 3_600;
                let span = 6 + (i as i64 % 7);
                Window::new(start, start + span * 86_400)
            })
            .collect();

        // ── 缓存档 ──
        //
        // ⚠️ **不 warm-up** —— warm-up 会把所有键都填上，
        //    命中率就永远是 100%，扫不出「命中率 → 收益」的关系。
        //
        //    这里让第一个请求就是「冷的」，之后才逐渐变热。
        let agg = std::sync::Arc::new(std::sync::Mutex::new(Aggregator::new(
            activities.to_vec(),
            Strategy::Optimized,
        )));
        let wins = windows.clone();
        let f: BenchFn = std::sync::Arc::new({
            let agg = std::sync::Arc::clone(&agg);
            move |i| {
                let mut guard = agg.lock().expect("聚合器锁");
                let w = wins[i % wins.len()];
                let _ = guard.aggregate(&w);
                true
            }
        });
        let cached = run_inprocess_named(&format!("optimized/{keys}键"), f, config).await;

        // ── 无缓存档（同一批窗口，变量单一）──
        let agg2 = std::sync::Arc::new(std::sync::Mutex::new(Aggregator::new(
            activities.to_vec(),
            Strategy::PreAggregated,
        )));
        let wins2 = windows.clone();
        let f: BenchFn = std::sync::Arc::new({
            let agg2 = std::sync::Arc::clone(&agg2);
            move |i| {
                let mut guard = agg2.lock().expect("聚合器锁");
                let w = wins2[i % wins2.len()];
                let _ = guard.aggregate(&w);
                true
            }
        });
        let uncached = run_inprocess_named(&format!("pre-agg/{keys}键"), f, config).await;

        // ★ **实测**命中率（不是估算）—— 从聚合器里读。
        let hit_rate = agg.lock().expect("聚合器锁").cache_hit_rate();

        let cached_qps = cached.qps;
        let uncached_qps = uncached.qps;
        results.push(ExperimentResult {
            name: format!("缓存命中率扫描（{keys} 个不同键）"),
            variant: format!("{keys}键"),
            report: cached,
        });
        results.push(ExperimentResult {
            name: format!("缓存命中率扫描（{keys} 个不同键）"),
            variant: format!("{keys}键-无缓存"),
            report: uncached,
        });

        agent_logging::info!(
            "{} 个不同键 → 实测命中率 {:.1}%，缓存档 QPS {:.0}，无缓存档 QPS {:.0}（{:.2}x）",
            keys,
            hit_rate * 100.0,
            cached_qps,
            uncached_qps,
            cached_qps / uncached_qps.max(f64::EPSILON)
        );
    }

    results
}

/// ★ 并发度扫描。
///
/// ## ⚠️ 实验前提：必须有**有限的并发资源**
///
/// 只有「模拟服务时间」没有「worker 上限」时，
/// `buffer_unordered(N)` 让 N 个任务**同时** sleep，谁也没排队 ——
/// 测出来的是一条平线。
pub async fn run_concurrency_sweep(
    activities: &[Activity],
    window: &Window,
    strategy: Strategy,
    levels: &[usize],
    requests: usize,
) -> Vec<ExperimentResult> {
    let mut results = Vec::new();

    for level in levels {
        let mut aggregator = Aggregator::new(activities.to_vec(), strategy);
        let _ = aggregator.aggregate(window);

        let agg = std::sync::Arc::new(std::sync::Mutex::new(aggregator));
        let win = *window;
        let f: BenchFn = std::sync::Arc::new(move |_| {
            let mut guard = agg.lock().expect("聚合器锁");
            let _ = guard.aggregate(&win);
            true
        });

        let config = LoadConfig {
            requests,
            concurrency: *level,
            service_time_ms: 50,
            server_workers: 8,
        };

        let report = run_inprocess(f, &config).await;
        results.push(ExperimentResult {
            name: format!("并发度扫描（并发 {level}）"),
            variant: format!("c{level}"),
            report,
        });
    }

    results
}

/// ★ 分段剖析 —— 把「时间花在哪」拆开。
///
/// ## 为什么需要它
///
/// 第一版优化只做「按天预聚合」（减少扫描量），实测 **1.00x**。
/// 分段剖析显示「过滤」只占 **1%**，真正贵的是**每条记录的派生计算**（占 **89%**）。
///
/// **教训**：「测不出差异」有两种原因 ——「实现错了」和「**优化点选错了**」。
/// 必须靠分段剖析才能分清。
#[derive(Debug, Clone)]
pub struct SegmentProfile {
    /// 环节名。
    pub segment: String,

    /// 耗时（毫秒）。
    pub elapsed_ms: f64,

    /// 占比（0.0 ~ 1.0）。
    pub share: f64,
}

/// 分段剖析 —— 对 `naive` 路径。
pub fn profile_naive(
    activities: &[Activity],
    window: &Window,
    rounds: usize,
) -> Vec<SegmentProfile> {
    let mut agg = Aggregator::new(activities.to_vec(), Strategy::Naive);

    // warm-up。
    let _ = agg.aggregate(window);

    let mut filter_ms = 0.0;
    let mut dedup_ms = 0.0;
    let mut stats_ms = 0.0;
    let mut day_ms = 0.0;
    let mut highlight_ms = 0.0;

    for _ in 0..rounds {
        let t = Instant::now();
        let filtered: Vec<&Activity> = agg
            .activities()
            .iter()
            .filter(|a| window.contains(a.ts))
            .collect();
        filter_ms += t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let deduped: Vec<&Activity> = filtered
            .iter()
            .copied()
            .filter(|a| seen.insert(a.id.as_str()))
            .collect();
        dedup_ms += t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let mut by_kind: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut by_scope: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for a in &deduped {
            *by_kind.entry(a.kind.clone()).or_insert(0) += 1;
            *by_scope.entry(a.scope.clone()).or_insert(0) += 1;
        }
        stats_ms += t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let mut by_day: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for a in &deduped {
            *by_day.entry(crate::model::day_label(a.ts)).or_insert(0) += 1;
        }
        day_ms += t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let mut highlights: Vec<crate::model::Highlight> = deduped
            .iter()
            .filter(|a| a.weight > 0 && !a.risk)
            .take(5)
            .map(|a| crate::model::Highlight {
                id: a.id.clone(),
                title: a.title.clone(),
                scope: a.scope.clone(),
                author: a.author.clone(),
                kind: a.kind.clone(),
                weight: a.weight,
            })
            .collect();
        highlights.sort_by(|a, b| b.weight.cmp(&a.weight).then(a.id.cmp(&b.id)));
        highlight_ms += t.elapsed().as_secs_f64() * 1000.0;
    }

    let total = filter_ms + dedup_ms + stats_ms + day_ms + highlight_ms;
    let pct = |ms: f64| if total > 0.0 { ms / total } else { 0.0 };

    let mut out = vec![
        SegmentProfile {
            segment: "过滤（全量扫描）".into(),
            elapsed_ms: filter_ms,
            share: pct(filter_ms),
        },
        SegmentProfile {
            segment: "去重（字符串 hash）".into(),
            elapsed_ms: dedup_ms,
            share: pct(dedup_ms),
        },
        SegmentProfile {
            segment: "统计（BTreeMap 增量插入）".into(),
            elapsed_ms: stats_ms,
            share: pct(stats_ms),
        },
        SegmentProfile {
            segment: "按天（字符串格式化）".into(),
            elapsed_ms: day_ms,
            share: pct(day_ms),
        },
        SegmentProfile {
            segment: "亮点（clone + 排序）".into(),
            elapsed_ms: highlight_ms,
            share: pct(highlight_ms),
        },
    ];
    out.sort_by(|a, b| {
        b.share
            .partial_cmp(&a.share)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// 从三档结果里算出对照。
pub fn compare_three_tier(results: &[ExperimentResult]) -> Vec<Comparison> {
    if results.len() < 2 {
        return Vec::new();
    }
    let baseline = &results[0].report;
    results[1..]
        .iter()
        .map(|r| Comparison::new(baseline, &r.report))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::demo_activities;

    fn window() -> Window {
        Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400)
    }

    #[tokio::test]
    async fn three_tier_produces_three_results() {
        let acts = demo_activities(1000);
        let cfg = LoadConfig {
            requests: 100,
            concurrency: 4,
            service_time_ms: 0,
            server_workers: 4,
        };
        let results = run_three_tier(&acts, &window(), &cfg).await;
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].variant, "naive");
        assert_eq!(results[1].variant, "pre-agg");
        assert_eq!(results[2].variant, "optimized");
    }

    #[tokio::test]
    async fn optimized_is_faster_than_naive() {
        let acts = demo_activities(3000);
        let cfg = LoadConfig {
            requests: 200,
            concurrency: 4,
            service_time_ms: 0,
            server_workers: 4,
        };
        let results = run_three_tier(&acts, &window(), &cfg).await;
        let naive = &results[0].report;
        let optimized = &results[2].report;
        assert!(
            optimized.qps >= naive.qps,
            "优化后不该更慢：naive={} optimized={}",
            naive.qps,
            optimized.qps
        );
    }

    #[test]
    fn profile_sums_to_one() {
        let acts = demo_activities(1000);
        let profile = profile_naive(&acts, &window(), 5);
        let total: f64 = profile.iter().map(|p| p.share).sum();
        assert!((total - 1.0).abs() < 1e-6, "占比之和应该是 1，实际 {total}");
    }

    #[test]
    fn profile_reveals_derived_computation_as_the_bottleneck() {
        let acts = demo_activities(3000);
        let profile = profile_naive(&acts, &window(), 5);
        // 「过滤」应该只占很小一部分 —— 这正是第一版优化只值 1.00x 的原因。
        let filter = profile.iter().find(|p| p.segment.contains("过滤")).unwrap();
        assert!(
            filter.share < 0.2,
            "过滤不应该占大头，实际 {:.1}%",
            filter.share * 100.0
        );
    }
}
