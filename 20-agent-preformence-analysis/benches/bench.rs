//! criterion 基准测试。
//!
//! ## criterion 对应 cProfile 的哪个角色
//!
//! 都是「**确定性重复测量**」，但关注点不同：
//!
//! | | cProfile | criterion |
//! |---|---|---|
//! | 回答的问题 | 「哪个函数慢」 | 「A 比 B 快多少，**这个差值显著吗**」 |
//! | 输出 | 逐函数耗时表 | 统计量（均值/中位数/置信区间）+ 显著性判定 |
//! | 侵入性 | 挂 `sys.setprofile`，**改变时序** | 反复跑，**取统计量** |
//!
//! criterion 最大的价值是**显著性判定**：
//! 「快 3%」在噪声面前毫无意义，
//! criterion 会告诉你这个差值**是否超过噪声带**。
//!
//! ⚠️ 这正是本仓库 09-rag-eval 里用过的思路：
//!    judge 自身抖动就有 ±0.05，小于 5% 的差值不算收益。

use criterion::{Criterion, criterion_group, criterion_main};

use agent_20::workload::RustWorkload;

/// 基准：单次检索。
///
/// ⚠️ 注意这里**每次都重建 workload** ——
///    BM25 索引建一次要花几毫秒，如果复用索引，
///    基准测的就只剩「查询打分」而不是「真实检索链路」。
fn bench_single_search(c: &mut Criterion) {
    let workload = RustWorkload::new();
    c.bench_function("bm25_single_search", |b| {
        b.iter(|| {
            for query in agent_20::workload::QUERIES {
                let _ = workload.run_once(query);
            }
        })
    });
}

/// 基准：全量检索工作负载。
fn bench_full_workload(c: &mut Criterion) {
    let workload = RustWorkload::new();
    c.bench_function("bm25_full_workload", |b| b.iter(|| workload.run(1)));
}

/// 基准：索引构建。
///
/// 单独测是因为它揭示了一个重要事实：
/// 「建索引贵、查询便宜」是检索系统的典型形状，
/// 所以索引应该**离线构建**而不是每次请求都建。
fn bench_index_build(c: &mut Criterion) {
    c.bench_function("bm25_index_build", |b| b.iter(RustWorkload::new));
}

criterion_group!(
    benches,
    bench_single_search,
    bench_full_workload,
    bench_index_build
);
criterion_main!(benches);
