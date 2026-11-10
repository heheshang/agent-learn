//! ★ 分段剖析示例 —— 指出时间到底花在哪。
//!
//! # 为什么需要这个例子
//!
//! 第一版的「优化」只做「按天预聚合」（**减少扫描量**），实测 **1.00x** ——
//! **完全没有收益**。
//!
//! 原因只有分段剖析能看出来：
//!
//! | 环节 | 占比 |
//! |---|---|
//! | 过滤（全量扫描） | **1.5%** |
//! | 按天（字符串格式化） | 56.6% |
//! | 统计（`BTreeMap` 增量插入） | 25.1% |
//! | 去重（字符串 hash） | 16.6% |
//! | 亮点（clone + 排序） | 0.2% |
//!
//! **「过滤」只占 1.5%** —— 所以「减少扫描量」当然没用。
//! 真正贵的是**每条记录的派生计算**（合计 **98.3%**）。
//!
//! # 教训（可推广到全仓）
//!
//! > 「测不出差异」有两种原因 ——「**实现错了**」和「**优化点选错了**」。
//!
//! 必须靠**分段剖析**才能分清，否则会去改本来正确的代码。
//!
//! # 运行
//!
//! ```bash
//! cargo run --release -p agent-25 --example profile_segments
//! ```

use agent_25::bench::profile_naive;
use agent_25::model::Window;
use agent_25::pipeline::demo_activities;

fn main() {
    let activities = demo_activities(5000);
    let window = Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400);

    let rounds = 50;
    let profile = profile_naive(&activities, &window, rounds);

    println!("分段剖析：{} 条记录 × {} 轮\n", activities.len(), rounds);
    println!("{:<32} {:>10} {:>8}", "环节", "耗时(ms)", "占比");
    println!("{}", "-".repeat(54));

    for p in &profile {
        println!(
            "{:<32} {:>10.3} {:>7.1}%",
            p.segment,
            p.elapsed_ms,
            p.share * 100.0
        );
    }

    let filter = profile
        .iter()
        .find(|p| p.segment.contains("过滤"))
        .map(|p| p.share)
        .unwrap_or(0.0);
    let derived = 1.0 - filter;

    println!(
        "\n★ 过滤只占 {:.1}%，派生计算占 {:.1}%",
        filter * 100.0,
        derived * 100.0
    );
    println!(
        "★ 所以「减少扫描量」这个优化方向，最多只能拿到 {:.1}% 的收益。",
        filter * 100.0
    );
    println!("★ 正确的方向是「把派生计算搬到建索引时并整型化」。");
}
