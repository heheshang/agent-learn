//! 20-agent-preformence-analysis —— 性能瓶颈分析。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- bench             # ★ 只测时（不采样，最快）
//! cargo run -- pprof             # ★ 采样剖析，产出火焰图
//! cargo run -- compare           # ★ Rust 侧耗时对比（多次取中位数）
//! cargo run -- tools             # 打印 Python 侧三工具的对应关系
//! cargo run -- all               # 全部
//! ```

use std::time::Instant;

use agent_logging::{info, init};
use anyhow::Result;

use agent_20::config::AppConfig;
use agent_20::profile::RustProfiler;
use agent_20::report::Report;
use agent_20::workload::{QUERIES, RustWorkload, TIMING_OUTER_REPEATS};

/// 跑几轮取**中位数**。
///
/// ⚠️ 为什么取中位数而不是平均值：
/// 单次运行的耗时噪声很大（机器负载、CPU 频率调节、缓存状态）。
/// 平均值会被偶发的长尾拉高，中位数更接近「典型情况」。
///
/// 这与本仓库 09-rag-eval 的经验一致：
/// judge 自身抖动就有 ±0.05，小于噪声带的差值不算收益。
fn median(mut values: Vec<u64>) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

/// 计时 demo：跑 N 轮，取中位数。
fn bench_demo(config: &AppConfig) -> Result<()> {
    let workload = RustWorkload::new();
    let mut samples: Vec<u64> = Vec::new();

    info!(
        corpus = workload.corpus_size(),
        repeats = config.repeats,
        "开始计时"
    );

    for _ in 0..config.repeats {
        let start = Instant::now();
        // ⚠️ 外层重复把耗时放大到毫秒级 ——
        //    否则单轮不到 1ms，取整后全是 0。
        let outcome = workload.run(TIMING_OUTER_REPEATS);
        samples.push(start.elapsed().as_micros() as u64);
        info!(
            elapsed_us = outcome.elapsed_us,
            hits = outcome.hits.len(),
            "一轮完成"
        );
    }

    let med = median(samples.clone());
    println!("轮次耗时（微秒）: {samples:?}");
    println!("中位数: {med} us");
    println!("语料规模: {} 块", workload.corpus_size());
    println!(
        "单轮检索: {} 个查询 × {} 块语料",
        QUERIES.len(),
        workload.corpus_size()
    );
    Ok(())
}

/// pprof 采样 demo：产出火焰图。
fn pprof_demo(config: &AppConfig) -> Result<()> {
    let workload = RustWorkload::new();
    let profiler = RustProfiler::new(&config.output_dir);

    info!(dir = %config.output_dir.display(), "开始采样");
    let artifact = profiler.sample_flamegraph("bm25_retrieval", || {
        // ⚠️ 必须跑够多轮，采样器才采得到样本。
        //
        // 实测：200 轮（约 130ms）只采到 **7** 个样本 ——
        // pprof 默认频率是 100Hz，130ms 理论上应该有 ~13 个，
        // 实际更少（信号投递有损耗）。
        //
        // 要拿到「有意义的火焰图」（几百个样本），
        // 至少要跑 **3000 轮**（约 2 秒）→ 约 200 个样本。
        //
        // ⚠️ 这和 Python 侧 py-spy「Samples: 2」的坑是**同一个**：
        //    采样器不是坏了，是被测程序跑得太快。
        workload.run(3000)
    })?;

    println!("火焰图: {}", artifact.path.display());
    println!("采样数: {}", artifact.samples);
    println!();
    println!("用浏览器打开 SVG 可以看到火焰图：");
    println!("  open {}", artifact.path.display());
    Ok(())
}

/// 对比 demo：多次计时 + 采样 + 出报告。
fn compare_demo(config: &AppConfig) -> Result<()> {
    let workload = RustWorkload::new();
    let profiler = RustProfiler::new(&config.output_dir);

    let mut timed_samples: Vec<u64> = Vec::new();
    for _ in 0..config.repeats {
        let start = Instant::now();
        let _ = workload.run(TIMING_OUTER_REPEATS);
        timed_samples.push(start.elapsed().as_micros() as u64);
    }

    let artifact = profiler.sample_flamegraph("bm25_retrieval", || workload.run(3000))?;

    let mut report = Report::new("Rust 侧性能剖析报告");
    report.push_timing("检索工作负载（微秒）", median(timed_samples));

    let out_path = config.output_dir.join("report.md");
    std::fs::create_dir_all(&config.output_dir)?;
    std::fs::write(&out_path, report.to_markdown())?;

    println!("{}", report.to_markdown());
    println!("火焰图: {}", artifact.path.display());
    println!("报告: {}", out_path.display());
    Ok(())
}

/// 打印 Python 侧三工具与 Rust 侧的对应关系。
fn tools_demo() -> Result<()> {
    println!("┌─────────────────────┬──────────────────────────┬──────────────────────────┐");
    println!("│ Python 工具          │ Rust 对应物               │ 共同点                   │");
    println!("├─────────────────────┼──────────────────────────┼──────────────────────────┤");
    println!("│ cProfile            │ criterion                │ 确定性重复测量           │");
    println!("│ py-spy              │ pprof                    │ 采样型、低开销、不改代码 │");
    println!("│ Scalene             │ （无直接对应）            │ —                        │");
    println!("└─────────────────────┴──────────────────────────┴──────────────────────────┘");
    println!();
    println!("⚠️ py-spy **不能**剖析 Rust 进程：");
    println!("   它读的是 CPython 的 PyInterpreterState / PyFrameObject，");
    println!("   Rust 进程里根本没有这些结构。");
    println!("   （macOS 上它还会先报「requires root」，掩盖真正的能力问题。）");
    Ok(())
}

fn main() -> Result<()> {
    init("agent-20");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let config = AppConfig::from_env();

    match demo.as_str() {
        "bench" => bench_demo(&config)?,
        "pprof" => pprof_demo(&config)?,
        "compare" => compare_demo(&config)?,
        "tools" => tools_demo()?,
        "all" => {
            tools_demo()?;
            println!();
            bench_demo(&config)?;
            println!();
            pprof_demo(&config)?;
        }
        other => {
            eprintln!("未知命令 {other}；可用：bench / pprof / compare / tools / all");
            std::process::exit(1);
        }
    }
    Ok(())
}
