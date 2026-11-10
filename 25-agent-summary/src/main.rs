//! 25-agent-summary 的二进制入口。
//!
//! ## 子命令
//!
//! | 命令 | 做什么 |
//! |---|---|
//! | `serve` | 起 HTTP 服务 |
//! | `bench` | 跑三档对照实验（进程内轨） |
//! | `bench-http` | 打真实 HTTP 服务（验证轨） |
//! | `profile` | 分段剖析 —— 指出时间花在哪 |
//! | `sweep` | 并发度扫描 |
//! | `report` | 生成周报 Markdown |
//!
//! ## ⚠️ 三条日志硬约定
//!
//! - **4xx 用 `warn!`、5xx 用 `error!`**（告警不能被 4xx 淹没）
//! - **降级必须打 `warn!`**（静默降级等于埋雷）
//! - **业务逻辑里不打 `println!`**（没法按模块关掉）

use std::time::Instant;

use agent_25::aggregate::Strategy;
use agent_25::bench::{compare_three_tier, profile_naive, run_three_tier_with_windows};
use agent_25::config::AppConfig;
use agent_25::loadtest::LoadConfig;
use agent_25::model::Window;
use agent_25::pipeline::{Pipeline, demo_activities};
use agent_25::render::render_report;
use agent_25::table::Table;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    agent_logging::init("agent-25");

    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    let config = AppConfig::from_env();

    match command {
        "serve" => serve(config).await,
        "bench" => bench(config).await,
        "bench-http" => bench_http(config).await,
        "profile" => profile(config),
        "sweep" => sweep(config).await,
        "cache-sweep" => cache_sweep(config).await,
        "report" => report(config),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => {
            eprintln!("未知子命令：{other}\n");
            print_help();
            std::process::exit(1);
        }
    }
}

/// 打印帮助。
fn print_help() {
    println!(
        "25-agent-summary —— Agent 全链路汇总与压测台

用法：agent-25 <子命令>

子命令：
  serve        起 HTTP 服务（默认端口 3170）
  bench        三档对照实验（进程内轨 —— 用来归因）
  bench-http   打真实 HTTP 服务（验证轨 —— 外部视角）
  profile      分段剖析 —— 指出时间花在哪
  sweep        并发度扫描
  cache-sweep  缓存命中率扫描
  report       生成周报 Markdown

环境变量（前缀 WK_）：
  WK_PORT              端口（默认 3170）
  WK_ACTIVITIES        活动记录数（默认 5000）
  WK_REQUESTS          压测请求数（默认 500）
  WK_CONCURRENCY       压测并发度（默认 16）
  WK_SERVICE_TIME_MS   模拟服务端时间（默认 0）
  WK_SERVER_WORKERS    服务端 worker 数（默认 8）
  WK_STRATEGY          默认策略（默认 optimized）
"
    );
}

/// 起 HTTP 服务。
async fn serve(config: AppConfig) -> anyhow::Result<()> {
    let activities = demo_activities(config.activities);
    let pipeline = Pipeline::new(activities, config.strategy);
    let state = agent_25::api::AppState::new(pipeline);

    let app = agent_25::api::router(state);
    let addr = config.bind_addr();

    agent_logging::info!("25-agent-summary 启动于 http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// 三档对照实验。
async fn bench(config: AppConfig) -> anyhow::Result<()> {
    let activities = demo_activities(config.activities);

    let load = LoadConfig {
        requests: config.requests,
        concurrency: config.concurrency,
        service_time_ms: config.service_time_ms,
        server_workers: config.server_workers,
    };

    agent_logging::info!(
        "三档对照实验：{} 条记录 / {} 次请求 / 并发 {}",
        activities.len(),
        load.requests,
        load.concurrency
    );

    // ★ 用**多个窗口** —— 全用同一个窗口会让缓存档 100% 命中，
    //   缓存收益被放大成几百倍，那个数字漂亮但不可信。
    let windows = build_windows(activities.len());
    let results = run_three_tier_with_windows(&activities, &windows, &load).await;
    let comparisons = compare_three_tier(&results);

    print_bench_table(&results, &comparisons);
    agent_logging::info!("本次用了 {} 个不同的查询窗口", windows.len());
    Ok(())
}

/// 造一批**不同的**查询窗口 —— 模拟真实负载的发散查询。
///
/// ## ⚠️ 窗口数量决定缓存命中率
///
/// 窗口越少 → 命中率越高 → 缓存收益被放大。
/// 这里用 16 个窗口，覆盖数据跨度的不同区段。
fn build_windows(activity_count: usize) -> Vec<Window> {
    let base = 1_700_000_000i64;
    let span_days = (activity_count as i64 / 250).clamp(4, 28);
    (0..16)
        .map(|i| {
            let start = base + (i * span_days / 16) * 86_400;
            Window::new(start, start + 6 * 86_400)
        })
        .collect()
}

/// 打印压测表格。
fn print_bench_table(
    results: &[agent_25::bench::ExperimentResult],
    comparisons: &[agent_25::metrics::Comparison],
) {
    let mut table = Table::new(&["变体", "QPS", "p50(ms)", "p99(ms)", "max(ms)", "成功率"]);

    for r in results {
        table.row(&[
            &r.variant,
            &format!("{:.1}", r.report.qps),
            &format!("{:.4}", r.report.p50_ms),
            &format!("{:.4}", r.report.p99_ms),
            &format!("{:.4}", r.report.max_ms),
            &format!("{:.1}%", r.report.success_rate * 100.0),
        ]);
    }

    println!("{}", table.render());

    if !comparisons.is_empty() {
        println!("\n相对基线的差值：");
        let mut t = Table::new(&["变体", "QPS 倍数", "p50 倍数", "p99 倍数"]);
        for c in comparisons {
            t.row(&[
                &c.variant,
                &format!("{:.2}x", c.qps_ratio),
                &format!("{:.3}x", c.p50_ratio),
                &format!("{:.3}x", c.p99_ratio),
            ]);
        }
        println!("{}", t.render());
    }
}

/// HTTP 压测（验证轨）。
async fn bench_http(config: AppConfig) -> anyhow::Result<()> {
    let url = std::env::var("WK_TARGET_URL")
        .unwrap_or_else(|_| format!("http://127.0.0.1:{}", config.port));

    let client = reqwest::Client::new();
    let load = LoadConfig {
        requests: config.requests,
        concurrency: config.concurrency,
        service_time_ms: config.service_time_ms,
        server_workers: config.server_workers,
    };

    agent_logging::info!("HTTP 压测目标：{}", url);

    let report = agent_25::loadtest::run_http(
        &client,
        &url,
        "/summary",
        serde_json::json!({"start": 1_700_000_000i64, "end": 1_700_000_000i64 + 7 * 86_400}),
        &load,
    )
    .await;

    let mut t = Table::new(&["请求数", "QPS", "p50(ms)", "p99(ms)", "成功率"]);
    t.row(&[
        &report.total_requests.to_string(),
        &format!("{:.1}", report.qps),
        &format!("{:.4}", report.p50_ms),
        &format!("{:.4}", report.p99_ms),
        &format!("{:.1}%", report.success_rate * 100.0),
    ]);
    println!("{}", t.render());
    Ok(())
}

/// 分段剖析。
fn profile(config: AppConfig) -> anyhow::Result<()> {
    let activities = demo_activities(config.activities);
    let window = Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400);

    let rounds = 20;
    let start = Instant::now();
    let profile = profile_naive(&activities, &window, rounds);
    let elapsed = start.elapsed();

    let mut t = Table::new(&["环节", "耗时(ms)", "占比"]);
    for p in &profile {
        t.row(&[
            &p.segment,
            &format!("{:.3}", p.elapsed_ms),
            &format!("{:.1}%", p.share * 100.0),
        ]);
    }
    println!("{}", t.render());
    agent_logging::info!(
        "剖析完成：{} 轮，耗时 {:.3} ms",
        rounds,
        elapsed.as_secs_f64() * 1000.0
    );

    // 结论提示。
    let top = profile
        .first()
        .map(|p| p.segment.clone())
        .unwrap_or_default();
    println!("\n★ 瓶颈是「{top}」—— 优化应该针对它，而不是「减少扫描量」。");
    Ok(())
}

/// 缓存命中率扫描。
async fn cache_sweep(config: AppConfig) -> anyhow::Result<()> {
    let activities = demo_activities(config.activities);
    let load = LoadConfig {
        requests: config.requests,
        concurrency: config.concurrency,
        service_time_ms: config.service_time_ms,
        server_workers: config.server_workers,
    };

    let levels = [1usize, 8, 64, 512, 5000];
    let results = agent_25::bench::run_cache_hit_sweep(&activities, &load, &levels).await;

    let mut t = Table::new(&["变体", "QPS", "p50(ms)", "p99(ms)"]);
    for r in &results {
        t.row(&[
            &r.variant,
            &format!("{:.1}", r.report.qps),
            &format!("{:.4}", r.report.p50_ms),
            &format!("{:.4}", r.report.p99_ms),
        ]);
    }
    println!("{}", t.render());
    Ok(())
}

/// 并发度扫描。
async fn sweep(config: AppConfig) -> anyhow::Result<()> {
    let activities = demo_activities(config.activities);
    let window = Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400);
    let levels = [1usize, 2, 4, 8, 16, 32, 64];

    let results = agent_25::bench::run_concurrency_sweep(
        &activities,
        &window,
        Strategy::Optimized,
        &levels,
        config.requests,
    )
    .await;

    let mut t = Table::new(&["并发", "QPS", "p50(ms)", "p99(ms)", "成功率"]);
    for r in &results {
        t.row(&[
            &r.variant,
            &format!("{:.1}", r.report.qps),
            &format!("{:.4}", r.report.p50_ms),
            &format!("{:.4}", r.report.p99_ms),
            &format!("{:.1}%", r.report.success_rate * 100.0),
        ]);
    }
    println!("{}", t.render());
    Ok(())
}

/// 生成周报。
fn report(config: AppConfig) -> anyhow::Result<()> {
    let activities = demo_activities(config.activities);
    let window = Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400);

    let mut pipeline = Pipeline::new(activities, config.strategy);
    let out = pipeline.run("cache", &window);

    let markdown = render_report(
        &out.digest,
        &window,
        None,
        &[(config.strategy, config.strategy.label().to_string())],
    );

    let dir = std::path::Path::new(&config.output_dir);
    std::fs::create_dir_all(dir)?;
    let path = dir.join("weekly-summary.md");
    std::fs::write(&path, &markdown)?;

    println!("{}", markdown);
    agent_logging::info!("周报已写入 {}", path.display());
    Ok(())
}
