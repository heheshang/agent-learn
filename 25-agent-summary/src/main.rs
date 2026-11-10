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
//! | **`agent`** | ★ **真调模型**跑一次 Agent 循环 |
//! | **`compare`** | ★ **协议对照** —— ReAct vs Function Calling |
//!
//! ## ⚠️ 三条日志硬约定
//!
//! - **4xx 用 `warn!`、5xx 用 `error!`**（告警不能被 4xx 淹没）
//! - **降级必须打 `warn!`**（静默降级等于埋雷）
//! - **业务逻辑里不打 `println!`**（没法按模块关掉）

use std::time::Instant;

use agent_25::agent_loop::{AgentConfig, FunctionAgent, ReactAgent};
use agent_25::aggregate::Strategy;
use agent_25::bench::{compare_three_tier, profile_naive, run_three_tier_with_windows};
use agent_25::config::AppConfig;
use agent_25::llm::{GenaiFunctionLlm, GenaiLlm, GenaiStructuredLlm, LlmOptions};
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
        "agent" => agent(config, &args).await,
        "compare" => compare(config, &args).await,
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
  agent        真调模型跑一次 Agent 循环（需要 DEEPSEEK_API_KEY）
  compare      协议对照 —— ReAct vs Function Calling（需要 DEEPSEEK_API_KEY）

环境变量（前缀 WK_）：
  WK_PORT              端口（默认 3170）
  WK_ACTIVITIES        活动记录数（默认 5000）
  WK_REQUESTS          压测请求数（默认 500）
  WK_CONCURRENCY       压测并发度（默认 16）
  WK_SERVICE_TIME_MS   模拟服务端时间（默认 0）
  WK_SERVER_WORKERS    服务端 worker 数（默认 8）
  WK_STRATEGY          默认策略（默认 optimized）
  WK_MODEL             模型名（默认 deepseek-flash）
  WK_MAX_TOKENS        最大输出 token（默认 2048）
  WK_TOOL_CHOICE       工具选择策略（默认 auto）—— none 表示「不把工具传给模型」
"
    );
}

/// 起 HTTP 服务。
async fn serve(config: AppConfig) -> anyhow::Result<()> {
    let activities = demo_activities(config.activities);
    let pipeline = Pipeline::new(activities, config.strategy);

    // ★ 有 API key 就挂上 LLM（`/agent` 端点可用），没有就离线模式。
    //
    // ⚠️ **不因为「没配 key」而拒绝启动** ——
    //    25 的核心产出（三档对照压测）是零网络的，
    //    LLM 是**可选增强**，不是启动前提。
    let has_key = std::env::var("DEEPSEEK_API_KEY")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);

    let state = if has_key {
        // ★ 提示词缓存路由键 —— **必须稳定**。
        //
        // system 提示 + 工具定义在 Agent 循环的每一轮里都一样，
        // 所以给它们一个固定的缓存键，
        // provider 可以把这段前缀路由到同一个缓存分片。
        //
        // ⚠️ 键里**不能**包含问题 —— 问题每次请求都不同，
        //    把变化的部分放进键等于每次都换分片，缓存永远命中不了。
        let options = LlmOptions::without_reasoning(0.0, config.max_tokens)
            .with_prompt_cache_key("agent-25-system-v1");

        // ★ 三条协议各挂一个 LLM ——
        //    `react` 用文本协议、`function` 用结构化工具调用、
        //    `structured` 用 JSON Schema 约束输出。
        let llm = std::sync::Arc::new(GenaiFunctionLlm::new(&config.model, options.clone()));
        let react_llm = std::sync::Arc::new(GenaiLlm::new(&config.model, options.clone()));
        let structured_llm = std::sync::Arc::new(GenaiStructuredLlm::new(
            &config.model,
            options.with_json_mode(true),
        ));

        agent_logging::info!(
            "已启用 LLM（模型 {}）—— /agent 端点可用（protocol: function / react / structured）",
            config.model
        );
        agent_25::api::AppState::with_all_llms(
            pipeline,
            llm,
            react_llm,
            structured_llm,
            config.tool_choice_mode(),
        )
    } else {
        agent_logging::warn!("没有 DEEPSEEK_API_KEY —— /agent 端点将返回 503；其余功能不受影响");
        agent_25::api::AppState::new(pipeline)
    };

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

/// ★ 真调模型跑一次 Agent 循环。
///
/// ## ⚠️ 需要 `DEEPSEEK_API_KEY`
///
/// 没有就**明确报错**（而不是静默降级成假模型）——
/// 静默降级会让人以为「测过了」。
async fn agent(config: AppConfig, args: &[String]) -> anyhow::Result<()> {
    let question = extract_positional(args, 2)
        .unwrap_or_else(|| "cache 模块上周做了哪些事？有没有风险项？".to_string());

    let activities = demo_activities(config.activities);
    let pipeline = Pipeline::new(activities, config.strategy);

    let window = Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400);

    let llm = std::sync::Arc::new(GenaiFunctionLlm::new(
        &config.model,
        LlmOptions::without_reasoning(0.0, config.max_tokens),
    ));

    let agent = FunctionAgent::new(
        llm,
        pipeline.registry().clone(),
        agent_25::agent_loop::AgentConfig {
            tool_choice: config.tool_choice_mode(),
            ..Default::default()
        },
    );

    let facts = pipeline.facts().to_vec();
    let recalled = agent_25::agent::recall_facts(&question, &facts, 3);

    agent_logging::info!("问题：{}", question);
    let run = agent.run(&question, &recalled, &window).await?;

    print_agent_run(&run);
    Ok(())
}

/// ★ 协议对照 —— ReAct vs Function Calling。
///
/// ## ★ 变量单一
///
/// **同一个问题、同一个工具集、同一个模型**，只换协议。
/// 差值归因到「协议本身」。
async fn compare(config: AppConfig, args: &[String]) -> anyhow::Result<()> {
    let question = extract_positional(args, 2)
        .unwrap_or_else(|| "cache 模块上周做了哪些事？有没有风险项？".to_string());

    let activities = demo_activities(config.activities);
    let pipeline = Pipeline::new(activities, config.strategy);
    let window = Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400);

    let facts = pipeline.facts().to_vec();
    let recalled = agent_25::agent::recall_facts(&question, &facts, 3);

    let options = LlmOptions::without_reasoning(0.0, config.max_tokens)
        .with_prompt_cache_key("agent-25-system-v1");

    // ── 协议 A：ReAct 文本协议 ────────────────────────────────
    let react_llm = std::sync::Arc::new(GenaiLlm::new(&config.model, options.clone()));
    let react = ReactAgent::new(
        react_llm,
        pipeline.registry().clone(),
        AgentConfig::default(),
    );
    let react_run = react.run(&question, &recalled, &window).await?;

    // ── 协议 B：Function Calling ──────────────────────────────
    let fn_llm = std::sync::Arc::new(GenaiFunctionLlm::new(&config.model, options.clone()));
    let function = FunctionAgent::new(
        fn_llm,
        pipeline.registry().clone(),
        AgentConfig {
            tool_choice: config.tool_choice_mode(),
            ..Default::default()
        },
    );
    let fn_run = function.run(&question, &recalled, &window).await?;

    // ── 协议 C：结构化输出（研究 → 成文 两阶段）────────────────
    //
    // ⚠️ 这一档**不是**「又一种工具调用协议」——
    //    它是「**研究**用工具、**成文**用 Schema 约束」的组合，
    //    所以它的 token 会**明显更高**（两次调用）。
    //
    //    但它换来的是**可校验**的结论（数字有出处、字段有 Schema）。
    let structured_llm = std::sync::Arc::new(GenaiStructuredLlm::new(
        &config.model,
        options.clone().with_json_mode(true),
    ));
    let structured_run = agent_25::structured::run_structured(
        &*structured_llm,
        std::sync::Arc::new(GenaiFunctionLlm::new(&config.model, options.clone())),
        &question,
        &recalled,
        &pipeline.registry().clone(),
        &window,
    )
    .await?;

    println!("=== 协议 A：ReAct 文本协议 ===");
    print_agent_run(&react_run);
    println!("\n=== 协议 B：Function Calling 结构化协议 ===");
    print_agent_run(&fn_run);
    println!("\n=== 协议 C：结构化输出（研究 → 成文）===");
    println!("答案：{}", structured_run.summary.answer);
    println!("接地校验：{}", structured_run.grounding.summary());
    println!(
        "模型：{} ｜ 证据：{} 条 ｜ 耗时：{} ms",
        structured_run.model,
        structured_run.evidence.len(),
        structured_run.elapsed_ms
    );

    println!("\n=== 对照 ===");
    let mut t = Table::new(&["指标", "ReAct", "Function Calling", "结构化输出"]);
    t.row(&[
        "耗时(ms)",
        &react_run.elapsed_ms.to_string(),
        &fn_run.elapsed_ms.to_string(),
        &structured_run.elapsed_ms.to_string(),
    ]);
    t.row(&[
        "工具调用次数",
        &react_run.tool_calls.to_string(),
        &fn_run.tool_calls.to_string(),
        &structured_run.evidence.len().to_string(),
    ]);
    t.row(&[
        "步数",
        &react_run.steps.len().to_string(),
        &fn_run.steps.len().to_string(),
        "-",
    ]);
    t.row(&[
        "达到步数上限",
        yes_no(react_run.hit_max_steps),
        yes_no(fn_run.hit_max_steps),
        "-",
    ]);

    // ★★ token 用量 —— **比耗时更稳定的归因依据**。
    //
    // ⚠️ 耗时受上游负载影响，同一次实验前后差 200ms 就可能翻转结论。
    //    token 数是确定性的。
    t.row(&[
        "输入 token",
        &react_run.usage.prompt_tokens.to_string(),
        &fn_run.usage.prompt_tokens.to_string(),
        &structured_run.usage.prompt_tokens.to_string(),
    ]);
    t.row(&[
        "输出 token",
        &react_run.usage.completion_tokens.to_string(),
        &fn_run.usage.completion_tokens.to_string(),
        &structured_run.usage.completion_tokens.to_string(),
    ]);
    t.row(&[
        "总 token",
        &react_run.usage.total_tokens.to_string(),
        &fn_run.usage.total_tokens.to_string(),
        &structured_run.usage.total_tokens.to_string(),
    ]);
    t.row(&[
        "输出被截断",
        yes_no(react_run.truncated),
        yes_no(fn_run.truncated),
        yes_no(structured_run.truncated),
    ]);
    t.row(&[
        "结论可校验",
        "否",
        "否",
        yes_no(structured_run.grounding.grounded),
    ]);

    println!("{}", t.render());

    Ok(())
}

/// 打印一次 Agent 运行的结果。
fn print_agent_run(run: &agent_25::agent_loop::AgentRun) {
    println!("问题：{}", run.question);
    println!("答案：{}", run.answer);
    println!(
        "模型：{} ｜ 工具调用：{} 次（尝试 {} 次）｜ 步数：{} ｜ 耗时：{} ms ｜ token：{}（输入 {} / 输出 {}）",
        run.model,
        run.tool_calls,
        run.tool_attempts,
        run.steps.len(),
        run.elapsed_ms,
        run.usage.total_tokens,
        run.usage.prompt_tokens,
        run.usage.completion_tokens
    );

    // ⚠️ 截断的答案看起来是**完整的** —— 必须显式提示。
    if run.truncated {
        println!("⚠️ 输出被 max_tokens 截断 —— 结论可能不完整");
    }

    if !run.tools_used.is_empty() {
        println!("用到的工具：{}", run.tools_used.join(", "));
    }

    if !run.steps.is_empty() {
        println!("\n步骤：");
        for step in &run.steps {
            let kind = if step.final_answer {
                "最终答案"
            } else {
                "工具调用"
            };
            println!("  第 {} 步（{kind}）", step.step);
            for call in &step.tool_calls {
                println!(
                    "    - {} → {}（尝试 {} 次）",
                    call.name,
                    if call.ok { "成功" } else { "失败" },
                    call.attempts
                );
                println!("      {}", first_line(&call.observation));
            }
        }
    }
}

/// 取第一行（观察可能很长）。
///
/// ## ⚠️ 必须按**字符**边界截断
///
/// ⚠️ **不能用 `&line[..100]`** —— 那是**字节**切片，
/// 中文一个字符 3 字节，切在中间会 **panic**：
///
/// ```text
/// panicked at 'end byte index 100 is not a char boundary;
/// it is inside '—' (bytes 98..101 of string)'
/// ```
///
/// 这在中文场景下**必然触发** —— 因为观察文本全是中文。
fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or("");

    // 按字符计数，超了就截断。
    for (count, (idx, _)) in line.char_indices().enumerate() {
        if count >= 100 {
            return format!("{}…", &line[..idx]);
        }
    }

    line.to_string()
}

/// 是 / 否。
fn yes_no(value: bool) -> &'static str {
    if value { "是" } else { "否" }
}

/// 取第 n 个位置参数。
fn extract_positional(args: &[String], index: usize) -> Option<String> {
    args.get(index)
        .filter(|s| !s.starts_with('-'))
        .map(|s| s.to_string())
}
