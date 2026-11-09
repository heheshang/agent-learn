//! 22-agent-async —— 手撕异步处理。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- tools                     # 工具清单（不调模型）
//! cargo run -- modes                     # ★ 四种并发模式说明
//! cargo run -- bench                     # ★ ★ 并发对照实验（四档模式）
//! cargo run -- bench-real                # ★ ★ 用真实天气接口跑对照实验
//! cargo run -- call get_weather '{"location":"北京"}'  # 直接调工具
//! cargo run -- run "北京和上海现在天气怎么样？"   # 跑一次 Agent
//! cargo run -- serve                     # 启动 HTTP 服务（:3140）
//! cargo run -- all                       # 全部
//! ```

use std::sync::Arc;

use agent_18::llm::GenaiFunctionLlm;
use agent_18::middleware::ErrorHandlingMiddleware;
use agent_logging::{info, init, warn};
use anyhow::Result;

use agent_22::agent::{Agent, AgentConfig};
use agent_22::api::AppStateInner;
use agent_22::bench;
use agent_22::config::AppConfig;
use agent_22::prompt;
use agent_22::tools::{self, CALCULATE_TOOL_NAME, GET_WEATHER_TOOL_NAME, SLOW_REPORT_TOOL_NAME};

/// 默认演示问题。
///
/// ⚠️ 必须是一个**会触发多个工具调用**的问题 ——
/// 这样才能观察到「一轮里的并发执行」。
const DEFAULT_QUESTION: &str = "北京和上海现在天气怎么样？请分别说明。";

/// 对照实验用的城市列表。
const BENCH_CITIES: [&str; 4] = ["北京", "上海", "广州", "深圳"];

#[tokio::main]
async fn main() -> Result<()> {
    init("agent-22");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "tools" => tools_demo()?,
        "modes" => modes_demo()?,
        "bench" => bench_demo(false).await?,
        "bench-real" => bench_demo(true).await?,
        "call" => {
            let tool = arg_or_default(2, GET_WEATHER_TOOL_NAME);
            let args = std::env::args()
                .nth(3)
                .unwrap_or_else(|| r#"{"location":"北京"}"#.to_string());
            call_demo(&tool, &args).await?;
        }
        "run" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            run_demo(&question).await?;
        }
        "serve" => serve().await?,
        "all" => {
            tools_demo()?;
            modes_demo()?;
            bench_demo(false).await?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: tools / modes / bench / bench-real / call / run / serve / all）"
            );
        }
    }

    Ok(())
}

/// 取第 n 个命令行参数（没有就用默认值）。
fn arg_or_default(index: usize, default: &str) -> String {
    std::env::args()
        .nth(index)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// 构造 Agent（真实模型）。
fn build_agent(config: &AppConfig) -> Arc<Agent> {
    let llm = Arc::new(GenaiFunctionLlm::new(
        &config.model,
        config.temperature,
        config.max_tokens,
        config.disable_reasoning,
    ));

    let registry = tools::demo_registry(200, Some(config.tool_timeout()));
    let blocking_tools = vec![SLOW_REPORT_TOOL_NAME.to_string()];

    let middleware = ErrorHandlingMiddleware::new(registry)
        .with_max_retries(config.max_retries)
        .with_base_delay(std::time::Duration::from_millis(config.base_delay_ms))
        .with_max_delay(std::time::Duration::from_millis(config.max_delay_ms));

    let agent_config = AgentConfig {
        max_steps: config.max_steps,
        tool_choice: config.tool_choice.clone(),
        max_concurrency: config.max_concurrency,
    };

    Arc::new(Agent::new(llm, middleware, agent_config, blocking_tools))
}

/// 打印工具清单。
fn tools_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let agent = build_agent(&config);

    println!("=== 工具清单 ===");
    print!("{}", prompt::tools_summary(&agent.tool_specs()));

    println!();
    println!("=== 三条执行轨道 ===");
    print_table(
        &["工具", "轨道", "说明"],
        &[
            vec![
                GET_WEATHER_TOOL_NAME.to_string(),
                "异步".to_string(),
                "真 HTTP（Open-Meteo），等网络".to_string(),
            ],
            vec![
                CALCULATE_TOOL_NAME.to_string(),
                "CPU".to_string(),
                "纯计算，并发无收益".to_string(),
            ],
            vec![
                SLOW_REPORT_TOOL_NAME.to_string(),
                "线程池".to_string(),
                "真阻塞，走 spawn_blocking".to_string(),
            ],
        ],
        &[14, 8, 32],
    );

    Ok(())
}

/// 打印四种并发模式。
fn modes_demo() -> Result<()> {
    println!("=== 四种并发模式（对应 LangChain） ===");
    println!();
    print_table(
        &["模式", "说明", "LangChain 对应"],
        &[
            vec![
                "sequential".to_string(),
                "一个接一个等，总耗时 = 之和".to_string(),
                "（手写 for 循环 await）".to_string(),
            ],
            vec![
                "join_all".to_string(),
                "全并发，按输入顺序返回".to_string(),
                "batch()".to_string(),
            ],
            vec![
                "as_completed".to_string(),
                "全并发，按完成顺序返回".to_string(),
                "batch_as_completed()".to_string(),
            ],
            vec![
                "bounded(n)".to_string(),
                "全并发，同时最多跑 n 个".to_string(),
                "max_concurrency=n".to_string(),
            ],
        ],
        &[16, 30, 26],
    );

    println!();
    println!("⚠️ 限流组只有在「任务数 > 限流数」时才和全并发有差别。");
    println!("⚠️ 并发 ≠ 并行：并发提升的是「等待」的吞吐，不是「计算」的吞吐。");

    Ok(())
}

/// 打印一张按**显示宽度**对齐的表。
///
/// ## ⚠️ 为什么用 `Vec<Vec<String>>` 而不是 `Vec<Vec<&str>>`
///
/// 因为表头和单元格里混着 `&str` 常量与 `String` 常量 ——
/// 统一成 `String` 最省心（`&str` 会自动 `to_string()`）。
fn print_table(headers: &[&str], rows: &[Vec<String>], widths: &[usize]) {
    let header_cells: Vec<&str> = headers.to_vec();
    println!(
        "{}",
        agent_22::table::render_row(&header_cells, widths, &[false; 8])
    );

    let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    println!("{}", sep.join("  "));

    for row in rows {
        let cells: Vec<&str> = row.iter().map(|c| c.as_str()).collect();
        println!(
            "{}",
            agent_22::table::render_row(&cells, widths, &[false; 8])
        );
    }
}

/// ★ 并发对照实验。
async fn bench_demo(real: bool) -> Result<()> {
    let config = AppConfig::from_env();
    let cities: Vec<String> = BENCH_CITIES.iter().map(|c| c.to_string()).collect();

    println!("=== 并发对照实验 ===");
    println!(
        "负载：{}",
        if real {
            "真实天气接口（Open-Meteo）"
        } else {
            "模拟 I/O 等待（sleep 120ms，更稳定可复现）"
        }
    );
    println!(
        "任务数：{}　并发上限：{}",
        cities.len(),
        config.max_concurrency
    );
    println!();

    let reports = bench::run_compare(cities, config.max_concurrency, !real).await?;

    print!("{}", bench::render_table(&reports));

    let sequential = reports
        .iter()
        .find(|r| r.mode == agent_22::concurrency::ExecutionMode::Sequential)
        .map(|r| r.total_ms())
        .unwrap_or(0);

    println!();
    println!("串行基准：{sequential} ms");
    for report in &reports {
        println!(
            "  {:<24} {:>6} ms　{:>5.2}x",
            report.mode.label(),
            report.total_ms(),
            report.speedup(sequential)
        );
    }

    Ok(())
}

/// 直接调工具。
async fn call_demo(tool: &str, args: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let agent = build_agent(&config);

    let arguments: serde_json::Value =
        serde_json::from_str(args).map_err(|e| anyhow::anyhow!("参数不是合法 JSON: {e}"))?;

    let start = std::time::Instant::now();
    let request = agent_18::middleware::ToolCallRequest {
        name: tool.to_string(),
        arguments,
        call_id: "direct".to_string(),
    };

    let outcome = agent.middleware().wrap_tool_call(&request).await;

    println!("工具    : {tool}");
    println!("耗时    : {} ms", start.elapsed().as_millis());
    println!("成功    : {}", outcome.ok);
    println!("尝试次数: {}", outcome.attempts);
    println!("走线程池: {}", agent.is_blocking(tool));
    println!();
    println!("--- 观察 ---");
    println!("{}", outcome.observation);

    Ok(())
}

/// 跑一次 Agent。
async fn run_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let agent = build_agent(&config);

    println!("=== Agent 运行 ===");
    println!("问题：{question}");
    println!();

    let run = agent.run(question).await?;

    for step in &run.steps {
        println!("── 第 {} 步 ──", step.step);
        if !step.text.is_empty() {
            println!("模型：{}", step.text);
        }
        for call in &step.tool_calls {
            println!(
                "  工具 {}（call_id={}，{} 次尝试{}）",
                call.name,
                call.call_id,
                call.attempts,
                if call.blocking { "，线程池" } else { "" }
            );
            println!("    {}", call.observation.lines().next().unwrap_or(""));
        }
        println!();
    }

    println!("=== 最终答案 ===");
    println!("{}", run.answer);
    println!();
    println!(
        "工具调用 {} 次 / {} 组　耗时 {} ms　模型 {}",
        run.tool_calls, run.tool_call_groups, run.elapsed_ms, run.model
    );

    Ok(())
}

/// 启动 HTTP 服务。
async fn serve() -> Result<()> {
    let config = AppConfig::from_env();

    info!("agent-22 启动中");
    println!("{}", config.summary());
    println!();

    let agent = build_agent(&config);
    let llm = Arc::new(GenaiFunctionLlm::new(
        &config.model,
        config.temperature,
        config.max_tokens,
        config.disable_reasoning,
    ));

    let state = Arc::new(AppStateInner::new(config.clone(), agent, llm));
    let app = agent_22::api::routes::router(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    let addr = listener.local_addr()?;

    println!("★ 监听 http://{addr}");
    println!("  GET  /            服务信息 + 工具清单");
    println!("  GET  /api/modes   四种并发模式说明");
    println!("  POST /api/run     跑一次 Agent（并发工具执行）");
    println!("  POST /api/bench   ★ 并发对照实验");
    println!("  POST /api/tools/call  直接调工具");
    println!();

    axum::serve(listener, app).await?;

    warn!("服务已退出");

    Ok(())
}
