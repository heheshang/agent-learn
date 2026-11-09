//! 18-agent-error —— 手撕 Agent 错误处理。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- tools                 # 工具清单（不调模型）
//! cargo run -- table                 # ★ 错误分类速查表
//! cargo run -- call strict_lookup '{"id":"nope"}'   # 直接调工具
//! cargo run -- run "拉一下 report/2026"              # 跑一次 Agent
//! cargo run -- compare               # ★ 三类错误对照
//! cargo run -- serve                 # 启动 HTTP 服务（:3110）
//! cargo run -- all                   # 全部
//! ```

use std::sync::Arc;

use agent_logging::{info, init};
use anyhow::Result;

use agent_18::agent::{Agent, AgentConfig};
use agent_18::api::AppStateInner;
use agent_18::config::AppConfig;
use agent_18::error::{ErrorCode, ErrorKind};
use agent_18::llm::GenaiFunctionLlm;
use agent_18::middleware::ErrorHandlingMiddleware;
use agent_18::tool::{
    FLAKY_FETCH_TOOL_NAME, FlakyTool, GUARDED_PAY_TOOL_NAME, GuardedTool, STRICT_LOOKUP_TOOL_NAME,
    StrictTool, ToolRegistry,
};

/// 默认演示问题。
///
/// ⚠️ 必须是一个**会触发工具失败**的问题 ——
/// 这样才能观察到「重试 → 降级」的完整过程。
const DEFAULT_QUESTION: &str = "帮我拉取一下 report/2026 的数据";

#[tokio::main]
async fn main() -> Result<()> {
    init("agent-18");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "tools" => tools_demo()?,
        "table" => table_demo()?,
        "call" => {
            let tool = arg_or_default(2, STRICT_LOOKUP_TOOL_NAME);
            let args = std::env::args()
                .nth(3)
                .unwrap_or_else(|| r#"{"id":"nope"}"#.to_string());
            call_demo(&tool, &args).await?;
        }
        "run" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            run_demo(&question).await?;
        }
        "compare" => compare_demo().await?,
        "serve" => serve().await?,
        "all" => {
            tools_demo()?;
            table_demo()?;
            compare_demo().await?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: tools / table / call / run / compare / serve / all）"
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

/// 构建工具注册表。
fn build_registry(config: &AppConfig) -> ToolRegistry {
    ToolRegistry::new(vec![
        Arc::new(FlakyTool::new(config.flaky_fail_times)),
        Arc::new(StrictTool::new(vec![
            "rec-1".to_string(),
            "rec-2".to_string(),
        ])),
        Arc::new(GuardedTool::new(config.pay_api_key.clone())),
    ])
}

/// 构建中间件。
fn build_middleware(config: &AppConfig) -> ErrorHandlingMiddleware {
    ErrorHandlingMiddleware::new(build_registry(config))
        .with_max_retries(config.max_retries)
        .with_base_delay(std::time::Duration::from_millis(config.base_delay_ms))
        .with_max_delay(std::time::Duration::from_millis(config.max_delay_ms))
}

/// 构建 LLM 客户端。
fn build_llm(config: &AppConfig) -> Arc<dyn agent_18::llm::FunctionLlm> {
    info!(
        model = %config.model,
        temperature = config.temperature,
        max_tokens = config.max_tokens,
        "创建 LLM 客户端"
    );
    Arc::new(GenaiFunctionLlm::new(
        &config.model,
        config.temperature,
        config.max_tokens,
        config.disable_reasoning,
    ))
}

/// 打印工具清单（不调模型）。
fn tools_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let registry = build_registry(&config);

    println!("\n=== 可用工具（{} 个）===\n", registry.len());
    print!(
        "{}",
        agent_18::prompt::render_tool_catalog(&registry.specs())
    );

    Ok(())
}

/// ★ 错误分类速查表。
fn table_demo() -> Result<()> {
    let codes = [
        ErrorCode::UpstreamTimeout,
        ErrorCode::UpstreamUnavailable,
        ErrorCode::RateLimited,
        ErrorCode::InvalidArguments,
        ErrorCode::UnknownTool,
        ErrorCode::NotFound,
        ErrorCode::RuleViolation,
        ErrorCode::Unauthorized,
        ErrorCode::InternalFault,
    ];

    println!("\n=== ★ 错误分类速查表 ===\n");

    // ⚠️ 必须按**显示宽度**补空格（汉字占 2 列），
    //    否则中英文混排的列会越往右越歪。
    let mut table = agent_18::table::Table::new(&["错误码", "类别", "重试", "降级", "HTTP"]);

    for code in codes {
        let kind = code.default_kind();
        let status = match kind {
            ErrorKind::Business => "400",
            ErrorKind::Transient => "503",
            ErrorKind::Fatal => "500",
        };
        table.row(&[
            code.id(),
            kind.name(),
            if kind.retryable() { "✅" } else { "❌" },
            if kind.degradable() { "✅" } else { "❌" },
            status,
        ]);
    }

    print!("{}", table.render());

    println!("\n── 决策规则 ──");
    println!("Transient：重试（指数退避）→ 重试耗尽后降级");
    println!("Business ：不重试，直接降级成观察");
    println!("Fatal    ：不重试、不降级，直接终止");

    println!("\n── 三条不变量 ──");
    println!("① 每个 tool_call 都必须生成恰好一条 tool 消息（少一条整个请求 400）");
    println!("② 致命错误也要生成回执，否则下一轮请求会 400");
    println!("③ 模型调用失败不能降级 —— 没有模型就没有答案");

    Ok(())
}

/// 直接调工具（不经模型）。
async fn call_demo(tool: &str, args_json: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let middleware = build_middleware(&config);

    let args: serde_json::Value =
        serde_json::from_str(args_json).map_err(|e| anyhow::anyhow!("参数不是合法 JSON: {e}"))?;

    println!("\n=== 直接调用工具（不经模型）===");
    println!("工具: {tool}");
    println!("参数: {args}\n");

    let outcome = middleware
        .wrap_tool_call(&agent_18::middleware::ToolCallRequest {
            name: tool.to_string(),
            arguments: args,
            call_id: "direct".to_string(),
        })
        .await;

    println!(
        "结果: {}（尝试 {} 次）\n",
        if outcome.ok {
            "✅ 成功"
        } else {
            "❌ 失败"
        },
        outcome.attempts
    );
    println!("{}", outcome.observation);

    if let Some(err) = &outcome.error {
        println!("\n错误码  : {}", err.diagnostic_code());
        println!("错误类别: {}", err.kind.name());
        println!("可否重试: {}", err.retryable());
        println!("可否降级: {}", err.degradable());
    }

    Ok(())
}

/// 跑一次 Agent。
async fn run_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let llm = build_llm(&config);
    let middleware = build_middleware(&config);

    let agent = Agent::new(
        llm,
        middleware,
        AgentConfig {
            max_steps: config.max_steps,
            tool_choice: config.tool_choice.clone(),
        },
    );

    println!("\n=== 跑一次 Agent（带错误处理）===");
    println!("问题: {question}");
    println!("tool_choice: {}\n", config.tool_choice.name());

    let run = agent.run(question).await?;

    println!("── 轨迹 ──");
    print!("{}", agent_18::prompt::render_trajectory(&run.steps));

    println!("── 最终答案 ──");
    println!("{}", run.answer);

    println!(
        "\n── 统计 ──\n工具调用 {} 次（含重试）| 逻辑调用 {} 次 | 重试 {} 次 | 降级 {} 次 | 耗时 {} ms",
        run.tool_calls,
        run.tool_call_groups,
        run.retried_calls,
        run.degraded_errors,
        run.elapsed_ms,
    );
    if run.hit_max_steps {
        println!("⚠️ 达到最大步数 —— 结果是降级返回的");
    }

    Ok(())
}

/// ★ 三类错误对照。
///
/// ## 对照的变量
///
/// **只有「工具失败类型」这一个变量不同** ——
/// 同一个中间件、同一套重试参数、同一个工具名。
async fn compare_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let middleware = build_middleware(&config);

    let scenarios = [
        (
            "Transient（可重试）",
            FLAKY_FETCH_TOOL_NAME,
            serde_json::json!({ "resource": "report/2026" }),
            "前几次失败，之后成功 —— 重试能救回来",
        ),
        (
            "Business（业务失败）",
            STRICT_LOOKUP_TOOL_NAME,
            serde_json::json!({ "id": "nope" }),
            "参数/目标有问题 —— 重试**一定**没用",
        ),
        (
            "Fatal（致命）",
            GUARDED_PAY_TOOL_NAME,
            serde_json::json!({ "amount": 100 }),
            "环境没配好 —— 重试和降级都是徒劳",
        ),
    ];

    println!("\n=== ★ 三类错误对照 ===");
    println!("（只有「工具失败类型」这一个变量不同）");
    println!("重试上限: {} 次\n", config.max_retries);

    for (label, tool, args, note) in scenarios {
        let request = agent_18::middleware::ToolCallRequest {
            name: tool.to_string(),
            arguments: args,
            call_id: "c1".to_string(),
        };

        let outcome = middleware.wrap_tool_call(&request).await;

        println!("── {label} ──");
        println!("工具    : {tool}");
        println!("说明    : {note}");
        println!(
            "结果    : {}",
            if outcome.ok {
                "✅ 成功"
            } else {
                "❌ 失败"
            }
        );
        println!("尝试次数: {}", outcome.attempts);

        if let Some(err) = &outcome.error {
            println!("错误码  : {}", err.diagnostic_code());
            println!("类别    : {}", err.kind.name());
        }

        println!(
            "处置    : {}",
            if outcome.ok {
                "直接返回结果"
            } else if outcome.fatal {
                "★ 终止（不降级）"
            } else if outcome.attempts > 1 {
                "重试后降级"
            } else {
                "直接降级（不重试）"
            }
        );
        println!("观察    : {}", outcome.observation);
        println!();
    }

    Ok(())
}

/// 启动 HTTP 服务。
async fn serve() -> Result<()> {
    let config = AppConfig::from_env();

    info!("配置:\n{}", config.summary());

    let llm = build_llm(&config);
    let middleware = build_middleware(&config);

    let agent = Arc::new(Agent::new(
        llm.clone(),
        middleware,
        AgentConfig {
            max_steps: config.max_steps,
            tool_choice: config.tool_choice.clone(),
        },
    ));

    let state = Arc::new(AppStateInner::new(config.clone(), agent, llm));
    let app = agent_18::api::routes::router(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    info!("服务已启动: http://{}", config.bind_addr);

    axum::serve(listener, app).await?;

    Ok(())
}
