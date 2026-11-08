//! 14-agent-tool —— 手撕自定义工具开发（LangChain Custom Tools + MCP）。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- tools                      # 工具清单 + Schema（不调模型、不联网）
//! cargo run -- call get_weather '{"location":"北京"}'   # 直接调工具 ★
//! cargo run -- run "北京现在天气怎么样？"    # 跑一次 Agent（工具集成到 ReAct 循环）
//! cargo run -- compare "北京现在天气怎么样？" # 三模式消融对照 ★
//! cargo run -- trace "北京现在天气怎么样？"   # 打印完整 ReAct 轨迹
//! cargo run -- mcp                        # 启动内置 MCP 服务器（:3071）
//! cargo run -- serve                      # 启动 HTTP 服务（:3070）
//! cargo run -- all                        # 全部
//! ```
//!
//! ⚠️ 默认问题**故意不等于** few-shot 示例里的问题（上海那条）——
//! 否则模型会直接抄示例的答案，消融实验就失真了。

use agent_logging::{debug, info, init, warn};
use std::sync::Arc;

use agent_14::agent::Agent;
use agent_14::api::AppStateInner;
use agent_14::config::{AppConfig, ToolMode};
use agent_14::mcp::McpClient;
use agent_14::tool::{Tool, ToolContext, ToolRegistry};
use agent_14::weather::WeatherTool;
use anyhow::Result;

/// 默认演示问题。
///
/// ⚠️ 不能用 few-shot 示例里的问题（上海）—— 模型会直接抄示例答案。
const DEFAULT_QUESTION: &str = "北京现在天气怎么样？";

#[tokio::main]
async fn main() -> Result<()> {
    init("agent-14");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "run" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            run_demo(&question, ToolMode::ReAct).await?;
        }
        "compare" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            compare_demo(&question).await?;
        }
        "trace" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            trace_demo(&question).await?;
        }
        "call" => {
            let tool = arg_or_default(2, "get_weather");
            let args = std::env::args()
                .nth(3)
                .unwrap_or_else(|| r#"{"location":"北京"}"#.to_string());
            call_demo(&tool, &args).await?;
        }
        "tools" => tools_demo()?,
        "mcp" => mcp_demo().await?,
        "parse" => {
            let sample = arg_or_default(
                2,
                "Thought: 查天气\nAction: get_weather\nAction Input: {\"location\": \"北京\"}",
            );
            parse_demo(&sample);
        }
        "serve" => serve().await?,
        "all" => {
            tools_demo()?;
            call_demo("get_weather", r#"{"location":"北京"}"#).await?;
            parse_demo(
                "Thought: 查天气\nAction: get_weather\nAction Input: {\"location\": \"北京\"}",
            );
            run_demo(DEFAULT_QUESTION, ToolMode::ReAct).await?;
            compare_demo(DEFAULT_QUESTION).await?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: run / compare / trace / call / tools / mcp / parse / serve / all）"
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

/// 构建工具注册表（含可选的 MCP 工具）。
async fn build_registry(config: &AppConfig) -> Result<(ToolRegistry, Option<Arc<McpClient>>)> {
    let weather = WeatherTool::new(
        config.tool.weather_default_days,
        config.tool.weather_max_days,
    );

    let mut tools: Vec<Arc<dyn Tool>> = vec![Arc::new(weather)];
    let mut mcp_client = None;

    // 连接 MCP 服务端（可选）。
    if config.tool.enable_mcp && !config.tool.mcp_endpoint.trim().is_empty() {
        let client = Arc::new(McpClient::new(&config.tool.mcp_endpoint));

        match client.list_tools().await {
            Ok(definitions) => {
                println!(
                    "MCP 已连接: {}（{} 个工具: {}）",
                    config.tool.mcp_endpoint,
                    definitions.len(),
                    definitions
                        .iter()
                        .map(|d| d.name.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                tools.extend(client.as_tools());
                mcp_client = Some(client);
                info!(
                    endpoint = %config.tool.mcp_endpoint,
                    tool_count = definitions.len(),
                    "MCP 已连接"
                );
            }
            Err(e) => {
                // MCP 连不上**不是致命错误** —— 本地工具还能用。
                eprintln!("⚠ MCP 连接失败（本地工具仍可用）: {e}");
                warn!(endpoint = %config.tool.mcp_endpoint, error = %e, "MCP 连接失败（本地工具仍可用）");
            }
        }
    }

    let registry = ToolRegistry::new(tools);

    // 工具白名单（动态工具选择）。
    let registry = if config.tool.allowed_tools.is_empty() {
        registry
    } else {
        let names: Vec<&str> = config
            .tool
            .allowed_tools
            .iter()
            .map(String::as_str)
            .collect();
        registry.allow_names(&names)
    };

    info!(
        tool_count = registry.len(),
        tools = ?registry.names(),
        mcp = mcp_client.is_some(),
        "工具注册表就绪"
    );

    Ok((registry, mcp_client))
}

/// 构建 Agent。
async fn build_agent(config: &AppConfig) -> Result<(Agent, Option<Arc<McpClient>>)> {
    let (registry, mcp_client) = build_registry(config).await?;

    let llm = agent_14::llm::make_llm(&config.model, config.temperature, config.max_tokens);

    let agent = Agent::new(llm, registry, config.clone());

    Ok((agent, mcp_client))
}

/// 跑一次 Agent。
async fn run_demo(question: &str, mode: ToolMode) -> Result<()> {
    let config = AppConfig::from_env();
    println!("\n=== Agent 运行（模式: {}）===", mode.name());
    println!("{}", config.summary());

    let (agent, _) = build_agent(&config).await?;

    let run = agent.run_with_mode(question, mode).await?;

    println!("--- ReAct 轨迹 ---");
    for step in &run.steps {
        println!("\n[步骤 {}]", step.step);
        if !step.thought.is_empty() {
            println!("  Thought     : {}", step.thought);
        }
        if let Some(tool) = &step.tool {
            let args = step
                .args
                .as_ref()
                .map(|a| serde_json::to_string(a).unwrap_or_default())
                .unwrap_or_default();
            println!("  Action      : {tool}");
            println!("  Action Input: {args}");
            if let Some(err) = &step.parse_error {
                println!("  ⚠ 参数解析  : {err}");
            }
            if let Some(obs) = &step.observation {
                println!("  Observation  :");
                for line in obs.lines() {
                    println!("    {line}");
                }
            }
            if step.tool_ok == Some(false) {
                println!("  ⚠ 工具执行失败（错误已作为 Observation 喂回模型）");
            }
            if step.direct == Some(true) {
                println!("  ⚡ 直通结束（工具输出直接作为最终答案）");
            }
        }
        // 原始输出 —— 排查「模型说了什么 vs 我们解析成了什么」时最有用。
        if std::env::var_os("TOOL_SHOW_RAW").is_some() {
            println!("  --- 原始输出 ---");
            for line in step.raw_output.lines() {
                println!("    | {line}");
            }
        }
    }

    println!("\n--- 最终答案 ---");
    println!("{}", run.answer);
    println!(
        "\n工具调用 {} 次 · 共 {} 步 · 用时 {}ms{}",
        run.tool_calls,
        run.steps.len(),
        run.elapsed_ms,
        if run.hit_max_steps {
            " · ⚠ 达到步数上限"
        } else {
            ""
        }
    );

    info!(
        mode = mode.name(),
        tool_calls = run.tool_calls,
        tools_used = ?run.tools_used,
        steps = run.steps.len(),
        hit_max_steps = run.hit_max_steps,
        elapsed_ms = run.elapsed_ms,
        "Agent 演示完成"
    );

    Ok(())
}

/// 三模式消融对照。
async fn compare_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    println!("\n=== 三模式消融对照 ★ ===");
    println!("问题: {question}\n");

    let (agent, _) = build_agent(&config).await?;

    for mode in ToolMode::all() {
        info!(mode = mode.name(), "消融：开始跑模式");
        let run = agent.run_with_mode(question, mode).await?;

        debug!(
            mode = mode.name(),
            tool_calls = run.tool_calls,
            tools_used = ?run.tools_used,
            steps = run.steps.len(),
            elapsed_ms = run.elapsed_ms,
            "消融：模式跑完"
        );

        println!("──────────────────────────────────────────");
        println!(
            "模式      : {}  (工具: {}, 思考: {})",
            mode.name(),
            if mode.uses_tools() { "有" } else { "无" },
            if mode.uses_thought() { "有" } else { "无" }
        );
        println!(
            "工具调用  : {} 次 · 步数 {} · 用时 {}ms{}",
            run.tool_calls,
            run.steps.len(),
            run.elapsed_ms,
            if run.hit_max_steps {
                " · 达到上限"
            } else {
                ""
            }
        );
        println!("答案      : {}", run.answer.replace('\n', " "));

        // 打印每一步调了什么工具 —— 这是对照实验最关键的证据。
        for step in &run.steps {
            if let Some(tool) = &step.tool {
                let args = step
                    .args
                    .as_ref()
                    .map(|a| serde_json::to_string(a).unwrap_or_default())
                    .unwrap_or_default();
                println!("    → {tool} {args}");
            }
        }
        println!();
    }

    Ok(())
}

/// 打印完整 ReAct 轨迹。
async fn trace_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let (agent, _) = build_agent(&config).await?;

    let run = agent.run_with_mode(question, ToolMode::ReAct).await?;

    println!("=== 完整 ReAct 轨迹 ===\n");
    println!("{}", run.trajectory());
    println!("--- 最终答案 ---\n{}", run.answer);

    Ok(())
}

/// 直接调用工具（不经模型）。
async fn call_demo(tool: &str, args_json: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let (registry, _) = build_registry(&config).await?;

    let args: serde_json::Value = match serde_json::from_str(args_json) {
        Ok(value) => value,
        Err(e) => anyhow::bail!("参数不是合法 JSON: {e}"),
    };

    println!("\n=== 直接调用工具 ===");
    println!("工具: {tool}");
    println!("参数: {args_json}\n");

    match registry
        .call_strict(tool, &args, &ToolContext::empty())
        .await
    {
        Ok(outcome) => {
            println!("--- 结果 ---");
            println!("{}", outcome.observation);
            if let Some(data) = &outcome.data {
                println!("\n--- 结构化数据 ---");
                println!("{}", serde_json::to_string_pretty(data)?);
            }
        }
        Err(e) => {
            // ⚠️ 这里是**用户**扮演模型角色 —— 所以错误原样返回。
            println!("--- 错误 ---");
            println!("{e}");
        }
    }

    Ok(())
}

/// 工具清单（不调模型、不联网）。
fn tools_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let weather = WeatherTool::new(
        config.tool.weather_default_days,
        config.tool.weather_max_days,
    );

    let spec = weather.spec();

    println!("\n=== 工具清单 ===");
    println!("[{}]", spec.name);
    println!("描述  : {}", spec.description);
    println!("Schema:");
    println!("{}", serde_json::to_string_pretty(&spec.schema)?);
    println!("渲染成提示词里的参数说明:");
    for line in spec.render_parameters().lines() {
        println!("  {line}");
    }

    println!("\n=== 参数校验演示（工具契约）===");
    for args in [
        serde_json::json!({}),
        serde_json::json!({ "location": "北京", "days": 30 }),
        serde_json::json!({ "location": "北京", "units": "kelvin" }),
        serde_json::json!({ "location": "北京", "extra": 1 }),
    ] {
        let outcome = registry_demo(&weather, &args);
        println!("\n参数: {args}");
        println!("结果: {outcome}");
    }

    Ok(())
}

/// 用一个工具跑一遍「校验 + 执行」，返回观察文本。
///
/// ## ⚠️ 这里**不能**用 `Handle::block_on`
///
/// `main` 是 `#[tokio::main]` —— 当前线程**就是** runtime 线程。
/// 在 runtime 线程里 `block_on` 会 panic：
/// 「Cannot start a runtime from within a runtime」。
///
/// 所以这个 demo 走**同步的注册表路径**（`call_strict` 的内部逻辑是纯计算，
/// 只有天气查询本身要发 HTTP，而 demo 里全是参数非法的用例，
/// 会在校验阶段就返回，根本不会发请求）。
fn registry_demo(tool: &WeatherTool, args: &serde_json::Value) -> String {
    let spec = tool.spec();
    let mut object = match args {
        serde_json::Value::Object(map) => map.clone(),
        _ => serde_json::Map::new(),
    };
    agent_14::schema::fill_defaults(&spec.schema, &mut object);
    let validation =
        agent_14::schema::validate(&spec.schema, &serde_json::Value::Object(object.clone()));

    if validation.is_ok() {
        // 参数合法 —— 这一步会真的联网，所以这里只提示不执行。
        return format!(
            "参数合法（会真的发请求）：{}",
            serde_json::to_string(&serde_json::Value::Object(object)).unwrap_or_default()
        );
    }

    validation.message()
}

/// 只测试输出解析器。
fn parse_demo(sample: &str) {
    println!("\n=== 输出解析器 ===");
    println!("--- 输入 ---\n{sample}\n--- 解析结果 ---");

    match agent_14::agent::parse_react_output(sample) {
        agent_14::agent::ParsedOutput::FinalAnswer { text } => {
            println!("类型    : final_answer");
            println!("答案    : {text}");
        }
        agent_14::agent::ParsedOutput::Action {
            thought,
            tool,
            args,
            parse_error,
        } => {
            println!("类型    : action");
            println!("思考    : {thought}");
            println!("工具    : {tool}");
            println!(
                "参数    : {}",
                serde_json::to_string_pretty(&args).unwrap_or_default()
            );
            if let Some(err) = parse_error {
                println!("⚠ 容错  : {err}");
            }
        }
        agent_14::agent::ParsedOutput::Unstructured { text } => {
            println!("类型    : unstructured（容错当成答案）");
            println!("内容    : {text}");
        }
    }
}

/// 启动内置 MCP 服务器。
async fn mcp_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let state = agent_14::mcp_server::McpServerState::new();
    let app = agent_14::mcp_server::router(state);

    let listener = tokio::net::TcpListener::bind(&config.mcp_bind_addr).await?;
    println!("=== 14-agent-tool 内置 MCP 服务器 ===");
    println!("监听地址: http://{}", config.mcp_bind_addr);
    info!(addr = %config.mcp_bind_addr, "MCP 服务器已启动");
    println!("\n端点:");
    println!("  POST /mcp      JSON-RPC 2.0（initialize / tools/list / tools/call）");
    println!("  GET  /health   健康检查");
    println!("\n用 curl 试试:");
    println!(
        r#"  curl -s -X POST http://{}/mcp -H 'Content-Type: application/json' -d '{{"jsonrpc":"2.0","id":1,"method":"tools/list"}}'"#,
        config.mcp_bind_addr
    );

    axum::serve(listener, app).await?;
    Ok(())
}

/// 启动 HTTP 服务。
async fn serve() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 14-agent-tool ===\n{}", config.summary());

    let (agent, mcp_client) = build_agent(&config).await?;

    let state = Arc::new(AppStateInner::new(
        config.clone(),
        Arc::new(agent),
        mcp_client,
    ));
    let app = agent_14::api::routes::router(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    println!("监听地址: http://{}", config.bind_addr);
    info!(addr = %config.bind_addr, "HTTP 服务已启动");
    println!("\n端点:");
    println!("  GET  /              服务信息 + 工具清单");
    println!("  GET  /health        健康检查");
    println!("  GET  /api/tools     工具清单（名字 / 描述 / Schema）");
    println!("  POST /api/tools/call 直接调用工具（不经模型）★");
    println!("  POST /api/run       跑一次 Agent");
    println!("  POST /api/compare   三模式消融对照 ★");
    println!("  POST /api/parse     只解析模型输出");

    axum::serve(listener, app).await?;
    Ok(())
}
