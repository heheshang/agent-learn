//! 13-agent-core —— 手撕 Agent 核心概念（ReAct + 工具 + 记忆 + 规划）。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- run "Agent 的三大组件是什么"   # 跑一次 Agent（ReAct 模式）
//! cargo run -- compare "Agent 的三大组件是什么" # 三模式消融对照 ★
//! cargo run -- trace "Agent 的三大组件是什么"   # 打印完整 ReAct 轨迹
//! cargo run -- tools                      # 工具说明（不调模型）
//! cargo run -- parse "Action: search..."  # 只测试输出解析器
//! cargo run -- serve                      # 启动 HTTP 服务（:3060）
//! cargo run -- all                        # 全部（需要 API key）
//! ```
//!
//! ⚠️ 默认问题**故意不等于** few-shot 示例里的问题（MRKL 那条）——
//! 否则模型会直接抄示例的答案，消融实验就失真了。

use agent_logging::{debug, info, init};
use std::sync::Arc;

use agent_13::agent::Agent;
use agent_13::api::AppStateInner;
use agent_13::config::{AgentMode, AppConfig};
use agent_13::tools::{CalculateTool, SearchTool, Tool, build_index};
use anyhow::Result;

/// 默认演示问题。
///
/// ⚠️ 不能用 few-shot 示例里的问题（MRKL）—— 模型会直接抄示例答案，
/// 三模式对照就变成了「示例 vs 示例」，测不出任何东西。
const DEFAULT_QUESTION: &str = "Agent 的三大组件是什么？";

#[tokio::main]
async fn main() -> Result<()> {
    init("agent-13");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "run" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            run_demo(&question, AgentMode::ReAct).await?;
        }
        "compare" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            compare_demo(&question).await?;
        }
        "trace" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            trace_demo(&question).await?;
        }
        "tools" => tools_demo()?,
        "parse" => {
            let sample = arg_or_default(
                2,
                "Thought: 需要查资料\nAction: search\nAction Input: {\"query\": \"ReAct\"}",
            );
            parse_demo(&sample);
        }
        "serve" => serve().await?,
        "all" => {
            tools_demo()?;
            parse_demo("Thought: 需要查资料\nAction: search\nAction Input: {\"query\": \"ReAct\"}");
            run_demo(DEFAULT_QUESTION, AgentMode::ReAct).await?;
            compare_demo(DEFAULT_QUESTION).await?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: run / compare / trace / tools / parse / serve / all）"
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

/// 构建 Agent（含工具）。
async fn build_agent(config: &AppConfig) -> Result<(Agent, usize)> {
    // BM25 索引：只需要知识库目录，不需要嵌入模型。
    let index = build_index(&config.data_dir)?;
    let chunks = index.len();

    let search = SearchTool::new(
        Arc::new(index),
        config.agent.search_top_k,
        config.tool.max_top_k,
        config.agent.max_observation_chars,
    );
    let calculate = CalculateTool;

    let llm = agent_13::llm::make_llm(&config.model, config.temperature, config.max_tokens);

    let agent = Agent::new(
        llm,
        vec![Arc::new(search), Arc::new(calculate)],
        config.clone(),
    );

    Ok((agent, chunks))
}

/// 跑一次 Agent。
async fn run_demo(question: &str, mode: AgentMode) -> Result<()> {
    let config = AppConfig::from_env();
    println!("\n=== Agent 运行（模式: {}）===", mode.name());
    println!("{}", config.summary());

    let (agent, chunks) = build_agent(&config).await?;
    println!("知识库块数: {chunks}\n");

    let run = agent.run_with_mode(question, mode).await?;

    println!("--- ReAct 轨迹 ---");
    for step in &run.steps {
        println!("\n[步骤 {}]", step.step);
        if !step.thought.is_empty() {
            println!("  Thought    : {}", step.thought);
        }
        if let Some(tool) = &step.tool {
            let args = step
                .args
                .as_ref()
                .map(|a| serde_json::to_string(a).unwrap_or_default())
                .unwrap_or_default();
            println!("  Action     : {tool}");
            println!("  Action Input: {args}");
            if let Some(err) = &step.parse_error {
                println!("  ⚠ 参数解析  : {err}");
            }
            if let Some(obs) = &step.observation {
                println!("  Observation :");
                for line in obs.lines() {
                    println!("    {line}");
                }
            }
            if step.tool_ok == Some(false) {
                println!("  ⚠ 工具执行失败（错误已作为 Observation 喂回模型）");
            }
        }
        // 原始输出 —— 排查「模型说了什么 vs 我们解析成了什么」时最有用。
        if std::env::var_os("AGENT_SHOW_RAW").is_some() {
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

    for mode in AgentMode::all() {
        info!(mode = mode.name(), "消融：开始跑模式");
        let run = agent.run_with_mode(question, mode).await?;

        debug!(
            mode = mode.name(),
            tool_calls = run.tool_calls,
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

    let run = agent.run_with_mode(question, AgentMode::ReAct).await?;

    println!("=== 完整 ReAct 轨迹 ===\n");
    println!("{}", run.trajectory());
    println!("--- 最终答案 ---\n{}", run.answer);

    Ok(())
}

/// 工具说明（不调模型）。
fn tools_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let index = build_index(&config.data_dir)?;

    println!("\n=== 工具清单 ===");
    println!(
        "知识库: {} ({} 个块)",
        config.data_dir.display(),
        index.len()
    );

    let search = SearchTool::new(
        Arc::new(index),
        config.agent.search_top_k,
        config.tool.max_top_k,
        config.agent.max_observation_chars,
    );

    println!("\n[1] {}", search.spec().name);
    println!("    描述: {}", search.spec().description);
    println!(
        "    Schema: {}",
        serde_json::to_string_pretty(&search.spec().schema)?
    );

    println!("\n[2] {}", CalculateTool.spec().name);
    println!("    描述: {}", CalculateTool.spec().description);
    println!(
        "    Schema: {}",
        serde_json::to_string_pretty(&CalculateTool.spec().schema)?
    );

    // 演示一次真实的工具调用（不调模型）。
    println!("\n=== 工具实测 ===");
    let outcome = search.run(&serde_json::json!({ "query": "ReAct Observation", "top_k": 2 }));
    println!("search 返回:\n{}", outcome.observation);

    let outcome = CalculateTool.run(&serde_json::json!({ "expression": "(12 + 34) * 2" }));
    println!("\ncalculate 返回: {}", outcome.observation);

    Ok(())
}

/// 只测试输出解析器。
fn parse_demo(sample: &str) {
    println!("\n=== 输出解析器 ===");
    println!("--- 输入 ---\n{sample}\n--- 解析结果 ---");

    match agent_13::agent::parse_react_output(sample) {
        agent_13::agent::ParsedOutput::FinalAnswer { text } => {
            println!("类型    : final_answer");
            println!("答案    : {text}");
        }
        agent_13::agent::ParsedOutput::Action {
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
        agent_13::agent::ParsedOutput::Unstructured { text } => {
            println!("类型    : unstructured（容错当成答案）");
            println!("内容    : {text}");
        }
    }
}

/// 启动 HTTP 服务。
async fn serve() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 13-agent-core ===\n{}", config.summary());

    let (agent, chunks) = build_agent(&config).await?;
    println!("知识库块数: {chunks}\n");

    let state = Arc::new(AppStateInner::new(config.clone(), Arc::new(agent), chunks));
    let app = agent_13::api::routes::router(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    println!("监听地址: http://{}", config.bind_addr);
    info!(addr = %config.bind_addr, chunks, "HTTP 服务已启动");
    println!("\n端点:");
    println!("  GET  /              服务信息 + 工具清单");
    println!("  GET  /health        健康检查");
    println!("  POST /api/run       跑一次 Agent");
    println!("  POST /api/compare   三模式消融对照 ★");
    println!("  POST /api/parse     只解析模型输出");

    axum::serve(listener, app).await?;
    Ok(())
}
