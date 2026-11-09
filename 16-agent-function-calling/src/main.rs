//! 16-agent-function-calling —— 手撕 Function Calling（结构化工具调用）。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- tools                      # 工具清单 + Schema（不调模型）
//! cargo run -- call get_weather '{"city":"北京"}'  # 直接调工具 ★
//! cargo run -- run "北京和上海现在天气怎么样？"    # 跑一次 Agent
//! cargo run -- trace "北京现在天气怎么样？"       # 打印结构化轨迹
//! cargo run -- compare "北京现在天气怎么样？"     # ★ tool_choice 四档对照
//! cargo run -- serve                      # 启动 HTTP 服务（:3090）
//! cargo run -- all                        # 全部
//! ```
//!
//! ⚠️ 默认问题**故意不是**「北京天气」——
//! 否则对照实验里 `none` 档会显得特别差（模型本来就知道北京天气），
//! 差值被「模型先验知识」污染。

use std::sync::Arc;

use agent_logging::{debug, info, init};
use anyhow::Result;

use agent_16::agent::Agent;
use agent_16::api::AppStateInner;
use agent_16::config::AppConfig;
use agent_16::functions::default_registry;
use agent_16::llm::{GenaiFunctionLlm, ToolChoiceMode};

/// 默认演示问题。
///
/// ⚠️ 必须是**模型不可能凭空知道**的问题 ——
/// 这样「用了工具」和「没用工具」的差值才能归因到工具，
/// 而不是「模型本来就知道」。
const DEFAULT_QUESTION: &str = "北京和上海现在天气怎么样？";

#[tokio::main]
async fn main() -> Result<()> {
    init("agent-16");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "run" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            run_demo(&question, ToolChoiceMode::Auto).await?;
        }
        "trace" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            trace_demo(&question).await?;
        }
        "compare" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            compare_demo(&question).await?;
        }
        "call" => {
            let tool = arg_or_default(2, "get_weather");
            let args = std::env::args()
                .nth(3)
                .unwrap_or_else(|| r#"{"location":"北京"}"#.to_string());
            call_demo(&tool, &args).await?;
        }
        "tools" => tools_demo()?,
        "serve" => serve().await?,
        "all" => {
            tools_demo()?;
            call_demo("calculate", r#"{"expression":"(12+34)*2"}"#).await?;
            run_demo(DEFAULT_QUESTION, ToolChoiceMode::Auto).await?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: run / trace / compare / call / tools / serve / all）"
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

/// 构建 BM25 索引（本地知识库）。
fn build_index(config: &AppConfig) -> Result<Arc<rag_08::bm25::Bm25Index>> {
    let data_dir = std::path::Path::new(&config.data_dir);

    if !data_dir.exists() {
        anyhow::bail!("数据目录不存在: {}（先跑 `make data`）", data_dir.display());
    }

    let splitter_config = rag_04::splitter::SplitterConfig::default();
    let index = rag_08::index::build_bm25(data_dir, &splitter_config)
        .map_err(|e| anyhow::anyhow!("构建知识库索引失败: {e}"))?;

    info!(
        dir = %data_dir.display(),
        documents = index.len(),
        "知识库索引构建完成"
    );

    Ok(Arc::new(index))
}

/// 构建 LLM 客户端。
fn build_llm(config: &AppConfig) -> Arc<dyn agent_16::llm::FunctionLlm> {
    info!(
        model = %config.model,
        temperature = config.temperature,
        max_tokens = config.max_tokens,
        reasoning = !config.disable_reasoning,
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
    let index = build_index(&config)?;
    let registry = default_registry(index);

    println!("\n=== 可用工具（{} 个）===\n", registry.len());

    for spec in registry.specs() {
        println!("■ {}（strict={}）", spec.name, spec.strict);
        println!("  描述: {}", spec.description.replace('\n', " "));
        println!("  参数:");
        for line in spec.render_parameters().lines() {
            println!("    {}", line);
        }
        println!("  Schema: {}", serde_json::to_string_pretty(&spec.schema)?);
        println!();
    }

    Ok(())
}

/// 直接调工具（不经模型）。
async fn call_demo(tool: &str, args_json: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let index = build_index(&config)?;
    let registry = default_registry(index);

    let args: serde_json::Value =
        serde_json::from_str(args_json).map_err(|e| anyhow::anyhow!("参数不是合法 JSON: {e}"))?;

    println!("\n=== 直接调用工具（不经模型）===");
    println!("工具: {tool}");
    println!("参数: {args}\n");

    let outcome = registry
        .call_strict(tool, &args, &agent_16::tool::ToolContext::empty())
        .await?;

    if outcome.ok {
        println!("✅ 成功\n");
        println!("{}", outcome.observation);
        if let Some(data) = &outcome.data {
            println!("\n结构化结果:\n{}", serde_json::to_string_pretty(data)?);
        }
    } else {
        println!("❌ 失败\n");
        println!("{}", outcome.observation);
    }

    Ok(())
}

/// 跑一次 Agent。
async fn run_demo(question: &str, tool_choice: ToolChoiceMode) -> Result<()> {
    let config = AppConfig::from_env();
    let index = build_index(&config)?;
    let registry = default_registry(index);
    let llm = build_llm(&config);

    let mut config = config;
    config.prompt.tool_choice = tool_choice.clone();

    let agent = Agent::new(llm, registry, config);

    println!("\n=== 跑一次 Function Calling Agent ===");
    println!("问题: {question}");
    println!("tool_choice: {}\n", tool_choice.name());

    let run = agent.run(question).await?;

    println!("── 轨迹 ──");
    print!("{}", agent_16::prompt::render_trajectory(&run.steps));

    println!("── 最终答案 ──");
    println!("{}", run.answer);

    println!(
        "\n── 统计 ──\n工具调用 {} 次 | 步数 {} | 耗时 {} ms | token {} (in) + {} (out)",
        run.tool_calls,
        run.steps.len(),
        run.elapsed_ms,
        run.usage.prompt_tokens,
        run.usage.completion_tokens,
    );
    if run.hit_max_steps {
        println!("⚠️ 达到最大步数 —— 结果是降级返回的");
    }

    Ok(())
}

/// 打印结构化轨迹（带思考内容）。
async fn trace_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let index = build_index(&config)?;
    let registry = default_registry(index);
    let llm = build_llm(&config);
    let agent = Agent::new(llm, registry, config);

    let run = agent.run(question).await?;

    println!("\n=== 完整结构化轨迹 ===\n");
    print!("{}", agent_16::prompt::render_trajectory(&run.steps));

    println!("=== 最终答案 ===\n{}", run.answer);

    Ok(())
}

/// ★ tool_choice 四档对照。
async fn compare_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let index = build_index(&config)?;
    let registry = default_registry(index);
    let llm = build_llm(&config);

    let mut modes = vec![
        ToolChoiceMode::Auto,
        ToolChoiceMode::None,
        ToolChoiceMode::Required,
    ];

    // `forced` 那一档需要知道工具名 —— 取注册表里的第一个。
    if let Some(first) = registry.names().first() {
        modes.push(ToolChoiceMode::Forced(first.clone()));
    }

    println!("\n=== tool_choice 四档对照 ===");
    println!("问题: {question}");
    println!("（只有 tool_choice 这一个变量不同）\n");

    for mode in modes {
        let mut cfg = config.clone();
        cfg.prompt.tool_choice = mode.clone();
        let agent = Agent::new(llm.clone(), registry.clone(), cfg);

        let run = agent.run(question).await?;

        println!("── {} ──", mode.name());
        println!("工具调用: {} 次 {:?}", run.tool_calls, run.tools_used);
        println!("耗时: {} ms", run.elapsed_ms);
        println!("答案: {}", truncate(&run.answer, 300));
        println!();
    }

    Ok(())
}

/// 截断长文本。
fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max).collect();
    format!("{kept}…")
}

/// 启动 HTTP 服务。
async fn serve() -> Result<()> {
    let config = AppConfig::from_env();

    info!("配置:\n{}", config.summary());

    let index = build_index(&config)?;
    let registry = default_registry(index);
    let llm = build_llm(&config);
    let agent = Arc::new(Agent::new(llm.clone(), registry, config.clone()));

    let state = Arc::new(AppStateInner::new(config.clone(), agent, llm));
    let app = agent_16::api::routes::router(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    info!("服务已启动: http://{}", config.bind_addr);

    axum::serve(listener, app).await?;

    Ok(())
}

/// 让 `debug!` 有用（避免未使用告警）。
#[allow(dead_code)]
fn _debug_marker() {
    debug!("agent-16");
}
