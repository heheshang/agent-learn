//! 19-agent-summary —— 研究助手 Agent。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- tools                 # 工具清单（不调模型）
//! cargo run -- search "rust 异步"     # ★ 直接调 Web 搜索工具
//! cargo run -- fetch "https://..."    # ★ 直接调网页抓取工具
//! cargo run -- research "本周进展"     # 只跑研究阶段（看证据）
//! cargo run -- weekly "本周重点"       # ★ 周度总结（研究 + 成文 + 校验）
//! cargo run -- plan "做一个研究助手"   # ★ 项目构建方案
//! cargo run -- compare                # ★ 接地校验对照
//! cargo run -- serve                 # 启动 HTTP 服务（:3120）
//! cargo run -- all                   # 全部
//! ```

use std::sync::Arc;

use agent_logging::{info, init, warn};
use anyhow::Result;

use agent_18::middleware::ErrorHandlingMiddleware;
use agent_19::agent::{Agent, AgentConfig};
use agent_19::api::AppStateInner;
use agent_19::config::AppConfig;
use agent_19::evidence::EvidenceCollector;
use agent_19::llm::GenaiFunctionLlm;
use agent_19::pipeline::{PlanPipeline, WeeklyPipeline};
use agent_19::search::WebBackend;
use agent_19::tools::MemoryTool;

/// 默认研究问题。
const DEFAULT_QUESTION: &str = "本项目在错误处理上做了什么设计决策？";

/// 默认周报关注点。
const DEFAULT_WEEKLY: &str = "错误处理模块与工具层的进展";

/// 默认项目需求。
const DEFAULT_REQUIREMENT: &str = "用 Rust + axum 构建一个集成 RAG 与 Web 搜索的研究助手";

#[tokio::main]
async fn main() -> Result<()> {
    init("agent-19");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "tools" => tools_demo()?,
        "search" => {
            let query = arg_or_default(2, "rust 异步");
            search_demo(&query).await?;
        }
        "fetch" => {
            let url = arg_or_default(2, "https://doc.rust-lang.org/book/");
            fetch_demo(&url).await?;
        }
        "research" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            research_demo(&question).await?;
        }
        "weekly" => {
            let question = arg_or_default(2, DEFAULT_WEEKLY);
            weekly_demo(&question).await?;
        }
        "plan" => {
            let requirement = arg_or_default(2, DEFAULT_REQUIREMENT);
            plan_demo(&requirement).await?;
        }
        "compare" => compare_demo()?,
        "serve" => serve().await?,
        "all" => {
            tools_demo()?;
            compare_demo()?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: tools / search / fetch / research / weekly / plan / compare / serve / all）"
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

/// 构建知识库索引。
///
/// ## ⚠️ 知识库不存在时**不报错**
///
/// 而是用空索引继续 —— 因为「知识库为空」是一个合法的状态
/// （新项目还没积累资料），不该让整个服务起不来。
fn build_index(config: &AppConfig) -> Arc<rag_08::bm25::Bm25Index> {
    if !config.kb_dir.exists() {
        warn!(
            dir = %config.kb_dir.display(),
            "知识库目录不存在 —— 用空索引继续（Web 搜索仍然可用）"
        );
        return Arc::new(rag_08::bm25::Bm25Index::build(&[]));
    }

    let splitter_config = rag_04::splitter::SplitterConfig::default();
    match rag_08::index::build_bm25(&config.kb_dir, &splitter_config) {
        Ok(index) => {
            info!(docs = index.len(), "知识库索引已构建");
            Arc::new(index)
        }
        Err(e) => {
            warn!(error = %e, "构建知识库索引失败 —— 用空索引继续");
            Arc::new(rag_08::bm25::Bm25Index::build(&[]))
        }
    }
}

/// 构建记忆存储 + 检索器。
fn build_memory(
    owner: &str,
) -> (
    Arc<dyn agent_17::longterm::MemoryStore>,
    Arc<dyn agent_17::longterm::MemoryRetriever>,
) {
    let store: Arc<dyn agent_17::longterm::MemoryStore> =
        Arc::new(agent_17::longterm::InMemoryMemoryStore::new());
    let retriever: Arc<dyn agent_17::longterm::MemoryRetriever> =
        Arc::new(agent_17::longterm::Bm25MemoryRetriever::new(store.clone()));
    let _ = owner;
    (store, retriever)
}

/// 构建 Web 后端。
fn build_backend(config: &AppConfig) -> Arc<dyn WebBackend> {
    match agent_19::search::build_backend(config) {
        Ok(b) => Arc::from(b),
        Err(e) => {
            warn!(error = %e, "构造 Web 后端失败 —— 退化为离线后端");
            Arc::new(agent_19::search::MockBackend::new(Vec::new()).failing())
        }
    }
}

/// 构建工具注册表。
fn build_registry(config: &AppConfig) -> agent_18::tool::ToolRegistry {
    let index = build_index(config);
    let backend = build_backend(config);
    let (store, retriever) = build_memory("default");

    let memory = MemoryTool::new(store, retriever, "default", 5, 20);

    agent_19::tools::default_registry(
        index,
        backend,
        memory,
        config.search_top_k,
        config.max_evidence_per_section,
    )
}

/// 构建中间件。
fn build_middleware(config: &AppConfig) -> ErrorHandlingMiddleware {
    ErrorHandlingMiddleware::new(build_registry(config))
        .with_max_retries(config.max_retries)
        .with_base_delay(std::time::Duration::from_millis(config.base_delay_ms))
        .with_max_delay(std::time::Duration::from_millis(config.max_delay_ms))
}

/// 构建 LLM 客户端。
fn build_llm(config: &AppConfig) -> Arc<dyn agent_19::llm::FunctionLlm> {
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
        agent_19::prompt::render_tool_catalog(&registry.specs())
    );

    Ok(())
}

/// ★ 直接调 Web 搜索工具。
///
/// ## ⚠️ 为什么走**工具**而不是直接调后端
///
/// 因为工具层会做**实体解码**（`&ensp;` → 空格）。
/// 直接调后端看到的原始 HTML 实体，和模型实际看到的不一样 ——
/// 那样演示就失真了。
async fn search_demo(query: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let middleware = build_middleware(&config);

    println!("\n=== 直接调用 Web 搜索工具 ===");
    println!("后端: {}", config.search_backend.name());
    println!("查询: {query}\n");

    let request = agent_18::middleware::ToolCallRequest {
        name: "web_search".to_string(),
        arguments: serde_json::json!({ "query": query, "top_k": config.search_top_k }),
        call_id: "demo".to_string(),
    };

    let outcome = middleware.wrap_tool_call(&request).await;

    if outcome.ok {
        // ⚠️ 必须剥掉 `@@DATA:` 行 ——
        //    那是给证据层用的结构化数据，不是给人看的。
        println!(
            "{}",
            agent_19::agent::render_observation(&outcome.observation)
        );
        println!("（尝试 {} 次）", outcome.attempts);
    } else {
        let error = outcome
            .error
            .unwrap_or_else(|| crate_error("未知的工具失败"));
        println!("❌ 搜索失败: {error}");
        println!("   错误码: {}", error.diagnostic_code());
        println!("   类别:   {}", error.kind.name());
        println!("   建议:   {}", error.hint.unwrap_or_default());
    }

    Ok(())
}

/// 构造一个内部错误（用于兜底）。
fn crate_error(msg: &str) -> agent_19::error::ToolError {
    agent_19::error::ToolError::internal("web_search", msg)
}

/// ★ 直接调网页抓取工具。
async fn fetch_demo(url: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let middleware = build_middleware(&config);

    println!("\n=== 直接调用网页抓取工具 ===");
    println!("链接: {url}\n");

    let request = agent_18::middleware::ToolCallRequest {
        name: "web_fetch".to_string(),
        arguments: serde_json::json!({ "url": url }),
        call_id: "demo".to_string(),
    };

    let outcome = middleware.wrap_tool_call(&request).await;

    if outcome.ok {
        println!(
            "{}",
            agent_19::agent::render_observation(&outcome.observation)
        );
    } else {
        let error = outcome
            .error
            .unwrap_or_else(|| crate_error("未知的工具失败"));
        println!("❌ 抓取失败: {error}");
        println!("   错误码: {}", error.diagnostic_code());
        println!("   类别:   {}", error.kind.name());
    }

    Ok(())
}

/// 只跑研究阶段。
async fn research_demo(question: &str) -> Result<()> {
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

    println!("\n=== 研究阶段（只拿证据，不成文）===");
    println!("问题: {question}\n");

    let system = agent_19::prompt::research_system();
    let (run, collector) = agent.research(question, &system).await?;

    print!("{}", agent_19::prompt::render_trajectory(&run.steps));

    println!("── 证据清单 ──");
    println!("{}", collector.render(config.max_evidence_per_section));

    println!(
        "\n── 统计 ──\n工具调用 {} 次（含重试）| 重试 {} 次 | 降级 {} 次 | 证据 {} 条 | 耗时 {} ms",
        run.tool_calls,
        run.retried_calls,
        run.degraded_errors,
        collector.len(),
        run.elapsed_ms,
    );

    Ok(())
}

/// ★ 周度总结。
async fn weekly_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
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

    let pipeline = WeeklyPipeline::new(agent, llm, config.max_evidence_per_section);

    println!("\n=== ★ 周度总结 ===");
    println!("关注点: {question}\n");

    let result = pipeline.run(question).await?;

    print!("{}", agent_19::prompt::render_trajectory(&result.run.steps));

    println!("── 周报 ──");
    println!("{}", result.markdown);

    let path = agent_19::pipeline::write_weekly_report(&config.output_dir, &result.markdown)?;
    println!("已写入: {}", path.display());

    Ok(())
}

/// ★ 项目构建方案。
async fn plan_demo(requirement: &str) -> Result<()> {
    let config = AppConfig::from_env();
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

    let pipeline = PlanPipeline::new(agent, llm, config.max_evidence_per_section);

    println!("\n=== ★ 项目构建方案 ===");
    println!("需求: {requirement}\n");

    let result = pipeline.run(requirement).await?;

    print!("{}", agent_19::prompt::render_trajectory(&result.run.steps));

    println!("── 方案 ──");
    println!("{}", result.markdown);

    let path = agent_19::pipeline::write_project_plan(&config.output_dir, &result.markdown)?;
    println!("已写入: {}", path.display());

    Ok(())
}

/// ★ 接地校验对照。
///
/// ## 对照的变量
///
/// **只有「模型是否引用证据」这一个变量不同** ——
/// 同一份证据清单、同一套校验代码。
///
/// 这能验证：`grounding` 三态确实能区分
/// 「有据可依」/「引用了不存在的编号」/「完全没引用」。
fn compare_demo() -> Result<()> {
    let mut collector = EvidenceCollector::new();
    collector.add(
        agent_19::evidence::EvidenceKind::Kb,
        "kb_search",
        "notes.md > 架构",
        "错误处理的降级策略是把失败转成观察喂回模型",
    );
    collector.add(
        agent_19::evidence::EvidenceKind::WebSearch,
        "web_search",
        "https://rust-lang.org/",
        "Rust Programming Language",
    );

    let cases = [
        (
            "全部引用真实证据",
            r#"{"overview":"正常","highlights":[{"text":"做了错误处理","evidence":["E1"]}],"risks":[],"next_steps":[]}"#,
        ),
        (
            "引用了不存在的编号",
            r#"{"overview":"部分悬空","highlights":[{"text":"做了错误处理","evidence":["E1"]},{"text":"编的结论","evidence":["E99"]}],"risks":[],"next_steps":[]}"#,
        ),
        (
            "一条证据都没引用",
            r#"{"overview":"完全没据","highlights":[{"text":"凭印象写的","evidence":[]}],"risks":[],"next_steps":[]}"#,
        ),
    ];

    println!("\n=== ★ 接地校验对照 ===");
    println!("（只有「模型是否引用证据」这一个变量不同）\n");

    for (label, raw) in cases {
        match agent_19::report::parse_weekly_summary(raw, &collector) {
            Ok(summary) => {
                println!("── {label} ──");
                println!("接地状态: {}", summary.grounding.name());
                println!(
                    "有效引用: {} | 悬空引用: {}",
                    summary.grounding_report.used_refs.len(),
                    summary.grounding_report.dangling_refs.len()
                );
                println!();
            }
            Err(e) => {
                println!("── {label} ──");
                println!("解析失败: {e}\n");
            }
        }
    }

    println!("── 证据清单 ──");
    println!("{}", collector.render(10));

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
    let app = agent_19::api::routes::router(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    info!("服务已启动: http://{}", config.bind_addr);

    axum::serve(listener, app).await?;

    Ok(())
}
