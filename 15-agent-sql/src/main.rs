//! 15-agent-sql —— 手撕 SQL Agent（LangChain SQL Agent）。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- schema                # 数据库概览（不调模型）
//! cargo run -- guard "DELETE FROM Track"  # 只跑护栏（不执行）★
//! cargo run -- call sql_db_query '{"query":"SELECT Name FROM Genre LIMIT 5"}'  # 直接调工具 ★
//! cargo run -- run "哪个流派的曲目平均时长最长？"   # 跑一次 Agent
//! cargo run -- trace "哪个流派的曲目平均时长最长？" # 打印完整轨迹
//! cargo run -- serve                 # 启动 HTTP 服务（:3080）
//! cargo run -- all                   # 全部
//! ```
//!
//! ⚠️ 默认问题**故意不等于** few-shot 示例里的问题
//! （示例问「有哪些流派」，默认问「平均时长最长」）——
//! 否则模型会直接抄示例的答案。

use std::path::PathBuf;
use std::sync::Arc;

use agent_14::tool::{Tool, ToolContext, ToolRegistry};
use agent_logging::{debug, info, init, warn};
use anyhow::Result;

use agent_15::agent::Agent;
use agent_15::api::AppStateInner;
use agent_15::checker::{LlmChecker, QueryChecker, StaticChecker};
use agent_15::config::{AppConfig, CheckerMode};
use agent_15::db::{self, Database};
use agent_15::tools::{ListTablesTool, QueryCheckerTool, QueryTool, SchemaTool};

/// 默认演示问题。
///
/// ⚠️ 不能用 few-shot 示例里的问题（「有哪些流派」）——
/// 模型会直接抄示例答案。
const DEFAULT_QUESTION: &str = "哪个流派的曲目平均时长最长？";

#[tokio::main]
async fn main() -> Result<()> {
    init("agent-15");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "run" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            run_demo(&question).await?;
        }
        "trace" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            trace_demo(&question).await?;
        }
        "call" => {
            let tool = arg_or_default(2, "sql_db_query");
            let args = std::env::args()
                .nth(3)
                .unwrap_or_else(|| r#"{"query":"SELECT Name FROM Genre LIMIT 5"}"#.to_string());
            call_demo(&tool, &args).await?;
        }
        "guard" => {
            let sql = arg_or_default(2, "DELETE FROM Track");
            guard_demo(&sql);
        }
        "schema" => schema_demo()?,
        "seed" => seed_demo()?,
        "serve" => serve().await?,
        "all" => {
            schema_demo()?;
            guard_demo("DELETE FROM Track");
            guard_demo("SELECT Name FROM Genre LIMIT 5");
            call_demo(
                "sql_db_query",
                r#"{"query":"SELECT Name FROM Genre LIMIT 5"}"#,
            )
            .await?;
            run_demo(DEFAULT_QUESTION).await?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: run / trace / call / guard / schema / seed / serve / all）"
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

/// 打开数据库（必要时自动生成种子库）。
fn open_database(config: &AppConfig) -> Result<Arc<Database>> {
    let db_path = PathBuf::from(&config.db.path);
    let seed_path = PathBuf::from(&config.db.seed_path);

    if !db_path.exists() {
        if config.db.auto_seed {
            info!(
                db = %db_path.display(),
                seed = %seed_path.display(),
                "数据库不存在 —— 用种子 SQL 生成"
            );
            db::load_seed(&seed_path, &db_path)?;
        } else {
            anyhow::bail!(
                "数据库文件不存在: {}（关掉 SQL_AUTO_SEED，或先跑 `make seed`）",
                db_path.display()
            );
        }
    }

    Ok(Arc::new(Database::open_readonly(&db_path)?))
}

/// 构建工具注册表。
fn build_registry(config: &AppConfig, db: Arc<Database>) -> Result<ToolRegistry> {
    let checker: Arc<dyn QueryChecker> = match config.checker {
        CheckerMode::Static => Arc::new(StaticChecker::new()),
        CheckerMode::Llm => {
            let llm = agent_15::llm::make_llm(&config.model, config.temperature, config.max_tokens);
            Arc::new(LlmChecker::new(llm))
        }
    };

    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ListTablesTool::new(db.clone())),
        Arc::new(SchemaTool::new(db.clone())),
        Arc::new(QueryTool::new(db).with_max_rows(config.db.max_rows)),
        Arc::new(QueryCheckerTool::new(checker)),
    ];

    let registry = ToolRegistry::new(tools);
    info!(
        tool_count = registry.len(),
        tools = ?registry.names(),
        checker = config.checker.name(),
        "工具注册表就绪"
    );

    Ok(registry)
}

/// 构建 Agent。
fn build_agent(config: &AppConfig) -> Result<(Arc<Agent>, Arc<Database>)> {
    let db = open_database(config)?;
    let registry = build_registry(config, db.clone())?;
    let llm = agent_15::llm::make_llm(&config.model, config.temperature, config.max_tokens);
    Ok((Arc::new(Agent::new(llm, registry, config.clone())), db))
}

/// 跑一次 Agent。
async fn run_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    println!("\n=== SQL Agent 运行 ===\n{}", config.summary());

    let (agent, _) = build_agent(&config)?;

    let run = agent.run(question).await?;

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
        }
        if std::env::var_os("SQL_SHOW_RAW").is_some() {
            println!("  --- 原始输出 ---");
            for line in step.raw_output.lines() {
                println!("    | {line}");
            }
        }
    }

    println!("\n--- 最终答案 ---");
    println!("{}", run.answer);

    // ⚠️ 「答案有没有数据支撑」必须显式告诉用户 ——
    // 一个看起来专业的编造答案比明显的错误更危险。
    if !run.grounded {
        println!("\n⚠️ **这个答案没有真实数据支撑** —— 没有一条查询被成功执行过。");
        println!("   里面的数字可能是模型凭空编的，请谨慎使用。");
    }
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

    // ── 安全审计摘要 ────────────────────────────────────────
    //
    // 这是本项目的特色输出 ——
    // 让用户一眼看到「这个 Agent 对数据库做了什么」。
    println!("\n--- 安全审计 ---");
    if run.executed_sql.is_empty() {
        println!("  执行过的 SQL: （无）");
    } else {
        for sql in &run.executed_sql {
            println!("  ✅ {sql}");
        }
    }
    if run.blocked_sql.is_empty() {
        println!("  被护栏拦截:   （无）");
    } else {
        for sql in &run.blocked_sql {
            println!("  🚫 {sql}");
        }
    }

    info!(
        tool_calls = run.tool_calls,
        tools_used = ?run.tools_used,
        executed = run.executed_sql.len(),
        blocked = run.blocked_sql.len(),
        steps = run.steps.len(),
        elapsed_ms = run.elapsed_ms,
        "Agent 演示完成"
    );

    Ok(())
}

/// 打印完整 ReAct 轨迹。
async fn trace_demo(question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let (agent, _) = build_agent(&config)?;

    let run = agent.run(question).await?;

    println!("=== 完整 ReAct 轨迹 ===\n");
    println!("{}", run.trajectory());
    println!("--- 最终答案 ---\n{}", run.answer);

    Ok(())
}

/// 直接调用工具（不经模型）。
async fn call_demo(tool: &str, args_json: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let db = open_database(&config)?;
    let registry = build_registry(&config, db)?;

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

/// 只跑护栏（不执行）。
fn guard_demo(sql: &str) {
    println!("\n=== SQL 护栏检查 ===");
    println!("SQL: {sql}\n");

    match agent_15::guard::check_sql(sql) {
        Ok(()) => {
            println!("✅ 允许执行（只读查询）");
        }
        Err(e) => {
            println!("🚫 拦截：{}", e.message());
        }
    }
}

/// 数据库概览（不调模型）。
fn schema_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let db = open_database(&config)?;

    println!("\n=== 数据库概览 ===\n");
    println!("{}", db::overview(&db)?);

    Ok(())
}

/// 生成种子数据库。
fn seed_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let db_path = PathBuf::from(&config.db.path);
    let seed_path = PathBuf::from(&config.db.seed_path);

    println!("\n=== 生成种子数据库 ===");
    db::load_seed(&seed_path, &db_path)?;

    let db = Database::open_readonly(&db_path)?;
    println!("{}", db::overview(&db)?);

    Ok(())
}

/// 启动 HTTP 服务。
async fn serve() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 15-agent-sql ===\n{}", config.summary());

    let (agent, db) = build_agent(&config)?;

    let state = Arc::new(AppStateInner::new(config.clone(), agent, db));
    let app = agent_15::api::routes::router(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    println!("监听地址: http://{}", config.bind_addr);
    info!(addr = %config.bind_addr, "HTTP 服务已启动");
    println!("\n端点:");
    println!("  GET  /              服务信息 + 工具清单");
    println!("  GET  /health        健康检查");
    println!("  GET  /api/tools     工具清单");
    println!("  GET  /api/schema    数据库概览 ★");
    println!("  POST /api/tools/call 直接调用工具（不经模型）★");
    println!("  POST /api/run       跑一次 Agent");
    println!("  POST /api/guard     只跑护栏（不执行）★");

    axum::serve(listener, app).await?;
    Ok(())
}

/// 让 `debug!` / `warn!` 在本文件里被引用（避免 unused import）。
#[allow(dead_code)]
fn _unused(level: &str) {
    match level {
        "debug" => debug!("x"),
        "warn" => warn!("x"),
        _ => {}
    }
}
