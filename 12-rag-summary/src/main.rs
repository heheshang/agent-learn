//! CLI 入口 —— 四种运行模式。
//!
//! | 子命令 | 说明 |
//! |---|---|
//! | `serve` | 启动 HTTP 服务（默认） |
//! | `demo` | 命令行跑一个问题的完整问答 |
//! | `compare` | 同题消融对照（阶梯各档） |
//! | `eval` | 跑评估集，产出消融报告 |
//!
//! ## 为什么要有 `compare` 和 `eval` 两个「重活」命令
//!
//! - `compare`：**同一个问题**横向对比各档策略 —— 快（只有检索），适合调试
//! - `eval`：**整个评估集**纵向跑指标 —— 慢（要生成 + judge），适合出报告
//!
//! 分开是因为它们的耗时差了两个数量级 ——
//! 混在一起会让「调试时想快速看一眼」变得很痛苦。

use std::process::ExitCode;
use std::sync::Arc;

use rag_12::api::{AppState, AppStateInner};
use rag_12::bootstrap;
use rag_12::config::{AppConfig, SearchMode};

/// CLI 参数。
#[derive(Debug, Clone)]
struct Cli {
    /// 子命令。
    command: Command,
    /// 演示/对照用的��题。
    question: Option<String>,
    /// 演示/对照用的模式。
    mode: Option<String>,
}

/// 子命令。
#[derive(Debug, Clone)]
enum Command {
    /// 启动 HTTP 服务。
    Serve,
    /// 命令行问答演示。
    Demo,
    /// 同题消融对照。
    Compare,
    /// 跑评估集。
    Eval,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = match parse_args() {
        Ok(cli) => cli,
        Err(message) => {
            eprintln!("错误: {message}\n");
            print_usage();
            return ExitCode::from(2);
        }
    };

    let config = AppConfig::from_env();

    match run(cli, config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\n运行失败: {e}");
            ExitCode::FAILURE
        }
    }
}

/// 解析命令行参数。
fn parse_args() -> Result<Cli, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        return Ok(Cli {
            command: Command::Serve,
            question: None,
            mode: None,
        });
    }

    let command = match args[0].as_str() {
        "serve" => Command::Serve,
        "demo" => Command::Demo,
        "compare" => Command::Compare,
        "eval" => Command::Eval,
        "--help" | "-h" | "help" => {
            print_usage();
            std::process::exit(0);
        }
        other => return Err(format!("未知子命令「{other}」")),
    };

    // 子命令之后的第一��位置参数是「问题」。
    let question = args.get(1).cloned();
    let mode = args.get(2).cloned();

    Ok(Cli {
        command,
        question,
        mode,
    })
}

/// 打印用法。
fn print_usage() {
    eprintln!(
        "12-rag-summary —— 摘要式 RAG（混合检索 + Reranker + Milvus）\n\
         \n\
         用法:\n\
           cargo run -p rag-12 -- serve                 启动 HTTP 服务（默认）\n\
           cargo run -p rag-12 -- demo \"<问题>\" [模式]    命令行问答\n\
           cargo run -p rag-12 -- compare \"<问题>\" [模式] 同题消融对照\n\
           cargo run -p rag-12 -- eval                  跑评估集出报告\n\
         \n\
         模式: dense | keyword | hybrid | hybrid_rerank | full\n\
         \n\
         环境变量:\n\
           SUMMARY_BIND_ADDR     监听地址（默认 127.0.0.1:3050）\n\
           SUMMARY_MODE          默认检索模式（默认 full）\n\
           SUMMARY_MILVUS_URI    Milvus 地址（默认 http://127.0.0.1:19530）\n\
           SUMMARY_COMPRESS_BUDGET  上下文 token 预算（默认 1200）"
    );
}

/// 执行子命令。
async fn run(cli: Cli, config: AppConfig) -> anyhow::Result<()> {
    match cli.command {
        Command::Serve => serve(config).await,
        Command::Demo => demo(cli, config).await,
        Command::Compare => compare(cli, config).await,
        Command::Eval => eval(config).await,
    }
}

/// 启动 HTTP 服务。
async fn serve(config: AppConfig) -> anyhow::Result<()> {
    println!("── 12-rag-summary ──");
    println!("{}", config.summary());
    println!();

    let (pipeline, index_report) = bootstrap::bootstrap(&config).await?;
    let state: AppState = Arc::new(AppStateInner::new(
        config.clone(),
        Arc::new(pipeline),
        Some(index_report),
    ));

    let router = rag_12::api::routes::router(state);
    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;

    println!("✓ 服务已启动: http://{}", config.bind_addr);
    println!("  端点: /  /health  /api/search  /api/ask  /api/compare  /api/eval  /api/reindex");
    println!();

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// 命令行问答演示。
async fn demo(cli: Cli, config: AppConfig) -> anyhow::Result<()> {
    let question = cli
        .question
        .ok_or_else(|| anyhow::anyhow!("demo 需要一个参数：问题"))?;

    let mode = match cli.mode.as_deref() {
        Some(value) => {
            SearchMode::parse(value).ok_or_else(|| anyhow::anyhow!("未知模式「{value}」"))?
        }
        None => config.retrieval.mode,
    };

    let (pipeline, _) = bootstrap::bootstrap(&config).await?;

    println!("── 问答 ──");
    println!("问题: {question}");
    println!("模式: {}", mode.name());
    println!();

    let outcome = pipeline.ask_with(&question, mode).await?;

    println!("【答案】");
    println!("{}", outcome.answer);
    println!();

    println!("【检索过程】");
    println!(
        "  路径: {} | 候选 {} 块 → 重排后 {} 块 | 改写 {} 个问法",
        outcome.retrieval.path.name(),
        outcome.retrieval.fused_hits.len(),
        outcome.retrieval.hit_count(),
        outcome.retrieval.queries.len(),
    );
    println!(
        "  耗时: 检索 {}ms | 重排 {}ms | 总计 {}ms",
        outcome.retrieval.retrieve_ms, outcome.retrieval.rerank_ms, outcome.elapsed_ms
    );
    if let Some(reason) = &outcome.retrieval.degrade_reason {
        println!("  ⚠ 降级: {reason}");
    }
    println!();

    println!("【上下文压缩】");
    println!("  {}", outcome.compression.stats.summary());
    println!();

    if !outcome.citations.is_empty() {
        println!("【引用】");
        for citation in &outcome.citations {
            println!(
                "  [{}] {} {}",
                citation.index,
                citation.chunk_id,
                citation.heading_path.join(" > ")
            );
        }
        println!();
    }

    Ok(())
}

/// 同题消融对照。
async fn compare(cli: Cli, config: AppConfig) -> anyhow::Result<()> {
    let question = cli
        .question
        .ok_or_else(|| anyhow::anyhow!("compare 需要一个参数：问题"))?;

    let modes: Vec<SearchMode> = match cli.mode.as_deref() {
        Some(value) => {
            vec![SearchMode::parse(value).ok_or_else(|| anyhow::anyhow!("未知模式「{value}」"))?]
        }
        None => SearchMode::all(),
    };

    let (pipeline, _) = bootstrap::bootstrap(&config).await?;

    println!("── 同题消融对照 ──");
    println!("问题: {question}");
    println!();

    let mut id_sets: Vec<Vec<String>> = Vec::new();

    for mode in &modes {
        println!("┌─ 模式: {}", mode.name());

        match pipeline.retrieve_with(&question, *mode).await {
            Ok(outcome) => {
                println!(
                    "│  路径: {} | 耗时 {}ms",
                    outcome.path.name(),
                    outcome.elapsed_ms
                );
                if let Some(reason) = &outcome.degrade_reason {
                    println!("│  ⚠ 降级: {reason}");
                }

                for hit in &outcome.hits {
                    println!("│    {}", hit.summary());
                }

                id_sets.push(outcome.hits.iter().map(|h| h.id.clone()).collect());
            }
            Err(e) => {
                println!("│  ✗ 失败: {e}");
                id_sets.push(Vec::new());
            }
        }

        println!("└─");
        println!();
    }

    println!("块 ID 平均两两重合度: {:.3}", overlap(&id_sets));

    Ok(())
}

/// 跑评估集，产出消融报告。
///
/// ## ⚠️ 全量评估很慢
///
/// 22 样本 × 5 变体 × (1 次生成 + 8 个确定性指标) —— 全是网络往返。
/// 所以支持用 `SUMMARY_EVAL_LIMIT` 限制每个变体的样本数（取前 N 条）。
async fn eval(config: AppConfig) -> anyhow::Result<()> {
    let dataset = rag_09::dataset::EvalDataset::load(&config.eval_dataset_path)?;
    let chunk_index = rag_12::eval::build_chunk_index(&config.data_dir, &config.splitter)?;
    let registry = rag_09::metrics::MetricRegistry::new(None, None);

    let limit = std::env::var("SUMMARY_EVAL_LIMIT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0);

    let (pipeline, _) = bootstrap::bootstrap(&config).await?;

    let runner =
        rag_12::eval::AblationRunner::new(Arc::new(pipeline), Arc::new(chunk_index), registry);

    println!("── 评估 ──");
    println!(
        "评估集: {}（{} 条）",
        config.eval_dataset_path.display(),
        dataset.cases.len()
    );
    println!("变体  : {}", SearchMode::all().len());
    match limit {
        Some(n) => println!("限制  : 每个变体只跑前 {n} 条"),
        None => println!("限制  : 无（全量）"),
    }
    println!();

    let result = runner
        .run_ladder_limited(&dataset, &SearchMode::all(), limit)
        .await?;

    print!("{}", result.to_table());
    println!();
    print!("{}", result.to_step_table());

    Ok(())
}

/// 计算平均两两重合度。
fn overlap(id_sets: &[Vec<String>]) -> f32 {
    if id_sets.len() < 2 {
        return 0.0;
    }

    let mut total = 0.0f32;
    let mut pairs = 0usize;

    for i in 0..id_sets.len() {
        for j in (i + 1)..id_sets.len() {
            let a: std::collections::HashSet<&String> = id_sets[i].iter().collect();
            let b: std::collections::HashSet<&String> = id_sets[j].iter().collect();
            let union = a.union(&b).count();
            if union == 0 {
                continue;
            }
            total += a.intersection(&b).count() as f32 / union as f32;
            pairs += 1;
        }
    }

    if pairs == 0 {
        0.0
    } else {
        total / pairs as f32
    }
}

/// 优雅关闭信号。
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }

    println!("\n服务已停止");
}
