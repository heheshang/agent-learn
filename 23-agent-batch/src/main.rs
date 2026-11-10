//! 23-agent-batch —— 手撕批处理优化。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- modes           # ★ 三档批策略说明
//! cargo run -- bench           # ★ ★ 嵌入批处理对照实验（本地 ONNX）
//! cargo run -- bench-rerank    # ★ ★ 重排批处理对照实验（本地 ONNX）
//! cargo run -- bench-remote    # ★ ★ 远程 HTTP 批处理对照实验（需 Ollama）
//! cargo run -- sweep           # ★ ★ 批大小扫描（找最优批大小）
//! cargo run -- embed "文本一" "文本二"   # 直接批量嵌入
//! cargo run -- rerank "查询" "候选一" "候选二"  # 直接批量重排
//! cargo run -- serve           # 启动 HTTP 服务（:3150）
//! cargo run -- all             # 全部
//! ```

use std::sync::Arc;

use agent_logging::{info, init, warn};
use anyhow::Result;
use rag_05::embed::Model;
use rag_08::reranker::RerankerModel;

use agent_23::api::AppStateInner;
use agent_23::batching::BatchStrategy;
use agent_23::bench;
use agent_23::config::AppConfig;
use agent_23::embedder::BatchEmbedder;
use agent_23::remote::RemoteEmbedder;
use agent_23::reranker::BatchReranker;

/// 嵌入模型名 → 枚举。
fn parse_embed_model(name: &str) -> Model {
    match name.to_ascii_lowercase().replace('-', "_").as_str() {
        "minilm" | "mini_lm" => Model::MiniLM,
        "bge" | "bge_small_zh" => Model::BGESmallZH,
        _ => Model::MultilingualE5Small,
    }
}

/// 重排模型名 → 枚举。
fn parse_rerank_model(name: &str) -> RerankerModel {
    match name.to_ascii_lowercase().replace('-', "_").as_str() {
        "m3" | "v2_m3" => RerankerModel::BgeRerankerV2M3,
        "jina" => RerankerModel::JinaRerankerV1TurboEn,
        _ => RerankerModel::BgeRerankerBase,
    }
}

/// 构造嵌入器与重排器。
fn build_models(config: &AppConfig) -> (Arc<BatchEmbedder>, Arc<BatchReranker>) {
    let embedder = BatchEmbedder::new(
        parse_embed_model(&config.embed_model),
        Some(rag_05::embed::cache_dir()),
        config.batch.clone(),
    );

    let reranker = BatchReranker::new(
        parse_rerank_model(&config.rerank_model),
        Some(rag_05::embed::cache_dir()),
        config.batch.clone(),
    );

    (Arc::new(embedder), Arc::new(reranker))
}

#[tokio::main]
async fn main() -> Result<()> {
    init("agent-23");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "modes" => modes_demo(),
        "bench" => bench_embed().await?,
        "bench-rerank" => bench_rerank().await?,
        "bench-remote" => bench_remote().await?,
        "sweep" => sweep().await?,
        "embed" => embed_demo()?,
        "rerank" => rerank_demo()?,
        "serve" => serve().await?,
        "all" => {
            modes_demo();
            bench_embed().await?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: modes / bench / bench-rerank / bench-remote / sweep / embed / rerank / serve / all）"
            );
        }
    }

    Ok(())
}

/// 打印三档批策略。
fn modes_demo() {
    println!("=== 三档批策略 ===");
    println!();

    let mut table = agent_23::table::Table::new(&["策略", "做法", "LangChain 对应", "省下什么"]);
    table.row(&[
        "sequential",
        "逐条调用，每条一次前向",
        "（基线）",
        "什么都不省",
    ]);
    table.row(&["fixed", "每 N 条一次前向", "batch()", "固定开销被 N 条分摊"]);
    table.row(&[
        "bucketed",
        "按长度分桶后定批",
        "batch() + 自行分桶",
        "同上，且 padding 浪费更少",
    ]);
    print!("{}", table.render());

    println!();
    println!("⚠️ 批处理**抬高单条延迟** —— 只适合「攒一批一起要」的场景。");
    println!("⚠️ 固定批里一条长文本会拖慢同批所有短文本（padding 浪费）。");
    println!("⚠️ 攒批必须有背压：最大批大小 / 最大等待时间 / 最大队列长度。");
    println!();
    println!("⚠️ LangChain 的 batch() 是**客户端并行**；");
    println!("   provider 的 Batch API 是**服务端排队**（更便宜但更慢）—— 两者不是一回事。");
}

/// ★ 嵌入批处理对照实验。
async fn bench_embed() -> Result<()> {
    let config = AppConfig::from_env();
    let (embedder, _) = build_models(&config);

    let corpus = bench::sample_corpus(config.sample_size);

    println!("=== 嵌入批处理对照实验 ===");
    println!("模型      : {}", embedder.model_name());
    println!("维度      : {}", embedder.dimension());
    println!("样本条数  : {}", corpus.len());
    println!("批大小    : {}", config.batch.batch_size);
    println!("轮数      : {}", config.rounds);
    println!();

    let report = bench::run_embed_compare(&embedder, &corpus, config.rounds).await?;

    print!("{}", report.render_table());
    println!();
    print!("{}", report.render_summary());

    Ok(())
}

/// ★ 重排批处理对照实验。
async fn bench_rerank() -> Result<()> {
    let config = AppConfig::from_env();
    let (_, reranker) = build_models(&config);

    let documents = bench::sample_documents(config.sample_size);

    println!("=== 重排批处理对照实验 ===");
    println!("模型      : {}", reranker.model_name());
    println!("候选数    : {}", documents.len());
    println!("批大小    : {}", config.batch.batch_size);
    println!("轮数      : {}", config.rounds);
    println!();

    let report = bench::run_rerank_compare(
        &reranker,
        "批处理如何提升吞吐量",
        &documents,
        3,
        config.rounds,
    )
    .await?;

    print!("{}", report.render_table());
    println!();
    print!("{}", report.render_summary());

    Ok(())
}

/// ★ 远程 HTTP 批处理对照实验。
async fn bench_remote() -> Result<()> {
    let config = AppConfig::from_env();

    let embedder = agent_23::remote::OllamaEmbedder::new(
        config.ollama_endpoint.clone(),
        config.ollama_model.clone(),
        config.batch.batch_size,
    );

    embedder
        .health_check()
        .await
        .map_err(|e| anyhow::anyhow!("远程后端不可用: {e}"))?;

    let corpus = bench::sample_corpus(config.sample_size);

    println!("=== 远程 HTTP 批处理对照实验 ===");
    println!(
        "后端      : {} @ {}",
        embedder.name(),
        config.ollama_endpoint
    );
    println!("样本条数  : {}", corpus.len());
    println!("批大小    : {}", config.batch.batch_size);
    println!();

    let report = agent_23::remote::compare_remote(
        &embedder,
        &corpus,
        config.batch.batch_size,
        config.batch.max_concurrency,
    )
    .await?;

    println!(
        "组 A 串行逐条 HTTP : {} ms　{:.1} 条/秒　单条 p50 {} ms",
        report.sequential.elapsed_ms,
        report.sequential.throughput_per_sec(),
        report.sequential.p50()
    );
    println!(
        "组 B 并发逐条 HTTP : {} ms　{:.1} 条/秒　单条 p50 {} ms　（并发上限 {}）",
        report.concurrent.elapsed_ms,
        report.concurrent.throughput_per_sec(),
        report.concurrent.p50(),
        config.batch.max_concurrency
    );
    println!(
        "组 C 批量 HTTP     : {} ms　{:.1} 条/秒　单条 p50 {} ms",
        report.batched.elapsed_ms,
        report.batched.throughput_per_sec(),
        report.batched.p50()
    );
    println!();
    println!(
        "客户端并行的收益 (B vs A) : {:.2}x",
        report.concurrency_speedup()
    );
    println!(
        "合并请求的增量收益 (C vs B): {:.2}x",
        report.incremental_speedup()
    );
    println!(
        "端到端收益     (C vs A) : {:.2}x",
        report.end_to_end_speedup()
    );
    println!();
    println!("⚠️ LangChain 的 batch() 对应的是**组 B**（客户端并行），");
    println!("   provider 的 Batch API 对应的是**组 C**（合并进一次请求）—— 两者不是一回事。");

    Ok(())
}

/// ★ 批大小扫描。
async fn sweep() -> Result<()> {
    let config = AppConfig::from_env();
    let (embedder, _) = build_models(&config);

    let corpus = bench::sample_corpus(config.sample_size);

    // 扫描的档位：1（基线）→ 2 的幂 → 128。
    let sizes: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64, 128];

    println!("=== 批大小扫描 ===");
    println!("模型      : {}", embedder.model_name());
    println!("样本条数  : {}", corpus.len());
    println!("轮数      : {}", config.rounds);
    println!("档位      : {sizes:?}");
    println!();

    let report =
        agent_23::sweep::sweep_batch_sizes(&embedder, &corpus, &sizes, config.rounds).await?;

    print!("{}", report.render_table());
    println!();
    print!("{}", report.render_summary());

    Ok(())
}

/// 直接批量嵌入。
fn embed_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let (embedder, _) = build_models(&config);

    let texts: Vec<String> = std::env::args().skip(2).collect();
    if texts.is_empty() {
        anyhow::bail!("请至少给一条文本");
    }

    let runtime = tokio::runtime::Runtime::new()?;
    let (vectors, stats) = runtime.block_on(embedder.embed_batch(&texts, BatchStrategy::Fixed))?;

    println!("条数      : {}", vectors.len());
    println!("维度      : {}", embedder.dimension());
    println!("{}", stats.summary_line());
    println!();
    for (i, v) in vectors.iter().enumerate() {
        let preview: Vec<String> = v.iter().take(4).map(|x| format!("{x:.4}")).collect();
        println!("  向量 {i}: [{}]", preview.join(", "));
    }

    Ok(())
}

/// 直接批量重排。
fn rerank_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let (_, reranker) = build_models(&config);

    let mut args = std::env::args().skip(2);
    let query = args.next().unwrap_or_else(|| "查询".to_string());
    let documents: Vec<rag_08::reranker::RerankDocument> = args
        .map(|text| {
            rag_08::reranker::RerankDocument::new(
                format!("doc-{}", text.chars().count()),
                "cli.md",
                vec![],
                text,
            )
        })
        .collect();

    if documents.is_empty() {
        anyhow::bail!("请至少给一条候选文档");
    }

    let runtime = tokio::runtime::Runtime::new()?;
    let (hits, stats) =
        runtime.block_on(reranker.rerank_batch(&query, &documents, 3, BatchStrategy::Fixed))?;

    println!("查询      : {query}");
    println!("候选数    : {}", documents.len());
    println!("{}", stats.summary_line());
    println!();
    for hit in &hits {
        println!("  [{}] {:.4}  {}", hit.rank + 1, hit.score, hit.text);
    }

    Ok(())
}

/// 启动 HTTP 服务。
async fn serve() -> Result<()> {
    let config = AppConfig::from_env();

    info!("agent-23 启动中");
    println!("{}", config.summary());
    println!();

    let (embedder, reranker) = build_models(&config);
    let state = Arc::new(AppStateInner::new(config.clone(), embedder, reranker));
    let app = agent_23::api::routes::router(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    let addr = listener.local_addr()?;

    println!("★ 监听 http://{addr}");
    println!("  GET  /                 服务信息 + 批策略");
    println!("  POST /api/embed        ★ 批量嵌入");
    println!("  POST /api/rerank       ★ 批量重排");
    println!("  POST /api/bench/embed  ★ ★ 嵌入对照实验");
    println!("  POST /api/bench/rerank ★ ★ 重排对照实验");
    println!("  POST /api/bench/remote ★ ★ 远程对照实验");
    println!("  POST /api/index        ★ 批量建索引");
    println!("  POST /api/search       ★ 批量检索");
    println!();

    axum::serve(listener, app).await?;

    warn!("服务已退出");

    Ok(())
}
