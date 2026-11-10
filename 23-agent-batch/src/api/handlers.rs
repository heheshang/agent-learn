//! API 处理器。

use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;

use agent_logging::info;

use crate::api::AppState;
use crate::api::models::*;
use crate::batching::BatchStrategy;
use crate::bench;
use crate::error::AppError;
use crate::pipeline;

/// 解析批策略（缺省 → `fixed`）。
fn parse_strategy(raw: Option<&str>) -> Result<BatchStrategy, AppError> {
    match raw {
        None => Ok(BatchStrategy::Fixed),
        Some(s) => BatchStrategy::parse(s).ok_or_else(|| {
            AppError::BadRequest(format!(
                "未知批策略: {s}（可选: sequential / fixed / bucketed）"
            ))
        }),
    }
}

/// 服务信息。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    let config = &state.config;

    Json(serde_json::json!({
        "service": "agent-23",
        "topic": "批处理优化 (Batching)",
        "bind_addr": config.bind_addr,
        "embed_model": config.embed_model,
        "rerank_model": config.rerank_model,
        "batch_size": config.batch.batch_size,
        "bucket_batch_size": config.batch.bucket_batch_size,
        "max_wait_ms": config.batch.max_wait_ms,
        "max_queue_len": config.batch.max_queue_len,
        "strategies": [
            {"name": "sequential", "label": "逐条调用", "desc": "每条一次前向 —— 基线"},
            {"name": "fixed", "label": "固定批", "desc": "每 N 条一次前向"},
            {"name": "bucketed", "label": "长度分桶", "desc": "按长度分桶后定批，减少 padding 浪费"},
        ],
        "endpoints": [
            "POST /api/embed",
            "POST /api/rerank",
            "POST /api/bench/embed",
            "POST /api/bench/rerank",
            "POST /api/bench/remote",
            "POST /api/index",
            "POST /api/search",
        ],
    }))
}

/// ★ 批量嵌入。
pub async fn embed(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<impl IntoResponse, AppError> {
    let strategy = parse_strategy(req.strategy.as_deref())?;

    let result = state.embedder.embed_batch(&req.texts, strategy).await?;

    let (vectors, stats) = result;

    Ok(Json(EmbedResponse {
        count: vectors.len(),
        dimension: state.embedder.dimension(),
        vectors,
        stats,
        strategy,
    }))
}

/// ★ 批量重排。
pub async fn rerank(
    State(state): State<AppState>,
    Json(req): Json<RerankRequest>,
) -> Result<impl IntoResponse, AppError> {
    let strategy = parse_strategy(req.strategy.as_deref())?;

    let documents: Vec<rag_08::reranker::RerankDocument> = req
        .documents
        .iter()
        .map(|d| d.to_rerank_document())
        .collect();

    let top_n = req.top_n.unwrap_or(3).max(1);

    let (hits, stats) = state
        .reranker
        .rerank_batch(&req.query, &documents, top_n, strategy)
        .await?;

    Ok(Json(RerankResponse {
        candidate_count: documents.len(),
        hits,
        stats,
        strategy,
    }))
}

/// 把对照报告转成 JSON。
fn report_to_json(report: &bench::CompareReport) -> serde_json::Value {
    let results: Vec<serde_json::Value> = report
        .results
        .iter()
        .map(|(strategy, stats)| {
            serde_json::json!({
                "strategy": strategy.label(),
                "stats": stats,
                "speedup_vs_sequential": stats.speedup_vs(report.baseline().elapsed_ms),
            })
        })
        .collect();

    serde_json::json!({
        "sample_size": report.sample_size,
        "rounds": report.rounds,
        "baseline_elapsed_ms": report.baseline().elapsed_ms,
        "speedups": report.speedups,
        "speedup_stddev": report.speedup_stddev(),
        "results": results,
        "table": report.render_table(),
        "summary": report.render_summary(),
    })
}

/// ★ ★ 嵌入批处理对照实验。
pub async fn bench_embed(
    State(state): State<AppState>,
    Json(req): Json<BenchEmbedRequest>,
) -> Result<impl IntoResponse, AppError> {
    let sample_size = req
        .sample_size
        .unwrap_or(state.config.sample_size)
        .clamp(1, 10_000);
    let rounds = req.rounds.unwrap_or(state.config.rounds).clamp(1, 20);

    let corpus = bench::sample_corpus(sample_size);

    let report = bench::run_embed_compare(&state.embedder, &corpus, rounds).await?;

    info!(
        sample_size,
        rounds,
        baseline_ms = report.baseline().elapsed_ms,
        "嵌入对照实验完成"
    );

    Ok(Json(report_to_json(&report)))
}

/// ★ ★ 重排批处理对照实验。
pub async fn bench_rerank(
    State(state): State<AppState>,
    Json(req): Json<BenchRerankRequest>,
) -> Result<impl IntoResponse, AppError> {
    let sample_size = req
        .sample_size
        .unwrap_or(state.config.sample_size)
        .clamp(1, 10_000);
    let rounds = req.rounds.unwrap_or(state.config.rounds).clamp(1, 20);
    let top_n = req.top_n.unwrap_or(3).clamp(1, 1000);

    let documents = bench::sample_documents(sample_size);

    let report =
        bench::run_rerank_compare(&state.reranker, &req.query, &documents, top_n, rounds).await?;

    Ok(Json(report_to_json(&report)))
}

/// ★ ★ 批大小扫描。
pub async fn bench_sweep(
    State(state): State<AppState>,
    Json(req): Json<BenchSweepRequest>,
) -> Result<impl IntoResponse, AppError> {
    let sample_size = req
        .sample_size
        .unwrap_or(state.config.sample_size)
        .clamp(1, 10_000);
    let rounds = req.rounds.unwrap_or(state.config.rounds).clamp(1, 20);

    let sizes = req
        .batch_sizes
        .clone()
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32, 64, 128]);

    let corpus = bench::sample_corpus(sample_size);

    let report = crate::sweep::sweep_batch_sizes(&state.embedder, &corpus, &sizes, rounds).await?;

    Ok(Json(serde_json::json!({
        "sample_size": sample_size,
        "rounds": rounds,
        "best_batch_size": report.best_batch_size,
        "points": report.points,
        "table": report.render_table(),
        "summary": report.render_summary(),
    })))
}

/// ★ ★ 远程 HTTP 批处理对照实验。
///
/// ⚠️ 需要 Ollama（或兼容服务）在跑；连不上直接返回 503。
pub async fn bench_remote(
    State(state): State<AppState>,
    Json(req): Json<BenchRemoteRequest>,
) -> Result<impl IntoResponse, AppError> {
    let sample_size = req
        .sample_size
        .unwrap_or(state.config.sample_size)
        .clamp(1, 2000);
    let batch_size = req
        .batch_size
        .unwrap_or(state.config.batch.batch_size)
        .clamp(1, 2048);

    let embedder = crate::remote::OllamaEmbedder::new(
        state.config.ollama_endpoint.clone(),
        state.config.ollama_model.clone(),
        batch_size,
    );

    embedder.health_check().await?;

    let corpus = bench::sample_corpus(sample_size);

    let max_concurrency = state.config.batch.max_concurrency;

    let report =
        crate::remote::compare_remote(&embedder, &corpus, batch_size, max_concurrency).await?;

    Ok(Json(serde_json::json!({
        "backend": report.backend,
        "batch_size": report.batch_size,
        "max_concurrency": max_concurrency,
        "sample_size": sample_size,
        "sequential": report.sequential,
        "concurrent": report.concurrent,
        "batched": report.batched,
        "end_to_end_speedup": report.end_to_end_speedup(),
        "concurrency_speedup": report.concurrency_speedup(),
        "incremental_speedup": report.incremental_speedup(),
        "note": "三组对照: 串行逐条 / 并发逐条(LangChain batch) / 批量 HTTP(provider Batch API)。端到端收益 = 批量 vs 串行；增量收益 = 批量 vs 并发",
    })))
}

/// ★ 批量建索引。
pub async fn index(
    State(state): State<AppState>,
    Json(req): Json<IndexRequest>,
) -> Result<impl IntoResponse, AppError> {
    let strategy = parse_strategy(req.strategy.as_deref())?;

    let documents: Vec<(String, String, Vec<String>)> = req
        .documents
        .into_iter()
        .map(|d| (d.id, d.text, d.heading_path))
        .collect();

    let (_, report) = pipeline::build_index(&state.embedder, &documents, strategy).await?;

    Ok(Json(IndexResponse {
        doc_count: report.doc_count,
        dimension: report.dimension,
        report,
    }))
}

/// ★ 批量检索。
pub async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<impl IntoResponse, AppError> {
    let strategy = parse_strategy(req.strategy.as_deref())?;

    let top_k = req.top_k.unwrap_or(20).clamp(1, 1000);
    let top_n = req.top_n.unwrap_or(3).clamp(1, 1000);

    // ⚠️ 这里用一个**临时索引**演示端到端链路 ——
    //    真正的服务应该维护一个常驻索引。
    //    所以语料用批处理自带的样本文档。
    let documents: Vec<(String, String, Vec<String>)> = bench::sample_documents(top_k)
        .into_iter()
        .map(|d| (d.id, d.text, d.heading_path))
        .collect();

    let (store, _) = pipeline::build_index(&state.embedder, &documents, strategy).await?;

    let report = pipeline::search(
        &state.embedder,
        &state.reranker,
        &store,
        &req.query,
        top_k,
        top_n,
        strategy,
    )
    .await?;

    Ok(Json(serde_json::json!({
        "query": req.query,
        "candidate_count": report.candidate_count,
        "recall_ms": report.recall_ms,
        "rerank_ms": report.rerank_ms,
        "stats": report.rerank_stats,
        "strategy": report.strategy,
        "hits": report.hits,
    })))
}

/// 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}
