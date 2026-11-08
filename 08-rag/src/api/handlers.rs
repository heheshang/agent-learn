//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `pipeline` / `hybrid` / `reranker` 里，
//! handler 不做任何业务判断。

use agent_logging::{info, warn};
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;

use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    AskRequest, AskResponse, CompareRequest, CompareResponse, ModeResult, ReindexResponse,
    SearchRequest, SearchResponse,
};
use crate::error::RagError;
use crate::hybrid::SearchMode;

/// GET /health —— 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET / —— 服务信息 + 索引状态。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "08-rag hybrid retrieval + rerank",
        "version": env!("CARGO_PKG_VERSION"),
        "index": {
            "chunks": state.index_stats.chunk_count,
            "bm25_documents": state.bm25_documents,
            "dimension": state.index_stats.dimension,
            "kind": state.index_stats.index_kind,
            "embedding_model": state.index_stats.embedding_model,
        },
        "retrieval": {
            "mode": state.config.retrieval.mode.name(),
            "top_k": state.config.retrieval.top_k,
            "per_path_k": state.config.retrieval.effective_per_path_k(),
            "min_score": state.config.retrieval.min_score,
            "fusion": "rrf",
            "rrf_k": crate::hybrid::RRF_K,
        },
        "rerank": {
            "enabled": state.config.rerank.enabled,
            "backend": state.config.rerank.backend.name(),
            "model": state.config.rerank.model.display_name(),
            "candidate_k": state.config.rerank.candidate_k,
            "top_n": state.config.rerank.top_n,
        },
        "generation": {
            "model": state.config.generation.model,
            "temperature": state.config.generation.temperature,
        },
        "modes": SearchMode::all()
            .iter()
            .map(|m| m.name())
            .collect::<Vec<_>>(),
        "endpoints": [
            "POST /api/ask",
            "POST /api/search",
            "POST /api/compare",
            "POST /api/reindex",
        ],
    }))
}

/// 解析检索模式（缺省用配置里的默认值）。
fn resolve_mode(raw: Option<&String>, fallback: SearchMode) -> Result<SearchMode, RagError> {
    match raw {
        None => Ok(fallback),
        Some(value) => SearchMode::parse(value).ok_or_else(|| {
            RagError::BadRequest(format!(
                "未知的检索模式: {value}（可选: vector / bm25 / hybrid）"
            ))
        }),
    }
}

/// POST /api/ask —— 混合检索 + 重排 + 生成。
#[axum::debug_handler]
pub async fn ask(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<AskRequest>,
) -> Result<Json<AskResponse>, RagError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("问答问题为空");
        return Err(RagError::BadRequest("问题不能为空".to_string()));
    }

    let mode = resolve_mode(payload.mode.as_ref(), state.config.retrieval.mode)?;

    // 复用流水线的 `ask_with` —— Prompt 组装和引用解析都在那里，
    // handler 不重复实现（两处实现迟早会漂移）。
    let outcome = state.pipeline.ask_with(question, mode).await?;

    info!(
        question = %question,
        mode = mode.name(),
        hits = outcome.retrieval.hits.len(),
        citations = outcome.citations.len(),
        reranked = outcome.retrieval.reranked,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "问答完成"
    );

    Ok(Json(AskResponse {
        answer: outcome.answer,
        citations: outcome.citations,
        hits: if payload.include_context {
            outcome.retrieval.hits
        } else {
            Vec::new()
        },
        mode,
        reranked: outcome.retrieval.reranked,
        degrade_reason: outcome.retrieval.degrade_reason,
        retrieve_ms: outcome.retrieval.retrieve_ms,
        rerank_ms: outcome.retrieval.rerank_ms,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// POST /api/search —— 混合检索 + 重排（不生成）。
///
/// 调试用：能直接看到「召回了什么」和「重排后顺序变了多少」。
#[axum::debug_handler]
pub async fn search(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<SearchRequest>,
) -> Result<Json<SearchResponse>, RagError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("搜索问题为空");
        return Err(RagError::BadRequest("问题不能为空".to_string()));
    }

    let mode = resolve_mode(payload.mode.as_ref(), state.config.retrieval.mode)?;

    let outcome = state.pipeline.retrieve_with(question, mode).await?;

    info!(
        question = %question,
        mode = mode.name(),
        hits = outcome.hits.len(),
        reranked = outcome.reranked,
        retrieve_ms = outcome.retrieve_ms,
        rerank_ms = outcome.rerank_ms,
        "搜索完成"
    );

    Ok(Json(SearchResponse {
        question: question.to_string(),
        mode,
        hits: outcome.hits,
        fused_hits: if payload.include_fused {
            outcome.fused_hits
        } else {
            Vec::new()
        },
        reranked: outcome.reranked,
        degrade_reason: outcome.degrade_reason,
        retrieve_ms: outcome.retrieve_ms,
        rerank_ms: outcome.rerank_ms,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// POST /api/compare —— 多模式对比。
///
/// **本项目的核心端点。**
///
/// 对同一个问题依次跑完多个模式，然后给出：
///
/// 1. 每个模式的命中块
/// 2. 每个模式是否命中了 `expected` 关键词
/// 3. 模式之间的**召回重叠度矩阵**
///
/// 第 3 项尤其重要 —— 它能直接告诉你「混合检索到底有没有带来新信息」。
#[axum::debug_handler]
pub async fn compare(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<CompareRequest>,
) -> Result<Json<CompareResponse>, RagError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("对比问题为空");
        return Err(RagError::BadRequest("问题不能为空".to_string()));
    }

    // 解析模式列表。
    let modes = match &payload.modes {
        None => SearchMode::all(),
        Some(list) => {
            let mut parsed = Vec::with_capacity(list.len());
            for raw in list {
                parsed.push(SearchMode::parse(raw).ok_or_else(|| {
                    RagError::BadRequest(format!(
                        "未知的检索模式: {raw}（可选: vector / bm25 / hybrid）"
                    ))
                })?);
            }
            if parsed.is_empty() {
                parsed = SearchMode::all();
            }
            parsed
        }
    };

    // 逐个模式跑。
    let mut results: Vec<ModeResult> = Vec::with_capacity(modes.len());

    for mode in modes {
        let mode_start = Instant::now();
        let outcome = state.pipeline.retrieve_with(question, mode).await?;

        let hit = payload.expected.as_ref().map(|keywords| {
            let all_text: String = outcome
                .hits
                .iter()
                .map(|h| format!("{} {}", h.text, h.heading_path.join(" ")))
                .collect::<Vec<_>>()
                .join(" ");
            keywords
                .iter()
                .any(|kw| !kw.is_empty() && all_text.contains(kw))
        });

        let hit_ids = outcome.hit_ids();

        results.push(ModeResult {
            mode,
            hits: outcome.hits,
            hit_ids,
            hit,
            reranked: outcome.reranked,
            degrade_reason: outcome.degrade_reason,
            elapsed_ms: mode_start.elapsed().as_millis() as u64,
        });
    }

    // 重叠度矩阵。
    let overlap = compute_overlap(&results);

    // 哪些模式命中了期望关键词。
    let successful_modes = results
        .iter()
        .filter(|r| r.hit == Some(true))
        .map(|r| r.mode.name().to_string())
        .collect();

    info!(
        question = %question,
        mode_count = results.len(),
        successful = ?successful_modes,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "多模式对比完成"
    );

    Ok(Json(CompareResponse {
        question: question.to_string(),
        results,
        overlap,
        successful_modes,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// 计算模式之间的召回重叠度矩阵。
///
/// `overlap[i][j]` = 模式 i 和模式 j 共同召回的块数。
fn compute_overlap(results: &[ModeResult]) -> Vec<Vec<usize>> {
    let n = results.len();
    let mut matrix = vec![vec![0usize; n]; n];

    for i in 0..n {
        for j in 0..n {
            let count = results[i]
                .hit_ids
                .iter()
                .filter(|id| results[j].hit_ids.contains(id))
                .count();
            matrix[i][j] = count;
        }
    }

    matrix
}

/// POST /api/reindex —— 重建索引。
///
/// 文档更新后调这个。重建期间会阻塞（嵌入模型是 CPU 密集的），
/// 所以这是个「管理端点」而不是「用户端点」。
#[axum::debug_handler]
pub async fn reindex(State(state): State<AppState>) -> Result<Json<ReindexResponse>, RagError> {
    let start = Instant::now();

    let embedder = crate::index::make_embedder(state.config.embedding_model.to_model())?;
    let options = crate::index::IndexOptions {
        splitter: state.config.splitter.clone(),
        ..Default::default()
    };

    let stats = crate::index::build_and_save(
        &embedder,
        &state.config.data_dir,
        &state.config.index_path,
        &options,
    )
    .await?;

    let bm25 = crate::index::build_bm25(&state.config.data_dir, &state.config.splitter)?;

    info!(
        chunk_count = stats.chunk_count,
        bm25_documents = bm25.len(),
        elapsed_ms = start.elapsed().as_millis() as u64,
        "索引重建完成"
    );

    Ok(Json(ReindexResponse {
        ok: true,
        stats,
        bm25_documents: bm25.len(),
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}
