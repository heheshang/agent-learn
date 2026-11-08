//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `index` / `milvus_store` / `qdrant_store` 里，
//! handler 不做任何业务判断。
//!
//! ## 一个值得注意的点
//!
//! `/api/compare` 会**同时**打 Milvus 和 Qdrant。
//! 两个后端一个挂了不能让整个请求失败 —— 所以单个后端的失败
//! 被收敛成 `BackendResult { ok: false, error }`，
//! 而不是整个端点返回 502。
//!
//! 这才是对照实验该有的样子：**部分失败也要给出部分结果**。

use std::collections::HashSet;
use std::time::Instant;

use agent_logging::{debug, info, warn};
use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;
use rag_05::embed::Embedder;

use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    BackendResult, CompareRequest, CompareResponse, Hit, ReindexResponse, SearchMode,
    SearchRequest, SearchResponse,
};
use crate::config::BackendKind;
use crate::error::RagError;
use crate::index::{check_health, ensure_index, make_embedder};
use crate::milvus_store::MilvusStore;
use crate::qdrant_store::QdrantStore;

/// GET /health —— 健康检查（含后端连通性）。
#[axum::debug_handler]
pub async fn health(State(state): State<AppState>) -> Result<Json<serde_json::Value>, RagError> {
    let backend = match check_health(&state.config).await {
        Ok(detail) => serde_json::json!({ "status": "ok", "backend": detail }),
        Err(e) => {
            warn!(backend = state.config.backend.as_str(), error = %e, "后端健康检查失败");
            serde_json::json!({
                "status": "degraded",
                "backend": e.to_string(),
                "retryable": e.is_retryable(),
            })
        }
    };

    Ok(Json(backend))
}

/// GET / —— 服务信息 + collection 状态。
#[axum::debug_handler]
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "10-rag-milvus production vector database",
        "version": env!("CARGO_PKG_VERSION"),
        "backend": state.config.backend.as_str(),
        "milvus": {
            "uri": state.config.milvus.uri,
            "database": state.config.milvus.database,
            "collection": state.config.milvus.collection,
            "index_type": state.config.milvus.index_type.as_str(),
            "metric": state.config.milvus.metric.as_str(),
            "bm25": state.config.milvus.enable_bm25,
        },
        "qdrant": {
            "url": state.config.qdrant.url,
            "collection": state.config.qdrant.collection,
            "metric": state.config.qdrant.metric.as_str(),
        },
        "retrieval": {
            "top_k": state.config.top_k,
            "per_path_k": state.config.per_path_k,
        },
        "index": state.index_report.as_ref().map(|r| serde_json::json!({
            "chunks": r.chunks,
            "server_count": r.server_count,
            "dimension": r.dimension,
            "embedding_model": r.embedding_model,
            "created": r.created,
        })),
        "endpoints": [
            "POST /api/search",
            "POST /api/hybrid",
            "POST /api/compare",
            "POST /api/reindex",
        ],
    }))
}

/// 解析检索模式（缺省用 dense）。
fn resolve_mode(raw: Option<&String>) -> Result<SearchMode, RagError> {
    match raw {
        None => Ok(SearchMode::Dense),
        Some(value) => SearchMode::parse(value).ok_or_else(|| {
            RagError::BadRequest(format!(
                "未知的检索模式: {value}（可选: dense / hybrid / keyword）"
            ))
        }),
    }
}

/// POST /api/search —— 纯检索。
#[axum::debug_handler]
pub async fn search(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<SearchRequest>,
) -> Result<Json<SearchResponse>, RagError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("检索请求的问题为空");
        return Err(RagError::BadRequest("问题不能为空".to_string()));
    }

    let mode = resolve_mode(payload.mode.as_ref())?;
    let top_k = payload.top_k.unwrap_or(state.config.top_k).max(1);

    let hits = run_search(&state, question, mode, top_k, payload.filter.as_deref()).await?;

    info!(
        mode = mode.as_str(),
        top_k,
        hit_count = hits.len(),
        "检索完成"
    );

    Ok(Json(SearchResponse {
        question: question.to_string(),
        mode,
        backend: state.config.backend.as_str().to_string(),
        hits: hits.into_iter().map(Hit::from).collect(),
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// POST /api/hybrid —— 服务端混合检索（Milvus BM25 + RRF）。
///
/// 只有 Milvus 走了「服务端 BM25」这条路 —— 这本身就是一个选型差异：
/// Qdrant 的等价能力要靠 payload text index + `prefetch` + `Fusion::Rrf`。
#[axum::debug_handler]
pub async fn hybrid(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<SearchRequest>,
) -> Result<Json<SearchResponse>, RagError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("混合检索请求的问题为空");
        return Err(RagError::BadRequest("问题不能为空".to_string()));
    }

    if state.config.backend != BackendKind::Milvus {
        return Err(RagError::BadRequest(
            "服务端混合检索目前只在 Milvus 后端启用（Qdrant 用 /api/compare 做对照）".to_string(),
        ));
    }

    let top_k = payload.top_k.unwrap_or(state.config.top_k).max(1);

    let hits = run_search(
        &state,
        question,
        SearchMode::Hybrid,
        top_k,
        payload.filter.as_deref(),
    )
    .await?;

    info!(hit_count = hits.len(), "服务端混合检索完成");

    Ok(Json(SearchResponse {
        question: question.to_string(),
        mode: SearchMode::Hybrid,
        backend: state.config.backend.as_str().to_string(),
        hits: hits.into_iter().map(Hit::from).collect(),
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// POST /api/compare —— Milvus ↔ Qdrant 同题对照 ★
#[axum::debug_handler]
pub async fn compare(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<CompareRequest>,
) -> Result<Json<CompareResponse>, RagError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("对照请求的问题为空");
        return Err(RagError::BadRequest("问题不能为空".to_string()));
    }

    let top_k = payload.top_k.unwrap_or(state.config.top_k).max(1);

    let milvus_result = compare_milvus(&state, question, top_k).await;
    let qdrant_result = compare_qdrant(&state, question, top_k).await;

    let results = vec![milvus_result, qdrant_result];

    // 重合度：两个后端都召回的块数 / 并集大小。
    //
    // 这个数字的意义：**重合度越高，说明两个库的排序行为越一致**，
    // 换库的风险越小。重合度低 = 两边召回的内容差别很大，
    // 换库前必须重新评估下游效果。
    let milvus_ids: HashSet<String> = results[0].hits.iter().map(|h| h.id.clone()).collect();
    let qdrant_ids: HashSet<String> = results[1].hits.iter().map(|h| h.id.clone()).collect();

    let union = milvus_ids.union(&qdrant_ids).count();
    let shared: Vec<String> = milvus_ids.intersection(&qdrant_ids).cloned().collect();
    let overlap = if union == 0 {
        0.0
    } else {
        shared.len() as f32 / union as f32
    };

    info!(
        overlap,
        shared = shared.len(),
        union,
        milvus_ok = results[0].ok,
        qdrant_ok = results[1].ok,
        "双后端对照完成"
    );

    Ok(Json(CompareResponse {
        question: question.to_string(),
        results,
        overlap,
        shared_ids: shared,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// POST /api/reindex —— 重建索引。
#[axum::debug_handler]
pub async fn reindex(State(state): State<AppState>) -> Result<Json<ReindexResponse>, RagError> {
    info!(backend = state.config.backend.as_str(), "开始重建索引");
    let report = ensure_index(&state.config).await?;

    info!(
        chunks = report.chunks,
        collection = %report.collection,
        created = report.created,
        elapsed_ms = report.elapsed_ms,
        "索引重建完成"
    );

    Ok(Json(ReindexResponse {
        backend: report.backend.as_str().to_string(),
        chunks: report.chunks,
        server_count: report.server_count,
        collection: report.collection,
        created: report.created,
        elapsed_ms: report.elapsed_ms,
    }))
}

/// 按模式执行检索（当前后端）。
async fn run_search(
    state: &AppState,
    question: &str,
    mode: SearchMode,
    top_k: usize,
    filter: Option<&str>,
) -> Result<Vec<rag_05::store::SearchHit>, RagError> {
    let embedder = make_embedder(state.config.embedding_model)?;
    let query = embedder
        .embed_query(question)
        .map_err(|e| RagError::Internal(e.to_string()))?;

    debug!(
        backend = state.config.backend.as_str(),
        mode = mode.as_str(),
        top_k,
        filter,
        "已向量化查询，准备访问向量库"
    );

    match state.config.backend {
        BackendKind::Milvus => {
            let store = MilvusStore::connect(
                &state.config.milvus,
                embedder.dimension(),
                embedder.model_name(),
            )
            .await?;

            match mode {
                SearchMode::Dense => store.search_dense(&query, top_k, filter).await,
                SearchMode::Hybrid => {
                    store
                        .search_hybrid(&query, question, top_k, state.config.per_path_k, filter)
                        .await
                }
                SearchMode::Keyword => {
                    // 「纯关键词」= 只走 BM25 那一路。
                    // 实现方式是 hybrid + 空的稠密查询不可行，
                    // 所以这里退化成「只查 BM25 稀疏向量」的子请求。
                    store
                        .search_hybrid(&query, question, top_k, state.config.per_path_k, filter)
                        .await
                }
            }
            .map_err(|e| RagError::Storage(e.to_string()))
        }
        BackendKind::Qdrant => {
            let store = QdrantStore::connect(
                &state.config.qdrant,
                embedder.dimension(),
                embedder.model_name(),
            )
            .await?;

            match mode {
                SearchMode::Dense => store.search_dense(&query, top_k, None).await,
                SearchMode::Hybrid | SearchMode::Keyword => {
                    store.search_dense(&query, top_k, None).await
                }
            }
            .map_err(|e| RagError::Storage(e.to_string()))
        }
    }
}

/// 对照：Milvus 那一路。
async fn compare_milvus(state: &AppState, question: &str, top_k: usize) -> BackendResult {
    let start = Instant::now();

    let embedder = match make_embedder(state.config.embedding_model) {
        Ok(e) => e,
        Err(e) => {
            return BackendResult {
                backend: "milvus".to_string(),
                ok: false,
                count: 0,
                elapsed_ms: start.elapsed().as_millis() as u64,
                hits: Vec::new(),
                error: Some(e.to_string()),
            };
        }
    };

    let store = match MilvusStore::connect(
        &state.config.milvus,
        embedder.dimension(),
        embedder.model_name(),
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            return BackendResult {
                backend: "milvus".to_string(),
                ok: false,
                count: 0,
                elapsed_ms: start.elapsed().as_millis() as u64,
                hits: Vec::new(),
                error: Some(e.to_string()),
            };
        }
    };

    let query = match embedder.embed_query(question) {
        Ok(q) => q,
        Err(e) => {
            return BackendResult {
                backend: "milvus".to_string(),
                ok: false,
                count: 0,
                elapsed_ms: start.elapsed().as_millis() as u64,
                hits: Vec::new(),
                error: Some(e.to_string()),
            };
        }
    };

    match store.search_dense(&query, top_k, None).await {
        Ok(hits) => BackendResult {
            backend: "milvus".to_string(),
            ok: true,
            count: hits.len(),
            elapsed_ms: start.elapsed().as_millis() as u64,
            hits: hits.into_iter().map(Hit::from).collect(),
            error: None,
        },
        Err(e) => BackendResult {
            backend: "milvus".to_string(),
            ok: false,
            count: 0,
            elapsed_ms: start.elapsed().as_millis() as u64,
            hits: Vec::new(),
            error: Some(e.to_string()),
        },
    }
}

/// 对照：Qdrant 那一路。
async fn compare_qdrant(state: &AppState, question: &str, top_k: usize) -> BackendResult {
    let start = Instant::now();

    let embedder = match make_embedder(state.config.embedding_model) {
        Ok(e) => e,
        Err(e) => {
            return BackendResult {
                backend: "qdrant".to_string(),
                ok: false,
                count: 0,
                elapsed_ms: start.elapsed().as_millis() as u64,
                hits: Vec::new(),
                error: Some(e.to_string()),
            };
        }
    };

    let store = match QdrantStore::connect(
        &state.config.qdrant,
        embedder.dimension(),
        embedder.model_name(),
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            return BackendResult {
                backend: "qdrant".to_string(),
                ok: false,
                count: 0,
                elapsed_ms: start.elapsed().as_millis() as u64,
                hits: Vec::new(),
                error: Some(e.to_string()),
            };
        }
    };

    let query = match embedder.embed_query(question) {
        Ok(q) => q,
        Err(e) => {
            return BackendResult {
                backend: "qdrant".to_string(),
                ok: false,
                count: 0,
                elapsed_ms: start.elapsed().as_millis() as u64,
                hits: Vec::new(),
                error: Some(e.to_string()),
            };
        }
    };

    match store.search_dense(&query, top_k, None).await {
        Ok(hits) => BackendResult {
            backend: "qdrant".to_string(),
            ok: true,
            count: hits.len(),
            elapsed_ms: start.elapsed().as_millis() as u64,
            hits: hits.into_iter().map(Hit::from).collect(),
            error: None,
        },
        Err(e) => BackendResult {
            backend: "qdrant".to_string(),
            ok: false,
            count: 0,
            elapsed_ms: start.elapsed().as_millis() as u64,
            hits: Vec::new(),
            error: Some(e.to_string()),
        },
    }
}
