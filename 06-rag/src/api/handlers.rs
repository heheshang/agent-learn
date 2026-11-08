//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `retrieval` / `generation` / `agent` 里，
//! handler 不做任何业务判断。

use agent_logging::{debug, info, warn};
use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;

use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    AgentResponse, AskRequest, AskResponse, ReindexResponse, SearchRequest, SearchResponse,
};
use crate::error::RagError;

/// GET /health —— 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET / —— 服务信息 + 索引状态。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "06-rag naive RAG",
        "version": env!("CARGO_PKG_VERSION"),
        "index": {
            "chunks": state.index_stats.chunk_count,
            "dimension": state.index_stats.dimension,
            "kind": state.index_stats.index_kind,
            "embedding_model": state.index_stats.embedding_model,
        },
        "retrieval": {
            "top_k": state.config.retrieval.top_k,
            "min_score": state.config.retrieval.min_score,
        },
        "generation": {
            "model": state.config.generation.model,
            "temperature": state.config.generation.temperature,
        },
        "endpoints": [
            "POST /api/ask",
            "POST /api/agent",
            "POST /api/search",
            "POST /api/reindex",
        ],
    }))
}

/// POST /api/ask —— Naive RAG 问答。
///
/// 流程：问题 → 检索（用原问题）→ 拼 Prompt → 生成 → 抽引用。
#[axum::debug_handler]
pub async fn ask(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<AskRequest>,
) -> Result<Json<AskResponse>, RagError> {
    let start = std::time::Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("问题为空");
        return Err(RagError::BadRequest("问题不能为空".to_string()));
    }

    // 1. 检索。Naive 的关键点：**用原问题**检索，不改写。
    let retrieval = match payload.top_k {
        Some(k) if k > 0 => state.retriever.retrieve_with_top_k(question, k).await?,
        Some(_) => {
            warn!(top_k = ?payload.top_k, "top_k 必须大于 0");
            return Err(RagError::BadRequest("top_k 必须大于 0".to_string()));
        }
        None => state.retriever.retrieve(question).await?,
    };

    // 2. 拼 Prompt。
    let prompt = crate::prompt::assemble(&retrieval, question);

    // 3. 生成 + 抽引用。
    let generated = state
        .generator
        .generate_with_citations(&prompt, &retrieval)
        .await?;

    info!(
        question = %question,
        retrieved = retrieval.chunks.len(),
        citations = generated.citations.len(),
        elapsed_ms = start.elapsed().as_millis() as u64,
        "问答完成"
    );

    Ok(Json(AskResponse {
        answer: generated.answer,
        citations: generated.citations,
        retrieved: if payload.include_context {
            retrieval.chunks
        } else {
            Vec::new()
        },
        elapsed_ms: start.elapsed().as_millis() as u64,
        usage: generated.usage,
        model: generated.model,
    }))
}

/// POST /api/agent —— Agent 问答。
///
/// 流程：问题 → 模型（可能自己调检索工具）→ 答案。
#[axum::debug_handler]
pub async fn agent(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<AskRequest>,
) -> Result<Json<AgentResponse>, RagError> {
    let start = std::time::Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("Agent 问题为空");
        return Err(RagError::BadRequest("问题不能为空".to_string()));
    }

    let result = state.agent.run(question).await?;

    info!(
        question = %question,
        iterations = result.iterations,
        tool_call_count = result.tool_calls.len(),
        hit_max_iterations = result.hit_max_iterations,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Agent 问答完成"
    );

    Ok(Json(AgentResponse {
        answer: result.answer,
        tool_calls: result.tool_calls,
        iterations: result.iterations,
        hit_max_iterations: result.hit_max_iterations,
        elapsed_ms: start.elapsed().as_millis() as u64,
        usage: result.usage,
        model: result.model,
    }))
}

/// POST /api/search —— 只检索不生成。
///
/// 调试用：能直接看到「召回了什么」，不用等模型把上下文消化完再猜。
#[axum::debug_handler]
pub async fn search(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<SearchRequest>,
) -> Result<Json<SearchResponse>, RagError> {
    let start = std::time::Instant::now();

    let query = payload.query.trim();
    if query.is_empty() {
        warn!("搜索查询词为空");
        return Err(RagError::BadRequest("查询词不能为空".to_string()));
    }

    let result = match payload.top_k {
        Some(k) if k > 0 => state.retriever.retrieve_with_top_k(query, k).await?,
        Some(_) => {
            warn!(top_k = ?payload.top_k, "top_k 必须大于 0");
            return Err(RagError::BadRequest("top_k 必须大于 0".to_string()));
        }
        None => state.retriever.retrieve(query).await?,
    };

    debug!(
        query = %query,
        hit_count = result.chunks.len(),
        elapsed_ms = start.elapsed().as_millis() as u64,
        "搜索完成"
    );

    Ok(Json(SearchResponse {
        query: result.query.clone(),
        chunks: result.chunks,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// POST /api/reindex —— 重建索引。
///
/// 文档更新后调这个。重建期间会阻塞（嵌入模型是 CPU 密集的），
/// 所以这是个「管理端点」而不是「用户端点」。

#[axum::debug_handler]
pub async fn reindex(State(state): State<AppState>) -> Result<Json<ReindexResponse>, RagError> {
    let start = std::time::Instant::now();

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

    info!(
        chunk_count = stats.chunk_count,
        document_count = stats.document_count,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "索引重建完成"
    );

    Ok(Json(ReindexResponse {
        ok: true,
        stats,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}
