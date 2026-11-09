//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `agent` / `checkpoint` / `longterm` 里，handler 不做任何业务判断。

use agent_logging::{debug, info, warn};
use axum::Json;
use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;

use std::collections::HashMap;

use crate::agent::AgentRun;
use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    ChatRequest, ChatResponse, CompareConclusion, CompareEntry, CompareRequest, CompareResponse,
    DeleteResponse, MemoryListResponse, ThreadDetailResponse, ThreadListResponse,
};
use crate::error::AppError;
use crate::state::ThreadId;

/// GET /health —— 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET / —— 服务信息。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "17-agent-memory 带记忆的 Agent 服务",
        "version": env!("CARGO_PKG_VERSION"),
        "model": {
            "name": state.config.model,
            "temperature": state.config.temperature,
            "max_tokens": state.config.max_tokens,
            "reasoning": !state.config.disable_reasoning,
        },
        "memory": {
            "checkpointer": state.config.storage.checkpointer.name(),
            "memory_store": state.config.storage.memory_store.name(),
            "data_dir": state.config.storage.data_dir,
            "extract_memories": state.config.extract_memories,
            "memory_top_k": state.config.memory_top_k,
            "history_keep": state.config.prompt.history_keep,
        },
        "endpoints": [
            "POST   /api/chat",
            "GET    /api/threads",
            "GET    /api/threads/{thread_id}",
            "DELETE /api/threads/{thread_id}",
            "GET    /api/memories",
            "DELETE /api/memories",
            "POST   /api/compare",
        ],
    }))
}

/// POST /api/chat —— **带记忆的对话**（核心端点）。
#[axum::debug_handler]
pub async fn chat(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    let question = payload.question.trim();
    if question.is_empty() {
        warn!("聊天请求的问题为空");
        return Err(AppError::BadRequest("问题不能为空".to_string()));
    }

    // ⚠️ `thread_id` 的校验在**入口**做 —— 它会成为存储层的 key。
    let thread_id = ThreadId::new(&payload.thread_id).map_err(AppError::BadRequest)?;

    let run = state
        .agent
        .run(&thread_id, &payload.owner, question)
        .await?;

    info!(
        thread = %thread_id,
        owner = %payload.owner,
        recalled = run.recalled.len(),
        extracted = run.extracted.len(),
        history = run.history_messages,
        elapsed_ms = run.elapsed_ms,
        "带记忆的对话完成"
    );

    Ok(Json(ChatResponse::from_run(run)))
}

/// GET /api/threads —— 列出所有线程。
#[axum::debug_handler]
pub async fn list_threads(
    State(state): State<AppState>,
) -> Result<Json<ThreadListResponse>, AppError> {
    let threads = state.agent.list_threads().await?;

    debug!(count = threads.len(), "已列出线程");

    Ok(Json(ThreadListResponse {
        count: threads.len(),
        threads,
    }))
}

/// GET /api/threads/{thread_id} —— 查看某个线程的消息历史。
#[axum::debug_handler]
pub async fn thread_detail(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<ThreadDetailResponse>, AppError> {
    let thread_id = ThreadId::new(&thread_id).map_err(AppError::BadRequest)?;

    let Some(thread) = state.agent.peek(&thread_id).await? else {
        return Err(AppError::NotFound(format!("线程不存在: {thread_id}")));
    };

    Ok(Json(ThreadDetailResponse {
        message_count: thread.message_count(),
        version: thread.version,
        thread_id: thread.thread_id,
        messages: thread.messages,
    }))
}

/// DELETE /api/threads/{thread_id} —— 清空某个线程的短期记忆。
#[axum::debug_handler]
pub async fn delete_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<DeleteResponse>, AppError> {
    let thread_id = ThreadId::new(&thread_id).map_err(AppError::BadRequest)?;

    let deleted = state.agent.forget_thread(&thread_id).await?;

    if deleted {
        info!(thread = %thread_id, "已清空线程的短期记忆");
    } else {
        warn!(thread = %thread_id, "线程不存在 —— 没有可清空的内容");
    }

    Ok(Json(DeleteResponse { deleted, count: 0 }))
}

/// GET /api/memories —— 列出某 owner 的长期记忆。
#[axum::debug_handler]
pub async fn list_memories(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<MemoryListResponse>, AppError> {
    let owner = params
        .get("owner")
        .cloned()
        .unwrap_or_else(|| "default".to_string());

    let memories = state.agent.list_memories(&owner).await?;

    Ok(Json(MemoryListResponse {
        count: memories.len(),
        owner,
        memories,
    }))
}

/// DELETE /api/memories —— 清空某 owner 的长期记忆。
#[axum::debug_handler]
pub async fn delete_memories(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<DeleteResponse>, AppError> {
    let owner = params
        .get("owner")
        .cloned()
        .unwrap_or_else(|| "default".to_string());

    let count = state.agent.forget_memories(&owner).await?;

    info!(owner = %owner, count, "已清空长期记忆");

    Ok(Json(DeleteResponse {
        deleted: count > 0,
        count,
    }))
}

/// POST /api/compare —— ★ **线程隔离 / 恢复对照**。
///
/// ## 它回答两个关键问题
///
/// 1. **线程隔离真的成立吗？**
///    两个线程各聊两轮，然后**实际检查**每个线程的历史条数。
///
///    如果状态串台了（比如两个线程共用了一份状态），
///    条数会翻倍 —— 一眼就能看出来。
///
/// 2. **进程重启后能恢复吗？**
///    同一个 `thread_id` 聊两轮，第二轮的历史条数必须比第一轮多。
///
/// ## 为什么「实际检查」比「声称隔离」有价值
///
/// 隔离是一个**否定性**性质 —— 「不该出现的东西没出现」。
/// 光看响应体看不出来，必须去**读回状态**逐条比对。
#[axum::debug_handler]
pub async fn compare(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<CompareRequest>,
) -> Result<Json<CompareResponse>, AppError> {
    if payload.threads.len() < 2 {
        return Err(AppError::BadRequest(
            "至少需要两个线程才能做隔离对照".to_string(),
        ));
    }

    let question = payload.question.trim();
    if question.is_empty() {
        return Err(AppError::BadRequest("问题不能为空".to_string()));
    }

    // 线程 id 先全部校验 —— 一个非法就整体拒绝。
    let thread_ids: Vec<ThreadId> = payload
        .threads
        .iter()
        .map(|raw| ThreadId::new(raw).map_err(AppError::BadRequest))
        .collect::<Result<_, _>>()?;

    // ── 第一轮：每个线程各聊一次 ─────────────────────────────
    let mut results: Vec<CompareEntry> = Vec::new();
    let mut first_round_counts: Vec<usize> = Vec::new();

    for thread_id in &thread_ids {
        let run = state.agent.run(thread_id, &payload.owner, question).await?;

        first_round_counts.push(run.history_messages);

        results.push(CompareEntry {
            thread_id: thread_id.to_string(),
            answer: run.answer,
            recalled: run.recalled.clone(),
            extracted: run.extracted.clone(),
            history_messages: run.history_messages,
        });
    }

    // ── 第二轮：再聊一次，验证「恢复」 ───────────────────────
    //
    // ⚠️ 同一个 thread_id 第二次调用 —— 历史条数必须增加。
    //    这证明 checkpointer 真的把状态存下来了。
    let mut second_round_counts: Vec<usize> = Vec::new();
    let mut second_round_recalled: Vec<usize> = Vec::new();

    for thread_id in &thread_ids {
        let run = state.agent.run(thread_id, &payload.owner, question).await?;
        second_round_counts.push(run.history_messages);
        second_round_recalled.push(run.recalled.len());
    }

    let thread_resumed = second_round_counts
        .iter()
        .zip(&first_round_counts)
        .all(|(second, first)| second > first);

    // ── 检查线程隔离 ─────────────────────────────────────────
    //
    // ⚠️ 判据：每个线程的历史条数必须**恰好**是「轮数 × 2」
    //    （每轮 = 1 条用户消息 + 1 条助手消息）。
    //
    // 如果两个线程共用了状态，条数会**翻倍**；
    // 如果某个线程的写回丢了，条数会**偏少**。
    //
    // 两种异常都能被这个判据抓到。
    //
    // ⚠️ 这里的轮数是**固定的 2**（上面明确跑了第一轮 + 第二轮），
    //    不是「线程数 + 1」—— 之前写错了，
    //    导致 `expected_per_thread` 随线程数变化，判据完全失效。
    const ROUNDS: usize = 2;
    let expected_per_thread = ROUNDS * 2;
    let threads_isolated = second_round_counts
        .iter()
        .all(|count| *count == expected_per_thread);

    // ── 检查长期记忆是否跨线程共享 ───────────────────────────
    //
    // ⚠️ 判据：**所有**线程在**第二轮**都召回了至少一条事实。
    //
    // 为什么必须是第二轮而不是第一轮 ——
    // 第一轮时记忆库还是**空的**（事实是这一轮结束之后才抽取入库的），
    // 用第一轮判定「长期记忆是否共享」永远得到 false。
    //
    // 第二轮时，事实已经入库，而且**每个线程**都要重新召回一次，
    // 所以这才真正检验到「跨线程共享」。
    let longterm_shared =
        !second_round_recalled.is_empty() && second_round_recalled.iter().all(|count| *count > 0);

    // 把第二轮的条数和召回数写回结果里（方便客户端看）。
    for (entry, count) in results.iter_mut().zip(&second_round_counts) {
        entry.history_messages = *count;
    }

    Ok(Json(CompareResponse {
        results,
        conclusion: CompareConclusion {
            threads_isolated,
            thread_resumed,
            longterm_shared,
        },
    }))
}

/// 从 [`AgentRun`] 取答案（调试用）。
#[allow(dead_code)]
fn answer_of(run: &AgentRun) -> &str {
    &run.answer
}
