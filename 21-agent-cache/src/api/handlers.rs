//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式。

use agent_logging::{debug, info, warn};
use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;

use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    CacheControlRequest, CacheEntryList, CacheStatsResponse, CallToolRequest, CallToolResponse,
    ClearResponse, RunRequest, RunResponse, ToolInfo, ToolListResponse,
};
use crate::cache::CacheControl;
use crate::error::AppError;

/// GET /health —— 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET / —— 服务信息 + 工具清单 + 缓存状态。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "21-agent-cache 带 Redis 缓存的 Agent 服务",
        "version": env!("CARGO_PKG_VERSION"),
        "agent": {
            "max_steps": state.config.agent.max_steps,
            "strict_prompt": state.config.prompt.strict,
            "tool_choice": state.config.prompt.tool_choice.name(),
        },
        "model": {
            "name": state.config.model,
            "temperature": state.config.temperature,
            "max_tokens": state.config.max_tokens,
            "reasoning": !state.config.disable_reasoning,
        },
        "cache": {
            "backend": state.cache.backend_kind(),
            "enabled": state.cache.is_enabled(),
            "namespace": state.cache.namespace(),
            "default_ttl": state.config.cache.default_ttl,
            "tool_ttl": state.config.cache.tool_ttl,
            "cacheable_tools": state.config.cache.cacheable_tools,
        },
        "tools": state.agent.registry().specs().iter().map(|spec| {
            serde_json::json!({
                "name": spec.name,
                "description": spec.description,
                "schema": spec.schema,
                "strict": spec.strict,
                "cacheable": state.cache.is_cacheable_tool(&spec.name),
            })
        }).collect::<Vec<_>>(),
        "endpoints": [
            "GET    /api/tools",
            "POST   /api/tools/call",
            "POST   /api/run",
            "GET    /api/cache/stats",
            "GET    /api/cache/entries",
            "DELETE /api/cache",
        ],
    }))
}

/// GET /api/tools —— 工具清单。
pub async fn list_tools(State(state): State<AppState>) -> impl IntoResponse {
    let tools = state
        .agent
        .registry()
        .specs()
        .iter()
        .map(|spec| ToolInfo {
            name: spec.name.clone(),
            description: spec.description.clone(),
            schema: spec.schema.clone(),
            strict: spec.strict,
            cacheable: state.cache.is_cacheable_tool(&spec.name),
        })
        .collect::<Vec<_>>();

    Json(ToolListResponse {
        count: tools.len(),
        tools,
    })
}

/// POST /api/tools/call —— 直接调用工具（不经模型）。
#[axum::debug_handler]
pub async fn call_tool(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<CallToolRequest>,
) -> Result<Json<CallToolResponse>, AppError> {
    let name = payload.tool.trim();
    if name.is_empty() {
        warn!("工具调用请求的工具名为空");
        return Err(AppError::BadRequest("工具名不能为空".to_string()));
    }

    let args = agent_16::agent::normalize_arguments(&payload.args);

    // ⚠️ 工具结果缓存：**只有白名单里的工具**才查/写。
    let cacheable = state.cache.is_cacheable_tool(name);

    let (outcome, cached) = if cacheable {
        match state.cache.get_tool_result(name, &args).await {
            Some(hit) => (
                agent_16::tool::ToolOutcome {
                    observation: hit.observation,
                    ok: true,
                    tool: hit.tool,
                    data: hit.data,
                },
                true,
            ),
            None => {
                let outcome = state
                    .agent
                    .registry()
                    .call_strict(name, &args, &agent_16::tool::ToolContext::empty())
                    .await?;

                if outcome.ok {
                    let result = crate::cache::CachedToolResult {
                        tool: outcome.tool.clone(),
                        observation: outcome.observation.clone(),
                        data: outcome.data.clone(),
                        stored_at_ms: crate::cache::now_ms(),
                    };
                    state.cache.set_tool_result(name, &args, &result).await;
                }

                (outcome, false)
            }
        }
    } else {
        let outcome = state
            .agent
            .registry()
            .call_strict(name, &args, &agent_16::tool::ToolContext::empty())
            .await?;
        (outcome, false)
    };

    if outcome.ok {
        info!(tool = %outcome.tool, cached, "工具直接调用成功");
    } else {
        warn!(tool = %outcome.tool, observation = %outcome.observation, "工具直接调用失败");
    }

    Ok(Json(CallToolResponse {
        tool: outcome.tool.clone(),
        ok: outcome.ok,
        observation: outcome.observation,
        data: outcome.data,
        cached,
    }))
}

/// POST /api/run —— 跑一次 Agent（可带缓存控制）。
#[axum::debug_handler]
pub async fn run(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<RunRequest>,
) -> Result<Json<RunResponse>, AppError> {
    let question = payload.question.trim();
    if question.is_empty() {
        warn!("运行请求的问题为空");
        return Err(AppError::BadRequest("问题不能为空".to_string()));
    }

    let control = build_control(payload.cache);

    let run = state.agent.run_with_control(question, &control).await?;
    let stats = state.cache.stats().snapshot();

    info!(
        cache_hits = run.cache_hits,
        cache_misses = run.cache_misses,
        tool_calls = run.tool_calls,
        elapsed_ms = run.elapsed_ms,
        "Agent 运行完成"
    );

    Ok(Json(RunResponse::from_run(run, stats)))
}

/// 把请求里的缓存控制转成内部类型。
fn build_control(request: Option<CacheControlRequest>) -> CacheControl {
    let Some(request) = request else {
        return CacheControl::default();
    };

    CacheControl {
        no_cache: request.no_cache,
        no_store: request.no_store,
        ttl: request.ttl,
        smaxage: request.smaxage,
    }
}

/// GET /api/cache/stats —— 缓存统计。
pub async fn cache_stats(State(state): State<AppState>) -> impl IntoResponse {
    Json(CacheStatsResponse {
        backend: state.cache.backend_kind().to_string(),
        enabled: state.cache.is_enabled(),
        namespace: state.cache.namespace().to_string(),
        stats: state.cache.stats().snapshot(),
    })
}

/// GET /api/cache/entries —— 列出缓存条目（排查用）。
///
/// ## ⚠️ 只返回**摘要**，不返回完整内容
///
/// 缓存里存的是模型的完整输出，可能包含敏感内容。
/// 排查时人眼看一下开头就够了。
pub async fn cache_entries(State(state): State<AppState>) -> impl IntoResponse {
    let namespace = state.cache.namespace().to_string();
    let entries = match state.cache.list_entries(&namespace).await {
        Ok(entries) => entries,
        Err(e) => {
            warn!(error = %e, "列出缓存条目失败");
            Vec::new()
        }
    };

    Json(CacheEntryList {
        count: entries.len(),
        entries,
    })
}

/// DELETE /api/cache —— 清空命名空间。
#[axum::debug_handler]
pub async fn clear_cache(State(state): State<AppState>) -> Result<Json<ClearResponse>, AppError> {
    let namespace = state.cache.namespace().to_string();

    let removed = state
        .cache
        .clear_namespace(&namespace)
        .await
        .map_err(|e| AppError::Cache(format!("清空缓存失败: {e}")))?;

    info!(namespace = %namespace, removed, "缓存命名空间已清空");
    Ok(Json(ClearResponse { removed, namespace }))
}

/// 调试用：打印配置摘要。
pub fn config_summary(state: &AppState) -> String {
    debug!("配置: {}", state.config.summary());
    state.config.summary()
}
