//! 处理器。

use agent_18::middleware::ToolCallRequest;
use axum::Json;
use axum::extract::State;

use crate::api::AppState;
use crate::api::models::{
    CallToolRequest, CallToolResponse, PlanRequest, PlanResponse, ResearchRequest,
    ResearchResponse, RunStats, WeeklyRequest, WeeklyResponse,
};
use crate::error::AppError;

/// 服务信息。
pub async fn root(State(state): State<AppState>) -> Json<serde_json::Value> {
    let specs = state.agent.tool_specs();

    Json(serde_json::json!({
        "service": "agent-19",
        "title": "研究助手 Agent（RAG + Web 搜索 + 周报 / 项目构建）",
        "model": state.llm.model_name(),
        "search_backend": state.config.search_backend.name(),
        "tools": specs.iter().map(|s| serde_json::json!({
            "name": s.name,
            "description": s.description,
            "schema": s.schema,
        })).collect::<Vec<_>>(),
        "endpoints": [
            "POST /api/research",
            "POST /api/weekly",
            "POST /api/plan",
            "POST /api/tools/call",
            "GET  /api/evidence",
        ],
    }))
}

/// 健康检查。
pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// ★ 研究阶段（拿证据，不成文）。
///
/// ## 为什么单独暴露这个端点
///
/// 因为**证据**才是这个项目最有价值的产物 ——
/// 想看「模型到底查到了什么」时，用它比看周报更直接。
pub async fn research(
    State(state): State<AppState>,
    Json(req): Json<ResearchRequest>,
) -> Result<Json<ResearchResponse>, AppError> {
    let system = crate::prompt::research_system();
    let (run, _) = state.agent.research(&req.question, &system).await?;

    Ok(Json(ResearchResponse::from_run(run)))
}

/// ★ 周度总结。
pub async fn weekly(
    State(state): State<AppState>,
    Json(req): Json<WeeklyRequest>,
) -> Result<Json<WeeklyResponse>, AppError> {
    let result = state.weekly.run(&req.question).await?;

    Ok(Json(WeeklyResponse {
        grounding: result.summary.grounding,
        stats: RunStats::from_run(&result.run),
        summary: result.summary,
        markdown: result.markdown,
    }))
}

/// ★ 项目构建方案。
pub async fn plan(
    State(state): State<AppState>,
    Json(req): Json<PlanRequest>,
) -> Result<Json<PlanResponse>, AppError> {
    let result = state.plan.run(&req.requirement).await?;

    Ok(Json(PlanResponse {
        grounding: result.plan.grounding,
        stats: RunStats::from_run(&result.run),
        plan: result.plan,
        markdown: result.markdown,
    }))
}

/// 直接调工具（不经模型）。
///
/// ## ⚠️ 一个刻意的设计选择
///
/// 「用户直接调工具」的错误**不**降级成观察 ——
/// 而是变成**真正的 HTTP 错误响应**。
///
/// 因为在这个端点里，**用户自己**扮演了「模型」的角色：
/// 他指定了工具和参数，他就该看到**真正的错误**。
pub async fn call_tool(
    State(state): State<AppState>,
    Json(req): Json<CallToolRequest>,
) -> Result<Json<CallToolResponse>, AppError> {
    let outcome = state
        .agent
        .middleware()
        .wrap_tool_call(&ToolCallRequest {
            name: req.tool.clone(),
            arguments: req.args,
            call_id: "direct".to_string(),
        })
        .await;

    if outcome.ok {
        return Ok(Json(CallToolResponse {
            ok: true,
            observation: outcome.observation,
            error_code: None,
            error_kind: None,
            attempts: outcome.attempts,
            retryable: None,
            degradable: None,
        }));
    }

    let error = outcome
        .error
        .unwrap_or_else(|| crate::error::ToolError::internal("unknown", "未知的工具失败"));

    Err(AppError::from_tool_error(&error))
}

/// ★ 证据来源说明。
///
/// ## 为什么做成 API
///
/// 「每种证据从哪来」是调用方最需要知道的契约 ——
/// 它决定了结论能被追查到哪一层。
pub async fn evidence(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "principle": "先查本地（零延迟、结果稳定），再查记忆，最后才联网（有时效性但可能失败）",
        "sources": [
            {
                "kind": "本地知识库",
                "tool": "kb_search",
                "backend": "BM25 倒排索引（08-rag）",
                "guarantee": "确定性 —— 同样的查询永远同样的结果"
            },
            {
                "kind": "长期记忆",
                "tool": "memory",
                "backend": "BM25 事实检索（17-agent-memory）",
                "guarantee": "跨会话持久 —— 记录过的事实会被召回"
            },
            {
                "kind": "Web 搜索",
                "tool": "web_search",
                "backend": state.config.search_backend.name(),
                "guarantee": "非确定性 —— 结果依赖外部服务，可能超时或被限流"
            },
            {
                "kind": "网页正文",
                "tool": "web_fetch",
                "backend": state.config.search_backend.name(),
                "guarantee": "只读 —— 不会修改任何外部状态"
            },
            {
                "kind": "计算",
                "tool": "calculate",
                "backend": "手写算术求值器（13-agent-core）",
                "guarantee": "确定性 —— 统计数字不走模型心算"
            }
        ],
        "invariants": [
            "每个 tool_call 都必须生成恰好一条 tool 消息（少一条整个请求 400）",
            "致命错误也要生成回执，否则下一轮请求会 400",
            "模型调用失败不能降级 —— 没有模型就没有答案",
            "工具失败必须声明类别：Transient 重试 / Business 降级 / Fatal 终止",
        ],
    }))
}
