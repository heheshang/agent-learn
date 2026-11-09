//! 处理器。
//!
//! ## ⚠️ 一个刻意的设计选择
//!
//! 「用户直接调工具」的错误**不**降级成观察 ——
//! 而是变成**真正的 HTTP 错误响应**。
//!
//! ## 为什么
//!
//! 在 Agent 循环里，用户扮演的是「旁观者」，模型才是执行者：
//! 工具失败 → 降级成观察 → **模型**读到并自我纠正。
//!
//! 但在「用户直接调工具」这个端点里，
//! **用户自己**扮演了「模型」的角色 ——
//! 他指定了工具和参数，他就该看到**真正的错误**，
//! 而不是一段被包装成「观察」的文本。
//!
//! 这和 14/15/16 的结论一致：
//! **工具失败不是请求失败 —— 除非用户直接扮演了模型。**

use axum::Json;
use axum::extract::State;

use crate::api::AppState;
use crate::api::models::{CallToolRequest, CallToolResponse, RunRequest, RunResponse};
use crate::error::{AppError, ErrorCode, ErrorKind};
use crate::middleware::ToolCallRequest;

/// 服务信息。
pub async fn root(State(state): State<AppState>) -> Json<serde_json::Value> {
    let specs = state.agent.tool_specs();

    Json(serde_json::json!({
        "service": "agent-18",
        "title": "手撕 Agent 错误处理（middleware 驱动的可诊断工具失败）",
        "model": state.llm.model_name(),
        "tools": specs.iter().map(|s| serde_json::json!({
            "name": s.name,
            "description": s.description,
            "schema": s.schema,
        })).collect::<Vec<_>>(),
        "endpoints": [
            "POST /api/run",
            "POST /api/tools/call",
            "GET  /api/errors/table",
        ],
    }))
}

/// 健康检查。
pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// 跑一次 Agent。
pub async fn run(
    State(state): State<AppState>,
    Json(req): Json<RunRequest>,
) -> Result<Json<RunResponse>, AppError> {
    let run = state.agent.run(&req.question).await?;
    Ok(Json(RunResponse::from_run(run)))
}

/// 直接调工具（不经模型）。
pub async fn call_tool(
    State(state): State<AppState>,
    Json(req): Json<CallToolRequest>,
) -> Result<Json<CallToolResponse>, AppError> {
    // ⚠️ 这里**不走**中间件的降级路径 ——
    //    用户直接调工具，错误就是 HTTP 错误。
    //
    // 但**重试**仍然适用 ——
    // 因为「上游临时故障」和「参数填错」在用户这一侧同样有区别。
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

    // ⚠️ 致命错误 → 500（服务端的问题）。
    //    其余 → 按类别映射（Business 400 / Transient 503）。
    let error = outcome
        .error
        .unwrap_or_else(|| crate::error::ToolError::internal("unknown", "未知的工具失败"));

    Err(AppError::from_tool_error(&error))
}

/// ★ 错误分类速查表。
///
/// ## 为什么把它做成 API
///
/// 「什么错误该重试」是最容易记错的地方 ——
/// 把它暴露出来，调用方可以自己查，
/// 也能让**回归测试**直接对着这张表断言。
pub async fn error_table() -> Json<serde_json::Value> {
    let codes = [
        ErrorCode::UpstreamTimeout,
        ErrorCode::UpstreamUnavailable,
        ErrorCode::RateLimited,
        ErrorCode::InvalidArguments,
        ErrorCode::UnknownTool,
        ErrorCode::NotFound,
        ErrorCode::RuleViolation,
        ErrorCode::Unauthorized,
        ErrorCode::InternalFault,
    ];

    let entries: Vec<serde_json::Value> = codes
        .iter()
        .map(|code| {
            let kind = code.default_kind();
            serde_json::json!({
                "code": code.id(),
                "summary": code.summary(),
                "kind": kind.name(),
                "retryable": kind.retryable(),
                "degradable": kind.degradable(),
                "http_status": match kind {
                    ErrorKind::Business => 400,
                    ErrorKind::Transient => 503,
                    ErrorKind::Fatal => 500,
                },
            })
        })
        .collect();

    Json(serde_json::json!({
        "principle": "重试和降级是两个正交的维度：重试=再调一次同样的工具，降级=把错误变成观察让模型换条路",
        "rules": [
            "Transient：重试（指数退避）→ 重试耗尽后降级",
            "Business：不重试，直接降级成观察",
            "Fatal：不重试、不降级，直接终止",
        ],
        "invariants": [
            "每个 tool_call 都必须生成恰好一条 tool 消息（少一条整个请求 400）",
            "致命错误也要生成回执，否则下一轮请求会 400",
            "模型调用失败不能降级 —— 没有模型就没有答案",
        ],
        "entries": entries,
    }))
}
