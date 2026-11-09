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
//! 这和 14/15/16/18 的结论一致：
//! **工具失败不是请求失败 —— 除非用户直接扮演了模型。**

use std::time::Instant;

use axum::Json;
use axum::extract::State;

use crate::api::AppState;
use crate::api::models::{
    BenchRequest, BenchResponse, CallToolRequest, CallToolResponse, ModeReport, RunRequest,
    RunResponse,
};
use crate::bench as bench_mod;
use crate::concurrency::ExecutionMode;
use crate::error::AppError;

/// 服务信息。
pub async fn root(State(state): State<AppState>) -> Json<serde_json::Value> {
    let specs = state.agent.tool_specs();

    Json(serde_json::json!({
        "service": "agent-22",
        "title": "手撕异步处理（并发工具执行 + 限流 + 线程池轨道）",
        "model": state.llm.model_name(),
        "max_concurrency": state.config.max_concurrency,
        "tools": specs.iter().map(|s| serde_json::json!({
            "name": s.name,
            "description": s.description,
            "schema": s.schema,
        })).collect::<Vec<_>>(),
        "endpoints": [
            "POST /api/run",
            "POST /api/tools/call",
            "POST /api/bench",
            "GET  /api/modes",
        ],
    }))
}

/// 健康检查。
pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// 四种并发模式说明。
pub async fn modes() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "modes": [
            {
                "name": ExecutionMode::Sequential.name(),
                "label": ExecutionMode::Sequential.label(),
                "description": "一个接一个等。总耗时 = 各任务耗时之和。",
                "langchain": "（无直接对应；对应手写 for 循环 await）"
            },
            {
                "name": ExecutionMode::JoinAll.name(),
                "label": ExecutionMode::JoinAll.label(),
                "description": "全部并发，**按输入顺序**返回。总耗时 ≈ 最慢的那个任务。",
                "langchain": "batch()"
            },
            {
                "name": ExecutionMode::AsCompleted.name(),
                "label": ExecutionMode::AsCompleted.label(),
                "description": "全部并发，**按完成顺序**返回（乱序）。每条结果带输入下标用于对账。",
                "langchain": "batch_as_completed()"
            },
            {
                "name": "bounded",
                "label": "限流并发（最多 n 个同时跑）",
                "description": "全部并发，但同时最多跑 n 个。防止打爆上游（429）/ 耗尽 fd。",
                "langchain": "RunnableConfig.max_concurrency"
            }
        ],
        "note": "⚠️ 限流组只有在「任务数 > 限流数」时才和全并发有差别。"
    }))
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
    let start = Instant::now();

    // ⚠️ 这里**不走** Agent 循环的降级路径 ——
    //    用户直接调工具，错误就是 HTTP 错误。
    //
    // 但**重试**仍然适用（中间件会重试 Transient）。
    let request = agent_18::middleware::ToolCallRequest {
        name: req.tool.clone(),
        arguments: req.args,
        call_id: "direct".to_string(),
    };

    let outcome = state.agent.middleware().wrap_tool_call(&request).await;

    // ⚠️ 致命错误 → 向上传播（不降级）。
    if let Some(err) = &outcome.error
        && outcome.fatal
    {
        return Err(AppError::from_tool_error(err));
    }

    // 降级的错误 → 400 / 503（按类别）。
    if !outcome.ok
        && let Some(err) = &outcome.error
    {
        return Err(AppError::from_tool_error(err));
    }

    let blocking = state.agent.is_blocking(&req.tool);

    Ok(Json(CallToolResponse {
        tool: req.tool,
        blocking,
        observation: outcome.observation,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// 并发对照实验。
pub async fn bench_handler(
    State(state): State<AppState>,
    Json(req): Json<BenchRequest>,
) -> Result<Json<BenchResponse>, AppError> {
    let cities = if req.cities.is_empty() {
        vec![
            "北京".to_string(),
            "上海".to_string(),
            "广州".to_string(),
            "深圳".to_string(),
        ]
    } else {
        req.cities
    };

    let limit = req.max_concurrency.unwrap_or(state.config.max_concurrency);

    let reports = bench_mod::run_compare(cities, limit, req.simulated).await?;

    let sequential_ms = reports
        .iter()
        .find(|r| r.mode == ExecutionMode::Sequential)
        .map(|r| r.total_ms())
        .unwrap_or(0);

    let mode_reports: Vec<ModeReport> = reports
        .iter()
        .map(|r| ModeReport::from_report(r, sequential_ms))
        .collect();

    let fastest = mode_reports
        .iter()
        .filter(|r| r.failure_count == 0)
        .max_by(|a, b| a.total_ms.cmp(&b.total_ms))
        .map(|r| r.mode.clone())
        .unwrap_or_else(|| "none".to_string());

    let best_speedup = mode_reports
        .iter()
        .filter(|r| r.failure_count == 0)
        .map(|r| r.speedup)
        .fold(0.0f64, f64::max);

    Ok(Json(BenchResponse {
        reports: mode_reports,
        sequential_ms,
        fastest,
        best_speedup,
    }))
}
