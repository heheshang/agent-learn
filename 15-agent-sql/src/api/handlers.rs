//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `agent` / `tools` / `guard` 里，handler 不做任何业务判断。

use agent_logging::{debug, info, warn};
use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;

use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    CallToolRequest, CallToolResponse, GuardRequest, GuardResponse, RunRequest, RunResponse,
    SchemaResponse, TableInfo, ToolInfo, ToolListResponse,
};
use crate::error::AppError;
use crate::guard;
use agent_14::tool::ToolContext;

/// GET /health —— 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET / —— 服务信息 + 工具清单。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "15-agent-sql SQL Agent 服务",
        "version": env!("CARGO_PKG_VERSION"),
        "agent": {
            "max_steps": state.config.agent.max_steps,
            "few_shot": state.config.prompt.few_shot,
            "checker": state.config.checker.name(),
        },
        "model": {
            "name": state.config.model,
            "temperature": state.config.temperature,
            "max_tokens": state.config.max_tokens,
        },
        "database": {
            "path": state.db.path().display().to_string(),
            "dialect": crate::prompt::DIALECT,
            "read_only": true,
        },
        "tools": state.agent.tool_specs().iter().map(|spec| {
            serde_json::json!({
                "name": spec.name,
                "description": spec.description,
                "schema": spec.schema,
                "parameters": spec.render_parameters(),
            })
        }).collect::<Vec<_>>(),
        "endpoints": [
            "GET  /api/tools",
            "GET  /api/schema",
            "POST /api/tools/call",
            "POST /api/run",
            "POST /api/guard",
        ],
    }))
}

/// GET /api/tools —— 工具清单。
///
/// 工具的「契约」是模型行为的一部分 —— 所以它必须**可检视**。
pub async fn list_tools(State(state): State<AppState>) -> impl IntoResponse {
    let tools = state
        .agent
        .tool_specs()
        .iter()
        .map(|spec| ToolInfo {
            name: spec.name.clone(),
            description: spec.description.clone(),
            schema: spec.schema.clone(),
            parameters: spec.render_parameters(),
        })
        .collect::<Vec<_>>();

    Json(ToolListResponse {
        count: tools.len(),
        tools,
    })
}

/// GET /api/schema —— 数据库概览。
///
/// 对应「用户问『这个库长什么样』」的场景 ——
/// Agent 自己会用工具查，但这个端点让用户**不用等模型**就能看到。
pub async fn schema(State(state): State<AppState>) -> Result<Json<SchemaResponse>, AppError> {
    let table_names = match state.db.list_tables() {
        Ok(names) => names,
        Err(e) => {
            warn!(error = %e, "读取表清单失败");
            return Err(AppError::Internal(e.to_string()));
        }
    };

    let mut tables = Vec::with_capacity(table_names.len());
    for name in table_names {
        let row_count = state.db.count_rows(&name).unwrap_or(-1);
        let columns = state.db.table_columns(&name).unwrap_or_default();
        let ddl = state.db.table_ddl(&name).ok().flatten();
        tables.push(TableInfo {
            name,
            row_count,
            columns,
            ddl,
        });
    }

    Ok(Json(SchemaResponse {
        path: state.db.path().display().to_string(),
        tables,
    }))
}

/// POST /api/tools/call —— **直接调用工具**（不经模型）。
///
/// ## 为什么这里工具错误会变成 HTTP 响应
///
/// 在 Agent 循环里，工具失败被转成 `Observation` 喂回模型 ——
/// 因为**模型**是那个需要自我纠正的角色。
///
/// 但在这里用户直接指定了工具和参数，他扮演的就是「模型」的角色。
/// 所以错误应该**原样返回给他**，而不是被包装成观察。
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

    let outcome = state
        .agent
        .registry()
        .call_strict(name, &payload.args, &ToolContext::empty())
        .await?;

    if outcome.ok {
        info!(tool = %outcome.tool, "工具直接调用成功");
    } else {
        warn!(tool = %outcome.tool, observation = %outcome.observation, "工具直接调用失败");
    }

    Ok(Json(CallToolResponse {
        tool: outcome.tool.clone(),
        ok: outcome.ok,
        observation: outcome.observation,
        data: outcome.data,
    }))
}

/// POST /api/run —— 跑一次 Agent。
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

    let run = state.agent.run(question).await?;

    info!(
        tool_calls = run.tool_calls,
        tools_used = ?run.tools_used,
        steps = run.steps.len(),
        blocked = run.blocked_sql.len(),
        elapsed_ms = run.elapsed_ms,
        "Agent 运行完成"
    );

    Ok(Json(RunResponse::from_run(run)))
}

/// POST /api/guard —— 只跑护栏（**不执行**）。
///
/// ## 为什么这个端点有价值
///
/// 1. **可测** —— CI 里可以跑一批「已知危险 SQL」，确认护栏没被改坏
/// 2. **可解释** —— 用户可以问「这条 SQL 会被拦吗？为什么？」
/// 3. **零成本** —— 不调模型、不碰数据库
///
/// 请求体：
///
/// ```json
/// { "query": "DELETE FROM Track" }
/// ```
#[axum::debug_handler]
pub async fn guard_check(JsonBody(payload): JsonBody<GuardRequest>) -> Json<GuardResponse> {
    let query = payload.query.trim().to_string();

    let (allowed, reason) = match guard::check_sql(&query) {
        Ok(()) => (true, None),
        Err(e) => (false, Some(e.message())),
    };

    debug!(allowed, "护栏检查完成");

    Json(GuardResponse {
        allowed,
        reason,
        query,
    })
}
