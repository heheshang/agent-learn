//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `agent` / `tool` 里，handler 不做任何业务判断。

use agent_logging::{debug, info, warn};
use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;
use std::sync::Arc;

use crate::agent::{Agent, AgentRun};
use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    CallToolRequest, CallToolResponse, CompareEntry, CompareRequest, CompareResponse, RunRequest,
    RunResponse, ToolInfo, ToolListResponse,
};
use crate::error::AppError;
use crate::llm::ToolChoiceMode;

/// GET /health —— 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET / —— 服务信息 + 工具清单。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "16-agent-function-calling Function Calling Agent 服务",
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
        "tools": state.agent.tool_specs().iter().map(|spec| {
            serde_json::json!({
                "name": spec.name,
                "description": spec.description,
                "schema": spec.schema,
                "strict": spec.strict,
                "parameters": spec.render_parameters(),
            })
        }).collect::<Vec<_>>(),
        "endpoints": [
            "GET  /api/tools",
            "POST /api/tools/call",
            "POST /api/run",
            "POST /api/compare",
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
            strict: spec.strict,
            parameters: spec.render_parameters(),
        })
        .collect::<Vec<_>>();

    Json(ToolListResponse {
        count: tools.len(),
        tools,
    })
}

/// POST /api/tools/call —— **直接调用工具**（不经模型）。
///
/// ## 为什么这里工具错误会变成 HTTP 响应
///
/// 在 Agent 循环里，工具失败被转成 `tool` 消息喂回模型 ——
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
        .call_strict(name, &payload.args, &crate::tool::ToolContext::empty())
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
        elapsed_ms = run.elapsed_ms,
        "Agent 运行完成"
    );

    Ok(Json(RunResponse::from_run(run)))
}

/// POST /api/compare —— ★ **tool_choice 四档对照**。
///
/// ## 它回答什么
///
/// 「**模型自己决定要不要用工具**」到底值多少？
///
/// | 档位 | 行为 |
/// |---|---|
/// | `auto` | 模型自己决定（基线） |
/// | `none` | 一个都不调 |
/// | `required` | 强制至少调一个 |
/// | `forced:工具名` | 必须调指定的那个 |
///
/// ## 为什么四档要在**同一个 Agent 实例**上跑
///
/// 工具集、模型、配置全都一样，**只有 `tool_choice` 变** ——
///
/// 这样差值才能归因到 `tool_choice` 这一个变量上。
#[axum::debug_handler]
pub async fn compare(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<CompareRequest>,
) -> Result<Json<CompareResponse>, AppError> {
    let question = payload.question.trim();
    if question.is_empty() {
        warn!("对照请求的问题为空");
        return Err(AppError::BadRequest("问题不能为空".to_string()));
    }

    let mut results = Vec::new();

    for mode in comparison_modes(&state) {
        let entry = run_one_mode(&state, mode, question).await;
        results.push(entry);
    }

    Ok(Json(CompareResponse { results }))
}

/// 四档对照的模式列表。
///
/// ⚠️ `forced` 那一档需要知道工具名 ——
/// 这里取**注册表里的第一个工具**。
fn comparison_modes(state: &AppState) -> Vec<ToolChoiceMode> {
    let mut modes = vec![
        ToolChoiceMode::Auto,
        ToolChoiceMode::None,
        ToolChoiceMode::Required,
    ];

    if let Some(first) = state.agent.registry().names().first() {
        modes.push(ToolChoiceMode::Forced(first.clone()));
    }

    modes
}

/// 跑一个模式并组装对照项。
///
/// ## ⚠️ 单档失败**不中断**整组对照
///
/// 对照实验的价值在于**横向比较** ——
/// 某一档挂了（模型不配合、上游超时），其余几档的数据仍然有价值。
async fn run_one_mode(state: &AppState, mode: ToolChoiceMode, question: &str) -> CompareEntry {
    let agent = build_agent_for_mode(state, mode.clone());

    match agent.run(question).await {
        Ok(run) => entry_from_run(mode, run),
        Err(e) => {
            warn!(mode = mode.name(), error = %e, "对照档位运行失败");
            CompareEntry {
                mode: mode.name().to_string(),
                answer: String::new(),
                tool_calls: 0,
                tools_used: Vec::new(),
                elapsed_ms: 0,
                usage: Default::default(),
                error: Some(e.to_string()),
            }
        }
    }
}

/// 按模式重建一个 Agent。
///
/// ## 为什么重建而不是改配置
///
/// `AppConfig` 是不可变地挂在 `AppState` 上的。
/// 重建一个共享同一份工具注册表、只换 `tool_choice` 的 Agent，
/// 比给配置加 `Arc<Mutex<>>` 干净得多。
fn build_agent_for_mode(state: &AppState, mode: ToolChoiceMode) -> Arc<Agent> {
    let mut config = state.config.clone();
    config.prompt.tool_choice = mode;

    // 复用同一个 LLM 实例 —— 保证四档的模型完全一致。
    // ⚠️ Agent 持有 `Arc<dyn FunctionLlm>`，这里从 state 里拿不到，
    //    所以由 `AppStateInner` 额外存一份。
    let llm = state.llm.clone();

    Arc::new(Agent::new(llm, state.agent.registry().clone(), config))
}

/// 从运行结果组装对照项。
fn entry_from_run(mode: ToolChoiceMode, run: AgentRun) -> CompareEntry {
    CompareEntry {
        mode: mode.name().to_string(),
        answer: run.answer,
        tool_calls: run.tool_calls,
        tools_used: run.tools_used,
        elapsed_ms: run.elapsed_ms,
        usage: run.usage,
        error: None,
    }
}

/// 调试用：打印配置摘要。
pub fn config_summary(state: &AppState) -> String {
    debug!("配置: {}", state.config.summary());
    state.config.summary()
}
