//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `agent` / `tool` 里，handler 不做任何业务判断。

use std::time::Instant;

use agent_logging::{debug, info, warn};
use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;

use crate::agent::ParsedOutput;
use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    CallToolRequest, CallToolResponse, CompareRequest, CompareResponse, ModeResult, ParseRequest,
    ParseResponse, RunRequest, RunResponse, ToolInfo, ToolListResponse,
};
use crate::config::ToolMode;
use crate::error::AppError;
use crate::tool::ToolContext;

/// GET /health —— 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET / —— 服务信息 + 工具清单。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "14-agent-tool 自定义工具服务",
        "version": env!("CARGO_PKG_VERSION"),
        "agent": {
            "mode": state.config.agent.mode.name(),
            "max_steps": state.config.agent.max_steps,
            "few_shot": state.config.prompt.few_shot,
        },
        "model": {
            "name": state.config.model,
            "temperature": state.config.temperature,
            "max_tokens": state.config.max_tokens,
        },
        "mcp": {
            "enabled": state.mcp_client.is_some(),
            "endpoint": state.mcp_client.as_ref().map(|c| c.endpoint().to_string()),
        },
        "tools": state.agent.tool_specs().iter().map(|spec| {
            serde_json::json!({
                "name": spec.name,
                "description": spec.description,
                "schema": spec.schema,
                "parameters": spec.render_parameters(),
                "return_direct": spec.return_direct,
            })
        }).collect::<Vec<_>>(),
        "modes": ToolMode::all().iter().map(|m| m.name()).collect::<Vec<_>>(),
        "endpoints": [
            "GET  /api/tools",
            "POST /api/tools/call",
            "POST /api/run",
            "POST /api/compare",
            "POST /api/parse",
        ],
    }))
}

/// GET /api/tools —— 工具清单。
///
/// ## 为什么这个端点值得存在
///
/// 工具的「契约」是模型行为的一部分 —— 所以它必须**可检视**。
/// 这个端点把每个工具的名字、描述、Schema、渲染后的参数说明
/// 全都列出来，一眼就能看出「模型到底被告知了什么」。
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
            return_direct: spec.return_direct,
        })
        .collect::<Vec<_>>();

    Json(ToolListResponse {
        count: tools.len(),
        tools,
        mcp_endpoint: state.mcp_client.as_ref().map(|c| c.endpoint().to_string()),
    })
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
///
/// 这正是 [`crate::tool::ToolRegistry::call_strict`] 的用途。
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

/// 解析运行模式（缺省用配置里的默认值）。
fn resolve_mode(raw: Option<&String>, fallback: ToolMode) -> Result<ToolMode, AppError> {
    match raw {
        None => Ok(fallback),
        Some(value) => ToolMode::parse(value).ok_or_else(|| {
            AppError::BadRequest(format!(
                "未知的运行模式: {value}（可选: direct / act_only / react）"
            ))
        }),
    }
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

    let mode = resolve_mode(payload.mode.as_ref(), state.config.agent.mode)?;

    let run = state.agent.run_with_mode(question, mode).await?;

    info!(
        mode = mode.name(),
        tool_calls = run.tool_calls,
        tools_used = ?run.tools_used,
        steps = run.steps.len(),
        elapsed_ms = run.elapsed_ms,
        "Agent 运行完成"
    );

    Ok(Json(RunResponse::from_run(run)))
}

/// POST /api/compare —— 三模式消融对照 ★。
///
/// **本项目的核心端点。**
///
/// 对同一个问题依次跑完 `direct` / `act_only` / `react` 三个模式。
///
/// 三者的**变量是单一的**：
///
/// - `direct` → `act_only`：加上了「工具」
/// - `act_only` → `react`：加上了「思考」
#[axum::debug_handler]
pub async fn compare(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<CompareRequest>,
) -> Result<Json<CompareResponse>, AppError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("消融对照请求的问题为空");
        return Err(AppError::BadRequest("问题不能为空".to_string()));
    }

    let modes = match &payload.modes {
        None => ToolMode::all(),
        Some(list) => {
            let mut parsed = Vec::with_capacity(list.len());
            for raw in list {
                parsed.push(ToolMode::parse(raw).ok_or_else(|| {
                    AppError::BadRequest(format!(
                        "未知的运行模式: {raw}（可选: direct / act_only / react）"
                    ))
                })?);
            }
            if parsed.is_empty() {
                parsed = ToolMode::all();
            }
            parsed
        }
    };

    let mut results: Vec<ModeResult> = Vec::with_capacity(modes.len());
    let mut summary: Vec<String> = Vec::with_capacity(modes.len());

    for mode in modes {
        let mode_start = Instant::now();
        info!(mode = mode.name(), "开始跑模式");
        let run = state.agent.run_with_mode(question, mode).await?;

        summary.push(format!(
            "{}: 工具调用 {} 次 ({}), {} 步, 用时 {}ms",
            mode.name(),
            run.tool_calls,
            if run.tools_used.is_empty() {
                "无".to_string()
            } else {
                run.tools_used.join("+")
            },
            run.steps.len(),
            mode_start.elapsed().as_millis()
        ));

        debug!(
            mode = mode.name(),
            tool_calls = run.tool_calls,
            tools_used = ?run.tools_used,
            steps = run.steps.len(),
            hit_max_steps = run.hit_max_steps,
            "模式跑完"
        );

        results.push(ModeResult {
            mode,
            answer: run.answer,
            tool_calls: run.tool_calls,
            tools_used: run.tools_used,
            hit_max_steps: run.hit_max_steps,
            steps: run.steps,
            elapsed_ms: mode_start.elapsed().as_millis() as u64,
        });
    }

    Ok(Json(CompareResponse {
        question: question.to_string(),
        results,
        summary,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// POST /api/parse —— 只解析模型输出（调试解析器）。
#[axum::debug_handler]
pub async fn parse(
    JsonBody(payload): JsonBody<ParseRequest>,
) -> Result<Json<ParseResponse>, AppError> {
    let parsed = crate::agent::parse_react_output(&payload.output);

    let response = match parsed {
        ParsedOutput::FinalAnswer { text } => ParseResponse {
            kind: "final_answer".to_string(),
            thought: String::new(),
            tool: None,
            args: None,
            parse_error: None,
            answer: Some(text),
        },
        ParsedOutput::Action {
            thought,
            tool,
            args,
            parse_error,
        } => ParseResponse {
            kind: "action".to_string(),
            thought,
            tool: Some(tool),
            args: Some(args),
            parse_error,
            answer: None,
        },
        ParsedOutput::Unstructured { text } => ParseResponse {
            kind: "unstructured".to_string(),
            thought: String::new(),
            tool: None,
            args: None,
            parse_error: Some("模型输出既没有 Final Answer 也没有 Action".to_string()),
            answer: Some(text),
        },
    };

    Ok(Json(response))
}
