//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `agent` / `tools` 里，handler 不做任何业务判断。

use std::time::Instant;

use agent_logging::{debug, info, warn};
use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;

use crate::agent::ParsedOutput;
use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    CompareRequest, CompareResponse, ModeResult, ParseRequest, ParseResponse, RunRequest,
    RunResponse,
};
use crate::config::AgentMode;
use crate::error::AgentError;

/// GET /health —— 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET / —— 服务信息 + 工具清单 + 知识库状态。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "13-agent-core ReAct agent",
        "version": env!("CARGO_PKG_VERSION"),
        "knowledge": {
            "data_dir": state.config.data_dir.display().to_string(),
            "chunks": state.knowledge_chunks,
        },
        "agent": {
            "mode": state.config.agent.mode.name(),
            "max_steps": state.config.agent.max_steps,
            "max_observation_chars": state.config.agent.max_observation_chars,
            "search_top_k": state.config.agent.search_top_k,
            "few_shot": state.config.prompt.few_shot,
        },
        "model": {
            "name": state.config.model,
            "temperature": state.config.temperature,
            "max_tokens": state.config.max_tokens,
        },
        "tools": state.agent.tools().iter().map(|t| {
            let spec = t.spec();
            serde_json::json!({
                "name": spec.name,
                "description": spec.description,
                "schema": spec.schema,
            })
        }).collect::<Vec<_>>(),
        "modes": AgentMode::all().iter().map(|m| m.name()).collect::<Vec<_>>(),
        "endpoints": [
            "POST /api/run",
            "POST /api/compare",
            "POST /api/parse",
        ],
    }))
}

/// 解析运行模式（缺省用配置里的默认值）。
fn resolve_mode(raw: Option<&String>, fallback: AgentMode) -> Result<AgentMode, AgentError> {
    match raw {
        None => Ok(fallback),
        Some(value) => AgentMode::parse(value).ok_or_else(|| {
            AgentError::BadRequest(format!(
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
) -> Result<Json<RunResponse>, AgentError> {
    let question = payload.question.trim();
    if question.is_empty() {
        warn!("运行请求的问题为空");
        return Err(AgentError::BadRequest("问题不能为空".to_string()));
    }

    let mode = resolve_mode(payload.mode.as_ref(), state.config.agent.mode)?;

    let run = state.agent.run_with_mode(question, mode).await?;

    info!(
        mode = mode.name(),
        tool_calls = run.tool_calls,
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
/// 对同一个问题依次跑完 `direct` / `act_only` / `react` 三个模式，
/// 然后给出各模式的答案、工具调用次数和步数。
///
/// 三者的**变量是单一的**：
///
/// - `direct` → `act_only`：加上了「工具」
/// - `act_only` → `react`：加上了「思考」
#[axum::debug_handler]
pub async fn compare(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<CompareRequest>,
) -> Result<Json<CompareResponse>, AgentError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("消融对照请求的问题为空");
        return Err(AgentError::BadRequest("问题不能为空".to_string()));
    }

    // 解析模式列表。
    let modes = match &payload.modes {
        None => AgentMode::all(),
        Some(list) => {
            let mut parsed = Vec::with_capacity(list.len());
            for raw in list {
                parsed.push(AgentMode::parse(raw).ok_or_else(|| {
                    AgentError::BadRequest(format!(
                        "未知的运行模式: {raw}（可选: direct / act_only / react）"
                    ))
                })?);
            }
            if parsed.is_empty() {
                parsed = AgentMode::all();
            }
            parsed
        }
    };

    // 逐个模式跑。
    let mut results: Vec<ModeResult> = Vec::with_capacity(modes.len());
    let mut summary: Vec<String> = Vec::with_capacity(modes.len());

    for mode in modes {
        let mode_start = Instant::now();
        info!(mode = mode.name(), "开始跑模式");
        let run = state.agent.run_with_mode(question, mode).await?;

        summary.push(format!(
            "{}: 工具调用 {} 次, {} 步, 用时 {}ms",
            mode.name(),
            run.tool_calls,
            run.steps.len(),
            mode_start.elapsed().as_millis()
        ));

        debug!(
            mode = mode.name(),
            tool_calls = run.tool_calls,
            steps = run.steps.len(),
            hit_max_steps = run.hit_max_steps,
            "模式跑完"
        );

        results.push(ModeResult {
            mode,
            answer: run.answer,
            tool_calls: run.tool_calls,
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
///
/// ## 为什么这个端点值得存在
///
/// 「几乎所有 Agent 演示代码的一半工作量都花在解析模型输出上」——
/// 这句话是博客里的原话。解析器是 Agent 系统里**最容易出错**的部分，
/// 所以给它一个独立的端点，能直接看到「模型这段输出被解析成了什么」。
#[axum::debug_handler]
pub async fn parse(
    JsonBody(payload): JsonBody<ParseRequest>,
) -> Result<Json<ParseResponse>, AgentError> {
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
