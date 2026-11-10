//! HTTP API 层 —— 路由、处理器、状态。
//!
//! ## ⚠️ 错误响应用 `AppError` 自己实现 `IntoResponse`
//!
//! 写成 `Result<_, Response>` 会让 `Err` 变体至少 128 字节，
//! 触发 clippy 的 `result_large_err`（`-D warnings` 门槛）。
//!
//! ## ⚠️ 状态码映射
//!
//! | 类别 | 状态码 |
//! |---|---|
//! | 参数不合法 | **400** |
//! | 上游暂时不可用 | **503**（不是 500 —— 503 的语义是「稍后重试可能成功」） |
//! | 内部错误 | **500** |
//! | 资源不存在 | **404** |

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::agent::ParsedOutput;
use crate::aggregate::Strategy;
use crate::error::{AppError, AppResult};
use crate::metrics::LoadReport;
use crate::model::{Activity, Window};
use crate::pipeline::Pipeline;

/// 应用状态。
pub struct AppState {
    /// 流水线（`Mutex` 包裹 —— 内部不是 `Sync`）。
    pub pipeline: std::sync::Mutex<Pipeline>,

    /// 压测报告（最近一次）。
    pub last_report: std::sync::Mutex<Option<LoadReport>>,

    /// ★ LLM（`None` = 没配 API key，走**离线模式**）。
    ///
    /// ## ⚠️ 为什么用 `Option`
    ///
    /// 没有 `DEEPSEEK_API_KEY` 时服务**仍然要能起来** ——
    /// 因为 25 的核心产出（三档对照压测）是**零网络**的。
    /// LLM 是**可选增强**，不是启动前提。
    pub llm: Option<Arc<dyn crate::llm::FunctionLlm>>,

    /// ★ ReAct 文本协议的 LLM（`None` = 不支持 `protocol=react`）。
    pub react_llm: Option<Arc<dyn crate::llm::Llm>>,

    /// ★ 结构化输出 LLM（`None` = 不支持 `protocol=structured`）。
    pub structured_llm: Option<Arc<dyn crate::llm::StructuredLlm>>,

    /// ★ 工具选择策略（来自 `WK_TOOL_CHOICE`）。
    pub tool_choice: crate::llm::ToolChoice,
}

impl AppState {
    /// 新建状态（**离线模式** —— 没有 LLM）。
    pub fn new(pipeline: Pipeline) -> Arc<Self> {
        Arc::new(Self {
            pipeline: std::sync::Mutex::new(pipeline),
            last_report: std::sync::Mutex::new(None),
            llm: None,
            react_llm: None,
            structured_llm: None,
            tool_choice: crate::llm::ToolChoice::Auto,
        })
    }

    /// 新建状态（**带 LLM**）。
    pub fn with_llm(pipeline: Pipeline, llm: Arc<dyn crate::llm::FunctionLlm>) -> Arc<Self> {
        Arc::new(Self {
            pipeline: std::sync::Mutex::new(pipeline),
            last_report: std::sync::Mutex::new(None),
            llm: Some(llm),
            react_llm: None,
            structured_llm: None,
            tool_choice: crate::llm::ToolChoice::Auto,
        })
    }

    /// ★ 新建状态（带**全套** LLM —— 三条协议都能跑）。
    pub fn with_all_llms(
        pipeline: Pipeline,
        llm: Arc<dyn crate::llm::FunctionLlm>,
        react_llm: Arc<dyn crate::llm::Llm>,
        structured_llm: Arc<dyn crate::llm::StructuredLlm>,
        tool_choice: crate::llm::ToolChoice,
    ) -> Arc<Self> {
        Arc::new(Self {
            pipeline: std::sync::Mutex::new(pipeline),
            last_report: std::sync::Mutex::new(None),
            llm: Some(llm),
            react_llm: Some(react_llm),
            structured_llm: Some(structured_llm),
            tool_choice,
        })
    }
}

/// 查询参数。
#[derive(Debug, Deserialize)]
pub struct ReportQuery {
    /// 窗口起点（Unix 秒）。
    pub start: Option<i64>,
    /// 窗口终点（Unix 秒）。
    pub end: Option<i64>,
}

/// 策略请求体。
#[derive(Debug, Deserialize)]
pub struct StrategyRequest {
    /// 策略名（`naive` / `pre-agg` / `optimized`）。
    pub strategy: String,
}

/// 聚合请求体。
#[derive(Debug, Deserialize)]
pub struct AggregateRequest {
    /// 窗口起点。
    pub start: i64,
    /// 窗口终点。
    pub end: i64,
    /// 查询语句（可选）。
    pub query: Option<String>,
}

/// ★ Agent 请求体。
#[derive(Debug, Deserialize)]
pub struct AgentRequest {
    /// 问题。
    pub question: String,

    /// 窗口起点（可选 —— 默认用全量数据）。
    pub start: Option<i64>,

    /// 窗口终点（可选）。
    pub end: Option<i64>,

    /// 协议：`function`（结构化）/ `react`（文本）/ `structured`（结构化输出）。
    ///
    /// ⚠️ 默认 `function` —— 结构化协议更稳（不用正则解析）。
    #[serde(default)]
    pub protocol: String,
}

/// 检索请求体。
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// 查询词。
    pub query: String,
    /// 返回条数。
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// 默认返回条数。
fn default_limit() -> usize {
    8
}

/// 汇总响应。
#[derive(Debug, Serialize)]
pub struct SummaryResponse {
    /// 命中数。
    pub hits: usize,
    /// 工具调用是否合法。
    pub valid_args: bool,
    /// 是否被护栏拦下。
    pub blocked: bool,
    /// 召回的事实数。
    pub recalled_facts: usize,
    /// 缓存键。
    pub cache_key: String,
    /// 汇总结论。
    pub digest: crate::model::Digest,
}

/// 建路由。
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        // ★ `/summary` 同时支持 GET 和 POST ——
        //    GET 带 query string（方便浏览器直接看），POST 带 JSON body。
        .route("/summary", get(summary_query).post(summary_body))
        .route("/search", post(search))
        .route("/strategy", post(set_strategy))
        .route("/bench", post(bench))
        .route("/modules", get(modules))
        // ★ Agent 端点 —— 真正调模型的链路。
        .route("/agent", post(agent_run))
        .route("/tools", get(tools))
        // ★ 直接调工具（不经模型）——
        //    这时用户扮演的是「模型」的角色，错误**原样返回**。
        .route("/tools/call", post(tool_call))
        .layer(agent_logging::trace_layer())
        .with_state(state)
}

/// 健康检查。
async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// 模块清单 —— 告诉调用方「这个项目汇总了哪些能力」。
async fn modules() -> Json<Value> {
    Json(json!({
        "modules": [
            { "id": "ingest", "name": "数据入口", "from": "04-rag / 08-rag" },
            { "id": "retrieve", "name": "检索", "from": "08-rag" },
            { "id": "agent", "name": "推理与记忆", "from": "13-agent-core / 17-agent-memory" },
            { "id": "tooling", "name": "工具与护栏", "from": "14 / 15 / 18" },
            { "id": "cache", "name": "缓存", "from": "21-agent-cache" },
            { "id": "concurrency", "name": "并发", "from": "22-agent-async" },
            { "id": "aggregate", "name": "聚合三档", "from": "本项目核心" },
        ],
        "strategies": ["naive", "pre-agg", "optimized"],
    }))
}

/// 生成周报（GET —— 参数走 query string）。
async fn summary_query(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ReportQuery>,
) -> AppResult<Json<Value>> {
    let (start, end) = match (q.start, q.end) {
        (Some(s), Some(e)) => (s, e),
        _ => {
            return Err(AppError::BadRequest(
                "必须同时提供 start 与 end（Unix 秒）".into(),
            ));
        }
    };
    summary_with_query(&state, start, end, None).await
}

/// 生成周报（POST —— 参数走 JSON body）。
async fn summary_body(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AggregateRequest>,
) -> AppResult<Json<Value>> {
    summary_with_query(&state, req.start, req.end, req.query).await
}

/// ★ 生成周报 —— **query 真的被用上**。
///
/// ## ⚠️ 修了第一版的一个真 bug
///
/// `AggregateRequest` 定义了 `query: Option<String>`，
/// 但 `build_summary` 里写死了 `pipeline.run("cache", &window)` ——
/// **调用方传的 query 被静默丢弃**。
///
/// 症状：看起来一切正常，但「发散查询」类实验（缓存命中率扫描）
/// 测的其实是同一个查询，结论全部失真。
async fn summary_with_query(
    state: &Arc<AppState>,
    start: i64,
    end: i64,
    query: Option<String>,
) -> AppResult<Json<Value>> {
    if end < start {
        return Err(AppError::BadRequest("end 不能小于 start".into()));
    }

    let window = Window::new(start, end);
    let mut pipeline = state
        .pipeline
        .lock()
        .map_err(|e| AppError::Internal(format!("状态锁中毒: {e}")))?;

    // ⚠️ 没传 query 时才用默认值 —— 传了就用传的。
    let query = query.unwrap_or_else(|| "cache".to_string());

    let out = pipeline.run(&query, &window);
    let markdown = crate::render::render_report(
        &out.digest,
        &window,
        None,
        &[(pipeline.strategy(), pipeline.strategy().label().to_string())],
    );

    Ok(Json(json!({
        "markdown": markdown,
        "digest": out.digest,
        "cache_key": out.cache_key,
        "strategy": pipeline.strategy().name(),
        "query": query,
    })))
}

/// 检索。
async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> AppResult<Json<Value>> {
    if req.query.trim().is_empty() {
        return Err(AppError::BadRequest("query 不能为空".into()));
    }
    if req.limit > 100 {
        return Err(AppError::BadRequest("limit 不能超过 100".into()));
    }

    let pipeline = state
        .pipeline
        .lock()
        .map_err(|e| AppError::Internal(format!("状态锁中毒: {e}")))?;

    let hits = pipeline.search(&req.query, req.limit);
    Ok(Json(json!({ "hits": hits, "count": hits.len() })))
}

/// 切换策略。
async fn set_strategy(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StrategyRequest>,
) -> AppResult<Json<Value>> {
    let strategy = Strategy::parse(&req.strategy).ok_or_else(|| {
        AppError::BadRequest(format!(
            "未知策略 {:?}，可选：naive / pre-agg / optimized",
            req.strategy
        ))
    })?;

    let mut pipeline = state
        .pipeline
        .lock()
        .map_err(|e| AppError::Internal(format!("状态锁中毒: {e}")))?;
    pipeline.set_strategy(strategy);

    Ok(Json(json!({ "strategy": strategy.name() })))
}

/// 跑一次压测（进程内轨）。
async fn bench(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AggregateRequest>,
) -> AppResult<Json<Value>> {
    let window = Window::new(req.start, req.end);
    let activities: Vec<Activity> = crate::pipeline::demo_activities(1000);

    let config = crate::loadtest::LoadConfig {
        requests: 200,
        concurrency: 8,
        service_time_ms: 0,
        server_workers: 4,
    };

    let results = crate::bench::run_three_tier(&activities, &window, &config).await;
    let comparisons = crate::bench::compare_three_tier(&results);

    if let Ok(mut guard) = state.last_report.lock() {
        *guard = Some(results[0].report.clone());
    }

    Ok(Json(json!({
        "results": results.iter().map(|r| json!({
            "variant": r.variant,
            "qps": r.report.qps,
            "p50_ms": r.report.p50_ms,
            "p99_ms": r.report.p99_ms,
            "success_rate": r.report.success_rate,
        })).collect::<Vec<_>>(),
        "comparisons": comparisons,
    })))
}

/// 工具调用示例（给 API 用）。
pub fn render_parsed(parsed: &ParsedOutput) -> Value {
    match parsed {
        ParsedOutput::FinalAnswer(a) => json!({ "type": "final", "answer": a }),
        ParsedOutput::Action { tool, args } => {
            json!({ "type": "action", "tool": tool, "args": args })
        }
        ParsedOutput::Unparsed(t) => json!({ "type": "unparsed", "text": t }),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Agent 与工具端点
// ─────────────────────────────────────────────────────────────────────

/// ★ 列出可用工具。
async fn tools(State(state): State<Arc<AppState>>) -> AppResult<Json<Value>> {
    let pipeline = state
        .pipeline
        .lock()
        .map_err(|e| AppError::Internal(format!("状态锁中毒: {e}")))?;

    let specs = pipeline.registry().specs();
    Ok(Json(json!({
        "tools": specs.iter().map(|s| json!({
            "name": s.name,
            "description": s.description,
            "schema": s.schema,
            "strict": s.strict,
        })).collect::<Vec<_>>(),
    })))
}

/// ★ 直接调工具（不经模型）。
///
/// ## ⚠️ 错误**原样返回**（400 / 500），不降级成观察
///
/// 因为这里用户扮演的是「模型」的角色 ——
/// 他应该看到真正的错误，而不是被包装成一段「观察」文本。
///
/// 这延续 14 / 15 / 16 的结论。
async fn tool_call(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ToolCallRequest>,
) -> AppResult<Json<Value>> {
    // ⚠️ **不能持着 `MutexGuard` 跨 `await`** ——
    //    `MutexGuard` 不是 `Send`，axum 的 Handler 要求 future 是 `Send`。
    let registry = {
        let pipeline = state
            .pipeline
            .lock()
            .map_err(|e| AppError::Internal(format!("状态锁中毒: {e}")))?;
        pipeline.registry().clone()
    };

    let outcome = registry
        .call_strict(&req.tool, &req.args, &crate::tool::ToolContext::empty())
        .await
        .map_err(|e| match e.kind {
            // Business → 400（调用方的问题）。
            crate::tooling::ErrorKind::Business => AppError::BadRequest(e.render()),
            // Transient → 503（稍后重试可能成功）。
            crate::tooling::ErrorKind::Transient => AppError::Unavailable(e.render()),
            // Fatal → 500（服务端的问题）。
            crate::tooling::ErrorKind::Fatal => AppError::Internal(e.render()),
        })?;

    Ok(Json(json!({
        "tool": outcome.tool,
        "ok": outcome.ok,
        "observation": outcome.observation,
        "data": outcome.data,
    })))
}

/// 工具调用请求体。
#[derive(Debug, Deserialize)]
pub struct ToolCallRequest {
    /// 工具名。
    pub tool: String,

    /// 参数。
    pub args: Value,
}

/// ★ 跑一次 Agent（真正调模型）。
///
/// ## ⚠️ 没有 LLM 时返回 **503**
///
/// 不是 400（参数没问题）、不是 500（不是 bug）。
/// 503 的语义是「服务端暂时不可用」——
/// 「没配 API key」正是这个意思。
async fn agent_run(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AgentRequest>,
) -> AppResult<Json<Value>> {
    if req.question.trim().is_empty() {
        return Err(AppError::BadRequest("question 不能为空".into()));
    }

    // ⚠️ 这里**不能**提前检查 `state.llm` ——
    //    `protocol=react` 用的是 `react_llm`，`protocol=structured`
    //    用的是 `structured_llm`。
    //    提前用 `state.llm` 把门，会让「只配了 react LLM」的部署
    //    永远返回 503 —— 而它本来是能跑的。
    //    所以 LLM 的可用性检查**下沉到各协议分支里**。

    // 窗口（没给就用全量）。
    let (start, end) = match (req.start, req.end) {
        (Some(s), Some(e)) => (s, e),
        _ => {
            let pipeline = state
                .pipeline
                .lock()
                .map_err(|e| AppError::Internal(format!("状态锁中毒: {e}")))?;
            let acts = pipeline.activities();
            let min = acts.iter().map(|a| a.ts).min().unwrap_or(0);
            let max = acts.iter().map(|a| a.ts).max().unwrap_or(0);
            (min, max + 1)
        }
    };

    if end < start {
        return Err(AppError::BadRequest("end 不能小于 start".into()));
    }

    let window = Window::new(start, end);

    // ⚠️ **不能持着 `MutexGuard` 跨 `await`** ——
    //    `MutexGuard` 不是 `Send`，axum 的 Handler 要求 future 是 `Send`。
    //
    //    所以：先取出需要的东西（registry 可 Clone、facts 可 Clone），
    //    **释放锁**，再跑 Agent。
    let (registry, facts) = {
        let pipeline = state
            .pipeline
            .lock()
            .map_err(|e| AppError::Internal(format!("状态锁中毒: {e}")))?;

        (pipeline.registry().clone(), pipeline.facts().to_vec())
    };

    // 记忆召回 —— ★ **必须在调模型之前**（事实要进 system 提示）。
    let recalled = crate::agent::recall_facts(&req.question, &facts, 3);

    // ⚠️ 工具选择策略来自配置（`WK_TOOL_CHOICE`）。
    let config = crate::agent_loop::AgentConfig {
        tool_choice: state.tool_choice.clone(),
        ..Default::default()
    };

    // ⚠️★ **协议必须真的分流** —— 原来这里无条件跑 `FunctionAgent`，
    //    `protocol` 字段是**摆设**（写 `react` 也跑 function）。
    //
    //    这会让「协议对照实验」彻底作废：
    //    两次请求走的是**同一条**链路，差值恒等于 0，
    //    但结论会写成「文本协议 vs 结构化协议没有差别」。
    let protocol = req.protocol.trim().to_ascii_lowercase();
    let protocol = if protocol.is_empty() {
        "function".to_string()
    } else {
        protocol
    };

    match protocol.as_str() {
        "react" => {
            let Some(llm) = state.react_llm.clone() else {
                return Err(AppError::Unavailable(
                    "服务端没有配置 ReAct 协议的 LLM".into(),
                ));
            };

            let agent = crate::agent_loop::ReactAgent::new(llm, registry, config);
            let run = agent.run(&req.question, &recalled, &window).await?;

            Ok(Json(json!({
                "protocol": "react",
                "question": run.question,
                "answer": run.answer,
                "steps": run.steps,
                "tool_calls": run.tool_calls,
                "tool_attempts": run.tool_attempts,
                "tools_used": run.tools_used,
                "hit_max_steps": run.hit_max_steps,
                "elapsed_ms": run.elapsed_ms,
                "model": run.model,
                "usage": run.usage,
                "truncated": run.truncated,
                "recalled_facts": run.recalled_facts,
            })))
        }

        "structured" => {
            let Some(structured_llm) = state.structured_llm.clone() else {
                return Err(AppError::Unavailable(
                    "服务端没有配置结构化输出的 LLM".into(),
                ));
            };

            // ⚠️ 阶段 A（研究）也要一个能调工具的 LLM。
            let Some(research_llm) = state.llm.clone() else {
                return Err(AppError::Unavailable(
                    "服务端没有配置 LLM —— 请设置 DEEPSEEK_API_KEY 后重启".into(),
                ));
            };

            let report = crate::structured::run_structured(
                &*structured_llm,
                research_llm,
                &req.question,
                &recalled,
                &registry,
                &window,
            )
            .await?;

            Ok(Json(json!({
                "protocol": "structured",
                "question": report.question,
                "answer": report.summary.answer,
                "summary": report.summary,
                "grounding": report.grounding,
                "evidence_count": report.evidence.len(),
                "elapsed_ms": report.elapsed_ms,
                "model": report.model,
                "usage": report.usage,
                "truncated": report.truncated,
                "recalled_facts": recalled.len(),
            })))
        }

        "function" => {
            let Some(llm) = state.llm.clone() else {
                return Err(AppError::Unavailable(
                    "服务端没有配置 LLM —— 请设置 DEEPSEEK_API_KEY 后重启".into(),
                ));
            };

            let agent = crate::agent_loop::FunctionAgent::new(llm, registry, config);
            let run = agent.run(&req.question, &recalled, &window).await?;

            Ok(Json(json!({
                "protocol": "function",
                "question": run.question,
                "answer": run.answer,
                "steps": run.steps,
                "tool_calls": run.tool_calls,
                "tool_attempts": run.tool_attempts,
                "tools_used": run.tools_used,
                "hit_max_steps": run.hit_max_steps,
                "elapsed_ms": run.elapsed_ms,
                "model": run.model,
                "usage": run.usage,
                "truncated": run.truncated,
                "recalled_facts": run.recalled_facts,
            })))
        }

        other => Err(AppError::BadRequest(format!(
            "未知的 protocol：{other}（可选：function / react / structured）"
        ))),
    }
}
