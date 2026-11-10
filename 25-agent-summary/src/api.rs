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
}

impl AppState {
    /// 新建状态。
    pub fn new(pipeline: Pipeline) -> Arc<Self> {
        Arc::new(Self {
            pipeline: std::sync::Mutex::new(pipeline),
            last_report: std::sync::Mutex::new(None),
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
    build_summary(&state, start, end).await
}

/// 生成周报（POST —— 参数走 JSON body）。
async fn summary_body(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AggregateRequest>,
) -> AppResult<Json<Value>> {
    build_summary(&state, req.start, req.end).await
}

/// 生成周报的公共逻辑。
async fn build_summary(state: &Arc<AppState>, start: i64, end: i64) -> AppResult<Json<Value>> {
    if end < start {
        return Err(AppError::BadRequest("end 不能小于 start".into()));
    }

    let window = Window::new(start, end);
    let mut pipeline = state
        .pipeline
        .lock()
        .map_err(|e| AppError::Internal(format!("状态锁中毒: {e}")))?;

    let out = pipeline.run("cache", &window);
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
