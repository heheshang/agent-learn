//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `pipeline` / `transform` / `fusion` 里，
//! handler 不做任何业务判断。

use std::time::Instant;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::extract::JsonBody;
use crate::api::models::{
    AskRequest, AskResponse, CompareRequest, CompareResponse, ReindexResponse, SearchRequest,
    SearchResponse, StrategyResult, TransformRequest, TransformResponse,
};
use crate::api::AppState;
use crate::error::QueryError;
use crate::fusion::FusionMethod;
use crate::transform::Strategy;

/// GET /health —— 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET / —— 服务信息 + 索引状态。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "07-query query transformation",
        "version": env!("CARGO_PKG_VERSION"),
        "index": {
            "chunks": state.index_stats.chunk_count,
            "dimension": state.index_stats.dimension,
            "kind": state.index_stats.index_kind,
            "embedding_model": state.index_stats.embedding_model,
        },
        "transform": {
            "default_strategy": state.config.transform.default_strategy.name(),
            "num_queries": state.config.transform.num_queries,
            "max_sub_questions": state.config.transform.max_sub_questions,
            "model": state.config.transform.transform_model,
        },
        "retrieval": {
            "top_k": state.config.retrieval.top_k,
            "per_query_k": state.config.retrieval.effective_per_query_k(),
            "fusion": state.config.retrieval.fusion.name(),
            "min_score": state.config.retrieval.min_score,
        },
        "generation": {
            "model": state.config.generation.model,
            "temperature": state.config.generation.temperature,
        },
        "strategies": Strategy::all()
            .iter()
            .map(|s| s.name())
            .collect::<Vec<_>>(),
        "endpoints": [
            "POST /api/ask",
            "POST /api/search",
            "POST /api/transform",
            "POST /api/compare",
            "POST /api/reindex",
        ],
    }))
}

/// 解析策略字符串（缺省用配置里的默认值）。
fn resolve_strategy(raw: Option<&String>, fallback: Strategy) -> Result<Strategy, QueryError> {
    match raw {
        None => Ok(fallback),
        Some(value) => Strategy::parse(value).ok_or_else(|| {
            QueryError::BadRequest(format!(
                "未知的改写策略: {value}（可选: none / hyde / multi_query / rewrite / decompose）"
            ))
        }),
    }
}

/// 解析融合方法（缺省用配置里的默认值）。
fn resolve_fusion(raw: Option<&String>, fallback: FusionMethod) -> Result<FusionMethod, QueryError> {
    match raw {
        None => Ok(fallback),
        Some(value) => FusionMethod::parse(value).ok_or_else(|| {
            QueryError::BadRequest(format!(
                "未知的融合方法: {value}（可选: rrf / max_score / concat）"
            ))
        }),
    }
}

/// POST /api/ask —— 改写 + 检索 + 生成。
#[axum::debug_handler]
pub async fn ask(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<AskRequest>,
) -> Result<Json<AskResponse>, QueryError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        return Err(QueryError::BadRequest("问题不能为空".to_string()));
    }

    let strategy = resolve_strategy(
        payload.strategy.as_ref(),
        state.config.transform.default_strategy,
    )?;

    let outcome = state.pipeline.ask(question, strategy).await?;

    Ok(Json(AskResponse {
        answer: outcome.answer,
        citations: outcome.citations,
        transformed_queries: outcome.retrieval.transformed_queries,
        hits: if payload.include_context {
            outcome.retrieval.hits
        } else {
            Vec::new()
        },
        degraded: outcome.retrieval.degraded,
        degrade_reason: outcome.retrieval.degrade_reason,
        transform_ms: outcome.retrieval.transform_ms,
        retrieve_ms: outcome.retrieval.retrieve_ms,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// POST /api/search —— 改写 + 检索（不生成）。
///
/// 调试用：能直接看到「改写成了什么」和「召回了什么」。
#[axum::debug_handler]
pub async fn search(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<SearchRequest>,
) -> Result<Json<SearchResponse>, QueryError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        return Err(QueryError::BadRequest("问题不能为空".to_string()));
    }

    let strategy = resolve_strategy(
        payload.strategy.as_ref(),
        state.config.transform.default_strategy,
    )?;
    let fusion = resolve_fusion(
        payload.fusion.as_ref(),
        state.config.retrieval.fusion,
    )?;

    let result = state.pipeline.retrieve_with(question, strategy, fusion).await?;

    Ok(Json(SearchResponse {
        question: result.original_question,
        strategy: result.strategy,
        transformed_queries: result.transformed_queries,
        hits: result.hits,
        runs: if payload.include_runs {
            result.runs
        } else {
            Vec::new()
        },
        degraded: result.degraded,
        degrade_reason: result.degrade_reason,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// POST /api/transform —— 只改写，不检索。
///
/// 单独暴露这个端点，是因为**改写的质量是可以单独评估的** ——
/// 不用等检索和生成，能快速迭代提示词。
#[axum::debug_handler]
pub async fn transform(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<TransformRequest>,
) -> Result<Json<TransformResponse>, QueryError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        return Err(QueryError::BadRequest("问题不能为空".to_string()));
    }

    let strategy = resolve_strategy(
        payload.strategy.as_ref(),
        state.config.transform.default_strategy,
    )?;

    let transformer = state.pipeline.transformer(strategy);
    let (queries, degraded, reason) = match transformer.transform(question).await {
        Ok(q) if !q.is_empty() => (q, false, None),
        Ok(_) => (
            vec![crate::transform::TransformedQuery::new(question, strategy)],
            true,
            Some("改写返回空结果".to_string()),
        ),
        Err(e) => {
            let fallback = vec![crate::transform::TransformedQuery::new(question, strategy)];
            if e.is_degradable() {
                (fallback, true, Some(format!("改写失败（{e}）")))
            } else {
                return Err(e);
            }
        }
    };

    Ok(Json(TransformResponse {
        question: question.to_string(),
        strategy,
        transformed_queries: queries,
        degraded,
        degrade_reason: reason,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// POST /api/compare —— 多策略对比。
///
/// **本项目的核心端点。**
///
/// 对同一个问题依次跑完多个策略，然后给出：
///
/// 1. 每个策略的改写结果
/// 2. 每个策略的命中块
/// 3. 每个策略是否命中了 `expected` 关键词
/// 4. 策略之间的**召回重叠度矩阵**
///
/// 第 4 项尤其重要 —— 它能直接告诉你「改写到底有没有带来新信息」。
#[axum::debug_handler]
pub async fn compare(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<CompareRequest>,
) -> Result<Json<CompareResponse>, QueryError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        return Err(QueryError::BadRequest("问题不能为空".to_string()));
    }

    let fusion = resolve_fusion(
        payload.fusion.as_ref(),
        state.config.retrieval.fusion,
    )?;

    // 解析策略列表。
    let strategies = match &payload.strategies {
        None => Strategy::all(),
        Some(list) => {
            let mut parsed = Vec::with_capacity(list.len());
            for raw in list {
                parsed.push(Strategy::parse(raw).ok_or_else(|| {
                    QueryError::BadRequest(format!(
                        "未知的改写策略: {raw}（可选: none / hyde / multi_query / rewrite / decompose）"
                    ))
                })?);
            }
            if parsed.is_empty() {
                parsed = Strategy::all();
            }
            parsed
        }
    };

    // 逐个策略跑。
    let mut results: Vec<StrategyResult> = Vec::with_capacity(strategies.len());

    for strategy in strategies {
        let strategy_start = Instant::now();
        let retrieval = state.pipeline.retrieve_with(question, strategy, fusion).await?;

        let hit = payload.expected.as_ref().map(|keywords| {
            let all_text: String = retrieval
                .hits
                .iter()
                .map(|h| format!("{} {}", h.text, h.heading_path.join(" ")))
                .collect::<Vec<_>>()
                .join(" ");
            keywords
                .iter()
                .any(|kw| !kw.is_empty() && all_text.contains(kw))
        });

        let hit_ids = retrieval.hit_ids();

        results.push(StrategyResult {
            strategy,
            transformed_queries: retrieval.transformed_queries,
            hits: retrieval.hits,
            hit_ids,
            hit,
            degraded: retrieval.degraded,
            elapsed_ms: strategy_start.elapsed().as_millis() as u64,
            runs: if payload.include_runs {
                retrieval.runs
            } else {
                Vec::new()
            },
        });
    }

    // 重叠度矩阵。
    let overlap = compute_overlap(&results);

    // 哪些策略命中了期望关键词。
    let successful_strategies = results
        .iter()
        .filter(|r| r.hit == Some(true))
        .map(|r| r.strategy.name().to_string())
        .collect();

    Ok(Json(CompareResponse {
        question: question.to_string(),
        results,
        overlap,
        successful_strategies,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// 计算策略之间的召回重叠度矩阵。
///
/// `overlap[i][j]` = 策略 i 和策略 j 共同召回的块数。
fn compute_overlap(results: &[StrategyResult]) -> Vec<Vec<usize>> {
    let n = results.len();
    let mut matrix = vec![vec![0usize; n]; n];

    for i in 0..n {
        for j in 0..n {
            let count = results[i]
                .hit_ids
                .iter()
                .filter(|id| results[j].hit_ids.contains(id))
                .count();
            matrix[i][j] = count;
        }
    }

    matrix
}

/// POST /api/reindex —— 重建索引。
///
/// 文档更新后调这个。重建期间会阻塞（嵌入模型是 CPU 密集的），
/// 所以这是个「管理端点」而不是「用户端点」。
#[axum::debug_handler]
pub async fn reindex(State(state): State<AppState>) -> Result<Json<ReindexResponse>, QueryError> {
    let start = Instant::now();

    let embedder = crate::index::make_embedder(state.config.embedding_model.to_model())?;
    let options = crate::index::IndexOptions::default();

    let stats = crate::index::build_and_save(
        &embedder,
        &state.config.data_dir,
        &state.config.index_path,
        &options,
    )
    .await?;

    Ok(Json(ReindexResponse {
        ok: true,
        stats,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// 便利函数：把策略解析的错误包装成统一格式（供测试用）。
#[cfg(test)]
pub fn parse_strategy(value: &str) -> crate::error::QueryResult<Strategy> {
    Strategy::parse(value).ok_or_else(|| QueryError::BadRequest(format!("未知的改写策略: {value}")))
}
