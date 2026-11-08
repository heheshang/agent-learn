//! HTTP 处理器 —— 薄的一层。
//!
//! 每个处理器都是同一个三段式：
//!
//! ```text
//! 1. 取参数（校验）──> 失败就返回 BadRequest
//! 2. 调核心层（pipeline / bootstrap）
//! 3. 组装响应
//! ```
//!
//! ⚠️ 处理器里**不做**任何检索策略判断 ——
//! 「用哪个模式」「要不要重排」这些都在 `SearchMode` 和配置里。
//! 这样 CLI（`demo` / `compare`）和 HTTP 走的是**同一套逻辑**。

use std::sync::Arc;
use std::time::Instant;

use agent_logging::{debug, info, warn};
use axum::Json;
use axum::extract::State;

use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    AskRequest, AskResponse, CompareRequest, CompareResponse, CompareVariant, ErrorResponse,
    EvalRequest, ReindexResponse, SearchRequest, SearchResponse,
};
use crate::config::SearchMode;
use crate::error::{RagError, RagResult};

/// `GET /` —— 服务信息。
pub async fn index(State(state): State<AppState>) -> Json<ErrorResponse> {
    let config = &state.config;
    let report = state.index_report.as_ref();

    let body = format!(
        "12-rag-summary —— 摘要式 RAG（混合检索 + Reranker + Milvus）\n\
         \n\
         端点:\n\
           GET  /health        健康检查\n\
           POST /api/search    纯检索\n\
           POST /api/ask       完整问答\n\
           POST /api/compare   同题消融对照\n\
           POST /api/eval      跑评估集\n\
           POST /api/reindex   重建索引\n\
         \n\
         当前配置:\n\
           模式       : {}\n\
           Milvus     : {} @ {}\n\
           Collection : {}\n\
           嵌入模型   : {}\n\
           检索       : top_k={} 每路召回={}\n\
           重排       : 启用={} 候选={} top_n={}\n\
           压缩       : 启用={} 预算={}\n\
           索引       : {}\n",
        config.retrieval.mode.name(),
        config.milvus.collection,
        config.milvus.uri,
        config.milvus.collection,
        crate::config::model_name(config.embedding_model),
        config.retrieval.top_k,
        config
            .retrieval
            .effective_per_path_k(config.rerank.candidate_k),
        config.rerank.enabled,
        config.rerank.candidate_k,
        config.rerank.top_n,
        config.compress.enabled,
        config.compress.budget_tokens,
        report
            .map(|r| r.summary())
            .unwrap_or_else(|| "未建立".to_string()),
    );

    Json(ErrorResponse {
        error: "OK".to_string(),
        message: body,
    })
}

/// `GET /health` —— 健康检查。
pub async fn health(State(state): State<AppState>) -> Result<Json<ErrorResponse>, RagError> {
    let backend = crate::bootstrap::check_health(&state.config).await?;

    Ok(Json(ErrorResponse {
        error: "OK".to_string(),
        message: format!(
            "{} | 模式={} | 服务端 BM25={}",
            backend,
            state.config.retrieval.mode.name(),
            state.config.milvus.enable_bm25
        ),
    }))
}

/// `POST /api/search` —— 纯检索。
pub async fn search(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<SearchRequest>,
) -> Result<Json<SearchResponse>, RagError> {
    let question = require_question(&payload.question)?;
    let mode = resolve_mode(&state, payload.mode.as_deref())?;

    let top_k = payload.top_k.unwrap_or(state.config.retrieval.top_k).max(1);

    let start = Instant::now();
    let outcome = state.pipeline.retrieve_with(&question, mode).await?;

    // 截断到请求的条数。
    let mut hits = outcome.hits.clone();
    hits.truncate(top_k);

    debug!(
        mode = mode.name(),
        path = outcome.path.name(),
        hit_count = hits.len(),
        elapsed_ms = outcome.elapsed_ms,
        "检索完成"
    );

    Ok(Json(SearchResponse {
        question,
        mode,
        path: outcome.path.name().to_string(),
        hits,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// `POST /api/ask` —— 完整问答。
pub async fn ask(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<AskRequest>,
) -> Result<Json<AskResponse>, RagError> {
    let question = require_question(&payload.question)?;
    let mode = resolve_mode(&state, payload.mode.as_deref())?;

    let outcome = state.pipeline.ask_with(&question, mode).await?;

    info!(
        mode = mode.name(),
        answer_len = outcome.answer.chars().count(),
        citations = outcome.citations.len(),
        passages = outcome.compression.stats.passages.len(),
        elapsed_ms = outcome.elapsed_ms,
        "问答完成"
    );

    Ok(Json(AskResponse {
        answer: outcome.answer,
        citations: outcome.citations,
        passages: outcome.compression.stats.passages.clone(),
        context: outcome.context,
        compression: outcome.compression.clone(),
        retrieval: outcome.retrieval,
        elapsed_ms: outcome.elapsed_ms,
    }))
}

/// `POST /api/compare` —— 同题消融对照。
///
/// 把同一个问题分别喂给阶梯上的每一档策略，
/// 返回各档的命中和**块 ID 重合度**。
pub async fn compare(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<CompareRequest>,
) -> Result<Json<CompareResponse>, RagError> {
    let question = require_question(&payload.question)?;
    let start = Instant::now();

    let modes = resolve_modes(payload.modes.as_deref())?;

    let mut variants = Vec::with_capacity(modes.len());
    let mut id_sets: Vec<Vec<String>> = Vec::with_capacity(modes.len());

    let mode_count = modes.len();
    for mode in &modes {
        match state.pipeline.retrieve_with(&question, *mode).await {
            Ok(outcome) => {
                let ids: Vec<String> = outcome.hits.iter().map(|h| h.id.clone()).collect();
                id_sets.push(ids.clone());

                variants.push(CompareVariant {
                    mode: mode.name().to_string(),
                    ok: true,
                    hits: outcome.hits,
                    elapsed_ms: outcome.elapsed_ms,
                    error: None,
                });
            }
            Err(e) => {
                id_sets.push(Vec::new());
                warn!(mode = mode.name(), error = %e, "消融档位检索失败");
                variants.push(CompareVariant {
                    mode: mode.name().to_string(),
                    ok: false,
                    hits: Vec::new(),
                    elapsed_ms: 0,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    let overlap = compute_overlap(&id_sets);

    info!(variant_count = mode_count, overlap, "消融对照完成");

    Ok(Json(CompareResponse {
        question,
        variants,
        overlap,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// `POST /api/eval` —— 跑评估集。
///
/// ## ⚠️ 这是「重活」
///
/// 22 个样本 × N 个变体，每个样本要 1 次生成 + 8 个确定性指标。
/// 全量跑要几分钟 —— 所以它**不是**同步阻塞的默认路径，
/// 而是提供一个明确要调用的端点。
pub async fn eval(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<EvalRequest>,
) -> Result<Json<crate::eval::AblationResult>, RagError> {
    let dataset = rag_09::dataset::EvalDataset::load(&state.config.eval_dataset_path)
        .map_err(|e| RagError::Internal(format!("加载评估集失败: {e}")))?;

    let chunk_index =
        crate::eval::build_chunk_index(&state.config.data_dir, &state.config.splitter)?;

    let registry = rag_09::metrics::MetricRegistry::new(None, None);

    let runner = crate::eval::AblationRunner::new(
        Arc::clone(&state.pipeline),
        Arc::new(chunk_index),
        registry,
    );

    let modes = resolve_modes(payload.modes.as_deref())?;

    let result = runner
        .run_ladder_limited(&dataset, &modes, payload.limit)
        .await?;

    info!(variant_count = modes.len(), "消融评估完成");

    Ok(Json(result))
}

/// `POST /api/reindex` —— 重建索引。
pub async fn reindex(State(state): State<AppState>) -> Result<Json<ReindexResponse>, RagError> {
    info!(collection = %state.config.milvus.collection, "开始重建索引");
    let report = crate::bootstrap::ensure_index(&state.config).await?;

    info!(
        chunks = report.chunks,
        server_count = report.server_count,
        bm25_docs = report.bm25_docs,
        created = report.created,
        elapsed_ms = report.elapsed_ms,
        "索引重建完成"
    );

    Ok(Json(ReindexResponse {
        collection: report.collection.clone(),
        chunks: report.chunks,
        server_count: report.server_count,
        bm25_docs: report.bm25_docs,
        created: report.created,
        elapsed_ms: report.elapsed_ms,
    }))
}

/// 校验问题非空。
fn require_question(question: &str) -> RagResult<String> {
    let trimmed = question.trim();
    if trimmed.is_empty() {
        warn!("请求里的问题为空");
        return Err(RagError::BadRequest("问题不能为空".to_string()));
    }
    Ok(trimmed.to_string())
}

/// 解析检索模式（缺省用配置里的默认）。
fn resolve_mode(state: &AppState, mode: Option<&str>) -> RagResult<SearchMode> {
    match mode {
        None => Ok(state.config.retrieval.mode),
        Some(value) => SearchMode::parse(value).ok_or_else(|| {
            RagError::BadRequest(format!(
                "未知模式「{value}」，可选: dense / keyword / hybrid / hybrid_rerank / full"
            ))
        }),
    }
}

/// 解析要对比的模式列表（缺省跑完整阶梯）。
fn resolve_modes(modes: Option<&[String]>) -> RagResult<Vec<SearchMode>> {
    match modes {
        None => Ok(SearchMode::all()),
        Some([]) => Ok(SearchMode::all()),
        Some(list) => {
            let mut parsed = Vec::with_capacity(list.len());
            for value in list {
                parsed.push(SearchMode::parse(value).ok_or_else(|| {
                    RagError::BadRequest(format!(
                        "未知模式「{value}」，可选: dense / keyword / hybrid / hybrid_rerank / full"
                    ))
                })?);
            }
            Ok(parsed)
        }
    }
}

/// 计算多组结果的**平均两两重合度**。
///
/// ```text
/// overlap(a, b) = |ids(a) ∩ ids(b)| / |ids(a) ∪ ids(b)|   （Jaccard）
/// ```
///
/// 重合度高 = 各档策略结果差不多；低 = 有独特贡献。
fn compute_overlap(id_sets: &[Vec<String>]) -> f32 {
    if id_sets.len() < 2 {
        return 0.0;
    }

    let mut total = 0.0f32;
    let mut pairs = 0usize;

    for i in 0..id_sets.len() {
        for j in (i + 1)..id_sets.len() {
            let a: std::collections::HashSet<&String> = id_sets[i].iter().collect();
            let b: std::collections::HashSet<&String> = id_sets[j].iter().collect();

            let union = a.union(&b).count();
            if union == 0 {
                continue;
            }

            total += a.intersection(&b).count() as f32 / union as f32;
            pairs += 1;
        }
    }

    if pairs == 0 {
        0.0
    } else {
        total / pairs as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_question() {
        assert!(require_question("").is_err());
        assert!(require_question("   ").is_err());
    }

    #[test]
    fn trims_question() {
        assert_eq!(require_question("  问题  ").unwrap(), "问题");
    }

    #[test]
    fn overlap_of_identical_sets_is_one() {
        let sets = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "b".to_string()],
        ];
        assert!((compute_overlap(&sets) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn overlap_of_disjoint_sets_is_zero() {
        let sets = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string(), "d".to_string()],
        ];
        assert!(compute_overlap(&sets).abs() < 1e-6);
    }

    #[test]
    fn overlap_of_partial_sets_is_fractional() {
        let sets = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["b".to_string(), "c".to_string()],
        ];
        // 交集 {b}，并集 {a,b,c} → 1/3
        assert!((compute_overlap(&sets) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn overlap_needs_at_least_two_sets() {
        assert_eq!(compute_overlap(&[vec!["a".to_string()]]), 0.0);
    }
}
