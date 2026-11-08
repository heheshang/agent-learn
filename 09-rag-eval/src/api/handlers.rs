//! 请求处理器。
//!
//! 每个 handler 都是「取参数 → 调核心层 → 组装响应」三段式 ——
//! 核心逻辑全在 `pipeline` / `metrics` / `report` 里，
//! handler 不做任何业务判断。

use std::time::Instant;

use agent_logging::{debug, info, warn};
use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;

use crate::api::AppState;
use crate::api::extract::JsonBody;
use crate::api::models::{
    ChunkIndexStats, DatasetCaseInfo, DatasetInfo, EvalRequest, EvalResponse, QuickEvalRequest,
    QuickEvalResponse, QuickMetricResult, VariantReport,
};
use crate::dataset::EvalCase;
use crate::error::EvalError;
use crate::metrics::types::EvalContext;
use crate::report::{ComparisonReport, Evaluator};

/// GET /health —— 健康检查。
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET / —— 服务信息 + 评估集状态。
pub async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "09-rag-eval — RAG 评估与对比",
        "version": env!("CARGO_PKG_VERSION"),
        "dataset": {
            "name": state.dataset.name,
            "case_count": state.dataset.len(),
        },
        "index": {
            "chunks": state.pipeline.index_stats().chunk_count,
            "dimension": state.pipeline.index_stats().dimension,
            "kind": state.pipeline.index_stats().index_kind,
            "embedding_model": state.pipeline.index_stats().embedding_model,
        },
        "metrics": state.metrics.names(),
        "variants": crate::pipeline::EvalVariant::all()
            .iter()
            .map(|v| v.name())
            .collect::<Vec<_>>(),
        "endpoints": [
            "GET  /api/dataset",
            "POST /api/eval ★",
            "POST /api/quick-eval",
        ],
    }))
}

/// GET /api/dataset —— 评估集详情。
///
/// ## 为什么这个端点重要
///
/// 因为评估集里最隐蔽的错误是**标注没对上**：
/// 标注的章节名在知识库里不存在 → 相关块解析成空 → 召回率恒为 0，
/// 而且**没有任何报错**。
///
/// 这个端点把「每个样本解析出了几个相关块」直接暴露出来 ——
/// 一眼就能看出标注是否有效。
pub async fn dataset(State(state): State<AppState>) -> impl IntoResponse {
    let chunk_index = state.pipeline.chunk_index();

    let cases: Vec<DatasetCaseInfo> = state
        .dataset
        .cases
        .iter()
        .map(|case| {
            let resolved = case.resolve_relevant_ids(chunk_index);
            DatasetCaseInfo {
                id: case.id.clone(),
                question: case.question.clone(),
                has_reference_answer: !case.reference_answer.trim().is_empty(),
                has_relevant: !resolved.is_empty(),
                resolved_relevant_count: resolved.len(),
            }
        })
        .collect();

    Json(DatasetInfo {
        name: state.dataset.name.clone(),
        case_count: state.dataset.len(),
        cases,
        chunk_index: ChunkIndexStats {
            chunk_count: chunk_index.chunk_count(),
            section_count: chunk_index.section_count(),
        },
    })
}

/// POST /api/eval —— 跑完整评估（多变体对比）。
///
/// **本项目的核心端点。**
///
/// 对同一个评估集依次跑完每个变体，然后给出：
///
/// 1. 每个变体的完整报告（聚合分数 + 逐样本明细）
/// 2. 第一个变体 vs 最后一个变体的**差值表**
#[axum::debug_handler]
pub async fn eval(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<EvalRequest>,
) -> Result<Json<EvalResponse>, EvalError> {
    let start = Instant::now();

    // 1. 解析变体。
    let variants = crate::api::models::resolve_variants(payload.variants.as_ref())
        .map_err(EvalError::BadRequest)?;

    // 2. 解析指标。
    let metrics = match &payload.metrics {
        None => state.metrics.metrics().to_vec(),
        Some(names) => state.metrics.select(names).map_err(|missing| {
            EvalError::BadRequest(format!(
                "未注册的指标: {}（可用: {}）",
                missing.join(", "),
                state.metrics.names().join(", ")
            ))
        })?,
    };

    let evaluator = Evaluator::new(metrics);

    // 3. 准备评估集（可能换了路径，或筛了样本 ID）。
    let dataset = build_dataset(
        &state,
        payload.dataset.as_deref(),
        payload.case_ids.as_ref(),
    )?;

    // 4. 逐变体跑。
    let mut reports = Vec::with_capacity(variants.len());
    for variant in variants {
        info!(variant = variant.name(), "开始评估变体");
        let (report, failed) = state
            .pipeline
            .run_variant(&dataset, variant, &evaluator)
            .await?;

        if failed > 0 {
            warn!(variant = variant.name(), failed, "部分样本生成失败");
        }

        reports.push(VariantReport {
            variant: variant.name().to_string(),
            report,
            failed_cases: failed,
        });
    }

    // 5. 对比报告（第一个 vs 最后一个）。
    let comparison = if reports.len() >= 2 {
        Some(ComparisonReport::compare(
            &reports[0].report,
            &reports[reports.len() - 1].report,
        ))
    } else {
        None
    };

    let elapsed_ms = start.elapsed().as_millis() as u64;
    info!(elapsed_ms, variant_count = reports.len(), "完整评估完成");

    Ok(Json(EvalResponse {
        dataset: dataset.name.clone(),
        variants: reports,
        comparison,
        elapsed_ms,
    }))
}
/// POST /api/quick-eval —— 快速评估单个问题。
///
/// ⚠️ **单样本分数不能用来做优化决策** —— 一个样本的波动可能很大。
/// 做决策要看 `/api/eval` 的整集聚合分数。
#[axum::debug_handler]
pub async fn quick_eval(
    State(state): State<AppState>,
    JsonBody(payload): JsonBody<QuickEvalRequest>,
) -> Result<Json<QuickEvalResponse>, EvalError> {
    let start = Instant::now();

    let question = payload.question.trim();
    if question.is_empty() {
        warn!("快速评估的问题为空");
        return Err(EvalError::BadRequest("问题不能为空".to_string()));
    }

    // 构造一个临时样本。
    let case =
        EvalCase::new("quick", question).with_answer(payload.reference_answer.unwrap_or_default());
    let case = if let Some(sections) = payload.relevant_sections {
        case.with_sections(sections)
    } else {
        case
    };

    // 选指标。
    let metrics = match &payload.metrics {
        None => state.metrics.metrics().to_vec(),
        Some(names) => state.metrics.select(names).map_err(|missing| {
            EvalError::BadRequest(format!(
                "未注册的指标: {}（可用: {}）",
                missing.join(", "),
                state.metrics.names().join(", ")
            ))
        })?,
    };

    // 跑系统。
    let mode = crate::api::models::resolve_mode(payload.mode.as_ref())
        .map_err(EvalError::BadRequest)?
        .unwrap_or(rag_08::hybrid::SearchMode::Hybrid);

    let system = &state.pipeline;
    let output = system
        .run_question(question, mode)
        .await
        .map_err(|e| EvalError::Retrieval(format!("跑系统失败: {e}")))?;

    // 算指标。
    let chunk_index = state.pipeline.chunk_index();
    let ctx = EvalContext::new(
        &case,
        output.answer.clone(),
        output.contexts.clone(),
        output.retrieved_ids.clone(),
        chunk_index,
    );

    let evaluator = Evaluator::new(metrics);
    let result = evaluator.evaluate_case(&ctx).await;

    let mut metric_results: Vec<QuickMetricResult> = result
        .scores
        .iter()
        .map(|(name, score)| QuickMetricResult {
            name: name.clone(),
            value: score.value,
            reason: score.reason.clone(),
            passed: score.passed(
                state
                    .metrics
                    .get(name)
                    .map(|m| m.threshold())
                    .unwrap_or(0.7),
            ),
        })
        .collect();

    // 失败的指标也要报出来 —— 否则用户以为「这个指标是 0 分」。
    for (name, error) in &result.errors {
        warn!(metric = name, error = %error, "指标计算失败");
        metric_results.push(QuickMetricResult {
            name: name.clone(),
            value: 0.0,
            reason: Some(format!("计算失败: {error}")),
            passed: false,
        });
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;
    debug!(
        question,
        metric_count = metric_results.len(),
        elapsed_ms,
        "快速评估完成"
    );

    Ok(Json(QuickEvalResponse {
        question: question.to_string(),
        answer: output.answer,
        retrieved_ids: output.retrieved_ids,
        metrics: metric_results,
        elapsed_ms,
    }))
}

/// 准备评估集（换路径 / 筛样本）。
fn build_dataset(
    state: &AppState,
    dataset_path: Option<&str>,
    case_ids: Option<&Vec<String>>,
) -> Result<crate::dataset::EvalDataset, EvalError> {
    let mut dataset = match dataset_path {
        Some(path) => crate::dataset::EvalDataset::load(std::path::Path::new(path))?,
        None => state.dataset.clone(),
    };

    if let Some(ids) = case_ids {
        dataset.cases.retain(|c| ids.contains(&c.id));
        if dataset.cases.is_empty() {
            return Err(EvalError::BadRequest(format!(
                "按 ID 筛选后没有剩下任何样本（请求了 {} 个 ID）",
                ids.len()
            )));
        }
    }

    dataset.validate()?;
    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use crate::metrics::MetricRegistry;
    use std::sync::Arc;

    #[test]
    fn all_refs_returns_every_metric() {
        let registry = MetricRegistry::deterministic_only();
        assert_eq!(registry.all_refs().len(), registry.names().len());
    }

    #[test]
    fn registry_reports_judge_metrics_when_llm_present() {
        // 用脚本化 LLM —— 零网络、零延迟。
        let llm: Arc<dyn rag_07::llm::Llm> = Arc::new(rag_07::llm::ScriptedLlm::always("{}"));
        let registry = MetricRegistry::new(Some(llm), None);

        assert!(registry.names().contains(&"faithfulness"));
        assert!(
            !registry.names().contains(&"answer_relevancy"),
            "没有嵌入器时不应该注册需要嵌入的指标"
        );
    }
}
