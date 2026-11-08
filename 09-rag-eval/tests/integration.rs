//! 集成测试 —— 端到端验证评估链路。
//!
//! ## 三个层次
//!
//! | 测试 | 验证什么 | 需要网络/模型 |
//! |---|---|---|
//! | 评估集加载与标注解析 | 数据层的正确性 | ❌ |
//! | 确定性指标端到端 | 指标计算的正确性 | ❌ |
//! | 报告与对比 | 报告聚合与差值计算 | ❌ |
//!
//! **全部不需要网络** —— LLM 用 07 的 `ScriptedLlm` 替身。
//!
//! ## 为什么不用真实模型
//!
//! 因为测试要**确定**。真实模型：
//!
//! - 要 API key
//! - 有抖动（同样输入不同输出）
//! - 慢（CI 里不可接受）
//!
//! 脚本化 LLM 让「提示词 → 解析 → 打分」这段逻辑能被充分覆盖，
//! 而模型的真实行为不在本项目的测试范围内。

use std::sync::Arc;

use rag_09::dataset::{ChunkIndex, EvalCase, EvalDataset};
use rag_09::metrics::Metric;
use rag_09::metrics::deterministic::{Mrr, NdcgAtK, PrecisionAtK, RecallAtK, TokenF1};
use rag_09::metrics::judge::Faithfulness;
use rag_09::metrics::types::EvalContext;
use rag_09::pipeline::EvalVariant;
use rag_09::report::{ComparisonReport, Evaluator, Verdict};

/// 造一个「章节 → 块 ID」的索引。
fn chunk_index() -> ChunkIndex {
    ChunkIndex::build(&[
        ("kb.md#0".to_string(), vec!["BM25 打分算法".to_string()]),
        ("kb.md#1".to_string(), vec!["BM25 打分算法".to_string()]),
        ("kb.md#2".to_string(), vec!["混合检索".to_string()]),
    ])
}

/// 造一个评估样本。
fn case() -> EvalCase {
    EvalCase::new("bm25-formula", "BM25 打分公式是什么")
        .with_answer("BM25 由 IDF、词频饱和和长度归一化三部分组成")
        .with_sections(vec!["BM25 打分算法".to_string()])
}

/// 造一个评估上下文。
fn ctx<'a>(
    case: &'a EvalCase,
    answer: &str,
    retrieved: &[&str],
    index: &'a ChunkIndex,
) -> EvalContext<'a> {
    EvalContext {
        case,
        answer: answer.to_string(),
        contexts: retrieved.iter().map(|s| s.to_string()).collect(),
        retrieved_ids: retrieved.iter().map(|s| s.to_string()).collect(),
        chunk_index: index,
    }
}

#[test]
fn dataset_loads_and_validates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cases.json");

    let dataset = EvalDataset::new(
        "integration",
        vec![
            case(),
            EvalCase::new("rrf", "为什么不能加权求和").with_sections(vec!["混合检索".to_string()]),
        ],
    );
    dataset.save(&path).unwrap();

    let loaded = EvalDataset::load(&path).unwrap();
    loaded.validate().unwrap();

    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded.name, "integration");
    assert_eq!(loaded.cases[0].question, "BM25 打分公式是什么");
}

#[test]
fn section_annotations_resolve_to_chunk_ids() {
    let index = chunk_index();
    let c = case();

    let ids = c.resolve_relevant_ids(&index);

    assert_eq!(ids.len(), 2, "「BM25 打分算法」下有两个块");
    assert!(ids.contains(&"kb.md#0".to_string()));
    assert!(ids.contains(&"kb.md#1".to_string()));
}

#[tokio::test]
async fn deterministic_metrics_work_end_to_end() {
    let index = chunk_index();
    let c = case();
    let ctx = ctx(
        &c,
        "BM25 由 IDF 和词频组成",
        &["kb.md#0", "kb.md#1"],
        &index,
    );

    let f1 = TokenF1::new().score(&ctx).await.unwrap();
    let recall = RecallAtK::new(5).score(&ctx).await.unwrap();
    let precision = PrecisionAtK::new(5).score(&ctx).await.unwrap();
    let ndcg = NdcgAtK::new(5).score(&ctx).await.unwrap();
    let mrr = Mrr.score(&ctx).await.unwrap();

    assert!(f1.value > 0.0, "答案和标准答案有共同 token");
    assert_eq!(recall.value, 1.0, "两个相关块都被召回了");
    assert_eq!(precision.value, 1.0, "召回的全是相关块");
    assert_eq!(ndcg.value, 1.0, "相关块排在最前面");
    assert_eq!(mrr.value, 1.0, "第一个结果就相关");
}

#[tokio::test]
async fn missing_relevant_chunk_lowers_recall() {
    let index = chunk_index();
    let c = case();
    // 只召回了一个相关块（另一个没进来），还混进了一个无关块。
    let ctx = ctx(&c, "答案", &["kb.md#0", "kb.md#9"], &index);

    let recall = RecallAtK::new(5).score(&ctx).await.unwrap();
    let precision = PrecisionAtK::new(5).score(&ctx).await.unwrap();
    let mrr = Mrr.score(&ctx).await.unwrap();

    assert_eq!(recall.value, 0.5, "两个相关块只召回了���个");
    assert_eq!(precision.value, 0.5, "两条里一条相关");
    assert_eq!(mrr.value, 1.0, "第一个就相关");
}

#[tokio::test]
async fn faithfulness_uses_scripted_llm() {
    let index = chunk_index();
    let c = case();
    let ctx = ctx(
        &c,
        "第一句有依据。第二句也有依据。",
        &["kb.md#0", "kb.md#1"],
        &index,
    );

    // 脚本化 LLM：一次调用返回两句话的批量判定。
    let llm: Arc<dyn rag_07::llm::Llm> = Arc::new(rag_07::llm::ScriptedLlm::always(
        r#"{"scores": [1.0, 1.0], "reasons": ["", ""]}"#,
    ));

    let metric = Faithfulness::new(llm);
    let score = metric.score(&ctx).await.unwrap();

    assert_eq!(score.value, 1.0, "两句话都有依据");
    assert!(score.reason.unwrap().contains("2/2"));
}

#[tokio::test]
async fn faithfulness_detects_hallucination() {
    let index = chunk_index();
    let c = case();
    let ctx = ctx(&c, "第一句有依据。第二句是编的。", &["kb.md#0"], &index);

    // 脚本化 LLM：一次调用返回两句话的批量判定（第二句没依据）。
    let llm: Arc<dyn rag_07::llm::Llm> = Arc::new(rag_07::llm::ScriptedLlm::always(
        r#"{"scores": [1.0, 0.0], "reasons": ["", "上下文里没有提到"]}"#,
    ));

    let metric = Faithfulness::new(llm);
    let score = metric.score(&ctx).await.unwrap();

    assert_eq!(score.value, 0.5, "两句里一句是幻觉");
    let reason = score.reason.unwrap();
    assert!(
        reason.contains("无依据"),
        "理由里应该指出哪句有问题：{reason}"
    );
}

#[tokio::test]
async fn judge_failure_is_recorded_not_fatal() {
    let index = chunk_index();
    let c = case();
    let ctx = ctx(&c, "答案", &["kb.md#0"], &index);

    // 脚本化 LLM 没有预设答案时会返回错误 —— 这模拟 judge 挂了。
    let llm: Arc<dyn rag_07::llm::Llm> = Arc::new(rag_07::llm::ScriptedLlm::new(vec![]));

    let metric = Faithfulness::new(llm);
    let err = metric.score(&ctx).await.unwrap_err();

    assert_eq!(err.status_code(), 502);
    assert!(err.is_degradable());
}

#[tokio::test]
async fn evaluator_aggregates_and_reports() {
    let index = chunk_index();
    let c = case();
    let ctx = ctx(&c, "BM25 由 IDF 组成", &["kb.md#0", "kb.md#1"], &index);

    let evaluator = Evaluator::new(vec![Arc::new(TokenF1::new()), Arc::new(RecallAtK::new(5))]);

    let dataset = EvalDataset::new("integration", vec![c.clone()]);
    let report = evaluator
        .evaluate(&dataset, &[ctx], "集成测试", "hybrid", 42)
        .await;

    assert_eq!(report.case_count, 1);
    assert_eq!(report.elapsed_ms, 42);
    assert_eq!(report.summary.len(), 2);

    let recall = report.metric("recall_at_k").unwrap();
    assert_eq!(recall.average, 1.0);
    assert_eq!(recall.pass_rate, 1.0);
    assert_eq!(recall.evaluated, 1);

    // 表格要能渲染。
    let table = report.to_table();
    assert!(table.contains("recall_at_k"));
    assert!(table.contains("token_f1"));
}

#[tokio::test]
async fn comparison_report_shows_improvement() {
    let index = chunk_index();

    // baseline：只召回了一个相关块。
    let baseline_case = case();
    let baseline_ctx = ctx(
        &baseline_case,
        "BM25 由 IDF 组成",
        &["kb.md#0", "kb.md#9"],
        &index,
    );

    // optimized：两个相关块都召回了。
    let optimized_case = case();
    let optimized_ctx = ctx(
        &optimized_case,
        "BM25 由 IDF 组成",
        &["kb.md#0", "kb.md#1"],
        &index,
    );

    let evaluator = Evaluator::new(vec![Arc::new(RecallAtK::new(5))]);

    let baseline = evaluator
        .evaluate(
            &EvalDataset::new("d", vec![baseline_case.clone()]),
            &[baseline_ctx],
            "基线",
            "vector",
            0,
        )
        .await;
    let optimized = evaluator
        .evaluate(
            &EvalDataset::new("d", vec![optimized_case.clone()]),
            &[optimized_ctx],
            "优化",
            "hybrid",
            0,
        )
        .await;

    let comparison = ComparisonReport::compare(&baseline, &optimized);

    assert_eq!(comparison.improved().len(), 1);
    let delta = comparison.improved()[0];
    assert_eq!(delta.name, "recall_at_k");
    assert_eq!(delta.baseline, 0.5);
    assert_eq!(delta.optimized, 1.0);
    assert_eq!(delta.delta, 0.5);
    assert_eq!(delta.verdict, Verdict::Improved);

    // 对比表要能渲染。
    let table = comparison.to_table();
    assert!(table.contains("recall_at_k"));
    assert!(table.contains("↑ 变好"));
}

#[test]
fn variants_parse_and_map_to_search_modes() {
    assert_eq!(EvalVariant::parse("hybrid"), Some(EvalVariant::Hybrid));
    assert_eq!(
        EvalVariant::parse("no_retrieval"),
        Some(EvalVariant::NoRetrieval)
    );
    assert_eq!(
        EvalVariant::Hybrid.search_mode(),
        Some(rag_08::hybrid::SearchMode::Hybrid)
    );
    assert_eq!(EvalVariant::NoRetrieval.search_mode(), None);
}
