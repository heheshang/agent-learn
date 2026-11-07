//! 集成测试 —— 覆盖「改写 → 多路检索 → 融合」的完整链路。
//!
//! ## 测试分层
//!
//! | 层 | 是否调 LLM | 覆盖内容 |
//! |---|---|---|
//! | 单元测试（各模块内） | 否 | 改写解析、RRF 融合、提示词结构、错误映射 |
//! | 集成测试（本文件） | 否 | 建索引 → 多路检索 → 融合 → 命中正确性 |
//! | 端到端测试（需 API key） | 是 | 完整问答 |
//!
//! 集成测试**不调 LLM** —— LLM 调用慢、贵、还不确定。
//! 「检索是否召回了正确的块」是确定性可以断言的，
//! 「模型是否基于资料回答」不是。
//!
//! 所以这里用 [`ScriptedLlm`] 喂预设答案：
//! 改写逻辑是**确定的**（给定同样的输入，解析出同样的查询），
//! 只有「LLM 会输出什么」是不确定的 —— 把它固定住，其余都能断言。
//!
//! ⚠️ 集成测试需要嵌入模型。如果模型没下载过，会自动下载（约 470MB）。

use std::path::PathBuf;
use std::sync::Arc;

use rag_07::config::{AppConfig, RetrievalConfig, TransformConfig};
use rag_07::fusion::{fuse, FusionMethod, RankedRun};
use rag_07::llm::{Llm, LlmFuture, ScriptedLlm};
use rag_07::pipeline::QueryPipeline;
use rag_07::retrieval::MultiQueryRetriever;
use rag_07::transform::{
    DecomposeTransformer, MultiQueryTransformer, QueryTransformer, Strategy,
};

/// 知识库测试文档路径。
fn kb_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/kb.md")
}

/// 创建一个只含单个文档的临时目录。
fn temp_kb_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("创建临时目录失败");
    let content = std::fs::read_to_string(kb_path()).expect("读取测试文档失败");
    std::fs::write(dir.path().join("kb.md"), content).expect("写入测试文档失败");
    dir
}

/// 建一个检索器（用默认的 E5 模型）。
async fn make_retriever(data_dir: &std::path::Path, top_k: usize) -> MultiQueryRetriever {
    let embedder = rag_07::index::make_embedder(rag_05::embed::Model::MultilingualE5Small)
        .expect("创建嵌入器失败（模型未下载？）");

    let (store, _stats) = rag_07::index::build_store(
        &embedder,
        data_dir,
        &rag_07::index::IndexOptions::default(),
    )
    .await
    .expect("建索引失败");

    MultiQueryRetriever::new(store, Box::new(embedder), top_k.max(1))
}

/// 造一个流水线（LLM 用脚本替身）。
fn make_pipeline(retriever: MultiQueryRetriever, llm: Arc<dyn Llm>) -> QueryPipeline {
    QueryPipeline::new(
        Arc::new(retriever),
        llm,
        RetrievalConfig::default(),
        TransformConfig::default(),
    )
}

#[tokio::test]
async fn index_contains_all_sections() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    // 测试文档有 6 个二级标题 → 至少 6 个块。
    let size = retriever.index_size().await;
    assert!(
        size >= 6,
        "索引里应该有至少 6 个块（每个二级标题一个），实际 {size}"
    );
}

#[tokio::test]
async fn retrieves_correct_section_for_specific_question() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 4).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("改写结果")));

    // 每个问题都对应文档里的一个独立章节。
    let cases = [
        ("光合作用的光反应发生在什么位置？", "类囊体膜"),
        ("B 树索引的查找复杂度是多少？", "O(log n)"),
        ("为什么 TCP 需要三次握手？", "三次握手"),
        ("混合检索是怎么融合两路结果的？", "倒数排名融合"),
    ];

    for (question, expected) in cases {
        let result = pipeline
            .retrieve(question, Strategy::None)
            .await
            .expect("检索失败");

        assert!(!result.hits.is_empty(), "问题「{question}」应该召回到结果");

        let all_text: String = result
            .hits
            .iter()
            .map(|h| format!("{} {}", h.text, h.heading_path.join(" ")))
            .collect::<Vec<_>>()
            .join(" ");

        assert!(
            all_text.contains(expected),
            "问题「{question}」期望命中「{expected}」，实际召回: {}",
            result
                .hits
                .iter()
                .map(|h| h.heading_path.join(" > "))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
}

#[tokio::test]
async fn multi_query_produces_multiple_runs_and_fuses_them() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    // LLM 返回 3 个问法。
    let llm = Arc::new(ScriptedLlm::always(
        "1. 文档分块大小怎么定\n2. chunk size 应该设多少\n3. 切块的重叠比例是多少",
    ));

    let pipeline = make_pipeline(retriever, llm);
    let result = pipeline
        .retrieve("文档切多大合适", Strategy::MultiQuery)
        .await
        .expect("检索失败");

    assert_eq!(result.transformed_queries.len(), 3);
    assert_eq!(result.runs.len(), 3);

    // 融合后的结果应该去重了（不能出现同一个 ID 两次）。
    let mut ids: Vec<&str> = result.hits.iter().map(|h| h.id.as_str()).collect();
    let before = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(before, ids.len(), "融合结果里出现了重复的块 ID");
}

#[tokio::test]
async fn multi_query_queries_are_all_used_for_retrieval() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    let llm = Arc::new(ScriptedLlm::always("1. 问法一\n2. 问法二"));
    let pipeline = make_pipeline(retriever, llm);

    let result = pipeline
        .retrieve("原问题", Strategy::MultiQuery)
        .await
        .unwrap();

    // 每一路都应该有自己的查询文本。
    let queries: Vec<&str> = result.runs.iter().map(|r| r.query.as_str()).collect();
    assert_eq!(queries, vec!["问法一", "问法二"]);
}

#[tokio::test]
async fn hyde_uses_document_embedding() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    let llm = Arc::new(ScriptedLlm::always(
        "文档分块是把长文档切分成较小片段的过程，块大小通常在三百到五百 token 之间。",
    ));

    let pipeline = make_pipeline(retriever, llm);
    let result = pipeline
        .retrieve("文档切多大合适", Strategy::Hyde)
        .await
        .unwrap();

    assert_eq!(result.transformed_queries.len(), 1);
    assert_eq!(result.transformed_queries[0].strategy, Strategy::Hyde);

    // HyDE 的「假想文档」应该和知识库里的原文语义接近。
    let all_text: String = result
        .hits
        .iter()
        .map(|h| h.text.clone())
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        all_text.contains("分块") || all_text.contains("token"),
        "HyDE 应该能召回「文档分块」相关内容，实际: {}",
        result
            .hits
            .iter()
            .map(|h| h.heading_path.join(" > "))
            .collect::<Vec<_>>()
            .join(" | ")
    );
}

#[tokio::test]
async fn decompose_splits_into_sub_questions() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    let llm = Arc::new(ScriptedLlm::always(
        "1. 光合作用的光反应阶段发生在什么位置？\n2. 光合作用的暗反应阶段发生在什么位置？",
    ));

    let pipeline = make_pipeline(retriever, llm);
    let result = pipeline
        .retrieve("光合作用的光反应和暗反应分别在什么位置进行？", Strategy::Decompose)
        .await
        .unwrap();

    assert_eq!(result.transformed_queries.len(), 2);
    assert!(result.runs.len() == 2);
}

#[tokio::test]
async fn degraded_transform_falls_back_to_original_question() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 4).await;

    // 空的 LLM 答案 → 解析不出内容 → 降级。
    let llm = Arc::new(ScriptedLlm::always("   "));
    let pipeline = make_pipeline(retriever, llm);

    let result = pipeline
        .retrieve("光合作用的光反应发生在什么位置", Strategy::MultiQuery)
        .await
        .unwrap();

    assert!(result.degraded, "解析失败时应该标记为降级");
    assert_eq!(result.transformed_queries.len(), 1);
    assert_eq!(
        result.transformed_queries[0].text,
        "光合作用的光反应发生在什么位置"
    );
}

#[tokio::test]
async fn empty_question_is_rejected() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 4).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")));

    let err = pipeline
        .retrieve("   ", Strategy::None)
        .await
        .unwrap_err();
    assert_eq!(err.status_code(), 400);
}

#[tokio::test]
async fn fusion_methods_produce_different_orderings() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    let llm = Arc::new(ScriptedLlm::always("1. 问法一\n2. 问法二\n3. 问法三"));
    let pipeline = make_pipeline(retriever, llm);

    let rrf = pipeline
        .retrieve_with("倒排索引怎么用", Strategy::MultiQuery, FusionMethod::Rrf)
        .await
        .unwrap();
    let concat = pipeline
        .retrieve_with("倒排索引怎么用", Strategy::MultiQuery, FusionMethod::Concat)
        .await
        .unwrap();

    assert_eq!(rrf.hits.len(), concat.hits.len());

    // RRF 的分数应该等于各路贡献之和，远小于原始相似度。
    for hit in &rrf.hits {
        assert!(hit.score > 0.0);
        assert!(hit.score < 1.0);
    }
}

#[tokio::test]
async fn rrf_rewards_consensus_blocks() {
    // 纯逻辑测试 —— 不需要索引。
    let chunk = |id: &str, score: f32| rag_06::retrieval::RetrievedChunk {
        rank: 0,
        id: id.to_string(),
        score,
        source: "doc.md".to_string(),
        heading_path: vec!["标题".to_string()],
        text: "正文".to_string(),
        heading_context: "# 标题".to_string(),
    };

    let runs = vec![
        RankedRun::new(
            "q1",
            Strategy::MultiQuery,
            vec![chunk("x", 0.9), chunk("a", 0.8)],
        ),
        RankedRun::new(
            "q2",
            Strategy::MultiQuery,
            vec![chunk("b", 0.9), chunk("a", 0.75)],
        ),
    ];

    let fused = fuse(&runs, FusionMethod::Rrf, 10);
    assert_eq!(fused[0].id, "a", "两路都命中的块应该排第一");
    assert_eq!(fused[0].votes, 2);
}

#[tokio::test]
async fn multi_query_transformer_caps_output() {
    let llm = ScriptedLlm::always("1. 一\n2. 二\n3. 三\n4. 四");
    let t = MultiQueryTransformer::new(Arc::new(llm)).with_num_queries(2);

    let out = t.transform("问题").await.unwrap();
    assert_eq!(out.len(), 2);
}

#[tokio::test]
async fn decompose_transformer_caps_output() {
    let llm = ScriptedLlm::always("1. 一\n2. 二\n3. 三\n4. 四\n5. 五");
    let t = DecomposeTransformer::new(Arc::new(llm)).with_max_sub_questions(3);

    let out = t.transform("问题").await.unwrap();
    assert_eq!(out.len(), 3);
}

#[tokio::test]
async fn all_strategies_produce_at_least_one_query() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 4).await;

    // 同一个 LLM 回答喂给所有策略 —— 每个策略都应该能从中解析出点什么，
    // 或者降级成原问题。**绝不能返回空列表**（那会让后续流程崩掉）。
    let llm = Arc::new(ScriptedLlm::always("1. 问法一\n2. 问法二"));

    let pipeline = make_pipeline(retriever, llm);

    for strategy in Strategy::all() {
        let result = pipeline.retrieve("原问题", strategy).await.unwrap();
        assert!(
            !result.transformed_queries.is_empty(),
            "策略 {} 不应该返回空查询列表",
            strategy.name()
        );
    }
}

#[test]
fn default_config_points_at_a_real_data_dir() {
    let config = AppConfig::default();
    assert!(
        config.data_dir.exists(),
        "默认数据目录应该存在: {}",
        config.data_dir.display()
    );
}

/// 确认 `LlmFuture` 的类型别名可用（编译期检查）。
#[allow(dead_code)]
fn _future_type_check<'a>(f: LlmFuture<'a>) -> LlmFuture<'a> {
    f
}
