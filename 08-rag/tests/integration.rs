//! 集成测试 —— 覆盖「混合检索 → 重排 → 生成」的完整链路。
//!
//! ## 测试分层
//!
//! | 层 | 是否调 LLM | 覆盖内容 |
//! |---|---|---|
//! | 单元测试（各模块内） | 否 | 分词、BM25 打分、RRF 融合、重排排序、错误映射 |
//! | 集成测试（本文件） | 否 | 建索引 → 双路检索 → 融合 → 重排 → 命中正确性 |
//! | 端到端测试（需 API key） | 是 | 完整问答 |
//!
//! 集成测试**不调 LLM** —— LLM 调用慢、贵、还不确定。
//!
//! 「检索是否召回了正确的块」是确定性可以断言的，
//! 「模型是否基于资料回答」不是。
//!
//! 所以这里用 [`ScriptedLlm`] 喂预设答案：
//! 生成逻辑是**确定的**（给定同样的输入，解析出同样的引用），
//! 只有「LLM 会输出什么」是不确定的 —— 把它固定住，其余都能断言。
//!
//! 重排用 [`KeywordReranker`]（确定性替身）——
//! 真实模型（`bge-reranker-base`）要下载 1.1GB，而且输出不保证稳定。
//!
//! ⚠️ 集成测试需要嵌入模型。如果模型没下载过，会自动下载（约 470MB）。

use std::path::PathBuf;
use std::sync::Arc;

use rag_08::bm25::Bm25Index;
use rag_08::config::{AppConfig, RerankConfig, RetrievalConfig};
use rag_08::hybrid::{HybridRetriever, SearchMode};
use rag_08::llm::{Llm, ScriptedLlm};
use rag_08::pipeline::HybridPipeline;
use rag_08::reranker::{KeywordReranker, Reranker};

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

/// 建一个混合检索器（向量索引 + BM25 索引）。
async fn make_retriever(data_dir: &std::path::Path, per_path_k: usize) -> HybridRetriever {
    let embedder = rag_08::index::make_embedder(rag_05::embed::Model::MultilingualE5Small)
        .expect("创建嵌入器失败（模型未下载？）");

    let (store, _stats) = rag_08::index::build_store(
        &embedder,
        data_dir,
        &rag_08::index::IndexOptions::default(),
    )
    .await
    .expect("建索引失败");

    let bm25 = rag_08::index::build_bm25(data_dir, &rag_04::splitter::SplitterConfig::default())
        .expect("建 BM25 索引失败");

    HybridRetriever::new(store, bm25, Box::new(embedder), per_path_k.max(1))
}

/// 造一个流水线（LLM 用脚本替身，重排用关键词替身）。
fn make_pipeline(
    retriever: HybridRetriever,
    llm: Arc<dyn Llm>,
    reranker: Option<Arc<dyn Reranker>>,
) -> HybridPipeline {
    HybridPipeline::new(
        Arc::new(retriever),
        reranker,
        llm,
        RetrievalConfig::default(),
        RerankConfig::default(),
    )
}

#[tokio::test]
async fn vector_and_bm25_indexes_have_matching_sizes() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    let vector_size = retriever.index_size().await;
    let bm25_size = retriever.bm25_size();

    assert!(vector_size >= 8, "向量索引里应该有至少 8 个块，实际 {vector_size}");
    assert_eq!(
        vector_size, bm25_size,
        "两套索引的块数必须一致 —— 否则融合时对不上"
    );
}

#[tokio::test]
async fn bm25_finds_exact_keyword_match() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")), None);

    // 「ECONNREFUSED」是一个罕见字符串 —— 向量检索很难命中，
    // 但 BM25 应该能精确命中「错误码速查」那一节。
    let outcome = pipeline
        .retrieve_with("ECONNREFUSED", SearchMode::Bm25)
        .await
        .expect("检索失败");

    assert!(!outcome.hits.is_empty(), "BM25 应该召回到结果");

    let all_text: String = outcome
        .hits
        .iter()
        .map(|h| format!("{} {}", h.text, h.heading_path.join(" ")))
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        all_text.contains("ECONNREFUSED"),
        "BM25 应该精确命中错误码，实际召回: {}",
        outcome
            .hits
            .iter()
            .map(|h| h.heading_path.join(" > "))
            .collect::<Vec<_>>()
            .join(" | ")
    );
}

#[tokio::test]
async fn hybrid_mode_finds_both_semantic_and_exact_matches() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")), None);

    // 这个问题用「光合作用的光反应发生在什么位置」这种语义问法。
    let outcome = pipeline
        .retrieve_with("光合作用的光反应发生在什么位置？", SearchMode::Hybrid)
        .await
        .expect("检索失败");

    let all_text: String = outcome
        .hits
        .iter()
        .map(|h| format!("{} {}", h.text, h.heading_path.join(" ")))
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        all_text.contains("类囊体膜"),
        "混合检索应该召回到光合作用相关内容，实际: {}",
        outcome
            .hits
            .iter()
            .map(|h| h.heading_path.join(" > "))
            .collect::<Vec<_>>()
            .join(" | ")
    );
}

#[tokio::test]
async fn hybrid_mode_deduplicates_across_paths() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")), None);

    let outcome = pipeline
        .retrieve_with("检索增强生成是什么", SearchMode::Hybrid)
        .await
        .unwrap();

    let mut ids: Vec<&str> = outcome.hits.iter().map(|h| h.id.as_str()).collect();
    let before = ids.len();
    ids.sort_unstable();
    ids.dedup();

    assert_eq!(before, ids.len(), "融合结果里出现了重复的块 ID");
}

#[tokio::test]
async fn hybrid_hits_record_consensus_votes() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")), None);

    let outcome = pipeline
        .retrieve_with("BM25 打分公式", SearchMode::Hybrid)
        .await
        .unwrap();

    assert!(!outcome.hits.is_empty());

    // 至少有一个块被两路同时命中 —— 否则「混合」根本没发生。
    let consensus = outcome.hits.iter().filter(|h| h.votes == 2).count();
    assert!(
        consensus > 0,
        "应该至少有一个块被两路同时命中，实际全部: {:?}",
        outcome
            .hits
            .iter()
            .map(|h| (h.id.clone(), h.votes))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn rerank_reorders_candidates() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    // 关键词重排器：包含「光合作用」的文档排前面。
    let reranker = Arc::new(KeywordReranker::new("光合作用"));
    let pipeline = make_pipeline(
        retriever,
        Arc::new(ScriptedLlm::always("x")),
        Some(reranker),
    );

    let outcome = pipeline
        .retrieve_with("光合作用", SearchMode::Hybrid)
        .await
        .unwrap();

    assert!(outcome.reranked, "重排应该执行了");

    // 重排后的第一名应该包含关键词。
    let first = &outcome.hits[0];
    assert!(
        first.text.contains("光合作用"),
        "重排后第一名应该包含关键词，实际: {}",
        first.text
    );
}

/// 创建一个含大量章节的临时目录（用于验证候选池大小）。
///
/// ⚠️ 默认的 `kb.md` 只有 11 个块 —— 分不清「每路召回 8 条」
/// 和「每路召回 20 条」的差别。所以这个测试要自己造一个更大的语料。
fn large_kb_dir(sections: usize) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("创建临时目录失败");

    let mut content = String::new();
    for i in 0..sections {
        content.push_str(&format!("## 章节 {i}\n\n"));
        // 每节写够长，确保每节至少产生一个块。
        content.push_str(&format!(
            "这是第 {i} 个章节的正文内容，用于验证候选池的召回数量。\
             混合检索需要足够多的候选块，交叉编码器才能从中挑出最相关的。\n\n"
        ));
    }

    std::fs::write(dir.path().join("kb.md"), content).expect("写入测试文档失败");
    dir
}

#[tokio::test]
async fn rerank_receives_enough_candidates() {
    // ⚠️ 回归测试：候选池必须真的被填满。
    //
    // 曾经的 bug：检索器固定每路召回 8 条，融合后候选不到 20 条，
    // 但配置里 `candidate_k = 20` —— 重排器永远拿不到 20 个候选，
    // 「送 20 条进重排」这个承诺是假的。
    //
    // 用一个 40 节的语料，才能真正区分「每路召回 8」和「每路召回 20」。
    let dir = large_kb_dir(40);
    let retriever = make_retriever(dir.path(), 8).await;

    let reranker = Arc::new(KeywordReranker::new("候选"));
    let pipeline = make_pipeline(
        retriever,
        Arc::new(ScriptedLlm::always("x")),
        Some(reranker),
    );

    let outcome = pipeline
        .retrieve_with("候选", SearchMode::Hybrid)
        .await
        .unwrap();

    // 融合后的候选数必须 ≥ candidate_k（20），
    // 否则说明每路召回没按 candidate_k 放大。
    let candidate_k = RerankConfig::default().candidate_k;
    assert!(
        outcome.fused_hits.len() >= candidate_k,
        "融合后的候选数应该至少等于 candidate_k ({candidate_k})，实际 {}",
        outcome.fused_hits.len()
    );
}

#[tokio::test]
async fn candidate_pool_never_exceeds_corpus_size() {
    // 语料比 candidate_k 还小的时候，候选数就等于语料大小 ——
    // 不应该报错，也不应该凭空造出重复块。
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")), None);

    let outcome = pipeline
        .retrieve_with("BM25", SearchMode::Hybrid)
        .await
        .unwrap();

    let corpus_size = 11; // kb.md 的块数
    assert!(outcome.fused_hits.len() <= corpus_size);

    let mut ids: Vec<&str> = outcome.hits.iter().map(|h| h.id.as_str()).collect();
    let before = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(before, ids.len(), "候选池里不应该有重复块");
}

#[tokio::test]
async fn rerank_can_be_disabled_for_baseline_comparison() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    // 不传 reranker —— 就是「对照组」。
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")), None);

    let outcome = pipeline
        .retrieve_with("光合作用", SearchMode::Hybrid)
        .await
        .unwrap();

    assert!(!outcome.reranked, "没有重排器时不应该标记为已重排");
    assert!(outcome.degrade_reason.is_none());
}

#[tokio::test]
async fn empty_question_is_rejected() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 4).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")), None);

    let err = pipeline
        .retrieve_with("   ", SearchMode::Hybrid)
        .await
        .unwrap_err();
    assert_eq!(err.status_code(), 400);
}

#[tokio::test]
async fn ask_produces_citations_mapped_to_hits() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    // 脚本化 LLM 返回带引用的答案。
    let llm = Arc::new(ScriptedLlm::always("光反应发生在类囊体膜上 [1]。"));
    let pipeline = make_pipeline(retriever, llm, None);

    let outcome = pipeline
        .ask_with("光合作用的光反应发生在什么位置？", SearchMode::Hybrid)
        .await
        .expect("问答失败");

    assert!(!outcome.answer.is_empty());
    assert_eq!(outcome.citations.len(), 1);
    assert_eq!(outcome.citations[0].index, 1);
    assert_eq!(outcome.citations[0].chunk_id, outcome.retrieval.hits[0].id);
}

#[tokio::test]
async fn out_of_range_citations_are_dropped() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;

    // 模型编造了一个不存在的引用编号 [99]。
    let llm = Arc::new(ScriptedLlm::always("答案里有个假引用 [99]。"));
    let pipeline = make_pipeline(retriever, llm, None);

    let outcome = pipeline
        .ask_with("BM25 是什么", SearchMode::Hybrid)
        .await
        .unwrap();

    assert!(
        outcome.citations.is_empty(),
        "越界的引用编号必须被丢掉"
    );
}

#[tokio::test]
async fn all_modes_return_results() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")), None);

    // 三种模式都应该能召回东西 —— 绝不能返回空列表
    // （那会让后续流程崩掉）。
    for mode in SearchMode::all() {
        let outcome = pipeline
            .retrieve_with("检索增强生成", mode)
            .await
            .unwrap_or_else(|e| panic!("模式 {} 检索失败: {e}", mode.name()));

        assert!(
            !outcome.hits.is_empty(),
            "模式 {} 不应该返回空结果",
            mode.name()
        );
    }
}

#[tokio::test]
async fn bm25_only_mode_marks_vector_scores_as_none() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")), None);

    let outcome = pipeline
        .retrieve_with("倒排索引", SearchMode::Bm25)
        .await
        .unwrap();

    assert!(outcome.hits.iter().all(|h| h.vector_score.is_none()));
    assert!(outcome.hits.iter().all(|h| h.bm25_score.is_some()));
}

#[tokio::test]
async fn vector_only_mode_marks_bm25_scores_as_none() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 8).await;
    let pipeline = make_pipeline(retriever, Arc::new(ScriptedLlm::always("x")), None);

    let outcome = pipeline
        .retrieve_with("倒排索引", SearchMode::Vector)
        .await
        .unwrap();

    assert!(outcome.hits.iter().all(|h| h.bm25_score.is_none()));
    assert!(outcome.hits.iter().all(|h| h.vector_score.is_some()));
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

#[test]
fn bm25_index_is_built_from_the_knowledge_base() {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    let index = Bm25Index::new();
    assert!(index.is_empty());

    let built = rag_08::index::build_bm25(&data_dir, &rag_04::splitter::SplitterConfig::default())
        .unwrap();
    assert!(!built.is_empty());
    assert!(built.vocab_size() > 50, "词表应该有一定规模");
}
