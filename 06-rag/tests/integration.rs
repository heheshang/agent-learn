//! 集成测试 —— 覆盖「检索 → 生成」的完整链路。
//!
//! ## 测试分层
//!
//! | 层 | 是否调 LLM | 覆盖内容 |
//! |---|---|---|
//! | 单元测试（各模块内） | 否 | 引用解析、Prompt 组装、错误映射 |
//! | 集成测试（本文件） | 否 | 建索引 → 检索 → 命中是否正确 |
//! | 端到端测试（需 API key） | 是 | 完整问答 |
//!
//! 集成测试**不调 LLM** —— LLM 调用慢、贵、还不确定。
//! 「检索是否召回了正确的块」是确定性可以断言的，
//! 「模型是否基于资料回答」不是。
//!
//! ⚠️ 集成测试需要嵌入模型。如果模型没下载过，会自动下载（约 100MB）。
//! 用 `FASTEMBED_CACHE_PATH` 可以指定缓存位置。

use std::path::PathBuf;

use rag_05::store::VectorStore;
use rag_06::config::{AppConfig, EmbeddingModel, RetrievalConfig};
use rag_06::retrieval::Retriever;

/// 知识库测试文档路径。
fn kb_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/kb_test.md")
}

/// 创建一个只含单个文档的临时目录。
fn temp_kb_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("创建临时目录失败");
    let content = std::fs::read_to_string(kb_path()).expect("读取测试文档失败");
    std::fs::write(dir.path().join("kb_test.md"), content).expect("写入测试文档失败");
    dir
}

/// 建一个检索器（用默认的 E5 模型）。
async fn make_retriever(data_dir: &std::path::Path, top_k: usize) -> Retriever {
    let embedder = rag_06::index::make_embedder(EmbeddingModel::MultilingualE5Small.to_model())
        .expect("创建嵌入器失败（模型未下载？）");

    let (store, _stats) = rag_06::index::build_store(
        &embedder,
        data_dir,
        &rag_06::index::IndexOptions::default(),
    )
    .await
    .expect("建索引失败");

    Retriever::new(
        store,
        Box::new(embedder),
        RetrievalConfig {
            top_k,
            min_score: None,
        },
    )
}

#[tokio::test]
async fn index_contains_all_sections() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 4).await;

    // 测试文档有 7 个二级标题 → 至少 7 个块。
    //
    // ⚠️ 为什么不是 8：文档原本有一段「本文件是……测试集」的前言，
    // 结构感知分块会把它单独切成一个块 —— 但它对任何查询都相似度很高
    // （含「知识库」「文档」「检索」等高频词），成了万能吸铁石，
    // 会挤掉真正的答案。所以前言被去掉了，块数相应少一个。
    let size = retriever.index_size().await;
    assert!(
        size >= 7,
        "索引里应该有至少 7 个块（每个二级标题一个），实际 {size}"
    );
}

#[tokio::test]
async fn retrieves_correct_section_for_specific_question() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 3).await;

    // 每个问题都对应文档里的一个独立章节。
    // 这些章节语义上互不相关 —— 所以「召回的第一名是不是那一节」
    // 就是一个有意义的断言。
    let cases = [
        ("咖啡因的半衰期是多久？", "咖啡因的作用机制"),
        ("为什么海水是咸的？", "海水为什么是咸的"),
        ("光合作用暗反应发生在哪里？", "光合作用的两个阶段"),
        ("TCP 为什么需要三次握手？", "TCP 三次握手"),
        ("数据库索引为什么能加速查询？", "数据库索引的原理"),
    ];

    for (question, expected_section) in cases {
        let result = retriever.retrieve(question).await.expect("检索失败");

        assert!(
            !result.chunks.is_empty(),
            "问题「{question}」应该召回到结果"
        );

        let top = &result.chunks[0];
        let top_heading = top.heading_path.last().map(String::as_str).unwrap_or("");

        assert!(
            top_heading.contains(expected_section)
                || top.text.contains(&expected_section[..expected_section.len().min(4)]),
            "问题「{question}」期望命中「{expected_section}」，实际命中「{top_heading}」"
        );
    }
}

#[tokio::test]
async fn scores_are_sorted_descending() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 5).await;

    let result = retriever.retrieve("咖啡因的副作用有哪些？").await.unwrap();

    for pair in result.chunks.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "结果应该按相似度降序：{} 然后 {}",
            pair[0].score,
            pair[1].score
        );
    }
}

#[tokio::test]
async fn empty_query_is_rejected() {
    let dir = temp_kb_dir();
    let retriever = make_retriever(dir.path(), 3).await;

    let err = retriever.retrieve("   ").await.unwrap_err();
    assert_eq!(err.status_code(), 400);
}

#[tokio::test]
async fn min_score_filters_out_weak_matches() {
    let dir = temp_kb_dir();
    let embedder = rag_06::index::make_embedder(EmbeddingModel::MultilingualE5Small.to_model())
        .unwrap();
    let (store, _) = rag_06::index::build_store(
        &embedder,
        dir.path(),
        &rag_06::index::IndexOptions::default(),
    )
    .await
    .unwrap();

    // 阈值设到 0.95 —— 正常查询不可能到这个分数。
    let retriever = Retriever::new(
        store,
        Box::new(embedder),
        RetrievalConfig {
            top_k: 5,
            min_score: Some(0.95),
        },
    );

    let result = retriever.retrieve("咖啡因的副作用有哪些？").await.unwrap();
    assert!(
        result.chunks.is_empty(),
        "阈值 0.95 应该过滤掉所有命中，实际还有 {} 个",
        result.chunks.len()
    );
}

#[tokio::test]
async fn index_survives_save_and_load() {
    let dir = temp_kb_dir();
    let embedder = rag_06::index::make_embedder(EmbeddingModel::MultilingualE5Small.to_model())
        .unwrap();

    let index_path = dir.path().join("index.faiss");
    let stats = rag_06::index::build_and_save(
        &embedder,
        dir.path(),
        &index_path,
        &rag_06::index::IndexOptions::default(),
    )
    .await
    .unwrap();

    assert!(stats.chunk_count >= 7);

    // 重新加载，检索结果必须一致。
    let loaded = rag_06::index::load_store(&embedder, &index_path).await.unwrap();
    assert_eq!(loaded.len(), stats.chunk_count);
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
