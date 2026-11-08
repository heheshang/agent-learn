//! 集成测试 —— 端到端验证流水线的每一层。
//!
//! ## 分两类
//!
//! | 类型 | 是否需要后端 | 覆盖什么 |
//! |---|---|---|
//! | 离线测试 | ❌ | 分块 → BM25 索引 → 融合 → 压缩 → 引用抽取的一致性 |
//! | 在线测试 | ✅ Milvus | 真实检索、服务端混合检索、索引幂等性 |
//!
//! ## ⚠️ 离线测试为什么不需要 Milvus
//!
//! 检索层被抽成了 [`Retriever`] trait —— 测试注入一个**确定性的替身**。
//! 这和 08-rag 用 `KeywordReranker` 替代 `LocalCrossEncoder`、
//! 07-query 用 `ScriptedLlm` 替代 `GenaiLlm` 是同一套思路：
//! **下游逻辑的正确性不应该依赖上游服务的可用性**。
//!
//! ## 在线测试的跳过策略
//!
//! 连不上 Milvus 时**跳过**而不是失败 —— 因为 CI 环境不一定有 Milvus。
//! 但设了 `RAG12_REQUIRE_BACKEND=1` 时**直接失败** ——
//! 这样本地开发能主动验证「后端真的通了」。
//!
//! 和 10-rag-milvus 的策略完全一致。

use std::sync::Arc;

use rag_12::config::{AppConfig, SearchMode};
use rag_12::retriever::{Hit, RetrievalPath, Retriever, build_bm25_index, fuse_rrf};

// ════════════════════════════════════════════════════════════════════
//  离线测试 —— 不需要任何后端
// ════════════════════════════════════════════════════════════════════

/// 项目自己的知识库目录。
fn kb_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}

/// 默认分块配置。
fn splitter_config() -> rag_04::splitter::SplitterConfig {
    rag_04::splitter::SplitterConfig::default()
}

#[test]
fn knowledge_base_is_loadable() {
    let documents = rag_04::loader::load_directory(&kb_dir()).unwrap();

    assert!(!documents.is_empty(), "知识库目录应该有文档");
    assert!(
        documents
            .iter()
            .any(|d| d.source.to_string_lossy().contains("kb.md")),
        "知识库应该包含 kb.md"
    );
}

#[test]
fn chunk_ids_are_stable_across_runs() {
    // ⚠️ 客户端 BM25 和 Milvus 索引必须用**完全相同**的分块配置，
    // 否则两路召回的块不是同一批，RRF 融合会静默失效。
    let first = rag_12::eval::build_chunk_index(&kb_dir(), &splitter_config()).unwrap();
    let second = rag_12::eval::build_chunk_index(&kb_dir(), &splitter_config()).unwrap();

    assert_eq!(first.chunk_count(), second.chunk_count());
    assert!(first.chunk_count() > 0);
}

#[test]
fn eval_sections_resolve_to_chunks() {
    // 评估集用章节标题标注相关块 —— 必须能翻译成块 ID。
    let index = rag_12::eval::build_chunk_index(&kb_dir(), &splitter_config()).unwrap();

    for section in [
        "BM25 打分算法",
        "混合检索",
        "重排与交叉编码器",
        "上下文压缩",
        "Milvus 向量数据库",
        "倒排索引",
    ] {
        assert!(
            !index.ids_by_section(section).is_empty(),
            "章节「{section}」应该能解析出块 ID"
        );
    }
}

#[test]
fn bm25_index_covers_the_same_chunks_as_the_chunk_index() {
    let chunk_index = rag_12::eval::build_chunk_index(&kb_dir(), &splitter_config()).unwrap();
    let bm25 = build_bm25_index(&kb_dir(), &splitter_config()).unwrap();

    assert_eq!(
        bm25.len(),
        chunk_index.chunk_count(),
        "BM25 索引和块索引必须覆盖同一批块"
    );
}

#[test]
fn bm25_finds_exact_keyword_hits() {
    let bm25 = build_bm25_index(&kb_dir(), &splitter_config()).unwrap();

    // 「ECONNREFUSED」是罕见字符串 —— 关键词路的主场。
    let hits = bm25.search("ECONNREFUSED", 5);
    assert!(!hits.is_empty(), "应该能召回错误码章节");

    let found = hits.iter().any(|h| h.text.contains("ECONNREFUSED"));
    assert!(found, "命中内容里应该包含 ECONNREFUSED");
}

#[test]
fn rrf_fusion_prefers_consensus() {
    // 构造：稠密路排第一的是 x，两路都命中的是 a。
    let dense = vec![
        rag_05::store::SearchHit {
            id: "x".to_string(),
            score: 0.95,
            source: "kb.md".to_string(),
            heading_path: vec!["混合检索".to_string()],
            text: "x 的正文".to_string(),
        },
        rag_05::store::SearchHit {
            id: "a".to_string(),
            score: 0.70,
            source: "kb.md".to_string(),
            heading_path: vec!["BM25 打分算法".to_string()],
            text: "a 的正文".to_string(),
        },
    ];

    let bm25 = vec![rag_08::bm25::Bm25Hit {
        id: "a".to_string(),
        score: 8.0,
        source: "kb.md".to_string(),
        heading_path: vec!["BM25 打分算法".to_string()],
        text: "a 的正文".to_string(),
        rank: 0,
    }];

    let fused = fuse_rrf(dense, bm25, 10);

    assert_eq!(fused[0].id, "a", "两路都命中的应该排前面");
    assert_eq!(fused[0].votes, 2);
}

#[test]
fn compression_keeps_citations_consistent() {
    let hits: Vec<Hit> = (0..6)
        .map(|i| Hit {
            rank: i,
            id: format!("kb.md#{i}"),
            score: 0.9 - i as f32 * 0.05,
            dense_score: Some(0.9),
            bm25_score: None,
            votes: 1,
            source: "kb.md".to_string(),
            heading_path: vec![format!("章节 {i}")],
            text: format!("这是第 {i} 段测试内容，用来占满 token 预算。"),
            path: RetrievalPath::ServerHybrid,
        })
        .collect();

    let compressor = rag_12::compress::ContextCompressor::new(rag_12::config::CompressConfig {
        budget_tokens: 60,
        min_tokens: 20,
        enabled: true,
    })
    .unwrap();

    let result = compressor.compress(&hits);

    // 引用编号必须连续 —— 否则 Prompt 里的 [n] 和实际片段对不上。
    for (index, passage) in result.passages.iter().enumerate() {
        assert_eq!(passage.rank, index);
        assert!(
            result.text.contains(&format!("[{}]", index + 1)),
            "上下文里必须出现引用标记 [{}]",
            index + 1
        );
    }

    // 编号不能超出实际片段数。
    assert!(
        !result
            .text
            .contains(&format!("[{}]", result.passages.len() + 1)),
        "不能引用不存在的片段"
    );
}

#[test]
fn compression_respects_budget() {
    let hits: Vec<Hit> = (0..10)
        .map(|i| Hit {
            rank: i,
            id: format!("kb.md#{i}"),
            score: 0.9,
            dense_score: Some(0.9),
            bm25_score: None,
            votes: 1,
            source: "kb.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: "很长的内容。".repeat(50),
            path: RetrievalPath::ServerHybrid,
        })
        .collect();

    let compressor = rag_12::compress::ContextCompressor::new(rag_12::config::CompressConfig {
        budget_tokens: 100,
        min_tokens: 20,
        enabled: true,
    })
    .unwrap();

    let result = compressor.compress(&hits);

    assert!(
        result.tokens_after <= result.tokens_before,
        "压缩后不应该变长"
    );
    assert!(result.dropped > 0, "超预算的块应该被砍掉");
}

// ════════════════════════════════════════════════════════════════════
//  离线流水线测试 —— 用确定性的检索替身
// ════════════════════════════════════════════════════════════════════

/// 确定性的检索替身 —— 永远返回固定的两条命中。
#[derive(Debug)]
struct FakeRetriever {
    hits: Vec<Hit>,
    /// 检索是否应该失败（用来测降级链）。
    fail: bool,
}

impl FakeRetriever {
    fn ok() -> Self {
        Self {
            hits: vec![
                Hit {
                    rank: 0,
                    id: "kb.md#0".to_string(),
                    score: 0.9,
                    dense_score: Some(0.9),
                    bm25_score: Some(5.0),
                    votes: 2,
                    source: "kb.md".to_string(),
                    heading_path: vec!["BM25 打分算法".to_string()],
                    text: "BM25 的核心是逆文档频率、词频饱和和长度归一化。".to_string(),
                    path: RetrievalPath::ServerHybrid,
                },
                Hit {
                    rank: 1,
                    id: "kb.md#1".to_string(),
                    score: 0.8,
                    dense_score: Some(0.8),
                    bm25_score: None,
                    votes: 1,
                    source: "kb.md".to_string(),
                    heading_path: vec!["混合检索".to_string()],
                    text: "混合检索把稀疏检索和稠密检索结合起来。".to_string(),
                    path: RetrievalPath::ServerHybrid,
                },
            ],
            fail: false,
        }
    }

    fn failing() -> Self {
        Self {
            hits: Vec::new(),
            fail: true,
        }
    }
}

impl Retriever for FakeRetriever {
    fn search<'a>(
        &'a self,
        _query: &'a str,
        k: usize,
        _per_path_k: usize,
        _mode: SearchMode,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = rag_12::RagResult<Vec<Hit>>> + Send + 'a>>
    {
        Box::pin(async move {
            if self.fail {
                return Err(rag_12::RagError::Retrieval("测试用的失败".to_string()));
            }
            Ok(self.hits.iter().take(k).cloned().collect())
        })
    }

    fn name(&self) -> &str {
        "fake"
    }
}

/// 构造一个离线可用的流水线。
fn offline_pipeline() -> Arc<rag_12::pipeline::SummaryPipeline> {
    use rag_07::llm::ScriptedLlm;

    let mut config = AppConfig::default();
    config.rerank.enabled = false;
    config.rewrite.enabled = false;
    config.compress.budget_tokens = 2000;

    let llm = Arc::new(ScriptedLlm::always("这是测试答案，引用 [1] 和 [2]。"))
        as Arc<dyn rag_07::llm::Llm>;

    Arc::new(rag_12::pipeline::SummaryPipeline::new(
        Arc::new(FakeRetriever::ok()),
        None,
        Arc::new(rag_12::compress::ContextCompressor::new(config.compress.clone()).unwrap()),
        None,
        llm,
        config,
    ))
}

#[tokio::test]
async fn offline_pipeline_returns_answer_and_citations() {
    let pipeline = offline_pipeline();

    let outcome = pipeline
        .ask_with("BM25 是什么", SearchMode::Hybrid)
        .await
        .unwrap();

    assert!(!outcome.answer.is_empty());
    assert_eq!(outcome.citations.len(), 2);
    assert_eq!(outcome.citations[0].index, 1);
    assert_eq!(outcome.citations[0].chunk_id, "kb.md#0");
}

#[tokio::test]
async fn offline_pipeline_records_retrieval_metadata() {
    let pipeline = offline_pipeline();

    let outcome = pipeline
        .retrieve_with("BM25 是什么", SearchMode::Hybrid)
        .await
        .unwrap();

    assert_eq!(outcome.mode, SearchMode::Hybrid);
    assert_eq!(outcome.path, RetrievalPath::ServerHybrid);
    assert_eq!(outcome.hit_count(), 2);
    assert_eq!(outcome.fused_hits.len(), 2);
    assert!(!outcome.reranked, "没装重排器就不该说重排过了");
}

#[tokio::test]
async fn empty_question_is_rejected() {
    let pipeline = offline_pipeline();

    let error = pipeline
        .ask_with("   ", SearchMode::Hybrid)
        .await
        .unwrap_err();
    assert_eq!(error.status_code(), 400);
}

#[tokio::test]
async fn retrieval_failure_propagates_as_error() {
    use rag_07::llm::ScriptedLlm;

    let config = AppConfig::default();
    let llm = Arc::new(ScriptedLlm::always("答案")) as Arc<dyn rag_07::llm::Llm>;

    let pipeline = rag_12::pipeline::SummaryPipeline::new(
        Arc::new(FakeRetriever::failing()),
        None,
        Arc::new(rag_12::compress::ContextCompressor::new(config.compress.clone()).unwrap()),
        None,
        llm,
        config,
    );

    let error = pipeline
        .retrieve_with("问题", SearchMode::Hybrid)
        .await
        .unwrap_err();
    assert_eq!(error.status_code(), 500);
}

// ════════════════════════════════════════════════════════════════════
//  API 测试 —— 用 axum 的 `oneshot`，不需要真的起服务器
// ════════════════════════════════════════════════════════════════════

/// 构造一个离线可用的应用状态。
fn test_state() -> rag_12::api::AppState {
    let config = AppConfig::default();

    Arc::new(rag_12::api::AppStateInner::new(
        config,
        offline_pipeline(),
        None,
    ))
}

#[tokio::test]
async fn index_endpoint_is_reachable() {
    use tower::ServiceExt;

    let app = rag_12::api::routes::router(test_state());
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn search_rejects_empty_question() {
    use tower::ServiceExt;

    let app = rag_12::api::routes::router(test_state());
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/search")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(r#"{"question":"  "}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn search_rejects_unknown_mode() {
    use tower::ServiceExt;

    let app = rag_12::api::routes::router(test_state());
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/search")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"question":"BM25 是什么","mode":"不存在的模式"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn malformed_json_returns_bad_request() {
    use tower::ServiceExt;

    let app = rag_12::api::routes::router(test_state());
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/search")
                .header("content-type", "application/json")
                .body(axum::body::Body::from("{ 不是 JSON"))
                .unwrap(),
        )
        .await
        .unwrap();

    // ⚠️ 解析失败也必须是 400 —— 不能是 422 或 500。
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn search_returns_hits() {
    use tower::ServiceExt;

    let app = rag_12::api::routes::router(test_state());
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/search")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"question":"BM25 是什么","mode":"hybrid"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let payload: rag_12::api::models::SearchResponse = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(payload.mode, SearchMode::Hybrid);
    assert_eq!(payload.hits.len(), 2);
    assert_eq!(payload.path, "server_hybrid");
}

#[tokio::test]
async fn ask_returns_answer_with_citations() {
    use tower::ServiceExt;

    let app = rag_12::api::routes::router(test_state());
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/ask")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"question":"BM25 是什么","mode":"hybrid"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let payload: rag_12::api::models::AskResponse = serde_json::from_slice(&bytes).unwrap();

    assert!(!payload.answer.is_empty());
    assert_eq!(payload.citations.len(), 2);
    assert_eq!(payload.passages.len(), 2);
}

#[tokio::test]
async fn compare_returns_all_variants_and_overlap() {
    use tower::ServiceExt;

    let app = rag_12::api::routes::router(test_state());
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/compare")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(r#"{"question":"BM25 是什么"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let payload: rag_12::api::models::CompareResponse = serde_json::from_slice(&bytes).unwrap();

    // 完整阶梯 = 5 档。
    assert_eq!(payload.variants.len(), 5);
    assert!(payload.variants.iter().all(|v| v.ok));
    // 各档命中相同 → 重合度 1.0。
    assert!((payload.overlap - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn eval_endpoint_requires_no_question() {
    use tower::ServiceExt;

    let app = rag_12::api::routes::router(test_state());
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/eval")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(r#"{"limit":1}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    // ⚠️ 评估跑的是**整个数据集**，不需要问题。
    // 如果 `EvalRequest` 误用了 `CompareRequest`（带必填 question），
    // 这里会变成 400 —— 所以这个测试守住「字段选对了」。
    assert_ne!(response.status(), 400, "评估端点不应该要求 question 字段");
}

// ════════════════════════════════════════════════════════════════════
//  在线测试 —— 需要 Milvus
// ════════════════════════════════════════════════════════════════════

/// 后端是否可用。
///
/// 不可用时默认**跳过**；`RAG12_REQUIRE_BACKEND=1` 时**失败**。
///
/// ⚠️ 必须是 `async fn` —— 在同步函数里 `block_on` 会撞上
/// 「Cannot start a runtime from within a runtime」。
async fn backend_available() -> bool {
    let config = AppConfig::default();

    rag_10::milvus_store::MilvusStore::connect(
        &config.milvus,
        rag_12::config::model_dimension(config.embedding_model),
        rag_12::config::model_name(config.embedding_model),
    )
    .await
    .is_ok()
}

/// 跳过后端测试（或直接失败）。
macro_rules! require_backend {
    () => {
        if !backend_available().await {
            if std::env::var("RAG12_REQUIRE_BACKEND").as_deref() == Ok("1") {
                panic!("RAG12_REQUIRE_BACKEND=1，但 Milvus 不可达");
            }
            eprintln!("跳过：Milvus 不可达");
            return;
        }
    };
}

#[tokio::test]
async fn milvus_index_is_idempotent() {
    require_backend!();

    let config = AppConfig::default();

    // 灌两次 —— count 不应该翻倍。
    let first = rag_12::bootstrap::ensure_index(&config).await.unwrap();
    let second = rag_12::bootstrap::ensure_index(&config).await.unwrap();

    assert_eq!(
        first.server_count, second.server_count,
        "重复灌库不应该让 count(*) 翻倍 —— 主键必须是稳定派生的"
    );
    assert!(second.server_count > 0, "服务端应该有数据");
}

#[tokio::test]
async fn milvus_dense_and_hybrid_both_return_results() {
    require_backend!();

    let config = AppConfig::default();
    rag_12::bootstrap::ensure_index(&config).await.unwrap();

    let (pipeline, _) = rag_12::bootstrap::bootstrap(&config).await.unwrap();

    let dense = pipeline
        .retrieve_with("BM25 打分公式是什么", SearchMode::Dense)
        .await
        .unwrap();
    assert!(!dense.hits.is_empty(), "稠密检索应该召回内容");

    let hybrid = pipeline
        .retrieve_with("BM25 打分公式是什么", SearchMode::Hybrid)
        .await
        .unwrap();
    assert!(!hybrid.hits.is_empty(), "混合检索应该召回内容");

    // 服务端混合检索必须走 server_hybrid 路径。
    assert_eq!(hybrid.path, RetrievalPath::ServerHybrid);
}

#[tokio::test]
async fn keyword_mode_uses_client_bm25() {
    require_backend!();

    let config = AppConfig::default();
    rag_12::bootstrap::ensure_index(&config).await.unwrap();

    let (pipeline, _) = rag_12::bootstrap::bootstrap(&config).await.unwrap();

    let keyword = pipeline
        .retrieve_with("ECONNREFUSED", SearchMode::Keyword)
        .await
        .unwrap();

    assert!(!keyword.hits.is_empty());
    assert_eq!(keyword.path, RetrievalPath::KeywordOnly);
}
