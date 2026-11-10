//! 集成测试 —— HTTP API 层 + 关键不变量。
//!
//! ## ⚠️ HTTP API 层用 `tower::ServiceExt::oneshot` 直接调路由
//!
//! **不开端口** —— 避免端口冲突、避免测试之间互相干扰。
//! （与 14/19/22 项目同一做法。）
//!
//! ## 零网络、零模型
//!
//! 集成测试用 [`agent_23::testkit`] 里的替身，
//! 所以 `cargo test` 不需要下载模型、不需要 Ollama。

use std::sync::Arc;

use agent_23::api::AppStateInner;
use agent_23::api::routes::router;
use agent_23::batching::{BatchAccumulator, BatchConfig, BatchStrategy};
use agent_23::config::AppConfig;
use agent_23::embedder::BatchEmbedder;
use agent_23::reranker::BatchReranker;
use agent_23::testkit::{ScriptedEmbedder, ScriptedReranker, corpus, documents};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::Value;
use tower::ServiceExt;

/// 构造一个用替身的后端状态。
fn test_state() -> Arc<AppStateInner> {
    let config = AppConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        sample_size: 8,
        rounds: 1,
        ..Default::default()
    };

    let embedder = Arc::new(BatchEmbedder::from_embedder(
        Arc::new(ScriptedEmbedder::new()),
        config.batch.clone(),
    ));

    let reranker = Arc::new(BatchReranker::from_reranker(
        Arc::new(ScriptedReranker::new()),
        config.batch.clone(),
    ));

    Arc::new(AppStateInner::new(config, embedder, reranker))
}

/// 发一个请求并解析 JSON 响应。
async fn call(app: axum::Router, req: Request<Body>) -> (StatusCode, Value) {
    let response = app.oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

/// POST 一个 JSON。
async fn post(app: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    call(app, req).await
}

// ── 服务信息 ─────────────────────────────────────────────────────────

#[tokio::test]
async fn root_lists_all_three_strategies() {
    let app = router(test_state());

    let (status, body) = call(
        app,
        Request::builder().uri("/").body(Body::empty()).unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    let strategies = body["strategies"].as_array().unwrap();
    assert_eq!(strategies.len(), 3);

    let names: Vec<&str> = strategies
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"sequential"));
    assert!(names.contains(&"fixed"));
    assert!(names.contains(&"bucketed"));
}

#[tokio::test]
async fn health_is_ok() {
    let app = router(test_state());

    let (status, body) = call(
        app,
        Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
}

// ── 批量嵌入 ─────────────────────────────────────────────────────────

#[tokio::test]
async fn embed_returns_vectors_in_input_order() {
    let app = router(test_state());

    let texts = corpus(6);
    let (status, body) = post(
        app,
        "/api/embed",
        serde_json::json!({"texts": texts, "strategy": "fixed"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["count"], 6);

    // ⚠️ 结果顺序必须与输入一致 —— 批处理的基本契约。
    let vectors = body["vectors"].as_array().unwrap();
    for (i, text) in texts.iter().enumerate() {
        let v0 = vectors[i][0].as_f64().unwrap();
        assert_eq!(v0 as usize, text.chars().count(), "第 {i} 条向量错位了");
    }
}

#[tokio::test]
async fn embed_rejects_unknown_strategy() {
    let app = router(test_state());

    let (status, body) = post(
        app,
        "/api/embed",
        serde_json::json!({"texts": ["a"], "strategy": "nope"}),
    )
    .await;

    // ⚠️ 调用方参数错 → 400，不是 500。
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["code"], "E_BAD_REQUEST");
}

#[tokio::test]
async fn embed_rejects_empty_texts() {
    let app = router(test_state());

    let (status, _) = post(app, "/api/embed", serde_json::json!({"texts": []})).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn embed_stats_report_throughput_and_latency() {
    let app = router(test_state());

    let (_, body) = post(
        app,
        "/api/embed",
        serde_json::json!({"texts": corpus(8), "strategy": "fixed"}),
    )
    .await;

    // ⚠️ 度量必须**同时**包含吞吐与延迟两个维度。
    let stats = &body["stats"];
    assert!(stats["throughput_per_sec"].is_number());
    assert!(stats["p50"].is_number());
    assert!(stats["p95"].is_number());
    assert_eq!(stats["total_items"], 8);
}

// ── 批量重排 ─────────────────────────────────────────────────────────

#[tokio::test]
async fn rerank_returns_hits_sorted_by_score() {
    let app = router(test_state());

    let docs: Vec<Value> = documents(5)
        .iter()
        .map(|d| {
            serde_json::json!({
                "id": d.id,
                "source": d.source,
                "heading_path": d.heading_path,
                "text": d.text,
            })
        })
        .collect();

    let (status, body) = post(
        app,
        "/api/rerank",
        serde_json::json!({
            "query": "第 1 条候选",
            "documents": docs,
            "top_n": 3,
            "strategy": "fixed",
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["candidate_count"], 5);

    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 3);

    // ⚠️ 分数必须单调不增。
    let scores: Vec<f64> = hits.iter().map(|h| h["score"].as_f64().unwrap()).collect();
    for pair in scores.windows(2) {
        assert!(pair[0] >= pair[1], "分数未降序: {scores:?}");
    }
}

#[tokio::test]
async fn rerank_rejects_empty_documents() {
    let app = router(test_state());

    let (status, _) = post(
        app,
        "/api/rerank",
        serde_json::json!({"query": "x", "documents": []}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ── 对照实验 ─────────────────────────────────────────────────────────

#[tokio::test]
async fn bench_embed_returns_all_three_strategies() {
    let app = router(test_state());

    let (status, body) = post(
        app,
        "/api/bench/embed",
        serde_json::json!({"sample_size": 8, "rounds": 1}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["sample_size"], 8);

    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);

    // ⚠️ 基线组必须是「逐条调用」。
    assert_eq!(results[0]["strategy"], "逐条调用");
}

#[tokio::test]
async fn bench_rerank_returns_all_three_strategies() {
    let app = router(test_state());

    let (status, body) = post(
        app,
        "/api/bench/rerank",
        serde_json::json!({"sample_size": 6, "rounds": 1, "top_n": 2}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["results"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn bench_sweep_finds_the_best_batch_size() {
    let app = router(test_state());

    let (status, body) = post(
        app,
        "/api/bench/sweep",
        serde_json::json!({"sample_size": 8, "rounds": 1, "batch_sizes": [1, 2, 4]}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["points"].as_array().unwrap().len(), 3);

    // ⚠️ 最优批大小必须是扫描过的档位之一。
    let best = body["best_batch_size"].as_u64().unwrap();
    assert!([1, 2, 4].contains(&best));
}

// ── 关键不变量 ───────────────────────────────────────────────────────

#[tokio::test]
async fn bucketed_embed_produces_identical_results_to_fixed() {
    // ⚠️ **最重要**的不变量：
    //    批处理只该影响**耗时**，不该影响**结果内容**。
    let embedder = BatchEmbedder::from_embedder(
        Arc::new(ScriptedEmbedder::new()),
        BatchConfig {
            batch_size: 3,
            bucket_batch_size: 3,
            ..Default::default()
        },
    );

    let texts = corpus(12);

    let (fixed, _) = embedder
        .embed_batch(&texts, BatchStrategy::Fixed)
        .await
        .unwrap();
    let (bucketed, _) = embedder
        .embed_batch(&texts, BatchStrategy::Bucketed)
        .await
        .unwrap();

    assert_eq!(fixed, bucketed, "分桶改变了结果内容 —— 这是 bug");
}

#[tokio::test]
async fn bucketed_rerank_produces_identical_results_to_sequential() {
    // ⚠️ 同样的不变量，作用在重排上。
    let reranker = BatchReranker::from_reranker(
        Arc::new(ScriptedReranker::new()),
        BatchConfig {
            batch_size: 4,
            bucket_batch_size: 4,
            ..Default::default()
        },
    );

    let docs = documents(12);

    let (seq, _) = reranker
        .rerank_batch("查询", &docs, 5, BatchStrategy::Sequential)
        .await
        .unwrap();
    let (bucketed, _) = reranker
        .rerank_batch("查询", &docs, 5, BatchStrategy::Bucketed)
        .await
        .unwrap();

    agent_23::testkit::assert_same_scores(&seq, &bucketed).unwrap();
}

#[tokio::test]
async fn accumulator_rejects_when_the_queue_is_full() {
    // ⚠️ 背压闸门必须**返回错误**，不能静默丢弃。
    let mut acc = BatchAccumulator::new(BatchConfig {
        batch_size: 1000,
        max_wait_ms: 10_000,
        max_queue_len: 3,
        ..Default::default()
    });

    for _ in 0..3 {
        assert!(acc.push("x".into()).unwrap().is_none());
    }

    let err = acc.push("y".into()).unwrap_err();
    assert_eq!(err.status_code(), 400);
    assert_eq!(acc.len(), 3);
}

#[tokio::test]
async fn padding_comparison_shows_bucketing_reduces_waste() {
    // ⚠️ 分桶收益的**直接证据**。
    let lengths: Vec<usize> = (0..16).map(|i| if i % 2 == 0 { 100 } else { 10 }).collect();

    let comparison = agent_23::padding::compare_padding(&lengths, 4);

    assert!(
        comparison.bucketed.padding_factor < comparison.fixed.padding_factor,
        "分桶后 padding 系数应更低"
    );
    assert!(comparison.improvement > 0.0);
}
