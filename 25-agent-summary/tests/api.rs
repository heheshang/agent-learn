//! API 层集成测试 —— 用 `tower::ServiceExt::oneshot` 直接调路由（**不开端口**）。
//!
//! ## ⚠️ 为什么不开端口
//!
//! 开端口会引入：端口冲突、启动时序、进程残留。
//! `oneshot` 把请求送进路由栈，测的是**完全相同的**处理逻辑，
//! 但零外部依赖、零时序问题。

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use agent_25::aggregate::Strategy;
use agent_25::api::{AppState, router};
use agent_25::pipeline::{Pipeline, demo_activities};

/// 建一个测试用的 app。
fn app() -> axum::Router {
    let activities = demo_activities(200);
    let pipeline = Pipeline::new(activities, Strategy::Optimized);
    router(AppState::new(pipeline))
}

/// 发一个请求，返回 (状态码, body 字符串)。
async fn call(app: &axum::Router, req: Request<Body>) -> (StatusCode, String) {
    let response = app.clone().oneshot(req).await.expect("请求应该成功");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("应该能读到 body");
    (status, String::from_utf8_lossy(&bytes).to_string())
}

/// GET 请求。
fn get(path: &str) -> Request<Body> {
    Request::builder()
        .uri(path)
        .method("GET")
        .body(Body::empty())
        .unwrap()
}

/// POST JSON 请求。
fn post(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .uri(path)
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn health_returns_ok() {
    let (status, body) = call(&app(), get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("ok"));
}

#[tokio::test]
async fn modules_lists_all_capabilities() {
    let (status, body) = call(&app(), get("/modules")).await;
    assert_eq!(status, StatusCode::OK);
    for m in [
        "ingest",
        "retrieve",
        "agent",
        "tooling",
        "cache",
        "concurrency",
        "aggregate",
    ] {
        assert!(body.contains(m), "模块清单里应该包含 {m}");
    }
    assert!(body.contains("naive"));
    assert!(body.contains("optimized"));
}

#[tokio::test]
async fn summary_requires_both_window_bounds() {
    let (status, _) = call(&app(), get("/summary?start=100")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn summary_rejects_reversed_window() {
    let (status, _) = call(&app(), get("/summary?start=200&end=100")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn summary_returns_markdown() {
    let (status, body) = call(&app(), get("/summary?start=1700000000&end=1700604800")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("周度总结"));
    assert!(body.contains("digest"));
}

#[tokio::test]
async fn search_rejects_empty_query() {
    let (status, _) = call(&app(), post("/search", serde_json::json!({"query": "  "}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn search_rejects_oversized_limit() {
    let (status, _) = call(
        &app(),
        post(
            "/search",
            serde_json::json!({"query": "cache", "limit": 1000}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn search_returns_hits() {
    let (status, body) = call(
        &app(),
        post("/search", serde_json::json!({"query": "cache", "limit": 5})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("hits"));
}

#[tokio::test]
async fn strategy_switch_accepts_known_names() {
    for name in ["naive", "pre-agg", "optimized"] {
        let (status, body) = call(
            &app(),
            post("/strategy", serde_json::json!({"strategy": name})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{name} 应该被接受");
        assert!(body.contains(name));
    }
}

#[tokio::test]
async fn strategy_switch_rejects_unknown_name() {
    let (status, _) = call(
        &app(),
        post("/strategy", serde_json::json!({"strategy": "nope"})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn bench_returns_three_tiers() {
    let (status, body) = call(
        &app(),
        post(
            "/bench",
            serde_json::json!({"start": 1_700_000_000i64, "end": 1_700_000_000i64 + 7 * 86_400}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("naive"));
    assert!(body.contains("pre-agg"));
    assert!(body.contains("optimized"));
}

#[tokio::test]
async fn unknown_route_is_404() {
    let (status, _) = call(&app(), get("/nope")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
