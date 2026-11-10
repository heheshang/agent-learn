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

// ─────────────────────────────────────────────────────────────────────
// P0 新增：Agent 与工具端点
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn tools_endpoint_lists_the_four_tools() {
    let (status, body) = call(&app(), get("/tools")).await;
    assert_eq!(status, StatusCode::OK);
    // ★ 四个工具 —— 工具多了会让模型凑数调用。
    for name in [
        "module_stats",
        "search_activities",
        "query_stats",
        "calculate",
    ] {
        assert!(body.contains(name), "工具清单里应该包含 {name}");
    }
}

#[tokio::test]
async fn direct_tool_call_returns_real_data() {
    let (status, body) = call(
        &app(),
        post(
            "/tools/call",
            serde_json::json!({"tool": "module_stats", "args": {"scope": "cache"}}),
        ),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("\"ok\":true"));
    // ★ 工具**真的执行了** —— 返回了结构化数据，不是编的。
    assert!(body.contains("\"data\""));
    assert!(body.contains("cache"));
}

#[tokio::test]
async fn direct_tool_call_with_bad_args_is_400() {
    let (status, body) = call(
        &app(),
        post(
            "/tools/call",
            serde_json::json!({"tool": "module_stats", "args": {}}),
        ),
    )
    .await;

    // ★ 用户直接调工具时，错误**原样返回**（400），不降级成观察。
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("E_INVALID_ARGUMENTS"));
}

#[tokio::test]
async fn direct_tool_call_with_unknown_tool_is_400() {
    let (status, body) = call(
        &app(),
        post(
            "/tools/call",
            serde_json::json!({"tool": "nope", "args": {}}),
        ),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    // ★ 错误里必须列出**可用工具** —— 模型据此纠正。
    assert!(body.contains("module_stats"));
}

#[tokio::test]
async fn sql_guard_blocks_writes_over_http() {
    let (status, body) = call(
        &app(),
        post(
            "/tools/call",
            serde_json::json!({
                "tool": "query_stats",
                "args": {"sql": "DELETE FROM activity"}
            }),
        ),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    // ★ 护栏真的拦住了（工具返回失败观察，不是 HTTP 错误）。
    assert!(body.contains("\"ok\":false"));
    assert!(body.contains("E_RULE_VIOLATION"));
}

#[tokio::test]
async fn agent_endpoint_without_llm_is_503() {
    let (status, body) = call(
        &app(),
        post(
            "/agent",
            serde_json::json!({"question": "cache 模块做了什么？"}),
        ),
    )
    .await;

    // ★ 没配 LLM 是 503（服务端暂时不可用），不是 400 也不是 500。
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body.contains("DEEPSEEK_API_KEY"));
}

#[tokio::test]
async fn unknown_protocol_is_400() {
    // ⚠️ 协议名拼错**不能**默默跑默认协议 ——
    //    那会让「协议对照实验」作废（两次跑的是同一条链路）。
    let (status, body) = call(
        &app(),
        post(
            "/agent",
            serde_json::json!({"question": "查一下", "protocol": "reactt"}),
        ),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("protocol"));
}

#[tokio::test]
async fn react_protocol_without_llm_is_503() {
    // ⚠️ LLM 可用性检查要**下沉到各协议分支** ——
    //    提前用 `state.llm` 把门，会让「只配了 react LLM」的部署
    //    永远 503。
    let (status, _body) = call(
        &app(),
        post(
            "/agent",
            serde_json::json!({"question": "查一下", "protocol": "react"}),
        ),
    )
    .await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn structured_protocol_without_llm_is_503() {
    let (status, _body) = call(
        &app(),
        post(
            "/agent",
            serde_json::json!({"question": "查一下", "protocol": "structured"}),
        ),
    )
    .await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn agent_endpoint_rejects_empty_question() {
    let (status, _) = call(
        &app(),
        post("/agent", serde_json::json!({"question": "   "})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn summary_uses_the_provided_query() {
    // ⚠️ 回归测试：第一版把 query **静默丢弃**了（写死 "cache"）。
    //    现在传了就必须用传的。
    let (status, body) = call(
        &app(),
        post(
            "/summary",
            serde_json::json!({
                "start": 1_700_000_000i64,
                "end": 1_700_000_000i64 + 7 * 86_400,
                "query": "vllm"
            }),
        ),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("\"query\":\"vllm\""));
}
