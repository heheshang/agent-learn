//! 集成测试 —— 覆盖 HTTP API 层与 mock vLLM 的**真实 HTTP 往返**。
//!
//! ## ⚠️ 为什么必须走真实 HTTP
//!
//! 14 项目的教训：**用 Rust 结构体构造数据会绕过 JSON 反序列化**，
//! 于是「字段名写错」这类 bug 在单测里根本发现不了 ——
//! 必须走**真实 HTTP 往返**才能暴露。
//!
//! 本项目的 mock vLLM 就是为了这个：它真的起一个 axum 服务，
//! 客户端真的发 HTTP 请求，服务端真的按 OpenAI 协议回应。
//!
//! 用 `tower::ServiceExt::oneshot` 直接调路由（不开端口），
//! 速度快且确定。

use agent_24::api::handlers::AppState;
use agent_24::api::router;
use agent_24::client::{BenchRequest, ClientConfig, VllmClient};
use agent_24::config::AppConfig;
use agent_24::scheduler::{BatchingMode, EngineConfig};
use agent_24::server::{MockState, routes as mock_routes};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

/// 构造一个请求。
fn request(method: &str, path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .header("Content-Type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// 读响应体。
async fn read_body(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// 测试用的应用状态。
fn state() -> AppState {
    AppState {
        config: AppConfig::default(),
    }
}

#[tokio::test]
async fn root_returns_service_info() {
    let app = router(state());
    let response = app
        .oneshot(request("GET", "/", serde_json::json!({})))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;
    assert_eq!(body["topic"], "批处理优化（Batching）");
    assert!(body["reference"].as_str().unwrap().contains("vllm.ai"));
    // ⚠️ 三个替代方案必须在（这是本项目的对照目标）。
    assert_eq!(body["alternatives"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn health_is_ok() {
    let app = router(state());
    let response = app
        .oneshot(request("GET", "/health", serde_json::json!({})))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn modes_endpoint_explains_both_batching_modes() {
    let app = router(state());
    let response = app
        .oneshot(request("GET", "/api/modes", serde_json::json!({})))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;
    let modes = body["modes"].as_array().unwrap();
    assert_eq!(modes.len(), 2);
    assert_eq!(modes[0]["name"], "static");
    assert_eq!(modes[1]["name"], "continuous");
    // ⚠️ 关键参数必须列出来（这是 vLLM 调参的入口）。
    assert!(
        modes[1]["key_params"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p.as_str().unwrap().contains("max-num-seqs"))
    );
}

#[tokio::test]
async fn simulate_returns_metrics_and_goodput() {
    let app = router(state());
    let response = app
        .oneshot(request(
            "POST",
            "/api/simulate",
            serde_json::json!({
                "num_requests": 30,
                "arrival_model": "burst",
                "length_distribution": "uniform",
                "min_input_tokens": 32,
                "max_input_tokens": 256,
                "min_output_tokens": 16,
                "max_output_tokens": 64,
                "max_num_seqs": 8
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;

    assert_eq!(body["metrics"]["total_requests"], 30);
    assert!(
        body["metrics"]["request_throughput_per_sec"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(body["metrics"]["ttft_p95_ms"].as_f64().unwrap() > 0.0);
    assert_eq!(body["outcomes"].as_array().unwrap().len(), 30);
    // ⚠️ goodput 必须存在 —— 只报吞吐是不够的。
    assert!(body["goodput"]["goodput_per_sec"].as_f64().unwrap() >= 0.0);
}

#[tokio::test]
async fn simulate_static_mode_differs_from_continuous() {
    let app = router(state());

    let base = serde_json::json!({
        "num_requests": 40,
        "arrival_model": "burst",
        "length_distribution": "uniform",
        "min_input_tokens": 32,
        "max_input_tokens": 512,
        "min_output_tokens": 16,
        "max_output_tokens": 128,
        "max_num_seqs": 8
    });

    let mut static_body = base.clone();
    static_body["mode"] = serde_json::json!("static");
    let mut cont_body = base.clone();
    cont_body["mode"] = serde_json::json!("continuous");

    let s = read_body(
        app.clone()
            .oneshot(request("POST", "/api/simulate", static_body))
            .await
            .unwrap(),
    )
    .await;
    let c = read_body(
        app.oneshot(request("POST", "/api/simulate", cont_body))
            .await
            .unwrap(),
    )
    .await;

    let s_wait = s["metrics"]["avg_queue_wait_ms"].as_f64().unwrap();
    let c_wait = c["metrics"]["avg_queue_wait_ms"].as_f64().unwrap();
    assert!(
        c_wait < s_wait,
        "连续批处理排队等待应更短：{c_wait} vs {s_wait}"
    );
}

#[tokio::test]
async fn bench_endpoint_runs_all_three_experiments() {
    let app = router(state());
    let response = app
        .oneshot(request(
            "POST",
            "/api/bench",
            serde_json::json!({
                "num_requests": 40,
                "arrival_model": "burst",
                "length_distribution": "uniform",
                "min_input_tokens": 32,
                "max_input_tokens": 256,
                "min_output_tokens": 16,
                "max_output_tokens": 64,
                "rounds": 1
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;

    let experiments = body["suite"]["experiments"].as_array().unwrap();
    assert_eq!(experiments.len(), 3, "默认应该跑全部三组实验");

    // 实验 A 有两档（静态 / 连续）。
    assert_eq!(experiments[0]["variants"].as_array().unwrap().len(), 2);
    // 实验 B 有五档（max_num_seqs 扫描）。
    assert_eq!(experiments[1]["variants"].as_array().unwrap().len(), 5);
    // 实验 C 有两档（chunked prefill 开/关）。
    assert_eq!(experiments[2]["variants"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn bench_endpoint_can_select_a_single_experiment() {
    let app = router(state());
    let response = app
        .oneshot(request(
            "POST",
            "/api/bench",
            serde_json::json!({
                "experiments": ["a"],
                "num_requests": 20,
                "arrival_model": "burst",
                "rounds": 1
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;
    let experiments = body["suite"]["experiments"].as_array().unwrap();
    assert_eq!(experiments.len(), 1);
}

#[tokio::test]
async fn env_endpoint_reports_effective_config() {
    let app = router(state());
    let response = app
        .oneshot(request("GET", "/api/env", serde_json::json!({})))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;
    assert!(body["summary"].as_str().unwrap().contains("max_num_seqs"));
    assert_eq!(body["rounds"], 3);
}

#[tokio::test]
async fn mock_vllm_serves_an_openai_compatible_response() {
    // ⚠️ ★ 真实 HTTP 往返 —— 这才能发现「字段名写错」这类 bug。
    let app = mock_routes(MockState::new(
        BatchingMode::Continuous,
        EngineConfig::default(),
    ));

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 32
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;

    // ⚠️ OpenAI 协议的字段名必须**一字不差** ——
    //    真 vLLM 的响应结构就是这个，客户端按这个解析。
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["completion_tokens"], 32);
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn mock_vllm_lists_models() {
    let app = mock_routes(MockState::new(
        BatchingMode::Continuous,
        EngineConfig::default(),
    ));

    let response = app
        .oneshot(request("GET", "/v1/models", serde_json::json!({})))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "mock-model");
}

#[tokio::test]
async fn mock_vllm_accepts_a_request_without_max_tokens() {
    // ⚠️ `max_tokens` 是可选的 —— 服务端不能因此报错。
    let app = mock_routes(MockState::new(
        BatchingMode::Continuous,
        EngineConfig::default(),
    ));

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}]
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_body(response).await;
    assert_eq!(body["usage"]["completion_tokens"], 64);
}

#[tokio::test]
async fn mock_vllm_handles_an_empty_message_list() {
    let app = mock_routes(MockState::new(
        BatchingMode::Continuous,
        EngineConfig::default(),
    ));

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock-model",
                "messages": []
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn client_reports_a_clear_error_when_the_server_is_unreachable() {
    // ⚠️ 连不上时必须给出**可执行**的错误信息，
    //    而不是让人对着 "connection refused" 发呆。
    let config = ClientConfig {
        // 选一个几乎肯定没人监听的端口。
        base_url: "http://127.0.0.1:1".to_string(),
        model: "mock".to_string(),
        api_key: "EMPTY".to_string(),
        timeout: std::time::Duration::from_secs(2),
        concurrency: 4,
    };

    let client = VllmClient::new(config).unwrap();
    let err = client.health().await.unwrap_err();

    let message = err.to_string();
    assert!(
        message.contains("连不上 vLLM"),
        "错误信息应说明连不上：{message}"
    );
    assert!(
        message.contains("127.0.0.1:1"),
        "错误信息应带上地址：{message}"
    );
}

#[tokio::test]
async fn client_marks_a_failed_request_as_not_ok() {
    let config = ClientConfig {
        base_url: "http://127.0.0.1:1".to_string(),
        model: "mock".to_string(),
        api_key: "EMPTY".to_string(),
        timeout: std::time::Duration::from_secs(2),
        concurrency: 4,
    };

    let client = VllmClient::new(config).unwrap();
    let response = client
        .chat(&BenchRequest {
            id: 7,
            prompt: "你好".to_string(),
            max_tokens: 16,
        })
        .await;

    assert!(!response.ok);
    assert!(response.error.is_some());
    assert_eq!(response.id, 7);
}
