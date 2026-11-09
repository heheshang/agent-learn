//! 集成测试 —— HTTP API 层 + Redis 真实后端。
//!
//! ## 测试策略
//!
//! | 层级 | 手段 | 是否需要外部依赖 |
//! |---|---|---|
//! | HTTP API 层 | `tower::ServiceExt::oneshot`（不开端口） | 否 |
//! | 缓存逻辑 | 内存后端 | 否 |
//! | Redis 后端 | 真实 Redis | **是**（连不上就跳过） |
//!
//! ## ⚠️ Redis 测试的跳过约定
//!
//! 和 10-rag-milvus 一样：连不上后端就**跳过**，
//! 设 `CACHE_REQUIRE_REDIS=1` 则**直接失败**。
//!
//! 理由：CI 环境不一定有 Redis，但**不应该**因为这个而让整个测试套件变红。
//! 而想要「必须验证 Redis」的场景（发布前）用环境变量强制。

use std::sync::Arc;

use agent_21::agent::CachedAgent;
use agent_21::api::AppStateInner;
use agent_21::api::routes::router;
use agent_21::cache::{
    CacheBackend, CacheControl, CacheService, CachedResponse, CachedUsage, LocalCache, RedisCache,
    now_ms,
};
use agent_21::config::AppConfig;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

// ─────────────────────────────────────────────────────────────────────
// 测试替身
// ─────────────────────────────────────────────────────────────────────

/// 一个可重复的 LLM —— 每次都返回同样的文本。
#[allow(dead_code)]
struct EchoLlm;

impl agent_21::llm::FunctionLlm for EchoLlm {
    fn chat_with_tools<'a>(
        &'a self,
        _system: &'a str,
        _messages: &'a [genai::chat::ChatMessage],
        _tools: &'a [agent_21::tools::ToolSpec],
        _tool_choice: agent_21::llm::ToolChoiceMode,
    ) -> agent_21::llm::FunctionLlmFuture<'a> {
        Box::pin(async move {
            Ok(agent_21::llm::ScriptedStep::Text("固定答案".to_string()).to_response())
        })
    }

    fn model_name(&self) -> &str {
        "echo"
    }
}

/// 一个计数 LLM —— 记录被调用的次数。
struct CountingLlm {
    calls: std::sync::atomic::AtomicUsize,
}

impl CountingLlm {
    fn new() -> Self {
        Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl agent_21::llm::FunctionLlm for CountingLlm {
    fn chat_with_tools<'a>(
        &'a self,
        _system: &'a str,
        _messages: &'a [genai::chat::ChatMessage],
        _tools: &'a [agent_21::tools::ToolSpec],
        _tool_choice: agent_21::llm::ToolChoiceMode,
    ) -> agent_21::llm::FunctionLlmFuture<'a> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Box::pin(async move {
            Ok(agent_21::llm::ScriptedStep::Text("固定答案".to_string()).to_response())
        })
    }

    fn model_name(&self) -> &str {
        "counting"
    }
}

/// 一个「慢」LLM —— 用来让缓存命中的耗时差异**可测量**。
struct SlowLlm {
    calls: std::sync::atomic::AtomicUsize,
}

impl SlowLlm {
    fn new() -> Self {
        Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl agent_21::llm::FunctionLlm for SlowLlm {
    fn chat_with_tools<'a>(
        &'a self,
        _system: &'a str,
        _messages: &'a [genai::chat::ChatMessage],
        _tools: &'a [agent_21::tools::ToolSpec],
        _tool_choice: agent_21::llm::ToolChoiceMode,
    ) -> agent_21::llm::FunctionLlmFuture<'a> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Box::pin(async move {
            // 模拟一次网络往返。
            tokio::time::sleep(std::time::Duration::from_millis(120)).await;
            Ok(agent_21::llm::ScriptedStep::Text("慢答案".to_string()).to_response())
        })
    }

    fn model_name(&self) -> &str {
        "slow"
    }
}

// ─────────────────────────────────────────────────────────────────────
// 辅助
// ─────────────────────────────────────────────────────────────────────

/// 构造一个使用内存后端的应用状态。
fn test_state(cache: Arc<CacheService>) -> Arc<AppStateInner> {
    let mut config = AppConfig::default();
    config.agent.max_steps = 3;

    let llm: Arc<dyn agent_21::llm::FunctionLlm> = Arc::new(CountingLlm::new());
    let registry = agent_21::tool::ToolRegistry::new(vec![]);
    let agent = Arc::new(CachedAgent::new(
        llm.clone(),
        registry,
        cache.clone(),
        config.clone(),
    ));

    Arc::new(AppStateInner::new(config, agent, cache, llm))
}

/// 发一个 JSON POST 请求。
async fn post(app: &axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);

    (status, json)
}

/// 发一个 GET 请求。
async fn get(app: &axum::Router, path: &str) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);

    (status, json)
}

/// Redis 是否可用。
async fn redis_backend(namespace: &str) -> Option<Arc<dyn CacheBackend>> {
    let url = std::env::var("CACHE_TEST_REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    match RedisCache::connect(&url).await {
        Ok(redis) => {
            // ⚠️ 测试之间可能互相干扰 —— 每次用**独立命名空间**。
            let service = CacheService::new(Arc::new(redis), 60).with_namespace(namespace);
            let backend = service.clone_backend();
            let _ = service.clear_namespace(namespace).await;
            Some(backend)
        }
        Err(e) => {
            let required = std::env::var("CACHE_REQUIRE_REDIS")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false);

            if required {
                panic!("Redis 必须可用（CACHE_REQUIRE_REDIS=1）但连接失败: {e}");
            }

            eprintln!("跳过 Redis 测试（连不上 {url}: {e}）");
            None
        }
    }
}

/// 唯一的命名空间（避免并行测试互相干扰）。
fn unique_namespace(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("test-{tag}-{nanos}")
}

// ─────────────────────────────────────────────────────────────────────
// HTTP API 层测试（零外部依赖）
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    let (status, body) = get(&app, "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn root_endpoint_exposes_cache_status() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    let (status, body) = get(&app, "/").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["cache"]["backend"], "local");
    assert_eq!(body["cache"]["enabled"], true);
    assert!(body["endpoints"].is_array());
}

#[tokio::test]
async fn run_endpoint_returns_answer_and_cache_stats() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache.clone()));

    let (status, body) = post(&app, "/api/run", json!({"question": "测试问题"})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["answer"], "固定答案");
    assert_eq!(body["cache_hits"], 0);
    assert_eq!(body["cache_misses"], 1);
    assert_eq!(body["cache_stats"]["misses"], 1);
}

#[tokio::test]
async fn run_endpoint_rejects_empty_question() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    let (status, body) = post(&app, "/api/run", json!({"question": "   "})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "Bad Request");
}

#[tokio::test]
async fn second_run_hits_the_cache() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache.clone()));

    let (_, first) = post(&app, "/api/run", json!({"question": "重复问题"})).await;
    let (_, second) = post(&app, "/api/run", json!({"question": "重复问题"})).await;

    assert_eq!(first["cache_hits"], 0);
    assert_eq!(second["cache_hits"], 1);
    // ⚠️ 答案必须**完全一致** —— 缓存不能改变语义。
    assert_eq!(first["answer"], second["answer"]);
}

#[tokio::test]
async fn no_cache_control_forces_a_real_call() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    post(&app, "/api/run", json!({"question": "问题"})).await;
    let (_, second) = post(
        &app,
        "/api/run",
        json!({"question": "问题", "cache": {"no_cache": true}}),
    )
    .await;

    assert_eq!(second["cache_hits"], 0);
    assert_eq!(second["cache_misses"], 1);
}

#[tokio::test]
async fn no_store_control_prevents_writing() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    post(
        &app,
        "/api/run",
        json!({"question": "问题", "cache": {"no_store": true}}),
    )
    .await;
    let (_, second) = post(&app, "/api/run", json!({"question": "问题"})).await;

    assert_eq!(second["cache_hits"], 0);
}

#[tokio::test]
async fn ttl_control_is_accepted() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    let (status, body) = post(
        &app,
        "/api/run",
        json!({"question": "问题", "cache": {"ttl": 30}}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["answer"], "固定答案");
}

#[tokio::test]
async fn smaxage_control_is_accepted() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    let (status, _) = post(
        &app,
        "/api/run",
        json!({"question": "问题", "cache": {"smaxage": 10}}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn cache_stats_endpoint_reports_hit_rate() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    post(&app, "/api/run", json!({"question": "问题"})).await;
    post(&app, "/api/run", json!({"question": "问题"})).await;

    let (status, body) = get(&app, "/api/cache/stats").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["backend"], "local");
    assert_eq!(body["stats"]["hits"], 1);
    assert_eq!(body["stats"]["misses"], 1);
    assert!((body["stats"]["hit_rate"].as_f64().unwrap() - 0.5).abs() < 1e-9);
}

#[tokio::test]
async fn cache_entries_endpoint_lists_entries() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    post(&app, "/api/run", json!({"question": "问题"})).await;

    let (status, body) = get(&app, "/api/cache/entries").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["count"], 1);
    assert!(
        body["entries"][0]["key"]
            .as_str()
            .unwrap()
            .contains("agent-cache")
    );
    // ⚠️ 预览应该只是摘要，不是全文。
    assert!(
        body["entries"][0]["preview"]
            .as_str()
            .unwrap()
            .contains("固定答案")
    );
}

#[tokio::test]
async fn clear_cache_endpoint_empties_the_namespace() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache.clone()));

    post(&app, "/api/run", json!({"question": "问题"})).await;

    let request = Request::builder()
        .method("DELETE")
        .uri("/api/cache")
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["removed"], 1);

    // 清空后再查 → 没有条目。
    let (_, entries) = get(&app, "/api/cache/entries").await;
    assert_eq!(entries["count"], 0);
}

#[tokio::test]
async fn tools_endpoint_lists_tools_with_cacheability() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    let (status, body) = get(&app, "/api/tools").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["count"], 0);
}

#[tokio::test]
async fn malformed_body_returns_bad_request() {
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
    let app = router(test_state(cache));

    let request = Request::builder()
        .method("POST")
        .uri("/api/run")
        .header("content-type", "application/json")
        .body(Body::from("{不是 JSON"))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// ─────────────────────────────────────────────────────────────────────
// 缓存语义测试
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn cache_hit_is_faster_than_cache_miss() {
    // ⚠️ 命中缓存**必须**比未命中快 —— 这是缓存存在的全部意义。
    //
    // 用一个「慢 LLM」（模拟 120ms 网络往返）来放大差异。
    let llm = Arc::new(SlowLlm::new());
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));

    let mut config = AppConfig::default();
    config.agent.max_steps = 1;

    let agent = CachedAgent::new(
        llm.clone(),
        agent_21::tool::ToolRegistry::new(vec![]),
        cache.clone(),
        config,
    );

    let start = std::time::Instant::now();
    agent.run("慢问题").await.unwrap();
    let miss_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    agent.run("慢问题").await.unwrap();
    let hit_elapsed = start.elapsed();

    assert_eq!(llm.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(
        hit_elapsed < miss_elapsed,
        "命中缓存 ({hit_elapsed:?}) 应该比未命中 ({miss_elapsed:?}) 快"
    );
}

#[tokio::test]
async fn disabled_cache_never_hits() {
    let llm = Arc::new(CountingLlm::new());
    let cache = Arc::new(CacheService::disabled(Arc::new(LocalCache::new())));

    let mut config = AppConfig::default();
    config.agent.max_steps = 1;

    let agent = CachedAgent::new(
        llm.clone(),
        agent_21::tool::ToolRegistry::new(vec![]),
        cache,
        config,
    );

    agent.run("问题").await.unwrap();
    agent.run("问题").await.unwrap();

    assert_eq!(llm.count(), 2);
}

#[tokio::test]
async fn different_models_do_not_share_cache_entries() {
    // ⚠️ 键里必须包含模型名 —— 否则换模型会返回**旧模型的答案**。
    let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));

    let mut config = AppConfig::default();
    config.agent.max_steps = 1;
    config.model = "model-a".to_string();

    let agent = CachedAgent::new(
        Arc::new(CountingLlm::new()),
        agent_21::tool::ToolRegistry::new(vec![]),
        cache.clone(),
        config.clone(),
    );

    agent.run("问题").await.unwrap();

    // 换模型 → 键变了 → 必须重新调模型。
    let mut config_b = config.clone();
    config_b.model = "model-b".to_string();
    let agent_b = CachedAgent::new(
        Arc::new(CountingLlm::new()),
        agent_21::tool::ToolRegistry::new(vec![]),
        cache,
        config_b,
    );

    let run = agent_b.run("问题").await.unwrap();
    assert_eq!(run.cache_hits, 0);
    assert_eq!(run.cache_misses, 1);
}

// ─────────────────────────────────────────────────────────────────────
// Redis 真实后端测试（连不上就跳过）
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn redis_backend_stores_and_reads() {
    let Some(backend) = redis_backend(&unique_namespace("basic")).await else {
        return;
    };

    let key = "agent-cache:test:redis:basic";
    let value = CachedResponse {
        tool_calls: Vec::new(),
        text: "Redis 里的答案".to_string(),
        reasoning: None,
        usage: CachedUsage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
        },
        model: "deepseek-flash".to_string(),
        stored_at_ms: now_ms(),
        key_desc: "test".to_string(),
    };

    backend.set(key, &value, 60).await.unwrap();

    let got = backend.get(key).await.unwrap().unwrap();
    assert_eq!(got.text, "Redis 里的答案");
    assert_eq!(got.usage.total_tokens, 3);

    assert!(backend.delete(key).await.unwrap());
    assert!(backend.get(key).await.unwrap().is_none());
}

#[tokio::test]
async fn redis_backend_expires_entries() {
    let Some(backend) = redis_backend(&unique_namespace("ttl")).await else {
        return;
    };

    let key = "agent-cache:test:redis:ttl";
    let value = CachedResponse {
        tool_calls: Vec::new(),
        text: "会过期".to_string(),
        reasoning: None,
        usage: CachedUsage::default(),
        model: "m".to_string(),
        stored_at_ms: now_ms(),
        key_desc: "test".to_string(),
    };

    // TTL = 1 秒。
    backend.set(key, &value, 1).await.unwrap();
    assert!(backend.get(key).await.unwrap().is_some());

    tokio::time::sleep(std::time::Duration::from_millis(1200)).await;
    assert!(backend.get(key).await.unwrap().is_none());
}

#[tokio::test]
async fn redis_backend_clears_only_its_namespace() {
    let namespace = unique_namespace("clear");
    let Some(backend) = redis_backend(&namespace).await else {
        return;
    };

    let value = CachedResponse {
        tool_calls: Vec::new(),
        text: "x".to_string(),
        reasoning: None,
        usage: CachedUsage::default(),
        model: "m".to_string(),
        stored_at_ms: now_ms(),
        key_desc: "test".to_string(),
    };

    backend
        .set(&format!("{namespace}:a"), &value, 60)
        .await
        .unwrap();
    backend
        .set(&format!("{namespace}:b"), &value, 60)
        .await
        .unwrap();
    backend.set("other-ns:c", &value, 60).await.unwrap();

    assert_eq!(backend.count_namespace(&namespace).await.unwrap(), 2);
    assert_eq!(backend.clear_namespace(&namespace).await.unwrap(), 2);
    assert_eq!(backend.count_namespace(&namespace).await.unwrap(), 0);
    // 另一个命名空间不受影响。
    assert_eq!(backend.count_namespace("other-ns").await.unwrap(), 1);

    // 清理。
    let _ = backend.clear_namespace("other-ns").await;
}

#[tokio::test]
async fn redis_serves_as_a_shared_cache_across_service_instances() {
    // ⚠️ 这是 Redis 相对内存缓存的**唯一本质优势**：
    //    两个独立的 CacheService 实例（模拟两个 worker）**共享**同一份缓存。
    //
    //    内存缓存做不到这一点 —— 它的缓存活在一个进程里。
    let namespace = unique_namespace("shared");
    let Some(backend) = redis_backend(&namespace).await else {
        return;
    };

    let service_a = CacheService::new(backend.clone(), 60).with_namespace(&namespace);
    let service_b = CacheService::new(backend.clone(), 60).with_namespace(&namespace);

    let key = agent_21::CacheKey::new(
        &namespace,
        "deepseek-flash",
        &[genai::chat::ChatMessage::user(
            genai::chat::MessageContent::from_text("共享问题".to_string()),
        )],
        "digest",
        "auto",
        0.0,
        2048,
        true,
    );

    let value = CachedResponse {
        tool_calls: Vec::new(),
        text: "共享答案".to_string(),
        reasoning: None,
        usage: CachedUsage::default(),
        model: "deepseek-flash".to_string(),
        stored_at_ms: now_ms(),
        key_desc: "test".to_string(),
    };

    service_a
        .set_response(&key, &value, &CacheControl::default())
        .await;

    // ⚠️ 实例 B 能读到实例 A 写的缓存。
    let hit = service_b
        .get_response(&key, &CacheControl::default())
        .await
        .unwrap();
    assert_eq!(hit.text, "共享答案");

    let _ = service_a.clear_namespace(&namespace).await;
}

#[tokio::test]
async fn local_cache_is_not_shared_across_instances() {
    // ⚠️ 与上一条**对照**：内存缓存**不共享**。
    //
    // 这就是 LiteLLM 文档说的
    // 「a proxy running four workers keeps four separate caches
    // and the hit rate drops roughly by the worker count」。
    let namespace = unique_namespace("local-not-shared");

    let service_a = CacheService::new(Arc::new(LocalCache::new()), 60).with_namespace(&namespace);
    let service_b = CacheService::new(Arc::new(LocalCache::new()), 60).with_namespace(&namespace);

    let key = agent_21::CacheKey::new(
        &namespace,
        "deepseek-flash",
        &[genai::chat::ChatMessage::user(
            genai::chat::MessageContent::from_text("问题".to_string()),
        )],
        "digest",
        "auto",
        0.0,
        2048,
        true,
    );

    let value = CachedResponse {
        tool_calls: Vec::new(),
        text: "答案".to_string(),
        reasoning: None,
        usage: CachedUsage::default(),
        model: "m".to_string(),
        stored_at_ms: now_ms(),
        key_desc: "test".to_string(),
    };

    service_a
        .set_response(&key, &value, &CacheControl::default())
        .await;

    // ⚠️ 实例 B **读不到** —— 内存缓存不跨实例共享。
    assert!(
        service_b
            .get_response(&key, &CacheControl::default())
            .await
            .is_none()
    );
}

#[tokio::test]
async fn unreachable_redis_degrades_instead_of_failing() {
    // ⚠️ 缓存不可用时**不能**让请求失败。
    //
    // 用一个确定连不上的地址构造 Redis 后端，
    // 验证「读失败 → 当 miss + 打 warn」这条降级路径。
    let backend = RedisCache::connect("redis://127.0.0.1:1")
        .await
        .expect_err("端口 1 不应该有 Redis");

    assert!(backend.contains("连接 Redis 失败") || backend.contains("Redis"));
}

#[tokio::test]
async fn service_survives_a_broken_backend() {
    // 用一个「总是失败」的假后端来验证降级路径。
    #[derive(Debug)]
    struct BrokenBackend;

    impl CacheBackend for BrokenBackend {
        fn kind(&self) -> &'static str {
            "broken"
        }

        fn get<'a>(
            &'a self,
            _key: &'a str,
        ) -> agent_21::cache::CacheFuture<'a, Result<Option<CachedResponse>, String>> {
            Box::pin(async move { Err("模拟 Redis 挂了".to_string()) })
        }

        fn set<'a>(
            &'a self,
            _key: &'a str,
            _value: &'a CachedResponse,
            _ttl_seconds: u64,
        ) -> agent_21::cache::CacheFuture<'a, Result<(), String>> {
            Box::pin(async move { Err("模拟 Redis 挂了".to_string()) })
        }

        fn delete<'a>(
            &'a self,
            _key: &'a str,
        ) -> agent_21::cache::CacheFuture<'a, Result<bool, String>> {
            Box::pin(async move { Err("模拟 Redis 挂了".to_string()) })
        }

        fn clear_namespace<'a>(
            &'a self,
            _namespace: &'a str,
        ) -> agent_21::cache::CacheFuture<'a, Result<usize, String>> {
            Box::pin(async move { Err("模拟 Redis 挂了".to_string()) })
        }

        fn count_namespace<'a>(
            &'a self,
            _namespace: &'a str,
        ) -> agent_21::cache::CacheFuture<'a, Result<usize, String>> {
            Box::pin(async move { Err("模拟 Redis 挂了".to_string()) })
        }

        fn list_entries<'a>(
            &'a self,
            _namespace: &'a str,
        ) -> agent_21::cache::CacheFuture<'a, Result<Vec<agent_21::cache::CacheEntry>, String>>
        {
            Box::pin(async move { Err("模拟 Redis 挂了".to_string()) })
        }
    }

    let cache = Arc::new(CacheService::new(Arc::new(BrokenBackend), 60));
    let key = agent_21::CacheKey::new(
        "ns",
        "m",
        &[genai::chat::ChatMessage::user(
            genai::chat::MessageContent::from_text("问题".to_string()),
        )],
        "d",
        "auto",
        0.0,
        2048,
        true,
    );

    // ⚠️ 读失败 → 当未命中（不抛错）。
    assert!(
        cache
            .get_response(&key, &CacheControl::default())
            .await
            .is_none()
    );

    // ⚠️ 写失败 → 忽略（不抛错）。
    let value = CachedResponse {
        tool_calls: Vec::new(),
        text: "x".to_string(),
        reasoning: None,
        usage: CachedUsage::default(),
        model: "m".to_string(),
        stored_at_ms: now_ms(),
        key_desc: "t".to_string(),
    };
    cache
        .set_response(&key, &value, &CacheControl::default())
        .await;

    // 降级次数被记录下来了。
    let stats = cache.stats().snapshot();
    assert_eq!(stats.degraded_reads, 1);
    assert_eq!(stats.degraded_writes, 1);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 1);
}
