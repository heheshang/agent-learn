//! 记忆层的集成测试 —— 零网络、零延迟、完全确定。
//!
//! ## 为什么要用 `ScriptedChatLlm` + `ScriptedExtractor`
//!
//! 本项目的核心行为有**五个**必须锁住的地方：
//!
//! 1. **线程隔离** —— 两个线程的历史互不串台
//! 2. **线程恢复** —— 同一个 `thread_id` 第二次调用能读到第一次的历史
//! 3. **跨线程长期记忆** —— 新线程能召回旧线程存下的事实
//! 4. **事实只进 system 提示**，不进消息数组
//! 5. **「记住失败」不能让「回答」失败**
//!
//! 这些行为用真实模型测是**不可靠**的 ——
//! 模型今天返回 2 个调用，明天返回 1 个，测试就会随机失败。
//! 所以用脚本把模型输出**钉死**。
//!
//! ## 测试分层
//!
//! | 层 | 测什么 | 需要模型？ |
//! |---|---|---|
//! | 本文件 | 记忆层的**控制流**（隔离 / 恢复 / 召回 / 降级） | ❌ |
//! | `api` 测试 | HTTP 契约（状态码、响应形状） | ❌ |
//! | 真实模型 | 端到端是否真的能跑通 | ✅（手动） |

use std::sync::Arc;

use agent_17::agent::{Agent, AgentConfig};
use agent_17::api::AppStateInner;
use agent_17::api::routes::router;
use agent_17::checkpoint::{file_checkpointer, in_memory_checkpointer};
use agent_17::config::AppConfig;
use agent_17::llm::{ChatLlm, ScriptedChatLlm};
use agent_17::longterm::{
    FactExtractor, MemoryFact, MemoryRetriever, MemorySearchFuture, MemoryStore, NoopExtractor,
    ScriptedExtractor, file_store, in_memory_store, rank_facts,
};
use agent_17::state::{MemoryMessage, ThreadId};

use serde_json::{Value, json};
use tower::ServiceExt;

// ─────────────────────────────────────────────────────────────────────
// 测试替身
// ─────────────────────────────────────────────────────────────────────

/// 一个只做 owner 过滤的检索器（不依赖 BM25 的确定性足够）。
struct OwnerRetriever {
    store: Arc<dyn MemoryStore>,
}

impl MemoryRetriever for OwnerRetriever {
    fn search<'a>(
        &'a self,
        query: &'a str,
        owner: &'a str,
        top_k: usize,
    ) -> MemorySearchFuture<'a, Vec<MemoryFact>> {
        let store = self.store.clone();
        let query = query.to_string();
        let owner = owner.to_string();
        Box::pin(async move {
            let facts = store.list(&owner).await?;
            Ok(rank_facts(&query, &facts, top_k))
        })
    }
}

/// 总是失败的抽取器。
struct FailingExtractor;

impl FactExtractor for FailingExtractor {
    fn extract<'a>(
        &'a self,
        _messages: &'a [MemoryMessage],
        _owner: &'a str,
    ) -> MemorySearchFuture<'a, Vec<MemoryFact>> {
        Box::pin(async move { Err(agent_17::error::AppError::Llm("抽取器挂了".to_string())) })
    }
}

/// 构造一个测试用的 Agent。
fn agent_with(
    llm: Arc<dyn ChatLlm>,
    extractor: Arc<dyn FactExtractor>,
    store: Arc<dyn MemoryStore>,
) -> Agent {
    let retriever = Arc::new(OwnerRetriever {
        store: store.clone(),
    });
    Agent::new(
        llm,
        in_memory_checkpointer(),
        store,
        retriever,
        extractor,
        AgentConfig::default(),
    )
}

fn tid(name: &str) -> ThreadId {
    ThreadId::new(name).unwrap()
}

fn fact(subject: &str, predicate: &str, value: &str, owner: &str) -> MemoryFact {
    MemoryFact::new(subject, predicate, value, owner)
}

// ─────────────────────────────────────────────────────────────────────
// 短期记忆：隔离 / 恢复
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn threads_do_not_leak_into_each_other() {
    // ⚠️ **线程隔离的核心断言**。
    let llm = Arc::new(ScriptedChatLlm::always("答案"));
    let agent = agent_with(llm.clone(), Arc::new(NoopExtractor), in_memory_store());

    agent
        .run(&tid("alice"), "alice", "我叫 Alice")
        .await
        .unwrap();
    agent.run(&tid("bob"), "bob", "我叫 Bob").await.unwrap();

    // 线程 bob 的模型调用里**不能**出现「我叫 Alice」。
    let calls = llm.calls();
    let bob_call = calls
        .iter()
        .find(|c| c.messages.iter().any(|m| m.content == "我叫 Bob"))
        .unwrap();
    assert!(
        !bob_call.messages.iter().any(|m| m.content == "我叫 Alice"),
        "线程 B 不应该看到线程 A 的消息"
    );

    // 两个线程各自的历史长度都是 2。
    let alice = agent.peek(&tid("alice")).await.unwrap().unwrap();
    let bob = agent.peek(&tid("bob")).await.unwrap().unwrap();
    assert_eq!(alice.message_count(), 2);
    assert_eq!(bob.message_count(), 2);
}

#[tokio::test]
async fn a_thread_resumes_with_its_previous_history() {
    // ⚠️ **恢复能力的核心断言**。
    let llm = Arc::new(ScriptedChatLlm::always("答案"));
    let agent = agent_with(llm.clone(), Arc::new(NoopExtractor), in_memory_store());

    agent.run(&tid("t1"), "alice", "我叫 Bob").await.unwrap();
    agent.run(&tid("t1"), "alice", "我叫什么").await.unwrap();

    // 第二次调用应该看到第一次的历史。
    let calls = llm.calls();
    assert_eq!(calls.len(), 2);
    assert!(calls[1].messages.iter().any(|m| m.content == "我叫 Bob"));
    assert!(calls[1].messages.iter().any(|m| m.content == "我叫什么"));
}

#[tokio::test]
async fn the_state_survives_a_checkpointer_restart() {
    // ⚠️ **「进程重启后还能接着聊」** ——
    //    第二个 checkpointer 实例（模拟重启）能读到第一个写的状态。
    let dir = tempfile::tempdir().unwrap();
    let tid = tid("persisted");

    {
        let cp = file_checkpointer(dir.path()).unwrap();
        let store = in_memory_store();
        let retriever = Arc::new(OwnerRetriever {
            store: store.clone(),
        });
        let agent = Agent::new(
            Arc::new(ScriptedChatLlm::always("答案")),
            cp,
            store,
            retriever,
            Arc::new(NoopExtractor),
            AgentConfig::default(),
        );
        agent.run(&tid, "alice", "我叫 Bob").await.unwrap();
    }

    // 新的 checkpointer —— 相当于进程重启了。
    let cp = file_checkpointer(dir.path()).unwrap();
    let loaded = cp.load(&tid).await.unwrap().unwrap();
    assert_eq!(loaded.message_count(), 2);
    assert_eq!(loaded.messages[0].content, "我叫 Bob");
}

// ─────────────────────────────────────────────────────────────────────
// 长期记忆：抽取 / 召回 / 合并
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_fact_is_extracted_and_recalled_across_threads() {
    let store = in_memory_store();
    let extractor = Arc::new(ScriptedExtractor::always(vec![fact(
        "用户", "名字", "Bob", "alice",
    )]));
    let llm = Arc::new(ScriptedChatLlm::always("你叫 Bob"));
    let agent = agent_with(llm.clone(), extractor, store.clone());

    // 线程 1：告诉模型名字。
    let run1 = agent.run(&tid("t1"), "alice", "我叫 Bob").await.unwrap();
    assert_eq!(run1.extracted.len(), 1);
    assert_eq!(run1.extracted[0].value, "Bob");

    // 事实真的进了长期记忆库。
    assert_eq!(store.list("alice").await.unwrap().len(), 1);

    // 线程 2（完全不同的线程）应该能召回这条事实。
    let run2 = agent
        .run(&tid("t2"), "alice", "我叫什么名字")
        .await
        .unwrap();
    assert_eq!(run2.recalled.len(), 1);
    assert_eq!(run2.recalled[0].value, "Bob");

    // ⚠️ 事实必须进 **system 提示** —— 否则模型看不到它，召回就白做了。
    let calls = llm.calls();
    assert!(calls[1].system.contains("Bob"));
    assert!(calls[1].system.contains("# Memory"));
}

#[tokio::test]
async fn recalled_facts_never_enter_the_message_array() {
    // ⚠️ **召回的事实不进消息数组** ——
    //    否则每轮都会把事实重复写进历史，越滚越长。
    let store = in_memory_store();
    store
        .put(fact("用户", "名字", "Bob", "alice"))
        .await
        .unwrap();

    let llm = Arc::new(ScriptedChatLlm::always("答案"));
    let agent = agent_with(llm, Arc::new(NoopExtractor), store);

    let run = agent.run(&tid("t1"), "alice", "我叫什么").await.unwrap();

    // 历史里只有「用户输入 + 模型回复」两条，没有 fact。
    assert_eq!(run.history_messages, 2);

    let state = agent.peek(&tid("t1")).await.unwrap().unwrap();
    assert!(
        state
            .messages
            .iter()
            .all(|m| m.role != agent_17::state::MemoryRole::Fact),
        "事实不该被写进短期记忆"
    );
}

#[tokio::test]
async fn a_changed_fact_is_merged_not_duplicated() {
    // ⚠️ 用户改口 → 同 (owner, subject, predicate) **合并**，旧值进 versions。
    let store = in_memory_store();
    store
        .put(fact("用户", "名字", "Bob", "alice"))
        .await
        .unwrap();
    store
        .put(fact("用户", "名字", "Robert", "alice"))
        .await
        .unwrap();

    let facts = store.list("alice").await.unwrap();
    assert_eq!(facts.len(), 1, "合并后应该还是一条");
    assert_eq!(facts[0].value, "Robert", "新值成为当前值");
    assert_eq!(facts[0].versions, vec!["Bob".to_string()], "旧值进历史");
}

#[tokio::test]
async fn owners_are_isolated_in_the_long_term_store() {
    // ⚠️ 长期记忆的**用户隔离** —— 和线程隔离同级别的重要。
    let store = in_memory_store();
    store
        .put(fact("用户", "名字", "Bob", "alice"))
        .await
        .unwrap();
    store
        .put(fact("用户", "名字", "Alice", "bob"))
        .await
        .unwrap();

    assert_eq!(store.list("alice").await.unwrap()[0].value, "Bob");
    assert_eq!(store.list("bob").await.unwrap()[0].value, "Alice");
}

#[tokio::test]
async fn a_failing_extractor_does_not_break_the_answer() {
    // ⚠️ **「记住失败不能让回答失败」** —— 这是硬原则。
    let llm = Arc::new(ScriptedChatLlm::always("答案"));
    let agent = agent_with(llm.clone(), Arc::new(FailingExtractor), in_memory_store());

    let run = agent.run(&tid("t1"), "alice", "我叫 Bob").await.unwrap();

    assert_eq!(run.answer, "答案");
    assert!(run.extracted.is_empty());
}

#[tokio::test]
async fn extraction_can_be_turned_off() {
    // 对照实验：关掉抽取 = 「没有长期记忆的 agent」。
    let store = in_memory_store();
    let llm = Arc::new(ScriptedChatLlm::always("答案"));
    let retriever = Arc::new(OwnerRetriever {
        store: store.clone(),
    });

    let agent = Agent::new(
        llm,
        in_memory_checkpointer(),
        store.clone(),
        retriever,
        Arc::new(NoopExtractor),
        AgentConfig {
            extract_memories: false,
            ..Default::default()
        },
    );

    let run = agent.run(&tid("t1"), "alice", "我叫 Bob").await.unwrap();
    assert!(run.extracted.is_empty());
    assert!(store.list("alice").await.unwrap().is_empty());
}

// ─────────────────────────────────────────────────────────────────────
// 存储层：目录隔离
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn memory_files_are_not_listed_as_threads() {
    // ⚠️ **一个静默的 bug** —— 短期记忆和长期记忆如果共用同一个目录，
    //    `GET /api/threads` 会把记忆文件也当成线程列出来。
    //
    //    这里直接锁住「两个存储用不同目录时不会互相污染」。
    let dir = tempfile::tempdir().unwrap();
    let threads_dir = dir.path().join("threads");
    let memories_dir = dir.path().join("memories");

    let cp = file_checkpointer(&threads_dir).unwrap();
    let store = file_store(&memories_dir).unwrap();

    cp.save(&{
        let mut state = agent_17::state::ThreadState::new("t1");
        state.merge_messages(vec![MemoryMessage::user("你好")]);
        state
    })
    .await
    .unwrap();

    store
        .put(fact("用户", "名字", "Bob", "alice"))
        .await
        .unwrap();

    // 线程列表里**只有** `t1`。
    let threads = cp.list_threads().await.unwrap();
    assert_eq!(threads, vec!["t1".to_string()]);
}

#[tokio::test]
async fn a_corrupted_thread_file_is_tolerated() {
    // ⚠️ 损坏的线程文件不能让整个服务挂掉 —— 按「无状态」处理。
    let dir = tempfile::tempdir().unwrap();
    let cp = file_checkpointer(dir.path()).unwrap();

    std::fs::write(dir.path().join("broken.json"), "{ 不是合法 JSON").unwrap();

    assert!(cp.load(&tid("broken")).await.unwrap().is_none());
}

// ─────────────────────────────────────────────────────────────────────
// HTTP API 层
// ─────────────────────────────────────────────────────────────────────

fn config() -> AppConfig {
    AppConfig::default()
}

/// 构造一个走 HTTP 层的 App（脚本化 LLM + 内存存储）。
fn app(steps: Vec<String>, extractor: Arc<dyn FactExtractor>) -> axum::Router {
    let llm = Arc::new(ScriptedChatLlm::new(steps));
    let store = in_memory_store();
    let retriever = Arc::new(OwnerRetriever {
        store: store.clone(),
    });
    let agent = Arc::new(Agent::new(
        llm.clone(),
        in_memory_checkpointer(),
        store,
        retriever,
        extractor,
        AgentConfig::default(),
    ));
    let state = Arc::new(AppStateInner::new(config(), agent, llm));
    router(state)
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn health_endpoint_works() {
    let response = app(vec![], Arc::new(NoopExtractor))
        .oneshot(request_get("/health"))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["status"], json!("ok"));
}

#[tokio::test]
async fn chat_endpoint_returns_the_answer_and_the_recalled_facts() {
    let store = in_memory_store();
    store
        .put(fact("用户", "名字", "Bob", "alice"))
        .await
        .unwrap();

    let llm = Arc::new(ScriptedChatLlm::always("你叫 Bob"));
    let retriever = Arc::new(OwnerRetriever {
        store: store.clone(),
    });
    let agent = Arc::new(Agent::new(
        llm.clone(),
        in_memory_checkpointer(),
        store,
        retriever,
        Arc::new(NoopExtractor),
        AgentConfig::default(),
    ));
    let state = Arc::new(AppStateInner::new(config(), agent, llm));
    let app = router(state);

    let response = app
        .oneshot(request_post(
            "/api/chat",
            json!({ "thread_id": "t1", "owner": "alice", "question": "我叫什么" }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["answer"], json!("你叫 Bob"));
    assert_eq!(json["recalled"][0]["value"], json!("Bob"));
    assert_eq!(json["history_messages"], json!(2));
}

#[tokio::test]
async fn chat_endpoint_rejects_an_invalid_thread_id() {
    // ⚠️ thread_id 会成为存储的 key —— 路径穿越必须在**入口**挡住。
    let response = app(vec![], Arc::new(NoopExtractor))
        .oneshot(request_post(
            "/api/chat",
            json!({ "thread_id": "../../etc/passwd", "question": "你好" }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_endpoint_rejects_an_empty_question() {
    let response = app(vec![], Arc::new(NoopExtractor))
        .oneshot(request_post(
            "/api/chat",
            json!({ "thread_id": "t1", "question": "   " }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn thread_detail_returns_the_stored_messages() {
    let app = app(vec!["你好 Bob".to_string()], Arc::new(NoopExtractor));

    // 先聊一轮，产生状态。
    let response = app
        .clone()
        .oneshot(request_post(
            "/api/chat",
            json!({ "thread_id": "t1", "owner": "alice", "question": "我叫 Bob" }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let response = app
        .clone()
        .oneshot(request_get("/api/threads/t1"))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["message_count"], json!(2));
    assert_eq!(json["messages"][0]["content"], json!("我叫 Bob"));
}

#[tokio::test]
async fn thread_detail_returns_404_for_an_unknown_thread() {
    let response = app(vec![], Arc::new(NoopExtractor))
        .oneshot(request_get("/api/threads/nope"))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn deleting_a_thread_clears_its_short_term_memory() {
    let app = app(vec!["答案".to_string()], Arc::new(NoopExtractor));

    app.clone()
        .oneshot(request_post(
            "/api/chat",
            json!({ "thread_id": "t1", "owner": "alice", "question": "你好" }),
        ))
        .await
        .unwrap();

    let response = app
        .clone()
        .oneshot(request_delete("/api/threads/t1"))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["deleted"], json!(true));

    // 再查应该是 404。
    let response = app
        .clone()
        .oneshot(request_get("/api/threads/t1"))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn compare_endpoint_reports_isolation_and_resumption() {
    // ⚠️ **隔离判据必须是「恰好 4 条」**（2 轮 × 2 条消息）。
    //
    //    之前写成了 `(线程数 + 1) * 2`，随线程数变化，判据完全失效。
    //
    // ⚠️ 脚本必须给够步数 —— 每个线程跑 2 轮，共 4 次模型调用。
    let extractor = Arc::new(ScriptedExtractor::always(vec![fact(
        "用户", "名字", "Bob", "alice",
    )]));
    let app = app(
        vec![
            "答案 1".to_string(),
            "答案 2".to_string(),
            "答案 3".to_string(),
            "答案 4".to_string(),
        ],
        extractor,
    );

    let response = app
        .oneshot(request_post(
            "/api/compare",
            json!({
                "owner": "alice",
                "threads": ["thread-a", "thread-b"],
                "question": "我叫什么名字？"
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let json = body_json(response).await;
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);

    // 两个线程各自 4 条（2 轮 × 2 条）。
    assert_eq!(results[0]["history_messages"], json!(4));
    assert_eq!(results[1]["history_messages"], json!(4));

    let conclusion = &json["conclusion"];
    assert_eq!(conclusion["threads_isolated"], json!(true));
    assert_eq!(conclusion["thread_resumed"], json!(true));
    // ⚠️ 长期记忆共享必须在**第二轮**判定 —— 第一轮记忆库还是空的。
    assert_eq!(conclusion["longterm_shared"], json!(true));
}

#[tokio::test]
async fn compare_endpoint_requires_at_least_two_threads() {
    let response = app(vec![], Arc::new(NoopExtractor))
        .oneshot(request_post(
            "/api/compare",
            json!({ "threads": ["only-one"], "question": "问题" }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn memories_endpoint_lists_and_clears_the_long_term_store() {
    let store = in_memory_store();
    store
        .put(fact("用户", "名字", "Bob", "alice"))
        .await
        .unwrap();

    let llm = Arc::new(ScriptedChatLlm::always("答案"));
    let retriever = Arc::new(OwnerRetriever {
        store: store.clone(),
    });
    let agent = Arc::new(Agent::new(
        llm.clone(),
        in_memory_checkpointer(),
        store,
        retriever,
        Arc::new(NoopExtractor),
        AgentConfig::default(),
    ));
    let state = Arc::new(AppStateInner::new(config(), agent, llm));
    let app = router(state);

    let response = app
        .clone()
        .oneshot(request_get("/api/memories?owner=alice"))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["count"], json!(1));
    assert_eq!(json["memories"][0]["value"], json!("Bob"));

    let response = app
        .clone()
        .oneshot(request_delete("/api/memories?owner=alice"))
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    let json = body_json(response).await;
    assert_eq!(json["deleted"], json!(true));
    assert_eq!(json["count"], json!(1));
}

// ─────────────────────────────────────────────────────────────────────
// 辅助
// ─────────────────────────────────────────────────────────────────────

fn request_get(path: &str) -> axum::http::Request<axum::body::Body> {
    axum::http::Request::builder()
        .method("GET")
        .uri(path)
        .body(axum::body::Body::empty())
        .unwrap()
}

fn request_post(path: &str, payload: Value) -> axum::http::Request<axum::body::Body> {
    axum::http::Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(payload.to_string()))
        .unwrap()
}

fn request_delete(path: &str) -> axum::http::Request<axum::body::Body> {
    axum::http::Request::builder()
        .method("DELETE")
        .uri(path)
        .body(axum::body::Body::empty())
        .unwrap()
}
