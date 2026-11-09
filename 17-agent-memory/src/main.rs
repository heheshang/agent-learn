//! 17-agent-memory —— 手撕 Agent Memory（消息状态 + thread_id + checkpointer）。
//!
//! 顶层入口：按名字跑各个 demo。
//!
//! 用法：
//! ```text
//! cargo run -- chat "我叫 Bob"              # 单轮对话（默认线程）
//! cargo run -- threads                      # 列出所有线程
//! cargo run -- history thread-a             # 查看某个线程的消息历史
//! cargo run -- memories                     # 列出长期记忆
//! cargo run -- compare                      # ★ 线程隔离 / 恢复对照
//! cargo run -- serve                        # 启动 HTTP 服务（:3100）
//! cargo run -- all                          # 全部
//! ```
//!
//! ⚠️ 默认问题**故意是**「我叫 Bob」——
//! 因为它是**能被记忆验证**的问题：
//! 第一轮告诉模型名字，第二轮问「我叫什么」，
//! 模型答对就说明记忆生效了。

use std::sync::Arc;

use agent_logging::{info, init};
use anyhow::Result;

use agent_17::agent::{Agent, AgentConfig};
use agent_17::api::AppStateInner;
use agent_17::checkpoint::{SharedCheckpointer, file_checkpointer, in_memory_checkpointer};
use agent_17::config::{AppConfig, CheckpointerKind, MemoryStoreKind};
use agent_17::llm::{ChatLlm, GenaiChatLlm};
use agent_17::longterm::{
    MemoryFact, MemoryRetriever, NoopExtractor, SharedFactExtractor, SharedMemoryStore, file_store,
    in_memory_store, rank_facts,
};
use agent_17::state::ThreadId;

/// 默认演示问题。
const DEFAULT_QUESTION: &str = "我叫 Bob";

/// 默认线程 id。
const DEFAULT_THREAD: &str = "demo-thread";

/// 默认 owner。
const DEFAULT_OWNER: &str = "default";

#[tokio::main]
async fn main() -> Result<()> {
    init("agent-17");

    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "chat" => {
            let question = arg_or_default(2, DEFAULT_QUESTION);
            let thread = arg_or_default(3, DEFAULT_THREAD);
            chat_demo(&thread, &question).await?;
        }
        "history" => {
            let thread = arg_or_default(2, DEFAULT_THREAD);
            history_demo(&thread).await?;
        }
        "threads" => {
            threads_demo().await?;
        }
        "memories" => {
            memories_demo().await?;
        }
        "compare" => {
            compare_demo().await?;
        }
        "serve" => serve().await?,
        "all" => {
            compare_demo().await?;
        }
        other => {
            anyhow::bail!(
                "未知 demo: {other}（可选: chat / history / threads / memories / compare / serve / all）"
            );
        }
    }

    Ok(())
}

/// 取第 n 个命令行参数（没有就用默认值）。
fn arg_or_default(index: usize, default: &str) -> String {
    std::env::args()
        .nth(index)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// 构建存储层。
///
/// ## ⚠️ 存储类型由配置决定
///
/// - `file` —— 真实持久化（进程重启后还能恢复）
/// - `memory` —— 临时（进程一重启就没了）
///
/// CLI 默认走 `memory` —— 因为 CLI 演示每次都是新进程，
/// 走文件模式会留下垃圾文件。
///
/// HTTP 服务走 `file` —— 那才是「持久化」该有的样子。
fn build_storage(
    config: &AppConfig,
    use_file: bool,
) -> Result<(SharedCheckpointer, SharedMemoryStore)> {
    if use_file {
        // ⚠️ 短期记忆和长期记忆**必须分目录** ——
        //    否则 `list_threads` 会把记忆文件当成线程列出来。
        let cp = file_checkpointer(config.storage.threads_dir())?;
        let store = file_store(config.storage.memories_dir())?;
        Ok((cp, store))
    } else {
        Ok((in_memory_checkpointer(), in_memory_store()))
    }
}

/// 构建 LLM 客户端。
fn build_llm(config: &AppConfig) -> Arc<dyn ChatLlm> {
    info!(
        model = %config.model,
        temperature = config.temperature,
        max_tokens = config.max_tokens,
        "创建 LLM 客户端"
    );
    Arc::new(GenaiChatLlm::new(
        &config.model,
        config.temperature,
        config.max_tokens,
        config.disable_reasoning,
    ))
}

/// 构建检索器。
///
/// ## 为什么 CLI 用的是「简单检索器」而不是 BM25
///
/// 事实很少（几条到几十条），线性打分完全够用，
/// 而且没有「索引过期」的问题。
fn build_retriever(store: SharedMemoryStore) -> Arc<dyn MemoryRetriever> {
    Arc::new(SimpleRetriever { store })
}

/// 一个简单的检索器 —— 线性 BM25 打分。
struct SimpleRetriever {
    store: SharedMemoryStore,
}

impl MemoryRetriever for SimpleRetriever {
    fn search<'a>(
        &'a self,
        query: &'a str,
        owner: &'a str,
        top_k: usize,
    ) -> agent_17::longterm::MemorySearchFuture<'a, Vec<MemoryFact>> {
        let store = self.store.clone();
        let query = query.to_string();
        let owner = owner.to_string();
        Box::pin(async move {
            let facts = store.list(&owner).await?;
            Ok(rank_facts(&query, &facts, top_k))
        })
    }
}

/// 构建 Agent。
fn build_agent(
    config: &AppConfig,
    llm: Arc<dyn ChatLlm>,
    cp: SharedCheckpointer,
    store: SharedMemoryStore,
    extractor: SharedFactExtractor,
) -> Arc<Agent> {
    let retriever = build_retriever(store.clone());

    let mut agent_config = AgentConfig {
        prompt: config.prompt.clone(),
        extract_memories: config.extract_memories,
        memory_top_k: config.memory_top_k,
    };
    agent_config.prompt.memory_top_k = config.memory_top_k;

    Arc::new(Agent::new(
        llm,
        cp,
        store,
        retriever,
        extractor,
        agent_config,
    ))
}

/// 单轮对话 demo。
async fn chat_demo(thread: &str, question: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let (cp, store) = build_storage(&config, false)?;
    let llm = build_llm(&config);

    // CLI 里用「抽取器直接返回预设事实」会失真 ——
    // 这里用真的 LLM 抽取器（和 HTTP 服务一致）。
    let extractor: SharedFactExtractor =
        Arc::new(agent_17::longterm::LlmFactExtractor::new(llm.clone()));
    let agent = build_agent(&config, llm, cp, store, extractor);

    let thread_id = ThreadId::new(thread).map_err(anyhow::Error::msg)?;

    println!("\n=== 带记忆的对话 ===");
    println!("线程: {thread_id}\n");

    // ── 第一轮：告诉模型名字 ────────────────────────────────
    let run1 = agent.run(&thread_id, DEFAULT_OWNER, question).await?;

    println!("── 第 1 轮 ──");
    println!("问题: {}", run1.question);
    println!("回答: {}", run1.answer);
    println!("历史条数: {}", run1.history_messages);
    if !run1.extracted.is_empty() {
        println!("抽取的事实:");
        for fact in &run1.extracted {
            println!("  · {}", fact.render());
        }
    }

    // ── 第二轮：问「我叫什么」 ──────────────────────────────
    let run2 = agent
        .run(&thread_id, DEFAULT_OWNER, "我叫什么名字？")
        .await?;

    println!("\n── 第 2 轮（同一线程）──");
    println!("问题: {}", run2.question);
    println!("回答: {}", run2.answer);
    println!("历史条数: {}", run2.history_messages);
    if !run2.recalled.is_empty() {
        println!("召回的记忆:");
        for fact in &run2.recalled {
            println!("  · {}", fact.render());
        }
    }

    println!("\n── 线程状态 ──");
    if let Some(state) = agent.peek(&thread_id).await? {
        println!(
            "线程 {} 共 {} 条消息（版本 {}）",
            state.thread_id,
            state.message_count(),
            state.version
        );
        for msg in &state.messages {
            println!("  [{}] {}", msg.role.name(), msg.content);
        }
    }

    Ok(())
}

/// 查看线程历史。
async fn history_demo(thread: &str) -> Result<()> {
    let config = AppConfig::from_env();
    let (cp, store) = build_storage(&config, false)?;
    let llm = build_llm(&config);
    let agent = build_agent(&config, llm, cp, store, Arc::new(NoopExtractor));

    let thread_id = ThreadId::new(thread).map_err(anyhow::Error::msg)?;

    println!("\n=== 线程历史: {thread_id} ===\n");

    match agent.peek(&thread_id).await? {
        Some(state) => {
            println!(
                "共 {} 条消息（版本 {}）\n",
                state.message_count(),
                state.version
            );
            for msg in &state.messages {
                println!("  [{}] {}", msg.role.name(), msg.content);
            }
        }
        None => {
            println!("线程不存在（还没有任何对话）");
        }
    }

    Ok(())
}

/// 列出所有线程。
async fn threads_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let (cp, store) = build_storage(&config, false)?;
    let llm = build_llm(&config);
    let agent = build_agent(&config, llm, cp, store, Arc::new(NoopExtractor));

    let threads = agent.list_threads().await?;

    println!("\n=== 线程列表（{} 个）===", threads.len());
    for t in &threads {
        println!("  · {t}");
    }

    Ok(())
}

/// 列出长期记忆。
async fn memories_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let (cp, store) = build_storage(&config, false)?;
    let llm = build_llm(&config);
    let agent = build_agent(&config, llm, cp, store, Arc::new(NoopExtractor));

    let memories = agent.list_memories(DEFAULT_OWNER).await?;

    println!("\n=== 长期记忆（{} 条）===", memories.len());
    for m in &memories {
        println!("  · {}", m.render());
        if !m.versions.is_empty() {
            println!("    历史版本: {}", m.versions.join(" → "));
        }
    }

    Ok(())
}

/// ★ 线程隔离 / 恢复对照。
///
/// ## 三个对照
///
/// 1. **短期记忆隔离** —— 两个线程各聊两轮，
///    每个线程的历史条数必须**恰好**是 4（2 轮 × 2 条消息）
/// 2. **短期记忆恢复** —— 同一线程第二次调用，历史条数必须增加
/// 3. **长期记忆共享** —— 新线程里也能召回旧线程存下的事实
async fn compare_demo() -> Result<()> {
    let config = AppConfig::from_env();
    let (cp, store) = build_storage(&config, false)?;
    let llm = build_llm(&config);

    let extractor: SharedFactExtractor =
        Arc::new(agent_17::longterm::LlmFactExtractor::new(llm.clone()));
    let agent = build_agent(&config, llm, cp, store, extractor);

    println!("\n=== ★ 线程隔离 / 恢复对照 ===\n");

    // ── 线程 A：告诉模型名字 ────────────────────────────────
    let thread_a = ThreadId::new("thread-a").map_err(anyhow::Error::msg)?;
    let thread_b = ThreadId::new("thread-b").map_err(anyhow::Error::msg)?;

    let a1 = agent
        .run(&thread_a, DEFAULT_OWNER, DEFAULT_QUESTION)
        .await?;
    println!("── 线程 A 第 1 轮 ──");
    println!("问题: {}", a1.question);
    println!("回答: {}", a1.answer);
    println!("历史条数: {}", a1.history_messages);
    for fact in &a1.extracted {
        println!("  抽取: {}", fact.render());
    }

    // ── 线程 B：完全不同的话题 ──────────────────────────────
    let b1 = agent.run(&thread_b, DEFAULT_OWNER, "我住在杭州").await?;
    println!("\n── 线程 B 第 1 轮 ──");
    println!("问题: {}", b1.question);
    println!("回答: {}", b1.answer);
    println!("历史条数: {}", b1.history_messages);
    for fact in &b1.extracted {
        println!("  抽取: {}", fact.render());
    }

    // ── 线程 A 第二轮：应该记得名字 ──────────────────────────
    let a2 = agent
        .run(&thread_a, DEFAULT_OWNER, "我叫什么名字？")
        .await?;
    println!("\n── 线程 A 第 2 轮（同一线程）──");
    println!("问题: {}", a2.question);
    println!("回答: {}", a2.answer);
    println!("历史条数: {}", a2.history_messages);
    if !a2.recalled.is_empty() {
        println!("召回的记忆:");
        for fact in &a2.recalled {
            println!("  · {}", fact.render());
        }
    }

    // ── 新线程 C：跨线程召回长期记忆 ────────────────────────
    let thread_c = ThreadId::new("thread-c").map_err(anyhow::Error::msg)?;
    let c1 = agent
        .run(&thread_c, DEFAULT_OWNER, "我叫什么名字？")
        .await?;
    println!("\n── 线程 C（全新线程）第 1 轮 ──");
    println!("问题: {}", c1.question);
    println!("回答: {}", c1.answer);
    println!("历史条数: {}", c1.history_messages);
    if !c1.recalled.is_empty() {
        println!("召回的记忆（跨线程!）:");
        for fact in &c1.recalled {
            println!("  · {}", fact.render());
        }
    }

    // ── 结论 ────────────────────────────────────────────────
    println!("\n── 结论 ──");

    let a_count = agent
        .peek(&thread_a)
        .await?
        .map(|s| s.message_count())
        .unwrap_or(0);
    let b_count = agent
        .peek(&thread_b)
        .await?
        .map(|s| s.message_count())
        .unwrap_or(0);

    let isolated = a_count == 4 && b_count == 2;
    println!(
        "线程隔离    : {}（线程 A {} 条 / 线程 B {} 条）",
        if isolated { "✅" } else { "❌" },
        a_count,
        b_count
    );

    let resumed = a2.history_messages > a1.history_messages;
    println!(
        "线程恢复    : {}（{} → {} 条）",
        if resumed { "✅" } else { "❌" },
        a1.history_messages,
        a2.history_messages
    );

    let shared = !c1.recalled.is_empty();
    println!(
        "长期记忆共享: {}（新线程召回了 {} 条事实）",
        if shared { "✅" } else { "❌" },
        c1.recalled.len()
    );

    Ok(())
}

/// 启动 HTTP 服务。
async fn serve() -> Result<()> {
    let config = AppConfig::from_env();

    info!("配置:\n{}", config.summary());

    // ⚠️ 服务走**文件**存储 —— 那才是「持久化」该有的样子。
    //
    // ⚠️ 两级记忆**分目录** —— 否则 `GET /api/threads` 会把记忆文件
    //    也当成线程列出来（静默的脏数据）。
    let cp: SharedCheckpointer = match config.storage.checkpointer {
        CheckpointerKind::Memory => in_memory_checkpointer(),
        CheckpointerKind::File => file_checkpointer(config.storage.threads_dir())?,
    };
    let store: SharedMemoryStore = match config.storage.memory_store {
        MemoryStoreKind::Memory => in_memory_store(),
        MemoryStoreKind::File => file_store(config.storage.memories_dir())?,
    };

    let llm = build_llm(&config);
    let extractor: SharedFactExtractor =
        Arc::new(agent_17::longterm::LlmFactExtractor::new(llm.clone()));
    let agent = build_agent(&config, llm.clone(), cp, store, extractor);

    let state = Arc::new(AppStateInner::new(config.clone(), agent, llm));
    let app = agent_17::api::routes::router(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    info!("服务已启动: http://{}", config.bind_addr);

    axum::serve(listener, app).await?;

    Ok(())
}
