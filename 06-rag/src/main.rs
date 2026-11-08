//! 06-rag —— 手撕 Naive RAG 系统的可执行入口。
//!
//! 有两种运行模式：
//!
//! ```bash
//! cargo run -p rag-06            # 启动 HTTP 服务
//! cargo run -p rag-06 -- demo    # 跑一遍端到端 demo（不启服务）
//! ```
//!
//! ## 服务启动流程
//!
//! ```text
//! 读配置 ──> 建/加载索引 ──> 装配 Retriever/Generator/Agent
//!        ──> 绑定端口 ──> axum::serve
//! ```
//!
//! 索引是**启动时**准备的（不存在就建，存在就加载）——
//! 这样第一个请求进来时不用等嵌入模型。

use std::sync::Arc;

use agent_logging::{info, init, stage_guard};
use anyhow::Result;
use rag_06::agent::{Agent, AgentConfig};
use rag_06::api::{AppState, AppStateInner, routes};
use rag_06::config::AppConfig;
use rag_06::generation::Generator;
use rag_06::retrieval::Retriever;

#[tokio::main]
async fn main() -> Result<()> {
    init("rag-06");

    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "serve".to_string());

    match mode.as_str() {
        "demo" => run_demo().await?,
        "serve" | "api" => run_server().await?,
        other => anyhow::bail!("未知模式: {other}（可选: serve / demo）"),
    }

    Ok(())
}

/// 启动 HTTP 服务。
async fn run_server() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 06-rag Naive RAG 服务 ===\n{}", config.summary());
    info!(
        bind_addr = %config.bind_addr,
        model = %config.generation.model,
        embedding_model = ?config.embedding_model,
        top_k = config.retrieval.top_k,
        min_score = ?config.retrieval.min_score,
        "服务配置已加载"
    );

    // 建索引或加载已有索引。
    println!("\n准备索引...");
    let _stage = stage_guard("准备索引");
    let (store, stats, created) = rag_06::index::ensure_index(&config).await?;
    if created {
        println!("  ✓ 已新建索引: {}", stats.summary());
        info!(stats = %stats.summary(), "索引已新建");
    } else {
        println!("  ✓ 已加载索引: {}", stats.summary());
        info!(stats = %stats.summary(), "索引已加载");
    }

    // 装配核心组件。
    let embedder = rag_06::index::make_embedder(config.embedding_model.to_model())?;
    let retriever = Arc::new(Retriever::new(
        store,
        Box::new(embedder),
        config.retrieval.clone(),
    ));

    let generator = Arc::new(Generator::new(config.generation.clone()));

    let agent = Arc::new(Agent::new(
        retriever.clone(),
        AgentConfig {
            generation: config.generation.clone(),
            ..Default::default()
        },
    ));

    let state: AppState = Arc::new(AppStateInner::new(
        config.clone(),
        retriever,
        generator,
        agent,
        stats,
    ));

    let app = routes::router(state);
    let listener = tokio::net::TcpListener::bind(&config.bind_addr)
        .await
        .map_err(|e| anyhow::anyhow!("绑定端口 {} 失败: {e}", config.bind_addr))?;

    let addr = listener.local_addr()?;
    println!("\n服务已启动: http://{addr}\n");
    info!(addr = %addr, "HTTP 服务已启动");
    println!("端点:");
    println!("  GET  /               服务信息 + 索引状态");
    println!("  GET  /health         健康检查");
    println!("  POST /api/ask        Naive RAG 问答");
    println!("  POST /api/agent      Agent 问答（模型自己决定检索）");
    println!("  POST /api/search     只检索不生成");
    println!("  POST /api/reindex    重建索引");
    println!("\n示例:");
    println!(
        "  curl -s http://{addr}/api/ask -H 'Content-Type: application/json' \\\n    \
         -d '{{\"question\":\"什么是 RAG？\"}}'"
    );

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("服务启动失败: {e}"))
}

/// 端到端 demo：检索 → 生成 → 引用。
async fn run_demo() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 06-rag 端到端 Demo ===\n{}", config.summary());

    let (store, stats, _) = rag_06::index::ensure_index(&config).await?;
    println!("\n索引: {}", stats.summary());

    let embedder = rag_06::index::make_embedder(config.embedding_model.to_model())?;
    let retriever = Retriever::new(store, Box::new(embedder), config.retrieval.clone());
    let generator = Generator::new(config.generation.clone());

    let questions = [
        "什么是检索增强生成（RAG）？",
        "分块的大小应该怎么定？",
        "为什么要在块上加标题路径？",
    ];

    for question in questions {
        println!("\n{}", "─".repeat(70));
        println!("问: {question}");

        // 1. 检索。
        let retrieval = retriever.retrieve(question).await?;
        println!("\n检索到 {} 个块:", retrieval.chunks.len());
        for chunk in &retrieval.chunks {
            let head = if chunk.heading_context.is_empty() {
                String::new()
            } else {
                format!(" {}", chunk.heading_context)
            };
            println!("  [{}] {:.4}{}", chunk.rank + 1, chunk.score, head);
            println!(
                "      {}",
                chunk
                    .text
                    .chars()
                    .take(70)
                    .collect::<String>()
                    .replace('\n', " ")
            );
        }

        // 2. 组 Prompt。
        let prompt = rag_06::prompt::assemble(&retrieval, question);

        // 3. 生成。
        let result = generator
            .generate_with_citations(&prompt, &retrieval)
            .await?;
        println!("\n答: {}", result.answer);
        info!(
            question = %question,
            retrieved = retrieval.chunks.len(),
            citations = result.citations.len(),
            total_tokens = result.usage.total_tokens,
            "Demo 问答完成"
        );
        println!(
            "\n引用: {} 条 | token: prompt={} completion={} total={}",
            result.citations.len(),
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
            result.usage.total_tokens
        );
        for c in &result.citations {
            println!(
                "  [{}] {} · {}",
                c.index,
                c.chunk_id,
                c.heading_path.join(" > ")
            );
        }
    }

    Ok(())
}
