//! 08-rag —— 手撕混合检索（Hybrid Search）与重排（Rerank）的可执行入口。
//!
//! 有三种运行模式：
//!
//! ```bash
//! cargo run -p rag-08            # 启动 HTTP 服务
//! cargo run -p rag-08 -- demo    # 跑一遍端到端 demo（不启服务）
//! cargo run -p rag-08 -- compare # 多模式对比 demo（本项目的核心产出）
//! ```
//!
//! ## 服务启动流程
//!
//! ```text
//! 读配置 ──> 建/加载索引（FAISS + BM25）──> 装配 Retriever / Reranker / Pipeline
//!        ──> 绑定端口 ──> axum::serve
//! ```
//!
//! 索引是**启动时**准备的（不存在就建，存在就加载）——
//! 这样第一个请求进来时不用等嵌入模型。

use std::sync::Arc;

use agent_logging::{info, init, stage_guard};
use anyhow::Result;
use rag_08::api::{AppState, AppStateInner, build_pipeline, routes};
use rag_08::config::AppConfig;
use rag_08::hybrid::SearchMode;
use rag_08::index::IndexStats;
use rag_08::llm::GenaiLlm;
use rag_08::pipeline::HybridPipeline;
use rag_08::reranker::Reranker;
use rag_08::tokenize::tokenize;

#[tokio::main]
async fn main() -> Result<()> {
    init("rag-08");

    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "serve".to_string());

    match mode.as_str() {
        "demo" => run_demo().await?,
        "compare" => run_compare().await?,
        "serve" | "api" => run_server().await?,
        other => anyhow::bail!("未知模式: {other}（可选: serve / demo / compare）"),
    }

    Ok(())
}

/// 装配好的运行时组件。
struct Runtime {
    pipeline: Arc<HybridPipeline>,
    stats: IndexStats,
}

/// 装配所有组件（serve / demo / compare 三种模式共用）。
///
/// 抽成函数是为了保证**三条路径用的是同一套装配逻辑** ——
/// 不然「demo 能跑但服务起不来」这种事会发生。
async fn bootstrap(config: &AppConfig) -> Result<Runtime> {
    // 建索引或加载已有索引。
    let (store, bm25, stats, created) = rag_08::index::ensure_index(config).await?;
    if created {
        println!("  ✓ 已新建索引: {}", stats.summary());
    } else {
        println!("  ✓ 已加载索引: {}", stats.summary());
    }
    println!(
        "  ✓ BM25 倒排索引: {} 个块 | 词表 {} 项",
        bm25.len(),
        bm25.vocab_size()
    );

    // 装配检索器。
    let embedder = rag_08::index::make_embedder(config.embedding_model.to_model())?;
    let retriever = Arc::new(rag_08::pipeline::make_retriever(
        store,
        bm25,
        Box::new(embedder),
        config,
    ));

    // 装配重排器。
    let reranker: Option<Arc<dyn Reranker>> = if config.rerank.enabled {
        let r = rag_08::reranker::make_reranker(
            config.rerank.backend,
            config.rerank.model,
            Some(rag_08::index::cache_dir()),
            config.rerank.cohere_api_key.as_deref(),
            &config.rerank.cohere_model,
        )?;
        println!("  ✓ 重排器: {}", r.name());
        Some(r)
    } else {
        println!("  ⚠ 重排已关闭（对照组）");
        None
    };

    // 装配 LLM。
    let llm = Arc::new(GenaiLlm::new(
        &config.generation.model,
        rag_08::llm::LlmOptions {
            temperature: config.generation.temperature,
            max_tokens: config.generation.max_tokens,
            // 生成侧保留思考 —— 问答质量需要推理。
            disable_reasoning: false,
        },
    )) as Arc<dyn rag_08::llm::Llm>;

    let pipeline = build_pipeline(retriever, reranker, llm, config);

    Ok(Runtime { pipeline, stats })
}

/// 启动 HTTP 服务。
async fn run_server() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 08-rag 混合检索 + 重排服务 ===\n{}", config.summary());
    info!(
        bind_addr = %config.bind_addr,
        mode = config.retrieval.mode.name(),
        top_k = config.retrieval.top_k,
        rerank_enabled = config.rerank.enabled,
        rerank_backend = config.rerank.backend.name(),
        rerank_model = %config.rerank.model.display_name(),
        generation_model = %config.generation.model,
        "服务配置已加载"
    );

    println!("\n准备索引...");
    let _stage = stage_guard("准备索引");
    let (store, bm25, stats, created) = rag_08::index::ensure_index(&config).await?;
    if created {
        println!("  ✓ 已新建索引: {}", stats.summary());
        info!(stats = %stats.summary(), "索引已新建");
    } else {
        println!("  ✓ 已加载索引: {}", stats.summary());
        info!(stats = %stats.summary(), "索引已加载");
    }
    println!(
        "  ✓ BM25 倒排索引: {} 个块 | 词表 {} 项",
        bm25.len(),
        bm25.vocab_size()
    );

    // ⚠️ 记住 BM25 的文档数 —— `bm25` 马上会被 move 进检索器。
    let bm25_documents = bm25.len();

    let embedder = rag_08::index::make_embedder(config.embedding_model.to_model())?;
    let retriever = Arc::new(rag_08::pipeline::make_retriever(
        store,
        bm25,
        Box::new(embedder),
        &config,
    ));

    let reranker: Option<Arc<dyn Reranker>> = if config.rerank.enabled {
        let r = rag_08::reranker::make_reranker(
            config.rerank.backend,
            config.rerank.model,
            Some(rag_08::index::cache_dir()),
            config.rerank.cohere_api_key.as_deref(),
            &config.rerank.cohere_model,
        )?;
        println!("  ✓ 重排器: {}", r.name());
        info!(reranker = %r.name(), "重排器就绪");
        Some(r)
    } else {
        println!("  ⚠ 重排已关闭（对照组）");
        info!("重排已关闭（对照组）");
        None
    };

    let llm = Arc::new(GenaiLlm::new(
        &config.generation.model,
        rag_08::llm::LlmOptions {
            temperature: config.generation.temperature,
            max_tokens: config.generation.max_tokens,
            // 生成侧保留思考 —— 问答质量需要推理。
            disable_reasoning: false,
        },
    )) as Arc<dyn rag_08::llm::Llm>;

    let pipeline = build_pipeline(retriever, reranker, llm, &config);

    let state: AppState = Arc::new(AppStateInner::new(
        config.clone(),
        pipeline,
        stats,
        bm25_documents,
    ));

    let app = routes::router(state);
    let listener = tokio::net::TcpListener::bind(&config.bind_addr)
        .await
        .map_err(|e| anyhow::anyhow!("绑定端口 {} 失败: {}", config.bind_addr, e))?;

    let addr = listener.local_addr()?;
    println!("\n服务已启动: http://{addr}\n");
    info!(addr = %addr, "HTTP 服务已启动");
    println!("端点:");
    println!("  GET  /               服务信息 + 索引状态");
    println!("  GET  /health         健康检查");
    println!("  POST /api/ask        混合检索 + 重排 + 生成");
    println!("  POST /api/search     混合检索 + 重排（不生成）");
    println!("  POST /api/compare    多模式对比 ★");
    println!("  POST /api/reindex    重建索引");
    println!("\n示例:");
    println!(
        "  curl -s http://{addr}/api/compare -H 'Content-Type: application/json' \\\n    \
         -d '{{\"question\":\"BM25 打分公式是什么\",\"modes\":[\"vector\",\"bm25\",\"hybrid\"],\"expected\":[\"BM25 打分算法\"]}}'"
    );

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("服务启动失败: {e}"))
}

/// 端到端 demo：混合检索 → 重排 → 生成。
async fn run_demo() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 08-rag 端到端 Demo ===\n{}", config.summary());

    let runtime = bootstrap(&config).await?;
    println!("\n索引: {}", runtime.stats.summary());

    let questions = ["BM25 的 IDF 是怎么算的", "TCP 三次握手是为什么"];

    for question in questions {
        println!("\n{}", "─".repeat(70));
        println!("问: {question}");
        println!("分词: {:?}", tokenize(question));

        let outcome = runtime.pipeline.retrieve(question).await?;

        println!("\n融合后命中 {} 个块:", outcome.fused_hits.len());
        for hit in &outcome.fused_hits {
            println!(
                "  [{}] rrf={:.6} 向量={:?} bm25={:?} 共识 {}/2  {}",
                hit.rank + 1,
                hit.score,
                hit.vector_score.map(|s| format!("{s:.4}")),
                hit.bm25_score.map(|s| format!("{s:.2}")),
                hit.votes,
                hit.heading_path.join(" > ")
            );
        }

        println!(
            "\n重排后命中 {} 个块（{}）:",
            outcome.hits.len(),
            if outcome.reranked {
                format!("耗时 {}ms", outcome.rerank_ms)
            } else {
                format!("未执行 — {}", outcome.degrade_reason.unwrap_or_default())
            }
        );
        for hit in &outcome.hits {
            println!(
                "  [{}] 共识 {}/2  {}",
                hit.rank + 1,
                hit.votes,
                hit.heading_path.join(" > ")
            );
            println!(
                "      {}",
                hit.text
                    .chars()
                    .take(60)
                    .collect::<String>()
                    .replace('\n', " ")
            );
        }
    }

    Ok(())
}

/// 多模式对比 demo —— 本项目的核心产出。
async fn run_compare() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 08-rag 多模式对比 Demo ===\n{}", config.summary());

    let runtime = bootstrap(&config).await?;
    println!("\n索引: {}", runtime.stats.summary());

    // 这些问题各有明确的「正确答案所在章节」——
    // 召回差异一眼就能看出来。
    //
    // ⚠️ 最后两个是**故意为 BM25 设计的**：
    // 「ECONNREFUSED」「BM25」这类罕见字符串，向量检索很难命中，
    // BM25 却能精确命中 —— 这正是混合检索存在的理由。
    let cases = [
        ("BM25 打分公式是什么", "BM25 打分算法"),
        ("ECONNREFUSED 是什么意思", "错误码速查"),
        ("数据库索引为什么能加速查询？", "数据库索引"),
        ("模型的知识有截止日期怎么办", "检索增强生成"),
        ("光合作用的光反应发生在什么位置？", "类囊体膜"),
    ];

    for (question, expected) in cases {
        println!("\n{}", "─".repeat(70));
        println!("问: {question}");
        println!("期望命中: 「{expected}」");

        for mode in SearchMode::all() {
            let outcome = runtime.pipeline.retrieve_with(question, mode).await?;

            let all_text: String = outcome
                .hits
                .iter()
                .map(|h| format!("{} {}", h.text, h.heading_path.join(" ")))
                .collect::<Vec<_>>()
                .join(" ");

            let hit = all_text.contains(expected);

            info!(
                question = %question,
                mode = mode.name(),
                hits = outcome.hits.len(),
                matched_expected = hit,
                retrieve_ms = outcome.retrieve_ms,
                rerank_ms = outcome.rerank_ms,
                "模式对比"
            );

            println!(
                "  {:<8} 命中 {} 个块 | {}  (检索 {}ms / 重排 {}ms)",
                mode.name(),
                outcome.hits.len(),
                if hit {
                    "✓ 召回了正确内容"
                } else {
                    "✗ 没召回到"
                },
                outcome.retrieve_ms,
                outcome.rerank_ms,
            );

            for hit_item in &outcome.hits {
                println!(
                    "      [{}] 共识 {}/2  {}",
                    hit_item.rank + 1,
                    hit_item.votes,
                    hit_item.heading_path.join(" > ")
                );
            }
        }
    }

    Ok(())
}
