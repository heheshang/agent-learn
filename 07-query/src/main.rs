//! 07-query —— 手撕 Query Transformation（查询改写）的可执行入口。
//!
//! 有两种运行模式：
//!
//! ```bash
//! cargo run -p rag-07            # 启动 HTTP 服务
//! cargo run -p rag-07 -- demo    # 跑一遍端到端 demo（不启服务）
//! cargo run -p rag-07 -- compare # 多策略对比 demo（本项目的核心产出）
//! ```
//!
//! ## 服务启动流程
//!
//! ```text
//! 读配置 ──> 建/加载索引 ──> 装配 Retriever / LLM / Pipeline
//!        ──> 绑定端口 ──> axum::serve
//! ```
//!
//! 索引是**启动时**准备的（不存在就建，存在就加载）——
//! 这样第一个请求进来时不用等嵌入模型。

use std::sync::Arc;

use agent_logging::{info, init, stage_guard};
use anyhow::Result;
use rag_07::api::{AppState, AppStateInner, build_pipeline, routes};
use rag_07::config::AppConfig;
use rag_07::llm::{GenaiLlm, LlmOptions};
use rag_07::pipeline::QueryPipeline;
use rag_07::retrieval::MultiQueryRetriever;
use rag_07::transform::Strategy;

#[tokio::main]
async fn main() -> Result<()> {
    init("rag-07");

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

/// 启动 HTTP 服务。
async fn run_server() -> Result<()> {
    let config = AppConfig::from_env();

    println!(
        "=== 07-query Query Transformation 服务 ===\n{}",
        config.summary()
    );
    info!(
        bind_addr = %config.bind_addr,
        default_strategy = config.transform.default_strategy.name(),
        transform_model = %config.transform.transform_model,
        generation_model = %config.generation.model,
        top_k = config.retrieval.top_k,
        fusion = config.retrieval.fusion.name(),
        "服务配置已加载"
    );

    // 建索引或加载已有索引。
    println!("\n准备索引...");
    let _stage = stage_guard("准备索引");
    let (store, stats, created) = rag_07::index::ensure_index(&config).await?;
    if created {
        println!("  ✓ 已新建索引: {}", stats.summary());
        info!(stats = %stats.summary(), "索引已新建");
    } else {
        println!("  ✓ 已加载索引: {}", stats.summary());
        info!(stats = %stats.summary(), "索引已加载");
    }

    // 装配核心组件。
    let embedder = rag_07::index::make_embedder(config.embedding_model.to_model())?;
    let retriever = Arc::new(MultiQueryRetriever::new(
        store,
        Box::new(embedder),
        config.retrieval.effective_per_query_k(),
    ));

    let llm = Arc::new(GenaiLlm::new(
        &config.transform.transform_model,
        LlmOptions {
            temperature: config.transform.transform_temperature,
            max_tokens: config.transform.transform_max_tokens,
            // 改写是短结构化输出，思考只会烧 token。
            disable_reasoning: true,
        },
    )) as Arc<dyn rag_07::llm::Llm>;

    let pipeline = build_pipeline(retriever, llm, &config);

    let state: AppState = Arc::new(AppStateInner::new(config.clone(), pipeline, stats));

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
    println!("  POST /api/ask        改写 → 检索 → 生成");
    println!("  POST /api/search     改写 → 检索（不生成）");
    println!("  POST /api/transform  只改写（迭代提示词用）");
    println!("  POST /api/compare    多策略对比 ★");
    println!("  POST /api/reindex    重建索引");
    println!("\n示例:");
    println!(
        "  curl -s http://{addr}/api/compare -H 'Content-Type: application/json' \\\n    \
         -d '{{\"question\":\"模型的知识有截止日期怎么办\",\"strategies\":[\"none\",\"hyde\",\"multi_query\"]}}'"
    );

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("服务启动失败: {e}"))
}

/// 端到端 demo：改写 → 检索 → 融合 → 生成。
async fn run_demo() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 07-query 端到端 Demo ===\n{}", config.summary());

    let (store, stats, _) = rag_07::index::ensure_index(&config).await?;
    println!("\n索引: {}", stats.summary());

    let embedder = rag_07::index::make_embedder(config.embedding_model.to_model())?;
    let retriever = Arc::new(MultiQueryRetriever::new(
        store,
        Box::new(embedder),
        config.retrieval.effective_per_query_k(),
    ));

    let llm = Arc::new(GenaiLlm::new(
        &config.transform.transform_model,
        LlmOptions {
            temperature: config.transform.transform_temperature,
            max_tokens: config.transform.transform_max_tokens,
            // 改写是短结构化输出，思考只会烧 token。
            disable_reasoning: true,
        },
    )) as Arc<dyn rag_07::llm::Llm>;

    let pipeline = QueryPipeline::new(
        retriever,
        llm,
        config.retrieval.clone(),
        config.transform.clone(),
    );

    let questions = [
        "模型的知识有截止日期怎么办",
        "光合作用的光反应和暗反应分别在什么位置进行？各自的作用是什么？",
    ];

    for question in questions {
        println!("\n{}", "─".repeat(70));
        println!("问: {question}");

        let outcome = pipeline
            .ask(question, config.transform.default_strategy)
            .await?;

        println!("\n改写结果（{}）:", outcome.retrieval.strategy.name());
        for q in &outcome.retrieval.transformed_queries {
            println!("  {}", q.summary());
        }

        println!("\n融合后命中 {} 个块:", outcome.retrieval.hits.len());
        for hit in &outcome.retrieval.hits {
            println!(
                "  [{}] {:.4} 共识 {}/{}  {}",
                hit.rank + 1,
                hit.best_score,
                hit.votes,
                outcome.retrieval.transformed_queries.len(),
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

        println!("\n答: {}", outcome.answer);
        info!(
            question = %question,
            strategy = outcome.retrieval.strategy.name(),
            hits = outcome.retrieval.hits.len(),
            citations = outcome.citations.len(),
            transform_ms = outcome.retrieval.transform_ms,
            retrieve_ms = outcome.retrieval.retrieve_ms,
            "Demo 问答完成"
        );
        println!(
            "\n引用: {} 条 | 改写 {}ms | 检索 {}ms",
            outcome.citations.len(),
            outcome.retrieval.transform_ms,
            outcome.retrieval.retrieve_ms
        );
    }

    Ok(())
}

/// 多策略对比 demo —— 本项目的核心产出。
async fn run_compare() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 07-query 多策略对比 Demo ===\n{}", config.summary());

    let (store, stats, _) = rag_07::index::ensure_index(&config).await?;
    println!("\n索引: {}", stats.summary());

    let embedder = rag_07::index::make_embedder(config.embedding_model.to_model())?;
    let retriever = Arc::new(MultiQueryRetriever::new(
        store,
        Box::new(embedder),
        config.retrieval.effective_per_query_k(),
    ));

    let llm = Arc::new(GenaiLlm::new(
        &config.transform.transform_model,
        LlmOptions {
            temperature: config.transform.transform_temperature,
            max_tokens: config.transform.transform_max_tokens,
            // 改写是短结构化输出，思考只会烧 token。
            disable_reasoning: true,
        },
    )) as Arc<dyn rag_07::llm::Llm>;

    let pipeline = QueryPipeline::new(
        retriever,
        llm,
        config.retrieval.clone(),
        config.transform.clone(),
    );

    // 这些问题各有明确的「正确答案所在章节」——
    // 召回差异一眼就能看出来。
    let cases = [
        ("模型的知识有截止日期怎么办", "检索增强生成"),
        ("光合作用的光反应和暗反应分别在什么位置进行？", "光合作用"),
        ("数据库索引为什么能加速查询？", "数据库索引"),
    ];

    for (question, expected) in cases {
        println!("\n{}", "─".repeat(70));
        println!("问: {question}");
        println!("期望命中: 「{expected}」");

        for strategy in Strategy::all() {
            let retrieval = pipeline.retrieve(question, strategy).await?;

            let all_text: String = retrieval
                .hits
                .iter()
                .map(|h| format!("{} {}", h.text, h.heading_path.join(" ")))
                .collect::<Vec<_>>()
                .join(" ");

            let hit = all_text.contains(expected);

            info!(
                question = %question,
                strategy = strategy.name(),
                transformed = retrieval.transformed_queries.len(),
                hits = retrieval.hits.len(),
                matched_expected = hit,
                retrieve_ms = retrieval.retrieve_ms,
                "策略对比"
            );

            println!(
                "  {:<12} {} 条改写 | 命中 {} 个块 | {}  ({}ms)",
                strategy.name(),
                retrieval.transformed_queries.len(),
                retrieval.hits.len(),
                if hit {
                    "✓ 召回了正确内容"
                } else {
                    "✗ 没召回到"
                },
                retrieval.retrieve_ms
            );

            for q in &retrieval.transformed_queries {
                println!("      {}", q.summary());
            }
            for hit_item in &retrieval.hits {
                println!(
                    "      [{}] {:.4} 共识 {}/{}  {}",
                    hit_item.rank + 1,
                    hit_item.best_score,
                    hit_item.votes,
                    retrieval.transformed_queries.len(),
                    hit_item.heading_path.join(" > ")
                );
            }
        }
    }

    Ok(())
}
