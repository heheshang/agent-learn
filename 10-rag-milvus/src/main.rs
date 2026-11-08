//! 10-rag-milvus —— 生产级向量数据库（Milvus）可执行入口。
//!
//! 有三种运行模式：
//!
//! ```bash
//! cargo run -p rag-10            # 启动 HTTP 服务
//! cargo run -p rag-10 -- demo    # 建索引 + 检索验证（不启服务）
//! cargo run -p rag-10 -- compare # Milvus ↔ Qdrant 同题对照 ★
//! ```
//!
//! ## 服务启动流程
//!
//! ```text
//! 读配置 ──> 健康检查后端 ──> 建/确认 collection ──> 灌入向量
//!        ──> 绑定端口 ──> axum::serve
//! ```
//!
//! ⚠️ 与 04~09 不同：这里的索引**在远端服务里**，
//! 所以启动时要先确认 Milvus 起来了 —— 起不来就直接报错，
//! 而不是让第一个请求去撞墙。

use std::sync::Arc;

use agent_logging::{info, init, warn};
use anyhow::Result;
use rag_05::embed::Embedder;
use rag_10::api::{AppState, AppStateInner, routes};
use rag_10::config::AppConfig;
use rag_10::index::{check_health, ensure_index, make_embedder};

#[tokio::main]
async fn main() -> Result<()> {
    init("rag-10");

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
        "=== 10-rag-milvus 生产级向量数据库 ===\n{}",
        config.summary()
    );

    println!("\n检查后端连通性...");
    match check_health(&config).await {
        Ok(detail) => {
            println!("  ✓ {detail}");
            info!(backend = config.backend.as_str(), detail = %detail, "后端连通性正常");
        }
        Err(e) => {
            println!("  ✗ {e}");
            warn!(backend = config.backend.as_str(), error = %e, "后端不可达");
            println!("\n提示：先启动 Milvus ——\n  cd 10-rag-milvus/docker && docker compose up -d");
            anyhow::bail!("后端不可达: {e}");
        }
    }

    println!("\n准备索引...");
    let report = ensure_index(&config).await?;
    println!("  ✓ {}", report.summary());

    let state: AppState = Arc::new(AppStateInner::new(config.clone(), Some(report)));

    let app = routes::router(state);
    let listener = tokio::net::TcpListener::bind(&config.bind_addr)
        .await
        .map_err(|e| anyhow::anyhow!("绑定端口 {} 失败: {}", config.bind_addr, e))?;

    let addr = listener.local_addr()?;
    println!("\n服务已启动: http://{addr}\n");
    info!(addr = %addr, backend = config.backend.as_str(), "HTTP 服务已启动");
    println!("端点:");
    println!("  GET  /               服务信息 + collection 状态");
    println!("  GET  /health         健康检查");
    println!("  POST /api/search     纯检索");
    println!("  POST /api/hybrid     服务端混合检索（Milvus BM25 + RRF）");
    println!("  POST /api/compare    Milvus ↔ Qdrant 同题对照 ★");
    println!("  POST /api/reindex    重建索引");
    println!("\n示例:");
    println!(
        "  curl -s http://{addr}/api/compare -H 'Content-Type: application/json' \\\n    \
         -d '{{\"question\":\"混合检索怎么融合两路结果\"}}'"
    );

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("服务启动失败: {e}"))
}

/// 端到端 demo：建索引 → 检索 → 打印命中。
async fn run_demo() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 10-rag-milvus 端到端 Demo ===\n{}", config.summary());

    println!("\n检查后端连通性...");
    match check_health(&config).await {
        Ok(detail) => {
            println!("  ✓ {detail}");
            info!(backend = config.backend.as_str(), detail = %detail, "后端连通性正常");
        }
        Err(e) => {
            println!("  ✗ {e}");
            warn!(backend = config.backend.as_str(), error = %e, "后端不可达");
            println!("\n提示：先启动 Milvus ——\n  cd 10-rag-milvus/docker && docker compose up -d");
            anyhow::bail!("后端不可达: {e}");
        }
    }

    println!("\n加载语料 → 分块 → 向量化...");
    let embedder = make_embedder(config.embedding_model)?;
    let index_config = rag_05::pipeline::IndexConfig {
        data_dir: config.data_dir.clone(),
        ..Default::default()
    };
    let build = rag_10::index::build_chunks(&embedder, &index_config)?;
    println!(
        "  ✓ {} 个块 → dim={}",
        build.chunks.len(),
        embedder.dimension()
    );

    println!("\n写入向量库...");
    let report = match config.backend {
        rag_10::config::BackendKind::Milvus => rag_10::index::index_milvus(&config, &build).await?,
        rag_10::config::BackendKind::Qdrant => rag_10::index::index_qdrant(&config, &build).await?,
    };
    println!("  ✓ {}", report.summary());

    // 检索验证。
    let questions = ["什么是向量数据库", "混合检索怎么做", "BM25 的 IDF 是什么"];

    for question in questions {
        println!("\n{}", "─".repeat(70));
        println!("问: {question}");

        let query = embedder.embed_query(question)?;

        match config.backend {
            rag_10::config::BackendKind::Milvus => {
                let store = rag_10::milvus_store::MilvusStore::connect(
                    &config.milvus,
                    embedder.dimension(),
                    embedder.model_name(),
                )
                .await?;

                let dense = store.search_dense(&query, config.top_k, None).await?;
                print_hits("稠密向量", &dense);

                if config.milvus.enable_bm25 {
                    let hybrid = store
                        .search_hybrid(&query, question, config.top_k, config.per_path_k, None)
                        .await?;
                    print_hits("混合检索（服务端 BM25 + RRF）", &hybrid);
                }
            }
            rag_10::config::BackendKind::Qdrant => {
                let store = rag_10::qdrant_store::QdrantStore::connect(
                    &config.qdrant,
                    embedder.dimension(),
                    embedder.model_name(),
                )
                .await?;
                let dense = store.search_dense(&query, config.top_k, None).await?;
                print_hits("稠密向量", &dense);
            }
        }
    }

    Ok(())
}

/// Milvus ↔ Qdrant 同题对照。
async fn run_compare() -> Result<()> {
    let config = AppConfig::from_env();

    println!(
        "=== 10-rag-milvus Milvus ↔ Qdrant 对照 ===\n{}",
        config.summary()
    );

    println!("\n检查后端连通性...");
    let mut qdrant_config = config.clone();
    qdrant_config.backend = rag_10::config::BackendKind::Qdrant;

    for (name, check) in [
        ("milvus", check_health(&config)),
        ("qdrant", check_health(&qdrant_config)),
    ] {
        match check.await {
            Ok(detail) => {
                println!("  ✓ {name}: {detail}");
                info!(backend = name, detail = %detail, "后端连通性正常");
            }
            Err(e) => {
                println!("  ✗ {name}: {e}");
                warn!(backend = name, error = %e, "后端不可达");
            }
        }
    }

    let embedder = make_embedder(config.embedding_model)?;
    let index_config = rag_05::pipeline::IndexConfig {
        data_dir: config.data_dir.clone(),
        ..Default::default()
    };
    let build = rag_10::index::build_chunks(&embedder, &index_config)?;

    println!("\n灌入两个后端...");
    let milvus_report = rag_10::index::index_milvus(&config, &build).await?;
    println!("  ✓ {}", milvus_report.summary());

    let mut qdrant_config = config.clone();
    qdrant_config.backend = rag_10::config::BackendKind::Qdrant;
    let qdrant_report = rag_10::index::index_qdrant(&qdrant_config, &build).await?;
    println!("  ✓ {}", qdrant_report.summary());

    let questions = [
        "什么是向量数据库",
        "混合检索如何融合 BM25 和向量",
        "BM25 的 IDF 公式",
        "为什么需要 HNSW 索引",
    ];

    for question in questions {
        println!("\n{}", "─".repeat(70));
        println!("问: {question}");

        let query = embedder.embed_query(question)?;

        let milvus_store = rag_10::milvus_store::MilvusStore::connect(
            &config.milvus,
            embedder.dimension(),
            embedder.model_name(),
        )
        .await?;

        let qdrant_store = rag_10::qdrant_store::QdrantStore::connect(
            &config.qdrant,
            embedder.dimension(),
            embedder.model_name(),
        )
        .await?;

        let m_start = std::time::Instant::now();
        let milvus_hits = match milvus_store.search_dense(&query, config.top_k, None).await {
            Ok(h) => h,
            Err(e) => {
                println!("  milvus  ✗ {e}");
                Vec::new()
            }
        };
        let m_ms = m_start.elapsed().as_millis();

        let q_start = std::time::Instant::now();
        let qdrant_hits = match qdrant_store.search_dense(&query, config.top_k, None).await {
            Ok(h) => h,
            Err(e) => {
                println!("  qdrant  ✗ {e}");
                Vec::new()
            }
        };
        let q_ms = q_start.elapsed().as_millis();

        for (name, hits, ms) in [
            ("milvus", &milvus_hits, m_ms),
            ("qdrant", &qdrant_hits, q_ms),
        ] {
            println!("  {name} ({} 条, {ms}ms):", hits.len());
            for hit in hits.iter() {
                println!(
                    "    {:.4}  {}  {}",
                    hit.score,
                    hit.id,
                    hit.heading_path.join(" > ")
                );
            }
        }

        // 重合度。
        let m_ids: std::collections::HashSet<&str> =
            milvus_hits.iter().map(|h| h.id.as_str()).collect();
        let q_ids: std::collections::HashSet<&str> =
            qdrant_hits.iter().map(|h| h.id.as_str()).collect();
        let union = m_ids.union(&q_ids).count();
        let shared = m_ids.intersection(&q_ids).count();
        let overlap = if union == 0 {
            0.0
        } else {
            shared as f32 / union as f32
        };
        println!("  重合度: {overlap:.2}（{shared}/{union}）");
        info!(question, overlap, shared, union, "双后端对照结果");
    }

    Ok(())
}

/// 打印一组命中。
fn print_hits(label: &str, hits: &[rag_05::store::SearchHit]) {
    println!("  {label}（{} 条）:", hits.len());
    for hit in hits {
        println!(
            "    {:.4}  {}  {}",
            hit.score,
            hit.id,
            hit.heading_path.join(" > ")
        );
        let preview: String = hit.text.chars().take(60).collect();
        println!("          {}", preview.replace('\n', " "));
    }
}
