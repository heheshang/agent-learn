//! 05-rag —— RAG Part 2：向量化与存储。
//!
//! 这一部分做 RAG 流水线的第三、四步：
//!
//! ```text
//! 块 ──Embedder──> 向量 ──Store──> 索引 ──search──> 命中的块
//! ```
//!
//! ## 核心认知
//!
//! **Embedding 决定「语义上什么算相关」，VectorStore 决定「在多少候选里找」。**
//!
//! - 换嵌入模型 = 重建索引（维度可能都变了）
//! - 换向量库 = 只改存储层，检索层不动
//!
//! ## 用法
//!
//! ```text
//! cargo run -- embed      # 向量化：看向量长什么样、相似度怎么算
//! cargo run -- build      # 建索引：Part 1 的块 → 向量 → FAISS
//! cargo run -- search     # 检索：FAISS 召回 top-k
//! cargo run -- compare    # 对比：Flat / IVF / HNSW 三种索引
//! cargo run -- persist    # 持久化：存盘再加载，验证结果一致
//! cargo run -- chroma     # Chroma：写入 + 服务端元数据过滤
//! cargo run -- all        # 全部
//! ```

use std::path::PathBuf;

use anyhow::Result;

use rag_05::{
    build_chunks, cosine_similarity, index_path, ChromaStore, Embedder, FaissStore, IndexConfig,
    IndexKind, LocalEmbedder, Model, VectorStore,
};

/// 测试数据目录 —— 复用 Part 1 的文档。
///
/// 用 `CARGO_MANIFEST_DIR` 定位，而不是相对路径 ——
/// `cargo run -p rag-05` 的工作目录是 workspace 根，
/// 不是 `05-rag/`，相对路径会找不到。
const DATA_DIR: &str = "../04-rag/data";

/// 数据目录的绝对路径。
fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DATA_DIR)
}

/// 索引目录的绝对路径。
///
/// 索引存在 **05-rag 自己**的目录下，而不是 04-rag ——
/// 04-rag 只负责「加载 + 分割」，索引是 Part 2 的产物。
fn index_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("index")
}

/// 构造本项目的索引配置。
fn index_config() -> IndexConfig {
    IndexConfig {
        data_dir: data_dir(),
        index_dir: Some(index_dir()),
        ..Default::default()
    }
}

/// Chroma 服务地址。
const CHROMA_ENDPOINT: &str = "http://localhost:8765";

/// Chroma 集合名。
const CHROMA_COLLECTION: &str = "rag05_demo";

#[tokio::main]
async fn main() -> Result<()> {
    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match demo.as_str() {
        "embed" => run_embed_demo().await?,
        "build" => run_build_demo().await?,
        "search" => run_search_demo().await?,
        "compare" => run_compare_demo().await?,
        "persist" => run_persist_demo().await?,
        "chroma" => run_chroma_demo().await?,
        "all" => {
            run_embed_demo().await?;
            run_build_demo().await?;
            run_search_demo().await?;
            run_compare_demo().await?;
            run_persist_demo().await?;
            run_chroma_demo().await?;
        }
        other => anyhow::bail!(
            "未知 demo: {other}（可选: embed / build / search / compare / persist / chroma / all）"
        ),
    }

    Ok(())
}

/// 创建配置好的嵌入器（默认多语言 E5，中英文都能用）。
fn make_embedder() -> Result<LocalEmbedder> {
    LocalEmbedder::new(Model::MultilingualE5Small, None).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Demo 1：向量化 —— 理解 embedding 的数值行为。
///
/// 重点看三件事：
/// 1. 向量长什么样（维度、取值范围）
/// 2. 语义相近 → 余弦相似度高
/// 3. 前缀（`query:` / `passage:`）对相似度的影响
async fn run_embed_demo() -> Result<()> {
    println!("\n=== 1. 向量化 ===");

    let embedder = make_embedder()?;
    println!("模型: {}", embedder.model_name());
    println!("维度: {}", embedder.dimension());
    println!(
        "前缀: 查询={} 文档={}",
        embedder.needs_query_prefix(),
        embedder.needs_document_prefix()
    );

    // ── 向量长什么样 ────────────────────────────────────────
    let samples = vec![
        "query: 什么是检索增强生成".to_string(),
        "passage: 检索增强生成是一种把外部知识库接进大语言模型的方法。".to_string(),
        "passage: 今天天气不错，适合去公园散步。".to_string(),
        "passage: 向量数据库支持按相似度做最近邻检索。".to_string(),
    ];

    let vectors = embedder.embed_documents(&samples)?;
    println!("\n--- 向量样本 ---");
    for (text, v) in samples.iter().zip(&vectors) {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let head: Vec<String> = v.iter().take(5).map(|x| format!("{x:.4}")).collect();
        println!(
            "  「{}…」 dim={} |norm|={:.4} 前5维=[{}]",
            text.chars().take(16).collect::<String>(),
            v.len(),
            norm,
            head.join(", ")
        );
    }

    // ── 相似度矩阵 ───────────────────────────────────────────
    println!("\n--- 余弦相似度矩阵 ---");
    println!("{:<40}", "文本");
    for i in 0..samples.len() {
        let label: String = samples[i].chars().take(14).collect();
        print!("  #{} {:<36}", i, label);
        for j in 0..samples.len() {
            print!(" {:>7.3}", cosine_similarity(&vectors[i], &vectors[j]));
        }
        println!();
    }
    println!("  （#0 是 query，#1..#3 是 passage；#0·#1 应该明显高于 #0·#2）");

    // ── 前缀的影响 ───────────────────────────────────────────
    println!("\n--- 前缀的影响（E5 模型）---");
    let no_prefix = embedder.embed_documents(&["什么是检索增强生成".to_string()])?;
    let with_prefix = embedder.embed_documents(&["query: 什么是检索增强生成".to_string()])?;
    let passage = embedder.embed_documents(&[
        "passage: 检索增强生成是一种把外部知识库接进大语言模型的方法。".to_string(),
    ])?;

    let d_no = cosine_similarity(&no_prefix[0], &passage[0]);
    let d_yes = cosine_similarity(&with_prefix[0], &passage[0]);
    println!("  查询无前缀 → 相似度 {d_no:.4}");
    println!("  查询有前缀 → 相似度 {d_yes:.4}");
    println!(
        "  差值 {:+.4}（前缀是 E5 这类非对称模型的核心约定，用错会明显掉分）",
        d_yes - d_no
    );

    // ── 对称 vs 非对称 ───────────────────────────────────────
    println!("\n--- 对称模型（MiniLM）对比 ---");
    let symmetric = LocalEmbedder::new(Model::MiniLM, None).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("  模型: {}", symmetric.model_name());
    println!(
        "  前缀: 查询={} 文档={}（对称模型两边都不加）",
        symmetric.needs_query_prefix(),
        symmetric.needs_document_prefix()
    );

    Ok(())
}

/// Demo 2：建索引 —— Part 1 的块 → 向量 → FAISS。
async fn run_build_demo() -> Result<()> {
    println!("\n=== 2. 建立 FAISS 索引 ===");

    let embedder = make_embedder()?;
    let config = index_config();

    let build = build_chunks(&embedder, &config)?;

    println!("\n--- 加载的文档 ---");
    for d in &build.documents {
        println!("  {}", d.summary());
    }

    println!("\n--- 分块 + 向量化 ---");
    println!("  块数: {}", build.chunks.len());
    println!("  向量维度: {}", embedder.dimension());

    // 展示一个块：原文 vs 送去做嵌入的文本 vs 向量。
    if let Some(c) = build.chunks.first() {
        println!("\n--- 第一个块 ---");
        println!("  ID: {}", c.id);
        println!("  标题路径: {}", c.heading_path.join(" > "));
        println!(
            "  正文前 80 字: {}",
            c.text.chars().take(80).collect::<String>()
        );
        let head: Vec<String> = c.vector.iter().take(5).map(|x| format!("{x:.4}")).collect();
        println!("  向量前 5 维: [{}]", head.join(", "));
    }

    // 建索引。
    let mut store = FaissStore::new(IndexKind::Flat, embedder.dimension(), embedder.model_name())?;
    store.add(&build.chunks).await?;

    println!("\n--- 索引 ---");
    println!("  类型: {}", store.index_description());
    println!("  条数: {}", store.len());
    println!("  模型: {}", store.model_name());

    // 存盘。
    let path = index_path(&config);
    std::fs::create_dir_all(path.parent().unwrap())?;
    store.save(&path).await?;
    println!("  已保存: {}", path.display());
    println!(
        "  元数据: {}",
        path.with_file_name(format!(
            "{}.meta.json",
            path.file_name().unwrap().to_string_lossy()
        ))
        .display()
    );

    Ok(())
}

/// Demo 3：检索 —— FAISS 召回 top-k。
async fn run_search_demo() -> Result<()> {
    println!("\n=== 3. 检索 ===");

    let embedder = make_embedder()?;
    let config = index_config();

    let build = build_chunks(&embedder, &config)?;
    let mut store = FaissStore::new(IndexKind::Flat, embedder.dimension(), embedder.model_name())?;
    store.add(&build.chunks).await?;

    let queries = [
        "分块的大小应该怎么定？",
        "为什么要加标题路径？",
        "向量检索是怎么算相似度的？",
    ];

    for q in queries {
        println!("\n--- 查询: {q} ---");
        let qv = embedder.embed_query(q)?;
        let hits = store.search(&qv, 3).await?;
        for (rank, hit) in hits.iter().enumerate() {
            println!("  #{} {}", rank + 1, hit.summary());
        }
    }

    println!("\n解读：");
    println!("  - 分数是余弦相似度（因为向量已归一化，内积 == 余弦）");
    println!("  - 越接近 1 越相关；负数表示语义上毫不相干");
    println!("  - 这里只做了召回（retrieve），还没做重排（rerank）—— Part 3");

    Ok(())
}

/// Demo 4：对比 —— 三种 FAISS 索引。
///
/// 这是 FAISS Intro 的核心：不同索引结构在「速度 / 精度 / 内存」上的取舍。
async fn run_compare_demo() -> Result<()> {
    println!("\n=== 4. 索引类型对比 ===");

    let embedder = make_embedder()?;
    let config = index_config();

    let build = build_chunks(&embedder, &config)?;
    let dim = embedder.dimension();

    let queries = [
        "分块的大小应该怎么定？",
        "为什么要加标题路径？",
        "向量检索是怎么算相似度的？",
    ];

    let kinds = [IndexKind::Flat, IndexKind::Ivf(4), IndexKind::Hnsw(16)];

    // 先算出 Flat（精确解）的结果作为 ground truth。
    let mut flat = FaissStore::new(IndexKind::Flat, dim, embedder.model_name())?;
    flat.add(&build.chunks).await?;

    let mut ground_truth: Vec<Vec<String>> = Vec::new();
    for q in &queries {
        let qv = embedder.embed_query(q)?;
        let hits = flat.search(&qv, 3).await?;
        ground_truth.push(hits.iter().map(|h| h.id.clone()).collect());
    }

    println!(
        "\n{:<12} {:>8} {:>10} {:>14} {:>10}",
        "索引", "条数", "耗时(ms)", "与精确解一致", "内存(MB)"
    );
    println!("{}", "-".repeat(60));

    for kind in kinds {
        let start = std::time::Instant::now();
        let mut store = FaissStore::new(kind, dim, embedder.model_name())?;
        store.add(&build.chunks).await?;
        let build_ms = start.elapsed().as_millis();

        // 检索并统计与精确解的一致率。
        let search_start = std::time::Instant::now();
        let mut agree = 0usize;
        let mut total = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let qv = embedder.embed_query(q)?;
            let hits = store.search(&qv, 3).await?;
            let got: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
            for id in &ground_truth[qi] {
                total += 1;
                if got.contains(&id.as_str()) {
                    agree += 1;
                }
            }
        }
        let search_ms = search_start.elapsed().as_millis();

        // 粗略估算内存：每条 4 字节 × 维度。
        let bytes = build.chunks.len() * dim * 4;
        let mb = bytes as f64 / (1024.0 * 1024.0);

        println!(
            "{:<12} {:>8} {:>10} {:>13}% {:>10.2}",
            kind.description(),
            store.len(),
            build_ms + search_ms,
            agree * 100 / total.max(1),
            mb
        );
    }

    println!("\n解读：");
    println!("  - Flat 是精确解，100% 一致，但查询是 O(N) 线性扫描");
    println!("  - IVF 先聚类再只搜部分簇，用精度换速度");
    println!("     nlist=4 在这个数据量下太小，训练点不足会退化");
    println!("  - HNSW 是图索引，单查询 O(log N)，召回率可调（efSearch）");
    println!("  - 数据量上千时三者差别不大，上万到十万才显出优势");

    Ok(())
}

/// Demo 5：持久化 —— 存盘、加载、验证结果一致。
async fn run_persist_demo() -> Result<()> {
    println!("\n=== 5. 持久化 ===");

    let embedder = make_embedder()?;
    let config = index_config();

    let build = build_chunks(&embedder, &config)?;
    let mut store = FaissStore::new(IndexKind::Flat, embedder.dimension(), embedder.model_name())?;
    store.add(&build.chunks).await?;

    let path = index_path(&config);
    std::fs::create_dir_all(path.parent().unwrap())?;
    store.save(&path).await?;

    let size = std::fs::metadata(&path)?.len();
    println!(
        "  已保存: {} ({:.1} KB)",
        path.display(),
        size as f64 / 1024.0
    );

    // 重新加载到一个全新的 store。
    let mut loaded = FaissStore::new(IndexKind::Flat, embedder.dimension(), embedder.model_name())?;
    loaded.load(&path).await?;
    println!("  已加载: {} 条", loaded.len());

    // 对比加载前后的检索结果 —— 必须完全一致。
    let q = "为什么要加标题路径？";
    let qv = embedder.embed_query(q)?;

    let before = store.search(&qv, 3).await?;
    let after = loaded.search(&qv, 3).await?;

    println!("\n--- 查询: {q} ---");
    println!("  存盘前:");
    for h in &before {
        println!("    {}", h.summary());
    }
    println!("  加载后:");
    for h in &after {
        println!("    {}", h.summary());
    }

    let same: Vec<&str> = before.iter().map(|h| h.id.as_str()).collect();
    let same2: Vec<&str> = after.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(same, same2, "加载前后检索结果应完全一致");
    println!("\n  ✓ 加载前后检索结果完全一致");

    // 演示「换了模型没重建索引」会被拦住。
    println!("\n--- 换模型会怎样 ---");
    let other = LocalEmbedder::new(Model::BGESmallZH, None).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut mismatched = FaissStore::new(IndexKind::Flat, other.dimension(), other.model_name())?;
    match mismatched.load(&path).await {
        Ok(()) => println!("  （竟然加载成功了？）"),
        Err(e) => println!("  ✓ 被拒绝: {e}"),
    }

    Ok(())
}

/// Demo 6：Chroma —— 写入 + 服务端元数据过滤。
///
/// 服务没起时优雅降级，不影响其它 demo。
async fn run_chroma_demo() -> Result<()> {
    println!("\n=== 6. Chroma ===");

    let embedder = make_embedder()?;
    let config = index_config();

    let build = build_chunks(&embedder, &config)?;

    let store = match ChromaStore::connect(
        CHROMA_ENDPOINT,
        CHROMA_COLLECTION,
        embedder.dimension(),
        embedder.model_name(),
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            println!("  Chroma 服务未连接（{e}）");
            println!("  启动方式:");
            println!("    uv venv .venv && uv pip install --python .venv/bin/python chromadb");
            println!("    .venv/bin/chroma run --path ./chroma-data --port 8765");
            println!("  跳过本 demo。");
            return Ok(());
        }
    };
    println!(
        "  已连接: {} (集合: {})",
        CHROMA_ENDPOINT,
        store.collection_name()
    );

    // 清空后重建，保证 demo 可重复运行。
    // 删除集合后要重新连接 —— 集合句柄已经失效了。
    store.drop_collection().await?;
    let mut store = ChromaStore::connect(
        CHROMA_ENDPOINT,
        CHROMA_COLLECTION,
        embedder.dimension(),
        embedder.model_name(),
    )
    .await?;

    store.add(&build.chunks).await?;
    println!(
        "  已写入 {} 条（服务端条数: {}）",
        build.chunks.len(),
        store.count().await?
    );

    // 普通检索。
    let q = "为什么要加标题路径？";
    let qv = embedder.embed_query(q)?;
    let hits = store.search_filtered(&qv, 3, None).await?;
    println!("\n--- 查询: {q} ---");
    for (rank, hit) in hits.iter().enumerate() {
        println!("  #{} {}", rank + 1, hit.summary());
    }

    // 服务端元数据过滤 —— Chroma 的核心优势。
    println!("\n--- 带过滤的检索（只搜 rag_guide.md）---");
    let filter = serde_json::json!({ "source": { "$eq": "rag_guide.md" } });
    let hits = store.search_filtered(&qv, 3, Some(filter)).await?;
    for (rank, hit) in hits.iter().enumerate() {
        println!("  #{} {}", rank + 1, hit.summary());
    }

    println!("\n解读：");
    println!("  - 过滤在 Chroma 服务端完成，只扫满足条件的向量");
    println!("  - FAISS 做不到这个：它只返回 ID，过滤得在 Rust 侧做");
    println!("    （先召回一堆再筛，白白浪费了 I/O）");

    Ok(())
}
