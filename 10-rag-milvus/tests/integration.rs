//! 集成测试 —— Milvus / Qdrant 的完整生命周期与混合检索。
//!
//! ## 测试分层
//!
//! | 层 | 需要 Docker | 覆盖内容 |
//! |---|---|---|
//! | 单元测试（各模块内） | 否 | 主键派生、分数归一化、配置解析、schema 映射 |
//! | 集成测试（本文件） | **是** | collection 生命周期、upsert 幂等、稠密检索、BM25 混合检索、RRF 融合、两库对照 |
//!
//! ## 为什么这些测试必须连真实服务
//!
//! 向量库的坑**几乎全在服务端**：
//!
//! - 不 `load_collection` 就搜不到数据
//! - 一致性级别选错会读到旧快照（`Bounded` 在 SDK 里被映射成字面 ts=2）
//! - 主键不自增稳定派生，重复灌库就会堆出重复数据
//! - BM25 函数字段声明错了，服务端直接拒绝建表
//!
//! 这些 mock 掉就没意义了 —— 所以本文件**要求真实 Milvus**。
//!
//! ## 跑法
//!
//! ```bash
//! cd 10-rag-milvus/docker && docker compose up -d
//! cargo test -p rag-10 --test integration -- --nocapture
//! ```
//!
//! 连不上服务时这些测试会**跳过**（而不是失败），
//! 免得没起 Docker 的同学被一堆红字淹没。
//! 但设了 `RAG10_REQUIRE_BACKEND=1` 时会直接失败 —— CI 用这个。

use std::collections::HashSet;

use rag_05::embed::Embedder;
use rag_10::config::{AppConfig, BackendKind, MilvusConfig, QdrantConfig};
use rag_10::index::make_embedder;
use rag_10::milvus_store::MilvusStore;
use rag_10::qdrant_store::QdrantStore;

/// 测试用 collection 的名前缀 —— 与默认配置错开，避免污染开发环境。
const TEST_COLLECTION_PREFIX: &str = "rag10_test";

/// 是否要求后端必须可用（CI 里设 `1`）。
fn require_backend() -> bool {
    std::env::var("RAG10_REQUIRE_BACKEND")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// 连不上后端时的统一处理：CI 直接失败，本地跳过。
fn skip_or_fail<T>(what: &str, result: Result<T, String>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(e) => {
            if require_backend() {
                panic!("{what} 失败（CI 要求后端可用）: {e}");
            }
            eprintln!("[skip] {what} 失败，跳过该测试: {e}");
            None
        }
    }
}

/// 构造测试配置。
///
/// ## 为什么 collection 名要带 tag + pid
///
/// Milvus 与 Qdrant 的 collection 是**全局命名空间**里的资源，
/// 而 Rust 的测试默认**并行**跑。如果所有测试共用一个 collection 名，
/// 它们会互相删对方的 collection —— 症状是随机失败：
///
/// ```text
/// Milvus: fail to wait growing segment ready, collection not found
/// Qdrant: Collection `rag10_test_chunks` already exists!
/// ```
///
/// 所以每个测试用**自己独立的 collection 名**（tag 区分测试 + pid 区分轮次），
/// 跑完再删掉 —— 这样既能并行，又不会互相干扰，
/// 上一轮崩掉留下的残骸也不会影响下一轮。
fn test_config(tag: &str) -> AppConfig {
    let name = format!("{TEST_COLLECTION_PREFIX}_{tag}_{}", std::process::id());

    let mut config = AppConfig::default();
    config.milvus.collection = name.clone();
    config.qdrant.collection = name;
    config.top_k = 5;
    config.per_path_k = 20;
    config
}

/// 连 Milvus（返回 None 表示不可达）。
async fn connect_milvus(config: &AppConfig) -> Option<MilvusStore> {
    let dimension = rag_10::config::model_dimension(config.embedding_model);
    let model = rag_10::config::model_name(config.embedding_model);

    skip_or_fail(
        "连接 Milvus",
        MilvusStore::connect(&config.milvus, dimension, model)
            .await
            .map_err(|e| e.to_string()),
    )
}

/// 连 Qdrant（返回 None 表示不可达）。
async fn connect_qdrant(config: &AppConfig) -> Option<QdrantStore> {
    let dimension = rag_10::config::model_dimension(config.embedding_model);
    let model = rag_10::config::model_name(config.embedding_model);

    skip_or_fail(
        "连接 Qdrant",
        QdrantStore::connect(&config.qdrant, dimension, model)
            .await
            .map_err(|e| e.to_string()),
    )
}

/// 测试用的向量维度 —— 必须与嵌入模型的维度一致（384）。
///
/// 向量库建表时就把维度写死了，插入维度不匹配的向量会被服务端拒绝：
/// `invalid parameter vector: vector dimension 4 does not match schema dimension 384`。
const TEST_DIM: usize = 384;

/// 造几条可断言的记录（向量是确定性的，不依赖嵌入模型下载）。
///
/// 用「正交单位向量」当向量 —— 在 384 维里取前 4 个坐标轴方向：
///
/// ```text
/// doc.md#0    = e0 = [1, 0, 0, 0, ...]
/// doc.md#1    = e1 = [0, 1, 0, 0, ...]
/// doc.md#2    = e2 = [0, 0, 1, 0, ...]
/// other.md#0  = e3 = [0, 0, 0, 1, ...]
/// ```
///
/// 检索时相似度关系完全可预测 —— 余弦相似度下，
/// 查 `e0` 只有 `doc.md#0` 得 1.0，其余全是 0.0。
fn make_records() -> Vec<rag_10::schema::ChunkRecord> {
    let unit = |idx: usize| {
        let mut v = vec![0.0f32; TEST_DIM];
        v[idx] = 1.0;
        v
    };

    vec![
        rag_10::schema::ChunkRecord {
            chunk_id: "doc.md#0".to_string(),
            source: "doc.md".to_string(),
            index: 0,
            heading_path: vec!["向量库".to_string(), "定义".to_string()],
            text: "向量数据库是专门存储和检索高维向量的数据库".to_string(),
            vector: unit(0),
        },
        rag_10::schema::ChunkRecord {
            chunk_id: "doc.md#1".to_string(),
            source: "doc.md".to_string(),
            index: 1,
            heading_path: vec!["向量库".to_string(), "索引".to_string()],
            text: "HNSW 是一种基于图的近似最近邻索引结构".to_string(),
            vector: unit(1),
        },
        rag_10::schema::ChunkRecord {
            chunk_id: "doc.md#2".to_string(),
            source: "doc.md".to_string(),
            index: 2,
            heading_path: vec!["向量库".to_string(), "混合检索".to_string()],
            text: "混合检索把稠密向量和 BM25 稀疏向量用 RRF 融合".to_string(),
            vector: unit(2),
        },
        rag_10::schema::ChunkRecord {
            chunk_id: "other.md#0".to_string(),
            source: "other.md".to_string(),
            index: 0,
            heading_path: vec!["其他".to_string()],
            text: "BM25 的核心是词频和逆文档频率的乘积".to_string(),
            vector: unit(3),
        },
    ]
}

/// 构造一个「指向第 idx 个坐标轴」的查询向量。
fn query_vector(idx: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; TEST_DIM];
    v[idx] = 1.0;
    v
}

/// Milvus：collection 生命周期（建 → 写 → 搜 → 删）。
#[tokio::test]
async fn milvus_collection_lifecycle() {
    let config = test_config("milvus_lifecycle");
    let Some(mut store) = connect_milvus(&config).await else {
        return;
    };

    // 先清干净，保证幂等。
    let _ = store.drop_collection().await;

    let created = store
        .ensure_collection(&config.milvus)
        .await
        .expect("建 collection 失败");
    assert!(created, "首次创建应返回 true");

    // 重复调用是幂等的 —— 不会报错，也不会重建。
    let created_again = store
        .ensure_collection(&config.milvus)
        .await
        .expect("重复 ensure_collection 失败");
    assert!(!created_again, "collection 已存在时应返回 false");

    let records = make_records();
    let written = store.upsert_records(&records).await.expect("写入失败");
    assert_eq!(written as usize, records.len());

    // 写入后能查到 —— 注意用的是 Strong 一致性。
    let count = store.count().await.expect("count 失败");
    assert_eq!(count, records.len(), "服务端条数应等于写入条数");

    // 稠密检索：查 unit(0) 方向，应该把 doc.md#0 排第一。
    let hits = store
        .search_dense(&query_vector(0), 3, None)
        .await
        .expect("稠密检索失败");
    assert!(!hits.is_empty(), "稠密检索应有结果");
    assert_eq!(hits[0].id, "doc.md#0", "最相似的应是 doc.md#0");
    assert!(hits[0].score > 0.99, "正交单位向量的余弦相似度应接近 1");

    // 标量过滤：只查 source == "other.md"。
    let filtered = store
        .search_dense(&query_vector(0), 10, Some("source == \"other.md\""))
        .await
        .expect("过滤检索失败");
    assert_eq!(filtered.len(), 1, "过滤后只应剩 1 条");
    assert_eq!(filtered[0].id, "other.md#0");

    // 删除。
    store.drop_collection().await.expect("删除 collection 失败");
    let exists = store
        .client()
        .has_collection(
            milvus::v2::request::collection::HasCollectionRequest::builder()
                .collection_name(config.milvus.collection.clone())
                .build()
                .expect("构造请求失败"),
        )
        .await
        .expect("查询 collection 存在性失败");
    assert!(!exists.exists(), "删除后不应再存在");
}

/// Milvus：重复 upsert 是幂等的 —— 条数不膨胀。
#[tokio::test]
async fn milvus_upsert_is_idempotent() {
    let config = test_config("milvus_idem");
    let Some(mut store) = connect_milvus(&config).await else {
        return;
    };

    let _ = store.drop_collection().await;
    store
        .ensure_collection(&config.milvus)
        .await
        .expect("建 collection 失败");

    let records = make_records();

    for round in 1..=3 {
        store.upsert_records(&records).await.expect("写入失败");
        let count = store.count().await.expect("count 失败");
        assert_eq!(
            count,
            records.len(),
            "第 {round} 轮 upsert 后条数不应变化（主键由 chunk_id 稳定派生）"
        );
    }

    // 检索结果里也不该出现重复 id。
    let hits = store
        .search_dense(&query_vector(0), 10, None)
        .await
        .expect("稠密检索失败");
    let ids: HashSet<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids.len(), hits.len(), "结果里出现了重复 id");

    store.drop_collection().await.expect("删除 collection 失败");
}

/// Milvus：服务端 BM25 混合检索 + RRF 融合。
#[tokio::test]
async fn milvus_hybrid_search_uses_server_side_bm25() {
    let config = test_config("milvus_hybrid");
    let Some(mut store) = connect_milvus(&config).await else {
        return;
    };

    if !config.milvus.enable_bm25 {
        eprintln!("[skip] 服务端 BM25 已关闭");
        return;
    }

    let _ = store.drop_collection().await;
    store
        .ensure_collection(&config.milvus)
        .await
        .expect("建 collection 失败");
    store
        .upsert_records(&make_records())
        .await
        .expect("写入失败");

    // 查询向量指向 unit(3)（BM25 那条），文本查「BM25」。
    // 稠密路径会命中 other.md#0，BM25 路径也命中它 —— 融合后应该稳定第一。
    let hits = store
        .search_hybrid(
            &query_vector(3),
            "BM25 词频 逆文档频率",
            3,
            config.per_path_k,
            None,
        )
        .await
        .expect("混合检索失败");

    assert!(!hits.is_empty(), "混合检索应有结果");
    assert_eq!(hits[0].id, "other.md#0", "BM25 命中应排第一");

    // RRF 的分数是「排名分」而非相似度 —— 量级远小于 1。
    assert!(
        hits[0].score > 0.0 && hits[0].score < 1.0,
        "RRF 融合分应在 (0,1) 之间，实际 {}",
        hits[0].score
    );

    store.drop_collection().await.expect("删除 collection 失败");
}

/// Milvus：L2 度量下的分数换算方向必须是「越大越相似」。
#[tokio::test]
async fn milvus_l2_scores_are_normalized() {
    let mut config = test_config("milvus_l2");
    config.milvus.metric = rag_10::config::MetricKind::L2;

    let Some(mut store) = connect_milvus(&config).await else {
        return;
    };

    let _ = store.drop_collection().await;
    store
        .ensure_collection(&config.milvus)
        .await
        .expect("建 collection 失败");

    // L2 用单位向量：正交的两个单位向量，L2 距离平方 = 2。
    // 换算后 score = 1 - d/2 = 0。
    let records = vec![rag_10::schema::ChunkRecord {
        chunk_id: "l2.md#0".to_string(),
        source: "l2.md".to_string(),
        index: 0,
        heading_path: vec![],
        text: "L2 度量的换算".to_string(),
        vector: query_vector(0),
    }];
    store.upsert_records(&records).await.expect("写入失败");

    let hits = store
        .search_dense(&query_vector(0), 1, None)
        .await
        .expect("L2 检索失败");
    assert_eq!(hits.len(), 1);
    // 完全相同的向量 → 距离 0 → score = 1。
    assert!(
        (hits[0].score - 1.0).abs() < 1e-4,
        "L2 换算后同向向量分数应接近 1，实际 {}",
        hits[0].score
    );

    store.drop_collection().await.expect("删除 collection 失败");
}

/// Qdrant：collection 生命周期 + upsert 幂等。
#[tokio::test]
async fn qdrant_collection_lifecycle() {
    let config = test_config("qdrant_lifecycle");
    let Some(mut store) = connect_qdrant(&config).await else {
        return;
    };

    let _ = store.drop_collection().await;

    let created = store
        .ensure_collection(&config.qdrant)
        .await
        .expect("建 collection 失败");
    assert!(created, "首次创建应返回 true");

    let records = make_records();
    for round in 1..=3 {
        store.upsert_records(&records).await.expect("写入失败");
        let count = store.count().await.expect("count 失败");
        assert_eq!(
            count,
            records.len(),
            "第 {round} 轮 upsert 后条数不应变化（point id 是稳定派生的 UUID）"
        );
    }

    let hits = store
        .search_dense(&query_vector(0), 3, None)
        .await
        .expect("稠密检索失败");
    assert!(!hits.is_empty(), "稠密检索应有结果");
    assert_eq!(hits[0].id, "doc.md#0");
    assert!(hits[0].score > 0.99);

    store.drop_collection().await.expect("删除 collection 失败");
}

/// Qdrant：payload 全文索引 + BM25 关键词检索。
#[tokio::test]
async fn qdrant_text_search_hits_keyword() {
    let config = test_config("qdrant_text");
    let Some(mut store) = connect_qdrant(&config).await else {
        return;
    };

    let _ = store.drop_collection().await;
    store
        .ensure_collection(&config.qdrant)
        .await
        .expect("建 collection 失败");
    store
        .upsert_records(&make_records())
        .await
        .expect("写入失败");

    // 「HNSW」只出现在 doc.md#1 里 —— 关键词检索应该精确命中。
    let hits = store
        .search_text("HNSW", 3, None)
        .await
        .expect("关键词检索失败");
    assert!(!hits.is_empty(), "关键词检索应有结果");
    assert_eq!(
        hits[0].id,
        "doc.md#1",
        "含 HNSW 的块应排第一，实际 {:?}",
        hits.iter().map(|h| &h.id).collect::<Vec<_>>()
    );

    store.drop_collection().await.expect("删除 collection 失败");
}

/// 两库同题对照：同一份数据、同一批查询，结果应当高度一致。
///
/// 这不是「Milvus 更好」或「Qdrant 更好」——
/// 而是在验证**公平对照的方法论**：语料、模型、查询全一样，
/// 差异只来自引擎实现。
#[tokio::test]
async fn both_backends_agree_on_top_result() {
    let config = test_config("compare");

    let (Some(mut milvus), Some(mut qdrant)) =
        (connect_milvus(&config).await, connect_qdrant(&config).await)
    else {
        return;
    };

    let _ = milvus.drop_collection().await;
    let _ = qdrant.drop_collection().await;
    milvus
        .ensure_collection(&config.milvus)
        .await
        .expect("Milvus 建 collection 失败");
    qdrant
        .ensure_collection(&config.qdrant)
        .await
        .expect("Qdrant 建 collection 失败");

    let records = make_records();
    milvus
        .upsert_records(&records)
        .await
        .expect("Milvus 写入失败");
    qdrant
        .upsert_records(&records)
        .await
        .expect("Qdrant 写入失败");

    // 用真实嵌入模型跑一遍，验证端到端链路。
    let embedder = match make_embedder(config.embedding_model) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[skip] 嵌入模型不可用: {e}");
            return;
        }
    };
    let query = match embedder.embed_query("向量数据库的索引结构") {
        Ok(q) => q,
        Err(e) => {
            eprintln!("[skip] 向量化失败: {e}");
            return;
        }
    };

    let milvus_hits = milvus
        .search_dense(&query, 3, None)
        .await
        .expect("Milvus 检索失败");
    let qdrant_hits = qdrant
        .search_dense(&query, 3, None)
        .await
        .expect("Qdrant 检索失败");

    assert!(!milvus_hits.is_empty(), "Milvus 应有结果");
    assert!(!qdrant_hits.is_empty(), "Qdrant 应有结果");
    assert_eq!(
        milvus_hits[0].id, qdrant_hits[0].id,
        "两个后端的第一名应一致（同样的向量、同样的距离度量）"
    );

    // 分数也应接近 —— 差值来自浮点累加顺序，通常在 1e-3 内。
    let diff = (milvus_hits[0].score - qdrant_hits[0].score).abs();
    assert!(
        diff < 1e-3,
        "两个后端的第一名分数应接近，差值 {diff}（milvus={} qdrant={}）",
        milvus_hits[0].score,
        qdrant_hits[0].score
    );

    milvus
        .drop_collection()
        .await
        .expect("删除 Milvus collection 失败");
    qdrant
        .drop_collection()
        .await
        .expect("删除 Qdrant collection 失败");
}

/// 真实语料端到端：加载 → 切块 → 向量化 → 灌库 → 混合检索。
///
/// 这是「整套系统能不能跑」的最后一公里检查。
#[tokio::test]
async fn end_to_end_with_real_corpus() {
    let config = test_config("e2e");
    let Some(mut store) = connect_milvus(&config).await else {
        return;
    };

    let embedder = match make_embedder(config.embedding_model) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[skip] 嵌入模型不可用: {e}");
            return;
        }
    };

    let index_config = rag_05::pipeline::IndexConfig {
        data_dir: config.data_dir.clone(),
        ..Default::default()
    };
    let build = match rag_10::index::build_chunks(&embedder, &index_config) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[skip] 构建语料失败: {e}");
            return;
        }
    };

    assert!(!build.chunks.is_empty(), "语料应切出至少一个块");

    let _ = store.drop_collection().await;
    store
        .ensure_collection(&config.milvus)
        .await
        .expect("建 collection 失败");

    let records = rag_10::schema::records_from_chunks(&build.chunks);
    store.upsert_records(&records).await.expect("写入失败");

    let count = store.count().await.expect("count 失败");
    assert_eq!(count, records.len(), "服务端条数应等于切块数");

    let query = embedder
        .embed_query("什么是向量数据库")
        .expect("向量化失败");
    let hits = store
        .search_dense(&query, config.top_k, None)
        .await
        .expect("稠密检索失败");

    assert_eq!(
        hits.len(),
        config.top_k.min(records.len()),
        "应返回 top_k 条（⚠️ 若少于 top_k，多半是又踩了 Bounded 一致性的坑）"
    );
    // 分数必须递减 —— 检索结果是有序的。
    for pair in hits.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "分数应单调不增：{} < {}",
            pair[0].score,
            pair[1].score
        );
    }

    store.drop_collection().await.expect("删除 collection 失败");
}

/// HTTP API 层：路由能起来，且 `/api/compare` 能同时返回两个后端的结果。
#[tokio::test]
async fn http_compare_endpoint_returns_both_backends() {
    let config = test_config("http_compare");

    let (Some(mut milvus), Some(mut qdrant)) =
        (connect_milvus(&config).await, connect_qdrant(&config).await)
    else {
        return;
    };

    let _ = milvus.drop_collection().await;
    let _ = qdrant.drop_collection().await;
    milvus
        .ensure_collection(&config.milvus)
        .await
        .expect("Milvus 建 collection 失败");
    qdrant
        .ensure_collection(&config.qdrant)
        .await
        .expect("Qdrant 建 collection 失败");

    let records = make_records();
    milvus
        .upsert_records(&records)
        .await
        .expect("Milvus 写入失败");
    qdrant
        .upsert_records(&records)
        .await
        .expect("Qdrant 写入失败");

    let mut config = config;
    config.backend = BackendKind::Milvus;
    let state = std::sync::Arc::new(rag_10::api::AppStateInner::new(config, None));
    let app = rag_10::api::routes::router(state);

    let body = serde_json::json!({ "question": "向量数据库", "top_k": 3 });

    use tower::util::ServiceExt;

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/compare")
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .expect("构造请求失败"),
        )
        .await
        .expect("请求失败");

    assert!(
        response.status().is_success(),
        "状态码 {}",
        response.status()
    );

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("读取响应体失败");
    // 响应模型只派生 Serialize —— 这里直接按 JSON 结构断言。
    let payload: serde_json::Value = serde_json::from_slice(&bytes).expect("解析响应失败");

    let results = payload["results"].as_array().expect("results 应是数组");
    assert_eq!(results.len(), 2, "应返回两个后端的结果");
    for result in results {
        assert_eq!(
            result["ok"],
            serde_json::Value::Bool(true),
            "后端失败: {result}"
        );
        assert!(
            result["count"].as_u64().unwrap_or(0) > 0,
            "后端应有命中: {result}"
        );
    }
    // 两边第一名应当一致（同一份数据 + 同一套模型 + 同一批查询）。
    let milvus_top = results[0]["hits"][0]["id"].as_str().unwrap_or_default();
    let qdrant_top = results[1]["hits"][0]["id"].as_str().unwrap_or_default();
    assert_eq!(
        milvus_top, qdrant_top,
        "两个后端的第一名应一致（milvus={milvus_top} qdrant={qdrant_top}）"
    );

    milvus
        .drop_collection()
        .await
        .expect("删除 Milvus collection 失败");
    qdrant
        .drop_collection()
        .await
        .expect("删除 Qdrant collection 失败");
}

/// 配置层：后端选择与 collection 名覆盖。
#[test]
fn config_overrides_are_applied() {
    let mut config = AppConfig::default();
    assert_eq!(config.backend, BackendKind::Milvus);
    assert_eq!(config.milvus.collection, "rag_chunks");

    // 直接改配置（不碰环境变量，避免影响并行测试）。
    config.milvus.collection = "abc".to_string();
    config.qdrant.collection = "abc".to_string();
    assert_eq!(config.milvus.collection, config.qdrant.collection);

    let milvus = MilvusConfig::default();
    assert!(milvus.enable_bm25, "默认应启用服务端 BM25");
    let qdrant = QdrantConfig::default();
    assert_eq!(qdrant.url, "http://127.0.0.1:6334");
}
