//! 索引流水线 —— 把 Part 1 的块灌进 Milvus / Qdrant。
//!
//! ```text
//! 文档 ──Loader──> 文本 ──Splitter──> 块 ──Embedder──> 向量 ──Store──> 远程索引
//! ```
//!
//! 与 05-rag 的差别只有一处，但那一处是**本质差别**：
//!
//! - 05-rag：索引在**本地文件**里，`save` / `load` 有意义
//! - 10-rag：索引在**远程服务**里，`save` / `load` 是 no-op
//!
//! 「重启服务要重新灌一遍数据」不是缺陷，而是生产向量库的常态 ——
//! 数据本来就在服务端，客户端只负责「确保它在」。

use std::time::Instant;

use agent_logging::{debug, info, stage_guard};
use milvus::v2::request::utility::CheckHealthRequest;
use rag_05::embed::{Embedder, LocalEmbedder, Model};
use rag_05::pipeline::{IndexBuild, IndexConfig};
use rag_05::store::VectorStore;

use crate::config::{AppConfig, BackendKind, model_dimension};
use crate::error::{RagError, RagResult};
use crate::milvus_store::MilvusStore;
use crate::qdrant_store::QdrantStore;
use crate::schema::records_from_chunks;
/// 索引构建结果。
#[derive(Debug, Clone)]
pub struct IndexReport {
    /// 用了哪个后端。
    pub backend: BackendKind,
    /// 灌进去多少条。
    pub chunks: usize,
    /// 服务端确认的条数。
    pub server_count: usize,
    /// collection 名。
    pub collection: String,
    /// 向量维度。
    pub dimension: usize,
    /// 嵌入模型名。
    pub embedding_model: String,
    /// 是否新建了 collection。
    pub created: bool,
    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

impl IndexReport {
    /// 一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "{} / {} | {} 块 | dim={} | {} | {}ms",
            self.backend.as_str(),
            self.collection,
            self.chunks,
            self.dimension,
            if self.created { "新建" } else { "已存在" },
            self.elapsed_ms
        )
    }
}

/// 缓存目录（绝对路径）。
///
/// 与 05/06/07/08 保持一致：优先 `FASTEMBED_CACHE_DIR`，
/// 否则用 workspace 根目录下的 `.fastembed_cache`。
///
/// 绝对路径是必须的 —— 否则 `cargo run` 和 `cargo test`
/// 的工作目录不同，会重复下载模型。
pub fn cache_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("FASTEMBED_CACHE_DIR")
        && !dir.is_empty()
    {
        return std::path::PathBuf::from(dir);
    }

    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join(".fastembed_cache"))
        .unwrap_or_else(|| std::path::PathBuf::from(".fastembed_cache"))
}

/// 创建嵌入器。
pub fn make_embedder(model: Model) -> RagResult<LocalEmbedder> {
    LocalEmbedder::new(model, Some(cache_dir())).map_err(|e| RagError::Internal(e.to_string()))
}

/// 加载 + 分割 + 向量化（复用 05-rag 的流水线）。
pub fn build_chunks(embedder: &dyn Embedder, config: &IndexConfig) -> RagResult<IndexBuild> {
    rag_05::pipeline::build_chunks(embedder, config).map_err(|e| RagError::Internal(e.to_string()))
}

/// 把块灌进 Milvus。
pub async fn index_milvus(config: &AppConfig, build: &IndexBuild) -> RagResult<IndexReport> {
    let start = Instant::now();

    let dimension = model_dimension(config.embedding_model);
    let model_name = crate::config::model_name(config.embedding_model);

    let mut store = MilvusStore::connect(&config.milvus, dimension, model_name).await?;

    let created = store
        .ensure_collection(&config.milvus)
        .await
        .map_err(|e| RagError::Storage(e.to_string()))?;

    let records = records_from_chunks(&build.chunks);
    let inserted = store
        .upsert_records(&records)
        .await
        .map_err(|e| RagError::Storage(e.to_string()))?;

    let server_count = store
        .count()
        .await
        .map_err(|e| RagError::Storage(e.to_string()))?;

    info!(
        backend = "milvus",
        collection = %config.milvus.collection,
        inserted,
        server_count,
        created,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Milvus 索引写入完成"
    );

    Ok(IndexReport {
        backend: BackendKind::Milvus,
        chunks: inserted as usize,
        server_count,
        collection: config.milvus.collection.clone(),
        dimension,
        embedding_model: model_name.to_string(),
        created,
        elapsed_ms: start.elapsed().as_millis() as u64,
    })
}

/// 把块灌进 Qdrant。
pub async fn index_qdrant(config: &AppConfig, build: &IndexBuild) -> RagResult<IndexReport> {
    let start = Instant::now();

    let dimension = model_dimension(config.embedding_model);
    let model_name = crate::config::model_name(config.embedding_model);

    let mut store = QdrantStore::connect(&config.qdrant, dimension, model_name).await?;

    let created = store
        .ensure_collection(&config.qdrant)
        .await
        .map_err(|e| RagError::Storage(e.to_string()))?;

    let records = records_from_chunks(&build.chunks);
    let inserted = store
        .upsert_records(&records)
        .await
        .map_err(|e| RagError::Storage(e.to_string()))?;

    let server_count = store
        .count()
        .await
        .map_err(|e| RagError::Storage(e.to_string()))?;

    info!(
        backend = "qdrant",
        collection = %config.qdrant.collection,
        inserted,
        server_count,
        created,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Qdrant 索引写入完成"
    );

    Ok(IndexReport {
        backend: BackendKind::Qdrant,
        chunks: inserted,
        server_count,
        collection: config.qdrant.collection.clone(),
        dimension,
        embedding_model: model_name.to_string(),
        created,
        elapsed_ms: start.elapsed().as_millis() as u64,
    })
}

/// 确保索引就绪：加载语料 → 向量化 → 写入选定的后端。
pub async fn ensure_index(config: &AppConfig) -> RagResult<IndexReport> {
    let _guard = stage_guard("构建索引");
    let embedder = make_embedder(config.embedding_model)?;

    let index_config = IndexConfig {
        data_dir: config.data_dir.clone(),
        ..Default::default()
    };

    let build = build_chunks(&embedder, &index_config)?;

    match config.backend {
        BackendKind::Milvus => index_milvus(config, &build).await,
        BackendKind::Qdrant => index_qdrant(config, &build).await,
    }
}

/// 走一遍完整的「建索引 → 检索」验证（不需要 HTTP 服务）。
///
/// 这是 `cargo run -p rag-10 -- demo` 的核心：
/// 证明索引真的写进去了、检索真的能召回。
pub async fn smoke_test(config: &AppConfig) -> RagResult<(IndexReport, Vec<String>)> {
    let _guard = stage_guard("冒烟测试：建索引 + 检索");
    let embedder = make_embedder(config.embedding_model)?;

    let index_config = IndexConfig {
        data_dir: config.data_dir.clone(),
        ..Default::default()
    };

    let build = build_chunks(&embedder, &index_config)?;

    let (report, hits) = match config.backend {
        BackendKind::Milvus => {
            let report = index_milvus(config, &build).await?;
            let store =
                MilvusStore::connect(&config.milvus, report.dimension, &report.embedding_model)
                    .await?;
            let query = embedder
                .embed_query("向量数据库是什么")
                .map_err(|e| RagError::Internal(e.to_string()))?;
            let hits = store
                .search_dense(&query, config.top_k, None)
                .await
                .map_err(|e| RagError::Storage(e.to_string()))?;
            debug!(backend = "milvus", hit_count = hits.len(), "冒烟检索完成");
            (report, hits.into_iter().map(|h| h.summary()).collect())
        }
        BackendKind::Qdrant => {
            let report = index_qdrant(config, &build).await?;
            let store =
                QdrantStore::connect(&config.qdrant, report.dimension, &report.embedding_model)
                    .await?;
            let query = embedder
                .embed_query("向量数据库是什么")
                .map_err(|e| RagError::Internal(e.to_string()))?;
            let hits = store
                .search_dense(&query, config.top_k, None)
                .await
                .map_err(|e| RagError::Storage(e.to_string()))?;
            debug!(backend = "qdrant", hit_count = hits.len(), "冒烟检索完成");
            (report, hits.into_iter().map(|h| h.summary()).collect())
        }
    };

    Ok((report, hits))
}

/// 检查后端是否可达（健康检查）。
pub async fn check_health(config: &AppConfig) -> RagResult<String> {
    match config.backend {
        BackendKind::Milvus => {
            let store = MilvusStore::connect(
                &config.milvus,
                model_dimension(config.embedding_model),
                crate::config::model_name(config.embedding_model),
            )
            .await?;
            let health = store
                .client()
                .check_health(
                    CheckHealthRequest::builder()
                        .build()
                        .map_err(|e| RagError::Storage(e.to_string()))?,
                )
                .await
                .map_err(|e| RagError::Storage(e.to_string()))?;
            Ok(format!(
                "milvus healthy={} uri={}",
                health.is_healthy(),
                config.milvus.uri
            ))
        }
        BackendKind::Qdrant => {
            let store = QdrantStore::connect(
                &config.qdrant,
                model_dimension(config.embedding_model),
                crate::config::model_name(config.embedding_model),
            )
            .await?;
            let health = store
                .client()
                .health_check()
                .await
                .map_err(|e| RagError::Storage(e.to_string()))?;
            Ok(format!(
                "qdrant title={} url={}",
                health.title, config.qdrant.url
            ))
        }
    }
}

/// 让 `MilvusStore` 的 `VectorStore` 实现可用于泛型代码。
///
/// ⚠️ 注意：`VectorStore` 的方法用的是 RPITIT
/// （`impl Future<Output = ...> + Send`），所以它**不是 dyn 兼容的** ——
/// 不能写成 `&mut dyn VectorStore`。
/// 泛型代码应该写成 `fn f<S: VectorStore>(store: &mut S)`，
/// 或者直接用具体类型。
pub fn as_vector_store<S: VectorStore>(store: &mut S) -> &mut S {
    store
}

/// 让 `QdrantStore` 的 `VectorStore` 实现可用于泛型代码。
pub fn as_qdrant_vector_store<S: VectorStore>(store: &mut S) -> &mut S {
    store
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_is_absolute() {
        assert!(cache_dir().is_absolute());
    }

    #[test]
    fn report_summary_mentions_backend() {
        let report = IndexReport {
            backend: BackendKind::Milvus,
            chunks: 10,
            server_count: 10,
            collection: "c".to_string(),
            dimension: 384,
            embedding_model: "e5".to_string(),
            created: true,
            elapsed_ms: 5,
        };
        assert!(report.summary().contains("milvus"));
        assert!(report.summary().contains("新建"));
    }
}
