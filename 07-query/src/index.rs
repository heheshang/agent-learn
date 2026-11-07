//! 索引层 —— 建索引 / 加载索引。
//!
//! ## 为什么不直接用 06 的 `ensure_index`
//!
//! 06 的 [`rag_06::index::ensure_index`] 接收的是 **06 自己的** `AppConfig`，
//! 而本项目有独立的 `AppConfig`（多了改写相关的配置）。
//!
//! 所以这里复用 06 的**底层函数**（`build_and_save` / `load_store` / `make_embedder`），
//! 自己包一层「确保索引存在」的高层逻辑。
//!
//! 这比在 06 里加一个泛型参数更干净 —— 06 保持自己的完整性，
//! 07 通过组合复用它的能力。
//!
//! ## 索引的生命周期
//!
//! ```text
//! 索引不存在 ──> build_and_save ──> 落盘
//! 索引存在   ──> load_store    ──> 校验模型名
//! ```
//!
//! ⚠️ 加载时会校验嵌入模型名 —— 换模型没重建索引会在这里报错，
//! 而不是等到检索时返回一堆不相关的块。

use std::path::Path;

use rag_05::embed::Embedder;
use rag_05::store::{FaissStore, VectorStore};

use crate::config::AppConfig;
use crate::error::{QueryError, QueryResult};

/// 索引统计（复用 06 的类型，保持两个模块的响应格式一致）。
pub use rag_06::index::IndexStats;

/// 建索引需要的参数。
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// 分块配置。
    pub splitter: rag_04::splitter::SplitterConfig,

    /// FAISS 索引类型。默认 `Flat` —— 精确检索，教学场景当 ground truth 用。
    pub index_kind: rag_05::store::IndexKind,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            splitter: rag_04::splitter::SplitterConfig::default(),
            index_kind: rag_05::store::IndexKind::Flat,
        }
    }
}

/// 确保索引存在：不存在就建，存在就加载。
///
/// 返回的布尔值表示**这次是不是新建了索引**（用于日志）。
pub async fn ensure_index(config: &AppConfig) -> QueryResult<(FaissStore, IndexStats, bool)> {
    let embedder = make_embedder(config.embedding_model.to_model())?;
    let options = IndexOptions::default();

    if config.index_path.exists() {
        let store = load_store(&embedder, &config.index_path).await?;
        let stats = IndexStats {
            document_count: 0,
            chunk_count: store.len(),
            dimension: embedder.dimension(),
            index_kind: store.index_description(),
            embedding_model: embedder.model_name().to_string(),
        };
        return Ok((store, stats, false));
    }

    if !config.auto_index {
        return Err(QueryError::NotReady(format!(
            "索引不存在: {}，且未开启自动建索引（QUERY_AUTO_INDEX=false）",
            config.index_path.display()
        )));
    }

    let stats = build_and_save(
        &embedder,
        &config.data_dir,
        &config.index_path,
        &options,
    )
    .await?;
    let store = load_store(&embedder, &config.index_path).await?;

    Ok((store, stats, true))
}

/// 建索引并落盘。
pub async fn build_and_save(
    embedder: &dyn Embedder,
    data_dir: &Path,
    index_path: &Path,
    options: &IndexOptions,
) -> QueryResult<IndexStats> {
    let (store, stats) = build_store(embedder, data_dir, options).await?;

    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| QueryError::Internal(format!("创建索引目录失败: {e}")))?;
    }

    store
        .save(index_path)
        .await
        .map_err(|e| QueryError::Internal(format!("保存索引失败: {e}")))?;

    Ok(stats)
}

/// 建索引并返回 store（不落盘）。
pub async fn build_store(
    embedder: &dyn Embedder,
    data_dir: &Path,
    options: &IndexOptions,
) -> QueryResult<(FaissStore, IndexStats)> {
    let config = rag_05::pipeline::IndexConfig {
        data_dir: data_dir.to_path_buf(),
        splitter: options.splitter.clone(),
        ..Default::default()
    };

    let build = rag_05::pipeline::build_chunks(embedder, &config)
        .map_err(|e| QueryError::Internal(format!("构建索引失败: {e}")))?;

    let mut store = FaissStore::new(options.index_kind, embedder.dimension(), embedder.model_name())
        .map_err(|e| QueryError::Internal(format!("创建 FAISS 索引失败: {e}")))?;

    store
        .add(&build.chunks)
        .await
        .map_err(|e| QueryError::Internal(format!("写入索引失败: {e}")))?;

    let stats = IndexStats {
        document_count: build.documents.len(),
        chunk_count: build.chunks.len(),
        dimension: embedder.dimension(),
        index_kind: options.index_kind.description(),
        embedding_model: embedder.model_name().to_string(),
    };

    Ok((store, stats))
}

/// 从磁盘加载索引（会校验嵌入模型名）。
pub async fn load_store(
    embedder: &dyn Embedder,
    index_path: &Path,
) -> QueryResult<FaissStore> {
    let mut store =
        FaissStore::new(rag_05::store::IndexKind::Flat, embedder.dimension(), embedder.model_name())
            .map_err(|e| QueryError::Internal(format!("创建 FAISS 索引失败: {e}")))?;

    store.load(index_path).await.map_err(|e| match e {
        rag_05::store::StoreError::Faiss(msg) => {
            QueryError::Internal(format!("加载索引失败（可能是换了嵌入模型，需重建）: {msg}"))
        }
        other => QueryError::Internal(format!("加载索引失败: {other}")),
    })?;

    Ok(store)
}

/// 创建嵌入器（复用 06 的实现，含统一的缓存目录处理）。
pub fn make_embedder(model: rag_05::embed::Model) -> QueryResult<rag_05::embed::LocalEmbedder> {
    rag_06::index::make_embedder(model).map_err(|e| QueryError::Internal(e.to_string()))
}
