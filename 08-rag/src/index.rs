//! 索引层 —— 同时维护 **向量索引**（FAISS）和 **BM25 倒排索引**。
//!
//! ## 为什么两套索引要放在一起
//!
//! 混合检索需要**同一批块**的两种索引：一份能算余弦相似度，一份能算 BM25。
//! 两份索引的块 ID 必须一一对应，否则融合时对不上。
//!
//! ## 索引的生命周期
//!
//! ```text
//! 文档 ──Splitter──> 块 ──Embedder──> 向量 ──FAISS──> index.faiss
//!            │
//!            └──────────────────────> BM25 倒排索引（内存）
//! ```
//!
//! ## ⚠️ 一个重要的工程决策：BM25 索引**不落盘**
//!
//! BM25 索引只需要分块结果，**不需要嵌入模型** —— 分块是纯 CPU 的字符串操作，
//! 几百个块只要几毫秒。
//!
//! 所以启动时的流程是：
//!
//! ```text
//! 索引不存在 ──> 跑嵌入模型建 FAISS 索引 ──> 落盘
//! 索引存在   ──> 加载 FAISS 索引（几毫秒）+ 重建 BM25 索引（几毫秒）
//! ```
//!
//! 好处：
//!
//! 1. **不需要维护第二套持久化格式** —— 没有「两个索引版本不一致」的问题
//! 2. **BM25 索引永远和 FAISS 索引的块完全一致** —— 因为它们来自同一次分块
//! 3. **改文档后不用清理 BM25 缓存** —— 它每次启动都重建
//!
//! 代价：每次启动多花几毫秒。对教学项目完全可以接受。
//!
//! ⚠️ FAISS 索引加载时**校验嵌入模型名** —— 换模型没重建索引会在这里报错，
//! 而不是等到检索时返回一堆不相关的块。

use std::path::Path;

use rag_04::splitter::{Splitter, SplitterConfig, StructureAwareSplitter, TokenCounter};
use rag_05::embed::Embedder;
use rag_05::store::{FaissStore, IndexKind, VectorStore};

use crate::bm25::{Bm25Document, Bm25Index};
use crate::error::{RagError, RagResult};

/// 索引统计（复用 06/07 的类型，保持响应格式一致）。
pub use rag_06::index::IndexStats;

/// 建索引需要的参数。
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// 分块配置。
    pub splitter: SplitterConfig,

    /// FAISS 索引类型。默认 `Flat` —— 精确检索，教学场景当 ground truth 用。
    pub index_kind: IndexKind,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            splitter: SplitterConfig::default(),
            index_kind: IndexKind::Flat,
        }
    }
}

/// 确保索引存在：不存在就建，存在就加载。
///
/// 返回 `(FAISS store, BM25 索引, 统计, 这次是不是新建的)`。
///
/// ## 为什么 BM25 索引总是现建
///
/// 见模块文档 —— 它不需要模型，几毫秒就好，而且能保证和 FAISS 索引的块完全一致。
pub async fn ensure_index(
    config: &crate::config::AppConfig,
) -> RagResult<(FaissStore, Bm25Index, IndexStats, bool)> {
    let embedder = make_embedder(config.embedding_model.to_model())?;
    let options = IndexOptions {
        splitter: config.splitter.clone(),
        ..Default::default()
    };

    let (store, stats, created) = if config.index_path.exists() {
        let store = load_store(&embedder, &config.index_path).await?;
        let stats = IndexStats {
            document_count: 0,
            chunk_count: store.len(),
            dimension: embedder.dimension(),
            index_kind: store.index_description(),
            embedding_model: embedder.model_name().to_string(),
        };
        (store, stats, false)
    } else {
        if !config.auto_index {
            return Err(RagError::NotReady(format!(
                "索引不存在: {}，且未开启自动建索引（RAG_AUTO_INDEX=false）",
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
        (store, stats, true)
    };

    // BM25 索引：每次启动重建（不需要模型，几毫秒）。
    let bm25 = build_bm25(&config.data_dir, &options.splitter)?;

    Ok((store, bm25, stats, created))
}

/// 建 BM25 索引（**不需要嵌入模型**）。
///
/// 用的是和向量索引**完全相同**的分块配置 —— 这是两套索引能对齐的前提。
pub fn build_bm25(data_dir: &Path, splitter_config: &SplitterConfig) -> RagResult<Bm25Index> {
    let counter = TokenCounter::new()
        .map_err(|e| RagError::Internal(format!("创建 token 计数器失败: {e}")))?;
    let splitter = StructureAwareSplitter::new(splitter_config.clone());

    let documents = rag_04::loader::load_directory(data_dir)
        .map_err(|e| RagError::Internal(format!("加载文档失败: {e}")))?;

    let mut bm25_docs: Vec<Bm25Document> = Vec::new();

    for doc in &documents {
        for chunk in splitter.split(doc, &counter) {
            // ⚠️ 用 `text`（正文）而不是 `embedded_text`（含标题前缀）。
            //
            // 向量索引用 `embedded_text` —— 标题路径能帮语义检索。
            // 但 BM25 是**关键词**匹配：如果把「# 标题 > 子标题」也塞进去，
            // 标题词会主导打分，让「按标题命中」压过「按正文命中」。
            //
            // 实践中我们希望 BM25 找的是**正文里的关键词**。
            // 标题信息通过 `heading_path` 字段单独保留，供 Reranker 用。
            bm25_docs.push(Bm25Document {
                id: rag_05::embed::chunk_id(&chunk.source, chunk.index),
                source: chunk.source.clone(),
                heading_path: chunk.heading_path.clone(),
                text: chunk.text.clone(),
            });
        }
    }

    Ok(Bm25Index::build(&bm25_docs))
}

/// 建索引并落盘。
pub async fn build_and_save(
    embedder: &dyn Embedder,
    data_dir: &Path,
    index_path: &Path,
    options: &IndexOptions,
) -> RagResult<IndexStats> {
    let (store, stats) = build_store(embedder, data_dir, options).await?;

    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| RagError::Internal(format!("创建索引目录失败: {e}")))?;
    }

    store
        .save(index_path)
        .await
        .map_err(|e| RagError::Internal(format!("保存索引失败: {e}")))?;

    Ok(stats)
}

/// 建索引并返回 store（不落盘）。
pub async fn build_store(
    embedder: &dyn Embedder,
    data_dir: &Path,
    options: &IndexOptions,
) -> RagResult<(FaissStore, IndexStats)> {
    let config = rag_05::pipeline::IndexConfig {
        data_dir: data_dir.to_path_buf(),
        splitter: options.splitter.clone(),
        ..Default::default()
    };

    let build = rag_05::pipeline::build_chunks(embedder, &config)
        .map_err(|e| RagError::Internal(format!("构建索引失败: {e}")))?;

    let mut store = FaissStore::new(options.index_kind, embedder.dimension(), embedder.model_name())
        .map_err(|e| RagError::Internal(format!("创建 FAISS 索引失败: {e}")))?;

    store
        .add(&build.chunks)
        .await
        .map_err(|e| RagError::Internal(format!("写入索引失败: {e}")))?;

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
pub async fn load_store(embedder: &dyn Embedder, index_path: &Path) -> RagResult<FaissStore> {
    let mut store =
        FaissStore::new(IndexKind::Flat, embedder.dimension(), embedder.model_name())
            .map_err(|e| RagError::Internal(format!("创建 FAISS 索引失败: {e}")))?;

    store.load(index_path).await.map_err(|e| match e {
        rag_05::store::StoreError::Faiss(msg) => {
            RagError::Internal(format!("加载索引失败（可能是换了嵌入模型，需重建）: {msg}"))
        }
        other => RagError::Internal(format!("加载索引失败: {other}")),
    })?;

    Ok(store)
}

/// 创建嵌入器（复用 06 的实现，含统一的缓存目录处理）。
pub fn make_embedder(model: rag_05::embed::Model) -> RagResult<rag_05::embed::LocalEmbedder> {
    rag_06::index::make_embedder(model).map_err(|e| RagError::Internal(e.to_string()))
}

/// 缓存目录（绝对路径）。
///
/// 与 06/07 保持一致：优先 `FASTEMBED_CACHE_DIR`，否则用 workspace 根目录下的
/// `.fastembed_cache`。
///
/// ```text
/// cargo run  -p rag-08   → CWD = workspace 根 → 同一个缓存
/// cargo test -p rag-08   → CWD = 08-rag/     → 同一个缓存（靠绝对路径保证）
/// ```
pub fn cache_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("FASTEMBED_CACHE_DIR")
        && !dir.is_empty()
    {
        return std::path::PathBuf::from(dir);
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join(".fastembed_cache"))
        .unwrap_or_else(|| std::path::PathBuf::from(".fastembed_cache"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_is_absolute() {
        // 绝对路径是必须的 —— 否则从不同目录跑会重复下载模型。
        assert!(cache_dir().is_absolute());
    }

    #[test]
    fn bm25_index_is_built_without_embedding_model() {
        // BM25 不需要模型 —— 这正是每次启动重建它的理由。
        let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let index = build_bm25(&data_dir, &SplitterConfig::default()).unwrap();

        assert!(!index.is_empty(), "BM25 索引应该有文档");
        assert!(index.vocab_size() > 10);
    }
}
