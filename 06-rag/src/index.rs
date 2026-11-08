//! 索引层（Knowledge Base）—— 把文档变成可检索的知识库。
//!
//! 这一层是 04（加载 + 分割）和 05（向量化 + 存储）的编排：
//!
//! ```text
//! 目录 ──load_directory──> LoadedDoc ──StructureAwareSplitter──> Chunk
//!       ──Embedder──> EmbeddedChunk ──FaissStore──> 索引
//! ```
//!
//! ## 为什么索引要单独一层
//!
//! 因为**建索引是离线操作，检索是在线操作**，两者的成本模型完全不同：
//!
//! - 建索引要跑嵌入模型（CPU 密集、要下载模型），一次几分钟
//! - 检索只做一次向量查找，毫秒级
//!
//! 把它们混在一起，每次问问题都要重新嵌入整个知识库 —— 这在生产里是不可接受的。
//! 所以：**建一次索引 → 存盘 → 后续只加载 + 检索**。
//!
//! ## 索引的产物
//!
//! 存盘后是两个文件（FAISS 的机制，见 05-rag）：
//!
//! ```text
//! index.faiss             # 向量索引
//! index.faiss.meta.json   # sidecar 元数据（原文、标题路径、来源）
//! ```
//!
//! ⚠️ 换嵌入模型必须重建索引 —— 元数据里记了模型名，加载时会校验，
//! 不一致直接报错，不会静默返回垃圾结果。

use std::path::{Path, PathBuf};

use agent_logging::{info, warn};
use rag_04::splitter::SplitterConfig;
use rag_05::embed::{Embedder, LocalEmbedder, Model};
use rag_05::store::{FaissStore, IndexKind, VectorStore};

use crate::config::AppConfig;
use crate::error::{RagError, RagResult};

/// 索引的构建结果。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexStats {
    /// 加载到的文档数。
    pub document_count: usize,

    /// 产生的块数。
    pub chunk_count: usize,

    /// 向量维度。
    pub dimension: usize,

    /// 索引类型描述串。
    pub index_kind: String,

    /// 使用的嵌入模型名。
    pub embedding_model: String,
}

impl IndexStats {
    /// 一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "{} 篇文档 → {} 个块 | {} 维 | {} | {}",
            self.document_count,
            self.chunk_count,
            self.dimension,
            self.index_kind,
            self.embedding_model
        )
    }
}

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

/// 建索引并返回 store（不落盘）。
///
/// 调用方负责决定要不要 `store.save(...)`。
pub async fn build_store(
    embedder: &dyn Embedder,
    data_dir: &Path,
    options: &IndexOptions,
) -> RagResult<(FaissStore, IndexStats)> {
    // 用 05-rag 的流水线做「加载 → 分割 → 向量化」，
    // 它已经处理好了标题路径注入、token 精确计数这些细节。
    let config = rag_05::pipeline::IndexConfig {
        data_dir: data_dir.to_path_buf(),
        splitter: options.splitter.clone(),
        ..Default::default()
    };

    let build = rag_05::pipeline::build_chunks(embedder, &config)
        .map_err(|e| RagError::Internal(format!("构建索引失败: {e}")))?;

    let mut store = FaissStore::new(
        options.index_kind,
        embedder.dimension(),
        embedder.model_name(),
    )
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

    info!(
        document_count = stats.document_count,
        chunk_count = stats.chunk_count,
        dimension = stats.dimension,
        index_kind = %stats.index_kind,
        model = %stats.embedding_model,
        "索引构建完成"
    );

    Ok((store, stats))
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

    info!(path = %index_path.display(), "索引已落盘");
    Ok(stats)
}

/// 从磁盘加载索引。
///
/// ⚠️ 会校验嵌入模型名 —— 换模型没重建索引会在这里报错，
/// 而不是等到检索时返回一堆不相关的块。
pub async fn load_store(embedder: &dyn Embedder, index_path: &Path) -> RagResult<FaissStore> {
    let mut store = FaissStore::new(IndexKind::Flat, embedder.dimension(), embedder.model_name())
        .map_err(|e| RagError::Internal(format!("创建 FAISS 索引失败: {e}")))?;

    store.load(index_path).await.map_err(|e| match e {
        rag_05::store::StoreError::Faiss(msg) => {
            warn!(path = %index_path.display(), error = %msg, "索引加载失败（多半是换了嵌入模型）");
            RagError::Internal(format!("加载索引失败（可能是换了嵌入模型，需重建）: {msg}"))
        }
        other => {
            warn!(path = %index_path.display(), error = %other, "索引加载失败");
            RagError::Internal(format!("加载索引失败: {other}"))
        }
    })?;

    Ok(store)
}

/// 创建嵌入器。
///
/// ## 缓存目录
///
/// fastembed 的默认缓存目录是**相对路径** `.fastembed_cache`，
/// 相对于**当前工作目录**。这在 workspace 里是个坑：
///
/// ```text
/// cargo run  -p rag-06   → CWD = workspace 根 → 缓存到 <root>/.fastembed_cache
/// cargo test -p rag-06   → CWD = 06-rag/    → 缓存到 06-rag/.fastembed_cache  ← 重新下载 470MB！
/// ```
///
/// 所以这里显式传一个**绝对路径**：优先用 `FASTEMBED_CACHE_DIR` 环境变量，
/// 否则用 workspace 根目录下的 `.fastembed_cache`。
/// 这样无论从哪个目录跑，模型都只需要下载一次。
pub fn make_embedder(model: Model) -> RagResult<LocalEmbedder> {
    LocalEmbedder::new(model, Some(cache_dir())).map_err(|e| {
        RagError::Internal(format!(
            "嵌入模型初始化失败（首次使用需要联网下载模型）: {e}"
        ))
    })
}

/// 嵌入模型的缓存目录（绝对路径）。
///
/// 1. `FASTEMBED_CACHE_DIR` 环境变量（用户显式指定，最高优先级）
/// 2. workspace 根目录下的 `.fastembed_cache`
fn cache_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("FASTEMBED_CACHE_DIR")
        && !dir.is_empty()
    {
        return std::path::PathBuf::from(dir);
    }

    // `CARGO_MANIFEST_DIR` = <workspace>/06-rag，往上一级就是 workspace 根。
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join(".fastembed_cache"))
        .unwrap_or_else(|| PathBuf::from(".fastembed_cache"))
}

/// 确保索引存在：不存在就建，存在就加载。
///
/// 这是服务启动时的标准动作 —— 让「第一次跑」和「第二次跑」都能直接工作。
///
/// 返回的布尔值表示**这次是不是新建了索引**（用于日志）。
pub async fn ensure_index(config: &AppConfig) -> RagResult<(FaissStore, IndexStats, bool)> {
    let embedder = make_embedder(config.embedding_model.to_model())?;
    let options = IndexOptions {
        splitter: config.splitter.clone(),
        ..Default::default()
    };

    if config.index_path.exists() {
        let store = load_store(&embedder, &config.index_path).await?;
        let stats = IndexStats {
            document_count: 0,
            chunk_count: store.len(),
            dimension: embedder.dimension(),
            index_kind: store.index_description(),
            embedding_model: embedder.model_name().to_string(),
        };
        info!(
            path = %config.index_path.display(),
            chunk_count = stats.chunk_count,
            "索引已存在，直接加载"
        );
        return Ok((store, stats, false));
    }

    if !config.auto_index {
        warn!(
            path = %config.index_path.display(),
            "索引不存在且未开启自动建索引"
        );
        return Err(RagError::NotReady(format!(
            "索引不存在: {}，且未开启自动建索引（RAG_AUTO_INDEX=false）",
            config.index_path.display()
        )));
    }

    info!("索引不存在，开始自动构建（首次会下载嵌入模型，耗时较长）");
    let stats = build_and_save(&embedder, &config.data_dir, &config.index_path, &options).await?;
    let store = load_store(&embedder, &config.index_path).await?;

    Ok((store, stats, true))
}
