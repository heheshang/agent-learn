//! 索引流水线 —— 把 Part 1 的块变成可检索的向量索引。
//!
//! ```text
//! 文档 ──Loader──> 文本 ──Splitter──> 块 ──Embedder──> 向量 ──Store──> 索引
//! ```
//!
//! 这一层是 Part 1 和 Part 2 的接缝：Part 1 产出 `Chunk`，
//! Part 2 把 `Chunk` 变成 `EmbeddedChunk` 存进索引。

use std::path::{Path, PathBuf};

use anyhow::Result;
use rag_04::loader::{LoadedDoc, load_directory};
use rag_04::splitter::{Chunk, Splitter, SplitterConfig, StructureAwareSplitter, TokenCounter};

use crate::embed::{EmbeddedChunk, Embedder};
use crate::store::VectorStore;

/// 索引配置。
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// 文档目录。
    pub data_dir: PathBuf,

    /// 索引目录。`None` 时取 `data_dir` 的同级 `index/`。
    pub index_dir: Option<PathBuf>,

    /// 分块配置（复用 Part 1 的）。
    pub splitter: SplitterConfig,

    /// 索引文件名（FAISS 用）。
    pub index_name: String,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            index_dir: None,
            splitter: SplitterConfig::default(),
            index_name: "index.faiss".to_string(),
        }
    }
}

/// 索引构建结果。
#[derive(Debug)]
pub struct IndexBuild {
    /// 向量化后的块。
    pub chunks: Vec<EmbeddedChunk>,
    /// 加载到的文档。
    pub documents: Vec<LoadedDoc>,
    /// 原始块（未向量化），用于对比「送去做嵌入的文本 vs 原文」。
    #[allow(dead_code)]
    pub raw_chunks: Vec<Chunk>,
}

/// 加载 → 分割 → 向量化。
///
/// 这一步只产出 `EmbeddedChunk`，**不写索引** ——
/// 写索引交给具体的 store（FAISS / Chroma）。
pub fn build_chunks(embedder: &dyn Embedder, config: &IndexConfig) -> Result<IndexBuild> {
    let counter = TokenCounter::new()?;
    let splitter = StructureAwareSplitter::new(config.splitter.clone());

    let documents = load_directory(&config.data_dir)?;

    let mut raw_chunks = Vec::new();
    for doc in &documents {
        raw_chunks.extend(splitter.split(doc, &counter));
    }

    let chunks = embedder.embed_chunks(&raw_chunks)?;

    Ok(IndexBuild {
        chunks,
        documents,
        raw_chunks,
    })
}

/// 索引文件的完整路径。
///
/// 索引目录和 `data_dir` 平级 —— 数据在 `data/`，索引在 `index/`，
/// 两者都在同一个项目目录下。
pub fn index_path(config: &IndexConfig) -> PathBuf {
    config.index_dir().join(&config.index_name)
}

impl IndexConfig {
    /// 索引目录（默认取 `data_dir` 的同级 `index/`）。
    pub fn index_dir(&self) -> PathBuf {
        self.index_dir.clone().unwrap_or_else(|| {
            self.data_dir
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("index")
        })
    }
}

/// 把构建好的块写进 store。
#[allow(dead_code)]
pub async fn index_into_store<S: VectorStore>(store: &mut S, build: &IndexBuild) -> Result<()> {
    store.add(&build.chunks).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_path_sits_next_to_data_dir() {
        let config = IndexConfig {
            data_dir: PathBuf::from("data"),
            index_name: "index.faiss".to_string(),
            ..Default::default()
        };
        assert_eq!(
            index_path(&config),
            PathBuf::from("index").join("index.faiss")
        );
    }

    #[test]
    fn explicit_index_dir_wins() {
        let config = IndexConfig {
            data_dir: PathBuf::from("data"),
            index_dir: Some(PathBuf::from("/tmp/my-index")),
            index_name: "idx.faiss".to_string(),
            ..Default::default()
        };
        assert_eq!(
            index_path(&config),
            PathBuf::from("/tmp/my-index/idx.faiss")
        );
    }
}
