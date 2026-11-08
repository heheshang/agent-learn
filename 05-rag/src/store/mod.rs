//! 向量存储层（Vector Stores）—— 存向量、查相似。
//!
//! 这是 RAG 流水线的第四步：
//!
//! ```text
//! 向量 ──Store──> 索引 ──search(query)──> 命中的块
//! ```
//!
//! ## 两种存储的定位
//!
//! | | [FAISS](faiss_store) | [Chroma](chroma_store) |
//! |---|---|---|
//! | 形态 | 进程内 C++ 库 + 文件 | HTTP 服务 + 持久化目录 |
//! | 检索 | 纯向量最近邻 | 向量检索 + 元数据过滤 |
//! | 元数据 | 存在 Rust 侧（sidecar JSON） | 存在数据库里，可服务端过滤 |
//! | 部署 | 嵌进你的二进制 | 独立进程，可多客户端共享 |
//! | 适合 | 单机、大批量、只做向量检索 | 需要过滤、需要跨服务共享 |
//!
//! ## 共同的接口
//!
//! [`VectorStore`] trait 把两者抽象成同一套操作：
//! `add` / `search` / `len` / `save` / `load`。
//! 检索层只依赖这个 trait，所以从 FAISS 换到 Chroma 不用改上层代码。
//!
//! ## 为什么 trait 是异步的
//!
//! Chroma 是 HTTP 服务，天然异步；FAISS 是进程内调用，同步即可。
//! 为了让两者**真正可互换**，trait 的方法统一声明为 `async fn` ——
//! FAISS 的实现里用立即就绪的 future 包住同步逻辑，调用方写法一致。
//!
//! ## 为什么要额外存元数据
//!
//! FAISS 只返回**整数 ID**，不返回文本。但 RAG 最终要拿到原文去喂 LLM，
//! 所以每条向量都必须配一份元数据（来源文件、标题路径、正文）。
//! 这是本项目里 `EmbeddedChunk` 存在的原因。

pub mod chroma_store;
pub mod faiss_store;

use std::future::Future;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::embed::EmbeddedChunk;

pub use chroma_store::ChromaStore;
pub use faiss_store::{FaissStore, IndexKind};

/// 存储层错误。
#[derive(Debug, Error)]
pub enum StoreError {
    /// FAISS 操作失败。
    #[error("FAISS 操作失败: {0}")]
    Faiss(String),

    /// Chroma 操作失败。
    #[error("Chroma 操作失败: {0}")]
    Chroma(String),

    /// Milvus 操作失败。
    ///
    /// 加这个变体是为了让 10-rag-milvus 能把 Milvus 实现成同一个
    /// `VectorStore` trait —— 上层代码不需要知道底下是哪个库。
    #[error("Milvus 操作失败: {0}")]
    Milvus(String),

    /// Qdrant 操作失败。
    #[error("Qdrant 操作失败: {0}")]
    Qdrant(String),

    /// 索引文件读写失败。
    #[error("索引文件 IO 失败: {0}")]
    Io(#[from] std::io::Error),

    /// 索引文件里的 JSON 解析失败。
    #[error("索引元数据解析失败: {0}")]
    Json(#[from] serde_json::Error),

    /// 向量维度与索引不匹配。
    #[error("向量维度与索引不匹配: 索引 {expected}, 查询 {actual}")]
    DimensionMismatch {
        /// 索引的维度。
        expected: usize,
        /// 实际查询向量的维度。
        actual: usize,
    },

    /// 索引为空。
    #[error("索引为空，无法检索")]
    EmptyIndex,
}

/// 一条检索命中。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// 命中的块 ID（`{文件名}#{序号}`）。
    pub id: String,

    /// 相似度分数。越大越相似（余弦相似度，范围 `[-1, 1]`）。
    pub score: f32,

    /// 来源文件名。
    pub source: String,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 块正文。
    pub text: String,
}

impl SearchHit {
    /// 一行摘要。
    pub fn summary(&self) -> String {
        let head = if self.heading_path.is_empty() {
            "(无标题)".to_string()
        } else {
            self.heading_path.join(" > ")
        };
        let preview: String = self.text.chars().take(50).collect();
        format!(
            "{:.4} {} [{}] {}",
            self.score,
            self.id,
            head,
            preview.replace('\n', " ")
        )
    }
}

/// 向量存储 trait。
///
/// 上层（检索、评估、生成）只依赖这个 trait，
/// 于是 FAISS ↔ Chroma 可以自由替换。
///
/// 方法是异步的：Chroma 要发 HTTP 请求，FAISS 的实现则返回立即就绪的 future。
///
/// 返回值写成 `impl Future<Output = ...> + Send` 而不是 `async fn` ——
/// 后者虽然更简洁，但无法在 trait 里标注 `Send` 约束，
/// 而调用方（tokio 任务）需要这个保证。
pub trait VectorStore: Send + Sync {
    /// 存储的名字（用于展示）。
    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// 写入索引的模型名 —— **换了模型必须重建索引**。
    fn model_name(&self) -> &str;

    /// 向量维度。
    #[allow(dead_code)]
    fn dimension(&self) -> usize;

    /// 当前索引里的条数。
    fn len(&self) -> usize;

    /// 是否为空。
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 写入一批已向量化的记录（追加语义，重复 ID 会被跳过）。
    fn add(
        &mut self,
        chunks: &[EmbeddedChunk],
    ) -> impl Future<Output = Result<(), StoreError>> + Send;

    /// 检索：返回最相似的 `k` 条。
    ///
    /// `query` 必须是 L2 归一化后的向量 ——
    /// 这样内积就等于余弦相似度，分数方向也天然一致（越大越相似）。
    fn search(
        &mut self,
        query: &[f32],
        k: usize,
    ) -> impl Future<Output = Result<Vec<SearchHit>, StoreError>> + Send;

    /// 持久化到磁盘。
    fn save(&self, path: &Path) -> impl Future<Output = Result<(), StoreError>> + Send;

    /// 从磁盘加载。
    fn load(&mut self, path: &Path) -> impl Future<Output = Result<(), StoreError>> + Send;
}

/// 从 `EmbeddedChunk` 里取出一段连续向量（FAISS 要的是扁平数组）。
pub(crate) fn flatten(vectors: &[Vec<f32>]) -> Vec<f32> {
    vectors.iter().flat_map(|v| v.iter().copied()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_concatenates_rows() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(flatten(&rows), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn flatten_of_empty_is_empty() {
        assert!(flatten(&[]).is_empty());
    }
}
