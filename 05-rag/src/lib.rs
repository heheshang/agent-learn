//! 05-rag 的库入口 —— 把 Part 2（向量化 + 存储）暴露成可复用的库。
//!
//! 06-rag（Naive RAG 完整系统）会 `use rag_05::{embed, pipeline, store}` 复用这里的
//! `Embedder` / `LocalEmbedder` / `FaissStore` / `VectorStore`，
//! 把 04（加载 + 分割）和 05（向量化 + 存储）串成一条完整的 RAG 流水线。
//!
//! 二进制 demo 仍在 `main.rs` 里，它通过 `rag_05::` 路径访问这些模块。

pub mod embed;
pub mod pipeline;
pub mod store;

pub use embed::{EmbedError, EmbeddedChunk, Embedder, LocalEmbedder, Model, cosine_similarity};
pub use pipeline::{IndexBuild, IndexConfig, build_chunks, index_path};
pub use store::{ChromaStore, FaissStore, IndexKind, SearchHit, StoreError, VectorStore};
