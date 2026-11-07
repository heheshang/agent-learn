//! 04-rag 的库入口 —— 把 Part 1（加载 + 分割）暴露成可复用的库。
//!
//! 05-rag（RAG Part 2）会 `use rag_04::{loader, splitter}` 复用这里的
//! `LoadedDoc` / `Chunk` / `StructureAwareSplitter`，避免把分块逻辑抄一遍。
//!
//! 二进制 demo 仍在 `main.rs` 里，两者共享同一套模块。

pub mod loader;
pub mod splitter;

pub use loader::{DocumentLoader, LoadedDoc, LoaderError};
pub use splitter::{Chunk, Splitter, SplitterConfig, StructureAwareSplitter, TokenCounter};
