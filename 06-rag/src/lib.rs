//! 06-rag —— 手撕 Naive RAG 系统。
//!
//! 这一部分把前面 04（加载 + 分割）、05（向量化 + 存储）接上
//! **检索（Retrieval）** 和 **生成（Generation）**，形成一个完整的问答系统：
//!
//! ```text
//! 离线（索引阶段）
//!   文档 ──Loader──> 文本 ──Splitter──> 块 ──Embedder──> 向量 ──Store──> 索引
//!
//! 在线（查询阶段）
//!   问题 ──Embedder──> 查询向量 ──Retriever(top-k)──> 命中块
//!                                                        │
//!                                       Prompt 组装 ──────┘
//!                                            │
//!                                            └──> LLM ──> 带引用的答案
//! ```
//!
//! ## 模块划分
//!
//! | 模块 | 职责 | 对应概念 |
//! |---|---|---|
//! | [`config`] | 全局配置（模型、端口、检索参数） | — |
//! | [`index`] | 索引流水线：文档 → 块 → 向量 → 索引 | Semantic Search / Knowledge Base |
//! | [`retrieval`] | 检索器：问题 → top-k 命中块 | Retrieval 架构 |
//! | [`prompt`] | Prompt 组装：命中块 → 上下文 | Naive RAG 的「拼装」环节 |
//! | [`generation`] | 生成器：上下文 + 问题 → 带引用的答案 | Generator LLM |
//! | [`agent`] | 自主 Agent：工具调用循环 | LLM Powered Autonomous Agents |
//! | [`api`] | HTTP 接口层 | 文档问答 API |
//!
//! ## 「Naive」的边界
//!
//! 本项目**刻意**只实现最朴素的那条路径 —— 没有查询改写、没有混合检索、
//! 没有重排、没有元数据过滤、没有「检索不到就拒答」的阈值。
//!
//! 这不是遗漏，而是**教学基线**：先把这条链路跑通、跑对，
//! 之后每加一个增强手段，都能用「加之前 vs 加之后」的实测差值来说明它值不值。
//! 具体的增强手段见 [`agent`] 模块和 README 的「Naive 的边界」一节。

pub mod agent;
pub mod api;
pub mod config;
pub mod error;
pub mod generation;
pub mod index;
pub mod prompt;
pub mod retrieval;

pub use config::AppConfig;
pub use error::{RagError, RagResult};
