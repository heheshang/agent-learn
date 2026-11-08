//! # 12-rag-summary —— 摘要式 RAG（混合检索 + Reranker + Milvus）
//!
//! 把前面 11 个项目的能力**集成**成一个生产形态的服务：
//!
//! | 来源项目 | 复用的能力 |
//! |---|---|
//! | 04-rag | 文档加载 + 结构感知分块 |
//! | 05-rag | 嵌入模型 + 向量存储抽象 |
//! | 06-rag | Prompt 组装 |
//! | 07-query | 查询改写（多问法）+ LLM 抽象 + RRF 常数 |
//! | 08-rag | BM25 索引 + 交叉编码器 Reranker |
//! | 09-rag-eval | 评估集 + 指标注册表 + 对比报告 |
//! | 10-rag-milvus | Milvus 存储（服务端混合检索） |
//!
//! ## 数据流
//!
//! ```text
//! 问题
//!   │
//!   ├─> [可选] 多问法改写 ──> {q1, q2, q3}
//!   │
//!   ├─> 混合检索 ──> dense(HNSW) + BM25(倒排) ──> RRF 融合 ──> 候选池
//!   │                            │
//!   │                            └─> 服务端 RRF（一次 RPC）
//!   │
//!   ├─> 交叉编码器重排 ──> top_n
//!   │
//!   ├─> 上下文压缩（token 预算）
//!   │
//!   └─> Prompt 组装 ──> LLM ──> 带引用的摘要
//! ```
//!
//! ## 消融阶梯
//!
//! ```text
//! dense ──> keyword ──> hybrid ──> hybrid_rerank ──> full
//! ```
//!
//! 每一档比上一档多一个增强，所以相邻差值**可归因**。
//!
//! ## 降级链
//!
//! | 环节失败 | 降级行为 |
//! |---|---|
//! | 查询改写失败 | 退回原问题 |
//! | 服务端混合检索失败 | 退回客户端双路融合 |
//! | 重排失败 | 退回融合后的原始顺序 |
//! | 上下文压缩失败 | 退回未压缩片段 |
//! | 检索彻底失败 | 报错 |

pub mod api;
pub mod bootstrap;
pub mod compress;
pub mod config;
pub mod error;
pub mod eval;
pub mod pipeline;
pub mod retriever;
pub mod table;

pub use config::{AppConfig, SearchMode};
pub use error::{RagError, RagResult};
pub use pipeline::{AskOutcome, Citation, RetrievalOutcome, SummaryPipeline};
pub use retriever::{Hit, HybridRetriever, RetrievalPath};
