//! 10-rag-milvus —— **生产级向量数据库（Milvus）**。
//!
//! 09-rag-eval 把「检索准不准」量化了，但那套系统的索引是
//! **FAISS（进程内）+ Chroma（HTTP）** —— 都还是「单体玩具」级别。
//! 真实生产里的向量数据库要面对的是：亿级向量、水平分片、
//! 多副本、持久化、滚动升级、权限管控。
//!
//! ## 这一层要解决的核心问题
//!
//! | 问题 | 单体索引（FAISS）怎么做 | 生产向量库怎么做 |
//! |---|---|---|
//! | 数据存哪 | 进程内存 / 一个 `.faiss` 文件 | **对象存储**（MinIO / S3），分 segment 落盘 |
//! | 元数据存哪 | Rust 侧 sidecar JSON | **事务型 KV**（etcd），保证 collection schema 一致 |
//! | 水平扩展 | 改 FAISS 参数（加内存） | **分片 + 副本 + 路由层** |
//! | 向量检索 | 直接算距离 | **索引构建器**（HNSW / IVF / DISKANN / SCANN…） |
//! | 混合检索 | 自己写 BM25（08 做过） | **服务端 BM25 函数**（稀疏向量 + 倒排索引） |
//! | 标量过滤 | 检索回来自己过滤 | **服务端过滤**（标量索引 + segment pruning） |
//! | 在线扩字段 | 重建整个索引 | `add_collection_field`（在线加字段） |
//!
//! ## Milvus 的架构（这是本项目真正的学习目标）
//!
//! ```text
//!   你的应用
//!      │  gRPC (19530)
//!      ▼
//!  ┌──────────────────────────────────────────────────────┐
//!  │  Proxy ── 入口：解析请求、路由到目标分片、聚合结果      │
//!  └───────────────┬──────────────────────────────────────┘
//!                  │
//!      ┌───────────┴────────────┐
//!      ▼                        ▼
//!  ┌──────────┐            ┌──────────┐
//!  │ QueryNode│  查数据     │ DataNode│  写数据 / 建索引      │
//!  └──────────┘            └──────────┘
//!      │                        │
//!      ▼                        ▼
//!   MinIO/S3                MinIO/S3        ← 向量数据 + 索引文件
//!   （对象存储）              （对象存储）
//!      ▲
//!      │
//!    etcd                    ← collection schema / segment 元信息
//!  （元数据）                                        │
//!                                                   ▼
//!                                              Coordinator
//!                                      （建索引 / compaction / flush）
//! ```
//!
//! - **QueryNode**：只读。加载 segment 到内存（含索引），执行向量检索。
//! - **DataNode**：只写。接收写入请求，落到 WAL（etcd 里的日志），
//!   攒够后 flush 成 segment 写进对象存储，并触发索引构建。
//! - **Coordinator**：集群大脑。调度建索引任务、触发 compaction、监控健康。
//! - **Proxy**：无状态入口。客户端只跟它说话，它负责把请求路由到
//!   对应的 QueryNode，并把多个分片的结果归并（top-k merge）。
//!
//! **为什么 standalone 也要这么多组件？**
//! 因为 standalone 只是把上面所有角色塞进一个进程 ——
//! 架构是一样的，只是没有网络分片。学过 standalone 再看分布式，
//! 就只是「每个角色从 1 个变成 N 个」。
//!
//! ## 本项目的三个产出
//!
//! ### 1. `MilvusStore` —— 实现 `VectorStore` trait
//!
//! 05-rag 定义了 [`VectorStore`](rag_05::store::VectorStore) trait，FAISS / Chroma 都实现了它。
//! 本项目让 **Milvus** 也实现同一个 trait，于是「换向量数据库」
//! 变成一次**可运行、可对比**的实验 —— 而不是纸上谈兵。
//!
//! ### 2. 服务端混合检索
//!
//! 08-rag 的 BM25 是自己在 Rust 里写的倒排索引；
//! Milvus 把 BM25 做成了**服务端函数**：
//!
//! ```text
//! text 字段 (VarChar, enable_analyzer=true)
//!        │  服务端 BM25 函数
//!        ▼
//! sparse 字段 (SparseFloatVector)
//!        │  SPARSE_INVERTED_INDEX
//!        ▼
//!   可全文检索的稀疏向量
//! ```
//!
//! 于是「稠密向量 + BM25」可以在**一次 hybrid_search** 里完成，
//! 融合由 Milvus 的 `RRFRerank` / `WeightedRerank` 完成。
//!
//! ### 3. Milvus ↔ Qdrant 同题对照
//!
//! Qdrant 是 Milvus 最直接的替代品（同样 gRPC + HNSW + payload 过滤）。
//! 本项目把**同一份语料、同一套嵌入模型、同一批查询**分别灌进两者，
//! 用同一套指标对比 —— 这才是选型该有的做法。
//!
//! ## 部署
//!
//! ```bash
//! cd 10-rag-milvus/docker
//! docker compose up -d
//! ```
//!
//! 详见 [`docker/docker-compose.yml`](../../docker/docker-compose.yml) 的注释。

pub mod api;
pub mod config;
pub mod error;
pub mod index;
pub mod milvus_store;
pub mod qdrant_store;
pub mod schema;

pub use config::{AppConfig, BackendKind, MilvusConfig, QdrantConfig};
pub use error::{RagError, RagResult};
pub use index::{IndexReport, ensure_index, smoke_test};
pub use milvus_store::{MilvusSchema, MilvusStore};
pub use qdrant_store::QdrantStore;
pub use schema::{ChunkRecord, chunk_to_record, records_from_chunks};
