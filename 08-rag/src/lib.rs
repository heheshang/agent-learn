//! 08-rag —— 手撕 **混合检索（Hybrid Search）+ 重排（Rerank）** 层。
//!
//! 07-query 把完整的 RAG 链路跑通，并加上了查询改写。
//! 但它的检索**仍然只有一路**：向量检索。
//!
//! ## 这一层要解决的三个问题
//!
//! | 问题 | 后果 | 本项目的解法 |
//! |---|---|---|
//! | 向量检索对**精确关键词**不敏感 | 问「`ECONNREFUSED` 错误码」召不回 | 加一路 BM25 |
//! | 两路的**分数不可比** | 加权求和怎么调都不对 | 用 RRF（只看排名） |
//! | 向量检索的**排序不够准** | top-10 里真正相关的排在第 8 | 加 cross-encoder 重排 |
//!
//! ## 整体链路
//!
//! ```text
//!                    ┌───────────────────────────────┐
//!   用户问题 ───────> │  BM25 倒排索引（关键词）      │
//!                    │  FAISS 向量索引（语义）        │
//!                    └──────────────┬────────────────┘//!
//!                                   │ 两路各召回 N 条
//!                    ┌──────────────▼───────────────┐
//!                    │  RRF 融合（只看排名）           │
//!                    └──────────────┬───────────────┘
//!                                   │ 候选集（top-20）
//!                    ┌──────────────▼───────────────┐
//!                    │  Cross-encoder 重排（精排）    │
//!                    └──────────────┬───────────────┘
//!                                   │ top-n
//!                    ┌──────────────▼───────────────┐
//!                    │  Prompt 组装 ──> LLM ──> 答案  │
//!                    └──────────────────────────────┘
//! ```
//!
//! ## 模块划分
//!
//! | 模块 | 职责 | 对应概念 |
//! |---|---|---|
//! | [`tokenize`] | 中英混排分词（CJK unigram + bigram） | Tokenization |
//! | [`bm25`] | BM25 打分 + 倒排索引 | Sparse Retrieval / Keyword Search |
//! | [`hybrid`] | 双路召回 + RRF 融合 | Hybrid Search / RRF |
//! | [`reranker`] | cross-encoder 重排（本地 / Cohere） | Rerank / Node Postprocessor |
//! | [`index`] | 向量索引 + BM25 索引的构建与加载 | Knowledge Base |
//! | [`pipeline`] | 端到端编排 + 降级链 | Query Engine |
//! | [`api`] | HTTP 接口（含**多模式对比**端点） | — |
//!
//! ## 对应教程
//!
//! [LlamaIndex 的 Cohere Rerank 例子](https://developers.llamaindex.ai/python/examples/node_postprocessor/coherererank/)
//! 演示的是**两阶段检索**：先 `similarity_top_k=10` 召回，再用 rerank 精排到 `top_n=2`。
//!
//! 教程里那个例子特别有说服力：**只取 top-2 时答案���幻觉**，
//! 而 **top-10 召回 + rerank 到 top-2** 答案才正确 ——
//! 这正好说明「召回阶段的排序不够准，重排才是精度的来源」。
//!
//! ## 论文
//!
//! Modular RAG（[Gao et al., 2023](https://arxiv.org/abs/2310.11511)）——
//! 把 RAG 拆成可组合的模块（路由 / 检索 / 重排 / 生成 / 校验）。
//! 本项目实现的正是其中的 **Routing + Reranking** 两个模块。
//!
//! ## 为什么这个项目值得关注
//!
//! 因为它提供了**三个对照组**，让收益可以被量化：
//!
//! | 对照组 | 端点参数 | 说明 |
//! |---|---|---|
//! | 只走向量 | `mode=vector` | 06/07 的水平 |
//! | 只走 BM25 | `mode=bm25` | 纯关键词检索 |
//! | 混合 + 重排 | `mode=hybrid` | 本项目的产出 |
//!
//! ```bash
//! curl -s localhost:3020/api/compare -H 'Content-Type: application/json' -d '{
//!   "question": "BM25 是什么",
//!   "modes": ["vector", "bm25", "hybrid"],
//!   "expected": ["倒排索引"]
//! }'
//! ```
//!
//! 收益不靠嘴说，靠这张表。

pub mod api;
pub mod bm25;
pub mod config;
pub mod error;
pub mod hybrid;
pub mod index;
pub mod llm;
pub mod pipeline;
pub mod reranker;
pub mod tokenize;

pub use bm25::{Bm25Document, Bm25Hit, Bm25Index};
pub use config::{AppConfig, RerankConfig, RetrievalConfig};
pub use error::{RagError, RagResult};
pub use hybrid::{HybridHit, HybridRetriever, SearchMode};
pub use pipeline::{AskOutcome, Citation, HybridPipeline, RetrievalOutcome};
pub use reranker::{
    CohereReranker, LocalCrossEncoder, RerankBackend, RerankDocument, RerankedHit, Reranker,
    RerankerModel,
};
pub use tokenize::tokenize;
