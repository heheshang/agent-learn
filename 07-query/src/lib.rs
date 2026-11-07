//! 07-query —— 手撕 **Query Transformation（查询改写）** 层。
//!
//! 06-rag 把完整的 Naive RAG 链路跑通了，但那条链路有个致命的「朴素」之处：
//! **用户问什么，就拿什么去检索**。
//!
//! ```text
//! 用户问题 ──原样──> 向量检索 ──> 命中块
//! ```
//!
//! 问题在于：**人话和文档不是同一种语言**。
//!
//! | 用户问 | 文档写 | 向量能对上吗 |
//! |---|---|---|
//! | 「喝咖啡晚上睡不着是怎么回事」 | 「咖啡因的副作用包括失眠、心悸」 | 勉强 |
//! | 「模型的知识有截止日期怎么办」 | 「检索增强生成通过外部知识库缓解知识截止问题」 | 难 |
//! | 「文档切多大合适」 | 「块大小通常设定在三百至五百 token」 | 难 |
//!
//! Query Transformation 就是**在检索之前，把用户的问题翻译成「文档的语言」**。
//!
//! ## 本项目实现的三种改写策略
//!
//! | 策略 | 核心思想 | 对应 LlamaIndex |
//! |---|---|---|
//! | **HyDE** | 让 LLM 先「假想」一段答案文档，用这段文档去检索 | `HyDEQueryTemplate` |
//! | **Multi-Query** | 让 LLM 生成 N 个等价问法，各自检索后融合 | `MultiQueryRetriever` |
//! | **Rewrite** | 把口语问题改写成语义更紧凑的独立检索式 | `QueryRewritePrompt` |
//! | **Decompose** | 把复合问题拆成多个可独立检索的子问题 | `SubQuestionQueryEngine` |
//!
//! ## 整体链路
//!
//! ```text
//!                    ┌──────────────────────────────┐
//!   用户问题 ───────> │  QueryTransformer（可组合）    │
//!                    │  HyDE / MultiQuery / Rewrite  │
//!                    │  / Decompose                  │
//!                    └──────────────┬───────────────┘
//!                                   │ Vec<TransformedQuery>
//!                    ┌──────────────▼───────────────┐
//!                    │  多路检索：每个改写查一次       │
//!                    └──────────────┬───────────────┘
//!                                   │ Vec<RankedRun>
//!                    ┌──────────────▼───────────────┐
//!                    │  融合：RRF / MaxScore          │
//!                    └──────────────┬───────────────┘//!
//!                                   │ Vec<FusedHit>
//!                    ┌──────────────▼───────────────┐
//!                    │  top-k 上下文 ──> LLM ──> 答案  │
//!                    └──────────────────────────────┘
//! ```
//!
//! ## 模块划分
//!
//! | 模块 | 职责 | 对应概念 |
//! |---|---|---|
//! | [`transform`] | 改写策略（trait + 四种实现 + 解析 + 提示词） | Query Transformations |
//! | [`fusion`] | 多路结果融合（RRF / MaxScore） | Query Fusion Retriever |
//! | [`retrieval`] | 多路检索器（支持「按文档还是按查询嵌入」） | Multi-Query Retrieval |
//! | [`pipeline`] | 端到端编排：改写 → 多路检索 → 融合 | Transform Query Engine |
//! | [`api`] | HTTP 接口（含**多策略对比**端点） | — |
//!
//! ## 为什么这个项目值得关注
//!
//! 因为它是**第一个能直接量出收益的增强手段**。
//!
//! 06-rag 的 README 里列了一张「Naive 缺什么」的表，Query Transformation
//! 排在第一行。这个项目把它实现出来，并且提供一个 `/api/compare` 端点：
//!
//! ```bash
//! # 同一个问题，一次跑四种策略，直接看召回差异
//! curl -s localhost:3010/api/compare -H 'Content-Type: application/json' -d '{
//!   "question": "模型的知识有截止日期怎么办",
//!   "strategies": ["none", "hyde", "multi_query", "decompose"],
//!   "expected": ["检索增强生成"]
//! }'
//! ```
//!
//! 收益不靠嘴说，靠这张表。

pub mod api;
pub mod config;
pub mod error;
pub mod fusion;
pub mod index;
pub mod llm;
pub mod pipeline;
pub mod retrieval;
pub mod transform;

pub use config::{AppConfig, TransformConfig};
pub use error::{QueryError, QueryResult};
pub use fusion::{FusionMethod, FusedHit, RankedRun};
pub use llm::{GenaiLlm, Llm, LlmOptions};
pub use pipeline::{AskOutcome, QueryPipeline, TransformedRetrieval};
pub use retrieval::{EmbedMode, MultiQueryRetriever};
pub use transform::{QueryTransformer, Strategy, TransformedQuery};
