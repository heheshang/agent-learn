//! HTTP API 层 —— 查询改写对照实验接口。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 索引状态 |
//! | POST | `/api/ask` | **改写 + 检索 + 生成**（问答） |
//! | POST | `/api/search` | **只改写 + 检索**，不生成（调试用） |
//! | POST | `/api/transform` | **只改写**，不检索（看改写结果） |
//! | POST | `/api/compare` | **多策略对比**（本项目的核心端点） |
//! | POST | `/api/reindex` | 重建索引 |
//!
//! ## 为什么 `/api/compare` 是核心
//!
//! 因为本项目的价值不在「加了一个 HyDE」，而在**「证明了 HyDE 值多少」**。
//!
//! `/api/compare` 接受一组策略，对同一个问题依次跑完，
//! 返回每个策略的改写结果、命中块、以及**它们之间的召回重叠度**。
//!
//! ```bash
//! curl -s localhost:3010/api/compare -H 'Content-Type: application/json' -d '{
//!   "question": "模型的知识有截止日期怎么办",
//!   "strategies": ["none", "hyde", "multi_query", "decompose"],
//!   "expected": ["检索增强生成"]
//! }'
//! ```
//!
//! 响应里的 `hit` 字段直接告诉你：**哪些策略召回了正确答案**。
//!
//! ## 统一错误响应
//!
//! ```json
//! { "error": "Bad Request", "message": "请求参数不合法: 问题不能为空" }
//! ```
//!
//! 状态码由 [`crate::error::QueryError::status_code`] 决定。

pub mod extract;
pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::config::AppConfig;
use crate::llm::Llm;
use crate::pipeline::QueryPipeline;
use crate::retrieval::MultiQueryRetriever;

/// 应用共享状态。
///
/// 所有 handler 都拿到同一个 `Arc<AppState>`。
/// 里面装的都是不可变的东西（`Arc<QueryPipeline>`），只有索引本身在内部用 Mutex 串行化。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// 查询流水线（改写 → 检索 → 融合 → 生成）。
    pub pipeline: Arc<QueryPipeline>,

    /// 索引统计（建索引时更新）。
    pub index_stats: crate::index::IndexStats,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(
        config: AppConfig,
        pipeline: Arc<QueryPipeline>,
        index_stats: crate::index::IndexStats,
    ) -> Self {
        Self {
            config,
            pipeline,
            index_stats,
        }
    }
}

/// 构造流水线（供 main 和测试共用）。
pub fn build_pipeline(
    retriever: Arc<MultiQueryRetriever>,
    llm: Arc<dyn Llm>,
    config: &AppConfig,
) -> Arc<QueryPipeline> {
    Arc::new(QueryPipeline::new(
        retriever,
        llm,
        config.retrieval.clone(),
        config.transform.clone(),
    ))
}
