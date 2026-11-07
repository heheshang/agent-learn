//! HTTP API 层 —— 混合检索与重排的对照实验接口。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 索引状态 |
//! | POST | `/api/ask` | **混合检索 + 重排 + 生成**（问答） |
//! | POST | `/api/search` | **只检索 + 重排**，不生成（调试用） |
//! | POST | `/api/compare` | **多模式对比**（本项目的核心端点） |
//! | POST | `/api/reindex` | 重建索引 |
//!
//! ## 为什么 `/api/compare` 是核心
//!
//! 因为本项目的价值不在「加了一个 BM25」，而在**「证明了混合检索值多少」**。
//!
//! `/api/compare` 接受一组模式，对同一个问题依次跑完，
//! 返回每个模式的命中块、以及**它们之间的召回重叠度**。
//!
//! ```bash
//! curl -s localhost:3020/api/compare -H 'Content-Type: application/json' -d '{
//!   "question": "BM25 是什么",
//!   "modes": ["vector", "bm25", "hybrid"],
//!   "expected": ["倒排索引"]
//! }'
//! ```
//!
//! 响应里的 `hit` 字段直接告诉你：**哪些模式召回了正确答案**。
//!
//! ## 统一错误响应
//!
//! ```json
//! { "error": "Bad Request", "message": "请求参数不合法: 问题不能为空" }
//! ```

pub mod extract;
pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::config::AppConfig;
use crate::hybrid::HybridRetriever;
use crate::pipeline::HybridPipeline;
use crate::reranker::Reranker;

/// 应用共享状态。
///
/// 所有 handler 都拿到同一个 `Arc<AppState>`。
/// 里面装的都是不可变的东西（`Arc<HybridPipeline>`），
/// 只有索引本身在内部用 Mutex 串行化。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// 检索流水线（混合检索 → 重排 → 生成）。
    pub pipeline: Arc<HybridPipeline>,

    /// 索引统计（建索引时更新）。
    pub index_stats: crate::index::IndexStats,

    /// BM25 索引里的文档数。
    pub bm25_documents: usize,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(
        config: AppConfig,
        pipeline: Arc<HybridPipeline>,
        index_stats: crate::index::IndexStats,
        bm25_documents: usize,
    ) -> Self {
        Self {
            config,
            pipeline,
            index_stats,
            bm25_documents,
        }
    }
}

/// 构造流水线（供 main 和测试共用）。
pub fn build_pipeline(
    retriever: Arc<HybridRetriever>,
    reranker: Option<Arc<dyn Reranker>>,
    llm: Arc<dyn rag_07::llm::Llm>,
    config: &AppConfig,
) -> Arc<HybridPipeline> {
    Arc::new(HybridPipeline::new(
        retriever,
        reranker,
        llm,
        config.retrieval.clone(),
        config.rerank.clone(),
    ))
}
