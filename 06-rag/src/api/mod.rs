//! HTTP API 层 —— 文档问答接口。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 索引状态 |
//! | POST | `/api/ask` | **Naive RAG 问答**（检索 → 生成） |
//! | POST | `/api/agent` | **Agent 问答**（模型自己决定检索） |
//! | POST | `/api/search` | 只检索不生成（调试用，看召回了什么） |
//! | POST | `/api/reindex` | 重建索引 |
//!
//! ## 为什么要分 `/api/ask` 和 `/api/agent` 两个端点
//!
//! 因为它们的**行为差异是本项目要观察的核心现象**：
//!
//! - `/api/ask`：问题原样拿去检索，检索结果直接进 Prompt
//! - `/api/agent`：模型先改写查询词，再检索
//!
//! 同一个问题打两个端点，就能直接对比出「查询改写」带来的差异。
//! 这比读十篇架构文章都直观。
//!
//! ## 统一错误响应
//!
//! ```json
//! { "error": "Bad Request", "message": "请求参数不合法: 问题不能为空" }
//! ```
//!
//! 状态码由 [`crate::error::RagError::status_code`] 决定。

pub mod extract;
pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::agent::Agent;
use crate::config::AppConfig;
use crate::generation::Generator;
use crate::retrieval::Retriever;

/// 应用共享状态。
///
/// 所有 handler 都拿到同一个 `Arc<AppState>`。
/// 里面装的都是不可变的东西（`Arc<Retriever>` / `Arc<Agent>`），
/// 只有索引本身在内部用 Mutex 串行化。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// 检索器。
    pub retriever: Arc<Retriever>,

    /// 生成器。
    pub generator: Arc<Generator>,

    /// 自主 Agent。
    pub agent: Arc<Agent>,

    /// 索引统计（建索引时更新）。
    pub index_stats: crate::index::IndexStats,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(
        config: AppConfig,
        retriever: Arc<Retriever>,
        generator: Arc<Generator>,
        agent: Arc<Agent>,
        index_stats: crate::index::IndexStats,
    ) -> Self {
        Self {
            config,
            retriever,
            generator,
            agent,
            index_stats,
        }
    }
}
