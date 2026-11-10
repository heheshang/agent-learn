//! HTTP API 层。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 批策略说明 |
//! | POST | `/api/embed` | ★ 批量嵌入（一次请求嵌入 N 条） |
//! | POST | `/api/rerank` | ★ 批量重排 |
//! | POST | `/api/bench/embed` | ★ ★ 嵌入批处理对照实验 |
//! | POST | `/api/bench/rerank` | ★ ★ 重排批处理对照实验 |
//! | POST | `/api/bench/remote` | ★ ★ 远程 HTTP 批处理对照实验 |
//! | POST | `/api/index` | ★ 批量建索引 |
//! | POST | `/api/search` | ★ 批量检索（召回 + 批量重排） |

pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::config::AppConfig;
use crate::embedder::BatchEmbedder;
use crate::reranker::BatchReranker;

/// 应用共享状态。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// 批量嵌入器。
    pub embedder: Arc<BatchEmbedder>,

    /// 批量重排器。
    pub reranker: Arc<BatchReranker>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(
        config: AppConfig,
        embedder: Arc<BatchEmbedder>,
        reranker: Arc<BatchReranker>,
    ) -> Self {
        Self {
            config,
            embedder,
            reranker,
        }
    }
}
