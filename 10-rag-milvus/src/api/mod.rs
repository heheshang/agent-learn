//! HTTP API 层 —— Milvus / Qdrant 检索服务。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查（含后端连通性） |
//! | GET  | `/` | 服务信息 + collection 状态 |
//! | POST | `/api/search` | **纯检索**（稠密 / 混合 / 关键词） |
//! | POST | `/api/hybrid` | **服务端混合检索**（Milvus BM25 + RRF）★ |
//! | POST | `/api/compare` | **Milvus ↔ Qdrant 同题对照** ★ |
//! | POST | `/api/reindex` | 重建索引 |
//!
//! ## 为什么 `/api/compare` 是核心
//!
//! 选型不能靠感觉。本端点把**同一个问题**分别发给 Milvus 和 Qdrant，
//! 返回两边的命中、重合度、以及各自的耗时 ——
//! 用**实测数据**而不是宣传页做决定。
//!
//! ```bash
//! curl -s localhost:3030/api/compare -H 'Content-Type: application/json' -d '{
//!   "question": "混合检索怎么融合两路结果"
//! }'
//! ```
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
use crate::index::IndexReport;

/// 应用共享状态。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// 上次建索引的报告。
    pub index_report: Option<IndexReport>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(config: AppConfig, index_report: Option<IndexReport>) -> Self {
        Self {
            config,
            index_report,
        }
    }
}
