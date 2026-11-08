//! HTTP API 层 —— 摘要式 RAG 服务。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查（含 Milvus 连通性） |
//! | GET  | `/` | 服务信息 + 索引状态 |
//! | POST | `/api/search` | **纯检索**（不生成） |
//! | POST | `/api/ask` | **完整问答**（检索 → 重排 → 压缩 → 生成） |
//! | POST | `/api/compare` | **同题消融对照** ★ |
//! | POST | `/api/eval` | **跑评估集**（产出报告） |
//! | POST | `/api/reindex` | 重建索引 |
//!
//! ## 为什么 `/api/compare` 是核心
//!
//! 它把**同一个问题**分别喂给阶梯上的每一档策略，
//! 返回各档的命中、重合度、耗时 ——
//! 用**实测数据**回答「每一步优化到底值多少」。
//!
//! ```bash
//! curl -s localhost:3050/api/compare -H 'Content-Type: application/json' -d '{
//!   "question": "混合检索怎么融合两路结果"
//! }'
//! ```

pub mod extract;
pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::bootstrap::IndexReport;
use crate::config::AppConfig;
use crate::pipeline::SummaryPipeline;

/// 应用共享状态。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// 流水线。
    pub pipeline: Arc<SummaryPipeline>,

    /// 上次建索引的报告。
    pub index_report: Option<IndexReport>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(
        config: AppConfig,
        pipeline: Arc<SummaryPipeline>,
        index_report: Option<IndexReport>,
    ) -> Self {
        Self {
            config,
            pipeline,
            index_report,
        }
    }
}
