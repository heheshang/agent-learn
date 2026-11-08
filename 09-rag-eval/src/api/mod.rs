//! HTTP API 层 —— RAG 评估的对照实验接口。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 评估集状态 |
//! | GET  | `/api/dataset` | 评估集详情（样本 + 标注解析结果） |
//! | POST | `/api/eval` | **跑完整评估**（多变体对比）★ 核心端点 |
//! | POST | `/api/quick-eval` | 快速评估单个问题 |
//!
//! ## 为什么 `/api/eval` 是核心
//!
//! 因为本项目的价值不在「实现了几个指标」，而在**「证明了优化值多少」**。
//!
//! `/api/eval` 接受一组变体，对**同一个评估集**依次跑完，
//! 返回每个变体的报告 + 它们之间的差值。
//!
//! ```bash
//! curl -s localhost:3030/api/eval -H 'Content-Type: application/json' -d '{
//!   "variants": ["no_retrieval", "vector", "hybrid"]
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

use crate::config::EvalConfig;
use crate::dataset::EvalDataset;
use crate::metrics::MetricRegistry;
use crate::pipeline::EvalPipeline;

/// 应用共享状态。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 评估配置。
    pub config: EvalConfig,

    /// 评估流水线。
    pub pipeline: Arc<EvalPipeline>,

    /// 评估集。
    pub dataset: EvalDataset,

    /// 指标注册表。
    pub metrics: Arc<MetricRegistry>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(
        config: EvalConfig,
        pipeline: Arc<EvalPipeline>,
        dataset: EvalDataset,
        metrics: Arc<MetricRegistry>,
    ) -> Self {
        Self {
            config,
            pipeline,
            dataset,
            metrics,
        }
    }
}
