//! HTTP API 层 —— 高级数据处理的对照实验接口。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 数据目录状态 |
//! | POST | `/api/parse` | **解析一份 PDF** ★ 核心端点 |
//! | POST | `/api/compare` | **几何还原 vs 压平** 同题对照 ★ |
//! | POST | `/api/chunk` | 解析 + 分块（返回块列表） |
//!
//! ## 为什么 `/api/compare` 是最有价值的端点
//!
//! 因为本项目的核心主张是：**结构化保留了语义**。
//!
//! 这个主张必须能被**验证**，而不是「文档里说说」。
//! `/api/compare` 把同一份 PDF 分别用两种策略处理，
//! 返回两份结果让调用者自己看差异。
//!
//! ## 统一错误响应
//!
//! ```json
//! { "error": "Unprocessable Entity", "message": "文档解析失败: PDF 没有可解析的页面" }
//! ```

pub mod extract;
pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::config::AppConfig;

/// 应用共享状态。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(config: AppConfig) -> Self {
        Self { config }
    }
}
