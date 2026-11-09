//! HTTP API 层 —— SQL Agent 服务。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 工具清单 |
//! | GET  | `/api/tools` | 工具清单（名字 / 描述 / Schema） |
//! | GET  | `/api/schema` | 数据库概览（表 / 列 / 行数） |
//! | POST | `/api/tools/call` | **直接调用工具**（不经模型） |
//! | POST | `/api/run` | 跑一次 Agent |
//! | POST | `/api/guard` | 只跑护栏（不执行） |
//!
//! ## `/api/guard` 值得单独说
//!
//! 它对应「安全护栏」的**独立可测性** ——
//! 用户可以拿任意 SQL 问「这条会被拦吗？」，不用跑模型。
//!
//! 这在生产里很有用：可以在 CI 里跑一批「已知危险 SQL」，
//! 确认护栏没被改坏。

pub mod extract;
pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::agent::Agent;
use crate::config::AppConfig;
use crate::db::Database;

/// 应用共享状态。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// Agent。
    pub agent: Arc<Agent>,

    /// 数据库（只读连接）。
    pub db: Arc<Database>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(config: AppConfig, agent: Arc<Agent>, db: Arc<Database>) -> Self {
        Self { config, agent, db }
    }
}
