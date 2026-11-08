//! HTTP API 层 —— Agent 自主决策服务。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 工具清单 + 知识库状态 |
//! | POST | `/api/run` | **跑一次 Agent**（ReAct 循环） |
//! | POST | `/api/compare` | **三模式消融对照** ★ |
//! | POST | `/api/parse` | **只解析模型输出**（调试解析器） |
//!
//! ## 为什么 `/api/compare` 是核心
//!
//! ReAct 论文的核心主张是「带 `Thought:` 的 ReAct 优于 Act-only 基线」。
//! 这个端点把**同一个问题**分别喂给三种模式：
//!
//! | 模式 | 工具 | 思考 | 验证什么 |
//! |---|---|---|---|
//! | `direct` | ❌ | ❌ | 模型参数里的知识够不够 |
//! | `act_only` | ✅ | ❌ | 「行动」本身值多少 |
//! | `react` | ✅ | ✅ | 「思考」又贡献了多少 |
//!
//! 收益不靠嘴说，靠这张表。

pub mod extract;
pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::agent::Agent;
use crate::config::AppConfig;

/// 应用共享状态。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// Agent。
    pub agent: Arc<Agent>,

    /// 知识库里的文档块数。
    pub knowledge_chunks: usize,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(config: AppConfig, agent: Arc<Agent>, knowledge_chunks: usize) -> Self {
        Self {
            config,
            agent,
            knowledge_chunks,
        }
    }
}
