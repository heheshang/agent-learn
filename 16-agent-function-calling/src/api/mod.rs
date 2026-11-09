//! HTTP API 层 —— Function Calling Agent 服务。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 工具清单 |
//! | GET  | `/api/tools` | 工具清单（名字 / 描述 / Schema / strict） |
//! | POST | `/api/tools/call` | **直接调用工具**（不经模型） |
//! | POST | `/api/run` | 跑一次 Agent |
//! | POST | `/api/compare` | ★ **tool_choice 四档对照** |
//!
//! ## `/api/compare` 值得单独说
//!
//! 它回答一个关键问题：**「模型自己决定要不要用工具」到底值多少？**
//!
//! 四档对照（同一问题、同一个模型与工具集，只有 `tool_choice` 变）：
//!
//! | 档位 | 行为 | 想验证什么 |
//! |---|---|---|
//! | `auto` | 模型自己决定 | 基线 |
//! | `none` | 一个都不调 | 不用工具能答到什么程度 |
//! | `required` | 强制调 | 强制用工具的效果 |
//! | `forced:工具名` | 必须调那一个 | 定向调用的效果

pub mod extract;
pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::agent::Agent;
use crate::config::AppConfig;
use crate::llm::FunctionLlm;

/// 应用共享状态。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// Agent。
    pub agent: Arc<Agent>,

    /// LLM 客户端。
    ///
    /// ## 为什么单独存一份
    ///
    /// `/api/compare` 要在**同一个模型实例**上跑四档
    /// （只有 `tool_choice` 变），所以 LLM 必须能被复用。
    ///
    /// ⚠️ 对照实验的价值在于「只有一个变量不同」——
    /// 如果四档各自建了模型，差值就说不清是谁带来的。
    pub llm: Arc<dyn FunctionLlm>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(config: AppConfig, agent: Arc<Agent>, llm: Arc<dyn FunctionLlm>) -> Self {
        Self { config, agent, llm }
    }
}
