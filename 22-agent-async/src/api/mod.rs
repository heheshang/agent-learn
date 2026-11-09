//! HTTP API 层。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 工具清单 + 配置 |
//! | POST | `/api/run` | ★ 跑一次 Agent（**并发**执行工具） |
//! | POST | `/api/tools/call` | 直接调工具（不经模型） |
//! | POST | `/api/bench` | ★ **并发对照实验**（四档模式） |
//! | GET  | `/api/modes` | 四种并发模式说明 |

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
    pub llm: Arc<dyn FunctionLlm>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(config: AppConfig, agent: Arc<Agent>, llm: Arc<dyn FunctionLlm>) -> Self {
        Self { config, agent, llm }
    }
}
