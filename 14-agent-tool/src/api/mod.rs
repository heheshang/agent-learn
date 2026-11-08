//! HTTP API 层 —— 自定义工具服务。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 工具清单 |
//! | GET  | `/api/tools` | 工具清单（名字 / 描述 / Schema / 参数说明） |
//! | POST | `/api/tools/call` | **直接调用工具**（不经模型） |
//! | POST | `/api/run` | 跑一次 Agent（工具集成到 ReAct 循环） |
//! | POST | `/api/compare` | 三模式消融对照 ★ |
//! | POST | `/api/parse` | 只解析模型输出（调试解析器） |
//!
//! ## `/api/tools/call` 值得单独说
//!
//! 它对应 LangChain 教程里「工具可以被独立调用」这一条。
//!
//! 为什么重要：
//!
//! - **调试** —— 直接看工具返回什么，不用等模型
//! - **确定性** —— 同样的参数永远同样的结果（模型会飘，工具不会）
//! - **成本** —— 不调模型
//!
//! 而且它是本项目**唯一**能把工具错误直接变成 HTTP 响应的地方 ——
//! 因为在这里用户扮演了「模型」的角色。

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

    /// MCP 客户端（启用 MCP 时有值）。
    pub mcp_client: Option<Arc<crate::mcp::McpClient>>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(
        config: AppConfig,
        agent: Arc<Agent>,
        mcp_client: Option<Arc<crate::mcp::McpClient>>,
    ) -> Self {
        Self {
            config,
            agent,
            mcp_client,
        }
    }
}
