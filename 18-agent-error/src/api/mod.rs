//! HTTP API 层。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 工具清单 |
//! | POST | `/api/run` | 跑一次 Agent（含错误处理） |
//! | POST | `/api/tools/call` | 直接调工具（不经模型） |
//! | POST | `/api/errors/table` | ★ **错误分类速查表** |
//!
//! ## `/api/errors/table` 值得单独说
//!
//! 它把「哪个错误该重试、哪个该降级、哪个该终止」
//! 这张决策表**暴露成 API** ——
//! 因为这是本项目最容易被误用的地方，
//! 让调用方能自己查到规则，比藏在代码里强。

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
    /// 单独存一份是为了让「错误分类对照」端点
    /// 能在**同一个模型实例**上跑多个场景 ——
    /// 对照实验的价值在于「只有一个变量不同」。
    pub llm: Arc<dyn FunctionLlm>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(config: AppConfig, agent: Arc<Agent>, llm: Arc<dyn FunctionLlm>) -> Self {
        Self { config, agent, llm }
    }
}
