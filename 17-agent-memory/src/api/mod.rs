//! HTTP API 层 —— 带记忆的 Agent 服务。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 配置 |
//! | POST | `/api/chat` | **带记忆的对话**（核心端点） |
//! | GET  | `/api/threads` | 列出所有线程 |
//! | GET  | `/api/threads/{id}` | 查看某个线程的消息历史 |
//! | DELETE | `/api/threads/{id}` | 清空某个线程的短期记忆 |
//! | GET  | `/api/memories` | 列出某 owner 的长期记忆 |
//! | DELETE | `/api/memories` | 清空某 owner 的长期记忆 |
//! | POST | `/api/compare` | ★ **线程隔离 / 恢复对照** |
//!
//! ## `/api/compare` 值得单独说
//!
//! 它回答两个关键问题：
//!
//! 1. **线程隔离真的成立吗？** —— 两个线程各问一句，
//!    断言线程 B 的历史里**没有**线程 A 的内容。
//! 2. **进程重启后能恢复吗？** —— 同一个 `thread_id`
//!    再问一句，模型能看到之前的历史。
//!
//! ```bash
//! curl -s localhost:3100/api/compare -H 'Content-Type: application/json' -d '{
//!   "owner": "alice",
//!   "threads": ["thread-a", "thread-b"],
//!   "question": "我叫什么名字？"
//! }'
//! ```

pub mod extract;
pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::agent::Agent;
use crate::config::AppConfig;
use crate::llm::ChatLlm;

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
    /// `/api/compare` 要在**同一个模型实例**上跑多个场景，
    /// 所以 LLM 必须能被复用。
    ///
    /// ⚠️ 对照实验的价值在于「只有一个变量不同」——
    /// 如果各自建了模型，差值就说不清是谁带来的。
    pub llm: Arc<dyn ChatLlm>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(config: AppConfig, agent: Arc<Agent>, llm: Arc<dyn ChatLlm>) -> Self {
        Self { config, agent, llm }
    }
}
