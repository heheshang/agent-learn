//! HTTP API 层 —— 带缓存的 Agent 服务。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 工具清单 + 缓存状态 |
//! | GET  | `/api/tools` | 工具清单 |
//! | POST | `/api/tools/call` | 直接调用工具（不经模型） |
//! | POST | `/api/run` | ★ 跑一次 Agent（可带缓存控制） |
//! | GET  | `/api/cache/stats` | ★ 缓存统计（命中率） |
//! | GET  | `/api/cache/entries` | ★ 列出缓存条目（排查用） |
//! | DELETE | `/api/cache` | ★ 清空命名空间 |
//!
//! ## `/api/run` 的缓存控制
//!
//! 请求体可以带 `cache` 字段：
//!
//! ```json
//! {
//!   "question": "什么是缓存击穿？",
//!   "cache": { "no_cache": true }
//! }
//! ```

pub mod extract;
pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::agent::CachedAgent;
use crate::cache::CacheService;
use crate::config::AppConfig;

/// 应用共享状态。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// Agent。
    pub agent: Arc<CachedAgent>,

    /// 缓存服务。
    pub cache: Arc<CacheService>,

    /// LLM 客户端。
    ///
    /// 单独存一份是为了「对照实验」时能复用同一个模型实例 ——
    /// 保证「缓存开/关」这一个变量之外其它全都相同。
    pub llm: Arc<dyn agent_16::llm::FunctionLlm>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(
        config: AppConfig,
        agent: Arc<CachedAgent>,
        cache: Arc<CacheService>,
        llm: Arc<dyn agent_16::llm::FunctionLlm>,
    ) -> Self {
        Self {
            config,
            agent,
            cache,
            llm,
        }
    }
}
