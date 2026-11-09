//! HTTP API 层。
//!
//! ## 路由表
//!
//! | 方法 | 路径 | 说明 |
//! |---|---|---|
//! | GET  | `/health` | 健康检查 |
//! | GET  | `/` | 服务信息 + 工具清单 |
//! | POST | `/api/research` | ★ 只跑研究阶段（拿证据，不成文） |
//! | POST | `/api/weekly` | ★ **周度总结**（研究 + 成文 + 校验） |
//! | POST | `/api/plan` | ★ **项目构建方案**（研究 + 成文 + 校验） |
//! | POST | `/api/tools/call` | 直接调工具（不经模型） |
//! | GET  | `/api/evidence` | ★ 当前工具清单 + 证据来源说明 |

pub mod handlers;
pub mod models;
pub mod routes;

use std::sync::Arc;

use crate::agent::Agent;
use crate::config::AppConfig;
use crate::llm::FunctionLlm;
use crate::pipeline::{PlanPipeline, WeeklyPipeline};

/// 应用共享状态。
pub type AppState = Arc<AppStateInner>;

/// 状态内部结构。
pub struct AppStateInner {
    /// 应用配置。
    pub config: AppConfig,

    /// 研究 Agent。
    pub agent: Arc<Agent>,

    /// LLM 客户端。
    pub llm: Arc<dyn FunctionLlm>,

    /// 周报流水线。
    pub weekly: Arc<WeeklyPipeline>,

    /// 项目方案流水线。
    pub plan: Arc<PlanPipeline>,
}

impl AppStateInner {
    /// 创建状态。
    pub fn new(config: AppConfig, agent: Arc<Agent>, llm: Arc<dyn FunctionLlm>) -> Self {
        let weekly = Arc::new(WeeklyPipeline::new(
            agent.clone(),
            llm.clone(),
            config.max_evidence_per_section,
        ));
        let plan = Arc::new(PlanPipeline::new(
            agent.clone(),
            llm.clone(),
            config.max_evidence_per_section,
        ));

        Self {
            config,
            agent,
            llm,
            weekly,
            plan,
        }
    }
}
