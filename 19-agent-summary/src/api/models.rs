//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见。

use serde::{Deserialize, Serialize};

use crate::agent::AgentRun;
use crate::evidence::Evidence;
use crate::report::{Grounding, ProjectPlan, WeeklySummary};

/// 研究请求体。
///
/// ```json
/// { "question": "本周错误处理模块的进展" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ResearchRequest {
    /// 本轮问题。
    pub question: String,
}

/// 研究响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ResearchResponse {
    /// 模型的研究结论。
    pub answer: String,

    /// 每一步的记录。
    pub steps: Vec<crate::agent::StepRecord>,

    /// 收集到的证据。
    pub evidence: Vec<Evidence>,

    /// 工具调用总次数（**含**重试）。
    pub tool_calls: usize,

    /// 被重试过的调用次数。
    pub retried_calls: usize,

    /// 降级成观察的错误数。
    pub degraded_errors: usize,

    /// 用到的工具名。
    pub tools_used: Vec<String>,

    /// 是否达到步数上限。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

impl ResearchResponse {
    /// 从 [`AgentRun`] 构造。
    pub fn from_run(run: AgentRun) -> Self {
        Self {
            answer: run.answer,
            steps: run.steps,
            evidence: run.evidence,
            tool_calls: run.tool_calls,
            retried_calls: run.retried_calls,
            degraded_errors: run.degraded_errors,
            tools_used: run.tools_used,
            hit_max_steps: run.hit_max_steps,
            elapsed_ms: run.elapsed_ms,
        }
    }
}

/// 周报请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct WeeklyRequest {
    /// 本周关注点（可选）。
    #[serde(default)]
    pub question: String,
}

/// 周报响应体。
#[derive(Debug, Clone, Serialize)]
pub struct WeeklyResponse {
    /// 结构化周报。
    pub summary: WeeklySummary,

    /// 渲染好的 Markdown。
    pub markdown: String,

    /// 接地状态。
    pub grounding: Grounding,

    /// 研究阶段的统计。
    pub stats: RunStats,
}

/// 项目方案请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct PlanRequest {
    /// 需求描述。
    pub requirement: String,
}

/// 项目方案响应体。
#[derive(Debug, Clone, Serialize)]
pub struct PlanResponse {
    /// 结构化方案。
    pub plan: ProjectPlan,

    /// 渲染好的 Markdown。
    pub markdown: String,

    /// 接地状态。
    pub grounding: Grounding,

    /// 研究阶段的统计。
    pub stats: RunStats,
}

/// 一次运行的统计。
#[derive(Debug, Clone, Serialize)]
pub struct RunStats {
    /// 工具调用次数（含重试）。
    pub tool_calls: usize,

    /// 被重试过的调用次数。
    pub retried_calls: usize,

    /// 降级次数。
    pub degraded_errors: usize,

    /// 证据条数。
    pub evidence_count: usize,

    /// 用到的工具。
    pub tools_used: Vec<String>,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

impl RunStats {
    /// 从 [`AgentRun`] 构造。
    pub fn from_run(run: &AgentRun) -> Self {
        Self {
            tool_calls: run.tool_calls,
            retried_calls: run.retried_calls,
            degraded_errors: run.degraded_errors,
            evidence_count: run.evidence.len(),
            tools_used: run.tools_used.clone(),
            elapsed_ms: run.elapsed_ms,
        }
    }
}

/// 直接调工具的请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct CallToolRequest {
    /// 工具名。
    pub tool: String,

    /// 参数。
    #[serde(default)]
    pub args: serde_json::Value,
}

/// 直接调工具的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CallToolResponse {
    /// 工具是否成功。
    pub ok: bool,

    /// 观察文本。
    pub observation: String,

    /// 错误码（失败时）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,

    /// 错误类别（失败时）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,

    /// 实际尝试次数。
    pub attempts: usize,

    /// 该不该重试（失败时）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,

    /// 该不该降级（失败时）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degradable: Option<bool>,
}

/// 错误响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// 错误类别（HTTP 状态码的规范名）。
    pub error: String,

    /// 错误详情。
    pub message: String,
}
