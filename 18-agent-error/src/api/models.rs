//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见。

use serde::{Deserialize, Serialize};

use crate::agent::AgentRun;

/// 跑一次 Agent 的请求体。
///
/// ```json
/// { "question": "拉一下 report/2026" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct RunRequest {
    /// 本轮问题。
    pub question: String,
}

/// 跑一次 Agent 的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct RunResponse {
    /// 最终答案。
    pub answer: String,

    /// 每一步的记录。
    pub steps: Vec<crate::agent::StepRecord>,

    /// 工具调用总次数（**含**重试）。
    pub tool_calls: usize,

    /// 逻辑上的工具调用数（不含重试）。
    pub tool_call_groups: usize,

    /// 被重试过的调用次数。
    pub retried_calls: usize,

    /// 降级成观察的错误数。
    pub degraded_errors: usize,

    /// 是否达到步数上限。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

impl RunResponse {
    /// 从 [`AgentRun`] 构造。
    pub fn from_run(run: AgentRun) -> Self {
        Self {
            answer: run.answer,
            steps: run.steps,
            tool_calls: run.tool_calls,
            tool_call_groups: run.tool_call_groups,
            retried_calls: run.retried_calls,
            degraded_errors: run.degraded_errors,
            hit_max_steps: run.hit_max_steps,
            elapsed_ms: run.elapsed_ms,
        }
    }
}

/// 直接调工具的请求体。
///
/// ```json
/// { "tool": "strict_lookup", "args": { "id": "rec-1" } }
/// ```
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
