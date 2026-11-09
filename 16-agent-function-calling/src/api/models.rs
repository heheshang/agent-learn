//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见 ——
//! 想知道服务能接什么、返回什么，看这一个文件就够了。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{AgentRun, StepRecord};

/// 跑一次 Agent 的请求体。
///
/// ```json
/// { "question": "北京和上海现在天气怎么样？" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct RunRequest {
    /// 用户的问题。
    pub question: String,
}

/// 跑一次 Agent 的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct RunResponse {
    /// 最终答案。
    pub answer: String,

    /// 每一步的详细记录（工具调用 / 参数 / 结果）。
    pub steps: Vec<StepRecord>,

    /// 工具调用总次数。
    pub tool_calls: usize,

    /// 用到的工具名。
    pub tools_used: Vec<String>,

    /// 是否因为达到步数上限而中止。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,

    /// token 用量。
    pub usage: crate::agent::TokenUsage,
}

impl RunResponse {
    /// 从 [`AgentRun`] 构造响应。
    pub fn from_run(run: AgentRun) -> Self {
        Self {
            answer: run.answer,
            steps: run.steps,
            tool_calls: run.tool_calls,
            tools_used: run.tools_used,
            hit_max_steps: run.hit_max_steps,
            elapsed_ms: run.elapsed_ms,
            model: run.model,
            usage: run.usage,
        }
    }
}

/// 四档对照的请求体。
///
/// ```json
/// { "question": "北京现在天气怎么样？" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CompareRequest {
    /// 用户的问题。
    pub question: String,
}

/// 四档对照的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CompareResponse {
    /// 每一档的结果。
    pub results: Vec<CompareEntry>,
}

/// 对照表里的一项。
#[derive(Debug, Clone, Serialize)]
pub struct CompareEntry {
    /// 档位名（`auto` / `none` / `required` / `forced`）。
    pub mode: String,

    /// 该档位的最终答案。
    pub answer: String,

    /// 工具调用次数。
    pub tool_calls: usize,

    /// 用到的工具名。
    pub tools_used: Vec<String>,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,

    /// token 用量。
    pub usage: crate::agent::TokenUsage,

    /// 错误（如果这一档失败了）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// 直接调用工具的请求体。
///
/// ```json
/// { "tool": "get_weather", "args": { "city": "北京" } }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CallToolRequest {
    /// 工具名。
    pub tool: String,

    /// 工具参数（可选）。
    #[serde(default)]
    pub args: Value,
}

/// 直接调用工具的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CallToolResponse {
    /// 工具名。
    pub tool: String,

    /// 是否成功。
    pub ok: bool,

    /// 结果文本（失败时是错误描述）。
    pub observation: String,

    /// 结构化数据（如果工具返回了）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// 工具清单里的一项。
#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    /// 工具名。
    pub name: String,

    /// 工具描述。
    pub description: String,

    /// JSON Schema。
    pub schema: Value,

    /// 是否开启 strict 模式。
    pub strict: bool,

    /// 参数说明（渲染成人类可读文本）。
    pub parameters: String,
}

/// 工具清单响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ToolListResponse {
    /// 工具数量。
    pub count: usize,

    /// 工具清单。
    pub tools: Vec<ToolInfo>,
}

/// 错误响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// 错误类别（HTTP 状态码的规范名）。
    pub error: String,

    /// 错误详情。
    pub message: String,
}
