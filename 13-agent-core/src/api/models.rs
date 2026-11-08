//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见 ——
//! 想知道服务能接什么、返回什么，看这一个文件就够了。

use serde::{Deserialize, Serialize};

use crate::agent::{AgentRun, StepRecord};
use crate::config::AgentMode;

/// 跑一次 Agent 的请求体。
///
/// ```json
/// { "question": "ReAct 框架是什么？", "mode": "react" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct RunRequest {
    /// 用户的问题。
    pub question: String,

    /// 运行模式（可选，默认用配置里的）。
    #[serde(default)]
    pub mode: Option<String>,
}

/// 跑一次 Agent 的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct RunResponse {
    /// 最终答案。
    pub answer: String,

    /// 运行模式。
    pub mode: AgentMode,

    /// 每一步的详细记录（思考 / 动作 / 观察）。
    pub steps: Vec<StepRecord>,

    /// 完整的 ReAct 轨迹（模型看到的那份文本）。
    pub trajectory: String,

    /// 工具调用次数。
    pub tool_calls: usize,

    /// 是否因为达到步数上限而中止。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,
}

impl RunResponse {
    /// 从 [`AgentRun`] 构造响应。
    pub fn from_run(run: AgentRun) -> Self {
        // ⚠️ 先算轨迹再移动字段 —— `trajectory()` 借用 `steps`，
        // 直接 move 会触发「部分移动后借用」。
        let trajectory = run.trajectory();
        Self {
            answer: run.answer,
            mode: run.mode,
            steps: run.steps,
            trajectory,
            tool_calls: run.tool_calls,
            hit_max_steps: run.hit_max_steps,
            elapsed_ms: run.elapsed_ms,
            model: run.model,
        }
    }
}

/// 三模式消融对照的请求体。
///
/// ```json
/// { "question": "ReAct 框架是什么？" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CompareRequest {
    /// 用户的问题。
    pub question: String,

    /// 要对比的模式（可选，默认全部三种）。
    #[serde(default)]
    pub modes: Option<Vec<String>>,
}

/// 三模式消融对照的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CompareResponse {
    /// 原始问题。
    pub question: String,

    /// 每个模式的结果。
    pub results: Vec<ModeResult>,

    /// 结论摘要（哪个模式用了几次工具、答案是什么）。
    pub summary: Vec<String>,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 单个模式的对照结果。
#[derive(Debug, Clone, Serialize)]
pub struct ModeResult {
    /// 模式名。
    pub mode: AgentMode,

    /// 最终答案。
    pub answer: String,

    /// 工具调用次数。
    pub tool_calls: usize,

    /// 是否达到步数上限。
    pub hit_max_steps: bool,

    /// 每一步的记录。
    pub steps: Vec<StepRecord>,

    /// 该模式的耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 只解析模型输出的请求体。
///
/// ```json
/// { "output": "Thought: ...\nAction: search\nAction Input: {...}" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ParseRequest {
    /// 模型输出的原始文本。
    pub output: String,
}

/// 解析结果。
#[derive(Debug, Clone, Serialize)]
pub struct ParseResponse {
    /// 解析出的动作类型：`final_answer` / `action` / `unstructured`。
    pub kind: String,

    /// 思考内容。
    pub thought: String,

    /// 工具名（`action` 时有值）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,

    /// 工具参数。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,

    /// 参数解析错误。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parse_error: Option<String>,

    /// 最终答案文本。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
}

/// 错误响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// 错误类别（HTTP 状态码的规范名）。
    pub error: String,

    /// 错误详情。
    pub message: String,
}
