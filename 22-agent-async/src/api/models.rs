//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见。
//!
//! ## ⚠️ 不重复定义「长得一样但不是同一个类型」的结构
//!
//! `StepRecord` / `ToolCallRecord` 直接复用 [`crate::agent`] 里的 ——
//! 重复定义一份会让「API 层看到的」和「Agent 内部产出的」悄悄漂移。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{AgentRun, StepRecord};
use crate::concurrency::{BatchReport, ExecutionMode};

/// 统一错误响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// HTTP 状态的规范名（如 "Bad Request"）。
    pub error: String,

    /// 错误详情。
    pub message: String,
}

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

    /// 每一步的记录。
    pub steps: Vec<StepRecord>,

    /// 工具调用总次数（**含**重试）。
    pub tool_calls: usize,

    /// 逻辑上的工具调用组数（同一轮的并行调用算 1 组）。
    pub tool_call_groups: usize,

    /// 用到的工具名。
    pub tools_used: Vec<String>,

    /// 被重试过的调用次数。
    pub retried_calls: usize,

    /// 降级成观察的错误数。
    pub degraded_errors: usize,

    /// 是否达到步数上限。
    pub hit_max_steps: bool,

    /// 是否因为**致命错误**而终止。
    pub fatal_terminated: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 模型名。
    pub model: String,
}

impl RunResponse {
    /// 从 [`AgentRun`] 构造。
    pub fn from_run(run: AgentRun) -> Self {
        Self {
            answer: run.answer,
            steps: run.steps,
            tool_calls: run.tool_calls,
            tool_call_groups: run.tool_call_groups,
            tools_used: run.tools_used,
            retried_calls: run.retried_calls,
            degraded_errors: run.degraded_errors,
            hit_max_steps: run.hit_max_steps,
            fatal_terminated: run.fatal_terminated,
            elapsed_ms: run.elapsed_ms,
            model: run.model,
        }
    }
}

/// 直接调工具的请求体。
///
/// ```json
/// { "tool": "get_weather", "args": { "location": "北京" } }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CallToolRequest {
    /// 工具名。
    pub tool: String,

    /// 参数。
    #[serde(default)]
    pub args: Value,
}

/// 直接调工具的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CallToolResponse {
    /// 工具名。
    pub tool: String,

    /// 是否走了线程池轨道。
    pub blocking: bool,

    /// 结果（观察文本）。
    pub observation: String,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 并发对照实验的请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct BenchRequest {
    /// 要并发的城市列表（每个城市调一次天气接口）。
    ///
    /// 默认 4 个城市。
    #[serde(default)]
    pub cities: Vec<String>,

    /// ⚠️ 并发上限（`buffer_unordered(n)` 的 n）。
    ///
    /// 不填则用配置里的 `ASYNC_MAX_CONCURRENCY`。
    #[serde(default)]
    pub max_concurrency: Option<usize>,

    /// ⚠️ 是否用**假**工具（`sleep` 模拟 I/O）而不是真的调天气接口。
    ///
    /// 默认 `true` —— 因为真实天气接口有网络波动，
    /// 跑对照实验时用假的更稳定、更可复现。
    #[serde(default = "default_true")]
    pub simulated: bool,
}

/// 默认值 `true`（serde 的 `default` 需要一个返回默认值的函数）。
fn default_true() -> bool {
    true
}

/// 并发对照实验的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct BenchResponse {
    /// 每种模式的报告。
    pub reports: Vec<ModeReport>,

    /// 串行基准耗时（毫秒）。
    pub sequential_ms: u64,

    /// 最快的模式。
    pub fastest: String,

    /// 最快的加速比。
    pub best_speedup: f64,
}

/// 单种模式的报告。
#[derive(Debug, Clone, Serialize)]
pub struct ModeReport {
    /// 模式名。
    pub mode: String,

    /// 人类可读的说明。
    pub label: String,

    /// 总耗时（毫秒）。
    pub total_ms: u64,

    /// 成功数。
    pub success_count: usize,

    /// 失败数。
    pub failure_count: usize,

    /// 完成顺序（输入下标）。
    pub completion_order: Vec<usize>,

    /// 相对串行基准的加速比。
    pub speedup: f64,

    /// 每个任务的耗时（毫秒）。
    pub task_elapsed_ms: Vec<u64>,
}

impl ModeReport {
    /// 从 [`BatchReport`] 构造。
    pub fn from_report(report: &BatchReport, sequential_ms: u64) -> Self {
        Self {
            mode: report.mode.name().to_string(),
            label: report.mode.label(),
            total_ms: report.total_ms(),
            success_count: report.success_count(),
            failure_count: report.failure_count(),
            completion_order: report.completion_order.clone(),
            speedup: report.speedup(sequential_ms),
            task_elapsed_ms: report
                .results
                .iter()
                .map(|r| r.elapsed.as_millis() as u64)
                .collect(),
        }
    }
}

/// 执行模式（用于请求体）。
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeParam {
    /// 串行。
    Sequential,
    /// 全部并发 · 保序。
    JoinAll,
    /// 全部并发 · 乱序。
    AsCompleted,
    /// 限流并发。
    Bounded,
}

/// 从 `ModeParam` + 上限构造 [`ExecutionMode`]。
pub fn to_execution_mode(param: ModeParam, limit: usize) -> ExecutionMode {
    match param {
        ModeParam::Sequential => ExecutionMode::Sequential,
        ModeParam::JoinAll => ExecutionMode::JoinAll,
        ModeParam::AsCompleted => ExecutionMode::AsCompleted,
        // ⚠️ clamp 到 ≥ 1 —— `Bounded(0)` 会让请求永远挂住。
        ModeParam::Bounded => ExecutionMode::Bounded(limit.max(1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_request_defaults_to_simulated() {
        let req: BenchRequest = serde_json::from_str(r#"{}"#).unwrap();
        assert!(req.simulated);
        assert!(req.max_concurrency.is_none());
    }

    #[test]
    fn mode_param_maps_to_execution_mode() {
        assert_eq!(
            to_execution_mode(ModeParam::Sequential, 4),
            ExecutionMode::Sequential
        );
        assert_eq!(
            to_execution_mode(ModeParam::Bounded, 4),
            ExecutionMode::Bounded(4)
        );
    }

    #[test]
    fn bounded_mode_clamps_zero_limit() {
        // ⚠️ `Bounded(0)` 会挂住 —— 必须在转换层挡掉。
        assert_eq!(
            to_execution_mode(ModeParam::Bounded, 0),
            ExecutionMode::Bounded(1)
        );
    }
}
