//! LLM 层 —— 复用 16 的结构化工具调用协议。
//!
//! ## 为什么复用
//!
//! 错误处理发生在「工具调用」这一层，要精确知道
//! 「哪个 `call_id` 的哪个工具失败了」，就必须用**结构化协议**
//! （`tool_calls` 数组），而不是 13/14/15 的 ReAct 文本协议。
//!
//! 16 已经把结构化协议里最麻烦的两个坑踩过了：
//!
//! - 「每个 `tool_call` 都必须有回执，否则整个请求 400」
//! - 「开思考模式必须回传 `reasoning_content`」
//!
//! 所以这里直接复用它的 trait 和实现。
//!
//! ## 本项目新增的
//!
//! [`ScriptedLlm`] —— 除了预设响应，还能**预设失败**。
//! 这是测「模型调用失败怎么办」的前提：
//! 没有一个能稳定失败的 LLM 替身，就没法写确定性测试。

use agent_logging::{info, warn};
use std::sync::{Arc, Mutex};

use genai::chat::{ChatMessage, ChatResponse};

pub use agent_16::llm::{
    FunctionLlm, FunctionLlmFuture, GenaiFunctionLlm, ScriptedFunctionLlm, ScriptedStep,
    ScriptedToolCall, ToolChoiceMode,
};

use crate::tools::ToolSpec;

/// 16 的 `FunctionLlm` trait 返回的是 **16 的** `AppError`
/// （它的 trait 签名写死了自己的错误类型）。
///
/// 所以这里的脚本替身也必须用 16 的错误类型 ——
/// 本项目的 [`crate::error::AppError`] 通过 `From` 转换接收它。
type ScriptError = agent_16::error::AppError;

/// 脚本化 LLM 的一个步骤。
///
/// ## 比 16 的 `ScriptedStep` 多了「失败」
#[derive(Debug, Clone)]
pub enum LlmStep {
    /// 正常返回。
    Response(Box<ScriptedStep>),

    /// 调用失败（模拟网络错误 / 上游 5xx）。
    ///
    /// ## 为什么要有它
    ///
    /// 「模型调用失败」是 Agent 最常见的失败之一，
    /// 它和「工具调用失败」的处置**完全不同**：
    ///
    /// - 工具失败 → 降级成观察，模型还能继续
    /// - 模型失败 → **没有模型就没有答案**，必须终止
    ///
    /// 这个区分必须有测试覆盖，所以需要一个能稳定失败的 LLM。
    Failure(String),
}

/// 脚本化 LLM —— 可预设成功，也可预设失败。
///
/// ## 用法
///
/// ```ignore
/// let llm = ScriptedLlm::new(vec![
///     LlmStep::Failure("连接超时".to_string()),
///     LlmStep::Response(Box::new(ScriptedStep::Text("最终答案".to_string()))),
/// ]);
/// ```
pub struct ScriptedLlm {
    steps: Mutex<Vec<LlmStep>>,
    calls: Mutex<usize>,
}

impl std::fmt::Debug for ScriptedLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedLlm")
            .field(
                "remaining",
                &self.steps.lock().map(|s| s.len()).unwrap_or(0),
            )
            .field("calls", &self.calls.lock().map(|c| *c).unwrap_or(0))
            .finish()
    }
}

impl ScriptedLlm {
    /// 创建脚本化 LLM（按调用顺序返回）。
    pub fn new(steps: Vec<LlmStep>) -> Self {
        Self {
            steps: Mutex::new(steps),
            calls: Mutex::new(0),
        }
    }

    /// 调用次数。
    pub fn call_count(&self) -> usize {
        self.calls.lock().map(|c| *c).unwrap_or(0)
    }
}

impl FunctionLlm for ScriptedLlm {
    fn chat_with_tools<'a>(
        &'a self,
        _system: &'a str,
        _messages: &'a [ChatMessage],
        _tools: &'a [ToolSpec],
        _tool_choice: ToolChoiceMode,
    ) -> FunctionLlmFuture<'a> {
        let step = {
            let mut calls = self.calls.lock().expect("calls 锁中毒");
            *calls += 1;

            let mut steps = self.steps.lock().expect("steps 锁中毒");
            if steps.is_empty() {
                None
            } else {
                Some(steps.remove(0))
            }
        };

        Box::pin(async move {
            match step {
                Some(LlmStep::Failure(message)) => {
                    warn!(error = %message, "脚本化 LLM 预设失败");
                    Err(ScriptError::Llm(message))
                }
                Some(LlmStep::Response(step)) => Ok((*step).to_response()),
                None => Err(ScriptError::Llm("脚本化 LLM 没有更多预设响应".to_string())),
            }
        })
    }

    fn model_name(&self) -> &str {
        "scripted"
    }
}

/// 构造一个「带工具调用」的脚本步骤。
pub fn step_calls(calls: Vec<ScriptedToolCall>) -> LlmStep {
    LlmStep::Response(Box::new(ScriptedStep::calls(calls)))
}

/// 构造一个「文本响应」的脚本步骤。
pub fn step_text(text: impl Into<String>) -> LlmStep {
    LlmStep::Response(Box::new(ScriptedStep::Text(text.into())))
}

/// 构造一个「调用失败」的脚本步骤。
pub fn step_failure(message: impl Into<String>) -> LlmStep {
    LlmStep::Failure(message.into())
}

/// 记录一次成功的 LLM 调用（给 API 层展示用）。
pub fn describe_response(response: &ChatResponse) -> String {
    info!(tool_calls = response.tool_calls().len(), "模型响应");
    response.first_text().unwrap_or_default().to_string()
}

/// 便捷别名。
pub type SharedLlm = Arc<dyn FunctionLlm>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn returns_scripted_responses_in_order() {
        let llm = ScriptedLlm::new(vec![step_text("一"), step_text("二")]);

        let resp = llm
            .chat_with_tools("s", &[], &[], ToolChoiceMode::Auto)
            .await
            .unwrap();
        assert_eq!(resp.first_text(), Some("一"));

        let resp = llm
            .chat_with_tools("s", &[], &[], ToolChoiceMode::Auto)
            .await
            .unwrap();
        assert_eq!(resp.first_text(), Some("二"));

        assert!(
            llm.chat_with_tools("s", &[], &[], ToolChoiceMode::Auto)
                .await
                .is_err()
        );
        assert_eq!(llm.call_count(), 3);
    }

    #[tokio::test]
    async fn can_preset_failures() {
        // ⚠️ 「模型调用失败」必须有测试覆盖 ——
        // 它和「工具调用失败」的处置完全不同。
        let llm = ScriptedLlm::new(vec![step_failure("连接超时")]);

        let err = llm
            .chat_with_tools("s", &[], &[], ToolChoiceMode::Auto)
            .await
            .unwrap_err();

        assert_eq!(err.status_code(), 502);
    }

    #[tokio::test]
    async fn can_preset_tool_calls() {
        let llm = ScriptedLlm::new(vec![step_calls(vec![ScriptedToolCall::new(
            "c1",
            "flaky_fetch",
            json!({ "resource": "r" }),
        )])]);

        let resp = llm
            .chat_with_tools("s", &[], &[], ToolChoiceMode::Auto)
            .await
            .unwrap();

        assert_eq!(resp.tool_calls().len(), 1);
        assert_eq!(resp.tool_calls()[0].fn_name, "flaky_fetch");
    }
}
