//! LLM 层 —— 复用 16 的结构化工具调用协议 + 18 的可失败替身。
//!
//! ## 为什么直接复用
//!
//! 本项目要改的是「**工具怎么执行**」（并发），
//! 而不是「怎么和模型对话」。
//!
//! 所以协议层完全不动：
//! - `FunctionLlm` —— 带工具定义的一次对话
//! - `GenaiFunctionLlm` —— 真实模型
//! - `ScriptedLlm` —— 脚本替身（18 的版本，能**预设失败**）
//!
//! ## ⚠️ 沿用 16 的两个坑
//!
//! 1. 每个 `tool_call` 都必须有回执（少一条整个请求 400）
//! 2. 开思考模式必须回传 `reasoning_content`
//!
//! 这两个坑都在 16 修好了，本项目**不重写** ——
//! 重写等于把坑再踩一次。

use std::sync::Arc;

pub use agent_18::llm::{
    FunctionLlm, FunctionLlmFuture, GenaiFunctionLlm, LlmStep, ScriptedLlm, ScriptedStep,
    ScriptedToolCall, ToolChoice,
};

/// 便捷别名。
pub type SharedLlm = Arc<dyn FunctionLlm>;

/// 一次**纯文本**生成（不传工具）。
pub async fn generate_text(
    llm: &dyn FunctionLlm,
    system: &str,
    user: &str,
) -> Result<String, agent_16::error::AppError> {
    let response = llm
        .chat_with_tools(
            system,
            &[genai::chat::ChatMessage::user(user.to_string())],
            &[],
            ToolChoice::None,
        )
        .await?;

    Ok(response.first_text().unwrap_or_default().to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn generates_plain_text_without_tools() {
        let llm = ScriptedLlm::new(vec![step_text("最终答案")]);

        let text = generate_text(&llm, "系统提示", "问题").await.unwrap();

        assert_eq!(text, "最终答案");
        assert_eq!(llm.call_count(), 1);
    }

    #[tokio::test]
    async fn scripted_llm_can_preset_failures() {
        let llm = ScriptedLlm::new(vec![step_failure("连接超时")]);

        let err = generate_text(&llm, "s", "u").await.unwrap_err();

        assert_eq!(err.status_code(), 502);
    }

    #[tokio::test]
    async fn scripted_llm_can_preset_parallel_tool_calls() {
        // ⚠️ 本项目的核心场景：一轮里**多个**工具调用。
        let llm = ScriptedLlm::new(vec![step_calls(vec![
            ScriptedToolCall::new("c1", "get_weather", json!({ "location": "北京" })),
            ScriptedToolCall::new("c2", "get_weather", json!({ "location": "上海" })),
        ])]);

        let resp = llm
            .chat_with_tools(
                "s",
                &[genai::chat::ChatMessage::user("问题".to_string())],
                &[],
                ToolChoice::Auto,
            )
            .await
            .unwrap();

        assert_eq!(resp.tool_calls().len(), 2);
    }
}
