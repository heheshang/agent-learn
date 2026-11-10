//! LLM 层 —— 复用 18 的结构化工具调用协议。
//!
//! ## 为什么直接复用
//!
//! 本项目要同时做两件事：
//!
//! 1. **多工具编排**（RAG 检索 + Web 搜索 + SQL 查询 + 记忆读写）
//! 2. **结构化产出**（周报 / 项目方案要 JSON）
//!
//! 第 1 件事必须走**结构化协议**（`tool_calls` 数组）——
//! 只有这样才能精确知道「哪个 call_id 的哪个工具失败了」，
//! 错误处理（18 的中间件）才有意义。
//!
//! 第 2 件事不需要工具，只要一次纯文本调用 ——
//! 复用同一个 trait 时传**空 tools 列表**即可
//! （`GenaiFunctionLlm` 里有 `if !tools.is_empty()` 的判断）。
//!
//! ## 测试替身
//!
//! 复用 18 的 [`ScriptedLlm`] —— 它能预设**成功**也能预设**失败**。
//! 没有「能稳定失败的模型替身」，就没法写确定性测试。

use std::sync::Arc;

pub use agent_18::llm::{
    FunctionLlm, FunctionLlmFuture, GenaiFunctionLlm, ScriptedLlm, ScriptedStep, ScriptedToolCall,
    ToolChoice, parse_tool_choice, tool_choice_name,
};

use agent_16::tools::ToolSpec;

/// 便捷别名。
pub type SharedLlm = Arc<dyn FunctionLlm>;

/// 一次**纯文本**生成（不传工具）。
///
/// ## 为什么单独包一层
///
/// 报告生成要的是「一段 JSON 文本」，
/// 而不是结构化的 `tool_calls`。
/// 但底层接口是同一个 `chat_with_tools` ——
/// 传空 `tools` 就退化成普通对话。
///
/// 抽成独立函数的好处是**调用点能一眼看出**「这里不调工具」，
/// 避免有人误以为这里也能触发工具调用。
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

/// 一次**带工具**的调用。
pub async fn chat_with_tools(
    llm: &dyn FunctionLlm,
    system: &str,
    messages: &[genai::chat::ChatMessage],
    tools: &[ToolSpec],
    tool_choice: ToolChoice,
) -> Result<genai::chat::ChatResponse, agent_16::error::AppError> {
    llm.chat_with_tools(system, messages, tools, tool_choice)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn generates_plain_text_without_tools() {
        let llm = ScriptedLlm::new(vec![agent_18::llm::step_text("周报正文")]);

        let text = generate_text(&llm, "系统提示", "生成周报").await.unwrap();

        assert_eq!(text, "周报正文");
        // ⚠️ 纯文本生成**不能**触发工具调用。
        assert_eq!(llm.call_count(), 1);
    }

    #[tokio::test]
    async fn scripted_llm_can_preset_failures() {
        let llm = ScriptedLlm::new(vec![agent_18::llm::step_failure("连接超时")]);

        let err = generate_text(&llm, "s", "u").await.unwrap_err();

        assert_eq!(err.status_code(), 502);
    }

    #[tokio::test]
    async fn scripted_llm_can_preset_tool_calls() {
        let llm = ScriptedLlm::new(vec![agent_18::llm::step_calls(vec![
            ScriptedToolCall::new("c1", "web_search", json!({ "query": "rust" })),
        ])]);

        let resp = chat_with_tools(
            &llm,
            "s",
            &[genai::chat::ChatMessage::user("问题".to_string())],
            &[ToolSpec::new(
                "web_search",
                "搜索",
                json!({ "type": "object" }),
            )],
            ToolChoice::Auto,
        )
        .await
        .unwrap();

        assert_eq!(resp.tool_calls().len(), 1);
        assert_eq!(resp.tool_calls()[0].fn_name, "web_search");
    }
}
