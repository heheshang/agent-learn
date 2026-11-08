//! LLM 抽象层 —— 复用 07-query 的 `Llm` trait。
//!
//! 13-agent-core 已经把这层抽象做完了（`agent_13::llm`），
//! 本项目直接复用：
//!
//! - `GenaiLlm` —— 真实模型（DeepSeek / OpenAI 兼容端点）
//! - `ScriptedLlm` —— 按脚本返回预设答案，零网络、零延迟、完全确定
//!
//! ## 为什么这个抽象对本项目**格外**重要
//!
//! 本项目的工具会**真的发 HTTP 请求**（天气 API）。
//! 如果测试也要发网络请求，那测试就会：
//!
//! 1. 慢（每次几百毫秒到几秒）
//! 2. 不确定（上游数据会变）
//! 3. 依赖外网（CI 上会挂）
//!
//! 所以工具层必须能**注入假实现** —— 而本项目的工具是 `Box<dyn Tool>`，
//! 塞一个假工具进去就行，不需要碰网络。
//!
//! 有了 [`ScriptedLlm`]，整个 Agent 循环 + 工具调用
//! 都能在**零网络、零延迟、完全确定**的条件下被测试。

use std::sync::Arc;

pub use agent_13::llm::{GenaiLlm, Llm, LlmOptions, ScriptedLlm};

/// 创建一个生产用的 LLM（真实模型）。
///
/// ## ⚠️ 关于 `max_tokens` 和推理模型
///
/// 复用 07 的结论：**推理模型的思考 token 和正文共用 `max_tokens` 预算**。
/// Agent 的每一步输出很短，但如果模型开了思考，
/// 1024 的预算可能被思考吃光 → 正文为空 → 解析失败。
///
/// 所以这里默认**关掉思考**。
pub fn make_llm(model: &str, temperature: f64, max_tokens: u32) -> Arc<dyn Llm> {
    Arc::new(GenaiLlm::new(
        model,
        LlmOptions::without_reasoning(temperature, max_tokens),
    ))
}

/// 创建一个测试用的脚本化 LLM。
pub fn make_scripted_llm(answers: Vec<String>) -> Arc<dyn Llm> {
    Arc::new(ScriptedLlm::new(answers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn scripted_llm_returns_answers_in_order() {
        let llm = make_scripted_llm(vec!["一".to_string(), "二".to_string()]);

        assert_eq!(llm.chat("s", "u").await.unwrap(), "一");
        assert_eq!(llm.chat("s", "u").await.unwrap(), "二");
    }

    #[test]
    fn production_llm_disables_reasoning_by_default() {
        let options = LlmOptions::without_reasoning(0.0, 1024);
        assert!(options.disable_reasoning);
    }
}
