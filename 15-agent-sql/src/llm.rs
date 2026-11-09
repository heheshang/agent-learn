//! LLM 抽象层 —— 复用 07-query 的 `Llm` trait。
//!
//! 复用带来两个好处：
//!
//! 1. **零重复** —— `GenaiLlm` / `ScriptedLlm` / `LlmOptions` 都在 07 里，
//!    不用再写一遍
//! 2. **零网络确定性测试** —— `ScriptedLlm` 让整个 SQL Agent 循环
//!    （包括护栏触发、错误重试）都能被单元测试锁住
//!
//! ## ⚠️ 关于 `max_tokens` 和推理模型
//!
//! 复用 07 的结论：**推理模型的思考 token 和正文共用 `max_tokens` 预算**。
//!
//! SQL Agent 的每一步输出短（`Thought / Action / Action Input`），
//! 但 `Action Input` 里嵌着一条 SQL —— 有一定长度。
//!
//! 所以这里默认**关掉思考**：Agent 的每一步都是短输出，
//! 推理收益极低，而风险很高（思考吃光预算 → 正文为空 → 解析失败）。

use agent_logging::info;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub use rag_07::llm::{GenaiLlm, Llm, LlmFuture, LlmOptions, ScriptedLlm};

/// 创建一个生产用的 LLM（真实模型）。
pub fn make_llm(model: &str, temperature: f64, max_tokens: u32) -> Arc<dyn Llm> {
    info!(
        model,
        temperature, max_tokens, "创建 LLM 客户端（已关闭思考模式）"
    );
    Arc::new(GenaiLlm::new(
        model,
        LlmOptions::without_reasoning(temperature, max_tokens),
    ))
}

/// 创建一个测试用的脚本化 LLM。
pub fn make_scripted_llm(answers: Vec<String>) -> Arc<dyn Llm> {
    Arc::new(ScriptedLlm::new(answers))
}

/// LLM 的异步调用 future。
pub type AgentLlmFuture<'a> = LlmFuture<'a>;

/// 把一个立即返回的 future 包装成 `LlmFuture`。
pub fn boxed_ready<T: Send + 'static>(value: T) -> Pin<Box<dyn Future<Output = T> + Send>> {
    Box::pin(async move { value })
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

    #[tokio::test]
    async fn scripted_llm_records_calls() {
        let llm = ScriptedLlm::new(vec!["答案".to_string()]);
        llm.chat("system 提示", "user 提示").await.unwrap();

        let calls = llm.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "system 提示");
    }

    #[test]
    fn production_llm_disables_reasoning_by_default() {
        let options = LlmOptions::without_reasoning(0.0, 2048);
        assert!(options.disable_reasoning);
    }
}
