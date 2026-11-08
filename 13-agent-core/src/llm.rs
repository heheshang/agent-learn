//! LLM 抽象层 —— 把「调用模型」收敛成一个 trait。
//!
//! 复用 07-query 的 `Llm` trait（`rag_07::llm::Llm`），它已经有：
//!
//! - `GenaiLlm` —— 真实模型（DeepSeek / OpenAI 兼容端点）
//! - `ScriptedLlm` —— 按脚本返回预设答案，零网络、零延迟、完全确定
//!
//! ## 为什么 Agent 循环**必须**有可替换的 LLM
//!
//! Agent 循环是一个**状态机**：每一步的输出决定下一步的输入。
//! 如果 LLM 不可替换，测试就只能靠「真的调一次模型」——
//! 那意味着：
//!
//! 1. 测试**慢**（每次几秒）
//! 2. 测试**不确定**（模型输出有随机性）
//! 3. 测试**要花钱**（每次都调 API）
//! 4. 测试**测不了边界**（比如「模型连调 6 次工具」这条路径，
//!    靠真实模型根本触发不了）
//!
//! 有了 `ScriptedLlm`，这些全部变成**确定性的单元测试**。
//!
//! 这也是仓库里贯穿的一条原则：
//! **依赖构造时注入，不搞全局单例**。

use agent_logging::info;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub use rag_07::llm::{GenaiLlm, Llm, LlmFuture, LlmOptions, ScriptedLlm};

/// 创建一个生产用的 LLM（真实模型）。
///
/// ## ⚠️ 关于 `max_tokens` 和推理模型
///
/// 复用 07 的结论：**推理模型的思考 token 和正文共用 `max_tokens` 预算**。
/// Agent 的每一步输出很短（`Thought/Action/Action Input`），
/// 但如果模型开了思考，1024 的预算可能被思考吃光，
/// 导致正文为空 → 解析失败 → 走 `Unstructured` 分支。
///
/// 所以这里默认**关掉思考**：Agent 的每一步都是短输出，
/// 推理收益极低，而风险很高。
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
///
/// `answers` 按调用顺序依次返回。
pub fn make_scripted_llm(answers: Vec<String>) -> Arc<dyn Llm> {
    Arc::new(ScriptedLlm::new(answers))
}

/// LLM 的异步调用 future。
pub type AgentLlmFuture<'a> = LlmFuture<'a>;

/// 把一个立即返回的 future 包装成 `LlmFuture`。
///
/// 便于在测试里写「永远返回同一个答案」的假 LLM。
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
        // Agent 的每一步输出都很短 —— 思考模式会把 max_tokens 吃光。
        // 这个行为由 `LlmOptions::without_reasoning` 保证。
        let options = LlmOptions::without_reasoning(0.0, 1024);
        assert!(options.disable_reasoning);
    }
}
