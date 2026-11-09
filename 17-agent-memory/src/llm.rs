//! LLM 抽象层 —— 支持「多轮消息历史」的对话调用。
//!
//! ## 为什么不能直接用 07/13 的 `Llm` trait
//!
//! 那两个项目的 `Llm` trait 签名是：
//!
//! ```ignore
//! fn chat<'a>(&'a self, system: &'a str, user: &'a str) -> LlmFuture<'a>;
//! ```
//!
//! 它**只能发 system + user 两条消息** —— 拿不到完整的历史。
//!
//! 而本项目的核心恰恰是**带着历史去问模型**：
//! 「我记得你叫 Bob，我叫你什么？」
//!
//! 所以签名必须是消息数组：
//!
//! ```ignore
//! fn chat_messages<'a>(
//!     &'a self,
//!     system: &'a str,
//!     messages: &'a [MemoryMessage],
//! ) -> ChatLlmFuture<'a>;
//! ```
//!
//! ## 与 16 的 `FunctionLlm` 的区别
//!
//! 16 的 `FunctionLlm` 还带 `tools` 和 `tool_choice`。
//! 本项目**不需要工具** —— 记忆项目的主角是「状态管理」。
//!
//! 但我们**复用 16 的 `genai` 消息类型**（`ChatMessage` / `ContentPart`），
//! 这样「思考内容回传」等 16 踩过的坑自然被继承。
//!
//! ## 测试替身
//!
//! [`ScriptedChatLlm`] 按脚本返回预设响应 ——
//! 零网络、零延迟、完全确定。
//! 整个记忆层（含线程隔离、恢复、召回）都能被单元测试锁住。

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use agent_logging::{debug, warn};

use crate::error::{AppError, AppResult};
use crate::state::MemoryMessage;

/// LLM 调用的 future。
pub type ChatLlmFuture<'a> = Pin<Box<dyn Future<Output = AppResult<String>> + Send + 'a>>;

/// 支持消息历史的 LLM trait。
pub trait ChatLlm: Send + Sync {
    /// 带历史的一次对话。
    ///
    /// ## 参数
    ///
    /// - `system` —— 系统提示（每次调用都重新拼，因为可能含召回的记忆）
    /// - `messages` —— **完整**的历史（user / assistant / tool / fact）
    ///
    /// ## ⚠️ 历史里可能包含 `fact` 角色
    ///
    /// 长期记忆召回的事实会混在历史里一起发出去。
    /// 这一层不需要区分 —— 转换在 [`MemoryMessage::to_genai`] 里做。
    fn chat_messages<'a>(
        &'a self,
        system: &'a str,
        messages: &'a [MemoryMessage],
    ) -> ChatLlmFuture<'a>;

    /// 模型名（用于日志和响应体）。
    fn model_name(&self) -> &str;
}

/// 基于 `genai` 的真实 LLM 实现。
pub struct GenaiChatLlm {
    client: genai::Client,
    model: String,
    temperature: f64,
    max_tokens: u32,
    disable_reasoning: bool,
}

impl std::fmt::Debug for GenaiChatLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenaiChatLlm")
            .field("model", &self.model)
            .field("disable_reasoning", &self.disable_reasoning)
            .finish()
    }
}

impl GenaiChatLlm {
    /// 创建 LLM 客户端。
    pub fn new(model: &str, temperature: f64, max_tokens: u32, disable_reasoning: bool) -> Self {
        Self {
            client: genai::Client::builder().build(),
            model: model.to_string(),
            temperature,
            max_tokens,
            disable_reasoning,
        }
    }
}

impl ChatLlm for GenaiChatLlm {
    fn chat_messages<'a>(
        &'a self,
        system: &'a str,
        messages: &'a [MemoryMessage],
    ) -> ChatLlmFuture<'a> {
        Box::pin(async move {
            let mut options = genai::chat::ChatOptions::default()
                .with_temperature(self.temperature)
                .with_max_tokens(self.max_tokens);

            // ⚠️ 关思考。
            //
            // 推理模型的「思考 token」和「正文 token」**共用** `max_tokens` 预算，
            // 思考没结束正文就是空的。
            //
            // 对记忆项目来说，思考更是**纯浪费**：
            // 「记住我的名字」这类任务不需要推理，
            // 而且每轮都要把历史重发一遍，token 本来就紧张。
            if self.disable_reasoning {
                options = options.with_reasoning_effort(genai::chat::ReasoningEffort::Zero);
            }

            // 历史消息转成 `genai` 消息。
            let genai_messages: Vec<genai::chat::ChatMessage> =
                messages.iter().map(|m| m.to_genai()).collect();

            let mut request = genai::chat::ChatRequest::new(genai_messages);
            if !system.trim().is_empty() {
                request = request.with_system(system.to_string());
            }

            debug!(
                model = %self.model,
                system_len = system.chars().count(),
                messages = messages.len(),
                "发起带历史的 LLM 调用"
            );

            let response = self
                .client
                .exec_chat(&self.model, request, Some(&options))
                .await
                .map_err(|e| {
                    warn!(model = %self.model, error = %e, "LLM 调用失败");
                    AppError::Llm(format!("调用模型失败: {e}"))
                })?;

            let text = response.first_text().unwrap_or_default().to_string();

            debug!(
                model = %self.model,
                answer_len = text.chars().count(),
                "LLM 响应完成"
            );

            Ok(text)
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// 测试用的脚本化 LLM —— 按调用顺序返回预设响应。
///
/// ## 为什么不用 07 的 `ScriptedLlm`
///
/// 07 的 `ScriptedLlm` 是 `chat(system, user)` 两参数签名 ——
///
/// 本项目的历史可能有 20 条消息，测试要能**断言**「模型看到了第 3 轮的那句话」。
///
/// 所以本项目的脚本化 LLM 会**记录每次调用看到的完整历史**，
/// 测试可以断言 `calls()[0].messages[2].content == "我叫 Bob"`。
pub struct ScriptedChatLlm {
    steps: Mutex<Vec<String>>,
    calls: Mutex<Vec<ScriptedCall>>,
    fallback: Option<String>,
}

/// 记录下来的一次调用。
#[derive(Debug, Clone)]
pub struct ScriptedCall {
    /// 系统提示。
    pub system: String,

    /// 看到的完整历史。
    pub messages: Vec<MemoryMessage>,
}

impl ScriptedChatLlm {
    /// 创建脚本化 LLM（按调用顺序返回）。
    pub fn new(steps: Vec<String>) -> Self {
        Self {
            steps: Mutex::new(steps),
            calls: Mutex::new(Vec::new()),
            fallback: None,
        }
    }

    /// 创建脚本化 LLM，所有调用都返回同一结果。
    pub fn always(answer: &str) -> Self {
        Self {
            steps: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
            fallback: Some(answer.to_string()),
        }
    }

    /// 记录下来的调用。
    pub fn calls(&self) -> Vec<ScriptedCall> {
        self.calls.lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// 调用次数。
    pub fn call_count(&self) -> usize {
        self.calls.lock().map(|c| c.len()).unwrap_or(0)
    }
}

impl ChatLlm for ScriptedChatLlm {
    fn chat_messages<'a>(
        &'a self,
        system: &'a str,
        messages: &'a [MemoryMessage],
    ) -> ChatLlmFuture<'a> {
        let answer = {
            let mut calls = self.calls.lock().expect("calls 锁中毒");
            calls.push(ScriptedCall {
                system: system.to_string(),
                messages: messages.to_vec(),
            });

            let mut steps = self.steps.lock().expect("steps 锁中毒");
            if steps.is_empty() {
                self.fallback.clone()
            } else {
                Some(steps.remove(0))
            }
        };

        Box::pin(async move {
            answer.ok_or_else(|| AppError::Llm("脚本化 LLM 没有更多预设响应".to_string()))
        })
    }

    fn model_name(&self) -> &str {
        "scripted"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn scripted_llm_returns_answers_in_order() {
        let llm = ScriptedChatLlm::new(vec!["一".to_string(), "二".to_string()]);

        assert_eq!(llm.chat_messages("s", &[]).await.unwrap(), "一");
        assert_eq!(llm.chat_messages("s", &[]).await.unwrap(), "二");
        assert!(llm.chat_messages("s", &[]).await.is_err());
    }

    #[tokio::test]
    async fn scripted_llm_records_the_full_history() {
        // ⚠️ 这是记忆项目测试的关键能力 ——
        //    能断言「模型看到了第几轮的什么」。
        let llm = ScriptedChatLlm::always("答案");
        let history = vec![
            MemoryMessage::user("我叫 Bob"),
            MemoryMessage::assistant("你好 Bob"),
            MemoryMessage::user("我叫什么"),
        ];

        llm.chat_messages("系统提示", &history).await.unwrap();

        let calls = llm.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].system, "系统提示");
        assert_eq!(calls[0].messages.len(), 3);
        assert_eq!(calls[0].messages[0].content, "我叫 Bob");
        assert_eq!(calls[0].messages[2].content, "我叫什么");
    }

    #[tokio::test]
    async fn scripted_llm_falls_back_when_exhausted() {
        let llm = ScriptedChatLlm::always("固定答案");
        assert_eq!(llm.chat_messages("s", &[]).await.unwrap(), "固定答案");
        assert_eq!(llm.chat_messages("s", &[]).await.unwrap(), "固定答案");
        assert_eq!(llm.call_count(), 2);
    }

    #[test]
    fn model_name_is_reported() {
        let llm = ScriptedChatLlm::always("x");
        assert_eq!(llm.model_name(), "scripted");
    }
}
