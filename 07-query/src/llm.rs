//! LLM 抽象层 —— 把「调用模型」收敛成一个 trait。
//!
//! ## 为什么要有这一层
//!
//! 查询改写的每一种策略都要调 LLM，而 LLM 调用有三个现实问题：
//!
//! 1. **慢且贵** —— 一次改写就是一次完整的模型调用（几百毫秒到几秒）
//! 2. **不确定** —— 同样的输入，输出可能不一样
//! 3. **会挂** —— 网络抖动、上游限流、鉴权失败
//!
//! 前两个问题让「测试改写逻辑」变得不可能，第三个让「服务」容易整体崩掉。
//!
//! 所以定义一个 trait，把 LLM 换成可控的替身：
//!
//! ```text
//! 生产：GenaiLlm      → 真的调 DeepSeek / OpenAI
//! 测试：ScriptedLlm   → 按脚本返回预设答案，零网络、零延迟、完全确定
//! ```
//!
//! 这样**所有改写策略的逻辑（提示词怎么写、输出怎么解析、失败怎么降级）
//! 都能被确定性地测试**，这正是这个项目最需要的性质。

use std::future::Future;
use std::pin::Pin;

use genai::chat::{ChatMessage, ChatOptions, ChatRequest};
use genai::Client;

use crate::error::{QueryError, QueryResult};

/// LLM 调用参数。
#[derive(Debug, Clone)]
pub struct LlmOptions {
    /// 采样温度。
    ///
    /// 改写场景建议**低温度**（0.0~0.3）—— 改写不是创作，
    /// 是「换个说法」，不需要多样性。温度高会得到离谱的改写结果。
    pub temperature: f64,

    /// 最大输出 token。
    ///
    /// HyDE 要生成一整段假想文档（几百 token），Multi-Query 要生成多个问法，
    /// 所以这个值要比普通问答大。
    pub max_tokens: u32,
}

impl Default for LlmOptions {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            max_tokens: 512,
        }
    }
}

/// LLM 的异步调用 future。
///
/// ## 为什么是 `Pin<Box<dyn Future>>` 而不是 `impl Future`
///
/// 因为本模块需要 `Box<dyn Llm>` —— **trait 必须 dyn 兼容**。
///
/// 而 Rust 的两种「返回 future」的写法都**不 dyn 兼容**：
///
/// - `async fn` → 编译期展开成 `-> impl Future`，不是 dyn 兼容
/// - `fn f() -> impl Future` → RPITIT 同样不是 dyn 兼容
///
/// 所以只能显式装箱：
///
/// ```text
/// async fn      →  fn f() -> Pin<Box<dyn Future<Output = T> + Send + '_>>
/// ```
///
/// 这是一个额外的分配（一次 `Box`），相对于一次 LLM 网络调用（几十毫秒）
/// 完全不值得优化。
pub type LlmFuture<'a> = Pin<Box<dyn Future<Output = QueryResult<String>> + Send + 'a>>;

/// LLM trait。
pub trait Llm: Send + Sync {
    /// 一次性对话：给 system + user，返回模型输出文本。
    fn chat<'a>(&'a self, system: &'a str, user: &'a str) -> LlmFuture<'a>;

    /// 模型名（用于日志和响应体）。
    fn model_name(&self) -> &str;
}

/// 基于 `genai` 的真实 LLM 实现。
pub struct GenaiLlm {
    client: Client,
    model: String,
    options: LlmOptions,
}

impl std::fmt::Debug for GenaiLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenaiLlm")
            .field("model", &self.model)
            .finish()
    }
}

impl GenaiLlm {
    /// 创建 LLM 客户端。
    ///
    /// `genai::Client` 内部是 `Arc`，克隆开销很小。
    pub fn new(model: &str, options: LlmOptions) -> Self {
        Self {
            client: Client::builder().build(),
            model: model.to_string(),
            options,
        }
    }

    /// 当前模型名。
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl Llm for GenaiLlm {
    fn chat<'a>(&'a self, system: &'a str, user: &'a str) -> LlmFuture<'a> {
        Box::pin(async move {
            let options = ChatOptions::default()
                .with_temperature(self.options.temperature)
                .with_max_tokens(self.options.max_tokens);

            let request = ChatRequest::new(vec![
                ChatMessage::system(system),
                ChatMessage::user(user),
            ]);

            let response = self
                .client
                .exec_chat(&self.model, request, Some(&options))
                .await
                .map_err(|e| QueryError::Transformation(format!("调用模型失败: {e}")))?;

            Ok(response.into_first_text().unwrap_or_default())
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// 测试用的脚本化 LLM —— 按调用顺序返回预设答案。
///
/// 用于确定性地测试「提示词 → 解析」这段逻辑，不碰网络。
///
/// ## 用法
///
/// ```ignore
/// let llm = ScriptedLlm::new(vec![
///     "改写后的检索式".to_string(),
///     "问法一\n问法二".to_string(),
/// ]);
/// ```
pub struct ScriptedLlm {
    answers: std::sync::Mutex<Vec<String>>,
    calls: std::sync::Mutex<Vec<(String, String)>>,
    fallback: Option<String>,
}

impl ScriptedLlm {
    /// 创建脚本化 LLM。`answers` 按调用顺序依次返回。
    pub fn new(answers: Vec<String>) -> Self {
        Self {
            answers: std::sync::Mutex::new(answers),
            calls: std::sync::Mutex::new(Vec::new()),
            fallback: None,
        }
    }

    /// 创建脚本化 LLM，所有调用都返回同一个答案。
    pub fn always(answer: &str) -> Self {
        Self {
            answers: std::sync::Mutex::new(Vec::new()),
            calls: std::sync::Mutex::new(Vec::new()),
            fallback: Some(answer.to_string()),
        }
    }

    /// 记录下来的调用（system, user）。
    pub fn calls(&self) -> Vec<(String, String)> {
        self.calls.lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// 调用次数。
    pub fn call_count(&self) -> usize {
        self.calls.lock().map(|c| c.len()).unwrap_or(0)
    }
}

impl Llm for ScriptedLlm {
    fn chat<'a>(&'a self, system: &'a str, user: &'a str) -> LlmFuture<'a> {
        let answer = {
            let mut calls = self.calls.lock().expect("calls 锁中毒");
            calls.push((system.to_string(), user.to_string()));

            let mut answers = self.answers.lock().expect("answers 锁中毒");
            if answers.is_empty() {
                self.fallback.clone()
            } else {
                Some(answers.remove(0))
            }
        };

        Box::pin(async move {
            answer.ok_or_else(|| {
                QueryError::Transformation("脚本化 LLM 没有更多预设答案".to_string())
            })
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
        let llm = ScriptedLlm::new(vec!["一".to_string(), "二".to_string()]);

        assert_eq!(llm.chat("s", "u").await.unwrap(), "一");
        assert_eq!(llm.chat("s", "u").await.unwrap(), "二");
        assert!(llm.chat("s", "u").await.is_err());
        assert_eq!(llm.call_count(), 3);
    }

    #[tokio::test]
    async fn scripted_llm_records_calls() {
        let llm = ScriptedLlm::always("答案");
        llm.chat("system 提示", "user 提示").await.unwrap();

        let calls = llm.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "system 提示");
        assert_eq!(calls[0].1, "user 提示");
    }

    #[tokio::test]
    async fn scripted_llm_falls_back_when_exhausted() {
        let llm = ScriptedLlm::always("固定答案");
        assert_eq!(llm.chat("s", "u").await.unwrap(), "固定答案");
        assert_eq!(llm.chat("s", "u").await.unwrap(), "固定答案");
    }
}
