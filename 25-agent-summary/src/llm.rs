//! **LLM 抽象层** —— 把「调用模型」收敛成一个 trait。
//!
//! # 为什么 25 需要这一层
//!
//! 25 原本是**零 LLM 依赖**的纯 CPU 压测台。
//! 后果是 `agent::parse_react_output` 解析的永远是
//! **代码自己拼出来的字符串** ——
//!
//! ```ignore
//! let raw = format!("Thought: ...\nAction: module_stats\nAction Input: {{...}}");
//! ```
//!
//! 于是「这个解析器在**真实模型**输出下能不能工作」**没有证据**。
//! 而模型输出恰恰是 Agent 系统里**最不可控**的部分：
//! 会回显提示词、会同时给 `Action:` 和 `Final Answer:`、
//! 会把参数写成字符串而不是对象。
//!
//! 这一层把模型换成可控替身，让**整个 Agent 循环**能被确定性地测。
//!
//! # 两级抽象
//!
//! | trait | 用途 | 消息形状 |
//! |---|---|---|
//! | [`Llm`] | **ReAct 文本协议**（13 的路子） | system + user → 纯文本 |
//! | [`FunctionLlm`] | **结构化 function calling**（16 的路子） | 消息数组 + 工具定义 → 完整响应 |
//!
//! ## ⚠️ 为什么不能只留一个
//!
//! 两者**协议不同**，不是实现细节的差异：
//!
//! - ReAct 靠**正则解析**模型输出的纯文本 —— 解析器是「最易错的部分」
//! - Function Calling 靠**结构化字段** —— 不用解析，但必须处理
//!   「每个 `tool_call` 都要有回执」「`reasoning_content` 要回传」
//!
//! 25 同时保留两条，才能做**协议对照**：
//! 「文本协议 vs 结构化协议，在同一条链路上差多少」。
//!
//! # 三条硬约定
//!
//! ## ① 思考模式必须**显式**关掉
//!
//! 推理模型的「思考 token」和「正文 token」**共用** `max_tokens` 预算 ——
//! 思考没结束正文就是空的。
//!
//! 翻译成 genai 是 `ReasoningEffort::Zero`（DeepSeek 侧即
//! `thinking: {type: "disabled"}`）。
//!
//! ## ② 失败的 LLM 调用**不能**降级成观察
//!
//! 「没有模型就没有答案」—— 模型调用失败是 **502**，
//! 不是 400，也不是「让模型自己纠正」。
//!
//! ## ③ 测试替身必须**模拟真实行为的全部影响源**
//!
//! 只模拟「正在修的那一个」会让单测结论与真实模型**相反**
//! —— 这是 21-agent-cache 实测踩过的坑。
//! 所以 [`ScriptedLlm`] 同时模拟「措辞变化」和「参数形状变化」。

use agent_logging::{debug, warn};
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use serde_json::Value;

use crate::error::{AppError, AppResult};

/// LLM 调用参数。
#[derive(Debug, Clone)]
pub struct LlmOptions {
    /// 采样温度。
    ///
    /// 工具调用场景建议**低温度**（0.0~0.3）——
    /// 工具调用不是创作，是「按 Schema 填参数」。
    pub temperature: f64,

    /// 最大输出 token。
    ///
    /// ⚠️ 推理模型的思考 token 与正文**共用**这个额度 ——
    /// 调小会让正文被思考吃光（`finish_reason = length`，
    /// 正文长度 0）。
    pub max_tokens: u32,

    /// 是否关闭模型的「思考」模式。
    pub disable_reasoning: bool,

    /// ★ 是否请求 **JSON 模式**输出。
    ///
    /// ## ⚠️ `json_mode` 只保证「输出是 JSON」，**不保证符合你的 Schema**
    ///
    /// 字段名错了、类型错了、多了字段，它照样输出。
    /// 所以**本地 Schema 校验仍然是必需品** ——
    /// 见 [`crate::structured::parse_summary`]。
    ///
    /// ⚠️ 另外 `json_mode` 要求提示词里**明确说「输出 JSON」**，
    /// 否则部分 provider 会拒绝（genai 的 `JsonModeWithoutInstruction`）。
    pub json_mode: bool,

    /// ★ 提示词缓存路由键（`prompt_cache_key`）。
    ///
    /// ## ⚠️ 为什么 Agent 循环**特别需要**它
    ///
    /// 21-agent-cache 实测的核心结论：
    /// **中间轮不写缓存 → 模型每次措辞不同 → 喂给下一轮的历史不同
    /// → 下一轮的请求前缀也变了 → provider 侧的 prompt cache 也命中不了**。
    ///
    /// 本项目把「稳定的前缀」（system 提示 + 工具定义）单独编成一个
    /// **稳定的键**，provider 可以据此把同一段前缀路由到同一个缓存分片。
    ///
    /// ⚠️ **它不会让「措辞不同」变得相同** ——
    /// 那是 21 的本地缓存要解决的问题。这里解决的是
    /// 「**前缀相同但路由不同**」这一层。
    ///
    /// ⚠️ 不是所有 provider 都支持这个参数 ——
    /// genai 的 OpenAI 适配器会带上它，DeepSeek 会**忽略**它（无害）。
    pub prompt_cache_key: Option<String>,
}

impl Default for LlmOptions {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            max_tokens: 2048,
            // ⚠️ 默认**关**思考 —— 思考模式会额外要求
            //    `reasoning_content` 回传（少一条就 400），
            //    对「测 Agent 循环本身」这个目标来说变量越少越好。
            disable_reasoning: true,
            json_mode: false,
            prompt_cache_key: None,
        }
    }
}

impl LlmOptions {
    /// 构造「关闭思考」的参数。
    pub fn without_reasoning(temperature: f64, max_tokens: u32) -> Self {
        Self {
            temperature,
            max_tokens,
            disable_reasoning: true,
            json_mode: false,
            prompt_cache_key: None,
        }
    }

    /// ★ 开启 **JSON 模式**。
    ///
    /// 用于「研究 → 成文」两阶段里的**成文阶段**：
    /// 模型必须输出一个符合 Schema 的 JSON 对象。
    pub fn with_json_mode(mut self, enabled: bool) -> Self {
        self.json_mode = enabled;
        self
    }

    /// ★ 设置提示词缓存路由键。
    ///
    /// ⚠️ 键**必须稳定** —— 每次调用都换键等于「告诉 provider 别缓存」。
    pub fn with_prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }
}

/// LLM 的异步调用 future。
///
/// ## ⚠️ 为什么是 `Pin<Box<dyn Future>>` 而不是 `impl Future`
///
/// 因为本模块需要 `Box<dyn Llm>` —— **trait 必须 dyn 兼容**。
/// 而 Rust 的两种「返回 future」的写法都**不 dyn 兼容**：
///
/// - `async fn` → 编译期展开成 `-> impl Future`，不是 dyn 兼容
/// - `fn f() -> impl Future` → RPITIT 同样不是 dyn 兼容
///
/// 所以只能显式装箱。这是一次 `Box` 分配，
/// 相对于一次 LLM 网络调用（几十毫秒到几秒）完全不值得优化。
pub type LlmFuture<'a> = Pin<Box<dyn Future<Output = AppResult<LlmReply>> + Send + 'a>>;

/// ★ 一次调用的 token 用量。
///
/// ## ⚠️ 为什么要有它
///
/// 「耗时」不是一个**好度量** —— 它受上游负载、网络抖动影响，
/// 同一次实验前后差 200ms 就可能让结论翻转。
///
/// token 数是**确定性**的：同样的输入必然产生同样的 prompt token。
/// 所以 token 是比耗时更稳定的归因依据。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LlmUsage {
    /// 输入 token。
    pub prompt_tokens: u32,

    /// 输出 token。
    pub completion_tokens: u32,

    /// 总 token（provider 没报时用「输入 + 输出」兜底）。
    pub total_tokens: u32,
}

impl LlmUsage {
    /// 累加。
    pub fn merge(&mut self, other: &LlmUsage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// ★ 一次 LLM 调用的结果。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LlmReply {
    /// 模型输出的文本。
    pub text: String,

    /// token 用量。
    pub usage: LlmUsage,

    /// ★ 输出是否被 `max_tokens` **截断**了。
    ///
    /// ## ⚠️ 为什么必须单独拎出来
    ///
    /// 截断的答案看起来是**完整的** —— 它有头有尾，只是少了尾巴。
    /// 不检查这个字段，就会把「被截断的半截结论」当成正常答案返回，
    /// 而且**症状极隐蔽**：数字少了，但读起来完全通顺。
    ///
    /// genai 的 `StopReason::is_max_tokens()` 就是这个判据。
    pub truncated: bool,
}

/// LLM trait —— ReAct 文本协议。
pub trait Llm: Send + Sync {
    /// 一次性对话：给 system + user，返回模型输出文本。
    fn chat<'a>(&'a self, system: &'a str, user: &'a str) -> LlmFuture<'a>;

    /// 模型名（用于日志和响应体）。
    fn model_name(&self) -> &str;
}

/// 基于 `genai` 的真实 LLM 实现。
pub struct GenaiLlm {
    client: genai::Client,
    model: String,
    options: LlmOptions,
}

impl std::fmt::Debug for GenaiLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenaiLlm")
            .field("model", &self.model)
            .field("disable_reasoning", &self.options.disable_reasoning)
            .finish()
    }
}

impl GenaiLlm {
    /// 创建 LLM 客户端。
    pub fn new(model: &str, options: LlmOptions) -> Self {
        Self {
            client: genai::Client::builder().build(),
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
            let mut options = genai::chat::ChatOptions::default()
                .with_temperature(self.options.temperature)
                .with_max_tokens(self.options.max_tokens);

            // ⚠️ 关思考。
            //
            // 不能靠「模型名后缀」那套 —— `deepseek-flash` 结尾是 `flash`，
            // 不是推理强度关键字。这里必须**显式**设置。
            //
            // `ReasoningEffort::Zero` 在 DeepSeek 适配器里会被翻译成
            // `thinking: {"type": "disabled"}`。
            //
            // ⚠️ **只在需要关闭时才设置** —— 一旦设成 `Some(effort)`，
            // genai 就不再走「模型默认」那条路，会把推理强度**显式**传给厂商。
            if self.options.disable_reasoning {
                options = options.with_reasoning_effort(genai::chat::ReasoningEffort::Zero);
            }

            // ★ JSON 模式 —— 成文阶段用它把输出约束成 JSON。
            if self.options.json_mode {
                options = options.with_response_format(genai::chat::ChatResponseFormat::JsonMode);
            }

            // ★ 提示词缓存路由键（provider 支持时才生效）。
            if let Some(key) = &self.options.prompt_cache_key {
                options = options.with_prompt_cache_key(key.clone());
            }

            let request = genai::chat::ChatRequest::new(vec![
                genai::chat::ChatMessage::system(system),
                genai::chat::ChatMessage::user(user),
            ]);

            debug!(
                model = %self.model,
                system_len = system.chars().count(),
                user_len = user.chars().count(),
                disable_reasoning = self.options.disable_reasoning,
                json_mode = self.options.json_mode,
                "发起 LLM 调用"
            );

            let response = self
                .client
                .exec_chat(&self.model, request, Some(&options))
                .await
                .map_err(|e| {
                    warn!(model = %self.model, error = %e, "LLM 调用失败");
                    AppError::BadGateway(format!("调用模型失败: {e}"))
                })?;

            // ⚠️ **先取字段再消费 `response`** ——
            //    `into_first_text()` 会把 `response` **move 掉**，
            //    之后再读 `usage` / `stop_reason` 就是「use of moved value」。
            //
            //    这是 2021 edition 的 `Into` 惯用法最容易踩的坑：
            //    看起来「先转文本再读统计」很自然，
            //    但 `into_*` 系列方法都是**消耗性**的。
            let text = response.first_text().unwrap_or_default().to_string();
            let usage = usage_of(&response);
            let truncated = response
                .stop_reason
                .as_ref()
                .is_some_and(|r| r.is_max_tokens());

            let reply = LlmReply {
                text,
                usage,
                truncated,
            };

            if reply.truncated {
                // ⚠️ 必须 `warn!` —— 截断的答案看起来是完整的，
                //    静默返回会让人以为「就差一点」。
                warn!(
                    model = %self.model,
                    "模型输出被 max_tokens 截断 —— 答案可能不完整"
                );
            }

            Ok(reply)
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// ★ 从 genai 的响应里提取 token 用量。
///
/// ## ⚠️ provider 常常**不报** usage
///
/// 那样 `total` 要用「输入 + 输出」兜底 ——
/// 宁可给一个**估算值**，也不要给 0（0 会被误读成「没花 token」）。
pub(crate) fn usage_of(response: &genai::chat::ChatResponse) -> LlmUsage {
    let prompt = response.usage.prompt_tokens.unwrap_or_default().max(0) as u32;
    let completion = response.usage.completion_tokens.unwrap_or_default().max(0) as u32;
    let total = response
        .usage
        .total_tokens
        .map(|v| v.max(0) as u32)
        .unwrap_or(prompt + completion);

    LlmUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: total,
    }
}

/// 测试用的脚本化 LLM —— 按调用顺序返回预设答案。
///
/// ## ⚠️ 它模拟的不只是「答案」
///
/// 21-agent-cache 实测踩过的坑：
/// 假 LLM 原本只模拟「随机 `call_id`」不模拟「措辞变化」，
/// 导致**单测结论与真实模型相反**（单测说归一化后能命中，真实是 0 命中）。
///
/// 所以这里的预设项刻意分成几个字段，让测试能表达多种真实行为。
///
/// ⚠️ 同样地，它也模拟 **token 用量**与**截断**——
/// 因为这两个都会影响 Agent 循环的决策（截断的答案不该被当成完整答案）。
pub struct ScriptedLlm {
    answers: Mutex<Vec<String>>,
    calls: Mutex<Vec<ScriptedTextCall>>,
    fallback: Option<String>,
    usage: LlmUsage,
    truncated: bool,
}

/// 记录下来的一次调用（供测试断言）。
#[derive(Debug, Clone)]
pub struct ScriptedTextCall {
    /// system 提示。
    pub system: String,

    /// user 提示。
    pub user: String,
}

impl ScriptedLlm {
    /// 创建脚本化 LLM。`answers` 按调用顺序依次返回。
    pub fn new(answers: Vec<String>) -> Self {
        Self {
            answers: Mutex::new(answers),
            calls: Mutex::new(Vec::new()),
            fallback: None,
            usage: LlmUsage::default(),
            truncated: false,
        }
    }

    /// 创建脚本化 LLM，所有调用都返回同一个答案。
    pub fn always(answer: &str) -> Self {
        Self {
            answers: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
            fallback: Some(answer.to_string()),
            usage: LlmUsage::default(),
            truncated: false,
        }
    }

    /// ★ 设定每次调用上报的 token 用量。
    pub fn with_usage(mut self, usage: LlmUsage) -> Self {
        self.usage = usage;
        self
    }

    /// ★ 设定每次调用都被**截断**。
    ///
    /// 用于测「截断的答案必须被标记出来」。
    pub fn truncated(mut self) -> Self {
        self.truncated = true;
        self
    }

    /// 记录下来的调用。
    pub fn calls(&self) -> Vec<ScriptedTextCall> {
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
            calls.push(ScriptedTextCall {
                system: system.to_string(),
                user: user.to_string(),
            });

            let mut answers = self.answers.lock().expect("answers 锁中毒");
            if answers.is_empty() {
                self.fallback.clone()
            } else {
                Some(answers.remove(0))
            }
        };

        Box::pin(async move {
            let text = answer
                .ok_or_else(|| AppError::BadGateway("脚本化 LLM 没有更多预设答案".to_string()))?;
            Ok(LlmReply {
                text,
                usage: self.usage,
                truncated: self.truncated,
            })
        })
    }

    fn model_name(&self) -> &str {
        "scripted"
    }
}

// ─────────────────────────────────────────────────────────────────────
// 结构化输出（JSON Schema 约束）
// ─────────────────────────────────────────────────────────────────────

/// ★ 结构化输出模式的 LLM 调用 future。
pub type StructuredLlmFuture<'a> = Pin<Box<dyn Future<Output = AppResult<LlmReply>> + Send + 'a>>;

/// ★ 支持**结构化输出**的 LLM trait。
///
/// ## 与 [`FunctionLlm`] 的区别
///
/// | | `FunctionLlm` | `StructuredLlm` |
/// |---|---|---|
/// | 目的 | 让模型**调工具** | 让模型**按 Schema 回答** |
/// | 约束来源 | `tools` 字段 | `response_format` 字段 |
/// | 是否多轮 | 是（要回传 `tool` 消息） | **否**（一次成文） |
///
/// ## ★ 为什么成文阶段**不挂工具**
///
/// 19-agent-summary 实测：模型输出 JSON 外面包 ` ```json ` 围栏，
/// 前言里带 `{周报}` 这类占位花括号 ——
/// 「找第一个 `{` 到最后一个 `}`」会抓到 `{周报}`。
///
/// 正确做法是**逐字符扫配对括号（跳过字符串字面量）后逐个试解析**，
/// 判据是「能 `from_str` 成功」而不是「括号配对」。
/// 这个逻辑在 [`crate::agent::extract_json_object`]。
pub trait StructuredLlm: Send + Sync {
    /// ★ 按 JSON Schema 生成一个结构化回答。
    ///
    /// `schema` 是**期望**的 Schema；
    /// provider 若支持 `json_schema`（OpenAI / Anthropic / Gemini）会**强制**它，
    /// 只支持 `json_object` 的（DeepSeek）则退化成「保证是 JSON」，
    /// **仍需本地校验**（[`crate::structured::parse_summary`]）。
    fn structured<'a>(
        &'a self,
        system: &'a str,
        user: &'a str,
        schema: &'a Value,
    ) -> StructuredLlmFuture<'a>;

    /// 模型名。
    fn model_name(&self) -> &str;
}

/// ★ 真实结构化输出实现（基于 `genai`）。
pub struct GenaiStructuredLlm {
    client: genai::Client,
    model: String,
    options: LlmOptions,
}

impl std::fmt::Debug for GenaiStructuredLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenaiStructuredLlm")
            .field("model", &self.model)
            .field("json_mode", &self.options.json_mode)
            .finish()
    }
}

impl GenaiStructuredLlm {
    /// 创建结构化输出客户端。
    pub fn new(model: &str, options: LlmOptions) -> Self {
        Self {
            client: genai::Client::builder().build(),
            model: model.to_string(),
            options,
        }
    }
}

impl StructuredLlm for GenaiStructuredLlm {
    fn structured<'a>(
        &'a self,
        system: &'a str,
        user: &'a str,
        schema: &'a Value,
    ) -> StructuredLlmFuture<'a> {
        Box::pin(async move {
            let mut options = genai::chat::ChatOptions::default()
                .with_temperature(self.options.temperature)
                .with_max_tokens(self.options.max_tokens);

            if self.options.disable_reasoning {
                options = options.with_reasoning_effort(genai::chat::ReasoningEffort::Zero);
            }

            // ★ 提示词缓存路由键 —— 成文阶段的前缀（system + 证据）是稳定的。
            if let Some(key) = &self.options.prompt_cache_key {
                options = options.with_prompt_cache_key(key.clone());
            }

            // ★★ **provider 能力分层** —— 这是本模块最关键的设计。
            //
            // genai 的适配器会**自动**做方言改写：
            //
            // - OpenAI / Anthropic / Gemini → `response_format: {type: "json_schema", ...}`
            //   并且用 `sanitize_json_schema` 按 provider 方言改写 Schema
            //   （所有字段进 `required`、递归补 `additionalProperties: false`、
            //   内联 `$ref`）。
            // - DeepSeek（走 OpenAI 适配器）→ 只支持 `json_object`，
            //   **不支持** `json_schema`。
            //
            // ⚠️ 如果无条件传 `JsonSpec`，DeepSeek 侧会拿到
            //    `response_format: {"type": "json_schema"}` —— 它不认，
            //    可能直接报错，或者**静默忽略**（那就等于没约束）。
            //
            // 所以本项目**显式**区分：DeepSeek 用 `JsonMode`，
            // 其余用 `JsonSpec`。
            let response_format = if self.model.contains("deepseek") {
                genai::chat::ChatResponseFormat::JsonMode
            } else {
                genai::chat::ChatResponseFormat::JsonSpec(genai::chat::JsonSpec::new(
                    "summary",
                    schema.clone(),
                ))
            };
            options = options.with_response_format(response_format);

            let request = genai::chat::ChatRequest::new(vec![
                genai::chat::ChatMessage::system(system),
                genai::chat::ChatMessage::user(user),
            ]);

            debug!(
                model = %self.model,
                strict_schema = !self.model.contains("deepseek"),
                "发起结构化输出调用"
            );

            let response = self
                .client
                .exec_chat(&self.model, request, Some(&options))
                .await
                .map_err(|e| {
                    warn!(model = %self.model, error = %e, "结构化输出调用失败");
                    AppError::BadGateway(format!("调用模型失败: {e}"))
                })?;

            // ⚠️ **先取字段再消费 `response`** ——
            //    `into_first_text()` 会把 `response` **move 掉**，
            //    之后再读 `usage` / `stop_reason` 就是「use of moved value」。
            let text = response.first_text().unwrap_or_default().to_string();
            let usage = usage_of(&response);
            let truncated = response
                .stop_reason
                .as_ref()
                .is_some_and(|r| r.is_max_tokens());

            let reply = LlmReply {
                text,
                usage,
                truncated,
            };

            if reply.truncated {
                // ⚠️ 结构化输出被截断 = JSON **必然不完整** → 解析会失败。
                //    必须 `warn!`，不能静默。
                warn!(model = %self.model, "结构化输出被 max_tokens 截断 —— JSON 可能不完整");
            }

            Ok(reply)
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// 脚本化结构化输出 LLM —— 按脚本返回预设文本。
pub struct ScriptedStructuredLlm {
    answers: Mutex<Vec<String>>,
    calls: Mutex<Vec<ScriptedStructuredCall>>,
    fallback: Option<String>,
}

/// 记录下来的一次结构化调用。
#[derive(Debug, Clone)]
pub struct ScriptedStructuredCall {
    /// system 提示。
    pub system: String,

    /// user 提示。
    pub user: String,

    /// 传入的 Schema。
    pub schema: Value,
}

impl ScriptedStructuredLlm {
    /// 创建脚本化结构化 LLM。
    pub fn new(answers: Vec<String>) -> Self {
        Self {
            answers: Mutex::new(answers),
            calls: Mutex::new(Vec::new()),
            fallback: None,
        }
    }

    /// 创建「所有调用都返回同一答案」的脚本化结构化 LLM。
    pub fn always(answer: &str) -> Self {
        Self {
            answers: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
            fallback: Some(answer.to_string()),
        }
    }

    /// 记录下来的调用。
    pub fn calls(&self) -> Vec<ScriptedStructuredCall> {
        self.calls.lock().map(|c| c.clone()).unwrap_or_default()
    }
}

impl StructuredLlm for ScriptedStructuredLlm {
    fn structured<'a>(
        &'a self,
        system: &'a str,
        user: &'a str,
        schema: &'a Value,
    ) -> StructuredLlmFuture<'a> {
        let answer = {
            let mut calls = self.calls.lock().expect("calls 锁中毒");
            calls.push(ScriptedStructuredCall {
                system: system.to_string(),
                user: user.to_string(),
                schema: schema.clone(),
            });

            let mut answers = self.answers.lock().expect("answers 锁中毒");
            if answers.is_empty() {
                self.fallback.clone()
            } else {
                Some(answers.remove(0))
            }
        };

        Box::pin(async move {
            let text = answer
                .ok_or_else(|| AppError::BadGateway("脚本化 LLM 没有更多预设答案".to_string()))?;
            Ok(LlmReply {
                text,
                usage: LlmUsage::default(),
                truncated: false,
            })
        })
    }

    fn model_name(&self) -> &str {
        "scripted"
    }
}

// ─────────────────────────────────────────────────────────────────────
// 结构化 Function Calling
// ─────────────────────────────────────────────────────────────────────

/// 工具选择策略 —— ★ **直接用 `genai::chat::ToolChoice`**。
///
/// ## ⚠️ 四档的实测结论（16-agent-function-calling）
///
/// | 档位 | 实测行为 |
/// |---|---|
/// | `auto` | **唯一正确的默认档** —— 一次并行拿多个结果，token 最低 |
/// | `none` | ⚠️ **不可靠** —— 模型会用 `<toolcall>{...}</toolcall>` **文本**模拟调用 |
/// | `required` | ⚠️ 诱发**凑数调用**，让模型丧失判断力 |
/// | `forced:名字` | 同 `required` |
///
/// ⚠️ `none` 不可靠的根因：模型在训练时见过海量这种 XML 格式，
/// 提示词里出现工具名它就会模仿。
/// 所以**真正禁止用工具是不把 tools 传给模型** ——
/// [`GenaiFunctionLlm`] 就是这么实现的。
///
/// ## ⚠️ 为什么**不再**自己定义一份枚举
///
/// 本项目原先手写了一个 `ToolChoiceMode`（Auto / None / Required / Forced），
/// 它与 `genai::chat::ToolChoice` **一一对应**，只是多了一套
/// `parse` / `name` 的字符串转换。
///
/// ★ **重复定义的代价不是「多几行代码」，而是「两份语义会漂移」** ——
/// 比如哪天 provider 加了新档位，本地枚举不会跟着更新，
/// 于是「本地说支持、provider 不支持」，
/// 症状是 provider 报一个看不懂的 400。
///
/// 所以现在直接用 crate 的 [`genai::chat::ToolChoice`]，
/// 只保留两个**纯字符串**的辅助函数（[`parse_tool_choice`] /
/// [`tool_choice_name`]）—— 它们是 CLI/配置层的需要，不是语义定义。
pub use genai::chat::ToolChoice;

/// 从字符串解析工具选择策略。
///
/// `forced:工具名` 表示指定函数（对应 crate 的 `ToolChoice::Tool`）。
///
/// ⚠️ 这是**纯字符串转换**，不含任何语义判断 ——
/// 语义由 crate 的 [`genai::chat::ToolChoice`] 定义。
pub fn parse_tool_choice(value: &str) -> Option<ToolChoice> {
    let raw = value.trim();
    match raw.to_ascii_lowercase().as_str() {
        "auto" => Some(ToolChoice::Auto),
        "none" => Some(ToolChoice::None),
        "required" => Some(ToolChoice::Required),
        _ => raw
            .strip_prefix("forced:")
            .map(|name| ToolChoice::tool(name.trim().to_string())),
    }
}

/// 工具选择策略的**显示名**（用于日志与 CLI 输出）。
///
/// ⚠️ 只用于**展示**，不参与任何语义判断。
pub fn tool_choice_name(choice: &ToolChoice) -> &'static str {
    match choice {
        ToolChoice::Auto => "auto",
        ToolChoice::None => "none",
        ToolChoice::Required => "required",
        ToolChoice::Tool { .. } => "forced",
    }
}

/// LLM 调用的 future（结构化版）。
pub type FunctionLlmFuture<'a> =
    Pin<Box<dyn Future<Output = AppResult<genai::chat::ChatResponse>> + Send + 'a>>;

/// 支持 Function Calling 的 LLM trait。
pub trait FunctionLlm: Send + Sync {
    /// 带工具定义的一次对话。
    ///
    /// 返回**完整的** [`genai::chat::ChatResponse`] —— 因为调用方需要：
    ///
    /// - `tool_calls()` —— 结构化的工具调用
    /// - `first_text()` —— 模型同时输出的文本
    /// - `reasoning_content` —— 推理模型的思考内容（**必须回传**，否则 400）
    fn chat_with_tools<'a>(
        &'a self,
        system: &'a str,
        messages: &'a [genai::chat::ChatMessage],
        tools: &'a [crate::tool::ToolSpec],
        tool_choice: ToolChoice,
    ) -> FunctionLlmFuture<'a>;

    /// 模型名。
    fn model_name(&self) -> &str;
}

/// 基于 `genai` 的真实 Function Calling 实现。
pub struct GenaiFunctionLlm {
    client: genai::Client,
    model: String,
    options: LlmOptions,
}

impl std::fmt::Debug for GenaiFunctionLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenaiFunctionLlm")
            .field("model", &self.model)
            .field("disable_reasoning", &self.options.disable_reasoning)
            .finish()
    }
}

impl GenaiFunctionLlm {
    /// 创建 LLM 客户端。
    pub fn new(model: &str, options: LlmOptions) -> Self {
        Self {
            client: genai::Client::builder().build(),
            model: model.to_string(),
            options,
        }
    }
}

impl FunctionLlm for GenaiFunctionLlm {
    fn chat_with_tools<'a>(
        &'a self,
        system: &'a str,
        messages: &'a [genai::chat::ChatMessage],
        tools: &'a [crate::tool::ToolSpec],
        tool_choice: ToolChoice,
    ) -> FunctionLlmFuture<'a> {
        Box::pin(async move {
            let mut options = genai::chat::ChatOptions::default()
                .with_temperature(self.options.temperature)
                .with_max_tokens(self.options.max_tokens);

            if self.options.disable_reasoning {
                options = options.with_reasoning_effort(genai::chat::ReasoningEffort::Zero);
            }

            // ★ 提示词缓存路由键 —— Agent 循环里 system 提示 + 工具定义
            //    是**稳定前缀**，provider 可以据此命中缓存。
            if let Some(key) = &self.options.prompt_cache_key {
                options = options.with_prompt_cache_key(key.clone());
            }

            let mut request =
                genai::chat::ChatRequest::new(messages.to_vec()).with_system(system.to_string());

            // ⚠️★ **`tool_choice = none` 必须**「不传 tools」**来实现 ——
            //    不能只传 `tool_choice: "none"`。
            //
            // 16-agent-function-calling 实测的坑：
            // 传了 `tool_choice: "none"` 但仍把工具列表传过去，
            // 模型会用**文本**模拟调用：
            //
            // ```text
            // <toolcall>{"name":"get_weather"}</toolcall>
            // ```
            //
            // 根因是训练数据里见过海量这种格式 ——
            // **提示词里出现工具名它就会模仿**。
            //
            // 所以「真正禁止用工具」= **把 tools 字段整个去掉**。
            let effective_choice = tool_choice.clone();
            let tools = match effective_choice {
                ToolChoice::None => Vec::new(),
                _ => tools.to_vec(),
            };

            if !tools.is_empty() {
                request = request.with_tools(tools.iter().map(|t| t.to_genai_tool()));
                options = options.with_tool_choice(effective_choice.clone());
            }

            debug!(
                model = %self.model,
                system_len = system.chars().count(),
                messages = messages.len(),
                tools = tools.len(),
                tool_choice = tool_choice_name(&tool_choice),
                "发起带工具的 LLM 调用"
            );

            let response = self
                .client
                .exec_chat(&self.model, request, Some(&options))
                .await
                .map_err(|e| {
                    warn!(model = %self.model, error = %e, "带工具的 LLM 调用失败");
                    AppError::BadGateway(format!("调用模型失败: {e}"))
                })?;

            debug!(
                model = %self.model,
                tool_calls = response.tool_calls().len(),
                has_text = response.first_text().is_some(),
                "LLM 响应完成"
            );

            Ok(response)
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// 脚本化 Function Calling LLM —— 按脚本返回预设响应。
///
/// ## 为什么预设项是枚举而不是字符串
///
/// Function Calling 的响应是**结构化**的（不是纯文本），
/// 预设时必须能表达「这一轮返回哪些工具调用」。
pub struct ScriptedFunctionLlm {
    steps: Mutex<Vec<ScriptedStep>>,
    calls: Mutex<Vec<ScriptedFunctionCall>>,
    fallback: Option<ScriptedStep>,
}

/// 脚本的一步。
#[derive(Debug, Clone)]
pub enum ScriptedStep {
    /// 只返回文本（最终答案）。
    Text(String),

    /// 只返回工具调用。
    ToolCalls(Vec<ScriptedToolCall>),

    /// 同时返回文本和工具调用。
    TextAndCalls(String, Vec<ScriptedToolCall>),

    /// 返回思考内容 + 工具调用（模拟推理模型）。
    ReasoningAndCalls(String, Vec<ScriptedToolCall>),
}

/// 脚本里的一次工具调用。
#[derive(Debug, Clone)]
pub struct ScriptedToolCall {
    /// 调用 ID（回执时要用它对账）。
    pub call_id: String,

    /// 工具名。
    pub name: String,

    /// 参数。
    pub arguments: Value,
}

impl ScriptedToolCall {
    /// 构造一次脚本工具调用。
    pub fn new(call_id: &str, name: &str, arguments: Value) -> Self {
        Self {
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments,
        }
    }
}

impl ScriptedStep {
    /// 构造一个「返回工具调用」的步骤。
    pub fn calls(calls: Vec<ScriptedToolCall>) -> Self {
        ScriptedStep::ToolCalls(calls)
    }

    /// 构造一个「单个工具调用」的步骤。
    pub fn call(call_id: &str, name: &str, args: Value) -> Self {
        ScriptedStep::ToolCalls(vec![ScriptedToolCall::new(call_id, name, args)])
    }

    /// 构造一个「只返回文本」的步骤。
    pub fn text(text: impl Into<String>) -> Self {
        ScriptedStep::Text(text.into())
    }

    /// 转成 `ChatResponse`。
    pub fn to_response(&self) -> genai::chat::ChatResponse {
        let (text, reasoning_content, calls) = match self {
            ScriptedStep::Text(t) => (t.clone(), None, Vec::new()),
            ScriptedStep::ToolCalls(c) => (String::new(), None, c.clone()),
            ScriptedStep::TextAndCalls(t, c) => (t.clone(), None, c.clone()),
            ScriptedStep::ReasoningAndCalls(r, c) => (String::new(), Some(r.clone()), c.clone()),
        };

        let mut parts: Vec<genai::chat::ContentPart> = Vec::new();
        if let Some(reasoning) = &reasoning_content {
            parts.push(genai::chat::ContentPart::ReasoningContent(
                reasoning.clone(),
            ));
        }
        if !text.is_empty() {
            parts.push(genai::chat::ContentPart::Text(text));
        }
        for call in calls {
            parts.push(genai::chat::ContentPart::ToolCall(genai::chat::ToolCall {
                call_id: call.call_id,
                fn_name: call.name,
                fn_arguments: call.arguments,
                thought_signatures: None,
            }));
        }

        genai::chat::ChatResponse {
            content: genai::chat::MessageContent::from_parts(parts),
            // ⚠️ `reasoning_content` 字段也要填 ——
            //    真实适配器（genai 的 OpenAI 适配器）就是**两个地方都填**：
            //    `ChatResponse.reasoning_content` 给调用方读，
            //    `ContentPart::ReasoningContent` 给下一轮回传用。
            reasoning_content: reasoning_content.clone(),
            model_iden: genai::ModelIden::new(genai::adapter::AdapterKind::OpenAI, "scripted"),
            provider_model_iden: genai::ModelIden::new(
                genai::adapter::AdapterKind::OpenAI,
                "scripted",
            ),
            stop_reason: None,
            usage: genai::chat::Usage::default(),
            captured_raw_body: None,
            response_id: None,
        }
    }
}

/// 记录下来的一次调用（供测试断言）。
#[derive(Debug, Clone)]
pub struct ScriptedFunctionCall {
    /// system 提示。
    pub system: String,

    /// 历史消息数。
    pub message_count: usize,

    /// 工具名列表。
    pub tools: Vec<String>,

    /// 工具选择策略。
    pub tool_choice: String,
}

impl ScriptedFunctionLlm {
    /// 创建脚本化 LLM（按调用顺序返回）。
    pub fn new(steps: Vec<ScriptedStep>) -> Self {
        Self {
            steps: Mutex::new(steps),
            calls: Mutex::new(Vec::new()),
            fallback: None,
        }
    }

    /// 创建脚本化 LLM，所有调用都返回同一结果。
    pub fn always(step: ScriptedStep) -> Self {
        Self {
            steps: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
            fallback: Some(step),
        }
    }

    /// 记录下来的调用。
    pub fn calls(&self) -> Vec<ScriptedFunctionCall> {
        self.calls.lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// 调用次数。
    pub fn call_count(&self) -> usize {
        self.calls.lock().map(|c| c.len()).unwrap_or(0)
    }
}

impl FunctionLlm for ScriptedFunctionLlm {
    fn chat_with_tools<'a>(
        &'a self,
        system: &'a str,
        messages: &'a [genai::chat::ChatMessage],
        tools: &'a [crate::tool::ToolSpec],
        tool_choice: ToolChoice,
    ) -> FunctionLlmFuture<'a> {
        let step = {
            let mut calls = self.calls.lock().expect("calls 锁中毒");
            calls.push(ScriptedFunctionCall {
                system: system.to_string(),
                message_count: messages.len(),
                tools: tools.iter().map(|t| t.name.clone()).collect(),
                tool_choice: tool_choice_name(&tool_choice).to_string(),
            });

            let mut steps = self.steps.lock().expect("steps 锁中毒");
            if steps.is_empty() {
                self.fallback.clone()
            } else {
                Some(steps.remove(0))
            }
        };

        Box::pin(async move {
            let step = step
                .ok_or_else(|| AppError::BadGateway("脚本化 LLM 没有更多预设响应".to_string()))?;
            Ok(step.to_response())
        })
    }

    fn model_name(&self) -> &str {
        "scripted"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolSpec;
    use serde_json::json;

    #[tokio::test]
    async fn scripted_llm_returns_answers_in_order() {
        let llm = ScriptedLlm::new(vec!["一".to_string(), "二".to_string()]);

        assert_eq!(llm.chat("s", "u").await.unwrap().text, "一");
        assert_eq!(llm.chat("s", "u").await.unwrap().text, "二");
        assert!(llm.chat("s", "u").await.is_err());
        assert_eq!(llm.call_count(), 3);
    }

    #[tokio::test]
    async fn scripted_llm_records_calls() {
        let llm = ScriptedLlm::always("答案");
        llm.chat("system 提示", "user 提示").await.unwrap();

        let calls = llm.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].system, "system 提示");
        assert_eq!(calls[0].user, "user 提示");
    }

    #[test]
    fn default_options_disable_reasoning() {
        // ⚠️ 默认关思考 —— 少一个必须正确实现的约束。
        assert!(LlmOptions::default().disable_reasoning);
    }

    #[test]
    fn without_reasoning_sets_the_flag() {
        let options = LlmOptions::without_reasoning(0.0, 2048);
        assert!(options.disable_reasoning);
        assert_eq!(options.temperature, 0.0);
        assert_eq!(options.max_tokens, 2048);
    }

    #[test]
    fn json_mode_is_off_by_default() {
        // ⚠️ 默认**不开** JSON 模式 —— 只有成文阶段才需要。
        assert!(!LlmOptions::default().json_mode);
    }

    #[test]
    fn prompt_cache_key_is_off_by_default() {
        // ⚠️ 默认**不设**缓存键 —— 它是 provider 相关的优化，
        //    不该在没想清楚时默默生效。
        assert!(LlmOptions::default().prompt_cache_key.is_none());
    }

    #[test]
    fn prompt_cache_key_can_be_set() {
        let options = LlmOptions::default().with_prompt_cache_key("k");
        assert_eq!(options.prompt_cache_key.as_deref(), Some("k"));
    }

    #[tokio::test]
    async fn scripted_llm_reports_usage_and_truncation() {
        // ★ token 用量 + 截断标记都必须能被观测到。
        let llm = ScriptedLlm::always("答案")
            .with_usage(LlmUsage {
                prompt_tokens: 11,
                completion_tokens: 7,
                total_tokens: 18,
            })
            .truncated();

        let reply = llm.chat("s", "u").await.unwrap();

        assert_eq!(reply.text, "答案");
        assert_eq!(reply.usage.total_tokens, 18);
        assert!(reply.truncated);
    }

    #[tokio::test]
    async fn usage_merge_accumulates() {
        let mut usage = LlmUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        };
        usage.merge(&LlmUsage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
        });

        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 18);
    }

    #[tokio::test]
    async fn structured_llm_passes_the_schema_through() {
        // ★ Schema 必须**传下去** —— 适配器据此决定用
        //    `json_schema` 还是 `json_object`。
        let llm = ScriptedStructuredLlm::always(r#"{"answer":"ok","numbers":[]}"#);
        let schema = crate::structured::summary_schema();

        llm.structured("s", "u", &schema).await.unwrap();

        let calls = llm.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].schema, schema);
    }

    #[tokio::test]
    async fn scripted_structured_llm_returns_answers_in_order() {
        let llm = ScriptedStructuredLlm::new(vec!["一".to_string(), "二".to_string()]);
        let schema = crate::structured::summary_schema();

        assert_eq!(llm.structured("s", "u", &schema).await.unwrap().text, "一");
        assert_eq!(llm.structured("s", "u", &schema).await.unwrap().text, "二");
        assert!(llm.structured("s", "u", &schema).await.is_err());
    }

    #[test]
    fn tool_choice_modes_map_to_the_genai_variants() {
        // ⚠️ 现在直接用 crate 的 `ToolChoice` ——
        // 映射是**恒等**的（不再是「本地枚举 → crate 枚举」的翻译）。
        assert_eq!(ToolChoice::None, genai::chat::ToolChoice::None);
        assert_eq!(ToolChoice::Auto, genai::chat::ToolChoice::Auto);
        assert_eq!(ToolChoice::Required, genai::chat::ToolChoice::Required);
        assert_eq!(
            ToolChoice::tool("get_weather"),
            genai::chat::ToolChoice::tool("get_weather")
        );
    }

    #[test]
    fn parses_tool_choice_modes() {
        // ⚠️ `parse` 现在只是**纯字符串转换**，不含语义判断。
        assert_eq!(parse_tool_choice("auto"), Some(ToolChoice::Auto));
        assert_eq!(parse_tool_choice("NONE"), Some(ToolChoice::None));
        assert_eq!(parse_tool_choice("required"), Some(ToolChoice::Required));
        assert_eq!(
            parse_tool_choice("forced:get_weather"),
            Some(ToolChoice::tool("get_weather"))
        );
        assert_eq!(parse_tool_choice("nope"), None);
    }

    /// ★ 显示名只是**展示**，不参与语义。
    #[test]
    fn tool_choice_names_are_for_display_only() {
        assert_eq!(tool_choice_name(&ToolChoice::Auto), "auto");
        assert_eq!(tool_choice_name(&ToolChoice::None), "none");
        assert_eq!(tool_choice_name(&ToolChoice::Required), "required");
        assert_eq!(tool_choice_name(&ToolChoice::tool("x")), "forced");
    }

    #[tokio::test]
    async fn scripted_function_llm_returns_steps_in_order() {
        let llm = ScriptedFunctionLlm::new(vec![
            ScriptedStep::Text("一".to_string()),
            ScriptedStep::Text("二".to_string()),
        ]);

        let resp = llm
            .chat_with_tools("s", &[], &[], ToolChoice::Auto)
            .await
            .unwrap();
        assert_eq!(resp.first_text(), Some("一"));

        let resp = llm
            .chat_with_tools("s", &[], &[], ToolChoice::Auto)
            .await
            .unwrap();
        assert_eq!(resp.first_text(), Some("二"));

        assert!(
            llm.chat_with_tools("s", &[], &[], ToolChoice::Auto)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn scripted_function_llm_records_calls() {
        let llm = ScriptedFunctionLlm::always(ScriptedStep::Text("答案".to_string()));
        let specs = vec![ToolSpec::new("f", "d", json!({}))];

        llm.chat_with_tools("系统提示", &[], &specs, ToolChoice::Required)
            .await
            .unwrap();

        let calls = llm.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].system, "系统提示");
        assert_eq!(calls[0].tools, vec!["f".to_string()]);
        assert_eq!(calls[0].tool_choice, "required");
    }

    #[test]
    fn scripted_step_with_calls_becomes_tool_calls() {
        let step = ScriptedStep::call("c1", "module_stats", json!({"scope": "cache"}));
        let resp = step.to_response();
        assert_eq!(resp.tool_calls().len(), 1);
        assert_eq!(resp.tool_calls()[0].fn_name, "module_stats");
        assert_eq!(resp.tool_calls()[0].call_id, "c1");
    }

    #[test]
    fn scripted_step_keeps_reasoning_content_in_both_places() {
        let step = ScriptedStep::ReasoningAndCalls(
            "我在想".to_string(),
            vec![ScriptedToolCall::new("c1", "t", json!({}))],
        );
        let resp = step.to_response();
        assert_eq!(resp.reasoning_content.as_deref(), Some("我在想"));
    }
}
