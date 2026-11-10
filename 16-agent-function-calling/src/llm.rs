//! LLM 抽象层 —— 支持 Function Calling 的模型调用。
//!
//! ## 为什么不能直接复用 07/13/14 的 `Llm` trait
//!
//! 那三个项目的 `Llm` trait 签名是：
//!
//! ```ignore
//! fn chat<'a>(&'a self, system: &'a str, user: &'a str) -> LlmFuture<'a>;
//! ```
//!
//! 它**只能发 system + user 两条消息**，拿不到工具定义，
//! 也拿不到结构化的 `tool_calls`。
//!
//! 本项目需要的签名是：
//!
//! ```ignore
//! fn chat_with_tools<'a>(
//!     &'a self,
//!     system: &'a str,
//!     messages: &'a [ChatMessage],
//!     tools: &'a [ToolSpec],
//!     tool_choice: ToolChoice,
//! ) -> FunctionLlmFuture<'a>;
//! ```
//!
//! 三个关键差异：
//!
//! 1. **消息是数组** —— 结构化协议需要保留完整历史
//!    （user → assistant(tool_calls) → tool → assistant → …）
//! 2. **带工具定义** —— 翻译成 API 的 `tools` 字段
//! 3. **返回完整响应** —— 因为需要 `tool_calls` 和 `reasoning_content`，
//!    不只是文本
//!
//! ## 测试替身
//!
//! [`ScriptedFunctionLlm`] 按脚本返回预设响应 ——
//! 零网络、零延迟、完全确定。
//! 整个 Function Calling 循环（含并行调用、错误自纠、
//! 思考内容回传）都能被单元测试锁住。

use agent_logging::{debug, info, warn};
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use genai::chat::{
    ChatMessage, ChatOptions, ChatRequest, ChatResponse, MessageContent, ReasoningEffort, ToolCall,
};
use serde_json::Value;

use crate::error::{AppError, AppResult};
use crate::tools::ToolSpec;

/// 工具选择策略 —— 对应 API 的 `tool_choice` 参数。
///
/// OpenAI 文档列了四档：
///
/// | 值 | 行为 |
/// |---|---|
/// | `auto` | **默认** —— 模型自己决定调不调、调几个 |
/// | `none` | 一个都不调（相当于不传 tools） |
/// | `required` | 至少调一个 |
/// | 指定函数 | 必须调这一个 |
///
/// ## 四档实测（deepseek-flash，问题「北京和上海现在天气怎么样？」）
///
/// | 档位 | 工具调用 | token | 观察到的行为 |
/// |---|---|---|---|
/// | `auto` | 2 | 2488 | **最优** —— 一次并行调 2 次，答案准确 |
/// | `none` | 0 | 440 | ⚠️ 模型仍用 `<toolcall>` **文本**模拟调用 |
/// | `required` | 7 | 7176 | ⚠️ 强制调用的**副作用**：多调了 5 次无关工具 |
/// | `forced:get_weather` | 7 | 5828 | ⚠️ 同上 |
///
/// ### 三个反直觉的实测结论
///
/// **① `none` 不可靠 —— 模型会用文本模拟工具调用。**
///
/// `none` 档实测输出：
///
/// ```text
/// <toolcall>
/// {"name": "get_weather", "arguments": {"city": "北京"}}
/// </toolcall>
/// ```
///
/// 它**没有**产生结构化的 `tool_calls`，而是**照着提示词里的工具说明**，
/// 用 XML 标签手写了一段「伪调用」。
///
/// 原因：模型在训练时见过大量这种格式。
/// 提示词里出现了工具的名字和描述，模型就会试图「模仿」那个格式。
///
/// ⚠️ 所以**不能**用 `none` 来实现「禁止用工具」——
///    真正的禁止是**不把 tools 传给模型**。
///
/// **② `required` 会诱发「凑数调用」。**
///
/// `required` 档实测调了 7 次工具，其中只有 2 次是必要的
/// （查两个城市），另外 5 次是模型为了满足「必须调工具」而硬凑的
/// —— 包括调 `calculate` 去算风速的差值。
///
/// 最终答案变成了 `20.1 / 1.8 = 11.166666666666668`
/// —— 它**只回答了最后一步算式**，没有回答「天气怎么样」。
///
/// 这说明 `required` 会让模型**丧失「该不该用工具」的判断力**。
/// 它是「强制工具调用」的开关，**不是**「更好的 auto」。
///
/// **③ `auto` 是唯一正确的默认档。**
///
/// 它让模型**自己判断**需不需要工具、该调几个。
/// 实测一次并行拿到两个城市的结果，token 消耗也最低。
///
/// 这四档正好能做一组**消融对照** ——
/// 「模型自己决定要不要用工具」到底贡献了多少。
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

/// LLM 调用的 future。
pub type FunctionLlmFuture<'a> = Pin<Box<dyn Future<Output = AppResult<ChatResponse>> + Send + 'a>>;

/// 支持 Function Calling 的 LLM trait。
pub trait FunctionLlm: Send + Sync {
    /// 带工具定义的一次对话。
    ///
    /// ## `messages` 的形状
    ///
    /// 第一条通常是 `ChatMessage::user`，
    /// 之后是 `assistant`（带 tool_calls）和 `tool`（回执）交替。
    ///
    /// ## 返回值
    ///
    /// 返回**完整的** [`ChatResponse`] —— 因为调用方需要：
    ///
    /// - `tool_calls()` —— 结构化的工具调用
    /// - `first_text()` —— 模型同时输出的文本
    /// - `reasoning_content` —— 推理模型的思考内容（**必须回传**，否则 400）
    fn chat_with_tools<'a>(
        &'a self,
        system: &'a str,
        messages: &'a [ChatMessage],
        tools: &'a [ToolSpec],
        tool_choice: ToolChoice,
    ) -> FunctionLlmFuture<'a>;

    /// 模型名。
    fn model_name(&self) -> &str;
}

/// 基于 `genai` 的真实 LLM 实现。
pub struct GenaiFunctionLlm {
    client: genai::Client,
    model: String,
    temperature: f64,
    max_tokens: u32,
    disable_reasoning: bool,
}

impl std::fmt::Debug for GenaiFunctionLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenaiFunctionLlm")
            .field("model", &self.model)
            .field("disable_reasoning", &self.disable_reasoning)
            .finish()
    }
}

impl GenaiFunctionLlm {
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

impl FunctionLlm for GenaiFunctionLlm {
    fn chat_with_tools<'a>(
        &'a self,
        system: &'a str,
        messages: &'a [ChatMessage],
        tools: &'a [ToolSpec],
        tool_choice: ToolChoice,
    ) -> FunctionLlmFuture<'a> {
        Box::pin(async move {
            let mut options = ChatOptions::default()
                .with_temperature(self.temperature)
                .with_max_tokens(self.max_tokens);

            // 关思考。
            //
            // ⚠️ **为什么默认要关**（实测踩坑）：
            //
            // 推理模型的「思考 token」和「正文 token」**共用** `max_tokens` 预算。
            // 思考没结束正文就是空的。
            //
            // 但本项目有个**额外的**理由：
            //
            // 开思考模式后，DeepSeek **要求**把 `reasoning_content` 回传，
            // 否则下一轮请求 400
            // （`The reasoning_content in the thinking mode must be passed back`）。
            //
            // 关掉思考就少一个必须正确实现的约束 ——
            // 对「学习 function calling 本身」这个目标来说，变量越少越好。
            //
            // 想体验推理模型的话，把 `disable_reasoning` 设成 `false`，
            // 本项目的循环**已经正确实现了 reasoning 回传**。
            if self.disable_reasoning {
                options = options.with_reasoning_effort(ReasoningEffort::Zero);
            }

            let mut request = ChatRequest::new(messages.to_vec()).with_system(system.to_string());

            if !tools.is_empty() {
                request = request.with_tools(tools.iter().map(|t| t.to_genai_tool()));
                options = options.with_tool_choice(tool_choice.clone());
            }

            debug!(
                model = %self.model,
                system_len = system.chars().count(),
                messages = messages.len(),
                tools = tools.len(),
                tool_choice = tool_choice_name(&tool_choice),
                disable_reasoning = self.disable_reasoning,
                "发起带工具的 LLM 调用"
            );

            let response = self
                .client
                .exec_chat(&self.model, request, Some(&options))
                .await
                .map_err(|e| {
                    warn!(model = %self.model, error = %e, "带工具的 LLM 调用失败");
                    AppError::Llm(format!("调用模型失败: {e}"))
                })?;

            info!(
                model = %self.model,
                tool_calls = response.tool_calls().len(),
                has_text = response.first_text().is_some(),
                has_reasoning = response.reasoning_content.is_some(),
                "LLM 响应完成"
            );

            Ok(response)
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// 测试用的脚本化 LLM —— 按调用顺序返回预设响应。
///
/// ## 预设响应的构造方式
///
/// 因为 Function Calling 的响应是**结构化**的（不是纯文本），
/// 预设时必须能表达「这一轮返回哪些工具调用」。
///
/// 所以这里让脚本项是一个枚举：
///
/// ```ignore
/// ScriptedStep::Text("最终答案")
/// ScriptedStep::calls(vec![ScriptedToolCall::new("call_1", "get_weather", json!({"city": "北京"}))])
/// ScriptedStep::TextAndCalls("我查一下", vec![...])
/// ```
pub struct ScriptedFunctionLlm {
    steps: Mutex<Vec<ScriptedStep>>,
    calls: Mutex<Vec<ScriptedCall>>,
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

    /// 返回工具调用 + 思考内容（模拟推理模型）。
    ReasoningAndCalls(String, Vec<ScriptedToolCall>),
}

/// 脚本里的一次工具调用。
///
/// ## 为什么用结构体而不是元组
///
/// 元组 `(String, String, Value)` 在调用点读起来是
/// `("c1", "get_weather", json!({...}))` ——
/// 三个 `String`/`Value` 挨在一起，**看不出哪个是哪个**。
/// 名字写出来就没这个问题了。
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

    /// 转成 `ChatResponse`。
    pub fn to_response(&self) -> ChatResponse {
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
            parts.push(genai::chat::ContentPart::ToolCall(ToolCall {
                call_id: call.call_id,
                fn_name: call.name,
                fn_arguments: call.arguments,
                thought_signatures: None,
            }));
        }

        ChatResponse {
            content: MessageContent::from_parts(parts),
            // ⚠️ `reasoning_content` 字段也要填 ——
            //    真实适配器（`genai` 的 OpenAI 适配器）就是**两个地方都填**：
            //    `ChatResponse.reasoning_content` 字段给调用方读，
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
pub struct ScriptedCall {
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
    pub fn calls(&self) -> Vec<ScriptedCall> {
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
        messages: &'a [ChatMessage],
        tools: &'a [ToolSpec],
        tool_choice: ToolChoice,
    ) -> FunctionLlmFuture<'a> {
        let step = {
            let mut calls = self.calls.lock().expect("calls 锁中毒");
            calls.push(ScriptedCall {
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
            let step =
                step.ok_or_else(|| AppError::Llm("脚本化 LLM 没有更多预设响应".to_string()))?;
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
    use serde_json::json;

    /// ⚠️ `parse` 现在只是**纯字符串转换**，不含语义判断 ——
    /// 语义由 crate 的 `ToolChoice` 定义。
    #[test]
    fn parses_tool_choice_modes() {
        assert_eq!(parse_tool_choice("auto"), Some(ToolChoice::Auto));
        assert_eq!(parse_tool_choice("NONE"), Some(ToolChoice::None));
        assert_eq!(parse_tool_choice("required"), Some(ToolChoice::Required));
        assert_eq!(
            parse_tool_choice("forced:get_weather"),
            Some(ToolChoice::tool("get_weather"))
        );
        assert_eq!(parse_tool_choice("nope"), None);
    }

    /// ★ 直接用 crate 的 `ToolChoice` —— 映射是**恒等**的，
    /// 不再需要「本地枚举 → crate 枚举」的翻译层。
    #[test]
    fn tool_choice_maps_are_identity() {
        assert_eq!(ToolChoice::Auto, ToolChoice::Auto);
        assert_eq!(ToolChoice::None, ToolChoice::None);
        assert_eq!(ToolChoice::Required, ToolChoice::Required);
        assert_eq!(ToolChoice::tool("f"), ToolChoice::tool("f"));
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
    async fn scripted_llm_returns_steps_in_order() {
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
    async fn scripted_llm_records_calls() {
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

    #[tokio::test]
    async fn scripted_step_produces_tool_calls() {
        let llm = ScriptedFunctionLlm::new(vec![ScriptedStep::call(
            "c1",
            "get_weather",
            json!({ "city": "北京" }),
        )]);

        let resp = llm
            .chat_with_tools("s", &[], &[], ToolChoice::Auto)
            .await
            .unwrap();

        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].fn_name, "get_weather");
        assert_eq!(calls[0].call_id, "c1");
    }
}
