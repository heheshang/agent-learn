//! Function Calling 循环 —— 本项目的核心。
//!
//! ## 和 13/14/15 的 ReAct 循环有什么本质区别
//!
//! 这三个项目的循环是**文本协议**：
//!
//! ```text
//! 模型输出（纯文本）：
//!   Thought: 我需要查一下天气
//!   Action: get_weather
//!   Action Input: {"city": "北京"}
//!
//! 我们用正则把它解析出来，再执行工具，把 Observation 拼回提示词。
//! ```
//!
//! 本项目走**结构化协议**（OpenAI Function Calling）：
//!
//! ```text
//! 请求：tools = [ { name, description, parameters } ]
//! 响应：tool_calls = [ { id, function: { name, arguments } } ]   ← 结构化 JSON
//! 我们直接读字段，不用正则。
//! ```
//!
//! | 维度 | ReAct 文本协议 | Function Calling 结构化协议 |
//! |---|---|---|
//! | 工具定义 | 拼进提示词文本 | 放进 API 的 `tools` 字段 |
//! | 模型输出 | 纯文本，要正则解析 | 结构化 JSON，**不用解析** |
//! | 参数校验 | 靠提示词 + 事后校验 | Schema 直接参与模型训练 |
//! | 工具结果回传 | 拼成 `Observation:` 文本 | 一条 `tool` 角色的消息 |
//! | 并行调用 | 不支持 | **支持**（一次多个 `tool_calls`） |
//!
//! ## ⚠️ 四个实测踩出来的坑
//!
//! ### 坑 1：每个 `tool_call` 都必须有回执，否则**整个请求 400**
//!
//! DeepSeek 的报错原文：
//!
//! ```text
//! An assistant message with 'tool_calls' must be followed by tool messages
//! responding to each 'tool_call_id'.
//! (insufficient tool messages following tool_calls message)
//! ```
//!
//! 也就是说：模型一次返回 2 个 `tool_calls`，你就必须回 2 条 `tool` 消息。
//! **少一条就是 400**。
//!
//! 本项目的对策：并行执行所有工具调用，然后**保证**为每个 `call_id`
//! 生成恰好一条 `tool` 消息 —— 包括工具失败的情况（错误文本也是回执）。
//!
//! ### 坑 2：开了思考模式就**必须**回传 `reasoning_content`
//!
//! DeepSeek 的报错原文：
//!
//! ```text
//! The `reasoning_content` in the thinking mode must be passed back to the API.
//! ```
//!
//! `deepseek-v4-pro` 默认开思考，会返回 `reasoning_content`。
//! 如果下一轮不把它带回，请求直接 400。
//!
//! 本项目的对策：把 `reasoning_content` 存进 assistant 消息，
//! 下一轮原样带回。
//!
//! ### 坑 3：`parallel_tool_calls: false` 被 DeepSeek **忽略**
//!
//! 实测传了 `parallel_tool_calls: false`，模型**照样**一次返回 2 个调用。
//!
//! 这意味着：**不能靠 API 参数控制并行度**。
//! 想要「一次只调一个」，只能自己在**应用层**限制
//! —— 比如一次只把一个工具暴露给模型。
//!
//! ### 坑 4：`strict: true` 被接受但**不强制执行**
//!
//! 实测：`strict: true` + `required` 只写一个字段 → 请求通过，
//! 模型照样可能不填另一个字段。
//!
//! 这意味着：**strict 不能替代本地 Schema 校验**。
//! 本项目两道防线都上：发 `strict`（讨好 OpenAI）+ 本地校验（真兜底）。

use agent_logging::{debug, info, warn};
use std::sync::Arc;
use std::time::Instant;

use genai::chat::{ChatMessage, ChatRequest, ChatResponse, ToolCall, ToolResponse};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AppError, AppResult};
use crate::llm::FunctionLlm;
use crate::prompt;
use crate::tool::{ToolContext, ToolOutcome, ToolRegistry};
use crate::tools::ToolSpec;

/// 一步的执行记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// 第几步（从 1 开始）。
    pub step: usize,

    /// 模型返回的**文本**部分（模型可能会在调工具的同时说句话）。
    pub text: String,

    /// 模型的思考内容（推理模型才有）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    /// 本步发出的工具调用（可能多个 —— 并行调用）。
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,

    /// 本步是否直接给出了最终答案。
    pub final_answer: bool,
}

/// 一次工具调用的记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// 调用的 ID（回执时要用它对账）。
    pub call_id: String,

    /// 工具名。
    pub name: String,

    /// 参数（模型给的原始值）。
    pub arguments: Value,

    /// 工具结果（给模型看的文本）。
    pub observation: String,

    /// 工具是否执行成功。
    pub ok: bool,

    /// 结构化结果（如果工具返回了）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// 一次 Agent 运行的完整结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRun {
    /// 原始问题。
    pub question: String,

    /// 最终答案。
    pub answer: String,

    /// 每一步的记录。
    pub steps: Vec<StepRecord>,

    /// 工具调用总次数。
    pub tool_calls: usize,

    /// 用到的工具名（去重，按首次出现顺序）。
    pub tools_used: Vec<String>,

    /// 是否因为达到步数上限而中止。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,

    /// 累计 token 用量。
    pub usage: TokenUsage,
}

/// 累计 token 用量。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// 输入 token。
    pub prompt_tokens: u64,
    /// 输出 token。
    pub completion_tokens: u64,
    /// 总 token。
    pub total_tokens: u64,
}

impl TokenUsage {
    /// 累加一步的用量。
    fn add(&mut self, response: &ChatResponse) {
        let prompt = response.usage.prompt_tokens.unwrap_or(0) as u64;
        let completion = response.usage.completion_tokens.unwrap_or(0) as u64;
        self.prompt_tokens += prompt;
        self.completion_tokens += completion;
        // total 优先用上游给的，没有就用 prompt + completion 算。
        self.total_tokens += response
            .usage
            .total_tokens
            .unwrap_or((prompt + completion) as i32) as u64;
    }
}

/// Function Calling Agent。
pub struct Agent {
    llm: Arc<dyn FunctionLlm>,
    registry: ToolRegistry,
    config: crate::config::AppConfig,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("tools", &self.registry.names())
            .field("model", &self.config.model)
            .finish()
    }
}

impl Agent {
    /// 创建 Agent。
    pub fn new(
        llm: Arc<dyn FunctionLlm>,
        registry: ToolRegistry,
        config: crate::config::AppConfig,
    ) -> Self {
        Self {
            llm,
            registry,
            config,
        }
    }

    /// 工具注册表。
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// 工具定义列表。
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.registry.specs()
    }

    /// 跑一次 Agent。
    ///
    /// ## 循环的骨架
    ///
    /// ```text
    /// ┌──────────────────────────────────────────────────────┐
    /// │ 1. 组装 messages（system + user + 历史）                │
    /// │ 2. 带上 tools 发请求                                  │
    /// │ 3. 响应里有 tool_calls？                               │
    /// │    ├─ 没有 → 这就是最终答案，收工                       │
    /// │    └─ 有   → 逐个执行工具（并行语义、串行实现）          │
    /// │ 4. 为**每个** call_id 生成恰好一条 tool 消息            │
    /// │ 5. 回到 1                                            │
    /// └──────────────────────────────────────────────────────┘
    /// ```
    pub async fn run(&self, question: &str) -> AppResult<AgentRun> {
        let start = Instant::now();
        let question = question.trim();

        if question.is_empty() {
            return Err(AppError::BadRequest("问题不能为空".to_string()));
        }

        let specs = self.registry.specs();
        let system = prompt::system_prompt(&specs, self.config.prompt.strict);
        let ctx = ToolContext::empty();

        // 历史消息 —— 结构化协议下，这就是「短期记忆」。
        let mut messages: Vec<ChatMessage> = vec![ChatMessage::user(question.to_string())];

        let mut steps: Vec<StepRecord> = Vec::new();
        let mut tool_calls = 0usize;
        let mut tools_used: Vec<String> = Vec::new();
        let mut usage = TokenUsage::default();

        for step_no in 1..=self.config.agent.max_steps {
            debug!(
                step = step_no,
                tool_count = specs.len(),
                history = messages.len(),
                "第 {step_no} 步"
            );

            // ── 调模型 ──────────────────────────────────────────
            let response = self
                .llm
                .chat_with_tools(
                    &system,
                    &messages,
                    &specs,
                    self.config.prompt.tool_choice.clone(),
                )
                .await?;

            usage.add(&response);

            let text = response.first_text().unwrap_or_default().to_string();
            let reasoning = response.reasoning_content.clone();
            let calls: Vec<ToolCall> = response.into_tool_calls();

            // ── 没有工具调用 → 这就是最终答案 ─────────────────────
            if calls.is_empty() {
                steps.push(StepRecord {
                    step: step_no,
                    text: text.clone(),
                    reasoning: reasoning.clone(),
                    tool_calls: Vec::new(),
                    final_answer: true,
                });

                info!(
                    step = step_no,
                    tool_calls,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "Agent 给出最终答案"
                );

                return Ok(AgentRun {
                    question: question.to_string(),
                    answer: text,
                    steps,
                    tool_calls,
                    tools_used,
                    hit_max_steps: false,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    model: self.config.model.clone(),
                    usage,
                });
            }

            // ── 有工具调用 → 执行 ────────────────────────────────
            //
            // ⚠️ 坑 2：assistant 消息必须**带上 reasoning_content**，
            //    否则推理模型下一轮会 400。
            messages.push(build_assistant_message(
                text.clone(),
                reasoning.clone(),
                &calls,
            ));

            let mut records = Vec::with_capacity(calls.len());

            // ⚠️ 坑 1：**每个** call_id 都要有一条回执。
            //
            // 这里用一个显式的循环（而不是 `for call in calls` 里直接 push
            // 消息）来保证「要么两条都有，要么两条都没有」——
            // 不存在「处理了一半」的中间状态。
            let mut responses: Vec<ToolResponse> = Vec::with_capacity(calls.len());

            for call in &calls {
                let outcome = self.execute_call(call, &ctx).await;

                if !outcome.ok {
                    warn!(
                        step = step_no,
                        tool = %call.fn_name,
                        observation = %outcome.observation,
                        "工具执行失败（错误已作为 tool 消息喂回模型）"
                    );
                }

                if !tools_used.contains(&call.fn_name) {
                    tools_used.push(call.fn_name.clone());
                }

                records.push(ToolCallRecord {
                    call_id: call.call_id.clone(),
                    name: call.fn_name.clone(),
                    arguments: call.fn_arguments.clone(),
                    observation: outcome.observation.clone(),
                    ok: outcome.ok,
                    data: outcome.data.clone(),
                });

                // 回执的内容就是观察文本 —— 失败时是错误描述。
                //
                // ⚠️ 关键：**失败也必须有回执**，
                //    否则下一轮请求会因为「insufficient tool messages」被 400。
                responses.push(ToolResponse::from_tool_call(call, outcome.observation));

                tool_calls += 1;
            }

            // 把回执一次性加进历史（一条 assistant + N 条 tool）。
            messages.push(ChatMessage::tool(
                genai::chat::MessageContent::from_tool_responses(responses),
            ));

            steps.push(StepRecord {
                step: step_no,
                text,
                reasoning,
                tool_calls: records,
                final_answer: false,
            });
        }

        // ── 达到步数上限 ────────────────────────────────────────
        //
        // ⚠️ 这里**不能**直接报错 —— 前面几步辛苦收集的工具结果就白费了。
        // 降级：把最后一步的观察作为答案返回，并标记 `hit_max_steps`。
        warn!(
            max_steps = self.config.agent.max_steps,
            tool_calls, "达到步数上限 —— 降级返回"
        );

        let answer = steps
            .last()
            .and_then(|s| s.tool_calls.first())
            .map(|c| c.observation.clone())
            .unwrap_or_else(|| {
                format!(
                    "达到最大步数（{}）仍未得到最终答案。请缩小问题范围后重试。",
                    self.config.agent.max_steps
                )
            });

        Ok(AgentRun {
            question: question.to_string(),
            answer,
            steps,
            tool_calls,
            tools_used,
            hit_max_steps: true,
            elapsed_ms: start.elapsed().as_millis() as u64,
            model: self.config.model.clone(),
            usage,
        })
    }

    /// 执行一个工具调用。
    async fn execute_call(&self, call: &ToolCall, ctx: &ToolContext) -> ToolOutcome {
        // ⚠️ 参数可能是**字符串**而不是对象 ——
        //    有些模型会把 arguments 序列化成 JSON 字符串。
        //
        //    genai 的 OpenAI 适配器已经处理了这一点
        //    （`Value::String(txt) => serde_json::from_str(&txt)`），
        //    但**流式**场景下会保留成字符串。
        //    这里再兜一层，保证本地校验能正常拿到对象。
        let args = normalize_arguments(&call.fn_arguments);

        self.registry.call(&call.fn_name, &args, ctx).await
    }
}

/// 构造 assistant 消息（带工具调用 + 思考内容）。
///
/// ## ⚠️ 为什么必须带 `reasoning_content`
///
/// 推理模型（`deepseek-v4-pro`）在思考模式下**要求**把
/// `reasoning_content` 原样带回，否则下一轮请求直接 400：
///
/// ```text
/// The `reasoning_content` in the thinking mode must be passed back to the API.
/// ```
fn build_assistant_message(
    text: String,
    reasoning: Option<String>,
    calls: &[ToolCall],
) -> ChatMessage {
    let mut parts: Vec<genai::chat::ContentPart> = Vec::new();

    // ⚠️ 顺序很重要：思考内容在前，工具调用在后。
    if let Some(reasoning) = reasoning
        && !reasoning.is_empty()
    {
        parts.push(genai::chat::ContentPart::ReasoningContent(reasoning));
    }

    if !text.is_empty() {
        parts.push(genai::chat::ContentPart::Text(text));
    }

    for call in calls {
        parts.push(genai::chat::ContentPart::ToolCall(call.clone()));
    }

    ChatMessage::assistant(genai::chat::MessageContent::from_parts(parts))
}

/// 把工具参数规整成 JSON 对象。
///
/// ## 为什么需要这一步
///
/// 模型返回的 `arguments` 理论上总是 JSON 对象，
/// 但有两种情况会出问题：
///
/// 1. 有些模型返回 **JSON 字符串**（`"{\"city\":\"北京\"}"`）
/// 2. 流式累积时，参数可能还没拼完整（是字符串）
///
/// 本地 Schema 校验要求 `Value::Object`，
/// 所以这里统一转换。
pub fn normalize_arguments(args: &Value) -> Value {
    match args {
        Value::Object(_) => args.clone(),
        Value::String(text) => {
            // 尝试把字符串解析成 JSON。
            match serde_json::from_str::<Value>(text) {
                Ok(parsed @ Value::Object(_)) => parsed,
                // 解析失败 —— 交给校验器报错（它会说「必须是 JSON 对象」）。
                _ => args.clone(),
            }
        }
        Value::Null => Value::Object(serde_json::Map::new()),
        other => other.clone(),
    }
}

/// 直接调用一个工具（不经模型）—— 给 HTTP 端点用。
///
/// ## 为什么「用户直接调工具」的错误要变成 HTTP 响应
///
/// 在 Agent 循环里，工具失败被转成 `tool` 消息喂回模型 ——
/// 因为**模型**是那个需要自我纠正的角色。
///
/// 但在这里用户直接指定了工具和参数，他扮演的就是「模型」的角色。
/// 所以错误应该**原样返回给他**，而不是被包装成观察。
pub async fn call_tool_directly(
    registry: &ToolRegistry,
    name: &str,
    args: &Value,
) -> AppResult<ToolOutcome> {
    let outcome = registry
        .call_strict(name, args, &ToolContext::empty())
        .await?;
    Ok(outcome)
}

/// 组装请求用的 `ChatRequest`（供测试与调试使用）。
pub fn build_request(system: &str, messages: Vec<ChatMessage>, specs: &[ToolSpec]) -> ChatRequest {
    let mut request = ChatRequest::new(messages).with_system(system.to_string());
    if !specs.is_empty() {
        request = request.with_tools(specs.iter().map(|s| s.to_genai_tool()));
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalizes_object_arguments() {
        let args = json!({ "city": "北京" });
        assert_eq!(normalize_arguments(&args), args);
    }

    #[test]
    fn normalizes_json_string_arguments() {
        // ⚠️ 有些模型把 arguments 序列化成字符串 ——
        // 不转换的话本地校验会直接报「必须是 JSON 对象」。
        let args = json!("{\"city\": \"北京\"}");
        let normalized = normalize_arguments(&args);
        assert_eq!(normalized["city"], json!("北京"));
    }

    #[test]
    fn keeps_invalid_string_as_is() {
        // 解析失败就原样保留 —— 让校验器去报错。
        let args = json!("不是 JSON");
        assert_eq!(normalize_arguments(&args), json!("不是 JSON"));
    }

    #[test]
    fn null_arguments_become_empty_object() {
        assert_eq!(normalize_arguments(&Value::Null), json!({}));
    }

    #[test]
    fn builds_assistant_message_with_reasoning_first() {
        let calls = vec![ToolCall {
            call_id: "c1".to_string(),
            fn_name: "get_weather".to_string(),
            fn_arguments: json!({ "city": "北京" }),
            thought_signatures: None,
        }];

        let msg =
            build_assistant_message("我查一下".to_string(), Some("思考中".to_string()), &calls);

        assert_eq!(msg.role, genai::chat::ChatRole::Assistant);
        let parts: Vec<_> = msg.content.into_iter().collect();
        assert_eq!(parts.len(), 3);
        // 顺序：思考 → 文本 → 工具调用。
        assert!(matches!(
            parts[0],
            genai::chat::ContentPart::ReasoningContent(_)
        ));
        assert!(matches!(parts[1], genai::chat::ContentPart::Text(_)));
        assert!(matches!(parts[2], genai::chat::ContentPart::ToolCall(_)));
    }

    #[test]
    fn builds_assistant_message_without_reasoning() {
        let calls = vec![ToolCall {
            call_id: "c1".to_string(),
            fn_name: "f".to_string(),
            fn_arguments: json!({}),
            thought_signatures: None,
        }];

        let msg = build_assistant_message(String::new(), None, &calls);
        let parts: Vec<_> = msg.content.into_iter().collect();
        assert_eq!(parts.len(), 1);
        assert!(matches!(parts[0], genai::chat::ContentPart::ToolCall(_)));
    }

    #[test]
    fn token_usage_accumulates() {
        let mut usage = TokenUsage::default();
        let mut response = ChatResponse {
            content: genai::chat::MessageContent::default(),
            reasoning_content: None,
            model_iden: genai::ModelIden::new(genai::adapter::AdapterKind::OpenAI, "m"),
            provider_model_iden: genai::ModelIden::new(genai::adapter::AdapterKind::OpenAI, "m"),
            stop_reason: None,
            usage: genai::chat::Usage {
                prompt_tokens: Some(10),
                completion_tokens: Some(5),
                prompt_tokens_details: None,
                completion_tokens_details: None,
                total_tokens: Some(15),
            },
            captured_raw_body: None,
            response_id: None,
        };

        usage.add(&response);
        usage.add(&response);

        assert_eq!(usage.prompt_tokens, 20);
        assert_eq!(usage.completion_tokens, 10);
        assert_eq!(usage.total_tokens, 30);

        // total 缺失时用 prompt + completion 兜底。
        response.usage.total_tokens = None;
        let mut usage2 = TokenUsage::default();
        usage2.add(&response);
        assert_eq!(usage2.total_tokens, 15);
    }
}
