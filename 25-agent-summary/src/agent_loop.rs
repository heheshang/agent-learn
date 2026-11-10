//! **Agent 循环** —— ★ 25 之前完全缺失的那一环。
//!
//! # 为什么这是最关键的一块
//!
//! 25 之前有：
//!
//! - `agent::parse_react_output` —— 解析器
//! - `tooling::validate` —— 参数校验
//! - `pipeline::Pipeline::run` —— 把上面几个**串起来假装跑一遍**
//!
//! 但**没有一个循环**。`Pipeline::run` 里的「模型输出」是
//! `format!` 拼出来的字符串：
//!
//! ```ignore
//! let raw = format!("Thought: ...\nAction: module_stats\nAction Input: {{...}}");
//! ```
//!
//! 所以「这些零件串起来跑得对不对」**没有证据**。
//! 本模块补上这个循环 —— 而且**同时**实现两条协议。
//!
//! # 两条协议对照
//!
//! | 维度 | ReAct 文本协议 | Function Calling 结构化协议 |
//! |---|---|---|
//! | 工具定义 | 拼进提示词文本 | 放进 API 的 `tools` 字段 |
//! | 模型输出 | 纯文本，要**正则解析** | 结构化 JSON，**不用解析** |
//! | 回执 | 拼成 `Observation:` 文本 | 一条 `role="tool"` 的消息 |
//! | 并行调用 | 不支持 | 支持 |
//!
//! ## ★ 两条路跑**同一个工具集**
//!
//! 这样才能做「协议对照实验」：
//! **同样的工具、同样的问题，只换协议** —— 差值归因到协议本身。
//!
//! # 四条硬约定
//!
//! ## ① 每个工具调用都必须有**恰好一条**回执
//!
//! DeepSeek 原文：
//!
//! ```text
//! An assistant message with 'tool_calls' must be followed by tool messages
//! responding to each 'tool_call_id'.
//! (insufficient tool messages following tool_calls message)
//! ```
//!
//! **少一条就是 400**，而且错误信息完全不提示少了哪条。
//! 工具**失败**也算回执（错误文本就是回执内容）。
//!
//! ## ② 达到步数上限要**降级**，不能报错
//!
//! 前面几步辛苦收集的工具结果不能白费。
//! 把最后的观察作为答案返回，并标记 `hit_max_steps`。
//!
//! ## ③ 模型调用失败**不能**降级
//!
//! 「没有模型就没有答案」—— 这是 502。
//!
//! ## ④ 记忆召回必须在**调模型之前**
//!
//! 事实要进 system 提示。
//! 如果召回发生在调模型之后，模型根本看不到这些事实。

use std::sync::Arc;
use std::time::Instant;

use genai::chat::{ChatMessage, ToolCall, ToolResponse};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{MemoryFact, MemoryMessage, MemoryRole, ThreadState, parse_react_output};
use crate::error::AppResult;
use crate::llm::{FunctionLlm, Llm, ToolChoice};
use crate::model::Window;
use crate::tool::{ToolContext, ToolOutcome, ToolRegistry};

/// 一次工具调用的记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// 调用的 ID（回执时要用它对账）。
    pub call_id: String,

    /// 工具名。
    pub name: String,

    /// 参数（模型的原始值）。
    pub arguments: Value,

    /// 工具结果（给模型看的文本）。
    pub observation: String,

    /// 工具是否执行成功。
    pub ok: bool,

    /// 尝试次数（**重试次数必须在这里** —— 丢了就没法统计）。
    pub attempts: usize,
}

/// 一步的执行记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// 第几步（从 1 开始）。
    pub step: usize,

    /// 模型返回的文本部分。
    pub text: String,

    /// 本步发出的工具调用。
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,

    /// 本步是否直接给出了最终答案。
    pub final_answer: bool,
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

    /// ★ 实际尝试次数（含重试）。
    pub tool_attempts: usize,

    /// 用到的工具名（去重，按首次出现顺序）。
    pub tools_used: Vec<String>,

    /// 是否因为达到步数上限而中止。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,

    /// 累计 token 用量。
    ///
    /// ⚠️ **耗时不是好度量** —— 受上游负载、网络抖动影响，
    /// 同一次实验前后差 200ms 就可能让结论翻转。
    /// token 数是**确定性**的，更适合做归因依据。
    pub usage: crate::llm::LlmUsage,

    /// ★ 输出是否被 `max_tokens` **截断**了。
    ///
    /// ## ⚠️ 为什么必须单独拎出来
    ///
    /// 截断的答案看起来是**完整的** —— 有头有尾，只少了尾巴。
    /// 症状极隐蔽：数字少了，但读起来完全通顺。
    pub truncated: bool,

    /// 召回的事实数（**记忆召回在调模型之前**）。
    pub recalled_facts: usize,
}

/// Agent 循环的配置。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 最大步数。
    pub max_steps: usize,

    /// 工具选择策略（仅结构化协议用）。
    pub tool_choice: ToolChoice,

    /// 每次工具失败后的最大重试次数。
    ///
    /// ⚠️ **只有 Transient 才重试** —— 参数填错时重试是纯浪费
    /// （参数不会因为重试变对）。
    pub max_retries: usize,

    /// 重试退避基数（毫秒）。
    pub retry_base_ms: u64,

    /// 重试退避上限（毫秒）。
    pub retry_max_ms: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 6,
            tool_choice: ToolChoice::Auto,
            max_retries: 2,
            retry_base_ms: 100,
            retry_max_ms: 2000,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// ReAct 文本协议循环
// ─────────────────────────────────────────────────────────────────────

/// ReAct Agent —— 文本协议。
pub struct ReactAgent {
    llm: Arc<dyn Llm>,
    registry: ToolRegistry,
    config: AgentConfig,
}

impl std::fmt::Debug for ReactAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReactAgent")
            .field("tools", &self.registry.names())
            .field("model", &self.llm.model_name())
            .finish()
    }
}

impl ReactAgent {
    /// 创建 ReAct Agent。
    pub fn new(llm: Arc<dyn Llm>, registry: ToolRegistry, config: AgentConfig) -> Self {
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

    /// 跑一次 Agent。
    ///
    /// ## 循环骨架
    ///
    /// ```text
    /// ┌──────────────────────────────────────────────────────┐
    /// │ 1. 召回事实 → 拼进 system 提示（★ 必须在调模型前）  │
    /// │ 2. 调模型，拿纯文本                                  │
    /// │ 3. 解析输出                                          │
    /// │    ├─ Final Answer → 收工                            │
    /// │    ├─ Action        → 执行工具 → 拼 Observation       │
    /// │    └─ Unparsed      → 收工（容错）                   │
    /// │ 4. 把 Observation 拼进历史，回到 2                  │
    /// └──────────────────────────────────────────────────────┘
    /// ```
    pub async fn run(
        &self,
        question: &str,
        facts: &[MemoryFact],
        window: &Window,
    ) -> AppResult<AgentRun> {
        let start = Instant::now();
        let question = question.trim();

        if question.is_empty() {
            return Err(crate::error::AppError::BadRequest("问题不能为空".into()));
        }

        let specs = self.registry.specs();
        let system = react_system_prompt(&specs, facts);
        let ctx = ToolContext::with_window(*window);

        // 短期记忆：消息历史。
        let mut state = ThreadState::new("react");

        let mut steps: Vec<StepRecord> = Vec::new();
        let mut tool_calls = 0usize;
        let mut tool_attempts = 0usize;
        let mut tools_used: Vec<String> = Vec::new();
        // ★ token 用量累加 + 截断标记。
        let mut usage = crate::llm::LlmUsage::default();
        let mut truncated = false;

        for step_no in 1..=self.config.max_steps {
            // ── 组装历史 ─────────────────────────────────────────
            //
            // ⚠️ 事实**只进 system**，**不进消息数组** ——
            //    否则几轮之后历史全是重复事实。
            let transcript = render_transcript(&state);

            let user = format!("{transcript}\n\n问题：{question}");

            // ── 调模型 ───────────────────────────────────────────
            let reply = self.llm.chat(&system, &user).await?;
            let raw = reply.text;

            usage.merge(&reply.usage);
            truncated |= reply.truncated;

            // ⚠️★ **截断必须被检查** —— 这是一个**极其隐蔽**的错误。
            //
            // 被 `max_tokens` 截断的答案看起来是**完整的** ——
            // 它有头有尾，只是少了尾巴。
            //
            // 不检查的话，就会把「被截断的半截结论」当成正常答案返回，
            // 而且**症状极隐蔽**：数字少了，但读起来完全通顺。
            //
            // ⚠️ 这里**不能**直接报错（会丢掉前面几步的工具结果），
            //    也不能**默默返回**（会把半截结论当完整结论）。
            //    正确做法是**在结果里标出来**，让调用方知道。
            if reply.truncated {
                agent_logging::warn!(
                    step = step_no,
                    prompt_tokens = reply.usage.prompt_tokens,
                    completion_tokens = reply.usage.completion_tokens,
                    "模型输出被 max_tokens 截断 —— 结果可能不完整"
                );
            }

            // ── 解析 ─────────────────────────────────────────────
            let parsed = parse_react_output(&raw);

            match parsed {
                crate::agent::ParsedOutput::FinalAnswer(answer) => {
                    steps.push(StepRecord {
                        step: step_no,
                        text: answer.clone(),
                        tool_calls: Vec::new(),
                        final_answer: true,
                    });

                    return Ok(AgentRun {
                        question: question.to_string(),
                        answer,
                        steps,
                        tool_calls,
                        tool_attempts,
                        tools_used,
                        hit_max_steps: false,
                        elapsed_ms: start.elapsed().as_millis() as u64,
                        model: self.llm.model_name().to_string(),
                        usage,
                        truncated,
                        recalled_facts: facts.len(),
                    });
                }

                crate::agent::ParsedOutput::Action { tool, args } => {
                    // ── 执行工具（含重试）─────────────────────────
                    let (outcome, attempts) = self.execute_with_retry(&tool, &args, &ctx).await;

                    tool_calls += 1;
                    tool_attempts += attempts;

                    if !tools_used.contains(&tool) {
                        tools_used.push(tool.clone());
                    }

                    let record = ToolCallRecord {
                        call_id: format!("call-{}", step_no),
                        name: tool.clone(),
                        arguments: args.clone(),
                        observation: outcome.observation.clone(),
                        ok: outcome.ok,
                        attempts,
                    };

                    // ── 把观察写进历史 ────────────────────────────
                    //
                    // ⚠️ 观察里可能带 `@@DATA:{...}` 结构化尾巴 ——
                    //    发给模型前**必须剥掉**（那坨 JSON 占 token 还干扰模型）。
                    state.merge_messages(vec![
                        MemoryMessage::new(MemoryRole::User, question.to_string()),
                        MemoryMessage::new(MemoryRole::Assistant, raw.clone()),
                        MemoryMessage::new(
                            MemoryRole::Assistant,
                            format!("Observation: {}", strip_data_tail(&outcome.observation)),
                        ),
                    ]);

                    steps.push(StepRecord {
                        step: step_no,
                        text: raw.clone(),
                        tool_calls: vec![record],
                        final_answer: false,
                    });
                }

                crate::agent::ParsedOutput::Unparsed(text) => {
                    // ⚠️ 什么都没解析出来 —— **容错当成答案**，
                    //    不要报错（模型可能只是说了句客套话）。
                    steps.push(StepRecord {
                        step: step_no,
                        text: text.clone(),
                        tool_calls: Vec::new(),
                        final_answer: true,
                    });

                    return Ok(AgentRun {
                        question: question.to_string(),
                        answer: text,
                        steps,
                        tool_calls,
                        tool_attempts,
                        tools_used,
                        hit_max_steps: false,
                        elapsed_ms: start.elapsed().as_millis() as u64,
                        model: self.llm.model_name().to_string(),
                        usage,
                        truncated,
                        recalled_facts: facts.len(),
                    });
                }
            }
        }

        // ── 达到步数上限 → 降级 ─────────────────────────────────
        agent_logging::warn!(
            max_steps = self.config.max_steps,
            tool_calls,
            "ReAct 达到步数上限 —— 降级返回"
        );

        let answer = steps
            .last()
            .and_then(|s| s.tool_calls.first())
            .map(|c| c.observation.clone())
            .unwrap_or_else(|| {
                format!(
                    "达到最大步数（{}）仍未得到最终答案。请缩小问题范围后重试。",
                    self.config.max_steps
                )
            });

        Ok(AgentRun {
            question: question.to_string(),
            answer,
            steps,
            tool_calls,
            tool_attempts,
            tools_used,
            hit_max_steps: true,
            elapsed_ms: start.elapsed().as_millis() as u64,
            model: self.llm.model_name().to_string(),
            usage,
            truncated,
            recalled_facts: facts.len(),
        })
    }

    /// ★ 执行工具（**带重试**）。
    ///
    /// ## ⚠️ 只有 **Transient** 才重试
    ///
    /// 参数填错时重试是纯浪费 —— 参数不会因为重试变对。
    ///
    /// 这条由回归测试 `business_errors_are_never_retried` 锁住。
    async fn execute_with_retry(
        &self,
        name: &str,
        args: &Value,
        ctx: &ToolContext,
    ) -> (ToolOutcome, usize) {
        let mut attempt = 0usize;

        loop {
            attempt += 1;

            // ★ 用 `call_strict` —— 我们需要**错误对象**才能判断该不该重试，
            //   而不是已经被渲染成文本的观察。
            match self.registry.call_strict(name, args, ctx).await {
                Ok(outcome) => return (outcome, attempt),

                Err(err) => {
                    if err.retryable() && attempt <= self.config.max_retries {
                        let delay = crate::tooling::backoff(
                            std::time::Duration::from_millis(self.config.retry_base_ms),
                            std::time::Duration::from_millis(self.config.retry_max_ms),
                            attempt,
                        );

                        agent_logging::warn!(
                            tool = %name,
                            code = %err.code.id(),
                            attempt,
                            delay_ms = delay.as_millis() as u64,
                            "Transient 错误 —— 重试"
                        );

                        tokio::time::sleep(delay).await;
                        continue;
                    }

                    // 不可重试 → 转成观察喂回模型。
                    agent_logging::warn!(
                        tool = %name,
                        code = %err.code.id(),
                        kind = %err.kind.name(),
                        attempt,
                        "工具失败 —— 错误将作为观察喂回模型"
                    );

                    return (ToolOutcome::failed(name, err.render()), attempt);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────
// Function Calling 循环
// ─────────────────────────────────────────────────────────────────────

/// Function Calling Agent —— 结构化协议。
pub struct FunctionAgent {
    llm: Arc<dyn FunctionLlm>,
    registry: ToolRegistry,
    config: AgentConfig,
}

impl std::fmt::Debug for FunctionAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionAgent")
            .field("tools", &self.registry.names())
            .field("model", &self.llm.model_name())
            .finish()
    }
}

impl FunctionAgent {
    /// 创建 Function Calling Agent。
    pub fn new(llm: Arc<dyn FunctionLlm>, registry: ToolRegistry, config: AgentConfig) -> Self {
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

    /// 跑一次 Agent。
    ///
    /// ## ★ 与 ReAct 的三个关键差异
    ///
    /// 1. **回执按 `call_id` 对账** —— 每个 `call_id` 恰好一条 `tool` 消息
    /// 2. **支持并行调用** —— 一次多个 `tool_calls`
    /// 3. **不用解析** —— 直接读结构化字段
    pub async fn run(
        &self,
        question: &str,
        facts: &[MemoryFact],
        window: &Window,
    ) -> AppResult<AgentRun> {
        let start = Instant::now();
        let question = question.trim();

        if question.is_empty() {
            return Err(crate::error::AppError::BadRequest("问题不能为空".into()));
        }

        let specs = self.registry.specs();
        let system = function_system_prompt(&specs, facts, &self.config.tool_choice);
        let ctx = ToolContext::with_window(*window);

        // 历史消息 —— 结构化协议下，这就是「短期记忆」。
        let mut messages: Vec<ChatMessage> = vec![ChatMessage::user(question.to_string())];

        let mut steps: Vec<StepRecord> = Vec::new();
        let mut tool_calls = 0usize;
        let mut tool_attempts = 0usize;
        let mut tools_used: Vec<String> = Vec::new();
        // ★ token 用量累加 + 截断标记（与 ReAct 版同语义）。
        let mut usage = crate::llm::LlmUsage::default();
        let mut truncated_any = false;

        for step_no in 1..=self.config.max_steps {
            // ── 调模型 ───────────────────────────────────────────
            let response = self
                .llm
                .chat_with_tools(&system, &messages, &specs, self.config.tool_choice.clone())
                .await?;

            // ★ 累加 token 用量（用 `llm` 模块的统一口径，避免两处算法不一致）。
            usage.merge(&crate::llm::usage_of(&response));

            // ⚠️★ **截断必须被检查** —— 被 `max_tokens` 截断的答案
            //    看起来是**完整的**（有头有尾，只少了尾巴），
            //    症状极隐蔽：数字少了，但读起来完全通顺。
            //
            // ⚠️ `stop_reason` 是 `Option<StopReason>` ——
            //    要用 `as_ref()` 读，否则会把 `response` **move 掉**，
            //    后面再读 `first_text()` 就编译不过。
            let truncated = response
                .stop_reason
                .as_ref()
                .is_some_and(|r| r.is_max_tokens());
            if truncated {
                truncated_any = true;
                agent_logging::warn!(
                    step = step_no,
                    "模型输出被 max_tokens 截断 —— 结果可能不完整"
                );
            }

            let text = response.first_text().unwrap_or_default().to_string();

            // ★ 思考内容 —— **必须回传**，否则推理模型直接 400。
            //
            // ⚠️ 用 `as_ref()` 读，避免把 `response` move 掉
            //    （后面还要 `into_tool_calls()`）。
            let reasoning = response.reasoning_content.as_deref().map(str::to_string);

            let calls: Vec<ToolCall> = response.into_tool_calls();

            // ── 没有工具调用 → 这就是最终答案 ─────────────────────
            if calls.is_empty() {
                steps.push(StepRecord {
                    step: step_no,
                    text: text.clone(),
                    tool_calls: Vec::new(),
                    final_answer: true,
                });

                return Ok(AgentRun {
                    question: question.to_string(),
                    answer: text,
                    steps,
                    tool_calls,
                    tool_attempts,
                    tools_used,
                    hit_max_steps: false,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    model: self.llm.model_name().to_string(),
                    usage,
                    truncated: truncated_any,
                    recalled_facts: facts.len(),
                });
            }

            // ── 有工具调用 → 执行 ────────────────────────────────
            //
            // ⚠️ 思考内容要**带上** —— 推理模型要求原样回传。
            messages.push(build_assistant_message(
                text.clone(),
                reasoning.as_deref(),
                &calls,
            ));

            let mut records = Vec::with_capacity(calls.len());
            let mut responses: Vec<ToolResponse> = Vec::with_capacity(calls.len());

            // ⚠️ 坑 1：**每个** call_id 都要有一条回执。
            //
            // 这里用一个显式的循环来保证「要么每个都有，要么一个都没有」——
            // 不存在「处理了一半」的中间状态。
            for call in &calls {
                let args = normalize_arguments(&call.fn_arguments);
                let (outcome, attempts) = self.execute_with_retry(&call.fn_name, &args, &ctx).await;

                tool_calls += 1;
                tool_attempts += attempts;

                if !tools_used.contains(&call.fn_name) {
                    tools_used.push(call.fn_name.clone());
                }

                records.push(ToolCallRecord {
                    call_id: call.call_id.clone(),
                    name: call.fn_name.clone(),
                    arguments: call.fn_arguments.clone(),
                    observation: outcome.observation.clone(),
                    ok: outcome.ok,
                    attempts,
                });

                // ⚠️ 关键：**失败也必须有回执**，
                //    否则下一轮请求会因为「insufficient tool messages」被 400。
                //
                //    回执内容要**剥掉 `@@DATA:` 尾巴** ——
                //    那坨 JSON 占 token 还干扰模型。
                responses.push(ToolResponse::from_tool_call(
                    call,
                    strip_data_tail(&outcome.observation),
                ));
            }

            // 把回执一次性加进历史（一条 assistant + N 条 tool）。
            messages.push(ChatMessage::tool(
                genai::chat::MessageContent::from_tool_responses(responses),
            ));

            steps.push(StepRecord {
                step: step_no,
                text,
                tool_calls: records,
                final_answer: false,
            });
        }

        // ── 达到步数上限 → 降级 ─────────────────────────────────
        agent_logging::warn!(
            max_steps = self.config.max_steps,
            tool_calls,
            "Function Calling 达到步数上限 —— 降级返回"
        );

        let answer = steps
            .last()
            .and_then(|s| s.tool_calls.first())
            .map(|c| c.observation.clone())
            .unwrap_or_else(|| {
                format!(
                    "达到最大步数（{}）仍未得到最终答案。请缩小问题范围后重试。",
                    self.config.max_steps
                )
            });

        Ok(AgentRun {
            question: question.to_string(),
            answer,
            steps,
            tool_calls,
            tool_attempts,
            tools_used,
            // ★ 这里**必须**是 `true` —— 原来写死成 `false`，
            //    症状是「明明降级返回了，响应里却说没到步数上限」，
            //    调用方会以为这是一个正常答案。
            //
            //    对照组：ReAct 版的同一位置就是 `true`。
            hit_max_steps: true,
            elapsed_ms: start.elapsed().as_millis() as u64,
            model: self.llm.model_name().to_string(),
            usage,
            truncated: truncated_any,
            recalled_facts: facts.len(),
        })
    }

    /// ★ 执行工具（带重试）—— 与 ReAct 版同语义。
    async fn execute_with_retry(
        &self,
        name: &str,
        args: &Value,
        ctx: &ToolContext,
    ) -> (ToolOutcome, usize) {
        let mut attempt = 0usize;

        loop {
            attempt += 1;

            match self.registry.call_strict(name, args, ctx).await {
                Ok(outcome) => return (outcome, attempt),

                Err(err) => {
                    if err.retryable() && attempt <= self.config.max_retries {
                        let delay = crate::tooling::backoff(
                            std::time::Duration::from_millis(self.config.retry_base_ms),
                            std::time::Duration::from_millis(self.config.retry_max_ms),
                            attempt,
                        );

                        agent_logging::warn!(
                            tool = %name,
                            code = %err.code.id(),
                            attempt,
                            delay_ms = delay.as_millis() as u64,
                            "Transient 错误 —— 重试"
                        );

                        tokio::time::sleep(delay).await;
                        continue;
                    }

                    agent_logging::warn!(
                        tool = %name,
                        code = %err.code.id(),
                        kind = %err.kind.name(),
                        attempt,
                        "工具失败 —— 错误将作为观察喂回模型"
                    );

                    return (ToolOutcome::failed(name, err.render()), attempt);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// 辅助函数
// ─────────────────────────────────────────────────────────────────────

/// 构造 ReAct 的 system 提示。
///
/// ## ⚠️ 事实**只**进这一段，**不进消息数组**
///
/// 否则几轮之后历史全是重复事实。
fn react_system_prompt(specs: &[crate::tool::ToolSpec], facts: &[MemoryFact]) -> String {
    let mut out = String::from(
        "你是一个活动记录分析助手。你可以使用下面列出的工具来查询真实数据。\n\n\
         **必须严格按下面的格式输出，不要有任何多余的解释：**\n\n\
         Thought: 你的思考\n\
         Action: 工具名\n\
         Action Input: JSON 参数\n\n\
         或者直接给出最终答案：\n\n\
         Final Answer: 最终答案\n\n\
         可用工具：\n",
    );

    for spec in specs {
        out.push_str(&format!(
            "- {}：{}（参数 Schema：{}）\n",
            spec.name, spec.description, spec.schema
        ));
    }

    if !facts.is_empty() {
        out.push_str("\n# 记忆\n以下是关于用户的已知事实，可直接使用：\n");
        for fact in facts {
            out.push_str(&format!("- {}\n", fact.render()));
        }
    }

    out.push_str(
        "\n## 硬约束\n\
         - 数字必须来自工具结果，**不要心算**（用 calculate 工具）\n\
         - 没有工具结果支撑的内容**不要写进最终答案**\n",
    );

    out
}

/// 构造 Function Calling 的 system 提示。
///
/// ## ⚠️ **不重复**工具的详细参数
///
/// 它们已经在 API 的 `tools` 字段里了，模型结构化看得到。
/// 提示词里再写一遍 → 两份描述可能不一致 → 模型陷入拉锯战。
///
/// ## ⚠️ `tool_choice = none` 时**连工具名都不列**
///
/// 16-agent-function-calling 实测：
/// 模型会用 `<toolcall>{"name":"get_weather"}</toolcall>` **文本**模拟调用。
/// 根因是「提示词里出现工具名它就会模仿」。
///
/// 所以「真正禁止用工具」= 不传 `tools` **且** 提示词里不出现工具名。
fn function_system_prompt(
    specs: &[crate::tool::ToolSpec],
    facts: &[MemoryFact],
    tool_choice: &ToolChoice,
) -> String {
    let mut out =
        String::from("你是一个活动记录分析助手。请使用提供的工具查询真实数据，然后回答问题。\n");

    if !facts.is_empty() {
        out.push_str("\n# 记忆\n以下是关于用户的已知事实，可直接使用：\n");
        for fact in facts {
            out.push_str(&format!("- {}\n", fact.render()));
        }
    }

    out.push_str(
        "\n## 硬约束\n\
         - 数字必须来自工具结果，**不要心算**（用 calculate 工具）\n\
         - 没有工具结果支撑的内容**不要写进最终答案**\n\
         - 需要统计多个模块时，**并行**发起多个工具调用\n",
    );

    // ⚠️ `none` 档：不列工具名（模型看到名字就会模仿调用）。
    if *tool_choice == ToolChoice::None {
        out.push_str("\n本轮**不允许**调用任何工具，请直接根据已有信息回答。\n");
        return out;
    }

    // 只列工具名 —— 详细参数在 `tools` 字段里。
    out.push_str("\n可用工具：");
    out.push_str(
        &specs
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push('\n');

    out
}

/// 把线程状态渲染成对话文本。
fn render_transcript(state: &ThreadState) -> String {
    if state.message_count() == 0 {
        return String::new();
    }

    let mut out = String::from("以下是已经完成的步骤（不是示例）：\n");
    for msg in &state.messages {
        let role = match msg.role {
            MemoryRole::User => "用户",
            MemoryRole::Assistant => "助手",
            MemoryRole::System => "系统",
            MemoryRole::Fact => "事实",
        };
        out.push_str(&format!("{role}：{}\n", msg.content));
    }
    out
}

/// 构造 assistant 消息（带工具调用）。
///
/// ## ⚠️★ **必须回传 `reasoning_content`**
///
/// 推理模型（DeepSeek / Kimi）**要求**把上一轮的思考内容原样回传，
/// 否则整轮请求会被 **400** 拒绝：
///
/// ```text
/// The reasoning_content in the assistant message must be passed back to the API
/// ```
///
/// 而且顺序必须是「**思考内容在工具调用之前**」——
/// genai 的 OpenAI 适配器会把 `ContentPart::ReasoningContent`
/// 提升成请求体里的 `reasoning_content` 兄弟字段。
///
/// ⚠️ 这就是 16-agent-function-calling 实测踩出来的坑 2：
/// 本项目原来的实现**丢掉了** `reasoning_content`，
/// 一旦有人把 `disable_reasoning` 打开，整个链路直接 400。
fn build_assistant_message(
    text: String,
    reasoning: Option<&str>,
    calls: &[ToolCall],
) -> ChatMessage {
    let mut parts: Vec<genai::chat::ContentPart> = Vec::new();

    // ★ 思考内容**先**放（顺序不能反）。
    if let Some(reasoning) = reasoning.filter(|r| !r.trim().is_empty()) {
        parts.push(genai::chat::ContentPart::ReasoningContent(
            reasoning.to_string(),
        ));
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
/// ## ⚠️ 为什么需要这一步
///
/// 有些模型返回 **JSON 字符串**（`"{\"scope\":\"cache\"}"`）
/// 而不是对象 —— 本地 Schema 校验要求 `Value::Object`，
/// 所以这里统一转换。
pub fn normalize_arguments(args: &Value) -> Value {
    match args {
        Value::Object(_) => args.clone(),
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(parsed @ Value::Object(_)) => parsed,
            // 解析失败 —— 交给校验器报错（它会说「必须是 JSON 对象」）。
            _ => args.clone(),
        },
        Value::Null => Value::Object(serde_json::Map::new()),
        other => other.clone(),
    }
}

/// ★ 剥掉观察里的 `@@DATA:{...}` 结构化尾巴。
///
/// ## 为什么
///
/// 那坨 JSON 是给**收集端**用的（解开它拿结构化证据），
/// 但发给模型时：
///
/// 1. 占 token
/// 2. **干扰模型** —— 它会把 JSON 当成要回答的一部分
///
/// 所以「收集 → 解码 → 剥掉 → 发给模型」是一条固定的流水线。
pub fn strip_data_tail(observation: &str) -> String {
    match observation.find("\n@@DATA:") {
        Some(idx) => observation[..idx].trim_end().to_string(),
        None => observation.to_string(),
    }
}

/// ★ 从观察里**解回**结构化数据。
///
/// 与 [`strip_data_tail`] 是一对：一个是「发给模型前剥掉」，
/// 这个是「收集端解回来」。
pub fn extract_data(observation: &str) -> Option<Value> {
    let idx = observation.find("\n@@DATA:")?;
    let json = observation[idx + "\n@@DATA:".len()..].trim();
    serde_json::from_str(json).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ScriptedFunctionLlm, ScriptedLlm, ScriptedStep, ScriptedToolCall};
    use crate::tool::{CalculateTool, ModuleStatsTool, ToolSpec};
    use serde_json::json;
    use std::sync::Arc;

    fn window() -> Window {
        Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400)
    }

    fn activities(n: usize) -> Arc<Vec<crate::model::Activity>> {
        Arc::new(
            (0..n)
                .map(|i| crate::model::Activity {
                    id: format!("a-{i:04}"),
                    ts: 1_700_000_000 + (i as i64 % 7) * 86_400,
                    kind: ["feature", "fix"][i % 2].to_string(),
                    author: ["alice", "bob"][i % 2].to_string(),
                    scope: ["cache", "rag"][i % 2].to_string(),
                    title: format!("任务 {i}"),
                    weight: (i % 10) as i32,
                    risk: false,
                })
                .collect(),
        )
    }

    fn registry() -> ToolRegistry {
        ToolRegistry::new(vec![
            Arc::new(ModuleStatsTool::new(activities(20))),
            Arc::new(CalculateTool),
        ])
    }

    // ── ReAct ────────────────────────────────────────────────

    #[tokio::test]
    async fn react_runs_the_full_loop_and_calls_a_tool() {
        let llm = Arc::new(ScriptedLlm::new(vec![
            "Thought: 查一下\nAction: module_stats\nAction Input: {\"scope\": \"cache\"}"
                .to_string(),
            "Final Answer: cache 模块共 10 条活动。".to_string(),
        ]));

        let agent = ReactAgent::new(llm, registry(), AgentConfig::default());
        let run = agent
            .run("cache 模块做了什么？", &[], &window())
            .await
            .unwrap();

        assert_eq!(run.answer, "cache 模块共 10 条活动。");
        assert_eq!(run.tool_calls, 1);
        assert_eq!(run.steps.len(), 2);
        assert!(run.tools_used.contains(&"module_stats".to_string()));
        assert!(!run.hit_max_steps);
    }

    #[tokio::test]
    async fn react_tool_failure_becomes_an_observation_not_an_error() {
        let llm = Arc::new(ScriptedLlm::new(vec![
            "Action: module_stats\nAction Input: {}".to_string(),
            "Final Answer: 参数错了，已纠正。".to_string(),
        ]));

        let agent = ReactAgent::new(llm, registry(), AgentConfig::default());
        let run = agent.run("查一下", &[], &window()).await.unwrap();

        // ★ 工具失败**不是**请求失败 —— 模型拿到观察后继续。
        assert_eq!(run.tool_calls, 1);
        assert!(!run.steps[0].tool_calls[0].ok);
        assert!(
            run.steps[0].tool_calls[0]
                .observation
                .contains("E_INVALID_ARGUMENTS")
        );
        assert_eq!(run.answer, "参数错了，已纠正。");
    }

    #[tokio::test]
    async fn react_degrades_when_max_steps_is_hit() {
        let llm = Arc::new(ScriptedLlm::always(
            "Action: module_stats\nAction Input: {\"scope\": \"cache\"}",
        ));

        let config = AgentConfig {
            max_steps: 2,
            ..Default::default()
        };
        let agent = ReactAgent::new(llm, registry(), config);
        let run = agent.run("查一下", &[], &window()).await.unwrap();

        // ⚠️ 达到步数上限要**降级**，不能报错。
        assert!(run.hit_max_steps);
        assert!(!run.answer.is_empty());
    }

    #[tokio::test]
    async fn react_facts_go_into_the_system_prompt_only() {
        // ⚠️ `Arc<ScriptedLlm>` **不会**自动 coerce 成 `Arc<dyn Llm>` ——
        //    必须先转成 trait object 再 clone。
        //    保留一个具体类型的句柄，才能断言「事实进了 system 提示」。
        let concrete = Arc::new(ScriptedLlm::always("Final Answer: 好的"));
        let llm: Arc<dyn Llm> = concrete.clone();
        let agent = ReactAgent::new(llm, registry(), AgentConfig::default());

        let facts = vec![MemoryFact::new("用户", "名字", "Bob", "bob")];
        let run = agent.run("我叫什么？", &facts, &window()).await.unwrap();

        assert_eq!(run.recalled_facts, 1);

        // ★ 事实必须在 system 提示里（模型看得到）。
        let calls = concrete.calls();
        assert!(calls[0].system.contains("Bob"));
    }

    // ── Function Calling ─────────────────────────────────────

    #[tokio::test]
    async fn function_calling_runs_the_full_loop() {
        let llm = Arc::new(ScriptedFunctionLlm::new(vec![
            ScriptedStep::call("c1", "module_stats", json!({"scope": "cache"})),
            ScriptedStep::text("cache 模块共 10 条活动。"),
        ]));

        let agent = FunctionAgent::new(llm, registry(), AgentConfig::default());
        let run = agent
            .run("cache 模块做了什么？", &[], &window())
            .await
            .unwrap();

        assert_eq!(run.answer, "cache 模块共 10 条活动。");
        assert_eq!(run.tool_calls, 1);
        assert_eq!(run.steps[0].tool_calls[0].call_id, "c1");
    }

    #[tokio::test]
    async fn function_calling_produces_one_receipt_per_call_id() {
        // 模型一次返回 2 个调用 —— 必须生成 2 条回执。
        let llm = Arc::new(ScriptedFunctionLlm::new(vec![
            ScriptedStep::calls(vec![
                ScriptedToolCall::new("c1", "module_stats", json!({"scope": "cache"})),
                ScriptedToolCall::new("c2", "module_stats", json!({"scope": "rag"})),
            ]),
            ScriptedStep::text("两个模块都查完了。"),
        ]));

        let agent = FunctionAgent::new(llm, registry(), AgentConfig::default());
        let run = agent.run("两个模块", &[], &window()).await.unwrap();

        assert_eq!(run.tool_calls, 2);
        // ★ 每个 call_id 恰好一条回执 —— 少一条就是 400。
        let call_ids: Vec<&str> = run.steps[0]
            .tool_calls
            .iter()
            .map(|c| c.call_id.as_str())
            .collect();
        assert_eq!(call_ids, vec!["c1", "c2"]);
    }

    #[tokio::test]
    async fn function_calling_failure_still_produces_a_receipt() {
        let llm = Arc::new(ScriptedFunctionLlm::new(vec![
            ScriptedStep::call("c1", "module_stats", json!({})),
            ScriptedStep::text("参数不对，已纠正。"),
        ]));

        let agent = FunctionAgent::new(llm, registry(), AgentConfig::default());
        let run = agent.run("查一下", &[], &window()).await.unwrap();

        assert_eq!(run.tool_calls, 1);
        assert!(!run.steps[0].tool_calls[0].ok);
        assert!(
            run.steps[0].tool_calls[0]
                .observation
                .contains("E_INVALID_ARGUMENTS")
        );
    }

    #[tokio::test]
    async fn llm_failure_is_a_bad_gateway_not_a_degradation() {
        // 脚本耗尽 → LLM 调用失败。
        let llm = Arc::new(ScriptedFunctionLlm::new(vec![]));
        let agent = FunctionAgent::new(llm, registry(), AgentConfig::default());

        let err = agent.run("查一下", &[], &window()).await.unwrap_err();
        // ★ 模型失败是 502 —— **不能**降级（没有模型就没有答案）。
        assert_eq!(err.status_code(), axum::http::StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn function_calling_degrades_when_max_steps_is_hit() {
        // ⚠️ 回归测试：原来 `hit_max_steps` 被写死成 `false`，
        //    症状是「明明降级返回了，响应里却说没到步数上限」，
        //    调用方会以为这是一个正常答案。
        let llm = Arc::new(ScriptedFunctionLlm::always(ScriptedStep::call(
            "c1",
            "module_stats",
            json!({"scope": "cache"}),
        )));

        let config = AgentConfig {
            max_steps: 2,
            ..Default::default()
        };
        let agent = FunctionAgent::new(llm, registry(), config);
        let run = agent.run("查一下", &[], &window()).await.unwrap();

        assert!(run.hit_max_steps, "达到步数上限必须标记出来");
        assert!(!run.answer.is_empty());
    }

    #[tokio::test]
    async fn reasoning_content_is_echoed_back_to_the_model() {
        // ⚠️★ 推理模型（DeepSeek / Kimi）**要求**把思考内容原样回传，
        //    否则整个请求 400。
        //
        // 这里用一个「会拒绝没有 reasoning_content 的请求」的假 LLM
        // 来模拟 —— 断言 assistant 消息里**确实带了**思考内容。
        let llm = Arc::new(ScriptedFunctionLlm::new(vec![
            ScriptedStep::ReasoningAndCalls(
                "我先查一下 cache 模块".to_string(),
                vec![ScriptedToolCall::new(
                    "c1",
                    "module_stats",
                    json!({"scope": "cache"}),
                )],
            ),
            ScriptedStep::text("cache 模块共 10 条活动。"),
        ]));

        let agent = FunctionAgent::new(llm, registry(), AgentConfig::default());
        let run = agent
            .run("cache 模块做了什么？", &[], &window())
            .await
            .unwrap();

        assert_eq!(run.answer, "cache 模块共 10 条活动。");
        assert_eq!(run.tool_calls, 1);
    }

    #[test]
    fn assistant_message_carries_the_reasoning_content() {
        // ★ 直接测构造函数 —— 断言「思考内容在工具调用之前」。
        let calls = vec![ToolCall {
            call_id: "c1".into(),
            fn_name: "module_stats".into(),
            fn_arguments: json!({"scope": "cache"}),
            thought_signatures: None,
        }];

        let message = build_assistant_message("我先查一下".to_string(), Some("思考过程"), &calls);

        let parts = message.content.parts();
        assert_eq!(parts.len(), 3);
        assert!(matches!(
            parts[0],
            genai::chat::ContentPart::ReasoningContent(_)
        ));
        assert!(matches!(parts[1], genai::chat::ContentPart::Text(_)));
        assert!(matches!(parts[2], genai::chat::ContentPart::ToolCall(_)));
    }

    #[test]
    fn assistant_message_without_reasoning_is_unchanged() {
        let calls = vec![ToolCall {
            call_id: "c1".into(),
            fn_name: "module_stats".into(),
            fn_arguments: json!({}),
            thought_signatures: None,
        }];

        // ⚠️ 空字符串的思考内容**不能**变成一个空 part ——
        //    那会让请求多出一个空字段。
        let message = build_assistant_message("text".to_string(), Some("   "), &calls);
        let parts = message.content.parts();
        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], genai::chat::ContentPart::Text(_)));
    }

    #[tokio::test]
    async fn react_accumulates_token_usage() {
        // ★ token 用量是比耗时更稳定的归因依据。
        let llm = Arc::new(ScriptedLlm::always("Final Answer: 好的").with_usage(
            crate::llm::LlmUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
            },
        ));

        let agent = ReactAgent::new(llm, registry(), AgentConfig::default());
        let run = agent.run("查一下", &[], &window()).await.unwrap();

        assert_eq!(run.usage.prompt_tokens, 100);
        assert_eq!(run.usage.completion_tokens, 20);
        assert_eq!(run.usage.total_tokens, 120);
    }

    #[tokio::test]
    async fn truncated_output_is_marked() {
        // ⚠️ 被 `max_tokens` 截断的答案看起来是**完整的** ——
        //    有头有尾，只少了尾巴。必须标出来。
        let llm = Arc::new(ScriptedLlm::always("Final Answer: 被截断了").truncated());

        let agent = ReactAgent::new(llm, registry(), AgentConfig::default());
        let run = agent.run("查一下", &[], &window()).await.unwrap();

        assert!(run.truncated);
    }

    #[tokio::test]
    async fn both_protocols_use_the_same_tool_set() {
        // ★ 协议对照实验的前提：两边工具完全一样。
        let react = ReactAgent::new(
            Arc::new(ScriptedLlm::always("Final Answer: x")),
            registry(),
            AgentConfig::default(),
        );
        let function = FunctionAgent::new(
            Arc::new(ScriptedFunctionLlm::always(ScriptedStep::text("x"))),
            registry(),
            AgentConfig::default(),
        );

        assert_eq!(
            react.registry().names(),
            function.registry().names(),
            "两条协议必须用同一份工具注册表，否则对照实验不成立"
        );
    }

    // ── 辅助函数 ─────────────────────────────────────────────

    #[test]
    fn strips_the_data_tail() {
        let observation = "模块 cache：共 3 条\n@@DATA:{\"total\":3}";
        let stripped = strip_data_tail(observation);
        assert_eq!(stripped, "模块 cache：共 3 条");
        assert!(!stripped.contains("@@DATA"));
    }

    #[test]
    fn extracts_the_data_back() {
        let observation = "模块 cache：共 3 条\n@@DATA:{\"total\":3}";
        let data = extract_data(observation).unwrap();
        assert_eq!(data["total"], json!(3));
    }

    #[test]
    fn extract_data_returns_none_without_a_tail() {
        assert!(extract_data("普通观察").is_none());
    }

    #[test]
    fn normalizes_json_string_arguments() {
        let args = json!("{\"scope\": \"cache\"}");
        let normalized = normalize_arguments(&args);
        assert_eq!(normalized["scope"], json!("cache"));
    }

    #[test]
    fn function_prompt_does_not_duplicate_tool_parameters() {
        let specs = vec![ToolSpec::new(
            "module_stats",
            "统计模块",
            json!({"type": "object", "properties": {"scope": {"type": "string"}}}),
        )];
        let prompt = function_system_prompt(&specs, &[], &ToolChoice::Auto);

        // ★ 提示词里只列工具名，**不重复** Schema。
        assert!(prompt.contains("module_stats"));
        assert!(!prompt.contains("properties"));
    }

    #[test]
    fn function_prompt_hides_tool_names_when_tools_are_disabled() {
        // ⚠️★ 16-agent-function-calling 实测的坑：
        //    `tool_choice = none` 时模型会用**文本**模拟调用
        //    （`<toolcall>{"name":"..."}</toolcall>`），
        //    根因是「提示词里出现工具名它就会模仿」。
        //
        //    所以 `none` 档提示词里**不能出现工具名**。
        let specs = vec![ToolSpec::new(
            "module_stats",
            "统计模块",
            json!({"type": "object", "properties": {"scope": {"type": "string"}}}),
        )];
        let prompt = function_system_prompt(&specs, &[], &ToolChoice::None);

        assert!(!prompt.contains("module_stats"));
        assert!(prompt.contains("不允许"));
    }

    #[test]
    fn react_prompt_lists_the_tools_with_schemas() {
        let specs = vec![ToolSpec::new(
            "module_stats",
            "统计模块",
            json!({"type": "object", "properties": {"scope": {"type": "string"}}}),
        )];
        let prompt = react_system_prompt(&specs, &[]);

        // ReAct 协议**必须**在提示词里给全 Schema（模型看不到结构化字段）。
        assert!(prompt.contains("module_stats"));
        assert!(prompt.contains("properties"));
    }
}
