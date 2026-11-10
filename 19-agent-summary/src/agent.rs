//! Agent 循环 —— 「研究助手」的主循环。
//!
//! # 循环骨架
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │ 1. 调模型（带 5 个工具）                                   │
//! │ 2. 响应里有 tool_calls？                                 │
//! │    ├─ 没有 → 这是最终答案，收工                            │
//! │    └─ 有   → 对**每个** call：                            │
//! │             ① middleware.wrap_tool_call(request)         │
//! │               ├─ 成功   → 收集证据 + observation          │
//! │               ├─ 降级   → observation = 错误文本           │
//! │               └─ 致命   → observation = 错误文本 + **终止** │
//! │ 3. 为**每个** call_id 生成**恰好一条** tool 消息            │
//! │ 4. 回到 1                                                │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! # ⚠️ 三个不变量（全部继承自 18）
//!
//! ### 不变量 1：每个 `call_id` 恰好一条回执
//!
//! 少一条 → 整个请求 400（`insufficient tool messages following tool_calls message`）。
//! **致命错误也要生成回执** —— 否则同样 400。
//!
//! ### 不变量 2：致命错误终止后，**不返回模型的答案**
//!
//! 模型可能在上一步说了些话，但那不是「基于工具结果的答案」。
//! 鉴权失败时把一句闲聊当答案返回，等于**假装成功**。
//!
//! ### 不变量 3：模型调用失败**不能**降级
//!
//! 「没有模型就没有答案」—— 模型调用失败时没有观察可喂，只能终止。
//!
//! # 本项目新增的东西：证据收集
//!
//! 18 的循环只关心「工具调没调成」。
//! 本项目还要关心「**查到了什么**」——
//! 所以每次工具成功，都把 `data` 里的结构化结果转成 [`Evidence`] 收进
//! [`EvidenceCollector`]，供后面的报告生成引用。

use std::sync::Arc;
use std::time::Instant;

use agent_16::agent::{build_assistant_message, normalize_arguments};
use agent_18::middleware::{ErrorHandlingMiddleware, MiddlewareOutcome, ToolCallRequest};
use agent_logging::{debug, info, warn};
use genai::chat::{ChatMessage, MessageContent, ToolCall, ToolResponse};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AppError, AppResult};
use crate::evidence::EvidenceCollector;
use crate::llm::{FunctionLlm, ToolChoice};
use crate::tools::ToolSpec;

/// 一次工具调用的记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// 调用 ID。
    pub call_id: String,

    /// 工具名。
    pub name: String,

    /// 参数。
    pub arguments: Value,

    /// 给模型看的观察文本。
    pub observation: String,

    /// 工具是否成功。
    pub ok: bool,

    /// 实际尝试了几次。
    pub attempts: usize,

    /// 错误码（失败时）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,

    /// 错误类别（失败时）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,

    /// 这一步收集到的证据编号。
    #[serde(default)]
    pub evidence: Vec<String>,
}

/// 一步的执行记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// 第几步（从 1 开始）。
    pub step: usize,

    /// 模型输出的文本。
    pub text: String,

    /// 本步的工具调用。
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,

    /// 本步是否给出了最终答案。
    pub final_answer: bool,
}

/// 一次 Agent 运行的完整结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRun {
    /// 原始问题。
    pub question: String,

    /// 最终答案（模型的自然语言回答）。
    pub answer: String,

    /// 每一步的记录。
    pub steps: Vec<StepRecord>,

    /// 工具调用总次数（**含**重试）。
    pub tool_calls: usize,

    /// 逻辑上的工具调用数（不含重试）。
    pub tool_call_groups: usize,

    /// 被重试过的调用次数。
    pub retried_calls: usize,

    /// 降级成观察的错误数。
    pub degraded_errors: usize,

    /// 用到的工具名（去重，按首次出现顺序）。
    pub tools_used: Vec<String>,

    /// 收集到的证据。
    pub evidence: Vec<crate::evidence::Evidence>,

    /// 是否因为达到步数上限而中止。
    pub hit_max_steps: bool,

    /// 是否因为**致命错误**而终止。
    pub fatal_terminated: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,
}

/// Agent 循环配置。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 最大步数。
    pub max_steps: usize,

    /// 工具选择策略。
    pub tool_choice: ToolChoice,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 6,
            tool_choice: ToolChoice::Auto,
        }
    }
}

/// 研究助手 Agent。
pub struct Agent {
    llm: Arc<dyn FunctionLlm>,
    middleware: ErrorHandlingMiddleware,
    config: AgentConfig,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("tools", &self.middleware.registry().names())
            .field("max_steps", &self.config.max_steps)
            .field("model", &self.llm.model_name())
            .finish()
    }
}

impl Agent {
    /// 创建 Agent。
    pub fn new(
        llm: Arc<dyn FunctionLlm>,
        middleware: ErrorHandlingMiddleware,
        config: AgentConfig,
    ) -> Self {
        Self {
            llm,
            middleware,
            config,
        }
    }

    /// 工具定义列表。
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.middleware.registry().specs()
    }

    /// 中间件（API 层直接调工具时复用重试逻辑）。
    pub fn middleware(&self) -> &ErrorHandlingMiddleware {
        &self.middleware
    }

    /// 跑一次研究任务。
    ///
    /// 返回值里带**完整证据链** —— 报告生成阶段靠它做引用校验。
    pub async fn research(
        &self,
        question: &str,
        system: &str,
    ) -> AppResult<(AgentRun, EvidenceCollector)> {
        let start = Instant::now();
        let question = question.trim();

        if question.is_empty() {
            return Err(AppError::BadRequest("问题不能为空".to_string()));
        }

        let specs = self.tool_specs();
        let mut collector = EvidenceCollector::new();

        let mut messages: Vec<ChatMessage> = vec![ChatMessage::user(question.to_string())];
        let mut steps: Vec<StepRecord> = Vec::new();
        let mut tool_calls = 0usize;
        let mut tool_call_groups = 0usize;
        let mut retried_calls = 0usize;
        let mut degraded_errors = 0usize;
        let mut tools_used: Vec<String> = Vec::new();

        for step_no in 1..=self.config.max_steps {
            debug!(step = step_no, tools = specs.len(), "第 {step_no} 步");

            // ── 调模型 ──────────────────────────────────────────
            //
            // ⚠️ 模型调用失败**不能**降级 ——
            //    「没有模型就没有答案」，没有观察可喂。
            let response = self
                .llm
                .chat_with_tools(system, &messages, &specs, self.config.tool_choice.clone())
                .await?;

            let text = response.first_text().unwrap_or_default().to_string();
            // ⚠️★ 思考内容**必须回传**（16 的坑 2）——
            //    推理模型在思考模式下要求把 `reasoning_content` 原样带回，
            //    否则下一轮请求直接 400：
            //    `The reasoning_content in the thinking mode must be passed back to the API.`
            //
            //    本项目原先**没有**回传，思考模式一开就 400（潜伏的真 bug）。
            let reasoning = response.reasoning_content.clone();
            let calls: Vec<ToolCall> = response.into_tool_calls();

            // ── 没有工具调用 → 最终答案 ─────────────────────────
            if calls.is_empty() {
                steps.push(StepRecord {
                    step: step_no,
                    text: text.clone(),
                    tool_calls: Vec::new(),
                    final_answer: true,
                });

                info!(
                    step = step_no,
                    tool_calls,
                    evidence = collector.len(),
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "Agent 给出最终答案"
                );

                return Ok((
                    AgentRun {
                        question: question.to_string(),
                        answer: text,
                        steps,
                        tool_calls,
                        tool_call_groups,
                        retried_calls,
                        degraded_errors,
                        tools_used,
                        evidence: collector.all().into_iter().cloned().collect(),
                        hit_max_steps: false,
                        fatal_terminated: false,
                        elapsed_ms: start.elapsed().as_millis() as u64,
                        model: self.llm.model_name().to_string(),
                    },
                    collector,
                ));
            }

            // ── 有工具调用 → 走中间件 ──────────────────────────
            // ⚠️ 复用 16 的 `build_assistant_message`（带 reasoning_content）——
            //    本项目原先自己写了一份**不带**思考内容的版本，
            //    思考模式一开就会被 provider 拒（400）。
            messages.push(build_assistant_message(
                text.clone(),
                reasoning.clone(),
                &calls,
            ));

            let mut records = Vec::with_capacity(calls.len());
            let mut responses: Vec<ToolResponse> = Vec::with_capacity(calls.len());
            let mut fatal: Option<crate::error::ToolError> = None;

            for call in &calls {
                let request = ToolCallRequest {
                    name: call.fn_name.clone(),
                    arguments: normalize_arguments(&call.fn_arguments),
                    call_id: call.call_id.clone(),
                };

                let outcome = self.middleware.wrap_tool_call(&request).await;

                tool_calls += outcome.attempts;
                tool_call_groups += 1;
                if outcome.attempts > 1 {
                    retried_calls += 1;
                }
                if !outcome.ok {
                    degraded_errors += 1;
                }

                if !tools_used.contains(&call.fn_name) {
                    tools_used.push(call.fn_name.clone());
                }

                // ── 收集证据 ────────────────────────────────────
                //
                // ⚠️ 只有**成功**的调用才有证据 ——
                //    降级后的错误文本不是知识。
                let evidence_ids = if outcome.error.is_some() {
                    Vec::new()
                } else {
                    collector.add_from_tool_data(&call.fn_name, &extract_data(&outcome))
                };

                // ⚠️ 发给模型的观察文本要**剥掉数据行** ——
                //    那坨 JSON 对模型没用，只会占 token 和干扰阅读。
                let visible = render_observation(&outcome.observation);

                records.push(ToolCallRecord {
                    call_id: call.call_id.clone(),
                    name: call.fn_name.clone(),
                    arguments: request.arguments.clone(),
                    observation: visible.clone(),
                    ok: outcome.ok,
                    attempts: outcome.attempts,
                    error_code: outcome
                        .error
                        .as_ref()
                        .map(|e| e.diagnostic_code().to_string()),
                    error_kind: outcome.error.as_ref().map(|e| e.kind.name().to_string()),
                    evidence: evidence_ids,
                });

                // ⚠️ 不变量 1：**每个** call_id 恰好一条回执。
                //
                // 包括致命错误 —— 少一条就 400。
                responses.push(ToolResponse::from_tool_call(call, visible));

                // 致命错误 → 记下来，但**继续把回执生成完**。
                if outcome.fatal {
                    fatal = outcome.error.clone();
                }
            }

            messages.push(ChatMessage::tool(MessageContent::from_tool_responses(
                responses,
            )));

            steps.push(StepRecord {
                step: step_no,
                text,
                tool_calls: records,
                final_answer: false,
            });

            // ⚠️ 不变量 2：致命错误 → 终止，**不返回模型的答案**。
            if let Some(err) = fatal {
                warn!(
                    step = step_no,
                    code = err.diagnostic_code(),
                    "致命错误 —— 终止 Agent 循环"
                );

                return Err(AppError::from_tool_error(&err));
            }
        }

        // ── 达到步数上限 ────────────────────────────────────────
        warn!(
            max_steps = self.config.max_steps,
            tool_calls, "达到步数上限 —— 降级返回"
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

        Ok((
            AgentRun {
                question: question.to_string(),
                answer,
                steps,
                tool_calls,
                tool_call_groups,
                retried_calls,
                degraded_errors,
                tools_used,
                evidence: collector.all().into_iter().cloned().collect(),
                hit_max_steps: true,
                fatal_terminated: false,
                elapsed_ms: start.elapsed().as_millis() as u64,
                model: self.llm.model_name().to_string(),
            },
            collector,
        ))
    }
}

/// ⚠️ 从中间件结果里取回工具的 `data`。
///
/// ## 为什么这里有个「提取器」
///
/// 18 的 `MiddlewareOutcome` **只保留了 observation 文本**，
/// 没有保留 `ToolOutcome.data` —— 因为 18 的循环不需要结构化结果。
///
/// 但本项目**需要**：证据就是从 `data` 里提取的。
///
/// ## 采用的办法：数据行约定
///
/// 工具把结构化结果编码成 observation 的**最后一行**：
///
/// ```text
/// @@DATA:{"hits":[...]}
/// ```
///
/// 循环做两件事：
///
/// 1. **收集证据时** —— 用 [`parse_data_line`] 把这行解回来
/// 2. **发给模型时** —— 用 [`render_observation`] 把这行**剥掉**
///
/// ## 为什么不让模型看到这行
///
/// 那是一坨 JSON 噪声 —— 占 token、干扰阅读，
/// 而且模型可能尝试「引用」里面的字段名。
///
/// ## 为什么不用「工具名 + 观察文本」反推
///
/// 观察文本是**给模型看的**，格式可能变；
/// 而这行数据是**工具自己写的**，格式由本项目约定，稳定。
fn extract_data(outcome: &MiddlewareOutcome) -> Value {
    parse_data_line(&outcome.observation)
}

/// 结构化数据在 observation 里的标记前缀。
pub const DATA_PREFIX: &str = "@@DATA:";

/// 从 observation 文本里解析结构化数据。
///
/// ## 约定
///
/// 工具返回的 observation 末尾可以带一行：
///
/// ```text
/// @@DATA:{"hits":[...]}
/// ```
///
/// 这行**不会**给模型看（渲染前会被剥掉），
/// 但循环能拿到结构化结果去做证据收集。
pub fn parse_data_line(observation: &str) -> Value {
    observation
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(DATA_PREFIX))
        .and_then(|json| serde_json::from_str(json).ok())
        .unwrap_or(Value::Null)
}

/// 渲染 observation（剥掉结构化数据行）。
pub fn render_observation(observation: &str) -> String {
    observation
        .lines()
        .filter(|line| !line.starts_with(DATA_PREFIX))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalizes_object_arguments() {
        let args = json!({ "query": "rust" });
        assert_eq!(normalize_arguments(&args), args);
    }

    #[test]
    fn normalizes_json_string_arguments() {
        let args = json!("{\"query\": \"rust\"}");
        let normalized = normalize_arguments(&args);
        assert_eq!(normalized["query"], json!("rust"));
    }

    #[test]
    fn null_arguments_become_empty_object() {
        assert_eq!(normalize_arguments(&Value::Null), json!({}));
    }

    #[test]
    fn parses_data_line() {
        let observation = "一些观察文本\n@@DATA:{\"hits\":[{\"source\":\"a.md\"}]}";
        let data = parse_data_line(observation);
        assert_eq!(data["hits"][0]["source"], json!("a.md"));
    }

    #[test]
    fn observation_without_data_line_yields_null() {
        assert_eq!(parse_data_line("普通观察"), Value::Null);
    }

    #[test]
    fn render_strips_data_line() {
        let observation = "可见文本\n@@DATA:{\"x\":1}";
        assert_eq!(render_observation(observation), "可见文本");
    }
}
