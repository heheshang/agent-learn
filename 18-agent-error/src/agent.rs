//! Agent 循环 —— 把中间件接进结构化工具调用协议。
//!
//! ## 循环骨架
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │ 1. 调模型（带 tools）                                        │
//! │ 2. 响应里有 tool_calls？                                     │
//! │    ├─ 没有 → 最终答案，收工                                  │
//! │    └─ 有   → 对**每个** call：                                │
//! │             ② middleware.wrap_tool_call(request)            │
//! │               ├─ 成功   → observation = 结果                 │
//! │               ├─ 降级   → observation = 错误文本（可诊断）    │
//! │               └─ 致命   → observation = 错误文本 + **终止**   │
//! │ 3. 为**每个** call_id 生成**恰好一条** tool 消息               │
//! │ 4. 回到 1                                                   │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## ⚠️ 三个不变量
//!
//! ### 不变量 1：每个 `call_id` 恰好一条回执
//!
//! 这是 16 实测踩出来的坑：
//! 少一条 → 整个请求 400（`insufficient tool messages following tool_calls message`）。
//!
//! ⚠️ **致命错误也要生成回执** —— 否则同样 400。
//! 这就是为什么本项目把「降级」和「终止」分成两步：
//! 先把错误写成一条 `tool` 消息（保证回执完整），再终止循环。
//!
//! ### 不变量 2：致命错误终止后，**不返回模型的答案**
//!
//! 模型可能在上一步已经说了些话，但那不是「基于工具结果的答案」。
//! 鉴权失败时把一句闲聊当答案返回，等于**假装成功**。
//!
//! 所以终止时返回 `Err`，让 HTTP 层给出 500。
//!
//! ### 不变量 3：模型调用失败**不能**降级
//!
//! 「没有模型就没有答案」——
//! 模型调用失败时没有观察可喂，只能终止。
//! 这是 [`crate::error::ErrorKind`] 里没有「模型错误」这一类的��
//! 原因：模型失败只有一种处置。

use std::sync::Arc;
use std::time::Instant;

use agent_16::agent::{build_assistant_message, normalize_arguments};
use agent_logging::{debug, info, warn};
use genai::chat::{ChatMessage, MessageContent, ToolCall, ToolResponse};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AppError, AppResult};
use crate::llm::{FunctionLlm, ToolChoice};
use crate::middleware::{ErrorHandlingMiddleware, MiddlewareOutcome, ToolCallRequest};
use crate::prompt;
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

    /// 这个错误有没有被**重试**过。
    #[serde(default)]
    pub retried: bool,
}

impl ToolCallRecord {
    /// 从中间件结果构造。
    fn from_outcome(call: &ToolCall, outcome: &MiddlewareOutcome) -> Self {
        Self {
            call_id: call.call_id.clone(),
            name: call.fn_name.clone(),
            arguments: call.fn_arguments.clone(),
            observation: outcome.observation.clone(),
            ok: outcome.ok,
            attempts: outcome.attempts,
            error_code: outcome
                .error
                .as_ref()
                .map(|e| e.diagnostic_code().to_string()),
            error_kind: outcome.error.as_ref().map(|e| e.kind.name().to_string()),
            retried: outcome.attempts > 1,
        }
    }
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

    /// 最终答案。
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

    /// 用到的工具名（去重）。
    pub tools_used: Vec<String>,

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
            max_steps: 5,
            tool_choice: ToolChoice::Auto,
        }
    }
}

/// 带错误处理的 Agent。
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

    /// 中间件（供 API 层直接调工具时复用重试逻辑）。
    pub fn middleware(&self) -> &ErrorHandlingMiddleware {
        &self.middleware
    }

    /// 跑一次 Agent。
    pub async fn run(&self, question: &str) -> AppResult<AgentRun> {
        let start = Instant::now();
        let question = question.trim();

        if question.is_empty() {
            return Err(AppError::BadRequest("问题不能为空".to_string()));
        }

        let specs = self.tool_specs();
        let system = prompt::system_prompt();

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
                .chat_with_tools(&system, &messages, &specs, self.config.tool_choice.clone())
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
                    degraded_errors,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "Agent 给出最终答案"
                );

                return Ok(AgentRun {
                    question: question.to_string(),
                    answer: text,
                    steps,
                    tool_calls,
                    tool_call_groups,
                    retried_calls,
                    degraded_errors,
                    tools_used,
                    hit_max_steps: false,
                    fatal_terminated: false,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    model: self.llm.model_name().to_string(),
                });
            }

            // ── 有工具调用 → 走中间件 ──────────────────────────
            //
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

                records.push(ToolCallRecord::from_outcome(call, &outcome));

                // ⚠️ 不变量 1：**每个** call_id 恰好一条回执。
                //
                // 包括致命错误 —— 少一条就 400。
                responses.push(ToolResponse::from_tool_call(
                    call,
                    outcome.observation.clone(),
                ));

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

        Ok(AgentRun {
            question: question.to_string(),
            answer,
            steps,
            tool_calls,
            tool_call_groups,
            retried_calls,
            degraded_errors,
            tools_used,
            hit_max_steps: true,
            fatal_terminated: false,
            elapsed_ms: start.elapsed().as_millis() as u64,
            model: self.llm.model_name().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorCode;
    use crate::llm::{LlmStep, ScriptedLlm, ScriptedToolCall, step_calls, step_failure, step_text};
    use crate::middleware::ErrorHandlingMiddleware;
    use crate::tool::{
        FLAKY_FETCH_TOOL_NAME, FlakyTool, GUARDED_PAY_TOOL_NAME, GuardedTool,
        STRICT_LOOKUP_TOOL_NAME, StrictTool, ToolRegistry,
    };
    use serde_json::json;
    use std::sync::Arc;
    use std::time::Duration;

    fn middleware_with(
        tools: Vec<Arc<dyn crate::tool::ErrorAwareTool>>,
    ) -> ErrorHandlingMiddleware {
        ErrorHandlingMiddleware::new(ToolRegistry::new(tools)).with_base_delay(Duration::ZERO)
    }

    #[tokio::test]
    async fn successful_run_reports_no_errors() {
        let llm = Arc::new(ScriptedLlm::new(vec![
            step_calls(vec![ScriptedToolCall::new(
                "c1",
                FLAKY_FETCH_TOOL_NAME,
                json!({ "resource": "r" }),
            )]),
            step_text("最终答案"),
        ]));

        // flaky 工具第 1 次就成功（fail_times = 0）。
        let middleware = middleware_with(vec![Arc::new(FlakyTool::new(0))]);
        let agent = Agent::new(llm, middleware, AgentConfig::default());

        let run = agent.run("拉取一下").await.unwrap();

        assert_eq!(run.answer, "最终答案");
        assert_eq!(run.tool_call_groups, 1);
        assert_eq!(run.retried_calls, 0);
        assert_eq!(run.degraded_errors, 0);
        assert!(!run.fatal_terminated);
    }

    #[tokio::test]
    async fn transient_failure_is_retried_and_recorded() {
        // flaky 工具前 2 次失败 —— 中间件会重试到成功。
        let tool = Arc::new(FlakyTool::new(2));
        let llm = Arc::new(ScriptedLlm::new(vec![
            step_calls(vec![ScriptedToolCall::new(
                "c1",
                FLAKY_FETCH_TOOL_NAME,
                json!({ "resource": "r" }),
            )]),
            step_text("最终答案"),
        ]));

        let middleware = middleware_with(vec![tool.clone()]);
        let agent = Agent::new(llm, middleware, AgentConfig::default());

        let run = agent.run("拉取一下").await.unwrap();

        assert_eq!(run.answer, "最终答案");
        assert_eq!(run.retried_calls, 1);
        assert_eq!(run.degraded_errors, 0);
        // 工具总共被调了 3 次（1 次失败 + 2 次重试）。
        assert_eq!(tool.call_count(), 3);
        // 记录里要能看到重试痕迹。
        assert!(run.steps[0].tool_calls[0].retried);
        assert_eq!(run.steps[0].tool_calls[0].attempts, 3);
    }

    #[tokio::test]
    async fn business_failure_is_degraded_not_retried() {
        let tool = Arc::new(StrictTool::new(vec!["rec-1".to_string()]));
        let llm = Arc::new(ScriptedLlm::new(vec![
            step_calls(vec![ScriptedToolCall::new(
                "c1",
                STRICT_LOOKUP_TOOL_NAME,
                json!({ "id": "nope" }),
            )]),
            step_text("我查不到这条记录"),
        ]));

        let middleware = middleware_with(vec![tool]);
        let agent = Agent::new(llm, middleware, AgentConfig::default());

        let run = agent.run("查一下 nope").await.unwrap();

        // ⚠️ 没有重试。
        assert_eq!(run.retried_calls, 0);
        assert_eq!(run.degraded_errors, 1);
        // 观察文本要带错误码。
        assert!(
            run.steps[0].tool_calls[0]
                .observation
                .contains("E_NOT_FOUND")
        );
        assert_eq!(
            run.steps[0].tool_calls[0].error_code.as_deref(),
            Some(ErrorCode::NotFound.id())
        );
    }

    #[tokio::test]
    async fn fatal_failure_terminates_the_loop() {
        let llm = Arc::new(ScriptedLlm::new(vec![
            step_calls(vec![ScriptedToolCall::new(
                "c1",
                GUARDED_PAY_TOOL_NAME,
                json!({ "amount": 10 }),
            )]),
            // ⚠️ 这一步**不应该**被走到 —— 致命错误必须终止。
            step_text("不该出现的答案"),
        ]));

        let middleware = middleware_with(vec![Arc::new(GuardedTool::new(None))]);
        let agent = Agent::new(llm, middleware, AgentConfig::default());

        let err = agent.run("付 10 块").await.unwrap_err();

        // ⚠️ 鉴权失败 → 500（服务端自己的问题）。
        assert_eq!(err.status_code(), 500);
        assert!(err.to_string().contains("E_UNAUTHORIZED"));
    }

    #[tokio::test]
    async fn model_failure_is_not_degraded() {
        // ⚠️ 模型调用失败 → 502，不能降级。
        let llm = Arc::new(ScriptedLlm::new(vec![step_failure("连接超时")]));
        let middleware = middleware_with(vec![Arc::new(FlakyTool::new(0))]);
        let agent = Agent::new(llm, middleware, AgentConfig::default());

        let err = agent.run("你好").await.unwrap_err();
        assert_eq!(err.status_code(), 502);
    }

    #[tokio::test]
    async fn empty_question_is_rejected() {
        let llm = Arc::new(ScriptedLlm::new(vec![LlmStep::Response(Box::new(
            crate::llm::ScriptedStep::Text("x".to_string()),
        ))]));
        let middleware = middleware_with(vec![]);
        let agent = Agent::new(llm, middleware, AgentConfig::default());

        let err = agent.run("   ").await.unwrap_err();
        assert_eq!(err.status_code(), 400);
    }

    #[tokio::test]
    async fn hitting_max_steps_degrades_gracefully() {
        // 每次都调工具 → 一直走不到最终答案。
        let steps: Vec<LlmStep> = (0..5)
            .map(|_| {
                step_calls(vec![ScriptedToolCall::new(
                    "c1",
                    FLAKY_FETCH_TOOL_NAME,
                    json!({ "resource": "r" }),
                )])
            })
            .collect();

        let llm = Arc::new(ScriptedLlm::new(steps));
        let middleware = middleware_with(vec![Arc::new(FlakyTool::new(0))]);
        let agent = Agent::new(llm, middleware, AgentConfig::default());

        let run = agent.run("一直调").await.unwrap();

        assert!(run.hit_max_steps);
        assert_eq!(run.steps.len(), 5);
    }

    #[test]
    fn normalizes_string_arguments() {
        let args = json!("{\"resource\":\"r\"}");
        assert_eq!(normalize_arguments(&args)["resource"], json!("r"));
    }
}
