//! Agent 循环 —— **并发执行工具调用**。
//!
//! # 与 16 / 18 的 Agent 循环的差别
//!
//! 16/18 的循环是**串行**的：一轮里模型返回 N 个工具调用，
//! 就一个接一个地执行。
//!
//! 本项目把这一步改成**并发**：
//! 一轮里的 N 个工具调用**同时**跑。
//!
//! ```text
//! 16/18（串行）：            本项目（并发）：
//!   call1 ──► 结果1          ┌─ call1 ─┐
//!   call2 ──► 结果2          ├─ call2 ─┼─► 一起等
//!   call3 ──► 结果3          └─ call3 ─┘
//!   总耗时 = 1+2+3            总耗时 ≈ max(1,2,3)
//! ```
//!
//! # ⚠️ 并发执行时最容易踩的三个坑
//!
//! ## 坑 1：回执顺序必须与 `tool_calls` 顺序对应
//!
//! 16 实测：每个 `call_id` 必须有**恰好一条** `tool` 消息，
//! 少一条整个请求就 400。
//!
//! 并发执行后结果到达顺序是乱的 —— 但**回执必须按 `call_id` 对应**，
//! 不能按「谁先完成」排。所以本项目用 `index` 还原输入顺序。
//!
//! ## 坑 2：阻塞工具必须走线程池
//!
//! 一个**真阻塞**的工具（同步数据库驱动）如果直接放进 async 里
//! 会卡住整个事件循环。所以并发执行时，标记为 `is_blocking` 的工具
//! 要用 `tokio::task::spawn_blocking` 包起来。
//!
//! ## 坑 3：并发度不是越高越好
//!
//! 全部并发会打爆上游（429）、耗尽 fd、撑爆内存。
//! LangChain 的 `max_concurrency` 就是为此而生 —— 本项目用
//! [`crate::concurrency::ExecutionMode::Bounded`] 实现它。
//!
//! ## 坑 4：并发层会把语义字段丢掉（**自测时发现**）
//!
//! ⚠️ 本项目最初让 `run_one` 返回 `Result<String, ToolError>` ——
//! **只带观察文本**，把中间件的 `outcome.attempts`（重试次数）
//! 和错误的 `kind`（Fatal / Business / Transient）**丢在了并发层**。
//!
//! 后果：
//! - **致命错误被降级成观察** —— 模型以为「工具坏了」→ 转而编答案，
//!   把配置问题掩盖成幻觉（18 明确警告过）。
//! - **`retried_calls` 恒为 0** —— `attempts` 被写死成 1。
//!
//! 修法：并发层的任务输出换成 [`crate::concurrency::TaskResult`]，
//! 带 `observation` + `attempts` + `error`（**原始错误对象**）。

use std::sync::Arc;
use std::time::Instant;

use agent_16::agent::build_assistant_message;
use agent_16::llm::{FunctionLlm, ToolChoice};
use agent_16::tools::ToolSpec;
use agent_18::middleware::{ErrorHandlingMiddleware, MiddlewareOutcome, ToolCallRequest};
use agent_logging::{debug, info, warn};
use genai::chat::{ChatMessage, MessageContent, ToolCall, ToolResponse};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::concurrency::{ExecutionMode, Task, TaskFactory, TaskResult, compare_all, run};
use crate::error::{AppError, AppResult, ToolError};
use crate::prompt;
use crate::tools::AsyncTool;

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

    /// 实际尝试了几次（含重试）。
    pub attempts: usize,

    /// 这个工具是不是**阻塞**的（走了线程池）。
    pub blocking: bool,

    /// 错误码（失败时）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,

    /// 错误类别（失败时）：`transient` / `business` / `fatal`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,

    /// 是否被重试过（`attempts > 1`）。
    pub retried: bool,

    /// ⚠️ 致命错误的**原始对象**（Fatal 时才有）。
    ///
    /// ## 为什么单独存
    ///
    /// `error_code` / `error_kind` 是给人看的**投影**，
    /// 但终止 Agent 循环时需要构造 [`AppError`] ——
    /// 那个构造需要**原始错误对象**（含 `code` / `kind` / `hint`）。
    ///
    /// 用 `unauthorized` 重新造一个会丢掉真实的错误码。
    /// 所以这里保留原对象，但**不序列化**（避免 JSON 里出现重复信息）。
    #[serde(skip)]
    pub fatal_error: Option<ToolError>,
}

/// 一步的执行记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// 第几步（从 1 开始）。
    pub step: usize,

    /// 模型输出的文本。
    pub text: String,

    /// 本步的工具调用。
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

    /// 工具调用总次数。
    pub tool_calls: usize,

    /// 逻辑上的工具调用数（同一轮内的并行调用算 1 组）。
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
    ///
    /// ⚠️ Fatal 错误**不降级**（18 的结论）——
    /// 降级会让模型以为「工具坏了」转而编答案。
    pub fatal_terminated: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,
}

/// Agent 配置。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 最大步数。
    pub max_steps: usize,

    /// 工具选择策略。
    pub tool_choice: ToolChoice,

    /// ⚠️ 并发上限。
    pub max_concurrency: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 5,
            tool_choice: ToolChoice::Auto,
            max_concurrency: 4,
        }
    }
}

/// 带并发工具执行的 Agent。
pub struct Agent {
    llm: Arc<dyn FunctionLlm>,
    /// ⚠️ 用 `Arc` 包住 —— 并发任务需要 `'static` 的所有权。
    middleware: Arc<ErrorHandlingMiddleware>,
    config: AgentConfig,
    /// 哪些工具是**阻塞**的（走线程池轨道）。
    blocking_tools: Arc<Vec<String>>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("tools", &self.middleware.registry().names())
            .field("max_steps", &self.config.max_steps)
            .field("max_concurrency", &self.config.max_concurrency)
            .finish()
    }
}

impl Agent {
    /// 创建 Agent。
    pub fn new(
        llm: Arc<dyn FunctionLlm>,
        middleware: ErrorHandlingMiddleware,
        config: AgentConfig,
        blocking_tools: Vec<String>,
    ) -> Self {
        Self {
            llm,
            middleware: Arc::new(middleware),
            config,
            blocking_tools: Arc::new(blocking_tools),
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

    /// 这个工具是不是阻塞的。
    pub fn is_blocking(&self, name: &str) -> bool {
        self.blocking_tools.iter().any(|n| n == name)
    }

    /// 跑一次 Agent。
    pub async fn run(self: &Arc<Self>, question: &str) -> AppResult<AgentRun> {
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

            // ── 有工具调用 → **并发**执行 ───────────────────────
            // ⚠️ 复用 16 的 `build_assistant_message`（带 reasoning_content）——
            //    本项目原先自己写了一份**不带**思考内容的版本，
            //    思考模式一开就会被 provider 拒（400）。
            messages.push(build_assistant_message(
                text.clone(),
                reasoning.clone(),
                &calls,
            ));

            let records = self.execute_calls_concurrently(&calls).await;

            tool_call_groups += 1;
            for record in &records {
                tool_calls += record.attempts;
                if record.attempts > 1 {
                    retried_calls += 1;
                }
                if !record.ok {
                    degraded_errors += 1;
                }
                if !tools_used.contains(&record.name) {
                    tools_used.push(record.name.clone());
                }
            }

            // ⚠️ 不变量：**每个** call_id 恰好一条回执。
            //
            // 回执顺序必须与 `calls` 的**输入顺序**一致 ——
            // 不能按「谁先完成」排（并发下完成顺序是乱的）。
            let responses: Vec<ToolResponse> = calls
                .iter()
                .zip(records.iter())
                .map(|(call, record)| {
                    ToolResponse::from_tool_call(call, record.observation.clone())
                })
                .collect();

            messages.push(ChatMessage::tool(MessageContent::from_tool_responses(
                responses,
            )));

            // ⚠️ 不变量 2：致命错误 → **终止**，**不返回模型的答案**。
            //
            // 18 的结论：Fatal 不能降级 —— 降级会让模型以为「工具坏了」，
            // 转而编一个答案，把配置问题（鉴权失败等）掩盖成幻觉。
            //
            // ⚠️ 终止**发生在回执生成之后** —— 16 的坑 1：
            // 少一条回执整个请求就 400，包括致命错误这一条。
            //
            // ⚠️ 判定放在 `steps.push` **之前** ——
            //    致命错误时直接返回错误，不该把这一步记进 steps。
            if let Some(fatal) = records.iter().find(|r| r.fatal_error.is_some()) {
                let err = fatal
                    .fatal_error
                    .clone()
                    .unwrap_or_else(|| ToolError::internal(fatal.name.clone(), "致命错误"));

                warn!(
                    step = step_no,
                    tool = %fatal.name,
                    code = err.diagnostic_code(),
                    "致命错误 —— 终止 Agent 循环"
                );

                return Err(AppError::from_tool_error(&err));
            }

            steps.push(StepRecord {
                step: step_no,
                text,
                tool_calls: records,
                final_answer: false,
            });
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

    /// ⚠️ **并发**执行一轮里的所有工具调用。
    ///
    /// ## 返回顺序
    ///
    /// 返回的 `Vec` 与输入 `calls` **一一对应**（同下标）——
    /// 调用方靠这个对应关系生成回执。
    async fn execute_calls_concurrently(
        self: &Arc<Self>,
        calls: &[ToolCall],
    ) -> Vec<ToolCallRecord> {
        let tasks = self.build_tasks(calls);

        let report = run(ExecutionMode::Bounded(self.config.max_concurrency), tasks).await;

        debug!(
            calls = calls.len(),
            mode = report.mode.name(),
            total_ms = report.total_ms(),
            "并发执行工具调用完成"
        );

        // ⚠️ 按 `index` 还原输入顺序 —— 并发下完成顺序是乱的。
        let mut ordered: Vec<Option<ToolCallRecord>> = vec![None; calls.len()];
        for output in &report.results {
            let call = calls.get(output.index);
            let error = output.result.error.clone();
            let record = ToolCallRecord {
                call_id: call.map(|c| c.call_id.clone()).unwrap_or_default(),
                name: output.name.clone(),
                arguments: call.map(|c| c.fn_arguments.clone()).unwrap_or(Value::Null),
                observation: output.result.observation.clone(),
                ok: output.result.is_ok(),
                // ⚠️ **真实**尝试次数 —— 中间件重试了几次就是几，
                //    不能写死 1（否则 `retried_calls` 恒为 0）。
                attempts: output.result.attempts,
                blocking: self.is_blocking(&output.name),
                error_code: error.as_ref().map(|e| e.diagnostic_code().to_string()),
                error_kind: error.as_ref().map(|e| e.kind.name().to_string()),
                retried: output.result.attempts > 1,
                // ⚠️ 致命错误的**原始对象**必须保留 ——
                //    Agent 循环靠它决定「终止」还是「降级」。
                fatal_error: if output.result.is_fatal() {
                    error
                } else {
                    None
                },
            };
            ordered[output.index] = Some(record);
        }

        // ⚠️ 理论上不会有 `None`（每个任务都一定会产生结果），
        //    但这里兜底成「内部错误」而不是 panic ——
        //    Agent 循环绝不能因为一个工具出问题就整体崩掉。
        ordered
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                opt.unwrap_or_else(|| ToolCallRecord {
                    call_id: calls.get(i).map(|c| c.call_id.clone()).unwrap_or_default(),
                    name: "unknown".to_string(),
                    arguments: Value::Null,
                    observation: "[E_INTERNAL_FAULT] 内部错误：并发执行的结果缺失。".to_string(),
                    ok: false,
                    attempts: 1,
                    blocking: false,
                    error_code: Some("E_INTERNAL_FAULT".to_string()),
                    error_kind: Some("fatal".to_string()),
                    retried: false,
                    fatal_error: Some(ToolError::internal("unknown", "并发执行的结果缺失")),
                })
            })
            .collect()
    }

    /// 构造并发任务。
    ///
    /// ## ⚠️ 为什么 `self` 必须是 `Arc<Self>`
    ///
    /// `Task` 里的 future 要求 `'static`（并发原语要能自由地持有它），
    /// 而 `self.run_one(...)` 借用了 `self` —— 所以只能克隆 `Arc`。
    fn build_tasks(self: &Arc<Self>, calls: &[ToolCall]) -> Vec<Task> {
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                let request = ToolCallRequest {
                    name: call.fn_name.clone(),
                    arguments: agent_16::agent::normalize_arguments(&call.fn_arguments),
                    call_id: call.call_id.clone(),
                };

                let agent = self.clone();
                let fut = async move { agent.run_one(request).await };
                Task::new(call.fn_name.clone(), index, fut)
            })
            .collect()
    }

    /// 执行一次工具调用（走中间件）。
    ///
    /// ## ⚠️ 错误必须**向上传播**
    ///
    /// 中间件把错误**降级**成观察文本，但**分类信息不能丢** ——
    /// 否则上层看到的全是「成功」，无法判断「该不该终止」。
    ///
    /// 所以这里返回 [`TaskResult`]（带 `error` 对象 + `attempts`）：
    /// - 上层把它渲染成观察文本喂回模型（18 的语义）
    /// - 同时用它判断「是不是致命错误」（Fatal → 终止循环）
    /// - 用它统计「实际尝试了几次」（重试次数）
    ///
    /// ## ⚠️ 为什么中间件要用 `Arc` 包住
    ///
    /// 并发任务要求 future 是 `'static`，
    /// 而 `self.middleware` 是借用 —— 所以必须 `Arc` + `clone`。
    ///
    /// ⚠️ 中间件本身是**只读**的（工具注册表是 `Vec<Arc<dyn Tool>>`），
    /// 所以多个并发任务共享一个中间件实例是**安全**的。
    async fn run_one(self: Arc<Self>, request: ToolCallRequest) -> TaskResult {
        let outcome: MiddlewareOutcome = self.middleware.wrap_tool_call(&request).await;

        if !outcome.ok {
            warn!(
                tool = %request.name,
                code = outcome.error.as_ref().map(|e| e.diagnostic_code()),
                attempts = outcome.attempts,
                "工具调用失败（错误已作为 tool 消息喂回模型）"
            );
        }

        match outcome.error {
            Some(err) => TaskResult::failed(err.to_string(), outcome.attempts, err),
            None => TaskResult::ok(outcome.observation, outcome.attempts),
        }
    }
}

/// 跑一批「并发对照实验」—— 串行 / 全并发 / 乱序 / 限流 四档。
///
/// ## ⚠️ 每种模式都用**同一个工厂**重建任务
///
/// 这样「变量单一」—— 四组的任务内容、顺序完全一致，
/// 只有执行方式不同，差值才能归因到并发模式。
pub async fn bench(factory: TaskFactory, limit: usize) -> Vec<crate::concurrency::BatchReport> {
    compare_all(factory, limit).await
}

/// 让 `AsyncTool` 的导入不被判定为未使用（它在类型签名里出现）。
#[allow(dead_code)]
fn _tool_marker(tool: &dyn AsyncTool) -> &dyn AsyncTool {
    tool
}
