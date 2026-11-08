//! Agent 循环 —— 复用 13 的 ReAct 引擎，接上本项目的工具基础设施。
//!
//! ## 和 13-agent-core 的关系
//!
//! **输出解析器直接复用** [`agent_13::agent::parse_react_output`] ——
//! 包括那个「模型回显整个提示词」的坑的修复：
//! 必须找**最后一个**标记，而不是第一个。
//!
//! 为什么复用而不是重写？因为解析器是整个 Agent 系统里**最容易出错**的部分，
//! 重写一遍就等于把踩过的坑再踩一次。
//!
//! ## 本项目加的三件事
//!
//! | 特性 | 说明 |
//! |---|---|
//! | **异步工具** | 工具可以等 I/O（天气 API / MCP 服务端） |
//! | **工具注册表** | 参数校验 + 失败转观察，全部在注册表里统一处理 |
//! | **`return_direct`** | 工具输出**直接**作为最终答案，跳过最后一次模型调用 |
//!
//! ## `return_direct` 的价值
//!
//! LangChain 教程里的 `fetch_order_status` 例子：
//!
//! > The agent returns the tool output directly without another LLM call.
//!
//! 这能**省一次模型调用**（省钱 + 降低延迟），
//! 而且保证答案**不被模型改写** —— 温度为 0 时，改写也未必是好事。
//!
//! ⚠️ 但只在「工具输出本身就是完整答案」时才能用。
//! 天气工具的输出需要模型组织成自然语言，所以默认**不开**。

use agent_logging::{debug, info, warn};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AppError, AppResult};
use crate::llm::Llm;
use crate::prompt;
use crate::tool::{ToolContext, ToolOutcome, ToolRegistry, ToolSpec};

/// 模型一轮输出的解析结果（复用 13 的定义）。
pub use agent_13::agent::ParsedOutput;

/// 解析模型输出（复用 13 的解析器）。
///
/// ## 为什么直接复用
///
/// 解析器是 Agent 系统里**最容易出错**的部分，
/// 而且 13 已经踩过「模型回显整个提示词」的坑并修好了
/// （必须找**最后一个**标记，而不是第一个）。
/// 重写一遍等于把坑再踩一次。
pub use agent_13::agent::parse_react_output;

/// 一步的执行记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// 第几步（从 1 开始）。
    pub step: usize,

    /// 模型的原始输出。
    pub raw_output: String,

    /// 思考内容。
    pub thought: String,

    /// 工具名（最终答案那一步为空）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,

    /// 工具参数。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,

    /// 观察文本（工具结果）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation: Option<String>,

    /// 工具返回的结构化数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,

    /// 工具是否执行成功。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_ok: Option<bool>,

    /// 参数解析错误（如果有）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parse_error: Option<String>,

    /// 这一步是不是「直通」结束（工具输出直接作为最终答案）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direct: Option<bool>,
}

impl StepRecord {
    /// 这一步是不是最终答案。
    pub fn is_final(&self) -> bool {
        self.tool.is_none()
    }
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

    /// 工具调用次数。
    pub tool_calls: usize,

    /// 用到的工具名（去重，按首次出现顺序）。
    pub tools_used: Vec<String>,

    /// 是否因为达到步数上限而中止。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,
}

impl AgentRun {
    /// 完整的 ReAct 轨迹。
    pub fn trajectory(&self) -> String {
        let mut out = String::new();
        for step in &self.steps {
            if !step.thought.is_empty() {
                out.push_str(&format!("Thought: {}\n", step.thought));
            }
            if let Some(tool) = &step.tool {
                out.push_str(&format!("Action: {tool}\n"));
                if let Some(args) = &step.args {
                    out.push_str(&format!(
                        "Action Input: {}\n",
                        serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
                    ));
                }
                if let Some(obs) = &step.observation {
                    out.push_str(&format!("Observation: {obs}\n"));
                }
            }
        }
        out.push_str(&format!("Final Answer: {}\n", self.answer));
        out
    }
}

/// 自主 Agent。
pub struct Agent {
    llm: Arc<dyn Llm>,
    registry: ToolRegistry,
    config: crate::config::AppConfig,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("tools", &self.registry.names())
            .finish()
    }
}

impl Agent {
    /// 创建 Agent。
    pub fn new(
        llm: Arc<dyn Llm>,
        registry: ToolRegistry,
        config: crate::config::AppConfig,
    ) -> Self {
        Self {
            llm,
            registry,
            config,
        }
    }

    /// 当前配置。
    pub fn config(&self) -> &crate::config::AppConfig {
        &self.config
    }

    /// 工具注册表。
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// 工具定义列表。
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.registry.specs()
    }

    /// 运行 Agent 处理一个问题（用配置里的默认模式）。
    pub async fn run(&self, question: &str) -> AppResult<AgentRun> {
        self.run_with_mode(question, self.config.agent.mode).await
    }

    /// 运行 Agent 处理一个问题（**显式指定模式**）。
    ///
    /// ## 为什么模式要能按次覆盖
    ///
    /// 因为消融对照端点要在**同一个 Agent 实例**上跑三种模式 ——
    /// 工具集、LLM、配置全都一样，**只有模式变量不同**。
    pub async fn run_with_mode(
        &self,
        question: &str,
        mode: crate::config::ToolMode,
    ) -> AppResult<AgentRun> {
        let start = Instant::now();
        let question = question.trim();

        if question.is_empty() {
            return Err(AppError::BadRequest("问题不能为空".to_string()));
        }

        let mut trajectory = String::new();
        let mut steps: Vec<StepRecord> = Vec::new();
        let mut tool_calls = 0usize;
        let mut tools_used: Vec<String> = Vec::new();

        // `Direct` 模式不挂工具 —— 提示词里也不会出现工具说明。
        let active_tools = if mode.uses_tools() {
            self.registry.clone()
        } else {
            ToolRegistry::new(Vec::new())
        };

        let specs = active_tools.specs();
        let ctx = ToolContext::empty();
        let system = prompt::system_prompt_for(mode);
        let few_shot = self.config.prompt.few_shot;

        // ── 主循环 ──────────────────────────────────────────────
        for step_no in 1..=self.config.agent.max_steps {
            debug!(
                mode = mode.name(),
                step = step_no,
                tool_count = specs.len(),
                "第 {step_no} 步"
            );

            let user = prompt::build_prompt(mode, &specs, question, &trajectory, few_shot);

            // 调试：打印真实发给模型的提示词。
            //
            // ⚠️ 排查「模型为什么做出奇怪的动作」时，
            //    看「模型到底看到了什么」比看「模型说了什么」更有效 ——
            //    因为前者是我们能控制的，后者只是结果。
            if std::env::var_os("TOOL_SHOW_PROMPT").is_some() {
                eprintln!("\n========== 第 {step_no} 步的完整提示词 ==========");
                eprintln!("{user}");
                eprintln!("========== 提示词结束 ==========\n");
            }

            let raw_output = self.llm.chat(&system, &user).await?;
            let parsed = agent_13::agent::parse_react_output(&raw_output);

            match parsed {
                ParsedOutput::FinalAnswer { text } => {
                    steps.push(StepRecord {
                        step: step_no,
                        raw_output: raw_output.clone(),
                        thought: String::new(),
                        tool: None,
                        args: None,
                        observation: None,
                        data: None,
                        tool_ok: None,
                        parse_error: None,
                        direct: None,
                    });

                    info!(
                        mode = mode.name(),
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
                    });
                }

                ParsedOutput::Unstructured { text } => {
                    // 容错：把这段文本当成最终答案。
                    steps.push(StepRecord {
                        step: step_no,
                        raw_output: raw_output.clone(),
                        thought: String::new(),
                        tool: None,
                        args: None,
                        observation: None,
                        data: None,
                        tool_ok: None,
                        parse_error: Some("模型输出既没有 Final Answer 也没有 Action".to_string()),
                        direct: None,
                    });

                    warn!(
                        mode = mode.name(),
                        step = step_no,
                        "模型输出无法解析 —— 按最终答案处理"
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
                    });
                }

                ParsedOutput::Action {
                    thought,
                    tool,
                    args,
                    parse_error,
                } => {
                    // ── 执行工具 ────────────────────────────────────
                    //
                    // 失败**不中断** —— 注册表把错误变成了观察。
                    let outcome = active_tools.call(&tool, &args, &ctx).await;

                    debug!(
                        mode = mode.name(),
                        step = step_no,
                        tool = %tool,
                        ok = outcome.ok,
                        "工具已执行"
                    );

                    if !outcome.ok {
                        warn!(
                            mode = mode.name(),
                            step = step_no,
                            tool = %tool,
                            observation = %outcome.observation,
                            "工具执行失败（错误已作为观察喂回模型）"
                        );
                    }

                    if !tools_used.contains(&tool) {
                        tools_used.push(tool.clone());
                    }

                    // 直通工具：工具输出**直接**作为最终答案。
                    //
                    // ⚠️ 只有当模型确实调了这个工具、且它是 `return_direct` 时才直通。
                    //    工具失败时不直通 —— 错误不该直接甩给用户。
                    let is_direct = outcome.ok
                        && active_tools
                            .get(&tool)
                            .map(|t| t.spec().return_direct)
                            .unwrap_or(false);

                    steps.push(StepRecord {
                        step: step_no,
                        raw_output: raw_output.clone(),
                        thought: thought.clone(),
                        tool: Some(tool.clone()),
                        args: Some(args.clone()),
                        observation: Some(outcome.observation.clone()),
                        data: outcome.data.clone(),
                        tool_ok: Some(outcome.ok),
                        parse_error: parse_error.clone(),
                        direct: if is_direct { Some(true) } else { None },
                    });
                    tool_calls += 1;

                    if is_direct {
                        info!(
                            mode = mode.name(),
                            step = step_no,
                            tool = %tool,
                            "直通工具 —— 输出直接作为最终答案"
                        );
                        return Ok(AgentRun {
                            question: question.to_string(),
                            answer: outcome.observation.clone(),
                            steps,
                            tool_calls,
                            tools_used,
                            hit_max_steps: false,
                            elapsed_ms: start.elapsed().as_millis() as u64,
                            model: self.config.model.clone(),
                        });
                    }

                    // ── 追加到轨迹 ────────────────────────────────
                    if !thought.is_empty() {
                        trajectory.push_str(&format!("Thought: {thought}\n"));
                    }
                    trajectory.push_str(&format!("Action: {tool}\n"));
                    trajectory.push_str(&format!(
                        "Action Input: {}\n",
                        serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string())
                    ));
                    trajectory.push_str(&format!("Observation: {}\n", outcome.observation));
                }
            }
        }

        // ── 步数用尽 ────────────────────────────────────────────
        //
        // 模型还想继续，但被 `max_steps` 掐断了。
        // 这里**不**直接报错 —— 再问一次，让它基于已有观察作答。
        warn!(
            mode = mode.name(),
            max_steps = self.config.agent.max_steps,
            tool_calls,
            "达到最大步数 —— 发起收尾调用"
        );
        let user = prompt::build_prompt(mode, &specs, question, &trajectory, few_shot);
        let final_prompt = format!(
            "{user}\n\n你已经用完了所有步骤。请基于目前已经获得的观察，直接以 `Final Answer:` 开头给出最终答案。"
        );

        // ⚠️ 收尾那次调用失败**也不应该**让整个运行失败 ——
        // 前面几步辛苦收集的观察是有价值的。
        let (answer, raw_output) = match self.llm.chat(&system, &final_prompt).await {
            Ok(raw) => {
                let answer = match agent_13::agent::parse_react_output(&raw) {
                    ParsedOutput::FinalAnswer { text } => text,
                    ParsedOutput::Unstructured { text } => text,
                    ParsedOutput::Action { .. } => {
                        "已达到最大步数，模型仍在请求调用工具。".to_string()
                    }
                };
                (answer, raw)
            }
            Err(e) => {
                warn!(error = %e, "收尾调用失败 —— 降级为「已收集到的观察摘要」");
                let answer = format!(
                    "已达到最大步数（{} 步），且收尾请求失败：{e}。\
                     Agent 已经收集到的观察如上，可据此判断。",
                    self.config.agent.max_steps
                );
                (answer, String::new())
            }
        };

        steps.push(StepRecord {
            step: self.config.agent.max_steps + 1,
            raw_output,
            thought: String::new(),
            tool: None,
            args: None,
            observation: None,
            data: None,
            tool_ok: None,
            parse_error: None,
            direct: None,
        });

        info!(
            mode = mode.name(),
            hit_max_steps = true,
            tool_calls,
            tools_used = ?tools_used,
            step_count = steps.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Agent 运行结束"
        );

        Ok(AgentRun {
            question: question.to_string(),
            answer,
            steps,
            tool_calls,
            tools_used,
            hit_max_steps: true,
            elapsed_ms: start.elapsed().as_millis() as u64,
            model: self.config.model.clone(),
        })
    }
}

/// 直接调用工具（不经 Agent 循环）。
///
/// ## 为什么需要这个入口
///
/// 它对应 LangChain 教程里的「工具可以被独立调用」——
/// 用户直接指定工具名和参数，绕过模型。
///
/// 这在实践中很有用：
///
/// - **调试**：直接看工具返回什么，不用等模型
/// - **确定性**：同样的参数永远同样的结果（模型会飘，工具不会）
/// - **成本**：不调模型
pub async fn call_tool_directly(
    registry: &ToolRegistry,
    name: &str,
    args: &Value,
    ctx: &ToolContext,
) -> Result<ToolOutcome, crate::error::ToolError> {
    registry.call_strict(name, args, ctx).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ScriptedLlm;
    use serde_json::json;

    /// 确定性的假工具：回显。
    struct EchoTool;

    impl crate::tool::Tool for EchoTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(
                "echo",
                "回显文本",
                json!({
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"]
                }),
            )
        }

        fn call<'a>(
            &'a self,
            args: &'a Value,
            _ctx: &'a ToolContext,
        ) -> crate::tool::ToolFuture<'a> {
            let text = args
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Box::pin(async move { Ok(ToolOutcome::ok("echo", format!("回显: {text}"))) })
        }
    }

    /// 声明了 `return_direct` 的假工具 —— 工具输出**直接**作为最终答案。
    struct DirectTool;

    impl crate::tool::Tool for DirectTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(
                "direct_echo",
                "回显文本并直接结束",
                json!({
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"]
                }),
            )
            .with_return_direct()
        }

        fn call<'a>(
            &'a self,
            args: &'a Value,
            _ctx: &'a ToolContext,
        ) -> crate::tool::ToolFuture<'a> {
            let text = args
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Box::pin(async move { Ok(ToolOutcome::ok("direct_echo", format!("直接答案: {text}"))) })
        }
    }

    fn agent_with(answers: Vec<String>, max_steps: usize) -> (Agent, Arc<ScriptedLlm>) {
        let llm = Arc::new(ScriptedLlm::new(answers));
        let mut config = crate::config::AppConfig::default();
        config.agent.max_steps = max_steps;

        let registry = ToolRegistry::new(vec![Arc::new(EchoTool)]);
        (Agent::new(llm.clone(), registry, config), llm)
    }

    #[tokio::test]
    async fn calls_tool_then_answers() {
        let (agent, _llm) = agent_with(
            vec![
                "Thought: 查\nAction: echo\nAction Input: {\"text\": \"hi\"}".to_string(),
                "Final Answer: 好了".to_string(),
            ],
            5,
        );

        let run = agent.run("问题").await.unwrap();

        assert_eq!(run.tool_calls, 1);
        assert_eq!(run.steps.len(), 2);
        assert!(run.answer.contains("好了"));
        assert_eq!(run.tools_used, vec!["echo".to_string()]);
    }

    #[tokio::test]
    async fn tool_failure_is_fed_back_not_fatal() {
        let (agent, llm) = agent_with(
            vec![
                "Thought: 查\nAction: echo\nAction Input: {}".to_string(),
                "Final Answer: 我看到报错了".to_string(),
            ],
            5,
        );

        let run = agent.run("问题").await.unwrap();

        assert_eq!(run.steps[0].tool_ok, Some(false));
        assert!(run.steps[0].observation.as_ref().unwrap().contains("text"));

        // 错误文本必须进了下一轮的提示词。
        let calls = llm.calls();
        assert!(calls[1].1.contains("text"));
    }

    #[tokio::test]
    async fn return_direct_skips_the_final_model_call() {
        // ⚠️ 直通工具：工具输出**直接**作为最终答案，
        //    不再问模型「怎么组织语言」。
        //
        // 脚本里只有 1 个答案 —— 如果还去调第二次模型，
        // 就会因为「脚本化 LLM 没有更多预设答案」而失败。
        let llm = Arc::new(ScriptedLlm::new(vec![
            "Thought: 查\nAction: direct_echo\nAction Input: {\"text\": \"直接答案\"}".to_string(),
        ]));
        let mut config = crate::config::AppConfig::default();
        config.agent.max_steps = 5;

        let registry = ToolRegistry::new(vec![Arc::new(DirectTool)]);
        let agent = Agent::new(llm.clone(), registry, config);

        let run = agent.run("问题").await.unwrap();

        // 只调了 1 次模型 —— 直通生效了。
        assert_eq!(llm.call_count(), 1);
        assert_eq!(run.tool_calls, 1);
        assert!(run.answer.contains("直接答案"));
        assert!(!run.hit_max_steps);
        // 直通的那一步要标记出来。
        assert_eq!(run.steps[0].direct, Some(true));
    }

    #[tokio::test]
    async fn non_direct_tool_still_asks_the_model() {
        // 对照：`EchoTool` 没开 `return_direct` —— 所以要走收尾分支。
        //
        // ⚠️ `max_steps = 1` —— 第 1 步调完工具就耗尽了步数，
        //    于是进入「收尾再问一次」分支。
        //    脚本只有 1 个答案，所以收尾那次调用会失败 ——
        //    但**不能**让整个运行失败（前面收集的观察要保住）。
        let (agent, llm) = agent_with(
            vec!["Thought: 查\nAction: echo\nAction Input: {\"text\": \"hi\"}".to_string()],
            1,
        );

        let run = agent.run("问题").await.unwrap();

        assert!(run.hit_max_steps);
        assert_eq!(run.tool_calls, 1);
        // 1 步 + 1 次收尾。
        assert_eq!(llm.call_count(), 2);
        assert!(run.steps[0].direct.is_none());
        // 收尾失败要降级 —— 答案里说明发生了什么，观察不能丢。
        assert!(run.answer.contains("收尾请求失败"));
        assert_eq!(run.steps.len(), 2);
    }

    #[tokio::test]
    async fn empty_question_is_rejected() {
        let (agent, _llm) = agent_with(vec![], 5);
        let err = agent.run("   ").await.unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[tokio::test]
    async fn direct_tool_call_bypasses_the_model() {
        let (agent, _llm) = agent_with(vec![], 5);

        let outcome = call_tool_directly(
            agent.registry(),
            "echo",
            &json!({ "text": "直连" }),
            &ToolContext::empty(),
        )
        .await
        .unwrap();

        assert!(outcome.ok);
        assert!(outcome.observation.contains("直连"));
    }
}
