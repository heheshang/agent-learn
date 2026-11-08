//! ReAct 循环 —— 把「思考」和「行动」交织起来的执行引擎。
//!
//! ## 循环的全貌
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │ 1. 组装提示词（格式说明 + 工具说明 + few-shot + 问题 + 轨迹）│
//! ├────────────────────────────────────────────────────────────┤
//! │ 2. 调模型 → 拿到一段文本                                    │
//! ├────────────────────────────────────────────────────────────┤
//! │ 3. 解析：                                                   │
//! │    ├─ 有 `Final Answer:` → 收工                             │
//! │    ├─ 有 `Action:`       → 执行工具 → 追加 Observation       │
//! │    └─ 都没有             → 视为最终答案（容错）              │
//! ├────────────────────────────────────────────────────────────┤
//! │ 4. 回到 1，直到 max_steps                                   │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## 与「函数调用」式 Agent 的区别
//!
//! 03-genai 用的是 **OpenAI function calling**：
//! 模型返回结构化的 `tool_calls`，程序执行后回填。
//!
//! ReAct 用的是**文本协议**：模型返回 `Action: search` +
//! `Action Input: {...}`，程序**自己解析**。
//!
//! | | function calling | ReAct 文本协议 |
//! |---|---|---|
//! | 可靠性 | 高（provider 保证结构） | 低（要自己解析，可能格式错） |
//! | 通用性 | 受 provider 支持度限制 | 任何模型都能用 |
//! | 可解释性 | 低（看不到中间思考） | **高**（`Thought:` 全文可见） |
//!
//! 博客里那句「几乎所有 Agent 演示代码的一半工作量都花在解析模型输出上」
//! 说的就是这个代价 —— 本项目的 [`parse_react_output`] 就是那「一半工作量」。
//!
//! ## 为什么不用 function calling
//!
//! 因为**可解释性**是 ReAct 的全部意义所在。
//! 论文的核心主张是「推理轨迹让模型更可解释、更可信」——
//! 如果用 function calling，`Thought:` 就丢了。
//!
//! 而且文本协议让**对照实验**干净：function calling 的结构化输出
//! 会「暗示」模型该调哪个工具，而文本协议下模型完全自由。

use agent_logging::{debug, info, warn};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::{AgentConfig, AgentMode, AppConfig, PromptConfig};
use crate::error::{AgentError, AgentResult};
use crate::llm::Llm;
use crate::prompt;
use crate::tools::{Tool, ToolOutcome};

/// 模型一轮输出的解析结果。
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedOutput {
    /// 模型给出了最终答案。
    FinalAnswer {
        /// 答案正文（已去掉 `Final Answer:` 前缀）。
        text: String,
    },

    /// 模型要求调用某个工具。
    Action {
        /// 思考内容（`ActOnly` 模式下为空）。
        thought: String,

        /// 工具名。
        tool: String,

        /// 工具参数（解析失败时为空对象）。
        args: serde_json::Value,

        /// 参数解析是否失败（失败时 `args` 是空对象，
        /// 错误信息会被喂回模型让它纠正）。
        parse_error: Option<String>,
    },

    /// 模型既没给答案也没给动作 —— **容错**：当成最终答案。
    ///
    /// ## 为什么要有这个分支
    ///
    /// 真实模型经常「跑偏」：输出一段解释却不继续格式。
    /// 如果直接报错，用户看到的是 500 —— 而模型其实给了一个（可能对，
    /// 可能错的）答案。把这段文本当成答案返回，
    /// **至少服务是可用的**，而且轨迹里能看到模型跑偏了。
    Unstructured {
        /// 原始文本。
        text: String,
    },
}

impl ParsedOutput {
    /// 是不是最终答案。
    pub fn is_final(&self) -> bool {
        matches!(
            self,
            ParsedOutput::FinalAnswer { .. } | ParsedOutput::Unstructured { .. }
        )
    }
}

/// 解析模型输出。
///
/// ## 解析顺序很关键
///
/// 1. **先找 `Final Answer:`** —— 因为模型的思考里可能提到这几个字
///    （比如 few-shot 里的示例被复述），先判动作会误判。
/// 2. 再找 `Action:` —— 必须**同时**有 `Action Input:` 才算合法动作。
/// 3. 都没有 → [`ParsedOutput::Unstructured`]。
///
/// ## ⚠️ 必须找**最后一个**标记，而不是第一个
///
/// 这是本项目实测踩到的**真坑**。
///
/// 模型有时会**回显整个提示词**再继续 —— 因为提示词里带着 few-shot 示例
/// 和上一轮的轨迹，模型会顺手把「Question: 用户的问题」、
/// 格式说明、示例、轨迹全抄一遍，然后才输出自己的下一步。
///
/// 实测输出长这样：
///
/// ```text
/// Question: 用户的问题              ← 回显
/// Thought: 用户问的是...             ← 回显（示例里的）
/// Action: search                     ← 回显（上一轮的）
/// Action Input: {"query": "..."}     ← 回显（上一轮的）
/// Observation: [1] ...               ← 回显（上一轮的）
/// Thought: 第一条片段已经明确给出了... ← 模型自己的
/// Final Answer: 在 ReAct 框架中...     ← 模型自己的
/// ```
///
/// 如果用 `find`（第一个匹配），会匹配到**回显里的旧 `Action:`** ——
/// 于是程序把上一步的工具**再执行一遍**，然后把模型真正的答案
/// 当成 observation 丢掉。实测症状是答案变成不相关的内容。
///
/// 所以这里统一用 `rfind`（最后一个匹配）：
/// 模型自己的输出**永远在回显之后**，所以最后一个标记才是它真正想说的。
///
/// ## `Action Input` 的三种容错
///
/// 模型给出的参数可能是：
///
/// - 合法 JSON → 直接用
/// - **带 markdown 代码块**的 JSON（```` ```json {...} ``` ````）→ 剥掉围栏
/// - 单引号 / 无引号的伪 JSON → 尝试修复
/// - 完全不是 JSON → 当成字符串参数（很多工具只需要一个字符串参数）
///
/// 最后一条是**兜底**：`search` 的核心参数就是 `query` 字符串，
/// 模型写成 `Action Input: ReAct` 虽然不合规，但意图很清楚。
/// 把它当成 `query` 比报错有用得多。
pub fn parse_react_output(text: &str) -> ParsedOutput {
    let trimmed = text.trim();
    debug!(raw_len = trimmed.len(), "开始解析模型输出");

    // ── 1. 最终答案（找最后一个）────────────────────────────────
    if let Some(pos) = find_last_marker(trimmed, "Final Answer:") {
        let answer = trimmed[pos + "Final Answer:".len()..].trim();
        debug!("解析为最终答案");
        return ParsedOutput::FinalAnswer {
            text: answer.to_string(),
        };
    }

    // ── 2. 动作（找最后一个）────────────────────────────────────
    let thought = extract_thought(trimmed);

    if let Some(pos) = find_last_marker(trimmed, "Action:") {
        let rest = &trimmed[pos + "Action:".len()..];

        // 工具名 = `Action:` 后面的第一个非空 token（到行尾或空白为止）。
        let tool = rest
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
            .to_string();

        if !tool.is_empty() {
            // `Action Input:` 必须在 `Action:` **之后** ——
            // 否则模型把 Action Input 写在前面就找不到。
            //
            // ⚠️ 同样找**最后一个** —— 回显里可能有上一轮的 `Action Input:`。
            let input_raw = find_last_marker(rest, "Action Input:")
                .map(|p| rest[p + "Action Input:".len()..].trim().to_string())
                .or_else(|| {
                    // 容错：没有 `Action Input:` 标记时，
                    // 尝试把 `Action:` 那一行剩下的内容当参数。
                    rest.lines()
                        .nth(1)
                        .map(|l| l.trim().to_string())
                        .filter(|l| !l.is_empty())
                })
                .unwrap_or_default();

            let (args, parse_error) = parse_action_input(&input_raw);
            debug!(tool = %tool, parse_error = ?parse_error, "解析为工具调用");
            return ParsedOutput::Action {
                thought,
                tool,
                args,
                parse_error,
            };
        }
    }

    // ── 3. 兜底 ─────────────────────────────────────────────
    warn!("模型输出既没有 Final Answer 也没有 Action");
    ParsedOutput::Unstructured {
        text: trimmed.to_string(),
    }
}

/// 找出标记**最后一次**出现的位置（忽略大小写）。
///
/// ⚠️ 必须是最后一个 —— 理由见 [`parse_react_output`] 的文档。
fn find_last_marker(text: &str, marker: &str) -> Option<usize> {
    let lower = text.to_lowercase();
    lower.rfind(&marker.to_lowercase())
}

/// 抽取 `Thought:` 段落。
///
/// `ActOnly` 模式没有 `Thought:` —— 返回空串。
///
/// ## 为什么要找**最后一个**
///
/// 模型回显提示词时会把示例里的 `Thought:` 一起抄回来。
/// 只有**最后一个** `Thought:` 才是模型自己刚写的。
fn extract_thought(text: &str) -> String {
    // 先定位 `Action:` —— `Thought:` 段落在它之前。
    let action_pos = find_last_marker(text, "Action:").unwrap_or(text.len());

    let head = &text[..action_pos];

    let Some(pos) = find_last_marker(head, "Thought:") else {
        return String::new();
    };

    let rest = &head[pos + "Thought:".len()..];

    // `Thought:` 段落到下一个动作标记为止。
    let mut end = rest.len();
    for marker in ["Action:", "Final Answer:"] {
        if let Some(p) = find_last_marker(rest, marker) {
            end = end.min(p);
        }
    }

    rest[..end].trim().to_string()
}

/// 解析 `Action Input` 的内容。
///
/// 返回 `(参数, 错误信息)`。
pub fn parse_action_input(raw: &str) -> (serde_json::Value, Option<String>) {
    let cleaned = raw
        .trim()
        // 剥掉 markdown 代码围栏。
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    if cleaned.is_empty() {
        return (
            serde_json::Value::Object(serde_json::Map::new()),
            Some("Action Input 为空".to_string()),
        );
    }

    // 正常路径：合法 JSON。
    match serde_json::from_str::<serde_json::Value>(cleaned) {
        Ok(value) => {
            // JSON 合法但不是对象（比如 `"ReAct"` 或 `[1,2]`）——
            // 归一化成对象，键名用常见的参数名兜底。
            match value {
                serde_json::Value::Object(_) => (value, None),
                serde_json::Value::String(s) => (
                    serde_json::json!({ "query": s }),
                    Some("Action Input 是字符串，已按 `query` 参数处理".to_string()),
                ),
                other => (
                    serde_json::json!({ "input": other }),
                    Some("Action Input 不是对象，已按 `input` 参数处理".to_string()),
                ),
            }
        }
        Err(err) => {
            // 兜底：不是 JSON —— 当成字符串参数。
            //
            // `search` 的核心参数就是 `query`，
            // 模型写成 `Action Input: ReAct` 意图很清楚。
            (
                serde_json::json!({ "query": cleaned }),
                Some(format!(
                    "Action Input 不是合法 JSON（{err}），已按 `query` 字符串处理"
                )),
            )
        }
    }
}

/// 一步的执行记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// 第几步（从 1 开始）。
    pub step: usize,

    /// 模型的原始输出。
    pub raw_output: String,

    /// 思考内容（`ActOnly` 模式为空）。
    pub thought: String,

    /// 工具名（最终答案那一步为空）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,

    /// 工具参数。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,

    /// 观察文本（工具结果）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation: Option<String>,

    /// 工具是否执行成功。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_ok: Option<bool>,

    /// 参数解析错误（如果有）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parse_error: Option<String>,
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

    /// 运行模式。
    pub mode: AgentMode,

    /// 最终答案。
    pub answer: String,

    /// 每一步的记录。
    pub steps: Vec<StepRecord>,

    /// 工具调用次数。
    pub tool_calls: usize,

    /// 是否因为达到步数上限而中止。
    ///
    /// `true` 表示模型还想继续但被掐断了 ——
    /// 这通常说明工具描述不清楚，或者问题本身需要更多步骤。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,
}

impl AgentRun {
    /// 完整的 ReAct 轨迹（`Thought/Action/Action Input/Observation` + 最终答案）。
    ///
    /// 和提示词里喂给模型的是**同一份文本**（外加最后的 `Final Answer:`）——
    /// 所以模型看到的历史就是用户能看到的轨迹。
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
        // 最终答案也放进轨迹 —— 否则轨迹看起来是「跑了一半」。
        out.push_str(&format!("Final Answer: {}\n", self.answer));
        out
    }
}

/// 自主 Agent。
///
/// 持有 LLM、工具集和配置。
pub struct Agent {
    llm: Arc<dyn Llm>,
    tools: Vec<Arc<dyn Tool>>,
    config: AppConfig,
    agent_config: AgentConfig,
    prompt_config: PromptConfig,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("mode", &self.agent_config.mode)
            .field("tools", &self.tools.len())
            .finish()
    }
}

impl Agent {
    /// 创建 Agent。
    pub fn new(llm: Arc<dyn Llm>, tools: Vec<Arc<dyn Tool>>, config: AppConfig) -> Self {
        let agent_config = config.agent.clone();
        let prompt_config = config.prompt.clone();
        Self {
            llm,
            tools,
            config,
            agent_config,
            prompt_config,
        }
    }

    /// 当前配置。
    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    /// 工具集。
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    /// 运行 Agent 处理一个问题（用配置里的默认模式）。
    pub async fn run(&self, question: &str) -> AgentResult<AgentRun> {
        self.run_with_mode(question, self.agent_config.mode).await
    }

    /// 运行 Agent 处理一个问题（**显式指定模式**）。
    ///
    /// ## 为什么模式要能按次覆盖
    ///
    /// 因为消融对照端点要在**同一个 Agent 实例**上跑三种模式 ——
    /// 工具集、LLM、配置全都一样，**只有模式变量不同**。
    /// 如果模式只能在构造时定死，就得建三个 Agent 实例，
    /// 很难保证「除了模式以外其它一切相同」。
    pub async fn run_with_mode(&self, question: &str, mode: AgentMode) -> AgentResult<AgentRun> {
        let start = Instant::now();
        let question = question.trim();

        if question.is_empty() {
            return Err(AgentError::BadRequest("问题不能为空".to_string()));
        }

        let mut trajectory = String::new();
        let mut steps: Vec<StepRecord> = Vec::new();
        let mut tool_calls = 0usize;

        // `Direct` 模式不挂工具 —— 提示词里也不会出现工具说明。
        let active_tools: Vec<Arc<dyn Tool>> = if mode.uses_tools() {
            self.tools.clone()
        } else {
            Vec::new()
        };

        // ── ReAct 主循环 ───────────────────────────────────────
        for step_no in 1..=self.agent_config.max_steps {
            debug!(
                mode = mode.name(),
                step = step_no,
                "第 {step_no} 步：组装提示词"
            );

            let (system, user) = prompt::assemble(
                mode,
                &tool_specs_of(&active_tools),
                question,
                &trajectory,
                self.prompt_config.few_shot,
            );

            let raw_output = self.llm.chat(&system, &user).await?;
            debug!(
                mode = mode.name(),
                step = step_no,
                raw_len = raw_output.len(),
                "模型已返回"
            );
            let parsed = parse_react_output(&raw_output);

            match parsed {
                ParsedOutput::FinalAnswer { text } => {
                    steps.push(StepRecord {
                        step: step_no,
                        raw_output: raw_output.clone(),
                        thought: String::new(),
                        tool: None,
                        args: None,
                        observation: None,
                        tool_ok: None,
                        parse_error: None,
                    });
                    trajectory.push_str(&format!("Final Answer: {text}\n"));

                    info!(
                        mode = mode.name(),
                        step = step_no,
                        tool_calls,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        "Agent 给出最终答案"
                    );

                    return Ok(AgentRun {
                        question: question.to_string(),
                        mode,
                        answer: text,
                        steps,
                        tool_calls,
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
                        tool_ok: None,
                        parse_error: Some("模型输出既没有 Final Answer 也没有 Action".to_string()),
                    });
                    trajectory.push_str(&format!("Final Answer: {text}\n"));

                    warn!(
                        mode = mode.name(),
                        step = step_no,
                        "模型输出既没有 Final Answer 也没有 Action —— 按最终答案处理"
                    );

                    return Ok(AgentRun {
                        question: question.to_string(),
                        mode,
                        answer: text,
                        steps,
                        tool_calls,
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
                    // 执行工具。
                    let outcome = run_tool_in(&active_tools, &tool, &args);

                    debug!(
                        mode = mode.name(),
                        step = step_no,
                        tool = %tool,
                        ok = outcome.ok,
                        observation_len = outcome.observation.chars().count(),
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

                    // 记录这一步。
                    steps.push(StepRecord {
                        step: step_no,
                        raw_output: raw_output.clone(),
                        thought: thought.clone(),
                        tool: Some(tool.clone()),
                        args: Some(args.clone()),
                        observation: Some(outcome.observation.clone()),
                        tool_ok: Some(outcome.ok),
                        parse_error: parse_error.clone(),
                    });
                    tool_calls += 1;

                    // 追加到轨迹 —— 下一轮模型会看到它。
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

        // ── 步数用尽 ──────────────────────────────────────────
        //
        // 模型还想继续，但被 `max_steps` 掐断了。
        //
        // ⚠️ 这里**不**直接报错 —— 而是再问一次，让它基于已有观察作答。
        // 这样至少能拿到一个基于真实观察的答案，
        // 而不是一句「我需要更多步骤」。
        let hit_max_steps = true;
        warn!(
            mode = mode.name(),
            max_steps = self.agent_config.max_steps,
            tool_calls,
            "达到最大步数 —— 发起收尾调用"
        );

        let (system, user) = prompt::assemble(
            mode,
            &tool_specs_of(&active_tools),
            question,
            &trajectory,
            self.prompt_config.few_shot,
        );

        let final_prompt = format!(
            "{user}\n\n你已经用完了所有步骤。请基于目前已经获得的观察，直接以 `Final Answer:` 开头给出最终答案。"
        );

        // 兜底那次模型调用**失败也不应该让整个运行失败**。
        //
        // ⚠️ 这里是个真实踩过的坑：早期版本用 `?` 直接传播错误，
        // 结果模型在收尾时挂掉 → 整个 run 报错 →
        // **前面几步辛苦收集的观察全丢了**。
        //
        // 而那些观察是有价值的（用户能看到 Agent 查了什么）。
        // 所以这里降级：调用失败就用「已收集到的观察摘要」当答案。
        let (answer, raw_output) = match self.llm.chat(&system, &final_prompt).await {
            Ok(raw) => {
                let answer = match parse_react_output(&raw) {
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
                    self.agent_config.max_steps
                );
                (answer, String::new())
            }
        };

        steps.push(StepRecord {
            step: self.agent_config.max_steps + 1,
            raw_output,
            thought: String::new(),
            tool: None,
            args: None,
            observation: None,
            tool_ok: None,
            parse_error: None,
        });

        info!(
            mode = mode.name(),
            hit_max_steps,
            tool_calls,
            step_count = steps.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Agent 运行结束"
        );

        Ok(AgentRun {
            question: question.to_string(),
            mode,
            answer,
            steps,
            tool_calls,
            hit_max_steps,
            elapsed_ms: start.elapsed().as_millis() as u64,
            model: self.config.model.clone(),
        })
    }

    /// 工具定义列表。
    #[allow(dead_code)]
    fn tool_specs(&self) -> Vec<crate::tools::ToolSpec> {
        tool_specs_of(&self.tools)
    }
}

/// 工具定义列表（抽成自由函数，方便按模式过滤）。
fn tool_specs_of(tools: &[Arc<dyn Tool>]) -> Vec<crate::tools::ToolSpec> {
    tools.iter().map(|t| t.spec()).collect()
}

/// 执行工具（抽成自由函数）。
///
/// ## 工具失败**不返回错误**
///
/// 工具报错时，我们把错误信息当成一条 `Observation` 喂回模型 ——
/// 模型读到「参数不合法」就能自己纠正（换参数重试）。
///
/// 这正是 [`AgentError::is_degradable`] 只对工具错误返回 `true` 的语义。
fn run_tool_in(tools: &[Arc<dyn Tool>], name: &str, args: &serde_json::Value) -> ToolOutcome {
    let Some(tool) = tools.iter().find(|t| t.spec().name == name) else {
        return ToolOutcome::failed(
            name,
            format!(
                "错误：未知工具 `{name}`。可用工具：{}",
                tools
                    .iter()
                    .map(|t| t.spec().name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
    };

    tool.run(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_final_answer() {
        let parsed = parse_react_output("Thought: 我知道了\nFinal Answer: 92");
        match parsed {
            ParsedOutput::FinalAnswer { text } => assert_eq!(text, "92"),
            other => panic!("expected final answer, got {other:?}"),
        }
    }

    #[test]
    fn parses_action_with_thought() {
        let parsed = parse_react_output(
            "Thought: 需要查资料\nAction: search\nAction Input: {\"query\": \"ReAct\"}",
        );
        match parsed {
            ParsedOutput::Action {
                thought,
                tool,
                args,
                parse_error,
            } => {
                assert_eq!(thought, "需要查资料");
                assert_eq!(tool, "search");
                assert_eq!(args["query"], "ReAct");
                assert!(parse_error.is_none());
            }
            other => panic!("expected action, got {other:?}"),
        }
    }

    #[test]
    fn parses_action_without_thought() {
        let parsed =
            parse_react_output("Action: calculate\nAction Input: {\"expression\": \"1+1\"}");
        match parsed {
            ParsedOutput::Action { thought, tool, .. } => {
                assert!(thought.is_empty());
                assert_eq!(tool, "calculate");
            }
            other => panic!("expected action, got {other:?}"),
        }
    }

    #[test]
    fn final_answer_wins_over_action_marker() {
        // 模型的思考里可能提到 "Action:" —— 必须先判 Final Answer。
        let parsed = parse_react_output("Thought: 我要说 Action: 这个词\nFinal Answer: 真正的答案");
        match parsed {
            ParsedOutput::FinalAnswer { text } => assert_eq!(text, "真正的答案"),
            other => panic!("expected final answer, got {other:?}"),
        }
    }

    #[test]
    fn handles_markdown_fenced_json() {
        let parsed = parse_react_output(
            "Action: search\nAction Input: ```json\n{\"query\": \"ReAct\"}\n```",
        );
        match parsed {
            ParsedOutput::Action {
                args, parse_error, ..
            } => {
                assert_eq!(args["query"], "ReAct");
                assert!(parse_error.is_none());
            }
            other => panic!("expected action, got {other:?}"),
        }
    }

    #[test]
    fn falls_back_to_query_for_plain_text_input() {
        // 模型写成 `Action Input: ReAct` —— 意图清楚，按 query 处理。
        let parsed = parse_react_output("Action: search\nAction Input: ReAct");
        match parsed {
            ParsedOutput::Action {
                args, parse_error, ..
            } => {
                assert_eq!(args["query"], "ReAct");
                assert!(parse_error.is_some(), "应该记录这个容错");
            }
            other => panic!("expected action, got {other:?}"),
        }
    }

    #[test]
    fn unstructured_output_is_tolerated() {
        let parsed = parse_react_output("我觉得答案应该是 42。");
        match parsed {
            ParsedOutput::Unstructured { text } => assert!(text.contains("42")),
            other => panic!("expected unstructured, got {other:?}"),
        }
    }

    #[test]
    fn empty_output_is_unstructured() {
        assert!(matches!(
            parse_react_output(""),
            ParsedOutput::Unstructured { .. }
        ));
    }

    #[test]
    fn action_without_input_still_parses() {
        let parsed = parse_react_output("Action: search");
        match parsed {
            ParsedOutput::Action { tool, args, .. } => {
                assert_eq!(tool, "search");
                assert!(args.as_object().unwrap().is_empty());
            }
            other => panic!("expected action, got {other:?}"),
        }
    }

    #[test]
    fn trajectory_renders_the_full_loop() {
        let run = AgentRun {
            question: "q".to_string(),
            mode: AgentMode::ReAct,
            answer: "a".to_string(),
            steps: vec![StepRecord {
                step: 1,
                raw_output: String::new(),
                thought: "先查".to_string(),
                tool: Some("search".to_string()),
                args: Some(json!({ "query": "ReAct" })),
                observation: Some("[1] 内容".to_string()),
                tool_ok: Some(true),
                parse_error: None,
            }],
            tool_calls: 1,
            hit_max_steps: false,
            elapsed_ms: 0,
            model: "m".to_string(),
        };

        let traj = run.trajectory();
        assert!(traj.contains("Thought: 先查"));
        assert!(traj.contains("Action: search"));
        assert!(traj.contains("Action Input:"));
        assert!(traj.contains("Observation: [1] 内容"));
    }

    #[test]
    fn parse_action_input_rejects_empty() {
        let (args, err) = parse_action_input("");
        assert!(args.as_object().unwrap().is_empty());
        assert!(err.is_some());
    }

    // ── 回显（echo）场景的回归测试 ──────────────────────────────────
    //
    // 模型有时会**回显整个提示词**再继续。
    // 如果解析器找**第一个**标记，就会匹配到回显里的旧 `Action:`，
    // 把上一步的工具重跑一遍，然后丢掉模型真正的答案。
    //
    // 实测症状：答案变成「平均每轮调用 3 次工具」这种完全无关的内容。

    /// 构造一段「回显 + 真正输出」的文本。
    fn echoed_output() -> String {
        "Question: 用户的问题\n\
         Thought: 用户问的是 ReAct 框架里 Observation 的定义。\n\
         Action: search\n\
         Action Input: {\"query\": \"ReAct Observation\", \"top_k\": 2}\n\
         Observation: [1] 旧的第一条观察\n\
         Thought: 信息足够，可以直接作答了。\n\
         Final Answer: Observation 是工具执行后返回给模型的结果。"
            .to_string()
    }

    #[test]
    fn echo_does_not_cause_the_previous_action_to_repeat() {
        let parsed = parse_react_output(&echoed_output());
        match parsed {
            ParsedOutput::FinalAnswer { text } => {
                assert!(text.contains("Observation 是工具执行后返回给模型的结果"));
                assert!(!text.contains("旧的第一条观察"));
            }
            other => panic!("expected final answer, got {other:?}"),
        }
    }

    #[test]
    fn echo_uses_the_last_thought_not_the_echoed_one() {
        let parsed = parse_react_output(&echoed_output());
        match parsed {
            ParsedOutput::FinalAnswer { .. } => {}
            other => panic!("expected final answer, got {other:?}"),
        }
    }

    #[test]
    fn echo_with_new_action_uses_the_new_one() {
        // 回显之后模型又给了一个新动作 —— 必须用**新的**那个。
        let output = "Action: search\n\
                      Action Input: {\"query\": \"旧查询\"}\n\
                      Observation: 旧结果\n\
                      Thought: 换个关键词再试\n\
                      Action: search\n\
                      Action Input: {\"query\": \"新查询\"}\n";

        let parsed = parse_react_output(output);
        match parsed {
            ParsedOutput::Action {
                tool,
                args,
                thought,
                ..
            } => {
                assert_eq!(tool, "search");
                assert_eq!(args["query"], "新查询");
                assert_eq!(thought, "换个关键词再试");
            }
            other => panic!("expected action, got {other:?}"),
        }
    }

    #[test]
    fn echo_with_final_answer_picks_the_final_answer() {
        let output = "Thought: 旧思考\n\
                      Action: search\n\
                      Action Input: {\"query\": \"旧\"}\n\
                      Final Answer: 旧答案\n\
                      Final Answer: 新答案";

        let parsed = parse_react_output(output);
        match parsed {
            ParsedOutput::FinalAnswer { text } => assert_eq!(text, "新答案"),
            other => panic!("expected final answer, got {other:?}"),
        }
    }
}
