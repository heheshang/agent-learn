//! Agent 循环 —— 复用 13/14 的 ReAct 引擎，接上 SQL 工具。
//!
//! ## 复用与新增
//!
//! | 能力 | 来源 |
//! |---|---|
//! | 模型输出解析 | [`agent_13::agent::parse_react_output`]（复用） |
//! | 工具注册表（参数校验 + 失败转观察） | [`agent_14::tool::ToolRegistry`]（复用） |
//! | 提示词组装 | 本项目 [`crate::prompt`] |
//! | SQL 护栏 | 本项目 [`crate::guard`] |
//!
//! ## 为什么要复用 13/14 而不是重写
//!
//! 解析器是 Agent 系统里**最容易出错**的部分 ——
//! 13 已经踩过「模型回显整个提示词」的坑并修好了
//! （必须找**最后一个**标记，而不是第一个）。
//!
//! 重写一遍等于把坑再踩一次。
//!
//! ## 本项目的独有价值
//!
//! **护栏在循环里的位置**：
//!
//! ```text
//! 模型给出 Action: sql_db_query
//!    ↓
//! 注册表校验参数（Schema）
//!    ↓
//! 护栏检查 SQL（只读？单语句？）
//!    ↓ 通过 → 只读连接执行
//!    ↓ 不通过 → 错误文本作为 Observation 喂回模型
//! ```
//!
//! ⚠️ **护栏失败不终止循环** —— 它和 SQL 执行错误一样，
//! 是「可自我纠正的失败」。模型读到「被护栏拦截，请改成只读查询」
//! 就能改。这是本项目最重要的一条设计。

use std::sync::Arc;
use std::time::Instant;

use agent_14::tool::{ToolContext, ToolOutcome, ToolRegistry};
use agent_logging::{debug, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AppError, AppResult};
use crate::llm::Llm;

/// 模型一轮输出的解析结果（复用 13 的定义）。
pub use agent_13::agent::ParsedOutput;

/// 解析模型输出（复用 13 的解析器）。
pub use agent_13::agent::parse_react_output;

/// 「因为没查库就作答」而退回模型的最大次数。
///
/// ## 为什么要有上限
///
/// 模型可能**反复**「先给答案再被退回」——
/// 如果没有上限，循环会变成拉锯战，一路烧到 `max_steps`。
///
/// 设成 2 是因为：正常情况下**一次**就够了
/// （模型看到「你的数字没有来源」，下一轮就会去查库）；
/// 给到 2 是留一点余量给「模型需要先想清楚该查什么」。
///
/// 超过上限后就放行 —— 宁可给一个可能不准确的答案，
/// 也不能让请求永远返回不了。
pub const MAX_REJECT_ROUNDS: usize = 2;

/// 检测模型输出里**自己编造**的 `Observation:` 段。
///
/// ## ⚠️ 这是实测踩出来的坑（本项目最隐蔽的一个）
///
/// 模型的原始输出里，有时会出现**它自己写的** `Observation:` ——
///
/// ```text
/// Action: sql_db_query
/// Action Input: {"query": "SELECT ..."}
/// Observation: 列: Name | avg_ms      ← 模型自己写的！
/// 行数: 5
/// ---
/// 摇滚 | 352000.0                    ← 模型编的数字！
/// Final Answer: 平均时长最长的是摇滚，393600 毫秒
/// ```
///
/// 也就是说，模型**一次性把整条轨迹（包括观察和答案）都编了出来**，
/// 数据库**一次都没被访问**。
///
/// ## 为什么 13 的解析器挡不住
///
/// 13 的 `parse_react_output` 优先找 `Final Answer:`（找**最后一个**），
/// 所以这种输出会被当成「最终答案」——
/// 恰恰是**最危险**的那种处理：编造的数字被当成真实结果。
///
/// ## 判据
///
/// `Observation:` 这个标记**只应该出现在我们发给模型的提示词里**
/// （作为「已完成的步骤」的一部分）。
///
/// 模型**自己**的输出里出现 `Observation:`，只有一种可能：
/// 它在**伪造**工具结果。
///
/// ## 但不能一刀切
///
/// 模型有时会**回显整个提示词**（13 实测踩过的坑）——
/// 回显里就带着提示词中的 `Observation:`。
/// 它也可能在 `Thought` 里**引用**真实观察。
///
/// 所以判据是三个条件同时成立：
///
/// 1. `Observation:` 出现在**最后一个** `Action:` **之后**
///    （回显的观察都在 Action 之前）
/// 2. 观察的内容**不是**提示词里已有的内容
///    （引用的真实观察 → 命中提示词 → 放行）
/// 3. 观察内容**非空**
///
/// ## 返回值
///
/// `Some(伪造的观察文本)` 表示模型伪造了观察；
/// `None` 表示没有伪造。
pub fn detect_fabricated_observation(raw: &str, prompt: &str) -> Option<String> {
    let (obs_pos, text) = locate_fabricated_observation(raw)?;
    let _ = obs_pos;

    // ⚠️ 如果这段内容**就在提示词里** → 它是引用/回显，不是伪造。
    let normalized_prompt = normalize_ws(prompt);
    if normalized_prompt.contains(&normalize_ws(&text)) {
        return None;
    }

    Some(text)
}

/// 定位模型输出里**伪造**的 `Observation:` 段。
///
/// 返回 `(标记的字节偏移, 伪造观察的文本)`。
///
/// ## ⚠️ 为什么两个函数共用一套逻辑
///
/// [`detect_fabricated_observation`] 用来「问是不是伪造的」，
/// [`cut_at_fabricated_observation`] 用来「在哪切」。
///
/// 两者必须**判据一致** —— 否则会出现
/// 「检测说不是伪造，但切的时候切了」这种诡异 bug。
fn locate_fabricated_observation(raw: &str) -> Option<(usize, String)> {
    // 找最后一个 `Action:`（大小写不敏感）。
    let action_pos = find_last_ci(raw, "action:")?;

    // 在它**之后**找 `Observation:`。
    let obs_pos = find_ci(&raw[action_pos..], "observation:")? + action_pos;

    // 伪造的观察从 `Observation:` 之后开始，
    // 到下一个 `Final Answer:` 或 `Action:` 之前结束。
    let content_start = obs_pos + "observation:".len();
    let tail = &raw[content_start..];

    let end = find_ci(tail, "final answer:")
        .or_else(|| find_ci(tail, "action:"))
        .unwrap_or(tail.len());

    let text = tail[..end].trim().to_string();

    if text.is_empty() {
        return None;
    }

    Some((obs_pos, text))
}

/// 在模型输出里找到**伪造的 `Observation:` 标记**的位置。
///
/// 返回该标记在 `raw` 中的**字节偏移**（找不到返回 `None`）。
///
/// 用途：把它**之后**的内容全部切掉 ——
///
/// - 伪造的观察被丢弃
/// - 紧跟其后的 `Final Answer:` 也被丢弃（它建立在伪造数据上）
/// - 保留下来的最后一个 `Action:` 会被**真正执行**
pub fn cut_at_fabricated_observation(raw: &str, prompt: &str) -> Option<usize> {
    let (obs_pos, text) = locate_fabricated_observation(raw)?;

    // ⚠️ 如果这段内容**就在提示词里** → 它是引用/回显，不是伪造。
    //
    // 用「归一化空白后子串匹配」——
    // 模型引用时可能换行、缩进都变了。
    let normalized_prompt = normalize_ws(prompt);
    if normalized_prompt.contains(&normalize_ws(&text)) {
        return None;
    }

    Some(obs_pos)
}

/// 归一化空白（把连续空白压成一个空格，并去掉首尾空白）。
fn normalize_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// 在 `text` 里找 `needle`（大小写不敏感），返回**字节偏移**。
fn find_ci(text: &str, needle: &str) -> Option<usize> {
    let lower_text = text.to_lowercase();
    let lower_needle = needle.to_lowercase();
    lower_text.find(&lower_needle)
}

/// 在 `text` 里找 `needle`（大小写不敏感）的**最后一次**出现，返回字节偏移。
fn find_last_ci(text: &str, needle: &str) -> Option<usize> {
    let lower_text = text.to_lowercase();
    let lower_needle = needle.to_lowercase();
    lower_text.rfind(&lower_needle)
}

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

    /// 这一步是不是最终答案。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_answer: Option<bool>,
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

    /// 执行过的 SQL 语句（去重，按首次出现顺序）。
    ///
    /// ## 为什么单独收集
    ///
    /// 这是「安全审计」的核心证据 ——
    /// 用户能看到「这个 Agent 到底对数据库做了什么」。
    ///
    /// 而且它让「护栏拦了什么」变得可见：
    /// 被拦的 SQL 也会出现在这里（标记为未执行）。
    pub executed_sql: Vec<String>,

    /// 被护栏拦截的 SQL（去重）。
    ///
    /// ## 这是本项目最重要的可观测性
    ///
    /// 如果模型频繁试图写数据，说明提示词或 few-shot 需要调整。
    /// 没有这个指标，就只能靠肉眼看轨迹。
    pub blocked_sql: Vec<String>,

    /// 答案**是否有真实数据支撑**。
    ///
    /// ## ⚠️ 为什么这个字段是必需的（实测踩坑）
    ///
    /// 实测发现模型经常**不执行任何查询**就给出答案，
    /// 而且数字是**编的**（看起来非常合理）。
    ///
    /// 一个看起来专业的错误答案，比一个明显的错误**更危险** ——
    /// 调用方很难发现它是编的。
    ///
    /// 所以这个字段把「答案有没有数据来源」变成一个**可断言的布尔值**：
    ///
    /// - `true`  —— 至少有一条查询**成功执行**过，答案有据可查
    /// - `false` —— 模型没查库（或查询全被拦下），答案是凭空生成的
    ///
    /// 调用方（和测试）可以直接断言它。
    pub grounded: bool,

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

/// SQL Agent。
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
    pub fn tool_specs(&self) -> Vec<agent_14::tool::ToolSpec> {
        self.registry.specs()
    }

    /// 运行 Agent 处理一个问题。
    pub async fn run(&self, question: &str) -> AppResult<AgentRun> {
        let start = Instant::now();
        let question = question.trim();

        if question.is_empty() {
            return Err(AppError::BadRequest("问题不能为空".to_string()));
        }

        let mut trajectory = String::new();
        let mut steps: Vec<StepRecord> = Vec::new();
        let mut tool_calls = 0usize;
        let mut tools_used: Vec<String> = Vec::new();
        let mut executed_sql: Vec<String> = Vec::new();
        let mut blocked_sql: Vec<String> = Vec::new();

        // ⚠️ 是否已经**真正执行过**一条查询。
        //
        // 这是「必须先查库才允许作答」这条硬约束的状态位。
        //
        // ⚠️ 注意它和 `executed_sql` 不完全等价 ——
        //    `executed_sql` 是**去重后**的列表（给审计看），
        //    这个布尔值是**流程状态**（给循环看）。
        let mut has_queried = false;

        // ⚠️ 是否**尝试过**查询（不管成功还是被拦）。
        //
        // ## 为什么需要区分这两个状态
        //
        // 如果只查 `has_queried`（只看成功），
        // 那么「模型尝试写数据 → 被护栏拦了 → 模型回答『数据库是只读的』」
        // 这种**完全正确**的应答也会被误判成「没查库就作答」而被退回。
        //
        // 所以判据是：**一次查询都没尝试过**才退回。
        let mut query_attempted = false;

        // ⚠️ 「因为没查库而被退回」的次数。
        //
        // 必须有上限 —— 否则模型可能无限次地「先给答案」，
        // 循环变成「反复退回 → 反复作答」的拉锯战。
        let mut reject_rounds = 0usize;

        let specs = self.registry.specs();
        let ctx = ToolContext::empty();
        let system = crate::prompt::system_prompt();
        let few_shot = self.config.prompt.few_shot;

        // ── 主循环 ──────────────────────────────────────────────
        for step_no in 1..=self.config.agent.max_steps {
            debug!(step = step_no, tool_count = specs.len(), "第 {step_no} 步");

            let user = crate::prompt::build_prompt(&specs, question, &trajectory, few_shot);

            // 调试：打印真实发给模型的提示词。
            //
            // ⚠️ 排查「模型为什么做出奇怪的动作」时，
            //    看「模型到底看到了什么」比看「模型说了什么」更有效。
            if std::env::var_os("SQL_SHOW_PROMPT").is_some() {
                eprintln!("\n========== 第 {step_no} 步的完整提示词 ==========");
                eprintln!("{user}");
                eprintln!("========== 提示词结束 ==========\n");
            }

            let raw_output = self.llm.chat(&system, &user).await?;

            // ── 剥离模型**伪造**的 Observation ──────────────────
            //
            // ⚠️ 实测踩坑：模型有时会一口气把整条轨迹（含 Observation
            //    和 Final Answer）都编出来，数据库一次都没被访问。
            //
            //    这里在解析**之前**就把伪造段切掉 ——
            //    这样 13 的解析器看到的就只剩「Thought + Action + Action Input」，
            //    它会正常解析出一个**动作**，我们再去**真正执行**它。
            let (effective_output, fabricated) =
                match cut_at_fabricated_observation(&raw_output, &user) {
                    Some(pos) => {
                        warn!(
                            step = step_no,
                            "模型伪造了 Observation —— 已剥离，改用其中的 Action 真正执行"
                        );
                        (
                            raw_output[..pos].trim().to_string(),
                            Some(raw_output[pos..].to_string()),
                        )
                    }
                    None => (raw_output.clone(), None),
                };

            let parsed = parse_react_output(&effective_output);

            match parsed {
                ParsedOutput::FinalAnswer { text } => {
                    // ── 「必须先查库」硬约束 ──────────────────────
                    //
                    // ⚠️ 实测踩坑：模型经常只查 schema 就直接给答案，
                    //    而且数字是**编的**（样例数据只有 3 行，
                    //    不可能算出平均值）。
                    //
                    //    提示词里写了「答案里的每个数字都必须来自查询结果」，
                    //    但那是**软约束** —— 模型可以不听。
                    //
                    //    这里把它变成**机制**：
                    //    没**尝试**查过库 → 退回成 Observation → 让模型先去查。
                    //
                    // ⚠️ 判据是「**尝试过**」而不是「**成功过**」——
                    //    因为「尝试写数据 → 被护栏拦了 → 回答『数据库是只读的』」
                    //    是**完全正确**的应答，不能被误判成「没查库就作答」。
                    if self.config.agent.require_query
                        && !query_attempted
                        && reject_rounds < MAX_REJECT_ROUNDS
                    {
                        reject_rounds += 1;

                        let hint = "错误：你还没有执行过任何查询，\
                             所以这个答案里的数字**没有数据来源** —— \
                             它是凭空编的。请先用 `sql_db_query` 执行一条只读查询拿到真实数据，\
                             再基于查询结果给出最终答案。";

                        warn!(
                            step = step_no,
                            reject_round = reject_rounds,
                            "模型未查库就作答 —— 退回，要求先执行查询"
                        );

                        // 把这条「假答案」写进轨迹 —— 模型下一轮能看到自己刚才说了什么，
                        // 从而更可能真的去查库（而不是换个姿势再编一次）。
                        trajectory.push_str(&format!("Thought: {text}\n"));
                        trajectory.push_str(&format!("Observation: {hint}\n"));

                        steps.push(StepRecord {
                            step: step_no,
                            raw_output: raw_output.clone(),
                            thought: String::new(),
                            tool: None,
                            args: None,
                            observation: Some(hint.to_string()),
                            data: None,
                            tool_ok: Some(false),
                            parse_error: Some("未执行查询就给出最终答案 —— 已退回".to_string()),
                            final_answer: None,
                        });

                        continue;
                    }

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
                        final_answer: Some(true),
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
                        executed_sql,
                        blocked_sql,
                        grounded: has_queried,
                        hit_max_steps: false,
                        elapsed_ms: start.elapsed().as_millis() as u64,
                        model: self.config.model.clone(),
                    });
                }

                ParsedOutput::Unstructured { text } => {
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
                        final_answer: Some(true),
                    });

                    warn!(step = step_no, "模型输出无法解析 —— 按最终答案处理");

                    return Ok(AgentRun {
                        question: question.to_string(),
                        answer: text,
                        steps,
                        tool_calls,
                        tools_used,
                        executed_sql,
                        blocked_sql,
                        grounded: has_queried,
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
                    let outcome = self.registry.call(&tool, &args, &ctx).await;

                    debug!(
                        step = step_no,
                        tool = %tool,
                        ok = outcome.ok,
                        "工具已执行"
                    );

                    if !outcome.ok {
                        warn!(
                            step = step_no,
                            tool = %tool,
                            observation = %outcome.observation,
                            "工具执行失败（错误已作为观察喂回模型）"
                        );
                    }

                    if !tools_used.contains(&tool) {
                        tools_used.push(tool.clone());
                    }

                    // ── 记录 SQL（用于安全审计）────────────────────
                    //
                    // ⚠️ **被拦截的 SQL 也要记录** ——
                    //    它能证明护栏真的在起作用。
                    if tool == "sql_db_query"
                        && let Some(sql) = args.get("query").and_then(Value::as_str)
                    {
                        let sql = sql.trim().to_string();
                        // ⚠️ 无论成功还是被拦，都算「**尝试过**查询」。
                        query_attempted = true;
                        if outcome.ok {
                            // ⚠️ 只有**成功**执行的查询才算「查过库」——
                            //    护栏拦下的 SQL 不算数（没落地）。
                            has_queried = true;
                            if !executed_sql.contains(&sql) {
                                executed_sql.push(sql);
                            }
                        } else if !blocked_sql.contains(&sql) {
                            blocked_sql.push(sql);
                        }
                    }

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
                        final_answer: None,
                    });
                    tool_calls += 1;

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

                    // ⚠️ 如果这一轮模型伪造过观察，要**明确告诉它**。
                    //
                    // 否则模型下一轮可能继续「自己写 Observation」——
                    // 它以为那是合法的对话格式。
                    if fabricated.is_some() {
                        trajectory.push_str(
                            "Observation: 你上一条回复里自己写了一段 `Observation:` —— \
                             那是**伪造的**，工具还没有执行过这个动作。\
                             上面这个 Observation 才是工具的真实返回。\
                             以后不要自己写 Observation 行，只输出 Thought / Action / Action Input。\n",
                        );
                    }
                }
            }
        }

        // ── 步数用尽 ────────────────────────────────────────────
        warn!(
            max_steps = self.config.agent.max_steps,
            tool_calls, "达到最大步数 —— 发起收尾调用"
        );

        let user = crate::prompt::build_prompt(&specs, question, &trajectory, few_shot);
        let final_prompt = format!(
            "{user}\n\n你已经用完了所有步骤。请基于目前已经获得的观察，直接以 `Final Answer:` 开头给出最终答案。"
        );

        // ⚠️ 收尾那次调用失败**也不应该**让整个运行失败 ——
        // 前面几步辛苦收集的观察是有价值的。
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
            final_answer: Some(true),
        });

        info!(
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
            executed_sql,
            blocked_sql,
            grounded: has_queried,
            hit_max_steps: true,
            elapsed_ms: start.elapsed().as_millis() as u64,
            model: self.config.model.clone(),
        })
    }
}

/// 直接调用工具（不经模型）。
///
/// 对应 LangChain 里「工具可以被独立调用」这一条 ——
///
/// - **调试**：直接看工具返回什么，不用等模型
/// - **确定性**：同样的参数永远同样的结果
/// - **成本**：不调模型
pub async fn call_tool_directly(
    registry: &ToolRegistry,
    name: &str,
    args: &Value,
    ctx: &ToolContext,
) -> Result<ToolOutcome, agent_14::error::ToolError> {
    registry.call_strict(name, args, ctx).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::StaticChecker;
    use crate::db::Database;
    use crate::llm::ScriptedLlm;
    use crate::tools::{ListTablesTool, QueryCheckerTool, QueryTool, SchemaTool};
    use serde_json::json;

    fn setup_db() -> (tempfile::TempDir, Arc<Database>) {
        let dir = tempfile::tempdir().unwrap();
        let seed = dir.path().join("seed.sql");
        std::fs::write(
            &seed,
            r#"
            CREATE TABLE Genre (GenreId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
            CREATE TABLE Track (
                TrackId INTEGER PRIMARY KEY,
                Name TEXT NOT NULL,
                GenreId INTEGER NOT NULL REFERENCES Genre(GenreId),
                Milliseconds INTEGER NOT NULL
            );
            INSERT INTO Genre (GenreId, Name) VALUES (1, '摇滚'), (2, '金属');
            INSERT INTO Track (TrackId, Name, GenreId, Milliseconds) VALUES
                (1, '夜色', 1, 300000),
                (2, '回声', 1, 200000),
                (3, '侧影', 2, 400000);
            "#,
        )
        .unwrap();
        let db_path = dir.path().join("test.db");
        crate::db::load_seed(&seed, &db_path).unwrap();
        (dir, Arc::new(Database::open_readonly(&db_path).unwrap()))
    }

    fn build(answers: Vec<String>) -> (Agent, Arc<ScriptedLlm>) {
        let (_dir, db) = setup_db();
        let llm = Arc::new(ScriptedLlm::new(answers));

        let registry = ToolRegistry::new(vec![
            Arc::new(ListTablesTool::new(db.clone())),
            Arc::new(SchemaTool::new(db.clone())),
            Arc::new(QueryTool::new(db)),
            Arc::new(QueryCheckerTool::new(Arc::new(StaticChecker::new()))),
        ]);

        let mut config = crate::config::AppConfig::default();
        config.agent.max_steps = 6;

        (Agent::new(llm.clone(), registry, config), llm)
    }

    #[tokio::test]
    async fn runs_the_full_discovery_loop() {
        let (agent, _llm) = build(vec![
            "Thought: 先看表\nAction: sql_db_list_tables\nAction Input: {}".to_string(),
            "Thought: 看 schema\nAction: sql_db_schema\nAction Input: {\"table_names\": \"Genre\"}".to_string(),
            "Thought: 查\nAction: sql_db_query\nAction Input: {\"query\": \"SELECT Name FROM Genre LIMIT 5\"}".to_string(),
            "Final Answer: 有摇滚和金属".to_string(),
        ]);

        let run = agent.run("有哪些流派").await.unwrap();

        assert_eq!(run.tool_calls, 3);
        assert_eq!(run.steps.len(), 4);
        assert!(run.answer.contains("摇滚"));
        assert_eq!(run.tools_used.len(), 3);
    }

    #[tokio::test]
    async fn records_executed_sql() {
        let (agent, _llm) = build(vec![
            "Action: sql_db_query\nAction Input: {\"query\": \"SELECT Name FROM Genre LIMIT 5\"}"
                .to_string(),
            "Final Answer: 好了".to_string(),
        ]);

        let run = agent.run("问题").await.unwrap();

        assert_eq!(run.executed_sql.len(), 1);
        assert!(run.executed_sql[0].contains("SELECT Name FROM Genre"));
        assert!(run.blocked_sql.is_empty());
    }

    #[tokio::test]
    async fn records_blocked_sql() {
        // ⚠️ 关键测试：护栏拦下的 SQL 必须被**记录**，
        //    这样「护栏到底拦了什么」才可观测。
        let (agent, _llm) = build(vec![
            "Action: sql_db_query\nAction Input: {\"query\": \"DELETE FROM Track\"}".to_string(),
            "Final Answer: 数据库是只读的".to_string(),
        ]);

        let run = agent.run("删掉所有曲目").await.unwrap();

        assert_eq!(run.blocked_sql.len(), 1);
        assert!(run.blocked_sql[0].contains("DELETE"));
        assert!(run.executed_sql.is_empty());
    }

    #[tokio::test]
    async fn guard_rejection_is_fed_back_not_fatal() {
        let (agent, llm) = build(vec![
            "Action: sql_db_query\nAction Input: {\"query\": \"DROP TABLE Track\"}".to_string(),
            "Final Answer: 我不能删表".to_string(),
        ]);

        let run = agent.run("问题").await.unwrap();

        assert_eq!(run.steps[0].tool_ok, Some(false));
        assert!(run.steps[0].observation.as_ref().unwrap().contains("只读"));

        // 错误文本必须进了下一轮的提示词。
        let calls = llm.calls();
        assert!(calls[1].1.contains("只读"));
    }

    #[tokio::test]
    async fn sql_execution_error_is_fed_back() {
        let (agent, _llm) = build(vec![
            "Action: sql_db_query\nAction Input: {\"query\": \"SELECT nosuchcol FROM Track\"}"
                .to_string(),
            "Final Answer: 列名不对".to_string(),
        ]);

        let run = agent.run("问题").await.unwrap();

        assert_eq!(run.steps[0].tool_ok, Some(false));
        assert!(
            run.steps[0]
                .observation
                .as_ref()
                .unwrap()
                .contains("nosuchcol")
        );
        // 失败的 SQL 记在 blocked 里（因为没成功执行）。
        assert_eq!(run.blocked_sql.len(), 1);
    }

    #[tokio::test]
    async fn empty_question_is_rejected() {
        let (agent, _llm) = build(vec![]);
        let err = agent.run("   ").await.unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[tokio::test]
    async fn max_steps_triggers_a_degraded_finish() {
        // max_steps = 1，脚本只有 1 个答案 → 收尾调用失败 → 降级。
        let (agent, llm) = build(vec![
            "Action: sql_db_list_tables\nAction Input: {}".to_string(),
        ]);

        let mut config = agent.config().clone();
        config.agent.max_steps = 1;
        let agent = Agent::new(llm.clone(), agent.registry().clone(), config);

        let run = agent.run("问题").await.unwrap();

        assert!(run.hit_max_steps);
        assert!(run.answer.contains("收尾请求失败"));
        // 观察不能丢。
        assert_eq!(run.steps.len(), 2);
    }

    #[tokio::test]
    async fn direct_tool_call_bypasses_the_model() {
        let (agent, _llm) = build(vec![]);

        let outcome = call_tool_directly(
            agent.registry(),
            "sql_db_query",
            &json!({ "query": "SELECT Name FROM Genre LIMIT 5" }),
            &ToolContext::empty(),
        )
        .await
        .unwrap();

        assert!(outcome.ok);
        assert!(outcome.observation.contains("摇滚"));
    }

    #[tokio::test]
    async fn direct_tool_call_returns_guard_error_to_the_user() {
        // ⚠️ 用户直接调工具时，错误应该**原样返回** ——
        //    因为这时用户扮演了模型角色。
        let (agent, _llm) = build(vec![]);

        let err = call_tool_directly(
            agent.registry(),
            "sql_db_query",
            &json!({ "query": "DROP TABLE Track" }),
            &ToolContext::empty(),
        )
        .await
        .unwrap_err();

        assert!(err.to_string().contains("只读"));
    }

    // ── 坑 1：模型伪造 Observation（一口气编完整条轨迹）────────
    //
    // ⚠️ 实测症状：模型一次性输出
    //   Action: sql_db_query / Action Input: {...} /
    //   Observation: 列: Name | avg_ms / 摇滚 | 352000.0 / Final Answer: ...
    // 数据库一次都没被访问，数字全是编的。
    //
    // 修法：剥离伪造的 Observation 段，让剩下的 Action 被真正执行。

    #[tokio::test]
    async fn fabricated_observation_is_stripped_and_the_action_is_executed() {
        let (agent, _llm) = build(vec![
            // 伪造的完整轨迹（Observation 和 Final Answer 都是模型自己写的）
            "Thought: 查一下\n\
             Action: sql_db_query\n\
             Action Input: {\"query\": \"SELECT Name FROM Genre LIMIT 5\"}\n\
             Observation: 列: Name\n\
             行数: 5\n\
             ---\n\
             编造的行1\n\
             Final Answer: 编造的答案"
                .to_string(),
            "Final Answer: 真实答案：摇滚".to_string(),
        ]);

        let run = agent.run("有哪些流派").await.unwrap();

        // ⚠️ 关键：那条 SQL 必须**真的执行过**。
        assert_eq!(run.executed_sql.len(), 1);
        assert!(run.executed_sql[0].contains("SELECT Name FROM Genre"));

        // ⚠️ 编造的数字不能出现在最终答案里。
        assert!(run.answer.contains("真实答案"));
        assert!(!run.answer.contains("编造的答案"));
    }

    #[tokio::test]
    async fn fabricated_observation_alone_is_not_trusted() {
        // 只检测函数：伪造的观察应该被识别出来。
        let raw = "Thought: 查\n\
                   Action: sql_db_query\n\
                   Action Input: {\"query\": \"SELECT 1\"}\n\
                   Observation: 编造的结果\n\
                   Final Answer: 编造的答案";
        let prompt = "真实问题: xxx\n已完成的步骤：\n";

        let detected = detect_fabricated_observation(raw, prompt);
        assert!(detected.is_some());
        assert!(detected.unwrap().contains("编造的结果"));
    }

    #[tokio::test]
    async fn a_real_observation_quoted_from_the_prompt_is_not_fabrication() {
        // ⚠️ 回显场景：模型引用了提示词里**真实存在**的观察 → 不算伪造。
        let real_obs = "Observation: 可用数据表（2 张）:\nGenre（2 行）";
        let prompt = format!("真实问题: xxx\n已完成的步骤：\n{real_obs}\n");

        let raw = format!(
            "Thought: 我看到了 {real_obs}\n\
             Action: sql_db_query\n\
             Action Input: {{\"query\": \"SELECT 1\"}}\n\
             Observation: {real_obs}\n\
             Final Answer: 好了"
        );

        assert!(detect_fabricated_observation(&raw, &prompt).is_none());
    }

    // ── 坑 2：模型不查库就作答 ────────────────────────────────
    //
    // ⚠️ 实测症状：模型只查了 schema 就直接给答案，数字是编的。
    //    提示词里写了「答案里的每个数字都必须来自查询结果」，
    //    但那是**软约束** —— 模型可以不听。
    //
    // 修法：机制化兜底 —— 没查过库就把答案退回，要求先查。

    #[tokio::test]
    async fn answering_without_querying_is_rejected() {
        let (agent, _llm) = build(vec![
            // 第一轮：不查库，直接给答案。
            "Final Answer: 编造的答案：平均 393600 毫秒".to_string(),
            // 第二轮：老老实实查库。
            "Action: sql_db_query\nAction Input: {\"query\": \"SELECT Name FROM Genre LIMIT 5\"}"
                .to_string(),
            "Final Answer: 真实答案".to_string(),
        ]);

        let run = agent.run("哪个流派平均最长").await.unwrap();

        // 第一步必须被标记成「被退回」。
        assert_eq!(run.steps[0].tool_ok, Some(false));
        assert!(
            run.steps[0]
                .observation
                .as_ref()
                .unwrap()
                .contains("没有数据来源")
        );

        // 最终答案必须是**查过库之后**给出的。
        assert!(run.answer.contains("真实答案"));
        assert!(!run.answer.contains("编造的答案"));
        assert_eq!(run.executed_sql.len(), 1);
    }

    #[tokio::test]
    async fn the_query_requirement_can_be_switched_off() {
        let (agent, llm) = build(vec!["Final Answer: 直接作答".to_string()]);

        let mut config = agent.config().clone();
        config.agent.require_query = false;
        let agent = Agent::new(llm.clone(), agent.registry().clone(), config);

        let run = agent.run("问题").await.unwrap();

        assert_eq!(run.answer, "直接作答");
        assert!(run.executed_sql.is_empty());
    }

    #[tokio::test]
    async fn the_reject_loop_is_bounded() {
        // 模型反复「不查库就作答」→ 不能无限拉锯。
        let (agent, _llm) = build(vec![
            "Final Answer: 编造的答案 1".to_string(),
            "Final Answer: 编造的答案 2".to_string(),
            "Final Answer: 编造的答案 3".to_string(),
        ]);

        let run = agent.run("问题").await.unwrap();

        // 退回次数有上限（MAX_REJECT_ROUNDS），超出后放行。
        assert!(run.steps.len() <= MAX_REJECT_ROUNDS + 2);
    }

    #[tokio::test]
    async fn grounded_flag_reflects_whether_a_query_actually_ran() {
        // ⚠️ 「答案有没有数据支撑」必须是一个**可断言的布尔值**。
        //
        // 一个看起来专业的错误答案比明显的错误更危险 ——
        // 调用方很难发现它是编的。

        // 情形 1：查过库 → grounded = true。
        let (agent, _llm) = build(vec![
            "Action: sql_db_query\nAction Input: {\"query\": \"SELECT Name FROM Genre LIMIT 5\"}"
                .to_string(),
            "Final Answer: 有摇滚".to_string(),
        ]);
        let run = agent.run("有哪些流派").await.unwrap();
        assert!(run.grounded);
        assert_eq!(run.executed_sql.len(), 1);

        // 情形 2：没查库（且关了硬约束）→ grounded = false。
        let (agent, llm) = build(vec!["Final Answer: 直接作答".to_string()]);
        let mut config = agent.config().clone();
        config.agent.require_query = false;
        let agent = Agent::new(llm.clone(), agent.registry().clone(), config);

        let run = agent.run("问题").await.unwrap();
        assert!(!run.grounded);
        assert!(run.executed_sql.is_empty());
    }

    #[tokio::test]
    async fn a_blocked_query_still_counts_as_attempted() {
        // ⚠️ 「被护栏拦下」也算**尝试过查询** ——
        //    所以「数据库是只读的」这个回答**不该**被退回。
        let (agent, _llm) = build(vec![
            "Action: sql_db_query\nAction Input: {\"query\": \"DELETE FROM Track\"}".to_string(),
            "Final Answer: 数据库是只读的，我不能删".to_string(),
        ]);

        let run = agent.run("删掉所有曲目").await.unwrap();

        assert_eq!(run.steps.len(), 2);
        assert_eq!(run.answer, "数据库是只读的，我不能删");
        // ⚠️ 但护栏拦下的 SQL 没落地 → 不算 grounded。
        assert!(!run.grounded);
        assert_eq!(run.blocked_sql.len(), 1);
    }
}
