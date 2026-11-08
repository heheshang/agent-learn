//! ReAct 提示词 —— 论文 [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) 的核心。
//!
//! ## 论文的原始提示词格式
//!
//! ReAct 论文里 HotpotQA 任务用的提示词长这样：
//!
//! ```text
//! Answer the following questions as best you can. You have access to the following tools:
//!
//! {tool_descriptions}
//!
//! Use the following format:
//!
//! Question: the input question you must answer
//! Thought: you should always think about what to do
//! Action: the action to take, should be one of [{tool_names}]
//! Action Input: the input to the action
//! Observation: the result of the action
//! ... (this Thought/Action/Action Input/Observation can repeat N times)
//! Thought: I now know the final answer
//! Final Answer: the final answer to the original input question
//! ```
//!
//! ## 三个关键的措辞决定
//!
//! ### 1. `Thought: you should always think about what to do`
//!
//! 「always」这个词是刻意的 —— 它把「先想再做」变成**硬约束**，
//! 而不是「建议」。论文的消融实验证明：去掉 `Thought:` 后
//! 模型在 HotpotQA 上的表现明显下降（Act-only 基线）。
//!
//! ### 2. `Thought: I now know the final answer`
//!
//! 这一行是**收尾信号**。模型需要一个明确的「我该停了」的触发器。
//! 没有它，模型会在工具循环里反复转 —— 这正是 AutoGPT 最著名的失败模式。
//!
//! 本项目保留了这个设计，但**额外**加了 `Final Answer:` 作为终止标记，
//! 因为 `Final Answer:` 比「我知道了」更容易被程序可靠地解析。
//!
//! ### 3. `Action Input` 而不是 `Action Input:`
//!
//! 论文原版是 `Action Input: the input to the action`（带冒号）。
//! 本项目统一用 `Action Input:` 后直接跟值 ——
//! 因为值里可能包含冒号（URL、时间），带冒号的解析规则更脆。
//!
//! ## 本模块提供三种提示词变体
//!
//! | 变体 | `Thought:` | 工具 | 用途 |
//! |---|---|---|---|
//! | [`react_system_prompt`] | ✅ | ✅ | **默认**，对齐论文 |
//! | [`act_only_system_prompt`] | ❌ | ✅ | **对照组** —— 验证「思考」的贡献 |
//! | [`direct_system_prompt`] | ❌ | ❌ | **对照组** —— 验证「工具」的贡献 |
//!
//! 三者共用同一套 few-shot 示例结构和同一个终止标记，
//! 所以消融实验的**变量是单一的**。

use crate::config::AgentMode;
use crate::tools::{CALCULATE_TOOL_NAME, SEARCH_TOOL_NAME, ToolSpec};

/// 把工具定义渲染成提示词里的工具说明段落。
///
/// 格式对齐论文：`{name}: {description}`。
pub fn render_tool_descriptions(tools: &[ToolSpec]) -> String {
    tools
        .iter()
        .map(|t| format!("- {}: {}", t.name, t.description))
        .collect::<Vec<_>>()
        .join("\n")
}

/// 把工具名字列表渲染成 `Action:` 那一行的候选项。
pub fn render_tool_names(tools: &[ToolSpec]) -> String {
    tools
        .iter()
        .map(|t| t.name.clone())
        .collect::<Vec<_>>()
        .join(", ")
}

/// few-shot 示例（**ReAct 完整版**，带 `Thought:`）。
///
/// ## ⚠️ 示例的问题**故意不等于**常见测试题
///
/// 早期版本用的是「ReAct 框架里的 Observation 是什么？」——
/// 结果模型在第 2 步会说「用户重复了示例里的问题」，
/// 然后**照着示例的 Observation 编了一个不相关的答案**
/// （实测产出了「平均每轮调用 4 次工具」这种幻觉）。
///
/// 根因是：**示例的 Question 和真实 Question 长得太像**，
/// 模型分不清「下面这段 Observation 是我刚查到的」还是「示例里的」。
///
/// 现在的示例用的是**知识库里的另一个话题**（MRKL），
/// 和实际提问错开 —— 消除了这种混淆。
///
/// ## 为什么示例里要展示「工具没结果」的情况
///
/// 示例里刻意包含一次「检索无结果」，让模型知道：
/// 看到空的 `Observation` 时应该**换关键词再试**，而不是硬编一个答案。
/// 这直接对应论文里 ReAct 缓解「幻觉」和「错误传播」的机制。
pub const FEW_SHOT_REACT: &str = r#"Question: MRKL 系统里大语言模型扮演什么角色？
Thought: 用户问的是 MRKL 系统的架构，需要查一下本地知识库。用 "MRKL 路由器 专家模块" 作为关键词检索。
Action: search
Action Input: {"query": "MRKL 路由器 专家模块", "top_k": 2}
Observation: [1] # Agent 的三大组件
MRKL 系统把大语言模型当成**路由器**，把请求分发给「专家模块」——这些模块可以是神经网络模型，也可以是符号工具（计算器、汇率换算器、天气 API）。实验表明，当外部符号工具足够可靠时，「知道何时以及如何使用工具」比工具本身更关键。

[2] # 工具调用的可靠性问题
自然语言作为模型与外部组件之间的接口，可靠性有限：模型可能输出格式错误，偶尔会「反抗」指令。
Thought: 第一条片段已经给出了完整答案，可以直接作答了。
Final Answer: 在 MRKL 系统中，大语言模型扮演**路由器**的角色，负责把请求分发给各类专家模块（神经网络模型或符号工具）。

Question: (12 + 34) * 2 等于多少？
Thought: 这是一个需要精确数值的算术题。按提示词要求，涉及具体数值计算必须调用 calculate 工具，不要自己心算。
Action: calculate
Action Input: {"expression": "(12 + 34) * 2"}
Observation: (12 + 34) * 2 = 92
Thought: 工具返回了 92，我可以据此回答。
Final Answer: 92"#;

/// few-shot 示例（**Act-only 版**，去掉 `Thought:`）。
///
/// ## 为什么必须单独写一份
///
/// `ActOnly` 是**对照组** —— 用来复现 ReAct 论文的消融结论：
/// 「带 `Thought:` 的 ReAct 优于去掉 `Thought:` 的 Act-only 基线」。
///
/// 如果对照组也带着 `Thought:` 的示例，那「思考」的变量就不是单一的
/// —— 模型会从示例里学到 `Thought:` 的写法，消融实验就废了。
///
/// 所以这里的示例**结构完全相同，只是去掉了 `Thought:` 行**。
pub const FEW_SHOT_ACT: &str = r#"Question: MRKL 系统里大语言模型扮演什么角色？
Action: search
Action Input: {"query": "MRKL 路由器 专家模块", "top_k": 2}
Observation: [1] # Agent 的三大组件
MRKL 系统把大语言模型当成**路由器**，把请求分发给「专家模块」——这些模块可以是神经网络模型，也可以是符号工具（计算器、汇率换算器、天气 API）。实验表明，当外部符号工具足够可靠时，「知道何时以及如何使用工具」比工具本身更关键。

[2] # 工具调用的可靠性问题
自然语言作为模型与外部组件之间的接口，可靠性有限：模型可能输出格式错误，偶尔会「反抗」指令。
Final Answer: 在 MRKL 系统中，大语言模型扮演**路由器**的角色，负责把请求分发给各类专家模块（神经网络模型或符号工具）。

Question: (12 + 34) * 2 等于多少？
Action: calculate
Action Input: {"expression": "(12 + 34) * 2"}
Observation: (12 + 34) * 2 = 92
Final Answer: 92"#;

/// 组装 ReAct 循环的提示词（用户消息）。
///
/// 参数：
///
/// - `mode` —— 运行模式，决定用哪种格式
/// - `tools` —— 可用工具
/// - `question` —— 用户问题
/// - `trajectory` —— 已有的轨迹（`Thought/Action/Action Input/Observation` 拼接）
///
/// ## 为什么轨迹是**追加**而不是每次重发
///
/// 因为模型每一轮都会看到「到目前为止发生了什么」——
/// 这正是 ReAct 「交织」的本质：**思考和行动共享同一条上下文**。
/// 如果每轮只发最后一步，模型就失去了「我为什么调这个工具」的记忆。
pub fn build_react_prompt(
    mode: AgentMode,
    tools: &[ToolSpec],
    question: &str,
    trajectory: &str,
    few_shot_enabled: bool,
) -> String {
    let mut out = String::new();

    // 回答问题的格式说明。
    //
    // ⚠️ `Action / Action Input / Observation` 这三行**只在有工具时**输出。
    // `Direct` 模式没有工具 —— 如果还留着 `Action:` 这一行，
    // 模型就会去**编造**一个动作（提示词在暗示它可以行动）。
    out.push_str("请用下面的格式回答问题。\n\n");
    out.push_str("Question: 用户的问题\n");

    if mode.uses_tools() {
        if mode.uses_thought() {
            out.push_str("Thought: 你必须先思考下一步要做什么，然后再行动\n");
        }
        out.push_str("Action: 你要采取的行动\n");
        out.push_str("Action Input: 行动的参数（JSON 对象）\n");
        out.push_str("Observation: 行动的结果\n");
        out.push_str("（Thought / Action / Action Input / Observation 可以重复多次）\n");
    }
    out.push_str("Final Answer: 最终答案\n\n");

    // 可用工具。
    if mode.uses_tools() && !tools.is_empty() {
        out.push_str(&format!(
            "你可以使用这些工具：\n{}\n\n",
            render_tool_descriptions(tools)
        ));
        out.push_str(&format!(
            "`Action` 必须是下面这些工具之一：[{}]\n\n",
            render_tool_names(tools)
        ));
    } else {
        out.push_str("你没有可用的工具，请直接给出最终答案。\n\n");
    }

    // few-shot 示例。
    if few_shot_enabled && let Some(example) = few_shot_for(mode, true) {
        out.push_str("下面是几个示例：\n\n");
        out.push_str(example);
        out.push_str("\n\n");
    }

    // 实际问题 + 已有轨迹。
    out.push_str(&format!("Question: {question}\n"));
    if !trajectory.trim().is_empty() {
        out.push_str(trajectory.trim_end());
        out.push('\n');
    }
    out.push_str("现在请继续，按格式输出下一步。");

    out
}

/// 完整 ReAct 循环的 system 提示词。
///
/// ## 为什么 ReAct 用「user 消息」承载循环状态，而不是 system
///
/// 因为轨迹在**增长** —— 每一步都要追加。
/// 如果塞在 system 里，每轮都要重建 system 消息；
/// 而 `genai` 的 `ChatRequest` 里 system 是固定的一条。
///
/// 把「格式说明 + 示例」放 system（不变的部分），
/// 把「问题 + 轨迹」放 user（增长的部分）——
/// 这是 prompt 工程的常规做法，也和论文的
/// 「task-specific prompt + few-shot exemplars」结构一致。
pub fn react_system_prompt() -> String {
    concat!(
        "你是一个会使用工具的自主 Agent。\n",
        "你通过在「思考」和「行动」之间交替来解决问题：\n",
        "- 思考（Thought）让你规划、跟踪和更新行动计划，并处理异常；\n",
        "- 行动（Action）让你与外部信息源交互，拿到模型权重里没有的信息。\n",
        "\n",
        "工作守则：\n",
        "1. 每次只输出一个动作，然后等待 Observation。不要凭空猜测 Observation 的内容。\n",
        "2. 涉及具体数值计算时，必须调用 calculate 工具，不要自己心算。\n",
        "3. 需要外部事实（定义、机制、对比）时，调用 search 工具；查询词要用知识库里可能出现的词。\n",
        "4. 如果检索没有结果，换一组关键词再检索一次；最多换两次，之后就用已有信息作答。\n",
        "5. 信息足够时，以 `Final Answer:` 开头给出最终答案，并注明信息来源。\n",
    )
    .to_string()
}

/// Act-only 变体的 system 提示词（**对照组**：去掉思考）。
///
/// ## 为什么这个对照组有意义
///
/// ReAct 论文的核心结论之一是：
/// 「带 `Thought:` 的 ReAct 明显优于去掉 `Thought:` 的 Act-only 基线」。
///
/// 保留这个对照组，就能在自己的知识库上**复现**那个结论 ——
/// 而不是只在论文里读过。
pub fn act_only_system_prompt() -> String {
    concat!(
        "你是一个会使用工具的自主 Agent。\n",
        "你通过调用工具与外部信息源交互来解决问题。\n",
        "\n",
        "工作守则：\n",
        "1. 每次只输出一个动作，然后等待 Observation。不要凭空猜测 Observation 的内容。\n",
        "2. 涉及具体数值计算时，必须调用 calculate 工具。\n",
        "3. 需要外部事实时，调用 search 工具。\n",
        "4. 信息足够时，以 `Final Answer:` 开头给出最终答案。\n",
    )
    .to_string()
}

/// Direct 变体的 system 提示词（**对照组**：完全不调工具）。
///
/// 对应论文里的 **No-CoT / No-ReAct** 基线：
/// 直接问模型，让它凭参数里的知识回答。
pub fn direct_system_prompt() -> String {
    concat!(
        "你是一个知识问答助手。你没有可用的工具，请直接根据你已有的知识回答问题。\n",
        "如果问题涉及具体数值计算，你可以自己算。\n",
        "以 `Final Answer:` 开头给出最终答案。\n",
    )
    .to_string()
}

/// 按模式取 system 提示词。
pub fn system_prompt_for(mode: AgentMode) -> String {
    match mode {
        AgentMode::ReAct => react_system_prompt(),
        AgentMode::ActOnly => act_only_system_prompt(),
        AgentMode::Direct => direct_system_prompt(),
    }
}

/// 按模式取 few-shot 示例。
///
/// ## `Direct` 模式**不带示例**
///
/// 因为 few-shot 里全是「调工具」的轨迹 —— 给一个没有工具的模型看
/// 工具调用示例，只会诱导它**编造**工具调用。
pub fn few_shot_for(mode: AgentMode, enabled: bool) -> Option<&'static str> {
    if !enabled {
        return None;
    }
    match mode {
        AgentMode::ReAct => Some(FEW_SHOT_REACT),
        AgentMode::ActOnly => Some(FEW_SHOT_ACT),
        AgentMode::Direct => None,
    }
}

/// 组装完整的「system + user」提示词对。
///
/// 返回 `(system, user)`。
pub fn assemble(
    mode: AgentMode,
    tools: &[ToolSpec],
    question: &str,
    trajectory: &str,
    few_shot_enabled: bool,
) -> (String, String) {
    let system = system_prompt_for(mode);
    let user = build_react_prompt(mode, tools, question, trajectory, few_shot_enabled);

    (system, user)
}

/// 工具名字列表（给提示词用）。
pub fn default_tool_names() -> Vec<&'static str> {
    vec![SEARCH_TOOL_NAME, CALCULATE_TOOL_NAME]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{calculate_schema, search_schema};

    fn tools() -> Vec<ToolSpec> {
        vec![
            ToolSpec::new(
                SEARCH_TOOL_NAME,
                "在本地知识库中做关键词检索",
                search_schema(),
            ),
            ToolSpec::new(
                CALCULATE_TOOL_NAME,
                "计算一个算术表达式",
                calculate_schema(),
            ),
        ]
    }

    #[test]
    fn react_prompt_contains_the_four_action_markers() {
        let (_, user) = assemble(AgentMode::ReAct, &tools(), "问题", "", true);

        assert!(user.contains("Thought:"));
        assert!(user.contains("Action:"));
        assert!(user.contains("Action Input:"));
        assert!(user.contains("Observation:"));
        assert!(user.contains("Final Answer:"));
    }

    #[test]
    fn act_only_prompt_has_no_thought_marker() {
        let (_, user) = assemble(AgentMode::ActOnly, &tools(), "问题", "", true);

        assert!(!user.contains("Thought:"));
        assert!(user.contains("Action:"));
        assert!(user.contains("Final Answer:"));
    }

    #[test]
    fn direct_prompt_has_no_tools_and_no_action_marker() {
        let (_, user) = assemble(AgentMode::Direct, &tools(), "问题", "", false);

        assert!(!user.contains("Action:"));
        assert!(!user.contains("你可以使用这些工具"));
        assert!(user.contains("Final Answer:"));
    }

    #[test]
    fn prompt_lists_tool_names_in_action_candidates() {
        let (_, user) = assemble(AgentMode::ReAct, &tools(), "问题", "", true);
        assert!(user.contains("[search, calculate]"));
    }

    #[test]
    fn trajectory_is_appended_to_the_prompt() {
        let trajectory = "Thought: 需要查资料\nAction: search\nAction Input: {\"query\": \"ReAct\"}\nObservation: [1] ...";
        let (_, user) = assemble(AgentMode::ReAct, &tools(), "问题", trajectory, true);

        assert!(user.contains(trajectory));
        // 轨迹必须在问题**之后** —— 顺序反了模型会混淆。
        let q_pos = user.find("Question: 问题").unwrap();
        let t_pos = user.find("Thought: 需要查资料").unwrap();
        assert!(t_pos > q_pos);
    }

    #[test]
    fn few_shot_can_be_disabled() {
        let (_, with) = assemble(AgentMode::ReAct, &tools(), "问题", "", true);
        let (_, without) = assemble(AgentMode::ReAct, &tools(), "问题", "", false);

        assert!(with.contains("下面是几个示例"));
        assert!(!without.contains("下面是几个示例"));
        // 关掉示例后仍然要有格式说明。
        assert!(without.contains("Final Answer:"));
    }

    #[test]
    fn direct_mode_never_gets_tool_examples() {
        // Direct 模式没有工具 —— 给它看工具调用示例只会诱导模型编造调用。
        let (_, user) = assemble(AgentMode::Direct, &tools(), "问题", "", true);
        assert!(!user.contains("下面是几个示例"));
    }

    #[test]
    fn few_shot_trajectory_shows_the_full_loop() {
        // 示例必须演示完整循环，否则模型学不会「Observation 之后还要再 Thought」。
        assert!(FEW_SHOT_REACT.contains("Thought:"));
        assert!(FEW_SHOT_REACT.contains("Action: search"));
        assert!(FEW_SHOT_REACT.contains("Action Input:"));
        assert!(FEW_SHOT_REACT.contains("Observation:"));
        assert!(FEW_SHOT_REACT.contains("Final Answer:"));
    }

    #[test]
    fn act_few_shot_has_no_thought_marker() {
        // 对照组必须**结构相同、只少 Thought 行** ——
        // 否则「思考」这个变量就不是单一的。
        assert!(!FEW_SHOT_ACT.contains("Thought:"));
        assert!(FEW_SHOT_ACT.contains("Action: search"));
        assert!(FEW_SHOT_ACT.contains("Observation:"));
        assert!(FEW_SHOT_ACT.contains("Final Answer:"));
    }

    #[test]
    fn react_and_act_few_shot_have_the_same_number_of_examples() {
        // 两份示例的问题必须一致 —— 换了问题就不是同一个实验了。
        assert_eq!(
            FEW_SHOT_REACT.matches("Question:").count(),
            FEW_SHOT_ACT.matches("Question:").count()
        );
    }

    #[test]
    fn few_shot_questions_differ_from_the_common_test_question() {
        // ⚠️ 这是一条**回归测试**。
        //
        // 早期版本的示例问题就是「ReAct 框架里的 Observation 是什么？」，
        // 和 demo 的默认问题完全一样 —— 模型在第 2 步会说
        // 「用户重复了示例里的问题」，然后照着示例编一个不相关的答案。
        //
        // 所以示例的提问必须和真实提问**错开**。
        let common = "ReAct 框架里的 Observation 是什么？";
        assert!(!FEW_SHOT_REACT.contains(common));
        assert!(!FEW_SHOT_ACT.contains(common));
    }

    #[test]
    fn direct_mode_system_prompt_has_no_tool_mentions() {
        let system = system_prompt_for(AgentMode::Direct);
        assert!(!system.contains("search"));
        assert!(!system.contains("calculate"));
    }

    #[test]
    fn tool_descriptions_are_rendered_with_names() {
        let rendered = render_tool_descriptions(&tools());
        assert!(rendered.contains("search:"));
        assert!(rendered.contains("calculate:"));
    }
}
