//! 提示词组装 —— 工具契约的「说明书」。
//!
//! ## 工具描述是**模型行为的一部分**，不是文档
//!
//! LangChain 教程里那句警告值得再强调一次：
//!
//! > 工具描述写得不好时，模型会反复调同一个工具。
//!
//! 这意味着提示词里渲染的工具说明**必须精确**。
//! 本模块的渲染分三块：
//!
//! ```text
//! 你可以使用这些工具：
//!
//! - get_weather: 查询某个城市的实时天气和未来几天的预报。…
//!   参数：
//!   - location (string, 必填): 城市名，例如 "北京"
//!   - units (string, 选填, 默认 celsius): 温度单位
//! ```
//!
//! ## 为什么参数也要渲染进去
//!
//! LangChain 里模型是通过 **tool schema**（结构化）知道参数的。
//! 但 ReAct 文本协议没有这个通道 —— 参数只能**写在提示词里**。
//!
//! 所以本项目把 Schema 渲染成可读文本（[`crate::schema::render_schema`]），
//! 让模型在文本协议下也能正确填参数。
//!
//! ## 与 13-agent-core 的差异
//!
//! | | 13-agent-core | 14-agent-tool |
//! |---|---|---|
//! | 工具说明 | `- name: description` | 加上参数明细 |
//! | 工具来源 | 固定两个（BM25 + 算术） | 本地 + MCP，运行时决定 |
//! | 工具失败 | 错误文本当观察 | 同上（复用注册表） |
//! | 直通工具 | 无 | 支持 `return_direct` |

use crate::tool::ToolSpec;

/// 渲染工具说明段落（含参数明细）。
pub fn render_tool_descriptions(tools: &[ToolSpec]) -> String {
    let mut out = String::new();

    for tool in tools {
        out.push_str(&format!("- {}: {}\n", tool.name, tool.description));

        // 参数明细 —— 只在有参数时才渲染。
        let params = tool.render_parameters();
        if !params.is_empty() {
            out.push_str("  参数：\n");
            for line in params.lines() {
                out.push_str(&format!("    {line}\n"));
            }
        }

        // 直通工具要明确说 —— 否则模型还会多问一句。
        if tool.return_direct {
            out.push_str("  （这个工具的结果就是最终答案，不需要再加工）\n");
        }
    }

    out.trim_end().to_string()
}

/// 渲染工具名字列表（`Action:` 那一行的候选项）。
pub fn render_tool_names(tools: &[ToolSpec]) -> String {
    tools
        .iter()
        .map(|t| t.name.clone())
        .collect::<Vec<_>>()
        .join(", ")
}

/// 组装 Agent 循环的提示词（用户消息）。
///
/// 参数：
///
/// - `mode` —— 运行模式（决定有没有 `Thought:`、有没有工具）
/// - `tools` —— 可用工具
/// - `question` —— 用户问题
/// - `trajectory` —— 已有的轨迹（`Thought/Action/Action Input/Observation` 拼接）
///
/// ## 为什么轨迹是**追加**而不是每次重发
///
/// 因为模型每一轮都会看到「到目前为止发生了什么」——
/// 这正是 ReAct「交织」的本质：**思考和行动共享同一条上下文**。
pub fn build_prompt(
    mode: crate::config::ToolMode,
    tools: &[ToolSpec],
    question: &str,
    trajectory: &str,
    few_shot: bool,
) -> String {
    let mut out = String::new();

    // ── 格式说明 ──────────────────────────────────────────────
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

    // ── 可用工具 ──────────────────────────────────────────────
    if mode.uses_tools() && !tools.is_empty() {
        out.push_str("你可以使用这些工具：\n\n");
        out.push_str(&render_tool_descriptions(tools));
        out.push_str("\n\n");
        out.push_str(&format!(
            "`Action` 必须是下面这些工具之一：[{}]\n\n",
            render_tool_names(tools)
        ));
        out.push_str("每次只调用一个工具，然后根据 Observation 决定下一步。\n\n");
    } else {
        out.push_str("你没有可用的工具，请直接给出最终答案。\n\n");
    }

    // ── few-shot 示例 ─────────────────────────────────────────
    //
    // ⚠️ `Direct` 模式**不带示例** —— few-shot 里全是「调工具」的轨迹，
    // 给一个没有工具的模型看，只会诱导它**编造**工具调用。
    if few_shot && mode.uses_tools() {
        out.push_str(
            "下面是几个**示例**，它们演示工具怎么用（注意：示例里的城市和问题都不是真的）：\n\n",
        );
        out.push_str(few_shot_for(mode));
        out.push_str("\n\n");
    }

    // ── 真实问题 + 已有轨迹 ───────────────────────────────────
    //
    // ⚠️⚠️ **这一段必须和示例明显区分开** —— 这是本项目实测踩到的真坑。
    //
    // 早期版本这里直接写 `Question: {question}`，和示例的格式**一模一样**。
    // 结果模型在第 2 步把上一轮自己写的
    // `Question: 北京... / Thought / Action / Action Input / Observation`
    // 误认成了**第三个示例**，于是「顺着示例的格式」自己编了一个
    // `Question: 深圳未来三天天气怎么样？` 去调工具 ——
    // 深圳从没出现在提示词里，模型的答案是彻底跑偏的。
    //
    // 修法有两条，缺一不可：
    //
    // 1. **显式声明**「下面这段不是示例，是真实对话」
    // 2. **把轨迹标注成「已完成的步骤」** —— 让模型知道它是**历史**，
    //    不是可以照着续写的模板
    out.push_str("========== 以下是真实对话（不是示例）==========\n\n");
    out.push_str(&format!("真实问题: {question}\n"));
    if !trajectory.trim().is_empty() {
        out.push_str("\n已完成的步骤：\n");
        out.push_str(trajectory.trim_end());
        out.push('\n');
    }
    out.push_str("\n现在请基于以上真实对话继续，按格式输出下一步。");

    out
}

/// system 提示词（按模式取）。
///
/// ## 为什么每个模式的 system 提示词都不同
///
/// 消融实验要求**变量是单一的**。
///
/// - `Direct` 里如果提到 `get_weather`，模型可能直接调它 —— 而它根本没有工具
/// - `ActOnly` 里如果留着「你必须先思考」，就等于给了对照组「思考」的能力
///
/// 所以三个提示词各自独立。
pub fn system_prompt_for(mode: crate::config::ToolMode) -> String {
    match mode {
        crate::config::ToolMode::ReAct => react_system_prompt(),
        crate::config::ToolMode::ActOnly => act_only_system_prompt(),
        crate::config::ToolMode::Direct => direct_system_prompt(),
    }
}

/// ReAct 变体的 system 提示词。
pub fn react_system_prompt() -> String {
    concat!(
        "你是一个会使用工具的自主 Agent。\n",
        "你通过在「思考」和「行动」之间交替来解决问题：\n",
        "- 思考（Thought）让你规划、跟踪和更新行动计划；\n",
        "- 行动（Action）让你与外部信息源交互，拿到模型权重里没有的信息。\n",
        "\n",
        "工作守则：\n",
        "1. 每次只输出一个动作，然后等待 Observation。不要凭空猜测 Observation 的内容。\n",
        "2. 涉及实时信息（天气、行情、新闻）时，必须调用对应工具，不要凭记忆回答。\n",
        "3. 如果 Observation 里出现「错误」，说明参数填错了 —— 读懂错误信息后修正参数再试一次。\n",
        "4. 工具返回的结果就是事实依据，回答时要忠于 Observation 里的数值，不要自行改写或推算。\n",
        "5. 信息足够时，以 `Final Answer:` 开头给出最终答案。\n",
    )
    .to_string()
}

/// Act-only 变体的 system 提示词（**对照组**：去掉思考）。
pub fn act_only_system_prompt() -> String {
    concat!(
        "你是一个会使用工具的自主 Agent。\n",
        "你通过调用工具与外部信息源交互来解决问题。\n",
        "\n",
        "工作守则：\n",
        "1. 每次只输出一个动作，然后等待 Observation。不要凭空猜测 Observation 的内容。\n",
        "2. 涉及实时信息（天气、行情、新闻）时，必须调用对应工具。\n",
        "3. 如果 Observation 里出现「错误」，读懂错误信息后修正参数再试一次。\n",
        "4. 回答时要忠于 Observation 里的数值。\n",
        "5. 信息足够时，以 `Final Answer:` 开头给出最终答案。\n",
    )
    .to_string()
}

/// Direct 变体的 system 提示词（**对照组**：完全不调工具）。
pub fn direct_system_prompt() -> String {
    concat!(
        "你是一个知识问答助手。你没有可用的工具，请直接根据你已有的知识回答问题。\n",
        "以 `Final Answer:` 开头给出最终答案。\n",
    )
    .to_string()
}

/// few-shot 示例（**ReAct 完整版**，带 `Thought:`）。
///
/// ## ⚠️ 示例里的问题必须和真实提问**错开**
///
/// 这是 13-agent-core 实测踩到的坑：
/// 示例的 Question 和真实 Question 长得太像时，
/// 模型会分不清「我刚查到的 Observation」和「示例里的」，
/// 于是照着示例编一个不相关的答案。
///
/// 所以这里的示例用「上海」和「一个算术表达式」，
/// 而真实提问通常问的是「北京」。
pub const FEW_SHOT_REACT: &str = r#"Question: 上海现在的天气怎么样？
Thought: 用户问的是上海的实时天气，这需要调用 get_weather 工具查询，不能凭记忆回答。location 参数填 "上海"。
Action: get_weather
Action Input: {"location": "上海", "units": "celsius", "days": 1}
Observation: 上海, 上海市, 中国 当前实况：多云，21°C，体感 20°C，相对湿度: 68%，风速: 11.2 km/h，观测时间: 2026-11-08T10:00
Thought: 工具已经返回了上海的实时天气，数据充分，可以直接作答。
Final Answer: 上海现在多云，气温 21 摄氏度（体感 20 摄氏度），相对湿度 68%，风速约 11.2 km/h。

Question: (12 + 34) * 2 等于多少？
Thought: 这是一个需要精确数值的算术题。必须调用 calculate 工具，不要自己心算。
Action: calculate
Action Input: {"expression": "(12 + 34) * 2"}
Observation: (12 + 34) * 2 = 92
Thought: 工具返回了 92，我可以据此回答。
Final Answer: 92"#;

/// few-shot 示例（**Act-only 版**，去掉 `Thought:`）。
///
/// ## 为什么必须单独写一份
///
/// `ActOnly` 是**对照组** —— 用来验证「思考」的贡献。
///
/// 如果对照组也带着 `Thought:` 的示例，那「思考」的变量就不是单一的
/// —— 模型会从示例里学到 `Thought:` 的写法，消融实验就废了。
pub const FEW_SHOT_ACT: &str = r#"Question: 上海现在的天气怎么样？
Action: get_weather
Action Input: {"location": "上海", "units": "celsius", "days": 1}
Observation: 上海, 上海市, 中国 当前实况：多云，21°C，体感 20°C，相对湿度: 68%，风速: 11.2 km/h，观测时间: 2026-11-08T10:00
Final Answer: 上海现在多云，气温 21 摄氏度（体感 20 摄氏度），相对湿度 68%，风速约 11.2 km/h。

Question: (12 + 34) * 2 等于多少？
Action: calculate
Action Input: {"expression": "(12 + 34) * 2"}
Observation: (12 + 34) * 2 = 92
Final Answer: 92"#;

/// 按模式取 few-shot 示例。
///
/// `Direct` 模式**不带示例** —— 给一个没有工具的模型看
/// 工具调用示例，只会诱导它编造工具调用。
pub fn few_shot_for(mode: crate::config::ToolMode) -> &'static str {
    match mode {
        crate::config::ToolMode::ReAct => FEW_SHOT_REACT,
        crate::config::ToolMode::ActOnly => FEW_SHOT_ACT,
        crate::config::ToolMode::Direct => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ToolMode;
    use crate::weather::weather_schema;
    use serde_json::json;

    fn tools() -> Vec<ToolSpec> {
        vec![
            ToolSpec::new("get_weather", "查询某个城市的实时天气。", weather_schema()),
            ToolSpec::new(
                "calculate",
                "计算算术表达式。",
                json!({
                    "type": "object",
                    "properties": {
                        "expression": { "type": "string", "description": "算术表达式" }
                    },
                    "required": ["expression"]
                }),
            ),
        ]
    }

    #[test]
    fn prompt_lists_tool_names_in_action_candidates() {
        let prompt = build_prompt(ToolMode::ReAct, &tools(), "问题", "", true);
        assert!(prompt.contains("[get_weather, calculate]"));
    }

    #[test]
    fn prompt_renders_parameter_details() {
        // ⚠️ 关键：文本协议下模型只能从提示词里读参数定义。
        let prompt = build_prompt(ToolMode::ReAct, &tools(), "问题", "", true);

        assert!(prompt.contains("location (string, 必填)"));
        assert!(prompt.contains("城市名"));
        assert!(prompt.contains("默认 celsius"));
        assert!(prompt.contains("expression (string, 必填)"));
    }

    #[test]
    fn prompt_marks_return_direct_tools() {
        let mut tools = tools();
        tools[0] = tools[0].clone().with_return_direct();
        let prompt = build_prompt(ToolMode::ReAct, &tools, "问题", "", true);

        assert!(prompt.contains("结果就是最终答案"));
    }

    #[test]
    fn direct_mode_has_no_action_marker_and_no_tools() {
        // ⚠️ `Direct` 模式如果还留着 `Action:` 那一行，
        //    模型就会去**编造**一个动作。
        let prompt = build_prompt(ToolMode::Direct, &tools(), "问题", "", true);

        assert!(!prompt.contains("Action:"));
        assert!(prompt.contains("你没有可用的工具"));
        assert!(prompt.contains("Final Answer:"));
    }

    #[test]
    fn direct_mode_never_gets_tool_examples() {
        // 给一个没有工具的模型看工具调用示例，只会诱导它编造调用。
        let prompt = build_prompt(ToolMode::Direct, &tools(), "问题", "", true);
        assert!(!prompt.contains("下面是几个示例"));
    }

    #[test]
    fn act_only_mode_has_no_thought_marker() {
        let prompt = build_prompt(ToolMode::ActOnly, &tools(), "问题", "", true);

        assert!(!prompt.contains("Thought:"));
        assert!(prompt.contains("Action:"));
        assert!(prompt.contains("Final Answer:"));
    }

    #[test]
    fn trajectory_is_appended_after_the_question() {
        let trajectory = "Thought: 查天气\nAction: get_weather\nAction Input: {\"location\": \"上海\"}\nObservation: 21°C";
        let prompt = build_prompt(ToolMode::ReAct, &tools(), "问题", trajectory, true);

        let q_pos = prompt.find("真实问题: 问题").unwrap();
        let t_pos = prompt.find("Thought: 查天气").unwrap();
        assert!(t_pos > q_pos, "轨迹必须在问题之后");
    }

    #[test]
    fn real_dialogue_is_clearly_separated_from_examples() {
        // ⚠️ **回归测试**（实测踩过的真坑）。
        //
        // 早期版本真实问题写成 `Question: {question}`，
        // 和 few-shot 示例的格式**一模一样**。
        //
        // 结果模型在第 2 步把上一轮自己写的轨迹误认成了**第三个示例**，
        // 于是「顺着示例的格式」自己编了一个
        // `Question: 深圳未来三天天气怎么样？` 去调工具 ——
        // 深圳从没出现在提示词里，答案彻底跑偏。
        //
        // 所以真实对话必须**显式**和示例区分开。
        let prompt = build_prompt(ToolMode::ReAct, &tools(), "北京现在天气怎么样？", "", true);

        // 1. 真实对话要有明确的分隔标记。
        assert!(prompt.contains("以下是真实对话（不是示例）"));

        // 2. 真实问题的标签**不能**是 `Question:` —— 那是示例的格式。
        assert!(prompt.contains("真实问题: 北京现在天气怎么样？"));
        assert!(!prompt.contains("Question: 北京现在天气怎么样？"));

        // 3. 示例段落要在真实对话**之前**，且明确标注是示例。
        let example_pos = prompt.find("下面是几个**示例**").unwrap();
        let real_pos = prompt.find("以下是真实对话（不是示例）").unwrap();
        assert!(example_pos < real_pos);
    }

    #[test]
    fn trajectory_is_labelled_as_completed_steps() {
        // ⚠️ 轨迹必须被标注成「已完成的步骤」——
        //    让模型知道它是**历史**，不是可以照着续写的模板。
        let trajectory = "Thought: 查天气\nAction: get_weather\nAction Input: {\"location\": \"北京\"}\nObservation: 21°C";
        let prompt = build_prompt(ToolMode::ReAct, &tools(), "北京天气", trajectory, true);

        assert!(prompt.contains("已完成的步骤："));
    }

    #[test]
    fn few_shot_examples_do_not_look_like_the_real_question() {
        // ⚠️ 回归测试：示例里的城市必须和真实提问**错开**。
        //
        // 否则模型会分不清「我刚查到的 Observation」和「示例里的」。
        let prompt = build_prompt(
            ToolMode::ReAct,
            &tools(),
            "北京现在天气怎么样？",
            "Thought: 查\nAction: get_weather\nAction Input: {\"location\": \"北京\"}\nObservation: 21°C",
            true,
        );

        // 示例里的城市是上海 / 算术题，不该出现北京。
        // ⚠️ 但「北京」会出现在**真实问题**和**轨迹**里 ——
        //    所以这里只断言示例段落本身不含北京。
        let example_section = prompt
            .split("下面是几个**示例**")
            .nth(1)
            .unwrap()
            .split("以下是真实对话")
            .next()
            .unwrap();

        assert!(!example_section.contains("北京"));
        assert!(example_section.contains("上海"));
    }

    #[test]
    fn system_prompt_tells_model_to_retry_on_error() {
        let system = system_prompt_for(ToolMode::ReAct);
        assert!(system.contains("修正参数再试一次"));
        assert!(system.contains("忠于 Observation"));
    }

    #[test]
    fn direct_system_prompt_has_no_tool_mentions() {
        let system = system_prompt_for(ToolMode::Direct);
        // ⚠️ 不能提到任何**具体工具名** —— 否则模型会去编造调用。
        assert!(!system.contains("get_weather"));
        assert!(!system.contains("calculate"));
    }

    #[test]
    fn few_shot_shows_a_complete_tool_loop() {
        assert!(FEW_SHOT_REACT.contains("Action: get_weather"));
        assert!(FEW_SHOT_REACT.contains("Action Input:"));
        assert!(FEW_SHOT_REACT.contains("Observation:"));
        assert!(FEW_SHOT_REACT.contains("Final Answer:"));
        assert!(FEW_SHOT_REACT.contains("Thought:"));
    }

    #[test]
    fn act_few_shot_has_no_thought_marker() {
        // ⚠️ 对照组必须**结构相同、只少 Thought 行** ——
        //    否则「思考」这个变量就不是单一的。
        assert!(!FEW_SHOT_ACT.contains("Thought:"));
        assert!(FEW_SHOT_ACT.contains("Action: get_weather"));
        assert!(FEW_SHOT_ACT.contains("Observation:"));
        assert!(FEW_SHOT_ACT.contains("Final Answer:"));
    }

    #[test]
    fn react_and_act_few_shot_have_the_same_number_of_examples() {
        assert_eq!(
            FEW_SHOT_REACT.matches("Question:").count(),
            FEW_SHOT_ACT.matches("Question:").count()
        );
    }

    #[test]
    fn few_shot_question_differs_from_the_common_one() {
        // ⚠️ 回归测试：示例问题不能和真实提问撞车。
        let common = "北京现在天气怎么样？";
        assert!(!FEW_SHOT_REACT.contains(common));
        assert!(!FEW_SHOT_ACT.contains(common));
    }
}
