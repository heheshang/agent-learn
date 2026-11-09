//! 提示词组装 —— 对齐 OpenAI 的 **GPT Best Practices**。
//!
//! ## 参考资料的明确建议
//!
//! OpenAI 的 [GPT Best Practices](https://platform.openai.com/docs/guides/gpt-best-practices)
//! 给出了一套结构化的写法，本项目全部采纳：
//!
//! | 建议 | 本项目的落实 |
//! |---|---|
//! | 用 Markdown 标题和 XML 标签划分逻辑边界 | `# Identity` / `# Instructions` / `# Tools` 三段式 |
//! | **Identity** —— 说明助手的角色与目标 | 「你是一个可以使用外部工具的助手」 |
//! | **Instructions** —— 何时调、调了做什么 | 明确列出「必须用工具，不要凭记忆」 |
//! | **Examples** —— 给例子 | 附上「查两个城市 → 并行两次调用」的示例 |
//! | 「描述工具的**目的**和**每个参数**（及其格式）以及**输出代表什么**」 | 每个工具的 description 都写清三件事 |
//! | 「**不要让模型填你已经知道的参数**」 | 见下方「参数归位」一节 |
//!
//! ## 一个关键差异：工具描述**不写进提示词**
//!
//! 13/14/15 的 ReAct 提示词里有一大段「工具说明」——
//! 因为工具定义只能以**文本**形式传给模型。
//!
//! 本项目不需要：工具定义在 API 的 `tools` 字段里，
//! 模型**结构化地**看得到它。
//!
//! 提示词只负责说**行为规则**（「必须用工具」这类），
//! 不负责说**工具长什么样**。
//!
//! ## 「不要让模型填你已经知道的参数」
//!
//! OpenAI 最佳实践里的一条：
//!
//! > Don't make the model fill arguments you already know.
//!
//! 本项目的落实：`calculate` 工具的 `expression` 是**唯一**由模型决定的参数；
//! 而「查哪个城市」这种如果上下文已知的，就不该做成参数。
//!
//! 另一个例子：`search_knowledge` 的 `top_k` 有默认值 ——
//! 模型**不需要**每次都填，Schema 的 `default` 会在本地校验前填好。

use crate::tools::ToolSpec;

/// 构造 system 提示词。
///
/// ## 为什么不把工具清单塞进提示词
///
/// 因为结构化协议下，工具定义已经在 `tools` 字段里了。
/// 再塞一遍是**重复消耗 token**，而且两份描述可能不一致
/// （提示词里那份一旦忘了改，模型就会按旧的理解调工具）。
///
/// 提示词只写**行为规则**。
pub fn system_prompt(specs: &[ToolSpec], strict: bool) -> String {
    let mut out = String::new();

    // ── Identity ─────────────────────────────────────────────
    out.push_str("# Identity\n\n");
    out.push_str("你是一个可以使用外部工具的助手。你能够调用工具来获取实时信息、查询本地知识库和进行精确计算。\n\n");

    // ── Instructions ─────────────────────────────────────────
    out.push_str("# Instructions\n\n");
    out.push_str("- 需要**实时或精确**的信息时，必须调用工具获取，不要凭记忆回答。\n");
    out.push_str("- 可以一次性调用多个工具，它们会并行执行。\n");
    out.push_str(
        "- 工具返回错误时，请阅读错误信息并修正参数后重新调用，不要放弃也不要编造答案。\n",
    );
    out.push_str("- 所有事实性数字必须来自工具返回的结果。\n");

    if strict {
        out.push_str("- 调用工具时必须严格遵守参数 Schema，所有必填字段都要填写。\n");
    }

    // ── Tools（只列名字 + 一句话用途）──────────────────────────
    //
    // ⚠️ 详细参数在 `tools` 字段里，这里**只列名字和用途**。
    //
    // 目的是让模型知道「有哪些工具、各自管什么」，
    // 而不重复 Schema 的内容（那部分模型已经从结构化字段读到了）。
    if !specs.is_empty() {
        out.push_str("\n# Tools\n\n");
        out.push_str("你可以调用以下工具：\n\n");
        for spec in specs {
            out.push_str(&format!(
                "- `{}`：{}\n",
                spec.name,
                first_line(&spec.description)
            ));
        }
        out.push_str("\n每个工具的参数格式、字段含义和输出定义见工具定义本身。调用前请确认参数符合 Schema。\n");
    }

    // ── Examples ─────────────────────────────────────────────
    //
    // OpenAI 最佳实践：
    // 「Include examples and edge cases, especially to rectify any recurring failures.」
    out.push_str("\n# Examples\n\n");
    out.push_str(
        "- 用户问「北京和上海现在天气怎么样」：分别调用 `get_weather` 两次（一次一个城市），\
         然后根据两个结果一起回答。\n",
    );
    out.push_str(
        "- 工具返回「错误：找不到城市 `Atlantis`」：说明参数有问题，\
         换一个真实的城市名重新调用，或向用户澄清。\n",
    );

    out
}

/// 取描述的第一行（用于工具清单）。
///
/// ## 为什么只取第一行
///
/// 14 的天气工具描述是**多行**的（用 `\` 续行拼接），
/// 直接塞进清单会很难看。
fn first_line(description: &str) -> String {
    let line = description
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_string();

    if line.is_empty() {
        description.trim().to_string()
    } else {
        line
    }
}

/// 构造用户消息。
pub fn user_message(question: &str) -> String {
    question.trim().to_string()
}

/// 渲染一次运行的**可读轨迹**（给 CLI / API 展示用）。
///
/// ## 与 13/14 的 `trajectory()` 有什么不同
///
/// 13/14 的轨迹是**提示词的镜像**（`Thought: / Action: / Observation:`）——
/// 因为那些协议下模型看到的就是这些文本。
///
/// 本项目的模型看到的是**结构化消息**，所以轨迹应该展示
/// **结构化流程**：每步调了哪些工具、参数是什么、结果是什么。
pub fn render_trajectory(steps: &[crate::agent::StepRecord]) -> String {
    let mut out = String::new();

    for step in steps {
        out.push_str(&format!("── 第 {} 步 ──\n", step.step));

        if let Some(reasoning) = &step.reasoning
            && !reasoning.is_empty()
        {
            out.push_str(&format!("思考: {reasoning}\n"));
        }

        if !step.text.is_empty() {
            out.push_str(&format!("文本: {}\n", step.text));
        }

        for call in &step.tool_calls {
            out.push_str(&format!(
                "工具调用: {}({}) [{}]\n",
                call.name,
                serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
                call.call_id
            ));
            out.push_str(&format!(
                "结果{}: {}\n",
                if call.ok { "" } else { "（失败）" },
                call.observation
            ));
        }

        if step.final_answer {
            out.push_str("→ 最终答案\n");
        }

        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn specs() -> Vec<ToolSpec> {
        vec![
            ToolSpec::new(
                "get_weather",
                "查询某个城市的实时天气。\n需要实时天气时必须调用。",
                json!({ "type": "object" }),
            ),
            ToolSpec::new("calculate", "计算算术表达式。", json!({ "type": "object" })),
        ]
    }

    #[test]
    fn system_prompt_has_the_three_sections() {
        let prompt = system_prompt(&specs(), false);

        assert!(prompt.contains("# Identity"));
        assert!(prompt.contains("# Instructions"));
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("# Examples"));
    }

    #[test]
    fn system_prompt_lists_tool_names() {
        let prompt = system_prompt(&specs(), false);

        assert!(prompt.contains("`get_weather`"));
        assert!(prompt.contains("`calculate`"));
    }

    #[test]
    fn system_prompt_takes_only_the_first_line_of_the_description() {
        let prompt = system_prompt(&specs(), false);

        // 第一行在清单里。
        assert!(prompt.contains("查询某个城市的实时天气。"));
        // 第二行**不该**出现在清单里。
        assert!(!prompt.contains("需要实时天气时必须调用。"));
    }

    #[test]
    fn strict_flag_adds_the_schema_instruction() {
        let prompt = system_prompt(&specs(), true);
        assert!(prompt.contains("严格遵守参数 Schema"));
    }

    #[test]
    fn non_strict_prompt_has_no_schema_instruction() {
        let prompt = system_prompt(&specs(), false);
        assert!(!prompt.contains("严格遵守参数 Schema"));
    }

    #[test]
    fn system_prompt_without_tools_has_no_tools_section() {
        let prompt = system_prompt(&[], false);
        assert!(!prompt.contains("# Tools"));
    }

    #[test]
    fn system_prompt_states_that_errors_are_recoverable() {
        let prompt = system_prompt(&specs(), false);
        // 这一条是本项目的核心行为约定 —— 必须写进提示词。
        assert!(prompt.contains("修正参数后重新调用"));
    }

    #[test]
    fn user_message_is_trimmed() {
        assert_eq!(user_message("  你好  "), "你好");
    }

    #[test]
    fn renders_trajectory() {
        let steps = vec![crate::agent::StepRecord {
            step: 1,
            text: String::new(),
            reasoning: None,
            tool_calls: vec![crate::agent::ToolCallRecord {
                call_id: "c1".to_string(),
                name: "get_weather".to_string(),
                arguments: json!({ "city": "北京" }),
                observation: "北京 25C".to_string(),
                ok: true,
                data: None,
            }],
            final_answer: false,
        }];

        let text = render_trajectory(&steps);
        assert!(text.contains("第 1 步"));
        assert!(text.contains("get_weather"));
        assert!(text.contains("北京 25C"));
    }
}
