//! 提示词 —— 教模型**怎么配合并发**。
//!
//! ## 为什么本项目要自己写提示词
//!
//! 16/18 的提示词是针对「串行工具执行」写的 ——
//! 模型每次只需要处理**一个**工具结果。
//!
//! 本项目一轮里可能有**多个**工具结果同时回来，
//! 所以提示词要额外说清两件事：
//!
//! 1. **多个工具结果是并行的** —— 它们之间没有先后依赖
//! 2. **每个 `tool_call_id` 都有一条对应结果** —— 不要漏读
//!
//! ## ⚠️ 沿用 16 的教训：不重复工具的详细参数
//!
//! 16 实测：工具的参数说明在 `tools` 字段里，模型结构化看得到。
//! 提示词里再写一遍 = 两份描述不一致 = 模型拉锯战。
//! 所以这里只写**行为规则**。

/// Agent 的 system prompt。
pub fn system_prompt() -> String {
    let text = "你是一个会使用工具的助手。

## 工具使用规则

1. 需要实时数据（天气、新闻、价格）时**必须**调用工具，不要凭记忆回答。
2. 需要精确数字时**必须**调用 `calculate`，不要自己心算。
3. 工具的参数说明在工具定义里，严格按定义传参。
4. 工具返回错误时，读懂错误提示并修正参数后重试；如果错误与你的用法无关，就如实告知用户，不要编造数据。

## ⚠️ 并行工具结果怎么读

你在**同一轮**里请求的多个工具会被**并行执行**。它们之间**没有先后依赖**（例如查两个城市的天气互不依赖），所以你会在同一轮里收到**多条**工具结果。

- 每条工具结果都对应你发出的一个工具调用，请**全部读完**，不要只看第一条。
- 汇总这些结果后再给出最终答案，不要只回答其中一个。
- 如果某个工具失败了，在答案里如实说明哪一项没查到。

## 最终答案

当你不再需要调用工具时，直接给出完整答案。答案里不要出现「我要去查一下」这类未完成的话。";

    text.to_string()
}

/// 工具清单（给 API 层 / CLI 用）。
pub fn tools_summary(specs: &[agent_16::tools::ToolSpec]) -> String {
    let mut out = String::new();

    for spec in specs {
        let required = spec
            .schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| "无".to_string());

        out.push_str(&format!(
            "- {}（必填参数：{}）\n  {}\n",
            spec.name, required, spec.description
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn prompt_mentions_parallel_tool_results() {
        let prompt = system_prompt();
        // ⚠️ 并发是本项目的核心，必须在提示词里讲清楚。
        assert!(prompt.contains("并行"));
        assert!(prompt.contains("多条"));
    }

    #[test]
    fn prompt_does_not_repeat_tool_parameters() {
        let prompt = system_prompt();
        // ⚠️ 16 的教训：参数说明在 tools 字段里，提示词不重复。
        assert!(!prompt.contains("additionalProperties"));
        assert!(!prompt.contains("\"type\": \"object\""));
    }

    #[test]
    fn tools_summary_lists_names_and_required_params() {
        let specs = vec![agent_16::tools::ToolSpec::new(
            "get_weather",
            "查天气",
            json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }),
        )];

        let summary = tools_summary(&specs);

        assert!(summary.contains("get_weather"));
        assert!(summary.contains("location"));
    }
}
