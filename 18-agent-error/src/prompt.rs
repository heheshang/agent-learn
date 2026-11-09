//! 提示词组装 —— 教模型**怎么读错误**。
//!
//! ## 为什么这个项目的提示词比别的项目长
//!
//! 因为模型会读到**工具的错误文本**（降级后的观察）。
//! 如果不告诉它「错误分三类」，它会对所有错误一视同仁：
//!
//! - 看到「上游暂时不可用」→ 可能当成「工具坏了」，转而编答案
//! - 看到「参数不合法」→ 知道要改参数（这个不需要教）
//!
//! 所以提示词里要明确写：
//!
//! 1. **错误文本会带一个 `[E_XXX]` 码** —— 那是给你判断用的
//! 2. **遇到 `[E_UPSTREAM_*]` 应该是临时故障** —— 稍后重试或如实告知用户
//! 3. **遇到 `[E_INVALID_ARGUMENTS]` 就改参数重试** —— 不要放弃
//! 4. **任何情况下都不要编造工具没有返回的数据**

use crate::tools::ToolSpec;

/// 构造 system 提示词。
pub fn system_prompt() -> String {
    let mut out = String::new();

    out.push_str("# Identity\n\n");
    out.push_str(
        "你是一个可以使用外部工具的助手。你能够调用工具获取数据、执行查询和进行支付操作。\n\n",
    );

    out.push_str("# Instructions\n\n");
    out.push_str("- 需要外部数据时必须调用工具，不要凭记忆编造。\n");
    out.push_str("- 所有事实性数字必须来自工具返回的结果。\n");
    out.push_str(
        "- 工具失败时，工具结果里会带一个 `[E_XXX]` 错误码和一句建议，\
         请根据它决定下一步。\n",
    );
    out.push_str(
        "  · `[E_INVALID_ARGUMENTS]` / `[E_UNKNOWN_TOOL]` / `[E_NOT_FOUND]`：\
         **你的参数或目标有问题** —— 按提示修正后重新调用。\n",
    );
    out.push_str(
        "  · `[E_UPSTREAM_TIMEOUT]` / `[E_UPSTREAM_UNAVAILABLE]` / `[E_RATE_LIMITED]`：\
         **外部服务的临时故障** —— 可以换个工具、稍后再试，或者如实告诉用户服务暂时不可用。\n",
    );
    out.push_str(
        "  · `[E_UNAUTHORIZED]`：**环境没配好** —— 不要重试，\
         直接如实告诉用户「服务未配置 / 无权限」，**不要**编造一个看起来合理的结果。\n",
    );
    out.push_str("- 无论工具成功还是失败，都**不要**编造工具没有返回的数据。\n");

    out.push_str("\n# Tools\n\n");
    out.push_str("你可以调用以下工具：\n\n");
    out.push_str("- `flaky_fetch`：从不稳定的上游拉取数据。可能会临时失败，重试可能成功。\n");
    out.push_str("- `strict_lookup`：按 id 查询记录。id 不存在或格式不对会明确报错。\n");
    out.push_str("- `guarded_pay`：发起支付。需要 API key，未配置会直接失败。\n");
    out.push_str(
        "\n每个工具的参数格式、字段含义和输出定义见工具定义本身。调用前请确认参数符合 Schema。\n",
    );

    out.push_str("\n# Examples\n\n");
    out.push_str(
        "- 用户问「拉一下 report/2026」：调用 `flaky_fetch`，\
         如果返回 `[E_UPSTREAM_UNAVAILABLE]`，等待后重试一次；\
         连续失败就告诉用户「上游服务暂时不可用，请稍后再试」。\n",
    );
    out.push_str(
        "- 用户问「查 rec-9」：调用 `strict_lookup`；\
         如果返回 `[E_NOT_FOUND]` 且提示里列出了可用 id，\
         说明用户可能记错了 id —— 可以用列表里的 id 重查一次，\
         或向用户确认。\n",
    );
    out.push_str(
        "- 用户说「付 100 块」：调用 `guarded_pay`；\
         如果返回 `[E_UNAUTHORIZED]`，**不要重试**，\
         直接告诉用户支付服务未配置。\n",
    );

    out
}

/// 构造带工具清单的提示词（给 API 层展示用）。
///
/// ## 与 [`system_prompt`] 的区别
///
/// 这里的版本会把**真实注册表**里的工具列出来，
/// 而 `system_prompt()` 是写死的 —— 因为模型看到的工具定义
/// 已经在 API 的 `tools` 字段里了，提示词里只需要**行为规则**。
pub fn render_tool_catalog(specs: &[ToolSpec]) -> String {
    let mut out = String::new();

    for spec in specs {
        out.push_str(&format!(
            "■ {}（strict={}）\n  {}\n  参数:\n",
            spec.name,
            spec.strict,
            spec.description.replace('\n', " ")
        ));
        for line in spec.render_parameters().lines() {
            out.push_str(&format!("    {line}\n"));
        }
        out.push('\n');
    }

    out
}

/// 渲染一次运行的**可读轨迹**（给 CLI / API 展示用）。
pub fn render_trajectory(steps: &[crate::agent::StepRecord]) -> String {
    let mut out = String::new();

    for step in steps {
        out.push_str(&format!("── 第 {} 步 ──\n", step.step));

        if !step.text.is_empty() {
            out.push_str(&format!("文本: {}\n", step.text));
        }

        for call in &step.tool_calls {
            let retry_mark = if call.retried {
                format!("（重试 {} 次）", call.attempts)
            } else {
                String::new()
            };
            out.push_str(&format!(
                "工具调用: {}({}) [{}]{}\n",
                call.name,
                serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
                call.call_id,
                retry_mark
            ));

            match (&call.error_code, &call.error_kind) {
                (Some(code), Some(kind)) => out.push_str(&format!(
                    "结果: 失败 [{code} / {kind}] → {}\n",
                    call.observation
                )),
                _ => out.push_str(&format!("结果: {}\n", call.observation)),
            }
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

    #[test]
    fn prompt_explains_the_three_error_classes() {
        let prompt = system_prompt();

        // ⚠️ 三类错误码都要出现 —— 模型要知道怎么区分。
        assert!(prompt.contains("E_INVALID_ARGUMENTS"));
        assert!(prompt.contains("E_UPSTREAM_UNAVAILABLE"));
        assert!(prompt.contains("E_UNAUTHORIZED"));
    }

    #[test]
    fn prompt_forbids_fabrication() {
        let prompt = system_prompt();
        assert!(prompt.contains("不要"));
        assert!(prompt.contains("编造"));
    }

    #[test]
    fn prompt_tells_the_model_to_stop_retrying_on_auth_errors() {
        let prompt = system_prompt();
        // ⚠️ 鉴权失败时明确说「不要重试」。
        assert!(prompt.contains("不要重试"));
    }

    #[test]
    fn prompt_lists_the_tools() {
        let prompt = system_prompt();
        assert!(prompt.contains("flaky_fetch"));
        assert!(prompt.contains("strict_lookup"));
        assert!(prompt.contains("guarded_pay"));
    }

    #[test]
    fn renders_the_tool_catalog() {
        let specs = vec![ToolSpec::new(
            "flaky_fetch",
            "拉取数据。",
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": { "resource": { "type": "string" } },
                "required": ["resource"]
            }),
        )];

        let text = render_tool_catalog(&specs);
        assert!(text.contains("flaky_fetch"));
        assert!(text.contains("resource"));
    }

    #[test]
    fn renders_the_trajectory_with_error_codes() {
        let steps = vec![crate::agent::StepRecord {
            step: 1,
            text: String::new(),
            tool_calls: vec![crate::agent::ToolCallRecord {
                call_id: "c1".to_string(),
                name: "strict_lookup".to_string(),
                arguments: json!({ "id": "nope" }),
                observation: "[E_NOT_FOUND] 目标资源不存在：记录 nope 不存在。".to_string(),
                ok: false,
                attempts: 1,
                error_code: Some("E_NOT_FOUND".to_string()),
                error_kind: Some("business".to_string()),
                retried: false,
            }],
            final_answer: false,
        }];

        let text = render_trajectory(&steps);
        assert!(text.contains("E_NOT_FOUND"));
        assert!(text.contains("business"));
    }
}
