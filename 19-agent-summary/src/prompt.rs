//! 提示词组装。
//!
//! # 三个提示词
//!
//! | 提示词 | 用在哪 | 关键点 |
//! |---|---|---|
//! | [`research_system`] | Agent 循环 | 教模型**先查后答**、区分工具优先级 |
//! | [`weekly_prompt`] | 周报生成 | 要求输出**带证据编号的 JSON** |
//! | [`plan_prompt`] | 项目方案 | 同上，但结构不同 |
//!
//! # ⚠️ 三条硬规则
//!
//! ## 规则 1：提示词里**不重复**工具的参数说明
//!
//! 工具的参数格式在 API 的 `tools` 字段里，模型结构化看得到。
//! 提示词里再写一遍就有**两份描述不一致**的风险 ——
//! 16 实测过这个坑：Schema 写 `city` 而实现读 `location`，
//! 模型试了 3 种组合才放弃，烧掉 8 次工具调用。
//!
//! 所以提示词只写**行为规则**，不写参数。
//!
//! ## 规则 2：证据编号必须由**代码**校验
//!
//! 提示词说「只能引用给定的编号」，但模型仍会写 `E99`。
//! 所以真正的防线在 [`crate::report::check_grounding`]。
//!
//! ## 规则 3：JSON 输出要求要**极简**
//!
//! 模型对「输出 JSON」的理解偏差很大。
//! 所以提示词里给出**完整的字段清单和示例**，
//! 并且明确说「不要输出 JSON 以外的任何内容」。

use crate::evidence::EvidenceCollector;

/// 研究阶段的 system 提示词。
pub fn research_system() -> String {
    let mut out = String::new();

    out.push_str("# Identity\n\n");
    out.push_str(
        "你是一个**研究助手**。你的工作方式是：\
         先用工具查证，再用证据说话。\n\n",
    );

    out.push_str("# Instructions\n\n");
    out.push_str("## 工具优先级（重要）\n\n");
    out.push_str(
        "1. **先查本地知识库**（`kb_search`）—— 零延迟、结果稳定，最可能是项目内部信息。\n",
    );
    out.push_str("2. **再查长期记忆**（`memory`）—— 记录过的事实比重新搜索更可信。\n");
    out.push_str("3. **最后才搜互联网**（`web_search` + `web_fetch`）—— 外部信息有时效性，但更慢也可能失败。\n\n");
    out.push_str("## 事实性要求\n\n");
    out.push_str("- 所有事实性结论必须来自工具返回的结果。**不要凭记忆作答**。\n");
    out.push_str("- 统计数字（求和、平均、百分比）**必须**用 `calculate` 算，不要心算。\n");
    out.push_str("- 引用具体的版本号、API 名、文件路径时，请用 `web_fetch` 打开原文确认，不要只看搜索摘要。\n");
    out.push_str("- 如果工具都没查到，**如实说没查到**，并说明尝试过什么。\n\n");
    out.push_str("## 工具失败时\n\n");
    out.push_str("工具结果里会带一个 `[E_XXX]` 错误码和一句建议，请根据它决定下一步：\n");
    out.push_str(
        "  · `[E_INVALID_ARGUMENTS]` / `[E_NOT_FOUND]`：**参数或目标有问题** —— 按提示修正后重新调用。\n",
    );
    out.push_str(
        "  · `[E_UPSTREAM_TIMEOUT]` / `[E_UPSTREAM_UNAVAILABLE]`：**外部服务的临时故障** —— \
         换个工具、稍后再试，或者如实告诉用户「外部服务暂时不可用」。\n",
    );
    out.push_str(
        "  · `[E_UNAUTHORIZED]`：**环境没配好** —— 不要重试，直接如实说明服务不可用，\
         **不要**编一个看起来合理的结果。\n",
    );
    out.push_str("- 无论工具成功还是失败，都**不要**编造工具没有返回的数据。\n\n");

    out.push_str("# Output\n\n");
    out.push_str(
        "回答里要**点名你查到了什么**（文件、链接、版本号），\
         让读者能自己去核对。引用格式：`[E1]`、`[E2]`（工具返回后会给出编号）。\n",
    );

    out
}

/// 周报生成的用户提示词。
///
/// ## ⚠️ 为什么把「证据清单」塞进提示词
///
/// 因为模型要**引用**证据编号。
/// 如果不告诉它有哪些编号，它会自己编（`E1`、`E2`…… 恰好对上），
/// 或者编出 `E99`。
///
/// 所以这里把**真实存在的编号**列出来。
pub fn weekly_prompt(question: &str, collector: &EvidenceCollector, max_per_kind: usize) -> String {
    let mut out = String::new();

    out.push_str("# Task\n\n");
    out.push_str("请根据下面**已经查到的证据**，写一份周度总结。\n\n");

    if !question.trim().is_empty() {
        out.push_str(&format!("## 本周关注点\n\n{}\n\n", question.trim()));
    }

    out.push_str("# Evidence\n\n");
    if collector.is_empty() {
        out.push_str(
            "（**一条证据都没有**）—— 这种情况必须如实说明「本周没有查到可用资料」，\
             并把 highlights / risks / next_steps 都留空数组。**绝对不要**凭记忆编内容。\n\n",
        );
    } else {
        out.push_str(&format!(
            "以下是检索到的证据。**每一条结论都必须引用它的编号**（如 `[E1]`）——\
             引用不存在的编号会被系统判定为无效。\n\n{}\n\n",
            collector.render(max_per_kind)
        ));
    }

    out.push_str("# Output Format\n\n");
    out.push_str(
        "只输出一个 JSON 对象，不要有任何解释文字、不要用 Markdown 代码块包裹。字段如下：\n\n",
    );
    out.push_str(
        r#"{
  "overview": "一句话概览（100 字以内）",
  "highlights": [
    { "text": "本周完成的一件事", "evidence": ["E1"] }
  ],
  "risks": [
    { "text": "一个风险或阻塞", "evidence": ["E2"] }
  ],
  "next_steps": [
    { "text": "下周要做的一件事", "evidence": ["E3"] }
  ]
}
"#,
    );
    out.push_str("\n# Rules\n\n");
    out.push_str("- `evidence` 数组里**只能**出现证据清单里出现过的编号。\n");
    out.push_str(
        "- 每条要点**至少**一个证据编号；如果某条结论确实没有任何证据支撑，就**不要写这条**。\n",
    );
    out.push_str("- 没有内容的章节用**空数组**，不要写「无」之类的占位文字。\n");
    out.push_str("- 不要编造版本号、数字、人名或链接。\n");

    out
}

/// 项目方案的用户提示词。
pub fn plan_prompt(
    requirement: &str,
    collector: &EvidenceCollector,
    max_per_kind: usize,
) -> String {
    let mut out = String::new();

    out.push_str("# Task\n\n");
    out.push_str("请根据下面**已经查到的证据**，为这个需求设计一个项目构建方案。\n\n");

    out.push_str("## 需求\n\n");
    out.push_str(&format!("{}\n\n", requirement.trim()));

    out.push_str("# Evidence\n\n");
    if collector.is_empty() {
        out.push_str(
            "（**一条证据都没有**）—— 必须如实说明「缺少参考资料」，\
             tech_stack 和 modules 用空数组。**绝对不要**凭记忆编造技术选型。\n\n",
        );
    } else {
        out.push_str(&format!(
            "以下是检索到的证据。**技术选型和模块划分必须引用证据编号**。\n\n{}\n\n",
            collector.render(max_per_kind)
        ));
    }

    out.push_str("# Output Format\n\n");
    out.push_str(
        "只输出一个 JSON 对象，不要有任何解释文字、不要用 Markdown 代码块包裹。字段如下：\n\n",
    );
    out.push_str(
        r#"{
  "title": "项目名称",
  "goal": "项目要解决什么问题（一到两句）",
  "tech_stack": ["技术1", "技术2"],
  "modules": [
    {
      "name": "模块名",
      "purpose": "这个模块负责什么",
      "tech": ["用到的技术"],
      "tasks": ["具体任务1", "具体任务2"],
      "evidence": ["E1"]
    }
  ],
  "milestones": [
    { "text": "里程碑描述", "evidence": ["E2"] }
  ],
  "risks": [
    { "text": "风险描述", "evidence": ["E3"] }
  ]
}
"#,
    );
    out.push_str("\n# Rules\n\n");
    out.push_str("- `evidence` 数组里**只能**出现证据清单里出现过的编号。\n");
    out.push_str("- 每个模块**至少**一个证据编号；没有证据支撑的模块不要写。\n");
    out.push_str("- `tasks` 要具体到「能开工」的粒度，不要写「实现功能」这种空话。\n");
    out.push_str("- 不要编造版本号、API 名或不存在的库。\n");

    out
}

/// 渲染工具清单（给 CLI / API 展示用）。
pub fn render_tool_catalog(specs: &[crate::tools::ToolSpec]) -> String {
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

/// 渲染一次运行的**可读轨迹**。
pub fn render_trajectory(steps: &[crate::agent::StepRecord]) -> String {
    let mut out = String::new();

    for step in steps {
        out.push_str(&format!("── 第 {} 步 ──\n", step.step));

        if !step.text.is_empty() {
            out.push_str(&format!("文本: {}\n", step.text));
        }

        for call in &step.tool_calls {
            let retry_mark = if call.attempts > 1 {
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

            if !call.evidence.is_empty() {
                out.push_str(&format!("证据: {}\n", call.evidence.join(", ")));
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
    use crate::evidence::{EvidenceCollector, EvidenceKind};

    fn collector() -> EvidenceCollector {
        let mut c = EvidenceCollector::new();
        c.add(EvidenceKind::Kb, "kb_search", "notes.md", "本地内容");
        c.add(
            EvidenceKind::WebSearch,
            "web_search",
            "https://x.com",
            "网页内容",
        );
        c
    }

    #[test]
    fn research_prompt_states_tool_priority() {
        let p = research_system();

        // ⚠️ 优先级顺序必须写清楚 —— 这是本项目与「通用 Agent」的核心差异。
        assert!(p.contains("先查本地知识库"));
        assert!(p.contains("最后才搜互联网"));
    }

    #[test]
    fn research_prompt_teaches_error_codes() {
        let p = research_system();
        assert!(p.contains("E_INVALID_ARGUMENTS"));
        assert!(p.contains("E_UPSTREAM_UNAVAILABLE"));
        assert!(p.contains("E_UNAUTHORIZED"));
    }

    #[test]
    fn research_prompt_forbids_fabrication() {
        let p = research_system();
        // ⚠️ 这两条必须一字不差地出现 —— 它们是本项目的核心防线。
        //
        // 注意断言里**不含** `**`（Markdown 加粗标记）——
        // 提示词里是 `都**不要**编造`，加粗会插在词中间。
        assert!(p.contains("不要凭记忆作答"));
        assert!(p.contains("编造工具没有返回的数据"));
    }

    #[test]
    fn weekly_prompt_lists_evidence_ids() {
        let p = weekly_prompt("本周重点", &collector(), 10);

        assert!(p.contains("[E1]"));
        assert!(p.contains("[E2]"));
        assert!(p.contains("本周重点"));
    }

    #[test]
    fn weekly_prompt_handles_empty_evidence() {
        let c = EvidenceCollector::new();
        let p = weekly_prompt("问题", &c, 10);

        assert!(p.contains("一条证据都没有"));
        assert!(p.contains("绝对不要"));
    }

    #[test]
    fn plan_prompt_includes_requirement() {
        let p = plan_prompt("做一个研究助手", &collector(), 10);

        assert!(p.contains("做一个研究助手"));
        assert!(p.contains("modules"));
    }

    #[test]
    fn renders_tool_catalog() {
        let specs = vec![crate::tools::ToolSpec::new(
            "kb_search",
            "检索知识库",
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        )];

        let text = render_tool_catalog(&specs);
        assert!(text.contains("kb_search"));
        assert!(text.contains("query"));
    }

    #[test]
    fn renders_trajectory_with_evidence() {
        let steps = vec![crate::agent::StepRecord {
            step: 1,
            text: String::new(),
            tool_calls: vec![crate::agent::ToolCallRecord {
                call_id: "c1".to_string(),
                name: "kb_search".to_string(),
                arguments: serde_json::json!({ "query": "rust" }),
                observation: "命中 2 段".to_string(),
                ok: true,
                attempts: 1,
                error_code: None,
                error_kind: None,
                evidence: vec!["E1".to_string()],
            }],
            final_answer: false,
        }];

        let text = render_trajectory(&steps);
        assert!(text.contains("kb_search"));
        assert!(text.contains("E1"));
    }
}
