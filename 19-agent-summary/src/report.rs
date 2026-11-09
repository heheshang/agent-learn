//! 报告层 —— 周度总结与项目构建的结构化产物。
//!
//! # 两个产物
//!
//! | 产物 | 回答的问题 | 章节 |
//! |---|---|---|
//! | [`WeeklySummary`] | 「这周干了什么？」 | 概览 / 亮点 / 风险 / 下周计划 |
//! | [`ProjectPlan`] | 「这个项目该怎么搭？」 | 目标 / 模块 / 里程碑 / 风险 |
//!
//! # ⚠️ 为什么要「结构化」而不是「一段话」
//!
//! 一段 Markdown 文本**没法校验**：
//! - 你不知道它有没有引用证据
//! - 你没法渲染成表格
//! - 你没法 diff 两周的差异
//!
//! 所以模型必须输出 **JSON**，由本模块解析成结构体，
//! 再**机制化校验**（引用是否成立），最后才渲染成人看的文本。
//!
//! # ⚠️ 三个实测踩出来的坑
//!
//! ## 坑 1：模型输出的 JSON 外面包着 ```json 代码块
//!
//! 直接 `serde_json::from_str` 会失败。
//! 所以要先**剥围栏**再解析。
//!
//! ## 坑 2：模型会在 JSON 前后加「好的，这是周报：」这样的前言
//!
//! 所以不能「找第一个 `{` 到最后一个 `}`」——
//! 前言里可能也有花括号（比如「{项目}」）。
//! 正确做法是**从第一个 `{` 开始，逐字符扫描配对括号**
//! （跳过字符串里的括号），找到**配对完整**的那一段。
//!
//! ## 坑 3：模型可能引用不存在的证据编号
//!
//! 提示词写了「只能引用给定编号」，但模型仍会写 `E99`。
//! 所以校验必须在**代码里**做，不能靠提示词。

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::evidence::EvidenceCollector;

/// 一条带证据引用的条目。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bullet {
    /// 条目正文。
    pub text: String,

    /// 引用的证据编号（`E1`、`E2`…）。
    #[serde(default)]
    pub evidence: Vec<String>,
}

impl Bullet {
    /// 构造一条。
    pub fn new(text: impl Into<String>, evidence: Vec<String>) -> Self {
        Self {
            text: text.into(),
            evidence,
        }
    }
}

/// 报告的**接地状态**。
///
/// ## 这是本项目最核心的可断言字段
///
/// 和 15 的 `grounded` 布尔值同源，但更细：
///
/// | 状态 | 含义 |
/// |---|---|
/// | `Grounded` | 每条结论都引用了**真实存在**的证据 |
/// | `PartiallyGrounded` | 部分结论引用了不存在的证据 |
/// | `Ungrounded` | 一条证据都没引用 |
///
/// ## ⚠️ 默认值是 `Ungrounded`
///
/// 因为「默认接地」是最危险的默认值 ——
/// 模型没引用任何东西时，如果默认是 `Grounded`，
/// 校验就形同虚设。
/// **没引用 = 没接地**，这是唯一安全的默认。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grounding {
    /// 完全接地。
    Grounded,

    /// 部分接地。
    PartiallyGrounded,

    /// 完全没有接地。
    #[default]
    Ungrounded,
}

impl Grounding {
    /// 是否**完全**接地。
    pub fn is_grounded(&self) -> bool {
        matches!(self, Self::Grounded)
    }

    /// 人类可读名称。
    pub fn name(&self) -> &'static str {
        match self {
            Self::Grounded => "接地",
            Self::PartiallyGrounded => "部分接地",
            Self::Ungrounded => "未接地",
        }
    }
}

/// 接地校验的结果。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroundingReport {
    /// 总体状态。
    pub status: Grounding,

    /// 引用了但**不存在**的证据编号。
    pub dangling_refs: Vec<String>,

    /// 实际被引用到的证据编号。
    pub used_refs: Vec<String>,

    /// 收集到的证据总数。
    pub total_evidence: usize,
}

/// 周度总结。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeeklySummary {
    /// 一句话概览。
    #[serde(default)]
    pub overview: String,

    /// 本周亮点。
    #[serde(default)]
    pub highlights: Vec<Bullet>,

    /// 风险与阻塞。
    #[serde(default)]
    pub risks: Vec<Bullet>,

    /// 下周计划。
    #[serde(default)]
    pub next_steps: Vec<Bullet>,

    /// 接地状态。
    #[serde(default)]
    pub grounding: Grounding,

    /// 接地详情。
    #[serde(default)]
    pub grounding_report: GroundingReport,
}

/// 项目构建方案。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectPlan {
    /// 方案标题。
    #[serde(default)]
    pub title: String,

    /// 项目目标。
    #[serde(default)]
    pub goal: String,

    /// 技术选型。
    #[serde(default)]
    pub tech_stack: Vec<String>,

    /// 模块划分。
    #[serde(default)]
    pub modules: Vec<PlanModule>,

    /// 里程碑。
    #[serde(default)]
    pub milestones: Vec<Bullet>,

    /// 风险。
    #[serde(default)]
    pub risks: Vec<Bullet>,

    /// 接地状态。
    #[serde(default)]
    pub grounding: Grounding,

    /// 接地详情。
    #[serde(default)]
    pub grounding_report: GroundingReport,
}

/// 方案里的一个模块。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanModule {
    /// 模块名。
    #[serde(default)]
    pub name: String,

    /// 模块职责。
    #[serde(default)]
    pub purpose: String,

    /// 该模块用到的技术。
    #[serde(default)]
    pub tech: Vec<String>,

    /// 关键任务。
    #[serde(default)]
    pub tasks: Vec<String>,

    /// 引用的证据编号。
    #[serde(default)]
    pub evidence: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────
// JSON 提取
// ─────────────────────────────────────────────────────────────────────

/// 从模型的原始输出里**提取出 JSON 片段**。
///
/// ## ⚠️ 为什么不能简单地「找第一个 `{` 到最后一个 `}`」
///
/// 因为模型经常在 JSON 前后加前言/后语，
/// 而前言里可能也有花括号（中文里 `{}` 用来占位很常见）。
///
/// 正确做法：**收集所有配对完整的候选**，**逐个尝试解析**，
/// 取**第一个能解析成 JSON 的**。
///
/// ## 为什么是「第一个能解析的」而不是「最大的」
///
/// 前言里的 `{周报}` 也能解析成 JSON？—— 不能，
/// 它不是合法 JSON（键没有引号）。
/// 所以「能解析」这个判据已经足够严格。
///
/// ## 代码块围栏
///
/// 模型经常输出带 ` ```json ` 围栏的 JSON ——
/// 剥围栏由 [`strip_code_fence`] 处理。
pub fn extract_json(raw: &str) -> Option<String> {
    let text = strip_code_fence(raw);

    for (start, ch) in text.char_indices() {
        if ch != '{' {
            continue;
        }

        let Some(end) = scan_balanced(&text[start..]) else {
            continue;
        };

        let candidate = &text[start..start + end];

        // ⚠️ 必须**真的能解析**才算数 ——
        //    光是括号配对还不够（`{周报}` 括号也配对）。
        if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
            return Some(candidate.to_string());
        }
    }

    None
}

/// 剥掉 Markdown 代码块围栏。
///
/// ## ⚠️ 只剥**围栏本身**，不动内容
///
/// 里面的 JSON 可能包含换行（模型爱 pretty-print），
/// 所以要保留原文，只去掉首尾的 ``` 行。
pub fn strip_code_fence(raw: &str) -> &str {
    let trimmed = raw.trim();

    if !trimmed.starts_with("```") {
        return trimmed;
    }

    // 跳过第一行（```json 或 ```）。
    let Some(first_newline) = trimmed.find('\n') else {
        return trimmed;
    };

    let body = &trimmed[first_newline + 1..];

    // 找到结束的 ```。
    match body.rfind("```") {
        Some(idx) => body[..idx].trim(),
        None => body.trim(),
    }
}

/// 扫描配对的 JSON 对象，返回**长度**（字节）。
///
/// ## ⚠️ 字符串字面量里的括号要跳过
///
/// `{"text": "a } b"}` —— 这里的 `}` 不结束对象。
fn scan_balanced(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (i, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + ch.len_utf8());
                }
            }
            _ => {}
        }
    }

    None
}

// ─────────────────────────────────────────────────────────────────────
// 接地校验
// ─────────────────────────────────────────────────────────────────────

/// 校验一份报告的接地状态。
///
/// ## 判据
///
/// 1. 收集所有条目引用的编号
/// 2. 不在证据集合里的编号 → `dangling_refs`（**悬空引用**）
/// 3. **一条引用都没有** → `Ungrounded`
/// 4. **有悬空引用**（哪怕一条有效引用都没有）→ `PartiallyGrounded`
/// 5. 否则 → `Grounded`
///
/// ## ⚠️ 为什么「只引用了伪造编号」也算 `PartiallyGrounded`
///
/// 这是本项目最容易搞错的一个语义。
///
/// 两种「没接地」的性质**完全不同**：
///
/// | 情况 | 含义 | 该怎么对待 |
/// |---|---|---|
/// | 一条引用都没有 | 模型**什么都没查**（可能是真的没资料） | 诚实 —— 如实说明查不到 |
/// | 引用了 `E99`（不存在） | 模型**编造**了引用 | **幻觉** —— 必须暴露 |
///
/// 如果第二种情况被判成 `Ungrounded`，
/// 它就混在「诚实但没资料」里了 —— **幻觉被掩盖**。
///
/// 所以：**有悬空引用就至少是 `PartiallyGrounded`**，
/// 哪怕有效引用数为零。
pub fn check_grounding(refs: &[String], collector: &EvidenceCollector) -> GroundingReport {
    let mut dangling: BTreeSet<String> = BTreeSet::new();
    let mut used: BTreeSet<String> = BTreeSet::new();

    for r in refs {
        if collector.contains(r) {
            used.insert(r.clone());
        } else {
            dangling.insert(r.clone());
        }
    }

    let status = if !dangling.is_empty() {
        // ⚠️ 悬空引用优先 —— 编造引用比没引用更严重。
        Grounding::PartiallyGrounded
    } else if used.is_empty() {
        Grounding::Ungrounded
    } else {
        Grounding::Grounded
    };

    GroundingReport {
        status,
        dangling_refs: dangling.into_iter().collect(),
        used_refs: used.into_iter().collect(),
        total_evidence: collector.len(),
    }
}

/// 收集一份周报里的全部引用编号。
pub fn summary_refs(summary: &WeeklySummary) -> Vec<String> {
    let mut refs = Vec::new();
    for group in [&summary.highlights, &summary.risks, &summary.next_steps] {
        for b in group {
            refs.extend(b.evidence.iter().cloned());
        }
    }
    refs
}

/// 收集一份方案里的全部引用编号。
pub fn plan_refs(plan: &ProjectPlan) -> Vec<String> {
    let mut refs = Vec::new();
    for group in [&plan.milestones, &plan.risks] {
        for b in group {
            refs.extend(b.evidence.iter().cloned());
        }
    }
    for m in &plan.modules {
        refs.extend(m.evidence.iter().cloned());
    }
    refs
}

/// 解析并校验周报。
pub fn parse_weekly_summary(
    raw: &str,
    collector: &EvidenceCollector,
) -> Result<WeeklySummary, crate::error::AppError> {
    let json_text = extract_json(raw).ok_or_else(|| {
        crate::error::AppError::Report(format!(
            "模型输出里找不到 JSON 对象。原始输出前 200 字：{}",
            raw.chars().take(200).collect::<String>()
        ))
    })?;

    let mut summary: WeeklySummary = serde_json::from_str(&json_text)
        .map_err(|e| crate::error::AppError::Report(format!("周报 JSON 解析失败: {e}")))?;

    let refs = summary_refs(&summary);
    let report = check_grounding(&refs, collector);
    summary.grounding = report.status;
    summary.grounding_report = report;

    Ok(summary)
}

/// 解析并校验项目方案。
pub fn parse_project_plan(
    raw: &str,
    collector: &EvidenceCollector,
) -> Result<ProjectPlan, crate::error::AppError> {
    let json_text = extract_json(raw).ok_or_else(|| {
        crate::error::AppError::Report(format!(
            "模型输出里找不到 JSON 对象。原始输出前 200 字：{}",
            raw.chars().take(200).collect::<String>()
        ))
    })?;

    let mut plan: ProjectPlan = serde_json::from_str(&json_text)
        .map_err(|e| crate::error::AppError::Report(format!("项目方案 JSON 解析失败: {e}")))?;

    let refs = plan_refs(&plan);
    let report = check_grounding(&refs, collector);
    plan.grounding = report.status;
    plan.grounding_report = report;

    Ok(plan)
}

// ─────────────────────────────────────────────────────────────────────
// 渲染
// ─────────────────────────────────────────────────────────────────────

/// 渲染周报为 Markdown。
pub fn render_weekly_summary(summary: &WeeklySummary) -> String {
    let mut out = String::new();

    if !summary.overview.trim().is_empty() {
        out.push_str("## 概览\n\n");
        out.push_str(summary.overview.trim());
        out.push_str("\n\n");
    }

    for (title, items) in [
        ("本周亮点", &summary.highlights),
        ("风险与阻塞", &summary.risks),
        ("下周计划", &summary.next_steps),
    ] {
        out.push_str(&format!("## {title}\n\n"));
        if items.is_empty() {
            out.push_str("_（无）_\n\n");
            continue;
        }
        for b in items {
            let refs = if b.evidence.is_empty() {
                String::new()
            } else {
                format!(
                    " {}",
                    b.evidence
                        .iter()
                        .map(|e| format!("[{e}]"))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            };
            out.push_str(&format!("- {}{}\n", b.text.trim(), refs));
        }
        out.push('\n');
    }

    out.push_str(&format!(
        "---\n接地状态：**{}**（引用 {} / 证据 {}）\n",
        summary.grounding.name(),
        summary.grounding_report.used_refs.len(),
        summary.grounding_report.total_evidence
    ));

    if !summary.grounding_report.dangling_refs.is_empty() {
        out.push_str(&format!(
            "⚠️ 悬空引用（编号不存在）：{}\n",
            summary.grounding_report.dangling_refs.join(", ")
        ));
    }

    out
}

/// 渲染项目方案为 Markdown。
pub fn render_project_plan(plan: &ProjectPlan) -> String {
    let mut out = String::new();

    if !plan.title.trim().is_empty() {
        out.push_str(&format!("# {}\n\n", plan.title.trim()));
    }

    if !plan.goal.trim().is_empty() {
        out.push_str("## 目标\n\n");
        out.push_str(plan.goal.trim());
        out.push_str("\n\n");
    }

    if !plan.tech_stack.is_empty() {
        out.push_str("## 技术选型\n\n");
        for t in &plan.tech_stack {
            out.push_str(&format!("- {}\n", t.trim()));
        }
        out.push('\n');
    }

    if !plan.modules.is_empty() {
        out.push_str("## 模块划分\n\n");
        for m in &plan.modules {
            out.push_str(&format!("### {}\n\n", m.name.trim()));
            if !m.purpose.trim().is_empty() {
                out.push_str(&format!("{}\n\n", m.purpose.trim()));
            }
            if !m.tech.is_empty() {
                out.push_str(&format!("技术：{}\n\n", m.tech.join(" / ")));
            }
            if !m.tasks.is_empty() {
                for t in &m.tasks {
                    out.push_str(&format!("- [ ] {}\n", t.trim()));
                }
                out.push('\n');
            }
        }
    }

    for (title, items) in [("里程碑", &plan.milestones), ("风险", &plan.risks)] {
        if items.is_empty() {
            continue;
        }
        out.push_str(&format!("## {title}\n\n"));
        for b in items {
            let refs = if b.evidence.is_empty() {
                String::new()
            } else {
                format!(
                    " {}",
                    b.evidence
                        .iter()
                        .map(|e| format!("[{e}]"))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            };
            out.push_str(&format!("- {}{}\n", b.text.trim(), refs));
        }
        out.push('\n');
    }

    out.push_str(&format!(
        "---\n接地状态：**{}**（引用 {} / 证据 {}）\n",
        plan.grounding.name(),
        plan.grounding_report.used_refs.len(),
        plan.grounding_report.total_evidence
    ));

    if !plan.grounding_report.dangling_refs.is_empty() {
        out.push_str(&format!(
            "⚠️ 悬空引用（编号不存在）：{}\n",
            plan.grounding_report.dangling_refs.join(", ")
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{EvidenceCollector, EvidenceKind};
    use serde_json::json;

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

    // ── JSON 提取 ──────────────────────────────────────────────

    #[test]
    fn extracts_plain_json() {
        let raw = r#"{"overview": "本周把错误处理做完了"}"#;
        let extracted = extract_json(raw).unwrap();
        assert!(extracted.starts_with('{'));
        assert!(extracted.contains("overview"));
    }

    #[test]
    fn extracts_json_from_code_fence() {
        let raw = "```json\n{\n  \"overview\": \"带围栏\"\n}\n```";
        let extracted = extract_json(raw).unwrap();
        assert!(extracted.contains("带围栏"));
    }

    #[test]
    fn extracts_json_after_preamble() {
        // ⚠️ 前言里有花括号 —— 不能被误当成 JSON 的开头。
        let raw = "好的，这是 {周报}：\n{\"overview\": \"真正的 JSON\"}";
        let extracted = extract_json(raw).unwrap();
        assert!(extracted.contains("真正的 JSON"));
        assert!(!extracted.contains("周报}"));
    }

    #[test]
    fn handles_braces_inside_strings() {
        let raw = r#"{"text": "有个 } 在里面", "n": 1}"#;
        let extracted = extract_json(raw).unwrap();
        let v: serde_json::Value = serde_json::from_str(&extracted).unwrap();
        assert_eq!(v["n"], json!(1));
    }

    #[test]
    fn returns_none_when_no_json() {
        assert!(extract_json("完全没有 JSON 的一段话").is_none());
    }

    // ── 接地校验 ────────────────────────────────────────────────

    #[test]
    fn grounded_when_all_refs_exist() {
        let c = collector();
        let report = check_grounding(&["E1".to_string(), "E2".to_string()], &c);

        assert_eq!(report.status, Grounding::Grounded);
        assert!(report.dangling_refs.is_empty());
        assert_eq!(report.used_refs.len(), 2);
    }

    #[test]
    fn partially_grounded_when_ref_is_dangling() {
        let c = collector();
        let report = check_grounding(&["E1".to_string(), "E99".to_string()], &c);

        assert_eq!(report.status, Grounding::PartiallyGrounded);
        assert_eq!(report.dangling_refs, vec!["E99".to_string()]);
    }

    #[test]
    fn dangling_only_is_still_partially_grounded() {
        // ⚠️ **关键语义**：只引用了伪造编号（`E99`），
        // 不能被判成 `Ungrounded` ——
        // 「没引用」是诚实，「引用了不存在的」是**幻觉**，
        // 两者混在一起会掩盖幻觉。
        let c = collector();
        let report = check_grounding(&["E99".to_string()], &c);

        assert_eq!(report.status, Grounding::PartiallyGrounded);
        assert!(report.used_refs.is_empty());
        assert_eq!(report.dangling_refs, vec!["E99".to_string()]);
    }

    #[test]
    fn ungrounded_only_when_nothing_is_cited_at_all() {
        let c = collector();
        let report = check_grounding(&[], &c);

        assert_eq!(report.status, Grounding::Ungrounded);
        assert!(!report.status.is_grounded());
    }

    // ── 解析 ────────────────────────────────────────────────────

    #[test]
    fn parses_weekly_summary_and_checks_grounding() {
        let c = collector();
        let raw = r#"{
            "overview": "本周完成错误处理模块",
            "highlights": [
                { "text": "实现重试与降级", "evidence": ["E1"] }
            ],
            "risks": [
                { "text": "搜索后端不稳定", "evidence": ["E2"] }
            ],
            "next_steps": [
                { "text": "接入 Web 搜索", "evidence": ["E7"] }
            ]
        }"#;

        let summary = parse_weekly_summary(raw, &c).unwrap();

        assert_eq!(summary.highlights.len(), 1);
        assert_eq!(summary.highlights[0].text, "实现重试与降级");
        assert_eq!(summary.grounding, Grounding::PartiallyGrounded);
        assert_eq!(
            summary.grounding_report.dangling_refs,
            vec!["E7".to_string()]
        );
    }

    #[test]
    fn parses_project_plan() {
        let c = collector();
        let raw = r#"{
            "title": "研究助手",
            "goal": "整合 RAG 与 Web 搜索",
            "tech_stack": ["Rust", "axum"],
            "modules": [
                { "name": "检索层", "purpose": "召回证据", "tech": ["BM25"], "tasks": ["建索引"], "evidence": ["E1"] }
            ],
            "milestones": [
                { "text": "跑通端到端", "evidence": ["E2"] }
            ],
            "risks": []
        }"#;

        let plan = parse_project_plan(raw, &c).unwrap();

        assert_eq!(plan.title, "研究助手");
        assert_eq!(plan.tech_stack.len(), 2);
        assert_eq!(plan.modules[0].name, "检索层");
        assert_eq!(plan.grounding, Grounding::Grounded);
    }

    #[test]
    fn missing_sections_default_to_empty() {
        let c = collector();
        let raw = r#"{"overview": "只有概览"}"#;

        let summary = parse_weekly_summary(raw, &c).unwrap();

        assert!(summary.highlights.is_empty());
        assert_eq!(summary.grounding, Grounding::Ungrounded);
    }

    #[test]
    fn invalid_json_reports_error() {
        let c = collector();
        let raw = r#"{"overview": 123}"#;

        let err = parse_weekly_summary(raw, &c).unwrap_err();

        assert!(err.to_string().contains("周报 JSON 解析失败"));
    }

    // ── 渲染 ────────────────────────────────────────────────────

    #[test]
    fn renders_weekly_summary_markdown() {
        let c = collector();
        let raw = r#"{
            "overview": "概览文本",
            "highlights": [{ "text": "亮点一", "evidence": ["E1"] }],
            "risks": [],
            "next_steps": [{ "text": "下一步", "evidence": ["E2"] }]
        }"#;

        let summary = parse_weekly_summary(raw, &c).unwrap();
        let md = render_weekly_summary(&summary);

        assert!(md.contains("## 概览"));
        assert!(md.contains("## 本周亮点"));
        assert!(md.contains("亮点一"));
        assert!(md.contains("[E1]"));
        assert!(md.contains("接地状态"));
    }

    #[test]
    fn renders_plan_with_tasks() {
        let c = collector();
        let raw = r#"{
            "title": "研究助手",
            "goal": "目标",
            "tech_stack": ["Rust"],
            "modules": [
                { "name": "检索层", "purpose": "召回", "tech": ["BM25"], "tasks": ["建索引"], "evidence": ["E1"] }
            ],
            "milestones": [],
            "risks": []
        }"#;

        let plan = parse_project_plan(raw, &c).unwrap();
        let md = render_project_plan(&plan);

        assert!(md.contains("# 研究助手"));
        assert!(md.contains("### 检索层"));
        assert!(md.contains("- [ ] 建索引"));
    }
}
