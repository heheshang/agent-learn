//! **渲染层** —— 把汇总结果渲染成 Markdown 周报。
//!
//! # ⚠️ 成文用**确定性规则**，不用 LLM
//!
//! 这是有意取舍：
//!
//! - **用 LLM**：文笔好，但 token 数 / 思考长度 / 上游负载会**淹没**
//!   优化前后的差值 —— 上游慢 200ms，读者根本看不出「快了 5 倍」。
//! - **用规则**：文笔平淡，但**每个数字都可复现、可归因**。
//!
//! 这与「研究与成文拆两阶段」是同一个思路：
//! **证据要干净，表达才可信**。

use crate::model::{Digest, Highlight};

/// 渲染一份周报。
pub fn render_report(
    digest: &Digest,
    window: &crate::model::Window,
    comparison: Option<&crate::metrics::Comparison>,
    strategy_summaries: &[(crate::aggregate::Strategy, String)],
) -> String {
    let mut out = String::new();

    // ── 标题 ──
    out.push_str("# 周度总结\n\n");
    out.push_str(&format!(
        "> 窗口：`{}` ~ `{}`（{} 天）\n\n",
        window.start,
        window.end,
        window.days()
    ));

    // ── 总览 ──
    out.push_str("## 总览\n\n");
    out.push_str(&format!("- 记录总数：**{}** 条\n", digest.total));
    out.push_str(&format!("- 去重后：**{}** 条\n", digest.deduped));
    out.push_str(&format!("- 覆盖模块：**{}** 个\n", digest.scopes));
    out.push_str(&format!("- 覆盖天数：**{}** 天\n", digest.days));
    out.push_str(&format!("- 风险项：**{}** 个\n\n", digest.risks));

    // ── 亮点 ──
    if !digest.highlights.is_empty() {
        out.push_str("## 亮点\n\n");
        out.push_str("| 标题 | 模块 | 负责人 | 类型 | 权重 |\n");
        out.push_str("|---|---|---|---|---|\n");
        for h in &digest.highlights {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                h.title, h.scope, h.author, h.kind, h.weight
            ));
        }
        out.push('\n');
    }

    // ── 风险 ──
    if !digest.risks_list.is_empty() {
        out.push_str("## 风险\n\n");
        out.push_str("| 标题 | 模块 | 负责人 | 类型 | 权重 |\n");
        out.push_str("|---|---|---|---|---|\n");
        for h in &digest.risks_list {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                h.title, h.scope, h.author, h.kind, h.weight
            ));
        }
        out.push('\n');
    }

    // ── 分布 ──
    out.push_str("## 分布\n\n");
    out.push_str("### 按模块\n\n");
    out.push_str("| 模块 | 条数 |\n|---|---|\n");
    for (name, count) in &digest.by_scope {
        out.push_str(&format!("| {name} | {count} |\n"));
    }
    out.push('\n');

    out.push_str("### 按类型\n\n");
    out.push_str("| 类型 | 条数 |\n|---|---|\n");
    for (name, count) in &digest.by_kind {
        out.push_str(&format!("| {name} | {count} |\n"));
    }
    out.push('\n');

    out.push_str("### 按天\n\n");
    out.push_str("| 日期 | 条数 |\n|---|---|\n");
    for (name, count) in &digest.by_day {
        out.push_str(&format!("| {name} | {count} |\n"));
    }
    out.push('\n');

    // ── 三档对照 ──
    if let Some(c) = comparison {
        out.push_str("## 三档对照\n\n");
        out.push_str(
            "> 三档**必须产出完全一致的结果**，否则测的是「不同的功能」而不是「优化」。\n\n",
        );
        out.push_str("| 变体 | 说明 |\n|---|---|\n");
        for (s, summary) in strategy_summaries {
            out.push_str(&format!("| `{}` | {} |\n", s.name(), summary));
        }
        out.push('\n');
        out.push_str(&format!(
            "- QPS：**{:.2}x**（+{:.1}%）\n",
            c.qps_ratio, c.qps_gain_pct
        ));
        out.push_str(&format!("- p50：**{:.3}x**（越小越快）\n", c.p50_ratio));
        out.push_str(&format!(
            "- p99：**{:.3}x**（越小越快，↓{:.1}%）\n",
            c.p99_ratio, c.p99_drop_pct
        ));
        out.push('\n');
    }

    out
}

/// 渲染一组亮点成 Markdown 表格（给终端用）。
pub fn render_highlights_table(highlights: &[Highlight]) -> String {
    let mut out = String::new();
    out.push_str("| 标题 | 模块 | 负责人 | 权重 |\n");
    out.push_str("|---|---|---|---|\n");
    for h in highlights {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            h.title, h.scope, h.author, h.weight
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Strategy;

    fn digest() -> Digest {
        Digest {
            total: 100,
            deduped: 95,
            scopes: 6,
            risks: 4,
            days: 7,
            by_kind: vec![("feature".into(), 30)],
            by_author: vec![("alice".into(), 50)],
            by_scope: vec![("cache".into(), 20)],
            by_day: vec![("2023-11-14".into(), 14)],
            highlights: vec![Highlight {
                id: "a1".into(),
                title: "做了件事".into(),
                scope: "cache".into(),
                author: "alice".into(),
                kind: "feature".into(),
                weight: 9,
            }],
            risks_list: vec![Highlight {
                id: "a2".into(),
                title: "有风险".into(),
                scope: "rag".into(),
                author: "bob".into(),
                kind: "incident".into(),
                weight: 5,
            }],
        }
    }

    #[test]
    fn report_contains_all_sections() {
        let w = crate::model::Window::new(1_700_000_000, 1_700_000_604_800);
        let md = render_report(&digest(), &w, None, &[]);
        assert!(md.contains("# 周度总结"));
        assert!(md.contains("## 总览"));
        assert!(md.contains("## 亮点"));
        assert!(md.contains("## 风险"));
        assert!(md.contains("## 分布"));
        assert!(md.contains("做了件事"));
        assert!(md.contains("有风险"));
    }

    #[test]
    fn report_includes_comparison_when_present() {
        let w = crate::model::Window::new(0, 1);
        let base = crate::metrics::LoadReport::from_samples(
            "naive",
            &[crate::metrics::Sample {
                ok: true,
                elapsed_ms: 100.0,
                status: None,
                cache_hit: None,
            }],
            1000.0,
            1,
        );
        let fast = crate::metrics::LoadReport::from_samples(
            "optimized",
            &[crate::metrics::Sample {
                ok: true,
                elapsed_ms: 20.0,
                status: None,
                cache_hit: None,
            }],
            1000.0,
            1,
        );
        let c = crate::metrics::Comparison::new(&base, &fast);
        let md = render_report(&digest(), &w, Some(&c), &[(Strategy::Naive, "基线".into())]);
        assert!(md.contains("## 三档对照"));
        assert!(md.contains("naive"));
    }

    #[test]
    fn report_is_deterministic() {
        let w = crate::model::Window::new(0, 1);
        let a = render_report(&digest(), &w, None, &[]);
        let b = render_report(&digest(), &w, None, &[]);
        assert_eq!(a, b);
    }
}
