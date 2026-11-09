//! 证据层 —— 「模型说的」和「工具查到的」之间的桥。
//!
//! # 为什么要有这一层
//!
//! 15-agent-sql 踩过一个很隐蔽的坑：
//! **模型不查库就作答** —— 答案看起来很专业，但数字是编的。
//! 它的修法是「机制化兜底」：把没查过库的答案**退回**成观察喂回模型。
//!
//! 但那是**文本协议**下的粗粒度判断（查过库没有）。
//! 本项目要更精确：**每一条结论要能指回具体的一条证据**。
//!
//! # 证据是怎么来的
//!
//! 工具返回的 `data` 字段里带结构化结果。证据层在每次工具成功后
//! 把它们收集起来，分配**稳定的证据编号**（`E1`、`E2`…）：
//!
//! ```text
//! 工具调用 ──► data ──► 证据收集器 ──► [E1] KB 检索：notes.md#架构
//!                                        [E2] Web 搜索：https://rust-lang.org/
//! ```
//!
//! 然后在生成报告时，把证据清单交给模型，
//! 要求它**每条结论都标注用了哪几条证据**。
//!
//! # ⚠️ 校验是**机制化**的，不是提示词里的软约束
//!
//! 模型可能给结论标一个不存在的证据编号（`E99`）。
//! 所以报告生成后要**再校验一遍**：
//!
//! - 引用了不存在的证据编号 → 标记为 `unsupported`
//! - 一个证据都没引用 → 整篇报告标记为 `ungrounded`
//!
//! 这和 15 的 `grounded` 布尔值是同一个思路：
//! **把「答案有没有数据支撑」变成可断言的字段**。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 证据来源类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    /// 本地知识库检索。
    Kb,

    /// Web 搜索结果。
    WebSearch,

    /// 网页正文。
    WebFetch,

    /// 长期记忆。
    Memory,

    /// 算术计算。
    Calculation,
}

impl EvidenceKind {
    /// 人类可读的名称。
    pub fn name(&self) -> &'static str {
        match self {
            Self::Kb => "本地知识库",
            Self::WebSearch => "Web 搜索",
            Self::WebFetch => "网页正文",
            Self::Memory => "长期记忆",
            Self::Calculation => "计算",
        }
    }

    /// 从工具名推断来源类型。
    pub fn from_tool(tool: &str) -> Self {
        match tool {
            "kb_search" => Self::Kb,
            "web_search" => Self::WebSearch,
            "web_fetch" => Self::WebFetch,
            "memory" => Self::Memory,
            "calculate" => Self::Calculation,
            _ => Self::Kb,
        }
    }
}

/// 一条证据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// 证据编号（`E1`、`E2`…）。
    pub id: String,

    /// 来源类型。
    pub kind: EvidenceKind,

    /// 来源定位（文件路径 / 链接 / 事实描述）。
    pub source: String,

    /// 证据正文（给模型看的那部分）。
    pub text: String,

    /// 产生这条证据的工具名。
    pub tool: String,
}

/// 证据收集器。
///
/// ## 为什么按**编号数值**排序而不是字典序
///
/// `BTreeMap<String, _>` 会按**字典序**排 —— 于是
/// `E1, E10, E11, E2, ...`（`"E10" < "E2"`，因为 `'1' < '2'`）。
///
/// 这在报告里是**灾难**：证据清单会变成 `E1, E10, E11, E12, E2, E3...`，
/// 模型看到这种顺序会以为编号本身有含义。
///
/// 所以这里存 **`(序号, 证据)`**，排序用序号 —— 得到 `E1, E2, E3, ..., E10`。
///
/// ## 为什么用 `BTreeMap` 而不是 `Vec`
///
/// 需要按编号**有序**输出，而 `Vec` 只能按插入顺序 ——
/// 插入顺序和编号顺序在回溯时可能不一致
/// （比如先收集 `E2` 再补 `E1`）。
#[derive(Debug, Default)]
pub struct EvidenceCollector {
    items: BTreeMap<usize, Evidence>,
    counter: usize,
}

impl EvidenceCollector {
    /// 创建空收集器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 收集一条证据，返回它的编号。
    pub fn add(&mut self, kind: EvidenceKind, tool: &str, source: &str, text: &str) -> String {
        self.counter += 1;
        let id = format!("E{}", self.counter);

        self.items.insert(
            self.counter,
            Evidence {
                id: id.clone(),
                kind,
                source: source.to_string(),
                text: text.to_string(),
                tool: tool.to_string(),
            },
        );

        id
    }

    /// 从工具的 `data` 字段提取证据并收集。
    ///
    /// ## 不同工具的 `data` 结构
    ///
    /// | 工具 | data 里的证据 |
    /// |---|---|
    /// | `kb_search` | `hits[].source` + `hits[].heading_path` |
    /// | `web_search` | `results[].url` + `results[].title` + `results[].snippet` |
    /// | `web_fetch` | `url` |
    /// | `memory` | `facts[]` |
    /// | `calculate` | `expression` + `value` |
    ///
    /// 提取不到证据的工具**不算失败** —— 只是这条调用没有产出引用点。
    pub fn add_from_tool_data(&mut self, tool: &str, data: &Value) -> Vec<String> {
        let kind = EvidenceKind::from_tool(tool);
        let mut ids = Vec::new();

        match tool {
            "kb_search" => {
                if let Some(hits) = data.get("hits").and_then(Value::as_array) {
                    for hit in hits {
                        let source = hit
                            .get("source")
                            .and_then(Value::as_str)
                            .unwrap_or("未知文件");
                        let heading = hit
                            .get("heading_path")
                            .and_then(Value::as_array)
                            .map(|h| {
                                h.iter()
                                    .filter_map(|v| v.as_str())
                                    .collect::<Vec<_>>()
                                    .join(" > ")
                            })
                            .unwrap_or_default();
                        let location = if heading.is_empty() {
                            source.to_string()
                        } else {
                            format!("{source} > {heading}")
                        };

                        // ⚠️ 证据正文取**块正文**，不是标题路径 ——
                        //    否则每条证据的 text 都等于 source，
                        //    渲染出来是同一句话重复两遍。
                        //
                        // 块正文在 `text` 字段里（08 的 Bm25Hit 有正文）。
                        let text = hit
                            .get("text")
                            .and_then(Value::as_str)
                            .map(|t| t.trim().to_string())
                            .filter(|t| !t.is_empty())
                            .unwrap_or_else(|| location.clone());

                        ids.push(self.add(kind, tool, &location, &text));
                    }
                }
            }
            "web_search" => {
                if let Some(results) = data.get("results").and_then(Value::as_array) {
                    for r in results {
                        let url = r.get("url").and_then(Value::as_str).unwrap_or("");
                        let title = r.get("title").and_then(Value::as_str).unwrap_or("");
                        let snippet = r.get("snippet").and_then(Value::as_str).unwrap_or("");
                        let text = format!("{title}\n{snippet}").trim().to_string();
                        ids.push(self.add(kind, tool, url, &text));
                    }
                }
            }
            "web_fetch" => {
                if let Some(url) = data.get("url").and_then(Value::as_str) {
                    // ⚠️ 证据正文用**正文开头**，不是 URL ——
                    //    否则这条证据毫无信息量（source 和 text 一样）。
                    let text = data
                        .get("text")
                        .and_then(Value::as_str)
                        .map(|t| t.trim().to_string())
                        .filter(|t| !t.is_empty())
                        .unwrap_or_else(|| url.to_string());
                    ids.push(self.add(kind, tool, url, &text));
                }
            }
            "memory" => {
                if let Some(facts) = data.get("facts").and_then(Value::as_array) {
                    for f in facts {
                        let subject = f.get("subject").and_then(Value::as_str).unwrap_or("");
                        let predicate = f.get("predicate").and_then(Value::as_str).unwrap_or("");
                        let value = f.get("value").and_then(Value::as_str).unwrap_or("");
                        let text = format!("{subject}的{predicate}：{value}");
                        ids.push(self.add(kind, tool, &text, &text));
                    }
                }
            }
            "calculate" => {
                let expr = data.get("expression").and_then(Value::as_str).unwrap_or("");
                let value = data.get("value").and_then(Value::as_f64).unwrap_or(0.0);
                let text = format!("{expr} = {value}");
                ids.push(self.add(kind, tool, &text, &text));
            }
            _ => {}
        }

        ids
    }

    /// 全部证据（按编号**数值**排序）。
    pub fn all(&self) -> Vec<&Evidence> {
        self.items.values().collect()
    }

    /// 证据条数。
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// 是否没有证据。
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// 按编号取证据。
    pub fn get(&self, id: &str) -> Option<&Evidence> {
        self.items.values().find(|e| e.id == id)
    }

    /// 是否存在这个编号。
    ///
    /// ## ⚠️ 这里**不能**用 `items.contains_key(id)`
    ///
    /// 因为 key 现在是**序号**（`usize`），不是 `"E1"` 这样的字符串。
    /// 必须按 `id` 字段比对 —— 否则 `E1` 和 `E01` 会被当成同一个。
    pub fn contains(&self, id: &str) -> bool {
        self.items.values().any(|e| e.id == id)
    }

    /// 渲染成给模型看的证据清单。
    ///
    /// ## 格式
    ///
    /// ```text
    /// [E1] （本地知识库）notes.md > 架构
    ///      错误处理的降级策略是把失败转成观察喂回模型
    /// [E2] （Web 搜索）https://rust-lang.org/
    ///      Rust Programming Language
    /// ```
    pub fn render(&self, max_per_kind: usize) -> String {
        let mut by_kind: BTreeMap<&str, Vec<&Evidence>> = BTreeMap::new();

        for e in self.items.values() {
            by_kind.entry(e.kind.name()).or_default().push(e);
        }

        let mut out = String::new();
        for (kind, items) in by_kind {
            out.push_str(&format!("【{kind}】\n"));
            for e in items.iter().take(max_per_kind) {
                out.push_str(&format!("[{}] {}\n", e.id, e.source));
                let text = e.text.trim();
                if !text.is_empty() {
                    // 缩进两格，视觉上和来源区分开。
                    for line in text.lines().take(3) {
                        out.push_str(&format!("     {line}\n"));
                    }
                }
            }
            out.push('\n');
        }

        out.trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn assigns_stable_sequential_ids() {
        let mut c = EvidenceCollector::new();

        let a = c.add(EvidenceKind::Kb, "kb_search", "notes.md", "内容 A");
        let b = c.add(
            EvidenceKind::WebSearch,
            "web_search",
            "https://x.com",
            "内容 B",
        );

        assert_eq!(a, "E1");
        assert_eq!(b, "E2");
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn ids_sort_numerically_not_lexicographically() {
        // ⚠️ **回归测试**：实测踩过的坑。
        //
        // 用 `BTreeMap<String, _>` 存的话，字典序会给出
        // `E1, E10, E11, E2, ...` —— 报告里的证据清单会乱序，
        // 模型看到会以为编号本身有含义。
        let mut c = EvidenceCollector::new();
        for i in 0..12 {
            c.add(EvidenceKind::Kb, "kb_search", &format!("f{i}.md"), "内容");
        }

        let ids: Vec<&str> = c.all().iter().map(|e| e.id.as_str()).collect();

        assert_eq!(ids[0], "E1");
        assert_eq!(ids[1], "E2");
        assert_eq!(ids[9], "E10");
        assert_eq!(ids[11], "E12");
    }

    #[test]
    fn kb_evidence_uses_chunk_body_not_heading_path() {
        // ⚠️ **回归测试**：实测踩过的坑。
        //
        // 早期实现把 `source`（标题路径）同时当成证据正文，
        // 渲染出来是同一句话重复两遍 —— 模型拿不到任何内容。
        let mut c = EvidenceCollector::new();
        let data = json!({
            "hits": [{
                "source": "notes.md",
                "heading_path": ["错误处理"],
                "score": 1.0,
                "text": "降级策略是把失败转成观察喂回模型"
            }]
        });

        let ids = c.add_from_tool_data("kb_search", &data);
        let e = c.get(&ids[0]).unwrap();

        assert_eq!(e.source, "notes.md > 错误处理");
        assert_eq!(e.text, "降级策略是把失败转成观察喂回模型");
        // ⚠️ source 和 text **不能**是同一句。
        assert_ne!(e.source, e.text);
    }

    #[test]
    fn extracts_kb_hits() {
        let mut c = EvidenceCollector::new();
        let data = json!({
            "query": "降级",
            "hits": [
                { "source": "notes.md", "heading_path": ["架构"], "score": 1.0, "text": "降级策略" }
            ]
        });

        let ids = c.add_from_tool_data("kb_search", &data);

        assert_eq!(ids.len(), 1);
        assert_eq!(c.get(&ids[0]).unwrap().source, "notes.md > 架构");
    }

    #[test]
    fn extracts_web_results() {
        let mut c = EvidenceCollector::new();
        let data = json!({
            "results": [
                { "title": "Rust 官网", "url": "https://rust-lang.org", "snippet": "官方" }
            ]
        });

        let ids = c.add_from_tool_data("web_search", &data);

        assert_eq!(ids.len(), 1);
        assert_eq!(c.get(&ids[0]).unwrap().source, "https://rust-lang.org");
        assert!(c.get(&ids[0]).unwrap().text.contains("Rust 官网"));
    }

    #[test]
    fn extracts_calculate_evidence() {
        let mut c = EvidenceCollector::new();
        let data = json!({ "expression": "1+1", "value": 2.0 });

        let ids = c.add_from_tool_data("calculate", &data);

        assert_eq!(ids.len(), 1);
        assert!(c.get(&ids[0]).unwrap().text.contains('2'));
    }

    #[test]
    fn unknown_tool_yields_no_evidence() {
        let mut c = EvidenceCollector::new();

        let ids = c.add_from_tool_data("whatever", &json!({}));

        assert!(ids.is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn renders_grouped_by_kind() {
        let mut c = EvidenceCollector::new();
        c.add(EvidenceKind::Kb, "kb_search", "notes.md", "本地内容");
        c.add(
            EvidenceKind::WebSearch,
            "web_search",
            "https://x.com",
            "网页内容",
        );

        let text = c.render(10);

        assert!(text.contains("【本地知识库】"));
        assert!(text.contains("【Web 搜索】"));
        assert!(text.contains("[E1] notes.md"));
    }

    #[test]
    fn respects_max_per_kind() {
        let mut c = EvidenceCollector::new();
        for i in 0..5 {
            c.add(EvidenceKind::Kb, "kb_search", &format!("f{i}.md"), "内容");
        }

        let text = c.render(2);

        // 每类只渲染 2 条，但**证据本身还在**（编号不变）。
        assert_eq!(text.matches("[E").count(), 2);
        assert_eq!(c.len(), 5);
    }

    #[test]
    fn contains_checks_ids() {
        let mut c = EvidenceCollector::new();
        let id = c.add(EvidenceKind::Kb, "kb_search", "a", "b");

        assert!(c.contains(&id));
        assert!(!c.contains("E99"));
    }
}
