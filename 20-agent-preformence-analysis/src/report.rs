//! 报告：把剖析结果整理成可对比、可断言的形式。
//!
//! ## 核心设计：不要把不同工具的数字「归一化」成同一个数
//!
//! cProfile 的秒、py-spy 的采样占比、Scalene 的 CPU 百分比 ——
//! 三者**语义不同**，强行归一化会丢掉最有价值的信息：
//! 「三个工具看到的根本不是同一件事」。
//!
//! 所以这里保留各自单位，**并排展示**。

use std::fmt::Write as _;

/// 单个工具的摘要。
#[derive(Debug, Clone)]
pub struct ToolSummary {
    /// 工具名。
    pub tool: String,

    /// 指标单位（自耗时秒 / 采样占比% / CPU 占比%）。
    pub unit: String,

    /// 是否包含 IO 等待。
    pub includes_io: bool,

    /// Top 项：(名称, 数值)。
    pub rows: Vec<(String, f64)>,
}

impl ToolSummary {
    /// 渲染成 Markdown 表格。
    pub fn to_markdown(&self, top: usize) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "### {}", self.tool);
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "单位：{}；{}",
            self.unit,
            if self.includes_io {
                "**含 IO 等待**"
            } else {
                "**不含 IO 等待**"
            }
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "| 排名 | 函数 | 指标 |");
        let _ = writeln!(out, "|---|---|---|");
        for (rank, (name, value)) in self.rows.iter().take(top).enumerate() {
            let _ = writeln!(out, "| {} | `{}` | {:.2} |", rank + 1, name, value);
        }
        out
    }
}

/// 一份完整报告。
#[derive(Debug, Clone)]
pub struct Report {
    /// 标题。
    pub title: String,

    /// 各工具摘要。
    pub tools: Vec<ToolSummary>,

    /// 计时结果：(标签, 毫秒)。
    pub timings: Vec<(String, u64)>,
}

impl Report {
    /// 创建空报告。
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            tools: Vec::new(),
            timings: Vec::new(),
        }
    }

    /// 加一个工具摘要。
    pub fn push_tool(&mut self, summary: ToolSummary) {
        self.tools.push(summary);
    }

    /// 加一条计时。
    pub fn push_timing(&mut self, label: impl Into<String>, elapsed_ms: u64) {
        self.timings.push((label.into(), elapsed_ms));
    }

    /// 渲染成 Markdown。
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# {}\n", self.title);

        if !self.timings.is_empty() {
            let _ = writeln!(out, "## 耗时对比\n");
            let _ = writeln!(out, "| 项目 | 毫秒 |");
            let _ = writeln!(out, "|---|---|");
            for (label, ms) in &self.timings {
                let _ = writeln!(out, "| {label} | {ms} |");
            }
            let _ = writeln!(out);
        }

        for tool in &self.tools {
            let _ = writeln!(out, "{}", tool.to_markdown(10));
        }
        out
    }
}
