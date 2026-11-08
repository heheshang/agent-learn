//! 终端表格渲染 —— 处理中文宽度。
//!
//! ## 为什么不能用 `format!("{:<15}")`
//!
//! Rust 的 `{:width$}` 按**字符数**补空格，不按**显示宽度**。
//! 而中文汉字在等宽终端里占 **2 列**：
//!
//! ```text
//! "变体"        → 2 个字符 → 补 13 个空格 → 显示宽度 15 + 2 = 17
//! "dense"       → 5 个字符 → 补 10 个空格 → 显示宽度 15
//! ```
//!
//! 所以中英文混排的表格会**歪** —— 而且越往右偏得越多。
//!
//! ## 本模块的做法
//!
//! 按显示宽度补空格：
//!
//! - ASCII 可见字符 → 1 列
//! - CJK 统一表意文字、假名、韩文 → 2 列
//! - 其余 → 1 列（简化处理，够用）
//!
//! 这样中英文混排的列才能真正对齐。

/// 字符的显示宽度（列数）。
///
/// ## 支持范围
///
/// - ASCII 可见字符 → 1
/// - CJK 统一表意文字（U+4E00–U+9FFF）→ 2
/// - CJK 标点、全角字符（U+3000–U+303F、U+FF00–U+FF60）→ 2
/// - 假名、韩文 → 2
/// - 其余（含控制字符、组合记号）→ 1
pub fn char_width(c: char) -> usize {
    let code = c as u32;

    match code {
        // CJK 统一表意文字 + 扩展 A + 兼容表意文字
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF => 2,

        // CJK 符号和标点、假名、韩文、全角形式
        0x3000..=0x303F
        | 0x3040..=0x30FF
        | 0x31F0..=0x31FF
        | 0xAC00..=0xD7AF
        | 0xFF00..=0xFF60
        | 0xFFE0..=0xFFE6 => 2,

        // 零宽字符（组合记号、零宽空格）→ 不占宽度
        0x200B..=0x200F | 0xFE00..=0xFE0F | 0x0300..=0x036F => 0,

        _ => 1,
    }
}

/// 字符串的显示宽度。
pub fn display_width(text: &str) -> usize {
    text.chars().map(char_width).sum()
}

/// 按显示宽度**左对齐**补空格。
pub fn pad_right(text: &str, width: usize) -> String {
    let current = display_width(text);
    if current >= width {
        return text.to_string();
    }
    format!("{text}{}", " ".repeat(width - current))
}

/// 按显示宽度**右对齐**补空格。
pub fn pad_left(text: &str, width: usize) -> String {
    let current = display_width(text);
    if current >= width {
        return text.to_string();
    }
    format!("{}{text}", " ".repeat(width - current))
}

/// 截断到指定显示宽度（超出时加省略号）。
pub fn truncate(text: &str, width: usize) -> String {
    if display_width(text) <= width {
        return text.to_string();
    }

    let mut out = String::new();
    let mut used = 0usize;

    for c in text.chars() {
        let w = char_width(c);
        if used + w > width.saturating_sub(1) {
            break;
        }
        out.push(c);
        used += w;
    }

    format!("{out}…")
}

/// 一个简单的等宽表格渲染器。
///
/// ## 用法
///
/// ```text
/// let mut table = Table::new(&["变体", "recall@5"]);
/// table.row(&["dense", "1.000"]);
/// println!("{}", table.render());
/// ```
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    /// 每列的宽度（自动计算，也可以手动指定）。
    widths: Vec<usize>,
    /// 左对齐的列（默认全部左对齐）。
    right_align: Vec<bool>,
}

impl Table {
    /// 创建表格（列宽自动计算）。
    pub fn new(headers: &[&str]) -> Self {
        Self {
            headers: headers.iter().map(|h| h.to_string()).collect(),
            rows: Vec::new(),
            widths: headers.iter().map(|h| display_width(h)).collect(),
            right_align: vec![false; headers.len()],
        }
    }

    /// 指定某列右对齐（数值列）。
    pub fn right_align(mut self, columns: &[usize]) -> Self {
        for &index in columns {
            if index < self.right_align.len() {
                self.right_align[index] = true;
            }
        }
        self
    }

    /// 加一行。
    pub fn row(&mut self, cells: &[&str]) {
        let row: Vec<String> = cells.iter().map(|c| c.to_string()).collect();

        for (index, cell) in row.iter().enumerate() {
            if index < self.widths.len() {
                let width = display_width(cell);
                if width > self.widths[index] {
                    self.widths[index] = width;
                }
            }
        }

        self.rows.push(row);
    }

    /// 渲染成文本。
    pub fn render(&self) -> String {
        let mut out = String::new();

        // 顶边框
        out.push_str(&self.border('┌', '┬', '┐'));
        // 表头
        out.push_str(&self.line(&self.headers));
        // 表头分隔
        out.push_str(&self.border('├', '┼', '┤'));
        // 数据行
        for row in &self.rows {
            out.push_str(&self.line(row));
        }
        // 底边框
        out.push_str(&self.border('└', '┴', '┘'));

        out
    }

    /// 渲染一行（带边框）。
    fn line(&self, cells: &[String]) -> String {
        let mut out = String::from("│");

        for (index, width) in self.widths.iter().enumerate() {
            let empty = String::new();
            let cell = cells.get(index).unwrap_or(&empty);

            if self.right_align.get(index).copied().unwrap_or(false) {
                out.push_str(&format!(" {} │", pad_left(cell, *width)));
            } else {
                out.push_str(&format!(" {} │", pad_right(cell, *width)));
            }
        }

        out.push('\n');
        out
    }

    /// 渲染一条横线。
    fn border(&self, left: char, middle: char, right: char) -> String {
        let mut out = String::from(left);

        for (index, width) in self.widths.iter().enumerate() {
            out.push_str(&"─".repeat(width + 2));
            out.push(if index + 1 == self.widths.len() {
                right
            } else {
                middle
            });
        }

        out.push('\n');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_is_one_column() {
        assert_eq!(display_width("dense"), 5);
        assert_eq!(display_width(""), 0);
    }

    #[test]
    fn cjk_is_two_columns() {
        assert_eq!(display_width("变体"), 4);
        assert_eq!(display_width("中文"), 4);
    }

    #[test]
    fn mixed_width_is_sum() {
        assert_eq!(display_width("dense 变体"), 5 + 1 + 4);
    }

    #[test]
    fn padding_uses_display_width() {
        // "变体" 是 4 列 → 补到 10 列需要 6 个空格，不是 8 个。
        let padded = pad_right("变体", 10);
        assert_eq!(display_width(&padded), 10);
        // 字节数 = 2 个汉字（各 3 字节）+ 6 个空格 = 12。
        //
        // ⚠️ 断言**字节数**而不是 `chars().count()` ——
        // 后者只会数出 8（2 汉字 + 6 空格），看不出宽度逻辑对不对。
        assert_eq!(padded.len(), 12);
        assert_eq!(padded.chars().count(), 8);
    }

    #[test]
    fn left_padding_uses_display_width() {
        let padded = pad_left("1.5", 8);
        assert_eq!(display_width(&padded), 8);
    }

    #[test]
    fn truncation_respects_width() {
        let short = truncate("短", 10);
        assert_eq!(short, "短");

        let long = truncate("这是一个很长的标题需要被截断", 8);
        assert!(display_width(&long) <= 8);
        assert!(long.ends_with('…'));
    }

    #[test]
    fn table_rows_align() {
        let mut table = Table::new(&["变体", "分数"]);
        table.row(&["dense", "1.000"]);
        table.row(&["混合检索", "0.500"]);

        let rendered = table.render();
        let lines: Vec<&str> = rendered.lines().collect();

        // 每一行的显示宽度必须一致。
        let width = display_width(lines[0]);
        for line in &lines {
            assert_eq!(display_width(line), width, "行宽不一致: {line}");
        }
    }

    #[test]
    fn empty_table_renders() {
        let table = Table::new(&["a", "b"]);
        let rendered = table.render();
        assert!(rendered.contains('┌'));
        assert!(rendered.contains('┘'));
    }
}
