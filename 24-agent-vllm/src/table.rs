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
//! 本模块按**显示宽度**补空格：CJK 统一表意文字、假名、韩文 → 2 列，
//! 其余 → 1 列。
//!
//! ## ⚠️ 断言要用**显示列**，不是**字节下标**
//!
//! 中文是 3 字节 UTF-8，`line.find('命')` 拿的是**字节下标**，
//! 而对齐要看**显示列** ——
//! 要用 `display_width(&line[..byte_idx])` 换算（21 项目的教训）。
//!
//! ## ⚠️ 表头/数据行**不能** `trim_end`
//!
//! 分隔线是 `"-" 重复 width`（没有空格可删），
//! 去掉表头尾部的空格会让分隔线和表头**差一列**（22 项目的教训）。

/// 字符的显示宽度（列数）。
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

        // 零宽字符（组合记号、零宽空格）→ 0 列
        0x200B..=0x200F | 0xFE00..=0xFE0F | 0x0300..=0x036F => 0,

        // Emoji 与符号区（✅ ❌ ★ 等）→ 终端里通常占 2 列
        0x1F300..=0x1FAFF | 0x2600..=0x27BF | 0x2B00..=0x2BFF => 2,

        _ => 1,
    }
}

/// 字符串的显示宽度。
pub fn display_width(text: &str) -> usize {
    text.chars().map(char_width).sum()
}

/// 按显示宽度左对齐补空格。
pub fn pad_right(text: &str, width: usize) -> String {
    let current = display_width(text);
    if current >= width {
        return text.to_string();
    }
    format!("{text}{}", " ".repeat(width - current))
}

/// 按显示宽度**右**对齐补空格（数字列用）。
pub fn pad_left(text: &str, width: usize) -> String {
    let current = display_width(text);
    if current >= width {
        return text.to_string();
    }
    format!("{}{text}", " ".repeat(width - current))
}

/// 渲染一行「多列」文本 —— 每列按**显示宽度**对齐。
pub fn render_row(cells: &[&str], widths: &[usize], right_align: &[bool]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(cells.len());

    for (i, cell) in cells.iter().enumerate() {
        let width = widths.get(i).copied().unwrap_or(0);
        let right = right_align.get(i).copied().unwrap_or(false);
        parts.push(if right {
            pad_left(cell, width)
        } else {
            pad_right(cell, width)
        });
    }

    parts.join("  ")
}

/// 一个简单的终端表格。
#[derive(Debug, Clone)]
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    /// 创建表格。
    pub fn new(headers: &[&str]) -> Self {
        Self {
            headers: headers.iter().map(|h| h.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    /// 加一行。
    pub fn row(&mut self, cells: &[&str]) {
        self.rows
            .push(cells.iter().map(|c| c.to_string()).collect());
    }

    /// 渲染成字符串。
    pub fn render(&self) -> String {
        let columns = self.headers.len();

        // 每列的宽度 = 该列所有单元格（含表头）的最大显示宽度。
        let mut widths: Vec<usize> = self.headers.iter().map(|h| display_width(h)).collect();

        for row in &self.rows {
            for (i, cell) in row.iter().enumerate().take(columns) {
                let w = display_width(cell);
                if w > widths[i] {
                    widths[i] = w;
                }
            }
        }

        let mut out = String::new();

        let header_line: Vec<String> = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| pad_right(h, widths[i]))
            .collect();
        // ⚠️ **不** trim_end —— 去掉尾部空格会让分隔线宽度和表头差一列。
        out.push_str(&header_line.join("  "));
        out.push('\n');

        let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
        out.push_str(&sep.join("  "));
        out.push('\n');

        for row in &self.rows {
            let line: Vec<String> = row
                .iter()
                .enumerate()
                .take(columns)
                .map(|(i, c)| pad_right(c, widths[i]))
                .collect();
            out.push_str(&line.join("  "));
            out.push('\n');
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_characters_are_two_columns_wide() {
        assert_eq!(display_width("命"), 2);
        assert_eq!(display_width("dense"), 5);
        assert_eq!(display_width("命中"), 4);
    }

    #[test]
    fn emoji_is_two_columns_wide() {
        assert_eq!(char_width('✅'), 2);
        assert_eq!(char_width('★'), 2);
    }

    #[test]
    fn mixed_table_columns_line_up() {
        let mut table = Table::new(&["名称", "值"]);
        table.row(&["串行", "400"]);
        table.row(&["join_all", "100"]);

        let rendered = table.render();
        let lines: Vec<&str> = rendered.lines().collect();

        // ⚠️ 每行的显示宽度必须一致 —— 否则中文混排的表格会歪。
        let width = display_width(lines[0]);
        for line in &lines {
            assert_eq!(display_width(line), width, "行宽不一致：{line}");
        }
    }

    #[test]
    fn byte_index_must_be_converted_to_display_column() {
        let line = "命中有 400ms";
        let byte_idx = line.find('命').unwrap();
        assert_eq!(byte_idx, 0);
        assert_eq!(display_width(&line[..6]), 4);
    }

    #[test]
    fn render_row_aligns_by_display_width() {
        let widths = [14, 8, 20];
        let align = [false, false, true];

        let line = render_row(&["串行", "异步", "真 HTTP"], &widths, &align);

        let expected: usize = widths.iter().sum::<usize>() + 2 * (widths.len() - 1);
        assert_eq!(display_width(&line), expected);
    }

    #[test]
    fn render_row_right_aligns_numeric_columns() {
        let widths = [10, 8];
        let align = [false, true];

        let line = render_row(&["串行", "489"], &widths, &align);

        assert!(line.ends_with("489"));
        assert_eq!(display_width(&line), 10 + 8 + 2);
    }
}
