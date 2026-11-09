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

        // 零宽字符（组合记号、零宽空格）→ 不占宽度
        0x200B..=0x200F | 0xFE00..=0xFE0F | 0x0300..=0x036F => 0,

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

/// 一个简单的终端表格。
#[derive(Debug, Clone)]
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    /// 创建表格。
    pub fn new(headers: Vec<&str>) -> Self {
        Self {
            headers: headers.into_iter().map(|h| h.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    /// 加一行。
    pub fn push(&mut self, row: Vec<String>) {
        self.rows.push(row);
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

        // 表头
        let header_line: Vec<String> = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| pad_right(h, widths[i]))
            .collect();
        out.push_str(header_line.join("  ").trim_end());
        out.push('\n');

        // 分隔线
        let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
        out.push_str(&sep.join("  "));
        out.push('\n');

        // 数据行
        for row in &self.rows {
            let line: Vec<String> = row
                .iter()
                .enumerate()
                .take(columns)
                .map(|(i, c)| pad_right(c, widths[i]))
                .collect();
            out.push_str(line.join("  ").trim_end());
            out.push('\n');
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_is_one_column() {
        assert_eq!(char_width('a'), 1);
        assert_eq!(display_width("dense"), 5);
    }

    #[test]
    fn cjk_is_two_columns() {
        assert_eq!(char_width('变'), 2);
        assert_eq!(display_width("变体"), 4);
    }

    #[test]
    fn pads_by_display_width() {
        // 「变体」是 4 列宽，补到 6 列 → 加 2 个空格（不是 4 个）。
        assert_eq!(pad_right("变体", 6), "变体  ");
        assert_eq!(pad_right("dense", 6), "dense ");
    }

    #[test]
    fn does_not_truncate_long_text() {
        assert_eq!(pad_right("abcdef", 3), "abcdef");
    }

    #[test]
    fn table_columns_align() {
        let mut table = Table::new(vec!["档位", "命中"]);
        table.push(vec!["auto".to_string(), "3".to_string()]);
        table.push(vec!["无缓存".to_string(), "0".to_string()]);

        let rendered = table.render();
        let lines: Vec<&str> = rendered.lines().collect();

        // 表头、分隔线、两行数据。
        assert_eq!(lines.len(), 4);

        // ⚠️ 必须按**显示宽度**算列位置，不能用字节偏移 ——
        //    中文是 3 字节 UTF-8，字节偏移和显示列完全不是一回事。
        let column_of = |line: &str, needle: char| -> usize {
            let byte_idx = line.find(needle).expect("列存在");
            let prefix = &line[..byte_idx];
            display_width(prefix)
        };

        // 「命中」列的显示起始位置。
        let col = column_of(lines[0], '命');
        assert_eq!(column_of(lines[2], '3'), col);
        assert_eq!(column_of(lines[3], '0'), col);
    }

    #[test]
    fn mixed_cjk_and_ascii_rows_stay_aligned() {
        // ⚠️ 这是本模块存在的理由 ——
        //    「档位」列有中文，「命中」列有数字，两列必须对齐。
        let mut table = Table::new(vec!["档位", "命中次数"]);
        table.push(vec!["auto".to_string(), "3".to_string()]);
        table.push(vec!["无缓存".to_string(), "12".to_string()]);
        table.push(vec!["强制调用".to_string(), "0".to_string()]);

        let rendered = table.render();
        let lines: Vec<&str> = rendered.lines().collect();

        let column_of = |line: &str, needle: char| -> usize {
            let byte_idx = line.find(needle).expect("列存在");
            display_width(&line[..byte_idx])
        };

        let col = column_of(lines[0], '命');
        for (i, needle) in ['3', '1', '0'].iter().enumerate() {
            let line = lines[2 + i];
            assert_eq!(column_of(line, *needle), col, "第 {i} 行未对齐");
        }
    }

    #[test]
    fn renders_empty_table() {
        let table = Table::new(vec!["a", "b"]);
        let rendered = table.render();
        assert_eq!(rendered.lines().count(), 2);
    }
}
