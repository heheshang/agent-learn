//! 终端表格渲染 —— 处理中文显示宽度。
//!
//! ## 为什么不能直接用 `format!("{:<15}")`
//!
//! Rust 的 `{:width$}` 按**字符数**补空格，不按**显示宽度**。
//! 而中文汉字在等宽终端里占 **2 列**：
//!
//! ```text
//! "错误码"  → 3 个字符 → 补 12 个空格 → 显示宽度 15 + 3 = 18
//! "code"    → 4 个字符 → 补 11 个空格 → 显示宽度 15
//! ```
//!
//! 所以中英文混排的表格会**歪** —— 而且越往右偏得越多。
//!
//! ## 为什么不直接复用 `12-rag-summary` 的 `table.rs`
//!
//! `rag-12` 那个 crate 依赖 Milvus SDK / 交叉编码器等重依赖，
//! 为了「一个显示宽度函数」把那一串拖进来不划算。
//!
//! 这里只实现本项目需要的最小子集（`display_width` / `pad_right`），
//! 口径与 12 完全一致。
//!
//! ## 支持范围
//!
//! - ASCII 可见字符 → 1 列
//! - CJK 统一表意文字、假名、韩文、全角标点 → 2 列
//! - 零宽字符（组合记号、零宽空格）→ 0 列
//! - 其余 → 1 列

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

        // Emoji 与符号区（✅ ❌ ★ 等）→ 终端里通常占 2 列
        0x1F300..=0x1FAFF | 0x2600..=0x27BF | 0x2B00..=0x2BFF => 2,

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

/// 一个简单的等宽表格渲染器。
///
/// ## 用法
///
/// ```text
/// let mut table = Table::new(&["错误码", "类别"]);
/// table.row(&["E_NOT_FOUND", "business"]);
/// println!("{}", table.render());
/// ```
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    /// 每列的宽度（自动计算）。
    widths: Vec<usize>,
}

impl Table {
    /// 创建表格（列宽自动计算）。
    pub fn new(headers: &[&str]) -> Self {
        Self {
            headers: headers.iter().map(|h| h.to_string()).collect(),
            rows: Vec::new(),
            widths: headers.iter().map(|h| display_width(h)).collect(),
        }
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

        out.push_str(&self.border('┌', '┬', '┐'));
        out.push_str(&self.line(&self.headers));
        out.push_str(&self.border('├', '┼', '┤'));
        for row in &self.rows {
            out.push_str(&self.line(row));
        }
        out.push_str(&self.border('└', '┴', '┘'));

        out
    }

    /// 渲染一行（带边框）。
    fn line(&self, cells: &[String]) -> String {
        let mut out = String::from("│");

        for (index, width) in self.widths.iter().enumerate() {
            let empty = String::new();
            let cell = cells.get(index).unwrap_or(&empty);
            out.push_str(&format!(" {} │", pad_right(cell, *width)));
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
        assert_eq!(display_width("code"), 4);
        assert_eq!(display_width(""), 0);
    }

    #[test]
    fn cjk_is_two_columns() {
        assert_eq!(display_width("错误码"), 6);
        assert_eq!(display_width("中文"), 4);
    }

    #[test]
    fn mixed_width_is_sum() {
        assert_eq!(display_width("code 错误码"), 4 + 1 + 6);
    }

    #[test]
    fn padding_uses_display_width() {
        // "错误码" 是 6 列 → 补到 12 列需要 6 个空格，不是 9 个。
        let padded = pad_right("错误码", 12);
        assert_eq!(display_width(&padded), 12);
        // 字节数 = 3 个汉字（各 3 字节）+ 6 个空格 = 15。
        assert_eq!(padded.len(), 15);
        assert_eq!(padded.chars().count(), 9);
    }

    #[test]
    fn padding_does_not_truncate() {
        let padded = pad_right("很长的标题", 4);
        assert_eq!(padded, "很长的标题");
    }

    #[test]
    fn emoji_is_two_columns() {
        // ✅ / ❌ 在终端里占 2 列。
        assert_eq!(display_width("✅"), 2);
        assert_eq!(display_width("❌"), 2);
    }

    #[test]
    fn table_rows_align() {
        // ⚠️ 中英文混排时，每一行的显示宽度必须一致 ——
        //    否则表格会歪。
        let mut table = Table::new(&["错误码", "类别", "HTTP"]);
        table.row(&["E_NOT_FOUND", "business", "400"]);
        table.row(&["E_UPSTREAM_TIMEOUT", "transient", "503"]);

        let rendered = table.render();
        let lines: Vec<&str> = rendered.lines().collect();

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
