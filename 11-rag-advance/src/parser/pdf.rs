//! PDF 读取 —— 把 PDF 变成带坐标的文字单元。
//!
//! ## 这一层是整个项目的地基
//!
//! 后面三个解析器（表格 / 图片 / 版面）全都建立在同一件事上：
//! **拿到每个文字单元的坐标**。
//!
//! ## 用的什么
//!
//! `docling-pdf` 的**纯 Rust 文本层解析器**（lopdf 实现，零外部依赖）。
//!
//! ⚠️ 为什么不用完整的 docling ML 管线：
//!
//! - 它要跑 ONNX 版面模型（几十 MB）+ 渲染页面 → 秒级 per 页
//! - 更要命的是 `docling-onnx` 要求 `ort =2.0.0-rc.14`，
//!   而本 workspace 的 `fastembed` 锁死 `ort =2.0.0-rc.9`
//!   （精确版本约束）→ **同 workspace 直接冲突**
//!
//! 纯几何路线没有这些负担，而且对「规整表格」效果够用。

use agent_logging::{debug, info, warn};
use serde::{Deserialize, Serialize};

use crate::error::{AdvanceError, AdvanceResult};
use crate::parser::geometry::{BBox, PositionedText};

/// 页面尺寸。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PageSize {
    /// 宽（磅）。
    pub width: f32,
    /// 高（磅）。
    pub height: f32,
}

/// 一页的所有内容。
#[derive(Debug, Clone)]
pub struct Page {
    /// 页码（从 1 开始）。
    pub page_no: usize,

    /// 页面尺寸。
    pub size: PageSize,

    /// 行（prose cell）—— 已经按阅读顺序排好。
    ///
    /// 「行」是 docling 文本层切出来的单位：一行里的字是连着的。
    pub lines: Vec<PositionedText>,

    /// 词 —— 比行更细的粒度。
    ///
    /// 表格还原用的是词级坐标（一个单元格可能是多个词）。
    pub words: Vec<PositionedText>,
}

impl Page {
    /// 页面上的文字总量（字符数）。
    pub fn char_count(&self) -> usize {
        self.lines.iter().map(|l| l.text.chars().count()).sum()
    }

    /// 一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "第 {} 页: {} 行 / {} 词 / {} 字符 ({:.0}×{:.0} pt)",
            self.page_no,
            self.lines.len(),
            self.words.len(),
            self.char_count(),
            self.size.width,
            self.size.height
        )
    }
}

/// 一份解析好的 PDF。
#[derive(Debug, Clone)]
pub struct PdfDocument {
    /// 页。
    pub pages: Vec<Page>,

    /// 文件名。
    pub file_name: String,
}

impl PdfDocument {
    /// 总页数。
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// 总字符数。
    pub fn char_count(&self) -> usize {
        self.pages.iter().map(Page::char_count).sum()
    }

    /// 文档里**有没有**文字层。
    ///
    /// 没有文字层 = 扫描件，纯几何路线完全用不上，
    /// 必须走 OCR。这个判断很重要 —— 否则用户会看到一个空结果，
    /// 却不知道为什么。
    pub fn has_text_layer(&self) -> bool {
        self.char_count() > 0
    }

    /// 一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "{}: {} 页, {} 字符, 文字层={}",
            self.file_name,
            self.page_count(),
            self.char_count(),
            if self.has_text_layer() {
                "有"
            } else {
                "无（扫描件）"
            }
        )
    }
}

/// 解析一份 PDF。
pub fn parse_pdf(path: &std::path::Path) -> AdvanceResult<PdfDocument> {
    let bytes =
        std::fs::read(path).map_err(|e| AdvanceError::Parse(format!("读取 PDF 失败: {e}")))?;

    parse_pdf_bytes(&bytes, &file_name_of(path))
}

/// 解析 PDF 字节流。
pub fn parse_pdf_bytes(bytes: &[u8], file_name: &str) -> AdvanceResult<PdfDocument> {
    debug!(file_name, bytes = bytes.len(), "开始解析 PDF 文本层");

    let parsed = docling_pdf::textparse::pdf_all_cells(bytes);

    if parsed.is_empty() {
        warn!(file_name, "PDF 没有可解析的页面");
        return Err(AdvanceError::Parse(
            "PDF 没有可解析的页面（文件损坏、加密，或不是 PDF）".to_string(),
        ));
    }

    let mut pages = Vec::with_capacity(parsed.len());

    for (index, cells) in parsed.iter().enumerate() {
        let lines: Vec<PositionedText> = cells
            .prose
            .iter()
            .map(|c| PositionedText::new(c.text.clone(), BBox::from_cell(c.l, c.t, c.r, c.b)))
            .collect();

        let words: Vec<PositionedText> = cells
            .words
            .iter()
            .map(|c| PositionedText::new(c.text.clone(), BBox::from_cell(c.l, c.t, c.r, c.b)))
            .collect();

        // 页面尺寸：从单元格的坐标范围推断。
        //
        // 没有单元格的页面（比如整页都是图）推不出来，
        // 那就用 US Letter 的默认值兜底 —— 只影响「页面大小」这个展示字段。
        let size = infer_page_size(&lines, &words);

        pages.push(Page {
            page_no: index + 1,
            size,
            lines,
            words,
        });
    }

    info!(file_name, page_count = pages.len(), "PDF 文本层解析完成");

    Ok(PdfDocument {
        pages,
        file_name: file_name.to_string(),
    })
}
/// 从单元格的坐标范围推断页面尺寸。
///
/// ## 为什么不用 PDF 的 `/MediaBox`
///
/// `/MediaBox` 才是权威答案，但 docling 的纯文本层 API 不暴露它
/// （完整管线才有）。而坐标范围的**下界**其实就等于页边距，
/// 上界等于页宽 —— 所以 `max(right)` 就是页面宽度的一个下界估计。
///
/// 实践中：文档的右边距通常比左边距小或者相等，
/// 所以 `max(right)` 略小于真实页宽。这个误差对分块没有影响
/// （分块只看相对位置），所以可以接受。
fn infer_page_size(lines: &[PositionedText], words: &[PositionedText]) -> PageSize {
    let mut max_right = 0.0f32;
    let mut max_bottom = 0.0f32;

    for cell in lines.iter().chain(words.iter()) {
        max_right = max_right.max(cell.bbox.right);
        max_bottom = max_bottom.max(cell.bbox.bottom);
    }

    if max_right <= 0.0 || max_bottom <= 0.0 {
        return PageSize {
            width: 612.0,
            height: 792.0,
        };
    }

    // 加上一个典型右边距（72pt = 1 英寸）。
    PageSize {
        width: max_right + 72.0,
        height: max_bottom + 72.0,
    }
}

/// 取文件名（含扩展名）。
fn file_name_of(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "document.pdf".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_a_parse_error() {
        let err = parse_pdf_bytes(b"not a pdf at all", "x.pdf").unwrap_err();
        assert_eq!(err.status_code(), 422);
    }

    #[test]
    fn page_size_falls_back_to_letter() {
        let size = infer_page_size(&[], &[]);
        assert_eq!(size.width, 612.0);
        assert_eq!(size.height, 792.0);
    }

    #[test]
    fn page_size_accounts_for_margins() {
        let lines = vec![PositionedText::new(
            "x",
            BBox::new(72.0, 100.0, 400.0, 112.0),
        )];
        let size = infer_page_size(&lines, &[]);
        assert!(size.width > 400.0);
    }
}
