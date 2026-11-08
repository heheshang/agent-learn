//! 解析流水线 —— 把一份 PDF 变成结构化文档。
//!
//! ## 完整链路
//!
//! ```text
//! PDF 文件
//!    │
//!    ├──► pdf.rs      读出带坐标的文字单元
//!    │
//!    ├──► table.rs   几何聚类 → 表格网格
//!    │
//!    ├──► image.rs   lopdf → 图片文件
//!    │
//!    └──► layout.rs  字号 + 几何 → 标题层级
//!    │
//!    ▼
//! 结构化文档（标题树 + 段落 + 表格 + 图片）
//!    │
//!    ├──► markdown.rs  渲染成 Markdown（给 LLM / 给人看）
//!    │
//!    └──► chunk.rs     结构感知分块（给检索）
//! ```
//!
//! ## ⚠️ 关键设计：表格**不参与**普通的行合并
//!
//! 这是整个项目最重要的一条规则。
//!
//! 一个表格的 20 个单元格在几何上是 20 个「行」。如果让它们走
//! 「行合并成段」的逻辑，会得到一段乱七八糟的文本 ——
//! 表格的结构就此丢失。
//!
//! 正确的做法是**先识别表格，把表格区域从行流里摘掉**，
//! 剩下的行才做段落合并。
//!
//! 这就是「先切表格，再合并段落」的原因。

use std::collections::HashSet;
use std::path::Path;

use agent_logging::{debug, info};
use serde::{Deserialize, Serialize};

use crate::config::AppConfig;
use crate::error::AdvanceResult;
use crate::parser::geometry::{BBox, ElementKind, PositionedText};
use crate::parser::image::{ExtractedImage, ImageOptions, extract_images};
use crate::parser::layout::{LayoutOptions, analyze_page};
use crate::parser::pdf::{PdfDocument, parse_pdf};
use crate::parser::table::{Table, TableOptions, TableStrategy, reconstruct_table};

/// 文档里的一个结构化元素。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentElement {
    /// 类型。
    pub kind: ElementKind,

    /// 文本（表格是 Markdown，图片是描述）。
    pub text: String,

    /// 标题层级。
    pub level: Option<u8>,

    /// 页码。
    pub page_no: usize,

    /// 位置。
    pub bbox: BBox,

    /// 表格的网格（只有 `kind == Table` 时有）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table: Option<TableData>,
}

/// 表格数据（可序列化的网格）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableData {
    /// 网格内容。
    pub grid: Vec<Vec<String>>,
    /// 表头。
    pub header: Vec<String>,
    /// 还原策略。
    pub strategy: TableStrategy,
}

/// 一份解析后的文档。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDocument {
    /// 来源文件名。
    pub file_name: String,

    /// 结构化元素（按阅读顺序）。
    pub elements: Vec<DocumentElement>,

    /// 抽出来的图片。
    pub images: Vec<ExtractedImage>,

    /// 页数。
    pub page_count: usize,

    /// 解析统计。
    pub stats: ParseStats,
}

/// 解析统计。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParseStats {
    /// 原始字符数。
    pub raw_chars: usize,

    /// 结构化后的字符数。
    pub structured_chars: usize,

    /// 识别出的表格数。
    pub table_count: usize,

    /// 识别出的图片数。
    pub image_count: usize,

    /// 成功落盘的图片数。
    pub extracted_image_count: usize,

    /// 标题数。
    pub heading_count: usize,

    /// 段落数。
    pub paragraph_count: usize,

    /// 有没有文字层。
    pub has_text_layer: bool,
}

impl ParseStats {
    /// 一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "{} 字符 → {} 字符 | 标题 {} 段落 {} 表格 {} 图片 {}/{}",
            self.raw_chars,
            self.structured_chars,
            self.heading_count,
            self.paragraph_count,
            self.table_count,
            self.extracted_image_count,
            self.image_count,
        )
    }

    /// 结构化后的字符数比原始多了多少。
    ///
    /// 正数说明结构化「补充」了信息（表格的 Markdown 表头、
    /// 分隔行会带来额外字符）；负数说明丢了一些东西。
    pub fn expansion_ratio(&self) -> f32 {
        if self.raw_chars == 0 {
            return 0.0;
        }
        self.structured_chars as f32 / self.raw_chars as f32
    }
}

/// 解析一份 PDF。
pub fn parse_file(path: &Path, config: &AppConfig) -> AdvanceResult<ParsedDocument> {
    let pdf = parse_pdf(path)?;

    let image_options = ImageOptions {
        min_bytes: config.image.min_bytes,
        min_dimension: config.image.min_dimension,
    };

    let images = extract_images(path, &config.asset_dir, &image_options)?;

    parse_document(pdf, images, config)
}

/// 解析一份已经读进来的 PDF。
pub fn parse_document(
    pdf: PdfDocument,
    images: Vec<ExtractedImage>,
    config: &AppConfig,
) -> AdvanceResult<ParsedDocument> {
    let table_options = TableOptions {
        min_cells: config.table.min_cells,
        row_tolerance: config.table.row_tolerance,
        column_tolerance: config.table.column_tolerance,
    };
    let layout_options = LayoutOptions::default();

    let mut elements: Vec<DocumentElement> = Vec::new();
    let mut table_count = 0usize;

    for page in &pdf.pages {
        // ── ① 先找表格 ──────────────────────────────────────────────
        //
        // 表格必须先于段落合并处理 —— 否则单元格会被当成行、
        // 再被当成段落，结构就没了。
        let (tables, consumed) = detect_tables(page, &table_options);

        // ── ② 把表格区域从行流里摘掉 ──────────────────────────────
        let remaining: Vec<PositionedText> = page
            .lines
            .iter()
            .filter(|line| {
                !consumed
                    .iter()
                    .any(|region| line.bbox.coverage_in(region) > 0.5)
            })
            .cloned()
            .collect();

        // ── ③ 版面分析：先分类（标题 / 正文），再合并成段 ────────
        //
        // ⚠️ 顺序不能反 —— 先合并的话，标题信号就丢了
        // （合并后的段落框是所有行的并集，高度暴涨，
        //   每个段落看起来都像标题）。
        let layout = analyze_page(&remaining, &layout_options);

        // ── ④ 把表格**插回**阅读顺序 ──────────────────────────────
        //
        // ⚠️ 为什么不能先把表格全 push 进去？
        //
        // 因为表格**必须待在自己的章节里** —— 分块时表格块的
        // `heading_path` 取的是它前面的标题栈。
        //
        // 如果先把所有表格 push 完再 push 文字，那么表格块的
        // 标题路径就是**空**的（还没遇到任何标题）。
        //
        // 所以：把表格按 y 坐标排序，和文字元素**归并**成一条
        // 按阅读顺序排列的流。
        let mut table_elements: Vec<DocumentElement> = Vec::with_capacity(tables.len());
        for table in &tables {
            table_count += 1;
            table_elements.push(DocumentElement {
                kind: ElementKind::Table,
                text: table.to_markdown(),
                level: None,
                page_no: page.page_no,
                bbox: table.bbox,
                table: Some(TableData {
                    grid: table.grid.clone(),
                    header: table.header.clone(),
                    strategy: table.strategy,
                }),
            });
        }

        let mut text_elements: Vec<DocumentElement> = layout
            .elements
            .into_iter()
            .map(|element| DocumentElement {
                kind: element.kind,
                text: element.text,
                level: element.level,
                page_no: page.page_no,
                bbox: element.bbox,
                table: None,
            })
            .collect();

        // 归并：按 y 坐标（top）排序，y 相同时表格优先（表格在上）。
        text_elements.append(&mut table_elements);
        text_elements.sort_by(|a, b| {
            a.bbox
                .top
                .partial_cmp(&b.bbox.top)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    // 同一行：表格排在段落前面。
                    match (a.kind, b.kind) {
                        (ElementKind::Table, ElementKind::Paragraph) => std::cmp::Ordering::Less,
                        (ElementKind::Paragraph, ElementKind::Table) => std::cmp::Ordering::Greater,
                        _ => std::cmp::Ordering::Equal,
                    }
                })
        });

        elements.extend(text_elements);
    }

    // ── ⑤ 图片挂到文档里 ─────────────────────────────────────────
    //
    // 图片不进入正文流 —— 它是**独立于文字的元素**。
    // 图片的描述会被写进 chunk（见 chunk.rs），但正文流里
    // 只保留一个占位，免得打断段落。
    for image in &images {
        if image.status != crate::parser::image::ImageStatus::Extracted {
            continue;
        }
        elements.push(DocumentElement {
            kind: ElementKind::Image,
            text: String::new(),
            level: None,
            page_no: image.page_no,
            bbox: BBox::new(0.0, 0.0, 0.0, 0.0),
            table: None,
        });
    }

    // ── ⑥ 统计 ───────────────────────────────────────────────────
    let raw_chars = pdf.char_count();
    let structured_chars = elements.iter().map(|e| e.text.chars().count()).sum();
    let heading_count = elements
        .iter()
        .filter(|e| e.kind == ElementKind::Heading)
        .count();
    let paragraph_count = elements
        .iter()
        .filter(|e| e.kind == ElementKind::Paragraph)
        .count();
    let extracted_image_count = images
        .iter()
        .filter(|i| i.status == crate::parser::image::ImageStatus::Extracted)
        .count();

    let page_count = pdf.page_count();
    let has_text_layer = pdf.has_text_layer();

    info!(
        file_name = %pdf.file_name,
        page_count,
        element_count = elements.len(),
        table_count,
        heading_count,
        paragraph_count,
        extracted_image_count,
        has_text_layer,
        "文档结构化解析完成"
    );

    let stats = ParseStats {
        raw_chars,
        structured_chars,
        table_count,
        image_count: images.len(),
        extracted_image_count,
        heading_count,
        paragraph_count,
        has_text_layer,
    };

    Ok(ParsedDocument {
        file_name: pdf.file_name,
        elements,
        images,
        page_count,
        stats,
    })
}

/// 在一页里找出表格。
///
/// ## 思路
///
/// 1. 把**行**按水平位置聚类 —— 同一「视觉行」上的多个单元格
///    组成一条候选表格行
/// 2. 连续的候选行（垂直方向上挨着）组成一个表格区域
/// 3. 区域内的单元格送去 [`reconstruct_table`]
///
/// ## 为什么要先按「视觉行」聚类
///
/// 因为一个单元格里可能有多个词（"Column aware" 是两个词），
/// 但一个词也可能被拆到多个视觉行（长文本换行）。
///
/// 用**行**而不是词来切表，是因为「行」是排版单位：
/// 表格的每一行在版面上就是一条水平的文字带。
///
/// ## ⚠️ 不能对整页调用 [`reconstruct_table`]
///
/// 整页的文字块里既有表格单元格，也有普通段落。
/// 直接对全页做网格重建，会把段落也当成「只有一个单元格的行」，
/// 拼出一个 30 行 4 列的假表格 —— 里面 90% 是空的。
///
/// 所以必须**先定位表格区域**（把连续的「多列行」圈出来），
/// 再对每个区域单独重建。
pub fn detect_tables(
    page: &crate::parser::pdf::Page,
    options: &TableOptions,
) -> (Vec<Table>, Vec<BBox>) {
    let lines = &page.lines;
    if lines.len() < options.min_cells {
        return (Vec::new(), Vec::new());
    }

    // ① 按垂直位置把行聚成「视觉行」。
    let mut visual_rows: Vec<Vec<&PositionedText>> = Vec::new();
    let mut visual_row_centers: Vec<f32> = Vec::new();

    let mut heights: Vec<f32> = lines.iter().map(|l| l.bbox.height()).collect();
    heights.sort_by(f32::total_cmp);
    let median_height = heights[heights.len() / 2].max(4.0);
    let row_tol = median_height * options.row_tolerance;

    let mut by_y: Vec<&PositionedText> = lines.iter().collect();
    by_y.sort_by(|a, b| a.bbox.center_y().total_cmp(&b.bbox.center_y()));

    for line in by_y {
        let cy = line.bbox.center_y();
        if let Some(last) = visual_rows.last_mut()
            && let Some(&rc) = visual_row_centers.last()
            && (cy - rc).abs() <= row_tol
        {
            last.push(line);
            let n = last.len() as f32;
            *visual_row_centers.last_mut().unwrap() = rc + (cy - rc) / n;
            continue;
        }
        visual_rows.push(vec![line]);
        visual_row_centers.push(cy);
    }

    // ② 找出「多列的视觉行」—— 一行里有 ≥2 个水平分离的块。
    let col_tol = median_height * options.column_tolerance;
    let is_multi_column = |row: &[&PositionedText]| -> bool {
        if row.len() < 2 {
            return false;
        }
        let mut lefts: Vec<f32> = row.iter().map(|c| c.bbox.left).collect();
        lefts.sort_by(f32::total_cmp);
        let mut distinct = 1usize;
        for w in lefts.windows(2) {
            if (w[1] - w[0]).abs() > col_tol {
                distinct += 1;
            }
        }
        distinct >= 2
    };

    // ③ 连续的「多列行」组成一个候选表格区域。
    let mut tables = Vec::new();
    let mut consumed: Vec<BBox> = Vec::new();

    let mut current: Vec<&PositionedText> = Vec::new();

    let flush =
        |current: &mut Vec<&PositionedText>, tables: &mut Vec<Table>, consumed: &mut Vec<BBox>| {
            if current.len() < options.min_cells {
                current.clear();
                return;
            }

            let cells: Vec<PositionedText> = current.iter().map(|c| (*c).clone()).collect();

            // 只有「像表格」才还原 —— 否则两栏排版的段落会被误判。
            if crate::parser::table::looks_like_table(&cells)
                && let Some(table) = reconstruct_table(&cells, 0, options)
            {
                consumed.push(table.bbox);
                tables.push(table);
            }
            current.clear();
        };

    for row in &visual_rows {
        if is_multi_column(row) {
            current.extend(row.iter().copied());
        } else {
            flush(&mut current, &mut tables, &mut consumed);
        }
    }
    flush(&mut current, &mut tables, &mut consumed);

    // 过滤掉「伪表格」：那些其实是并排段落的区域。
    // `looks_like_table` 已经在 `flush` 里把关了，
    // 但相邻的多个表格会被合并成一个区域 —— 靠单元格数量再分一次。
    let tables = split_oversized_tables(tables, options);

    debug!(
        page_no = page.page_no,
        table_count = tables.len(),
        consumed_regions = consumed.len(),
        "本页表格检测完成"
    );

    (tables, consumed)
}

/// 把「合并过头的表格」拆回来。
///
/// ## 问题
///
/// 相邻的两张表（中间只有一行间隔）会被「连续多列行」的规则
/// 合并成一个大表格 —— 中间那道间隔看不出来。
///
/// ## 怎么拆
///
/// 看每一行内部的**水平间隙**。两张表之间的间隔处，
/// 左右两个表块的左边缘距离会突然变大（或者出现一个「整行宽度」的块）。
///
/// 这里用一个更简单的判据：按行分组，如果某一行内部的列数
/// 和整张表的列数不一致，说明这里是两张表 —— 在那里切开。
fn split_oversized_tables(tables: Vec<Table>, options: &TableOptions) -> Vec<Table> {
    let mut out = Vec::new();

    for table in tables {
        if table.row_count() < 4 {
            out.push(table);
            continue;
        }

        // 找出列数突变的位置。
        let base_cols = table.column_count();
        let mut splits: Vec<usize> = Vec::new();

        for (ri, row) in table.grid.iter().enumerate() {
            let filled = row.iter().filter(|c| !c.trim().is_empty()).count();
            if filled < base_cols / 2 {
                splits.push(ri);
            }
        }

        if splits.is_empty() {
            out.push(table);
            continue;
        }

        // 按切点拆成多张表。
        let mut start = 0usize;
        for split in splits {
            if split > start {
                let mut piece = table.clone();
                piece.grid = table.grid[start..split].to_vec();
                if piece.grid.len() >= 2 && piece.grid[0].len() >= 2 {
                    piece.header = piece.grid[0].clone();
                    out.push(piece);
                }
            }
            start = split + 1;
        }
        if start < table.grid.len() {
            let mut piece = table.clone();
            piece.grid = table.grid[start..].to_vec();
            if piece.grid.len() >= 2 && piece.grid[0].len() >= 2 {
                piece.header = piece.grid[0].clone();
                out.push(piece);
            }
        }

        let _ = options;
    }

    out
}

/// 压平成纯文本（对照基线）。
///
/// 这是「不解析结构」的做法 —— 直接把所有文字按阅读顺序拼起来。
/// 它的意义是给「结构化到底值多少」提供一个对照组。
pub fn flatten_to_text(pdf: &PdfDocument) -> String {
    let mut out = String::new();
    for page in &pdf.pages {
        for line in &page.lines {
            out.push_str(line.text.trim());
            out.push('\n');
        }
    }
    out
}

/// 渲染成 Markdown。
pub fn to_markdown(doc: &ParsedDocument) -> String {
    let mut out = String::new();
    let mut heading_stack: Vec<(u8, String)> = Vec::new();

    for element in &doc.elements {
        match element.kind {
            ElementKind::Heading => {
                let level = element.level.unwrap_or(1).clamp(1, 6);
                // 标题层级变了 → 把不再适用的旧标题弹栈。
                while heading_stack.last().is_some_and(|(l, _)| *l >= level) {
                    heading_stack.pop();
                }
                out.push_str(&"#".repeat(level as usize));
                out.push(' ');
                out.push_str(&element.text);
                out.push_str("\n\n");
                heading_stack.push((level, element.text.clone()));
            }
            ElementKind::Paragraph => {
                out.push_str(&element.text);
                out.push_str("\n\n");
            }
            ElementKind::Table => {
                out.push_str(&element.text);
                out.push_str("\n\n");
            }
            ElementKind::Image => {
                // 图片在 Markdown 里留一个占位符 —— 具体描述由
                // 视觉模型补上后写进 chunk。
                out.push_str("<!-- image -->\n\n");
            }
            ElementKind::ListItem | ElementKind::Furniture => {}
        }
    }

    out
}

/// 统计文档里出现过的页码集合（用于校验分块的页码引用）。
pub fn page_numbers(doc: &ParsedDocument) -> HashSet<usize> {
    doc.elements.iter().map(|e| e.page_no).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::pdf::Page;
    use crate::parser::pdf::PageSize;

    fn page(lines: Vec<PositionedText>) -> Page {
        Page {
            page_no: 1,
            size: PageSize {
                width: 612.0,
                height: 792.0,
            },
            lines,
            words: Vec::new(),
        }
    }

    fn cell(text: &str, left: f32, top: f32) -> PositionedText {
        PositionedText::new(text, BBox::new(left, top, left + 80.0, top + 14.0))
    }

    #[test]
    fn detects_a_table_and_removes_it_from_the_line_stream() {
        let lines = vec![
            cell("Heading", 72.0, 0.0),
            cell("A", 72.0, 40.0),
            cell("B", 200.0, 40.0),
            cell("1", 72.0, 60.0),
            cell("2", 200.0, 60.0),
        ];

        let (tables, consumed) = detect_tables(&page(lines), &TableOptions::default());

        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].column_count(), 2);
        assert_eq!(consumed.len(), 1);
    }

    #[test]
    fn a_single_column_document_is_not_a_table() {
        let lines = vec![
            cell("Line one", 72.0, 0.0),
            cell("Line two", 72.0, 20.0),
            cell("Line three", 72.0, 40.0),
            cell("Line four", 72.0, 60.0),
        ];

        let (tables, _) = detect_tables(&page(lines), &TableOptions::default());
        assert!(tables.is_empty());
    }

    #[test]
    fn stats_expansion_ratio_is_computed() {
        let stats = ParseStats {
            raw_chars: 100,
            structured_chars: 150,
            ..Default::default()
        };
        assert!((stats.expansion_ratio() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn stats_expansion_ratio_handles_zero() {
        let stats = ParseStats::default();
        assert_eq!(stats.expansion_ratio(), 0.0);
    }
}
