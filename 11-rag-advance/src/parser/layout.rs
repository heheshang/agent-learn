//! 版面分析 —— 把一页的文字块排成**有结构的文档**。
//!
//! ## 这一层解决什么
//!
//! 表格解析给出的是「哪些字在哪个格子里」，但没说**谁是谁的标题**。
//! 没有标题层级，结构感知分块就无从谈起 ——
//! 04-rag 的 [`StructureAwareSplitter`] 是沿着标题边界切的。
//!
//! ## 标题是怎么认出来的
//!
//! 这里有一个**不依赖版面模型**的办法：**行高**。
//!
//! PDF 的文字层给每个文字块一个矩形框。**行高和字号成正比**
//! （单倍行距下 `行高 ≈ 字号 × 1.2`），所以：
//!
//! - 标题的行框比正文高
//! - 标题通常不以句号结尾
//! - 标题通常比较短
//!
//! 判定规则：
//!
//! ```text
//! ① 候选：行高 > 正文行高 × 阈值，且长度 < 阈值
//! ② 排除：像正文（以句末标点结尾 / 太长）
//! ③ 分级：按行高排序，行高越大层级越高
//! ```
//!
//! ## ⚠️ 顺序很重要：先分类，再合并
//!
//! 如果先「把行合并成段」再判断标题，标题信号就丢了 ——
//! 因为合并后的段落框是所有行的**并集**，高度暴涨，
//! 每个段落看起来都像标题。
//!
//! 所以正确的顺序是：
//!
//! ```text
//! 行 ──► 分类（标题 / 正文）──► 合并（只合并连续的正文行）
//! ```
//!
//! ## 为什么不用版面模型
//!
//! docling 的完整 ML 管线能识别标题，但那要跑 ONNX 版面模型
//! （几十 MB）+ 渲染页面。本项目走纯几何路线：
//!
//! - 快（毫秒级）
//! - 零模型依赖
//! - 在**有文字层的 PDF**（绝大多数企业文档）上效果够用
//!
//! 代价是：**扫描件**（没有文字层）完全处理不了 —— 那种 PDF 需要 OCR，
//! 属于另一条技术路线。

use serde::{Deserialize, Serialize};

use crate::parser::geometry::{BBox, ElementKind, PositionedText};

/// 标题识别参数。
#[derive(Debug, Clone)]
pub struct LayoutOptions {
    /// 标题的行高至少要比正文大多少倍。
    pub heading_scale: f32,

    /// 标题的最大字符数。
    pub max_heading_chars: usize,

    /// 正文的最大字符数 —— 超过就一定不是标题。
    pub max_paragraph_chars: usize,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            heading_scale: 1.15,
            max_heading_chars: 80,
            max_paragraph_chars: 2000,
        }
    }
}

/// 一个带字号的文字块。
#[derive(Debug, Clone)]
pub struct SizedText {
    /// 文字。
    pub text: String,
    /// 位置。
    pub bbox: BBox,
    /// 字号（磅）。没有信息时用 `None`。
    pub font_size: Option<f32>,
}

/// 版面分析的结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layout {
    /// 排好序的元素。
    pub elements: Vec<Element>,

    /// 正文的典型行高。
    pub body_line_height: f32,
}

/// 版面元素。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    /// 类型。
    pub kind: ElementKind,
    /// 位置。
    pub bbox: BBox,
    /// 文本。
    pub text: String,
    /// 标题层级。
    pub level: Option<u8>,
}

impl Element {
    /// 构造标题元素。
    pub fn heading(text: impl Into<String>, bbox: BBox, level: u8) -> Self {
        Self {
            kind: ElementKind::Heading,
            bbox,
            text: text.into(),
            level: Some(level),
        }
    }

    /// 构造段落元素。
    pub fn paragraph(text: impl Into<String>, bbox: BBox) -> Self {
        Self {
            kind: ElementKind::Paragraph,
            bbox,
            text: text.into(),
            level: None,
        }
    }
}

/// 一行的分类结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    /// 标题。
    Heading,
    /// 正文。
    Body,
}

/// 分析一页的版面。
///
/// `lines` 必须已经按**阅读顺序**排好（从上到下）。
///
/// ## 算法
///
/// ```text
/// ① 估算正文行高（众数）
/// ② 逐行分类：标题 / 正文
/// ③ 连续的正文行合并成段；标题独立成段
/// ④ 给标题分级（按行高降序）
/// ```
pub fn analyze_page(lines: &[PositionedText], options: &LayoutOptions) -> Layout {
    if lines.is_empty() {
        return Layout {
            elements: Vec::new(),
            body_line_height: 0.0,
        };
    }

    let body_line_height = estimate_body_line_height(lines);
    let threshold = body_line_height * options.heading_scale;

    // ① 逐行分类。
    let kinds: Vec<LineKind> = lines
        .iter()
        .map(|line| classify_line(line, threshold, options))
        .collect();

    // ② 给标题分级：按行高降序去重，最大的 = 1 级。
    let mut heading_heights: Vec<f32> = lines
        .iter()
        .zip(kinds.iter())
        .filter(|(_, kind)| **kind == LineKind::Heading)
        .map(|(line, _)| line.bbox.height())
        .collect();
    heading_heights.sort_by(|a, b| b.total_cmp(a));
    heading_heights.dedup_by(|a, b| (*a - *b).abs() < 0.5);

    let level_of = |height: f32| -> u8 {
        heading_heights
            .iter()
            .position(|&h| (h - height).abs() < 0.5)
            .map(|i| (i + 1).min(6) as u8)
            .unwrap_or(6)
    };

    // ③ 分组成元素。
    let mut elements: Vec<Element> = Vec::new();
    let mut buffer: Vec<&PositionedText> = Vec::new();

    let flush = |buffer: &mut Vec<&PositionedText>, elements: &mut Vec<Element>| {
        if buffer.is_empty() {
            return;
        }

        let text = buffer
            .iter()
            .map(|l| l.text.trim())
            .collect::<Vec<_>>()
            .join(" ");

        let bbox = buffer
            .iter()
            .map(|l| l.bbox)
            .reduce(|a, b| a.union(&b))
            .unwrap_or(BBox::new(0.0, 0.0, 0.0, 0.0));

        buffer.clear();

        if text.chars().count() > options.max_paragraph_chars {
            // 超长的块多半是解析噪声 —— 丢掉。
            return;
        }

        elements.push(Element::paragraph(text, bbox));
    };

    for (line, kind) in lines.iter().zip(kinds.iter()) {
        match kind {
            LineKind::Heading => {
                // 标题独立成段 —— 先把缓冲区冲掉。
                flush(&mut buffer, &mut elements);

                let text = line.text.trim();
                if text.is_empty() {
                    continue;
                }
                let level = level_of(line.bbox.height());
                elements.push(Element::heading(text, line.bbox, level));
            }
            LineKind::Body => {
                // 连续的行合并 —— 但如果上一行是段落结尾（句号），
                // 就另起一段。
                let starts_new_paragraph = buffer.last().is_some_and(|prev| {
                    prev.text
                        .trim_end()
                        .ends_with(['.', '。', '?', '？', '!', '！'])
                });

                if starts_new_paragraph {
                    flush(&mut buffer, &mut elements);
                }

                buffer.push(line);
            }
        }
    }
    flush(&mut buffer, &mut elements);

    Layout {
        elements,
        body_line_height,
    }
}

/// 估算正文的典型行高。
///
/// ## 做法：取**众数**的行高
///
/// 不能取平均 —— 一份文档里行高分布是：
///
/// ```text
/// 标题 16pt × 5 处
/// 小标题 13pt × 8 处
/// 正文 10pt × 200 处   ← 主体
/// 脚注 8pt × 20 处
/// ```
///
/// 平均值会被大量正文拉到 10 附近（这倒是想要的），
/// 但如果文档里表格很多、表格文字是 7pt，平均值会掉到 8.5，
/// 于是 10pt 的正文会被误判成标题。
///
/// 所以取**众数**（出现次数最多的行高），它一定是正文。
fn estimate_body_line_height(lines: &[PositionedText]) -> f32 {
    let mut buckets: Vec<(f32, usize)> = Vec::new();

    for line in lines {
        // 按 0.5pt 分桶 —— 小数位没有意义。
        let bucket = (line.bbox.height() * 2.0).round() / 2.0;
        match buckets.iter_mut().find(|(b, _)| (*b - bucket).abs() < 0.01) {
            Some((_, count)) => *count += 1,
            None => buckets.push((bucket, 1)),
        }
    }

    if buckets.is_empty() {
        return 12.0;
    }

    // 众数；并列时取**较大的**（正文通常比脚注大）。
    buckets.sort_by(|a, b| a.0.total_cmp(&b.0));
    let max_count = buckets.iter().map(|(_, c)| *c).max().unwrap_or(0);
    let candidates: Vec<f32> = buckets
        .iter()
        .filter(|(_, c)| *c == max_count)
        .map(|(s, _)| *s)
        .collect();

    candidates.into_iter().fold(0.0f32, f32::max)
}

/// 判断一行是标题还是正文。
fn classify_line(line: &PositionedText, threshold: f32, options: &LayoutOptions) -> LineKind {
    let text = line.text.trim();
    let char_len = text.chars().count();

    if char_len > options.max_heading_chars {
        return LineKind::Body;
    }

    if looks_like_sentence(text) {
        return LineKind::Body;
    }

    if line.bbox.height() >= threshold {
        LineKind::Heading
    } else {
        LineKind::Body
    }
}

/// 判断一段文字**像不像完整的句子**（即不像标题）。
///
/// 标题的特征是「短且不以句号结尾」。
/// 中英文的句末标点都要检查。
fn looks_like_sentence(text: &str) -> bool {
    let trimmed = text.trim_end();

    // 以句号 / 问号 / 感叹号 / 分号结尾 → 是句子。
    for ending in ['.', '?', '!', ';', '。', '？', '！', '；'] {
        if trimmed.ends_with(ending) {
            return true;
        }
    }

    // 结尾是逗号 → 也是一个长句的片段。
    if trimmed.ends_with(',') || trimmed.ends_with('，') {
        return true;
    }

    false
}

/// 把行合并成段落（**不识别标题**的简化版）。
///
/// 只在「已经确定这些行都是正文」时用。
/// 标题识别请用 [`analyze_page`] —— 它先分类再合并。
pub fn merge_lines(
    lines: &[PositionedText],
    line_height: f32,
    _options: &LayoutOptions,
) -> Vec<SizedText> {
    let mut merged: Vec<SizedText> = Vec::new();

    for line in lines {
        let text = line.text.trim();
        if text.is_empty() {
            continue;
        }

        let should_merge = match merged.last() {
            None => false,
            Some(prev) => {
                // 行间距明显变大 → 新段。
                let gap = line.bbox.top - prev.bbox.bottom;
                let gap_large = gap > line_height * 0.8;
                let prev_ends_sentence = prev
                    .text
                    .trim_end()
                    .ends_with(['.', '。', '?', '？', '!', '！']);

                !gap_large && !prev_ends_sentence
            }
        };

        if should_merge {
            let prev = merged.last_mut().unwrap();
            prev.text.push(' ');
            prev.text.push_str(text);
            prev.bbox = prev.bbox.union(&line.bbox);
        } else {
            merged.push(SizedText {
                text: text.to_string(),
                bbox: line.bbox,
                font_size: None,
            });
        }
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, height: f32, top: f32) -> PositionedText {
        PositionedText::new(text, BBox::new(72.0, top, 400.0, top + height))
    }

    #[test]
    fn body_line_height_is_the_mode_not_the_mean() {
        // 3 个标题（16pt），20 个正文（10pt），5 个脚注（8pt）。
        let mut lines = Vec::new();
        for i in 0..3 {
            lines.push(line(&format!("标题{i}"), 16.0, i as f32 * 30.0));
        }
        for i in 0..20 {
            lines.push(line(&format!("正文{i}"), 10.0, 100.0 + i as f32 * 15.0));
        }
        for i in 0..5 {
            lines.push(line(&format!("脚注{i}"), 8.0, 500.0 + i as f32 * 12.0));
        }

        let layout = analyze_page(&lines, &LayoutOptions::default());

        assert!(
            (layout.body_line_height - 10.0).abs() < 0.01,
            "正文行高应该是众数 10，实际 {}",
            layout.body_line_height
        );
    }

    #[test]
    fn headings_are_detected_and_graded() {
        let lines = vec![
            line("Advanced Data Processing", 16.0, 0.0),
            line("1. Why flat text fails", 13.0, 40.0),
            line("Body paragraph that is long enough.", 10.0, 80.0),
            line("Body paragraph number two here.", 10.0, 100.0),
        ];

        let layout = analyze_page(&lines, &LayoutOptions::default());

        let headings: Vec<&Element> = layout
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Heading)
            .collect();

        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].level, Some(1));
        assert_eq!(headings[1].level, Some(2));
    }

    #[test]
    fn a_sentence_ending_in_a_period_is_not_a_heading() {
        // 行高很大，但以句号结尾 —— 是句子不是标题。
        let lines = vec![
            line(
                "This is a long sentence that ends with a period.",
                20.0,
                0.0,
            ),
            line("Body", 10.0, 30.0),
        ];
        let layout = analyze_page(&lines, &LayoutOptions::default());
        assert!(
            !layout
                .elements
                .iter()
                .any(|e| e.kind == ElementKind::Heading)
        );
    }

    #[test]
    fn a_very_long_block_is_never_a_heading() {
        let long = "x".repeat(200);
        let lines = vec![line(&long, 20.0, 0.0), line("Body", 10.0, 30.0)];
        let layout = analyze_page(&lines, &LayoutOptions::default());
        assert!(
            !layout
                .elements
                .iter()
                .any(|e| e.kind == ElementKind::Heading)
        );
    }

    #[test]
    fn body_lines_merge_into_one_paragraph() {
        let lines = vec![
            line("First line of a paragraph", 10.0, 0.0),
            line("second line continues it", 10.0, 12.0),
        ];

        let layout = analyze_page(&lines, &LayoutOptions::default());

        let paragraphs: Vec<&Element> = layout
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Paragraph)
            .collect();

        assert_eq!(paragraphs.len(), 1);
        assert!(paragraphs[0].text.contains("continues it"));
    }

    #[test]
    fn a_heading_breaks_the_paragraph() {
        let lines = vec![
            line("Body before the heading", 10.0, 0.0),
            line("A Heading", 16.0, 30.0),
            line("Body after the heading", 10.0, 60.0),
        ];

        let layout = analyze_page(&lines, &LayoutOptions::default());

        // 三段：正文 / 标题 / 正文。
        assert_eq!(layout.elements.len(), 3);
        assert_eq!(layout.elements[1].kind, ElementKind::Heading);
        assert_eq!(layout.elements[0].kind, ElementKind::Paragraph);
        assert_eq!(layout.elements[2].kind, ElementKind::Paragraph);
    }

    #[test]
    fn a_sentence_end_starts_a_new_paragraph() {
        let lines = vec![
            line("First sentence.", 10.0, 0.0),
            line("Second sentence.", 10.0, 12.0),
        ];
        let layout = analyze_page(&lines, &LayoutOptions::default());
        let paragraphs = layout
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Paragraph)
            .count();
        assert_eq!(paragraphs, 2);
    }

    #[test]
    fn merge_lines_still_merges_body_lines() {
        let lines = vec![
            PositionedText::new(
                "First line of a paragraph",
                BBox::new(0.0, 0.0, 100.0, 12.0),
            ),
            PositionedText::new(
                "second line continues it",
                BBox::new(0.0, 12.0, 100.0, 24.0),
            ),
            PositionedText::new("A new paragraph.", BBox::new(0.0, 60.0, 100.0, 72.0)),
        ];

        let merged = merge_lines(&lines, 12.0, &LayoutOptions::default());

        assert_eq!(merged.len(), 2);
        assert!(merged[0].text.contains("continues it"));
        assert_eq!(merged[1].text, "A new paragraph.");
    }
}
