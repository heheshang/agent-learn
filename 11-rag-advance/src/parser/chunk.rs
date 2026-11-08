//! 结构感知分块 —— 把结构化文档变成可检索的块。
//!
//! ## 和 04-rag 的分块有什么不同
//!
//! 04-rag 的 [`StructureAwareSplitter`] 沿着 **Markdown 标题边界**切。
//! 本项目的输入不是 Markdown，而是一棵**元素树**
//! （标题 / 段落 / 表格 / 图片）。
//!
//! 所以这里的切分规则是：
//!
//! ```text
//! 标题 ──► 开一个新的「章节」
//! 段落 ──► 累积到当前章节的缓冲区
//! 表格 ──► **独立成块**（永不与段落混在一起）★
//! 图片 ──► **独立成块**（或者并入相邻块）★
//! ```
//!
//! ## ★ 为什么表格必须独立成块
//!
//! 这是本项目最重要的设计决策。
//!
//! 考虑这张表：
//!
//! ```text
//! | Strategy | Cost |
//! | -------- | ---- |
//! | Flat     | low  |
//! | Layout   | high |
//! ```
//!
//! **方案 A：表格和正文混在一个块里**
//!
//! ```text
//! 表格的每一行会散落在块里的不同位置 —— 一行在块首、
//! 一行在块尾。检索到「Flat 的成本是多少」时，块里的
//! 「low」和「Flat」之间隔着半个块，向量的语义被稀释了。
//! ```
//!
//! **方案 B：表格独立成块**
//!
//! ```text
//! 整张表是一个块，Markdown 表格形式。表头「Strategy | Cost」
//! 和数据行「Flat | low」在同一个块里，而且**位置相邻**。
//! 嵌入模型看到的语义是完整的。
//! ```
//!
//! 方案 B 还有一个额外好处：**表格块可以被整块引用**。
//! 用户问「Flat 策略的成本是多少」，答案可以直接引用这张表。
//!
//! ## ★ 图片为什么也要独立
//!
//! 因为**图片的语义不在图片里，在它的描述里**。
//!
//! 图片的描述（多模态模型生成的）是一段文本，
//! 它需要和其他文本块**语义相邻**才能被检索到。
//! 但它同时需要保留「我是第几页的第几张图」这个出处信息。
//!
//! 所以图片块 = 描述文本 + 出处元数据。

use agent_logging::debug;
use serde::{Deserialize, Serialize};

use crate::parser::geometry::ElementKind;
use crate::parser::pipeline::{DocumentElement, ParsedDocument};

/// 一个可检索的块。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// 块 ID（`{文件名}#{序号}`）。
    pub id: String,

    /// 来源文件名。
    pub source: String,

    /// 块序号。
    pub index: usize,

    /// 标题路径（章节层级）。
    pub heading_path: Vec<String>,

    /// 块正文。
    pub text: String,

    /// 送去做嵌入的文本（含标题路径前缀）。
    pub embedded_text: String,

    /// 块类型。
    pub kind: ElementKind,

    /// 页码。
    pub page_no: usize,
}

impl Chunk {
    /// 一行摘要。
    pub fn summary(&self) -> String {
        let head = if self.heading_path.is_empty() {
            "(无标题)".to_string()
        } else {
            self.heading_path.join(" > ")
        };
        let preview: String = self.text.chars().take(50).collect();
        format!(
            "#{} [{}] p{} {} | {}",
            self.index,
            head,
            self.page_no,
            kind_name(self.kind),
            preview.replace('\n', " ")
        )
    }
}

/// 元素类型的中文名。
pub fn kind_name(kind: ElementKind) -> &'static str {
    match kind {
        ElementKind::Heading => "标题",
        ElementKind::Paragraph => "段落",
        ElementKind::ListItem => "列表",
        ElementKind::Table => "表格",
        ElementKind::Image => "图片",
        ElementKind::Furniture => "页眉页脚",
    }
}

/// 分块配置。
#[derive(Debug, Clone)]
pub struct ChunkOptions {
    /// 段落块的最大 token 数 —— 超过就切开。
    pub max_paragraph_tokens: usize,

    /// 是否给块附加标题路径前缀。
    pub add_heading_context: bool,

    /// 图片描述是否独立成块。
    ///
    /// 关掉的话，图片描述会并入**下一个**文本块。
    pub isolate_images: bool,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            max_paragraph_tokens: 300,
            add_heading_context: true,
            isolate_images: true,
        }
    }
}

/// 把结构化文档切成块。
pub fn chunk_document(doc: &ParsedDocument, options: &ChunkOptions) -> Vec<Chunk> {
    let counter = match rag_04::splitter::TokenCounter::new() {
        Ok(counter) => counter,
        // tokenizer 起不来（几乎不可能）就退化成「按字符估」。
        Err(_) => return chunk_without_tokenizer(doc, options),
    };

    let mut chunks = Vec::new();
    let mut heading_path: Vec<String> = Vec::new();
    let mut index = 0usize;

    // 当前正在累积的段落缓冲区。
    let mut buffer = String::new();
    let mut buffer_page = 0usize;

    let flush_buffer = |buffer: &mut String,
                        buffer_page: &mut usize,
                        heading_path: &Vec<String>,
                        chunks: &mut Vec<Chunk>,
                        index: &mut usize| {
        let text = buffer.trim().to_string();
        if text.is_empty() {
            buffer.clear();
            return;
        }

        let embedded_text = build_embedded_text(&text, heading_path, options);
        chunks.push(Chunk {
            id: format!("{}#{}", doc.file_name, *index),
            source: doc.file_name.clone(),
            index: *index,
            heading_path: heading_path.clone(),
            text,
            embedded_text,
            kind: ElementKind::Paragraph,
            page_no: *buffer_page,
        });
        *index += 1;
        buffer.clear();
    };

    for element in &doc.elements {
        match element.kind {
            ElementKind::Heading => {
                // 标题切换 → 先把缓冲区里的段落冲掉，再换章节。
                flush_buffer(
                    &mut buffer,
                    &mut buffer_page,
                    &heading_path,
                    &mut chunks,
                    &mut index,
                );

                let level = element.level.unwrap_or(1) as usize;
                // 维护标题栈：把层级 ≥ 当前标题的旧标题弹掉。
                while heading_path.len() >= level && !heading_path.is_empty() {
                    heading_path.pop();
                }
                while heading_path.len() < level.saturating_sub(1) {
                    heading_path.push(String::new());
                }
                if heading_path.len() == level.saturating_sub(1) {
                    heading_path.push(element.text.clone());
                } else {
                    heading_path[level - 1] = element.text.clone();
                }
                heading_path.truncate(level);
            }

            ElementKind::Paragraph => {
                if buffer.is_empty() {
                    buffer_page = element.page_no;
                }
                // 超长段落 → 先冲掉，再自己开新块。
                if counter.count(&buffer) + counter.count(&element.text)
                    > options.max_paragraph_tokens
                    && !buffer.is_empty()
                {
                    flush_buffer(
                        &mut buffer,
                        &mut buffer_page,
                        &heading_path,
                        &mut chunks,
                        &mut index,
                    );
                    buffer_page = element.page_no;
                }
                if !buffer.is_empty() {
                    buffer.push(' ');
                }
                buffer.push_str(&element.text);
            }

            ElementKind::Table => {
                // ★ 表格独立成块。
                flush_buffer(
                    &mut buffer,
                    &mut buffer_page,
                    &heading_path,
                    &mut chunks,
                    &mut index,
                );

                let text = element.text.trim().to_string();
                if text.is_empty() {
                    continue;
                }
                let embedded_text = build_embedded_text(&text, &heading_path, options);
                chunks.push(Chunk {
                    id: format!("{}#{}", doc.file_name, index),
                    source: doc.file_name.clone(),
                    index,
                    heading_path: heading_path.clone(),
                    text,
                    embedded_text,
                    kind: ElementKind::Table,
                    page_no: element.page_no,
                });
                index += 1;
            }

            ElementKind::Image => {
                let text = element.text.trim().to_string();
                if text.is_empty() {
                    // 没有描述的图片不单独成块（没有可检索的内容）。
                    continue;
                }

                if options.isolate_images {
                    flush_buffer(
                        &mut buffer,
                        &mut buffer_page,
                        &heading_path,
                        &mut chunks,
                        &mut index,
                    );
                    let embedded_text = build_embedded_text(&text, &heading_path, options);
                    chunks.push(Chunk {
                        id: format!("{}#{}", doc.file_name, index),
                        source: doc.file_name.clone(),
                        index,
                        heading_path: heading_path.clone(),
                        text,
                        embedded_text,
                        kind: ElementKind::Image,
                        page_no: element.page_no,
                    });
                    index += 1;
                } else {
                    // 并入当前缓冲区。
                    if buffer.is_empty() {
                        buffer_page = element.page_no;
                    }
                    if !buffer.is_empty() {
                        buffer.push(' ');
                    }
                    buffer.push_str(&text);
                }
            }

            ElementKind::ListItem | ElementKind::Furniture => {}
        }
    }

    // 收尾。
    flush_buffer(
        &mut buffer,
        &mut buffer_page,
        &heading_path,
        &mut chunks,
        &mut index,
    );

    debug!(
        file_name = %doc.file_name,
        chunk_count = chunks.len(),
        "结构感知分块完成"
    );

    chunks
}

/// 构造送去做嵌入的文本。
///
/// 标题路径前缀是提升召回最有效的单个技巧 ——
/// 同样一句话在不同章节里含义完全不同。
fn build_embedded_text(text: &str, heading_path: &[String], options: &ChunkOptions) -> String {
    if !options.add_heading_context || heading_path.is_empty() {
        return text.to_string();
    }

    let context = heading_path
        .iter()
        .filter(|h| !h.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" > ");

    if context.is_empty() {
        return text.to_string();
    }

    format!("# {context}\n\n{text}")
}

/// 不带 tokenizer 的退化实现（按字符数估算 token）。
fn chunk_without_tokenizer(doc: &ParsedDocument, options: &ChunkOptions) -> Vec<Chunk> {
    let max_chars = options.max_paragraph_tokens * 2;

    let mut chunks = Vec::new();
    let mut heading_path: Vec<String> = Vec::new();
    let mut index = 0usize;
    let mut buffer = String::new();
    let mut buffer_page = 0usize;

    let push = |text: String,
                kind: ElementKind,
                page_no: usize,
                heading_path: &Vec<String>,
                chunks: &mut Vec<Chunk>,
                index: &mut usize| {
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        let embedded_text = build_embedded_text(&text, heading_path, options);
        chunks.push(Chunk {
            id: format!("{}#{}", doc.file_name, *index),
            source: doc.file_name.clone(),
            index: *index,
            heading_path: heading_path.clone(),
            text,
            embedded_text,
            kind,
            page_no,
        });
        *index += 1;
    };

    for element in &doc.elements {
        match element.kind {
            ElementKind::Heading => {
                if !buffer.trim().is_empty() {
                    push(
                        std::mem::take(&mut buffer),
                        ElementKind::Paragraph,
                        buffer_page,
                        &heading_path,
                        &mut chunks,
                        &mut index,
                    );
                }
                let level = element.level.unwrap_or(1) as usize;
                heading_path.truncate(level.saturating_sub(1));
                heading_path.push(element.text.clone());
            }
            ElementKind::Paragraph => {
                if buffer.is_empty() {
                    buffer_page = element.page_no;
                }
                if buffer.chars().count() > max_chars {
                    push(
                        std::mem::take(&mut buffer),
                        ElementKind::Paragraph,
                        buffer_page,
                        &heading_path,
                        &mut chunks,
                        &mut index,
                    );
                    buffer_page = element.page_no;
                }
                if !buffer.is_empty() {
                    buffer.push(' ');
                }
                buffer.push_str(&element.text);
            }
            ElementKind::Table | ElementKind::Image => {
                if !buffer.trim().is_empty() {
                    push(
                        std::mem::take(&mut buffer),
                        ElementKind::Paragraph,
                        buffer_page,
                        &heading_path,
                        &mut chunks,
                        &mut index,
                    );
                }
                push(
                    element.text.clone(),
                    element.kind,
                    element.page_no,
                    &heading_path,
                    &mut chunks,
                    &mut index,
                );
            }
            _ => {}
        }
    }

    if !buffer.trim().is_empty() {
        push(
            buffer,
            ElementKind::Paragraph,
            buffer_page,
            &heading_path,
            &mut chunks,
            &mut index,
        );
    }

    chunks
}

/// 统计各类块的数量。
///
/// 返回 `(类型名, 数量)` —— 名字用中文，便于直接展示。
pub fn count_by_kind(chunks: &[Chunk]) -> Vec<(String, usize)> {
    use std::collections::BTreeMap;

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for chunk in chunks {
        *counts.entry(kind_name(chunk.kind).to_string()).or_insert(0) += 1;
    }
    counts.into_iter().collect()
}

/// 渲染成可读的报告（给 demo 用）。
pub fn render_report(doc: &ParsedDocument, chunks: &[Chunk]) -> String {
    let mut out = String::new();

    out.push_str(&format!("文件: {}\n", doc.file_name));
    out.push_str(&format!("页数: {}\n", doc.page_count));
    out.push_str(&format!("统计: {}\n\n", doc.stats.summary()));

    out.push_str("── 块 ──\n");
    for chunk in chunks {
        out.push_str(&format!("  {}\n", chunk.summary()));
    }

    out
}

/// 便捷函数：解析 → 分块。
pub fn parse_and_chunk(doc: &ParsedDocument, options: &ChunkOptions) -> Vec<Chunk> {
    chunk_document(doc, options)
}

/// 供 `DocumentElement` 使用的辅助：判断元素是否为空。
pub fn is_empty_element(element: &DocumentElement) -> bool {
    element.text.trim().is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::geometry::BBox;
    use crate::parser::pipeline::ParseStats;

    fn element(
        kind: ElementKind,
        text: &str,
        level: Option<u8>,
        page_no: usize,
    ) -> DocumentElement {
        DocumentElement {
            kind,
            text: text.to_string(),
            level,
            page_no,
            bbox: BBox::new(0.0, 0.0, 100.0, 10.0),
            table: None,
        }
    }

    fn doc(elements: Vec<DocumentElement>) -> ParsedDocument {
        ParsedDocument {
            file_name: "test.pdf".to_string(),
            elements,
            images: Vec::new(),
            page_count: 1,
            stats: ParseStats::default(),
        }
    }

    #[test]
    fn tables_become_their_own_chunks() {
        let document = doc(vec![
            element(ElementKind::Heading, "章节", Some(1), 1),
            element(ElementKind::Paragraph, "正文段落。", None, 1),
            element(
                ElementKind::Table,
                "| A | B |\n| --- | --- |\n| 1 | 2 |",
                None,
                1,
            ),
            element(ElementKind::Paragraph, "另一段。", None, 1),
        ]);

        let chunks = chunk_document(&document, &ChunkOptions::default());

        let table_chunks: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| c.kind == ElementKind::Table)
            .collect();

        assert_eq!(table_chunks.len(), 1);
        // 表格块里必须同时有表头和数据行 —— 否则列关系丢了。
        assert!(table_chunks[0].text.contains("A | B"));
        assert!(table_chunks[0].text.contains("1 | 2"));
    }

    #[test]
    fn a_table_never_mixes_with_a_paragraph() {
        let document = doc(vec![
            element(ElementKind::Paragraph, "前面的段落。", None, 1),
            element(ElementKind::Table, "| A |\n| --- |\n| 1 |", None, 1),
            element(ElementKind::Paragraph, "后面的段落。", None, 1),
        ]);

        let chunks = chunk_document(&document, &ChunkOptions::default());

        for chunk in &chunks {
            if chunk.kind == ElementKind::Table {
                assert!(!chunk.text.contains("段落"), "表格块里混进了段落");
            } else {
                assert!(!chunk.text.contains('|'), "段落块里混进了表格");
            }
        }
    }

    #[test]
    fn heading_path_is_tracked_and_prefixed() {
        let document = doc(vec![
            element(ElementKind::Heading, "顶层", Some(1), 1),
            element(ElementKind::Heading, "子节", Some(2), 1),
            element(ElementKind::Paragraph, "内容。", None, 1),
        ]);

        let chunks = chunk_document(&document, &ChunkOptions::default());
        let last = chunks.last().unwrap();

        assert_eq!(last.heading_path, vec!["顶层", "子节"]);
        assert!(last.embedded_text.starts_with("# 顶层 > 子节"));
    }

    #[test]
    fn a_new_h1_flushes_the_previous_section() {
        let document = doc(vec![
            element(ElementKind::Heading, "第一章", Some(1), 1),
            element(ElementKind::Paragraph, "第一章的内容。", None, 1),
            element(ElementKind::Heading, "第二章", Some(1), 1),
            element(ElementKind::Paragraph, "第二章的内容。", None, 1),
        ]);

        let chunks = chunk_document(&document, &ChunkOptions::default());

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading_path, vec!["第一章"]);
        assert_eq!(chunks[1].heading_path, vec!["第二章"]);
    }

    #[test]
    fn images_without_descriptions_are_dropped() {
        let document = doc(vec![
            element(ElementKind::Heading, "章节", Some(1), 1),
            element(ElementKind::Image, "", None, 1),
        ]);

        let chunks = chunk_document(&document, &ChunkOptions::default());
        assert!(chunks.iter().all(|c| c.kind != ElementKind::Image));
    }

    #[test]
    fn images_with_descriptions_become_their_own_chunks() {
        let document = doc(vec![
            element(ElementKind::Heading, "章节", Some(1), 1),
            element(ElementKind::Paragraph, "文字。", None, 1),
            element(
                ElementKind::Image,
                "一张柱状图，展示三种策略的耗时。",
                None,
                1,
            ),
        ]);

        let chunks = chunk_document(&document, &ChunkOptions::default());

        let image_chunks: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| c.kind == ElementKind::Image)
            .collect();
        assert_eq!(image_chunks.len(), 1);
        assert!(image_chunks[0].text.contains("柱状图"));
    }
}
