//! 集成测试 —— 从「真实的 PDF 文件」到「可检索的块」。
//!
//! ## 和单元测试的分工
//!
//! 单元测试（`src/**/tests`）测的是**算法**：给一组人造的坐标，
//! 看表格能不能还原、标题能不能认出来。
//!
//! 集成测试测的是**链路**：真的造一个 PDF，真的解析它，
//! 看端到端的产出对不对。
//!
//! ## 为什么自己造 PDF
//!
//! 因为项目里不能带二进制测试文件（git diff 看不出来、没法 review）。
//! 用 Python 手写 PDF 语法 —— 只依赖 zlib（Python 标准库）。
//!
//! 但测试不能依赖 Python —— CI 上不一定有。所以这里用 **Rust** 造 PDF：
//! `tests/fixtures.rs` 里有一个极简的 PDF 构造器。

mod fixtures;

use rag_11::config::AppConfig;
use rag_11::parser::chunk::{ChunkOptions, chunk_document};
use rag_11::parser::geometry::ElementKind;
use rag_11::parser::pipeline::{detect_tables, parse_file, to_markdown};
use rag_11::parser::table::TableOptions;

/// 指向临时目录的配置。
///
/// 测试不能写进项目目录 —— 并行跑测试时会互相踩。
fn test_config(root: &std::path::Path) -> AppConfig {
    AppConfig {
        data_dir: root.to_path_buf(),
        asset_dir: root.join("assets"),
        ..AppConfig::default()
    }
}

/// 造一份带标题 + 表格 + 段落的 PDF。
///
/// 和 `data/make_complex_pdf.py` 的输出结构一致，
/// 但用 Rust 造，测试不依赖 Python。
fn build_test_pdf() -> Vec<u8> {
    let mut ops: Vec<String> = Vec::with_capacity(16);

    // 标题（16pt）
    ops.push("BT /F1 16 Tf 72 720 Td (Test Document Title) Tj ET".to_string());
    // 小标题（13pt）
    ops.push("BT /F1 13 Tf 72 690 Td (Section One) Tj ET".to_string());
    // 段落（10pt）—— 一个**跨两行**的段落：只有最后一行以句号结尾。
    //
    // ⚠️ 这一条很重要：真实的 PDF 里，一个段落的**中间行**不以句号结尾
    // （只有最后一行有）。如果中间行也带句号，
    // 「句号 = 段落边界」的启发式就会把它切开 —— 那是 fixture 的问题，
    // 不是解析器的问题。
    ops.push(
        "BT /F1 10 Tf 72 670 Td (This is a body paragraph that should be merged) Tj ET".to_string(),
    );
    ops.push("BT /F1 10 Tf 72 655 Td (across two visual lines into one.) Tj ET".to_string());

    // 表格：4 列 × 3 行（含表头）
    let cols = [72.0, 220.0, 370.0, 520.0];
    let rows = [600.0f32, 580.0, 560.0];
    let table = [
        ["Name", "Type", "Cost", "Notes"],
        ["Flat", "none", "low", "fast"],
        ["Layout", "full", "high", "slow"],
    ];

    // 横线
    for (i, _) in rows.iter().enumerate() {
        let y = rows[i];
        ops.push(format!("{} {} m {} {} l S", cols[0], y, cols[3], y));
    }
    // 底边
    ops.push(format!(
        "{} {} m {} {} l S",
        cols[0],
        rows[2] - 20.0,
        cols[3],
        rows[2] - 20.0
    ));
    // 竖线
    for col in cols {
        ops.push(format!(
            "{} {} m {} {} l S",
            col,
            rows[0],
            col,
            rows[2] - 20.0
        ));
    }
    // 单元格文字
    for (ri, row) in table.iter().enumerate() {
        for (ci, text) in row.iter().enumerate() {
            ops.push(format!(
                "BT /F1 10 Tf {} {} Td ({}) Tj ET",
                cols[ci] + 4.0,
                rows[ri] - 12.0,
                text
            ));
        }
    }

    fixtures::pdf_from_ops(&ops)
}

#[test]
fn parses_headings_paragraphs_and_tables() {
    let dir = tempfile::tempdir().unwrap();
    let pdf_path = dir.path().join("test.pdf");
    std::fs::write(&pdf_path, build_test_pdf()).unwrap();

    let config = test_config(dir.path());

    let doc = parse_file(&pdf_path, &config).unwrap();

    // 文本层存在。
    assert!(doc.stats.has_text_layer);
    assert_eq!(doc.page_count, 1);

    // 标题识别出来了。
    assert!(
        doc.stats.heading_count >= 2,
        "应该识别出至少 2 个标题，实际 {}",
        doc.stats.heading_count
    );

    // 标题文本正确。
    let heading_texts: Vec<&str> = doc
        .elements
        .iter()
        .filter(|e| e.kind == ElementKind::Heading)
        .map(|e| e.text.as_str())
        .collect();
    assert!(
        heading_texts
            .iter()
            .any(|t| t.contains("Test Document Title")),
        "标题列表: {heading_texts:?}"
    );

    // 表格识别出来了，而且列数正确。
    assert_eq!(doc.stats.table_count, 1, "应该识别出 1 张表");
    let table = doc
        .elements
        .iter()
        .find(|e| e.kind == ElementKind::Table)
        .expect("应该有表格元素");

    let data = table.table.as_ref().unwrap();
    assert_eq!(data.grid.len(), 3, "3 行（含表头）");
    assert_eq!(data.grid[0].len(), 4, "4 列");
    assert_eq!(data.grid[0][0], "Name");
    assert_eq!(data.grid[1][0], "Flat");
    assert_eq!(data.grid[2][3], "slow");
}

#[test]
fn table_markdown_preserves_the_column_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let pdf_path = dir.path().join("test.pdf");
    std::fs::write(&pdf_path, build_test_pdf()).unwrap();

    let config = test_config(dir.path());

    let doc = parse_file(&pdf_path, &config).unwrap();
    let markdown = to_markdown(&doc);

    // 关键断言：表头和数据在同一行 —— 列对应关系没丢。
    assert!(
        markdown.contains("| Name | Type | Cost | Notes |"),
        "Markdown 里应该有完整的表头行:\n{markdown}"
    );
    assert!(
        markdown.contains("| Flat | none | low | fast |"),
        "Markdown 里应该有完整的数据行:\n{markdown}"
    );
}

#[test]
fn paragraphs_are_merged_into_one_element() {
    let dir = tempfile::tempdir().unwrap();
    let pdf_path = dir.path().join("test.pdf");
    std::fs::write(&pdf_path, build_test_pdf()).unwrap();

    let config = test_config(dir.path());

    let doc = parse_file(&pdf_path, &config).unwrap();

    let paragraph = doc
        .elements
        .iter()
        .find(|e| e.kind == ElementKind::Paragraph && e.text.contains("body paragraph"))
        .expect("应该有段落元素");

    // 两行合并成一段。
    assert!(
        paragraph.text.contains("two visual lines"),
        "两行应该合并成一段，实际: {}",
        paragraph.text
    );
}

#[test]
fn tables_become_their_own_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let pdf_path = dir.path().join("test.pdf");
    std::fs::write(&pdf_path, build_test_pdf()).unwrap();

    let config = test_config(dir.path());

    let doc = parse_file(&pdf_path, &config).unwrap();
    let chunks = chunk_document(&doc, &ChunkOptions::default());

    let table_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.kind == ElementKind::Table)
        .collect();

    assert_eq!(table_chunks.len(), 1);

    // 表格块里必须有表头 + 数据行，且分属不同的「格」。
    let text = &table_chunks[0].text;
    assert!(text.contains("| Name | Type | Cost | Notes |"));
    assert!(text.contains("| Flat | none | low | fast |"));
}

#[test]
fn table_chunks_carry_their_section_heading() {
    // 回归测试：表格必须待在**自己的章节**里。
    //
    // 曾经的 bug：解析时先把所有表格 push 进元素列表，
    // 再 push 文字。结果表格块的 `heading_path` 是空的 ——
    // 因为 push 表格的时候还没遇到任何标题。
    //
    // 表格在文档里的位置是「第二章的那个表」，
    // 标题路径必须反映这一点。
    let dir = tempfile::tempdir().unwrap();
    let pdf_path = dir.path().join("test.pdf");
    std::fs::write(&pdf_path, build_test_pdf()).unwrap();

    let config = test_config(dir.path());

    let doc = parse_file(&pdf_path, &config).unwrap();
    let chunks = chunk_document(&doc, &ChunkOptions::default());

    let table_chunk = chunks
        .iter()
        .find(|c| c.kind == ElementKind::Table)
        .expect("应该有表格块");

    assert!(
        !table_chunk.heading_path.is_empty(),
        "表格块应该带标题路径，实际是空的"
    );
    assert!(
        table_chunk
            .heading_path
            .iter()
            .any(|h| h.contains("Section One")),
        "表格块应该在「Section One」章节里，实际: {:?}",
        table_chunk.heading_path
    );
}

#[test]
fn chunks_carry_their_heading_path() {
    let dir = tempfile::tempdir().unwrap();
    let pdf_path = dir.path().join("test.pdf");
    std::fs::write(&pdf_path, build_test_pdf()).unwrap();

    let config = test_config(dir.path());

    let doc = parse_file(&pdf_path, &config).unwrap();
    let chunks = chunk_document(&doc, &ChunkOptions::default());

    // 至少有一个块带有标题路径。
    assert!(
        chunks.iter().any(|c| !c.heading_path.is_empty()),
        "应该有块携带标题路径"
    );

    // 嵌入文本带标题前缀。
    let with_context = chunks
        .iter()
        .find(|c| c.embedded_text.starts_with("# "))
        .expect("应该有块带标题前缀");
    assert!(with_context.embedded_text.contains("Test Document Title"));
}

#[test]
fn detection_is_stable_across_runs() {
    // 同一个文件解析两次，结果必须一样 —— 否则索引不可复现。
    let dir = tempfile::tempdir().unwrap();
    let pdf_path = dir.path().join("test.pdf");
    std::fs::write(&pdf_path, build_test_pdf()).unwrap();

    let config = test_config(dir.path());

    let first = parse_file(&pdf_path, &config).unwrap();
    let second = parse_file(&pdf_path, &config).unwrap();

    assert_eq!(first.stats.table_count, second.stats.table_count);
    assert_eq!(first.elements.len(), second.elements.len());
    assert_eq!(first.stats.raw_chars, second.stats.raw_chars);
}

#[test]
fn detect_tables_does_not_treat_paragraphs_as_tables() {
    // 只有正文、没有表格的 PDF —— 不应该检出表格。
    let ops = vec![
        "BT /F1 10 Tf 72 720 Td (Only plain prose in this document.) Tj ET".to_string(),
        "BT /F1 10 Tf 72 700 Td (A second paragraph with no grid at all.) Tj ET".to_string(),
    ];
    let dir = tempfile::tempdir().unwrap();
    let pdf_path = dir.path().join("prose.pdf");
    std::fs::write(&pdf_path, fixtures::pdf_from_ops(&ops)).unwrap();

    let pdf = rag_11::parser::pdf::parse_pdf(&pdf_path).unwrap();
    let (tables, _) = detect_tables(&pdf.pages[0], &TableOptions::default());

    assert!(tables.is_empty(), "纯正文不应该被识别成表格");
}

#[test]
fn a_corrupt_file_produces_a_parse_error() {
    let dir = tempfile::tempdir().unwrap();
    let pdf_path = dir.path().join("broken.pdf");
    std::fs::write(&pdf_path, b"this is definitely not a pdf").unwrap();

    let config = test_config(dir.path());

    let err = parse_file(&pdf_path, &config).unwrap_err();
    assert_eq!(err.status_code(), 422, "解析失败应该是 422，不是 500");
}
