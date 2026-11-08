//! 解析器集合 —— 把复杂文档变成结构化数据。
//!
//! ## 分层
//!
//! ```text
//! pdf.rs      读出带坐标的文字单元           ← 地基
//!    │
//!    ├──► table.rs   几何聚类 → 表格网格
//!    ├──► image.rs   lopdf → 图片文件
//!    └──► layout.rs  字号 + 几何 → 标题层级
//!    │
//!    ▼
//! pipeline.rs   编排：先切表格，再合并段落
//!    │
//!    ├──► chunk.rs    结构感知分块
//!    └──► markdown.rs 渲染成 Markdown
//! ```

pub mod chunk;
pub mod geometry;
pub mod image;
pub mod layout;
pub mod pdf;
pub mod pipeline;
pub mod table;
pub mod vision;

pub use chunk::{Chunk, ChunkOptions, chunk_document, count_by_kind};
pub use geometry::{BBox, ElementKind, LayoutElement, PositionedText};
pub use image::{ExtractedImage, ImageOptions, ImageStatus, extract_images};
pub use layout::{Layout, LayoutOptions};
pub use pdf::{Page, PageSize, PdfDocument, parse_pdf, parse_pdf_bytes};
pub use pipeline::{
    DocumentElement, ParseStats, ParsedDocument, TableData, parse_document, parse_file, to_markdown,
};
pub use table::{Table, TableOptions, TableStrategy, reconstruct_table};
pub use vision::{VisionDescriber, describe_images};
