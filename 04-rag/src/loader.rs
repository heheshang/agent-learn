//! 文档加载层（Loaders）—— 把各种格式的文件变成统一的文本。
//!
//! 这一层解决 RAG 流水线的第一步：**把字节变成文本**。
//! 输出统一为 [`LoadedDoc`]，里面装着正文 + 元数据，后续的分块器只认这个结构。
//!
//! ## 三种 loader
//!
//! | loader | 适用 | 特点 |
//! |---|---|---|
//! | [`MarkdownLoader`] | `.md` | 直接读，保留标题结构 |
//! | [`PdfTextLoader`] | `.pdf`（文本层） | 纯 Rust 快基线，**丢结构** |
//! | [`DoclingLoader`] | `.pdf` / `.html` / ... | 结构感知，转成 Markdown 保留层级 |
//!
//! ## 为什么需要结构感知
//!
//! `pdf-extract` 抽出来的是一堆扁平的行，标题和正文混在一起，
//! 分块器只能按字符数硬切。而 Docling 会把 PDF 解析成带层级的
//! 中间表示，再导出成 Markdown —— 标题变成了 `#` / `##`，
//! 分块器就能沿着标题边界切，这是「结构感知分块」的前提。

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// 加载失败。
#[derive(Debug, Error)]
pub enum LoaderError {
    /// 文件读不出来。
    #[error("读取文件失败 {path}: {source}")]
    Io {
        /// 文件路径。
        path: PathBuf,
        /// 底层 IO 错误。
        #[source]
        source: std::io::Error,
    },

    /// 不认识的扩展名。
    #[error("不支持的文件类型: {0}")]
    #[allow(dead_code)]
    UnsupportedFormat(String),

    /// PDF 解析失败。
    #[error("PDF 解析失败 {path}: {message}")]
    PdfParse {
        /// 文件路径。
        path: PathBuf,
        /// 错误描述。
        message: String,
    },

    /// Docling 转换失败。
    #[error("Docling 转换失败 {path}: {message}")]
    DoclingConvert {
        /// 文件路径。
        path: PathBuf,
        /// 错误描述。
        message: String,
    },
}

/// 一份加载好的文档。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedDoc {
    /// 来源文件路径。
    pub source: PathBuf,

    /// 文件名（含扩展名），做展示和引用。
    pub file_name: String,

    /// 文档格式（`markdown` / `pdf` / ...）。
    pub format: String,

    /// 正文。Markdown 保留原始标记；PDF 走 Docling 时是转换出的 Markdown。
    pub text: String,

    /// 页面数（仅 PDF 有意义）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_count: Option<usize>,

    /// 加载器名字，用来对比不同 loader 的效果。
    pub loader: String,
}

impl LoadedDoc {
    /// 正文字符数。
    pub fn char_len(&self) -> usize {
        self.text.chars().count()
    }

    /// 正文行数。
    pub fn line_count(&self) -> usize {
        self.text.lines().count()
    }

    /// 打印一段摘要。
    pub fn summary(&self) -> String {
        let mut s = format!(
            "[{}] {} ({} 字符, {} 行)",
            self.loader,
            self.file_name,
            self.char_len(),
            self.line_count()
        );
        if let Some(p) = self.page_count {
            s.push_str(&format!(", {p} 页"));
        }
        s
    }
}

/// 文档加载器 trait。
///
/// 存在这个 trait 是为了：上层流水线只依赖 `Box<dyn DocumentLoader>`，
/// 于是「本地 PDF loader」和「远程 Unstructured API loader」可以互换。
pub trait DocumentLoader: Send + Sync {
    /// 这个 loader 能处理哪些扩展名（小写，不含点）。
    fn supported_extensions(&self) -> &[&str];

    /// 加载一个文件。
    fn load(&self, path: &Path) -> Result<LoadedDoc, LoaderError>;

    /// 是否支持这个文件。
    fn supports(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .is_some_and(|e| self.supported_extensions().contains(&e.as_str()))
    }
}

/// Markdown loader —— 直接读文件。
///
/// Markdown 本身就是文本，而且标题层级用 `#` 标记，
/// 天然带着结构信息，不需要额外解析。
#[derive(Debug, Default, Clone, Copy)]
pub struct MarkdownLoader;

impl DocumentLoader for MarkdownLoader {
    fn supported_extensions(&self) -> &[&str] {
        &["md", "markdown", "txt"]
    }

    fn load(&self, path: &Path) -> Result<LoadedDoc, LoaderError> {
        let text = std::fs::read_to_string(path).map_err(|source| LoaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;

        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        Ok(LoadedDoc {
            source: path.to_path_buf(),
            file_name,
            format: "markdown".to_string(),
            text,
            page_count: None,
            loader: "markdown".to_string(),
        })
    }
}

/// PDF 纯文本 loader —— 基于 `pdf-extract`。
///
/// 优点：纯 Rust、快、零外部依赖。
/// 缺点：**丢结构**。抽出来是扁平行，标题和正文没有区分，
/// 多栏排版还会串行。这也是为什么还需要 Docling。
#[derive(Debug, Default, Clone, Copy)]
pub struct PdfTextLoader;

impl DocumentLoader for PdfTextLoader {
    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }

    fn load(&self, path: &Path) -> Result<LoadedDoc, LoaderError> {
        // 逐页抽取，这样能同时拿到页数。
        let pages =
            pdf_extract::extract_text_by_pages(path).map_err(|e| LoaderError::PdfParse {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        let page_count = pages.len();
        let text = pages.join("\n\n");

        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        Ok(LoadedDoc {
            source: path.to_path_buf(),
            file_name,
            format: "pdf".to_string(),
            text,
            page_count: Some(page_count),
            loader: "pdf-extract".to_string(),
        })
    }
}

/// Docling loader —— 结构感知解析。
///
/// 把 PDF / HTML 解析成 `DoclingDocument`（带层级的中间表示），
/// 再导出成 Markdown。因为标题被还原成了 `#` / `##`，
/// 后续的分块器可以沿标题边界切，保留语义结构。
///
/// 这是 Unstructured / MinerU 这类「重型解析服务」在 Rust 里的本地替代品：
/// 同样的目标（结构感知），但不需要跑一个 Python 服务。
///
/// ## ⚠️ 版面分析（识别 PDF 标题）需要额外模型
///
/// Rust 版 docling 的 PDF 后端分两档：
///
/// - **本 crate 默认（`pdf-text` feature）**：只抽 PDF 的文本层。
///   快、零额外依赖，但**标题不会变成 `#`** —— 所以结构感知分块对 PDF 无效，
///   PDF 只能走「按 token 硬切」的路子。
/// - **完整 ML 管线（docling 的 `pdf` feature）**：版面分析能识别标题，
///   输出 `## Section 1 ...`，结构感知分块才真正生效。
///   代价：编译期多拉 ONNX runtime，运行期需要 `.models/layout_heron_int8.onnx`。
///
/// 完整管线因为依赖冲突（ONNX runtime 版本与 workspace 里的 rig/fastembed 不兼容）
/// 放在独立的 `04-rag/layout-demo` crate 里演示。见那边�� README。
#[derive(Debug, Default, Clone, Copy)]
pub struct DoclingLoader;

impl DocumentLoader for DoclingLoader {
    fn supported_extensions(&self) -> &[&str] {
        &["pdf", "html", "htm"]
    }

    fn load(&self, path: &Path) -> Result<LoadedDoc, LoaderError> {
        let source =
            docling::SourceDocument::from_file(path).map_err(|e| LoaderError::DoclingConvert {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        // `text_layer_only` = 只抽文本层，不跑版面分析（因此不需要 ONNX 模型）。
        let result = docling::DocumentConverter::new()
            .text_layer_only(true)
            .convert(source)
            .map_err(|e| LoaderError::DoclingConvert {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        let text = result.document.export_to_markdown();

        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        Ok(LoadedDoc {
            source: path.to_path_buf(),
            file_name,
            format: "pdf".to_string(),
            text,
            page_count: None, // Docling 的页数在 DoclingDocument 内部，这里先不暴露
            loader: "docling".to_string(),
        })
    }
}

/// 按扩展名自动挑一个 loader。
///
/// 返回 `None` 表示没有 loader 支持这个文件。
pub fn loader_for(path: &Path) -> Option<Box<dyn DocumentLoader>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())?;

    match ext.as_str() {
        "md" | "markdown" | "txt" => Some(Box::new(MarkdownLoader)),
        "pdf" => Some(Box::new(DoclingLoader)),
        "html" | "htm" => Some(Box::new(DoclingLoader)),
        other => {
            eprintln!("跳过不支持的文件类型: {other} ({})", path.display());
            None
        }
    }
}

/// 加载目录下所有受支持的文件。
pub fn load_directory(dir: &Path) -> Result<Vec<LoadedDoc>, LoaderError> {
    let mut docs = Vec::new();

    let entries = std::fs::read_dir(dir).map_err(|source| LoaderError::Io {
        path: dir.to_path_buf(),
        source,
    })?;

    // 排序保证加载顺序稳定（否则每次 run 结果可能不同）。
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    paths.sort();

    for path in paths {
        if let Some(loader) = loader_for(&path)
            && loader.supports(&path)
        {
            docs.push(loader.load(&path)?);
        }
    }

    Ok(docs)
}
