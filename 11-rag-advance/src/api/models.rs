//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见 ——
//! 想知道服务能接什么、返回什么，看这一个文件就够了。

use serde::{Deserialize, Serialize};

use crate::parser::chunk::Chunk;
use crate::parser::pipeline::{DocumentElement, ParseStats};
use crate::parser::table::TableStrategy;

/// 解析请求体。
///
/// ```json
/// { "path": "11-rag-advance/data/complex_layout.pdf" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ParseRequest {
    /// PDF 的路径（相对于服务的 `data_dir`，或绝对路径）。
    pub path: String,

    /// 表格还原策略（可选，默认 `geometric`）。
    #[serde(default)]
    pub table_strategy: Option<String>,

    /// 是否返回 Markdown 渲染结果。
    #[serde(default = "default_true")]
    pub include_markdown: bool,
}

fn default_true() -> bool {
    true
}

/// 对照请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct CompareRequest {
    /// PDF 的路径。
    pub path: String,
}

/// 解析响应。
#[derive(Debug, Clone, Serialize)]
pub struct ParseResponse {
    /// 来源文件名。
    pub file_name: String,

    /// 页数。
    pub page_count: usize,

    /// 解析统计。
    pub stats: ParseStats,

    /// 结构化元素。
    pub elements: Vec<DocumentElement>,

    /// 渲染出的 Markdown（可选）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markdown: Option<String>,
}

/// 对照响应。
#[derive(Debug, Clone, Serialize)]
pub struct CompareResponse {
    /// 来源文件名。
    pub file_name: String,

    /// 几何还原的表格数。
    pub geometric_tables: usize,

    /// 压平后的表格数。
    pub flat_tables: usize,

    /// 几何还原的结果。
    pub geometric: Vec<CompareSide>,

    /// 压平的结果。
    pub flat: Vec<CompareSide>,

    /// 结论摘要。
    pub verdict: String,
}

/// 对照的一边。
#[derive(Debug, Clone, Serialize)]
pub struct CompareSide {
    /// 策略。
    pub strategy: TableStrategy,

    /// 表格（Markdown）。
    pub tables: Vec<String>,

    /// 纯文本（把所有字拼起来）。
    pub plain_text: String,

    /// 字符数。
    pub char_count: usize,
}

/// 分块请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkRequest {
    /// PDF 的路径。
    pub path: String,

    /// 段落块的最大 token 数（可选）。
    #[serde(default)]
    pub max_paragraph_tokens: Option<usize>,
}

/// 分块响应。
#[derive(Debug, Clone, Serialize)]
pub struct ChunkResponse {
    /// 来源文件名。
    pub file_name: String,

    /// 解析统计。
    pub stats: ParseStats,

    /// 块列表。
    pub chunks: Vec<Chunk>,

    /// 各类块的数量。
    pub kind_counts: Vec<(String, usize)>,
}

/// 错误响应。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// HTTP 状态码的规范名。
    pub error: String,

    /// 错误详情。
    pub message: String,
}

/// 服务信息。
#[derive(Debug, Clone, Serialize)]
pub struct ServiceInfo {
    /// 服务名。
    pub service: String,

    /// 版本。
    pub version: String,

    /// 文档目录。
    pub data_dir: String,

    /// 图片目录。
    pub asset_dir: String,

    /// 配置摘要。
    pub config: String,
}
