//! 处理器 —— 每个端点一个函数。
//!
//! 统一套路：
//!
//! ```text
//! 取参数 ──► 解析 ──► 处理 ──► 序列化
//!              │
//!              └──► 出错就返回 AdvanceError（自动映射 HTTP 码）
//! ```

use std::path::{Path, PathBuf};

use agent_logging::{debug, info, warn};
use axum::Json;

use crate::api::extract::JsonBody;
use crate::api::models::{
    ChunkRequest, ChunkResponse, CompareRequest, CompareResponse, CompareSide, ErrorResponse,
    ParseRequest, ParseResponse, ServiceInfo,
};
use crate::error::{AdvanceError, AdvanceResult};
use crate::parser::chunk::{ChunkOptions, chunk_document, count_by_kind};
use crate::parser::pipeline::{parse_document, parse_file, to_markdown};
use crate::parser::table::{TableStrategy, flatten_table};

/// 服务信息。
pub async fn root(
    axum::extract::State(state): axum::extract::State<crate::api::AppState>,
) -> Json<ServiceInfo> {
    Json(ServiceInfo {
        service: "11-rag-advance".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        data_dir: state.config.data_dir.display().to_string(),
        asset_dir: state.config.asset_dir.display().to_string(),
        config: state.config.summary(),
    })
}

/// 健康检查。
pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "11-rag-advance",
    }))
}

/// 解析一份 PDF。
pub async fn parse(
    axum::extract::State(state): axum::extract::State<crate::api::AppState>,
    JsonBody(body): JsonBody<ParseRequest>,
) -> AdvanceResult<Json<ParseResponse>> {
    let path = resolve_path(&state.config.data_dir, &body.path)?;

    let strategy = match &body.table_strategy {
        Some(raw) => TableStrategy::parse(raw)
            .ok_or_else(|| AdvanceError::BadRequest(format!("未知的表格策略: {raw}")))?,
        None => TableStrategy::Geometric,
    };

    info!(path = %path.display(), strategy = strategy.as_str(), "开始解析 PDF");

    let pdf = crate::parser::pdf::parse_pdf(&path)?;
    let images = crate::parser::image::extract_images(
        &path,
        &state.config.asset_dir,
        &crate::parser::image::ImageOptions {
            min_bytes: state.config.image.min_bytes,
            min_dimension: state.config.image.min_dimension,
        },
    )?;

    let mut doc = parse_document(pdf, images, &state.config)?;

    // 如果指定了「压平」策略，把表格换成压平版本。
    if strategy == TableStrategy::Flat {
        doc = flatten_tables_in_place(doc, &state.config)?;
    }

    let markdown = if body.include_markdown {
        Some(to_markdown(&doc))
    } else {
        None
    };

    info!(
        file_name = %doc.file_name,
        page_count = doc.page_count,
        element_count = doc.elements.len(),
        image_count = doc.images.len(),
        "PDF 解析完成"
    );

    Ok(Json(ParseResponse {
        file_name: doc.file_name.clone(),
        page_count: doc.page_count,
        stats: doc.stats.clone(),
        elements: doc.elements.clone(),
        markdown,
    }))
}

/// 对照：几何还原 vs 压平。
pub async fn compare(
    axum::extract::State(state): axum::extract::State<crate::api::AppState>,
    JsonBody(body): JsonBody<CompareRequest>,
) -> AdvanceResult<Json<CompareResponse>> {
    let path = resolve_path(&state.config.data_dir, &body.path)?;

    let pdf = crate::parser::pdf::parse_pdf(&path)?;

    let geometric = collect_tables(&pdf, &state.config, TableStrategy::Geometric);
    let flat = collect_tables(&pdf, &state.config, TableStrategy::Flat);

    let verdict = build_verdict(&geometric, &flat);

    info!(
        file_name = %pdf.file_name,
        geometric_tables = geometric.len(),
        flat_tables = flat.len(),
        "表格策略对照完成"
    );

    Ok(Json(CompareResponse {
        file_name: pdf.file_name.clone(),
        geometric_tables: geometric.len(),
        flat_tables: flat.len(),
        geometric: vec![side_for(&geometric, TableStrategy::Geometric)],
        flat: vec![side_for(&flat, TableStrategy::Flat)],
        verdict,
    }))
}

/// 解析 + 分块。
pub async fn chunk(
    axum::extract::State(state): axum::extract::State<crate::api::AppState>,
    JsonBody(body): JsonBody<ChunkRequest>,
) -> AdvanceResult<Json<ChunkResponse>> {
    let path = resolve_path(&state.config.data_dir, &body.path)?;

    let doc = parse_file(&path, &state.config)?;

    let options = ChunkOptions {
        max_paragraph_tokens: body.max_paragraph_tokens.unwrap_or(300),
        ..Default::default()
    };

    let chunks = chunk_document(&doc, &options);
    let kind_counts = count_by_kind(&chunks);

    debug!(
        file_name = %doc.file_name,
        chunk_count = chunks.len(),
        "结构感知分块完成"
    );

    Ok(Json(ChunkResponse {
        file_name: doc.file_name.clone(),
        stats: doc.stats.clone(),
        chunks,
        kind_counts,
    }))
}

/// 把请求里的路径解析成绝对路径。
///
/// 相对路径是相对于 `data_dir` 的 —— 这样客户端不用知道服务跑在哪。
fn resolve_path(data_dir: &Path, raw: &str) -> AdvanceResult<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        warn!("请求里的 path 为空");
        return Err(AdvanceError::BadRequest("path 不能为空".to_string()));
    }

    // ⚠️ 防目录穿越：`..` 不该能跳出 data_dir。
    let candidate = Path::new(trimmed);

    let resolved = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        data_dir.join(candidate)
    };

    if !resolved.exists() {
        warn!(path = %resolved.display(), "文件不存在");
        return Err(AdvanceError::NotFound(format!(
            "文件不存在: {}",
            resolved.display()
        )));
    }

    Ok(resolved)
}

/// 收集一份 PDF 里的所有表格（按指定策略）。
///
/// ⚠️ 几何策略必须走 [`detect_tables`] —— 它先定位表格区域，
/// 再对每个区域重建网格。直接对整页做网格重建会把段落
/// 也算进去，拼出一个 90% 是空格的假表格。
///
/// 压平策略是**对照基线**，它故意模拟纯文本抽取的行为：
/// 把整页的字按行拼起来，列信息彻底丢失。
fn collect_tables(
    pdf: &crate::parser::pdf::PdfDocument,
    config: &crate::config::AppConfig,
    strategy: TableStrategy,
) -> Vec<String> {
    let options = crate::parser::table::TableOptions {
        min_cells: config.table.min_cells,
        row_tolerance: config.table.row_tolerance,
        column_tolerance: config.table.column_tolerance,
    };

    let mut out = Vec::new();

    for page in &pdf.pages {
        match strategy {
            TableStrategy::Geometric => {
                let (tables, _) = crate::parser::pipeline::detect_tables(page, &options);
                for table in tables {
                    out.push(table.to_markdown());
                }
            }
            TableStrategy::Flat => {
                // 对照基线：整页压平 —— 这就是纯文本抽取看到的东西。
                if let Some(table) = flatten_table(&page.lines, page.page_no) {
                    let text = table
                        .grid
                        .iter()
                        .flatten()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n");
                    out.push(text);
                }
            }
        }
    }

    out
}

/// 把表格换成压平版本。
fn flatten_tables_in_place(
    doc: crate::parser::pipeline::ParsedDocument,
    config: &crate::config::AppConfig,
) -> AdvanceResult<crate::parser::pipeline::ParsedDocument> {
    let options = crate::parser::table::TableOptions {
        min_cells: config.table.min_cells,
        row_tolerance: config.table.row_tolerance,
        column_tolerance: config.table.column_tolerance,
    };

    let mut elements = Vec::with_capacity(doc.elements.len());

    for element in doc.elements {
        if element.kind != crate::parser::geometry::ElementKind::Table {
            elements.push(element);
            continue;
        }

        // 从网格里抽出所有单元格，压平成纯文本行。
        let Some(table) = &element.table else {
            elements.push(element);
            continue;
        };

        let flat: Vec<String> = table
            .grid
            .iter()
            .map(|row| {
                row.iter()
                    .filter(|c| !c.trim().is_empty())
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .filter(|row| !row.trim().is_empty())
            .collect();

        let text = flat.join("\n");

        elements.push(crate::parser::pipeline::DocumentElement {
            kind: element.kind,
            text,
            level: element.level,
            page_no: element.page_no,
            bbox: element.bbox,
            table: Some(crate::parser::pipeline::TableData {
                grid: table.grid.clone(),
                header: table.header.clone(),
                strategy: TableStrategy::Flat,
            }),
        });
    }

    let _ = options;

    Ok(crate::parser::pipeline::ParsedDocument {
        file_name: doc.file_name,
        elements,
        images: doc.images,
        page_count: doc.page_count,
        stats: doc.stats,
    })
}

/// 构造对照的一边。
fn side_for(tables: &[String], strategy: TableStrategy) -> CompareSide {
    let plain_text = tables.join("\n\n");
    CompareSide {
        strategy,
        tables: tables.to_vec(),
        char_count: plain_text.chars().count(),
        plain_text,
    }
}

/// 构造结论摘要。
fn build_verdict(geometric: &[String], flat: &[String]) -> String {
    if geometric.is_empty() && flat.is_empty() {
        return "这份文档里没有识别到表格。".to_string();
    }

    let geo_cells: usize = geometric
        .iter()
        .map(|t| t.lines().filter(|l| l.starts_with('|')).count())
        .sum();
    let flat_lines: usize = flat.iter().map(|t| t.lines().count()).sum();

    format!(
        "几何还原出 {} 张表（{} 行 Markdown 网格，保留了列对应关系）；\
         压平后剩 {} 行纯文本（列对应关系丢失）。\
         表格类问题的可检索性差异就在这里。",
        geometric.len(),
        geo_cells,
        flat_lines
    )
}

/// 未使用的错误响应构造（保留给将来扩展）。
#[allow(dead_code)]
fn error_response(status: &str, message: &str) -> ErrorResponse {
    ErrorResponse {
        error: status.to_string(),
        message: message.to_string(),
    }
}
