//! 11-rag-advance —— 高级数据处理可执行入口。
//!
//! 有四种运行模式：
//!
//! ```bash
//! cargo run -p rag-11            # 启动 HTTP 服务
//! cargo run -p rag-11 -- demo    # 解析一份 PDF，打印结构化结果
//! cargo run -p rag-11 -- compare # 几何还原 vs 压平 同题对照 ★
//! cargo run -p rag-11 -- chunk   # 解析 + 分块
//! ```

use std::sync::Arc;

use agent_logging::{debug, info, init, stage_guard, warn};
use anyhow::Result;
use rag_11::api::{AppState, AppStateInner, routes};
use rag_11::config::AppConfig;
use rag_11::parser::chunk::{ChunkOptions, chunk_document, count_by_kind};
use rag_11::parser::pipeline::{parse_file, to_markdown};
use rag_11::parser::table::TableStrategy;

#[tokio::main]
async fn main() -> Result<()> {
    init("rag-11");

    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "serve".to_string());

    match mode.as_str() {
        "demo" => run_demo().await?,
        "compare" => run_compare().await?,
        "chunk" => run_chunk().await?,
        "serve" | "api" => run_server().await?,
        other => anyhow::bail!("未知模式: {other}（可选: serve / demo / compare / chunk）"),
    }

    Ok(())
}

/// 启动 HTTP 服务。
async fn run_server() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 11-rag-advance 高级数据处理 ===\n{}", config.summary());

    // 确认数据目录存在 —— 不存在的话每个请求都会 404，
    // 不如启动时就说清楚。
    if !config.data_dir.exists() {
        println!(
            "\n提示：文档目录不存在（{}），请先放入 PDF 文件。",
            config.data_dir.display()
        );
        warn!(data_dir = %config.data_dir.display(), "文档目录不存在");
    }

    let state: AppState = Arc::new(AppStateInner::new(config.clone()));

    let app = routes::router(state);
    let listener = tokio::net::TcpListener::bind(&config.bind_addr)
        .await
        .map_err(|e| anyhow::anyhow!("绑定端口 {} 失败: {}", config.bind_addr, e))?;

    let addr = listener.local_addr()?;
    println!("\n服务已启动: http://{addr}\n");
    info!(addr = %addr, "HTTP 服务已启动");
    println!("端点:");
    println!("  GET  /               服务信息 + 配置");
    println!("  GET  /health         健康检查");
    println!("  POST /api/parse      解析 PDF → 结构化结果 ★");
    println!("  POST /api/compare    几何还原 vs 压平 对照 ★");
    println!("  POST /api/chunk      解析 + 分块");
    println!("\n示例:");
    println!(
        "  curl -s http://{addr}/api/parse -H 'Content-Type: application/json' \\\n    \
         -d '{{\"path\":\"complex_layout.pdf\"}}'"
    );

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("服务启动失败: {e}"))
}

/// 端到端 demo：解析 → 打印结构。
async fn run_demo() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 11-rag-advance 解析 Demo ===\n{}", config.summary());

    let path = pick_pdf(&config)?;
    println!("\n解析: {}", path.display());
    info!(path = %path.display(), "开始解析 PDF");

    let doc = parse_file(&path, &config)?;

    info!(
        file_name = %doc.file_name,
        page_count = doc.page_count,
        element_count = doc.elements.len(),
        image_count = doc.images.len(),
        has_text_layer = doc.stats.has_text_layer,
        "PDF 解析完成"
    );

    println!("\n── 统计 ──");
    println!("  {}", doc.stats.summary());
    println!(
        "  文字层: {}",
        if doc.stats.has_text_layer {
            "有"
        } else {
            "无（扫描件）"
        }
    );

    println!("\n── 结构化元素（前 25 个）──");
    for element in doc.elements.iter().take(25) {
        let kind = rag_11::parser::chunk::kind_name(element.kind);
        let level = element.level.map(|l| format!(" H{l}")).unwrap_or_default();
        println!(
            "  [p{}{level} {kind}] {}",
            element.page_no,
            element.text.replace('\n', " ⏎ ")
        );
    }

    println!("\n── 表格（Markdown）──");
    for element in doc
        .elements
        .iter()
        .filter(|e| e.kind == rag_11::parser::geometry::ElementKind::Table)
    {
        println!("\n{}", element.text);
    }

    println!("\n── 图片 ──");
    for image in &doc.images {
        println!("  {}", image.summary());
    }

    println!("\n── 渲染后的 Markdown（前 60 行）──");
    for line in to_markdown(&doc).lines().take(60) {
        println!("  {line}");
    }

    Ok(())
}

/// 对照：几何还原 vs 压平。
async fn run_compare() -> Result<()> {
    let config = AppConfig::from_env();

    println!(
        "=== 11-rag-advance 几何还原 vs 压平 ===\n{}",
        config.summary()
    );

    let path = pick_pdf(&config)?;
    println!("\n解析: {}", path.display());

    let pdf = rag_11::parser::pdf::parse_pdf(&path)?;
    let options = rag_11::parser::table::TableOptions {
        min_cells: config.table.min_cells,
        row_tolerance: config.table.row_tolerance,
        column_tolerance: config.table.column_tolerance,
    };

    for page in &pdf.pages {
        println!("\n{}", "─".repeat(72));
        println!("第 {} 页", page.page_no);

        println!("\n【几何还原】");
        let (tables, _) = rag_11::parser::pipeline::detect_tables(page, &options);
        debug!(
            page_no = page.page_no,
            table_count = tables.len(),
            "几何还原表格识别完成"
        );
        if tables.is_empty() {
            println!("  （未识别到表格）");
        }
        for table in &tables {
            println!("  {} 行 × {} 列", table.row_count(), table.column_count());
            println!("{}", table.to_markdown());
        }

        println!("\n【压平对照】");
        // 对照基线：整页压平成纯文本行 —— 列对应关系彻底丢失。
        match rag_11::parser::table::flatten_table(&page.lines, page.page_no) {
            Some(table) => {
                println!("  {} 行 × 1 列（列信息丢失）", table.row_count());
                for row in &table.grid {
                    println!("  {}", row.join(" "));
                }
            }
            None => println!("  （未识别到表格）"),
        }
    }

    println!("\n{}", "─".repeat(72));
    println!("结论：几何还原保留了「哪一格属于哪一列」，压平没有。");
    println!("      前者能回答跨行跨列的问题，后者只能回答「文档里出现过这个词」。");

    Ok(())
}

/// 解析 + 分块。
async fn run_chunk() -> Result<()> {
    let config = AppConfig::from_env();

    println!("=== 11-rag-advance 分块 Demo ===\n{}", config.summary());

    let path = pick_pdf(&config)?;
    println!("\n解析: {}", path.display());

    let _guard = stage_guard("解析并分块");
    let doc = parse_file(&path, &config)?;
    let chunks = chunk_document(&doc, &ChunkOptions::default());

    info!(
        file_name = %doc.file_name,
        chunk_count = chunks.len(),
        "结构感知分块完成"
    );

    println!("\n── 统计 ──");
    println!("  {}", doc.stats.summary());
    println!("  块数: {}", chunks.len());
    for (kind, count) in count_by_kind(&chunks) {
        println!("    {kind}: {count}");
    }

    println!("\n── 块 ──");
    for chunk in &chunks {
        println!("  {}", chunk.summary());
    }

    Ok(())
}

/// 挑一份要处理的 PDF。
///
/// 优先用命令行给的路径，否则从 `data_dir` 里找第一份 PDF。
fn pick_pdf(config: &AppConfig) -> Result<std::path::PathBuf> {
    if let Some(arg) = std::env::args().nth(2) {
        let path = std::path::PathBuf::from(&arg);
        if !path.exists() {
            anyhow::bail!("文件不存在: {}", path.display());
        }
        return Ok(path);
    }

    let mut pdfs: Vec<std::path::PathBuf> = std::fs::read_dir(&config.data_dir)
        .map_err(|e| anyhow::anyhow!("读取数据目录失败: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
        })
        .collect();
    pdfs.sort();

    pdfs.into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("数据目录里没有 PDF: {}", config.data_dir.display()))
}

/// 让 `TableStrategy` 的导入不被当成死代码。
#[allow(dead_code)]
fn _keep_strategy_used(s: TableStrategy) -> &'static str {
    s.as_str()
}
