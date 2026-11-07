//! 04-rag —— RAG Part 1：加载与分割。
//!
//! 这一部分只做流水线的头两步：
//!
//! ```text
//! 文件 ──Loader──> 文本 ──Splitter──> 块
//! ```
//!
//! 后面（嵌入、向量库、检索、生成）在 Part 2。
//!
//! ## 核心认知
//!
//! **Loader 决定你拿到什么文本，Splitter 决定模型能检索到什么。**
//!
//! 大多数 RAG 效果差不是模型的问题，而是这两步没做对：
//! - 用平铺文本抽取处理 PDF，标题结构全丢了 → 块边界全是乱的
//! - 按字符数硬切，切碎了句子和表格 → 块本身语义就不完整
//! - 不加标题路径 → 「什么是 RAG」在「概述」和「深入」两节检索结果无法区分
//!
//! 用法：
//! ```text
//! cargo run -- load       # 加载：对比三种 loader 的输出
//! cargo run -- split      # 分割：对比三种 splitter 的输出
//! cargo run -- compare    # 对比：同一文档不同策略的效果差异
//! cargo run -- pipeline   # 全流程：加载 → 分割 → 统计
//! ```

use std::path::{Path, PathBuf};

use anyhow::Result;

use rag_04::loader::{DoclingLoader, DocumentLoader, MarkdownLoader, PdfTextLoader};
use rag_04::splitter::{
    Chunk, MarkdownSplitter, RecursiveSplitter, Splitter, SplitterConfig, StructureAwareSplitter,
    TokenCounter,
};

/// 测试数据目录。
const DATA_DIR: &str = "data";

fn main() -> Result<()> {
    let demo = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DATA_DIR);

    match demo.as_str() {
        "load" => run_load_demo(&data_dir)?,
        "split" => run_split_demo(&data_dir)?,
        "compare" => run_compare_demo(&data_dir)?,
        "pipeline" => run_pipeline_demo(&data_dir)?,
        "all" => {
            run_load_demo(&data_dir)?;
            run_split_demo(&data_dir)?;
            run_compare_demo(&data_dir)?;
            run_pipeline_demo(&data_dir)?;
        }
        other => {
            anyhow::bail!("未知 demo: {other}（可选: load / split / compare / pipeline / all）")
        }
    }

    Ok(())
}

/// Demo 1：加载 —— 对比三种 loader 的输出。
///
/// 重点看 PDF 的两种加载方式差在哪：
/// - `pdf-extract`：扁平的纯文本行，标题和正文混在一起
/// - `docling`：还原成 Markdown，标题变成 `#` / `##`
fn run_load_demo(dir: &Path) -> Result<()> {
    println!("\n=== 1. 文档加载 ===");

    let md_path = dir.join("rag_guide.md");
    let pdf_path = dir.join("rag_notes.pdf");

    // ── Markdown ──────────────────────────────────────────────
    let md = MarkdownLoader.load(&md_path)?;
    println!("\n[MarkdownLoader] {}", md.summary());
    println!("--- 前 15 行 ---");
    for line in md.text.lines().take(15) {
        println!("  {line}");
    }

    // ── PDF：纯文本抽取 ───────────────────────────────────────
    println!("\n---");
    let pdf_flat = PdfTextLoader.load(&pdf_path)?;
    println!("[PdfTextLoader] {}", pdf_flat.summary());
    println!("--- 前 15 行 ---");
    for line in pdf_flat.text.lines().take(15) {
        println!("  {line}");
    }

    // ── PDF：结构感知（Docling）────────────────────────────────
    println!("\n---");
    match DoclingLoader.load(&pdf_path) {
        Ok(pdf_structured) => {
            println!("[DoclingLoader] {}", pdf_structured.summary());
            println!("--- 前 15 行 ---");
            for line in pdf_structured.text.lines().take(15) {
                println!("  {line}");
            }
            println!("\n--- 标题行（结构保留下来了）---");
            for line in pdf_structured.text.lines() {
                if line.starts_with('#') {
                    println!("  {line}");
                }
            }
        }
        Err(e) => {
            println!("[DoclingLoader] 失败: {e}");
            println!("（需要 docling 的 pdf-text feature，转换失败不影响其他 demo）");
        }
    }

    Ok(())
}

/// Demo 2：分割 —— 对比三种 splitter。
///
/// 同一份 Markdown，三种切法，看块的边界落在哪里。
fn run_split_demo(dir: &Path) -> Result<()> {
    println!("\n=== 2. 文本分割 ===");

    let md = MarkdownLoader.load(&dir.join("rag_guide.md"))?;
    let counter = TokenCounter::new()?;

    let config = SplitterConfig {
        chunk_size: 120,
        chunk_overlap: 20,
        add_heading_context: true,
    };

    // ── 递归分块 ──────────────────────────────────────────────
    let recursive = RecursiveSplitter::new(config.clone()).split(&md, &counter);
    print_chunks(
        "RecursiveSplitter（不看结构，按段落/句子降级切）",
        &recursive,
    );

    // ── Markdown 分块 ─────────────────────────────────────────
    let markdown = MarkdownSplitter::new(config.clone()).split(&md, &counter);
    print_chunks("MarkdownSplitter（CommonMark 块级元素）", &markdown);

    // ── 结构感知分块 ──────────────────────────────────────────
    let structure = StructureAwareSplitter::new(config.clone()).split(&md, &counter);
    print_chunks(
        "StructureAwareSplitter（沿标题层级，块不跨标题）",
        &structure,
    );

    Ok(())
}

/// Demo 3：对比 —— 量化不同策略的差异。
///
/// 三个关键指标：
/// - 块数 / 平均大小
/// - 有多少块是「跨标题」的（语义污染）
/// - 有多少块超预算（超出模型上下文）
fn run_compare_demo(dir: &Path) -> Result<()> {
    println!("\n=== 3. 策略对比 ===");

    let md = MarkdownLoader.load(&dir.join("rag_guide.md"))?;
    let counter = TokenCounter::new()?;

    let config = SplitterConfig {
        chunk_size: 120,
        chunk_overlap: 20,
        add_heading_context: true,
    };

    let recursive = RecursiveSplitter::new(config.clone());
    let markdown = MarkdownSplitter::new(config.clone());
    let structure = StructureAwareSplitter::new(config.clone());

    let splitters: Vec<(&str, Vec<Chunk>)> = vec![
        (recursive.name(), recursive.split(&md, &counter)),
        (markdown.name(), markdown.split(&md, &counter)),
        (structure.name(), structure.split(&md, &counter)),
    ];

    println!(
        "\n{:<18} {:>6} {:>10} {:>10} {:>12}",
        "策略", "块数", "平均 token", "最大 token", "标题路径覆盖率"
    );
    println!("{}", "-".repeat(60));

    for (name, chunks) in &splitters {
        if chunks.is_empty() {
            continue;
        }
        let avg = chunks.iter().map(|c| c.token_count).sum::<usize>() / chunks.len();
        let max = chunks.iter().map(|c| c.token_count).max().unwrap_or(0);
        let with_heading = chunks.iter().filter(|c| !c.heading_path.is_empty()).count();
        let coverage = with_heading * 100 / chunks.len();

        println!(
            "{:<18} {:>6} {:>10} {:>10} {:>11}%",
            name,
            chunks.len(),
            avg,
            max,
            coverage
        );
    }

    // 关键差异：结构感知分块的标题路径覆盖率应该是 100%，
    // 因为它保证每个块都在某个标题下。
    println!("\n解读：");
    println!("  - 标题路径覆盖率：块里有多少带上了「所在章节」信息。");
    println!("    这直接影响检索 —— 没带路径的块，语义是悬空的。");
    println!("  - structure-aware 的覆盖率是 100%，因为它按标题切。");
    println!("  - recursive 的块可能横跨两个章节，标题路径会指向其中之一，");
    println!("    另一部分的内容就被错误地归属了。");

    // 展示一个跨标题污染的例子。
    println!("\n--- recursive 中标题路径最长（可能跨标题）的块 ---");
    let mut sorted = splitters[0].1.clone();
    sorted.sort_by_key(|c| c.heading_path.len());
    for c in sorted.iter().rev().take(2) {
        println!("  {}", c.summary());
    }

    // 标题路径本身就是元数据的一部分，展示一个多级路径的块。
    if let Some(c) = all_deepest(&splitters[1].1) {
        println!("\n--- markdown 策略里路径最深的块（上下文最丰富）---");
        println!("  {}", c.summary());
    }

    Ok(())
}

/// 找出标题路径最深的块。
fn all_deepest(chunks: &[Chunk]) -> Option<&Chunk> {
    chunks.iter().max_by_key(|c| c.heading_path.len())
}

/// Demo 4：全流程 —— 加载 → 分割 → 统计。
///
/// 这是 Part 2 的接入点：切出来的 `Chunk` 直接就能送去算嵌入。
fn run_pipeline_demo(dir: &Path) -> Result<()> {
    println!("\n=== 4. 全流程 ===");

    let counter = TokenCounter::new()?;
    let config = SplitterConfig::default();
    let splitter = StructureAwareSplitter::new(config.clone());

    let docs = rag_04::loader::load_directory(dir)?;
    println!("\n加载了 {} 份文档:", docs.len());
    for d in &docs {
        println!("  {}", d.summary());
    }

    let mut all_chunks = Vec::new();
    for doc in &docs {
        let chunks = splitter.split(doc, &counter);
        println!("\n  {} → {} 块", doc.file_name, chunks.len());
        all_chunks.extend(chunks);
    }

    println!("\n--- 全部块（{} 个）---", all_chunks.len());
    for c in &all_chunks {
        println!("  {}", c.summary());
    }

    // 统计：这是 Part 2 做嵌入前要关心的数字。
    let total_tokens: usize = all_chunks.iter().map(|c| c.token_count).sum();
    println!("\n--- 统计 ---");
    println!("  块总数: {}", all_chunks.len());
    println!("  总 token: {total_tokens}");
    println!(
        "  平均块大小: {} token",
        all_chunks.iter().map(|c| c.token_count).sum::<usize>() / all_chunks.len().max(1)
    );
    println!(
        "  超预算块数 (>{}/token): {}",
        config.chunk_size,
        all_chunks
            .iter()
            .filter(|c| c.token_count > config.chunk_size)
            .count()
    );
    println!(
        "  带标题路径的块: {}/{}",
        all_chunks
            .iter()
            .filter(|c| !c.heading_path.is_empty())
            .count(),
        all_chunks.len()
    );

    // 展示一个块送去做嵌入时实际用的文本。
    if let Some(c) = all_chunks.iter().find(|c| c.heading_path.len() >= 2) {
        println!("\n--- 送去做嵌入的文本示例 ---");
        println!("来源: {}", c.source);
        println!("标题路径: {}", c.heading_path.join(" > "));
        println!("token: {}", c.token_count);
        println!("---");
        println!("{}", c.embedded_text);
        println!("---");
    }

    Ok(())
}

/// 打印一组块。
fn print_chunks(title: &str, chunks: &[Chunk]) {
    println!("\n{}", "=".repeat(70));
    println!("{title} —— {} 块", chunks.len());
    println!("{}", "=".repeat(70));
    for c in chunks {
        println!("  {}", c.summary());
    }
}
