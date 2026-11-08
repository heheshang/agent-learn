//! 09-rag-eval 的可执行入口。
//!
//! 有四种运行模式：
//!
//! ```bash
//! cargo run -p rag-09 -- serve     # 启动 HTTP 服务
//! cargo run -p rag-09 -- eval      # 跑完整评估（多变体对比）★ 核心产出
//! cargo run -p rag-09 -- compare   # 只跑「优化前 vs 优化后」两个变体的对比
//! cargo run -p rag-09 -- demo      # 快速演示：跑一条评估集，打印报告
//! ```
//!
//! ## 整体流程
//!
//! ```text
//! 读配置 ──> 加载评估集 ──> 建/加载索引 ──> 建块索引（章节 → 块 ID）
//!                                                     │
//!                              ┌──────────────────────┴──────────────────┐
//!                              ▼                                         ▼
//!                      跑各变体（检索+生成）                    指标计算（确定性 + judge）
//!                              │                                         │
//!                              └──────────────────────┬──────────────────┘
//!                                                     ▼
//!                                              报告 + 对比报告
//! ```

use std::sync::Arc;
use std::time::Instant;

use agent_logging::{info, init, stage_guard, warn};
use anyhow::Result;
use rag_09::api::{AppState, AppStateInner, routes};
use rag_09::config::EvalConfig;
use rag_09::dataset::EvalDataset;
use rag_09::metrics::MetricRegistry;
use rag_09::pipeline::{EvalVariant, bootstrap};
use rag_09::report::{ComparisonReport, Evaluator};

#[tokio::main]
async fn main() -> Result<()> {
    init("rag-09");

    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "eval".to_string());

    match mode.as_str() {
        "eval" => run_eval().await?,
        "compare" => run_compare().await?,
        "demo" => run_demo().await?,
        "serve" | "api" => run_server().await?,
        other => anyhow::bail!("未知模式: {other}（可选: eval / compare / demo / serve）"),
    }

    Ok(())
}

/// 装配好的运行时组件。
struct Runtime {
    config: EvalConfig,
    pipeline: Arc<rag_09::pipeline::EvalPipeline>,
    dataset: EvalDataset,
    metrics: Arc<MetricRegistry>,
}

/// 装配所有组件（四种模式共用）。
///
/// 抽成函数是为了保证**四条路径用的是同一套装配逻辑** ——
/// 不然「demo 能跑但服务起不来」这种事会发生。
async fn bootstrap_all(config: EvalConfig) -> Result<Runtime> {
    let _guard = stage_guard("装配评估组件");

    // 1. 加载评估集。
    let dataset = EvalDataset::load(&config.dataset_path)?;
    dataset.validate()?;
    println!("  ✓ 评估集: {}（{} 个样本）", dataset.name, dataset.len());

    // 2. 建/加载索引 + 建块索引。
    let rag_config = config.to_rag_config();
    let pipeline = Arc::new(
        bootstrap(&rag_config)
            .await?
            .with_concurrency(config.concurrency),
    );
    println!(
        "  ✓ 索引: {} 个块 | {} 维 | {}",
        pipeline.index_stats().chunk_count,
        pipeline.index_stats().dimension,
        pipeline.index_stats().embedding_model
    );
    println!(
        "  ✓ 块索引: {} 个章节 / {} 个块",
        pipeline.chunk_index().section_count(),
        pipeline.chunk_index().chunk_count()
    );

    // 3. ⚠️ 检查评估集的标注是否有效 —— 这是一个「静默失败」高发区。
    warn_if_annotations_broken(&dataset, pipeline.chunk_index());
    // 4. 建指标注册表。
    //
    // ⚠️ judge 用**独立的 LLM 实例**，温度固定为 0 ——
    // 否则 judge 自身的抖动会淹没真实的指标差异。
    let judge_llm: Option<Arc<dyn rag_07::llm::Llm>> = if config.judge_enabled {
        Some(Arc::new(rag_08::llm::GenaiLlm::new(
            &config.judge_model,
            rag_08::llm::LlmOptions {
                temperature: config.judge_temperature,
                max_tokens: config.judge_max_tokens,
                // judge 是结构化短输出任务，思考只会烧 token。
                disable_reasoning: config.judge_disable_reasoning,
            },
        )) as Arc<dyn rag_07::llm::Llm>)
    } else {
        println!("  ⚠ judge 指标已关闭（只跑确定性指标）");
        None
    };

    let embedder: Arc<dyn rag_05::embed::Embedder> =
        Arc::new(rag_09::pipeline::bootstrap_embedder(&rag_config)?);

    let metrics = Arc::new(MetricRegistry::new(judge_llm, Some(embedder)));
    println!("  ✓ 指标: {}", metrics.names().join(", "));

    Ok(Runtime {
        config,
        pipeline,
        dataset,
        metrics,
    })
}

/// 检查评估集标注是否有效。
///
/// ## 为什么这一步很重要
///
/// 标注的章节名在知识库里找不到 → 相关块解析成空 → 召回率恒为 0。
///
/// **而且不会报任何错** —— 报告看起来很正常，只是所有检索指标都是 0。
/// 这个检查把「静默失败」变成「显式警告」。
fn warn_if_annotations_broken(dataset: &EvalDataset, chunk_index: &rag_09::dataset::ChunkIndex) {
    let broken: Vec<&str> = dataset
        .cases
        .iter()
        .filter(|c| !c.relevant_chunk_ids.is_empty() || !c.relevant_sections.is_empty())
        .filter(|c| c.resolve_relevant_ids(chunk_index).is_empty())
        .map(|c| c.id.as_str())
        .collect();

    if broken.is_empty() {
        return;
    }

    warn!(
        count = broken.len(),
        cases = ?broken,
        "评估集标注没对上 —— 相关块解析为空，召回指标会是 0"
    );
    println!(
        "  ⚠ 以下样本的相关块标注没对上（召回指标会是 0）: {}",
        broken.join(", ")
    );
    println!("    → 检查章节名是否和知识库标题完全一致，或直接改用 relevant_chunk_ids");
}

/// 跑完整评估（所有变体）。
async fn run_eval() -> Result<()> {
    let config = EvalConfig::from_env();

    println!("=== 09-rag-eval 完整评估 ===\n{}", config.summary());

    let runtime = bootstrap_all(config).await?;

    let evaluator = Evaluator::new(runtime.metrics.metrics().to_vec());
    let mut reports = Vec::new();
    let start = Instant::now();

    for variant in &runtime.config.variants {
        println!("\n正在评估变体: {} ...", variant.name());
        info!(variant = variant.name(), "开始评估变体");

        let (report, failed) = runtime
            .pipeline
            .run_variant(&runtime.dataset, *variant, &evaluator)
            .await?;

        if failed > 0 {
            println!("  ⚠ {} 个样本生成失败，已跳过", failed);
            warn!(variant = variant.name(), failed, "部分样本生成失败");
        }

        print!("{}", report.to_table());
        reports.push(report);
    }

    // 对比报告：第一个变体 vs 最后一个变体。
    if reports.len() >= 2 {
        let comparison = ComparisonReport::compare(&reports[0], &reports[reports.len() - 1]);
        println!("{}", comparison.to_table());
    }

    // 落盘。
    save_reports(&runtime, &reports)?;

    println!("\n总耗时: {}ms", start.elapsed().as_millis());
    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        "完整评估结束"
    );

    Ok(())
}

/// 只跑「优化前 vs 优化后」两个变体的对比。
///
/// 这是本项目**最核心的产出** —— 收益不靠嘴说，靠差值证明。
async fn run_compare() -> Result<()> {
    let mut config = EvalConfig::from_env();

    // 对比模式固定用这两个变体 —— 除非用户显式指定了别的。
    if std::env::var("EVAL_VARIANTS").is_err() {
        config.variants = vec![EvalVariant::Vector, EvalVariant::Hybrid];
    }

    println!("=== 09-rag-eval 优化前后对比 ===\n{}", config.summary());

    let runtime = bootstrap_all(config).await?;

    let evaluator = Evaluator::new(runtime.metrics.metrics().to_vec());
    let mut reports = Vec::new();

    for variant in &runtime.config.variants {
        println!("\n正在评估变体: {} ...", variant.name());
        info!(variant = variant.name(), "开始评估变体");

        let (report, failed) = runtime
            .pipeline
            .run_variant(&runtime.dataset, *variant, &evaluator)
            .await?;

        if failed > 0 {
            println!("  ⚠ {} 个样本生成失败，已跳过", failed);
            warn!(variant = variant.name(), failed, "部分样本生成失败");
        }

        reports.push(report);
    }

    if reports.len() >= 2 {
        let comparison = ComparisonReport::compare(&reports[0], &reports[reports.len() - 1]);
        println!("{}", comparison.to_table());

        // 结论。
        println!("=== 结论 ===");
        if comparison.improved().is_empty() {
            println!("没有指标出现明显提升（差值 ≥ 0.05）");
        } else {
            for delta in comparison.improved() {
                println!(
                    "  ↑ {}: {:.4} → {:.4}（{:+.4}）",
                    delta.name, delta.baseline, delta.optimized, delta.delta
                );
            }
        }
        for delta in comparison.regressed() {
            println!(
                "  ↓ {}: {:.4} → {:.4}（{:+.4}）—— 变差了，要查原因",
                delta.name, delta.baseline, delta.optimized, delta.delta
            );
        }
    }

    save_reports(&runtime, &reports)?;

    Ok(())
}

/// 快速演示：只跑评估集的第一个样本，打印详细结果。
async fn run_demo() -> Result<()> {
    let config = EvalConfig::from_env();

    println!("=== 09-rag-eval 快速演示 ===\n{}", config.summary());

    let runtime = bootstrap_all(config).await?;

    let case = runtime
        .dataset
        .cases
        .first()
        .ok_or_else(|| anyhow::anyhow!("评估集为空"))?;

    println!("\n样本: {}", case.id);
    println!("问题: {}", case.question);
    println!("标准答案: {}", case.reference_answer);

    let chunk_index = runtime.pipeline.chunk_index();
    let relevant = case.resolve_relevant_ids(chunk_index);
    println!("相关块（解析后）: {}", relevant.len());
    for id in &relevant {
        println!("  - {id}（章节: {:?}）", chunk_index.section_by_id(id));
    }

    // 跑两个变体做对比。
    for variant in [EvalVariant::Vector, EvalVariant::Hybrid] {
        println!("\n{}", "─".repeat(60));
        println!("变体: {}", variant.name());

        let output = runtime
            .pipeline
            .run_question(&case.question, variant.search_mode().unwrap())
            .await?;

        println!("检索到 {} 个块:", output.retrieved_ids.len());
        for (i, id) in output.retrieved_ids.iter().enumerate() {
            let marker = if relevant.contains(id) {
                "✓ 相关"
            } else {
                "  "
            };
            println!("  [{}] {marker} {id}", i + 1);
        }

        println!("\n答案:\n{}", output.answer);

        // 算指标。
        let ctx = rag_09::EvalContext::new(
            case,
            output.answer.clone(),
            output.contexts.clone(),
            output.retrieved_ids.clone(),
            chunk_index,
        );

        println!("\n指标:");
        for metric in runtime.metrics.metrics() {
            match metric.score(&ctx).await {
                Ok(score) => println!(
                    "  {:<24} {:.4}  {}",
                    metric.name(),
                    score.value,
                    score.reason.unwrap_or_default()
                ),
                Err(e) => println!("  {:<24} 计算失败: {e}", metric.name()),
            }
        }
    }

    Ok(())
}

/// 保存报告到磁盘。
fn save_reports(runtime: &Runtime, reports: &[rag_09::EvalReport]) -> Result<()> {
    let dir = &runtime.config.report_dir;
    std::fs::create_dir_all(dir)?;

    for report in reports {
        let path = dir.join(format!("{}.json", report.variant));
        std::fs::write(&path, serde_json::to_string_pretty(report)?)?;
        println!("  ✓ 报告已保存: {}", path.display());
    }

    if reports.len() >= 2 {
        let comparison = ComparisonReport::compare(&reports[0], &reports[reports.len() - 1]);
        let path = dir.join("comparison.json");
        std::fs::write(&path, serde_json::to_string_pretty(&comparison)?)?;
        println!("  ✓ 对比报告已保存: {}", path.display());

        let path = dir.join("comparison.txt");
        std::fs::write(&path, comparison.to_table())?;
    }

    Ok(())
}

/// 启动 HTTP 服务。
async fn run_server() -> Result<()> {
    let config = EvalConfig::from_env();

    println!("=== 09-rag-eval 评估服务 ===\n{}", config.summary());

    let runtime = bootstrap_all(config).await?;

    let state: AppState = Arc::new(AppStateInner::new(
        runtime.config.clone(),
        runtime.pipeline.clone(),
        runtime.dataset.clone(),
        runtime.metrics.clone(),
    ));

    let app = routes::router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3030")
        .await
        .map_err(|e| anyhow::anyhow!("绑定端口 3030 失败: {e}"))?;

    let addr = listener.local_addr()?;
    println!("\n服务已启动: http://{addr}\n");
    info!(addr = %addr, "HTTP 服务已启动");
    println!("端点:");
    println!("  GET  /               服务信息 + 评估集状态");
    println!("  GET  /health         健康检查");
    println!("  GET  /api/dataset    评估集详情（标注是否有效）");
    println!("  POST /api/eval       完整评估（多变体对比）★");
    println!("  POST /api/quick-eval 快速评估单个问题");
    println!("\n示例:");
    println!(
        "  curl -s http://{addr}/api/eval -H 'Content-Type: application/json' \\\n    \
         -d '{{\"variants\":[\"no_retrieval\",\"vector\",\"hybrid\"]}}'"
    );

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("服务启动失败: {e}"))
}
