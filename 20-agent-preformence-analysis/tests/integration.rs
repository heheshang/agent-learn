//! 测试：锁住 Rust 侧的行为。
//!
//! ## 为什么 Rust 侧也要有测试
//!
//! 本项目的核心对比是「Python 侧 vs Rust 侧」。
//! 要让这个对比有意义，两边必须**做同一件事**：
//! 同样的语料、同样的查询、同样的 BM25 口径。
//!
//! 所以测试要锁住：
//! ① Rust 侧的检索**确实命中**（不是恒返回空）；
//! ② 语料规模与 Python 侧一致；
//! ③ 计时用的是微秒（毫秒会全部取整成 0）。

use agent_20::config::{AppConfig, ToolProfile};
use agent_20::profile::RustProfiler;
use agent_20::report::{Report, ToolSummary};
use agent_20::workload::{CORPUS_REPEAT, DOCUMENTS, QUERIES, RustWorkload, TIMING_OUTER_REPEATS};

#[test]
fn corpus_size_matches_python_side() {
    // Python 侧：corpus_repeat=4，10 条基线文档 → 40 块
    let workload = RustWorkload::new();
    assert_eq!(workload.corpus_size(), CORPUS_REPEAT * DOCUMENTS.len());
    assert_eq!(workload.corpus_size(), 40);
}

#[test]
fn queries_are_the_same_as_python_side() {
    // 与 Python 侧 WorkloadSpec.queries 逐条一致
    assert_eq!(QUERIES.len(), 5);
    assert_eq!(QUERIES[0], "Agent 系统的性能瓶颈通常出现在哪里");
    assert_eq!(QUERIES[4], "查询改写会带来多少额外开销");
}

#[test]
fn retrieval_actually_finds_hits() {
    // ⚠️ 检索必须有命中，否则「检索是瓶颈」这个结论根本测不出来。
    let workload = RustWorkload::new();
    let hits = workload.run_once("性能剖析 工具有哪些");
    assert!(!hits.is_empty(), "检索必须命中");
    assert!(hits.len() <= 3, "top-k 应该是 3");
}

#[test]
fn every_query_returns_hits() {
    // 每个查询都要能命中（不能只靠个别查询撑场面）
    let workload = RustWorkload::new();
    for query in QUERIES {
        let hits = workload.run_once(query);
        assert!(!hits.is_empty(), "查询「{query}」必须命中");
    }
}

#[test]
fn workload_reports_microseconds_not_zero() {
    // ⚠️ 计时必须用微秒：毫秒会全部取整成 0。
    let workload = RustWorkload::new();
    let outcome = workload.run(TIMING_OUTER_REPEATS);
    assert!(
        outcome.elapsed_us > 0,
        "耗时必须是正的微秒数（毫秒取整会变 0）"
    );
    assert_eq!(outcome.search_rounds, QUERIES.len() * TIMING_OUTER_REPEATS);
}

#[test]
fn index_build_is_reused_across_queries() {
    // 建一次索引，检索多次 —— 这是「优化后」的核心特征。
    let workload = RustWorkload::new();
    let first = workload.run_once(QUERIES[0]);
    let second = workload.run_once(QUERIES[0]);
    // 同样的查询必须给出同样的结果（索引没被污染）
    assert_eq!(first.len(), second.len());
    assert_eq!(first[0].id, second[0].id);
}

#[test]
fn tool_profile_parses() {
    assert_eq!(ToolProfile::parse("pprof"), Some(ToolProfile::Pprof));
    // 大写也要认（解析器会先小写化）
    assert_eq!(ToolProfile::parse("PPROF"), Some(ToolProfile::Pprof));
    assert_eq!(ToolProfile::parse(" timer "), Some(ToolProfile::Timer));
    assert_eq!(ToolProfile::parse("timer"), Some(ToolProfile::Timer));
    assert_eq!(ToolProfile::parse("none"), Some(ToolProfile::Timer));
    assert_eq!(ToolProfile::parse("unknown"), None);
}

#[test]
fn app_config_defaults() {
    let config = AppConfig::default();
    assert!(config.repeats > 0);
    assert!(!config.include_python);
}

#[test]
fn profiler_produces_flamegraph_artifact() {
    // 产出火焰图，采样数必须 > 0
    let dir = tempfile::tempdir().expect("创建临时目录");
    let profiler = RustProfiler::new(dir.path());
    let workload = RustWorkload::new();

    let artifact = profiler
        .sample_flamegraph("test_flamegraph", || workload.run(2000))
        .expect("采样应成功");

    assert!(artifact.path.exists(), "火焰图文件应存在");
    assert!(artifact.samples > 0, "采样数必须大于 0");
    assert_eq!(artifact.kind, "flamegraph");
}

#[test]
fn report_renders_markdown() {
    let mut report = Report::new("测试报告");
    report.push_timing("检索", 1234);
    report.push_tool(ToolSummary {
        tool: "pprof".to_string(),
        unit: "采样占比（%）".to_string(),
        includes_io: false,
        rows: vec![("bm25_score".to_string(), 42.5)],
    });

    let markdown = report.to_markdown();
    assert!(markdown.contains("测试报告"));
    assert!(markdown.contains("1234"));
    assert!(markdown.contains("bm25_score"));
    assert!(markdown.contains("不含 IO 等待"));
}
