//! 配置：环境变量前缀用 `PERF_`（Performance）。
//!
//! 与其它项目错开：13 用 `AGENT_`，14 用 `TOOL_`，15 用 `SQL_`，
//! 16 用 `FC_`，17 用 `MEM_`，18 用 `ERR_`，19 用 `SUM_`。

use std::path::PathBuf;

/// 要剖析的工具。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolProfile {
    /// pprof（采样型，对应 py-spy 的角色）。
    Pprof,

    /// 不剖析，只跑计时（最轻量，用来取基线）。
    Timer,
}

impl ToolProfile {
    /// 解析命令行 / 环境变量里的名字。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pprof" => Some(Self::Pprof),
            "timer" | "none" => Some(Self::Timer),
            _ => None,
        }
    }

    /// 名称。
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pprof => "pprof",
            Self::Timer => "timer",
        }
    }
}

/// 应用配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// 剖析产物输出目录。
    pub output_dir: PathBuf,

    /// 要剖析的工具。
    pub tool: ToolProfile,

    /// 工作负载重复次数。
    ///
    /// ⚠️ 为什么要重复跑：
    /// 单次运行的耗时**噪声很大**（机器负载、CPU 频率、缓存状态）。
    /// 剖析结论要可靠，至少跑几次取中位数。
    pub repeats: usize,

    /// 是否把 Python 靶子的剖析也一起跑（需要 Docker）。
    pub include_python: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            // ⚠️ 用 `perf-out/` 而不是 `target/perf` ——
            //    `target/` 是 Cargo 的构建目录（已被根 .gitignore 忽略），
            //    剖析产物混进去会让人搞不清哪些是编译产物、哪些是分析结果。
            output_dir: PathBuf::from("perf-out"),
            tool: ToolProfile::Timer,
            repeats: 3,
            include_python: false,
        }
    }
}

impl AppConfig {
    /// 从环境变量读取配置。
    ///
    /// | 变量 | 默认 | 说明 |
    /// |---|---|---|
    /// | `PERF_OUTPUT_DIR` | `perf-out` | 产物目录 |
    /// | `PERF_TOOL` | `timer` | `pprof` / `timer` |
    /// | `PERF_REPEATS` | `3` | 重复次数 |
    /// | `PERF_INCLUDE_PYTHON` | `0` | 是否跑 Python 侧剖析 |
    pub fn from_env() -> Self {
        let default = Self::default();

        let output_dir = std::env::var("PERF_OUTPUT_DIR")
            .ok()
            .map(PathBuf::from)
            .unwrap_or(default.output_dir);

        let tool = std::env::var("PERF_TOOL")
            .ok()
            .and_then(|v| ToolProfile::parse(&v))
            .unwrap_or(default.tool);

        let repeats = std::env::var("PERF_REPEATS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(default.repeats);

        let include_python = matches!(
            std::env::var("PERF_INCLUDE_PYTHON").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );

        Self {
            output_dir,
            tool,
            repeats,
            include_python,
        }
    }
}
