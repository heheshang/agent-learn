//! Rust 侧的剖析驱动。
//!
//! ## 工具对应关系
//!
//! Python 世界的三件套在 Rust 世界的对应物：
//!
//! | Python | Rust | 共同点 | 差异 |
//! |---|---|---|---|
//! | cProfile | `criterion` | 确定性重复测量 + 统计显著性 | criterion 会告诉你差值**是否显著** |
//! | py-spy | `pprof` | 采样型、低开销、不改代码 | pprof 直接出**火焰图** |
//! | Scalene | （无直接对应） | — | Rust 没有「Python 层 vs C 层」之分 |
//!
//! ## 为什么 py-spy **不能**剖析 Rust 进程
//!
//! py-spy 是**为 CPython 写的**：它读目标进程的 `PyInterpreterState`，
//! 顺着 `PyFrameObject` 链表还原**Python 帧**。
//!
//! Rust 进程里根本没有这些结构 —— py-spy attach 上去会：
//! - 找不到解释器状态 → 采到 0 个样本；或
//! - 直接报错（符号找不到）。
//!
//! ⚠️ **实测坑**：在 macOS 上它会先报「requires root」，
//!    你以为是自己权限问题，sudo 之后才发现是「压根不支持 Rust」。
//!    —— 这是个**双重误导**：权限问题掩盖了能力问题。
//!
//! Rust 侧的正解就是 `pprof`：它是**语言无关**的采样器
//! （靠 `signal` + `backtrace`），所以能剖析任何 Rust 程序。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// 剖析产物。
#[derive(Debug, Clone)]
pub struct ProfileArtifact {
    /// 产物类型（`flamegraph` / `svg` / `text`）。
    pub kind: String,

    /// 产物路径。
    pub path: PathBuf,

    /// 采样数（0 表示未采样）。
    pub samples: u64,
}

/// Rust 侧剖析器。
pub struct RustProfiler {
    output_dir: PathBuf,
}

impl RustProfiler {
    /// 创建剖析器。
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
        }
    }

    /// 产物目录。
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// 用 `pprof` 采样一段闭包，产出火焰图（SVG）。
    ///
    /// ## 与 Python 侧 py-spy 的对应关系
    ///
    /// py-spy 是**外部进程** attach 目标（不用重启、不用改代码）；
    /// pprof 是**库内嵌**（在进程里注册信号处理器）。
    ///
    /// 两者都是采样器，但 pprof 需要目标程序主动调用它 ——
    /// 这就是它相对 py-spy 的**劣势**：不能 attach 正在跑的进程。
    /// 代价换来的是「跨平台、零外部依赖、语言无关」。
    pub fn sample_flamegraph<F, T>(&self, label: &str, f: F) -> Result<ProfileArtifact>
    where
        F: FnOnce() -> T,
    {
        std::fs::create_dir_all(&self.output_dir)
            .with_context(|| format!("创建产物目录 {} 失败", self.output_dir.display()))?;

        let svg_path = self.output_dir.join(format!("{label}.flamegraph.svg"));

        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(199)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .context("启动 pprof 采样器失败")?;

        let _result = f();

        // ⚠️ 必须先 `report` 再让 `guard` 被 drop ——
        //    drop 时采样器才真正停止，顺序反了会拿到不完整的栈。
        let report = guard.report().build().context("构建 pprof 报告失败")?;

        // 采样数 = 所有栈的「样本计数」之和。
        //
        // ⚠️ `report.data` 的 value 类型是 `isize`（pprof 的内部表示），
        //    直接 `sum::<u64>()` 会因为 `Iterator::Item` 类型不匹配而编译失败。
        //    这里显式转成 `u64` 再累加。
        let samples: u64 = report
            .data
            .values()
            .map(|count| (*count).max(0) as u64)
            .sum();

        let mut file = std::fs::File::create(&svg_path)
            .with_context(|| format!("创建火焰图文件 {} 失败", svg_path.display()))?;
        report.flamegraph(&mut file).context("写出火焰图失败")?;

        Ok(ProfileArtifact {
            kind: "flamegraph".to_string(),
            path: svg_path,
            samples,
        })
    }

    /// 只计时，不采样（取基线用）。
    ///
    /// ⚠️ **为什么要单独测基线**：
    /// pprof 的采样器本身有开销（每个采样点要抓栈）。
    /// 拿「挂了采样器」的耗时做对比，测的是采样器的开销。
    pub fn timed<F, T>(&self, f: F) -> (T, u64)
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_millis() as u64;
        (result, elapsed)
    }
}
