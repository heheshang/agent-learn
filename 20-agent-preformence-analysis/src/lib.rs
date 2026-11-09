//! # agent-20 —— 性能瓶颈分析（cProfile / py-spy / Scalene）
//!
//! 这个项目回答一个问题：**Agent 系统慢了，到底慢在哪？**
//!
//! ## 为什么要有这个项目
//!
//! 前面 19 个项目把「功能」做出来了 —— 检索、工具、记忆、错误处理、
//! 接地校验。但**功能正确 ≠ 跑得快**。
//!
//! 一个真实的 Agent 系统的延迟由三部分构成：
//!
//! ```text
//! 总延迟 = CPU 时间 + IO 等待 + 其它
//!            ↑          ↑
//!            可优化    优化不了（网络往返、模型生成速度）
//! ```
//!
//! **分不清这三者，就无法优化** ——
//! 你可能花三天优化了一段只占 2% 的 CPU 热点，
//! 而真正的瓶颈是每次 800ms 的 LLM 网络往返。
//!
//! ## 核心产出
//!
//! 三个剖析工具对**同一份程序**给出**三张不同的「罪魁祸首」名单**：
//!
//! | 工具 | 类型 | 量的东西 | 含 IO 等待？ |
//! |---|---|---|---|
//! | **cProfile** | 确定性统计 | 每次函数调用的**自耗时** | **含** |
//! | **py-spy** | 采样 | 栈顶帧的采样占比 | **不含**（默认排除空闲线程） |
//! | **Scalene** | 插桩 | CPU 占比（分 Python / C 两部分） | **不含**（单列） |
//!
//! 它们都没错 —— 它们量的**根本不是同一件事**。
//! 这个差异本身就是本项目最有价值的产出。
//!
//! ## 两个语言侧
//!
//! | 目录 | 内容 |
//! |---|---|
//! | [`profiler/`] | **Python 靶子**：被剖析对象 + 三工具驱动 + 对比报告 |
//! | 本 crate | **Rust 对照**：用 `pprof`（对应 py-spy）+ `criterion`（对应 cProfile）|
//!
//! ## ⚠️ 一个必须知道的约束
//!
//! **py-spy 在 macOS 上恒需 root**（官方文档：「OSX always requires running as root」），
//! 因为它靠 `task_for_pid` 读目标进程内存，而 macOS 的 SIP 不允许普通用户
//! attach 别的进程。**重签名（`codesign` 加 `get-task-allow`）也绕不过去**（实测过）。
//!
//! 所以本项目的 py-spy 一律**在 Docker 容器里跑**（Linux 下 attach 自己
//! spawn 的子进程不需要 root，只需 `SYS_PTRACE` 能力）。
//! cProfile 与 Scalene 是「插桩运行」型，不需要容器，宿主机 venv 里就能跑。

pub mod config;
pub mod profile;
pub mod report;
pub mod workload;

pub use config::{AppConfig, ToolProfile};
pub use profile::{ProfileArtifact, RustProfiler};
pub use report::{Report, ToolSummary};
pub use workload::{RustWorkload, WorkloadOutcome};
