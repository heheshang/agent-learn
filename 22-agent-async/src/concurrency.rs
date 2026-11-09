//! 并发原语 —— **本项目的核心**。
//!
//! # 为什么需要一个专门的并发模块
//!
//! FastAPI 教程把「并发 vs 串行」讲得很形象：
//!
//! > 你在柜台点单，然后**拿着号码牌坐下等** —— 这期间你可以跟同伴聊天。
//! > （并发）如果 8 个收银员同时也是厨师，你就只能**站在柜台前等** ——
//! > （并行）
//!
//! Rust 里没有「async 函数 vs 普通函数」的语法差异，
//! 但**有**一个完全对应的选择：
//!
//! | | FastAPI | Rust / tokio |
//! |---|---|---|
//! | 等待 I/O | `async def` + `await` | `async fn` + `.await` |
//! | 同步（阻塞）代码 | `def` → **丢进线程池** | `fn` → **丢进 `spawn_blocking`** |
//!
//! ## ⚠️ 三个必须记住的坑
//!
//! **① 同步的 CPU 密集代码放进 `async` 里会阻塞整个事件循环。**
//!
//! 一个 `async` 任务里做 100ms 的纯计算，其间**整个运行时都卡住** ——
//! 别的任务一个都跑不了。这正是 FastAPI 里「`def` 会跑在线程池」的原因。
//! 本项目用 [`crate::tools::BlockingTool`] 显式表达这个「线程池轨」。
//!
//! **② `buffer_unordered(0)` 会永远挂住。**
//!
//! 并发上限必须 **≥ 1**。配置层已经 clamp 过，但本模块也再挡一次 ——
//! 见 [`run`]。
//!
//! **③ 并发执行的返回顺序 ≠ 输入顺序。**
//!
//! LangChain 的 `batch_as_completed()` 文档里明说：
//! 「结果可能乱序到达，**每条结果带上输入的下标**用于对账」。
//! 本项目照做：每条 [`TaskOutput`] 都带 `index`，
//! 用 [`BatchReport::ordered_results`] 还原输入顺序。

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};

use crate::error::ToolError;

/// 一个并发任务的 future。
pub type TaskFuture = Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send>>;

/// 一个并发任务。
///
/// ## 为什么用「持有 future」而不是「持有闭包」
///
/// `tokio::spawn` 要求 future 是 `'static`（不借用调用栈），
/// 而本项目的工具调用要借用 `&ToolRegistry` / `&str` 提示词。
/// 所以只能**就地组合**（`join_all` / `FuturesUnordered` / `buffer_unordered`），
/// 而这些原语都接受「已经构造好的 future」。
pub struct Task {
    /// 任务名（工具名）。
    pub name: String,

    /// 输入下标（用于乱序时对账）。
    pub index: usize,

    /// 任务本体。
    pub future: TaskFuture,
}

impl Task {
    /// 构造一个任务。
    pub fn new<F>(name: impl Into<String>, index: usize, fut: F) -> Self
    where
        F: Future<Output = Result<String, ToolError>> + Send + 'static,
    {
        Self {
            name: name.into(),
            index,
            future: Box::pin(fut),
        }
    }
}

/// 一个任务的执行结果。
#[derive(Debug, Clone)]
pub struct TaskOutput {
    /// 输入下标（对账用）。
    pub index: usize,

    /// 任务名。
    pub name: String,

    /// 结果：观察文本 / 错误。
    pub result: Result<String, ToolError>,

    /// 这个任务自己花了多久。
    pub elapsed: Duration,
}

/// 执行模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// **串行**：一个接一个等。
    ///
    /// 对应 FastAPI 里的「站在柜台前等」—— 不并发。
    Sequential,

    /// **全部并发 + 保序返回**（`join_all`）。
    ///
    /// 对应 LangChain 的 `batch()` —— 所有输入都跑完，按输入顺序返回。
    JoinAll,

    /// **全部并发 + 谁先完成谁先返回**（`FuturesUnordered`）。
    ///
    /// 对应 LangChain 的 `batch_as_completed()` —— 结果**乱序**到达。
    AsCompleted,

    /// **限流并发**（`buffer_unordered(n)`）。
    ///
    /// 对应 LangChain 的 `max_concurrency` —— 同时最多跑 n 个。
    Bounded(usize),
}

impl ExecutionMode {
    /// 模式名（用于日志与报告）。
    pub fn name(&self) -> &'static str {
        match self {
            ExecutionMode::Sequential => "sequential",
            ExecutionMode::JoinAll => "join_all",
            ExecutionMode::AsCompleted => "as_completed",
            ExecutionMode::Bounded(_) => "bounded",
        }
    }

    /// 人类可读的说明。
    pub fn label(&self) -> String {
        match self {
            ExecutionMode::Sequential => "串行（一个接一个等）".to_string(),
            ExecutionMode::JoinAll => "全部并发 · 保序返回".to_string(),
            ExecutionMode::AsCompleted => "全部并发 · 谁先完成谁先返回".to_string(),
            ExecutionMode::Bounded(n) => format!("限流并发（最多 {n} 个同时跑）"),
        }
    }
}

/// 一批任务的执行报告。
#[derive(Debug, Clone)]
pub struct BatchReport {
    /// 执行模式。
    pub mode: ExecutionMode,

    /// 整批的总耗时。
    pub total: Duration,

    /// 结果（**按完成顺序** —— `AsCompleted` / `Bounded` 下与输入顺序不同）。
    pub results: Vec<TaskOutput>,

    /// 完成顺序（存的是输入下标）。
    pub completion_order: Vec<usize>,
}

impl BatchReport {
    /// 按**输入顺序**还原结果。
    ///
    /// ⚠️ 这一步是必须的：并发模式下结果的物理顺序是完成顺序，
    /// 要和输入对齐必须靠 `index`。
    pub fn ordered_results(&self) -> Vec<&TaskOutput> {
        let mut ordered: Vec<&TaskOutput> = self.results.iter().collect();
        ordered.sort_by_key(|r| r.index);
        ordered
    }

    /// 成功的任务数。
    pub fn success_count(&self) -> usize {
        self.results.iter().filter(|r| r.result.is_ok()).count()
    }

    /// 失败的任务数。
    pub fn failure_count(&self) -> usize {
        self.results.iter().filter(|r| r.result.is_err()).count()
    }

    /// 最慢的那个任务。
    pub fn slowest(&self) -> Option<&TaskOutput> {
        self.results.iter().max_by_key(|r| r.elapsed)
    }

    /// 总耗时（毫秒）。
    pub fn total_ms(&self) -> u64 {
        self.total.as_millis() as u64
    }

    /// 相对「串行」的加速比。
    ///
    /// 需要传入串行基准耗时。
    pub fn speedup(&self, sequential_ms: u64) -> f64 {
        if sequential_ms == 0 {
            return 0.0;
        }
        sequential_ms as f64 / self.total_ms().max(1) as f64
    }
}

/// 执行一批任务 —— **四种模式的核心实现**。
///
/// ## ⚠️ 并发上限必须 ≥ 1
///
/// `ExecutionMode::Bounded(0)` 会让 `buffer_unordered(0)` 永远不 poll，
/// 表现为「请求挂住不返回」。这里 clamp 掉，配置层也 clamp 过一次。
pub async fn run(mode: ExecutionMode, tasks: Vec<Task>) -> BatchReport {
    let start = Instant::now();

    let (results, completion_order) = match mode {
        ExecutionMode::Sequential => {
            let mut results = Vec::with_capacity(tasks.len());
            for task in tasks {
                let index = task.index;
                let name = task.name.clone();
                let start = Instant::now();
                let result = task.future.await;
                results.push(TaskOutput {
                    index,
                    name,
                    result,
                    elapsed: start.elapsed(),
                });
            }
            let order = results.iter().map(|r| r.index).collect();
            (results, order)
        }

        ExecutionMode::JoinAll => {
            // ⚠️ `join_all` **保序** —— 返回顺序 == 输入顺序。
            let futures = tasks.into_iter().map(|task| {
                let index = task.index;
                let name = task.name.clone();
                async move {
                    let start = Instant::now();
                    let result = task.future.await;
                    TaskOutput {
                        index,
                        name,
                        result,
                        elapsed: start.elapsed(),
                    }
                }
            });
            let results = futures::future::join_all(futures).await;
            let order = results.iter().map(|r| r.index).collect();
            (results, order)
        }

        ExecutionMode::AsCompleted => {
            // ⚠️ `FuturesUnordered` **不保序** —— 谁先完成谁先出来。
            let mut set = FuturesUnordered::new();
            for task in tasks {
                let index = task.index;
                let name = task.name.clone();
                set.push(async move {
                    let start = Instant::now();
                    let result = task.future.await;
                    TaskOutput {
                        index,
                        name,
                        result,
                        elapsed: start.elapsed(),
                    }
                });
            }

            let mut results = Vec::new();
            while let Some(output) = set.next().await {
                results.push(output);
            }
            let order = results.iter().map(|r| r.index).collect();
            (results, order)
        }

        ExecutionMode::Bounded(limit) => {
            // ⚠️ 上限至少 1 —— `buffer_unordered(0)` 会永远挂住。
            let limit = limit.max(1);

            let stream = futures::stream::iter(tasks.into_iter().map(|task| {
                let index = task.index;
                let name = task.name.clone();
                async move {
                    let start = Instant::now();
                    let result = task.future.await;
                    TaskOutput {
                        index,
                        name,
                        result,
                        elapsed: start.elapsed(),
                    }
                }
            }))
            .buffer_unordered(limit);

            futures::pin_mut!(stream);

            let mut results = Vec::new();
            while let Some(output) = stream.next().await {
                results.push(output);
            }
            let order = results.iter().map(|r| r.index).collect();
            (results, order)
        }
    };

    BatchReport {
        mode,
        total: start.elapsed(),
        results,
        completion_order,
    }
}

/// 任务工厂 —— 用来**重复构造**同一批任务。
///
/// ## 为什么需要工厂而不是「一批任务」
///
/// `Task` 里装的是 future，而 future **不可克隆**。
/// 所以要跑四遍对照实验，就必须能**重建**这批任务。
/// 工厂闭包就是这个「重建器」。
pub type TaskFactory = Box<dyn Fn() -> Vec<Task> + Send + Sync>;

/// 把一组「同构任务」跑遍四种模式 —— **对照实验的入口**。
///
/// ## 为什么要跑遍四种
///
/// 因为它们**不是**同一个东西：
///
/// - `Sequential` 是**基线**（不并发）
/// - `JoinAll` 与 `AsCompleted` 的**结果顺序不同**（前者保序，后者乱序）
/// - `Bounded` 是**唯一**会限流的
///
/// 只测一个的话，很容易得出「并发就是快」这种没有信息量的结论。
///
/// ## ⚠️ 每种模式都用**同一个工厂**重建任务
///
/// 这样「变量单一」—— 四组的任务内容、顺序完全一致，
/// 只有执行方式不同，差值才能归因到并发模式。
pub async fn compare_all(factory: TaskFactory, limit: usize) -> Vec<BatchReport> {
    let mut reports = Vec::new();

    reports.push(run(ExecutionMode::Sequential, factory()).await);
    reports.push(run(ExecutionMode::JoinAll, factory()).await);
    reports.push(run(ExecutionMode::AsCompleted, factory()).await);
    reports.push(run(ExecutionMode::Bounded(limit), factory()).await);

    reports
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 造一批「模拟 I/O 等待」的任务。
    fn io_tasks(count: usize, ms: u64) -> Vec<Task> {
        (0..count).map(|i| io_task("t", i, ms)).collect()
    }

    /// 一个「模拟 I/O 等待」的任务：睡 `ms` 毫秒后返回。
    fn io_task(name: &str, index: usize, ms: u64) -> Task {
        // ⚠️ `name` 必须 own 一份 —— future 要求 `'static`。
        let name = name.to_string();
        Task::new(name.clone(), index, async move {
            tokio::time::sleep(Duration::from_millis(ms)).await;
            Ok(format!("{name} 完成"))
        })
    }

    #[tokio::test]
    async fn sequential_is_slower_than_concurrent_for_io_wait() {
        let seq = run(ExecutionMode::Sequential, io_tasks(4, 100)).await;
        let par = run(ExecutionMode::JoinAll, io_tasks(4, 100)).await;

        // ⚠️ 串行 4×100ms，并发约 100ms。
        assert!(
            seq.total_ms() >= 380,
            "串行应至少 ~400ms，实际 {}",
            seq.total_ms()
        );
        assert!(
            par.total_ms() < 300,
            "并发应远快于串行，实际 {}ms",
            par.total_ms()
        );
    }

    #[tokio::test]
    async fn join_all_preserves_input_order() {
        // ⚠️ 故意让后面的任务先完成 ——
        //    如果实现退化成「谁先完成谁先返回」，顺序就会乱。
        let tasks = vec![
            Task::new("slow", 0, async {
                tokio::time::sleep(Duration::from_millis(80)).await;
                Ok("slow".to_string())
            }),
            Task::new("fast", 1, async {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok("fast".to_string())
            }),
        ];

        let report = run(ExecutionMode::JoinAll, tasks).await;

        let names: Vec<&str> = report
            .ordered_results()
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(names, vec!["slow", "fast"]);
    }

    #[tokio::test]
    async fn as_completed_returns_in_completion_order() {
        let tasks = vec![
            Task::new("slow", 0, async {
                tokio::time::sleep(Duration::from_millis(80)).await;
                Ok("slow".to_string())
            }),
            Task::new("fast", 1, async {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok("fast".to_string())
            }),
        ];

        let report = run(ExecutionMode::AsCompleted, tasks).await;

        // ⚠️ 完成顺序：fast 先。
        let names: Vec<&str> = report.results.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["fast", "slow"]);

        // ⚠️ 但按 index 还原后仍是输入顺序。
        let names: Vec<&str> = report
            .ordered_results()
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(names, vec!["slow", "fast"]);
    }

    #[tokio::test]
    async fn bounded_limits_concurrency() {
        let counter = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let tasks: Vec<Task> = (0..6)
            .map(|i| {
                let counter = counter.clone();
                let peak = peak.clone();
                Task::new("t", i, async move {
                    let now = counter.fetch_add(1, Ordering::SeqCst) + 1;
                    // 记录并发峰值。
                    peak.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(40)).await;
                    counter.fetch_sub(1, Ordering::SeqCst);
                    Ok("ok".to_string())
                })
            })
            .collect();

        let report = run(ExecutionMode::Bounded(2), tasks).await;

        let observed = peak.load(Ordering::SeqCst);
        assert!(observed <= 2, "并发峰值应 ≤ 2，实际 {observed} —— 限流失效");
        assert_eq!(report.results.len(), 6);
    }

    #[tokio::test]
    async fn bounded_with_zero_limit_does_not_hang() {
        // ⚠️ `buffer_unordered(0)` 会永远挂住 ——
        //    所以 `run` 里必须 clamp 到 1。
        //    这个测试用 `tokio::time::timeout` 卡住它，防止回归。
        let tasks = vec![io_task("t", 0, 10)];

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            run(ExecutionMode::Bounded(0), tasks),
        )
        .await;

        assert!(result.is_ok(), "Bounded(0) 挂住了 —— clamp 失效");
    }

    #[tokio::test]
    async fn failures_are_kept_per_task() {
        let tasks = vec![
            Task::new("ok", 0, async { Ok("成功".to_string()) }),
            Task::new("bad", 1, async {
                Err(ToolError::internal("bad", "炸了"))
            }),
        ];

        let report = run(ExecutionMode::JoinAll, tasks).await;

        assert_eq!(report.success_count(), 1);
        assert_eq!(report.failure_count(), 1);
    }

    #[tokio::test]
    async fn speedup_is_computed_against_sequential_baseline() {
        let seq = run(ExecutionMode::Sequential, io_tasks(4, 100)).await;
        let par = run(ExecutionMode::JoinAll, io_tasks(4, 100)).await;

        let speedup = par.speedup(seq.total_ms());
        assert!(speedup > 2.0, "加速比应 > 2x，实际 {speedup:.2}x");
    }

    #[tokio::test]
    async fn compare_all_runs_the_same_tasks_in_four_modes() {
        // ⚠️ 工厂闭包必须能**重建**任务（future 不可克隆）。
        let factory: TaskFactory =
            Box::new(|| (0..3).map(|i| io_task("t", i, 30)).collect::<Vec<Task>>());

        let reports = compare_all(factory, 2).await;

        assert_eq!(reports.len(), 4);
        assert_eq!(reports[0].mode, ExecutionMode::Sequential);
        assert_eq!(reports[1].mode, ExecutionMode::JoinAll);
        assert_eq!(reports[2].mode, ExecutionMode::AsCompleted);
        assert_eq!(reports[3].mode, ExecutionMode::Bounded(2));

        // 每组都必须有 3 条结果。
        for report in &reports {
            assert_eq!(report.results.len(), 3);
        }
    }
}
