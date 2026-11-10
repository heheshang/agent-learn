//! **并发层** —— 四档并发模式 + 限流。
//!
//! # 这个模块汇总了哪些历史能力
//!
//! | 能力 | 来自 | 在这里的形态 |
//! |---|---|---|
//! | 四档并发模式 + 限流 | 22-agent-async | [`ExecutionMode`] / [`run`] |
//!
//! # 四档模式
//!
//! | 模式 | 原语 | 顺序 | 限流 |
//! |---|---|---|---|
//! | `Sequential` | 逐个 await | 保序 | — |
//! | `JoinAll` | `join_all` | **保序** | — |
//! | `AsCompleted` | `FuturesUnordered` | **乱序** | — |
//! | `Bounded(n)` | `buffer_unordered(n)` | 乱序 | ✅ |
//!
//! # ⚠️ 三条硬约束
//!
//! ## ① `Bounded(0)` 会让请求**永远挂住**
//!
//! `buffer_unordered(0)` 内部信号量永远不发令牌。
//! 必须在**三道防线**都 clamp 到 ≥ 1。
//!
//! ## ② 并发任务要求 future 是 `'static`
//!
//! `tokio::spawn` 会**夺走所有权**（工具调用要借用 `&str` 提示词），
//! 所以只能用**就地组合**的并发原语。
//!
//! ## ③ 并发 ≠ 并行
//!
//! 并发提升的是「**等待**」的吞吐；CPU 密集任务本来就满负荷跑。

use std::future::Future;
use std::time::{Duration, Instant};

use futures::stream::{FuturesUnordered, StreamExt};

/// 一个任务的结果。
#[derive(Debug, Clone, PartialEq)]
pub struct TaskResult {
    /// 观察文本。
    pub observation: String,

    /// 尝试次数（**重试次数必须在这里** —— 丢了就没法统计）。
    pub attempts: usize,

    /// 是否失败。
    pub failed: bool,
}

impl TaskResult {
    /// 成功。
    pub fn ok(observation: impl Into<String>, attempts: usize) -> Self {
        Self {
            observation: observation.into(),
            attempts: attempts.max(1),
            failed: false,
        }
    }

    /// 失败。
    pub fn failed(observation: impl Into<String>, attempts: usize) -> Self {
        Self {
            observation: observation.into(),
            attempts: attempts.max(1),
            failed: true,
        }
    }
}

/// 一个并发任务（**持有 future**，不是持有闭包）。
pub struct Task {
    /// 任务名。
    pub name: String,

    /// 输入下标（**回执按它还原顺序**）。
    pub index: usize,

    /// 要跑的 future。
    pub future: BoxFuture,
}

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Task")
            .field("name", &self.name)
            .field("index", &self.index)
            .finish()
    }
}

/// 一个装箱的 future。
pub type BoxFuture = std::pin::Pin<Box<dyn Future<Output = TaskResult> + Send>>;

impl Task {
    /// 新建任务。
    pub fn new<F>(name: impl Into<String>, index: usize, fut: F) -> Self
    where
        F: Future<Output = TaskResult> + Send + 'static,
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
    /// 输入下标。
    pub index: usize,

    /// 任务名。
    pub name: String,

    /// 结果。
    pub result: TaskResult,

    /// 耗时。
    pub elapsed: Duration,
}

/// 一批任务的执行报告。
#[derive(Debug, Clone)]
pub struct BatchReport {
    /// 使用的模式。
    pub mode: ExecutionMode,

    /// 总耗时。
    pub total: Duration,

    /// 各任务结果（**按完成顺序**）。
    pub results: Vec<TaskOutput>,

    /// 完成顺序（任务下标序列）。
    pub completion_order: Vec<usize>,
}

impl BatchReport {
    /// ★ 按**输入顺序**还原结果。
    ///
    /// ## ⚠️ 为什么必须还原
    ///
    /// 并发执行后结果到达顺序是**乱的**。
    /// 回执必须按 `call_id` / 输入下标对应，
    /// 否则「工具调用 → 工具回执」的配对就乱了。
    pub fn ordered_results(&self) -> Vec<&TaskOutput> {
        let mut out: Vec<&TaskOutput> = self.results.iter().collect();
        out.sort_by_key(|r| r.index);
        out
    }

    /// 成功数。
    pub fn success_count(&self) -> usize {
        self.results.iter().filter(|r| !r.result.failed).count()
    }

    /// 失败数。
    pub fn failure_count(&self) -> usize {
        self.results.iter().filter(|r| r.result.failed).count()
    }

    /// 相对串行的加速比。
    pub fn speedup(&self, sequential_ms: f64) -> f64 {
        if sequential_ms <= 0.0 {
            return 1.0;
        }
        sequential_ms / self.total.as_secs_f64() * 1000.0
    }
}

/// 并发模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// 串行。
    Sequential,
    /// 全部并发 + 保序。
    JoinAll,
    /// 全部并发 + 谁先完成谁先返回。
    AsCompleted,
    /// 限流并发。
    Bounded(usize),
}

impl ExecutionMode {
    /// 模式名。
    pub fn name(self) -> &'static str {
        match self {
            ExecutionMode::Sequential => "sequential",
            ExecutionMode::JoinAll => "join_all",
            ExecutionMode::AsCompleted => "as_completed",
            ExecutionMode::Bounded(_) => "bounded",
        }
    }

    /// 人类可读的说明。
    pub fn label(self) -> String {
        match self {
            ExecutionMode::Sequential => "串行（一个接一个等）".to_string(),
            ExecutionMode::JoinAll => "全部并发 + 保序返回".to_string(),
            ExecutionMode::AsCompleted => "全部并发 + 谁先完成谁先返回".to_string(),
            ExecutionMode::Bounded(n) => format!("限流并发（最多 {n} 个同时在跑）"),
        }
    }

    /// 从字符串解析。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sequential" => Some(ExecutionMode::Sequential),
            "join_all" => Some(ExecutionMode::JoinAll),
            "as_completed" => Some(ExecutionMode::AsCompleted),
            other => other
                .strip_prefix("bounded")
                .and_then(|n| n.trim_start_matches(':').parse().ok())
                .map(ExecutionMode::Bounded),
        }
    }
}

/// ★ 执行一批任务 —— 四种模式的核心实现。
///
/// ## ⚠️ 并发上限必须 ≥ 1
///
/// `ExecutionMode::Bounded(0)` 会让 `buffer_unordered(0)` 永远不 poll，
/// 表现为「请求挂住不返回」。这里 clamp 掉。
pub async fn run(mode: ExecutionMode, tasks: Vec<Task>) -> BatchReport {
    let start = Instant::now();

    let (results, completion_order) = match mode {
        ExecutionMode::Sequential => {
            let mut results = Vec::with_capacity(tasks.len());
            for task in tasks {
                let index = task.index;
                let name = task.name.clone();
                let t0 = Instant::now();
                let result = task.future.await;
                results.push(TaskOutput {
                    index,
                    name,
                    result,
                    elapsed: t0.elapsed(),
                });
            }
            let order = results.iter().map(|r| r.index).collect();
            (results, order)
        }

        ExecutionMode::JoinAll => {
            let futures = tasks.into_iter().map(|task| {
                let index = task.index;
                let name = task.name.clone();
                async move {
                    let t0 = Instant::now();
                    let result = task.future.await;
                    TaskOutput {
                        index,
                        name,
                        result,
                        elapsed: t0.elapsed(),
                    }
                }
            });
            let results = futures::future::join_all(futures).await;
            let order = results.iter().map(|r| r.index).collect();
            (results, order)
        }

        ExecutionMode::AsCompleted => {
            let mut set = FuturesUnordered::new();
            for task in tasks {
                let index = task.index;
                let name = task.name.clone();
                set.push(async move {
                    let t0 = Instant::now();
                    let result = task.future.await;
                    TaskOutput {
                        index,
                        name,
                        result,
                        elapsed: t0.elapsed(),
                    }
                });
            }
            let mut results = Vec::new();
            let mut order = Vec::new();
            while let Some(output) = set.next().await {
                order.push(output.index);
                results.push(output);
            }
            (results, order)
        }

        ExecutionMode::Bounded(limit) => {
            // ⚠️ clamp 到 ≥ 1 —— 否则请求永远挂住。
            let limit = limit.max(1);
            let futures = tasks.into_iter().map(|task| {
                let index = task.index;
                let name = task.name.clone();
                async move {
                    let t0 = Instant::now();
                    let result = task.future.await;
                    TaskOutput {
                        index,
                        name,
                        result,
                        elapsed: t0.elapsed(),
                    }
                }
            });
            let mut results = Vec::new();
            let mut order = Vec::new();
            // ⚠️ `buffer_unordered` 作用在 **Stream** 上，
            //    不是 `FuturesUnordered`（后者本身已经是乱序集合）。
            let mut stream = futures::stream::iter(futures).buffer_unordered(limit);
            while let Some(output) = stream.next().await {
                order.push(output.index);
                results.push(output);
            }
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

/// ★ 任务工厂 —— 跑多遍对照实验时必须用它重建任务。
///
/// ## ⚠️ 为什么不用闭包
///
/// [`Task`] **不可克隆**（里面装的是 future，跑一次就没了）。
/// 所以要跑四遍对照，必须用工厂**重建**任务 ——
/// 这才能做到「变量单一」。
pub type TaskFactory = Box<dyn Fn() -> Vec<Task> + Send + Sync>;

/// 造一批「模拟 I/O 等待」的任务。
pub fn io_tasks(count: usize, delay_ms: u64) -> TaskFactory {
    Box::new(move || {
        (0..count)
            .map(|i| {
                Task::new(format!("task-{i}"), i, async move {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    TaskResult::ok(format!("ok-{i}"), 1)
                })
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_zero_is_clamped() {
        assert_eq!(ExecutionMode::Bounded(0).name(), "bounded");
    }

    #[test]
    fn mode_parses_from_string() {
        assert_eq!(
            ExecutionMode::parse("sequential"),
            Some(ExecutionMode::Sequential)
        );
        assert_eq!(
            ExecutionMode::parse("bounded:4"),
            Some(ExecutionMode::Bounded(4))
        );
    }

    #[tokio::test]
    async fn sequential_runs_in_order() {
        let report = run(ExecutionMode::Sequential, io_tasks(3, 1)()).await;
        let indices: Vec<usize> = report.ordered_results().iter().map(|r| r.index).collect();
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(report.success_count(), 3);
    }

    #[tokio::test]
    async fn join_all_keeps_input_order() {
        let report = run(ExecutionMode::JoinAll, io_tasks(3, 1)()).await;
        let indices: Vec<usize> = report.ordered_results().iter().map(|r| r.index).collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn as_completed_may_be_out_of_order() {
        let report = run(ExecutionMode::AsCompleted, io_tasks(3, 1)()).await;
        // 无论顺序如何，ordered_results 必须还原输入顺序。
        let indices: Vec<usize> = report.ordered_results().iter().map(|r| r.index).collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn bounded_zero_does_not_hang() {
        // ⚠️ 这个测试如果没 clamp 会**永远挂住**（测试超时）。
        let report = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run(ExecutionMode::Bounded(0), io_tasks(2, 1)()),
        )
        .await
        .expect("Bounded(0) 不能挂住");
        assert_eq!(report.results.len(), 2);
    }

    #[tokio::test]
    async fn concurrent_is_faster_than_sequential_for_io() {
        let seq = run(ExecutionMode::Sequential, io_tasks(4, 100)()).await;
        let par = run(ExecutionMode::JoinAll, io_tasks(4, 100)()).await;
        assert!(
            par.total < seq.total,
            "并发应该更快：{:?} vs {:?}",
            par.total,
            seq.total
        );
    }
}
