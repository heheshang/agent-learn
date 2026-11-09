//! 并发对照实验 —— **把「并发到底快多少」量化出来**。
//!
//! # 实验设计
//!
//! ## 变量单一
//!
//! 四组跑的是**同一批任务**（同一个工厂重建），
//! 只有执行方式不同：
//!
//! | 组 | 模式 | 对应 LangChain |
//! |---|---|---|
//! | A | `Sequential` —— 一个接一个等 | （无对应，直接串行调） |
//! | B | `JoinAll` —— 全并发，**保序**返回 | `batch()` |
//! | C | `AsCompleted` —— 全并发，**乱序**返回 | `batch_as_completed()` |
//! | D | `Bounded(n)` —— 限流并发 | `max_concurrency=n` |
//!
//! ## 两种「负载」
//!
//! ### ① 模拟负载（`simulated = true`，**默认**）
//!
//! 用 `tokio::time::sleep` 模拟 I/O 等待。
//!
//! ⚠️ **为什么默认用模拟的**：
//! 真实天气接口有网络抖动和限流，
//! 跑对照实验时用假的更稳定、更可复现 ——
//! 这正是 20 项目的教训（「差值必须用不挂剖析器的纯净运行测」）。
//!
//! ### ② 真实负载（`simulated = false`）
//!
//! 真调 Open-Meteo 的天气接口。
//! 结论会受网络环境影响，但能证明「真实场景下并发也有效」。
//!
//! # ⚠️ 一个反直觉的结论
//!
//! **限流组（D）不一定比全并发组（B）慢。**
//!
//! 当任务数 ≤ 限流数时，B 和 D **完全等价**。
//! 只有任务数 > 限流数时，D 才会体现「排队」。
//!
//! 所以对照实验必须**同时改变任务数和限流数**，
//! 否则会得出「限流没用」的错误结论。

use std::time::Duration;

use crate::concurrency::{BatchReport, ExecutionMode, Task, TaskFactory};
use crate::error::{AppError, AppResult, ToolError};
use crate::tools::AsyncTool;

/// 模拟 I/O 等待的时长（毫秒）。
pub const SIMULATED_IO_MS: u64 = 120;

/// 构造一批「模拟 I/O」的任务。
///
/// ## ⚠️ 必须用**工厂**
///
/// `Task` 里装的是 future，不可克隆。
/// 跑四组对照实验就要重建四遍 —— 工厂就是重建器。
pub fn simulated_factory(cities: Vec<String>) -> TaskFactory {
    Box::new(move || {
        cities
            .iter()
            .enumerate()
            .map(|(index, city)| {
                let city = city.clone();
                Task::new("simulated", index, async move {
                    // ⚠️ 模拟的是「等网络往返」——
                    //    sleep 在 async 里**不会**阻塞事件循环，
                    //    所以并发才真的能重叠。
                    tokio::time::sleep(Duration::from_millis(SIMULATED_IO_MS)).await;
                    Ok(format!("{city} 的模拟查询结果"))
                })
            })
            .collect::<Vec<Task>>()
    })
}

/// 构造一批「真调天气接口」的任务。
pub fn real_factory(cities: Vec<String>) -> TaskFactory {
    Box::new(move || {
        cities
            .iter()
            .enumerate()
            .map(|(index, city)| {
                let city = city.clone();
                Task::new("get_weather", index, async move {
                    let tool = crate::tools::WeatherTool::new(1, 1);
                    let args = serde_json::json!({ "location": city });
                    let ctx = crate::tools::ToolContext::with_timeout(Duration::from_secs(15));

                    tool.call(&args, &ctx).await.map_err(|e: ToolError| e)
                })
            })
            .collect::<Vec<Task>>()
    })
}

/// 构造一批「真阻塞」的任务（演示线程池轨道）。
///
/// ## ⚠️ 这是**同步阻塞**的负载
///
/// 它的「慢」来自 CPU / 阻塞等待，**不是** I/O 等待。
/// 所以并发对它**没有收益** —— 这正是要澄清的：
/// **并发 ≠ 并行**。
pub fn blocking_factory(topics: Vec<String>, block_ms: u64) -> TaskFactory {
    Box::new(move || {
        topics
            .iter()
            .enumerate()
            .map(|(index, topic)| {
                let topic = topic.clone();
                Task::new("slow_report", index, async move {
                    let tool = crate::tools::BlockingTool::new(block_ms);
                    let args = serde_json::json!({ "topic": topic });
                    let ctx = crate::tools::ToolContext::empty();

                    tool.call(&args, &ctx).await
                })
            })
            .collect::<Vec<Task>>()
    })
}

/// 跑一次完整对照实验。
pub async fn run_compare(
    cities: Vec<String>,
    limit: usize,
    simulated: bool,
) -> AppResult<Vec<BatchReport>> {
    if cities.is_empty() {
        return Err(AppError::BadRequest("城市列表不能为空".to_string()));
    }

    let factory: TaskFactory = if simulated {
        simulated_factory(cities.clone())
    } else {
        real_factory(cities.clone())
    };

    Ok(crate::concurrency::compare_all(factory, limit.max(1)).await)
}

/// 把四份报告渲染成终端表格。
pub fn render_table(reports: &[BatchReport]) -> String {
    let sequential_ms = reports
        .iter()
        .find(|r| r.mode == ExecutionMode::Sequential)
        .map(|r| r.total_ms())
        .unwrap_or(0);

    let mut out = String::new();
    out.push_str("执行模式                 总耗时(ms)   加速比    成功  失败  完成顺序\n");
    out.push_str("------------------------  ----------  -------  ----  ----  --------\n");

    for report in reports {
        let speedup = report.speedup(sequential_ms);
        let order = report
            .completion_order
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");

        out.push_str(&format!(
            "{:<24}  {:>10}  {:>6.2}x  {:>4}  {:>4}  {}\n",
            report.mode.label(),
            report.total_ms(),
            speedup,
            report.success_count(),
            report.failure_count(),
            order
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn simulated_compare_shows_speedup() {
        let cities = vec![
            "北京".to_string(),
            "上海".to_string(),
            "广州".to_string(),
            "深圳".to_string(),
        ];

        let reports = run_compare(cities, 4, true).await.unwrap();

        assert_eq!(reports.len(), 4);

        let seq = reports
            .iter()
            .find(|r| r.mode == ExecutionMode::Sequential)
            .unwrap();
        let par = reports
            .iter()
            .find(|r| r.mode == ExecutionMode::JoinAll)
            .unwrap();

        // ⚠️ 串行 4×120ms，并发约 120ms —— 加速比应 > 2x。
        assert!(
            par.speedup(seq.total_ms()) > 2.0,
            "加速比应 > 2x，实际 {:.2}x",
            par.speedup(seq.total_ms())
        );
    }

    #[tokio::test]
    async fn empty_city_list_is_rejected() {
        let err = run_compare(Vec::new(), 4, true).await.unwrap_err();
        assert_eq!(err.status_code(), 400);
    }

    #[tokio::test]
    async fn table_contains_all_four_modes() {
        let cities = vec!["北京".to_string(), "上海".to_string()];
        let reports = run_compare(cities, 2, true).await.unwrap();

        let table = render_table(&reports);

        assert!(table.contains("串行"));
        assert!(table.contains("全部并发"));
        assert!(table.contains("限流并发"));
    }
}
