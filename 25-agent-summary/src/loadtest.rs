//! **压测轨** —— 进程内（归因）+ HTTP（验证）。
//!
//! # 两条轨的分工
//!
//! | 轨 | 测什么 | 用途 |
//! |---|---|---|
//! | **进程内**（`run_inprocess`） | 纯被测逻辑 | **归因** —— 差值归因到具体环节 |
//! | **HTTP**（`run_http`） | 完整栈（含 HTTP 开销） | **验证** —— 外部视角的真实表现 |
//!
//! # ★ 这两条轨**不能混着比**
//!
//! 进程内测出「算法快了 5.16x」，HTTP 轨测出「QPS 只快了 1.03x」——
//! 这**不是矛盾**，而是：
//!
//! > **归因到哪一层，取决于你在哪一层测。**
//!
//! 根因是 HTTP 栈开销（1~4ms）比被测逻辑（0.1ms）大一个数量级。
//! 进程内轨用来**归因**，HTTP 轨用来**验证**。

use std::sync::Arc;
use std::time::Instant;

use futures::stream::{self, StreamExt};

use crate::metrics::{LoadReport, Sample};

/// 压测配置。
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// 请求数。
    pub requests: usize,

    /// 并发度。
    pub concurrency: usize,

    /// 模拟的服务端时间（毫秒）—— 0 表示不模拟。
    pub service_time_ms: u64,

    /// 服务端 worker 数 —— **限流的关键**。
    ///
    /// ⚠️ 没有它就**测不出排队效应**：
    /// `buffer_unordered(N)` 让 N 个任务**同时**进入 sleep，谁也没排队，
    /// 于是「并发 8 的 p50」≈「并发 1 的 p50」，是一条平线。
    pub server_workers: usize,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            requests: 500,
            concurrency: 16,
            service_time_ms: 0,
            server_workers: 8,
        }
    }
}

/// 一个被测操作。
///
/// ## 为什么用闭包而不是 trait
///
/// 因为闭包能直接捕获 `Arc` 这类**准备好的状态**，
/// 压测时不需要每次重新构造（那是**另一个**环节的成本）。
pub type BenchFn = Arc<dyn Fn(usize) -> bool + Send + Sync>;

/// 进程内压测。
///
/// ## 为什么用 `buffer_unordered` 而不是 `tokio::spawn`
///
/// `spawn` 要求 future 是 `'static`，而这里要借用 `&BenchFn`。
/// 只能**就地组合**并发原语。
pub async fn run_inprocess(f: BenchFn, config: &LoadConfig) -> LoadReport {
    run_inprocess_named("inprocess", f, config).await
}

/// 进程内压测（**带变体名**）。
///
/// ⚠️ 变体名会出现在报告里 —— 压测表要能区分「哪一档」，
/// 不能全都叫 `inprocess`。
pub async fn run_inprocess_named(variant: &str, f: BenchFn, config: &LoadConfig) -> LoadReport {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(config.server_workers.max(1)));
    let service_time = std::time::Duration::from_millis(config.service_time_ms);

    let start = Instant::now();
    let mut samples: Vec<Sample> = Vec::with_capacity(config.requests);

    let tasks = stream::iter(0..config.requests).map(|i| {
        let f = Arc::clone(&f);
        let semaphore = Arc::clone(&semaphore);
        async move {
            // ⚠️ 计时**必须从抢 permit 之前**开始 ——
            //    否则「排队时间」被漏掉，测出来的只是「服务时间」。
            //
            //    排队时间是客户端**真实感受到**的一部分：
            //    请求发出 → 排队 → 才开始服务。
            let t0 = Instant::now();

            // ★ 先抢 permit —— 抢不到就在这里**排队**。
            let _permit = semaphore.acquire_owned().await;

            if !service_time.is_zero() {
                tokio::time::sleep(service_time).await;
            }
            let ok = f(i);
            Sample {
                ok,
                elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
                status: None,
                cache_hit: None,
            }
        }
    });

    let mut stream = tasks.buffer_unordered(config.concurrency.max(1));
    while let Some(sample) = stream.next().await {
        samples.push(sample);
    }

    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    LoadReport::from_samples(variant, &samples, wall_ms, config.concurrency)
}

/// HTTP 压测 —— 打真实的服务。
pub async fn run_http(
    client: &reqwest::Client,
    base_url: &str,
    path: &str,
    body: serde_json::Value,
    config: &LoadConfig,
) -> LoadReport {
    let start = Instant::now();
    let mut samples: Vec<Sample> = Vec::with_capacity(config.requests);

    let url = format!("{base_url}{path}");
    let tasks = stream::iter(0..config.requests).map(|_| {
        let client = client.clone();
        let url = url.clone();
        let body = body.clone();
        async move {
            let t0 = Instant::now();
            let outcome = client.post(&url).json(&body).send().await;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            match outcome {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let ok = resp.status().is_success();
                    Sample {
                        ok,
                        elapsed_ms,
                        status: Some(status),
                        cache_hit: None,
                    }
                }
                Err(_) => Sample {
                    ok: false,
                    elapsed_ms,
                    status: None,
                    cache_hit: None,
                },
            }
        }
    });

    let mut stream = tasks.buffer_unordered(config.concurrency.max(1));
    while let Some(sample) = stream.next().await {
        samples.push(sample);
    }

    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    LoadReport::from_samples("http", &samples, wall_ms, config.concurrency)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(requests: usize, concurrency: usize) -> LoadConfig {
        LoadConfig {
            requests,
            concurrency,
            service_time_ms: 0,
            server_workers: 8,
        }
    }

    #[tokio::test]
    async fn inprocess_counts_requests() {
        let f: BenchFn = Arc::new(|_| true);
        let report = run_inprocess(f, &config(100, 10)).await;
        assert_eq!(report.total_requests, 100);
        assert_eq!(report.failed, 0);
    }

    #[tokio::test]
    async fn inprocess_records_failures() {
        let f: BenchFn = Arc::new(|i| i % 2 == 0);
        let report = run_inprocess(f, &config(100, 10)).await;
        assert_eq!(report.failed, 50);
        assert!((report.success_rate - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn without_worker_limit_there_is_no_queueing() {
        // ⚠️ 这是**前提测试** —— 锁住「没有 worker 上限就没有排队」这个事实。
        //
        // 没有 permit 限制时，并发 8 的任务**同时**进入 sleep，
        // 谁也没排队 → p50 ≈ 并发 1 的 p50。
        let f: BenchFn = Arc::new(|_| true);
        let mut cfg = config(40, 8);
        cfg.service_time_ms = 50;
        cfg.server_workers = 1000; // 相当于无限 worker

        let high = run_inprocess(Arc::clone(&f), &cfg).await;

        cfg.concurrency = 1;
        let low = run_inprocess(f, &cfg).await;

        // 没有排队 → 延迟基本持平。
        assert!(
            high.p50_ms < low.p50_ms * 1.5,
            "没有 worker 上限时不该出现排队：high={:.1} low={:.1}",
            high.p50_ms,
            low.p50_ms
        );
    }

    #[tokio::test]
    async fn worker_limit_creates_queueing() {
        // ★ 对照组 —— 有 worker 上限时，高并发应该排队（p50 明显上升）。
        let f: BenchFn = Arc::new(|_| true);
        let mut cfg = config(40, 8);
        cfg.service_time_ms = 50;
        cfg.server_workers = 2; // 只有 2 个 worker

        let high = run_inprocess(Arc::clone(&f), &cfg).await;

        cfg.concurrency = 1;
        let low = run_inprocess(f, &cfg).await;

        assert!(
            high.p50_ms > low.p50_ms * 1.5,
            "有 worker 上限时高并发应该排队：high={:.1} low={:.1}",
            high.p50_ms,
            low.p50_ms
        );
    }
}
