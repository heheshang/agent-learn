//! 远程批量嵌入后端 —— 走真实 HTTP。
//!
//! # 为什么需要远程后端
//!
//! 本地 ONNX（fastembed）的批处理是「**一次前向**吃 N 条」；
//! 而远程服务的批处理是「**一次 HTTP 请求**带 N 条」。
//!
//! 两者省下的东西**不同**：
//!
//! | 后端 | 批处理省下的 | 瓶颈 |
//! |---|---|---|
//! | 本地 ONNX | 权重复制 + 前向的固定开销 | 算力 |
//! | 远程 HTTP | **网络往返**（31 次 RTT → 1 次）+ 服务端调度开销 | 网络 |
//!
//! ⚠️ 远程批处理的收益**完全取决于上游是否真的并行处理**。
//! 有些服务端收到 `input: [32 条]` 后仍然**逐条**跑模型 ——
//! 那你只是把 32 次 HTTP 往返压缩成 1 次，
//! **模型计算时间一分不省**。
//!
//! 所以本项目把远程后端也做成可对照的：
//! 「远程批量」vs「远程逐条」两个数一比，
//! 就知道这个上游**值不值得用批量**。
//!
//! # 后端实现
//!
//! | 后端 | 接口 | 说明 |
//! |---|---|---|
//! | [`OllamaEmbedder`] | `POST /api/embed` | 本地 Ollama 服务，免 key，支持 `input` 数组 |
//! | [`OpenAiEmbedder`] | `POST /v1/embeddings` | 兼容 OpenAI 协议，需 key |
//!
//! ## ⚠️ 批大小上限
//!
//! 两种协议都有**单请求条数上限**（OpenAI 是 2048，Ollama 无硬限制但受内存约束）。
//! 超限会被服务端拒绝 —— 所以本项目在**客户端**就分块，
//! 并且把「批大小」做成**可调**，因为**最优批大小是服务端相关的**：
//! 太大可能触发限流（429）或 OOM，太小则往返开销占主导。

use std::time::{Duration, Instant};

use futures::StreamExt;

use agent_logging::{debug, info, warn};
use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};
use crate::metrics::BatchStats;

/// 远程嵌入的 future 类型（`async fn` 在 trait 里不是 dyn 兼容）。
///
/// ## ⚠️ 为什么要显式装箱
///
/// `async fn` 在 trait 里**不是 dyn 兼容**的 ——
/// 所以 `&dyn RemoteEmbedder` 编不过。
/// 解决办法是像 07 的 `LlmFuture` / 14 的 `ToolFuture` 那样
/// 显式返回 `Pin<Box<dyn Future + Send + '_>>`。
pub type RemoteEmbedFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = AppResult<Vec<Vec<f32>>>> + Send + 'a>>;

/// 远程嵌入后端的通用接口。
///
/// ## 为什么抽 trait
///
/// 因为「远程批量」的收益**强依赖上游实现**：
/// 有的上游真并行，有的只是把 N 次循环包在一个 HTTP 请求里。
/// 要比较不同上游，必须有一个统一的调用面。
pub trait RemoteEmbedder: Send + Sync {
    /// 后端名。
    fn name(&self) -> &str;

    /// 单次请求**最多**能带多少条。
    ///
    /// 超出会被服务端拒绝 —— 客户端必须自己分块。
    fn max_batch_size(&self) -> usize;

    /// 嵌入一批（调用方保证 `texts.len() <= max_batch_size()`）。
    fn embed_batch<'a>(&'a self, texts: &'a [String]) -> RemoteEmbedFuture<'a>;
}

/// Ollama 的嵌入后端。
///
/// ```http
/// POST http://127.0.0.1:11434/api/embed
/// Content-Type: application/json
///
/// { "model": "qwen3-embedding:0.6b", "input": ["文本一", "文本二"] }
/// ```
///
/// 响应：
///
/// ```json
/// { "embeddings": [[0.1, 0.2, ...], [0.3, ...]] }
/// ```
///
/// ## ⚠️ Ollama **没有**批大小硬限制
///
/// 但也不能无限发 —— 一次塞几千条会让服务端内存暴涨。
/// 所以本项目默认按 32 条分块（可配置）。
pub struct OllamaEmbedder {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    max_batch: usize,
    timeout: Duration,
}

impl std::fmt::Debug for OllamaEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaEmbedder")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("max_batch", &self.max_batch)
            .finish()
    }
}

/// Ollama 的请求体。
#[derive(Debug, Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

/// Ollama 的响应体。
#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaEmbedder {
    /// 创建 Ollama 后端。
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>, max_batch: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.into(),
            model: model.into(),
            max_batch: max_batch.clamp(1, 2048),
            timeout: Duration::from_secs(300),
        }
    }

    /// 从环境变量创建。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `BATCH_OLLAMA_ENDPOINT` | 服务地址 | `http://127.0.0.1:11434` |
    /// | `BATCH_OLLAMA_MODEL` | 模型名 | `qwen3-embedding:0.6b` |
    pub fn from_env() -> Self {
        let endpoint = std::env::var("BATCH_OLLAMA_ENDPOINT")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());

        let model = std::env::var("BATCH_OLLAMA_MODEL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "qwen3-embedding:0.6b".to_string());

        let max_batch = std::env::var("BATCH_OLLAMA_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);

        Self::new(endpoint, model, max_batch)
    }

    /// 检查后端是否可达。
    pub async fn health_check(&self) -> AppResult<()> {
        let url = format!("{}/api/tags", self.endpoint.trim_end_matches('/'));
        let response = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| AppError::Upstream(format!("无法连接 Ollama（{url}）: {e}")))?;

        if !response.status().is_success() {
            return Err(AppError::Upstream(format!(
                "Ollama 健康检查失败: HTTP {}",
                response.status()
            )));
        }
        Ok(())
    }
}

impl RemoteEmbedder for OllamaEmbedder {
    fn name(&self) -> &str {
        &self.model
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch
    }

    fn embed_batch<'a>(&'a self, texts: &'a [String]) -> RemoteEmbedFuture<'a> {
        Box::pin(async move {
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            let url = format!("{}/api/embed", self.endpoint.trim_end_matches('/'));

            debug!(count = texts.len(), model = %self.model, "发起远程批量嵌入请求");

            let response = self
                .client
                .post(&url)
                .timeout(self.timeout)
                .json(&OllamaEmbedRequest {
                    model: &self.model,
                    input: texts,
                })
                .send()
                .await
                .map_err(|e| AppError::Upstream(format!("远程批量嵌入请求失败: {e}")))?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                // ⚠️ 429（限流）说明批太大或请求太密 —— 明确提示。
                if status.as_u16() == 429 {
                    warn!(count = texts.len(), "远程服务限流（429），请调小批大小");
                    return Err(AppError::Upstream(format!(
                        "远程服务限流（429），当前批大小 {} 可能过大: {body}",
                        texts.len()
                    )));
                }
                return Err(AppError::Upstream(format!(
                    "远程批量嵌入失败: HTTP {status}: {body}"
                )));
            }

            let body: OllamaEmbedResponse = response
                .json()
                .await
                .map_err(|e| AppError::Upstream(format!("解析远程响应失败: {e}")))?;

            // ⚠️ 条数必须对得上 —— 少一条就是「结果与输入错位」。
            if body.embeddings.len() != texts.len() {
                warn!(
                    input = texts.len(),
                    output = body.embeddings.len(),
                    "远程嵌入数量不匹配"
                );
                return Err(AppError::Upstream(format!(
                    "远程嵌入数量不匹配: 输入 {} 条, 输出 {} 条",
                    texts.len(),
                    body.embeddings.len()
                )));
            }

            Ok(body.embeddings)
        })
    }
}

/// 远程嵌入的对照实验报告。
///
/// ## ⚠️ **三组**，不是两组
///
/// 这是本项目**最反直觉的一次修正**。
///
/// 最初的对照只有两组：
///
/// | 组 | 做法 |
/// |---|---|
/// | A | **串行**逐条 HTTP（每次 1 条，一次一个请求） |
/// | B | **批量** HTTP（每次 N 条） |
///
/// 实测结果是 **0.99x** —— 批量**一点也没快**。
/// 看起来像是「远程批量没用」，但这个结论是**错的**。
///
/// ## ⚠️ 为什么是错的：漏掉了「客户端并行」这一档
///
/// LangChain 的 `batch()` 的定义是：
///
/// > Send multiple requests to a model in a batch for more efficient processing
/// > —— **客户端并行**发 N 个独立请求
///
/// 所以 `batch()` 的真正对照不是「串行逐条」，
/// 而是「**并发**逐条」！
///
/// 三组的完整关系：
///
/// | 组 | 做法 | 对应 LangChain | 少了什么 |
/// |---|---|---|---|
/// | A | **串行**逐条 HTTP | （无） | 既没并行也没合并 |
/// | B | **并发**逐条 HTTP（`max_concurrency`） | `batch()` | 少了「合并进一次请求」 |
/// | C | **批量** HTTP（一次 N 条） | provider 的 Batch API | 少了「客户端并行」 |
///
/// 真正的收益分解是：
///
/// ```text
/// A（串行逐条）
///   ↓ 客户端并行（省网络往返）
/// B（并发逐条）          ← LangChain 的 batch() 在这一层
///   ↓ 合并进一次请求（省请求开销 + 服务端调度）
/// C（批量 HTTP）          ← provider Batch API 在这一层
/// ```
///
/// ⚠️ **两个维度是正交的**：
/// 「并发」省的是**往返次数的墙钟时间**，
/// 「批量」省的是**每次请求的固定开销**。
/// 只做前者（B）通常比两者都做（C）更快 ——
/// 因为服务端并行度才是瓶颈。
///
/// 所以本项目的结论要**两个数一起看**：
///
/// - `C vs A` —— 批处理**端到端**的收益
/// - `C vs B` —— 「合并进一次请求」相对「客户端并行」的**增量**收益
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteCompareReport {
    /// 组 A：串行逐条 HTTP。
    pub sequential: BatchStats,

    /// 组 B：并发逐条 HTTP（LangChain `batch()` 的对应物）。
    pub concurrent: BatchStats,

    /// 组 C：批量 HTTP（一次请求 N 条）。
    pub batched: BatchStats,

    /// 批大小。
    pub batch_size: usize,

    /// 后端名。
    pub backend: String,
}

impl RemoteCompareReport {
    /// 端到端加速比（批量 vs 串行逐条）。
    pub fn end_to_end_speedup(&self) -> f64 {
        self.batched.speedup_vs(self.sequential.elapsed_ms)
    }

    /// **增量**加速比（批量 vs 并发逐条）。
    ///
    /// ⚠️ 这个数才是「合并进一次请求」本身的收益 ——
    /// 排除了「客户端并行」的贡献。
    pub fn incremental_speedup(&self) -> f64 {
        self.batched.speedup_vs(self.concurrent.elapsed_ms)
    }

    /// 客户端并行的收益（并发 vs 串行）。
    pub fn concurrency_speedup(&self) -> f64 {
        self.concurrent.speedup_vs(self.sequential.elapsed_ms)
    }
}

/// 远程对照实验：**串行逐条** vs **并发逐条** vs **批量**。
///
/// ## ⚠️ 三组必须用**同样的输入**
///
/// 且**顺序**相同 —— 否则吞吐数字没有可比性。
pub async fn compare_remote(
    embedder: &dyn RemoteEmbedder,
    texts: &[String],
    batch_size: usize,
    max_concurrency: usize,
) -> AppResult<RemoteCompareReport> {
    if texts.is_empty() {
        return Err(AppError::BadRequest("输入文本列表不能为空".to_string()));
    }

    let batch_size = batch_size.clamp(1, embedder.max_batch_size());
    let max_concurrency = max_concurrency.clamp(1, 64);

    // ── 组 A：串行逐条 ────────────────────────────────────────────
    let start = Instant::now();
    let mut seq_latencies = Vec::with_capacity(texts.len());

    for text in texts {
        let one_start = Instant::now();
        // ⚠️ 失败就**中断**整个对照实验 ——
        //    带着失败样本比出来的比值没有意义。
        embedder.embed_batch(std::slice::from_ref(text)).await?;
        seq_latencies.push(one_start.elapsed().as_millis() as u64);
    }

    let seq_elapsed = start.elapsed();
    let sequential = BatchStats::new(
        texts.len(),
        texts.len(),
        seq_elapsed,
        seq_latencies.clone(),
        seq_latencies,
        1,
        false,
    );

    // ── 组 B：并发逐条（LangChain `batch()` 的对应物）────────────
    //
    // ⚠️ 用 `buffer_unordered(max_concurrency)` 而**不是** `join_all` ——
    //    `join_all` 是**无限并发**，那会让「并发上限」这个变量失效，
    //    对照实验就没法控制「并发多少」这个维度了。
    //
    // ⚠️ 也不能用 `tokio::spawn` —— 它要求 future 是 `'static`，
    //    而这里要借用 `embedder` 和 `texts`。
    //    （与 22 项目「就地组合的并发原语」同一思路。）
    let start = Instant::now();

    let futures: Vec<_> = texts
        .iter()
        .map(|text| {
            let one_start = Instant::now();
            async move {
                let result = embedder.embed_batch(std::slice::from_ref(text)).await;
                (result, one_start.elapsed().as_millis() as u64)
            }
        })
        .collect();

    // ⚠️ `buffer_unordered` 是**限流**并发 ——
    //    同时最多跑 `max_concurrency` 个。
    let stream = futures::stream::iter(futures).buffer_unordered(max_concurrency);
    futures::pin_mut!(stream);

    let mut conc_latencies = Vec::with_capacity(texts.len());
    while let Some((result, latency)) = stream.next().await {
        // ⚠️ 失败就**中断**整个对照实验 ——
        //    带着失败样本比出来的比值没有意义。
        result?;
        conc_latencies.push(latency);
    }

    let conc_elapsed = start.elapsed();
    let concurrent = BatchStats::new(
        texts.len(),
        texts.len(),
        conc_elapsed,
        conc_latencies.clone(),
        conc_latencies,
        1,
        false,
    );

    // ── 组 C：批量 HTTP（一次请求 N 条）──────────────────────────
    let start = Instant::now();
    let mut batch_latencies = Vec::new();
    let mut item_latencies = Vec::with_capacity(texts.len());

    let chunks: Vec<&[String]> = texts.chunks(batch_size).collect();
    for chunk in &chunks {
        let batch_start = Instant::now();
        embedder.embed_batch(chunk).await?;
        let elapsed = batch_start.elapsed().as_millis() as u64;

        batch_latencies.push(elapsed);
        // ⚠️ 同一批里的每条**同时**完成 → 延迟相同。
        for _ in *chunk {
            item_latencies.push(elapsed);
        }
    }

    let batch_elapsed = start.elapsed();
    let batched = BatchStats::new(
        texts.len(),
        chunks.len(),
        batch_elapsed,
        item_latencies,
        batch_latencies,
        batch_size,
        false,
    );

    info!(
        backend = %embedder.name(),
        items = texts.len(),
        sequential_ms = sequential.elapsed_ms,
        concurrent_ms = concurrent.elapsed_ms,
        batched_ms = batched.elapsed_ms,
        concurrency_speedup = concurrent.speedup_vs(sequential.elapsed_ms),
        incremental_speedup = batched.speedup_vs(concurrent.elapsed_ms),
        "远程批处理对照完成"
    );

    Ok(RemoteCompareReport {
        sequential,
        concurrent,
        batched,
        batch_size,
        backend: embedder.name().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// 确定性的替身后端 —— 记录每次请求带了多少条。
    struct ScriptedRemote {
        calls: Mutex<Vec<usize>>,
        delay_ms: u64,
    }

    impl ScriptedRemote {
        fn new(delay_ms: u64) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                delay_ms,
            }
        }
    }

    impl RemoteEmbedder for ScriptedRemote {
        fn name(&self) -> &str {
            "scripted"
        }

        fn max_batch_size(&self) -> usize {
            64
        }

        fn embed_batch<'a>(
            &'a self,
            texts: &'a [String],
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = AppResult<Vec<Vec<f32>>>> + Send + 'a>,
        > {
            Box::pin(async move {
                self.calls.lock().unwrap().push(texts.len());
                if self.delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
                }
                Ok(texts.iter().map(|t| vec![t.len() as f32, 0.0]).collect())
            })
        }
    }

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("文本 {i}")).collect()
    }

    #[tokio::test]
    async fn sequential_uses_one_request_per_item() {
        let remote = ScriptedRemote::new(0);
        compare_remote(&remote, &texts(5), 2, 4).await.unwrap();

        let calls = remote.calls.lock().unwrap().clone();
        // 组 A 5 次（每次 1 条）+ 组 B 5 次（每次 1 条）+ 组 C 3 次（2+2+1）。
        assert_eq!(calls[0..5], vec![1, 1, 1, 1, 1]);
        assert_eq!(calls[5..10], vec![1, 1, 1, 1, 1]);
        assert_eq!(calls[10..], vec![2, 2, 1]);
    }

    #[tokio::test]
    async fn concurrent_group_also_uses_one_request_per_item() {
        // ⚠️ 组 B（并发逐条）也是**每次 1 条** ——
        //    它与组 A 的区别只在「同时发」还是「排队发」。
        let remote = ScriptedRemote::new(0);
        let report = compare_remote(&remote, &texts(10), 5, 4).await.unwrap();

        assert_eq!(report.concurrent.total_batches, 10);
        assert_eq!(report.sequential.total_batches, 10);
        assert_eq!(report.batched.total_batches, 2);
        assert_eq!(report.batched.batch_size, 5);
    }

    #[tokio::test]
    async fn concurrency_beats_sequential_when_requests_overlap() {
        // 每请求 10ms 固定延迟。
        // 串行：10 × 10ms = 100ms
        // 并发（4 路）：10 / 4 批 × 10ms ≈ 30ms
        let remote = ScriptedRemote::new(10);
        let report = compare_remote(&remote, &texts(10), 10, 4).await.unwrap();

        assert!(
            report.concurrent.elapsed_ms < report.sequential.elapsed_ms,
            "并发组应更快: {} vs {}",
            report.concurrent.elapsed_ms,
            report.sequential.elapsed_ms
        );
    }

    #[tokio::test]
    async fn incremental_speedup_is_measured_against_the_concurrent_group() {
        // ⚠️ 「合并进一次请求」的收益必须**相对并发逐条**来算 ——
        //    否则会把「客户端并行」的贡献也算进去。
        let remote = ScriptedRemote::new(10);
        let report = compare_remote(&remote, &texts(10), 10, 4).await.unwrap();

        assert!(
            (report.incremental_speedup()
                - report.batched.speedup_vs(report.concurrent.elapsed_ms))
            .abs()
                < 1e-9
        );
        assert!(
            (report.end_to_end_speedup() - report.batched.speedup_vs(report.sequential.elapsed_ms))
                .abs()
                < 1e-9
        );
    }

    #[tokio::test]
    async fn batch_size_is_clamped_to_backend_limit() {
        let remote = ScriptedRemote::new(0);
        // 后端上限 64，请求 1000 → 必须被 clamp。
        let report = compare_remote(&remote, &texts(100), 1000, 4).await.unwrap();
        assert_eq!(report.batch_size, 64);
        assert_eq!(report.batched.total_batches, 2);
    }

    #[tokio::test]
    async fn empty_input_is_rejected() {
        let remote = ScriptedRemote::new(0);
        let err = compare_remote(&remote, &[], 4, 4).await.unwrap_err();
        assert_eq!(err.status_code(), 400);
    }

    #[tokio::test]
    async fn batched_shares_one_request_across_all_items() {
        // ⚠️ 核心洞察：**一次请求服务多条** —— 这是批处理省下的东西。
        let remote = ScriptedRemote::new(10);
        let report = compare_remote(&remote, &texts(10), 10, 4).await.unwrap();

        // 逐条：10 次请求
        assert_eq!(report.sequential.total_batches, 10);
        // 批量：1 次请求
        assert_eq!(report.batched.total_batches, 1);

        // 批量组的单条延迟 = 该批耗时（10 条共享同一次延迟）
        assert_eq!(report.batched.p50(), report.batched.batch_latencies_ms[0]);
    }

    #[tokio::test]
    async fn batched_total_elapsed_is_shorter_than_sequential() {
        // 每请求 10ms 固定延迟 → 逐条 10×10ms，批量 1×10ms。
        let remote = ScriptedRemote::new(10);
        let report = compare_remote(&remote, &texts(10), 10, 4).await.unwrap();

        assert!(
            report.batched.elapsed_ms < report.sequential.elapsed_ms,
            "批量组应更快: {} vs {}",
            report.batched.elapsed_ms,
            report.sequential.elapsed_ms
        );
    }
}
