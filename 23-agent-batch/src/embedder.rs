//! 批量嵌入 —— 本项目的核心产出之一。
//!
//! # 与基线的差别
//!
//! 基线（05 项目的 [`rag_05::LocalEmbedder`]）：
//!
//! ```text
//! for text in texts {
//!     model.embed(vec![text], None)   ← 每次一个 batch_size=1 的前向
//! }
//! ```
//!
//! 本项目：
//!
//! ```text
//! for chunk in texts.chunks(32) {
//!     model.embed(chunk.to_vec(), None)  ← 一次前向吃 32 条
//! }
//! ```
//!
//! 唯一的差别是**一次前向喂多少条**。
//! 剩下的（padding 浪费、背压、分桶）都在别处解决。
//!
//! # ⚠️ 为什么**不**直接调 `LocalEmbedder::embed_documents`
//!
//! 因为它的 `embed_documents` 内部**硬编码**了「一条一批」：
//!
//! ```text
//! // rag_05::embed::LocalEmbedder::embed_documents
//! self.model.embed(texts.to_vec(), None)
//! ```
//!
//! 等等 —— 它其实**已经是批量的**（`texts.to_vec()` 一次全喂）。
//! 那基线在哪？基线在**调用方**：05 的流水线是
//! 「加载 → 分块 → 一次性 embed 全部」的批处理形态，
//! 而**在线单条查询**走的是 `embed_query` —— 那才是真正的逐条。
//!
//! 所以本项目的对照组这样设计（**变量单一**）：
//!
//! | 组 | 做法 | 用途 |
//! |---|---|---|
//! | A | 逐条调 `embed_query`（每条一次前向） | 基线 |
//! | B | 固定批（`batch_size` 条一次前向） | 主要对照组 |
//! | C | 长度分桶后定批 | 减少 padding 浪费 |
//!
//! 三组**跑的是同一批文本**，只有批的大小策略不同。

use std::sync::Arc;
use std::time::Instant;

use agent_logging::{debug, info, warn};
use rag_05::embed::{Embedder, LocalEmbedder, Model};

use crate::batching::{BatchConfig, BatchInfo, BatchStrategy, bucket_by_length, chunk_fixed};
use crate::error::{AppError, AppResult};
use crate::metrics::BatchStats;

/// 批量嵌入器。
///
/// ## 为什么 `Arc` 包住模型
///
/// ONNX 推理是 **CPU 密集的同步调用**，
/// 必须放到 `spawn_blocking` 的阻塞线程池里跑，
/// 否则会卡住 tokio 的工作线程。
///
/// 而 `spawn_blocking` 要求闭包是 `'static` —— 不能借用 `&self`。
/// 所以这里克隆一份 `Arc` 塞进闭包，而不是借用 `self`。
///
/// ⚠️ 这个坑来自 08 项目的 `LocalCrossEncoder` —— 同一个模式。
pub struct BatchEmbedder {
    /// 底层嵌入器。
    ///
    /// ⚠️ 用 `Arc<dyn Embedder>` 而不是 `Arc<LocalEmbedder>` ——
    ///    这样测试能注入替身，且本地模型与远程模型共用同一条代码路径。
    inner: Arc<dyn Embedder>,
    config: BatchConfig,
}

impl std::fmt::Debug for BatchEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchEmbedder")
            .field("model", &self.inner.model_name())
            .field("dimension", &self.inner.dimension())
            .field("config", &self.config)
            .finish()
    }
}

impl BatchEmbedder {
    /// 创建批量嵌入器（会下载模型，首次较慢）。
    pub fn new(model: Model, cache_dir: Option<std::path::PathBuf>, config: BatchConfig) -> Self {
        let inner = LocalEmbedder::new(model, cache_dir)
            .expect("嵌入模型初始化失败（请检查模型缓存目录与网络）");

        Self {
            inner: Arc::new(inner),
            config,
        }
    }

    /// 用已有的嵌入器包装（测试用替身）。
    pub fn from_embedder(embedder: Arc<dyn Embedder>, config: BatchConfig) -> Self {
        // ⚠️ `LocalEmbedder` 是具体类型，但 `Embedder` 是 trait。
        //    为了让测试能注入替身，这里用 `Arc<dyn Embedder>`。
        //    代价：本地模型走不到 `Arc<LocalEmbedder>` 的快路径，
        //    但 `embed_documents` 本身是 trait 方法，行为一致。
        Self {
            inner: embedder,
            config,
        }
    }

    /// 当前配置。
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// 底层嵌入器的 `Arc`（供扫描模块复用同一模型实例）。
    pub fn inner_arc(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.inner)
    }

    /// 模型名。
    pub fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    /// 向量维度。
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// ★ 用指定策略批量嵌入。
    ///
    /// 返回：`(向量列表, 度量)`。
    ///
    /// ## ⚠️ 结果顺序**严格等于输入顺序**
    ///
    /// 这是批处理的基本契约。
    /// 分桶策略内部会排序，但返回前**必须还原**（坑 3）。
    pub async fn embed_batch(
        &self,
        texts: &[String],
        strategy: BatchStrategy,
    ) -> AppResult<(Vec<Vec<f32>>, BatchStats)> {
        if texts.is_empty() {
            return Err(AppError::BadRequest("输入文本列表不能为空".to_string()));
        }

        let start = Instant::now();

        let (vectors, batches, item_latencies, batch_latencies) = match strategy {
            BatchStrategy::Sequential => self.embed_sequential(texts).await?,
            BatchStrategy::Fixed => self.embed_fixed(texts).await?,
            BatchStrategy::Bucketed => self.embed_bucketed(texts).await?,
        };

        let elapsed = start.elapsed();

        let stats = BatchStats::new(
            texts.len(),
            batches.len(),
            elapsed,
            item_latencies,
            batch_latencies,
            match strategy {
                BatchStrategy::Sequential => 1,
                BatchStrategy::Fixed => self.config.batch_size,
                BatchStrategy::Bucketed => self.config.bucket_batch_size,
            },
            matches!(strategy, BatchStrategy::Bucketed),
        );

        info!(
            strategy = strategy.label(),
            items = texts.len(),
            batches = batches.len(),
            elapsed_ms = stats.elapsed_ms,
            throughput = stats.throughput_per_sec(),
            "批量嵌入完成"
        );

        Ok((vectors, stats))
    }

    /// 组 A：逐条调用（基线）。
    ///
    /// 每条文本**独立**跑一次前向 —— 固定开销被反复支付 N 次。
    async fn embed_sequential(
        &self,
        texts: &[String],
    ) -> AppResult<(Vec<Vec<f32>>, Vec<BatchInfo>, Vec<u64>, Vec<u64>)> {
        let mut vectors = Vec::with_capacity(texts.len());
        let mut batches = Vec::new();
        let mut item_latencies = Vec::with_capacity(texts.len());
        let mut batch_latencies = Vec::with_capacity(texts.len());

        for (i, text) in texts.iter().enumerate() {
            let batch_start = Instant::now();
            let vec = self.embed_one(text).await?;
            let elapsed = batch_start.elapsed().as_millis() as u64;

            vectors.push(vec);
            item_latencies.push(elapsed);
            batch_latencies.push(elapsed);
            batches.push(BatchInfo {
                index: i,
                size: 1,
                elapsed_ms: elapsed,
            });
        }

        Ok((vectors, batches, item_latencies, batch_latencies))
    }

    /// 组 B：固定大小批。
    async fn embed_fixed(
        &self,
        texts: &[String],
    ) -> AppResult<(Vec<Vec<f32>>, Vec<BatchInfo>, Vec<u64>, Vec<u64>)> {
        let chunks = chunk_fixed(texts, self.config.batch_size);
        let mut vectors = Vec::with_capacity(texts.len());
        let mut batches = Vec::new();
        let mut item_latencies = Vec::with_capacity(texts.len());
        let mut batch_latencies = Vec::with_capacity(chunks.len());

        for (bi, indices) in chunks.iter().enumerate() {
            let batch_texts: Vec<String> = indices.iter().map(|i| texts[*i].clone()).collect();
            let batch_start = Instant::now();

            let batch_vectors = self.embed_many(&batch_texts).await?;
            let elapsed = batch_start.elapsed().as_millis() as u64;

            // ⚠️ 一批里的**每一条**延迟 = 该批的耗时（它们同时完成）。
            //    这正是「批处理抬高单条延迟」的量化体现。
            for _ in indices {
                item_latencies.push(elapsed);
            }
            batch_latencies.push(elapsed);
            batches.push(BatchInfo {
                index: bi,
                size: indices.len(),
                elapsed_ms: elapsed,
            });

            vectors.extend(batch_vectors);
        }

        Ok((vectors, batches, item_latencies, batch_latencies))
    }

    /// 组 C：长度分桶后定批。
    async fn embed_bucketed(
        &self,
        texts: &[String],
    ) -> AppResult<(Vec<Vec<f32>>, Vec<BatchInfo>, Vec<u64>, Vec<u64>)> {
        let buckets = bucket_by_length(texts, self.config.bucket_batch_size);
        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut batches = Vec::new();
        let mut item_latencies = vec![0u64; texts.len()];
        let mut batch_latencies = Vec::with_capacity(buckets.len());

        for (bi, bucket) in buckets.iter().enumerate() {
            let batch_texts: Vec<String> = bucket.iter().map(|(i, _)| texts[*i].clone()).collect();
            let batch_start = Instant::now();

            let batch_vectors = self.embed_many(&batch_texts).await?;
            let elapsed = batch_start.elapsed().as_millis() as u64;

            // ⚠️⚠️ **关键**：按原始下标**放回原位**。
            //    分桶内部把输入按长度排过序，
            //    不还原的话结果就与输入**错位**了 —— 静默的数据错误。
            for (slot, ((original_index, _), vector)) in
                bucket.iter().zip(batch_vectors).enumerate()
            {
                vectors[*original_index] = Some(vector);
                item_latencies[*original_index] = elapsed;
                let _ = slot;
            }

            batch_latencies.push(elapsed);
            batches.push(BatchInfo {
                index: bi,
                size: bucket.len(),
                elapsed_ms: elapsed,
            });
        }

        // ⚠️ 理论上不该有空位 —— 但**不能**用 unwrap 把 panic 留给线上。
        //    缺一条就是「结果条数对不上」，必须报出来。
        let vectors: Vec<Vec<f32>> = vectors
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                v.ok_or_else(|| AppError::Internal(format!("分桶嵌入缺少第 {i} 条的结果")))
            })
            .collect::<AppResult<Vec<_>>>()?;

        Ok((vectors, batches, item_latencies, batch_latencies))
    }

    /// 嵌入一条（走 trait 的 `embed_query`，自动加查询前缀）。
    async fn embed_one(&self, text: &str) -> AppResult<Vec<f32>> {
        let embedder = Arc::clone(&self.inner);
        let text = text.to_string();

        let result = tokio::task::spawn_blocking(move || embedder.embed_query(&text))
            .await
            .map_err(|e| AppError::Internal(format!("嵌入任务失败: {e}")))?;

        result.map_err(|e| AppError::Inference(e.to_string()))
    }

    /// 嵌入一批（**一条前向**）。
    async fn embed_many(&self, texts: &[String]) -> AppResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let embedder = Arc::clone(&self.inner);
        let texts = texts.to_vec();

        debug!(count = texts.len(), "发起批量前向");

        let expected = texts.len();
        let result = tokio::task::spawn_blocking(move || embedder.embed_documents(&texts))
            .await
            .map_err(|e| AppError::Internal(format!("批量嵌入任务失败: {e}")))?;

        let vectors = result.map_err(|e| AppError::Inference(e.to_string()))?;

        // ⚠️ 条数必须对得上 —— 少一条就是「结果与输入错位」，
        //    属于**静默的数据错误**，必须在源头拦住。
        if vectors.len() != expected {
            warn!(input = expected, output = vectors.len(), "嵌入数量不匹配");
            return Err(AppError::Inference(format!(
                "嵌入数量不匹配: 输入 {} 条, 输出 {} 条",
                expected,
                vectors.len()
            )));
        }

        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// 确定性的替身嵌入器 —— 记录每次调用喂进去多少条。
    struct ScriptedEmbedder {
        dimension: usize,
        calls: Mutex<Vec<usize>>,
    }

    impl ScriptedEmbedder {
        fn new() -> Self {
            Self {
                dimension: 4,
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl Embedder for ScriptedEmbedder {
        fn model_name(&self) -> &str {
            "scripted"
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn embed_documents(
            &self,
            texts: &[String],
        ) -> Result<Vec<Vec<f32>>, rag_05::embed::EmbedError> {
            self.calls.lock().unwrap().push(texts.len());
            Ok(texts
                .iter()
                .map(|t| vec![t.chars().count() as f32, 1.0, 0.0, 0.0])
                .collect())
        }

        fn embed_query(&self, text: &str) -> Result<Vec<f32>, rag_05::embed::EmbedError> {
            // ⚠️ 必须**也**记录 —— 逐条策略走的是 `embed_query`，
            //    不记录就测不到「N 次单条前向」。
            self.calls.lock().unwrap().push(1);
            Ok(vec![text.chars().count() as f32, 1.0, 0.0, 0.0])
        }
    }

    fn scripted(config: BatchConfig) -> BatchEmbedder {
        BatchEmbedder::from_embedder(Arc::new(ScriptedEmbedder::new()), config)
    }

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("文本 {i}")).collect()
    }

    #[tokio::test]
    async fn sequential_calls_one_item_at_a_time() {
        let embedder = scripted(BatchConfig {
            batch_size: 8,
            ..Default::default()
        });

        let (vectors, stats) = embedder
            .embed_batch(&texts(5), BatchStrategy::Sequential)
            .await
            .unwrap();

        assert_eq!(vectors.len(), 5);
        assert_eq!(stats.total_batches, 5);
        assert_eq!(stats.total_items, 5);
    }

    #[tokio::test]
    async fn fixed_batches_group_items() {
        let embedder = scripted(BatchConfig {
            batch_size: 8,
            ..Default::default()
        });

        let (vectors, stats) = embedder
            .embed_batch(&texts(10), BatchStrategy::Fixed)
            .await
            .unwrap();

        assert_eq!(vectors.len(), 10);
        // 10 条 / 批大小 8 → 2 批
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.batch_latencies_ms.len(), 2);
    }

    #[tokio::test]
    async fn bucketed_results_match_input_order() {
        let embedder = scripted(BatchConfig {
            bucket_batch_size: 2,
            ..Default::default()
        });

        // 长度差异很大 → 分桶一定会打乱顺序 → 正好验证还原逻辑。
        let input = vec![
            "短".to_string(),
            "很长的一段内容".repeat(5),
            "中".to_string(),
            "很长的一段内容".repeat(6),
        ];

        let (vectors, stats) = embedder
            .embed_batch(&input, BatchStrategy::Bucketed)
            .await
            .unwrap();

        assert_eq!(vectors.len(), input.len());
        assert!(stats.bucketed);

        // ⚠️ 每条向量的第一个分量 = 文本长度 ——
        //    用它验证「结果确实对回了原来的输入」。
        for (original, vector) in input.iter().zip(vectors.iter()) {
            assert_eq!(vector[0] as usize, original.chars().count());
        }
    }

    #[tokio::test]
    async fn bucketed_order_matches_fixed_order() {
        // ⚠️ 最重要的不变量：**批处理不改变结果内容，只改变顺序与耗时**。
        let embedder = scripted(BatchConfig {
            batch_size: 3,
            bucket_batch_size: 3,
            ..Default::default()
        });

        let input = vec![
            "aaaa".to_string(),
            "b".to_string(),
            "cccccccc".to_string(),
            "dd".to_string(),
            "eeeeeeeeee".to_string(),
            "f".to_string(),
        ];

        let (fixed, _) = embedder
            .embed_batch(&input, BatchStrategy::Fixed)
            .await
            .unwrap();
        let (bucketed, _) = embedder
            .embed_batch(&input, BatchStrategy::Bucketed)
            .await
            .unwrap();

        assert_eq!(fixed, bucketed, "分桶改变了结果内容 —— 这是 bug");
    }

    #[tokio::test]
    async fn empty_input_is_rejected() {
        let embedder = scripted(BatchConfig::default());
        let err = embedder
            .embed_batch(&[], BatchStrategy::Fixed)
            .await
            .unwrap_err();
        assert_eq!(err.status_code(), 400);
    }

    #[tokio::test]
    async fn fixed_strategy_really_batches_the_underlying_calls() {
        let inner = Arc::new(ScriptedEmbedder::new());
        let embedder = BatchEmbedder::from_embedder(
            inner.clone() as Arc<dyn Embedder>,
            BatchConfig {
                batch_size: 4,
                ..Default::default()
            },
        );

        embedder
            .embed_batch(&texts(8), BatchStrategy::Fixed)
            .await
            .unwrap();

        let calls = inner.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![4, 4], "固定批应产生两次 4 条的前向");
    }

    #[tokio::test]
    async fn sequential_strategy_produces_one_call_per_item() {
        let inner = Arc::new(ScriptedEmbedder::new());
        let embedder = BatchEmbedder::from_embedder(
            inner.clone() as Arc<dyn Embedder>,
            BatchConfig::default(),
        );

        embedder
            .embed_batch(&texts(3), BatchStrategy::Sequential)
            .await
            .unwrap();

        let calls = inner.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![1, 1, 1], "逐条策略应产生三次单条前向");
    }
}
