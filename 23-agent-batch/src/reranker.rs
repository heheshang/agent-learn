//! 批量重排 —— 本项目的核心产出之二。
//!
//! # 重排为什么特别适合批处理
//!
//! 重排是 **cross-encoder**：把「查询 + 文档」拼在一起送进模型。
//!
//! ```text
//! 查询 + 文档₁ ──encode──> 分数₁
//! 查询 + 文档₂ ──encode──> 分数₂
//! 查询 + 文档₃ ──encode──> 分数₃
//! ```
//!
//! 每个候选都要跑一次模型 —— 这是重排**慢**的根源，
//! 也是它**最值得批处理**的地方：
//!
//! - 一次前向的**固定开销**（权重复制到计算设备）被 N 个候选分摊
//! - 32 个候选塞进一次前向，比 32 次单条快得多
//!
//! # 与 08 项目的关系
//!
//! 08 的 [`rag_08::LocalCrossEncoder`] 内部用 fastembed 的
//! `rerank(query, documents, false, None)` —— 那里 `batch_size = None`
//! 意味着走 fastembed 的默认块大小（256），
//! 所以它**已经**是「一次前向算一批候选」了。
//!
//! 那本项目加了什么？**对照、度量与正确性验证**：
//!
//! | 组 | 做法 | 用途 |
//! |---|---|---|
//! | A | 逐个候选调（每次一个前向） | 基线 |
//! | B | 一次前向算一批候选 | 主要对照组 |
//! | C | 长度分桶后分批 | 减少 padding 浪费 |
//!
//! 并且**验证一个关键不变量**：
//! **批处理后的排序结果必须与逐条完全一致** ——
//! 分数不变、顺序不变、选出的 top_n 不变。
//!
//! ⚠️ 这个不变量不是理所当然的：
//! fastembed 的 rerank 内部用 `par_chunks` 分块，
//! **每块是独立前向**，而 padding 是 `BatchLongest`（按块内最长条补）——
//! 换批大小会改变 padding，进而**可能**改变浮点结果。
//! 所以这个不变量**必须实测验证**，不能假设。
//! 回归测试 `batched_rerank_matches_sequential_rerank` 锁住它。

use std::sync::Arc;
use std::time::Instant;

use agent_logging::{debug, info, warn};
use rag_08::reranker::{RerankDocument, RerankedHit, Reranker, RerankerModel};

use crate::batching::{BatchConfig, BatchInfo, BatchStrategy, bucket_by_length, chunk_fixed};
use crate::error::{AppError, AppResult};
use crate::metrics::BatchStats;

/// 批量重排器。
pub struct BatchReranker {
    inner: Arc<dyn Reranker>,
    config: BatchConfig,
}

impl std::fmt::Debug for BatchReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchReranker")
            .field("model", &self.inner.name())
            .field("config", &self.config)
            .finish()
    }
}

impl BatchReranker {
    /// 创建批量重排器（会下载模型，首次较慢）。
    pub fn new(
        model: RerankerModel,
        cache_dir: Option<std::path::PathBuf>,
        config: BatchConfig,
    ) -> Self {
        let inner = rag_08::LocalCrossEncoder::new(model, cache_dir)
            .expect("重排模型初始化失败（请检查模型缓存目录与网络）");

        Self {
            inner: Arc::new(inner),
            config,
        }
    }

    /// 用已有的重排器包装（测试用替身）。
    pub fn from_reranker(reranker: Arc<dyn Reranker>, config: BatchConfig) -> Self {
        Self {
            inner: reranker,
            config,
        }
    }

    /// 当前配置。
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// 模型名。
    pub fn model_name(&self) -> &str {
        self.inner.name()
    }

    /// ★ 批量重排。
    ///
    /// 返回：`(重排结果, 度量)`。
    ///
    /// ## ⚠️ 结果**必须与逐条一致**
    ///
    /// 这是批处理的**正确性契约**：
    /// 批处理只该影响**耗时**，不该影响**结果**。
    pub async fn rerank_batch(
        &self,
        query: &str,
        documents: &[RerankDocument],
        top_n: usize,
        strategy: BatchStrategy,
    ) -> AppResult<(Vec<RerankedHit>, BatchStats)> {
        if documents.is_empty() {
            return Err(AppError::BadRequest("候选文档列表不能为空".to_string()));
        }

        let start = Instant::now();

        let (hits, batches, item_latencies, batch_latencies) = match strategy {
            BatchStrategy::Sequential => self.rerank_sequential(query, documents, top_n).await?,
            BatchStrategy::Fixed => self.rerank_fixed(query, documents, top_n).await?,
            BatchStrategy::Bucketed => self.rerank_bucketed(query, documents, top_n).await?,
        };

        let elapsed = start.elapsed();

        let stats = BatchStats::new(
            documents.len(),
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
            candidates = documents.len(),
            batches = batches.len(),
            hits = hits.len(),
            elapsed_ms = stats.elapsed_ms,
            "批量重排完成"
        );

        Ok((hits, stats))
    }

    /// 组 A：逐个候选（每次一个前向）。
    ///
    /// ⚠️ 每次调用都只送**一个**文档，
    /// 所以固定开销被支付了 N 次 —— 这是基线。
    async fn rerank_sequential(
        &self,
        query: &str,
        documents: &[RerankDocument],
        top_n: usize,
    ) -> AppResult<(Vec<RerankedHit>, Vec<BatchInfo>, Vec<u64>, Vec<u64>)> {
        let mut all_scores: Vec<(usize, f32)> = Vec::with_capacity(documents.len());
        let mut batches = Vec::new();
        let mut item_latencies = Vec::with_capacity(documents.len());
        let mut batch_latencies = Vec::with_capacity(documents.len());

        for (i, doc) in documents.iter().enumerate() {
            let batch_start = Instant::now();
            let hits = self.rerank_one(query, std::slice::from_ref(doc)).await?;
            let elapsed = batch_start.elapsed().as_millis() as u64;

            if let Some(hit) = hits.first() {
                all_scores.push((i, hit.score));
            }

            item_latencies.push(elapsed);
            batch_latencies.push(elapsed);
            batches.push(BatchInfo {
                index: i,
                size: 1,
                elapsed_ms: elapsed,
            });
        }

        let hits = build_hits(all_scores, documents, top_n);
        Ok((hits, batches, item_latencies, batch_latencies))
    }

    /// 组 B：固定大小批。
    async fn rerank_fixed(
        &self,
        query: &str,
        documents: &[RerankDocument],
        top_n: usize,
    ) -> AppResult<(Vec<RerankedHit>, Vec<BatchInfo>, Vec<u64>, Vec<u64>)> {
        let chunks = chunk_fixed(&texts_of(documents), self.config.batch_size);
        let mut all_scores: Vec<(usize, f32)> = Vec::with_capacity(documents.len());
        let mut batches = Vec::new();
        let mut item_latencies = vec![0u64; documents.len()];
        let mut batch_latencies = Vec::with_capacity(chunks.len());

        for (bi, indices) in chunks.iter().enumerate() {
            let subset: Vec<RerankDocument> =
                indices.iter().map(|i| documents[*i].clone()).collect();

            let batch_start = Instant::now();
            let hits = self.rerank_one(query, &subset).await?;
            let elapsed = batch_start.elapsed().as_millis() as u64;

            // ⚠️ `hits` 里的 `original_index` 是**该子集内**的下标，
            //    要映射回**全局**下标 —— 否则分数会挂到错误的文档上。
            for hit in &hits {
                if let Some(global) = indices.get(hit.original_index) {
                    all_scores.push((*global, hit.score));
                    item_latencies[*global] = elapsed;
                }
            }

            batch_latencies.push(elapsed);
            batches.push(BatchInfo {
                index: bi,
                size: indices.len(),
                elapsed_ms: elapsed,
            });
        }

        let hits = build_hits(all_scores, documents, top_n);
        Ok((hits, batches, item_latencies, batch_latencies))
    }

    /// 组 C：长度分桶后分批。
    async fn rerank_bucketed(
        &self,
        query: &str,
        documents: &[RerankDocument],
        top_n: usize,
    ) -> AppResult<(Vec<RerankedHit>, Vec<BatchInfo>, Vec<u64>, Vec<u64>)> {
        // ⚠️ 分桶用**送进模型的文本**（标题 + 正文）的长度 ——
        //    因为 padding 是按实际送进去的序列算的。
        let buckets = bucket_by_length(&texts_of(documents), self.config.bucket_batch_size);

        let mut all_scores: Vec<(usize, f32)> = Vec::with_capacity(documents.len());
        let mut batches = Vec::new();
        let mut item_latencies = vec![0u64; documents.len()];
        let mut batch_latencies = Vec::with_capacity(buckets.len());

        for (bi, bucket) in buckets.iter().enumerate() {
            let subset: Vec<RerankDocument> =
                bucket.iter().map(|(i, _)| documents[*i].clone()).collect();

            let batch_start = Instant::now();
            let hits = self.rerank_one(query, &subset).await?;
            let elapsed = batch_start.elapsed().as_millis() as u64;

            for hit in &hits {
                if let Some((global, _)) = bucket.get(hit.original_index) {
                    all_scores.push((*global, hit.score));
                    item_latencies[*global] = elapsed;
                }
            }

            batch_latencies.push(elapsed);
            batches.push(BatchInfo {
                index: bi,
                size: bucket.len(),
                elapsed_ms: elapsed,
            });
        }

        let hits = build_hits(all_scores, documents, top_n);
        Ok((hits, batches, item_latencies, batch_latencies))
    }

    /// 一次前向算一批（返回**全部**结果，不截断）。
    ///
    /// ## ⚠️ 为什么不用 `top_n` 截断
    ///
    /// 因为要**跨批**合并后再取 top_n。
    /// 如果每批只取 top_n，合并后的 top_n 可能漏掉
    /// 「在别的批里排第 1 的那个文档」——
    /// 各批的 top_n 合起来**不一定是**全局 top_n。
    ///
    /// ⚠️ 这是「分批求 top-k」的经典陷阱。
    ///
    /// ## 为什么直接 `await` 08 的 `rerank` 而不额外 `spawn_blocking`
    ///
    /// 因为 08 的 `LocalCrossEncoder::rerank` **内部已经**
    /// 把 ONNX 推理放进了 `spawn_blocking`（因为推理是同步阻塞的）。
    /// 再包一层只会多一次线程切换。
    async fn rerank_one(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> AppResult<Vec<RerankedHit>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        debug!(count = documents.len(), "发起批量重排前向");

        let hits = self.inner.rerank(query, documents, documents.len()).await;

        let hits = hits.map_err(|e| AppError::Inference(e.to_string()))?;

        // ⚠️ 结果条数必须与输入一致 ——
        //    少一条就意味着某个候选的分数丢了（静默的错误）。
        if hits.len() != documents.len() {
            warn!(
                input = documents.len(),
                output = hits.len(),
                "重排结果数量不匹配"
            );
            return Err(AppError::Inference(format!(
                "重排结果数量不匹配: 输入 {} 条, 输出 {} 条",
                documents.len(),
                hits.len()
            )));
        }

        Ok(hits)
    }
}

/// 候选文档的文本（用于分桶的长度估算）。
fn texts_of(documents: &[RerankDocument]) -> Vec<String> {
    documents
        .iter()
        .map(|d| {
            if d.heading_path.is_empty() {
                d.text.clone()
            } else {
                format!("# {}\n{}", d.heading_path.join(" > "), d.text)
            }
        })
        .collect()
}

/// 把「(全局下标, 分数)」列表组装成最终的 top_n 结果。
///
/// ## ⚠️ 排序必须是**稳定**且**含 tie-breaker** 的
///
/// 分数相同时要保留**原始顺序**（下标小的在前）。
/// `sort_by` 是稳定排序，但比较器必须**包含下标**作为 tie-breaker ——
/// 否则同分文档的相对顺序在不同批大小下可能不同，
/// 而那正是「批处理改变了结果」的隐蔽形态。
fn build_hits(
    scores: Vec<(usize, f32)>,
    documents: &[RerankDocument],
    top_n: usize,
) -> Vec<RerankedHit> {
    let mut scores = scores;
    scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    scores
        .into_iter()
        .take(top_n)
        .enumerate()
        .filter_map(|(rank, (index, score))| {
            let doc = documents.get(index)?;
            Some(RerankedHit {
                rank,
                id: doc.id.clone(),
                score,
                original_index: index,
                source: doc.source.clone(),
                heading_path: doc.heading_path.clone(),
                text: doc.text.clone(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs(n: usize) -> Vec<RerankDocument> {
        (0..n)
            .map(|i| {
                RerankDocument::new(
                    format!("doc-{i}"),
                    "test.md",
                    vec![format!("标题 {i}")],
                    format!("正文内容 {}", "x".repeat(i * 10 + 1)),
                )
            })
            .collect()
    }

    #[test]
    fn build_hits_sorts_by_score_descending() {
        let documents = docs(3);
        let hits = build_hits(vec![(0, 0.1), (1, 0.9), (2, 0.5)], &documents, 3);

        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, "doc-1");
        assert_eq!(hits[1].id, "doc-2");
        assert_eq!(hits[2].id, "doc-0");
        assert_eq!(hits[0].rank, 0);
    }

    #[test]
    fn build_hits_keeps_original_order_on_ties() {
        // ⚠️ 同分时必须按原始下标排 —— 否则批大小会影响结果顺序。
        let documents = docs(3);
        let hits = build_hits(vec![(2, 0.5), (0, 0.5), (1, 0.5)], &documents, 3);

        assert_eq!(hits[0].id, "doc-0");
        assert_eq!(hits[1].id, "doc-1");
        assert_eq!(hits[2].id, "doc-2");
    }

    #[test]
    fn build_hits_respects_top_n() {
        let documents = docs(5);
        let hits = build_hits((0..5).map(|i| (i, i as f32)).collect(), &documents, 2);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn texts_of_includes_heading() {
        let documents = docs(1);
        let texts = texts_of(&documents);
        assert!(texts[0].starts_with("# "));
    }
}
