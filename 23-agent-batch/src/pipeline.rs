//! 端到端流水线 —— **批量索引** + **批量检索**。
//!
//! # 批处理在 RAG 链路上的两个位置
//!
//! ```text
//! ┌─── 离线：批量索引（吞吐场景）───────────────────────────┐
//! │  文档语料                                            │
//! │     ↓ 批量嵌入（一次前向 32 条）                      │
//! │  向量库                                              │
//! └──────────────────────────────────────────────────────┘
//!
//! ┌─── 在线：批量检索（延迟场景）───────────────────────────┐
//! │  查询 → 向量检索 → 候选集 top-20                     │
//! │     ↓ 批量重排（一次前向 20 个候选）                  │
//! │  top-3                                               │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! ## ⚠️ 两个位置的优化目标**相反**
//!
//! | 位置 | 场景 | 优化目标 | 批处理的影响 |
//! |---|---|---|---|
//! | **批量索引** | 离线灌库 | **吞吐** | 吞吐大幅提升，延迟无所谓 |
//! | **批量检索** | 在线查询 | **延迟** | 吞吐提升但**单条延迟被抬高** |
//!
//! 所以「批量检索」只在**一次要精排很多候选**时才划算 ——
//! 比如「一次查询召回 100 条再精排」。
//! 典型的 top-20 → top-3 场景，批处理的收益有限。
//!
//! ## 为什么向量检索用**暴力**而不是 FAISS
//!
//! 因为本项目要测的是**批处理**的收益，不是检索算法。
//! 用暴力检索（余弦相似度全量扫描）能把变量锁死在「批处理」这一个维度 ——
//! 引入 FAISS 会带来索引构建、参数调优等额外变量。
//!
//! （FAISS 的用法见 05-rag。）

use std::collections::HashMap;

use agent_logging::{debug, info};
use rag_05::embed::cosine_similarity;
use rag_08::reranker::{RerankDocument, RerankedHit};
use serde::{Deserialize, Serialize};

use crate::batching::BatchStrategy;
use crate::embedder::BatchEmbedder;
use crate::error::{AppError, AppResult};
use crate::metrics::BatchStats;
use crate::reranker::BatchReranker;

/// 批量索引的结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexReport {
    /// 索引里的文档数。
    pub doc_count: usize,

    /// 向量维度。
    pub dimension: usize,

    /// 嵌入的度量。
    pub embed_stats: BatchStats,

    /// 使用的批策略。
    pub strategy: BatchStrategy,

    /// 建索引总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 批量检索的结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchReport {
    /// 重排后的命中。
    pub hits: Vec<RerankedHit>,

    /// 召回的候选数。
    pub candidate_count: usize,

    /// 检索（召回）耗时（毫秒）。
    pub recall_ms: u64,

    /// 重排耗时（毫秒）。
    pub rerank_ms: u64,

    /// 重排的度量。
    pub rerank_stats: BatchStats,

    /// 使用的批策略。
    pub strategy: BatchStrategy,
}

/// 一个内存向量库（暴力检索）。
#[derive(Debug, Clone)]
pub struct InMemoryVectorStore {
    /// 文档 ID → 文档全文。
    texts: HashMap<String, String>,

    /// 文档 ID → 向量。
    vectors: HashMap<String, Vec<f32>>,

    /// 文档 ID → 元信息（标题路径等）。
    headings: HashMap<String, Vec<String>>,
}

impl InMemoryVectorStore {
    /// 创建空库。
    pub fn new() -> Self {
        Self {
            texts: HashMap::new(),
            vectors: HashMap::new(),
            headings: HashMap::new(),
        }
    }

    /// 文档数。
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// 插入一条。
    pub fn insert(
        &mut self,
        id: impl Into<String>,
        text: impl Into<String>,
        heading_path: Vec<String>,
        vector: Vec<f32>,
    ) {
        let id = id.into();
        self.texts.insert(id.clone(), text.into());
        self.headings.insert(id.clone(), heading_path);
        self.vectors.insert(id, vector);
    }

    /// 取一条的文本。
    pub fn text(&self, id: &str) -> Option<&String> {
        self.texts.get(id)
    }

    /// 取一条的标题路径。
    pub fn heading_path(&self, id: &str) -> Option<&Vec<String>> {
        self.headings.get(id)
    }

    /// 暴力检索：返回 (id, 分数) 按分数降序。
    ///
    /// ⚠️ 这是**精确**检索 —— 没有近似。
    /// 这样测出来的差值能**归因**到批处理，而不是检索算法。
    pub fn search(&self, query_vector: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let mut hits: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(id, v)| (id.clone(), cosine_similarity(query_vector, v)))
            .collect();

        // ⚠️ 排序必须**稳定且含 tie-breaker** ——
        //    否则同分文档的相对顺序不确定，
        //    会让「批处理是否改变了结果」这个判断变得不可靠。
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        hits.truncate(top_k);
        hits
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// ★ 批量索引：把一批文档一次性灌进向量库。
///
/// ## ⚠️ 结果顺序必须与输入顺序一致
///
/// 索引写入的顺序会影响「同分文档的相对顺序」，
/// 所以必须保证批处理**没有改变**写入顺序。
pub async fn build_index(
    embedder: &BatchEmbedder,
    documents: &[(String, String, Vec<String>)],
    strategy: BatchStrategy,
) -> AppResult<(InMemoryVectorStore, IndexReport)> {
    if documents.is_empty() {
        return Err(AppError::BadRequest("文档列表不能为空".to_string()));
    }

    let start = std::time::Instant::now();

    let texts: Vec<String> = documents.iter().map(|(_, text, _)| text.clone()).collect();
    let (vectors, embed_stats) = embedder.embed_batch(&texts, strategy).await?;

    let mut store = InMemoryVectorStore::new();
    for ((id, text, heading), vector) in documents.iter().zip(vectors) {
        store.insert(id.clone(), text.clone(), heading.clone(), vector);
    }

    let elapsed = start.elapsed().as_millis() as u64;
    let doc_count = store.len();

    info!(
        doc_count,
        elapsed_ms = elapsed,
        strategy = strategy.label(),
        "批量索引完成"
    );

    Ok((
        store,
        IndexReport {
            doc_count,
            dimension: embedder.dimension(),
            embed_stats,
            strategy,
            elapsed_ms: elapsed,
        },
    ))
}

/// ★ 批量检索：召回 + 批量重排。
///
/// ## ⚠️ 两个阶段的度量要**分开**记
///
/// 「召回」是向量检索（毫秒级），
/// 「重排」是模型前向（秒级）。
/// 混在一起就看不出批处理到底优化了哪一段。
pub async fn search(
    embedder: &BatchEmbedder,
    reranker: &BatchReranker,
    store: &InMemoryVectorStore,
    query: &str,
    top_k: usize,
    top_n: usize,
    strategy: BatchStrategy,
) -> AppResult<SearchReport> {
    if store.is_empty() {
        return Err(AppError::BadRequest("向量库为空，无法检索".to_string()));
    }
    if query.trim().is_empty() {
        return Err(AppError::BadRequest("查询不能为空".to_string()));
    }

    // ── 阶段 1：召回 ────────────────────────────────────────────
    let recall_start = std::time::Instant::now();

    let (query_vectors, _) = embedder.embed_batch(&[query.to_string()], strategy).await?;
    let query_vector = query_vectors
        .into_iter()
        .next()
        .ok_or_else(|| AppError::Internal("查询向量为空".to_string()))?;

    let candidates = store.search(&query_vector, top_k);
    let recall_ms = recall_start.elapsed().as_millis() as u64;

    // ── 阶段 2：批量重排 ────────────────────────────────────────
    let rerank_documents: Vec<RerankDocument> = candidates
        .iter()
        .map(|(id, _)| {
            RerankDocument::new(
                id.clone(),
                "index",
                store.heading_path(id).cloned().unwrap_or_default(),
                store.text(id).cloned().unwrap_or_default(),
            )
        })
        .collect();

    debug!(candidates = rerank_documents.len(), "开始批量重排");

    let (hits, rerank_stats) = reranker
        .rerank_batch(query, &rerank_documents, top_n, strategy)
        .await?;

    Ok(SearchReport {
        hits,
        candidate_count: candidates.len(),
        recall_ms,
        rerank_ms: rerank_stats.elapsed_ms,
        rerank_stats,
        strategy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(n: usize) -> InMemoryVectorStore {
        let mut store = InMemoryVectorStore::new();
        for i in 0..n {
            store.insert(
                format!("doc-{i}"),
                format!("文本 {i}"),
                vec!["标题".to_string()],
                vec![i as f32, 1.0, 0.0],
            );
        }
        store
    }

    #[test]
    fn search_returns_top_k_sorted_by_score() {
        let store = store_with(5);
        let hits = store.search(&[4.0, 1.0, 0.0], 3);

        assert_eq!(hits.len(), 3);
        // 与 [4,1,0] 最像的是 doc-4（点积最大）。
        assert_eq!(hits[0].0, "doc-4");
        assert_eq!(hits[1].0, "doc-3");
    }

    #[test]
    fn search_respects_top_k() {
        let store = store_with(10);
        assert_eq!(store.search(&[1.0, 1.0, 0.0], 4).len(), 4);
    }

    #[test]
    fn search_on_empty_store_returns_nothing() {
        let store = InMemoryVectorStore::new();
        assert!(store.search(&[1.0], 5).is_empty());
    }

    #[test]
    fn search_ties_are_broken_by_id() {
        // ⚠️ 同分时必须按 ID 排 —— 否则顺序不确定。
        let mut store = InMemoryVectorStore::new();
        for id in ["doc-c", "doc-a", "doc-b"] {
            store.insert(id, "文本", vec![], vec![1.0, 0.0, 0.0]);
        }

        let hits = store.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(
            hits.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["doc-a", "doc-b", "doc-c"]
        );
    }

    #[tokio::test]
    async fn search_on_empty_store_is_rejected() {
        let store = InMemoryVectorStore::new();
        let err = crate::pipeline::search(
            &crate::embedder::BatchEmbedder::from_embedder(
                std::sync::Arc::new(crate::testkit::ScriptedEmbedder::new()),
                Default::default(),
            ),
            &crate::reranker::BatchReranker::new(
                rag_08::reranker::RerankerModel::BgeRerankerBase,
                Some(crate::testkit::cache_dir()),
                Default::default(),
            ),
            &store,
            "查询",
            10,
            3,
            BatchStrategy::Fixed,
        )
        .await
        .unwrap_err();

        assert_eq!(err.status_code(), 400);
    }
}
