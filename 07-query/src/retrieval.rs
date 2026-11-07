//! 检索层 —— 支持「按问题嵌入」和「按文档嵌入」两种模式。
//!
//! ## 这是本项目和 06-rag 最大的实现差异
//!
//! 06-rag 的 [`rag_06::retrieval::Retriever`] 只有一个模式：
//! **把用户问题当 query 嵌入**（`embed_query`）。
//!
//! 但 HyDE 的核心恰恰要求**反过来**：它生成的是一段「假想的答案文档」，
//! 这段东西在语义角色上是**文档**，不是问题 —— 必须用
//! `embed_documents`（`passage:` 前缀）嵌入。
//!
//! 用错了会明显掉效果：
//!
//! | 嵌入方式 | 前缀 | 用在 HyDE 上会怎样 |
//! |---|---|---|
//! | `embed_query` | `query: ` | 向量落在模型「问题区域」，但去匹配「文档区域」 |
//! | `embed_documents` | `passage: ` | 向量和文档在同一分布，检索最准 |
//!
//! E5 这类非对称模型对前缀很敏感 —— 这是 E5 训练时就设计成
//! 「用 `query:` 前缀去检索 `passage:` 前缀的文档」的非对称结构。
//!
//! ## 所以本模块的接口多了一个参数
//!
//! [`EmbedMode`] 明确告诉检索器「这一路查询该用哪种方式嵌入」。

use std::sync::Arc;

use rag_05::embed::Embedder;
use rag_05::store::{FaissStore, SearchHit, VectorStore};
use serde::{Deserialize, Serialize};

use crate::error::{QueryError, QueryResult};
use crate::fusion::RankedRun;
use crate::transform::Strategy;

pub use rag_06::retrieval::RetrievedChunk;

/// 嵌入模式 —— 决定这一路查询用 query 还是 passage 的方式嵌入。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbedMode {
    /// 按**问题**嵌入（`query:` 前缀）。
    ///
    /// 用于：原问题、Multi-Query 的问法、Decompose 的子问题、Rewrite 的检索式。
    Query,

    /// 按**文档**嵌入（`passage:` 前缀）。
    ///
    /// 用于：**HyDE** 生成的假想答案文档。
    Document,
}

impl EmbedMode {
    /// 该策略默认用哪种模式。
    ///
    /// 这个映射是**策略的内在属性**，不是配置项：
    ///
    /// | 策略 | 模式 | 理由 |
    /// |---|---|---|
    /// | None | Query | 原问题本来就是问题 |
    /// | HyDE | **Document** | 生成的是文档，必须按文档嵌入 |
    /// | MultiQuery | Query | 生成的是问法 |
    /// | Rewrite | Query | 生成的是检索式（保留疑问句形态） |
    /// | Decompose | Query | 生成的是子问题 |
    pub fn for_strategy(strategy: Strategy) -> Self {
        match strategy {
            Strategy::Hyde => EmbedMode::Document,
            _ => EmbedMode::Query,
        }
    }
}

/// 多路检索器。
///
/// ## 并发模型
///
/// 多个改写查询的检索是**互不依赖**的，所以应该**并发**执行 ——
///
/// ```text
/// 串行：Q1 → Q2 → Q3     总耗时 = t1 + t2 + t3
/// 并发：Q1 ─┐
///       Q2 ─┼─→ 融合      总耗时 ≈ max(t1, t2, t3) + 排队
///       Q3 ─┘
/// ```
///
/// Multi-Query 有 3 路，省下大约 2/3 的嵌入等待时间。
///
/// ## 锁的处理
///
/// FAISS 的 `search` 需要 `&mut self`（内部搜索缓冲区非线程安全），
/// 所以 store 用 `tokio::sync::Mutex` 串行化。
///
/// ⚠️ 用 `std::sync::Mutex` 会让 future 变成非 `Send`（`MutexGuard` 不是 `Send`），
/// axum 直接拒绝编译 —— 原因和 06-rag 完全一样。
pub struct MultiQueryRetriever {
    store: Arc<tokio::sync::Mutex<FaissStore>>,
    embedder: Arc<dyn Embedder>,
    /// 每路召回多少条。
    ///
    /// ⚠️ 这个值要比最终 `top_k` **大**。
    ///
    /// 因为多路检索 + 融合后，真正有用的块可能来自任何一路。
    /// 如果每路只取 4 条，最终 top-4 里可能有 2 条是重复的，
    /// 实际只有 2 条有效信息。
    ///
    /// 经验值：每路 `top_k * 2`（最低 8，上限 50）。
    pub per_query_k: usize,
}

impl std::fmt::Debug for MultiQueryRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiQueryRetriever")
            .field("per_query_k", &self.per_query_k)
            .finish()
    }
}

impl MultiQueryRetriever {
    /// 创建检索器。
    pub fn new(store: FaissStore, embedder: Box<dyn Embedder>, per_query_k: usize) -> Self {
        Self {
            store: Arc::new(tokio::sync::Mutex::new(store)),
            embedder: Arc::from(embedder),
            per_query_k: per_query_k.max(1),
        }
    }

    /// 设置每路召回数量。
    pub fn with_per_query_k(mut self, per_query_k: usize) -> Self {
        self.per_query_k = per_query_k.max(1);
        self
    }

    /// 索引里的条数。
    pub async fn index_size(&self) -> usize {
        self.store.lock().await.len()
    }

    /// 检索一路：查询文本 → 命中块。
    pub async fn retrieve(
        &self,
        query: &str,
        mode: EmbedMode,
        k: usize,
    ) -> QueryResult<Vec<RetrievedChunk>> {
        let query = query.trim();
        if query.is_empty() {
            return Err(QueryError::BadRequest("查询不能为空".to_string()));
        }

        let vector = self.embed_one(query, mode).await?;

        let mut guard = self.store.lock().await;
        let hits = guard
            .search(&vector, k.max(1))
            .await
            .map_err(|e| QueryError::Retrieval(format!("向量检索失败: {e}")))?;
        drop(guard);

        Ok(hits.iter().enumerate().map(|(rank, hit)| to_chunk(rank, hit)).collect())
    }

    /// 并发检索多路，返回每路的结果。
    ///
    /// ## 为什么嵌入和 FAISS 搜索要分成两步
    ///
    /// - **嵌入**是纯 CPU 计算，可以真正并行（`spawn_blocking` 到阻塞线程池）
    /// - **FAISS 搜索**受 mutex 限制，本来就是串行的
    ///
    /// 如果把两者塞进同一个 future，tokio 只会并发调度，
    /// 实际还是要排队等锁 —— 没意义。
    ///
    /// 所以：**先把所有查询的向量算出来，再依次做向量检索**。
    pub async fn retrieve_multi(
        &self,
        queries: &[(String, EmbedMode, Strategy)],
        k: usize,
    ) -> QueryResult<Vec<RankedRun>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        // 1. 并发嵌入。
        let mut tasks = Vec::with_capacity(queries.len());
        for (text, mode, _) in queries {
            let embedder = Arc::clone(&self.embedder);
            let text = text.clone();
            let mode = *mode;

            tasks.push(tokio::task::spawn_blocking(move || {
                embed_text(embedder.as_ref(), &text, mode)
            }));
        }

        let mut vectors = Vec::with_capacity(tasks.len());
        for task in tasks {
            let result = task
                .await
                .map_err(|e| QueryError::Internal(format!("嵌入任务失败: {e}")))?;
            vectors.push(result?);
        }

        // 2. 依次做向量检索。
        let mut runs = Vec::with_capacity(queries.len());
        for ((text, _mode, strategy), vector) in queries.iter().zip(vectors) {
            let mut guard = self.store.lock().await;
            let hits = guard
                .search(&vector, k.max(1))
                .await
                .map_err(|e| QueryError::Retrieval(format!("向量检索失败: {e}")))?;
            drop(guard);

            let chunks: Vec<RetrievedChunk> = hits
                .iter()
                .enumerate()
                .map(|(rank, hit)| to_chunk(rank, hit))
                .collect();

            runs.push(RankedRun::new(text.clone(), *strategy, chunks));
        }

        Ok(runs)
    }

    /// 单路嵌入（包在 `spawn_blocking` 里）。
    ///
    /// ONNX 推理是**同步的 CPU 密集**调用，直接在 async 上下文里调用会
    /// **阻塞 tokio 的工作线程** —— 一次几十到几百毫秒，期间同线程上的
    /// 其它请求全被卡住。`spawn_blocking` 把它们丢到专门的阻塞线程池。
    async fn embed_one(&self, text: &str, mode: EmbedMode) -> QueryResult<Vec<f32>> {
        let embedder = Arc::clone(&self.embedder);
        let text = text.to_string();

        tokio::task::spawn_blocking(move || embed_text(embedder.as_ref(), &text, mode))
            .await
            .map_err(|e| QueryError::Internal(format!("嵌入任务失败: {e}")))?
    }
}

/// 按指定模式嵌入一段文本。
fn embed_text(
    embedder: &dyn Embedder,
    text: &str,
    mode: EmbedMode,
) -> QueryResult<Vec<f32>> {
    match mode {
        EmbedMode::Query => embedder
            .embed_query(text)
            .map_err(|e| QueryError::Retrieval(format!("查询向量化失败: {e}"))),
        EmbedMode::Document => {
            // 文档方式：走 `embed_documents`，让模型加上 `passage: ` 前缀。
            //
            // ⚠️ 不能直接调 `embed_query` 然后手动加前缀 ——
            // `needs_document_prefix()` 是模型自己的规则，绕过它容易出错。
            let texts = vec![text.to_string()];
            let mut vecs = embedder
                .embed_documents(&texts)
                .map_err(|e| QueryError::Retrieval(format!("文档向量化失败: {e}")))?;

            vecs.pop()
                .ok_or_else(|| QueryError::Retrieval("嵌入返回空结果".to_string()))
        }
    }
}

/// 把 `SearchHit` 转成 `RetrievedChunk`。
fn to_chunk(rank: usize, hit: &SearchHit) -> RetrievedChunk {
    let heading_context = if hit.heading_path.is_empty() {
        String::new()
    } else {
        format!("# {}", hit.heading_path.join(" > "))
    };

    RetrievedChunk {
        rank,
        id: hit.id.clone(),
        score: hit.score,
        source: hit.source.clone(),
        heading_path: hit.heading_path.clone(),
        text: hit.text.clone(),
        heading_context,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用的假嵌入器 —— 返回固定向量，不依赖模型下载。
    struct NoopEmbedder;

    impl Embedder for NoopEmbedder {
        fn model_name(&self) -> &str {
            "noop"
        }

        fn dimension(&self) -> usize {
            4
        }

        fn embed_documents(
            &self,
            texts: &[String],
        ) -> Result<Vec<Vec<f32>>, rag_05::embed::EmbedError> {
            Ok(texts
                .iter()
                .map(|t| vec![t.chars().count() as f32, 1.0, 0.0, 0.0])
                .collect())
        }
    }

    #[test]
    fn hyde_uses_document_mode() {
        assert_eq!(
            EmbedMode::for_strategy(Strategy::Hyde),
            EmbedMode::Document
        );
    }

    #[test]
    fn question_like_strategies_use_query_mode() {
        for s in [
            Strategy::None,
            Strategy::MultiQuery,
            Strategy::Rewrite,
            Strategy::Decompose,
        ] {
            assert_eq!(EmbedMode::for_strategy(s), EmbedMode::Query);
        }
    }

    #[tokio::test]
    async fn rejects_empty_query() {
        // 空查询在嵌入前就被拦下 —— 不需要真的 embedder。
        let retriever = MultiQueryRetriever::new(
            FaissStore::new(rag_05::store::IndexKind::Flat, 384, "test").unwrap(),
            Box::new(NoopEmbedder),
            8,
        );

        let err = retriever
            .retrieve("   ", EmbedMode::Query, 4)
            .await
            .unwrap_err();
        assert_eq!(err.status_code(), 400);
    }
}
