//! 检索层（Retrieval）—— 找出与问题最相关的块。
//!
//! ```text
//! 问题 ──Embedder──> 查询向量 ──FaissStore.search──> top-k 命中块
//! ```
//!
//! ## Naive RAG 的检索有多朴素
//!
//! 朴素版检索**只有一步**：把问题变成向量，去索引里找最相似的 k 个。
//!
//! 刻意没做的事（每一项都是后面要加的增强手段）：
//!
//! | 没做的事 | 会导致什么 | 增强手段 |
//! |---|---|---|
//! | 查询改写（把口语问题改成检索式） | 问「它多少钱」检索不到「价格」 | HyDE / 查询分解 |
//! | 混合检索（向量 + BM25 关键词） | 问具体编号、API 名时召不回来 | BM25 + RRF |
//! | 重排（cross-encoder 精排） | top-10 里有 3 个真正相关的排在第 8 | Cross-encoder rerank |
//! | 元数据过滤 | 混进了别的文档的块 | 按 source/时间过滤 |
//! | 相似度阈值 | 什么都不相关时也硬答 | `min_score` |
//!
//! ## 为什么「先召回再精排」是两步
//!
//! 向量检索（双编码器 bi-encoder）把问题和文档**分别**编码成向量，
//! 文档向量可以离线算好、存起来 —— 这就是它快的原因，也是它只能做「粗筛」的原因：
//! 问题和文档之间没有交互，只能比距离。
//!
//! 重排模型（cross-encoder）把「问题 + 文档」**拼在一起**送进模型，
//! 能看到交互 —— 所以更准，但每个候选都要跑一次模型，慢得多。
//!
//! 所以标准做法是：**向量召回 20~100 条 → 重排取 top 3~5**。

use rag_05::embed::Embedder;
use rag_05::store::{FaissStore, SearchHit, VectorStore};
use serde::{Deserialize, Serialize};

use crate::config::RetrievalConfig;
use crate::error::{RagError, RagResult};

/// 一条检索命中（带排名）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedChunk {
    /// 排名（0 = 最相关）。
    pub rank: usize,

    /// 命中块的 ID（`{文件名}#{序号}`）。
    pub id: String,

    /// 相似度分数（余弦相似度，越大越相关）。
    pub score: f32,

    /// 来源文件名。
    pub source: String,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 块正文。
    pub text: String,

    /// 标题路径渲染成一行（`# 文档 > ## 章节`），空则为空串。
    pub heading_context: String,
}

impl RetrievedChunk {
    /// 从 `SearchHit` + 排名构造。
    fn from_hit(rank: usize, hit: &SearchHit) -> Self {
        let heading_context = if hit.heading_path.is_empty() {
            String::new()
        } else {
            format!("# {}", hit.heading_path.join(" > "))
        };

        Self {
            rank,
            id: hit.id.clone(),
            score: hit.score,
            source: hit.source.clone(),
            heading_path: hit.heading_path.clone(),
            text: hit.text.clone(),
            heading_context,
        }
    }

    /// 引用标记，如 `[1]`。用于在答案里标注来源。
    pub fn citation(&self) -> String {
        format!("[{}]", self.rank + 1)
    }
}

/// 检索结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// 原始问题。
    pub query: String,

    /// 命中的块（按相关度降序）。
    pub chunks: Vec<RetrievedChunk>,

    /// 实际召回数量（可能被阈值过滤掉了几个）。
    pub retrieved_count: usize,

    /// 最高分（用于「检索质量」判断）。
    pub top_score: Option<f32>,
}

impl RetrievalResult {
    /// 没有任何命中。
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// 拼成上下文的纯文本。
    ///
    /// 每个块前面加 `[n]` 引用标记 —— 这样模型可以在答案里写 `[1]`，
    /// 用户能顺着标记回到原文。这是 RAG 相对纯生成的核心价值：
    /// **答案可溯源**。
    pub fn to_context(&self) -> String {
        self.chunks
            .iter()
            .map(|c| {
                let head = if c.heading_context.is_empty() {
                    String::new()
                } else {
                    format!("{}\n", c.heading_context)
                };
                format!("[{}] {}{}", c.rank + 1, head, c.text)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// 检索器。
///
/// 持有一个 FAISS store 和一个嵌入器，按配置做 top-k 检索。
///
/// ## 为什么 store 用 `Arc<Mutex<>>`
///
/// FAISS 的 `IndexImpl` 是 `Send + Sync` 的，但 `search` 需要 `&mut self`
/// —— 因为 FAISS 内部的搜索缓冲区不是线程安全的（多个线程同时搜索会写坏缓冲区）。
///
/// 所以共享一个 store 需要串行化。
///
/// ### 为什么用 `tokio::sync::Mutex` 而不是 `std::sync::Mutex`
///
/// `std::sync::MutexGuard` **不是 `Send`**。而检索是 `async fn`，
/// 持锁期间要 `.await`（等待 store.search 完成）——
/// 这会让整个 handler 的 future 变成非 `Send`，axum 直接拒绝编译：
///
/// ```text
/// error: future cannot be sent between threads safely
///   future returned by `ask` is not `Send`
///   the trait `Send` is not implemented for `std::sync::MutexGuard<'_, FaissStore>`
/// ```
///
/// `tokio::sync::MutexGuard` 在 `T: Send` 时是 `Send`，所以能跨 await 持有。
/// 代价是：锁等待是异步的（会 yield 而不是阻塞线程）—— 这正是我们想要的。
///
/// ⚠️ 不要用 `std::sync::Mutex` + `block_in_place` 绕过 —— 那在
/// `current_thread` runtime 里会 panic。
///
/// ## 并发量的真实上限
///
/// 所有检索共享一把锁，所以**并发检索实际上是被串行化的**。
/// 对教学项目够用（检索是毫秒级的纯内存操作）。
/// 真正的并发方案是每个请求一个 store 实例，或用 FAISS 的
/// `SearchParameters` 做无锁搜索。
pub struct Retriever {
    store: std::sync::Arc<tokio::sync::Mutex<FaissStore>>,
    embedder: Box<dyn Embedder>,
    config: RetrievalConfig,
}

impl std::fmt::Debug for Retriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Retriever")
            .field("config", &self.config)
            .finish()
    }
}

impl Retriever {
    /// 创建检索器。
    pub fn new(store: FaissStore, embedder: Box<dyn Embedder>, config: RetrievalConfig) -> Self {
        Self {
            store: std::sync::Arc::new(tokio::sync::Mutex::new(store)),
            embedder,
            config,
        }
    }

    /// 当前配置。
    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    /// 索引里的条数。
    pub async fn index_size(&self) -> usize {
        self.store.lock().await.len()
    }

    /// 检索：问题 → top-k 命中块。
    pub async fn retrieve(&self, query: &str) -> RagResult<RetrievalResult> {
        self.retrieve_with_top_k(query, self.config.top_k).await
    }

    /// 检索：问题 → top-k 命中块（显式指定 k）。
    ///
    /// Agent 场景下用这个 —— 模型可以在工具调用里自己指定要几条。
    pub async fn retrieve_with_top_k(&self, query: &str, top_k: usize) -> RagResult<RetrievalResult> {
        let query = query.trim();
        if query.is_empty() {
            return Err(RagError::BadRequest("问题不能为空".to_string()));
        }

        // 1. 问题 → 向量。用 `embed_query` 而不是 `embed_documents` ——
        //    E5 这类非对称模型要求查询加 `query: ` 前缀，文档加 `passage: `。
        //    用错前缀会明显掉召回率。
        let query_vector = self
            .embedder
            .embed_query(query)
            .map_err(|e| RagError::Retrieval(format!("问题向量化失败: {e}")))?;

        // 2. 向量 → top-k。多召回一些，再按阈值过滤。
        //
        //    ⚠️ 这里 `k` 用 `top_k` 而不是更大的值，因为朴素版没有重排 ——
        //    召回顺序就是最终顺序。多召回只会引入更多噪声。
        let mut guard = self
            .store
            .lock()
            .await;

        let hits = guard
            .search(&query_vector, top_k.max(1))
            .await
            .map_err(|e| RagError::Retrieval(format!("向量检索失败: {e}")))?;

        drop(guard);

        // 3. 阈值过滤（默认关闭 —— 这是 Naive 的边界之一）。
        let filtered: Vec<SearchHit> = match self.config.min_score {
            Some(min) => hits
                .into_iter()
                .filter(|h| h.score >= min)
                .collect(),
            None => hits,
        };

        let retrieved_count = filtered.len();
        let chunks: Vec<RetrievedChunk> = filtered
            .iter()
            .enumerate()
            .map(|(rank, hit)| RetrievedChunk::from_hit(rank, hit))
            .collect();

        let top_score = chunks.first().map(|c| c.score);

        Ok(RetrievalResult {
            query: query.to_string(),
            chunks,
            retrieved_count,
            top_score,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(rank: usize, score: f32) -> RetrievedChunk {
        RetrievedChunk {
            rank,
            id: format!("doc.md#{rank}"),
            score,
            source: "doc.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: "正文".to_string(),
            heading_context: "# 标题".to_string(),
        }
    }

    #[test]
    fn citation_is_one_based() {
        assert_eq!(chunk(0, 0.9).citation(), "[1]");
        assert_eq!(chunk(1, 0.8).citation(), "[2]");
    }

    #[test]
    fn context_marks_each_chunk_with_its_number() {
        let result = RetrievalResult {
            query: "q".to_string(),
            chunks: vec![chunk(0, 0.9), chunk(1, 0.8)],
            retrieved_count: 2,
            top_score: Some(0.9),
        };
        let ctx = result.to_context();
        assert!(ctx.contains("[1]"));
        assert!(ctx.contains("[2]"));
        assert!(ctx.contains("# 标题"));
    }

    #[test]
    fn empty_result_has_empty_context() {
        let result = RetrievalResult {
            query: "q".to_string(),
            chunks: vec![],
            retrieved_count: 0,
            top_score: None,
        };
        assert!(result.is_empty());
        assert_eq!(result.to_context(), "");
    }
}
