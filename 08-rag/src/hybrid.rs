//! 混合检索（Hybrid Search）—— BM25 关键词 + 向量语义，两路并行。
//!
//! ## 为什么混合
//!
//! 两路的**失败模式正好互补**：
//!
//! | 场景 | 向量检索 | BM25 |
//! |---|---|---|
//! | 「怎么优化检索速度」 | ✅ 语义匹配 | ⚠️ 关键词对不上 |
//! | 「`ECONNREFUSED` 是什么错误」 | ❌ 罕见 token 被语义稀释 | ✅ 精确命中 |
//! | 「`bge-reranker-v2-m3`」 | ❌ 型号名几乎没有语义 | ✅ 精确命中 |
//!
//! 单独用任何一路，都会在另一路的强项上翻车。
//! 混合检索把两路都跑一遍，再合并 —— 这就是工业界的默认做法。
//!
//! ## 怎么合并：RRF
//!
//! **不能直接加权求和**，因为两路的分数根本不可比：
//!
//! | | 分数来源 | 典型范围 |
//! |---|---|---|
//! | 向量检索 | 余弦相似度 | 0.55 ~ 0.85 |
//! | BM25 | 对数加权和 | 0 ~ 15+ |
//!
//! 一个 0.85 的余弦相似度对上 15 分的 BM25 —— 加权系数怎么调都不对。
//!
//! 所以用 **RRF（Reciprocal Rank Fusion，倒数排名融合）**：
//!
//! ```text
//! score(d) = Σ  1 / (k + rank_i(d))
//!              i∈lists
//! ```
//!
//! **只看排名，不看分数** —— 于是跨路径天然可比。
//!
//! `k` 复用 `rag_07::fusion::RRF_K`（= 60），和 07-query 保持一致。
//!
//! ## 检索模式
//!
//! | 模式 | 说明 | 用途 |
//! |---|---|---|
//! | [`SearchMode::Vector`] | 只走向量 | **对照组** |
//! | [`SearchMode::Bm25`] | 只走 BM25 | **对照组** |
//! | [`SearchMode::Hybrid`] | 两路都走 + RRF | **默认** |
//!
//! 有了两个对照组，才能量化「混合到底值多少」——
//! 这和 07-query 里 `Strategy::None` 的作用完全一样。

use std::collections::HashMap;
use std::sync::Arc;

use rag_05::embed::Embedder;
use rag_05::store::{FaissStore, SearchHit, VectorStore};
use serde::{Deserialize, Serialize};

use crate::bm25::{Bm25Hit, Bm25Index};
use crate::error::{RagError, RagResult};

/// RRF 的常数 `k` —— 复用 07-query 的值，保证两个项目的融合行为一致。
pub use rag_07::fusion::RRF_K;

/// 检索模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// 只走向量检索（**对照组**）。
    Vector,

    /// 只走 BM25 关键词检索（**对照组**）。
    Bm25,

    /// 两路并行 + RRF 融合（**默认**）。
    Hybrid,
}

impl SearchMode {
    /// 模式名。
    pub fn name(self) -> &'static str {
        match self {
            SearchMode::Vector => "vector",
            SearchMode::Bm25 => "bm25",
            SearchMode::Hybrid => "hybrid",
        }
    }

    /// 从字符串解析。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "vector" | "dense" | "semantic" => Some(SearchMode::Vector),
            "bm25" | "keyword" | "sparse" => Some(SearchMode::Bm25),
            "hybrid" | "mixed" => Some(SearchMode::Hybrid),
            _ => None,
        }
    }

    /// 所有模式。
    pub fn all() -> Vec<Self> {
        vec![SearchMode::Vector, SearchMode::Bm25, SearchMode::Hybrid]
    }
}

/// 一条混合检索的命中。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridHit {
    /// 排名（0 = 最相关）。
    pub rank: usize,

    /// 块 ID。
    pub id: String,

    /// 融合分数（RRF）。
    pub score: f32,

    /// 向量检索的相似度（没被向量路召回时是 `None`）。
    pub vector_score: Option<f32>,

    /// BM25 分数（没被 BM25 路召回时是 `None`）。
    pub bm25_score: Option<f32>,

    /// 两路都命中了它 —— **共识度**。
    ///
    /// 这是判断混合结果是否可信的关键指标：
    /// 共识度 2/2 的块几乎一定对；1/2 的块可能只是某一路的噪声。
    pub votes: usize,

    /// 来源文件名。
    pub source: String,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 块正文。
    pub text: String,
}

impl HybridHit {
    /// 引用标记（1 开始）。
    pub fn citation(&self) -> String {
        format!("[{}]", self.rank + 1)
    }

    /// 渲染成上下文片段（格式和 06/07 保持一致）。
    pub fn to_context(&self) -> String {
        let head = if self.heading_path.is_empty() {
            String::new()
        } else {
            format!("# {}\n", self.heading_path.join(" > "))
        };
        format!("[{}] {}{}", self.rank + 1, head, self.text)
    }
}

/// 混合检索器。
///
/// 持有三个东西：
///
/// - `store`：FAISS 向量索引
/// - `bm25`：BM25 倒排索引
/// - `embedder`：把查询变成向量
///
/// ## 锁的处理
///
/// FAISS 的 `search` 需要 `&mut self`（内部搜索缓冲区非线程安全），
/// 所以用 `tokio::sync::Mutex` 串行化。
///
/// ⚠️ 不能用 `std::sync::Mutex` —— 它的 `MutexGuard` 不是 `Send`，
/// 跨 `.await` 持有会让 handler 的 future 变成非 `Send`，axum 直接拒绝编译。
/// （和 06/07 的原因完全一样。）
pub struct HybridRetriever {
    store: Arc<tokio::sync::Mutex<FaissStore>>,
    bm25: Arc<Bm25Index>,
    embedder: Arc<dyn Embedder>,
    /// 默认的每路召回数量。
    ///
    /// ⚠️ 这个值要比最终 `top_k` **大**。
    ///
    /// 因为两路的结果会重叠 —— 如果每路只取 `top_k` 条，
    /// 融合后可能挤满重复内容，实际有效信息不足一半。
    ///
    /// 经验值：每路 `top_k * 2`（最低 8，上限 50）。
    ///
    /// 它是**默认值** —— [`HybridRetriever::search`] 允许按次覆盖，
    /// 因为重排场景需要「每路多召回一些」来喂饱重排器。
    pub per_path_k: usize,
}

impl std::fmt::Debug for HybridRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridRetriever")
            .field("per_path_k", &self.per_path_k)
            .field("bm25_docs", &self.bm25.len())
            .finish()
    }
}

impl HybridRetriever {
    /// 创建混合检索器。
    pub fn new(
        store: FaissStore,
        bm25: Bm25Index,
        embedder: Box<dyn Embedder>,
        per_path_k: usize,
    ) -> Self {
        Self {
            store: Arc::new(tokio::sync::Mutex::new(store)),
            bm25: Arc::new(bm25),
            embedder: Arc::from(embedder),
            per_path_k: per_path_k.max(1),
        }
    }

    /// 设置每路召回数量。
    pub fn with_per_path_k(mut self, per_path_k: usize) -> Self {
        self.per_path_k = per_path_k.max(1);
        self
    }

    /// 向量索引里的条数。
    pub async fn index_size(&self) -> usize {
        self.store.lock().await.len()
    }

    /// BM25 索引里的文档数。
    pub fn bm25_size(&self) -> usize {
        self.bm25.len()
    }

    /// 检索：查询 → 命中块。
    ///
    /// `per_path_k` 为 `None` 时用 [`HybridRetriever::per_path_k`] 这个默认值。
    ///
    /// ## 为什么允许按次覆盖
    ///
    /// 重排场景需要**更多候选** —— reranker 只能重排它拿到的东西。
    /// 如果每路只召回 8 条，融合后可能只剩 10 条左右的候选，
    /// 重排器的工作范围就被卡死了。
    ///
    /// 所以重排流水线会传 `max(per_path_k, rerank_candidate_k)`。
    pub async fn search(
        &self,
        query: &str,
        mode: SearchMode,
        top_k: usize,
    ) -> RagResult<Vec<HybridHit>> {
        self.search_with_k(query, mode, top_k, None).await
    }

    /// 检索（显式指定每路召回数量）。
    pub async fn search_with_k(
        &self,
        query: &str,
        mode: SearchMode,
        top_k: usize,
        per_path_k: Option<usize>,
    ) -> RagResult<Vec<HybridHit>> {
        let query = query.trim();
        if query.is_empty() {
            return Err(RagError::BadRequest("查询不能为空".to_string()));
        }

        let per_path_k = per_path_k.unwrap_or(self.per_path_k).max(1);

        match mode {
            SearchMode::Vector => {
                let hits = self.search_vector(query, per_path_k).await?;
                Ok(finish(hits, Vec::new(), top_k))
            }
            SearchMode::Bm25 => {
                let hits = self.bm25.search(query, per_path_k);
                Ok(finish(Vec::new(), hits, top_k))
            }
            SearchMode::Hybrid => {
                // 两路**并发**跑：向量路要走嵌入（几十到几百毫秒），
                // BM25 路是纯内存操作（微秒级）。
                //
                // 先把 BM25 算完（同步、便宜），再 await 向量路 ——
                // 这样 BM25 的耗时几乎被完全掩盖在嵌入的等待里。
                let bm25_hits = self.bm25.search(query, per_path_k);

                let vector_hits = self.search_vector(query, per_path_k).await?;

                Ok(finish(vector_hits, bm25_hits, top_k))
            }
        }
    }

    /// 向量检索一路。
    async fn search_vector(&self, query: &str, k: usize) -> RagResult<Vec<SearchHit>> {
        // 嵌入是 CPU 密集的同步调用 —— 必须放到阻塞线程池，
        // 否则会卡住 tokio 的工作线程。
        let vector = {
            let embedder = Arc::clone(&self.embedder);
            let query = query.to_string();
            tokio::task::spawn_blocking(move || embedder.embed_query(&query))
                .await
                .map_err(|e| RagError::Internal(format!("嵌入任务失败: {e}")))?
                .map_err(|e| RagError::Retrieval(format!("查询向量化失败: {e}")))?
        };

        let mut guard = self.store.lock().await;
        let hits = guard
            .search(&vector, k.max(1))
            .await
            .map_err(|e| RagError::Retrieval(format!("向量检索失败: {e}")))?;
        drop(guard);

        Ok(hits)
    }
}

/// 融合两路结果并截断到 `top_k`。
///
/// ## RRF 的实现
///
/// ```text
/// score(d) = Σ  1 / (k + rank_i(d))
/// ```
///
/// 按 ID 去重（同一个块在两路都命中时只占一个位置），
/// 同时记录 `vector_score` / `bm25_score` / `votes`。
///
/// ## 排序的次级键
///
/// RRF 分数相同时，按 `max(vector_score, bm25_score)` 降序，
/// 再按 ID 升序 —— 保证结果**可复现**，不依赖 HashMap 的迭代顺序。
fn finish(
    vector_hits: Vec<SearchHit>,
    bm25_hits: Vec<Bm25Hit>,
    top_k: usize,
) -> Vec<HybridHit> {
    // key = 块 ID，value = 累加器。
    let mut acc: HashMap<String, HybridHit> = HashMap::new();

    for (idx, hit) in vector_hits.iter().enumerate() {
        let rank = idx + 1;
        let contribution = 1.0 / (RRF_K + rank as f32);

        let entry = acc.entry(hit.id.clone()).or_insert_with(|| HybridHit {
            rank: 0,
            id: hit.id.clone(),
            score: 0.0,
            vector_score: None,
            bm25_score: None,
            votes: 0,
            source: hit.source.clone(),
            heading_path: hit.heading_path.clone(),
            text: hit.text.clone(),
        });

        entry.score += contribution;
        entry.votes += 1;
        entry.vector_score = Some(hit.score);
    }

    for (idx, hit) in bm25_hits.iter().enumerate() {
        let rank = idx + 1;
        let contribution = 1.0 / (RRF_K + rank as f32);

        let entry = acc.entry(hit.id.clone()).or_insert_with(|| HybridHit {
            rank: 0,
            id: hit.id.clone(),
            score: 0.0,
            vector_score: None,
            bm25_score: None,
            votes: 0,
            source: hit.source.clone(),
            heading_path: hit.heading_path.clone(),
            text: hit.text.clone(),
        });

        entry.score += contribution;
        entry.votes += 1;
        entry.bm25_score = Some(hit.score);
    }

    let mut hits: Vec<HybridHit> = acc.into_values().collect();

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let a_best = a.vector_score.unwrap_or(0.0).max(a.bm25_score.unwrap_or(0.0));
                let b_best = b.vector_score.unwrap_or(0.0).max(b.bm25_score.unwrap_or(0.0));
                b_best
                    .partial_cmp(&a_best)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.id.cmp(&b.id))
    });

    hits.truncate(top_k);
    for (i, hit) in hits.iter_mut().enumerate() {
        hit.rank = i;
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vhit(id: &str, score: f32) -> SearchHit {
        SearchHit {
            id: id.to_string(),
            score,
            source: "doc.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: format!("正文 {id}"),
        }
    }

    fn bhit(id: &str, score: f32) -> Bm25Hit {
        Bm25Hit {
            id: id.to_string(),
            score,
            source: "doc.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: format!("正文 {id}"),
            rank: 0,
        }
    }

    #[test]
    fn dedupes_across_paths() {
        let hits = finish(vec![vhit("a", 0.9), vhit("b", 0.8)], vec![bhit("a", 5.0)], 10);

        assert_eq!(hits.len(), 2, "同一个 ID 只能占一个位置");
        let a = hits.iter().find(|h| h.id == "a").unwrap();
        assert_eq!(a.votes, 2, "两路都命中 → 共识度 2");
        assert!(a.vector_score.is_some());
        assert!(a.bm25_score.is_some());
    }

    #[test]
    fn rewards_consensus() {
        // 「a」两路都命中，「x」只在向量路排第一。
        // RRF 下共识的「a」应该排前面。
        let hits = finish(vec![vhit("x", 0.9), vhit("a", 0.8)], vec![bhit("a", 5.0)], 10);

        assert_eq!(hits[0].id, "a");
        assert_eq!(hits[0].votes, 2);
    }

    #[test]
    fn single_path_mode_has_no_votes_beyond_one() {
        let hits = finish(vec![vhit("a", 0.9)], Vec::new(), 10);
        assert_eq!(hits[0].votes, 1);
        assert!(hits[0].bm25_score.is_none());
    }

    #[test]
    fn respects_top_k() {
        let hits = finish(
            vec![vhit("a", 0.9), vhit("b", 0.8), vhit("c", 0.7)],
            Vec::new(),
            2,
        );
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn ranks_are_sequential() {
        let hits = finish(vec![vhit("a", 0.9), vhit("b", 0.8)], Vec::new(), 10);
        assert_eq!(hits[0].rank, 0);
        assert_eq!(hits[1].rank, 1);
        assert_eq!(hits[0].citation(), "[1]");
    }

    #[test]
    fn fusion_is_deterministic() {
        let first = finish(vec![vhit("a", 0.9), vhit("b", 0.8)], vec![bhit("c", 3.0)], 10);
        let second = finish(vec![vhit("a", 0.9), vhit("b", 0.8)], vec![bhit("c", 3.0)], 10);

        let ids1: Vec<&str> = first.iter().map(|h| h.id.as_str()).collect();
        let ids2: Vec<&str> = second.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn empty_inputs_produce_empty_result() {
        assert!(finish(Vec::new(), Vec::new(), 10).is_empty());
    }

    #[test]
    fn parses_mode_names() {
        assert_eq!(SearchMode::parse("hybrid"), Some(SearchMode::Hybrid));
        assert_eq!(SearchMode::parse("Vector"), Some(SearchMode::Vector));
        assert_eq!(SearchMode::parse("bm25"), Some(SearchMode::Bm25));
        assert_eq!(SearchMode::parse("unknown"), None);
    }

    #[test]
    fn all_modes_contains_three() {
        assert_eq!(SearchMode::all().len(), 3);
    }
}
