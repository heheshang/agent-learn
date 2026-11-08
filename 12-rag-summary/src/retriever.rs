//! 检索层 —— 在 Milvus 上做「稠密 + 稀疏」混合检索。
//!
//! ## 本项目与前两个项目的检索差异
//!
//! | 项目 | 稠密路 | 稀疏路 | 融合位置 | 融合算法 |
//! |---|---|---|---|---|
//! | 05-rag | FAISS | ❌ | — | — |
//! | 08-rag | FAISS | 客户端 BM25 | 客户端 | RRF |
//! | 10-rag-milvus | Milvus | 服务端 BM25 | 服务端 | RRF |
//! | **12-rag-summary** | **Milvus** | **客户端 BM25 + 服务端 BM25** | **客户端 + 服务端（双层）** | **RRF** |
//!
//! ## 为什么要有「双层」融合
//!
//! 服务端 hybrid（10-rag-milvus）一次 RPC 出结果，但它有一个结构性限制：
//! **分词发生在服务端**，用的是 Milvus 的 `standard` 分析器 ——
//! 它对英文友好，对中文会把整句当成一个 token。
//!
//! 客户端 BM25（08-rag）用自己的中英混排分词（CJK 单字 + 双字），
//! 中文效果好得多，但索引在**进程内**，数据量一大就吃不消。
//!
//! 所以本项目两条路都留：
//!
//! - **服务端 hybrid**：生产主路径，一次 RPC，大规模可用
//! - **客户端 BM25**：精确对照 + Milvus 不可达时的降级路径
//!
//! ## 两条路的分数不可比 —— 这是 RRF 存在的理由
//!
//! | 路 | 分数来源 | 范围 |
//! |---|---|---|
//! | 稠密 | 余弦相似度 | 0.55 ~ 0.85 |
//! | BM25 | 对数加权和 | 0 ~ 15+ |
//!
//! 直接加权求和怎么做都对不上。所以用 RRF —— **只看排名**。
//!
//! ## 降级链
//!
//! ```text
//! 服务端 hybrid 失败 ──> 客户端 dense + 客户端 BM25 + RRF
//!            │
//!            └──> 客户端也失败（Milvus 完全不可达）──> 直接报错
//! ```
//!
//! ## 复用 10-rag-milvus 的 `MilvusStore`
//!
//! 稠密检索和服务端 hybrid 的 SDK 调用**完全复用** [`MilvusStore`]：
//!
//! - `search_dense` —— 稠密检索（含一致性级别、分数方向归一化）
//! - `search_hybrid` —— 服务端 hybrid（含 RRF 重排、`Strong` 一致性级别）
//!
//! 本模块只补上 10-rag-milvus **没有**的那一块：
//! **客户端 BM25 + 客户端 RRF 融合**。

use std::collections::HashMap;
use std::sync::Arc;

use agent_logging::{debug, warn};
use rag_05::embed::Embedder;
use rag_08::bm25::{Bm25Document, Bm25Hit, Bm25Index};
use rag_08::hybrid::RRF_K;
use rag_10::milvus_store::MilvusStore;
use serde::{Deserialize, Serialize};

use rag_04::splitter::Splitter;

use crate::config::SearchMode;
use crate::error::{RagError, RagResult};

/// 一条命中。
///
/// ## 为什么要自己定义，不直接用 `rag_08::hybrid::HybridHit`
///
/// 因为本项目的检索**有三层信息**要记录：
///
/// - `dense_score` / `bm25_score`：两路各自的分数（不同时为 `None`）
/// - `votes`：几路命中了它（共识度）
/// - `path`：这个结果**来自哪一层**（服务端 / 客户端回退）
///
/// `rag_08::hybrid::HybridHit` 的 `source` 字段是「来源文件名」，
/// 语义不同 —— 所以这里定义自己的类型，避免把两个概念混在一起。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    /// 排名（0 = 最相关）。
    pub rank: usize,

    /// 块 ID。
    pub id: String,

    /// 融合分数（RRF）。
    pub score: f32,

    /// 稠密路的分数（未被稠密路召回时是 `None`）。
    pub dense_score: Option<f32>,

    /// BM25 路的分数（未被关键词路召回时是 `None`）。
    pub bm25_score: Option<f32>,

    /// 共识度：几路命中了它。
    ///
    /// 2 = 两路都命中（几乎一定对）；1 = 只有一路命中（可能是噪声）。
    pub votes: usize,

    /// 来源文件名。
    pub source: String,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 块正文。
    pub text: String,

    /// 检索路径：服务端 hybrid 还是客户端回退。
    pub path: RetrievalPath,
}

/// 检索路径 —— 用来在报告里说清「这次结果是怎么来的」。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalPath {
    /// Milvus 服务端混合检索（dense + 服务端 BM25 + 服务端 RRF）。
    ServerHybrid,

    /// 客户端双路（dense + 客户端 BM25）+ 客户端 RRF。
    ClientFusion,

    /// 纯稠密（Milvus `search_dense`）。
    DenseOnly,

    /// 纯关键词（客户端 BM25）。
    KeywordOnly,
}

impl RetrievalPath {
    /// 路径名。
    pub fn name(self) -> &'static str {
        match self {
            RetrievalPath::ServerHybrid => "server_hybrid",
            RetrievalPath::ClientFusion => "client_fusion",
            RetrievalPath::DenseOnly => "dense_only",
            RetrievalPath::KeywordOnly => "keyword_only",
        }
    }
}

impl Hit {
    /// 引用标记（1 开始）。
    pub fn citation(&self) -> String {
        format!("[{}]", self.rank + 1)
    }

    /// 渲染成上下文片段（格式和 06/07/08 保持一致）。
    pub fn to_context(&self) -> String {
        let head = if self.heading_path.is_empty() {
            String::new()
        } else {
            format!("# {}\n", self.heading_path.join(" > "))
        };
        format!("[{}] {}{}", self.rank + 1, head, self.text)
    }

    /// 一行摘要。
    pub fn summary(&self) -> String {
        let preview: String = self.text.chars().take(40).collect();
        format!(
            "[{}] {} dense={:?} bm25={:?} votes={} | {}",
            self.rank + 1,
            self.id,
            self.dense_score,
            self.bm25_score,
            self.votes,
            preview.replace('\n', " ")
        )
    }
}

/// 检索器 trait。
///
/// ## 为什么要抽成 trait
///
/// 三个理由，和 08-rag 抽 `Reranker` 是同一套逻辑：
///
/// 1. **测试能注入替身** —— API 层的参数校验、错误处理不应该依赖
///    一个真实运行的 Milvus。
/// 2. **失败模式隔离** —— 「Milvus 不可达」和「BM25 索引为空」
///    是完全不同的错误，不该混在一个具体类型里。
/// 3. **未来可替换** —— 加一个云端向量库时，流水线一行都不用改。
///
/// 返回值用显式装箱的 future（而不是 `async fn`）——
/// 因为 `async fn` 不是 dyn 兼容的。见 08-rag 的 `Reranker`。
pub trait Retriever: Send + Sync + std::fmt::Debug {
    /// 混合检索。
    ///
    /// # 参数
    ///
    /// - `query`：用户查询
    /// - `k`：返回条数
    /// - `per_path_k`：每路召回数
    /// - `mode`：检索模式 —— 决定走哪条路
    fn search<'a>(
        &'a self,
        query: &'a str,
        k: usize,
        per_path_k: usize,
        mode: SearchMode,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RagResult<Vec<Hit>>> + Send + 'a>>;

    /// 检索器名字（用于日志和响应体）。
    fn name(&self) -> &str;
}

/// 检索器 —— 封装 Milvus + 客户端 BM25 的混合检索。
pub struct HybridRetriever {
    /// Milvus 存储（复用 10-rag-milvus 的实现）。
    store: Arc<MilvusStore>,

    /// 嵌入器。
    embedder: Arc<dyn Embedder>,

    /// 客户端 BM25 索引（降级路径 + 精确对照用）。
    bm25: Arc<Bm25Index>,

    /// 服务端是否启用了 BM25。
    server_bm25: bool,
}

impl std::fmt::Debug for HybridRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridRetriever")
            .field("collection", &self.store.collection_name())
            .field("server_bm25", &self.server_bm25)
            .field("bm25_docs", &self.bm25.len())
            .finish()
    }
}

impl HybridRetriever {
    /// 创建检索器。
    pub fn new(store: Arc<MilvusStore>, embedder: Arc<dyn Embedder>, bm25: Arc<Bm25Index>) -> Self {
        let server_bm25 = store.has_bm25();
        Self {
            store,
            embedder,
            bm25,
            server_bm25,
        }
    }

    /// 服务端是否启用了 BM25。
    pub fn has_server_bm25(&self) -> bool {
        self.server_bm25
    }

    /// 客户端 BM25 索引的文档数。
    pub fn bm25_size(&self) -> usize {
        self.bm25.len()
    }

    /// 纯稠密检索（对照组 A）。
    pub async fn search_dense(&self, query: &str, k: usize) -> RagResult<Vec<Hit>> {
        let vector = self.embed(query).await?;

        let hits = self
            .store
            .search_dense(&vector, k.max(1), None)
            .await
            .map_err(|e| RagError::Retrieval(format!("Milvus 稠密检索失败: {e}")))?;

        debug!(path = "dense", hit_count = hits.len(), "稠密检索完成");

        Ok(assign_ranks(to_hits(
            hits.into_iter()
                .map(|h| (h.id, h.score, h.source, h.heading_path, h.text))
                .collect(),
            RetrievalPath::DenseOnly,
        )))
    }

    /// 纯关键词检索（对照组 B —— 客户端 BM25）。
    pub async fn search_keyword(&self, query: &str, k: usize) -> RagResult<Vec<Hit>> {
        let bm25_hits = self.bm25.search(query, k.max(1));
        debug!(
            path = "keyword",
            hit_count = bm25_hits.len(),
            "客户端 BM25 检索完成"
        );
        Ok(assign_ranks(to_hits(
            bm25_hits
                .into_iter()
                .map(|h| (h.id, h.score, h.source, h.heading_path, h.text))
                .collect(),
            RetrievalPath::KeywordOnly,
        )))
    }

    /// 服务端混合检索（生产主路径）。
    ///
    /// dense + BM25 两个 `SubSearchRequest`，服务端用 RRF 融合。
    /// 一次 RPC 出结果 —— 这就是「用生产向量库」的价值。
    pub async fn search_server_hybrid(
        &self,
        query: &str,
        k: usize,
        per_path_k: usize,
    ) -> RagResult<Vec<Hit>> {
        let vector = self.embed(query).await?;

        let hits = self
            .store
            .search_hybrid(&vector, query, k.max(1), per_path_k.max(1), None)
            .await
            .map_err(|e| RagError::Retrieval(format!("Milvus 混合检索失败: {e}")))?;

        debug!(
            path = "server_hybrid",
            hit_count = hits.len(),
            "服务端混合检索完成"
        );

        Ok(assign_ranks(to_hits(
            hits.into_iter()
                .map(|h| (h.id, h.score, h.source, h.heading_path, h.text))
                .collect(),
            RetrievalPath::ServerHybrid,
        )))
    }

    /// 客户端双路融合（dense + 客户端 BM25 + RRF）。
    ///
    /// 用途有两个：
    ///
    /// 1. **降级路径** —— Milvus 不可达时的兜底
    /// 2. **精确对照** —— 服务端 BM25 vs 客户端 BM25，中文场景差异明显
    pub async fn search_client_fusion(
        &self,
        query: &str,
        k: usize,
        per_path_k: usize,
    ) -> RagResult<Vec<Hit>> {
        let vector = self.embed(query).await?;

        let dense_hits = self
            .store
            .search_dense(&vector, per_path_k.max(1), None)
            .await
            .map_err(|e| RagError::Retrieval(format!("Milvus 稠密检索失败: {e}")))?;

        let bm25_hits = self.bm25.search(query, per_path_k.max(1));

        let fused = fuse_rrf(dense_hits, bm25_hits, k);
        debug!(
            path = "client_fusion",
            hit_count = fused.len(),
            "客户端双路融合完成"
        );
        Ok(assign_ranks(fused))
    }

    /// 嵌入查询（放到阻塞线程池，避免卡住 tokio 工作线程）。
    async fn embed(&self, query: &str) -> RagResult<Vec<f32>> {
        let embedder = Arc::clone(&self.embedder);
        let query = query.to_string();

        tokio::task::spawn_blocking(move || embedder.embed_query(&query))
            .await
            .map_err(|e| RagError::Internal(format!("嵌入任务失败: {e}")))?
            .map_err(|e| RagError::Retrieval(format!("查询向量化失败: {e}")))
    }
}

impl Retriever for HybridRetriever {
    /// 按模式检索。
    ///
    /// ## 模式 → 路径的映射
    ///
    /// | 模式 | 路径 |
    /// |---|---|
    /// | `Dense` | Milvus 纯稠密 |
    /// | `Keyword` | 客户端 BM25 |
    /// | `Hybrid` / `HybridRerank` / `Full` | 服务端混合（失败则降级到客户端融合） |
    ///
    /// ## 降级链
    ///
    /// ```text
    /// 服务端 hybrid ──失败──> 客户端融合（dense + 客户端 BM25）
    ///       │                        │
    ///       └──都失败──> 报错（没数据就没有答案）
    /// ```
    ///
    /// 返回值里的 `path` 会说清最终走的是哪条路 —— 报告里能看到
    /// 「这次是降级了还是正常」。
    fn search<'a>(
        &'a self,
        query: &'a str,
        k: usize,
        per_path_k: usize,
        mode: SearchMode,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RagResult<Vec<Hit>>> + Send + 'a>> {
        Box::pin(async move {
            match mode {
                SearchMode::Dense => self.search_dense(query, k).await,
                SearchMode::Keyword => self.search_keyword(query, k).await,
                SearchMode::Hybrid | SearchMode::HybridRerank | SearchMode::Full => {
                    match self.search_server_hybrid(query, k, per_path_k).await {
                        Ok(hits) => Ok(hits),
                        Err(server_err) => {
                            // 服务端失败 —— 降级到客户端融合。
                            warn!(error = %server_err, "服务端混合检索失败，降级到客户端融合");
                            match self.search_client_fusion(query, k, per_path_k).await {
                                Ok(client) if !client.is_empty() => Ok(client),
                                _ => Err(server_err),
                            }
                        }
                    }
                }
            }
        })
    }

    fn name(&self) -> &str {
        "milvus_hybrid"
    }
}

/// 客户端 RRF 融合。
///
/// ```text
/// score(d) = Σ  1 / (k + rank_i(d))
/// ```
///
/// 和 08-rag 的实现同构 —— 但这里融合的是「Milvus 稠密」和
/// 「客户端 BM25」两路。
///
/// ## 排序的次级键
///
/// RRF 分数相同时，按 `max(dense, bm25)` 降序，再按 ID 升序 ——
/// 保证结果**可复现**，不依赖 HashMap 的迭代顺序。
pub fn fuse_rrf(
    dense_hits: Vec<rag_05::store::SearchHit>,
    bm25_hits: Vec<Bm25Hit>,
    top_k: usize,
) -> Vec<Hit> {
    let mut acc: HashMap<String, Hit> = HashMap::new();

    for (index, hit) in dense_hits.iter().enumerate() {
        let rank = index + 1;
        let contribution = 1.0 / (RRF_K + rank as f32);

        let entry = acc.entry(hit.id.clone()).or_insert_with(|| Hit {
            rank: 0,
            id: hit.id.clone(),
            score: 0.0,
            dense_score: None,
            bm25_score: None,
            votes: 0,
            source: hit.source.clone(),
            heading_path: hit.heading_path.clone(),
            text: hit.text.clone(),
            path: RetrievalPath::ClientFusion,
        });

        entry.score += contribution;
        entry.votes += 1;
        entry.dense_score = Some(hit.score);
    }

    for (index, hit) in bm25_hits.iter().enumerate() {
        let rank = index + 1;
        let contribution = 1.0 / (RRF_K + rank as f32);

        let entry = acc.entry(hit.id.clone()).or_insert_with(|| Hit {
            rank: 0,
            id: hit.id.clone(),
            score: 0.0,
            dense_score: None,
            bm25_score: None,
            votes: 0,
            source: hit.source.clone(),
            heading_path: hit.heading_path.clone(),
            text: hit.text.clone(),
            path: RetrievalPath::ClientFusion,
        });

        entry.score += contribution;
        entry.votes += 1;
        entry.bm25_score = Some(hit.score);
    }

    let mut hits: Vec<Hit> = acc.into_values().collect();

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let a_best = a
                    .dense_score
                    .unwrap_or(0.0)
                    .max(a.bm25_score.unwrap_or(0.0));
                let b_best = b
                    .dense_score
                    .unwrap_or(0.0)
                    .max(b.bm25_score.unwrap_or(0.0));
                b_best
                    .partial_cmp(&a_best)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.id.cmp(&b.id))
    });

    hits.truncate(top_k);
    assign_ranks(hits)
}

/// 把中间结构转成 [`Hit`]。
///
/// ## ⚠️ 分数语义必须准确
///
/// | 路径 | `score` | `dense_score` | `bm25_score` |
/// |---|---|---|---|
/// | `DenseOnly` | 余弦相似度 | ✅ 同 `score` | `None` |
/// | `KeywordOnly` | BM25 分 | `None` | ✅ 同 `score` |
/// | `ClientFusion` | RRF 融合分 | ✅ 该路的稠密分 | ✅ 该路的 BM25 分 |
/// | `ServerHybrid` | 服务端 RRF 融合分 | `None` | `None` |
///
/// ⚠️ **服务端混合检索的分数不能拆成 dense / bm25** ——
///
/// 服务端返回的是**融合后**的 RRF 分数，两路的原始分数已经被
/// 服务端合成为一个值，客户端拿不到分项。
///
/// 所以这里必须填 `None` —— 如果填成 dense 分数，
/// 报告里就会出现「dense=0.0164」这种**看起来像余弦相似度、
/// 实则是 RRF 分数**的值，误导性极强。
fn to_hits(raw: Vec<(String, f32, String, Vec<String>, String)>, path: RetrievalPath) -> Vec<Hit> {
    raw.into_iter()
        .map(|(id, score, source, heading_path, text)| Hit {
            rank: 0,
            id,
            score,
            dense_score: match path {
                RetrievalPath::DenseOnly => Some(score),
                RetrievalPath::ClientFusion => Some(score),
                RetrievalPath::KeywordOnly | RetrievalPath::ServerHybrid => None,
            },
            bm25_score: match path {
                RetrievalPath::KeywordOnly => Some(score),
                _ => None,
            },
            votes: 1,
            source,
            heading_path,
            text,
            path,
        })
        .collect()
}

/// 重新分配排名。
fn assign_ranks(mut hits: Vec<Hit>) -> Vec<Hit> {
    for (index, hit) in hits.iter_mut().enumerate() {
        hit.rank = index;
    }
    hits
}

/// 用分块结果构建客户端 BM25 索引。
///
/// ## ⚠️ 必须和建索引用**完全相同**的分块配置
///
/// 否则块 ID 对不上 —— 客户端 BM25 召回的块和 Milvus 里的块
/// 是两批不同的东西，RRF 融合时会「两路命中的不是同一块」。
///
/// 这个坑和 09-rag-eval 里的一模一样：不报错，只是静默地降低效果。
pub fn build_bm25_index(
    data_dir: &std::path::Path,
    splitter_config: &rag_04::splitter::SplitterConfig,
) -> RagResult<Bm25Index> {
    let counter = rag_04::splitter::TokenCounter::new()
        .map_err(|e| RagError::Internal(format!("创建 token 计数器失败: {e}")))?;
    let splitter = rag_04::splitter::StructureAwareSplitter::new(splitter_config.clone());

    let documents = rag_04::loader::load_directory(data_dir)
        .map_err(|e| RagError::Internal(format!("加载文档失败: {e}")))?;

    let mut bm25_docs: Vec<Bm25Document> = Vec::new();

    for doc in &documents {
        for chunk in splitter.split(doc, &counter) {
            // ⚠️ 用 `text`（正文）而不是 `embedded_text`（含标题前缀）。
            //
            // 向量索引用 `embedded_text` —— 标题路径能帮语义检索。
            // 但 BM25 是**关键词**匹配：如果把标题也塞进去，
            // 标题词会主导打分，让「按标题命中」压过「按正文命中」。
            bm25_docs.push(Bm25Document {
                id: rag_05::embed::chunk_id(&chunk.source, chunk.index),
                source: chunk.source.clone(),
                heading_path: chunk.heading_path.clone(),
                text: chunk.text.clone(),
            });
        }
    }

    Ok(Bm25Index::build(&bm25_docs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(id: &str, score: f32) -> rag_05::store::SearchHit {
        rag_05::store::SearchHit {
            id: id.to_string(),
            score,
            source: "doc.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: format!("正文 {id}"),
        }
    }

    fn bm25(id: &str, score: f32) -> Bm25Hit {
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
    fn fusion_dedupes_across_paths() {
        let hits = fuse_rrf(
            vec![dense("a", 0.9), dense("b", 0.8)],
            vec![bm25("a", 5.0)],
            10,
        );

        assert_eq!(hits.len(), 2, "同一个 ID 只能占一个位置");
        let a = hits.iter().find(|h| h.id == "a").unwrap();
        assert_eq!(a.votes, 2, "两路都命中 → 共识度 2");
        assert!(a.dense_score.is_some());
        assert!(a.bm25_score.is_some());
    }

    #[test]
    fn fusion_rewards_consensus() {
        // 「a」两路都命中，「x」只在稠密路排第一。
        let hits = fuse_rrf(
            vec![dense("x", 0.9), dense("a", 0.8)],
            vec![bm25("a", 5.0)],
            10,
        );

        assert_eq!(hits[0].id, "a");
        assert_eq!(hits[0].votes, 2);
    }

    #[test]
    fn fusion_respects_top_k() {
        let hits = fuse_rrf(
            vec![dense("a", 0.9), dense("b", 0.8), dense("c", 0.7)],
            Vec::new(),
            2,
        );
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn fusion_ranks_are_sequential() {
        let hits = fuse_rrf(vec![dense("a", 0.9)], Vec::new(), 10);
        assert_eq!(hits[0].rank, 0);
        assert_eq!(hits[0].citation(), "[1]");
    }

    #[test]
    fn fusion_is_deterministic() {
        let first = fuse_rrf(
            vec![dense("a", 0.9), dense("b", 0.8)],
            vec![bm25("c", 3.0)],
            10,
        );
        let second = fuse_rrf(
            vec![dense("a", 0.9), dense("b", 0.8)],
            vec![bm25("c", 3.0)],
            10,
        );

        let ids1: Vec<&str> = first.iter().map(|h| h.id.as_str()).collect();
        let ids2: Vec<&str> = second.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn empty_inputs_produce_empty_result() {
        assert!(fuse_rrf(Vec::new(), Vec::new(), 10).is_empty());
    }

    #[test]
    fn keyword_only_path_records_bm25_score() {
        let hits = to_hits(
            vec![(
                "a".to_string(),
                7.5,
                "d".to_string(),
                vec![],
                "t".to_string(),
            )],
            RetrievalPath::KeywordOnly,
        );
        assert!(hits[0].dense_score.is_none());
        assert_eq!(hits[0].bm25_score, Some(7.5));
    }

    #[test]
    fn dense_only_path_records_dense_score() {
        let hits = to_hits(
            vec![(
                "a".to_string(),
                0.9,
                "d".to_string(),
                vec![],
                "t".to_string(),
            )],
            RetrievalPath::DenseOnly,
        );
        assert_eq!(hits[0].dense_score, Some(0.9));
        assert!(hits[0].bm25_score.is_none());
    }

    #[test]
    fn context_format_matches_previous_projects() {
        let hits = to_hits(
            vec![(
                "a".to_string(),
                0.9,
                "d".to_string(),
                vec!["标题".to_string()],
                "正文".to_string(),
            )],
            RetrievalPath::DenseOnly,
        );
        let ctx = hits[0].to_context();

        assert!(ctx.contains("[1]"));
        assert!(ctx.contains("# 标题"));
    }

    #[test]
    fn bm25_index_is_built_from_knowledge_base() {
        let data_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let index =
            build_bm25_index(&data_dir, &rag_04::splitter::SplitterConfig::default()).unwrap();

        assert!(!index.is_empty(), "BM25 索引应该有文档");
        assert!(index.vocab_size() > 10);
    }
}
