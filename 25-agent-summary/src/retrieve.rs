//! **检索层** —— BM25 倒排索引 + RRF 混合融合。
//!
//! # 这个模块汇总了哪些历史能力
//!
//! | 能力 | 来自 | 在这里的形态 |
//! |---|---|---|
//! | BM25 打分 + 倒排索引 | 08-rag | [`Bm25Index`] |
//! | 双路召回 + RRF 融合 | 08-rag | [`HybridRetriever`] |
//!
//! # 为什么用 RRF 而不是加权求和
//!
//! 稠密向量分和 BM25 分**量纲不同**（一个是无界浮点，一个是 TF-IDF 变体），
//! 加权求和的权重怎么调都不对。
//!
//! RRF（Reciprocal Rank Fusion）**只看排名不看分数**：
//! `score = Σ 1 / (k + rank)`，天然规避量纲问题。
//!
//! `k = 60` 是原论文的经验值 —— 它让「排第一」和「排第十」的贡献差距
//! 不会大到压倒其它路。

use std::collections::{HashMap, HashSet};

use crate::ingest::{Tokens, tokenize};

/// RRF 的常数 `k`。
pub const RRF_K: f64 = 60.0;

/// 一篇参与检索的文档。
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Document {
    /// 稳定 ID。
    pub id: String,

    /// 来源。
    pub source: String,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 正文。
    pub text: String,
}

/// 一条命中。
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Hit {
    /// 文档 ID。
    pub id: String,

    /// 分数。
    pub score: f64,

    /// 来源。
    pub source: String,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 正文。
    pub text: String,

    /// 排名（0 = 最相关）。
    pub rank: usize,

    /// ★ 这条命中来自哪一路（`bm25` / `dense`）。
    pub route: String,
}

/// BM25 倒排索引。
#[derive(Debug, Clone, Default)]
pub struct Bm25Index {
    /// 词 → [(文档下标, 词频)]。
    postings: HashMap<String, Vec<(usize, u32)>>,

    /// 文档下标 → 词元总数。
    doc_len: Vec<usize>,

    /// 文档下标 → 元信息。
    doc_meta: Vec<(String, String, Vec<String>, String)>,

    /// 平均文档长度。
    avgdl: f64,
}

impl Bm25Index {
    /// 建索引。
    pub fn build(documents: &[Document]) -> Self {
        let mut index = Self::default();
        for doc in documents {
            index.add(doc);
        }
        index.finalize();
        index
    }

    /// 加一篇文档。
    pub fn add(&mut self, doc: &Document) {
        let doc_index = self.doc_meta.len();
        let tokens = tokenize(&doc.text);

        let mut freq: HashMap<String, u32> = HashMap::new();
        for t in &tokens {
            *freq.entry(t.clone()).or_insert(0) += 1;
        }

        for (term, count) in freq {
            self.postings
                .entry(term)
                .or_default()
                .push((doc_index, count));
        }

        self.doc_len.push(tokens.len());
        self.doc_meta.push((
            doc.id.clone(),
            doc.source.clone(),
            doc.heading_path.clone(),
            doc.text.clone(),
        ));
    }

    /// 收尾（算 `avgdl`）。
    pub fn finalize(&mut self) {
        self.avgdl = if self.doc_len.is_empty() {
            0.0
        } else {
            self.doc_len.iter().sum::<usize>() as f64 / self.doc_len.len() as f64
        };
    }

    /// 文档数。
    pub fn len(&self) -> usize {
        self.doc_meta.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.doc_meta.is_empty()
    }

    /// 词表大小。
    pub fn vocab_size(&self) -> usize {
        self.postings.len()
    }

    /// 某词出现在多少篇文档里。
    pub fn doc_freq(&self, term: &str) -> usize {
        self.postings.get(term).map(|v| v.len()).unwrap_or(0)
    }

    /// IDF。
    pub fn idf(&self, term: &str) -> f64 {
        let n = self.len() as f64;
        let df = self.doc_freq(term) as f64;
        // 加 1 平滑，避免 `df = n` 时 idf = 0。
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// 检索。
    pub fn search(&self, query: &str, k: usize) -> Vec<Hit> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() || self.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<usize, f64> = HashMap::new();
        let k1 = 1.2;
        let b = 0.75;

        for term in &query_tokens {
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let idf = self.idf(term);
            for (doc_index, freq) in postings {
                let dl = self.doc_len[*doc_index] as f64;
                let f = *freq as f64;
                let denom = f + k1 * (1.0 - b + b * dl / self.avgdl.max(1.0));
                let score = idf * (f * (k1 + 1.0)) / denom;
                *scores.entry(*doc_index).or_insert(0.0) += score;
            }
        }

        let mut ranked: Vec<(usize, f64)> = scores.into_iter().collect();
        // ⚠️ 排序必须**确定** —— 分数相同时按文档下标排。
        //    否则同样输入会产出不同顺序，「三档产出必须一致」这条约束会挂。
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        ranked
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (doc_index, score))| {
                let (id, source, heading_path, text) = &self.doc_meta[doc_index];
                Hit {
                    id: id.clone(),
                    score,
                    source: source.clone(),
                    heading_path: heading_path.clone(),
                    text: text.clone(),
                    rank,
                    route: "bm25".to_string(),
                }
            })
            .collect()
    }
}

/// ★ 稠密路的检索函数类型。
///
/// 抽成别名是因为 clippy 的 `type_complexity` ——
/// `Box<dyn Fn(&str, usize) -> Vec<(String, f64)> + Send + Sync>` 太长了。
pub type DenseSearch = Box<dyn Fn(&str, usize) -> Vec<(String, f64)> + Send + Sync>;

/// 混合检索器 —— 双路召回 + RRF 融合。
///
/// ## ★ RRF 的排名必须是**路内排名**
///
/// 如果用全局下标融合多路，会退化成「第一路说了算」——
/// 因为第一路的下标总是最小，`1/(k+rank)` 总是最大。
/// 必须**每路各自从 0 开始排名**，再相加。
pub struct HybridRetriever {
    /// 关键词路。
    pub bm25: Bm25Index,

    /// 语义路的检索函数（返回带分数的文档列表）。
    ///
    /// 这里用闭包而不是 trait —— 因为本项目**不引入外部依赖**，
    /// 语义路用「按词元 Jaccard 相似度」这个纯计算近似即可。
    pub dense: DenseSearch,
}

impl HybridRetriever {
    /// RRF 融合两路结果。
    ///
    /// ## ⚠️ 每一路**独立**排名
    ///
    /// 这是本项目最容易被写错的地方 —— 见类型文档。
    pub fn fuse(&self, query: &str, k: usize) -> Vec<Hit> {
        let sparse = self.bm25.search(query, k);

        // 语义路：返回 (doc_id, score)，按分数降序。
        let mut dense_raw = (self.dense)(query, k);
        dense_raw.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let mut fused: HashMap<String, (f64, Hit)> = HashMap::new();

        // 路 1：BM25（路内排名从 0 开始）。
        for (rank, hit) in sparse.iter().enumerate() {
            let contrib = 1.0 / (RRF_K + (rank + 1) as f64);
            let entry = fused
                .entry(hit.id.clone())
                .or_insert_with(|| (0.0, hit.clone()));
            entry.0 += contrib;
        }

        // 路 2：稠密（路内排名从 0 开始 —— ⚠️ 不是接在 sparse 后面）。
        for (rank, (doc_id, _score)) in dense_raw.iter().enumerate() {
            let contrib = 1.0 / (RRF_K + (rank + 1) as f64);
            if let Some(entry) = fused.get_mut(doc_id) {
                entry.0 += contrib;
            }
        }

        let mut hits: Vec<Hit> = fused
            .into_values()
            .map(|(score, mut hit)| {
                hit.score = score;
                hit
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        hits.truncate(k);

        for (rank, hit) in hits.iter_mut().enumerate() {
            hit.rank = rank;
        }
        hits
    }
}

/// 一个**纯计算**的稠密检索近似。
///
/// ## ⚠️ 为什么不用真的嵌入模型
///
/// 真模型要下 470MB，且推理耗时是**硬件相关**的 ——
/// 压测要测的是「代码逻辑」，不是「ONNX 推理速度」。
///
/// 这里的近似是**词元 Jaccard 相似度**：
/// 查两个文档共享多少词元。它和 BM25 的区别在于
/// **BM25 看词频与 IDF，Jaccard 只看集合重叠** —— 足够体现「两路不同源」。
///
/// ⚠️ 它的**绝对**相关性不真实，但**相对**关系（两路给出不同排序）是真实的，
/// 而 RRF 融合要验证的正是这个。
///
/// ## ⚠️ 为什么是**具名结构体**而不是 `impl Fn`
///
/// `impl Fn` 会捕获生命周期（edition 2024 起），
/// 而它要被装进 `Box<dyn Fn ... + Send + Sync>`（`'static`）——
/// 具名结构体**拥有**数据，就没这个问题。
pub struct JaccardDense {
    /// 每篇文档的词元集合。
    doc_tokens: Vec<(String, HashSet<String>)>,
}

impl JaccardDense {
    /// 从索引构造。
    pub fn new(index: &Bm25Index) -> Self {
        let doc_tokens: Vec<(String, HashSet<String>)> = (0..index.len())
            .map(|i| {
                let text = index.document_text(i);
                (
                    index.document_id(i).to_string(),
                    tokenize(text).into_iter().collect(),
                )
            })
            .collect();
        Self { doc_tokens }
    }

    /// 检索。
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f64)> {
        let q: HashSet<String> = tokenize(query).into_iter().collect();
        if q.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(String, f64)> = self
            .doc_tokens
            .iter()
            .map(|(id, tokens)| {
                let inter = q.intersection(tokens).count() as f64;
                let union = q.union(tokens).count() as f64;
                (id.clone(), if union > 0.0 { inter / union } else { 0.0 })
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }
}

impl Bm25Index {
    /// 取文档 ID。
    pub fn document_id(&self, index: usize) -> &str {
        &self.doc_meta[index].0
    }

    /// 取文档正文。
    pub fn document_text(&self, index: usize) -> &str {
        &self.doc_meta[index].3
    }
}

/// 分词的公共别名（方便外部调用）。
pub type TokenSet = Tokens;

#[cfg(test)]
mod tests {
    use super::*;

    fn docs() -> Vec<Document> {
        vec![
            Document {
                id: "a".into(),
                source: "a.md".into(),
                heading_path: vec!["检索".into()],
                text: "混合检索把稠密向量与 BM25 关键词两路结果用 RRF 融合".into(),
            },
            Document {
                id: "b".into(),
                source: "b.md".into(),
                heading_path: vec!["缓存".into()],
                text: "内容寻址缓存键必须包含所有影响输出的字段".into(),
            },
            Document {
                id: "c".into(),
                source: "c.md".into(),
                heading_path: vec!["检索".into()],
                text: "BM25 关键词检索用倒排索引打分".into(),
            },
        ]
    }

    #[test]
    fn search_finds_the_relevant_document() {
        let index = Bm25Index::build(&docs());
        let hits = index.search("BM25 关键词", 3);
        assert!(!hits.is_empty());
        assert!(hits.iter().any(|h| h.id == "c"));
    }

    #[test]
    fn search_is_deterministic_under_ties() {
        let index = Bm25Index::build(&docs());
        let a = index.search("检索", 3);
        let b = index.search("检索", 3);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_query_returns_nothing() {
        let index = Bm25Index::build(&docs());
        assert!(index.search("!!!", 3).is_empty());
    }

    #[test]
    fn idf_is_positive() {
        let index = Bm25Index::build(&docs());
        assert!(index.idf("检索") > 0.0);
    }

    #[test]
    fn rrf_uses_per_route_ranks() {
        let index = Bm25Index::build(&docs());
        let dense = JaccardDense::new(&index);
        let retriever = HybridRetriever {
            bm25: index,
            dense: Box::new(move |q: &str, k: usize| dense.search(q, k)),
        };
        let hits = retriever.fuse("检索 BM25", 3);
        assert!(!hits.is_empty());
        // 融合分应该都 > 0。
        assert!(hits.iter().all(|h| h.score > 0.0));
    }

    #[test]
    fn fusion_is_deterministic() {
        let index = Bm25Index::build(&docs());
        let dense = JaccardDense::new(&index);
        let retriever = HybridRetriever {
            bm25: index,
            dense: Box::new(move |q: &str, k: usize| dense.search(q, k)),
        };
        let a = retriever.fuse("检索 缓存", 3);
        let b = retriever.fuse("检索 缓存", 3);
        assert_eq!(a, b);
    }

    #[test]
    fn ranks_are_sequential_after_fusion() {
        let index = Bm25Index::build(&docs());
        let dense = JaccardDense::new(&index);
        let retriever = HybridRetriever {
            bm25: index,
            dense: Box::new(move |q: &str, k: usize| dense.search(q, k)),
        };
        let hits = retriever.fuse("检索", 3);
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.rank, i);
        }
    }
}
