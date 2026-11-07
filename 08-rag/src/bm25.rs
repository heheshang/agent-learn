//! BM25 —— 稀疏检索（关键词检索）。
//!
//! ## BM25 是什么
//!
//! BM25 是最经典的**稀疏检索**算法（Robertson / Zaragoza 等人，1994 年提出），
//! 也是 Elasticsearch、Lucene 默认的打分函数。
//!
//! 它给每个「查询词 → 文档」的匹配打分，然后按分数排序。
//!
//! ## 打分公式
//!
//! ```text
//!                        f(t,D) · (k₁ + 1)
//!  score(D,Q) = Σ  IDF(t) · ─────────────────────
//!  t∈Q                    f(t,D) + k₁ · (1 - b + b·|D|/avgdl)
//! ```
//!
//! 拆开看，三个部分各管一件事：
//!
//! | 部分 | 作用 | 类比 |
//! |---|---|---|
//! | `IDF(t)` | 这个词**有多稀有** | 罕见词比常见词更能区分文档 |
//! | `f(t,D)` | 这个词在文档里**出现几次** | 出现多次 → 更相关（但有饱和） |
//! | `|D|/avgdl` | 文档**长度归一化** | 长文档不该仅因为长就占优 |
//!
//! ### IDF：词频的逆比
//!
//! ```text
//! IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
//! ```
//!
//! - `N` = 文档总数，`df(t)` = 包含词 `t` 的文档数
//! - 出现在**所有**文档里的词（`df = N`），IDF ≈ 0 —— 它没有区分度
//!   （比如「的」「是」「系统」）
//! - 只出现在**一个**文档里的词，IDF 最大 —— 它几乎是指针
//!
//! ### 词频饱和：`k₁`
//!
//! 朴素的想法是「出现 100 次比出现 10 次相关 10 倍」。**这是错的。**
//!
//! 如果一个文档里「检索」出现 100 次，它几乎可以肯定是关于检索的 ——
//! 但它比出现 10 次的相关性并没有高 10 倍。第 11 次出现带来的
//! 信息量远小于第 1 次。
//!
//! `k₁` 控制这个饱和速度：默认 `1.2`。
//!
//! - `k₁ = 0` → 词频完全不参与打分（纯布尔检索）
//! - `k₁` 越大 → 词频的贡献越接近线性（饱和越慢）
//!
//! ### 长度归一化：`b`
//!
//! 短文档里出现 3 次「检索」比长文档里出现 3 次更能说明问题。
//!
//! `b` 控制长度惩罚的强度：默认 `0.75`。
//!
//! - `b = 0` → 完全不做长度归一化
//! - `b = 1` → 完整的长度归一化
//! - `b = 0.75` → 经验折中值
//!
//! ## 为什么稀疏检索仍然是必要的
//!
//! 因为**向量检索对精确匹配是盲的**。
//!
//! | 查询 | 向量检索 | BM25 |
//! |---|---|---|
//! | 「`ECONNREFUSED` 错误码」 | ❌ 语义相近但对不上 | ✅ 精确命中 |
//! | 「`bge-reranker-v2-m3`」 | ❌ 罕见 token 被稀释 | ✅ 精确命中 |
//! | 「怎么优化检索速度」 | ✅ 语义匹配 | ⚠️ 关键词对不上时召不回 |
//!
//! **两者的失败模式正好互补** —— 这就是混合检索（Hybrid Search）的理论基础。
//!
//! ## 倒排索引
//!
//! 为了让打分高效，本模块维护一个**倒排索引**：
//!
//! ```text
//! 词 → [(文档下标, 该词词频), ...]
//! ```
//!
//! 查询时只遍历「查询里出现的词」对应的 posting list，
//! 而不用扫描全部文档 —— 这是 BM25 能在百万级文档上跑得动的原因。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tokenize::tokenize;

/// BM25 的词频饱和参数 `k₁`。
pub const K1: f32 = 1.2;

/// BM25 的长度归一化参数 `b`。
pub const B: f32 = 0.75;

/// 一篇参与 BM25 检索的文档。
#[derive(Debug, Clone)]
pub struct Bm25Document {
    /// 稳定 ID（与向量库里的 ID 一致）。
    pub id: String,
    /// 来源文件名。
    pub source: String,
    /// 标题路径。
    pub heading_path: Vec<String>,
    /// 文档正文。
    pub text: String,
}

/// 一条 BM25 命中。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Hit {
    /// 命中的文档 ID。
    pub id: String,
    /// BM25 分数（越大越相关）。
    pub score: f32,
    /// 来源文件名。
    pub source: String,
    /// 标题路径。
    pub heading_path: Vec<String>,
    /// 文档正文。
    pub text: String,
    /// 命中的排名（0 = 最相关）。
    pub rank: usize,
}

/// BM25 索引。
///
/// ## 数据结构
///
/// ```text
/// postings:  词 → [(文档下标, 词频)]
/// doc_len:   文档下标 → 词元总数
/// doc_meta:  文档下标 → (ID, 来源, 标题路径, 正文)
/// avgdl:     平均文档长度
/// ```
pub struct Bm25Index {
    /// 倒排索引：词 → [(文档下标, 词频)]。
    postings: HashMap<String, Vec<(usize, u32)>>,
    /// 每个文档的词元数。
    doc_len: Vec<u32>,
    /// 每个文档的元数据。
    doc_meta: Vec<(String, String, Vec<String>, String)>,
    /// 文档总数。
    num_docs: usize,
    /// 平均文档长度。
    avgdl: f32,
}

impl std::fmt::Debug for Bm25Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bm25Index")
            .field("num_docs", &self.num_docs)
            .field("vocab_size", &self.postings.len())
            .field("avgdl", &self.avgdl)
            .finish()
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

impl Bm25Index {
    /// 创建空索引。
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_len: Vec::new(),
            doc_meta: Vec::new(),
            num_docs: 0,
            avgdl: 0.0,
        }
    }

    /// 从文档列表构建索引。
    pub fn build(documents: &[Bm25Document]) -> Self {
        let mut index = Self::new();

        for doc in documents {
            index.add(doc);
        }

        index.finalize();
        index
    }

    /// 往索引里加一篇文档。
    ///
    /// ## 为什么不直接用 `build`
    ///
    /// 因为有些场景是**增量**的 —— 比如重建索引时，一批一批地加。
    /// 但注意：加完之后**必须**调 [`Bm25Index::finalize`] 刷新 `avgdl`，
    /// 否则长度归一化会用到过期的平均值。
    pub fn add(&mut self, doc: &Bm25Document) {
        let doc_index = self.doc_meta.len();

        let tokens = tokenize(&doc.text);
        let mut freq: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *freq.entry(token.clone()).or_insert(0) += 1;
        }

        for (term, tf) in freq {
            self.postings.entry(term).or_default().push((doc_index, tf));
        }

        self.doc_len.push(tokens.len() as u32);
        self.doc_meta.push((
            doc.id.clone(),
            doc.source.clone(),
            doc.heading_path.clone(),
            doc.text.clone(),
        ));
        self.num_docs += 1;
    }

    /// 刷新派生统计量（`avgdl`）。
    pub fn finalize(&mut self) {
        self.avgdl = if self.doc_len.is_empty() {
            0.0
        } else {
            let total: u32 = self.doc_len.iter().sum();
            total as f32 / self.doc_len.len() as f32
        };
    }

    /// 索引里的文档数。
    pub fn len(&self) -> usize {
        self.num_docs
    }

    /// 索引是否为空。
    pub fn is_empty(&self) -> bool {
        self.num_docs == 0
    }

    /// 词表大小（不同的词的数量）。
    pub fn vocab_size(&self) -> usize {
        self.postings.len()
    }

    /// 某个词出现在多少篇文档里（文档频率 `df`）。
    pub fn doc_freq(&self, term: &str) -> usize {
        self.postings.get(term).map(|p| p.len()).unwrap_or(0)
    }

    /// 逆文档频率（IDF）。
    ///
    /// 用的是 BM25 标准公式（带 `ln(1 + ...)` 平滑）：
    ///
    /// ```text
    /// IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
    /// ```
    ///
    /// `+1` 和 `ln` 的作用：保证 IDF **永远为正**。
    /// 没有平滑的话，出现在超过一半文档里的词会得到负 IDF ——
    /// 那会让「文档包含了太多常见词」反而**降低**分数，明显不合理。
    pub fn idf(&self, term: &str) -> f32 {
        let n = self.num_docs as f32;
        let df = self.doc_freq(term) as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// 检索：查询词 → top-k 命中。
    ///
    /// ## 算法
    ///
    /// 1. 分词
    /// 2. 对每个查询词，取出它的 posting list
    /// 3. 对每个 (文档, 词频) 计算 BM25 增量分，累加到该文档的分数上
    /// 4. 按分数降序排序，取前 k 条
    ///
    /// ## 关于查询词的重复
    ///
    /// 查询里同一个词出现两次，**只算一次**。
    ///
    /// 原因：IDF 已经编码了「这个词有多稀有」。重复查询词是想强调，
    /// 但在 BM25 的框架里，重复只会线性放大同一个词的贡献，
    /// 而这偏离了「词集合」的语义。业界实现（Lucene / ES）也是去重的。
    pub fn search(&self, query: &str, k: usize) -> Vec<Bm25Hit> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        // 查询词去重：同一个词在查询里出现多次只算一次
        // （IDF 已经编码了「这个词有多稀有」，重复只会线性放大它的贡献）。
        let mut query_terms: Vec<&str> = Vec::new();
        for term in tokenize(query) {
            if self.postings.contains_key(&term) && !query_terms.contains(&term.as_str()) {
                query_terms.push(self.postings.get_key_value(&term).unwrap().0.as_str());
            }
        }

        if query_terms.is_empty() {
            return Vec::new();
        }

        // 文档下标 → 累计分数。
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in &query_terms {
            let idf = self.idf(term);
            if idf <= 0.0 {
                continue;
            }

            let postings = &self.postings[*term];
            for (doc_index, tf) in postings {
                let score = self.bm25_term_score(*doc_index, *tf, idf);
                *scores.entry(*doc_index).or_insert(0.0) += score;
            }
        }

        // 按分数降序。分数相同时按文档下标升序 —— 保证结果**可复现**。
        let mut ranked: Vec<(usize, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        ranked
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (doc_index, score))| {
                let (id, source, heading_path, text) = &self.doc_meta[doc_index];
                Bm25Hit {
                    id: id.clone(),
                    score,
                    source: source.clone(),
                    heading_path: heading_path.clone(),
                    text: text.clone(),
                    rank,
                }
            })
            .collect()
    }

    /// 单个词对单个文档的 BM25 贡献。
    ///
    /// ```text
    ///              f(t,D) · (k₁ + 1)
    ///  contribution = ─────────────────────
    ///                f(t,D) + k₁ · (1 - b + b·|D|/avgdl)
    /// ```
    ///
    /// 分母里的 `k₁ · (1 - b + b·|D|/avgdl)` 叫**长度归一化因子**：
    ///
    /// - 文档长度 = 平均长度（`|D| = avgdl`）→ 因子 = `k₁`
    /// - 文档比平均长 2 倍 → 因子 ≈ `k₁ · (1 + b)`（惩罚更重）
    /// - 文档比平均短一半 → 因子 ≈ `k₁ · (1 - b/2)`（惩罚更轻）
    fn bm25_term_score(&self, doc_index: usize, tf: u32, idf: f32) -> f32 {
        let doc_len = self.doc_len[doc_index] as f32;

        // avgdl 为 0（所有文档都空）时避免除零。
        let avgdl = if self.avgdl > 0.0 { self.avgdl } else { 1.0 };

        let length_norm = 1.0 - B + B * (doc_len / avgdl);
        let tf = tf as f32;

        idf * (tf * (K1 + 1.0)) / (tf + K1 * length_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, text: &str) -> Bm25Document {
        Bm25Document {
            id: id.to_string(),
            source: "test.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: text.to_string(),
        }
    }

    fn sample_index() -> Bm25Index {
        Bm25Index::build(&[
            doc("a", "检索增强生成通过外部知识库缓解知识截止问题"),
            doc("b", "倒排索引从关键词映射到文档列表"),
            doc("c", "光合作用分为光反应和暗反应两个阶段"),
            doc("d", "数据库索引最常见的是 B 树索引"),
        ])
    }

    #[test]
    fn finds_exact_keyword_match() {
        let index = sample_index();
        let hits = index.search("倒排索引", 5);

        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, "b", "精确关键词应该排在第一");
    }

    #[test]
    fn rare_terms_outrank_common_ones() {
        let index = Bm25Index::build(&[
            doc("a", "系统 系统 系统 系统"),
            doc("b", "系统 系统 系统 光合作用"),
        ]);

        // 「光合作用」只在一个文档里 → IDF 高；
        // 「系统」在两个文档里 → IDF 低。
        let hits = index.search("系统 光合作用", 5);
        assert_eq!(hits[0].id, "b");
    }

    #[test]
    fn idf_is_always_positive() {
        let index = sample_index();
        // 「的」之类的常见词虽然 IDF 很小，但绝不会变成负数。
        for term in ["检索", "索引", "系统"] {
            assert!(index.idf(term) >= 0.0, "IDF({term}) 必须是正的");
        }
    }

    #[test]
    fn common_term_gets_lower_idf_than_rare_term() {
        let index = Bm25Index::build(&[
            doc("a", "系统"),
            doc("b", "系统"),
            doc("c", "光合作用"),
        ]);

        assert!(index.idf("光合作用") > index.idf("系统"));
    }

    #[test]
    fn longer_documents_are_penalized() {
        // 同样出现 1 次，但短文档应该得分更高。
        let index = Bm25Index::build(&[
            doc("short", "检索"),
            doc("long", &format!("检索 {}", "无关内容 ".repeat(50))),
        ]);

        let hits = index.search("检索", 5);
        assert_eq!(hits[0].id, "short");
    }

    #[test]
    fn term_frequency_saturates() {
        // 出现 2 次应该比出现 1 次得分高，但不应该高 2 倍。
        let index = Bm25Index::build(&[
            doc("once", "检索"),
            doc("twice", "检索 检索"),
            doc("many", &format!("检索 {}", "检索 ".repeat(20))),
        ]);

        let hits = index.search("检索", 5);
        let by_id = |id: &str| hits.iter().find(|h| h.id == id).unwrap().score;

        assert!(by_id("twice") > by_id("once"));
        // 20 次不应该比 2 次高 10 倍 —— 饱和效应。
        assert!(
            by_id("many") < by_id("twice") * 5.0,
            "词频应该饱和: many={} twice={}",
            by_id("many"),
            by_id("twice")
        );
    }

    #[test]
    fn empty_query_returns_nothing() {
        let index = sample_index();
        assert!(index.search("", 5).is_empty());
        // 用一个拉丁词 —— 中文查询很难「完全不命中」，
        // 因为单个汉字（比如「的」）几乎在每篇文档里都出现过。
        assert!(index.search("zzzqqqxyw", 5).is_empty());
    }

    #[test]
    fn empty_index_returns_nothing() {
        let index = Bm25Index::new();
        assert!(index.search("检索", 5).is_empty());
        assert!(index.is_empty());
    }

    #[test]
    fn respects_top_k() {
        let index = sample_index();
        let hits = index.search("索引", 1);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn results_are_deterministic() {
        let index = sample_index();
        let first = index.search("索引 检索", 5);
        let second = index.search("索引 检索", 5);

        let ids1: Vec<&str> = first.iter().map(|h| h.id.as_str()).collect();
        let ids2: Vec<&str> = second.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn ranks_are_sequential() {
        let index = sample_index();
        let hits = index.search("索引", 5);
        for (i, hit) in hits.iter().enumerate() {
            assert_eq!(hit.rank, i);
        }
    }

    #[test]
    fn cjk_bigrams_make_phrase_matching_work() {
        let index = Bm25Index::build(&[
            doc("a", "光合作用的光反应发生在类囊体膜上"),
            doc("b", "数据库索引的查找复杂度是 O(log n)"),
        ]);

        // 「类囊体膜」是三个字，只有 bigram 能把它和别的文档区分开。
        let hits = index.search("类囊体膜", 5);
        assert_eq!(hits[0].id, "a");
    }

    #[test]
    fn repeated_query_terms_count_once() {
        let index = sample_index();
        let once = index.search("检索", 5);
        let twice = index.search("检索 检索", 5);

        assert_eq!(once[0].score, twice[0].score);
    }

    #[test]
    fn tracks_document_frequency() {
        let index = sample_index();
        assert_eq!(index.doc_freq("检索"), 1);
        assert_eq!(index.doc_freq("索引"), 2);
        assert_eq!(index.doc_freq("不存在的词"), 0);
    }

    #[test]
    fn vocab_size_counts_distinct_terms() {
        let index = sample_index();
        // 4 篇文档，每篇的词都不完全重叠 → 词表应该远大于 4。
        assert!(index.vocab_size() > 4);
    }

    #[test]
    fn incremental_add_matches_bulk_build() {
        let docs = vec![
            doc("a", "检索增强生成"),
            doc("b", "倒排索引"),
            doc("c", "光合作用"),
        ];

        let bulk = Bm25Index::build(&docs);

        let mut incremental = Bm25Index::new();
        for d in &docs {
            incremental.add(d);
        }
        incremental.finalize();

        let bulk_hits = bulk.search("检索 索引", 5);
        let inc_hits = incremental.search("检索 索引", 5);

        assert_eq!(bulk_hits.len(), inc_hits.len());
        for (b, i) in bulk_hits.iter().zip(inc_hits.iter()) {
            assert_eq!(b.id, i.id);
            assert!((b.score - i.score).abs() < 1e-5);
        }
    }
}
