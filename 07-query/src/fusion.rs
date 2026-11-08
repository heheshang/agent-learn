//! 查询融合（Query Fusion）—— 把多路检索结果合并成一份。
//!
//! ## 为什么需要融合
//!
//! Multi-Query / Decompose 各自检索一次，得到 N 份「排序列表」。
//! 它们之间会**大量重复**（同一个块可能在每一路里都排第一），
//! 也会有**长尾**（某个块只在一路里出现过）。
//!
//! 融合要回答两个问题：
//!
//! 1. **去重** —— 同一个块不能占多个位置（否则 top-k 全是重复内容）
//! 2. **排序** —— 怎么把 N 份列表合成一份可信的全局排序
//!
//! ## 两种融合算法
//!
//! ### RRF（Reciprocal Rank Fusion，倒数排名融合）
//!
//! ```text
//! score(d) = Σ  1 / (k + rank_i(d))
//!              i∈lists
//! ```
//!
//! `k` 是常数（通常 60），`rank_i(d)` 是文档 d 在第 i 路里的排名（从 1 开始）。
//!
//! **它的妙处在于只看排名，不看分数。**
//!
//! 为什么这很重要？因为不同检索路径的分数**根本不可比**：
//!
//! | 路径 | 分数来源 | 分数范围 |
//! |---|---|---|
//! | 原问题 | 余弦相似度 | 0.62 ~ 0.78 |
//! | HyDE 假想文档 | 余弦相似度 | 0.55 ~ 0.83 |
//! | Multi-Query 变体 | 余弦相似度 | 0.60 ~ 0.75 |
//!
//! 你不能直接把 0.78 和 0.55 比大小 —— 但「排第 1」和「排第 20」
//! 是可以比的。RRF 只用排名，所以天然免疫这个问题。
//!
//! 这也是为什么 RRF 是工业界的默认选择。
//!
//! ### MaxScore（最高分胜出）
//!
//! ```text
//! score(d) = max_i score_i(d)
//! ```
//!
//! 简单直接：谁的分最高就用谁的。
//!
//! | | RRF | MaxScore |
//! |---|---|---|
//! | 需要分数可比 | ❌ 不需要 | ✅ 需要 |
//! | 奖励「多路都命中」 | ✅ 是 | ❌ 否 |
//! | 对单路高分极度敏感 | 较稳健 | ❌ 容易 |
//!
//! 本项目两个都实现，可以用 `/api/compare` 对比。

use agent_logging::debug;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::retrieval::RetrievedChunk;

/// 融合方法。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionMethod {
    /// RRF（倒数排名融合）—— **默认**。
    ///
    /// 只看排名，跨路径可比，奖励多路共识。
    Rrf,

    /// 最高分胜出。
    ///
    /// 直接取最大分数。
    MaxScore,

    /// 不融合 —— 把所有路径的结果简单拼接（按每路顺序）。
    ///
    /// 这是**对照基线**，用来证明「融合本身」有没有价值。
    Concat,
}

impl FusionMethod {
    /// 方法名。
    pub fn name(self) -> &'static str {
        match self {
            FusionMethod::Rrf => "rrf",
            FusionMethod::MaxScore => "max_score",
            FusionMethod::Concat => "concat",
        }
    }

    /// 从字符串解析。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "rrf" => Some(FusionMethod::Rrf),
            "max_score" | "maxscore" => Some(FusionMethod::MaxScore),
            "concat" => Some(FusionMethod::Concat),
            _ => None,
        }
    }

    /// 所有方法。
    pub fn all() -> Vec<Self> {
        vec![
            FusionMethod::Rrf,
            FusionMethod::MaxScore,
            FusionMethod::Concat,
        ]
    }
}

/// RRF 的常数 `k`。
///
/// ## 这个数字在干什么
///
/// `score(d) = 1 / (k + rank)`。
///
/// `k` 的作用是**压制头部的排他性**：
///
/// - `k = 0` → 第 1 名得 1.0 分，第 10 名得 0.1 分 —— 头部极度强势，
///   一路的第 1 名会碾压其它路的第 1 名
/// - `k = 60`（默认）→ 第 1 名得 0.0164，第 10 名得 0.0143 —— 差距很小，
///   **「多路都排在中上」比「一路排第一」更有价值**
///
/// 这是 RRF 的核心设计哲学：**共识 > 单点最优**。
///
/// `k = 60` 是 Cormack 等人在原论文里的经验值，也是各搜索引擎的通用默认。
pub const RRF_K: f32 = 60.0;

/// 一路检索的结果。
///
/// 「一路」= 一个改写查询对应的一次检索。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedRun {
    /// 这一路的查询文本。
    pub query: String,

    /// 这一路的策略（HyDE / MultiQuery / ...）。
    pub strategy: crate::transform::Strategy,

    /// 命中块（按相关度降序）。
    pub chunks: Vec<RetrievedChunk>,
}

impl RankedRun {
    /// 构造一路结果。
    pub fn new(
        query: impl Into<String>,
        strategy: crate::transform::Strategy,
        chunks: Vec<RetrievedChunk>,
    ) -> Self {
        Self {
            query: query.into(),
            strategy,
            chunks,
        }
    }

    /// 这一路的查询数（1 = 单路）。
    pub fn is_single(&self) -> bool {
        true
    }
}

/// 一条融合后的命中。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedHit {
    /// 融合后的排名（0 = 最相关）。
    pub rank: usize,

    /// 块 ID。
    pub id: String,

    /// 融合分数。
    ///
    /// - RRF：各路 `1/(k+rank)` 之和
    /// - MaxScore：各路最高相似度
    /// - Concat：无意义（等于 0）
    pub score: f32,

    /// 原始最高相似度（各路里的最大值）—— 便于展示「这个块本身有多相关」。
    pub best_score: f32,

    /// 有几路检索命中了它 —— **共识度**。
    ///
    /// 这是判断融合结果是否可信的关键指标：
    /// 共识度 3/3 的块几乎一定对；共识度 1/3 的块可能只是某一路的噪声。
    pub votes: usize,

    /// 来源文件名。
    pub source: String,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 块正文。
    pub text: String,

    /// 命中它的每一路（查询文本）。
    pub matched_queries: Vec<String>,
}

impl FusedHit {
    /// 引用标记（1 开始）。
    pub fn citation(&self) -> String {
        format!("[{}]", self.rank + 1)
    }

    /// 渲染成上下文片段（和 06-rag 的格式保持一致）。
    pub fn to_context(&self) -> String {
        let head = if self.heading_path.is_empty() {
            String::new()
        } else {
            format!("# {}\n", self.heading_path.join(" > "))
        };
        format!("[{}] {}{}", self.rank + 1, head, self.text)
    }
}

/// 融合多路结果。
///
/// # 参数
///
/// - `runs`：各路检索结果
/// - `method`：融合方法
/// - `top_k`：最终保留多少条
pub fn fuse(runs: &[RankedRun], method: FusionMethod, top_k: usize) -> Vec<FusedHit> {
    let hits = match method {
        FusionMethod::Rrf => fuse_rrf(runs, top_k),
        FusionMethod::MaxScore => fuse_max_score(runs, top_k),
        FusionMethod::Concat => fuse_concat(runs, top_k),
    };

    debug!(
        method = method.name(),
        run_count = runs.len(),
        hit_count = hits.len(),
        top_k,
        "融合完成"
    );

    hits
}

/// RRF 融合。
///
/// 对每个块累加 `1 / (k + rank)`，然后按融合分数降序。
///
/// ## 实现要点
///
/// 1. **按 ID 去重** —— 同一个块在多路里出现只算一次（但累加多路的分数）
/// 2. **保留 `best_score`** —— 融合分数和「这个块本身有多相关」是两回事
/// 3. **记录 `votes`** —— 共识度
/// 4. **稳定的次级排序** —— 融合分数相同时按 `best_score` 排，再按 ID 排
///    （保证结果**可复现**，不依赖 HashMap 的迭代顺序）
fn fuse_rrf(runs: &[RankedRun], top_k: usize) -> Vec<FusedHit> {
    // key = 块 ID，value = 累加器
    let mut acc: HashMap<String, HitAccumulator> = HashMap::new();

    for run in runs {
        for (idx, chunk) in run.chunks.iter().enumerate() {
            // 排名从 1 开始。
            let rank = idx + 1;
            let contribution = 1.0 / (RRF_K + rank as f32);

            let entry = acc
                .entry(chunk.id.clone())
                .or_insert_with(|| HitAccumulator {
                    source: chunk.source.clone(),
                    heading_path: chunk.heading_path.clone(),
                    text: chunk.text.clone(),
                    best_score: f32::NEG_INFINITY,
                    fused_score: 0.0,
                    votes: 0,
                    matched_queries: Vec::new(),
                });

            entry.fused_score += contribution;
            entry.votes += 1;
            entry.best_score = entry.best_score.max(chunk.score);

            // 记录命中它的查询（去重）。
            if !entry.matched_queries.contains(&run.query) {
                entry.matched_queries.push(run.query.clone());
            }
        }
    }

    let mut hits: Vec<FusedHit> = acc
        .into_iter()
        .map(|(id, a)| FusedHit {
            rank: 0,
            id,
            score: a.fused_score,
            best_score: a.best_score,
            votes: a.votes,
            source: a.source,
            heading_path: a.heading_path,
            text: a.text,
            matched_queries: a.matched_queries,
        })
        .collect();

    // 排序：融合分数降序 → 最高相似度降序 → ID 升序（保证确定性）。
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.best_score
                    .partial_cmp(&a.best_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.id.cmp(&b.id))
    });

    hits.truncate(top_k);
    assign_ranks(&mut hits);
    hits
}

/// MaxScore 融合。
///
/// 取每个块在各路里的**最高**相似度作为分数。
///
/// ⚠️ 这个方法假设各路分数可比 —— 对同一嵌入模型的多路查询基本成立，
/// 但如果一路用的是 HyDE（分数分布不同），就不可靠了。
fn fuse_max_score(runs: &[RankedRun], top_k: usize) -> Vec<FusedHit> {
    let mut acc: HashMap<String, HitAccumulator> = HashMap::new();

    for run in runs {
        for chunk in &run.chunks {
            let entry = acc
                .entry(chunk.id.clone())
                .or_insert_with(|| HitAccumulator {
                    source: chunk.source.clone(),
                    heading_path: chunk.heading_path.clone(),
                    text: chunk.text.clone(),
                    best_score: f32::NEG_INFINITY,
                    fused_score: 0.0,
                    votes: 0,
                    matched_queries: Vec::new(),
                });

            entry.best_score = entry.best_score.max(chunk.score);
            entry.fused_score = entry.best_score;
            entry.votes += 1;

            if !entry.matched_queries.contains(&run.query) {
                entry.matched_queries.push(run.query.clone());
            }
        }
    }

    let mut hits: Vec<FusedHit> = acc
        .into_iter()
        .map(|(id, a)| FusedHit {
            rank: 0,
            id,
            score: a.best_score,
            best_score: a.best_score,
            votes: a.votes,
            source: a.source,
            heading_path: a.heading_path,
            text: a.text,
            matched_queries: a.matched_queries,
        })
        .collect();

    hits.sort_by(|a, b| {
        b.best_score
            .partial_cmp(&a.best_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });

    hits.truncate(top_k);
    assign_ranks(&mut hits);
    hits
}

/// 简单拼接融合（对照基线）。
///
/// 把各路结果依次拼起来，去重后取前 top_k。
///
/// 这个方法**没有排序逻辑** —— 第一路的第 1 名一定排在
/// 第二路的第 1 名前面。用它来对比 RRF，能直观看出
/// 「融合算法」比「简单拼接」好在哪。
fn fuse_concat(runs: &[RankedRun], top_k: usize) -> Vec<FusedHit> {
    let mut seen: HashMap<String, FusedHit> = HashMap::new();
    let mut ordered: Vec<String> = Vec::new();

    for run in runs {
        for chunk in &run.chunks {
            if seen.contains_key(&chunk.id) {
                continue;
            }
            seen.insert(
                chunk.id.clone(),
                FusedHit {
                    rank: 0,
                    id: chunk.id.clone(),
                    score: chunk.score,
                    best_score: chunk.score,
                    votes: 1,
                    source: chunk.source.clone(),
                    heading_path: chunk.heading_path.clone(),
                    text: chunk.text.clone(),
                    matched_queries: vec![run.query.clone()],
                },
            );
            ordered.push(chunk.id.clone());
        }
    }

    let mut hits: Vec<FusedHit> = ordered
        .into_iter()
        .filter_map(|id| seen.remove(&id))
        .collect();

    hits.truncate(top_k);
    assign_ranks(&mut hits);
    hits
}

/// 重新分配排名。
fn assign_ranks(hits: &mut [FusedHit]) {
    for (i, hit) in hits.iter_mut().enumerate() {
        hit.rank = i;
    }
}

/// 融合过程中的累加器。
struct HitAccumulator {
    source: String,
    heading_path: Vec<String>,
    text: String,
    best_score: f32,
    fused_score: f32,
    votes: usize,
    matched_queries: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::RetrievedChunk;
    use crate::transform::Strategy;

    fn chunk(id: &str, score: f32) -> RetrievedChunk {
        RetrievedChunk {
            rank: 0,
            id: id.to_string(),
            score,
            source: "doc.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: format!("正文 {id}"),
            heading_context: "# 标题".to_string(),
        }
    }

    fn run(query: &str, ids: &[(&str, f32)]) -> RankedRun {
        RankedRun::new(
            query,
            Strategy::MultiQuery,
            ids.iter().map(|(id, s)| chunk(id, *s)).collect(),
        )
    }

    #[test]
    fn rrf_prefers_consensus() {
        // A 在两路都排第 2，B 只在一路排第 1。
        // RRF 下 A 的分数更高 —— 共识胜出。
        let runs = vec![
            run("q1", &[("x", 0.9), ("a", 0.8)]),
            run("q2", &[("b", 0.9), ("a", 0.75)]),
        ];

        let fused = fuse(&runs, FusionMethod::Rrf, 10);
        assert_eq!(fused[0].id, "a");
        assert_eq!(fused[0].votes, 2);
    }

    #[test]
    fn rrf_deduplicates_ids() {
        let runs = vec![
            run("q1", &[("a", 0.9), ("b", 0.8)]),
            run("q2", &[("a", 0.85), ("b", 0.7)]),
        ];

        let fused = fuse(&runs, FusionMethod::Rrf, 10);
        assert_eq!(fused.len(), 2);
        assert!(fused.iter().all(|h| h.votes == 2));
    }

    #[test]
    fn rrf_records_matched_queries() {
        let runs = vec![run("查询一", &[("a", 0.9)]), run("查询二", &[("a", 0.8)])];

        let fused = fuse(&runs, FusionMethod::Rrf, 10);
        assert_eq!(fused[0].matched_queries.len(), 2);
        assert!(fused[0].matched_queries.contains(&"查询一".to_string()));
    }

    #[test]
    fn max_score_prefers_single_best() {
        // 与 RRF 相反：单路的最高分胜出。
        let runs = vec![
            run("q1", &[("x", 0.9), ("a", 0.8)]),
            run("q2", &[("b", 0.9), ("a", 0.75)]),
        ];

        let fused = fuse(&runs, FusionMethod::MaxScore, 10);
        // x 和 b 都是 0.9，按 ID 升序打破平局 → b 在前。
        assert_eq!(fused[0].best_score, 0.9);
    }

    #[test]
    fn concat_preserves_run_order() {
        let runs = vec![
            run("q1", &[("a", 0.9), ("b", 0.8)]),
            run("q2", &[("c", 0.95), ("d", 0.7)]),
        ];

        let fused = fuse(&runs, FusionMethod::Concat, 10);
        let ids: Vec<&str> = fused.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn respects_top_k() {
        let runs = vec![run("q1", &[("a", 0.9), ("b", 0.8), ("c", 0.7)])];
        let fused = fuse(&runs, FusionMethod::Rrf, 2);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn ranks_are_sequential() {
        let runs = vec![run("q1", &[("a", 0.9), ("b", 0.8)])];
        let fused = fuse(&runs, FusionMethod::Rrf, 10);
        assert_eq!(fused[0].rank, 0);
        assert_eq!(fused[1].rank, 1);
        assert_eq!(fused[0].citation(), "[1]");
    }

    #[test]
    fn fusion_is_deterministic() {
        let runs = vec![
            run("q1", &[("a", 0.9), ("b", 0.8)]),
            run("q2", &[("c", 0.9), ("a", 0.7)]),
        ];

        // 跑两次，结果必须完全一致 —— 否则线上会出现「同样的问题，
        // 每次召回的块顺序都不一样」这种无法调试的问题。
        let first = fuse(&runs, FusionMethod::Rrf, 10);
        let second = fuse(&runs, FusionMethod::Rrf, 10);
        let ids1: Vec<&str> = first.iter().map(|h| h.id.as_str()).collect();
        let ids2: Vec<&str> = second.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn empty_runs_produce_empty_result() {
        let fused = fuse(&[], FusionMethod::Rrf, 10);
        assert!(fused.is_empty());
    }
}
