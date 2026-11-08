//! 端到端流水线 —— 改写 → 检索 → 重排 → 压缩 → 生成。
//!
//! ```text
//!                                  ┌─ 多问法改写（可选）
//!                                  │
//! 问题 ───────────────────────────> ├─> 混合检索（Milvus dense + BM25 + RRF）
//!                                  │         │
//!                                  │         ├─> 候选池（candidate_k）
//!                                  │         │
//!                                  │         └─> cross-encoder 重排 ──> top_n
//!                                  │
//!                                  └─> 上下文压缩（token 预算）
//!                                            │
//!                                            └─> Prompt 组装 ──> LLM ──> 带引用的答案
//! ```
//!
//! ## 与 08-rag 的关系
//!
//! 08-rag 是「本地版」：FAISS + 客户端 BM25 + 本地重排。
//! 本项目是「生产版」：Milvus 服务端混合 + 本地重排 + 上下文压缩
//! + 可选的多问法改写。
//!
//! 重排器、LLM 抽象、引用抽取**全部复用** 08 / 07 的实现 ——
//! 本项目只重写「检索层」和「压缩层」。
//!
//! ## 降级链
//!
//! 这个流水线**每一环都有降级路径**：
//!
//! | 环节失败 | 降级行为 |
//! |---|---|
//! | 多问法改写失败 | 退回原问题继续检索 |
//! | 服务端混合检索失败 | 退回客户端双路融合 |
//! | 重排失败 | 退回融合后的原始顺序 |
//! | 上下文压缩失败 | 退回未压缩的完整片段 |
//! | 检索彻底失败 | 报错（没有数据就没答案） |
//! | 生成失败 | 报错（检索结果仍随错误一起返回） |
//!
//! ⚠️ 注意最后一条：生成失败时**检索结果仍然是有价值的**。

use std::sync::Arc;
use std::time::Instant;

use agent_logging::{debug, info, warn};
use rag_07::llm::Llm;
use rag_07::transform::{QueryTransformer, Strategy, TransformedQuery};
use rag_08::reranker::{RerankDocument, Reranker};
use serde::{Deserialize, Serialize};

use crate::compress::{CompressedContext, ContextCompressor};
use crate::config::{AppConfig, SearchMode};
use crate::error::{RagError, RagResult};
use crate::retriever::{Hit, RetrievalPath, Retriever};

/// 一次「检索 + 重排 + 压缩」的完整结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalOutcome {
    /// 原始问题。
    pub query: String,

    /// 使用的检索模式。
    pub mode: SearchMode,

    /// 实际走的检索路径。
    pub path: RetrievalPath,

    /// 重排前的候选（融合结果）。
    pub fused_hits: Vec<Hit>,

    /// 重排后的命中（最终结果）。
    ///
    /// 重排被禁用或失败时，它等于 `fused_hits`（截断到 `top_n`）。
    pub hits: Vec<Hit>,

    /// 重排是否真的执行了。
    pub reranked: bool,

    /// 降级原因（如果有）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,

    /// 使用的查询变体（多问法模式下会有多条）。
    pub queries: Vec<String>,

    /// 检索耗时（毫秒）。
    pub retrieve_ms: u64,

    /// 重排耗时（毫秒）。
    pub rerank_ms: u64,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

impl RetrievalOutcome {
    /// 最终命中的块数。
    pub fn hit_count(&self) -> usize {
        self.hits.len()
    }

    /// 命中的块 ID 列表（用于计算召回重叠度）。
    pub fn hit_ids(&self) -> Vec<String> {
        self.hits.iter().map(|h| h.id.clone()).collect()
    }
}

/// 压缩后的上下文 + 压缩统计。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressOutcome {
    /// 压缩统计。
    #[serde(flatten)]
    pub stats: CompressedContext,

    /// 压缩是否真的执行了。
    pub compressed: bool,
}

/// 一次问答的完整结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskOutcome {
    /// 最终答案。
    pub answer: String,

    /// 引用到的块。
    pub citations: Vec<Citation>,

    /// 检索过程（融合 + 重排 + 压缩）。
    pub retrieval: RetrievalOutcome,

    /// 压缩后的上下文文本。
    pub context: String,

    /// 压缩统计。
    pub compression: CompressOutcome,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 一条引用。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// 引用编号（从 1 开始，对应 Prompt 里的 `[n]`）。
    pub index: usize,

    /// 被引用的块 ID。
    pub chunk_id: String,

    /// 来源文件名。
    pub source: String,

    /// 标题路径。
    pub heading_path: Vec<String>,
}

/// 摘要式 RAG 流水线。
pub struct SummaryPipeline {
    retriever: Arc<dyn Retriever>,
    reranker: Option<Arc<dyn Reranker>>,
    compressor: Arc<ContextCompressor>,
    transformer: Option<Arc<dyn QueryTransformer>>,
    llm: Arc<dyn Llm>,
    config: AppConfig,
}

impl std::fmt::Debug for SummaryPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SummaryPipeline")
            .field("mode", &self.config.retrieval.mode.name())
            .field("rerank", &self.config.rerank.enabled)
            .field("rewrite", &self.rewrite_enabled())
            .finish()
    }
}

impl SummaryPipeline {
    /// 创建流水线。
    pub fn new(
        retriever: Arc<dyn Retriever>,
        reranker: Option<Arc<dyn Reranker>>,
        compressor: Arc<ContextCompressor>,
        transformer: Option<Arc<dyn QueryTransformer>>,
        llm: Arc<dyn Llm>,
        config: AppConfig,
    ) -> Self {
        Self {
            retriever,
            reranker,
            compressor,
            transformer,
            llm,
            config,
        }
    }

    /// 是否启用查询改写。
    pub fn rewrite_enabled(&self) -> bool {
        self.transformer.is_some() && self.config.rewrite.enabled
    }

    /// 当前配置。
    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    /// 检索 + 重排 + 压缩（**不生成**）。
    pub async fn retrieve(&self, query: &str) -> RagResult<RetrievalOutcome> {
        self.retrieve_with(query, self.config.retrieval.mode).await
    }

    /// 检索 + 重排 + 压缩（指定检索模式）。
    pub async fn retrieve_with(
        &self,
        query: &str,
        mode: SearchMode,
    ) -> RagResult<RetrievalOutcome> {
        let start = Instant::now();
        let query = query.trim();

        if query.is_empty() {
            return Err(RagError::BadRequest("问题不能为空".to_string()));
        }

        // ── 0. 查询改写（可选）────────────────────────────────────
        let (queries, degrade_notes) = self.transform_queries(query, mode).await;

        // ── 1. 混合检索 ──────────────────────────────────────────
        let retrieve_start = Instant::now();

        let per_path_k = self
            .config
            .retrieval
            .effective_per_path_k(self.config.rerank.candidate_k);
        let candidate_pool = self.config.rerank.candidate_k.max(self.config.rerank.top_n);

        // 多问法模式下每一路各自检索，再做**跨查询融合**。
        //
        // ⚠️ 这是「两层融合」的第二层：
        // 第一层是「单查询内的 dense + BM25 融合」，
        // 第二层是「多个查询变体之间的融合」。
        let (fused_hits, path) = if queries.len() > 1 {
            let mut routes: Vec<Vec<Hit>> = Vec::with_capacity(queries.len());
            let mut last_path = RetrievalPath::ServerHybrid;

            for q in &queries {
                let hits = self
                    .retriever
                    .search(q, candidate_pool, per_path_k, mode)
                    .await?;
                last_path = hits.first().map(|h| h.path).unwrap_or(last_path);
                routes.push(hits);
            }

            (fuse_across_queries(routes, candidate_pool), last_path)
        } else {
            let hits = self
                .retriever
                .search(&queries[0], candidate_pool, per_path_k, mode)
                .await?;
            let path = hits
                .first()
                .map(|h| h.path)
                .unwrap_or(RetrievalPath::DenseOnly);
            (hits, path)
        };

        let retrieve_ms = retrieve_start.elapsed().as_millis() as u64;

        debug!(
            mode = mode.name(),
            query_count = queries.len(),
            candidate_pool,
            fused = fused_hits.len(),
            retrieve_ms,
            "混合检索完成"
        );

        // 阈值过滤（只对「能拿到稠密分数」的命中生效）。
        //
        // ⚠️ 服务端混合检索的分数是**融合后的 RRF 分**，
        // 客户端拿不到分项的稠密相似度 —— 拿 RRF 分和余弦阈值比
        // 是量纲错误。所以这里靠 `dense_score` 是否存在来判断：
        // 没有稠密分的块**原样保留**，而不是被误杀。
        let mut fused_hits = fused_hits;
        if let Some(min) = self.config.retrieval.min_score {
            fused_hits.retain(|h| h.dense_score.is_none_or(|s| s >= min));
            for (i, hit) in fused_hits.iter_mut().enumerate() {
                hit.rank = i;
            }
        }

        // ── 2. 重排 ──────────────────────────────────────────────
        let rerank_start = Instant::now();
        let (hits, reranked, mut degrade_reason) = self.rerank(query, &fused_hits).await;
        let rerank_ms = rerank_start.elapsed().as_millis() as u64;

        if let Some(note) = degrade_notes {
            degrade_reason = Some(match degrade_reason {
                Some(existing) => format!("{existing}；{note}"),
                None => note,
            });
        }

        if let Some(reason) = &degrade_reason {
            warn!(mode = mode.name(), reason = %reason, "检索链路发生降级");
        }

        Ok(RetrievalOutcome {
            query: query.to_string(),
            mode,
            path,
            fused_hits,
            hits,
            reranked,
            degrade_reason,
            queries,
            retrieve_ms,
            rerank_ms,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// 完整问答：改写 → 检索 → 重排 → 压缩 → 生成。
    pub async fn ask(&self, query: &str) -> RagResult<AskOutcome> {
        self.ask_with(query, self.config.retrieval.mode).await
    }

    /// 完整问答（指定检索模式 —— 消融实验用）。
    pub async fn ask_with(&self, query: &str, mode: SearchMode) -> RagResult<AskOutcome> {
        let start = Instant::now();

        let retrieval = self.retrieve_with(query, mode).await?;

        if retrieval.hits.is_empty() {
            return Err(RagError::NotReady("检索没有召回到任何内容".to_string()));
        }

        // ── 3. 上下文压缩 ────────────────────────────────────────
        let compressed = self.compressor.compress(&retrieval.hits);
        let compress_enabled = self.config.compress.enabled;

        debug!(
            enabled = compress_enabled,
            tokens_before = compressed.tokens_before,
            tokens_after = compressed.tokens_after,
            dropped = compressed.dropped,
            truncated = compressed.truncated,
            passages = compressed.passages.len(),
            "上下文压缩完成"
        );

        // ── 4. Prompt 组装 ───────────────────────────────────────
        // 复用 06-rag 的 system 提示词（格式与前几个项目保持一致），
        // 但 user 提示词用压缩后的上下文。
        let prompt = rag_06::prompt::assemble_no_context(query);
        let user_prompt = build_user_prompt(&compressed.text, query);

        // ── 5. 生成 ──────────────────────────────────────────────
        let answer = self
            .llm
            .chat(&prompt.system, &user_prompt)
            .await
            .map_err(|e| RagError::Generation(e.to_string()))?;

        let citations = extract_citations(&answer, &compressed.passages);

        info!(
            mode = mode.name(),
            answer_len = answer.chars().count(),
            citations = citations.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "生成完成"
        );

        Ok(AskOutcome {
            answer,
            citations,
            context: compressed.text.clone(),
            compression: CompressOutcome {
                stats: compressed,
                compressed: compress_enabled,
            },
            retrieval,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// 查询改写（可选）。
    ///
    /// ## 降级语义
    ///
    /// 改写失败**不报错** —— 退回原问题继续检索。
    /// 召回差一点，但答案仍然能给出。
    async fn transform_queries(
        &self,
        query: &str,
        mode: SearchMode,
    ) -> (Vec<String>, Option<String>) {
        if !mode.uses_query_rewrite() {
            return (vec![query.to_string()], None);
        }

        let Some(transformer) = &self.transformer else {
            return (vec![query.to_string()], None);
        };

        match transformer.transform(query).await {
            Ok(queries) if !queries.is_empty() => {
                let texts: Vec<String> = queries
                    .iter()
                    .map(|q: &TransformedQuery| q.text.trim().to_string())
                    .filter(|t| !t.is_empty())
                    .collect();

                if texts.is_empty() {
                    warn!("改写结果为空，降级为原问题");
                    (
                        vec![query.to_string()],
                        Some("改写结果为空，降级为原问题".to_string()),
                    )
                } else {
                    debug!(query_count = texts.len(), "查询改写完成");
                    (texts, None)
                }
            }
            Ok(_) => (
                vec![query.to_string()],
                Some("改写返回空结果，降级为原问题".to_string()),
            ),
            Err(e) => {
                warn!(error = %e, "查询改写失败，降级为原问题");
                (
                    vec![query.to_string()],
                    Some(format!("改写失败（{e}），降级为原问题")),
                )
            }
        }
    }

    /// 重排（带降级）。
    ///
    /// ## 降级语义
    ///
    /// 重排失败 → **不报错**，退回「融合后的原始顺序」。
    ///
    /// 理由：重排是**精排**环节，它的作用是「把对的排到前面」。
    /// 它挂了，结果**依然可用**（只是顺序差一点）。
    async fn rerank(&self, query: &str, fused_hits: &[Hit]) -> (Vec<Hit>, bool, Option<String>) {
        let Some(reranker) = &self.reranker else {
            return (
                fused_hits
                    .iter()
                    .take(self.config.rerank.top_n)
                    .cloned()
                    .collect(),
                false,
                None,
            );
        };

        if fused_hits.is_empty() {
            return (Vec::new(), false, None);
        }

        let documents: Vec<RerankDocument> = fused_hits
            .iter()
            .map(|h| {
                RerankDocument::new(
                    h.id.clone(),
                    h.source.clone(),
                    h.heading_path.clone(),
                    h.text.clone(),
                )
            })
            .collect();

        match reranker
            .rerank(query, &documents, self.config.rerank.top_n)
            .await
        {
            Ok(reranked) => {
                let by_id: std::collections::HashMap<&str, &Hit> =
                    fused_hits.iter().map(|h| (h.id.as_str(), h)).collect();

                let hits: Vec<Hit> = reranked
                    .into_iter()
                    .filter_map(|r| {
                        let original = by_id.get(r.id.as_str())?;
                        let mut hit = (*original).clone();
                        hit.rank = r.rank;
                        hit.text = r.text.clone();
                        Some(hit)
                    })
                    .collect();

                debug!(
                    hit_count = hits.len(),
                    top_n = self.config.rerank.top_n,
                    "重排完成"
                );
                (hits, true, None)
            }
            Err(e) => {
                warn!(error = %e, "重排失败，降级为融合后的原始顺序");
                (
                    fused_hits
                        .iter()
                        .take(self.config.rerank.top_n)
                        .cloned()
                        .collect(),
                    false,
                    Some(format!("重排失败（{e}），降级为融合后的原始顺序")),
                )
            }
        }
    }
}

/// 跨查询变体的融合（第二层 RRF）。
///
/// 多个问法各自检索后，结果会有大量重复（同一个块在每一路里都排前面）。
/// 这里按块 ID 去重并累加 RRF 分数 —— 和 07-query 的 `fuse_rrf` 同构。
///
/// ## ⚠️ 排名必须是「路内排名」
///
/// RRF 的公式是 `1 / (k + rank_i(d))`，其中 `rank_i(d)` 是文档 d
/// **在第 i 路里**的名次。
///
/// 所以这里按**每一路自己的下标**算排名，而不是把各路拼起来后的全局下标。
/// 用全局下标的话，第二路的第 1 名会被当成第 `len(route1)+1` 名 ——
/// RRF 分数被严重低估，多路融合就退化成了「第一路说了算」。
///
/// ## 为什么要用 RRF 而不是 MaxScore
///
/// 各路都是向量检索，分数**可比** —— 但它们是**不同查询**下的相似度，
/// 分布仍有差异（一个问法更泛、更容易给出高相似度）。
/// RRF 只看排名，天然免疫这个问题。
pub fn fuse_across_queries(routes: Vec<Vec<Hit>>, top_k: usize) -> Vec<Hit> {
    use std::collections::HashMap;

    let mut acc: HashMap<String, Hit> = HashMap::new();

    for route in &routes {
        for (index, hit) in route.iter().enumerate() {
            // ⚠️ 路内排名，从 1 开始。
            let rank = index + 1;
            let contribution = 1.0 / (rag_08::hybrid::RRF_K + rank as f32);

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
                path: hit.path,
            });

            entry.score += contribution;
            entry.votes += 1;
            entry.dense_score = hit.dense_score.or(entry.dense_score);
            entry.bm25_score = hit.bm25_score.or(entry.bm25_score);
        }
    }

    let mut fused: Vec<Hit> = acc.into_values().collect();

    fused.sort_by(|a, b| {
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

    fused.truncate(top_k);
    for (index, hit) in fused.iter_mut().enumerate() {
        hit.rank = index;
    }

    fused
}

/// 组装 User 提示词：上下文 + 问题。
fn build_user_prompt(context: &str, query: &str) -> String {
    let context_section = if context.is_empty() {
        "（本次检索没有召回任何资料片段）".to_string()
    } else {
        context.to_string()
    };

    format!("【参考资料】\n{context_section}\n\n【问题】\n{query}")
}

/// 从答案里抽取引用编号。
///
/// 只接受落在实际块数范围内的编号 —— 模型编造的 `[9]` 要丢掉
/// （留着等于给用户一个假引用入口）。
///
/// ⚠️ 这里用的是**压缩后**的块列表 —— 因为引用编号对应的是
/// 真正进入上下文的那些块。
fn extract_citations(answer: &str, passages: &[Hit]) -> Vec<Citation> {
    let mut citations: Vec<Citation> = Vec::new();
    let mut seen: Vec<usize> = Vec::new();

    for group in find_bracket_groups(answer) {
        for number in group {
            if number == 0 || number > passages.len() {
                continue;
            }
            if seen.contains(&number) {
                continue;
            }
            seen.push(number);

            let hit = &passages[number - 1];
            citations.push(Citation {
                index: number,
                chunk_id: hit.id.clone(),
                source: hit.source.clone(),
                heading_path: hit.heading_path.clone(),
            });
        }
    }

    citations.sort_by_key(|c| c.index);
    citations
}

/// 找出所有 `[...]` 里的数字组。
fn find_bracket_groups(answer: &str) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let bytes = answer.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'[' {
            let close = answer[i..]
                .char_indices()
                .take_while(|(offset, _)| *offset < 64)
                .find(|(_, c)| *c == ']')
                .map(|(offset, _)| i + offset);

            if let Some(close) = close {
                let inner = &answer[i + 1..close];
                let numbers: Vec<usize> = inner
                    .split(',')
                    .filter_map(|part| part.trim().parse::<usize>().ok())
                    .collect();

                if !numbers.is_empty() {
                    groups.push(numbers);
                }

                i = close + 1;
                continue;
            }
        }
        i += 1;
    }

    groups
}

/// 从环境变量解析改写策略（供 bootstrap 使用）。
pub fn rewrite_strategy_name(strategy: Strategy) -> &'static str {
    strategy.name()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retriever::RetrievalPath;

    fn hit(id: &str) -> Hit {
        Hit {
            rank: 0,
            id: id.to_string(),
            score: 0.03,
            dense_score: Some(0.8),
            bm25_score: None,
            votes: 1,
            source: "doc.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: format!("正文 {id}"),
            path: RetrievalPath::ServerHybrid,
        }
    }

    #[test]
    fn extracts_valid_citations_only() {
        let passages = vec![hit("a"), hit("b"), hit("c")];
        let cites = extract_citations("见 [1] 和 [3]，还有 [9]。", &passages);

        assert_eq!(cites.len(), 2);
        assert_eq!(cites[0].index, 1);
        assert_eq!(cites[1].index, 3);
    }

    #[test]
    fn citations_are_deduplicated() {
        let passages = vec![hit("a"), hit("b")];
        let cites = extract_citations("[1] 和 [1] 重复。", &passages);
        assert_eq!(cites.len(), 1);
    }

    #[test]
    fn cross_query_fusion_dedupes_ids() {
        let routes = vec![vec![hit("a"), hit("b")], vec![hit("a"), hit("c")]];
        let fused = fuse_across_queries(routes, 10);

        assert_eq!(fused.len(), 3, "同一个块只能占一个位置");
    }

    #[test]
    fn cross_query_fusion_rewards_consensus() {
        // 「a」两路都命中，「x」只在一路。
        let routes = vec![vec![hit("x"), hit("a")], vec![hit("a")]];
        let fused = fuse_across_queries(routes, 10);

        assert_eq!(fused[0].id, "a");
        assert_eq!(fused[0].votes, 2);
    }

    /// 回归测试：RRF 的排名必须是**路内排名**。
    ///
    /// 如果错用全局下标，第二路的第 1 名会被当成第 N+1 名 ——
    /// 分数被严重低估，多路融合就退化成「第一路说了算」。
    #[test]
    fn cross_query_fusion_uses_per_route_ranks() {
        // 两路完全一样：["a", "b"] 和 ["a", "b"]。
        // 「a」在两路都排第 1 → 分数 = 2 / (60 + 1) ≈ 0.0328。
        let routes = vec![vec![hit("a"), hit("b")], vec![hit("a"), hit("b")]];
        let fused = fuse_across_queries(routes, 10);

        let expected = 2.0 / 61.0;
        assert!(
            (fused[0].score - expected).abs() < 1e-6,
            "路内排名下的 RRF 分数应为 {expected}, 实际 {}",
            fused[0].score
        );
    }

    #[test]
    fn cross_query_fusion_respects_top_k() {
        let routes = vec![vec![hit("a"), hit("b"), hit("c")]];
        assert_eq!(fuse_across_queries(routes, 2).len(), 2);
    }

    #[test]
    fn cross_query_fusion_is_deterministic() {
        let first = fuse_across_queries(vec![vec![hit("a"), hit("b")], vec![hit("a")]], 10);
        let second = fuse_across_queries(vec![vec![hit("a"), hit("b")], vec![hit("a")]], 10);

        let ids1: Vec<&str> = first.iter().map(|h| h.id.as_str()).collect();
        let ids2: Vec<&str> = second.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn user_prompt_marks_empty_context() {
        let prompt = build_user_prompt("", "问题");
        assert!(prompt.contains("没有召回任何资料片段"));
    }

    #[test]
    fn user_prompt_includes_context_and_question() {
        let prompt = build_user_prompt("[1] 内容", "问题是什么");
        assert!(prompt.contains("[1] 内容"));
        assert!(prompt.contains("问题是什么"));
    }
}
