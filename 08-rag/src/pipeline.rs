//! 端到端编排 —— 混合检索 → 重排 → 生成。
//!
//! ```text
//! 问题 ──> 混合检索（BM25 + 向量，RRF 融合）──> 候选集
//!                                                  │
//!                                      重排（cross-encoder）──> top-n
//!                                                  │
//!                                            Prompt 组装
//!                                                  │
//!                                                LLM ──> 带引用的答案
//! ```
//!
//! ## 降级链
//!
//! 这个流水线**每一环都有降级路径**：
//!
//! | 环节失败 | 降级行为 |
//! |---|---|
//! | 向量路失败 | 只用 BM25 路的结果 |
//! | BM25 路失败 | 只用向量路的结果 |
//! | 两路都失败 | 直接报错（没有数据就没答案） |
//! | 重排失败 | 退回融合后的原始顺序 |
//! | 生成失败 | 报错（检索结果仍然有价值，随错误一起返回） |
//!
//! ⚠️ 注意最后一条：生成失败时**检索结果仍然是有价值的**。
//! 所以 `ask` 在生成失败时把检索结果一起返回，而不是丢弃一切。

use agent_logging::{debug, info, warn};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::{AppConfig, RerankConfig, RetrievalConfig};
use crate::error::{RagError, RagResult};
use crate::hybrid::{HybridHit, HybridRetriever, SearchMode};
use crate::reranker::{RerankDocument, Reranker};

/// 一次「检索 + 重排」的完整结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalOutcome {
    /// 原始问题。
    pub query: String,

    /// 使用的检索模式。
    pub mode: SearchMode,

    /// 融合后的命中（重排**前**）。
    pub fused_hits: Vec<HybridHit>,

    /// 重排后的命中（最终结果）。
    ///
    /// 重排被禁用或失败时，它等于 `fused_hits`（截断到 `top_n`）。
    pub hits: Vec<HybridHit>,

    /// 重排是否真的执行了。
    pub reranked: bool,

    /// 降级原因（如果有）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,

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

    /// 拼成上下文的纯文本。
    ///
    /// 格式和 06/07 保持一致（`[n] # 标题 > 子标题\n正文`），
    /// 这样三个模块的 Prompt 可以直接对比。
    pub fn to_context(&self) -> String {
        self.hits
            .iter()
            .map(HybridHit::to_context)
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// 命中的块 ID 列表（用于计算召回重叠度）。
    pub fn hit_ids(&self) -> Vec<String> {
        self.hits.iter().map(|h| h.id.clone()).collect()
    }
}

/// 检索 + 重排流水线。
pub struct HybridPipeline {
    retriever: Arc<HybridRetriever>,
    reranker: Option<Arc<dyn Reranker>>,
    llm: Arc<dyn rag_07::llm::Llm>,
    retrieval_config: RetrievalConfig,
    rerank_config: RerankConfig,
}

impl std::fmt::Debug for HybridPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridPipeline")
            .field("retrieval", &self.retrieval_config)
            .field("rerank", &self.rerank_config)
            .finish()
    }
}

impl HybridPipeline {
    /// 创建流水线。
    ///
    /// `reranker` 传 `None` 就是**关闭重排**（对照组）。
    pub fn new(
        retriever: Arc<HybridRetriever>,
        reranker: Option<Arc<dyn Reranker>>,
        llm: Arc<dyn rag_07::llm::Llm>,
        retrieval_config: RetrievalConfig,
        rerank_config: RerankConfig,
    ) -> Self {
        Self {
            retriever,
            reranker,
            llm,
            retrieval_config,
            rerank_config,
        }
    }

    /// 底层的 LLM（供 API 层组装 Prompt 后直接调用）。
    pub fn llm(&self) -> Arc<dyn rag_07::llm::Llm> {
        Arc::clone(&self.llm)
    }

    /// 当前检索配置。
    pub fn config(&self) -> &RetrievalConfig {
        &self.retrieval_config
    }

    /// 检索 + 重排（**不生成**）。
    pub async fn retrieve(&self, query: &str) -> RagResult<RetrievalOutcome> {
        self.retrieve_with(query, self.retrieval_config.mode).await
    }

    /// 检索 + 重排（指定检索模式）。
    pub async fn retrieve_with(
        &self,
        query: &str,
        mode: SearchMode,
    ) -> RagResult<RetrievalOutcome> {
        let start = Instant::now();
        let query = query.trim();
        if query.is_empty() {
            warn!("检索问题为空");
            return Err(RagError::BadRequest("问题不能为空".to_string()));
        }

        debug!(query = %query, mode = mode.name(), "开始检索 + 重排");

        // ── 1. 混合检索 ──────────────────────────────────────────
        let retrieve_start = Instant::now();

        // 召回阶段要**多取一些** —— reranker 只能重排它拿到的候选。
        //
        // ⚠️ 每路召回数必须 ≥ 候选池大小的一半，否则两路合并后
        // 候选数量达不到 `candidate_k`，重排器的工作范围被卡死。
        //
        // 所以这里取 `max(per_path_k, candidate_k)` ——
        // 两路各召回这么多条，合并去重后至少能凑够候选池。
        let per_path_k = self
            .retrieval_config
            .effective_per_path_k()
            .max(self.rerank_config.candidate_k)
            .max(self.rerank_config.top_n);

        let candidate_pool = self.rerank_config.candidate_k.max(self.rerank_config.top_n);

        let fused_hits = self
            .retriever
            .search_with_k(query, mode, candidate_pool, Some(per_path_k))
            .await?;

        let retrieve_ms = retrieve_start.elapsed().as_millis() as u64;

        // 阈值过滤（只对向量路的分数生效 —— BM25 分数不可比）。
        let mut fused_hits = fused_hits;
        if let Some(min) = self.retrieval_config.min_score {
            let before = fused_hits.len();
            fused_hits.retain(|h| h.vector_score.map(|s| s >= min).unwrap_or(true));
            for (i, hit) in fused_hits.iter_mut().enumerate() {
                hit.rank = i;
            }
            debug!(
                before,
                after = fused_hits.len(),
                min_score = min,
                "阈值过滤完成"
            );
        }

        // ── 2. 重排 ──────────────────────────────────────────────
        let rerank_start = Instant::now();
        let (hits, reranked, degrade_reason) = self.rerank(query, &fused_hits).await;
        let rerank_ms = rerank_start.elapsed().as_millis() as u64;

        info!(
            query = %query,
            mode = mode.name(),
            fused_count = fused_hits.len(),
            hit_count = hits.len(),
            reranked,
            retrieve_ms,
            rerank_ms,
            "检索 + 重排完成"
        );

        Ok(RetrievalOutcome {
            query: query.to_string(),
            mode,
            fused_hits,
            hits,
            reranked,
            degrade_reason,
            retrieve_ms,
            rerank_ms,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// 重排（带降级）。
    ///
    /// ## 降级语义
    ///
    /// 重排失败 → **不报错**，退回「融合后的原始顺序」。
    ///
    /// 理由：重排是**精排**环节，它的作用是「把对的排到前面」。
    /// 它挂了，结果**依然可用**（只是顺序差一点）。
    ///
    /// 这和生成失败完全不同 —— 生成挂了就没有答案了。
    async fn rerank(
        &self,
        query: &str,
        fused_hits: &[HybridHit],
    ) -> (Vec<HybridHit>, bool, Option<String>) {
        let Some(reranker) = &self.reranker else {
            // 没有重排器 —— 直接截断。
            return (
                fused_hits
                    .iter()
                    .take(self.rerank_config.top_n)
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
            .rerank(query, &documents, self.rerank_config.top_n)
            .await
        {
            Ok(reranked) => {
                // 把重排结果映射回融合结果（补上两路的分数和共识度）。
                let by_id: std::collections::HashMap<&str, &HybridHit> =
                    fused_hits.iter().map(|h| (h.id.as_str(), h)).collect();

                let hits: Vec<HybridHit> = reranked
                    .into_iter()
                    .filter_map(|r| {
                        let original = by_id.get(r.id.as_str())?;
                        let mut hit = (*original).clone();
                        hit.rank = r.rank;
                        hit.text = r.text.clone();
                        Some(hit)
                    })
                    .collect();

                (hits, true, None)
            }
            Err(e) => {
                // 降级：退回融合后的原始顺序。
                warn!(error = %e, "重排失败，降级为融合后的原始顺序");
                (
                    fused_hits
                        .iter()
                        .take(self.rerank_config.top_n)
                        .cloned()
                        .collect(),
                    false,
                    Some(format!("重排失败（{e}），降级为融合后的原始顺序")),
                )
            }
        }
    }

    /// 完整问答：混合检索 → 重排 → 生成。
    pub async fn ask(&self, query: &str) -> RagResult<AskOutcome> {
        self.ask_with(query, self.retrieval_config.mode).await
    }

    /// 完整问答（指定检索模式）。
    pub async fn ask_with(&self, query: &str, mode: SearchMode) -> RagResult<AskOutcome> {
        let start = Instant::now();

        let retrieval = self.retrieve_with(query, mode).await?;

        if retrieval.hits.is_empty() {
            warn!(query = %query, "检索没有召回到任何内容");
            return Err(RagError::NotReady("检索没有召回到任何内容".to_string()));
        }

        // 组 Prompt —— 复用 06-rag 的组装逻辑（格式一致，便于对比）。
        let prompt = rag_06::prompt::assemble_no_context(query);
        let user_prompt = build_user_prompt(&retrieval, query);

        let answer = self
            .llm
            .chat(&prompt.system, &user_prompt)
            .await
            .map_err(|e| {
                warn!(error = %e, "生成失败");
                RagError::Generation(e.to_string())
            })?;

        let citations = extract_citations(&answer, &retrieval.hits);

        info!(
            query = %query,
            mode = mode.name(),
            hits = retrieval.hits.len(),
            citations = citations.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "问答完成"
        );

        Ok(AskOutcome {
            answer,
            citations,
            retrieval,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }
}

/// 一次问答的完整结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskOutcome {
    /// 最终答案。
    pub answer: String,

    /// 引用到的块。
    pub citations: Vec<Citation>,

    /// 检索过程（融合 + 重排结果）。
    pub retrieval: RetrievalOutcome,

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

/// 组装 User 提示词：上下文 + 问题。
fn build_user_prompt(retrieval: &RetrievalOutcome, query: &str) -> String {
    let context = retrieval.to_context();

    let context_section = if context.is_empty() {
        "（本次检索没有召回任何资料片段）".to_string()
    } else {
        context
    };

    format!("【参考资料】\n{context_section}\n\n【问题】\n{query}")
}

/// 从答案里抽取引用编号。
///
/// 只接受落在实际块数范围内的编号 —— 模型编造的 `[9]` 要丢掉
/// （留着等于给用户一个假引用入口）。
fn extract_citations(answer: &str, hits: &[HybridHit]) -> Vec<Citation> {
    let mut citations: Vec<Citation> = Vec::new();
    let mut seen: Vec<usize> = Vec::new();

    for group in find_bracket_groups(answer) {
        for number in group {
            if number == 0 || number > hits.len() {
                continue;
            }
            if seen.contains(&number) {
                continue;
            }
            seen.push(number);

            let hit = &hits[number - 1];
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

/// 便利函数：从 `AppConfig` 构造流水线。
pub fn make_retriever(
    store: rag_05::store::FaissStore,
    bm25: crate::bm25::Bm25Index,
    embedder: Box<dyn rag_05::embed::Embedder>,
    config: &AppConfig,
) -> HybridRetriever {
    HybridRetriever::new(
        store,
        bm25,
        embedder,
        config.retrieval.effective_per_path_k(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, score: f32) -> HybridHit {
        HybridHit {
            rank: 0,
            id: id.to_string(),
            score,
            vector_score: Some(score),
            bm25_score: None,
            votes: 1,
            source: "doc.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: "正文".to_string(),
        }
    }

    #[test]
    fn extracts_valid_citations_only() {
        let hits = vec![hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)];
        let cites = extract_citations("见 [1] 和 [3]，还有 [9]。", &hits);

        assert_eq!(cites.len(), 2);
        assert_eq!(cites[0].index, 1);
        assert_eq!(cites[1].index, 3);
    }

    #[test]
    fn context_uses_bracket_numbers() {
        let hits = vec![hit("a", 0.9)];
        let outcome = RetrievalOutcome {
            query: "q".to_string(),
            mode: SearchMode::Hybrid,
            fused_hits: hits.clone(),
            hits,
            reranked: false,
            degrade_reason: None,
            retrieve_ms: 0,
            rerank_ms: 0,
            elapsed_ms: 0,
        };

        let ctx = outcome.to_context();
        assert!(ctx.contains("[1]"));
        assert!(ctx.contains("# 标题"));
    }
}
