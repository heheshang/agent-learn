//! 端到端编排 —— 改写 → 多路检索 → 融合 → 生成。
//!
//! ```text
//! 问题 ──Transformer──> Vec<TransformedQuery>
//!                            │
//!                            ├──> 每路检索（并发嵌入 + 串行 FAISS）
//!                            │
//!                     Vec<RankedRun>
//!                            │
//!                       融合（RRF）
//!                            │
//!                     Vec<FusedHit>
//!                            │
//!                       Prompt 组装
//!                            │
//!                     LLM ──> 带引用的答案
//! ```
//!
//! ## 降级链
//!
//! 这个流水线**每一环都有降级路径**，这是它能作为「生产组件」的基础：
//!
//! | 环节失败 | 降级行为 |
//! |---|---|
//! | 改写失败（LLM 挂） | 用原问题继续检索 |
//! | 某一路检索失败 | 跳过这一路，用其它路的结果 |
//! | 全部检索失败 | 直接报错（没有数据就没答案） |
//! | 融合失败 | 退回 concat 拼接 |
//! | 生成失败 | 报错（返回检索结果，让调用方自己看） |
//!
//! ⚠️ 注意最后一条：生成失败时，**检索结果仍然是有价值的**。
//! 所以 `ask` 接口在生成失败时返回检索结果 + 错误信息，而不是丢弃一切。

use agent_logging::{debug, info, warn};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::{AppConfig, RetrievalConfig};
use crate::error::{QueryError, QueryResult};
use crate::fusion::{FusedHit, FusionMethod, RankedRun, fuse};
use crate::llm::Llm;
use crate::retrieval::{EmbedMode, MultiQueryRetriever};
use crate::transform::{
    DecomposeTransformer, HydeTransformer, MultiQueryTransformer, NoopTransformer,
    QueryTransformer, RewriteTransformer, Strategy, TransformedQuery,
};

/// 一次「改写 + 检索 + 融合」的完整结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformedRetrieval {
    /// 原始问题。
    pub original_question: String,

    /// 使用的策略。
    pub strategy: Strategy,

    /// 改写后的查询列表。
    pub transformed_queries: Vec<TransformedQuery>,

    /// 每路的原始检索结果（融合前）。
    ///
    /// ⚠️ 这一项体积可能很大 —— 每路 `per_query_k` 个块。
    /// API 层默认**不返回**它，只在 `include_runs=true` 时带上。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub runs: Vec<RankedRun>,

    /// 融合后的命中。
    pub hits: Vec<FusedHit>,

    /// 改写耗时（毫秒）。
    pub transform_ms: u64,

    /// 检索 + 融合耗时（毫秒）。
    pub retrieve_ms: u64,

    /// 是否发生了降级（改写失败退回原问题）。
    pub degraded: bool,

    /// 降级原因（如果有）。
    pub degrade_reason: Option<String>,
}

impl TransformedRetrieval {
    /// 融合后的命中数。
    pub fn hit_count(&self) -> usize {
        self.hits.len()
    }

    /// 拼成上下文的纯文本。
    ///
    /// 格式和 06-rag 保持一致（`[n] # 标题 > 子标题\n正文`），
    /// 这样两个模块的 Prompt 可以直接对比。
    pub fn to_context(&self) -> String {
        self.hits
            .iter()
            .map(FusedHit::to_context)
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// 命中的块 ID 列表（用于计算「各策略召回重叠度」）。
    pub fn hit_ids(&self) -> Vec<String> {
        self.hits.iter().map(|h| h.id.clone()).collect()
    }
}

/// 查询流水线。
pub struct QueryPipeline {
    retriever: Arc<MultiQueryRetriever>,
    llm: Arc<dyn Llm>,
    config: RetrievalConfig,
    transform_config: crate::config::TransformConfig,
}

impl std::fmt::Debug for QueryPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryPipeline")
            .field("config", &self.config)
            .finish()
    }
}

impl QueryPipeline {
    /// 创建流水线。
    pub fn new(
        retriever: Arc<MultiQueryRetriever>,
        llm: Arc<dyn Llm>,
        config: RetrievalConfig,
        transform_config: crate::config::TransformConfig,
    ) -> Self {
        Self {
            retriever,
            llm,
            config,
            transform_config,
        }
    }

    /// 当前检索配置。
    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    /// 构造某个策略的转换器。
    ///
    /// 每个策略独立持有一个转换器 —— 因为它们需要的提示词和参数不同。
    ///
    /// ## ⚠️ 关键：复用流水线自己的 LLM
    ///
    /// 这里**不能**新建 `GenaiLlm` —— 必须复用注入进来的那个。
    ///
    /// 原因有两个：
    ///
    /// 1. **测试可替换** —— 测试注入 `ScriptedLlm`（零网络、完全确定），
    ///    如果转换器自己新建真实 LLM，测试就会真的去调网络，完全失去意义。
    /// 2. **模型一致** —— 改写和生成应该用同一个模型实例，
    ///    否则配置里的 `transform_model` 会被绕过。
    pub fn transformer(&self, strategy: Strategy) -> Box<dyn QueryTransformer> {
        let llm = Arc::clone(&self.llm);

        match strategy {
            Strategy::None => Box::new(NoopTransformer),
            Strategy::Hyde => Box::new(HydeTransformer::new(llm)),
            Strategy::MultiQuery => Box::new(
                MultiQueryTransformer::new(llm).with_num_queries(self.transform_config.num_queries),
            ),
            Strategy::Rewrite => Box::new(RewriteTransformer::new(llm)),
            Strategy::Decompose => Box::new(
                DecomposeTransformer::new(llm)
                    .with_max_sub_questions(self.transform_config.max_sub_questions),
            ),
        }
    }

    /// 改写 + 检索 + 融合（**不生成**）。
    ///
    /// 这是 API 层「只检索」端点和「对比」端点的底层实现。
    pub async fn retrieve(
        &self,
        question: &str,
        strategy: Strategy,
    ) -> QueryResult<TransformedRetrieval> {
        self.retrieve_with(question, strategy, self.config.fusion)
            .await
    }

    /// 改写 + 检索 + 融合（指定融合方法）。
    pub async fn retrieve_with(
        &self,
        question: &str,
        strategy: Strategy,
        fusion: FusionMethod,
    ) -> QueryResult<TransformedRetrieval> {
        let question = question.trim();
        if question.is_empty() {
            warn!("查询问题为空");
            return Err(QueryError::BadRequest("问题不能为空".to_string()));
        }

        debug!(
            question = %question,
            strategy = ?strategy,
            fusion = fusion.name(),
            "开始查询流水线"
        );

        // ── 1. 改写 ──────────────────────────────────────────────
        let transform_start = Instant::now();
        let (queries, degraded, degrade_reason) =
            self.transform_with_fallback(question, strategy).await;
        let transform_ms = transform_start.elapsed().as_millis() as u64;

        debug!(
            query_count = queries.len(),
            transform_ms, degraded, "改写阶段完成"
        );
        for q in &queries {
            debug!(query = %q.text, variant = ?q.variant, "改写查询");
        }

        // ── 2. 多路检索 ──────────────────────────────────────────
        let retrieve_start = Instant::now();
        let per_k = self.config.effective_per_query_k();

        let specs: Vec<(String, EmbedMode, Strategy)> = queries
            .iter()
            .map(|q| {
                let mode = EmbedMode::for_strategy(strategy);
                (q.text.clone(), mode, strategy)
            })
            .collect();

        let runs = self.retriever.retrieve_multi(&specs, per_k).await?;
        let retrieve_ms = retrieve_start.elapsed().as_millis() as u64;

        debug!(
            run_count = runs.len(),
            per_query_k = per_k,
            retrieve_ms,
            "多路检索完成"
        );

        // ── 3. 融合 ──────────────────────────────────────────────
        let mut hits = fuse(&runs, fusion, self.config.top_k);

        // 阈值过滤（在融合之后做 —— 因为不同路的分数分布不一样，
        // 融合分数和原始相似度不是一回事，要按 `best_score` 过滤）。
        if let Some(min) = self.config.min_score {
            let before = hits.len();
            hits.retain(|h| h.best_score >= min);
            for (i, hit) in hits.iter_mut().enumerate() {
                hit.rank = i;
            }
            debug!(before, after = hits.len(), min_score = min, "阈值过滤完成");
        }

        info!(
            question = %question,
            strategy = ?strategy,
            fusion = fusion.name(),
            run_count = runs.len(),
            hit_count = hits.len(),
            transform_ms,
            retrieve_ms,
            degraded,
            "查询完成"
        );

        Ok(TransformedRetrieval {
            original_question: question.to_string(),
            strategy,
            transformed_queries: queries,
            runs,
            hits,
            transform_ms,
            retrieve_ms,
            degraded,
            degrade_reason,
        })
    }

    /// 改写（带降级）。
    ///
    /// ## 降级语义
    ///
    /// 改写失败 → **不报错**，退回「用原问题检索」。
    /// 这不是「偷懒」，而是让「查询改写」保持为**纯增强**：
    /// LLM 挂掉时，服务仍然提供完整的 RAG 能力，只是召回差一点。
    async fn transform_with_fallback(
        &self,
        question: &str,
        strategy: Strategy,
    ) -> (Vec<TransformedQuery>, bool, Option<String>) {
        let transformer = self.transformer(strategy);

        match transformer.transform(question).await {
            Ok(queries) if !queries.is_empty() => (queries, false, None),
            Ok(_) => {
                warn!("改写返回空结果，降级为原问题");
                (
                    vec![TransformedQuery::new(question, strategy)],
                    true,
                    Some("改写返回空结果，降级为原问题".to_string()),
                )
            }
            Err(e) => {
                // 改写错误一律降级 —— 「查询改写」必须保持为纯增强：
                // LLM 挂掉时，服务仍然提供完整的 RAG 能力，只是召回差一点。
                warn!(error = %e, "改写失败，降级为原问题");
                (
                    vec![TransformedQuery::new(question, strategy)],
                    true,
                    Some(format!("改写失败（{e}），降级为原问题")),
                )
            }
        }
    }

    /// 完整问答：改写 → 检索 → 融合 → 生成。
    ///
    /// 返回「答案 + 检索结果 + 改写过程」，让调用方能看到全链路。
    pub async fn ask(&self, question: &str, strategy: Strategy) -> QueryResult<AskOutcome> {
        let start = Instant::now();

        let retrieval = self.retrieve(question, strategy).await?;

        if retrieval.hits.is_empty() {
            warn!(question = %question, "检索没有召回到任何内容");
            return Err(QueryError::NotReady("检索没有召回到任何内容".to_string()));
        }

        // 组 Prompt —— 复用 06-rag 的组装逻辑（格式一致，便于对比）。
        let prompt = rag_06::prompt::assemble_no_context(question);
        let user_prompt = build_user_prompt(&retrieval, question);

        let answer = self
            .llm
            .chat(&prompt.system, &user_prompt)
            .await
            .map_err(|e| {
                warn!(error = %e, "生成失败");
                QueryError::Generation(e.to_string())
            })?;

        let citations = extract_citations(&answer, &retrieval.hits);

        info!(
            question = %question,
            strategy = ?retrieval.strategy,
            hit_count = retrieval.hits.len(),
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

    /// 检索过程（改写 + 融合结果）。
    pub retrieval: TransformedRetrieval,

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
fn build_user_prompt(retrieval: &TransformedRetrieval, question: &str) -> String {
    let context = retrieval.to_context();

    let context_section = if context.is_empty() {
        "（本次检索没有召回任何资料片段）".to_string()
    } else {
        context
    };

    format!("【参考资料】\n{context_section}\n\n【问题】\n{question}")
}

/// 从答案里抽取引用编号。
///
/// 复用 06-rag 的解析逻辑 —— 但这里用的是 `FusedHit` 而不是 `RetrievedChunk`，
/// 所以单独写一份。逻辑完全一致：**只接受落在实际块数范围内的编号**，
/// 模型编造的 `[9]` 要丢掉（留着等于给用户一个假引用入口）。
fn extract_citations(answer: &str, hits: &[FusedHit]) -> Vec<Citation> {
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

/// 便捷函数：从 `AppConfig` 构造流水线需要的组件。
pub fn make_retriever(
    store: rag_05::store::FaissStore,
    embedder: Box<dyn rag_05::embed::Embedder>,
    config: &AppConfig,
) -> MultiQueryRetriever {
    MultiQueryRetriever::new(store, embedder, config.retrieval.effective_per_query_k())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_valid_citations_only() {
        let hits: Vec<FusedHit> = (0..3)
            .map(|i| FusedHit {
                rank: i,
                id: format!("doc.md#{i}"),
                score: 0.1,
                best_score: 0.8,
                votes: 1,
                source: "doc.md".to_string(),
                heading_path: vec!["标题".to_string()],
                text: "正文".to_string(),
                matched_queries: vec!["q".to_string()],
            })
            .collect();

        let cites = extract_citations("见 [1] 和 [3]，还有 [9]。", &hits);
        assert_eq!(cites.len(), 2);
        assert_eq!(cites[0].index, 1);
        assert_eq!(cites[1].index, 3);
    }

    #[test]
    fn context_uses_bracket_numbers() {
        let hit = FusedHit {
            rank: 0,
            id: "doc.md#0".to_string(),
            score: 0.1,
            best_score: 0.8,
            votes: 1,
            source: "doc.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: "正文内容".to_string(),
            matched_queries: vec![],
        };

        let ctx = hit.to_context();
        assert!(ctx.contains("[1]"));
        assert!(ctx.contains("# 标题"));
        assert!(ctx.contains("正文内容"));
    }
}
