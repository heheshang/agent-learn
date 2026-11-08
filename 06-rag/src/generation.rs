//! 生成层（Generation）—— 把上下文 + 问题变成最终答案。
//!
//! ```text
//! Prompt ──LLM──> 答案文本
//! ```
//!
//! ## 这是 Naive RAG 的最后一步，也是最「薄」的一步
//!
//! 朴素版生成只做三件事：
//!
//! 1. 把组好的 Prompt 发给 LLM
//! 2. 拿到答案文本
//! 3. 把答案和「引用了哪些块」一起返回
//!
//! 没有：答案校验（groundedness check）、自我批判（critique）、
//! 答案重排序、多候选对比、引用抽取（把答案里的 `[1]` 映射回具体块）。
//!
//! ## 引用是怎么「落地」的
//!
//! 模型在答案里写 `[1]`，但**模型并不知道** `[1]` 到底是哪个块 ——
//! 是我们在 Prompt 里用 `[1] # 标题\n正文` 的形式告诉它的。
//!
//! 所以要把「可溯源」真正做实，需要两步：
//!
//! 1. Prompt 里给每个块编号（本项目已做）
//! 2. 解析答案里的 `[n]`，映射回具体的块和原文位置（本模块已做）
//!
//! 只有第 1 步，答案是「看起来带引用」；两步都做，引用才能点击、核对。

use agent_logging::{debug, info, warn};
use genai::Client;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest};
use serde::{Deserialize, Serialize};

use crate::config::GenerationConfig;
use crate::error::{RagError, RagResult};
use crate::prompt::AssembledPrompt;
use crate::retrieval::RetrievalResult;

/// 生成结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// 模型生成的答案文本。
    pub answer: String,

    /// 答案里引用到的块（按引用编号去重）。
    ///
    /// 空数组表示模型没有标注引用 —— 这本身是个信号：
    /// 说明模型可能是凭先验知识答的，而不是基于资料。
    pub citations: Vec<Citation>,

    /// 消耗的 token 数（prompt / completion / total）。
    pub usage: TokenUsage,

    /// 使用的模型名。
    pub model: String,
}

/// 一条引用。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// 引用编号（对应 Prompt 里的 `[n]`，从 1 开始）。
    pub index: usize,

    /// 被引用的块 ID。
    pub chunk_id: String,

    /// 来源文件名。
    pub source: String,

    /// 标题路径。
    pub heading_path: Vec<String>,
}

/// token 用量。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// 输入 token 数。
    pub prompt_tokens: u32,
    /// 输出 token 数。
    pub completion_tokens: u32,
    /// 总 token 数。
    pub total_tokens: u32,
}

/// 生成器。
///
/// 持有一个 `genai::Client`（克隆开销很小 —— 内部是 `Arc`）。
pub struct Generator {
    client: Client,
    config: GenerationConfig,
}

impl std::fmt::Debug for Generator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generator")
            .field("config", &self.config)
            .finish()
    }
}

impl Generator {
    /// 创建生成器。
    pub fn new(config: GenerationConfig) -> Self {
        Self {
            client: Client::builder().build(),
            config,
        }
    }

    /// 当前配置。
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }

    /// 底层 client（Agent 层要复用它做工具调用）。
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// 生成答案。
    ///
    /// 只做「发请求 → 取文本」，引用解析在 [`generate_with_citations`] 里做。
    pub async fn generate(&self, prompt: &AssembledPrompt) -> RagResult<String> {
        let options = ChatOptions::default()
            .with_temperature(self.config.temperature)
            .with_max_tokens(self.config.max_tokens);

        let request = ChatRequest::new(vec![
            ChatMessage::system(&prompt.system),
            ChatMessage::user(&prompt.user),
        ]);

        debug!(
            model = %self.config.model,
            system_len = prompt.system.chars().count(),
            user_len = prompt.user.chars().count(),
            "发起模型生成请求"
        );

        let response = self
            .client
            .exec_chat(&self.config.model, request, Some(&options))
            .await
            .map_err(|e| {
                warn!(model = %self.config.model, error = %e, "模型调用失败");
                RagError::Generation(format!("调用模型失败: {e}"))
            })?;

        Ok(response.into_first_text().unwrap_or_default())
    }

    /// 生成答案，并把答案里的 `[n]` 映射回具体的块。
    ///
    /// 这是「引用可溯源」的落地步骤 —— 见模块文档。
    pub async fn generate_with_citations(
        &self,
        prompt: &AssembledPrompt,
        retrieval: &RetrievalResult,
    ) -> RagResult<GenerationResult> {
        let request = ChatRequest::new(vec![
            ChatMessage::system(&prompt.system),
            ChatMessage::user(&prompt.user),
        ]);

        let options = ChatOptions::default()
            .with_temperature(self.config.temperature)
            .with_max_tokens(self.config.max_tokens);

        debug!(
            model = %self.config.model,
            chunk_count = retrieval.chunks.len(),
            "发起带引用的生成请求"
        );

        let response = self
            .client
            .exec_chat(&self.config.model, request, Some(&options))
            .await
            .map_err(|e| {
                warn!(model = %self.config.model, error = %e, "模型调用失败");
                RagError::Generation(format!("调用模型失败: {e}"))
            })?;

        let answer = response.first_text().unwrap_or_default().to_string();
        let citations = extract_citations(&answer, retrieval);

        info!(
            model = %self.config.model,
            answer_chars = answer.chars().count(),
            citation_count = citations.len(),
            prompt_tokens = response.usage.prompt_tokens.unwrap_or(0),
            completion_tokens = response.usage.completion_tokens.unwrap_or(0),
            "生成完成"
        );

        // 没有引用是个**信号** —— 说明模型可能凭先验知识答的，
        // 而不是基于检索到的资料。值得单独 warn 一行。
        if citations.is_empty() {
            warn!("答案里没有任何引用标记 —— 模型可能没有依据检索资料作答");
        }

        Ok(GenerationResult {
            answer,
            citations,
            usage: TokenUsage {
                prompt_tokens: response.usage.prompt_tokens.unwrap_or(0) as u32,
                completion_tokens: response.usage.completion_tokens.unwrap_or(0) as u32,
                total_tokens: response.usage.total_tokens.unwrap_or(0) as u32,
            },
            model: self.config.model.clone(),
        })
    }
}

/// 从答案文本里抽取引用编号。
///
/// 匹配 `[1]`、`[2,3]`、`[1, 2]` 这类形式 —— 方括号里是逗号分隔的数字。
///
/// ⚠️ 只接受 1..=chunks.len() 范围内的编号。模型经常会编造不存在的编号
/// （比如引用了 `[9]` 但只给了 3 个块），那些要丢掉 ——
/// 留着等于给用户一个假的引用入口。
pub fn extract_citations(answer: &str, retrieval: &RetrievalResult) -> Vec<Citation> {
    let mut citations: Vec<Citation> = Vec::new();
    let mut seen: Vec<usize> = Vec::new();

    for group in find_bracket_groups(answer) {
        for number in group {
            // 编号从 1 开始；0 和超界的都丢弃。
            if number == 0 || number > retrieval.chunks.len() {
                continue;
            }
            if seen.contains(&number) {
                continue;
            }
            seen.push(number);

            let chunk = &retrieval.chunks[number - 1];
            citations.push(Citation {
                index: number,
                chunk_id: chunk.id.clone(),
                source: chunk.source.clone(),
                heading_path: chunk.heading_path.clone(),
            });
        }
    }

    citations.sort_by_key(|c| c.index);
    citations
}

/// 找出所有 `[...]` 里的数字组。
///
/// 返回形如 `[[1], [2, 3]]` —— 外层是每个方括号，内层是该括号里的数字。
fn find_bracket_groups(answer: &str) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let bytes = answer.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'[' {
            // 找配对的 `]`。限制在 64 字符内，避免吃到无关的方括号。
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

                // 至少要有一个合法数字，才当成引用组。
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::RetrievedChunk;

    fn retrieval(n: usize) -> RetrievalResult {
        RetrievalResult {
            query: "q".to_string(),
            chunks: (0..n)
                .map(|i| RetrievedChunk {
                    rank: i,
                    id: format!("doc.md#{i}"),
                    score: 0.9,
                    source: "doc.md".to_string(),
                    heading_path: vec!["章节".to_string()],
                    text: "正文".to_string(),
                    heading_context: "# 章节".to_string(),
                })
                .collect(),
            retrieved_count: n,
            top_score: Some(0.9),
        }
    }

    #[test]
    fn extracts_single_citation() {
        let r = retrieval(3);
        let cites = extract_citations("RAG 是 [1] 说的。", &r);
        assert_eq!(cites.len(), 1);
        assert_eq!(cites[0].index, 1);
        assert_eq!(cites[0].chunk_id, "doc.md#0");
    }

    #[test]
    fn extracts_multiple_citations() {
        let r = retrieval(3);
        let cites = extract_citations("见 [1] 和 [3]。", &r);
        assert_eq!(cites.len(), 2);
        assert_eq!(cites[0].index, 1);
        assert_eq!(cites[1].index, 3);
    }

    #[test]
    fn handles_comma_separated_group() {
        let r = retrieval(3);
        let cites = extract_citations("参考 [1, 3]。", &r);
        assert_eq!(cites.len(), 2);
    }

    #[test]
    fn drops_out_of_range_numbers() {
        let r = retrieval(2);
        // 9 超过了块数，必须丢掉 —— 留着就是假引用。
        let cites = extract_citations("见 [1] 和 [9]。", &r);
        assert_eq!(cites.len(), 1);
        assert_eq!(cites[0].index, 1);
    }

    #[test]
    fn deduplicates_repeated_citations() {
        let r = retrieval(2);
        let cites = extract_citations("[1] 提到 [1] 再次提到 [1]。", &r);
        assert_eq!(cites.len(), 1);
    }

    #[test]
    fn answer_without_citations_yields_empty() {
        let r = retrieval(2);
        let cites = extract_citations("没有任何引用标记。", &r);
        assert!(cites.is_empty());
    }

    #[test]
    fn ignores_non_numeric_brackets() {
        let r = retrieval(2);
        let cites = extract_citations("见 [注] 和 [1]。", &r);
        assert_eq!(cites.len(), 1);
    }
}
