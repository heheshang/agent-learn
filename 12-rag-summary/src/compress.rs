//! 上下文压缩（Context Compression）—— 把召回的块压进 token 预算。
//!
//! ## 为什么「摘要型 RAG」特别需要压缩
//!
//! 普通 RAG 的任务是「问答」—— 输出和输入差不多长。
//! 摘要型 RAG 的任务是「把一堆片段压成一段话」—— **输出远短于输入**。
//!
//! 这带来一个反直觉的问题：**上下文越长，效果可能越差**。
//!
//! 1. **稀释** —— 50 个块里只有 3 个相关，其余 47 个是噪声。
//!    「中间迷失」（lost in the middle）现象让模型更容易忽略中间内容。
//! 2. **干扰** —— 不相关的片段会诱导模型去引用它们（幻觉）。
//! 3. **成本** —— 上下文 token 直接等于钱和延迟。
//!
//! ## 为什么用「抽取式」而不是「生成式」
//!
//! | | 抽取式（本项目） | 生成式 |
//! |---|---|---|
//! | 做法 | 挑选 + 截断**原文** | 让 LLM 总结 |
//! | 可溯源 | ✅ 原句还在 | ❌ 概括后无法回溯 |
//! | 新失败点 | 无 | 引入 LLM 调用（慢、贵、会挂） |
//! | 信息损失 | 只丢「不相关的部分」 | 可能丢「相关的部分」 |
//!
//! 生成式压缩听起来更「智能」，但它**用一个可靠性问题换另一个**：
//! 你本来担心「上下文太长」，现在要担心「压缩时把关键信息压掉了」。
//! 而且压缩本身要用 LLM —— 一次额外的网络往返。
//!
//! 所以本项目用**抽取式**：只做「挑选 + 按句子边界截断」，
//! 保证每一句进入上下文的文字都**逐字来自原始语料**。
//!
//! ## 压缩算法
//!
//! ```text
//! 剩余预算 = budget_tokens
//! for 每块（按相关度降序）:
//!     块成本 = token(标题 + 正文)
//!     if 剩余预算 ≥ 块成本:
//!         整块保留
//!     else if 剩余预算 ≥ min_tokens:
//!         按句子边界截断到剩余预算
//!     else:
//!         停止（预算耗尽）
//! ```
//!
//! ## ⚠️ 引用编号必须保持稳定
//!
//! 压缩之后，块的**数量**会变（少了几块）—— 但引用编号 `[1]`、`[2]`
//! 必须和实际进入上下文的块一一对应。
//!
//! 所以压缩时**重新编号**，而不是保留原始 rank。
//! 否则模型会引用一个「已经被砍掉的块」。

use rag_04::splitter::TokenCounter;

use crate::config::CompressConfig;
use crate::error::{RagError, RagResult};
use crate::retriever::Hit;

/// 压缩结果。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompressedContext {
    /// 压缩后的上下文片段（按相关度降序）。
    pub passages: Vec<Hit>,

    /// 渲染好的上下文文本（`[n] # 标题\n正文` 格式）。
    pub text: String,

    /// 压缩前后的 token 数。
    pub tokens_before: usize,

    /// 压缩后的 token 数。
    pub tokens_after: usize,

    /// 被砍掉的块数。
    pub dropped: usize,

    /// 是否被截断（块被砍了一半）。
    pub truncated: bool,
}

impl CompressedContext {
    /// 上下文文本。
    pub fn to_context(&self) -> String {
        self.text.clone()
    }

    /// 压缩率（`0.0` = 没压，`1.0` = 压到 0）。
    pub fn compression_ratio(&self) -> f32 {
        if self.tokens_before == 0 {
            return 0.0;
        }
        1.0 - (self.tokens_after as f32 / self.tokens_before as f32)
    }

    /// 一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "{} → {} token ({}%) | 保留 {} 块，砍掉 {} 块{}",
            self.tokens_before,
            self.tokens_after,
            (self.compression_ratio() * 100.0).round() as i32,
            self.passages.len(),
            self.dropped,
            if self.truncated { "，有截断" } else { "" }
        )
    }
}

/// 上下文压缩器。
pub struct ContextCompressor {
    counter: TokenCounter,
    config: CompressConfig,
}

impl std::fmt::Debug for ContextCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextCompressor")
            .field("budget_tokens", &self.config.budget_tokens)
            .field("min_tokens", &self.config.min_tokens)
            .finish()
    }
}

impl ContextCompressor {
    /// 创建压缩器。
    pub fn new(config: CompressConfig) -> RagResult<Self> {
        let counter = TokenCounter::new()
            .map_err(|e| RagError::Internal(format!("创建 token 计数器失败: {e}")))?;

        Ok(Self { counter, config })
    }

    /// 当前配置。
    pub fn config(&self) -> &CompressConfig {
        &self.config
    }

    /// 压缩上下文。
    ///
    /// `hits` 必须**按相关度降序**排列 —— 压缩优先保留更相关的块。
    pub fn compress(&self, hits: &[Hit]) -> CompressedContext {
        let tokens_before: usize = hits.iter().map(|h| self.count_hit(h)).sum();

        if !self.config.enabled || hits.is_empty() {
            let passages = hits.to_vec();
            return CompressedContext {
                text: render_context(&passages),
                tokens_before,
                tokens_after: tokens_before,
                passages,
                dropped: 0,
                truncated: false,
            };
        }

        let mut remaining = self.config.budget_tokens;
        let mut passages: Vec<Hit> = Vec::new();
        let mut dropped = 0usize;
        let mut truncated = false;

        for hit in hits {
            if remaining == 0 {
                dropped += 1;
                continue;
            }

            let cost = self.count_hit(hit);

            if cost <= remaining {
                // 整块放得下 —— 完整保留。
                remaining -= cost;
                passages.push(hit.clone());
                continue;
            }

            // 放不下整块 —— 尝试按句子边界截断。
            //
            // ⚠️ 只有当剩余预算 ≥ `min_tokens` 时才截断 ——
            // 否则塞进去的是半句话，价值为负还占位置。
            if remaining >= self.config.min_tokens {
                let truncated_text = self.truncate_to_budget(hit, remaining);
                if !truncated_text.is_empty() {
                    let mut partial = hit.clone();
                    partial.text = truncated_text;
                    let used = self.count_hit(&partial);
                    remaining -= used.min(remaining);
                    passages.push(partial);
                    truncated = true;
                    continue;
                }
            }

            // 预算不够放一个最小片段 —— 这个块直接砍掉。
            dropped += 1;
        }

        // ⚠️ 重新编号 —— 引用编号必须和实际进入上下文的块对应。
        for (index, hit) in passages.iter_mut().enumerate() {
            hit.rank = index;
        }

        let tokens_after: usize = passages.iter().map(|h| self.count_hit(h)).sum();

        CompressedContext {
            text: render_context(&passages),
            tokens_before,
            tokens_after,
            passages,
            dropped,
            truncated,
        }
    }

    /// 计算一个块进入上下文的 token 成本（标题 + 正文 + 引用标记）。
    fn count_hit(&self, hit: &Hit) -> usize {
        let head = if hit.heading_path.is_empty() {
            String::new()
        } else {
            format!("# {}\n", hit.heading_path.join(" > "))
        };
        let marker = format!("[{}] ", hit.rank + 1);
        self.counter.count(&format!("{marker}{head}{}", hit.text))
    }

    /// 把一个块按句子边界截断到给定 token 预算。
    fn truncate_to_budget(&self, hit: &Hit, budget: usize) -> String {
        let head = if hit.heading_path.is_empty() {
            String::new()
        } else {
            format!("# {}\n", hit.heading_path.join(" > "))
        };
        // 标题和引用标记也要占预算。
        let overhead = self.counter.count(&format!("[{}] {head}", hit.rank + 1));
        let body_budget = budget.saturating_sub(overhead);

        if body_budget == 0 {
            return String::new();
        }

        // 按句子边界切分。
        //
        // ⚠️ 为什么按句子而不是按字符？
        // 因为截断到句子中间会产生「半句话」—— 语义不完整，
        // 对模型是纯噪声。宁可少要一点，也要句子完整。
        let sentences = split_sentences(&hit.text);

        let mut kept = String::new();
        for sentence in sentences {
            let candidate = if kept.is_empty() {
                sentence.clone()
            } else {
                format!("{kept}{sentence}")
            };

            if self.counter.count(&candidate) > body_budget {
                break;
            }

            kept = candidate;
        }

        // 一个完整句子都放不下 —— 放弃截断（返回空）。
        if kept.trim().is_empty() {
            return String::new();
        }

        // 标记截断 —— 让模型知道这是片段，不是完整段落。
        //
        // ⚠️ 这个省略号很重要：不加的话，模型可能把片段当成
        // 完整论述来引用，产生「引用了不存在的内容」的幻觉。
        format!("{kept}…")
    }
}

/// 把文本按句子边界切分。
///
/// 支持中英文标点，以及换行（段落边界）。
///
/// ## 为什么保留标点
///
/// 句子切分后要重新拼回去，所以**切点处的标点要跟着前一句** ——
/// 否则拼出来的文本会缺标点，读起来像被人为截断过。
fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut sentences = Vec::new();
    let mut current = String::new();

    for (index, ch) in chars.iter().enumerate() {
        current.push(*ch);

        let is_terminator = matches!(ch, '。' | '！' | '？' | '；' | '\n' | '.' | '!' | '?');

        // 英文句号后面如果是数字（小数点），不切。
        //
        // ⚠️ 判断依据是「小数点**前**是数字且**后**也是数字」——
        // 只看后一位不够（"1. 版本" 里的点后面是空格）。
        let is_decimal_point = *ch == '.'
            && index > 0
            && chars[index - 1].is_ascii_digit()
            && chars.get(index + 1).is_some_and(|c| c.is_ascii_digit());

        if is_terminator && !is_decimal_point {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// 把若干块渲染成上下文文本。
pub fn render_context(hits: &[Hit]) -> String {
    hits.iter()
        .map(Hit::to_context)
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retriever::{Hit, RetrievalPath};

    fn hit(text: &str) -> Hit {
        Hit {
            rank: 0,
            id: "doc.md#0".to_string(),
            score: 0.9,
            dense_score: Some(0.9),
            bm25_score: None,
            votes: 1,
            source: "doc.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: text.to_string(),
            path: RetrievalPath::ServerHybrid,
        }
    }

    fn compressor(budget: usize, min: usize) -> ContextCompressor {
        ContextCompressor::new(CompressConfig {
            budget_tokens: budget,
            min_tokens: min,
            enabled: true,
        })
        .unwrap()
    }

    #[test]
    fn keeps_everything_when_under_budget() {
        let c = compressor(10_000, 50);
        let hits = vec![hit("第一段内容。"), hit("第二段内容。")];

        let result = c.compress(&hits);

        assert_eq!(result.passages.len(), 2);
        assert_eq!(result.dropped, 0);
        assert_eq!(result.tokens_after, result.tokens_before);
    }

    #[test]
    fn drops_blocks_when_budget_is_exhausted() {
        let c = compressor(20, 50);
        let hits = vec![
            hit("第一段内容很长很长很长。"),
            hit("第二段内容很长很长很长。"),
        ];

        let result = c.compress(&hits);

        assert!(result.dropped > 0, "超预算的块应该被砍掉");
        assert!(result.tokens_after <= result.tokens_before);
    }

    #[test]
    fn disabled_compressor_is_a_passthrough() {
        let c = ContextCompressor::new(CompressConfig {
            budget_tokens: 10,
            min_tokens: 5,
            enabled: false,
        })
        .unwrap();

        let hits = vec![hit("很长的内容。"), hit("也很长。")];
        let result = c.compress(&hits);

        assert_eq!(result.passages.len(), 2);
        assert_eq!(result.dropped, 0);
    }

    #[test]
    fn renumbers_citations_after_compression() {
        let c = compressor(10_000, 50);
        let mut hits = vec![hit("第一段。"), hit("第二段。"), hit("第三段。")];
        // 模拟重排后的原始排名。
        for (i, h) in hits.iter_mut().enumerate() {
            h.rank = i;
        }

        let result = c.compress(&hits);

        for (i, p) in result.passages.iter().enumerate() {
            assert_eq!(p.rank, i, "引用编号必须连续且从 0 开始");
        }
        assert!(result.text.contains("[1]"));
        assert!(result.text.contains("[3]"));
    }

    #[test]
    fn truncated_passage_is_marked_with_ellipsis() {
        // 预算很小 → 触发截断。
        let c = compressor(40, 10);
        let long = "这是第一句话。这是第二句话。这是第三句话。这是第四句话。这是第五句话。";
        let result = c.compress(&[hit(long)]);

        if result.truncated {
            assert!(
                result.text.contains('…'),
                "被截断的片段必须标记出来，否则模型会当成完整论述"
            );
        }
    }

    #[test]
    fn splits_sentences_on_chinese_punctuation() {
        let sentences = split_sentences("第一句。第二句！第三句？");
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].ends_with('。'));
    }

    #[test]
    fn splits_sentences_on_english_punctuation() {
        let sentences = split_sentences("First one. Second one! Third?");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn keeps_decimal_points_intact() {
        let sentences = split_sentences("版本是 1.5 的模型。");
        assert_eq!(sentences.len(), 1);
        assert!(sentences[0].contains("1.5"));
    }

    #[test]
    fn splits_on_newlines() {
        let sentences = split_sentences("第一行\n第二行");
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn empty_input_produces_empty_sentences() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   \n  ").is_empty());
    }

    #[test]
    fn compression_ratio_is_reported() {
        let c = compressor(10_000, 50);
        let result = c.compress(&[hit("内容。")]);

        assert_eq!(result.compression_ratio(), 0.0);
        assert!(result.summary().contains("token"));
    }

    #[test]
    fn empty_hits_produce_empty_context() {
        let c = compressor(100, 10);
        let result = c.compress(&[]);

        assert!(result.passages.is_empty());
        assert!(result.text.is_empty());
        assert_eq!(result.tokens_before, 0);
    }
}
