//! Decompose 转换器 —— 把复合问题拆成多个独立子问题。
//!
//! ## 机制
//!
//! ```text
//! 复合问题 ──LLM──> 子问题1 / 子问题2 / 子问题3
//!                        │         │         │
//!                        └── 各自检索 ─┘
//!                                │
//!                             RRF 融合
//! ```
//!
//! ## 核心洞察：向量的「平均效应」
//!
//! 一个包含多个检索意图的句子，它的向量是**所有意图的加权平均**。
//!
//! 举例：
//!
//! ```text
//! 「光合作用的光反应和暗反应分别在什么位置进行？各自的作用是什么？」
//!                    ↑ 位置意图        ↑ 作用意图
//! ```
//!
//! 这个句子的向量落在「位置」和「作用」之间的中间地带 ——
//! 它离「光反应发生在类囊体膜上」不够近，也离「光反应合成 ATP」不够近。
//! 结果就是**两个意图都没召回好**。
//!
//! 拆开之后：
//!
//! ```text
//! 「光合作用的光反应阶段发生在什么位置？」  → 向量很尖，召回准
//! 「光合作用的光反应阶段的主要作用是什么？」→ 向量很尖，召回准
//! ```
//!
//! ## 与 Multi-Query 的本质区别
//!
//! 这两个策略**经常被混淆**，但它们解决的是不同的问题：
//!
//! | | Multi-Query | Decompose |
//! |---|---|---|
//! | 拆的是什么 | **措辞** | **意图** |
//! | 产出 | 同一问题的 N 种说法 | N 个不同的问题 |
//! | 解决的问题 | 单一意图召不全 | 多意图互相干扰 |
//! | 典型输入 | 「模型知识截止怎么办」 | 「A 和 B 的区别是什么？」 |
//!
//! 一个实用的判据：**如果 N 个变体之间可以互相替代，是 Multi-Query；
//! 如果 N 个变体之间不能互相替代（缺一个就不完整），是 Decompose。**
//!
//! ## 融合的额外考量
//!
//! Decompose 的子问题**覆盖不同方面**，所以它们召回的块可能完全不重叠。
//! 融合时用 RRF（Reciprocal Rank Fusion）比 MaxScore 更合适 ——
//! RRF 只看排名不看分数，天然适合「多路异构结果」的合并。
//! 详见 [`crate::fusion`]。

use crate::llm::Llm;
use agent_logging::{debug, warn};

use super::prompts::DECOMPOSE_PROMPT;
use super::{QueryTransformer, Strategy, TransformFuture, TransformedQuery, parse_numbered_lines};

/// Decompose 转换器。
pub struct DecomposeTransformer {
    llm: std::sync::Arc<dyn Llm>,
    prompt_template: String,
    /// 最多接受几个子问题。
    max_sub_questions: usize,
}

impl std::fmt::Debug for DecomposeTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecomposeTransformer")
            .field("max_sub_questions", &self.max_sub_questions)
            .finish()
    }
}

impl DecomposeTransformer {
    /// 创建转换器（用默认提示词，默认最多 4 个子问题）。
    pub fn new(llm: std::sync::Arc<dyn Llm>) -> Self {
        Self {
            llm,
            prompt_template: DECOMPOSE_PROMPT.to_string(),
            max_sub_questions: 4,
        }
    }

    /// 创建转换器（自定义提示词模板，必须含 `{question}`）。
    pub fn with_prompt(llm: std::sync::Arc<dyn Llm>, prompt_template: impl Into<String>) -> Self {
        Self {
            llm,
            prompt_template: prompt_template.into(),
            max_sub_questions: 4,
        }
    }

    /// 设置子问题数量上限。
    ///
    /// 提示词里要求「2 到 4 个」，因为：
    ///
    /// - 1 个 → 没拆，退化成单查询
    /// - \> 4 个 → 分解过细，检索次数爆炸，且子问题之间开始重复
    ///
    /// 5 个以上的复合问题应该**递归分解**，而不是一次性拆到底。
    pub fn with_max_sub_questions(mut self, max_sub_questions: usize) -> Self {
        self.max_sub_questions = max_sub_questions.max(1);
        self
    }

    /// 当前子问题数量上限。
    pub fn max_sub_questions(&self) -> usize {
        self.max_sub_questions
    }
}

impl QueryTransformer for DecomposeTransformer {
    fn strategy(&self) -> Strategy {
        Strategy::Decompose
    }

    fn transform<'a>(&'a self, question: &'a str) -> TransformFuture<'a> {
        Box::pin(async move {
            let question = question.trim();
            if question.is_empty() {
                return Err(crate::error::QueryError::BadRequest(
                    "问题不能为空".to_string(),
                ));
            }

            let prompt = self.prompt_template.replace("{question}", question);
            let raw = self.llm.chat("", &prompt).await?;

            let items = parse_numbered_lines(&raw, self.max_sub_questions);

            if items.is_empty() {
                warn!("Decompose 输出解析不出子问题");
                return Err(crate::error::QueryError::Transformation(
                    "Decompose 输出解析不出子问题".to_string(),
                ));
            }

            debug!(question = %question, sub_count = items.len(), "Decompose 拆分子问题");
            for (i, item) in items.iter().enumerate() {
                debug!(index = i + 1, sub_question = %item, "子问题");
            }

            Ok(items
                .into_iter()
                .enumerate()
                .map(|(i, text)| TransformedQuery {
                    text,
                    strategy: Strategy::Decompose,
                    variant: Some(i),
                    note: Some("子问题".to_string()),
                })
                .collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::llm::ScriptedLlm;

    #[tokio::test]
    async fn splits_compound_question() {
        let llm = ScriptedLlm::always(
            "1. 光合作用的光反应阶段发生在什么位置？\n2. 光合作用的暗反应阶段发生在什么位置？\n3. 光合作用的光反应阶段的主要作用是什么？",
        );
        let t = DecomposeTransformer::new(Arc::new(llm));

        let out = t
            .transform("光合作用的光反应和暗反应分别在什么位置进行？")
            .await
            .unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|q| q.strategy == Strategy::Decompose));
        assert_eq!(out[0].variant, Some(0));
    }

    #[tokio::test]
    async fn caps_at_max_sub_questions() {
        let llm = ScriptedLlm::always("1. 一\n2. 二\n3. 三\n4. 四\n5. 五");
        let t = DecomposeTransformer::new(Arc::new(llm)).with_max_sub_questions(3);

        let out = t.transform("问题").await.unwrap();
        assert_eq!(out.len(), 3);
    }

    #[tokio::test]
    async fn single_sub_question_is_valid() {
        // 只有一个检索意图时，拆出 1 个是正常的。
        let llm = ScriptedLlm::always("1. 咖啡因的半衰期是多久");
        let t = DecomposeTransformer::new(Arc::new(llm));

        let out = t.transform("咖啡因半衰期").await.unwrap();
        assert_eq!(out.len(), 1);
    }
}
