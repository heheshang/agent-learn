//! Multi-Query 转换器 —— 一个问题生成 N 个等价问法。
//!
//! ## 机制
//!
//! ```text
//! 问题 ──LLM──> 问法1 / 问法2 / 问法3
//!                  │        │        │
//!                  └── 各自检索 ──┘
//!                          │
//!                       RRF 融合
//! ```
//!
//! ## 为什么比 HyDE 更「广」但更「浅」
//!
//! | | HyDE | Multi-Query |
//! |---|---|---|
//! | 覆盖 | 一个点（最可能的答案） | 一个面（多种问法） |
//! | 检索次数 | 1 | N |
//! | 优势 | 单次检索精度高 | 召回率高，不容易「完全搜不到」 |
//! | 劣势 | 假想文档写偏了 → 全盘皆错 | 可能召回到「同义但冗余」的结果 |
//!
//! 实践中两者**互补**：HyDE 保证「搜得到」，Multi-Query 保证「搜得全」。
//! 本项目的 `/api/compare` 端点就是用来量这个差值的。
//!
//! ## 一个隐含的坑：结果冗余
//!
//! 3 个等价的问法很可能召回**同一批块**（只是顺序略有不同）。
//! 所以融合时必须做**去重** —— 否则 top-k 里塞满同一个块的三个副本，
//! 上下文里全是重复内容，把 token 预算浪费光。
//!
//! 去重在 [`crate::fusion`] 里做（按块 ID）。

use crate::llm::Llm;
use agent_logging::{debug, warn};

use super::prompts::MULTI_QUERY_PROMPT;
use super::{QueryTransformer, Strategy, TransformFuture, TransformedQuery, parse_numbered_lines};

/// Multi-Query 转换器。
pub struct MultiQueryTransformer {
    llm: std::sync::Arc<dyn Llm>,
    prompt_template: String,
    /// 最多接受几个问法。
    num_queries: usize,
}

impl std::fmt::Debug for MultiQueryTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiQueryTransformer")
            .field("num_queries", &self.num_queries)
            .finish()
    }
}

impl MultiQueryTransformer {
    /// 创建转换器（用默认提示词，默认 3 个问法）。
    pub fn new(llm: std::sync::Arc<dyn Llm>) -> Self {
        Self {
            llm,
            prompt_template: MULTI_QUERY_PROMPT.to_string(),
            num_queries: 3,
        }
    }

    /// 创建转换器（自定义提示词模板，必须含 `{question}`）。
    pub fn with_prompt(llm: std::sync::Arc<dyn Llm>, prompt_template: impl Into<String>) -> Self {
        Self {
            llm,
            prompt_template: prompt_template.into(),
            num_queries: 3,
        }
    }

    /// 设置问法数量上限。
    ///
    /// 数量不是越多越好：
    ///
    /// - 太少（1 个）→ 退化成单查询，没有融合的意义
    /// - 太多（> 5）→ 检索次数线性增长，延迟和成本都上去，
    ///   而且第 4、5 个问法往往和前 3 个高度重复，边际收益趋近于 0
    ///
    /// 经验值 3~4。
    pub fn with_num_queries(mut self, num_queries: usize) -> Self {
        self.num_queries = num_queries.max(1);
        self
    }

    /// 当前问法数量上限。
    pub fn num_queries(&self) -> usize {
        self.num_queries
    }
}

impl QueryTransformer for MultiQueryTransformer {
    fn strategy(&self) -> Strategy {
        Strategy::MultiQuery
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

            let items = parse_numbered_lines(&raw, self.num_queries);

            if items.is_empty() {
                warn!("Multi-Query 输出解析不出问法");
                return Err(crate::error::QueryError::Transformation(
                    "Multi-Query 输出解析不出问法".to_string(),
                ));
            }

            debug!(question = %question, query_count = items.len(), "Multi-Query 生成问法");
            for (i, item) in items.iter().enumerate() {
                debug!(variant = i + 1, query = %item, "生成的问法");
            }

            Ok(items
                .into_iter()
                .enumerate()
                .map(|(i, text)| TransformedQuery {
                    text,
                    strategy: Strategy::MultiQuery,
                    variant: Some(i),
                    note: None,
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
    async fn produces_multiple_queries() {
        let llm = ScriptedLlm::always(
            "1. 模型知识截止日期如何解决\n2. RAG 如何缓解知识截止\n3. 如何给模型补充新知识",
        );
        let t = MultiQueryTransformer::new(Arc::new(llm));

        let out = t.transform("模型的知识有截止日期怎么办").await.unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].variant, Some(0));
        assert_eq!(out[2].variant, Some(2));
        assert!(out.iter().all(|q| q.strategy == Strategy::MultiQuery));
    }

    #[tokio::test]
    async fn caps_at_num_queries() {
        let llm = ScriptedLlm::always("1. 一\n2. 二\n3. 三\n4. 四\n5. 五");
        let t = MultiQueryTransformer::new(Arc::new(llm)).with_num_queries(2);

        let out = t.transform("问题").await.unwrap();
        assert_eq!(out.len(), 2);
    }

    #[tokio::test]
    async fn drops_meta_preamble() {
        let llm =
            ScriptedLlm::always("Here are 3 different phrasings:\n1. 问法一\n2. 问法二\n3. 问法三");
        let t = MultiQueryTransformer::new(Arc::new(llm));

        let out = t.transform("问题").await.unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text, "问法一");
    }

    #[tokio::test]
    async fn errors_when_no_queries_parsed() {
        let llm = ScriptedLlm::always("Here are the queries:");
        let t = MultiQueryTransformer::new(Arc::new(llm));
        assert!(t.transform("问题").await.unwrap_err().is_degradable());
    }
}
