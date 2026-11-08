//! Rewrite 转换器 —— 把问题改写成一个自包含的检索式。
//!
//! ## 机制
//!
//! ```text
//! 问题（含指代）──LLM──> 自包含的检索式 ──> 向量检索
//! ```
//!
//! ## 它解决的具体问题：向量对指代无能为力
//!
//! 嵌入模型把「它」编码成一个固定方向的向量 —— 所有含「它」的句子
//! 都在向量空间里挨得很近，但**离任何具体内容都很远**。
//!
//! 所以：
//!
//! ```text
//! 「它训练成本高吗」  →  向量集中在「它/是否/成本高」的泛化方向
//!                    →  召回的全是「关于它的泛泛讨论」
//! ```
//!
//! Rewrite 把它补全成「RAG 和微调相比，哪种方式的训练成本更高」，
//! 向量才有具体的方向。
//!
//! ## 什么时候该用
//!
//! | 场景 | 该用吗 |
//! |---|---|
//! | 单轮问答，问题是完整名词短语 | **不用** —— 改写只会引入噪声 |
//! | 多轮对话，问题里有指代 | **用** —— 这是它唯一不可替代的场景 |
//! | 问题里有口语、缩写、错别字 | **用** —— 帮模型对齐术语 |
//!
//! 本项目默认在 `single_turn` 场景下也能用 Rewrite，但 README 里会强调：
//! **Rewrite 的收益高度依赖问题形态**，它不是「越大越通用」的策略。

use crate::llm::Llm;
use agent_logging::{debug, warn};

use super::prompts::REWRITE_PROMPT;
use super::{QueryTransformer, Strategy, TransformFuture, TransformedQuery, parse_single};

/// Rewrite 转换器。
pub struct RewriteTransformer {
    llm: std::sync::Arc<dyn Llm>,
    prompt_template: String,
}

impl std::fmt::Debug for RewriteTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RewriteTransformer").finish()
    }
}

impl RewriteTransformer {
    /// 创建转换器（用默认提示词）。
    pub fn new(llm: std::sync::Arc<dyn Llm>) -> Self {
        Self {
            llm,
            prompt_template: REWRITE_PROMPT.to_string(),
        }
    }

    /// 创建转换器（自定义提示词模板，必须含 `{question}`）。
    pub fn with_prompt(llm: std::sync::Arc<dyn Llm>, prompt_template: impl Into<String>) -> Self {
        Self {
            llm,
            prompt_template: prompt_template.into(),
        }
    }
}

impl QueryTransformer for RewriteTransformer {
    fn strategy(&self) -> Strategy {
        Strategy::Rewrite
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

            let Some(text) = parse_single(&raw) else {
                warn!("Rewrite 输出解析不出内容");
                return Err(crate::error::QueryError::Transformation(
                    "Rewrite 输出解析不出内容".to_string(),
                ));
            };

            debug!(question = %question, rewritten = %text, "Rewrite 改写完成");

            Ok(vec![TransformedQuery {
                text,
                strategy: Strategy::Rewrite,
                variant: Some(0),
                note: Some("自包含检索式".to_string()),
            }])
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::llm::ScriptedLlm;

    #[tokio::test]
    async fn produces_single_rewritten_query() {
        let llm = ScriptedLlm::always("RAG 和微调相比，哪种方式的训练成本更高");
        let t = RewriteTransformer::new(Arc::new(llm));

        let out = t.transform("那它训练成本高吗").await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].strategy, Strategy::Rewrite);
        assert!(out[0].text.contains("RAG"));
    }

    #[tokio::test]
    async fn strips_surrounding_quotes() {
        // LLM 有时会给结果加引号 —— 检索时引号是噪声，要剥掉。
        let llm = ScriptedLlm::always("「改写后的检索式」");
        let t = RewriteTransformer::new(Arc::new(llm));

        let out = t.transform("问题").await.unwrap();
        assert_eq!(out[0].text, "改写后的检索式");
    }

    #[tokio::test]
    async fn keeps_quotes_inside_text() {
        // 只剥「首尾成对」的引号，句子中间的引号不动。
        let llm = ScriptedLlm::always("什么是 \"RAG\" 架构");
        let t = RewriteTransformer::new(Arc::new(llm));

        let out = t.transform("问题").await.unwrap();
        assert_eq!(out[0].text, "什么是 \"RAG\" 架构");
    }

    #[tokio::test]
    async fn errors_when_unparseable() {
        let llm = ScriptedLlm::always("   ");
        let t = RewriteTransformer::new(Arc::new(llm));
        assert!(t.transform("问题").await.unwrap_err().is_degradable());
    }
}
