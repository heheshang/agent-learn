//! HyDE 转换器 —— 用「假想答案文档」去检索。
//!
//! ## 机制
//!
//! ```text
//! 问题 ──LLM──> 假想答案文档（陈述句、含术语）
//!                    │
//!                    ├──> 用【文档】的方式嵌入（passage 前缀）
//!                    └──> 向量检索 ──> 命中真实文档
//! ```
//!
//! ## 为什么有效
//!
//! 嵌入模型训练时的数据是**「文档 ↔ 文档」**的配对（query→passage 除外，
//! 但那是少数），而不是「问题 ↔ 文档」。
//!
//! 所以：
//!
//! - 用**问题**的向量去找**文档** → 有分布错位（distribution shift）
//! - 用**假想文档**的向量去找**文档** → 分布对齐了
//!
//! 这是 HyDE 全部魔力的来源 —— 它不改变检索算法，只改变**送进检索器的那段文本**。
//!
//! ## 实现上的两个关键细节
//!
//! ### 1. 用 `embed_documents` 而不是 `embed_query`
//!
//! 本项目的 E5 模型区分 `query:` / `passage:` 前缀。
//! HyDE 生成的明明是**文档**，所以必须走 `embed_documents` 拿 `passage:` 前缀。
//!
//! 用错了会明显掉效果 —— 前缀不匹配会让向量落在模型没见过的区域。
//! 这个细节由 [`crate::retrieval::EmbedMode`] 统一处理。

use crate::llm::Llm;
use agent_logging::{debug, warn};

use super::prompts::HYDE_PROMPT;
use super::{QueryTransformer, Strategy, TransformFuture, TransformedQuery, parse_single};

/// HyDE 转换器。
pub struct HydeTransformer {
    llm: std::sync::Arc<dyn Llm>,
    prompt_template: String,
}

impl std::fmt::Debug for HydeTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HydeTransformer").finish()
    }
}

impl HydeTransformer {
    /// 创建转换器（用默认提示词）。
    pub fn new(llm: std::sync::Arc<dyn Llm>) -> Self {
        Self {
            llm,
            prompt_template: HYDE_PROMPT.to_string(),
        }
    }

    /// 创建转换器（自定义提示词模板，必须含 `{question}`）。
    pub fn with_prompt(llm: std::sync::Arc<dyn Llm>, prompt_template: impl Into<String>) -> Self {
        Self {
            llm,
            prompt_template: prompt_template.into(),
        }
    }

    /// 当前提示词模板。
    pub fn prompt(&self) -> &str {
        &self.prompt_template
    }
}

impl QueryTransformer for HydeTransformer {
    fn strategy(&self) -> Strategy {
        Strategy::Hyde
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

            // 解析：取第一段「像文档」的内容。
            let Some(text) = parse_single(&raw) else {
                warn!("HyDE 输出解析不出内容");
                return Err(crate::error::QueryError::Transformation(
                    "HyDE 输出解析不出内容".to_string(),
                ));
            };

            debug!(
                question = %question,
                doc_chars = text.chars().count(),
                "HyDE 生成假想文档"
            );

            Ok(vec![TransformedQuery {
                text,
                strategy: Strategy::Hyde,
                variant: Some(0),
                note: Some("假想答案文档（按文档方式嵌入）".to_string()),
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
    async fn produces_single_hypothetical_document() {
        let llm = ScriptedLlm::always(
            "检索增强生成通过从外部知识库检索相关文档来缓解模型的知识截止问题。",
        );
        let t = HydeTransformer::new(Arc::new(llm));

        let out = t.transform("模型的知识有截止日期怎么办").await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].strategy, Strategy::Hyde);
        assert_eq!(out[0].variant, Some(0));
        assert!(out[0].text.contains("检索增强生成"));
        assert!(out[0].note.is_some());
    }

    /// 回归测试：HyDE 的提示词要求 150~250 字的输出。
    ///
    /// 如果解析器用「列表项 200 字上限」去过滤，HyDE 会**永远解析不出内容**
    /// → 永远降级成原问题 → HyDE 静默失效（代码不报错，只是效果为零）。
    #[tokio::test]
    async fn accepts_long_hypothetical_document() {
        let paragraph = "模型的知识截止日期问题可以通过检索增强生成来解决。".repeat(12);
        assert!(paragraph.chars().count() > 200);

        let llm = ScriptedLlm::always(&paragraph);
        let t = HydeTransformer::new(Arc::new(llm));

        let out = t.transform("模型的知识有截止日期怎么办").await.unwrap();
        assert_eq!(out.len(), 1);
        assert!(
            out[0].text.contains("检索增强生成"),
            "长文档应该被完整保留，实际: {}",
            out[0].text
        );
    }

    #[tokio::test]
    async fn fills_question_placeholder() {
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = calls.clone();

        // 用一个会记录调用的替身，方便断言提示词内容。
        struct RecordingLlm {
            sink: std::sync::Arc<std::sync::Mutex<Vec<(String, String)>>>,
        }

        impl Llm for RecordingLlm {
            fn chat<'a>(&'a self, system: &'a str, user: &'a str) -> crate::llm::LlmFuture<'a> {
                self.sink
                    .lock()
                    .unwrap()
                    .push((system.to_string(), user.to_string()));
                Box::pin(async move { Ok("文档正文".to_string()) })
            }

            fn model_name(&self) -> &str {
                "recording"
            }
        }

        let t = HydeTransformer::new(Arc::new(RecordingLlm { sink }));

        t.transform("我的问题").await.unwrap();

        let recorded = calls.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert!(recorded[0].1.contains("我的问题"));
        assert!(!recorded[0].1.contains("{question}"));
    }

    #[tokio::test]
    async fn rejects_empty_question() {
        let llm = ScriptedLlm::always("x");
        let t = HydeTransformer::new(Arc::new(llm));
        assert_eq!(t.transform("  ").await.unwrap_err().status_code(), 400);
    }

    #[tokio::test]
    async fn errors_when_output_unparseable() {
        let llm = ScriptedLlm::always("   \n  \n ");
        let t = HydeTransformer::new(Arc::new(llm));
        let err = t.transform("问题").await.unwrap_err();
        assert!(err.is_degradable());
    }
}
