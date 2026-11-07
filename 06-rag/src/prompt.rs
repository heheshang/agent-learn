//! Prompt 组装 —— 把检索结果变成模型能用的上下文。
//!
//! ## Naive RAG 的 Prompt 长什么样
//!
//! Naive 版本只有两个部分：
//!
//! ```text
//! [System]  你是问答助手。只根据下面提供的资料回答问题。
//!           如果资料里没有答案，就说「根据已有资料无法回答」。
//!           回答要用 `[1]`、`[2]` 标注引用来源。
//!
//! [User]    资料：
//!           [1] # 标题 > 子标题
//!           正文...
//!
//!           [2] # 另一个标题
//!           正文...
//!
//!           问题：什么是 RAG？
//! ```
//!
//! 朴素的地方在于：**检索到什么就拼什么**，不排序、不去重、不压缩、
//! 不做「上下文超预算就砍掉最不相关的块」。
//!
//! ## Prompt 里每句话的作用
//!
//! | 指令 | 防的是什么 |
//! |---|---|
//! | 「只根据提供的资料回答」 | 模型用自己的先验知识编答案（幻觉） |
//! | 「资料里没有答案就说无法回答」 | 强行作答，比不知道更糟 |
//! | 「用 `[n]` 标注引用」 | 答案不可溯源，用户无法核对 |
//!
//! 这三条是 RAG 提示词里最基础、也最有效的三条。
//!
//! ## 上下文顺序有讲究
//!
//! 常见做法是**最相关的放最后**（靠近问题）—— 大模型对上下文中间的内容
//! 注意力较弱（"lost in the middle" 现象），首尾的信息更容易被用到。
//!
//! 但朴素版**不重排** —— 检索顺序即最终顺序（最相关在前）。
//! 本项目保留这个朴素行为，并在 README 里对比说明。

use crate::retrieval::RetrievalResult;

/// 组装好的 Prompt。
#[derive(Debug, Clone)]
pub struct AssembledPrompt {
    /// System 提示词。
    pub system: String,

    /// User 提示词（包含资料 + 问题）。
    pub user: String,
}

impl AssembledPrompt {
    /// Prompt 的总字符数（粗略估算 token 数的下界）。
    pub fn char_len(&self) -> usize {
        self.system.chars().count() + self.user.chars().count()
    }
}

/// System 提示词模板。
///
/// 分成「角色 / 约束 / 引用格式」三块，是为了让模型每一件事都只做一件事。
pub const SYSTEM_PROMPT: &str = "\
你是一个技术文档问答助手。请严格遵守以下规则：

1. 只根据【参考资料】中的内容回答问题。参考资料里没有的信息，不要编造。
2. 如果参考资料不足以回答问题，直接说明「根据现有资料无法回答该问题」，并指出缺少哪部分信息。
3. 回答中引用资料的地方，用 [1]、[2] 这样的方括号编号标注来源编号。
4. 回答用简洁的中文，不要复述整段资料，只抽取与问题相关的信息。";

/// 组装 Prompt。
///
/// # 参数
///
/// - `result`：检索结果（命中块 + 引用编号）
/// - `question`：用户问题
pub fn assemble(result: &RetrievalResult, question: &str) -> AssembledPrompt {
    let context = result.to_context();

    // ⚠️ 资料为空时也要给模型一个明确的「没有资料」信号，
    // 而不是给一个空的资料块 —— 后者容易让模型误以为
    // 「资料里确实没写，所以我知道答案」。
    let context_section = if context.is_empty() {
        "（本次检索没有召回任何资料片段）".to_string()
    } else {
        context
    };

    let user = format!(
        "【参考资料】\n{context_section}\n\n【问题】\n{question}",
    );

    AssembledPrompt {
        system: SYSTEM_PROMPT.to_string(),
        user,
    }
}

/// 组装 Prompt（带自定义 system 提示）。
///
/// Agent 场景下用 —— 自主 Agent 的 system 提示要额外说明
/// 「你可以调用工具」以及工具的使用规则。
pub fn assemble_with_system(
    result: &RetrievalResult,
    question: &str,
    system: &str,
) -> AssembledPrompt {
    let mut prompt = assemble(result, question);
    prompt.system = system.to_string();
    prompt
}

/// 构造「无检索结果」的场景提示。
///
/// 当检索没召回到任何东西时，Naive RAG 仍然会把问题发给模型（因为
/// 也许模型自己知道）。但更好的做法是**直接拒答**，省一次 LLM 调用。
///
/// 本项目提供这个函数，让调用方自己决定要不要用 —— 这就是「阈值驱动的拒答」
/// 那条增强手段的最小实现。
pub fn assemble_no_context(question: &str) -> AssembledPrompt {
    AssembledPrompt {
        system: SYSTEM_PROMPT.to_string(),
        user: format!(
            "【参考资料】\n（没有召回到任何相关资料）\n\n【问题】\n{question}\n\n\
             请直接说明「根据现有资料无法回答该问题」，不要凭自己的知识作答。"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::RetrievedChunk;

    fn result_with_texts(texts: &[&str]) -> RetrievalResult {
        RetrievalResult {
            query: "q".to_string(),
            chunks: texts
                .iter()
                .enumerate()
                .map(|(i, t)| RetrievedChunk {
                    rank: i,
                    id: format!("doc.md#{i}"),
                    score: 0.9 - i as f32 * 0.1,
                    source: "doc.md".to_string(),
                    heading_path: vec!["章节".to_string()],
                    text: t.to_string(),
                    heading_context: "# 章节".to_string(),
                })
                .collect(),
            retrieved_count: texts.len(),
            top_score: Some(0.9),
        }
    }

    #[test]
    fn includes_context_and_question() {
        let result = result_with_texts(&["第一段", "第二段"]);
        let prompt = assemble(&result, "什么是 RAG？");

        assert!(prompt.user.contains("第一段"));
        assert!(prompt.user.contains("第二段"));
        assert!(prompt.user.contains("什么是 RAG？"));
        assert!(prompt.user.contains("[1]"));
        assert!(prompt.user.contains("[2]"));
    }

    #[test]
    fn empty_context_is_explicit() {
        let result = result_with_texts(&[]);
        let prompt = assemble(&result, "问题");

        assert!(prompt.user.contains("没有召回任何资料片段"));
    }

    #[test]
    fn no_context_variant_tells_model_to_refuse() {
        let prompt = assemble_no_context("问题");
        assert!(prompt.user.contains("无法回答该问题"));
    }

    #[test]
    fn custom_system_replaces_default() {
        let result = result_with_texts(&["x"]);
        let prompt = assemble_with_system(&result, "q", "你是检索助手");
        assert_eq!(prompt.system, "你是检索助手");
    }
}
