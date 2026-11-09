//! 提示词组装 —— 把「短期记忆 + 长期记忆」拼成一次模型调用。
//!
//! ## 三段式结构
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │ # Identity                                      │  你是谁
//! ├─────────────────────────────────────────────────┤
//! │ # Instructions                                 │  行为规则
//! ├─────────────────────────────────────────────────┤
//! │ # Memory（长期记忆召回的事实）                  │  ★ 跨会话记住的东西
//! ├─────────────────────────────────────────────────┤
//! │ # Conversation（短期记忆历史）                  │  ★ 本线程说过的话
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## 为什么把「长期记忆」和「对话历史」**分开**成两段
//!
//! 这是本项目最关键的一个提示词决策。
//!
//! 两者性质完全不同：
//!
//! | | 长期记忆 | 对话历史 |
//! |---|---|---|
//! | 来源 | 之前的会话（别的线程） | 本线程 |
//! | 内容 | **结论**（「用户名字是 Bob」） | **过程**（「我叫 Bob」） |
//! | 长度 | 几条（受 top_k 限制） | 可能几十条 |
//! | 稳定性 | 稳定（跨会话） | 每次都变 |
//!
//! 如果把它们混在一起丢给模型，模型会**分不清**：
//! 「Bob 是我现在告诉你的，还是你记得的？」
//!
//! 分段之后：
//!
//! - 「Memory」段 = **权威的已知事实**（模型应当直接采信）
//! - 「Conversation」段 = **本轮对话的上下文**
//!
//! ## ⚠️ 长期记忆的事实**必须标成「可能过时」
//!
//! 用户可能改口（「我以前叫 Bob，现在叫 Robert」）。
//! 所以 Memory 段的措辞是「你记得的关于用户的信息（可能已经过时）」，
//! 并且指令里明确说「如果用户当前的说法与记忆冲突，**以当前说法为准**」。
//!
//! 不写这一句，模型会用旧记忆纠正用户 —— 这是很糟糕的体验。

use crate::longterm::MemoryFact;
use crate::state::MemoryMessage;

/// 提示词配置。
#[derive(Debug, Clone)]
pub struct PromptConfig {
    /// 是否注入长期记忆。
    pub use_longterm: bool,

    /// 长期记忆召回条数上限。
    pub memory_top_k: usize,

    /// 历史里保留的消息条数（`0` = 全部保留）。
    ///
    /// 对应 LangChain 的 `trim_messages` / `keep=("messages", 20)`。
    pub history_keep: usize,

    /// 身份描述。
    pub identity: String,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            use_longterm: true,
            memory_top_k: 5,
            history_keep: 20,
            identity: "你是一个有记忆的 AI 助手。你记得用户在过去会话中告诉过你的信息。\
                      当用户询问你之前说过的内容时，直接用记忆里的事实回答。"
                .to_string(),
        }
    }
}

/// 构造 system 提示词。
///
/// ## 参数
///
/// - `facts` —— 召回的长期记忆事实（可能为空）
/// - `config` —— 提示词配置
pub fn system_prompt(facts: &[MemoryFact], config: &PromptConfig) -> String {
    let mut out = String::new();

    // ── Identity ─────────────────────────────────────────────
    out.push_str("# Identity\n\n");
    out.push_str(&config.identity);
    out.push_str("\n\n");

    // ── Instructions ─────────────────────────────────────────
    out.push_str("# Instructions\n\n");
    out.push_str("- 你可以看到本次对话的完整历史，请利用它理解上下文。\n");
    out.push_str("- 回答里如果用到了记忆中的信息，不要说「根据我的记忆」，直接自然地说出来。\n");
    out.push_str("- 如果用户的当前说法与记忆里的信息冲突，**以当前说法为准**，并更新你的记忆。\n");
    out.push_str("- 不要编造记忆里没有的信息。\n");

    // ── Memory（长期记忆）─────────────────────────────────────
    //
    // ⚠️ 这段是「跨会话记忆」的唯一出口 ——
    //    没有它，长期记忆就只是数据库里的一堆死数据。
    if config.use_longterm && !facts.is_empty() {
        out.push_str("\n# Memory\n\n");
        out.push_str("你记得的关于用户的信息（来自之前的会话，可能已经过时）：\n\n");
        for (i, fact) in facts.iter().enumerate() {
            out.push_str(&format!("{}. {}\n", i + 1, fact.render()));
        }
    }

    out
}

/// 构造要发给模型的**消息数组**。
///
/// ## 关键设计：召回的事实**不进消息数组**
///
/// 召回的事实放在 **system 提示**里（`# Memory` 段），
/// **不**作为消息加入历史。
///
/// ## ⚠️ 为什么
///
/// 1. **事实不是对话** —— 它没有「谁说的」这个语义，
///    塞进消息数组会污染对话轮次。
/// 2. **事实不该被持久化进短期记忆** —— 否则每次对话都会把召回的事实
///    再存一遍，几轮之后历史里全是重复的事实。
/// 3. **上下文可控** —— 事实在 system 里，位置固定、顺序固定、
///    数量受 `top_k` 限制。
///
/// ## 截断策略
///
/// 历史太长时，保留**最近 N 条**（`config.history_keep`），
/// 但**永远保留系统提示**（见 [`ThreadState::recent`]）。
pub fn build_messages(history: &[MemoryMessage], config: &PromptConfig) -> Vec<MemoryMessage> {
    if config.history_keep == 0 {
        return history.to_vec();
    }

    // 用 `ThreadState::recent` 的语义，但这里只有切片没有 state ——
    // 所以手动实现同样的规则（保留 System + 最近 N 条非 System）。
    let mut out: Vec<MemoryMessage> = Vec::new();

    for msg in history
        .iter()
        .filter(|m| m.role == crate::state::MemoryRole::System)
    {
        out.push(msg.clone());
    }

    let non_system: Vec<&MemoryMessage> = history
        .iter()
        .filter(|m| m.role != crate::state::MemoryRole::System)
        .collect();

    let start = non_system.len().saturating_sub(config.history_keep);
    for msg in &non_system[start..] {
        out.push((*msg).clone());
    }

    out
}

/// 构造「记忆抽取」的提示词。
///
/// ## 对应 Mem0 的什么
///
/// Mem0 的 `memory.add(messages)` 内部会调 LLM 做事实抽取，
/// 提示词大意是「从这段对话里提取关于用户的原子事实」。
///
/// ## 输出格式
///
/// 严格的 JSON 数组 —— 这样能用 `serde_json` 直接解析。
///
/// ```json
/// [{"subject": "用户", "predicate": "名字", "value": "Bob", "importance": 0.8}]
/// ```
pub fn extraction_prompt(messages: &[MemoryMessage], owner: &str) -> String {
    let mut out = String::new();

    out.push_str("# Task\n\n");
    out.push_str("从下面的对话里提取**关于用户**的原子事实。\n\n");

    out.push_str("# Rules\n\n");
    out.push_str("- 只提取**事实**，不要提取闲聊、客套、情绪。\n");
    out.push_str("- 每条事实是「主语 + 谓语 + 值」三段。\n");
    out.push_str("- 主语通常是「用户」；如果是关于项目/技术的，主语写那个名字。\n");
    out.push_str("- 谓语要短且稳定（如「名字」「偏好」「职业」「语言」）。\n");
    out.push_str("- 同一件���的多个说法只保留一条。\n");
    out.push_str("- **没有事实就返回空数组** `[]` —— 不要为了凑数编造。\n");
    out.push_str("- importance 是 0~1 的小数，表示这条事实的重要程度。\n");

    out.push_str("\n# Output Format\n\n");
    out.push_str("严格的 JSON 数组，不要 markdown 代码块，不要任何解释文字：\n\n");
    out.push_str(
        r#"[{"subject": "用户", "predicate": "名字", "value": "Bob", "importance": 0.8}]"#,
    );

    out.push_str(&format!("\n\n# Owner\n\n{owner}\n\n"));

    out.push_str("# Dialogue\n\n");
    for msg in messages {
        let role = msg.role.name();
        // 只给「有信息量」的消息 —— 事实都在 user 侧。
        if msg.content.trim().is_empty() {
            continue;
        }
        out.push_str(&format!("[{role}] {}\n", msg.content.trim()));
    }

    out
}

/// 解析抽取器返回的 JSON。
///
/// ## ⚠️ 为什么不能直接 `serde_json::from_str`
///
/// 模型经常会：
///
/// - 用 markdown 代码块包裹（```json ... ```）
/// - 在 JSON 前后加解释文字（「好的，提取结果如下：」）
///
/// 直接解析会失败。所以这里做**容错抽取**：
/// 找到第一个 `[` 到最后一个 `]` 之间的内容。
///
/// 这和 09-rag-eval 的 `extract_json_object` 是同一套思路
/// （那边的对象是 `{...}`，这里是数组 `[...]`）。
pub fn parse_extraction(text: &str) -> Vec<ExtractedFact> {
    let Some(array) = extract_json_array(text) else {
        return Vec::new();
    };

    serde_json::from_str::<Vec<ExtractedFact>>(array).unwrap_or_default()
}

/// 抽取器输出的一条原始事实。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ExtractedFact {
    /// 主语。
    #[serde(default = "default_subject")]
    pub subject: String,

    /// 谓语。
    #[serde(default)]
    pub predicate: String,

    /// 值。
    #[serde(default)]
    pub value: String,

    /// 重要度。
    #[serde(default = "default_importance")]
    pub importance: f32,
}

fn default_subject() -> String {
    "用户".to_string()
}

fn default_importance() -> f32 {
    0.5
}

impl ExtractedFact {
    /// 转成 [`MemoryFact`]（带上 owner）。
    pub fn to_fact(&self, owner: &str) -> crate::longterm::MemoryFact {
        crate::longterm::MemoryFact::new(&self.subject, &self.predicate, &self.value, owner)
            .with_importance(self.importance)
    }
}

/// 从一段文本里抠出 JSON 数组。
///
/// 找第一个 `[` 到**配对**的 `]` —— 括号计数，跳过字符串字面量里的括号。
fn extract_json_array(text: &str) -> Option<&str> {
    let start = text.find('[')?;
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for i in start..bytes.len() {
        let c = bytes[i];

        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
            continue;
        }

        match c {
            b'"' => in_string = true,
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].trim());
                }
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::MemoryRole;

    fn fact(subject: &str, predicate: &str, value: &str) -> MemoryFact {
        MemoryFact::new(subject, predicate, value, "default")
    }

    #[test]
    fn system_prompt_has_the_three_sections() {
        let prompt = system_prompt(&[], &PromptConfig::default());
        assert!(prompt.contains("# Identity"));
        assert!(prompt.contains("# Instructions"));
        // 没有事实时不该有 Memory 段。
        assert!(!prompt.contains("# Memory"));
    }

    #[test]
    fn system_prompt_injects_recalled_facts() {
        let facts = vec![
            fact("用户", "名字", "Bob"),
            fact("用户", "偏好", "深色模式"),
        ];
        let prompt = system_prompt(&facts, &PromptConfig::default());

        assert!(prompt.contains("# Memory"));
        assert!(prompt.contains("用户的名字：Bob"));
        assert!(prompt.contains("用户的偏好：深色模式"));
    }

    #[test]
    fn system_prompt_warns_that_memories_may_be_stale() {
        // ⚠️ 这一句很重要 —— 防止模型用旧记忆纠正用户。
        let prompt = system_prompt(&[fact("用户", "名字", "Bob")], &PromptConfig::default());
        assert!(prompt.contains("可能已经过时"));
        assert!(prompt.contains("以当前说法为准"));
    }

    #[test]
    fn memory_section_can_be_disabled() {
        let config = PromptConfig {
            use_longterm: false,
            ..Default::default()
        };
        let prompt = system_prompt(&[fact("用户", "名字", "Bob")], &config);
        assert!(!prompt.contains("# Memory"));
    }

    #[test]
    fn build_messages_trims_to_the_tail() {
        let mut history = vec![MemoryMessage::system("规则")];
        for i in 0..10 {
            history.push(MemoryMessage::user(format!("第 {i} 条")));
        }

        let config = PromptConfig {
            history_keep: 3,
            ..Default::default()
        };
        let messages = build_messages(&history, &config);

        // 系统提示 + 最近 3 条。
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, MemoryRole::System);
        assert_eq!(messages[1].content, "第 7 条");
        assert_eq!(messages[3].content, "第 9 条");
    }

    #[test]
    fn build_messages_with_zero_keep_returns_everything() {
        let history = vec![
            MemoryMessage::user("一"),
            MemoryMessage::assistant("二"),
            MemoryMessage::user("三"),
        ];
        let config = PromptConfig {
            history_keep: 0,
            ..Default::default()
        };
        assert_eq!(build_messages(&history, &config).len(), 3);
    }

    #[test]
    fn extraction_prompt_includes_the_dialogue() {
        let messages = vec![
            MemoryMessage::user("我叫 Bob"),
            MemoryMessage::assistant("你好 Bob"),
        ];
        let prompt = extraction_prompt(&messages, "default");

        assert!(prompt.contains("[user] 我叫 Bob"));
        assert!(prompt.contains("[assistant] 你好 Bob"));
        assert!(prompt.contains("default"));
    }

    #[test]
    fn extraction_prompt_skips_empty_messages() {
        let messages = vec![MemoryMessage::user("   ")];
        let prompt = extraction_prompt(&messages, "default");
        assert!(!prompt.contains("[user]"));
    }

    #[test]
    fn parses_a_plain_json_array() {
        let json = r#"[{"subject":"用户","predicate":"名字","value":"Bob","importance":0.8}]"#;
        let facts = parse_extraction(json);

        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].predicate, "名字");
        assert_eq!(facts[0].value, "Bob");
        assert_eq!(facts[0].importance, 0.8);
    }

    #[test]
    fn parses_a_json_array_wrapped_in_a_code_block() {
        // ⚠️ 模型很爱用 markdown 代码块 —— 必须能容错。
        let text = "好的，提取结果如下：\n```json\n[{\"subject\":\"用户\",\"predicate\":\"名字\",\"value\":\"Bob\"}]\n```";
        let facts = parse_extraction(text);

        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].value, "Bob");
    }

    #[test]
    fn parsing_garbage_returns_empty() {
        assert!(parse_extraction("完全没有事实").is_empty());
        assert!(parse_extraction("").is_empty());
        assert!(parse_extraction("[{不是合法 JSON}]").is_empty());
    }

    #[test]
    fn extracted_fact_carries_defaults() {
        let facts = parse_extraction(r#"[{"value":"Bob"}]"#);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].subject, "用户");
        assert_eq!(facts[0].importance, 0.5);
    }

    #[test]
    fn extracted_fact_converts_to_memory_fact() {
        let facts = parse_extraction(
            r#"[{"subject":"用户","predicate":"名字","value":"Bob","importance":0.9}]"#,
        );
        let fact = facts[0].to_fact("alice");

        assert_eq!(fact.owner, "alice");
        assert_eq!(fact.subject, "用户");
        assert_eq!(fact.importance, 0.9);
    }
}
