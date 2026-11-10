//! **推理层** —— ReAct 循环、输出解析、算术求值、记忆。
//!
//! # 这个模块汇总了哪些历史能力
//!
//! | 能力 | 来自 | 在这里的形态 |
//! |---|---|---|
//! | ReAct 输出解析 | 13-agent-core | [`parse_react_output`] |
//! | 手写算术求值器 | 13-agent-core | [`calculate`] |
//! | 线程消息合并 + 事实召回 | 17-agent-memory | [`ThreadState`] / [`MemoryFact`] |

use serde_json::{Value, json};

/// 解析出的 ReAct 输出。
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedOutput {
    /// 最终答案。
    FinalAnswer(String),
    /// 要调工具。
    Action {
        /// 工具名。
        tool: String,
        /// 参数（JSON）。
        args: Value,
    },
    /// 什么都没解析出来（容错当成答案）。
    Unparsed(String),
}

/// ★ ReAct 输出解析。
///
/// ## ⚠️ 必须找**最后一个**标记，不能找第一个
///
/// 模型有时会**回显整个提示词**再继续。
/// 如果找第一个 `Action:`，会匹配到回显里的旧动作 ——
/// 上一步工具被重跑、模型真正的答案被丢弃。
///
/// 症状很隐蔽：答案变成「平均每轮调用 3 次工具」这种无关内容。
///
/// ## ⚠️ 工具名要**截断**成标识符
///
/// `Action:Thought:Thought:` 曾被解析出工具名 `Thought:Thought` ——
/// 那是个永远匹配不上任何工具的名字。
/// 修法是「取开头那段连续的字母 / 数字 / 下划线」。
///
/// ## ⚠️ `Final Answer:` 优先级高于 `Action:`
///
/// 模型有时同时给两者，这时答案优先（它已经收工了）。
pub fn parse_react_output(text: &str) -> ParsedOutput {
    // ★ 找**最后一个**标记。
    let final_at = text.rfind("Final Answer:");
    let action_at = text.rfind("Action:");
    let input_at = text.rfind("Action Input:");

    // 答案优先 —— 但只在它出现在 Action 之后时才让位。
    if let Some(pos) = final_at {
        let action_is_later = action_at.is_some_and(|a| a > pos);
        if !action_is_later {
            let answer = text[pos + "Final Answer:".len()..].trim().to_string();
            return ParsedOutput::FinalAnswer(answer);
        }
    }

    if let Some(pos) = action_at {
        let after = &text[pos + "Action:".len()..];
        let tool = take_identifier(after);

        let args = match input_at {
            Some(ip) if ip > pos => {
                let raw = &text[ip + "Action Input:".len()..];
                parse_action_input(raw)
            }
            _ => json!({}),
        };

        return ParsedOutput::Action { tool, args };
    }

    ParsedOutput::Unparsed(text.trim().to_string())
}

/// ★ 取开头那段连续的标识符（字母 / 数字 / 下划线）。
fn take_identifier(text: &str) -> String {
    text.trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// 解析 `Action Input:` 后面的参数。
///
/// 优先找**能成功反序列化**的 JSON 对象 —— 而不是「第一个 `{` 到最后一个 `}`」，
/// 因为模型的输出里可能有 `{周报}` 这类**括号配对但不是合法 JSON** 的占位。
///
/// 判据是「能 `from_str` 成功」而不是「括号配对」。
pub fn parse_action_input(raw: &str) -> Value {
    let cleaned = strip_code_fence(raw.trim());

    if let Ok(v) = serde_json::from_str::<Value>(&cleaned) {
        return v;
    }

    // 逐字符扫配对括号（跳过字符串字面量里的括号），逐个试解析。
    if let Some(v) = scan_balanced_json(&cleaned) {
        return v;
    }

    // 退化：当成纯文本参数。
    json!({ "input": cleaned })
}

/// 去掉 markdown 代码围栏。
fn strip_code_fence(text: &str) -> String {
    let t = text.trim();
    if let Some(rest) = t.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        let rest = rest.strip_suffix("```").unwrap_or(rest);
        return rest.trim().to_string();
    }
    t.to_string()
}

/// ★ 逐字符扫配对括号，跳过字符串字面量。
fn scan_balanced_json(text: &str) -> Option<Value> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] != '{' {
            i += 1;
            continue;
        }
        let start = i;
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;

        while i < chars.len() {
            let c = chars[i];
            if in_string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    in_string = false;
                }
            } else {
                match c {
                    '"' => in_string = true,
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            let candidate: String = chars[start..=i].iter().collect();
                            if let Ok(v) = serde_json::from_str::<Value>(&candidate) {
                                return Some(v);
                            }
                            break;
                        }
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        i += 1;
    }

    None
}

/// ★ 手写算术求值器 —— 只支持 `+ - * /` 和括号。
///
/// ## 为什么不用模型的「心算」
///
/// 统计数字必须**来自计算**，不能来自模型生成的文本 ——
/// 模型会「看起来很专业但数字是编的」。
///
/// 递归下降，优先级：`()` > `* /` > `+ -`。
pub fn calculate(expression: &str) -> Result<f64, String> {
    let mut parser = Parser {
        chars: expression.chars().collect(),
        pos: 0,
    };
    let value = parser.parse_expr()?;
    parser.skip_whitespace();
    if parser.pos != parser.chars.len() {
        return Err(format!(
            "表达式尾部有多余字符: {}",
            parser.chars[parser.pos..].iter().collect::<String>()
        ));
    }
    Ok(value)
}

/// 算术解析器。
struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    /// 跳过空白。
    fn skip_whitespace(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    /// 解析 `+ -`。
    fn parse_expr(&mut self) -> Result<f64, String> {
        let mut left = self.parse_term()?;
        loop {
            self.skip_whitespace();
            match self.chars.get(self.pos) {
                Some('+') => {
                    self.pos += 1;
                    let right = self.parse_term()?;
                    left += right;
                }
                Some('-') => {
                    self.pos += 1;
                    let right = self.parse_term()?;
                    left -= right;
                }
                _ => return Ok(left),
            }
        }
    }

    /// 解析 `* /`。
    fn parse_term(&mut self) -> Result<f64, String> {
        let mut left = self.parse_factor()?;
        loop {
            self.skip_whitespace();
            match self.chars.get(self.pos) {
                Some('*') => {
                    self.pos += 1;
                    let right = self.parse_factor()?;
                    left *= right;
                }
                Some('/') => {
                    self.pos += 1;
                    let right = self.parse_factor()?;
                    if right == 0.0 {
                        return Err("除数为零".into());
                    }
                    left /= right;
                }
                _ => return Ok(left),
            }
        }
    }

    /// 解析数字或括号。
    fn parse_factor(&mut self) -> Result<f64, String> {
        self.skip_whitespace();
        match self.chars.get(self.pos) {
            Some('(') => {
                self.pos += 1;
                let v = self.parse_expr()?;
                self.skip_whitespace();
                if self.chars.get(self.pos) != Some(&')') {
                    return Err("缺少右括号".into());
                }
                self.pos += 1;
                Ok(v)
            }
            Some('-') => {
                self.pos += 1;
                Ok(-self.parse_factor()?)
            }
            Some(c) if c.is_ascii_digit() => {
                let start = self.pos;
                while self
                    .chars
                    .get(self.pos)
                    .is_some_and(|c| c.is_ascii_digit() || *c == '.')
                {
                    self.pos += 1;
                }
                self.chars[start..self.pos]
                    .iter()
                    .collect::<String>()
                    .parse::<f64>()
                    .map_err(|e| format!("数字解析失败: {e}"))
            }
            Some(c) => Err(format!("意外的字符: {c}")),
            None => Err("表达式意外结束".into()),
        }
    }
}

/// ─────────────────────────────────────────────────────────────────────
/// 记忆
/// ─────────────────────────────────────────────────────────────────────
/// 消息角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRole {
    /// 系统提示。
    System,
    /// 用户。
    User,
    /// 助手。
    Assistant,
    /// ★ 事实 —— 本项目特有的角色。
    ///
    /// 事实是**权威背景**，不是「某人在说话」——
    /// 所以它映射成 system 而不是 user/assistant。
    Fact,
}

/// 一条消息。
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryMessage {
    /// 消息 ID。
    pub id: String,

    /// 角色。
    pub role: MemoryRole,

    /// 内容。
    pub content: String,
}

impl MemoryMessage {
    /// 构造一条消息。
    pub fn new(role: MemoryRole, content: impl Into<String>) -> Self {
        Self {
            id: next_message_id(),
            role,
            content: content.into(),
        }
    }

    /// 用户消息。
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MemoryRole::User, content)
    }

    /// 助手消息。
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MemoryRole::Assistant, content)
    }

    /// 事实消息。
    pub fn fact(content: impl Into<String>) -> Self {
        Self::new(MemoryRole::Fact, content)
    }
}

/// 消息 ID 计数器。
static MESSAGE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 生成消息 ID。
pub fn next_message_id() -> String {
    let n = MESSAGE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("msg-{n:08}")
}

/// 一个线程的状态。
#[derive(Debug, Clone, Default)]
pub struct ThreadState {
    /// 线程 ID。
    pub thread_id: String,

    /// 消息历史。
    pub messages: Vec<MemoryMessage>,
}

impl ThreadState {
    /// 新建线程。
    pub fn new(thread_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            messages: Vec::new(),
        }
    }

    /// ★ 合并一批消息（LangGraph `add_messages` reducer 的语义）。
    ///
    /// - **同 id** → **原地替换**（保持位置不变）
    /// - **异 id** → 追加到末尾
    ///
    /// ## ⚠️ 为什么「替换要原地」而不是「删了再加」
    ///
    /// 因为位置**有语义** —— 消息顺序就是对话顺序。
    /// 挪到末尾会让工具调用和回执的相对顺序乱掉。
    pub fn merge_messages(&mut self, incoming: Vec<MemoryMessage>) {
        for msg in incoming {
            if let Some(pos) = self.messages.iter().position(|m| m.id == msg.id) {
                self.messages[pos] = msg;
            } else {
                self.messages.push(msg);
            }
        }
    }

    /// 消息条数。
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// 取最近 N 条。
    pub fn recent(&self, keep: usize) -> Vec<MemoryMessage> {
        if self.messages.len() <= keep {
            return self.messages.clone();
        }
        self.messages[self.messages.len() - keep..].to_vec()
    }
}

/// 一条长期记忆事实。
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryFact {
    /// 事实 ID。
    pub id: String,

    /// 主语。
    pub subject: String,

    /// 谓语。
    pub predicate: String,

    /// 值。
    pub value: String,

    /// 归属人。
    pub owner: String,

    /// 重要度（0.0 ~ 1.0）。
    pub importance: f32,
}

impl MemoryFact {
    /// 构造一条事实。
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
        owner: impl Into<String>,
    ) -> Self {
        Self {
            id: format!("fact-{}", next_message_id()),
            subject: subject.into(),
            predicate: predicate.into(),
            value: value.into(),
            owner: owner.into(),
            importance: 0.5,
        }
    }

    /// ★ 去重键 —— 用 `(owner, subject, predicate)`，**不是整条内容**。
    ///
    /// 因为「用户叫 Bob」和「用户名字是 Bob」是**同一件事**。
    /// 用整条内容做键，模型换个说法就产生重复事实。
    pub fn dedup_key(&self) -> (String, String, String) {
        (
            self.owner.clone(),
            self.subject.clone(),
            self.predicate.clone(),
        )
    }

    /// 渲染成给模型看的一行。
    pub fn render(&self) -> String {
        format!("{}的{}：{}", self.subject, self.predicate, self.value)
    }

    /// 用于检索的文本。
    pub fn searchable_text(&self) -> String {
        format!("{} {} {}", self.subject, self.predicate, self.value)
    }
}

/// ★ 事实召回 —— 用 BM25 检索。
///
/// ## ⚠️ 零命中时要**退化**
///
/// BM25 是词元匹配，「用户 / 名字 / Bob」和「我叫什么」词元交集为空 ——
/// 模型会**彻底失忆**。
///
/// 修法：零命中时退化为「按 importance 取前 k 条」——
/// 事实只有几十条，全给不贵，模型自己会挑。
pub fn recall_facts(query: &str, facts: &[MemoryFact], top_k: usize) -> Vec<MemoryFact> {
    if facts.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let docs: Vec<crate::retrieve::Document> = facts
        .iter()
        .map(|f| crate::retrieve::Document {
            id: f.id.clone(),
            source: f.owner.clone(),
            heading_path: vec![format!("{} / {}", f.subject, f.predicate)],
            text: f.searchable_text(),
        })
        .collect();

    let index = crate::retrieve::Bm25Index::build(&docs);
    let hits = index.search(query, top_k);

    if hits.is_empty() {
        // ★ 零命中退化：按 importance 取前 k。
        let mut sorted: Vec<&MemoryFact> = facts.iter().collect();
        sorted.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        return sorted.into_iter().take(top_k).cloned().collect();
    }

    let by_id: std::collections::HashMap<&str, &MemoryFact> =
        facts.iter().map(|f| (f.id.as_str(), f)).collect();
    hits.into_iter()
        .filter_map(|h| by_id.get(h.id.as_str()).map(|f| (*f).clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_final_answer() {
        let out = parse_react_output("Thought: 好了\nFinal Answer: 42");
        assert_eq!(out, ParsedOutput::FinalAnswer("42".into()));
    }

    #[test]
    fn parses_action_with_args() {
        let out = parse_react_output("Action: search\nAction Input: {\"q\": \"x\"}");
        match out {
            ParsedOutput::Action { tool, args } => {
                assert_eq!(tool, "search");
                assert_eq!(args["q"], "x");
            }
            other => panic!("应该是 Action，实际是 {other:?}"),
        }
    }

    #[test]
    fn takes_the_last_marker_not_the_first() {
        // 模型回显了提示词（含旧的 Action），真正的动作在后面。
        let text = "Action: old\nAction Input: {\"q\": 1}\n...继续...\nAction: new\nAction Input: {\"q\": 2}";
        match parse_react_output(text) {
            ParsedOutput::Action { tool, args } => {
                assert_eq!(tool, "new");
                assert_eq!(args["q"], 2);
            }
            other => panic!("应该是 Action，实际是 {other:?}"),
        }
    }

    #[test]
    fn tool_name_is_truncated_to_identifier() {
        let out = parse_react_output("Action:Thought:Thought:");
        match out {
            ParsedOutput::Action { tool, .. } => assert_eq!(tool, "Thought"),
            other => panic!("应该是 Action，实际是 {other:?}"),
        }
    }

    #[test]
    fn final_answer_wins_over_earlier_action() {
        let out = parse_react_output("Action: search\nAction Input: {}\nFinal Answer: 完成");
        assert_eq!(out, ParsedOutput::FinalAnswer("完成".into()));
    }

    #[test]
    fn parses_json_with_code_fence() {
        let args = parse_action_input("```json\n{\"scope\": \"cache\"}\n```");
        assert_eq!(args["scope"], "cache");
    }

    #[test]
    fn skips_placeholder_braces() {
        // `{周报}` 括号配对但不是合法 JSON。
        let args = parse_action_input("前言 {周报} 然后 {\"scope\": \"cache\"}");
        assert_eq!(args["scope"], "cache");
    }

    #[test]
    fn falls_back_to_text_arg() {
        let args = parse_action_input("随便一段话");
        assert_eq!(args["input"], "随便一段话");
    }

    #[test]
    fn calculates_arithmetic() {
        assert_eq!(calculate("1 + 2 * 3").unwrap(), 7.0);
        assert_eq!(calculate("(1 + 2) * 3").unwrap(), 9.0);
        assert_eq!(calculate("10 / 4").unwrap(), 2.5);
        assert_eq!(calculate("2 * (3 + 4) - 1").unwrap(), 13.0);
    }

    #[test]
    fn rejects_bad_expressions() {
        assert!(calculate("1 +").is_err());
        assert!(calculate("(1 + 2").is_err());
        assert!(calculate("1 / 0").is_err());
        assert!(calculate("1 2").is_err());
    }

    #[test]
    fn merge_replaces_in_place() {
        let mut state = ThreadState::new("t1");
        let msg = MemoryMessage::user("原内容");
        state.merge_messages(vec![msg.clone()]);
        state.merge_messages(vec![MemoryMessage::assistant("回答")]);

        let mut updated = msg.clone();
        updated.content = "新内容".into();
        state.merge_messages(vec![updated]);

        assert_eq!(state.message_count(), 2);
        // ★ 位置不变 —— 仍是第一条。
        assert_eq!(state.messages[0].content, "新内容");
    }

    #[test]
    fn merge_appends_new_ids() {
        let mut state = ThreadState::new("t1");
        state.merge_messages(vec![MemoryMessage::user("a")]);
        state.merge_messages(vec![MemoryMessage::user("b")]);
        assert_eq!(state.message_count(), 2);
    }

    #[test]
    fn fact_dedup_key_ignores_value() {
        let a = MemoryFact::new("用户", "名字", "Bob", "bob");
        let b = MemoryFact::new("用户", "名字", "Bobby", "bob");
        assert_eq!(a.dedup_key(), b.dedup_key());
    }

    #[test]
    fn recall_falls_back_when_bm25_misses() {
        let facts = vec![
            MemoryFact::new("用户", "名字", "Bob", "bob"),
            MemoryFact::new("用户", "偏好语言", "中文", "bob"),
        ];
        // 「我叫什么」与事实的词元交集为空 → BM25 零命中。
        let recalled = recall_facts("我叫什么", &facts, 2);
        assert_eq!(recalled.len(), 2);
    }

    #[test]
    fn recall_finds_matching_fact() {
        let facts = vec![
            MemoryFact::new("用户", "名字", "Bob", "bob"),
            MemoryFact::new("项目", "阶段", "压测", "team"),
        ];
        let recalled = recall_facts("名字 Bob", &facts, 1);
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].value, "Bob");
    }
}
