//! 会话状态模型 —— 「消息状态 + thread_id + checkpointer」的三件套。
//!
//! ## 对应 LangChain 的什么
//!
//! LangChain 的 [short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
//! 文档里，短期记忆就是三样东西：
//!
//! | LangChain 概念 | 本项目的对应物 |
//! |---|---|
//! | `AgentState.messages` | [`ThreadState::messages`]（消息数组） |
//! | `configurable.thread_id` | [`ThreadId`]（线程标识） |
//! | `checkpointer` | [`Checkpointer`] trait（状态持久化） |
//!
//! ## 三个必须做对的地方
//!
//! ### ① 消息状态必须**可序列化**
//!
//! checkpointer 要把状态写进数据库 / 文件，所以状态里的东西
//! **必须能序列化成 JSON**。
//!
//! ⚠️ 这里用**自己的** [`MemoryMessage`] 类型，而不是直接存 `genai::ChatMessage` ——
//! 因为 `genai::ChatMessage` 的 `MessageContent` 内部结构
//! （`Vec<ContentPart>`）序列化后是 `{"parts": [...]}` 这种嵌套形态，
//! 读回来很难看，而且把第三方类型的内部结构**焊死**在持久化格式里很危险。
//!
//! 自己的类型还带来一个好处：**换 LLM 库不用改数据库**。
//!
//! ### ② 线程隔离必须在**存储层**做
//!
//! 不能「先查出所有消息再在内存里过滤 thread_id」——
//! 那既慢又容易漏。本项目的 [`Checkpointer`] 实现**按 thread_id 分键存储**，
//! 读的时候直接按 key 取，物理上不可能串台。
//!
//! ### ③ 状态更新必须是**追加式**的
//!
//! LangGraph 的 `add_messages` reducer 是「按 id 追加或替换」。
//! 本项目用同样的语义：每条消息有唯一 id，
//! 写入时**同 id 替换、异 id 追加**。
//!
//! 这样「删除消息」（LangChain 的 `RemoveMessage`）就能自然实现：
//! 只要重放一遍不含被删消息的历史即可。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use genai::chat::{ChatMessage, ChatRole, ContentPart, MessageContent};

/// 消息的角色。
///
/// ## 为什么不用 `genai::ChatRole`
///
/// `genai::ChatRole` 只有四个变体（System / User / Assistant / Tool），
/// 但**短期记忆**还需要区分「这条消息是用户说的还是助手说的」之外的东西：
/// 比如「这是从长期记忆里召回的事实」—— 它不是用户也不是助手，
/// 而是一段**注入到上下文里的背景**。
///
/// LangChain 的做法是给消息加 `additional_kwargs`（额外元数据）。
/// 本项目用 `MemoryRole::Fact` 显式表达这件事 ——
///
/// ```text
/// system   —— 身份与规则
/// user     —— 用户输入
/// assistant —— 助手回复
/// tool    —— 工具回执
/// fact    —— ★ 长期记忆召回的事实（注入到上下文，不是对话的一部分）
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRole {
    /// 系统提示（身份、规则）。
    System,
    /// 用户输入。
    User,
    /// 助手回复。
    Assistant,
    /// 工具回执。
    Tool,
    /// 从长期记忆召回的事实。
    Fact,
}

impl MemoryRole {
    /// 角色名。
    pub fn name(&self) -> &'static str {
        match self {
            MemoryRole::System => "system",
            MemoryRole::User => "user",
            MemoryRole::Assistant => "assistant",
            MemoryRole::Tool => "tool",
            MemoryRole::Fact => "fact",
        }
    }

    /// 转成 `genai` 的角色。
    ///
    /// ## ⚠️ `Fact` 映射成 `System`
    ///
    /// 召回的事实要作为「背景信息」喂给模型，
    /// 而 `genai` 的 `ChatRole` 没有 `fact` 这个角色。
    ///
    /// 映射成 `System` 是安全的：事实是**权威上下文**，
    /// 不是「某人在说话」。用 `User` 反而会让模型以为那是用户的诉求。
    pub fn to_genai(&self) -> ChatRole {
        match self {
            MemoryRole::System | MemoryRole::Fact => ChatRole::System,
            MemoryRole::User => ChatRole::User,
            MemoryRole::Assistant => ChatRole::Assistant,
            MemoryRole::Tool => ChatRole::Tool,
        }
    }

    /// 从 `genai` 的角色反推。
    ///
    /// ⚠️ `System` 有歧义 —— 可能是真的系统提示，
    /// 也可能是被映射过来的 `Fact`。
    ///
    /// 这里**统一映射成 `System`**（不是 `Fact`），
    /// 因为从 checkpoint 恢复时我们分不清。
    /// `Fact` 的语义只在「注入时」有意义（那一轮知道它是 fact），
    /// 恢复之后它就是一段普通的系统级上下文。
    pub fn from_genai(role: &ChatRole) -> Self {
        match role {
            ChatRole::System => MemoryRole::System,
            ChatRole::User => MemoryRole::User,
            ChatRole::Assistant => MemoryRole::Assistant,
            ChatRole::Tool => MemoryRole::Tool,
        }
    }
}

/// 一条会话消息。
///
/// ## 为什么自己定义而不直接用 `genai::ChatMessage`
///
/// 1. **持久化格式稳定** —— checkpointer 写的是 JSON，
///    把第三方类型的内部结构（`{"parts":[...]}`）焊死进数据库很危险：
///    升级依赖就可能读不出历史。
/// 2. **换 LLM 库不用迁移数据** —— 状态模型是**我们的**，不是供应商的。
/// 3. **能表达 `genai` 表达不了的东西** —— 比如 `Fact` 角色。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMessage {
    /// 消息的唯一 id。
    ///
    /// ## 为什么要 id
    ///
    /// LangGraph 的 `add_messages` reducer 靠 id 判断
    /// 「是追加新消息」还是「替换已有消息」。
    /// 有了 id，「删除一条消息」就退化成「重放不含它的历史」。
    pub id: String,

    /// 角色。
    pub role: MemoryRole,

    /// 正文。
    pub content: String,

    /// 工具调用 id（仅 `tool` 角色有）。
    ///
    /// ⚠️ **必须有它** —— Function Calling 协议要求
    /// 每条 `tool` 消息的 `tool_call_id` 与前面的 `tool_call` 对上，
    /// 少一条就是整个请求 400（16 项目实测踩过）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// 工具名（仅 `tool` 角色有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,

    /// 思考内容（推理模型才有）。
    ///
    /// ⚠️ 同样**必须回传** —— 开了思考模式后不回传，
    /// DeepSeek 会 400（`must be passed back to the API`）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl MemoryMessage {
    /// 构造一条消息（自动生成 id）。
    pub fn new(role: MemoryRole, content: impl Into<String>) -> Self {
        Self {
            id: new_message_id(),
            role,
            content: content.into(),
            tool_call_id: None,
            tool_name: None,
            reasoning: None,
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

    /// 系统消息。
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MemoryRole::System, content)
    }

    /// 长期记忆召回的事实。
    pub fn fact(content: impl Into<String>) -> Self {
        Self::new(MemoryRole::Fact, content)
    }

    /// 工具回执。
    ///
    /// ⚠️ `tool_call_id` 是**必填** —— 没有它，
    /// 模型下一轮会收到「insufficient tool messages」而 400。
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            tool_call_id: Some(tool_call_id.into()),
            tool_name: Some(tool_name.into()),
            ..Self::new(MemoryRole::Tool, content)
        }
    }

    /// 挂上思考内容。
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// 挂上工具调用 id。
    pub fn with_tool_call_id(mut self, call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(call_id.into());
        self
    }

    /// 消息 id。
    pub fn id(&self) -> &str {
        &self.id
    }

    /// 角色。
    pub fn role(&self) -> MemoryRole {
        self.role
    }

    /// 正文。
    pub fn content(&self) -> &str {
        &self.content
    }

    /// 转成 `genai` 的消息。
    ///
    /// ## ⚠️ 三个必须做对的细节
    ///
    /// 1. **思考内容必须在工具调用之前** —— 顺序错了 DeepSeek 会 400。
    /// 2. **`tool` 消息必须带 `tool_call_id`** —— 靠 [`ToolResponse`] 承载。
    /// 3. **`Fact` 映射成 system** —— 见 [`MemoryRole::to_genai`]。
    pub fn to_genai(&self) -> ChatMessage {
        match self.role {
            MemoryRole::Tool => {
                // 工具回执 —— 用 `ToolResponse` 承载 `tool_call_id`。
                let response = genai::chat::ToolResponse {
                    call_id: self.tool_call_id.clone().unwrap_or_default(),
                    fn_name: self.tool_name.clone(),
                    content: self.content.clone(),
                    parts: None,
                };
                ChatMessage::tool(MessageContent::from_tool_responses(vec![response]))
            }
            _ => {
                let mut parts: Vec<ContentPart> = Vec::new();

                // 思考内容在前。
                if let Some(reasoning) = &self.reasoning
                    && !reasoning.is_empty()
                {
                    parts.push(ContentPart::ReasoningContent(reasoning.clone()));
                }

                if !self.content.is_empty() {
                    parts.push(ContentPart::Text(self.content.clone()));
                }

                if parts.is_empty() {
                    // 空的助手消息（模型只调了工具没说话）——
                    // 塞一个空文本，否则某些适配器会把它丢掉。
                    parts.push(ContentPart::Text(String::new()));
                }

                ChatMessage {
                    role: self.role.to_genai(),
                    content: MessageContent::from_parts(parts),
                    options: None,
                }
            }
        }
    }
}

/// 生成一个消息 id。
///
/// ## 为什么要自己生成
///
/// `genai` 没有提供消息 id（它不持久化消息）。
/// 而 LangGraph 的 `add_messages` reducer **依赖 id** 来判断
/// 「追加」还是「替换」。
///
/// ## id 的构成
///
/// ```text
/// msg_<序号>_<单调递增计数>
/// ```
///
/// 用**单调递增计数**而不是随机数，是因为：
///
/// - 同一线程内**天然唯一**（计数器不回绕）；
/// - **可排序** —— 消息按 id 排序 = 按插入顺序排序；
/// - **不泄漏信息** —— 随机 uuid 会在日志里占地方且不可读。
///
/// ⚠️ 跨线程可能重复（每个线程一个计数器）——
/// 但这没关系，因为 id 只在**单个线程内**用于去重。
pub fn new_message_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("msg_{n:08}")
}

/// 一个线程的会话状态。
///
/// ## 对应 LangChain 的 `AgentState`
///
/// LangChain 的 `AgentState` 有一个 `messages` 字段（用 `add_messages` reducer 管理）
/// 以及用户自定义的扩展字段。本项目的 [`ThreadState`] 对应前者，
/// 扩展字段用 `extra`（一个 JSON 对象）承载 —— 和 LangChain 的
/// `state_schema` 扩展机制一致。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreadState {
    /// 线程 id。
    pub thread_id: String,

    /// 消息历史（按插入顺序）。
    ///
    /// ## ⚠️ 为什么这里**不**做去重
    ///
    /// LangGraph 的 `add_messages` reducer 会在**写入时**去重
    /// （同 id 替换、异 id 追加）。
    ///
    /// 本项目把去重逻辑放在 [`ThreadState::merge_messages`] 里 ——
    /// 因为**读**的时候去重会导致「删了又加回来」的诡异行为。
    pub messages: Vec<MemoryMessage>,

    /// 用户自定义的扩展状态。
    ///
    /// 对应 LangChain 里 `class CustomAgentState(AgentState): user_id: str`。
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,

    /// 状态版本号（每次写入 +1）。
    ///
    /// ## 用途
    ///
    /// 用来检测**并发写冲突** —— 两个请求同时写同一个线程时，
    /// 后写的会覆盖先写的。版本号让这种情况能被观测到。
    #[serde(default)]
    pub version: u64,
}

impl ThreadState {
    /// 创建一个空线程状态。
    pub fn new(thread_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            messages: Vec::new(),
            extra: HashMap::new(),
            version: 0,
        }
    }

    /// 合并一批新消息（LangGraph `add_messages` reducer 的语义）。
    ///
    /// ## 规则
    ///
    /// - **同 id** → 替换旧的那条（**原地替换，保持位置不变**）
    /// - **异 id** → 追加到末尾
    ///
    /// ## ⚠️ 为什么「替换要原地」而不是「删了再加」
    ///
    /// 因为位置**有语义** —— 消息顺序就是对话顺序。
    /// 如果替换时把消息挪到末尾，工具调用和回执的相对顺序就乱了，
    /// 而 Function Calling 协议**要求** `tool` 消息紧跟在对应的
    /// `assistant(tool_calls)` 后面。
    pub fn merge_messages(&mut self, incoming: Vec<MemoryMessage>) {
        for msg in incoming {
            if let Some(pos) = self.messages.iter().position(|m| m.id == msg.id) {
                self.messages[pos] = msg;
            } else {
                self.messages.push(msg);
            }
        }
    }

    /// 删除指定 id 的消息（LangChain `RemoveMessage` 的语义）。
    ///
    /// ## 返回值
    ///
    /// 被删掉的消息（如果存在）。
    pub fn remove_message(&mut self, id: &str) -> Option<MemoryMessage> {
        let pos = self.messages.iter().position(|m| m.id == id)?;
        Some(self.messages.remove(pos))
    }

    /// 清空消息历史（`REMOVE_ALL_MESSAGES`）。
    ///
    /// ⚠️ **系统提示会被一起清掉** —— 这是 `REMOVE_ALL_MESSAGES` 的语义。
    /// 所以本项目在「清空」后**不自动补回**系统提示：
    /// 调用方（[`crate::agent`]）在每次组装上下文时会重新加系统提示，
    /// 所以清空是安全的。
    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }

    /// 消息条数。
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// 取最近 N 条消息（不含系统提示）。
    ///
    /// ## 对应 LangChain 的 `trim_messages`
    ///
    /// LangChain 的 trim 策略有 `keep=("messages", 20)` 这种。
    /// 本项目实现的是**按条数**的最简版本 ——
    /// 保留系统提示 + 最近 N 条非系统消息。
    pub fn recent(&self, keep: usize) -> Vec<MemoryMessage> {
        let mut out: Vec<MemoryMessage> = Vec::new();

        // 系统提示永远保留 —— 它是身份，不是历史。
        for msg in self
            .messages
            .iter()
            .filter(|m| m.role == MemoryRole::System)
        {
            out.push(msg.clone());
        }

        let non_system: Vec<&MemoryMessage> = self
            .messages
            .iter()
            .filter(|m| m.role != MemoryRole::System)
            .collect();

        let start = non_system.len().saturating_sub(keep);
        for msg in &non_system[start..] {
            out.push((*msg).clone());
        }

        out
    }
}

/// 线程标识。
///
/// ## 为什么用 newtype 而不是裸 `String`
///
/// `thread_id` 会出现在**很多**函数的签名里
/// （checkpointer 的所有方法、agent 的入口、API 的参数）。
/// 用裸 `String` 的话，下面这个 bug 编译器抓不到：
///
/// ```ignore
/// checkpointer.save(&state, &some_other_string)  // 传错了线程
/// ```
///
/// 用 newtype 后，编译器会要求显式转换，意图更清楚。
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ThreadId(String);

impl ThreadId {
    /// 创建线程 id。
    ///
    /// ## 校验规则
    ///
    /// - 非空
    /// - 不超过 128 字符
    /// - 只允许字母、数字、`-`、`_`、`.`
    ///
    /// ## ⚠️ 为什么限制字符集
    ///
    /// 因为线程 id 会**成为存储的 key**（文件名 / 数据库主键）。
    /// 如果允许 `/` 之类的字符，就可能出现**路径穿越**：
    /// `thread_id = "../../etc/passwd"`。
    ///
    /// 这个约束必须在**入口**就卡住 —— 不能等到存储层才发现。
    pub fn new(raw: impl Into<String>) -> Result<Self, String> {
        let raw = raw.into();
        let trimmed = raw.trim();

        if trimmed.is_empty() {
            return Err("thread_id 不能为空".to_string());
        }

        if trimmed.chars().count() > 128 {
            return Err("thread_id 过长（最多 128 个字符）".to_string());
        }

        if !trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        {
            return Err(
                "thread_id 只能包含字母、数字、减号、下划线和点（避免路径穿越）".to_string(),
            );
        }

        Ok(Self(trimmed.to_string()))
    }

    /// 内部字符串。
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// 消耗自身返回内部字符串。
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for ThreadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ThreadId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_ids_are_unique_and_monotonic() {
        let a = new_message_id();
        let b = new_message_id();
        assert_ne!(a, b);
        // 单调递增 —— 数字部分必须比前一个大。
        let na: u64 = a.trim_start_matches("msg_").parse().unwrap();
        let nb: u64 = b.trim_start_matches("msg_").parse().unwrap();
        assert!(nb > na);
    }

    #[test]
    fn roles_map_to_genai() {
        assert_eq!(MemoryRole::User.to_genai(), ChatRole::User);
        assert_eq!(MemoryRole::Assistant.to_genai(), ChatRole::Assistant);
        assert_eq!(MemoryRole::Tool.to_genai(), ChatRole::Tool);
        // ⚠️ Fact 映射成 system（它是背景，不是用户诉求）。
        assert_eq!(MemoryRole::Fact.to_genai(), ChatRole::System);
        assert_eq!(MemoryRole::System.to_genai(), ChatRole::System);
    }

    #[test]
    fn fact_role_is_preserved_in_our_own_type() {
        let msg = MemoryMessage::fact("用户叫 Bob");
        assert_eq!(msg.role, MemoryRole::Fact);
        assert_eq!(msg.role.name(), "fact");
    }

    #[test]
    fn tool_messages_carry_the_call_id() {
        let msg = MemoryMessage::tool_result("call_1", "get_weather", "晴");
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(msg.tool_name.as_deref(), Some("get_weather"));
        assert_eq!(msg.role, MemoryRole::Tool);
    }

    #[test]
    fn converts_to_genai_message() {
        let msg = MemoryMessage::user("你好");
        let genai_msg = msg.to_genai();

        assert_eq!(genai_msg.role, ChatRole::User);
        let parts: Vec<_> = genai_msg.content.into_iter().collect();
        assert_eq!(parts.len(), 1);
        assert!(matches!(parts[0], ContentPart::Text(_)));
    }

    #[test]
    fn reasoning_comes_before_text_in_genai() {
        // ⚠️ 16 实测：思考内容顺序错了 DeepSeek 会 400。
        let msg = MemoryMessage::assistant("答案").with_reasoning("思考中");
        let genai_msg = msg.to_genai();
        let parts: Vec<_> = genai_msg.content.into_iter().collect();

        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], ContentPart::ReasoningContent(_)));
        assert!(matches!(parts[1], ContentPart::Text(_)));
    }

    #[test]
    fn merge_replaces_in_place_by_id() {
        let mut state = ThreadState::new("t1");
        let mut msg = MemoryMessage::user("第一");
        msg.id = "m1".to_string();
        state.merge_messages(vec![msg]);

        let mut second = MemoryMessage::user("第二");
        second.id = "m2".to_string();
        state.merge_messages(vec![second]);

        assert_eq!(state.messages.len(), 2);

        // ⚠️ 同 id 替换 —— 且**位置不变**。
        let mut replaced = MemoryMessage::user("第一（改）");
        replaced.id = "m1".to_string();
        state.merge_messages(vec![replaced]);

        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.messages[0].content, "第一（改）");
        assert_eq!(state.messages[1].content, "第二");
    }

    #[test]
    fn merge_appends_new_ids() {
        let mut state = ThreadState::new("t1");
        state.merge_messages(vec![MemoryMessage::user("一")]);
        state.merge_messages(vec![MemoryMessage::assistant("二")]);
        assert_eq!(state.messages.len(), 2);
    }

    #[test]
    fn remove_message_deletes_by_id() {
        let mut state = ThreadState::new("t1");
        let mut msg = MemoryMessage::user("要删的");
        msg.id = "m1".to_string();
        state.merge_messages(vec![msg]);
        state.merge_messages(vec![MemoryMessage::assistant("保留")]);

        let removed = state.remove_message("m1");
        assert!(removed.is_some());
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].content, "保留");
        assert!(state.remove_message("不存在").is_none());
    }

    #[test]
    fn clear_messages_removes_everything() {
        let mut state = ThreadState::new("t1");
        state.merge_messages(vec![MemoryMessage::user("一")]);
        state.clear_messages();
        assert_eq!(state.message_count(), 0);
    }

    #[test]
    fn recent_keeps_system_and_tail() {
        let mut state = ThreadState::new("t1");
        state.merge_messages(vec![MemoryMessage::system("规则")]);
        for i in 0..5 {
            state.merge_messages(vec![MemoryMessage::user(format!("第 {i} 条"))]);
        }

        let recent = state.recent(2);
        // 系统提示 + 最近 2 条。
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].role, MemoryRole::System);
        assert_eq!(recent[1].content, "第 3 条");
        assert_eq!(recent[2].content, "第 4 条");
    }

    #[test]
    fn recent_handles_fewer_messages_than_keep() {
        let mut state = ThreadState::new("t1");
        state.merge_messages(vec![MemoryMessage::user("只有一条")]);
        let recent = state.recent(10);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn thread_id_validates_input() {
        assert!(ThreadId::new("thread-1").is_ok());
        assert!(ThreadId::new("thread_1.2").is_ok());
        assert!(ThreadId::new("  ").is_err());
        assert!(ThreadId::new("").is_err());
        // ⚠️ 路径穿越必须被挡住。
        assert!(ThreadId::new("../../etc/passwd").is_err());
        assert!(ThreadId::new("a/b").is_err());
        assert!(ThreadId::new("x".repeat(129)).is_err());
    }

    #[test]
    fn thread_id_trims_whitespace() {
        let id = ThreadId::new("  abc  ").unwrap();
        assert_eq!(id.as_str(), "abc");
    }
}
