//! 缓存键构造 —— 对齐 LiteLLM 的 `Cache.get_cache_key`。
//!
//! ## 参考
//!
//! LiteLLM 文档（*Custom Cache Keys*）给出的默认实现是：
//!
//! ```python
//! def custom_get_cache_key(*args, **kwargs):
//!     return (
//!         kwargs.get("model", "")
//!         + str(kwargs.get("messages", ""))
//!         + str(kwargs.get("temperature", ""))
//!         + str(kwargs.get("logit_bias", ""))
//!     )
//! ```
//!
//! 注意它的形态：**把所有影响输出的参数拼成一个字符串**。
//! 本项目把这个思想做成 `CacheKey` —— 但**不照抄字符串拼接**，
//! 而是改成「结构化 → 规范化 JSON → SHA-256」。
//!
//! ## 为什么不用字符串拼接
//!
//! 字符串拼接有三个真实的问题：
//!
//! 1. **分隔符歧义** —— `("ab", "c")` 和 `("a", "bc")` 拼出来一样。
//!    问「北京天气」和「北京」「天气」两个问题如果只是简单拼接，
//!    有可能撞键。
//! 2. **顺序敏感 → 顺序也敏感** —— 消息顺序换了（完全不同的对话）
//!    哈希就变了，这本来是对的；但**不该变的参数**（比如缓存版本号）
//!    混在里面会让「升级缓存格式」变成「全部失效」且难以排查。
//! 3. **无法审计** —— 你看不出一个键是由什么算出来的。
//!
//! 结构化 + SHA-256 三条都解决：
//!
//! - 规范化 JSON **有明确边界**（`serde_json` 的对象序列化是确定的）
//! - 字段分组，版本号单独一段
//! - 键本体可反查（我们把「键的构成」也存进 value 里）
//!
//! ## ⚠️ 最重要的一个约束：**键必须包含所有影响输出的东西**
//!
//! 这是 LiteLLM 文档里那句
//!
//! > Exact-match caches (redis, s3, gcs, local, disk) key on a hash of the whole request,
//! > so any change to the conversation is a miss.
//!
//! 的实际含义。
//!
//! 漏掉任何一个字段，就会在**两个本该不同的请求**之间返回错答。
//! 本项目的 [`CacheKey`] 强制包含：
//!
//! | 字段 | 为什么必须有 |
//! |---|---|
//! | `model` | 不同模型的答案完全不同 |
//! | `messages` | 对话历史 —— 任何一处改动都必须 miss |
//! | `tools` | 工具定义变了，模型的行为就变了 |
//! | `tool_choice` | `auto` / `none` / `required` 的输出不同 |
//! | `temperature` | 影响采样结果 |
//! | `max_tokens` | 影响输出长度 |
//! | `disable_reasoning` | 开/关思考模式输出结构不同 |
//! | `version` | 缓存格式升级时用来**批量失效** |
//!
//! ## ⚠️ 一个容易漏的字段：**工具执行的副作用**
//!
//! 上表漏了一样东西 —— **工具的结果**。
//!
//! 假设问题「北京现在天气怎么样？」：
//! 第一轮模型输出 `tool_calls: [get_weather(北京)]`，
//! 这一轮的**请求**完全相同（同样的 messages + tools），
//! 所以键相同 —— 缓存会命中，返回**半小时前的天气**。
//!
//! 这不是 bug，这是**缓存的定义**（exact-match cache）。
//! 但它意味着：**缓存粒度决定了「答案有多旧」**。
//!
//! 本项目的做法（见 [`crate::cache::CacheService`]）：
//! **只缓存「纯文本的最终答案」那一轮**，
//! 带 `tool_calls` 的中间轮**不缓存**。
//! 理由：中间轮的缓存命中会跳过工具执行，
//! 导致「模型已经决定了调哪个工具」被冻结 ——
//! 工具结果的时效性问题还在，但它至少不缓存**模型的决策**。
//!
//! 这个取舍在 README 的「三档缓存语义」里量化对比。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use genai::chat::ChatMessage;

/// 缓存键的**格式版本**。
///
/// ## 为什么要有版本号
///
/// 缓存里存的是 JSON。任何一次「存储格式变更」都会让旧数据无法解析。
///
/// 两种处理方式：
///
/// 1. **解析失败就当 miss**（本项目采用）—— 简单，但要付出「一次解析失败」的代价。
/// 2. **版本号变更 → 键前缀变更 → 旧数据自然过期** —— 更干净。
///
/// 本项目两者都做：键里带版本号前缀，解析失败也当 miss。
/// 版本号变更后，旧的键**根本不会被查**，所以旧数据只是占着内存直到 TTL 过期。
pub const CACHE_KEY_VERSION: &str = "v1";

/// 缓存键的**默认命名空间**。
///
/// ## 命名空间的意义
///
/// LiteLLM 的 `Cache` 有 `namespace` 参数，
/// 语义缓存还有 `semantic_cache_scope`（`key` / `end_user`）。
///
/// 命名空间的作用是**分区**：不同租户 / 不同业务线的缓存互不干扰，
/// 并且可以**整块失效**（`DEL namespace:*`）。
pub const DEFAULT_NAMESPACE: &str = "agent-cache";

/// 缓存键的构成部分。
///
/// ## 为什么要拆成结构体而不是直接哈希
///
/// 因为**审计价值**：命中/未命中日志里要能说清
/// 「这个键是由哪些部分算出来的」。
/// 如果只存一个哈希串，出了「为什么没命中」的问题就只能猜。
///
/// 序列化用 `BTreeMap` 而不是 `HashMap` ——
/// `HashMap` 的迭代顺序**不确定**，同样的内容序列化两次可能得到不同的字符串，
/// 键就不稳定了。这是缓存键最经典的自伤。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheKey {
    /// 格式版本（`v1`）。
    pub version: String,

    /// 命名空间。
    pub namespace: String,

    /// 模型名。
    pub model: String,

    /// 对话消息（规范化后的形式）。
    pub messages: Vec<NormalizedMessage>,

    /// 工具定义的哈希（工具多时只存哈希，避免键无限长）。
    pub tools_digest: String,

    /// 工具选择策略。
    pub tool_choice: String,

    /// 采样温度。
    pub temperature: f64,

    /// 输出 token 上限。
    pub max_tokens: u32,

    /// 是否关闭了思考模式。
    pub disable_reasoning: bool,
}

/// 规范化后的消息。
///
/// ## 为什么要「规范化」而不是直接用 `ChatMessage`
///
/// 三个原因：
///
/// 1. **`ChatMessage` 是第三方类型** —— 它的 serde 表现（`{"parts":[...]}`）
///    是**内部实现细节**。把它焊死在缓存格式里，
///    升级 `genai` 版本就可能让全部历史缓存无法反序列化。
///    17-agent-memory 已经踩过这个坑（「不要把第三方类型的内部结构焊死在持久化格式里」）。
/// 2. **`options` 字段与缓存无关** —— 它是 per-message 的选项（如 cache control），
///    不影响输出，不该进键。
/// 3. **可比性** —— 规范化后能直接比较两段对话是否相同。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedMessage {
    /// 角色（`system` / `user` / `assistant` / `tool`）。
    pub role: String,

    /// 文本内容（多段拼接）。
    pub text: String,

    /// 工具调用（assistant 消息才有）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<NormalizedToolCall>,

    /// 思考内容（assistant 消息才有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    /// 工具结果（tool 消息才有）。
    ///
    /// ## ⚠️ 这个字段是**补上的一个真实缺陷**
    ///
    /// 原来的实现里，`normalize_message` 对 `ContentPart::ToolResponse`
    /// 落进了 `_ => {}` 分支 —— **工具结果的内容完全没进缓存键**。
    ///
    /// 症状：两个内容天差地别的 `tool` 消息，只要 `call_id` 相同，
    /// 算出来的键**完全相同**。
    ///
    /// 这直接违反了本项目自己的硬约束①
    /// 「缓存键必须包含所有影响输出的东西」——
    /// 工具结果显然影响输出（下一轮模型的输入就是它）。
    ///
    /// ⚠️ 注意：补上这个字段**不会**让 agentic 链路的命中率变成非零
    /// —— 因为 `call_id` 仍然每次都变（那是另一个原因，见 `call_id` 字段）。
    /// 补它的意义是**正确性**：不能让「不同输入共享同一个答案」。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_results: Vec<NormalizedToolResult>,
}

/// 规范化后的工具结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedToolResult {
    /// 对应的调用 ID（用于对账）。
    pub call_id: String,

    /// 工具名。
    pub name: String,

    /// 工具返回的正文。
    pub content: String,
}

/// 规范化后的工具调用。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedToolCall {
    /// 调用 ID。
    ///
    /// ⚠️ **必须进键** —— 它的值随模型输出变化，
    /// 所以「同一个问题」两次调用产生的 `call_id` 不同，
    /// 这也意味着**带工具调用的那一轮几乎不会命中缓存**。
    /// 这正是我们「不缓存中间轮」的技术理由之一。
    pub call_id: String,

    /// 工具名。
    pub name: String,

    /// 参数。
    pub arguments: Value,
}

impl CacheKey {
    /// 构造一个缓存键。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        namespace: impl Into<String>,
        model: impl Into<String>,
        messages: &[ChatMessage],
        tools_digest: impl Into<String>,
        tool_choice: impl Into<String>,
        temperature: f64,
        max_tokens: u32,
        disable_reasoning: bool,
    ) -> Self {
        Self {
            version: CACHE_KEY_VERSION.to_string(),
            namespace: namespace.into(),
            model: model.into(),
            messages: messages.iter().map(normalize_message).collect(),
            tools_digest: tools_digest.into(),
            tool_choice: tool_choice.into(),
            temperature,
            max_tokens,
            disable_reasoning,
        }
    }

    /// 计算成最终的 Redis key。
    ///
    /// 形态：`{namespace}:{version}:{sha256前16位}`
    ///
    /// ## 为什么只取前 16 位
    ///
    /// 64 位十六进制（SHA-256 全长）作 key 太长，Redis key 和日志都难读。
    /// 16 位 = 64 bit，碰撞概率在「单机缓存」场景下可忽略
    /// （生日界：10 万个键时碰撞概率约 2.7e-10）。
    ///
    /// 如果要跑多机 Redis 集群、键数量到千万级，才需要加长。
    pub fn redis_key(&self) -> String {
        let digest = self.digest();
        format!("{}:{}:{}", self.namespace, self.version, &digest[..16])
    }

    /// 内容的 SHA-256 十六进制摘要。
    ///
    /// ## ⚠️ 序列化必须**确定**
    ///
    /// 用 `serde_json::to_string` 对一个 `struct`（字段顺序固定）序列化是确定的。
    /// 但如果中间塞了 `HashMap`，顺序就不确定了 —— 所以 [`CacheKey`] 里
    /// 所有集合都是 `Vec` / `BTreeMap`。
    pub fn digest(&self) -> String {
        let json = serde_json::to_string(self).expect("缓存键序列化不会失败");
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        let out = hasher.finalize();
        out.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// 键的**人类可读摘要**（用于日志与 API 响应）。
    ///
    /// 只展示「哪些维度不同」，不展示完整内容 ——
    /// 完整内容可能包含用户的隐私数据。
    pub fn describe(&self) -> String {
        format!(
            "ns={} model={} msgs={} tools={} choice={} temp={} max_tokens={}",
            self.namespace,
            self.model,
            self.messages.len(),
            &self.tools_digest[..self.tools_digest.len().min(8)],
            self.tool_choice,
            self.temperature,
            self.max_tokens
        )
    }
}

/// 把 `ChatMessage` 规范化。
fn normalize_message(msg: &ChatMessage) -> NormalizedMessage {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();
    let mut reasoning = None;

    for part in msg.content.parts() {
        match part {
            genai::chat::ContentPart::Text(t) => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
            }
            genai::chat::ContentPart::ReasoningContent(r) => {
                reasoning = Some(r.clone());
            }
            genai::chat::ContentPart::ToolCall(call) => {
                tool_calls.push(NormalizedToolCall {
                    call_id: call.call_id.clone(),
                    name: call.fn_name.clone(),
                    arguments: call.fn_arguments.clone(),
                });
            }
            // ⚠️ **工具结果必须进键** ——
            //
            // 原来这里落进 `_ => {}`，导致工具结果的内容对缓存键**完全不可见**：
            // 两个内容不同、`call_id` 相同的 `tool` 消息会算出**同一个键**。
            // 那意味着「不同输入 → 同一个答案」，直接违反硬约束①。
            genai::chat::ContentPart::ToolResponse(resp) => {
                tool_results.push(NormalizedToolResult {
                    call_id: resp.call_id.clone(),
                    name: resp.fn_name.clone().unwrap_or_default(),
                    content: resp.content.clone(),
                });
            }
            _ => {}
        }
    }

    NormalizedMessage {
        role: format!("{:?}", msg.role).to_lowercase(),
        text,
        tool_calls,
        tool_results,
        reasoning,
    }
}

/// 计算一组工具定义的摘要。
///
/// ## 为什么工具只存摘要不存全文
///
/// 工具定义可能有几百行 Schema。放进键里会让键变得很长，
/// 而且每次算哈希都要重新序列化一遍（浪费）。
///
/// 用摘要的代价是：**极小概率的碰撞**（两个不同的工具集摘要相同）。
/// 64 位摘要下的碰撞概率可以忽略。
pub fn tools_digest(specs: &[agent_16::tools::ToolSpec]) -> String {
    // 用 `BTreeMap` 而不是 `Vec` —— 保证顺序无关。
    //
    // ⚠️ 工具的**顺序**其实不影响模型行为（`tools` 字段是个数组，
    // 但模型看到的是「有哪些工具」）。所以按名字排序是对的 ——
    // 否则同一个工具集换个顺序就 miss，白白损失命中率。
    let map: BTreeMap<&str, &Value> = specs
        .iter()
        .map(|spec| (spec.name.as_str(), &spec.schema))
        .collect();

    let json = serde_json::to_string(&map).expect("工具定义序列化不会失败");
    let mut hasher = Sha256::new();
    hasher.update(json.as_bytes());
    let out = hasher.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use genai::chat::MessageContent;
    use serde_json::json;

    fn user(text: &str) -> ChatMessage {
        ChatMessage::user(MessageContent::from_text(text.to_string()))
    }

    fn key_for(messages: &[ChatMessage]) -> CacheKey {
        CacheKey::new(
            DEFAULT_NAMESPACE,
            "deepseek-flash",
            messages,
            tools_digest(&[]),
            "auto",
            0.0,
            2048,
            true,
        )
    }

    #[test]
    fn same_input_gives_same_key() {
        let a = key_for(&[user("你好")]);
        let b = key_for(&[user("你好")]);
        assert_eq!(a.redis_key(), b.redis_key());
    }

    #[test]
    fn different_question_gives_different_key() {
        let a = key_for(&[user("你好")]);
        let b = key_for(&[user("再见")]);
        assert_ne!(a.redis_key(), b.redis_key());
    }

    #[test]
    fn different_model_gives_different_key() {
        let a = key_for(&[user("你好")]);
        let mut b = a.clone();
        b.model = "gpt-4o".to_string();
        assert_ne!(a.redis_key(), b.redis_key());
    }

    #[test]
    fn different_temperature_gives_different_key() {
        let a = key_for(&[user("你好")]);
        let mut b = a.clone();
        b.temperature = 0.7;
        assert_ne!(a.redis_key(), b.redis_key());
    }

    #[test]
    fn different_tool_choice_gives_different_key() {
        let a = key_for(&[user("你好")]);
        let mut b = a.clone();
        b.tool_choice = "required".to_string();
        assert_ne!(a.redis_key(), b.redis_key());
    }

    #[test]
    fn message_order_matters() {
        let a = key_for(&[user("第一"), user("第二")]);
        let b = key_for(&[user("第二"), user("第一")]);
        assert_ne!(a.redis_key(), b.redis_key());
    }

    #[test]
    fn key_is_stable_across_calls() {
        // ⚠️ 键必须是**内容寻址** —— 重启进程后同样的输入要得到同样的键，
        //    否则缓存永远命中不了。
        let first = key_for(&[user("稳定性测试")]).redis_key();
        let second = key_for(&[user("稳定性测试")]).redis_key();
        assert_eq!(first, second);
        assert!(first.starts_with("agent-cache:v1:"));
    }

    #[test]
    fn namespace_changes_the_key() {
        let a = key_for(&[user("你好")]);
        let mut b = a.clone();
        b.namespace = "other".to_string();
        assert_ne!(a.redis_key(), b.redis_key());
    }

    #[test]
    fn version_changes_the_key() {
        // 版本号变更 → 旧缓存自然失效（键前缀变了）。
        let a = key_for(&[user("你好")]);
        let mut b = a.clone();
        b.version = "v2".to_string();
        assert_ne!(a.redis_key(), b.redis_key());
    }

    #[test]
    fn tools_digest_is_order_independent() {
        let spec_a = agent_16::tools::ToolSpec::new("a", "d", json!({"type": "object"}));
        let spec_b = agent_16::tools::ToolSpec::new("b", "d", json!({"type": "object"}));

        let d1 = tools_digest(&[spec_a.clone(), spec_b.clone()]);
        let d2 = tools_digest(&[spec_b, spec_a]);
        assert_eq!(d1, d2);
    }

    #[test]
    fn tools_digest_changes_with_schema() {
        let s1 = agent_16::tools::ToolSpec::new("a", "d", json!({"type": "object"}));
        let s2 = agent_16::tools::ToolSpec::new("a", "d", json!({"type": "object", "x": 1}));
        assert_ne!(tools_digest(&[s1]), tools_digest(&[s2]));
    }

    #[test]
    fn tool_call_id_is_part_of_the_key() {
        // ⚠️ call_id 随模型输出变化 —— 这让「带工具调用的轮」几乎必然 miss。
        //    这正是「不缓存中间轮」的技术理由。
        let assistant_msg = |call_id: &str| {
            let call = genai::chat::ToolCall {
                call_id: call_id.to_string(),
                fn_name: "get_weather".to_string(),
                fn_arguments: json!({"location": "北京"}),
                thought_signatures: None,
            };
            ChatMessage::assistant(MessageContent::from_parts(vec![
                genai::chat::ContentPart::ToolCall(call),
            ]))
        };

        let a = key_for(&[user("q"), assistant_msg("call_1")]);
        let b = key_for(&[user("q"), assistant_msg("call_2")]);
        assert_ne!(a.redis_key(), b.redis_key());
    }

    #[test]
    fn tool_result_content_is_part_of_the_key() {
        // ⚠️ **回归测试**：修复前 `ContentPart::ToolResponse` 落进了 `_ => {}`，
        //    工具结果的内容**完全没进缓存键** ——
        //    两个内容不同、`call_id` 相同的 `tool` 消息会算出**同一个键**，
        //    违反硬约束①「键必须包含所有影响输出的东西」。
        let tool_msg = |call_id: &str, content: &str| {
            ChatMessage::tool(genai::chat::MessageContent::from_parts(vec![
                genai::chat::ContentPart::ToolResponse(genai::chat::ToolResponse::new(
                    call_id, content,
                )),
            ]))
        };

        let a = key_for(&[tool_msg("call_1", "搜索结果 A")]);
        let b = key_for(&[tool_msg("call_1", "完全不同的搜索结果 B")]);

        assert_ne!(
            a.redis_key(),
            b.redis_key(),
            "工具结果内容不同，缓存键必须不同"
        );

        // 内容相同 → 键相同（不能把键算成永远不稳定的）。
        let a2 = key_for(&[tool_msg("call_1", "搜索结果 A")]);
        assert_eq!(a.redis_key(), a2.redis_key());
    }

    #[test]
    fn describe_mentions_key_dimensions() {
        let text = key_for(&[user("你好")]).describe();
        assert!(text.contains("model=deepseek-flash"));
        assert!(text.contains("msgs=1"));
    }
}
