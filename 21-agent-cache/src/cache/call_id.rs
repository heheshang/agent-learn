//! 确定性 `call_id` —— 让 agentic 链路也能命中精确匹配缓存。
//!
//! ## 要解决的问题
//!
//! 本项目实测出的核心结论是：
//! **精确匹配缓存在 agentic 链路上必然 miss**。
//!
//! 根因是缓存键里含有 `call_id` —— 模型每次生成的工具调用 ID 都不同
//! （`call_abc123` vs `call_9f8e7d6c`），
//! 于是「同一个问题跑两次」算出来的键完全不同。
//!
//! ## ⚠️ 但「不能去掉 `call_id`」这个说法需要**精确化**
//!
//! 16 的坑 1 说：「`call_id` 是 tool 消息回执的**对账依据**，
//! 少一条回执整个请求就 400」。
//!
//! 这条约束的准确范围是：
//! **在**同一次请求内**，assistant 消息的 `tool_call_id`
//! 与 tool 消息的 `tool_call_id` 必须一一对应。
//!
//! 它**并不要求** `call_id` 跨请求唯一 —— 更**不要求**它跨请求随机。
//!
//! 所以有第三条路：**不是去掉，而是重写**。
//! 把 `call_id` 换成一个**由内容派生**的确定性值：
//!
//! ```text
//! call_id = "call_" + sha256(步数 | 工具名 | 规范化参数 | 同参序号)[..16]
//! ```
//!
//! ## 为什么这样是**安全的**
//!
//! 1. **对账依然成立** —— 我们把 assistant 消息和 tool 消息**一起**重写，
//!    两边用的是同一个新 ID，配对关系不变。
//! 2. **比「从键里剔除」更正确** —— 如果只是从**键**里剔除 `call_id`
//!    （消息本身还带原始 ID），那模型看到的 prompt 里 ID 文本仍在变化，
//!    理论上可能影响输出。而**重写消息本身**让模型每次看到的是**同一个** ID，
//!    请求变得**完全确定**。
//! 3. **ID 格式合法** —— 用 `call_<16 位十六进制>`，
//!    符合 DeepSeek / OpenAI 的 `call_` 前缀惯例。
//!
//! ## ⚠️ 为什么不把步数去掉
//!
//! 步数参与派生，是为了让**同一条会话里**不同步的调用 ID 不撞车。
//! 撞车不会立刻报错（provider 只要求「最近的 assistant 消息」能配上），
//! 但会制造出「历史里有多个同 ID 的 tool_call」的畸形请求 ——
//! 那种请求的语义是模糊的，排查起来极其痛苦。
//!
//! ## ⚠️ 同参序号的必要性
//!
//! 模型可能**并行**发出多个参数完全相同的调用（例如连查两次同一个工具）。
//! 这时它们的 ID 必须不同，否则请求里会出现两个同 ID 的 `tool_call`
//! 和两条同 ID 的 tool 消息 —— 同样畸形。
//!
//! 「同参序号」= 在本轮内，此前出现过多少个**完全相同**的 (工具, 参数) 组合。
//! 它只依赖**内容**，不依赖顺序 —— 所以它是稳定的。

use std::collections::BTreeMap;

use serde_json::Value;
use sha2::{Digest, Sha256};

/// 派生前缀的**版本号**。
///
/// ## ⚠️ 为什么要版本号
///
/// 派生规则一旦改变（例如加了某个字段），所有 ID 都会变，
/// 于是**所有历史缓存的键**都会变 —— 缓存全量失效（但不会出错，
/// 只是miss 一次）。
///
/// 版本号让「改规则」变成一件**显式**的事：
/// 改了就 bump，自然失效，不改就一直兼容。
pub const CALL_ID_VERSION: &str = "v1";

/// 计算规范化后的参数 JSON。
///
/// ## 为什么需要「规范化」
///
/// 模型返回的参数有两种形态（见 16 的 `normalize_arguments`）：
///
/// - `{"city": "北京"}` —— JSON 对象
/// - `"{\"city\": \"北京\"}"` —— **JSON 字符串**
///
/// 两者语义相同，但如果直接对原始值做哈希，同一个工具调用
/// 会因为模型这次返回对象、下次返回字符串而得到**不同的 ID**。
///
/// 另外 JSON 对象的**键序**也不保证稳定 ——
/// `serde_json` 的 `Map` 默认是 `BTreeMap`（有序），
/// 但开了 `preserve_order` feature 就是插入序，
/// 所以**不能**依赖它的顺序。
///
/// 因此这里**递归排序键**后重新拼接，得到一个与顺序无关的规范形式。
pub fn canonical_args(args: &Value) -> String {
    match args {
        Value::Object(map) => {
            // 递归排序 → 与键的插入顺序无关。
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let inner: Vec<String> = keys
                .iter()
                .map(|k| {
                    format!(
                        "{}:{}",
                        Value::String((*k).clone()),
                        canonical_args(&map[*k])
                    )
                })
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(canonical_args).collect();
            format!("[{}]", inner.join(","))
        }
        other => other.to_string(),
    }
}

/// 派生一个确定性的 `call_id`。
///
/// ## 参数
///
/// - `step_no` —— 第几步（从 1 开始）。**参与**派生，避免同一条会话里不同步撞 ID。
/// - `tool` —— 工具名。
/// - `canonical_args` —— 用 [`canonical_args`] 规范化过的参数 JSON。
/// - `occurrence` —— 本轮内此前出现过多少个**完全相同**的 (工具, 参数) 组合。
///
/// ## 输出
///
/// `call_<16 位十六进制>` —— 符合主流 provider 的 ID 惯例。
pub fn deterministic_call_id(
    step_no: usize,
    tool: &str,
    canonical_args: &str,
    occurrence: usize,
) -> String {
    let payload = format!(
        "{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}",
        CALL_ID_VERSION, step_no, tool, canonical_args, occurrence
    );
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    let out = hasher.finalize();
    let hex: String = out.iter().map(|b| format!("{b:02x}")).collect();
    format!("call_{}", &hex[..16])
}

/// 给一批工具调用分配确定性 ID。
///
/// ## 返回值
///
/// 与输入**一一对应**的新 `ToolCall` 列表（ID 已替换）。
///
/// ## ⚠️ 「同参序号」的实现细节
///
/// 计数键是 `(工具名, 规范化参数)` —— 只有**完全相同**的调用才会共享序号空间。
/// 参数不同的调用各自从 0 开始，所以它们的 ID **互不影响**（也就不会被模型的
/// 调用顺序影响）。
///
/// 这一点很重要：模型返回并行调用的**顺序**不保证稳定，
/// 如果 ID 依赖绝对位置，同一个问题的两次运行就可能算出不同的 ID ——
/// 那就白改了。
pub fn assign_deterministic_call_ids(
    step_no: usize,
    calls: &mut [genai::chat::ToolCall],
    args_of: impl Fn(&genai::chat::ToolCall) -> Value,
) {
    let mut seen: BTreeMap<(String, String), usize> = BTreeMap::new();

    for call in calls.iter_mut() {
        let args = args_of(call);
        let canonical = canonical_args(&args);
        let counter = seen
            .entry((call.fn_name.clone(), canonical.clone()))
            .or_insert(0);
        let occurrence = *counter;
        *counter += 1;

        call.call_id = deterministic_call_id(step_no, &call.fn_name, &canonical, occurrence);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn same_input_gives_same_id() {
        let a = deterministic_call_id(1, "search", "{\"q\":\"x\"}", 0);
        let b = deterministic_call_id(1, "search", "{\"q\":\"x\"}", 0);
        assert_eq!(a, b);
        assert!(a.starts_with("call_"));
        assert_eq!(a.len(), "call_".len() + 16);
    }

    #[test]
    fn different_step_gives_different_id() {
        // ⚠️ 步数必须参与派生 —— 否则同一条会话里不同步会撞 ID。
        let a = deterministic_call_id(1, "search", "{\"q\":\"x\"}", 0);
        let b = deterministic_call_id(2, "search", "{\"q\":\"x\"}", 0);
        assert_ne!(a, b);
    }

    #[test]
    fn different_tool_gives_different_id() {
        let a = deterministic_call_id(1, "search", "{}", 0);
        let b = deterministic_call_id(1, "calculate", "{}", 0);
        assert_ne!(a, b);
    }

    #[test]
    fn different_args_give_different_id() {
        let a = deterministic_call_id(1, "search", "{\"q\":\"x\"}", 0);
        let b = deterministic_call_id(1, "search", "{\"q\":\"y\"}", 0);
        assert_ne!(a, b);
    }

    #[test]
    fn occurrence_disambiguates_identical_calls() {
        // ⚠️ 并行发出两个完全相同的调用时，ID 必须不同。
        let a = deterministic_call_id(1, "search", "{\"q\":\"x\"}", 0);
        let b = deterministic_call_id(1, "search", "{\"q\":\"x\"}", 1);
        assert_ne!(a, b);
    }

    #[test]
    fn canonical_args_is_key_order_independent() {
        let a = canonical_args(&json!({"a": 1, "b": 2}));
        let b = canonical_args(&json!({"b": 2, "a": 1}));
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_args_is_nested_key_order_independent() {
        let a = canonical_args(&json!({"outer": {"x": 1, "y": 2}}));
        let b = canonical_args(&json!({"outer": {"y": 2, "x": 1}}));
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_args_preserves_array_order() {
        // ⚠️ 数组的顺序**是**语义的一部分，不能排序。
        let a = canonical_args(&json!({"list": [1, 2]}));
        let b = canonical_args(&json!({"list": [2, 1]}));
        assert_ne!(a, b);
    }

    #[test]
    fn canonical_args_handles_string_form() {
        // 模型有时把 arguments 返回成 JSON 字符串。
        // 调用方应该先用 `normalize_arguments` 规整 —— 这里只验证函数本身不 panic。
        let s = canonical_args(&Value::String("{\"a\":1}".to_string()));
        assert!(s.contains("a"));
    }
}
