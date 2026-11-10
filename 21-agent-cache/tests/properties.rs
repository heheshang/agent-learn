//! 属性测试（property-based testing）—— 21-agent-cache 的 **缓存键构造**。
//!
//! ## 缓存键为什么最需要属性测试
//!
//! 缓存键是**精确匹配缓存**的心脏。21 的硬约束①写得很直白：
//!
//! > **缓存键必须包含所有影响输出的东西**
//!
//! 漏掉任何一个字段，就会在**两个本该不同的请求**之间返回错答 ——
//! 而且症状是「答案看起来很合理」，极难排查。
//!
//! 这类 bug **没法靠单元测试穷举**：
//! 每个字段都要单独写一个测试（现有单元测试就是这么做的，
//! 见 `key.rs` 的 `different_model_gives_different_key` 等），
//! 但**漏掉一个字段**这件事本身无法被「逐字段测试」发现 ——
//! 因为没写的那个测试根本不存在。
//!
//! 属性测试换了个思路：**不测「每个字段都进键了」**，
//! 而是测「**任意两个输入，只要有一个字段不同，键就必须不同**」——
//! 用「构造两个只差一个字段的键」把这件事变成可生成的性质。
//!
//! ## 不变量清单
//!
//! | 不变量 | 含义 |
//! |---|---|
//! | `same_input_same_key` | 相同输入 → 相同键（内容寻址） |
//! | `any_field_change_changes_key` | **任一**字段变化 → 键必变（硬约束①） |
//! | `message_content_change_changes_key` | 消息正文变化 → 键必变 |
//! | `tool_result_change_changes_key` | 工具结果变化 → 键必变（21 修过的真 bug） |
//! | `key_is_valid_redis_key` | 键符合 Redis 键的形态约定 |
//! | `digest_is_stable` | 摘要是稳定的 SHA-256 十六进制串 |

use agent_21::cache::key::{CacheKey, DEFAULT_NAMESPACE};
use agent_21::cache::tools_digest;
use genai::chat::{ChatMessage, ContentPart, MessageContent, ToolCall, ToolResponse};
use proptest::prelude::*;
use serde_json::json;

/// 造一条 user 消息。
fn user(text: &str) -> ChatMessage {
    ChatMessage::user(MessageContent::from_text(text.to_string()))
}

/// 造一条带工具调用的 assistant 消息。
fn assistant_with_call(call_id: &str, name: &str, args: serde_json::Value) -> ChatMessage {
    ChatMessage::assistant(MessageContent::from_parts(vec![ContentPart::ToolCall(
        ToolCall {
            call_id: call_id.to_string(),
            fn_name: name.to_string(),
            fn_arguments: args,
            thought_signatures: None,
        },
    )]))
}

/// 造一条 tool 回执消息。
fn tool_result(call_id: &str, content: &str) -> ChatMessage {
    ChatMessage::tool(MessageContent::from_parts(vec![ContentPart::ToolResponse(
        ToolResponse::new(call_id, content),
    )]))
}

/// 造一个缓存键（除消息外的参数固定）。
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

/// 随机消息文本。
///
/// 刻意排除空串 —— 空文本与「没有文本」在规范化后可能等价，
/// 会让「内容变化 → 键必变」这条性质出现平凡反例。
fn text() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9一-鿿 ]{1,30}"
}

/// 随机工具名。
fn tool_name() -> impl Strategy<Value = String> {
    "[a-z_]{1,12}"
}

/// 随机 `call_id`。
fn call_id() -> impl Strategy<Value = String> {
    "call_[a-z0-9]{4,12}"
}

/// 随机参数。
fn args() -> impl Strategy<Value = serde_json::Value> {
    prop::sample::select(vec![
        json!({}),
        json!({"location": "北京"}),
        json!({"query": "ReAct"}),
        json!({"n": 3}),
        json!({"deep": {"nested": [1, 2, 3]}}),
    ])
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// 相同输入 → 相同键（**内容寻址**的基本要求）。
    ///
    /// ⚠️ 这条如果挂了，缓存就永远命中不了 ——
    /// 而症状是「命中率 0」，很容易被误判成「缓存不work」。
    #[test]
    fn same_input_same_key(
        msgs in prop::collection::vec(
            prop_oneof![
                text().prop_map(|t| user(&t)),
                (call_id(), tool_name(), args())
                    .prop_map(|(id, name, a)| assistant_with_call(&id, &name, a)),
                (call_id(), text())
                    .prop_map(|(id, c)| tool_result(&id, &c)),
            ],
            0..6,
        ),
    ) {
        let a = key_for(&msgs).redis_key();
        let b = key_for(&msgs).redis_key();
        prop_assert_eq!(a, b, "相同输入必须得到相同键");
    }

    /// ⭐⭐ **硬约束①**：消息内容变化 → 键必变。
    ///
    /// 这条性质覆盖了 21 修过的一个**真 bug**：
    /// `ContentPart::ToolResponse` 曾落进 `_ => {}` 分支，
    /// 工具结果内容**完全没进键** ——
    /// 两个内容天差地别的 `tool` 消息会算出**同一个键**。
    #[test]
    fn message_content_change_changes_key(
        prefix in prop::collection::vec(text(), 0..4),
        suffix in prop::collection::vec(text(), 0..4),
        content_a in text(),
        content_b in text(),
    ) {
        prop_assume!(content_a != content_b);

        let mut msgs_a: Vec<ChatMessage> = prefix.iter().map(|t| user(t)).collect();
        msgs_a.push(tool_result("call_1", &content_a));
        msgs_a.extend(suffix.iter().map(|t| user(t)));

        let mut msgs_b = msgs_a.clone();
        // 只替换工具结果的内容，其它一切不变。
        let last = msgs_b.len() - suffix.len() - 1;
        msgs_b[last] = tool_result("call_1", &content_b);

        prop_assert_ne!(
            key_for(&msgs_a).redis_key(),
            key_for(&msgs_b).redis_key(),
            "工具结果内容不同，缓存键必须不同"
        );
    }

    /// ⭐ 工具调用参数变化 → 键必变。
    #[test]
    fn tool_call_arguments_change_changes_key(
        call_a in args(),
        call_b in args(),
    ) {
        prop_assume!(call_a != call_b);

        let a = key_for(&[assistant_with_call("call_1", "get_weather", call_a)]);
        let b = key_for(&[assistant_with_call("call_1", "get_weather", call_b)]);

        prop_assert_ne!(
            a.redis_key(),
            b.redis_key(),
            "工具参数不同，缓存键必须不同"
        );
    }

    /// ⭐ 工具名变化 → 键必变。
    #[test]
    fn tool_name_change_changes_key(
        name_a in tool_name(),
        name_b in tool_name(),
    ) {
        prop_assume!(name_a != name_b);

        let a = key_for(&[assistant_with_call("call_1", &name_a, json!({}))]);
        let b = key_for(&[assistant_with_call("call_1", &name_b, json!({}))]);

        prop_assert_ne!(
            a.redis_key(),
            b.redis_key(),
            "工具名不同，缓存键必须不同"
        );
    }

    /// ⭐ 消息**顺序**变化 → 键必变。
    ///
    /// 顺序不同 = 完全不同的对话，必须 miss。
    #[test]
    fn message_order_change_changes_key(
        a in text(),
        b in text(),
    ) {
        prop_assume!(a != b);

        let first = key_for(&[user(&a), user(&b)]);
        let second = key_for(&[user(&b), user(&a)]);

        prop_assert_ne!(
            first.redis_key(),
            second.redis_key(),
            "消息顺序不同，缓存键必须不同"
        );
    }

    /// ⭐⭐ 硬约束①的**全字段版本**：任一字段变化 → 键必变。
    ///
    /// 现有的单元测试是「一个字段一个测试」，
    /// 而这条性质把它们合起来验证 ——
    /// 新增字段时只要忘了加进键，这条立刻失败。
    #[test]
    fn any_field_change_changes_key(
        namespace_a in "[a-z]{1,8}",
        namespace_b in "[a-z]{1,8}",
        model_a in "[a-z]{1,8}",
        model_b in "[a-z]{1,8}",
        choice_a in "(auto|none|required)",
        choice_b in "(auto|none|required)",
        temp_a in 0.0f64..2.0,
        temp_b in 0.0f64..2.0,
        max_a in 1u32..8192,
        max_b in 1u32..8192,
        reasoning_a in any::<bool>(),
        reasoning_b in any::<bool>(),
    ) {
        let msgs = vec![user("同样的消息")];

        // 前提：每一对「基字段 / 变体字段」必须**真的不同**。
        //
        // ⚠️ 少了这个前提，性质就是错的 ——
        // 随机生成的 `namespace_a` 和 `namespace_b` 可能相同，
        // 那时「键相同」是**正确行为**，不是 bug。
        prop_assume!(namespace_a != namespace_b, "命名空间必须不同");
        prop_assume!(model_a != model_b, "模型名必须不同");
        prop_assume!(choice_a != choice_b, "工具选择策略必须不同");
        prop_assume!(
            (temp_a - temp_b).abs() > f64::EPSILON,
            "温度必须不同"
        );
        prop_assume!(max_a != max_b, "max_tokens 必须不同");
        prop_assume!(reasoning_a != reasoning_b, "思考开关必须不同");

        let base = CacheKey::new(
            &namespace_a, &model_a, &msgs, "digest", &choice_a, temp_a, max_a, reasoning_a,
        );

        // 逐个字段替换，每个都必须产生不同的键。
        let variants = vec![
            CacheKey::new(
                &namespace_b, &model_a, &msgs, "digest", &choice_a, temp_a, max_a, reasoning_a,
            ),
            CacheKey::new(
                &namespace_a, &model_b, &msgs, "digest", &choice_a, temp_a, max_a, reasoning_a,
            ),
            CacheKey::new(
                &namespace_a, &model_a, &msgs, "digest", &choice_b, temp_a, max_a, reasoning_a,
            ),
            CacheKey::new(
                &namespace_a, &model_a, &msgs, "digest2", &choice_a, temp_a, max_a, reasoning_a,
            ),
            CacheKey::new(
                &namespace_a, &model_a, &msgs, "digest", &choice_a, temp_b, max_a, reasoning_a,
            ),
            CacheKey::new(
                &namespace_a, &model_a, &msgs, "digest", &choice_a, temp_a, max_b, reasoning_a,
            ),
            CacheKey::new(
                &namespace_a, &model_a, &msgs, "digest", &choice_a, temp_a, max_a, reasoning_b,
            ),
        ];

        for variant in variants {
            prop_assert_ne!(
                base.redis_key(),
                variant.redis_key(),
                "任一字段变化都必须产生不同的缓存键"
            );
        }
    }

    /// 键的形态符合 Redis 键约定：`{namespace}:{version}:{16 位十六进制}`。
    #[test]
    fn key_shape_is_valid(
        _namespace in "[a-z]{1,10}",
        msgs in prop::collection::vec(text(), 0..5),
    ) {
        let msgs: Vec<ChatMessage> = msgs.iter().map(|t| user(t)).collect();
        let key = key_for(&msgs).redis_key();

        let parts: Vec<&str> = key.split(':').collect();
        prop_assert_eq!(parts.len(), 3, "键必须是三段结构，实际是 {:?}", key);
        prop_assert_eq!(parts[0], DEFAULT_NAMESPACE);
        prop_assert_eq!(parts[1], "v1");
        prop_assert_eq!(
            parts[2].len(),
            16,
            "摘要段必须是 16 位十六进制，实际是 {:?}",
            parts[2]
        );
        prop_assert!(
            parts[2].chars().all(|c| c.is_ascii_hexdigit()),
            "摘要段必须全是十六进制，实际是 {:?}",
            parts[2]
        );
    }

    /// 摘要是稳定的 64 位 SHA-256 十六进制串。
    #[test]
    fn digest_is_stable_sha256(
        msgs in prop::collection::vec(text(), 0..5),
    ) {
        let msgs: Vec<ChatMessage> = msgs.iter().map(|t| user(t)).collect();
        let key = key_for(&msgs);

        let digest = key.digest();
        prop_assert_eq!(digest.len(), 64, "SHA-256 十六进制串应是 64 位");
        prop_assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
        prop_assert_eq!(digest, key.digest(), "摘要必须稳定");
    }
}
