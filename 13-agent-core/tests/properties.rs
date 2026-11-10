//! 属性测试（property-based testing）—— 13-agent-core 的 ReAct 输出解析器。
//!
//! ## 为什么解析器需要属性测试
//!
//! [`parse_react_output`] 的输入是**模型的原始输出** ——
//! 这是整个仓库里**最不可信的一段文本**：
//!
//! - 模型可能回显整个提示词（含 few-shot 示例）
//! - 模型可能输出任意 Unicode（中文、emoji、零宽字符）
//! - 模型可能给出格式完全错误的 JSON
//!
//! 单元测试只能覆盖「人想得到的」输入。
//! 属性测试让**随机输入**去撞不变量，失败时自动收缩成最小反例。
//!
//! ## ⚠️ 这个文件真的抓到过两个 bug
//!
//! ### bug 1：Unicode 小写化导致越界 panic
//!
//! 最初的 `find_last_marker` 实现是：
//!
//! ```text
//! let lower = text.to_lowercase();
//! lower.rfind(&marker.to_lowercase())
//! ```
//!
//! Unicode 小写化**不保证字节长度不变** —— `İ`（U+0130）小写化成
//! `i` + 组合字符（2 字节 → 3 字节）。
//! 于是 `lower` 的偏移量比 `text` 大，用它切 `text` 直接越界 panic：
//!
//! ```text
//! text = "İİİAction: x" → lower.rfind("action:") = 9
//! text[9 + 7..] → start byte index 16 is out of bounds for string of length 15
//! ```
//!
//! 修复见 `agent.rs` 的 `find_last_marker`。
//!
//! ### bug 2：工具名没做「标识符截断」
//!
//! `Action:Thought:Thought:` 会被解析出工具名 `Thought:Thought` ——
//! 一个**永远不匹配任何工具**的名字（`trim_matches` 只削首尾，内部冒号留着）。
//! 修复改成「取开头那段连续的字母/数字/下划线」。
//!
//! ## 不变量清单
//!
//! | 不变量 | 含义 |
//! |---|---|
//! | `never_panics` | 任意输入都不 panic（**最重要的那条**） |
//! | `final_answer_wins` | 有 `Final Answer:` 时一定解析成最终答案 |
//! | `action_tool_name_is_sane` | 工具名是纯标识符 |
//! | `input_is_always_an_object` | 参数永远是 JSON 对象 |
//! | `last_marker_wins` | 多个标记时取**最后一个** |
//! | `no_tool_name_falls_back` | 没有工具名时回退成 `Unstructured` |

use agent_13::agent::{ParsedOutput, parse_action_input, parse_react_output};
use proptest::prelude::*;

/// 四个标记关键词（都在解析器里被识别）。
const MARKERS: [&str; 4] = ["Action:", "Action Input:", "Final Answer:", "Thought:"];

/// 生成**不含任何标记**的噪声文本。
///
/// ## 为什么排除 `:`
///
/// 四个标记**全都含冒号**。所以噪声里只要没有 `:`，
/// 就保证不会出现半个标记 —— 这样「最后一个标记」类性质
/// 可以精确构造出「恰好 N 个标记」的输入，
/// 不会因为随机噪声里藏着标记而导致大量 reject。
fn noise() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            // 常见 ASCII 词元（不含冒号）
            Just("search".to_string()),
            Just("calculate".to_string()),
            Just("query".to_string()),
            Just("hello world".to_string()),
            // 标点与空白（不含冒号）
            Just(",".to_string()),
            Just(".".to_string()),
            Just("!".to_string()),
            Just("?".to_string()),
            Just(" ".to_string()),
            Just("\n".to_string()),
            Just("\"".to_string()),
            Just("'".to_string()),
            Just("[]{}".to_string()),
            // 会改变字节长度的字符（小写化元凶）
            Just("İ".to_string()),
            Just("ẞ".to_string()),
            // 零宽 / 组合字符
            Just("\u{200B}".to_string()),
            Just("\u{0301}".to_string()),
            // CJK 与 emoji
            Just("检索".to_string()),
            Just("🤖".to_string()),
            // 任意单个字符
            any::<char>().prop_map(|c| c.to_string()),
        ],
        0..30,
    )
    .prop_map(|parts| parts.concat())
}

/// 生成一段「可能来自模型」的文本 —— 噪声 + 随机插入的标记。
///
/// 用于「任意输入都不 panic」这类**不需要精确构造**的性质。
fn model_output() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            prop::sample::select(&MARKERS).prop_map(|s| s.to_string()),
            noise(),
        ],
        0..20,
    )
    .prop_map(|parts| parts.concat())
}

/// 找出 `needle` 在 `text` 中**最后一次**出现的位置（ASCII 不区分大小写）。
///
/// ## ⚠️ 返回的是**原串**的字节偏移
///
/// 不能简单地 `text.to_lowercase().rfind(...)` ——
/// Unicode 小写化会改变字节长度，偏移就对不上了
/// （这正是被测代码曾经的 bug）。
///
/// 这里只在**字符边界**上做 ASCII 大小写比较，
/// 所以偏移永远是原串的偏移。
fn last_marker_pos(text: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    let needle_len = needle.len();
    text.char_indices().rev().find_map(|(idx, _)| {
        let end = idx + needle_len;
        let slice = text.get(idx..end)?;
        slice.eq_ignore_ascii_case(needle).then_some(idx)
    })
}

/// 解析出「最后一个 `Action:` 之后的工具名」（标识符部分）。
fn tool_name_after_last_action(text: &str) -> Option<String> {
    let pos = last_marker_pos(text, "action:")?;
    let rest = text.get(pos + "Action:".len()..)?;
    Some(
        rest.split_whitespace()
            .next()
            .unwrap_or("")
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect(),
    )
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    /// ⭐ 最重要的不变量：**任意输入都不 panic**。
    ///
    /// 这是「模型输出完全不可信」这一前提下的底线要求。
    #[test]
    fn never_panics(raw in model_output()) {
        // 只要不 panic 就算过 —— proptest 会自动把 panic 收缩成最小反例。
        let _ = parse_react_output(&raw);
    }

    /// `parse_action_input` 同样不能 panic。
    #[test]
    fn action_input_never_panics(raw in model_output()) {
        let _ = parse_action_input(&raw);
    }

    /// 出现 `Final Answer:` 时，一定解析成最终答案（优先级最高）。
    #[test]
    fn final_answer_wins(raw in model_output()) {
        let parsed = parse_react_output(&raw);

        // 只有当文本里**确实有** `final answer:`（忽略大小写）时才断言。
        if raw.to_lowercase().contains("final answer:") {
            prop_assert!(
                matches!(parsed, ParsedOutput::FinalAnswer { .. }),
                "含 `Final Answer:` 的输入必须解析成最终答案，实际是 {:?}",
                parsed
            );
        }
    }

    /// 工具名必须是纯标识符（字母/数字/下划线）。
    ///
    /// ⚠️ 这条性质抓到过一个真 bug：
    /// `Action:Thought:Thought:` 曾被解析成工具名 `Thought:Thought`
    /// —— 一个永远匹配不上任何工具的名字。
    #[test]
    fn action_tool_name_is_sane(raw in model_output()) {
        if let ParsedOutput::Action { tool, .. } = parse_react_output(&raw) {
            prop_assert!(!tool.is_empty(), "工具名不能为空");
            prop_assert!(
                tool.chars().all(|c| c.is_alphanumeric() || c == '_'),
                "工具名只能含字母/数字/下划线，实际是 {:?}",
                tool
            );
        }
    }

    /// 参数永远是 JSON **对象**（不是数组、不是标量）。
    ///
    /// 这条不变量很重要：下游代码直接 `args.get("query")`，
    /// 如果参数不是对象就会 panic 或静默失效。
    #[test]
    fn action_args_are_always_an_object(raw in model_output()) {
        if let ParsedOutput::Action { args, .. } = parse_react_output(&raw) {
            prop_assert!(
                args.is_object(),
                "参数必须是 JSON 对象，实际是 {:?}",
                args
            );
        }
    }

    /// `parse_action_input` 的返回参数也永远是对象。
    #[test]
    fn parsed_action_input_is_always_an_object(raw in model_output()) {
        let (args, _err) = parse_action_input(&raw);
        prop_assert!(args.is_object(), "参数必须是 JSON 对象，实际是 {:?}", args);
    }

    /// 多个 `Final Answer:` 时取**最后一个** ——
    /// 这是 13 的核心修复（模型回显提示词时旧标记在前）。
    ///
    /// ⚠️ 断言方式：对拍一个**独立的参考实现**，
    /// 而不是假设「我拼接的 suffix 一定不含标记」。
    /// 随机噪声里完全可能再出现一个 `Final Answer:`。
    #[test]
    fn last_final_answer_wins(
        prefix in noise(),
        middle in noise(),
        suffix in noise(),
    ) {
        let raw = format!("{prefix}Final Answer: 第一个{middle}Final Answer: 第二个{suffix}");

        // 参考实现：最后一个标记的**起始位置**。
        let last_pos = last_marker_pos(&raw, "final answer:")
            .expect("文本里一定有两个标记");
        let expected = raw[last_pos + "Final Answer:".len()..].trim().to_string();

        match parse_react_output(&raw) {
            ParsedOutput::FinalAnswer { text } => {
                prop_assert_eq!(text, expected, "应该取最后一个 `Final Answer:` 的内容");
            }
            other => prop_assert!(false, "应当是最终答案，实际是 {:?}", other),
        }
    }

    /// 多个 `Action:` 时取**最后一个**。
    ///
    /// ⚠️ 前提：文本里**没有** `Final Answer:` ——
    /// 因为 `Final Answer:` 优先级**高于** `Action:`（这是 13 的设计，
    /// 见 `parse_react_output` 的文档：模型的思考里可能提到这几个字，
    /// 先判动作会误判）。
    #[test]
    fn last_action_wins(
        prefix in noise(),
        middle in noise(),
        suffix in noise(),
    ) {
        let raw = format!("{prefix}Action: search{middle}Action: calculate{suffix}");

        // 前提过滤：出现 `Final Answer:` 时结果必然是最终答案，
        // 那由 `final_answer_wins` 单独覆盖。
        prop_assume!(last_marker_pos(&raw, "final answer:").is_none());

        // 参考实现：最后一个 `Action:` 之后的第一个标识符 token。
        let expected = tool_name_after_last_action(&raw)
            .expect("文本里一定有两个标记");

        // 前提过滤：最后一个 `Action:` 之后**必须**有工具名。
        //
        // 否则（`Action:` 出现在文本末尾，后面什么都没有）
        // 解析不出工具名 → 结果是 `Unstructured`，
        // 那是**正确行为**（没有工具名可执行），由下一条性质单独覆盖。
        prop_assume!(!expected.is_empty());

        match parse_react_output(&raw) {
            ParsedOutput::Action { tool, .. } => {
                prop_assert_eq!(tool, expected, "应该取最后一个 `Action:`");
            }
            other => prop_assert!(false, "应当是工具调用，实际是 {:?}", other),
        }
    }

    /// 最后一个 `Action:` 之后**没有工具名**时，不能解析成工具调用。
    ///
    /// 因为那样会拿一个空工具名去执行 —— 下游会报「未知工具 ` `」。
    /// 正确行为是回退到 `Unstructured`（当作答案）。
    #[test]
    fn action_without_tool_name_falls_back_to_unstructured(
        prefix in noise(),
        suffix in "[,!?. \n]{0,8}",
    ) {
        let raw = format!("{prefix}Action:{suffix}");
        prop_assume!(last_marker_pos(&raw, "final answer:").is_none());

        // 「最后一个 Action: 之后没有标识符」的前置条件。
        let name = tool_name_after_last_action(&raw).expect("至少有一个标记");
        prop_assume!(name.is_empty());

        prop_assert!(
            matches!(parse_react_output(&raw), ParsedOutput::Unstructured { .. }),
            "没有工具名时必须回退到 Unstructured，实际是 {:?}",
            parse_react_output(&raw)
        );
    }

    /// `Thought:` 段一定出现在 `Action:` **之前**——
    /// 也就是说解析出来的 thought 不会是 Action 之后的内容。
    #[test]
    fn thought_is_extracted_before_action(
        thought in "[a-zA-Z ]{0,20}",
        _rest in noise(),
    ) {
        let raw = format!("Thought: {thought}\nAction: search\nAction Input: {{\"q\":1}}");

        if let ParsedOutput::Action { thought: parsed, .. } = parse_react_output(&raw) {
            prop_assert!(
                !parsed.contains("Action"),
                "Thought 不该包含 Action 之后的内容，实际是 {:?}",
                parsed
            );
        }
    }
}
