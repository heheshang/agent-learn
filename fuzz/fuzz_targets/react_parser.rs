//! 模糊测试目标 —— 13-agent-core 的 **ReAct 输出解析器**。
//!
//! ## 这个目标在找什么
//!
//! **panic**。解析器的输入是模型的原始输出，完全不可信。
//! 任何 panic 都会让整个 Agent 请求 500 —— 对一个「读用户问题」的服务来说，
//! 这是可以直接被外部触发的可用性问题。
//!
//! ## 与 `proptest` 的分工
//!
//! | | proptest（13 的 `tests/properties.rs`） | libFuzzer（本文件） |
//! |---|---|---|
//! | 输入来源 | 结构化策略（按语义生成） | **任意字节序列** |
//! | 覆盖 | 语义边界（标记、工具名、参数） | 字节级畸形（非法 UTF-8、超长、混合） |
//! | 失败产物 | 收缩后的最小反例（人可读） | 语料文件（可复现） |
//!
//! 两者**互补**：proptest 保证「语义正确」，libFuzzer 保证「字节级健壮」。
//!
//! ## ⚠️ 已经抓到过的 bug
//!
//! `find_last_marker` 曾用 `to_lowercase()` 后的偏移量切原串，
//! 遇到 `İ`（小写化会变长的字符）就越界 panic。
//! 见 `13-agent-core/src/agent.rs` 里 `find_last_marker` 的文档。

#![no_main]

use libfuzzer_sys::fuzz_target;
use agent_13::agent::{parse_action_input, parse_react_output};

fuzz_target!(|data: &[u8]| {
    // ⚠️ libFuzzer 给的是**任意字节**，不保证是合法 UTF-8。
    //
    // 这里用 `from_utf8` 过滤掉非法序列 ——
    // 因为 Rust 的 `&str` 保证了 UTF-8，
    // 而「非法 UTF-8 怎么处理」是**调用方**（HTTP 层）的事，不是解析器的事。
    //
    // 但我们**同时**测两种解释方式：
    // - 合法 UTF-8 → 走正常路径
    // - 非法 UTF-8 → 用 `from_utf8_lossy` 替换成 U+FFFD 再走一遍
    //   （这模拟了「服务端收到乱码后替换处理」的真实场景）
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = parse_react_output(text);
        let _ = parse_action_input(text);
    }

    let lossy = String::from_utf8_lossy(data);
    let _ = parse_react_output(&lossy);
    let _ = parse_action_input(&lossy);
});
