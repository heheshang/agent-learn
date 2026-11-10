//! 模糊测试目标 —— 08-rag 的 **中英混排分词器**。
//!
//! ## 这个目标在找什么
//!
//! 1. **panic** —— 分词器被 13/14/15/21 全链路复用，挂了就是全挂。
//! 2. **空词元** —— 空串会让 BM25 的词频统计失去意义。
//! 3. **脚本混用** —— 一个词元里既有 CJK 又有拉丁，
//!    会让 BM25 的长度归一化失真。

#![no_main]

use libfuzzer_sys::fuzz_target;
use rag_08::tokenize::tokenize;

/// 判断一个字符是否属于 CJK。
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xF900..=0xFAFF
        | 0x3040..=0x30FF
        | 0xAC00..=0xD7AF
        | 0x20000..=0x2A6DF)
}

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    // 1. 不能 panic。
    let tokens = tokenize(&text);

    // 2. 词元永不为空。
    for token in &tokens {
        assert!(!token.is_empty(), "词元不能是空串（input={text:?}）");
    }

    // 3. 词元不能混脚本。
    for token in &tokens {
        let has_cjk = token.chars().all(is_cjk);
        let has_latin = token.chars().any(|c| c.is_ascii_alphanumeric());
        assert!(
            !(has_cjk && has_latin),
            "词元不能混脚本：{token:?}（input={text:?}）"
        );
    }

    // 4. 拉丁词元必须全小写。
    for token in &tokens {
        if token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !token.is_empty() {
            assert_eq!(
                token.to_lowercase(),
                *token,
                "拉丁词元必须全小写：{token:?}（input={text:?}）"
            );
        }
    }

    // 5. 分词必须确定。
    assert_eq!(tokens, tokenize(&text), "分词必须确定（input={text:?}）");
});
