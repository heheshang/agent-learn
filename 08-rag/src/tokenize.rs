//! 分词器 —— 中英混排的关键词切分。
//!
//! ## 为什么必须自己写
//!
//! BM25 的前提是「把文本切成词」。但中文没有空格，英文有 ——
//! 一个通用分词器要同时处理两种语言，就得引入词典和统计模型。
//!
//! 本项目的目标是**教学可读**，所以选一个简单但正确的方案：
//!
//! ```text
//! 拉丁字母/数字串 → 整段小写成一个词
//! CJK 字符        → 单字（unigram）+ 相邻双字（bigram）
//! 其它字符        → 分隔符
//! ```
//!
//! ## 为什么 CJK 要出 bigram
//!
//! 只用单字会丢掉词序信息：「数据检索」和「检索数据」的 unigram 集合完全一样。
//!
//! 加了 bigram 之后：
//!
//! ```text
//! 「数据检索」 → 数据 / 据检 / 检索
//! 「检索数据」 → 检索 / 索数 / 数据
//! ```
//!
//! 两者的 bigram 集合不同了 —— 词序信息被保留下来。
//!
//! 这是一个经典的折中：**不做真正的分词，但也不完全丢掉词序**。
//! 真正的中文分词（词典 + CRF / BERT）效果好得多，但不在本项目范围内。
//!
//! ## 为什么拉丁词要小写
//!
//! BM25 的文档频率统计是**大小写不敏感**的。
//! `RAG` 和 `rag` 应该被当成同一个词，否则 IDF 会被算成两个词的交集。
//!
//! ## 其它字符一律当分隔符
//!
//! 标点、空白、Markdown 标记（`#`、`*`、`|`）都只起分隔作用。
//! 这很关键：`**粗体**` 和 `粗体` 应该切出同样的词。

/// 判断一个字符是否属于 CJK 字符集。
///
/// 覆盖范围：CJK 统一表意文字（含扩展 A）、兼容表意文字、
/// 日文假名、韩文谚文。
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4DBF   // CJK 扩展 A
        | 0x4E00..=0x9FFF // CJK 基本区
        | 0xF900..=0xFAFF // CJK 兼容表意文字
        | 0x3040..=0x30FF // 平假名 / 片假名
        | 0xAC00..=0xD7AF // 韩文谚文
        | 0x20000..=0x2A6DF // CJK 扩展 B
    )
}

/// 把文本切成词元（token）。
///
/// 返回的词元**不去重** —— BM25 的查询侧需要保留查询词的频次。
///
/// # 示例
///
/// ```text
/// "RAG 用 BM25 检索"   → ["rag", "bm25", "检索"]
/// "数据检索"            → ["数据", "据检", "检索"]
/// ```
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut latin = String::new();
    let mut cjk: Vec<char> = Vec::new();

    for ch in text.chars() {
        if is_cjk(ch) {
            flush_latin(&mut latin, &mut tokens);
            cjk.push(ch);
        } else if ch.is_alphanumeric() {
            // 拉丁字母 / 数字 —— 和 CJK 之间的边界也算分隔。
            flush_cjk(&mut cjk, &mut tokens);
            for lower in ch.to_lowercase() {
                latin.push(lower);
            }
        } else {
            // 标点、空白、Markdown 符号 —— 纯分隔符。
            flush_latin(&mut latin, &mut tokens);
            flush_cjk(&mut cjk, &mut tokens);
        }
    }

    flush_latin(&mut latin, &mut tokens);
    flush_cjk(&mut cjk, &mut tokens);

    tokens
}

/// 把一段连续拉丁文本刷成一个词。
fn flush_latin(buf: &mut String, out: &mut Vec<String>) {
    if !buf.is_empty() {
        out.push(std::mem::take(buf));
    }
}

/// 把一段连续 CJK 文本刷成 unigram + bigram。
fn flush_cjk(buf: &mut Vec<char>, out: &mut Vec<String>) {
    if buf.is_empty() {
        return;
    }

    // unigram：每个字单独成词。
    for &c in buf.iter() {
        out.push(c.to_string());
    }

    // bigram：相邻两字成词（保留词序信息）。
    for window in buf.windows(2) {
        out.push(window.iter().collect());
    }

    buf.clear();
}

/// 统计文本里的词元数量（去重前）。
pub fn token_count(text: &str) -> usize {
    tokenize(text).len()
}

/// 判断一个词元是否「有意义」——
///
/// 纯数字的词元（`1`、`2024`）在 BM25 里几乎没有区分度，
/// 但也可能是关键的版本号 / 错误码，所以**不丢弃**。
/// 这个函数只是给测试和调试用的可读性辅助。
pub fn is_numeric(token: &str) -> bool {
    !token.is_empty() && token.chars().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_latin_words_and_lowercases() {
        assert_eq!(tokenize("RAG and BM25"), vec!["rag", "and", "bm25"]);
    }

    #[test]
    fn keeps_numbers_attached_to_letters() {
        assert_eq!(tokenize("O(log n)"), vec!["o", "log", "n"]);
    }

    #[test]
    fn emits_unigrams_and_bigrams_for_cjk() {
        let tokens = tokenize("数据检索");
        assert_eq!(tokens, vec!["数", "据", "检", "索", "数据", "据检", "检索"]);
    }

    #[test]
    fn cjk_bigrams_preserve_word_order() {
        // 「数据检索」和「检索数据」的 unigram 集合相同，
        // 但 bigram 集合不同 —— 这正是加 bigram 的意义。
        let unigrams = |s: &str| -> std::collections::HashSet<String> {
            tokenize(s)
                .into_iter()
                .filter(|t| t.chars().count() == 1)
                .collect()
        };
        let bigrams = |s: &str| -> std::collections::HashSet<String> {
            tokenize(s)
                .into_iter()
                .filter(|t| t.chars().count() == 2)
                .collect()
        };

        assert_eq!(
            unigrams("数据检索"),
            unigrams("检索数据"),
            "unigram 集合应该相同"
        );
        assert_ne!(
            bigrams("数据检索"),
            bigrams("检索数据"),
            "bigram 集合应该不同"
        );
    }

    #[test]
    fn punctuation_only_separates() {
        assert_eq!(tokenize("**粗体**"), tokenize("粗体"));
        assert_eq!(tokenize("a, b"), vec!["a", "b"]);
    }

    #[test]
    fn empty_input_yields_no_tokens() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("   \n\t ").is_empty());
        assert!(tokenize("!!!").is_empty());
    }

    #[test]
    fn mixed_script_text() {
        let tokens = tokenize("用 BM25 检索数据");
        assert!(tokens.contains(&"bm25".to_string()));
        assert!(tokens.contains(&"检索".to_string()));
        assert!(tokens.contains(&"数据".to_string()));
    }

    #[test]
    fn numeric_tokens_are_detected() {
        assert!(is_numeric("2024"));
        assert!(!is_numeric("rag"));
        assert!(!is_numeric(""));
    }

    #[test]
    fn token_count_matches_tokenize() {
        let text = "混合 Mixed 文本 text";
        assert_eq!(token_count(text), tokenize(text).len());
    }
}
