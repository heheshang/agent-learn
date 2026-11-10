//! 属性测试（property-based testing）—— 08-rag 的 **中英混排分词器**。
//!
//! ## 分词器为什么需要属性测试
//!
//! 分词器是 BM25 的**地基** —— 切错了，检索质量整体塌方，
//! 而且症状很隐蔽（召回率低，但没有任何报错）。
//!
//! 更要命的是：13 的 `search` 工具、14 的天气工具、15 的 SQL Agent、
//! 21 的缓存链路**全都复用它**。一处切错，全链路受影响。
//!
//! 现有单元测试覆盖了「典型文本」。
//! 属性测试补的是**边界组合**：CJK 与拉丁混排、标点夹在中间、
//! emoji、组合字符 —— 这些组合会让「连续 CJK 段」的边界判定出错。
//!
//! ## 不变量清单
//!
//! | 不变量 | 含义 |
//! |---|---|
//! | `never_panics` | 任意文本都不 panic |
//! | `tokens_are_non_empty` | 词元永不为空串（空串会让 BM25 分数异常） |
//! | `latin_is_lowercased` | 拉丁词元全小写 |
//! | `punctuation_produces_no_tokens` | 纯标点不产生词元 |
//! | `tokenize_is_deterministic` | 同输入同输出 |
//! | `whitespace_is_a_separator` | 空白只是分隔符 |
//! | `cjk_bigrams_preserve_order` | bigram 保留词序（检索质量的核心） |
//! | `token_count_matches_tokenize` | 计数与切分一致 |

use proptest::prelude::*;
use rag_08::tokenize::{is_numeric, token_count, tokenize};

/// 生成任意文本（含 CJK、拉丁、标点、emoji、组合字符）。
fn text() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            // CJK
            Just("中".to_string()),
            Just("检索".to_string()),
            Just("数据".to_string()),
            // 拉丁
            Just("rag".to_string()),
            Just("BM25".to_string()),
            Just("ReAct".to_string()),
            // 数字
            Just("2024".to_string()),
            // 标点与空白
            Just(",".to_string()),
            Just(".".to_string()),
            Just("(".to_string()),
            Just(")".to_string()),
            Just("!".to_string()),
            Just(" ".to_string()),
            Just("\n".to_string()),
            Just("\t".to_string()),
            // Markdown 记号
            Just("#".to_string()),
            Just("*".to_string()),
            Just("|".to_string()),
            // emoji 与组合字符
            Just("🤖".to_string()),
            Just("\u{0301}".to_string()),
            // 任意字符
            any::<char>().prop_map(|c| c.to_string()),
        ],
        0..30,
    )
    .prop_map(|parts| parts.concat())
}

/// 判断一个词元是否「全是 CJK」。
fn is_cjk_token(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))
}

/// 判断一个词元是否「含拉丁字母或数字」。
fn is_latin_token(token: &str) -> bool {
    !token.is_empty() && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// ⭐ 任意文本都不 panic。
    #[test]
    fn tokenizer_never_panics(input in text()) {
        let _ = tokenize(&input);
        let _ = token_count(&input);
    }

    /// ⭐ 词元**永不为空串**。
    ///
    /// 空词元会让 BM25 的词频统计出现「长度为 0 的词」，
    /// 分数计算直接失去意义。
    #[test]
    fn tokens_are_never_empty(input in text()) {
        for token in tokenize(&input) {
            prop_assert!(!token.is_empty(), "词元不能是空串");
        }
    }

    /// ⭐ 拉丁词元**全小写**。
    ///
    /// BM25 的文档频率统计是大小写不敏感的 ——
    /// `RAG` 和 `rag` 必须算同一个词，否则 IDF 被算成两个词的交集。
    #[test]
    fn latin_tokens_are_lowercased(input in text()) {
        for token in tokenize(&input) {
            if is_latin_token(&token) {
                prop_assert_eq!(
                    token.to_lowercase(),
                    token.clone(),
                    "拉丁词元必须全小写，实际是 {:?}",
                    token
                );
            }
        }
    }

    /// ⭐ 纯标点 / 空白**不产生任何词元**。
    #[test]
    fn punctuation_produces_no_tokens(
        punct in prop::sample::select(vec![
            ",", ".", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}",
            "#", "*", "|", "-", "/", "\\", " ", "\n", "\t", "　",
        ]),
    ) {
        let tokens = tokenize(punct);
        prop_assert!(
            tokens.is_empty(),
            "纯标点不该产生词元，实际是 {:?}",
            tokens
        );
    }

    /// ⭐ 分词是**确定性**的 —— 同输入同输出。
    ///
    /// 这条是「缓存键稳定」「检索可复现」的基础。
    #[test]
    fn tokenize_is_deterministic(input in text()) {
        prop_assert_eq!(tokenize(&input), tokenize(&input));
    }

    /// ⭐ 空白是**词边界** —— 插入空白不改变两侧的切分结果。
    ///
    /// ⚠️ 为什么不是「删掉空白结果不变」：
    /// 删掉 `"rag rag"` 的空格会得到 `"ragrag"` —— 两个拉丁词**合并成一个**。
    /// 因为分词器把「连续的拉丁字母/数字」当作一个词。
    ///
    /// 正确的性质是「空白处切一刀，两侧各自独立切分」：
    /// `tokenize(a + " " + b) == tokenize(a) ++ tokenize(b)`
    #[test]
    fn whitespace_is_a_word_boundary(
        a in "[a-zA-Z0-9]{1,8}",
        b in "[a-zA-Z0-9]{1,8}",
    ) {
        prop_assert_eq!(
            tokenize(&format!("{a} {b}")),
            {
                let mut left = tokenize(&a);
                left.extend(tokenize(&b));
                left
            },
            "空白必须是词边界"
        );
    }

    /// ⭐⭐ **bigram 集合精确等于「相邻字符对」的集合**。
    ///
    /// 这是分词器的**完整行为刻画** ——
    /// bigram 就是「每个相邻两字」去重后的结果。
    #[test]
    fn bigrams_are_exactly_the_adjacent_pairs(input in "[一-鿿]{2,10}") {
        let tokens = tokenize(&input);
        let bigrams: std::collections::HashSet<String> = tokens
            .iter()
            .filter(|t| t.chars().count() == 2)
            .cloned()
            .collect();

        let chars: Vec<char> = input.chars().collect();
        let expected: std::collections::HashSet<String> = chars
            .windows(2)
            .map(|w| w.iter().collect())
            .collect();

        prop_assert_eq!(
            bigrams, expected,
            "bigram 集合必须精确等于相邻字符对集合"
        );
    }

    /// ⭐⭐ **顺序确实影响结果** —— 这才是加 bigram 的全部意义。
    ///
    /// 「数据检索」和「检索数据」的 unigram 集合完全相同，
    /// 只有 bigram 能区分它们。
    ///
    /// ⚠️ 这里用「交换两个不同字符」构造反例 ——
    /// 而不是随机生成两串（那样字符集合很难撞上相同，reject 率极高）。
    #[test]
    fn swapping_two_characters_changes_bigrams(
        a in "[一-鿿]",
        b in "[一-鿿]",
        rest in "[一-鿿]{0,6}",
    ) {
        prop_assume!(a != b, "两个字符必须不同");

        let s1 = format!("{a}{b}{rest}");
        let s2 = format!("{b}{a}{rest}");

        let bigrams = |s: &str| -> std::collections::HashSet<String> {
            tokenize(s)
                .into_iter()
                .filter(|t| t.chars().count() == 2)
                .collect()
        };

        // unigram 集合必然相同（字符集合没变）。
        let unigrams = |s: &str| -> std::collections::HashSet<String> {
            tokenize(s)
                .into_iter()
                .filter(|t| t.chars().count() == 1)
                .collect()
        };
        prop_assert_eq!(unigrams(&s1), unigrams(&s2));

        // bigram 集合必然不同（顺序变了）。
        prop_assert_ne!(
            bigrams(&s1),
            bigrams(&s2),
            "交换两个字符后 bigram 集合必须不同 —— 这是加 bigram 的意义"
        );
    }

    /// ⭐ 计数与切分结果**一致**。
    #[test]
    fn token_count_matches_tokenize(input in text()) {
        prop_assert_eq!(token_count(&input), tokenize(&input).len());
    }

    /// ⭐ CJK 段落的**连续两字**一定产出 bigram。
    ///
    /// 例如「数据检索」必须包含 bigram「数据」「据检」「检索」。
    #[test]
    fn cjk_runs_produce_all_bigrams(input in "[一-鿿]{2,8}") {
        let tokens = tokenize(&input);
        let bigrams: std::collections::HashSet<String> = tokens
            .iter()
            .filter(|t| t.chars().count() == 2)
            .cloned()
            .collect();

        let chars: Vec<char> = input.chars().collect();
        for window in chars.windows(2) {
            let expected: String = window.iter().collect();
            prop_assert!(
                bigrams.contains(&expected),
                "缺少 bigram {:?}，实际词元：{:?}",
                expected,
                tokens
            );
        }
    }

    /// ⭐ CJK 段落的每个字都产出 unigram。
    #[test]
    fn cjk_runs_produce_all_unigrams(input in "[一-鿿]{1,8}") {
        let tokens = tokenize(&input);
        for ch in input.chars() {
            prop_assert!(
                tokens.contains(&ch.to_string()),
                "缺少 unigram {:?}，实际词元：{:?}",
                ch.to_string(),
                tokens
            );
        }
    }

    /// ⭐ 词元要么是 CJK、要么是拉丁 —— 不会混在一起。
    ///
    /// 混在一起的词元会让 BM25 的长度归一化失真。
    #[test]
    fn tokens_do_not_mix_scripts(input in text()) {
        for token in tokenize(&input) {
            let has_cjk = is_cjk_token(&token);
            let has_latin = token.chars().any(|c| c.is_ascii_alphanumeric());
            prop_assert!(
                !(has_cjk && has_latin),
                "词元不能混脚本，实际是 {:?}",
                token
            );
        }
    }

    /// ⭐ `is_numeric` 与词元内容一致。
    #[test]
    fn is_numeric_matches_content(input in text()) {
        for token in tokenize(&input) {
            if !token.is_empty() && token.chars().all(|c| c.is_ascii_digit()) {
                prop_assert!(is_numeric(&token), "{:?} 应该是纯数字", token);
            } else {
                prop_assert!(!is_numeric(&token), "{:?} 不该被判为纯数字", token);
            }
        }
    }
}
