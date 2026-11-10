//! 属性测试（property-based testing）—— 15-agent-sql 的 **SQL 安全护栏**。
//!
//! ## 为什么护栏最需要属性测试
//!
//! 护栏是**安全关键**模块：它决定「模型生成的 SQL 能不能碰数据库」。
//! 它的输入是**模型生成的任意 SQL** —— 和 13 的 ReAct 解析器一样，
//! 完全不可信。
//!
//! 单元测试只能覆盖「人想得到的」攻击手法。
//! 属性测试让**随机拼接**的 SQL 去撞安全不变量，
//! 失败时自动收缩成最小反例 —— 往往就是一条「从没想过但能绕过」的语句。
//!
//! ## 本项目自己写护栏的理由（README 里的三层防线）
//!
//! ```text
//! 第 1 层：护栏（静态检查）  ← 本文件测的
//! 第 2 层：只读连接（运行时）
//! 第 3 层：行数上限 + 超时
//! ```
//!
//! 护栏**不能**简单用 `contains` ——
//! `WHERE name = 'DROP TABLE'` 里的 DROP 是**数据**不是操作，
//! `SELECT 1 /* DROP */` 里的 DROP 是**注释**。
//! 所以必须做真正的词法分析（见 `tokenize_sql`）。
//!
//! ## 不变量清单
//!
//! | 不变量 | 含义 |
//! |---|---|
//! | `never_panics` | 任意 SQL 都不 panic |
//! | `dangerous_keywords_always_rejected` | **黑名单关键字裸词出现必被拦**（安全底线） |
//! | `dangerous_keywords_in_strings_allowed` | 字符串/注释里的关键字**不该**被误伤 |
//! | `multiple_statements_always_rejected` | 分号后有内容必被拦 |
//! | `non_readonly_prefix_always_rejected` | 非只读开头必被拦 |
//! | `tokenizer_never_leaks_comments` | 注释里的内容不进词法单元 |
//! | `tokenizer_never_leaks_strings` | 字符串字面量整体算一个 `Str` |

use agent_15::error::GuardError;
use agent_15::guard::{FORBIDDEN_KEYWORDS, Token, check_sql, is_read_only, tokenize_sql};
use proptest::prelude::*;

/// 危险的写操作关键字（挑代表性的几个）。
const DANGEROUS: [&str; 6] = ["insert", "update", "delete", "drop", "alter", "pragma"];

/// 只读关键字。
const READONLY: [&str; 4] = ["select", "with", "values", "explain"];

/// 生成一段任意 SQL 噪声（**含**分号、注释符、引号 —— 什么都可能）。
///
/// 用于「不 panic」「必被拦」这类**不需要精确构造**的性质。
fn sql_noise() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            // 只读关键字
            prop::sample::select(&READONLY).prop_map(|s| s.to_string()),
            // 危险关键字（裸词）
            prop::sample::select(&DANGEROUS).prop_map(|s| s.to_string()),
            // 危险关键字（藏在字符串里）
            Just("'DROP TABLE'".to_string()),
            Just("'it''s a delete'".to_string()),
            // 危险关键字（藏在注释里）
            Just("-- DROP".to_string()),
            Just("/* UPDATE */".to_string()),
            // 引号与标识符
            Just("'".to_string()),
            Just("\"".to_string()),
            Just("`".to_string()),
            Just("[".to_string()),
            Just("]".to_string()),
            // 分号（多语句）
            Just(";".to_string()),
            // 注释符
            Just("--".to_string()),
            Just("/*".to_string()),
            Just("*/".to_string()),
            // 普通内容
            Just("t".to_string()),
            Just("Name".to_string()),
            Just("1".to_string()),
            Just(" ".to_string()),
            Just("\n".to_string()),
            Just("(".to_string()),
            Just(")".to_string()),
            Just(",".to_string()),
            Just("*".to_string()),
            Just("=".to_string()),
            Just("FROM".to_string()),
            Just("WHERE".to_string()),
            // 任意字符
            any::<char>().prop_map(|c| c.to_string()),
        ],
        0..40,
    )
    .prop_map(|parts| parts.concat())
}

/// 生成一段**不含分号**的 SQL 噪声。
///
/// ## 为什么需要单独一个版本
///
/// 「不该被拦」类性质（如「字符串里的危险关键字是合法的」）
/// 需要排除一种**正当的拦截**：**多语句**。
///
/// 因为一旦 SQL 里有分号且分号后还有内容，
/// 护栏会以 `MultipleStatements` 拦掉 —— 那是**正确行为**，
/// 与「关键字误伤」是两回事。不排除它，
/// 性质就会把「正确拦截」当成「误伤」报出来。
fn sql_noise_no_semicolon() -> impl Strategy<Value = String> {
    sql_noise().prop_filter("噪声里不含分号", |s| !s.contains(';'))
}

/// 生成一段**不含分号、也不含注释符**的 SQL 噪声。
///
/// 用于「分号后必须有真实内容」的性质 ——
/// 因为未闭合的块注释（`/* DROP`）会把后面整段吞掉，
/// 那时「分号后有内容」并不成立。
fn sql_noise_plain() -> impl Strategy<Value = String> {
    sql_noise().prop_filter("噪声里不含分号与注释符", |s| {
        !s.contains(';') && !s.contains("--") && !s.contains("/*") && !s.contains("*/")
    })
}

/// 生成一条「裸词包含危险关键字」的 SQL。
///
/// 「裸词」的构造方式：关键字前后用空格隔开，保证是独立的标识符。
fn sql_with_bare_dangerous_word() -> impl Strategy<Value = String> {
    prop::sample::select(&DANGEROUS).prop_map(|word| format!("SELECT * FROM t WHERE x = {word}"))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    /// ⭐ 任意 SQL 都不 panic。
    #[test]
    fn guard_never_panics(sql in sql_noise()) {
        let _ = check_sql(&sql);
        let _ = is_read_only(&sql);
        let _ = tokenize_sql(&sql);
    }

    /// ⭐⭐ **安全底线**：黑名单关键字以**裸词**形式出现时，必被拦截。
    ///
    /// 这条性质是整个护栏的价值所在 ——
    /// 一旦它被违反，就意味着「模型可以用一条 SQL 删库」。
    #[test]
    fn dangerous_keywords_as_bare_words_are_always_rejected(
        sql in sql_with_bare_dangerous_word(),
    ) {
        let result = check_sql(&sql);

        prop_assert!(
            result.is_err(),
            "含裸词危险关键字的 SQL 必须被拦截，但它通过了：{}",
            sql
        );

        // 且必须是「关键字黑名单」或「非只读开头」这两种拦截之一。
        match result {
            Err(GuardError::ForbiddenKeyword { .. }) | Err(GuardError::NotReadOnly(_)) => {}
            other => prop_assert!(false, "拦截原因不对，实际是 {:?}", other),
        }
    }

    /// ⚠️ 危险关键字藏在**字符串字面量**里时**不该**被拦。
    ///
    /// 这是「误伤优先」原则的另一面 ——
    /// `WHERE name = 'DROP TABLE'` 是**合法查询**，
    /// 拦掉它会让模型白白重写一次。
    #[test]
    fn dangerous_keywords_inside_strings_are_allowed(sql in sql_noise_no_semicolon()) {
        // 构造一条「只在字符串里含危险关键字」的合法查询。
        let sql = format!("SELECT * FROM t WHERE name = 'DROP TABLE' {sql}");

        // 但如果额外拼进去的内容里**又**出现了裸词危险关键字，
        // 那就该拦 —— 这条交给上面那条性质管。
        let has_bare_dangerous = tokenize_sql(&sql).iter().any(|tok| match tok {
            Token::Word(w) => FORBIDDEN_KEYWORDS.contains(&w.as_str()),
            _ => false,
        });
        prop_assume!(!has_bare_dangerous);

        prop_assert!(
            check_sql(&sql).is_ok(),
            "字符串里的危险关键字不该被拦：{}",
            sql
        );
    }

    /// ⚠️ 危险关键字藏在**注释**里时也**不该**被拦。
    #[test]
    fn dangerous_keywords_inside_comments_are_allowed(sql in sql_noise_no_semicolon()) {
        let sql = format!("SELECT * FROM t /* DROP TABLE */ {sql}");

        let has_bare_dangerous = tokenize_sql(&sql).iter().any(|tok| match tok {
            Token::Word(w) => FORBIDDEN_KEYWORDS.contains(&w.as_str()),
            _ => false,
        });
        prop_assume!(!has_bare_dangerous);

        prop_assert!(
            check_sql(&sql).is_ok(),
            "注释里的危险关键字不该被拦：{}",
            sql
        );
    }

    /// ⭐ **多语句必被拦** —— `SELECT 1; DROP TABLE x` 是经典注入手法。
    ///
    /// ⚠️ 关键细节：分号后必须有**真实的词法内容**。
    /// 末尾单个分号（`SELECT 1;`）是合法的单语句 —— 这是 15 的明确设计。
    #[test]
    fn multiple_statements_are_always_rejected(
        first in sql_noise_plain(),
        second in sql_noise_plain(),
    ) {
        // 第二段以裸词危险关键字开头，确保「分号后有内容」这件事成立。
        let sql = format!("SELECT 1; {first} DROP {second}");

        let result = check_sql(&sql);
        prop_assert!(
            result.is_err(),
            "多语句必须被拦截，但它通过了：{}",
            sql
        );
    }

    /// 末尾单个分号是**合法**的（不算多语句）。
    #[test]
    fn trailing_semicolon_is_allowed(sql in sql_noise_no_semicolon()) {
        let sql = format!("SELECT * FROM t {sql};");

        let has_bare_dangerous = tokenize_sql(&sql).iter().any(|tok| match tok {
            Token::Word(w) => FORBIDDEN_KEYWORDS.contains(&w.as_str()),
            _ => false,
        });
        prop_assume!(!has_bare_dangerous);

        prop_assert!(
            check_sql(&sql).is_ok(),
            "末尾单个分号不该被拦：{}",
            sql
        );
    }

    /// 非只读开头的语句必被拦。
    #[test]
    fn non_readonly_prefix_is_always_rejected(
        word in prop::sample::select(&DANGEROUS),
        rest in sql_noise(),
    ) {
        let sql = format!("{word} t {rest}");

        prop_assert!(
            check_sql(&sql).is_err(),
            "以危险关键字开头的 SQL 必须被拦截，但它通过了：{}",
            sql
        );
    }

    /// 只读开头的语句，只要没有裸词危险关键字，就该放行。
    ///
    /// 这条是「误伤优先」的另一半 —— 护栏不能把正常查询也拦掉。
    #[test]
    fn clean_readonly_queries_are_allowed(sql in sql_noise()) {
        let sql = format!("SELECT * FROM t {sql}");

        let has_bare_dangerous = tokenize_sql(&sql).iter().any(|tok| match tok {
            Token::Word(w) => FORBIDDEN_KEYWORDS.contains(&w.as_str()),
            _ => false,
        });
        prop_assume!(!has_bare_dangerous);

        // 单语句要求：分号后面不能再有内容。
        let has_trailing = sql
            .trim()
            .trim_end_matches(';')
            .contains(';');
        prop_assume!(!has_trailing);

        prop_assert!(
            check_sql(&sql).is_ok(),
            "干净的只读查询不该被拦：{}",
            sql
        );
    }

    /// ⚠️ 词法分析**不把注释内容变成单词**。
    #[test]
    fn comments_never_become_words(
        comment_body in "[a-zA-Z ] {0,20}",
    ) {
        let sql = format!("SELECT 1 /* {comment_body} */");
        let words: Vec<String> = tokenize_sql(&sql)
            .into_iter()
            .filter_map(|t| match t {
                Token::Word(w) => Some(w),
                _ => None,
            })
            .collect();

        prop_assert!(
            !words.contains(&"drop".to_string()),
            "注释内容不该变成单词，实际词表：{:?}",
            words
        );
    }

    /// ⚠️ 词法分析把整个字符串字面量算成**一个** `Str`。
    #[test]
    fn strings_are_single_tokens(
        content in "[a-zA-Z ]{0,20}",
    ) {
        let sql = format!("SELECT '{content}'");
        let str_count = tokenize_sql(&sql)
            .iter()
            .filter(|t| matches!(t, Token::Str))
            .count();

        prop_assert_eq!(
            str_count, 1,
            "字符串字面量应该只算一个 Str token，实际是 {}",
            str_count
        );
    }

    /// ⚠️ 转义的单引号（`''`）不打断字符串。
    #[test]
    fn escaped_quotes_do_not_break_strings(
        content in "[a-zA-Z]{0,15}",
    ) {
        // `'it''s {content}'` —— 中间的 `''` 是转义，不是字符串结束。
        let sql = format!("SELECT 'it''s {content}' FROM t");
        let str_count = tokenize_sql(&sql)
            .iter()
            .filter(|t| matches!(t, Token::Str))
            .count();

        prop_assert_eq!(
            str_count, 1,
            "转义单引号不该把字符串拆开，实际 Str token 数：{}",
            str_count
        );
    }

    /// 超长 SQL 必被拦。
    #[test]
    fn overly_long_sql_is_rejected(
        padding in "[a-z ]{8000,8100}",
    ) {
        let sql = format!("SELECT {padding}");
        let result = check_sql(&sql);

        prop_assert!(
            matches!(result, Err(GuardError::TooLong { .. })),
            "超长 SQL 必须以 TooLong 被拦，实际是 {:?}",
            result
        );
    }

    /// 空 SQL 必被拦。
    #[test]
    fn empty_sql_is_rejected(ws in "[ \t\n]{0,5}") {
        prop_assert!(
            matches!(check_sql(&ws), Err(GuardError::Empty)),
            "空 SQL 必须被拦"
        );
    }
}
