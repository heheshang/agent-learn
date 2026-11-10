//! 模糊测试目标 —— 15-agent-sql 的 **SQL 安全护栏**。
//!
//! ## 这个目标在找什么
//!
//! **两类问题**：
//!
//! 1. **panic** —— 护栏是数据库前的第一道门，它挂了等于门没了。
//! 2. **绕过** —— 一条危险 SQL 被 `check_sql` 放行。
//!    这是**最严重的**：它意味着「模型可以让数据库执行写操作」。
//!
//! ## 绕过怎么判定
//!
//! 用**独立实现**的黑名单做对照：
//!
//! - 护栏放行（`Ok`）
//! - 但独立扫描发现 SQL 里有**裸词**危险关键字
//!
//! → 判定为绕过，panic。
//!
//! 独立扫描只做「去空白、去字符串字面量、去注释」后的裸词匹配 ——
//! 这正是护栏应该做的事，两边独立实现才能互为对照。
//!
//! ## ⚠️ 为什么要「独立实现」
//!
//! 如果直接用护栏自己的 `tokenize_sql` 做对照，
//! 那就变成「用被测代码验证被测代码」——
//! 护栏的词法分析有 bug 时，两边会一起错，检测不出来。
//!
//! 这里的独立实现故意写得**更简单、更保守**：
//! 只做最朴素的「按空白切词 + 去掉引号内内容」。
//! 它会**误报**（比如把注释里的词也算进去），
//! 所以判定绕过时还要求护栏**确实放行**了 ——
//! 误报方向是「报出可疑但不 panic 就不算失败」。

#![no_main]

use libfuzzer_sys::fuzz_target;
use agent_15::guard::{FORBIDDEN_KEYWORDS, check_sql, tokenize_sql};

/// 危险关键字的「宽松匹配」集合（含各种大小写与常见变形）。
fn is_dangerous_word(word: &str) -> bool {
    let w = word.to_ascii_lowercase();
    FORBIDDEN_KEYWORDS.iter().any(|k| *k == w)
}

/// 独立实现的「裸词扫描」。
///
/// 故意写得简单：去掉字符串字面量与注释，然后按非标识符字符切词。
/// 它比护栏**更保守**（会多报），所以只用来发现「护栏漏放」的方向。
fn has_bare_dangerous_word(sql: &str) -> bool {
    // 去掉单引号字符串（含 '' 转义）。
    let mut cleaned = String::with_capacity(sql.len());
    let mut chars = sql.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' {
            // 跳过字符串内容，遇到未转义的 ' 结束。
            while let Some(c2) = chars.next() {
                if c2 == '\'' {
                    // 两个连续单引号 = 转义，继续留在字符串里。
                    if chars.peek() == Some(&'\'') {
                        chars.next();
                        continue;
                    }
                    break;
                }
            }
            cleaned.push(' ');
        } else {
            cleaned.push(c);
        }
    }

    // 去掉行注释与块注释。
    let cleaned = strip_comments(&cleaned);

    // 按非标识符字符切词。
    cleaned
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .any(is_dangerous_word)
}

/// 去掉注释（简单实现：找 `--` 到行尾、找 `/*` 到 `*/`）。
fn strip_comments(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len());
    let bytes: Vec<char> = sql.chars().collect();
    let mut i = 0;

    while i < bytes.len() {
        // 行注释
        if bytes[i] == '-' && i + 1 < bytes.len() && bytes[i + 1] == '-' {
            while i < bytes.len() && bytes[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // 块注释
        if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == '*' && bytes[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

fuzz_target!(|data: &[u8]| {
    let sql = String::from_utf8_lossy(data);

    // 1. 不能 panic。
    let result = check_sql(&sql);

    // 2. 不能绕过。
    //
    // ⚠️ 只有护栏**放行**且独立扫描发现裸词危险关键字时才算绕过。
    //
    // 误报方向（独立扫描报了但护栏拦了）不算失败 ——
    // 那是「护栏比独立实现更严」，是好事。
    if result.is_ok() && has_bare_dangerous_word(&sql) {
        panic!("护栏漏放了含裸词危险关键字的 SQL: {sql:?}");
    }

    // 3. 自洽性：护栏用 `tokenize_sql` 判定，
    //    那么「护栏放行」时，tokenize 结果里不该有黑名单裸词。
    //
    //    这条比第 2 条更严格（用的是护栏自己的词法分析），
    //    能抓到「黑名单检查根本没生效」这类实现缺失。
    if result.is_ok() {
        let leaked = tokenize_sql(&sql).iter().any(|tok| match tok {
            agent_15::guard::Token::Word(w) => FORBIDDEN_KEYWORDS.contains(&w.as_str()),
            _ => false,
        });
        assert!(
            !leaked,
            "护栏放行了含裸词危险关键字的 SQL: {sql:?}"
        );
    }
});
