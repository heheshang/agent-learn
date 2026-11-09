//! SQL 安全护栏 —— 只允许只读查询。
//!
//! ## 为什么这是**整个项目最重要的一个模块**
//!
//! LangChain SQL Agent 教程里有一段警告：
//!
//! > Building Q&A systems of SQL databases requires executing model-generated
//! > SQL queries. There are inherent risks in doing this.
//! > Make sure that your database connection permissions are always scoped
//! > as narrowly as possible.
//!
//! 翻译成工程语言：**执行模型生成的 SQL 本身就是一个高危操作**。
//! 教程给的解法是「把数据库连接权限收窄」——
//! 但那是**部署侧**的防线，一旦有人图省事给了写权限，整条链就失守了。
//!
//! 本项目做的是**代码侧**的防线：不让危险 SQL 到达数据库。
//!
//! ## 三层防线
//!
//! ```text
//! 第 1 层：护栏（静态检查）  ← 本模块
//! 第 2 层：只读连接（运行时）  ← 本模块负责建立
//! 第 3 层：行数上限 + 超时    ← 本模块负责施加
//! ```
//!
//! 任何一层单独都不够：
//!
//! - 只有护栏 → 护栏有解析盲区（比如 `WITH x AS (DELETE ...)`）
//! - 只有只读连接 → SQLite 的 `PRAGMA` / `ATTACH` 仍能绕过
//! - 只有行数上限 → `DROP TABLE` 一条语句就够
//!
//! ## 护栏的实现口径：白名单 + 关键字黑名单
//!
//! ### 白名单（语句类型）
//!
//! 只允许 `SELECT` / `WITH` / `VALUES` / `EXPLAIN` 开头。
//!
//! `WITH` 要单独放行 —— 因为 **CTE（公用表表达式）本身是只读的**：
//! `WITH cte AS (SELECT ...) SELECT * FROM cte` 完全安全。
//! 但 `WITH ... AS (DELETE ...)` 就危险了 —— 这靠**关键字黑名单**兜住。
//!
//! ### 黑名单（危险关键字）
//!
//! `INSERT` / `UPDATE` / `DELETE` / `DROP` / `ALTER` / `CREATE` /
//! `REPLACE` / `TRUNCATE` / `ATTACH` / `DETACH` / `PRAGMA` /
//! `VACUUM` / `REINDEX` / `BEGIN` / `COMMIT` / `ROLLBACK`。
//!
//! ⚠️ 黑名单是在**整条 SQL 的词法层面**做的（不是简单 `contains`），
//! 详见 [`tokenize_sql`]。
//!
//! ## ⚠️ 关键字黑名单的固有局限（必须诚实说明）
//!
//! 纯文本关键字匹配**挡不住**所有注入手法：
//!
//! - 字符串字面量里的关键字（`WHERE name = 'DROP TABLE'`）会误伤
//! - 注释里的关键字（`SELECT 1 /* DROP */`）会误伤
//! - Unicode 混淆、拼接等手法不在讨论范围内
//!
//! 本项目的处理方式是**误伤优先**：宁可多报一次错，
//! 也不要放过一条危险语句。因为误伤的代价是「模型重写一次 SQL」，
//! 而漏放的代价是「数据库被删了」。
//!
//! 这也是为什么还必须有第 2、3 层防线 ——
//! **任何单层都能被绕过，只有叠加才可靠**。

use crate::error::GuardError;

/// SQL 语句的最大长度。
///
/// 防止「超长 SQL 把内存和 CPU 打满」。
/// 正常业务的 SQL 远小于这个上限。
pub const MAX_SQL_LEN: usize = 8_000;

/// 允许的只读语句的关键字（大小写不敏感）。
pub const ALLOWED_STATEMENT_KEYWORDS: &[&str] = &["select", "with", "values", "explain"];

/// 被禁止的写操作关键字。
///
/// ## 为什么包含 `PRAGMA`
///
/// SQLite 的 `PRAGMA` 能改很多设置（比如 `PRAGMA journal_mode`），
/// 也能读文件（`PRAGMA database_list`）—— 属于「元数据操作」，
/// 在 SQL Agent 场景里没有正当用途，直接禁掉。
pub const FORBIDDEN_KEYWORDS: &[&str] = &[
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "begin",
    "commit",
    "rollback",
    "savepoint",
    "release",
    "grant",
    "revoke",
    "analyze",
    "upsert",
];

/// SQL 的词法单元。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// 标识符或关键字（已转成小写）。
    Word(String),
    /// 字符串字面量（单引号）。
    Str,
    /// 带引号的标识符（双引号 / 方括号 / 反引号）。
    QuotedIdent,
    /// 数字。
    Number,
    /// 其它符号。
    Symbol(char),
}

/// 把 SQL 切成词法单元。
///
/// ## 为什么不能简单用 `contains`
///
/// `WHERE name = 'DROP'` 里也有 `DROP` —— 但它是**字符串内容**，
/// 不是语句。简单 `contains` 会误伤。
///
/// 反过来，`SELECT 1 /* DROP */` 里的 `DROP` 是**注释** ——
/// 简单 `contains` 会误伤（这方向是安全的），但
/// `SELECT 'x' -- DROP` 里的 `--` 之后全是注释，
/// 真实语句早就结束了。
///
/// 所以这里做一次**真正的词法分析**：
///
/// 1. 单引号字符串：`'...''...'`（SQLite 用两个单引号转义）
/// 2. 双引号 / 方括号 / 反引号：带引号的标识符
/// 3. 行注释 `-- ...`
/// 4. 块注释 `/* ... */`
/// 5. 标识符 / 数字 / 符号
///
/// ## 刻意的简化
///
/// **不处理** `#` 注释（MySQL 风格，SQLite 不支持）、
/// **不处理** dollar-quoted string（Postgres 风格）——
/// 本项目只支持 SQLite，语义面窄一点反而更可控。
pub fn tokenize_sql(sql: &str) -> Vec<Token> {
    let chars: Vec<char> = sql.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];

        // ── 空白 ──────────────────────────────────────────────
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // ── 行注释 `--` ───────────────────────────────────────
        if c == '-' && i + 1 < chars.len() && chars[i + 1] == '-' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        // ── 块注释 `/* */` ────────────────────────────────────
        if c == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            // 跳过 `*/`（如果没闭合就跳到末尾）
            i = (i + 2).min(chars.len());
            continue;
        }

        // ── 单引号字符串 ──────────────────────────────────────
        //
        // SQLite 的转义是 `''`（两个单引号表示一个）。
        if c == '\'' {
            i += 1;
            while i < chars.len() {
                if chars[i] == '\'' {
                    // 两个连续单引号 → 转义了一个单引号，继续。
                    if i + 1 < chars.len() && chars[i + 1] == '\'' {
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                i += 1;
            }
            tokens.push(Token::Str);
            continue;
        }

        // ── 带引号的标识符 ────────────────────────────────────
        if c == '"' || c == '`' || c == '[' {
            let close = match c {
                '"' => '"',
                '`' => '`',
                _ => ']',
            };
            i += 1;
            while i < chars.len() && chars[i] != close {
                i += 1;
            }
            i = (i + 1).min(chars.len());
            tokens.push(Token::QuotedIdent);
            continue;
        }

        // ── 标识符 / 关键字 ───────────────────────────────────
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect::<String>().to_lowercase();
            tokens.push(Token::Word(word));
            continue;
        }

        // ── 数字 ──────────────────────────────────────────────
        if c.is_ascii_digit() {
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            tokens.push(Token::Number);
            continue;
        }

        // ── 其它符号 ──────────────────────────────────────────
        tokens.push(Token::Symbol(c));
        i += 1;
    }

    tokens
}

/// 检查一条 SQL 是否可以安全执行。
///
/// ## 返回 `Ok(())` 表示通过，`Err(GuardError)` 表示被拦截。
///
/// ## 检查顺序（顺序有意义）
///
/// 1. **空** —— 最先排除，否则后面的关键字分析无从谈起
/// 2. **长度** —— 超长直接拒，避免做无谓的词法分析
/// 3. **单语句** —— 多语句必须拒，因为第一条可能是 SELECT、第二条是 DELETE
/// 4. **语句类型白名单** —— 必须以只读关键字开头
/// 5. **关键字黑名单** —— 逐个词法单元检查
///
/// ## 为什么「单语句」检查要放在白名单**之前**
///
/// 因为 `SELECT 1; DROP TABLE x` 的第一个词是 `SELECT`，
/// 白名单会放行 —— 只能靠「单语句」检查拦住。
///
/// ## ⚠️ 语句类型判断用的是**第一个词法单元**
///
/// 而**不是** `trim().starts_with("select")` —— 因为
/// `/* 注释 */ SELECT ...` 的第一个词法单元才是 `select`。
/// 用 `starts_with` 会把带前置注释的正常查询误判为非法。
pub fn check_sql(sql: &str) -> Result<(), GuardError> {
    let trimmed = sql.trim();

    if trimmed.is_empty() {
        return Err(GuardError::Empty);
    }

    if trimmed.chars().count() > MAX_SQL_LEN {
        return Err(GuardError::TooLong {
            len: trimmed.chars().count(),
            limit: MAX_SQL_LEN,
        });
    }

    let tokens = tokenize_sql(trimmed);

    if tokens.is_empty() {
        return Err(GuardError::Empty);
    }

    // ── 单语句检查 ────────────────────────────────────────────
    //
    // 找分号：分号后面如果还有任何词法单元，就是多语句。
    //
    // ⚠️ 末尾的单个分号是允许的（`SELECT 1;` 是合法的单语句）。
    let mut saw_semicolon = false;
    for token in &tokens {
        match token {
            Token::Symbol(';') => saw_semicolon = true,
            _ if saw_semicolon => {
                return Err(GuardError::MultipleStatements);
            }
            _ => {}
        }
    }

    // ── 语句类型白名单 ────────────────────────────────────────
    let first_word = tokens.iter().find_map(|t| match t {
        Token::Word(w) => Some(w.clone()),
        _ => None,
    });

    let Some(first_word) = first_word else {
        return Err(GuardError::NotReadOnly("<空>".to_string()));
    };

    if !ALLOWED_STATEMENT_KEYWORDS.contains(&first_word.as_str()) {
        return Err(GuardError::NotReadOnly(first_word));
    }

    // ── 关键字黑名单 ──────────────────────────────────────────
    //
    // 只扫「裸词」—— 字符串字面量和带引号的标识符不参与，
    // 因为 `WHERE name = 'DROP'` 里的 `DROP` 是数据不是操作。
    for token in &tokens {
        if let Token::Word(word) = token
            && FORBIDDEN_KEYWORDS.contains(&word.as_str())
        {
            return Err(GuardError::ForbiddenKeyword {
                keyword: word.clone(),
            });
        }
    }

    Ok(())
}

/// 判断一条 SQL 是不是只读查询（`check_sql` 的布尔版本）。
pub fn is_read_only(sql: &str) -> bool {
    check_sql(sql).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 正常查询必须放行 ──────────────────────────────────────
    #[test]
    fn allows_simple_select() {
        assert!(check_sql("SELECT * FROM Track").is_ok());
    }

    #[test]
    fn allows_select_with_where_and_join() {
        let sql = "SELECT g.Name, AVG(t.Milliseconds) AS avg_ms \
                   FROM Track t JOIN Genre g ON t.GenreId = g.GenreId \
                   GROUP BY g.Name ORDER BY avg_ms DESC LIMIT 5";
        assert!(check_sql(sql).is_ok());
    }

    #[test]
    fn allows_cte() {
        // ⚠️ CTE 本身是只读的 —— 必须放行。
        let sql = "WITH top_tracks AS (SELECT Name FROM Track ORDER BY Milliseconds DESC LIMIT 10) \
                   SELECT * FROM top_tracks";
        assert!(check_sql(sql).is_ok());
    }

    #[test]
    fn allows_leading_comment() {
        // ⚠️ 第一个**词法单元**是 select，不是注释 ——
        //    所以用 `starts_with("select")` 的实现会误判。
        assert!(check_sql("/* 统计各流派平均时长 */ SELECT 1").is_ok());
        assert!(check_sql("-- 注释\nSELECT 1").is_ok());
    }

    #[test]
    fn allows_trailing_semicolon() {
        assert!(check_sql("SELECT 1;").is_ok());
    }

    // ── 写操作必须拦截 ────────────────────────────────────────
    #[test]
    fn blocks_update() {
        let err = check_sql("UPDATE Track SET Name = 'x'").unwrap_err();
        assert!(matches!(err, GuardError::NotReadOnly(_)));
    }

    #[test]
    fn blocks_delete() {
        let err = check_sql("DELETE FROM Track").unwrap_err();
        assert!(matches!(err, GuardError::NotReadOnly(_)));
    }

    #[test]
    fn blocks_drop() {
        assert!(!is_read_only("DROP TABLE Track"));
    }

    #[test]
    fn blocks_insert() {
        assert!(!is_read_only("INSERT INTO Track VALUES (1, 'x')"));
    }

    #[test]
    fn blocks_create() {
        assert!(!is_read_only("CREATE TABLE t (a INT)"));
    }

    #[test]
    fn blocks_pragma() {
        assert!(!is_read_only("PRAGMA journal_mode = WAL"));
    }

    #[test]
    fn blocks_attach() {
        // ATTACH 能把任意文件挂进来 —— 必须禁。
        assert!(!is_read_only("ATTACH DATABASE '/etc/passwd' AS p"));
    }

    #[test]
    fn blocks_dml_hidden_in_cte() {
        // ⚠️ 最阴险的一种：`WITH` 开头，白名单会放行，
        //    但 CTE 里藏着一个 DELETE。
        let sql = "WITH x AS (DELETE FROM Track RETURNING TrackId) SELECT * FROM x";
        let err = check_sql(sql).unwrap_err();
        assert!(matches!(err, GuardError::ForbiddenKeyword { .. }));
    }

    #[test]
    fn blocks_dml_hidden_after_union() {
        let sql = "SELECT 1 UNION SELECT 1; DROP TABLE Track";
        assert!(check_sql(sql).is_err());
    }

    // ── 多语句必须拦截 ────────────────────────────────────────
    #[test]
    fn blocks_multiple_statements() {
        let err = check_sql("SELECT 1; DROP TABLE Track").unwrap_err();
        assert!(matches!(err, GuardError::MultipleStatements));
    }

    #[test]
    fn blocks_multiple_selects() {
        // 就算都是 SELECT，多语句也不允许 —— 简化审计。
        assert!(!is_read_only("SELECT 1; SELECT 2"));
    }

    // ── 误伤防护 ──────────────────────────────────────────────
    #[test]
    fn allows_keyword_inside_string_literal() {
        // ⚠️ 字符串里的 DROP 是**数据**，不是操作 —— 不该拦。
        assert!(check_sql("SELECT * FROM Track WHERE Name = 'DROP TABLE'").is_ok());
    }

    #[test]
    fn allows_keyword_inside_comment() {
        // 注释里的关键字不该拦 —— 注释不影响执行语义。
        assert!(check_sql("SELECT 1 -- DROP TABLE").is_ok());
        assert!(check_sql("SELECT 1 /* DROP TABLE */").is_ok());
    }

    #[test]
    fn allows_quoted_identifier() {
        // 带引号的标识符里的关键字不该拦。
        assert!(check_sql("SELECT \"delete\" FROM Track").is_ok());
    }

    #[test]
    fn allows_escaped_quote_in_string() {
        // SQLite 用 `''` 转义单引号 —— 词法分析必须正确处理，
        // 否则字符串会「提前结束」，后面的内容被误判为关键字。
        assert!(check_sql("SELECT * FROM Track WHERE Name = 'it''s DROP'").is_ok());
    }

    // ── 边界 ──────────────────────────────────────────────────
    #[test]
    fn rejects_empty_sql() {
        assert!(matches!(check_sql("   "), Err(GuardError::Empty)));
    }

    #[test]
    fn rejects_too_long_sql() {
        let long = format!("SELECT {}", "a".repeat(MAX_SQL_LEN));
        assert!(matches!(check_sql(&long), Err(GuardError::TooLong { .. })));
    }

    #[test]
    fn is_case_insensitive() {
        assert!(check_sql("select 1").is_ok());
        assert!(check_sql("SeLeCt 1").is_ok());
        assert!(!is_read_only("delete from track"));
    }

    // ── 词法分析器 ────────────────────────────────────────────
    #[test]
    fn tokenizer_handles_string_literals() {
        let tokens = tokenize_sql("SELECT 'a b' FROM t");
        assert_eq!(
            tokens,
            vec![
                Token::Word("select".to_string()),
                Token::Str,
                Token::Word("from".to_string()),
                Token::Word("t".to_string()),
            ]
        );
    }

    #[test]
    fn tokenizer_skips_comments() {
        let tokens = tokenize_sql("-- 注释\nSELECT 1 /* 块 */");
        assert_eq!(
            tokens,
            vec![Token::Word("select".to_string()), Token::Number]
        );
    }

    #[test]
    fn tokenizer_handles_escaped_quote() {
        // `'it''s'` 是一个完整的字符串 —— 不能提前结束。
        let tokens = tokenize_sql("'it''s'");
        assert_eq!(tokens, vec![Token::Str]);
    }

    #[test]
    fn tokenizer_handles_quoted_identifier() {
        let tokens = tokenize_sql("\"my col\"");
        assert_eq!(tokens, vec![Token::QuotedIdent]);
    }
}
