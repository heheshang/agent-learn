//! SQL 检查器 —— 对应 LangChain 的 `sql_db_query_checker`。
//!
//! ## 教程里的做法
//!
//! ```python
//! @tool
//! def sql_db_query_checker(query: str) -> str:
//!     """Use this tool to double check if your query is correct before executing it."""
//!     trigger_prompt = """{query}
//! Double check the sqlite query above for common mistakes, including:
//! - Using NOT IN with NULL values
//! - Using UNION when UNION ALL should have been used
//! - Using BETWEEN for exclusive ranges
//! - Data type mismatch in predicates
//! - Properly quoting identifiers
//! - Using the correct number of arguments for functions
//! - Casting to the correct data type
//! - Using the proper columns for joins
//!
//! If there are any of the above mistakes, rewrite the query. If there are no
//! mistakes, just reproduce the original query.
//!
//! Output the final SQL query only.
//!
//! SQL Query: """.format(query=query)
//!
//!     response = model.invoke(trigger_prompt)
//!     return response.text.strip()
//! ```
//!
//! ## 本项目的做法：拆成 trait，两种实现
//!
//! ```text
//! QueryChecker (trait)
//!    ├── StaticChecker —— 纯本地启发式（默认，零成本、确定性）
//!    └── LlmChecker   —— 调模型做语义检查（对齐教程）
//! ```
//!
//! ## 为什么默认用静态检查器
//!
//! 三个理由：
//!
//! 1. **确定性** —— 静态检查器每次给同样的结果，可以被单元测试锁住
//! 2. **零成本** —— 不多一次模型调用（教程的版本每次查询都多一次 LLM 往返）
//! 3. **可解释** —— 告诉模型「你在第几处犯了什么错」，比「模型觉得不对」有用
//!
//! 但静态检查器**抓不到语义错误**（比如「该 JOIN 哪张表」），
//! 所以保留了 LLM 版本 —— 用户可以按需切换。

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use agent_logging::{debug, warn};

use crate::guard::{Token, tokenize_sql};

/// 检查结果。
#[derive(Debug, Clone)]
pub struct CheckOutcome {
    /// 检查后的 SQL（可能已被改写）。
    pub sql: String,

    /// 发现的问题（空 = 没问题）。
    pub issues: Vec<String>,
}

impl CheckOutcome {
    /// 是否没有发现问题。
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }
}

/// SQL 检查器。
///
/// ## 为什么是 async
///
/// LLM 版本需要等模型返回 —— 和仓库里其他 trait 一样，
/// 用 `Pin<Box<dyn Future>>` 显式装箱（`async fn` 不是 dyn 兼容）。
pub trait QueryChecker: Send + Sync {
    /// 检查一条 SQL。
    fn check<'a>(&'a self, sql: &'a str)
    -> Pin<Box<dyn Future<Output = CheckOutcome> + Send + 'a>>;
}

/// 静态检查器 —— 纯本地的启发式规则。
///
/// ## 检查的七类问题（对齐教程里列的那份清单）
///
/// | # | 规则 | 为什么要查 |
/// |---|---|---|
/// | 1 | `NOT IN` + `NULL` | `x NOT IN (1, NULL)` 恒为 NULL（**结果全错**） |
/// | 2 | `UNION` 而非 `UNION ALL` | `UNION` 会去重，改变行数语义 |
/// | 3 | `BETWEEN` 用于开区间 | `BETWEEN` 是闭区间 |
/// | 4 | 引号不匹配的标识符 | `SELECT "col FROM t` 语法错 |
/// | 5 | `SELECT *` | 取了所有列，可能被护栏/规范拒绝 |
/// | 6 | 没有 `LIMIT` 的 `SELECT` | 可能拉回海量数据 |
/// | 7 | `JOIN` 但没有 `ON` | 笛卡尔积 |
///
/// ## 规则 1 是**唯一会导致「答案静默错误」**的规则
///
/// 其它规则要么报错，要么只是性能/风格问题。
/// 但 `NOT IN` + `NULL` 不报错 —— 查询**成功执行**，只是结果永远是空。
/// 这正是「静默错误」最危险的地方。
#[derive(Debug, Default)]
pub struct StaticChecker {
    /// 是否检查「缺少 LIMIT」。
    ///
    /// ## ⚠️ 默认**关闭**
    ///
    /// `LIMIT` 是**性能建议**，不是**正确性错误**。
    ///
    /// 「检查器发现了问题」这句话对模型是有分量的 ——
    /// 如果每次都因为「没写 LIMIT」报一遍，模型会以为检查器在挑刺，
    /// 真正的错误（`NOT IN` + `NULL`）反而被淹没了。
    ///
    /// 所以：默认只查**正确性**问题，
    /// `LIMIT` 检查作为可选规则（[`StaticChecker::with_limit_check`]）单独开。
    pub require_limit: bool,

    /// 是否检查 `SELECT *`。
    pub forbid_star: bool,
}

impl StaticChecker {
    /// 创建检查器（默认规则 —— 只查正确性问题）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 开启「缺少 `LIMIT`」的检查（性能建议）。
    pub fn with_limit_check(mut self) -> Self {
        self.require_limit = true;
        self
    }

    /// 关闭「缺少 LIMIT」的检查。
    pub fn without_limit_check(mut self) -> Self {
        self.require_limit = false;
        self
    }

    /// 开启「禁止 `SELECT *`」的检查。
    pub fn with_star_check(mut self) -> Self {
        self.forbid_star = true;
        self
    }

    /// 跑一遍所有规则。
    pub fn check_sync(&self, sql: &str) -> CheckOutcome {
        let mut issues: Vec<String> = Vec::new();
        let tokens = tokenize_sql(sql);

        // ── 规则 1：`NOT IN` + `NULL` ───────────────────────────
        if let Some(issue) = check_not_in_null(&tokens) {
            issues.push(issue);
        }

        // ── 规则 2：`UNION` 而非 `UNION ALL` ───────────────────
        if let Some(issue) = check_union_without_all(&tokens) {
            issues.push(issue);
        }

        // ── 规则 3：`BETWEEN` 的闭区间语义 ────────────────────
        if let Some(issue) = check_between(&tokens) {
            issues.push(issue);
        }

        // ── 规则 4：引号不匹配 ────────────────────────────────
        if let Some(issue) = check_unbalanced_quotes(sql) {
            issues.push(issue);
        }

        // ── 规则 5：`SELECT *` ─────────────────────────────────
        if self.forbid_star && has_select_star(&tokens) {
            issues.push(
                "使用了 `SELECT *` —— 请只查询与问题相关的列（教程明确要求这一点）".to_string(),
            );
        }

        // ── 规则 6：缺少 `LIMIT` ──────────────────────────────
        if self.require_limit && is_select(&tokens) && !has_limit(&tokens) {
            issues.push(
                "查询没有 `LIMIT` —— 请加上 LIMIT（默认 5 条）以避免拉回过多数据".to_string(),
            );
        }

        // ── 规则 7：`JOIN` 没有 `ON` ───────────────────────────
        if let Some(issue) = check_join_without_on(&tokens) {
            issues.push(issue);
        }

        let mut fixed = sql.trim().to_string();

        // ── 自动修正 ──────────────────────────────────────────
        //
        // 「能确定地改对」的才改：
        //
        // - `NOT IN` + `NULL` → 改成 `NOT EXISTS` 或提示用 `IS NULL`
        // - `UNION` → `UNION ALL`（**有争议**：UNION 可能才是本意）
        //
        // ⚠️ 只改**确定安全**的那一类 —— 其余只提示，不擅自改。
        if issues.iter().any(|i| i.contains("NOT IN")) {
            fixed = suggest_not_in_fix(&fixed);
        }

        if issues.iter().any(|i| i.contains("UNION")) {
            // ⚠️ `UNION` → `UNION ALL` 会改变语义（去重 vs 不去重）。
            //    教程的检查器是让**模型**决定，本项目采取保守策略：
            //    只提示，不自动改 —— 因为模型能自己判断。
        }

        debug!(issue_count = issues.len(), "静态 SQL 检查完成");

        CheckOutcome { sql: fixed, issues }
    }
}

impl QueryChecker for StaticChecker {
    fn check<'a>(
        &'a self,
        sql: &'a str,
    ) -> Pin<Box<dyn Future<Output = CheckOutcome> + Send + 'a>> {
        let outcome = self.check_sync(sql);
        Box::pin(async move { outcome })
    }
}

/// 规则 1：`NOT IN` 子查询/列表里出现 `NULL`。
///
/// ## 为什么这是**静默错误**
///
/// ```sql
/// SELECT * FROM Track WHERE GenreId NOT IN (SELECT GenreId FROM X WHERE X IS NULL)
/// ```
///
/// SQL 三值逻辑：`x NOT IN (1, NULL)` 等价于 `x <> 1 AND x <> NULL`，
/// 而任何值 `<> NULL` 都是 NULL —— 所以整个 WHERE 恒为假，
/// **一行都不返回，但查询「成功」了**。
///
/// 这类错误不会报错，只会让答案悄悄变成空。
fn check_not_in_null(tokens: &[Token]) -> Option<String> {
    let words: Vec<&str> = tokens
        .iter()
        .filter_map(|t| match t {
            Token::Word(w) => Some(w.as_str()),
            _ => None,
        })
        .collect();

    let not_in_pos = words.iter().position(|w| *w == "in")?;
    // `NOT IN` —— 前一个词是 `not`。
    if not_in_pos == 0 || words[not_in_pos - 1] != "not" {
        return None;
    }

    // 从 `IN` 往后找 `NULL` 关键字（最多看到下一个子查询结束）。
    let mut depth = 0usize;
    for token in &tokens[not_in_pos..] {
        match token {
            Token::Symbol('(') => depth += 1,
            Token::Symbol(')') => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            Token::Word(w) if w == "null" && depth >= 1 => {
                return Some(
                    "`NOT IN` 的列表或子查询里包含 NULL —— 这会让条件恒为 NULL，\
                     查询成功但一行都不返回（静默错误）。\
                     请改用 `NOT EXISTS`，或把 NULL 过滤掉（`WHERE x IS NOT NULL`）。"
                        .to_string(),
                );
            }
            _ => {}
        }
    }

    None
}

/// 规则 2：`UNION`（没有 `ALL`）。
fn check_union_without_all(tokens: &[Token]) -> Option<String> {
    let words: Vec<&str> = tokens
        .iter()
        .filter_map(|t| match t {
            Token::Word(w) => Some(w.as_str()),
            _ => None,
        })
        .collect();

    let pos = words.iter().position(|w| *w == "union")?;
    let next = words.get(pos + 1).copied();
    if next == Some("all") {
        return None;
    }

    Some(
        "使用了 `UNION`（会去重）—— 如果你只是想把两个结果集拼起来，\
         `UNION ALL` 更合适（更快，且不会改变行数）。\
         确认确实需要去重的话可以忽略这条。"
            .to_string(),
    )
}

/// 规则 3：`BETWEEN` 的闭区间语义。
fn check_between(tokens: &[Token]) -> Option<String> {
    let words: Vec<&str> = tokens
        .iter()
        .filter_map(|t| match t {
            Token::Word(w) => Some(w.as_str()),
            _ => None,
        })
        .collect();

    if !words.contains(&"between") {
        return None;
    }

    Some(
        "`BETWEEN` 是**闭区间**（包含两端）。\
         如果你想表达「大于 a 且小于 b」的开区间，\
         请改用 `> a AND < b`。"
            .to_string(),
    )
}

/// 规则 4：引号不匹配。
fn check_unbalanced_quotes(sql: &str) -> Option<String> {
    let mut single = false;
    let mut double = false;
    let mut in_single_escaped = false;

    let chars: Vec<char> = sql.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if single {
            if c == '\'' {
                if in_single_escaped {
                    in_single_escaped = false;
                } else if i + 1 < chars.len() && chars[i + 1] == '\'' {
                    in_single_escaped = true;
                    i += 1;
                } else {
                    single = false;
                }
            }
        } else if double {
            if c == '"' {
                double = false;
            }
        } else {
            match c {
                '\'' => single = true,
                '"' => double = true,
                _ => {}
            }
        }
        i += 1;
    }

    if single {
        return Some("单引号没有闭合 —— SQL 会报语法错误".to_string());
    }
    if double {
        return Some("双引号没有闭合 —— SQL 会报语法错误".to_string());
    }
    None
}

/// 规则 5：是否有 `SELECT *`。
fn has_select_star(tokens: &[Token]) -> bool {
    let mut saw_select = false;
    for token in tokens {
        match token {
            Token::Word(w) if w == "select" => saw_select = true,
            Token::Symbol('*') if saw_select => return true,
            Token::Word(w) if w == "from" => saw_select = false,
            _ => {}
        }
    }
    false
}

/// 规则 6：是否有 `LIMIT`。
fn has_limit(tokens: &[Token]) -> bool {
    tokens
        .iter()
        .any(|t| matches!(t, Token::Word(w) if w == "limit"))
}

/// 是否是 `SELECT` 查询。
fn is_select(tokens: &[Token]) -> bool {
    matches!(tokens.first(), Some(Token::Word(w)) if w == "select")
        || matches!(tokens.first(), Some(Token::Word(w)) if w == "with")
}

/// 规则 7：`JOIN` 但没有 `ON`。
fn check_join_without_on(tokens: &[Token]) -> Option<String> {
    let words: Vec<&str> = tokens
        .iter()
        .filter_map(|t| match t {
            Token::Word(w) => Some(w.as_str()),
            _ => None,
        })
        .collect();

    let join_pos = words.iter().position(|w| w.ends_with("join"))?;

    // `CROSS JOIN` 本来就没有 ON。
    //
    // ⚠️ `cross` 在 `join` **之前** —— 所以要看前一个词。
    if join_pos > 0 && words[join_pos - 1] == "cross" {
        return None;
    }

    // JOIN 之后到下一个子句之间应该出现 `on`。
    //
    // ⚠️ `words` 是 `Vec<&str>`，所以 `iter()` 给的是 `&&str`，
    //    `take_while` 的闭包参数是 `&&&str` —— 要解引用**两次**。
    let has_on = words[join_pos + 1..]
        .iter()
        .copied()
        .take_while(|w| {
            !matches!(
                *w,
                "join" | "where" | "group" | "order" | "limit" | "having" | "union"
            )
        })
        .any(|w| w == "on");

    if !has_on {
        return Some(
            "`JOIN` 没有 `ON` 条件 —— 这会产生笛卡尔积（行数暴涨）。\
             请补上正确的关联条件。"
                .to_string(),
        );
    }

    None
}

/// 为 `NOT IN` + `NULL` 生成修正建议。
///
/// ## ⚠️ 这里只做**保守**的改写
///
/// 不能直接改成 `NOT IN` 的等价物 —— SQL 没有简单的等价替换。
/// 所以这里改写成 `NOT EXISTS` 形式，但那需要重写整个查询结构。
///
/// 真正安全的做法是：**只加提示，让模型自己改**。
/// 但我们仍然提供一个**明确的方向**（在 SQL 后面附加注释），
/// 让模型知道该往哪个方向改。
fn suggest_not_in_fix(sql: &str) -> String {
    format!(
        "-- ⚠ 提醒：这条查询里的 `NOT IN` 遇到了 NULL，结果会恒为空。\
         请改写成 `NOT EXISTS (SELECT 1 FROM ... WHERE ... IS NOT NULL)` 的形式。\n{sql}"
    )
}

/// LLM 检查器 —— 调模型做语义检查（对齐 LangChain 教程）。
///
/// ## 与教程的差异
///
/// 教程直接 `model.invoke(trigger_prompt)`。本项目用 [`crate::llm::Llm`]
/// 抽象，所以可以注入 [`crate::llm::ScriptedLlm`] 做确定性测试。
///
/// ## 提示词对齐教程
///
/// 教程列的七类常见错误，本项目**逐条**照抄 —— 因为那是
/// 「SQL 里最容易犯的错」的清单。
pub struct LlmChecker {
    llm: Arc<dyn crate::llm::Llm>,
}

impl LlmChecker {
    /// 创建检查器。
    pub fn new(llm: Arc<dyn crate::llm::Llm>) -> Self {
        Self { llm }
    }
}

impl QueryChecker for LlmChecker {
    fn check<'a>(
        &'a self,
        sql: &'a str,
    ) -> Pin<Box<dyn Future<Output = CheckOutcome> + Send + 'a>> {
        Box::pin(async move {
            let prompt = build_check_prompt(sql);

            match self.llm.chat(LLM_CHECK_SYSTEM, &prompt).await {
                Ok(raw) => {
                    let cleaned = extract_sql(&raw);
                    if cleaned.is_empty() {
                        // 模型没给出 SQL —— 保留原样，但记一条问题。
                        return CheckOutcome {
                            sql: sql.to_string(),
                            issues: vec!["检查器没有返回可解析的 SQL —— 保留原查询".to_string()],
                        };
                    }
                    if cleaned == sql.trim() {
                        CheckOutcome {
                            sql: cleaned,
                            issues: Vec::new(),
                        }
                    } else {
                        CheckOutcome {
                            sql: cleaned,
                            issues: vec!["检查器改写了这条 SQL".to_string()],
                        }
                    }
                }
                Err(e) => {
                    // ⚠️ 检查器失败**不应该是致命错误** ——
                    //    降级成「跳过检查」，让查询照常进行。
                    warn!(error = %e, "LLM 检查器失败 —— 跳过检查");
                    CheckOutcome {
                        sql: sql.to_string(),
                        issues: vec![format!("检查器调用失败（已跳过检查）: {e}")],
                    }
                }
            }
        })
    }
}

/// LLM 检查器的 system 提示词。
///
/// 对齐 LangChain 教程里 `sql_db_query_checker` 的那份清单。
pub const LLM_CHECK_SYSTEM: &str = concat!(
    "你是一个 SQL 审查员。用户会给你一条 SQLite 查询，\
     请检查它有没有下列常见错误：\n",
    "- NOT IN 遇到 NULL 值\n",
    "- 本该用 UNION ALL 却用了 UNION\n",
    "- 用 BETWEEN 表达开区间\n",
    "- 谓词里的数据类型不匹配\n",
    "- 标识符的引号用错了\n",
    "- 函数的参数个数不对\n",
    "- 类型转换（CAST）用错了\n",
    "- JOIN 用了错误的列\n",
    "\n",
    "如果有错误，就改写这条查询；如果没有错误，就原样输出。\
     只输出最终的 SQL 语句，不要任何解释、不要 Markdown 代码块。"
);

/// 构造检查提示词（对齐教程的 `trigger_prompt`）。
pub fn build_check_prompt(sql: &str) -> String {
    format!(
        "{sql}\n\
         Double check the sqlite query above for common mistakes, including:\n\
         - Using NOT IN with NULL values\n\
         - Using UNION when UNION ALL should have been used\n\
         - Using BETWEEN for exclusive ranges\n\
         - Data type mismatch in predicates\n\
         - Properly quoting identifiers\n\
         - Using the correct number of arguments for functions\n\
         - Casting to the correct data type\n\
         - Using the proper columns for joins\n\
         \n\
         If there are any of the above mistakes, rewrite the query. \
         If there are no mistakes, just reproduce the original query.\n\
         \n\
         Output the final SQL query only.\n\
         \n\
         SQL Query:"
    )
}

/// 从模型输出里抽出 SQL 语句。
///
/// ## ⚠️ 必须剥掉 Markdown 代码块
///
/// 模型经常返回带围栏的 SQL：
///
/// 模型经常返回带围栏的 SQL（` ```sql ` 开头，` ``` ` 结尾）。
/// 直接当 SQL 用会报语法错。
///
/// ## ⚠️ 必须跳过**解释性文字**
///
/// 模型也可能先说一句「这条查询没问题。」，然后才给出 SQL。
///
/// 「第一行」不一定是 SQL —— 所以要**找第一行看起来像 SQL 的**。
/// 判据：去掉空白后以 SQL 关键字开头。
pub fn extract_sql(raw: &str) -> String {
    let mut text = raw.trim();

    // 剥掉 ```sql / ``` 围栏。
    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        let after = after.strip_prefix("sql").unwrap_or(after);
        let after = after.strip_prefix("SQL").unwrap_or(after);
        match after.find("```") {
            Some(end) => text = after[..end].trim(),
            None => text = after.trim(),
        }
    }

    // 找第一行「像 SQL 的」。
    //
    // 判据：首个词是 SQL 关键字（select / with / values / explain）。
    let first_line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("--") && looks_like_sql(line))
        .unwrap_or("");

    // 去掉行尾分号。
    first_line.trim_end_matches(';').trim().to_string()
}

/// 判断一行文字「像不像 SQL」。
fn looks_like_sql(line: &str) -> bool {
    let lowered = line.to_ascii_lowercase();
    crate::guard::ALLOWED_STATEMENT_KEYWORDS
        .iter()
        .any(|kw| lowered.starts_with(kw))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 规则 1：NOT IN + NULL ──────────────────────────────────
    #[test]
    fn detects_not_in_with_null() {
        let checker = StaticChecker::new();
        let outcome = checker.check_sync(
            "SELECT * FROM Track WHERE GenreId NOT IN (SELECT GenreId FROM X WHERE Y IS NULL)",
        );
        assert!(!outcome.is_clean());
        assert!(outcome.issues.iter().any(|i| i.contains("NOT IN")));
        // 修正建议里要给出方向。
        assert!(outcome.sql.contains("NOT EXISTS"));
    }

    #[test]
    fn accepts_not_in_without_null() {
        let checker = StaticChecker::new();
        let outcome = checker.check_sync("SELECT * FROM Track WHERE GenreId NOT IN (1, 2, 3)");
        assert!(!outcome.issues.iter().any(|i| i.contains("NOT IN")));
    }

    #[test]
    fn accepts_plain_in() {
        let checker = StaticChecker::new();
        let outcome = checker.check_sync("SELECT * FROM Track WHERE GenreId IN (1, NULL)");
        // ⚠️ `IN` 遇到 NULL 是**正确**的（只是匹配不到 NULL），不报。
        assert!(!outcome.issues.iter().any(|i| i.contains("NOT IN")));
    }

    // ── 规则 2：UNION ─────────────────────────────────────────
    #[test]
    fn detects_union_without_all() {
        let checker = StaticChecker::new();
        let outcome = checker.check_sync("SELECT a FROM t1 UNION SELECT a FROM t2");
        assert!(outcome.issues.iter().any(|i| i.contains("UNION")));
    }

    #[test]
    fn accepts_union_all() {
        let checker = StaticChecker::new();
        let outcome = checker.check_sync("SELECT a FROM t1 UNION ALL SELECT a FROM t2");
        assert!(!outcome.issues.iter().any(|i| i.contains("UNION")));
    }

    #[test]
    fn does_not_rewrite_union_automatically() {
        // ⚠️ 保守策略：`UNION` → `UNION ALL` 会改变语义，只提示不改。
        let checker = StaticChecker::new();
        let outcome = checker.check_sync("SELECT a FROM t1 UNION SELECT a FROM t2");
        assert!(outcome.sql.contains("UNION SELECT"));
        assert!(!outcome.sql.contains("UNION ALL"));
    }

    // ── 规则 3：BETWEEN ───────────────────────────────────────
    #[test]
    fn detects_between() {
        let checker = StaticChecker::new();
        let outcome =
            checker.check_sync("SELECT * FROM Track WHERE Milliseconds BETWEEN 100 AND 200");
        assert!(outcome.issues.iter().any(|i| i.contains("BETWEEN")));
    }

    // ── 规则 4：引号 ──────────────────────────────────────────
    #[test]
    fn detects_unbalanced_quotes() {
        let checker = StaticChecker::new();
        let outcome = checker.check_sync("SELECT * FROM Track WHERE Name = 'abc");
        assert!(outcome.issues.iter().any(|i| i.contains("单引号")));
    }

    #[test]
    fn accepts_balanced_quotes() {
        let checker = StaticChecker::new();
        let outcome = checker.check_sync("SELECT * FROM Track WHERE Name = 'abc'");
        assert!(!outcome.issues.iter().any(|i| i.contains("引号")));
    }

    // ── 规则 5/6：SELECT * / LIMIT ────────────────────────────
    #[test]
    fn detects_missing_limit() {
        // ⚠️ LIMIT 检查默认**关闭** —— 它是性能建议，不是正确性错误。
        let checker = StaticChecker::new().with_limit_check();
        let outcome = checker.check_sync("SELECT Name FROM Genre");
        assert!(outcome.issues.iter().any(|i| i.contains("LIMIT")));
    }

    #[test]
    fn limit_check_is_off_by_default() {
        // ⚠️ 回归测试：默认不报 LIMIT —— 否则「检查器发现问题」这句话
        //    会被性能建议淹没，真正的错误反而被忽略。
        let checker = StaticChecker::new();
        let outcome = checker.check_sync("SELECT Name FROM Genre");
        assert!(!outcome.issues.iter().any(|i| i.contains("LIMIT")));
    }

    #[test]
    fn accepts_query_with_limit() {
        let checker = StaticChecker::new().with_limit_check();
        let outcome = checker.check_sync("SELECT Name FROM Genre LIMIT 5");
        assert!(!outcome.issues.iter().any(|i| i.contains("LIMIT")));
    }

    #[test]
    fn limit_check_can_be_disabled() {
        let checker = StaticChecker::new()
            .with_limit_check()
            .without_limit_check();
        let outcome = checker.check_sync("SELECT Name FROM Genre");
        assert!(!outcome.issues.iter().any(|i| i.contains("LIMIT")));
    }

    #[test]
    fn star_check_is_opt_in() {
        let checker = StaticChecker::new();
        assert!(
            !checker
                .check_sync("SELECT * FROM Genre LIMIT 5")
                .issues
                .iter()
                .any(|i| i.contains("SELECT *"))
        );

        let strict = StaticChecker::new().with_star_check();
        assert!(
            strict
                .check_sync("SELECT * FROM Genre LIMIT 5")
                .issues
                .iter()
                .any(|i| i.contains("SELECT *"))
        );
    }

    // ── 规则 7：JOIN 没有 ON ─────────────────────────────────
    #[test]
    fn detects_join_without_on() {
        let checker = StaticChecker::new();
        let outcome = checker.check_sync("SELECT * FROM Track JOIN Genre LIMIT 5");
        assert!(outcome.issues.iter().any(|i| i.contains("ON")));
    }

    #[test]
    fn accepts_join_with_on() {
        let checker = StaticChecker::new();
        let outcome = checker
            .check_sync("SELECT * FROM Track JOIN Genre ON Track.GenreId = Genre.GenreId LIMIT 5");
        assert!(!outcome.issues.iter().any(|i| i.contains("ON")));
    }

    #[test]
    fn accepts_cross_join() {
        let checker = StaticChecker::new();
        let outcome = checker.check_sync("SELECT * FROM Track CROSS JOIN Genre LIMIT 5");
        assert!(!outcome.issues.iter().any(|i| i.contains("ON")));
    }

    // ── SQL 抽取 ──────────────────────────────────────────────
    #[test]
    fn extracts_sql_from_markdown_fence() {
        let raw = "```sql\nSELECT Name FROM Genre\n```";
        assert_eq!(extract_sql(raw), "SELECT Name FROM Genre");
    }

    #[test]
    fn extracts_sql_from_plain_text() {
        assert_eq!(
            extract_sql("SELECT Name FROM Genre;"),
            "SELECT Name FROM Genre"
        );
    }

    #[test]
    fn extraction_skips_explanation_lines() {
        let raw = "这条查询没问题。\nSELECT Name FROM Genre";
        assert_eq!(extract_sql(raw), "SELECT Name FROM Genre");
    }

    #[test]
    fn check_prompt_mentions_all_seven_mistakes() {
        let prompt = build_check_prompt("SELECT 1");
        for keyword in [
            "NOT IN",
            "UNION",
            "BETWEEN",
            "type mismatch",
            "quoting",
            "arguments",
            "Casting",
            "joins",
        ] {
            assert!(prompt.contains(keyword), "提示词应提到 {keyword}");
        }
    }

    // ── LLM 检查器（用 ScriptedLlm 确定性测试）───────────────
    #[tokio::test]
    async fn llm_checker_uses_model_output() {
        use crate::llm::ScriptedLlm;

        let llm = Arc::new(ScriptedLlm::new(vec![
            "```sql\nSELECT Name FROM Genre LIMIT 5\n```".to_string(),
        ]));
        let checker = LlmChecker::new(llm);

        let outcome = checker.check("SELECT Name FROM Genre").await;

        assert_eq!(outcome.sql, "SELECT Name FROM Genre LIMIT 5");
        assert!(outcome.issues.iter().any(|i| i.contains("改写")));
    }

    #[tokio::test]
    async fn llm_checker_degrades_when_model_fails() {
        use crate::llm::ScriptedLlm;

        // 脚本化 LLM 没有预设答案 → 调用失败。
        let llm = Arc::new(ScriptedLlm::new(vec![]));
        let checker = LlmChecker::new(llm);

        let outcome = checker.check("SELECT 1").await;

        // ⚠️ 检查器失败**不致命** —— 保留原 SQL。
        assert_eq!(outcome.sql, "SELECT 1");
        assert!(outcome.issues.iter().any(|i| i.contains("跳过检查")));
    }
}
