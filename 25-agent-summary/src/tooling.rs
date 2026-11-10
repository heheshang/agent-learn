//! **工具层** —— Schema 校验、工具执行、安全护栏、错误分类。
//!
//! # 这个模块汇总了哪些历史能力
//!
//! | 能力 | 来自 | 在这里的形态 |
//! |---|---|---|
//! | JSON Schema 子集校验 | 14-agent-tool | [`validate`] |
//! | SQL 安全护栏（词法分析） | 15-agent-sql | [`guard`] |
//! | 错误三分类 + 稳定错误码 | 18-agent-error | [`ErrorKind`] / [`ErrorCode`] |
//!
//! # 三条硬约定
//!
//! ## ① 工具失败**不是**请求失败
//!
//! 工具报错时把错误信息当成一条 `Observation` 喂回模型，
//! 模型读到「参数 `location` 必须是字符串」就能自己纠正。
//!
//! ## ② 参数错了**别发请求**
//!
//! Schema 本地校验，错误信息必须**可执行**（指出哪个参数、期望什么、实际是什么）。
//!
//! ## ③ 只有 Transient 该重试
//!
//! 参数填错时重试是纯浪费 —— 参数不会因为重试变对。

use serde_json::{Value, json};

/// 一个校验问题。
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    /// 出问题的路径（`<root>` 表示根）。
    pub path: String,

    /// 问题描述。
    pub message: String,
}

/// 校验结果。
#[derive(Debug, Clone, PartialEq)]
pub struct Validation {
    /// 所有问题（**一次性报全**，让模型一次改对）。
    pub issues: Vec<Issue>,
}

impl Validation {
    /// 是否通过。
    pub fn is_ok(&self) -> bool {
        self.issues.is_empty()
    }

    /// 人类可读的消息。
    pub fn message(&self) -> String {
        if self.issues.is_empty() {
            return "参数校验通过".to_string();
        }
        self.issues
            .iter()
            .map(|i| format!("{}：{}", i.path, i.message))
            .collect::<Vec<_>>()
            .join("；")
    }
}

/// 用 Schema 校验参数。
///
/// ## 支持的 Schema 子集
///
/// - `type`：`object` / `string` / `number` / `integer` / `boolean` / `array`
/// - `properties` + `required`
/// - `enum`
/// - `minimum` / `maximum`
/// - `minLength` / `maxLength`
/// - `default`（**缺失时填默认值**）
/// - `additionalProperties: false`（拒绝未声明的字段）
pub fn validate(schema: &Value, args: &Value) -> Validation {
    let mut issues = Vec::new();

    let Some(object) = args.as_object() else {
        issues.push(Issue {
            path: "<root>".into(),
            message: format!("必须是 JSON 对象，实际是 {}", type_name(args)),
        });
        return Validation { issues };
    };

    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        // 必填检查。
        if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
            for key in required {
                let Some(name) = key.as_str() else { continue };
                if !object.contains_key(name) {
                    issues.push(Issue {
                        path: name.to_string(),
                        message: "缺少必填参数".to_string(),
                    });
                }
            }
        }

        // 逐个字段校验。
        for (name, value) in object {
            match props.get(name) {
                Some(field_schema) => {
                    check_value(name, field_schema, value, &mut issues);
                }
                None => {
                    if schema.get("additionalProperties").and_then(|v| v.as_bool()) == Some(false) {
                        issues.push(Issue {
                            path: name.clone(),
                            message: "未声明的参数".to_string(),
                        });
                    }
                }
            }
        }
    }

    Validation { issues }
}

/// 校验单个字段。
fn check_value(path: &str, schema: &Value, value: &Value, issues: &mut Vec<Issue>) {
    // 枚举。
    if let Some(allowed) = schema.get("enum").and_then(|e| e.as_array())
        && !allowed.contains(value)
    {
        let list = allowed
            .iter()
            .map(|v| v.as_str().unwrap_or("?"))
            .collect::<Vec<_>>()
            .join(" / ");
        issues.push(Issue {
            path: path.to_string(),
            message: format!("必须是 [{list}] 之一，实际是 {}", render(value)),
        });
        return;
    }

    let Some(expected) = schema.get("type").and_then(|t| t.as_str()) else {
        return;
    };

    let ok = match expected {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    };

    if !ok {
        issues.push(Issue {
            path: path.to_string(),
            message: format!("必须是 {expected}，实际是 {}", type_name(value)),
        });
        return;
    }

    if let Some(s) = value.as_str() {
        if let Some(min) = schema.get("minLength").and_then(|v| v.as_u64())
            && (s.chars().count() as u64) < min
        {
            issues.push(Issue {
                path: path.to_string(),
                message: format!("长度至少 {min}，实际 {}", s.chars().count()),
            });
        }
        if let Some(max) = schema.get("maxLength").and_then(|v| v.as_u64())
            && (s.chars().count() as u64) > max
        {
            issues.push(Issue {
                path: path.to_string(),
                message: format!("长度最多 {max}，实际 {}", s.chars().count()),
            });
        }
    }

    if let Some(n) = value.as_f64() {
        if let Some(min) = schema.get("minimum").and_then(|v| v.as_f64())
            && n < min
        {
            issues.push(Issue {
                path: path.to_string(),
                message: format!("不能小于 {min}，实际 {n}"),
            });
        }
        if let Some(max) = schema.get("maximum").and_then(|v| v.as_f64())
            && n > max
        {
            issues.push(Issue {
                path: path.to_string(),
                message: format!("不能大于 {max}，实际 {n}"),
            });
        }
    }
}

/// 填默认值 —— Schema 里声明了 `default` 但参数没给时补上。
pub fn fill_defaults(schema: &Value, args: &mut serde_json::Map<String, Value>) {
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return;
    };
    for (name, field_schema) in props {
        if args.contains_key(name) {
            continue;
        }
        if let Some(default) = field_schema.get("default") {
            args.insert(name.clone(), default.clone());
        }
    }
}

/// JSON 值的类型名。
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// 值的简短渲染（用于错误消息）。
fn render(value: &Value) -> String {
    match value {
        Value::String(s) => format!("\"{s}\""),
        other => other.to_string(),
    }
}

/// 工具的 Schema —— 一个示例工具（按模块统计）。
pub fn module_stats_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "description": "模块名，例如 \"cache\"、\"rag\"。",
                "minLength": 1
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
                "description": "返回条数上限。"
            },
            "include_risks": {
                "type": "boolean",
                "default": true,
                "description": "是否包含风险项。"
            }
        },
        "required": ["scope"],
        "additionalProperties": false
    })
}

/// ─────────────────────────────────────────────────────────────────────
/// 错误处理
/// ─────────────────────────────────────────────────────────────────────
/// 错误的处置类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// **可重试** —— 网络抖动、上游超时。
    Transient,
    /// **业务失败** —— 参数非法、资源不存在。不可重试，但可降级。
    Business,
    /// **致命** —— 鉴权失败、内部缺陷。不可重试、不可降级。
    Fatal,
}

impl ErrorKind {
    /// 名称。
    pub fn name(self) -> &'static str {
        match self {
            ErrorKind::Transient => "Transient",
            ErrorKind::Business => "Business",
            ErrorKind::Fatal => "Fatal",
        }
    }

    /// 该不该重试。
    pub fn retryable(self) -> bool {
        matches!(self, ErrorKind::Transient)
    }

    /// 该不该降级成观察。
    pub fn degradable(self) -> bool {
        matches!(self, ErrorKind::Transient | ErrorKind::Business)
    }
}

/// 稳定的错误码。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// 参数非法。
    InvalidArguments,
    /// 未知工具。
    UnknownTool,
    /// 超时。
    Timeout,
    /// 上游错误。
    Upstream,
    /// 限流。
    RateLimited,
    /// 未找到。
    NotFound,
    /// 规则违反。
    RuleViolation,
    /// 未授权。
    Unauthorized,
    /// 内部错误。
    Internal,
}

impl ErrorCode {
    /// 码的字符串（**稳定契约** —— 文案会变，码不会）。
    pub fn id(self) -> &'static str {
        match self {
            ErrorCode::InvalidArguments => "E_INVALID_ARGUMENTS",
            ErrorCode::UnknownTool => "E_UNKNOWN_TOOL",
            ErrorCode::Timeout => "E_UPSTREAM_TIMEOUT",
            ErrorCode::Upstream => "E_UPSTREAM_ERROR",
            ErrorCode::RateLimited => "E_RATE_LIMITED",
            ErrorCode::NotFound => "E_NOT_FOUND",
            ErrorCode::RuleViolation => "E_RULE_VIOLATION",
            ErrorCode::Unauthorized => "E_UNAUTHORIZED",
            ErrorCode::Internal => "E_INTERNAL",
        }
    }

    /// 默认处置类别。
    pub fn default_kind(self) -> ErrorKind {
        match self {
            ErrorCode::InvalidArguments
            | ErrorCode::UnknownTool
            | ErrorCode::NotFound
            | ErrorCode::RuleViolation => ErrorKind::Business,
            ErrorCode::Timeout | ErrorCode::Upstream | ErrorCode::RateLimited => {
                ErrorKind::Transient
            }
            ErrorCode::Unauthorized | ErrorCode::Internal => ErrorKind::Fatal,
        }
    }

    /// 一句话摘要。
    pub fn summary(self) -> &'static str {
        match self {
            ErrorCode::InvalidArguments => "参数不合法",
            ErrorCode::UnknownTool => "未知工具",
            ErrorCode::Timeout => "上游超时",
            ErrorCode::Upstream => "上游错误",
            ErrorCode::RateLimited => "被限流",
            ErrorCode::NotFound => "资源不存在",
            ErrorCode::RuleViolation => "违反规则",
            ErrorCode::Unauthorized => "未授权",
            ErrorCode::Internal => "内部错误",
        }
    }
}

/// 一次工具失败。
#[derive(Debug, Clone, PartialEq)]
pub struct ToolError {
    /// 错误码。
    pub code: ErrorCode,

    /// 处置类别。
    pub kind: ErrorKind,

    /// 出错的工具名。
    pub tool: String,

    /// 原因。
    pub message: String,

    /// 给模型的**可执行建议**。
    ///
    /// ⚠️ 没有 hint 的 Business 错误视为缺陷 ——
    ///    Business 错误的价值就在于「模型能据此纠正」。
    pub hint: Option<String>,
}

impl ToolError {
    /// 构造错误（类别取错误码的默认值）。
    pub fn new(code: ErrorCode, tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            kind: code.default_kind(),
            tool: tool.into(),
            message: message.into(),
            hint: None,
        }
    }

    /// 覆盖处置类别。
    pub fn with_kind(mut self, kind: ErrorKind) -> Self {
        self.kind = kind;
        self
    }

    /// 附加可执行建议。
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// 该不该重试。
    pub fn retryable(&self) -> bool {
        self.kind.retryable()
    }

    /// 该不该降级。
    pub fn degradable(&self) -> bool {
        self.kind.degradable()
    }

    /// ★ 渲染成给模型看的三段式文本。
    ///
    /// ```text
    /// [E_INVALID_ARGUMENTS] 归类：Business。原因：scope 不能为空。建议：传入模块名。
    /// ```
    ///
    /// 错误码是**稳定契约**，归类告诉模型「该不该再试」，
    /// 建议告诉模型「**怎么改**」。
    pub fn render(&self) -> String {
        let hint = self.hint.as_deref().unwrap_or("请根据原因调整后重试。");
        format!(
            "[{}] 归类：{}。原因：{}。建议：{}",
            self.code.id(),
            self.kind.name(),
            self.message,
            hint
        )
    }
}

/// 指数退避 —— 第 n 次失败后等多久。
///
/// `delay = min(base × 2^(n-1), max)`
pub fn backoff(
    base: std::time::Duration,
    max: std::time::Duration,
    attempt: usize,
) -> std::time::Duration {
    if base.is_zero() {
        return std::time::Duration::ZERO;
    }
    // ⚠️ shift 要 clamp —— `1u32 << 40` 会溢出。
    let shift = attempt.saturating_sub(1).min(16);
    let factor = 1u32 << shift;
    base.saturating_mul(factor).min(max)
}

/// ─────────────────────────────────────────────────────────────────────
/// SQL 安全护栏
/// ─────────────────────────────────────────────────────────────────────
/// 一个 SQL 词法单元。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// 标识符 / 关键字。
    Word(String),
    /// 字符串字面量。
    Str(String),
    /// 数字。
    Number(String),
    /// 标点 / 运算符。
    Punct(char),
}

/// 护栏的判定结果。
#[derive(Debug, Clone, PartialEq)]
pub enum GuardError {
    /// 空语句。
    Empty,
    /// 超过长度上限。
    TooLong {
        /// 实际长度。
        len: usize,
        /// 上限。
        limit: usize,
    },
    /// 不是只读语句。
    NotReadOnly,
    /// 多语句。
    MultipleStatements,
    /// 命中危险关键字。
    ForbiddenKeyword(String),
}

/// SQL 长度上限。
pub const MAX_SQL_LEN: usize = 8000;

/// SQL 词法分析。
///
/// ## ⚠️ 为什么不能用 `contains`
///
/// `WHERE name = 'DROP TABLE'` 里的 DROP 是**数据**不是操作，
/// `SELECT 1 /* DROP */` 里的 DROP 是**注释**。
/// 必须做真正的词法分析，且黑名单只在**裸词**上匹配。
///
/// ## 误伤优先于漏放
///
/// 误伤代价是模型重写一次 SQL，漏放代价是数据库被删。
pub fn tokenize_sql(sql: &str) -> Vec<Token> {
    let chars: Vec<char> = sql.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        // 空白。
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // 行注释 `--`。
        if c == '-' && i + 1 < chars.len() && chars[i + 1] == '-' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        // 块注释 `/* */`。
        if c == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(chars.len());
            continue;
        }

        // 单引号字符串（`''` 是转义）。
        if c == '\'' {
            i += 1;
            let mut buf = String::new();
            while i < chars.len() {
                if chars[i] == '\'' {
                    if i + 1 < chars.len() && chars[i + 1] == '\'' {
                        buf.push('\'');
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                buf.push(chars[i]);
                i += 1;
            }
            out.push(Token::Str(buf));
            continue;
        }

        // 双引号 / 反引号标识符。
        if c == '"' || c == '`' {
            let quote = c;
            i += 1;
            let mut buf = String::new();
            while i < chars.len() && chars[i] != quote {
                buf.push(chars[i]);
                i += 1;
            }
            i = (i + 1).min(chars.len());
            out.push(Token::Word(buf));
            continue;
        }

        // 数字。
        if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            out.push(Token::Number(chars[start..i].iter().collect()));
            continue;
        }

        // 标识符 / 关键字。
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Token::Word(chars[start..i].iter().collect()));
            continue;
        }

        // 标点。
        out.push(Token::Punct(c));
        i += 1;
    }

    out
}

/// 只读语句的白名单前缀。
const READ_ONLY_PREFIXES: &[&str] = &["SELECT", "WITH", "VALUES", "EXPLAIN"];

/// 危险关键字黑名单。
const FORBIDDEN_KEYWORDS: &[&str] = &[
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
    "GRANT",
    "REVOKE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "LOCK",
    "UNLOCK",
    "ANALYZE",
    "EXEC",
    "EXECUTE",
    "CALL",
    "MERGE",
    "UPSERT",
];

/// SQL 安全检查。
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

    // 单语句检查：第一个裸词之后不能再来一个语句起始词。
    let words: Vec<String> = tokens
        .iter()
        .filter_map(|t| match t {
            Token::Word(w) => Some(w.to_uppercase()),
            _ => None,
        })
        .collect();

    if words.is_empty() {
        return Err(GuardError::NotReadOnly);
    }

    // 只读白名单。
    if !READ_ONLY_PREFIXES.contains(&words[0].as_str()) {
        return Err(GuardError::NotReadOnly);
    }

    // 多语句检查。
    //
    // ## ⚠️ 判断「语句边界」必须懂 SQL 的 CTE 结构
    //
    // ```sql
    // WITH x AS (SELECT 1) SELECT * FROM x
    // ```
    //
    // 这里有**两个** `SELECT` 关键字，但它们属于**同一条语句**：
    // 第一个在括号内（子查询），第二个是 CTE 定义之后的「主查询」。
    //
    // 判据：
    // 1. 找**括号外**的第一个语句关键字 → 语句头
    // 2. 若语句头是 `WITH`，则**再允许**一个括号外的语句关键字（主查询）
    // 3. 此后再出现括号外的语句关键字 → 才是真的多语句
    //
    // ⚠️ 不能简单数 `SELECT` 的个数 —— 那会把合法的 CTE 误判成多语句。
    let stmt_starts = [
        "SELECT", "WITH", "VALUES", "EXPLAIN", "INSERT", "UPDATE", "DELETE",
    ];

    // 括号外的语句关键字序列。
    let mut outer_starts: Vec<&str> = Vec::new();
    let mut depth = 0i32;
    for token in &tokens {
        match token {
            Token::Punct('(') => depth += 1,
            Token::Punct(')') => depth -= 1,
            Token::Word(w) => {
                let upper = w.to_uppercase();
                if depth == 0 && stmt_starts.contains(&upper.as_str()) {
                    outer_starts.push(stmt_starts.iter().find(|s| **s == upper).unwrap());
                }
            }
            _ => {}
        }
    }

    // 语句头。
    let Some(head) = outer_starts.first() else {
        return Err(GuardError::NotReadOnly);
    };

    // 允许的语句关键字个数：`WITH` 后面还有一个主查询。
    let allowed = if *head == "WITH" { 2 } else { 1 };
    if outer_starts.len() > allowed {
        return Err(GuardError::MultipleStatements);
    }

    // 危险关键字（只在裸词上匹配）。
    for w in &words {
        if FORBIDDEN_KEYWORDS.contains(&w.as_str()) {
            return Err(GuardError::ForbiddenKeyword(w.clone()));
        }
    }

    Ok(())
}

/// 是否只读。
pub fn is_read_only(sql: &str) -> bool {
    check_sql(sql).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_args_pass() {
        let schema = module_stats_schema();
        let args = json!({"scope": "cache"});
        assert!(validate(&schema, &args).is_ok());
    }

    #[test]
    fn missing_required_is_reported() {
        let schema = module_stats_schema();
        let v = validate(&schema, &json!({}));
        assert!(!v.is_ok());
        assert!(v.message().contains("scope"));
    }

    #[test]
    fn wrong_type_is_reported() {
        let schema = module_stats_schema();
        let v = validate(&schema, &json!({"scope": 123}));
        assert!(!v.is_ok());
        assert!(v.message().contains("string"));
    }

    #[test]
    fn unknown_field_is_rejected_when_closed() {
        let schema = module_stats_schema();
        let v = validate(&schema, &json!({"scope": "a", "extra": 1}));
        assert!(!v.is_ok());
    }

    #[test]
    fn enum_violation_is_reported() {
        let schema = json!({"type": "object", "properties": {"m": {"enum": ["a", "b"]}}});
        let v = validate(&schema, &json!({"m": "c"}));
        assert!(!v.is_ok());
    }

    #[test]
    fn all_issues_are_reported_at_once() {
        let schema = module_stats_schema();
        let v = validate(&schema, &json!({"limit": "x", "extra": 1}));
        // 缺 scope + limit 类型错 + 多余字段 → 至少 2 个问题。
        assert!(v.issues.len() >= 2);
    }

    #[test]
    fn defaults_are_filled() {
        let schema = module_stats_schema();
        let mut args = json!({"scope": "cache"}).as_object().unwrap().clone();
        fill_defaults(&schema, &mut args);
        assert_eq!(args.get("limit"), Some(&json!(10)));
        assert_eq!(args.get("include_risks"), Some(&json!(true)));
    }

    #[test]
    fn error_kinds_follow_the_matrix() {
        assert!(ErrorCode::Timeout.default_kind().retryable());
        assert!(!ErrorCode::InvalidArguments.default_kind().retryable());
        assert!(!ErrorCode::Internal.default_kind().degradable());
        assert!(ErrorCode::InvalidArguments.default_kind().degradable());
    }

    #[test]
    fn error_text_has_three_segments() {
        let e = ToolError::new(ErrorCode::InvalidArguments, "search", "scope 不能为空")
            .with_hint("传入模块名");
        let text = e.render();
        assert!(text.contains("E_INVALID_ARGUMENTS"));
        assert!(text.contains("Business"));
        assert!(text.contains("传入模块名"));
    }

    #[test]
    fn backoff_grows_exponentially_and_caps() {
        let base = std::time::Duration::from_millis(100);
        let max = std::time::Duration::from_secs(1);
        assert_eq!(backoff(base, max, 1).as_millis(), 100);
        assert_eq!(backoff(base, max, 2).as_millis(), 200);
        assert_eq!(backoff(base, max, 3).as_millis(), 400);
        assert_eq!(backoff(base, max, 10).as_millis(), 1000);
    }

    #[test]
    fn guard_allows_read_only() {
        assert!(check_sql("SELECT * FROM t").is_ok());
        assert!(check_sql("WITH x AS (SELECT 1) SELECT * FROM x").is_ok());
    }

    #[test]
    fn guard_blocks_writes() {
        assert!(matches!(
            check_sql("DELETE FROM t"),
            Err(GuardError::NotReadOnly)
        ));
        assert!(matches!(
            check_sql("DROP TABLE t"),
            Err(GuardError::NotReadOnly)
        ));
    }

    #[test]
    fn guard_ignores_keywords_inside_strings() {
        // DROP 在这里只是数据。
        assert!(check_sql("SELECT * FROM t WHERE name = 'DROP TABLE'").is_ok());
    }

    #[test]
    fn guard_ignores_keywords_inside_comments() {
        assert!(check_sql("SELECT 1 /* DROP */").is_ok());
        assert!(check_sql("SELECT 1 -- DROP").is_ok());
    }

    #[test]
    fn guard_blocks_multiple_statements() {
        assert!(matches!(
            check_sql("SELECT 1; SELECT 2"),
            Err(GuardError::MultipleStatements)
        ));
    }

    #[test]
    fn guard_blocks_empty_and_overlong() {
        assert_eq!(check_sql("   "), Err(GuardError::Empty));
        let long = format!("SELECT {}", "a".repeat(MAX_SQL_LEN));
        assert!(matches!(check_sql(&long), Err(GuardError::TooLong { .. })));
    }

    #[test]
    fn guard_allows_cte_with_subquery() {
        // ⚠️ 子查询里的 SELECT 不是「第二条语句」。
        assert!(check_sql("WITH x AS (SELECT 1) SELECT * FROM x").is_ok());
        assert!(check_sql("SELECT * FROM (SELECT 1) t").is_ok());
    }

    #[test]
    fn guard_still_blocks_semicolon_statements() {
        assert!(matches!(
            check_sql("SELECT 1; DELETE FROM t"),
            Err(GuardError::MultipleStatements)
        ));
    }

    #[test]
    fn guard_allows_leading_comment() {
        // 第一个**词法单元**是 select，而不是注释。
        assert!(check_sql("/* hi */ SELECT 1").is_ok());
    }
}
