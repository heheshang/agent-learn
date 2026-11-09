//! SQL 工具集 —— 对应 LangChain 的四个 `sql_db_*` 工具。
//!
//! | LangChain | 本项目 |
//! |---|---|
//! | `sql_db_list_tables` | [`ListTablesTool`] |
//! | `sql_db_schema` | [`SchemaTool`] |
//! | `sql_db_query` | [`QueryTool`] |
//! | `sql_db_query_checker` | [`QueryCheckerTool`] |
//!
//! ## 与 LangChain 的**关键差异**
//!
//! 教程里的 `sql_db_query` 是这样的：
//!
//! ```python
//! @tool
//! def sql_db_query(query: str) -> str:
//!     """Input to this tool is a detailed and correct SQL query..."""
//!     con = sqlite3.connect("Chinook.db")
//!     try:
//!         cursor.execute(query)          # ← 直接执行，模型说什么就执行什么
//!         res = cursor.fetchall()
//!         return str(res)
//!     except Exception as e:
//!         return f"Error: {e}"
//! ```
//!
//! 教程自己也标注了警告：
//!
//! > The following database tools are minimal wrappers for demonstration
//! > purposes only. They are not intended to be secure or used in production.
//!
//! ## 本项目加了什么
//!
//! 1. **护栏**（[`crate::guard`]）—— 模型生成的 SQL 必须先过静态检查
//! 2. **只读连接**（[`crate::db`]）—— 运行时兜底
//! 3. **行数上限** —— 防止一条 `SELECT *` 把内存打满
//! 4. **可执行错误文案** —— 错误信息必须让模型能自我纠正
//!
//! ## 为什么 `sql_db_query_checker` 值得单独实现成一个「工具」
//!
//! 教程里它是一个 `@tool`，由**模型自己决定**要不要调用。
//! 本项目保留这个设计（对齐教程），但把它做成**可插拔**的：
//! 需要 LLM 时用 [`QueryCheckerTool`]，不需要时（测试、单测）用
//! [`crate::checker::StaticChecker`] —— 一个不依赖模型的静态检查器。
//!
//! 理由：LangChain 的 `sql_db_query_checker` 内部**直接调模型**
//! （`response = model.invoke(trigger_prompt)`），
//! 所以它没法被确定性测试。本项目把它拆成 trait，
//! 于是「检查器」这件事可以被离线测。

use std::sync::Arc;

use agent_14::tool::{Tool, ToolContext, ToolFuture, ToolOutcome, ToolSpec};
use agent_logging::{debug, info, warn};
use serde_json::{Value, json};

use crate::checker::QueryChecker;
use crate::db::Database;
use crate::error::AppError;
use crate::guard;

/// 列出所有数据表。
///
/// 对应 LangChain 的 `sql_db_list_tables`。
///
/// ## 教程里的实现
///
/// ```python
/// cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
/// tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
/// return ", ".join(tables)
/// ```
///
/// 本项目返回的是**带行数的列表** —— 因为「有多少数据」对模型判断
/// 「这个表值得查吗」很有用（教程版本只给表名）。
pub struct ListTablesTool {
    db: Arc<Database>,
}

impl ListTablesTool {
    /// 创建工具。
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
}

impl Tool for ListTablesTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "sql_db_list_tables",
            "列出数据库中所有的数据表（不含 SQLite 内部表），并给出每张表的行数。\
             输入是空字符串。你在开始查询前应该先调用它，看看有哪些表可用。",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, _args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let tables = self.db.list_tables().map_err(to_tool_error)?;

            if tables.is_empty() {
                return Ok(ToolOutcome::ok(
                    "sql_db_list_tables",
                    "数据库里没有数据表。",
                ));
            }

            let mut lines = Vec::with_capacity(tables.len());
            for table in &tables {
                let count = self.db.count_rows(table).unwrap_or(-1);
                lines.push(format!("{table}（{count} 行）"));
            }

            Ok(ToolOutcome::ok(
                "sql_db_list_tables",
                format!("可用数据表（{} 张）:\n{}", tables.len(), lines.join("\n")),
            ))
        })
    }
}

/// 查看表的结构（DDL + 样例行）。
///
/// 对应 LangChain 的 `sql_db_schema`。
///
/// ## 与教程版本的差异
///
/// 教程版返回「DDL + 3 行样例」。本项目额外返回：
///
/// - **每列的类型和是否主键**（更结构化）
/// - **行数**（模型判断「这表大不大」）
///
/// 额外的信息让模型少写错一次 SQL。
pub struct SchemaTool {
    db: Arc<Database>,
    sample_rows: usize,
}

impl SchemaTool {
    /// 创建工具（默认取 3 行样例，和教程一致）。
    pub fn new(db: Arc<Database>) -> Self {
        Self { db, sample_rows: 3 }
    }

    /// 设置样例行数。
    pub fn with_sample_rows(mut self, rows: usize) -> Self {
        self.sample_rows = rows;
        self
    }
}

impl Tool for SchemaTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "sql_db_schema",
            "查看指定数据表的结构：建表语句、每列的类型与主键标记、行数，以及几行样例数据。\
             输入是逗号分隔的表名，例如 \"Track, Genre\"。\
             调用前请先用 sql_db_list_tables 确认表名存在。",
            json!({
                "type": "object",
                "properties": {
                    "table_names": {
                        "type": "string",
                        "description": "逗号分隔的表名列表，例如 \"Track, Genre\""
                    }
                },
                "required": ["table_names"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let raw = args
                .get("table_names")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();

            if raw.is_empty() {
                return Err(agent_14::error::ToolError::invalid_args(vec![
                    "参数 `table_names` 不能为空。请传入逗号分隔的表名，例如 \"Track, Genre\"。"
                        .to_string(),
                ]));
            }

            // 先取所有合法表名 —— 不在列表里的直接报错（对应教程的
            // `if table not in valid_tables` 分支）。
            let valid = self.db.list_tables().map_err(to_tool_error)?;

            let mut sections: Vec<String> = Vec::new();
            // ⚠️ 记录是否有表不存在 —— 有的话整个调用算**失败**。
            //
            // 为什么不把错误混在正常结果里（教程的做法）：
            // 混在一起的话，模型可能只看到「有内容」就以为成功了，
            // 然后去查一张不存在的表。
            //
            // 单独标成失败，模型能明确知道「表名错了」，然后用
            // `sql_db_list_tables` 拿正确表名重试。
            let mut has_missing = false;

            for name in raw.split(',') {
                let name = name.trim();
                if name.is_empty() {
                    continue;
                }

                if !valid.contains(&name.to_string()) {
                    // ⚠️ 错误信息必须**列出可用表名** —— 否则模型只能瞎猜。
                    has_missing = true;
                    sections.push(format!(
                        "错误：表 `{name}` 不存在。可用的表：{}",
                        valid.join(", ")
                    ));
                    continue;
                }

                let ddl = self.db.table_ddl(name).map_err(to_tool_error)?;
                let columns = self.db.table_columns(name).map_err(to_tool_error)?;
                let count = self.db.count_rows(name).unwrap_or(-1);

                let mut section = String::new();
                if let Some(ddl) = ddl {
                    section.push_str(&ddl);
                    section.push_str("\n\n");
                }

                section.push_str(&format!(
                    "列信息（共 {} 列，表内 {count} 行）:\n",
                    columns.len()
                ));
                for col in &columns {
                    let mut flags = Vec::new();
                    if col.primary_key {
                        flags.push("主键");
                    }
                    if col.not_null {
                        flags.push("NOT NULL");
                    }
                    let flag_str = if flags.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", flags.join(", "))
                    };
                    section.push_str(&format!(
                        "  - {}: {}{}\n",
                        col.name, col.data_type, flag_str
                    ));
                }

                match self.db.sample_rows(name, self.sample_rows) {
                    Ok(rows) if !rows.is_empty() => {
                        let names: Vec<&str> = rows[0]
                            .as_object()
                            .map(|o| o.keys().map(|k| k.as_str()).collect())
                            .unwrap_or_default();
                        section.push_str(&format!("\n样例数据（前 {} 行）:\n", rows.len()));
                        section.push_str(&format!("{}\n", names.join(" | ")));
                        for row in &rows {
                            let cells: Vec<String> = names
                                .iter()
                                .map(|n| match row.get(*n) {
                                    Some(Value::String(s)) => s.clone(),
                                    Some(other) => other.to_string(),
                                    None => "NULL".to_string(),
                                })
                                .collect();
                            section.push_str(&format!("{}\n", cells.join(" | ")));
                        }
                    }
                    Ok(_) => {
                        section.push_str("\n（该表当前没有数据）\n");
                    }
                    Err(e) => {
                        section.push_str(&format!("\n（读取样例数据失败: {e}）\n"));
                    }
                }

                sections.push(section);
            }

            let text = sections.join("\n\n");

            if has_missing {
                // ⚠️ 有表不存在 —— 整个调用算失败，让模型能明确知道要改表名。
                Err(agent_14::error::ToolError::invalid_args(vec![text]))
            } else {
                Ok(ToolOutcome::ok("sql_db_schema", text))
            }
        })
    }
}

/// 执行只读 SQL 查询。
///
/// 对应 LangChain 的 `sql_db_query`。
///
/// ## ⚠️ 这是**唯一**会碰数据库的工具 —— 护栏在这里施加
///
/// 流程：
///
/// ```text
/// 模型给的 SQL
///    ↓
/// 护栏静态检查（只读？单语句？有危险关键字？）
///    ↓ 不通过 → 错误文案喂回模型
/// 护栏通过
///    ↓
/// 只读连接执行（运行时兜底）
///    ↓
/// 行数截断 → 结果
/// ```
///
/// ## 护栏拦截时**不执行**，但**仍然是「工具失败」不是「请求失败」**
///
/// 因为模型读到「这条 SQL 被护栏拦截，请改成只读查询」后，
/// 能自己写一条合法的。这正是 LangChain 里
/// `handle_tool_errors` 的精神。
pub struct QueryTool {
    db: Arc<Database>,
    max_rows: usize,
    /// 是否允许写操作（默认**不允许**）。
    ///
    /// ⚠️ 这个开关存在的唯一理由是**教学演示**：
    /// 让人亲眼看到「护栏到底拦了什么」。
    ///
    /// 生产环境**绝对不能**打开 ——
    /// 所以它默认 `false`，且打开时必须显式调用
    /// [`QueryTool::allow_writes`]。
    allow_writes: bool,
}

impl QueryTool {
    /// 创建工具（默认只读 + 最多 100 行）。
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            db,
            max_rows: 100,
            allow_writes: false,
        }
    }

    /// 设置最大返回行数。
    pub fn with_max_rows(mut self, max_rows: usize) -> Self {
        self.max_rows = max_rows.clamp(1, 1000);
        self
    }

    /// ⚠️ **打开写操作许可**（仅用于教学演示）。
    ///
    /// 打开后护栏不再拦截写操作 —— 但**只读连接**仍然会拒绝它们
    /// （第 2 层防线依然在）。
    ///
    /// 所以这个开关的实际效果是：
    /// 「模型能看到『这条 SQL 被护栏拦了』的反馈」，
    /// 而数据库本身**依然安全**。
    pub fn allow_writes(mut self) -> Self {
        self.allow_writes = true;
        self
    }
}

impl Tool for QueryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "sql_db_query",
            "执行一条只读 SQL 查询并返回结果。\
             输入必须是完整的 SQL 查询语句（SQLite 方言）。\
             只允许 SELECT / WITH / VALUES / EXPLAIN 开头的只读查询，\
             禁止 INSERT / UPDATE / DELETE / DROP 等写操作，也禁止多条语句。\
             如果返回的是错误信息，请读懂错误原因后修正 SQL 再试一次。\
             遇到 Unknown column 之类的错误，请先用 sql_db_schema 查看正确的字段名。",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要执行的只读 SQL 查询语句，例如 \"SELECT Name FROM Genre\""
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let sql = args
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();

            if sql.is_empty() {
                return Err(agent_14::error::ToolError::invalid_args(vec![
                    "参数 `query` 不能为空。请传入一条只读 SQL 查询。".to_string(),
                ]));
            }

            // ── 护栏 ────────────────────────────────────────────
            //
            // ⚠️ `allow_writes` 只影响**护栏**（第 1 层），
            //    第 2 层的只读连接依然会拒绝写操作。
            if !self.allow_writes
                && let Err(guard_err) = guard::check_sql(&sql)
            {
                warn!(error = %guard_err, "SQL 被安全护栏拦截");
                // ⚠️ 返回 `Err`（不是 `Ok(failed)`）——
                //    注册表会把它转成一条 `Observation` 喂回模型，
                //    模型读到「被护栏拦截，请改成只读查询」就能自己纠正。
                //
                //    用 `InvalidArgs` 而不是 `Execution` ——
                //    因为这在语义上是「参数（SQL）不合法」，
                //    用户直接调工具时应该拿到 400 而不是 500。
                return Err(agent_14::error::ToolError::invalid_args(vec![
                    guard_err.message(),
                ]));
            }

            // ── 执行 ────────────────────────────────────────────
            debug!(sql = %sql, "开始执行只读查询");

            match self.db.query(&sql, self.max_rows) {
                Ok(result) => {
                    info!(
                        row_count = result.rows.len(),
                        truncated = result.truncated,
                        elapsed_ms = result.elapsed_ms,
                        "查询执行成功"
                    );
                    Ok(ToolOutcome::ok_with_data(
                        "sql_db_query",
                        result.render(),
                        serde_json::to_value(&result).unwrap_or(Value::Null),
                    ))
                }
                Err(e) => {
                    // ⚠️ 执行错误（语法错、列不存在、只读限制…）——
                    //    全部转成「可自我纠正」的观察。
                    warn!(error = %e, "SQL 执行失败");
                    Err(agent_14::error::ToolError::execution(
                        "sql_db_query",
                        format!("错误：{e}\n请根据这个错误修正 SQL 后重试。"),
                    ))
                }
            }
        })
    }
}

/// 检查 SQL 是否正确（对应 LangChain 的 `sql_db_query_checker`）。
///
/// ## 与教程版本的差异
///
/// 教程里这个工具**直接调模型**：
///
/// ```python
/// response = model.invoke(trigger_prompt)
/// return response.text.strip()
/// ```
///
/// 这带来两个问题：
///
/// 1. **没法确定性测试** —— 每次结果都不同
/// 2. **多一次模型调用** —— 慢且贵
///
/// 本项目把它抽象成 [`QueryChecker`] trait：
///
/// - [`crate::checker::StaticChecker`] —— 纯本地的启发式检查（默认，零成本）
/// - [`crate::checker::LlmChecker`] —— 调模型做语义检查（对齐教程）
///
/// 这样「检查器」这件事可以离线测，也可以在两种模式间切换。
pub struct QueryCheckerTool {
    checker: Arc<dyn QueryChecker>,
}

impl QueryCheckerTool {
    /// 创建工具。
    pub fn new(checker: Arc<dyn QueryChecker>) -> Self {
        Self { checker }
    }
}

impl Tool for QueryCheckerTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "sql_db_query_checker",
            "在执行一条 SQL 之前，用它检查查询有没有常见错误\
             （NOT IN 遇到 NULL、UNION 与 UNION ALL 混用、BETWEEN 用于开区间、\
             谓词类型不匹配、标识符引号、函数参数个数、类型转换、JOIN 用了错误的列）。\
             如果有错误就改写，没有错误就原样输出。\
             输出只能是 SQL 语句本身，不要有任何解释。",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检查的 SQL 查询语句"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let sql = args
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();

            if sql.is_empty() {
                return Err(agent_14::error::ToolError::invalid_args(vec![
                    "参数 `query` 不能为空。".to_string(),
                ]));
            }

            let checked = self.checker.check(&sql).await;

            if checked.issues.is_empty() {
                Ok(ToolOutcome::ok("sql_db_query_checker", checked.sql))
            } else {
                // ⚠️ 有问题时，把「修正后的 SQL + 问题列表」都返回 ——
                //    模型能同时看到「改成了什么」和「为什么改」。
                Ok(ToolOutcome::ok(
                    "sql_db_query_checker",
                    format!(
                        "检查发现 {} 个问题（已自动修正）：\n{}\n修正后的 SQL：\n{}",
                        checked.issues.len(),
                        checked.issues.join("\n"),
                        checked.sql
                    ),
                ))
            }
        })
    }
}

/// 把 `AppError` 转成工具错误。
fn to_tool_error(e: AppError) -> agent_14::error::ToolError {
    match e {
        AppError::Tool(msg) => agent_14::error::ToolError::execution("sql_db", msg),
        other => agent_14::error::ToolError::execution("sql_db", other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::StaticChecker;

    fn setup() -> (tempfile::TempDir, Arc<Database>) {
        let dir = tempfile::tempdir().unwrap();
        let seed = dir.path().join("seed.sql");
        std::fs::write(
            &seed,
            r#"
            CREATE TABLE Genre (GenreId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
            CREATE TABLE Track (
                TrackId INTEGER PRIMARY KEY,
                Name TEXT NOT NULL,
                GenreId INTEGER NOT NULL REFERENCES Genre(GenreId),
                Milliseconds INTEGER NOT NULL
            );
            INSERT INTO Genre (GenreId, Name) VALUES (1, '摇滚'), (2, '金属');
            INSERT INTO Track (TrackId, Name, GenreId, Milliseconds) VALUES
                (1, '夜色', 1, 300000),
                (2, '回声', 1, 200000),
                (3, '侧影', 2, 400000);
            "#,
        )
        .unwrap();
        let db_path = dir.path().join("test.db");
        crate::db::load_seed(&seed, &db_path).unwrap();
        (dir, Arc::new(Database::open_readonly(&db_path).unwrap()))
    }

    #[tokio::test]
    async fn list_tables_returns_names_and_counts() {
        let (_dir, db) = setup();
        let tool = ListTablesTool::new(db);
        let outcome = tool.call(&json!({}), &ToolContext::empty()).await.unwrap();

        assert!(outcome.ok);
        assert!(outcome.observation.contains("Genre"));
        assert!(outcome.observation.contains("Track"));
        assert!(outcome.observation.contains("行"));
    }

    #[tokio::test]
    async fn schema_returns_ddl_columns_and_samples() {
        let (_dir, db) = setup();
        let tool = SchemaTool::new(db);
        let outcome = tool
            .call(
                &json!({ "table_names": "Track, Genre" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap();

        assert!(outcome.ok);
        assert!(outcome.observation.contains("CREATE TABLE"));
        assert!(outcome.observation.contains("Milliseconds"));
        assert!(outcome.observation.contains("夜色"));
        assert!(outcome.observation.contains("主键"));
    }

    #[tokio::test]
    async fn schema_reports_unknown_table_with_available_list() {
        let (_dir, db) = setup();
        let tool = SchemaTool::new(db);
        let err = tool
            .call(&json!({ "table_names": "Nope" }), &ToolContext::empty())
            .await
            .unwrap_err();

        // ⚠️ 必须是**可执行**的错误：告诉模型有哪些表可用。
        assert!(err.to_string().contains("不存在"));
        assert!(err.to_string().contains("Track"));
    }

    #[tokio::test]
    async fn schema_rejects_empty_table_names() {
        let (_dir, db) = setup();
        let tool = SchemaTool::new(db);
        let err = tool
            .call(&json!({}), &ToolContext::empty())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("table_names"));
    }

    #[tokio::test]
    async fn query_executes_a_valid_select() {
        let (_dir, db) = setup();
        let tool = QueryTool::new(db);
        let outcome = tool
            .call(
                &json!({ "query": "SELECT Name FROM Genre" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap();

        assert!(outcome.ok, "{}", outcome.observation);
        assert!(outcome.observation.contains("摇滚"));
        assert!(outcome.data.is_some());
    }

    #[tokio::test]
    async fn query_blocks_delete() {
        let (_dir, db) = setup();
        let tool = QueryTool::new(db);
        let err = tool
            .call(
                &json!({ "query": "DELETE FROM Track" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap_err();

        // ⚠️ 关键：护栏拦截**不执行**，但也不是请求失败 ——
        //    错误信息必须让模型能改成只读查询。
        assert!(err.to_string().contains("只读"));
        assert!(err.to_string().contains("SELECT"));
    }

    #[tokio::test]
    async fn query_blocks_update_and_drop() {
        let (_dir, db) = setup();
        let tool = QueryTool::new(db);

        for sql in [
            "UPDATE Track SET Name = 'x'",
            "DROP TABLE Track",
            "INSERT INTO Genre VALUES (9, 'x')",
        ] {
            let err = tool
                .call(&json!({ "query": sql }), &ToolContext::empty())
                .await
                .unwrap_err();
            assert!(err.to_string().contains("只读"), "应该被拦截: {sql}");
        }
    }

    #[tokio::test]
    async fn query_blocks_multiple_statements() {
        let (_dir, db) = setup();
        let tool = QueryTool::new(db);
        let err = tool
            .call(
                &json!({ "query": "SELECT 1; DROP TABLE Track" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap_err();

        assert!(err.to_string().contains("单条"));
    }

    #[tokio::test]
    async fn query_reports_syntax_error_as_correctable() {
        let (_dir, db) = setup();
        let tool = QueryTool::new(db);
        let err = tool
            .call(&json!({ "query": "SELECT FROM" }), &ToolContext::empty())
            .await
            .unwrap_err();

        assert!(err.to_string().contains("修正"));
    }

    #[tokio::test]
    async fn query_truncates_rows() {
        let (_dir, db) = setup();
        let tool = QueryTool::new(db).with_max_rows(2);
        let outcome = tool
            .call(
                &json!({ "query": "SELECT * FROM Track" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap();

        assert!(outcome.ok);
        assert!(outcome.observation.contains("截断"));
    }

    #[tokio::test]
    async fn query_rejects_empty_sql() {
        let (_dir, db) = setup();
        let tool = QueryTool::new(db);
        let err = tool
            .call(&json!({}), &ToolContext::empty())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("query"));
    }

    #[tokio::test]
    async fn checker_tool_returns_sql_unchanged_when_no_issues() {
        let checker = Arc::new(StaticChecker::new());
        let tool = QueryCheckerTool::new(checker);
        let outcome = tool
            .call(
                &json!({ "query": "SELECT Name FROM Genre LIMIT 5" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap();

        assert!(outcome.ok);
        assert_eq!(outcome.observation, "SELECT Name FROM Genre LIMIT 5");
    }

    #[tokio::test]
    async fn checker_tool_reports_issues() {
        let checker = Arc::new(StaticChecker::new());
        let tool = QueryCheckerTool::new(checker);
        let outcome = tool
            .call(
                &json!({ "query": "SELECT * FROM Track WHERE Name NOT IN ('a', NULL)" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap();

        assert!(outcome.ok);
        assert!(outcome.observation.contains("问题"));
        assert!(outcome.observation.contains("NOT IN"));
    }

    #[test]
    fn tool_specs_have_descriptions() {
        let (_dir, db) = setup();
        for spec in [
            ListTablesTool::new(db.clone()).spec(),
            SchemaTool::new(db.clone()).spec(),
            QueryTool::new(db.clone()).spec(),
            QueryCheckerTool::new(Arc::new(StaticChecker::new())).spec(),
        ] {
            assert!(!spec.name.is_empty());
            assert!(!spec.description.is_empty());
        }
    }
}
