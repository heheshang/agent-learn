//! 数据库层 —— 只读连接、种子库加载、Schema 探查。
//!
//! ## 三件事
//!
//! | 函数 | 职责 |
//! |---|---|
//! | [`Database::open_readonly`] | 打开**只读**连接（第 2 层防线） |
//! | [`load_seed`] | 从 `seed.sql` 生成数据库文件 |
//! | [`Database::list_tables`] / [`Database::describe_table`] | 供 Agent 探查 Schema |
//!
//! ## 只读连接是**运行时**防线，和护栏互补
//!
//! 护栏（[`crate::guard`]）是**静态**分析 —— 它拦的是「看起来危险」的语句。
//! 但静态分析有盲区（复杂表达式、编码技巧、护栏本身的解析 bug）。
//!
//! 只读连接是**运行时**防线 —— 由 SQLite 自己保证：
//! 即使一条 `DELETE` 绕过了护栏，数据库也**拒绝执行**。
//!
//! 两者叠加才是「纵深防御」。
//!
//! ## ⚠️ 为什么用 `Mutex<Connection>` 而不是 `Connection`
//!
//! `rusqlite::Connection` 内部用 `RefCell`，所以它**不是 `Sync`** ——
//! 意味着 `Arc<Connection>` 不能放进 axum 的 `State`（要求 `Send + Sync`），
//! 也不能作为 `Tool` 的字段（`Tool: Send + Sync`）。
//!
//! 两个可选方案：
//!
//! | 方案 | 优点 | 缺点 |
//! |---|---|---|
//! | `Mutex<Connection>` | 连接复用，无额外开销 | 查询串行化 |
//! | `Connection` 池（`r2d2`） | 并发查询 | 多一个依赖 + 复杂度 |
//!
//! 本项目选 `Mutex` —— 因为：
//!
//! 1. **SQLite 的并发写本来就有限制**，而且我们是只读的
//! 2. **查询都是毫秒级**，串行化的代价可以忽略
//! 3. **少一个依赖** —— 学习项目应该把复杂度花在刀刃上
//!
//! ⚠️ 但**锁 poisoning** 要处理：某个查询 panic 了，
//! 后续所有查询都会失败。所以下面用的是 `lock().unwrap_or_else(|e| e.into_inner())`
//! —— 拿回被毒化的锁并继续用（SQLite 连接在 panic 后通常还是可用的）。

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use agent_logging::{debug, info, warn};
use rusqlite::Connection;

use crate::error::{AppError, AppResult};

/// 数据库连接（只读）。
pub struct Database {
    /// 只读连接。
    ///
    /// 用 `Mutex` 包一层是为了满足 `Sync` 要求（axum State / Tool trait）。
    conn: Mutex<Connection>,

    path: PathBuf,
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database")
            .field("path", &self.path)
            .finish()
    }
}

impl Database {
    /// 打开**只读**连接。
    ///
    /// ## 第 2 层防线
    ///
    /// 用 `SQLITE_OPEN_READ_ONLY` 打开 —— SQLite 层面拒绝一切写操作。
    /// 即使护栏有漏，数据库也不会被改。
    ///
    /// ## ⚠️ 不存在的文件会**报错**（不是新建空库）
    ///
    /// `SQLITE_OPEN_READ_ONLY` 打不开不存在的文件。
    /// 这是**故意**的 —— 「库不存在」应该是一个明确的错误，
    /// 而不是静默返回一个空库（那会让 Agent 以为「库里没有数据」）。
    pub fn open_readonly(path: &Path) -> AppResult<Self> {
        if !path.exists() {
            return Err(AppError::BadRequest(format!(
                "数据库文件不存在: {}（先跑 `make seed` 生成种子库）",
                path.display()
            )));
        }

        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
        )
        .map_err(|e| AppError::Internal(format!("打开只读连接失败（{}）: {e}", path.display())))?;

        info!(path = %path.display(), "已打开只读数据库连接");
        Ok(Self {
            conn: Mutex::new(conn),
            path: path.to_path_buf(),
        })
    }

    /// 数据库文件路径。
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 拿到连接锁。
    ///
    /// ## ⚠️ 为什么忽略 poisoning
    ///
    /// 如果某个查询 panic 了，`Mutex` 会被标记为 poisoned，
    /// 之后所有 `lock()` 都返回 `Err`。
    ///
    /// 对本项目来说，连接本身通常还是可用的（SQLite 不会因为一次
    /// 查询 panic 就损坏），所以拿回被毒化的锁继续用比「整个服务死掉」好。
    ///
    /// 但我们**记一条 warn 日志** —— 静默忽略 panic 是排查噩梦。
    fn conn(&self) -> MutexGuard<'_, Connection> {
        match self.conn.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("数据库连接锁被毒化（之前有查询 panic 过）—— 继续使用");
                poisoned.into_inner()
            }
        }
    }

    /// 列出所有用户表（排除 `sqlite_%` 内部表）。
    ///
    /// ## 对应 LangChain 的 `sql_db_list_tables`
    ///
    /// 教程里的实现：
    ///
    /// ```python
    /// cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    /// tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
    /// ```
    pub fn list_tables(&self) -> AppResult<Vec<String>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT name FROM sqlite_master WHERE type='table' \
             AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )?;

        let tables = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;

        debug!(count = tables.len(), "已列出数据表");
        Ok(tables)
    }

    /// 查一张表的建表语句（DDL）。
    ///
    /// 对应 LangChain 的 `sql_db_schema` 里的 `SELECT sql FROM sqlite_master`。
    pub fn table_ddl(&self, table: &str) -> AppResult<Option<String>> {
        let conn = self.conn();
        let mut stmt =
            conn.prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name = ?1")?;
        let mut rows = stmt.query([table])?;
        match rows.next()? {
            Some(row) => Ok(row.get::<_, Option<String>>(0)?),
            None => Ok(None),
        }
    }

    /// 查一张表的列信息。
    ///
    /// ## 用 `pragma_table_info` 表值函数
    ///
    /// 直接写 `PRAGMA table_info("表名")` 需要把表名拼进 SQL ——
    /// 而 `pragma_table_info(?)` 可以**用绑定参数**传表名，
    /// 彻底避开标识符注入。
    ///
    /// （护栏禁止的是 `PRAGMA` 作为**用户提交的 SQL**；
    /// 这里的 `pragma_table_info` 是**读元数据的表值函数**，
    /// 在只读连接上完全合法，语义也不同。）
    pub fn table_columns(&self, table: &str) -> AppResult<Vec<ColumnInfo>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT cid, name, type, \"notnull\", dflt_value, pk FROM pragma_table_info(?1)",
        )?;

        let columns = stmt
            .query_map([table], |row| {
                Ok(ColumnInfo {
                    cid: row.get(0)?,
                    name: row.get(1)?,
                    data_type: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    not_null: row.get::<_, i64>(3)? != 0,
                    default_value: row.get::<_, Option<String>>(4)?,
                    primary_key: row.get::<_, i64>(5)? != 0,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(columns)
    }

    /// 取一张表的样例行（最多 `limit` 行）。
    ///
    /// 对应 LangChain 的 `sql_db_schema` 里的 `SELECT * FROM t LIMIT 3`。
    ///
    /// ## ⚠️ 表名是**标识符**，不能用参数占位
    ///
    /// SQL 的参数占位符只能替代**值**，不能替代**标识符**。
    /// 所以表名必须**内联**到 SQL 文本里 ——
    /// 这就必须做**标识符转义**（把 `"` 变成 `""`）。
    ///
    /// ## ⚠️ 转义是必需的，不是可选的
    ///
    /// 表名来自 Agent（模型生成），如果不转义，
    /// 模型给出 `Track"; DROP TABLE x --` 就能注入。
    ///
    /// 但注意：这里执行的是**内部固定模板**，不是模型生成的 SQL，
    /// 所以即便表名被污染，注入的也只是 `SELECT * FROM ...` 的片段，
    /// 且 `LIMIT` 参数是绑定参数。**但还是转义** ——
    /// 纵深防御不能省。
    pub fn sample_rows(&self, table: &str, limit: usize) -> AppResult<Vec<serde_json::Value>> {
        let quoted = quote_identifier(table);
        let sql = format!("SELECT * FROM {quoted} LIMIT ?1");

        let limit = limit.clamp(1, 50);

        let conn = self.conn();
        let mut stmt = conn.prepare(&sql)?;
        let column_names: Vec<String> = stmt.column_names().map(|s| s.to_string()).collect();

        let rows = stmt
            .query_map([limit as i64], |row| row_to_json(row, &column_names))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    /// 执行一条**已经通过护栏**的只读查询。
    ///
    /// ## ⚠️ 调用方必须先跑 [`crate::guard::check_sql`]
    ///
    /// 这里**不再重复检查** —— 重复检查会让人误以为
    /// 「跳过检查也能安全」，反而削弱了护栏的意义。
    ///
    /// 护栏在工具层（`sql_db_query` 工具）统一施加。
    ///
    /// ## 行数上限
    ///
    /// `rusqlite` 的 `query_map` 是**惰性求值** —— 逐行拉取，
    /// 我们只拉 N 行就不会把全表读进内存。
    pub fn query(&self, sql: &str, max_rows: usize) -> AppResult<QueryResult> {
        let start = std::time::Instant::now();

        let conn = self.conn();
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| AppError::Tool(format!("SQL 语法错误: {e}")))?;

        let column_names: Vec<String> = stmt.column_names().map(|s| s.to_string()).collect();

        let mut rows: Vec<serde_json::Value> = Vec::new();
        let mut truncated = false;

        {
            let mut mapped = stmt
                .query([])
                .map_err(|e| AppError::Tool(format!("SQL 执行失败: {e}")))?;

            while let Some(row) = mapped
                .next()
                .map_err(|e| AppError::Tool(format!("SQL 执行失败: {e}")))?
            {
                if rows.len() >= max_rows {
                    truncated = true;
                    break;
                }

                rows.push(row_to_json(row, &column_names)?);
            }
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;

        debug!(
            elapsed_ms,
            row_count = rows.len(),
            truncated,
            "查询执行完成"
        );

        Ok(QueryResult {
            columns: column_names,
            rows,
            truncated,
            elapsed_ms,
        })
    }

    /// 表里的行数（用于给模型一个「大概有多大」的直觉）。
    pub fn count_rows(&self, table: &str) -> AppResult<i64> {
        let quoted = quote_identifier(table);
        let sql = format!("SELECT COUNT(*) FROM {quoted}");
        let conn = self.conn();
        let count: i64 = conn.query_row(&sql, [], |row| row.get(0))?;
        Ok(count)
    }
}

/// 把一行 SQLite 数据转成 JSON 对象。
fn row_to_json(
    row: &rusqlite::Row<'_>,
    column_names: &[String],
) -> rusqlite::Result<serde_json::Value> {
    let mut obj = serde_json::Map::new();
    for (idx, name) in column_names.iter().enumerate() {
        let value = match row.get_ref(idx)? {
            rusqlite::types::ValueRef::Null => serde_json::Value::Null,
            rusqlite::types::ValueRef::Integer(i) => serde_json::Value::from(i),
            rusqlite::types::ValueRef::Real(f) => serde_json::Value::from(f),
            rusqlite::types::ValueRef::Text(t) => {
                serde_json::Value::from(String::from_utf8_lossy(t).to_string())
            }
            rusqlite::types::ValueRef::Blob(b) => {
                serde_json::Value::from(format!("<{} bytes>", b.len()))
            }
        };
        obj.insert(name.clone(), value);
    }
    Ok(serde_json::Value::Object(obj))
}

/// 列信息。
#[derive(Debug, Clone, serde::Serialize)]
pub struct ColumnInfo {
    /// 列序号。
    pub cid: i64,

    /// 列名。
    pub name: String,

    /// 数据类型。
    pub data_type: String,

    /// 是否 NOT NULL。
    pub not_null: bool,

    /// 默认值。
    pub default_value: Option<String>,

    /// 是否主键。
    pub primary_key: bool,
}

/// 查询结果。
#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryResult {
    /// 列名。
    pub columns: Vec<String>,

    /// 行数据（每行是一个 JSON 对象）。
    pub rows: Vec<serde_json::Value>,

    /// 是否因为超过行数上限而被截断。
    pub truncated: bool,

    /// 执行耗时（毫秒）。
    pub elapsed_ms: u64,
}

impl QueryResult {
    /// 渲染成给模型看的文本。
    ///
    /// ## 为什么要渲染成文本
    ///
    /// ReAct 文本协议里，工具结果只能是一段**文本**。
    /// 所以结构化结果必须再渲染一次。
    ///
    /// ## 格式
    ///
    /// ```text
    /// 列: Genre | AvgMs
    /// 行数: 5
    /// ---
    /// 摇滚 | 298123.4
    /// 金属 | 391234.5
    /// ```
    pub fn render(&self) -> String {
        if self.rows.is_empty() {
            return format!(
                "查询成功，但结果为空（0 行）。\n列: {}",
                self.columns.join(" | ")
            );
        }

        let mut out = String::new();
        out.push_str(&format!("列: {}\n", self.columns.join(" | ")));
        out.push_str(&format!("行数: {}\n", self.rows.len()));
        if self.truncated {
            out.push_str("⚠ 结果已截断（超过行数上限）\n");
        }
        out.push_str("---\n");

        for row in &self.rows {
            let cells: Vec<String> = self
                .columns
                .iter()
                .map(|col| {
                    row.get(col)
                        .map(render_cell)
                        .unwrap_or_else(|| "NULL".to_string())
                })
                .collect();
            out.push_str(&cells.join(" | "));
            out.push('\n');
        }

        out
    }
}

/// 渲染一个单元格的值。
fn render_cell(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "NULL".to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// 转义 SQL 标识符。
///
/// 把 `"` 变成 `""`，然后用双引号包起来。
///
/// ## 为什么必须转义
///
/// 表名来自模型 —— 模型可能给出 `Track"; DROP TABLE x --`。
/// 不转义的话，注入就成功了。
///
/// ## 但这只是**第一层**
///
/// 主要防线是 [`crate::guard`] —— 这里转义是因为
/// 「内部固定模板 + 模型提供的标识符」这个组合仍然需要转义。
pub fn quote_identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// 从 `seed.sql` 生成数据库文件。
///
/// ## 为什么需要它
///
/// 仓库里存的是**文本 SQL**（可审阅、可 diff），
/// 真正的 `.db` 是**可再生**的（已在 `.gitignore` 里忽略）。
///
/// ## 幂等
///
/// 如果目标文件已存在，先删掉重建 —— 保证每次都是同一份种子数据
/// （`seed.sql` 里已经用固定随机种子生成好了数据，所以是确定的）。
pub fn load_seed(seed_sql: &Path, db_path: &Path) -> AppResult<()> {
    if let Some(parent) = db_path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    if db_path.exists() {
        warn!(path = %db_path.display(), "种子库已存在 —— 删除重建");
        std::fs::remove_file(db_path)?;
    }

    let sql = std::fs::read_to_string(seed_sql).map_err(|e| {
        AppError::Internal(format!("读取种子 SQL 失败（{}）: {e}", seed_sql.display()))
    })?;

    let conn = Connection::open(db_path)?;
    conn.execute_batch(&sql)
        .map_err(|e| AppError::Internal(format!("执行种子 SQL 失败: {e}")))?;

    // 统计一下规模，写进日志。
    let tables: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        [],
        |row| row.get(0),
    )?;

    info!(
        path = %db_path.display(),
        tables,
        "种子库已生成"
    );

    Ok(())
}

/// 打印数据库概览（给 CLI 用）。
pub fn overview(db: &Database) -> AppResult<String> {
    let tables = db.list_tables()?;
    let mut out = format!("数据库: {}\n", db.path().display());
    out.push_str(&format!("数据表 ({} 张):\n", tables.len()));

    for table in &tables {
        let count = db.count_rows(table)?;
        let columns = db.table_columns(table)?;
        let col_desc: Vec<String> = columns
            .iter()
            .map(|c| {
                if c.primary_key {
                    format!("{}(PK,{})", c.name, c.data_type)
                } else {
                    format!("{}({})", c.name, c.data_type)
                }
            })
            .collect();
        out.push_str(&format!(
            "  {:<12} {:>5} 行  |  {}\n",
            table,
            count,
            col_desc.join(", ")
        ));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (tempfile::TempDir, PathBuf) {
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
        load_seed(&seed, &db_path).unwrap();
        (dir, db_path)
    }

    #[test]
    fn load_seed_creates_the_database() {
        let (_dir, db_path) = temp_db();
        assert!(db_path.exists());
    }

    #[test]
    fn load_seed_is_idempotent() {
        let (dir, db_path) = temp_db();
        let seed = dir.path().join("seed.sql");
        // 再跑一次 —— 应该重建而不是报错。
        load_seed(&seed, &db_path).unwrap();
        let db = Database::open_readonly(&db_path).unwrap();
        assert_eq!(db.count_rows("Track").unwrap(), 3);
    }

    #[test]
    fn lists_user_tables_only() {
        let (_dir, db_path) = temp_db();
        let db = Database::open_readonly(&db_path).unwrap();
        let tables = db.list_tables().unwrap();
        assert_eq!(tables, vec!["Genre".to_string(), "Track".to_string()]);
    }

    #[test]
    fn reads_table_columns() {
        let (_dir, db_path) = temp_db();
        let db = Database::open_readonly(&db_path).unwrap();
        let cols = db.table_columns("Track").unwrap();

        assert_eq!(cols.len(), 4);
        assert_eq!(cols[0].name, "TrackId");
        assert!(cols[0].primary_key);
        assert_eq!(cols[2].name, "GenreId");
        assert!(cols[3].not_null);
    }

    #[test]
    fn reads_sample_rows() {
        let (_dir, db_path) = temp_db();
        let db = Database::open_readonly(&db_path).unwrap();
        let rows = db.sample_rows("Track", 2).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["Name"], "夜色");
        assert_eq!(rows[0]["TrackId"], 1);
    }

    #[test]
    fn query_returns_structured_rows() {
        let (_dir, db_path) = temp_db();
        let db = Database::open_readonly(&db_path).unwrap();

        let result = db
            .query(
                "SELECT g.Name AS genre, COUNT(*) AS n FROM Track t \
                 JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name ORDER BY n DESC",
                100,
            )
            .unwrap();

        assert_eq!(result.columns, vec!["genre".to_string(), "n".to_string()]);
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0]["genre"], "摇滚");
        assert_eq!(result.rows[0]["n"], 2);
    }

    #[test]
    fn query_truncates_at_max_rows() {
        let (_dir, db_path) = temp_db();
        let db = Database::open_readonly(&db_path).unwrap();

        let result = db.query("SELECT * FROM Track", 2).unwrap();

        assert_eq!(result.rows.len(), 2);
        assert!(result.truncated);
    }

    #[test]
    fn query_reports_syntax_error_as_tool_error() {
        let (_dir, db_path) = temp_db();
        let db = Database::open_readonly(&db_path).unwrap();

        let err = db.query("SELECT FROM WHERE", 10).unwrap_err();
        // ⚠️ 语法错误必须是「可降级」的工具错误 ——
        //    模型读到它就能改对 SQL。
        assert!(matches!(err, AppError::Tool(_)));
    }

    #[test]
    fn query_reports_missing_column_as_tool_error() {
        let (_dir, db_path) = temp_db();
        let db = Database::open_readonly(&db_path).unwrap();

        let err = db.query("SELECT nosuchcol FROM Track", 10).unwrap_err();
        assert!(matches!(err, AppError::Tool(_)));
        assert!(err.to_string().contains("nosuchcol"));
    }

    #[test]
    fn readonly_connection_rejects_writes() {
        // ⚠️ **第 2 层防线的核心测试**：
        //    即使 SQL 绕过了护栏，只读连接也必须拒绝。
        let (_dir, db_path) = temp_db();
        let db = Database::open_readonly(&db_path).unwrap();

        let err = db.query("DELETE FROM Track", 10).unwrap_err();
        assert!(matches!(err, AppError::Tool(_)));
        assert!(
            err.to_string().to_lowercase().contains("readonly") || err.to_string().contains("只读"),
            "错误信息应说明是只读限制: {err}"
        );
    }

    #[test]
    fn opening_a_missing_database_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = Database::open_readonly(&dir.path().join("nope.db")).unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[test]
    fn quote_identifier_escapes_quotes() {
        assert_eq!(quote_identifier("Track"), "\"Track\"");
        assert_eq!(
            quote_identifier("Track\"; DROP TABLE x --"),
            "\"Track\"\"; DROP TABLE x --\""
        );
    }

    #[test]
    fn render_shows_columns_and_rows() {
        let result = QueryResult {
            columns: vec!["genre".to_string(), "n".to_string()],
            rows: vec![serde_json::json!({"genre": "摇滚", "n": 2})],
            truncated: false,
            elapsed_ms: 1,
        };

        let text = result.render();
        assert!(text.contains("genre | n"));
        assert!(text.contains("摇滚 | 2"));
    }

    #[test]
    fn render_handles_empty_result() {
        let result = QueryResult {
            columns: vec!["a".to_string()],
            rows: vec![],
            truncated: false,
            elapsed_ms: 0,
        };
        assert!(result.render().contains("结果为空"));
    }
}
