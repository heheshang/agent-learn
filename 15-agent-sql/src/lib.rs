//! # 15-agent-sql —— 手撕 **SQL Agent**（LangChain SQL Agent）
//!
//! 这个项目回答一个问题：**怎么让 Agent 安全地查数据库？**
//!
//! 参考资料：
//!
//! - 教程：[Build a SQL agent](https://docs.langchain.com/oss/python/langchain/sql-agent)
//!   —— 四个 `sql_db_*` 工具、ReAct Agent、SQL 自检、Human-in-the-loop
//!
//! ## 核心产出
//!
//! 一个能根据**自然语言**查询 SQLite 数据库的 Agent，
//! 并给出一套**可复现的**安全防线。
//!
//! ## 八个模块
//!
//! | 模块 | 职责 | 对应 LangChain |
//! |---|---|---|
//! | [`guard`] | **SQL 安全护栏**（只读 / 单语句 / 关键字） | 教程里的警告 |
//! | [`db`] | 只读连接 + 种子库加载 + Schema 探查 | `sqlite3.connect` |
//! | [`tools`] | 四个 SQL 工具 | `sql_db_*` 四个工具 |
//! | [`checker`] | SQL 自检（静态 / LLM 两种实现） | `sql_db_query_checker` |
//! | [`prompt`] | 提示词组装 | `system_prompt` |
//! | [`agent`] | ReAct 循环 | `create_agent` |
//! | [`config`] | 配置 | — |
//! | [`api`] | HTTP API | LangSmith Studio |
//!
//! ## 一条贯穿全项目的原则
//!
//! > **执行模型生成的 SQL 本身就是一个高危操作。**
//!
//! 教程自己也警告了：
//!
//! > There are inherent risks in doing this. Make sure that your database
//! > connection permissions are always scoped as narrowly as possible.
//!
//! 但教程给的示例代码是**直接执行**模型给的 SQL：
//!
//! ```python
//! cursor.execute(query)   # 模型说什么就执行什么
//! ```
//!
//! 它自己也标注了：「not intended to be secure or used in production」。
//!
//! ## 本项目的三层防线
//!
//! ```text
//! 第 1 层：护栏（静态检查）   —— 拦掉危险语句
//! 第 2 层：只读连接（运行时） —— SQLite 层面拒绝一切写操作
//! 第 3 层：行数上限 + 标识符转义 —— 限制影响面
//! ```
//!
//! 任何一层单独都有盲区 —— **叠加才可靠**。
//!
//! ## 与 13/14 的关系
//!
//! | 能力 | 来源 |
//! |---|---|
//! | 模型输出解析 | [`agent_13::agent::parse_react_output`]（复用） |
//! | 工具注册表 | [`agent_14::tool::ToolRegistry`]（复用） |
//! | LLM 抽象 | [`rag_07::llm::Llm`]（复用） |
//!
//! **不复用就等于把踩过的坑再踩一次** ——
//! 尤其 13/14 那个「模型回显整个提示词」的坑。

pub mod agent;
pub mod api;
pub mod checker;
pub mod config;
pub mod db;
pub mod error;
pub mod guard;
pub mod llm;
pub mod prompt;
pub mod tools;

pub use agent::{Agent, AgentRun, StepRecord, call_tool_directly};
pub use checker::{CheckOutcome, LlmChecker, QueryChecker, StaticChecker};
pub use config::{AgentConfig, AppConfig, CheckerMode, DbConfig, PromptConfig};
pub use db::{ColumnInfo, Database, QueryResult};
pub use error::{AppError, AppResult, GuardError};
pub use guard::{check_sql, is_read_only, tokenize_sql};
pub use llm::{GenaiLlm, Llm, LlmOptions, ScriptedLlm};
pub use tools::{ListTablesTool, QueryCheckerTool, QueryTool, SchemaTool};
