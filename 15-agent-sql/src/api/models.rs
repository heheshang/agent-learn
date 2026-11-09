//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见 ——
//! 想知道服务能接什么、返回什么，看这一个文件就够了。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{AgentRun, StepRecord};

/// 跑一次 Agent 的请求体。
///
/// ```json
/// { "question": "哪个流派的曲目平均时长最长？" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct RunRequest {
    /// 用户的问题。
    pub question: String,
}

/// 跑一次 Agent 的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct RunResponse {
    /// 最终答案。
    pub answer: String,

    /// 每一步的详细记录（思考 / 动作 / 观察）。
    pub steps: Vec<StepRecord>,

    /// 完整的 ReAct 轨迹（模型看到的那份文本）。
    pub trajectory: String,

    /// 工具调用次数。
    pub tool_calls: usize,

    /// 用到的工具名。
    pub tools_used: Vec<String>,

    /// 执行过的 SQL（安全审计用）。
    pub executed_sql: Vec<String>,

    /// 被护栏拦截的 SQL（安全审计用）。
    pub blocked_sql: Vec<String>,

    /// 答案**是否有真实数据支撑**。
    ///
    /// ⚠️ `false` 表示模型没执行过任何查询就给出了答案 ——
    /// 那个答案里的数字是**编的**，不要直接采信。
    pub grounded: bool,

    /// 是否因为达到步数上限而中止。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,
}

impl RunResponse {
    /// 从 [`AgentRun`] 构造响应。
    pub fn from_run(run: AgentRun) -> Self {
        // ⚠️ 先算轨迹再移动字段 —— `trajectory()` 借用 `steps`。
        let trajectory = run.trajectory();
        Self {
            answer: run.answer,
            steps: run.steps,
            trajectory,
            tool_calls: run.tool_calls,
            tools_used: run.tools_used,
            executed_sql: run.executed_sql,
            blocked_sql: run.blocked_sql,
            grounded: run.grounded,
            hit_max_steps: run.hit_max_steps,
            elapsed_ms: run.elapsed_ms,
            model: run.model,
        }
    }
}

/// 直接调用工具的请求体。
///
/// ```json
/// { "tool": "sql_db_query", "args": { "query": "SELECT Name FROM Genre LIMIT 5" } }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CallToolRequest {
    /// 工具名。
    pub tool: String,

    /// 工具参数（可选）。
    #[serde(default)]
    pub args: Value,
}

/// 直接调用工具的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CallToolResponse {
    /// 工具名。
    pub tool: String,

    /// 是否成功。
    pub ok: bool,

    /// 观察文本（成功时是结果，失败时是错误描述）。
    pub observation: String,

    /// 结构化数据（如果工具返回了）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// 工具清单里的一项。
#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    /// 工具名。
    pub name: String,

    /// 工具描述。
    pub description: String,

    /// JSON Schema。
    pub schema: Value,

    /// 参数说明（渲染成人类可读文本）。
    pub parameters: String,
}

/// 工具清单响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ToolListResponse {
    /// 工具数量。
    pub count: usize,

    /// 工具清单。
    pub tools: Vec<ToolInfo>,
}

/// 数据库概览响应体。
#[derive(Debug, Clone, Serialize)]
pub struct SchemaResponse {
    /// 数据库文件路径。
    pub path: String,

    /// 数据表清单（含行数与列信息）。
    pub tables: Vec<TableInfo>,
}

/// 一张表的概览。
#[derive(Debug, Clone, Serialize)]
pub struct TableInfo {
    /// 表名。
    pub name: String,

    /// 行数。
    pub row_count: i64,

    /// 列信息。
    pub columns: Vec<crate::db::ColumnInfo>,

    /// 建表语句（DDL）。
    pub ddl: Option<String>,
}

/// 护栏检查的请求体。
///
/// ```json
/// { "query": "DELETE FROM Track" }
/// ```
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GuardRequest {
    /// 要检查的 SQL。
    pub query: String,
}

/// 护栏检查的响应体。
#[derive(Debug, Clone, serde::Serialize)]
pub struct GuardResponse {
    /// 是否允许执行。
    pub allowed: bool,

    /// 被拦截的原因（允许时为 `null`）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,

    /// 原始 SQL。
    pub query: String,
}

/// 错误响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// 错误类别（HTTP 状态码的规范名）。
    pub error: String,

    /// 错误详情。
    pub message: String,
}
