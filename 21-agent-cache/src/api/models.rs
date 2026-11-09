//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{AgentRun, StepRecord};
use crate::cache::CacheStatsSnapshot;

/// 缓存条目（用于 `/api/cache/entries` 排查）。
///
/// ⚠️ 直接复用 [`crate::cache::backend::CacheEntry`] ——
/// 不重复定义一份「长得一样但不是同一个类型」的结构。
pub use crate::cache::backend::CacheEntry;

/// 跑一次 Agent 的请求体。
///
/// ```json
/// { "question": "什么是缓存击穿？" }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct RunRequest {
    /// 用户的问题。
    pub question: String,

    /// 缓存控制参数（可选）。
    ///
    /// ## ⚠️ `no_cache` 和 `no_store` 的语义
    ///
    /// - `no_cache: true` —— **不读**缓存（但仍然写）
    /// - `no_store: true` —— **不写**缓存（但仍然读）
    ///
    /// 名字来自 LiteLLM，语义容易记反 —— 见 [`crate::cache::control`]。
    #[serde(default)]
    pub cache: Option<CacheControlRequest>,
}

/// 单次请求的缓存控制。
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
pub struct CacheControlRequest {
    /// 不读缓存（但会写）。
    #[serde(default)]
    pub no_cache: bool,

    /// 不写缓存（但会读）。
    #[serde(default)]
    pub no_store: bool,

    /// 写入时的 TTL（秒）。
    #[serde(default)]
    pub ttl: Option<u64>,

    /// 只接受这个时间范围内的新鲜缓存（秒）。
    #[serde(default)]
    pub smaxage: Option<u64>,
}

/// 跑一次 Agent 的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct RunResponse {
    /// 最终答案。
    pub answer: String,

    /// 每一步的记录。
    pub steps: Vec<StepRecord>,

    /// 工具调用总次数。
    pub tool_calls: usize,

    /// 用到的工具名。
    pub tools_used: Vec<String>,

    /// 是否达到步数上限。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 模型名。
    pub model: String,

    /// token 用量。
    pub usage: crate::agent::TokenUsage,

    /// 本次命中缓存的轮数。
    pub cache_hits: usize,

    /// 本次真实调模型的轮数。
    pub cache_misses: usize,

    /// 缓存统计快照。
    pub cache_stats: CacheStatsSnapshot,
}

impl RunResponse {
    /// 从 [`AgentRun`] + 缓存统计构造响应。
    pub fn from_run(run: AgentRun, stats: CacheStatsSnapshot) -> Self {
        Self {
            answer: run.answer,
            steps: run.steps,
            tool_calls: run.tool_calls,
            tools_used: run.tools_used,
            hit_max_steps: run.hit_max_steps,
            elapsed_ms: run.elapsed_ms,
            model: run.model,
            usage: run.usage,
            cache_hits: run.cache_hits,
            cache_misses: run.cache_misses,
            cache_stats: stats,
        }
    }
}

/// 缓存统计响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CacheStatsResponse {
    /// 后端类型（`redis` / `local`）。
    pub backend: String,

    /// 是否启用。
    pub enabled: bool,

    /// 命名空间。
    pub namespace: String,

    /// 统计。
    pub stats: CacheStatsSnapshot,
}

/// 缓存条目列表。
#[derive(Debug, Clone, Serialize)]
pub struct CacheEntryList {
    /// 条目数。
    pub count: usize,

    /// 条目。
    pub entries: Vec<CacheEntry>,
}

/// 清空命名空间的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ClearResponse {
    /// 删掉的条目数。
    pub removed: usize,

    /// 命名空间。
    pub namespace: String,
}

/// 直接调用工具的请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct CallToolRequest {
    /// 工具名。
    pub tool: String,

    /// 工具参数。
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

    /// 结果文本。
    pub observation: String,

    /// 结构化数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,

    /// 是否来自缓存。
    pub cached: bool,
}

/// 工具清单里的一项。
#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    /// 工具名。
    pub name: String,

    /// 描述。
    pub description: String,

    /// JSON Schema。
    pub schema: Value,

    /// 是否 strict。
    pub strict: bool,

    /// 结果是否可缓存。
    pub cacheable: bool,
}

/// 工具清单响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ToolListResponse {
    /// 工具数量。
    pub count: usize,

    /// 工具清单。
    pub tools: Vec<ToolInfo>,
}

/// 错误响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// 错误类别（HTTP 状态码的规范名）。
    pub error: String,

    /// 错误详情。
    pub message: String,
}
