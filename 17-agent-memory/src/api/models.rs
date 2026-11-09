//! 请求 / 响应模型。
//!
//! 单独放一个文件，是为了让「API 契约」一眼可见 ——
//! 想知道服务能接什么、返回什么，看这一个文件就够了。

use serde::{Deserialize, Serialize};

use crate::agent::AgentRun;
use crate::longterm::MemoryFact;
use crate::state::MemoryMessage;

/// 聊天的请求体。
///
/// ```json
/// {
///   "thread_id": "thread-a",
///   "owner": "alice",
///   "question": "我叫 Bob，我叫什么名字？"
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    /// 线程 id —— **短期记忆的隔离边界**。
    ///
    /// 同一个 `thread_id` = 同一段对话。
    /// 不同 `thread_id` 之间**完全隔离**。
    pub thread_id: String,

    /// 长期记忆的归属（通常是用户 id）。
    ///
    /// ## 默认值
    ///
    /// 缺省时用 `"default"` —— 单用户场景不需要区分。
    #[serde(default = "default_owner")]
    pub owner: String,

    /// 本轮问题。
    pub question: String,
}

fn default_owner() -> String {
    "default".to_string()
}

/// 聊天的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    /// 线程 id。
    pub thread_id: String,

    /// 模型回复。
    pub answer: String,

    /// 本轮召回了哪些长期记忆。
    #[serde(default)]
    pub recalled: Vec<MemoryFact>,

    /// 本轮抽取出了哪些新事实。
    #[serde(default)]
    pub extracted: Vec<MemoryFact>,

    /// 写回之后，线程里有多少条消息。
    pub history_messages: usize,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

impl ChatResponse {
    /// 从 [`AgentRun`] 构造响应。
    pub fn from_run(run: AgentRun) -> Self {
        Self {
            thread_id: run.thread_id,
            answer: run.answer,
            recalled: run.recalled,
            extracted: run.extracted,
            history_messages: run.history_messages,
            elapsed_ms: run.elapsed_ms,
        }
    }
}

/// 线程列表的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ThreadListResponse {
    /// 线程数量。
    pub count: usize,

    /// 线程 id 列表。
    pub threads: Vec<String>,
}

/// 线程详情的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ThreadDetailResponse {
    /// 线程 id。
    pub thread_id: String,

    /// 消息条数。
    pub message_count: usize,

    /// 状态版本号（每次写入 +1）。
    pub version: u64,

    /// 消息历史。
    pub messages: Vec<MemoryMessage>,
}

/// 长期记忆列表的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct MemoryListResponse {
    /// owner。
    pub owner: String,

    /// 事实数量。
    pub count: usize,

    /// 事实列表。
    pub memories: Vec<MemoryFact>,
}

/// 删除操作的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct DeleteResponse {
    /// 是否真的删掉了。
    pub deleted: bool,

    /// 删除的条数（清空长期记忆时用）。
    #[serde(default)]
    pub count: usize,
}

/// 隔离 / 恢复对照的请求体。
///
/// ```json
/// {
///   "owner": "alice",
///   "threads": ["thread-a", "thread-b"],
///   "question": "我叫什么名字？"
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CompareRequest {
    /// 长期记忆的归属。
    #[serde(default = "default_owner")]
    pub owner: String,

    /// 要对照的线程 id 列表（至少两个）。
    pub threads: Vec<String>,

    /// 每个线程问的问题。
    ///
    /// ## ⚠️ 为什么默认问题是「我叫什么名字」
    ///
    /// 因为它**只能靠记忆回答** ——
    /// 模型不可能凭空知道用户的名字。
    ///
    /// 这样「答对了」就一定是「记忆生效了」，
    /// 而不是「模型瞎猜对了」。
    #[serde(default = "default_compare_question")]
    pub question: String,
}

fn default_compare_question() -> String {
    "我叫什么名字？".to_string()
}

/// 隔离 / 恢复对照的响应体。
#[derive(Debug, Clone, Serialize)]
pub struct CompareResponse {
    /// 每个线程的结果。
    pub results: Vec<CompareEntry>,

    /// 结论。
    pub conclusion: CompareConclusion,
}

/// 对照表里的一项。
#[derive(Debug, Clone, Serialize)]
pub struct CompareEntry {
    /// 线程 id。
    pub thread_id: String,

    /// 模型回复。
    pub answer: String,

    /// 本轮召回了哪些长期记忆。
    #[serde(default)]
    pub recalled: Vec<MemoryFact>,

    /// 本轮抽取出了哪些新事实。
    ///
    /// ## 为什么也要带上它
    ///
    /// 「召回了什么」和「抽出了什么」是**两个方向**：
    ///
    /// - `recalled` —— 读长期记忆（跨线程共享的证据）
    /// - `extracted` —— 写长期记忆（事实确实入库的证据）
    ///
    /// 只看 `recalled` 说不清「是没抽到」还是「抽到了但没召回」。
    #[serde(default)]
    pub extracted: Vec<MemoryFact>,

    /// 写回之后，线程里有多少条消息。
    pub history_messages: usize,
}

/// 对照结论。
#[derive(Debug, Clone, Serialize)]
pub struct CompareConclusion {
    /// **线程隔离**：每个线程的历史**只**包含自己的消息。
    ///
    /// ## 这个字段是本端点的核心断言
    ///
    /// 它不是「我们觉得隔离了」，而是**实际检查过**：
    /// 每个线程的历史里不包含其它线程的任何消息。
    pub threads_isolated: bool,

    /// **恢复能力**：同一个 thread_id 第二次调用时，历史条数增加了。
    ///
    /// 这说明「第二次调用读到了第一次写下的状态」。
    pub thread_resumed: bool,

    /// 跨线程的**长期记忆**是否召回了事实。
    ///
    /// ## 为什么它和线程隔离是两回事
    ///
    /// - 线程隔离 = **短期**记忆互不干扰
    /// - 长期记忆共享 = **跨**线程知道同一件事
    ///
    /// 两者**同时成立**才说明两级记忆都正常：
    /// 短期记忆该隔离的隔离，长期记忆该共享的共享。
    pub longterm_shared: bool,
}

/// 错误响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// 错误类别（HTTP 状态码的规范名）。
    pub error: String,

    /// 错误详情。
    pub message: String,
}
