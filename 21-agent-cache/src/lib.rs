//! # 21-agent-cache —— 手撕 **Redis 缓存**（缓存 LLM 响应）
//!
//! 这个项目回答一个问题：**Agent 系统的 LLM 响应怎么缓存？缓存什么、什么时候不该缓存？**
//!
//! 参考资料：
//!
//! - 教程：[Using Redis with FastAPI](https://redis.io/tutorials/develop/python/fastapi/)
//!   —— cache-aside 模式、TTL、序列化/反序列化（datetime 的 object_hook）
//! - 工具：[LiteLLM Caching](https://docs.litellm.ai/docs/caching)
//!   —— 四种 cache-control（`no-cache` / `no-store` / `ttl` / `s-maxage`）、
//!   自定义 cache key、Redis 是「唯一跨 worker 共享」的缓存
//!
//! ## 核心产出
//!
//! 一个**缓存层对 Agent 循环透明**的 Function Calling Agent：
//!
//! ```text
//! 用户问题 ──▶ 算缓存键（内容寻址 SHA-256）
//!                │
//!                ├─▶ 命中 → 直接返回缓存响应（**不调模型**）
//!                └─▶ 未命中 → 调模型 → 写缓存（TTL）
//! ```
//!
//! ## 三个最值得记住的结论
//!
//! ### ① 缓存键必须包含**所有**影响输出的东西
//!
//! LiteLLM 文档原话：「Exact-match caches key on a hash of the whole request,
//! so any change to the conversation is a miss.」
//!
//! 漏掉任何一个字段（模型名 / 温度 / 工具定义 / 消息历史），
//! 就会在**两个本该不同的请求**之间返回错答 ——
//! 而且症状是「答案看起来很合理」，极难排查。
//!
//! ### ② `no-cache` 和 `no-store` 的名字有误导性
//!
//! - `no-cache` = 「这次**不要用**缓存」（但仍然**写**）
//! - `no-store` = 「这次**不要存**」（但仍然**读**）
//!
//! ### ③ ⚠️⚠️ exact-match cache 对 agentic traffic **并非**必然失灵
//!
//! 这是本项目**实测**出来的核心结论，而且它**推翻了本项目自己的前一版结论**。
//!
//! 最初的观察是：缓存键里含有 `call_id`（模型每次生成的工具调用 ID 都不同），
//! 而 `call_id` 是 tool 回执的**对账依据**（见 16 的坑 1），
//! 所以「带工具调用的那一轮」几乎必然 miss。
//!
//! 于是当时的结论是「agentic 链路的精确匹配缓存无效」。
//!
//! ⚠️ **但这个结论是错的。**
//!
//! 实测发现真正的元凶是另一件事：
//! **中间轮不写缓存** → 每次都真调模型 → 模型每次的**措辞**不同
//! → 那个措辞在 assistant 消息正文里 → 进消息历史 → 进缓存键
//! → 后面所有轮次的键全都不稳定。
//!
//! 修掉之后（缓存中间轮），agentic 链路命中率 **0% → 100%**，
//! 耗时 **4465ms → 6ms**，token **100% 节省**，答案完全一致。
//!
//! 教训：「精确匹配对 agentic 无效」是一个**实现细节**的结论，
//! 不是**技术原理**的结论。写「必然 miss」之前，
//! 先问「**是哪个具体的实现选择**造成的」。
//!
//! LiteLLM 文档对**语义缓存**的警告仍然成立：
//! 「Semantic caches embed the prompt and serve the closest match... which
//! suits single-shot prompts and **goes badly wrong on agentic traffic**.」
//!
//! 本项目**不做语义缓存**（那是另一个项目的事），
//! 但把「exact-match 在什么场景有效」这件事**量化**出来。
//!
//! ## 模块划分
//!
//! | 模块 | 职责 | 对应 LiteLLM 概念 |
//! |---|---|---|
//! | [`cache::key`] | 缓存键构造（内容寻址） | `Cache.get_cache_key` |
//! | [`cache::backend`] | Redis / 内存两种后端 | `Cache(type=...)` |
//! | [`cache::control`] | 单次请求的缓存控制 | `cache={no-cache,no-store,ttl,s-maxage}` |
//! | [`cache::call_id`] | 确定性 `call_id`（让 agentic 也能命中） | — |
//! | [`cache::service`] | 组合 + **降级** | `litellm.cache` 门面 |
//! | [`agent`] | 带缓存的 Function Calling 循环 | — |
//! | [`api`] | HTTP 接口 | — |

pub mod agent;
pub mod api;
pub mod cache;
pub mod config;
pub mod error;
pub mod table;

// ── 复用 16-agent-function-calling 的协议层 ────────────────────────
//
// 本项目**不重写** Function Calling —— 直接把 16 的模块 re-export 出来，
// 好处：
// 1. 16 修过的坑（call_id 回执、reasoning 回传、Schema 校验）自动继承；
// 2. 本项目的 diff 聚焦在「缓存」这一个主题上。
pub use agent_16::functions;
pub use agent_16::llm;
pub use agent_16::prompt;
pub use agent_16::tool;
pub use agent_16::tools;

pub use agent::{AgentRun, CachedAgent, CallSource, StepRecord, TokenUsage, ToolCallRecord};
pub use cache::{
    CACHE_KEY_VERSION, CALL_ID_VERSION, CacheBackend, CacheControl, CacheKey, CacheService,
    CacheStats, CacheStatsSnapshot, CachedResponse, CachedToolCall, CachedToolResult, LocalCache,
    RedisCache, assign_deterministic_call_ids, canonical_args, deterministic_call_id, tools_digest,
};
pub use config::{AgentConfig, AppConfig, CacheConfig, PromptConfig, RedisConfig};
pub use error::{AppError, AppResult, ToolError};
pub use table::Table;
