//! 缓存层 —— 本项目的核心。
//!
//! ## 模块划分
//!
//! | 模块 | 职责 | 对应 LiteLLM 概念 |
//! |---|---|---|
//! | [`key`] | 缓存键构造（内容寻址） | `Cache.get_cache_key` |
//! | [`backend`] | Redis / 内存两种后端 | `Cache(type=...)` |
//! | [`control`] | 单次请求的缓存控制 | `cache={no-cache,no-store,ttl,s-maxage}` |
//! | [`service`] | 把上面三者组合起来 + 降级 | `litellm.cache` 门面 |
//!
//! ## 数据流
//!
//! ```text
//! Agent 的一次 LLM 轮次
//!   │
//!   ├─▶ CacheKey::new(model, messages, tools, ...)
//!   │       └─▶ digest() = SHA-256(规范化 JSON)
//!   │
//!   ├─▶ CacheService::get_response(key, control)
//!   │       ├─ 命中 → 直接返回缓存的响应（不调模型）
//!   │       └─ 未命中 → 调模型 → CacheService::set_response(...)
//! ```

pub mod backend;
pub mod call_id;
pub mod control;
pub mod key;
pub mod service;

pub use backend::{
    CacheBackend, CacheEntry, CacheFuture, CachedResponse, CachedToolCall, CachedToolResult,
    CachedUsage, LocalCache, RedisCache, now_ms,
};
pub use call_id::{
    CALL_ID_VERSION, assign_deterministic_call_ids, canonical_args, deterministic_call_id,
};
pub use control::CacheControl;
pub use key::{CACHE_KEY_VERSION, CacheKey, DEFAULT_NAMESPACE, tools_digest};
pub use service::{CacheService, CacheStats, CacheStatsSnapshot};
