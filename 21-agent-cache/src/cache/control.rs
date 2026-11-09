//! 缓存控制 —— 对齐 LiteLLM 的 4 个 cache-control 参数。
//!
//! ## 参考
//!
//! LiteLLM 文档（*Cache Controls*）列出四个参数：
//!
//! | 参数 | 语义 |
//! |---|---|
//! | `no-cache` | **不返回**缓存响应，直接调真实端点（但**仍然会写**缓存） |
//! | `no-store` | **不写**缓存（但**仍然会读**缓存） |
//! | `ttl` | 缓存存活时间（秒） |
//! | `s-maxage` | **只接受**在这个时间范围内的缓存 |
//!
//! ## ⚠️ `no-cache` 和 `no-store` 的区别是最容易搞错的地方
//!
//! 名字有误导性：
//!
//! - `no-cache` —— 说的不是「不缓存」，而是「**这次不要用缓存**」。
//!   请求仍然会**写入**缓存。
//! - `no-store` —— 说的是「**这次不要存**」。
//!   请求仍然会**读取**缓存。
//!
//! ## ⚠️ `ttl` 和 `s-maxage` 的区别
//!
//! - `ttl` —— **写入时**决定这条缓存活多久（Redis 的 `EX`）。
//! - `s-maxage` —— **读取时**决定「太老的缓存我不认」。
//!
//! 两者可以同时用：`ttl=600` + `s-maxage=60` 意味着
//! 「存 10 分钟，但只有 1 分钟内的新鲜度才认」。
//!
//! 这两个是**不同的维度**，实现上也是两处不同的检查。

use serde::{Deserialize, Serialize};

/// 单次请求的缓存控制参数。
///
/// ## 默认值
///
/// `no_cache = false`、`no_store = false`、`ttl = None`、`smaxage = None`
/// —— 即「正常缓存，使用默认 TTL，无新鲜度上限」。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CacheControl {
    /// 不读缓存（但会写）。
    ///
    /// 对应 LiteLLM 的 `no-cache`。
    pub no_cache: bool,

    /// 不写缓存（但会读）。
    ///
    /// 对应 LiteLLM 的 `no-store`。
    pub no_store: bool,

    /// 写入时的 TTL（秒）。
    ///
    /// 对应 LiteLLM 的 `ttl`。`None` 表示用后端的默认 TTL。
    pub ttl: Option<u64>,

    /// 读取时的新鲜度上限（秒）。
    ///
    /// 对应 LiteLLM 的 `s-maxage`。
    pub smaxage: Option<u64>,
}

impl CacheControl {
    /// 构造「正常缓存」控制。
    pub fn normal() -> Self {
        Self::default()
    }

    /// 构造「绕过缓存读，但仍然写」。
    ///
    /// ## 典型用途
    ///
    /// 「强制重新生成一次，同时把新结果存起来」——
    /// 比如用户点了「刷新」。
    pub fn bypass_read() -> Self {
        Self {
            no_cache: true,
            ..Self::default()
        }
    }

    /// 构造「读缓存，但不写」。
    ///
    /// ## ⚠️ 典型用途
    ///
    /// 「这次是一次性查询，别污染我的缓存」——
    /// 比如**包含随机数/时间戳**的请求（比如带唯一 ID 的追踪查询）。
    ///
    /// 这种请求如果写进缓存，会把一条**永远不会被再次请求**的键
    /// 留在 Redis 里，白占内存。
    pub fn read_only() -> Self {
        Self {
            no_store: true,
            ..Self::default()
        }
    }

    /// 设置 TTL。
    pub fn with_ttl(mut self, seconds: u64) -> Self {
        self.ttl = Some(seconds);
        self
    }

    /// 设置新鲜度上限。
    pub fn with_smaxage(mut self, seconds: u64) -> Self {
        self.smaxage = Some(seconds);
        self
    }

    /// 是否应该**跳过读**。
    pub fn skip_read(&self) -> bool {
        self.no_cache
    }

    /// 是否应该**跳过写**。
    pub fn skip_write(&self) -> bool {
        self.no_store
    }

    /// 检查一条缓存是否**太老**。
    ///
    /// ## ⚠️ 这里比较的是 `stored_at_ms`，不是 Redis 的 TTL
    ///
    /// Redis 的 TTL 是「还剩多久过期」，
    /// 而 `s-maxage` 要的是「这条缓存有多老」。
    ///
    /// 两者不等价：
    /// - TTL=600、缓存写入 500 秒前 → TTL 还剩 100 秒，但缓存已经 500 秒老了
    /// - 如果 `smaxage=60`，这条缓存**必须被拒绝**
    ///
    /// 所以必须自己存写入时间戳（见 [`CachedResponse::stored_at_ms`]）。
    pub fn is_too_old(&self, stored_at_ms: u64, now_ms: u64) -> bool {
        let Some(smaxage) = self.smaxage else {
            return false;
        };

        let age_seconds = now_ms.saturating_sub(stored_at_ms) / 1000;
        age_seconds > smaxage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_normal_caching() {
        let control = CacheControl::default();
        assert!(!control.skip_read());
        assert!(!control.skip_write());
        assert!(control.ttl.is_none());
        assert!(control.smaxage.is_none());
    }

    #[test]
    fn no_cache_skips_read_but_not_write() {
        // ⚠️ 名字有误导性：no-cache 是「不用缓存读」，不是「不缓存」。
        let control = CacheControl::bypass_read();
        assert!(control.skip_read());
        assert!(!control.skip_write());
    }

    #[test]
    fn no_store_skips_write_but_not_read() {
        let control = CacheControl::read_only();
        assert!(!control.skip_read());
        assert!(control.skip_write());
    }

    #[test]
    fn smaxage_rejects_old_entries() {
        let control = CacheControl::normal().with_smaxage(60);
        let now = 1_000_000u64;

        // 30 秒前的缓存 —— 新鲜。
        assert!(!control.is_too_old(now - 30_000, now));
        // 90 秒前的缓存 —— 太老。
        assert!(control.is_too_old(now - 90_000, now));
    }

    #[test]
    fn no_smaxage_accepts_everything() {
        let control = CacheControl::normal();
        assert!(!control.is_too_old(0, 1_000_000));
    }

    #[test]
    fn smaxage_and_ttl_are_independent_dimensions() {
        let control = CacheControl::normal().with_ttl(600).with_smaxage(60);
        assert_eq!(control.ttl, Some(600));
        assert_eq!(control.smaxage, Some(60));
        // TTL 是 10 分钟，但只认 1 分钟内的新鲜度。
        assert!(control.is_too_old(1_000_000 - 120_000, 1_000_000));
    }
}
