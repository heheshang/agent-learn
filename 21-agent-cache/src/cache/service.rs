//! 缓存服务层 —— 把「键构造 / 后端 / 控制参数 / 降级」组合起来。
//!
//! ## 职责划分
//!
//! ```text
//! Agent 循环
//!    │  ① 算键（CacheKey）
//!    ▼
//! CacheService
//!    │  ② 读：控制参数 → 后端.get → s-maxage 检查
//!    │  ③ 写：控制参数 → 后端.set(TTL)
//!    ▼
//! CacheBackend（Redis / Local）
//! ```
//!
//! ## 降级语义（本项目最重要的设计约束）
//!
//! 缓存是**优化**，不是**功能**。
//!
//! 所以后端出错时：
//!
//! - **读失败** → 当作未命中（记 `degraded_reads`）
//! - **写失败** → 忽略（记 `degraded_writes`）
//! - **任何失败都要打 `warn!`** —— 静默降级等于埋雷
//!
//! Agent 循环**永远不会**因为缓存不可用而失败。
//!
//! ## ⚠️ 一个反直觉但必须坚持的选择：读失败**不重试**
//!
//! 缓存是**旁路**（cache-aside），不在关键路径上。
//! Redis 超时了，重试只会让请求**更慢**（多等一个超时），
//! 而收益是「可能命中缓存」。
//!
//! 正确的做法是：立即降级，让请求走模型（慢但正确）。
//!
//! 这与 18-agent-error 的结论一致：
//! 「Transient 才重试，Business 只降级」——
//! 但缓存超时在这里连「降级」都不是，是**直接当 miss**。

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_logging::{debug, info, warn};

use crate::cache::backend::{CacheBackend, CacheEntry, CachedResponse, CachedToolResult, now_ms};
use crate::cache::control::CacheControl;
use crate::cache::key::CacheKey;

/// 缓存统计。
///
/// ## 为什么要有它
///
/// 「缓存有没有用」不能靠感觉，要靠**命中率**说话。
/// 而命中率的定义必须先定死：
///
/// - 分母是**请求数**还是**缓存查询次数**？
///   本项目用**缓存查询次数**（每次 Agent 循环可能查多次），
///   并且另外单独统计「端到端请求级」的命中率。
#[derive(Debug, Default)]
pub struct CacheStats {
    /// 缓存查询次数。
    hits: AtomicU64,
    /// 缓存未命中次数。
    misses: AtomicU64,
    /// 写入次数。
    writes: AtomicU64,
    /// 写入被跳过次数（`no-store`）。
    write_skipped: AtomicU64,
    /// 读被跳过次数（`no-cache`）。
    read_skipped: AtomicU64,
    /// 因太老被拒绝次数（`s-maxage`）。
    stale_rejected: AtomicU64,
    /// 降级读次数（后端出错）。
    degraded_reads: AtomicU64,
    /// 降级写次数（后端出错）。
    degraded_writes: AtomicU64,
    /// 因命中而节省的 token（估算：命中条目的 total_tokens）。
    saved_tokens: AtomicU64,
}

impl CacheStats {
    /// 缓存查询次数。
    pub fn queries(&self) -> u64 {
        self.hits.load(Ordering::Relaxed) + self.misses.load(Ordering::Relaxed)
    }

    /// 命中次数。
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// 未命中次数。
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// 写入次数。
    pub fn writes(&self) -> u64 {
        self.writes.load(Ordering::Relaxed)
    }

    /// 写跳过次数。
    pub fn write_skipped(&self) -> u64 {
        self.write_skipped.load(Ordering::Relaxed)
    }

    /// 读跳过次数。
    pub fn read_skipped(&self) -> u64 {
        self.read_skipped.load(Ordering::Relaxed)
    }

    /// 太老被拒次数。
    pub fn stale_rejected(&self) -> u64 {
        self.stale_rejected.load(Ordering::Relaxed)
    }

    /// 降级读次数。
    pub fn degraded_reads(&self) -> u64 {
        self.degraded_reads.load(Ordering::Relaxed)
    }

    /// 降级写次数。
    pub fn degraded_writes(&self) -> u64 {
        self.degraded_writes.load(Ordering::Relaxed)
    }

    /// 节省的 token。
    pub fn saved_tokens(&self) -> u64 {
        self.saved_tokens.load(Ordering::Relaxed)
    }

    /// 命中率（0.0 ~ 1.0）。
    ///
    /// ## ⚠️ 分母是「真正查了缓存的次数」
    ///
    /// 不是「总请求数」—— 因为 `no-cache` 的请求根本没查缓存，
    /// 把它算进分母会**稀释**命中率，让指标失真。
    pub fn hit_rate(&self) -> f64 {
        let queries = self.queries();
        if queries == 0 {
            return 0.0;
        }
        self.hits() as f64 / queries as f64
    }

    /// 快照（可序列化）。
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            queries: self.queries(),
            hits: self.hits(),
            misses: self.misses(),
            writes: self.writes(),
            write_skipped: self.write_skipped(),
            read_skipped: self.read_skipped(),
            stale_rejected: self.stale_rejected(),
            degraded_reads: self.degraded_reads(),
            degraded_writes: self.degraded_writes(),
            saved_tokens: self.saved_tokens(),
            hit_rate: self.hit_rate(),
        }
    }
}

/// 缓存统计快照。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheStatsSnapshot {
    /// 查询次数。
    pub queries: u64,
    /// 命中次数。
    pub hits: u64,
    /// 未命中次数。
    pub misses: u64,
    /// 写入次数。
    pub writes: u64,
    /// 写跳过次数。
    pub write_skipped: u64,
    /// 读跳过次数。
    pub read_skipped: u64,
    /// 太老被拒次数。
    pub stale_rejected: u64,
    /// 降级读次数。
    pub degraded_reads: u64,
    /// 降级写次数。
    pub degraded_writes: u64,
    /// 节省的 token。
    pub saved_tokens: u64,
    /// 命中率。
    pub hit_rate: f64,
}

/// 缓存服务。
///
/// ## 可克隆的原因
///
/// 对照实验需要「同一个后端 + 两套服务」（一组开缓存、一组关缓存），
/// 保证唯一区别是缓存开关。
#[derive(Clone)]
pub struct CacheService {
    backend: Arc<dyn CacheBackend>,
    stats: Arc<CacheStats>,
    default_ttl: u64,
    /// 工具结果缓存的 TTL（秒）。
    tool_ttl: u64,
    /// 命名空间。
    namespace: String,
    /// 哪些工具的结果**允许**被缓存。
    ///
    /// ## ⚠️ 默认**不含** `get_weather`
    ///
    /// 天气是**实时数据**。缓存它意味着「10 分钟前的天气」。
    ///
    /// 这个白名单是**显式**的：要缓存一个工具，就必须把它写进来。
    /// 「默认全缓存 + 黑名单」是错的默认值 ——
    /// 因为新加的工具会自动进入缓存，而作者可能没意识到它的时效性。
    cacheable_tools: Vec<String>,
    /// 是否启用了缓存（关掉时所有读返回 miss、所有写跳过）。
    enabled: bool,
}

impl std::fmt::Debug for CacheService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheService")
            .field("backend", &self.backend.kind())
            .field("enabled", &self.enabled)
            .field("default_ttl", &self.default_ttl)
            .finish()
    }
}

impl CacheService {
    /// 创建缓存服务。
    pub fn new(backend: Arc<dyn CacheBackend>, default_ttl: u64) -> Self {
        Self {
            backend,
            stats: Arc::new(CacheStats::default()),
            default_ttl,
            tool_ttl: default_ttl,
            namespace: crate::cache::key::DEFAULT_NAMESPACE.to_string(),
            cacheable_tools: Vec::new(),
            enabled: true,
        }
    }

    /// 构造**关闭状态**的缓存服务（所有请求都直通模型）。
    ///
    /// ## 为什么要有这个
    ///
    /// 「缓存关掉」必须是**一等公民** ——
    /// 否则对照组实验要靠「把 Redis 停了」来实现，
    /// 而那会引入「连不上 Redis」这个**额外变量**。
    ///
    /// 正确做法：同一个后端、同一个 Agent，只把 `enabled` 翻成 false。
    pub fn disabled(backend: Arc<dyn CacheBackend>) -> Self {
        let mut service = Self::new(backend, 0);
        service.enabled = false;
        service
    }

    /// 用**给定后端**构造关闭状态的缓存服务。
    pub fn disabled_backend(backend: Arc<dyn CacheBackend>) -> Self {
        Self::disabled(backend)
    }

    /// 克隆后端句柄。
    ///
    /// 用于对照实验：把同一个后端同时给「缓存开」和「缓存关」两组，
    /// 保证**唯一区别是缓存开关**。
    pub fn clone_backend(&self) -> Arc<dyn CacheBackend> {
        Arc::clone(&self.backend)
    }

    /// 取出后端句柄。
    pub fn into_backend(self) -> Arc<dyn CacheBackend> {
        self.backend
    }

    /// 设置可缓存的工具白名单。
    pub fn with_cacheable_tools(mut self, tools: Vec<String>) -> Self {
        self.cacheable_tools = tools;
        self
    }

    /// 设置工具结果的 TTL。
    pub fn with_tool_ttl(mut self, seconds: u64) -> Self {
        self.tool_ttl = seconds;
        self
    }

    /// 统计句柄。
    pub fn stats(&self) -> Arc<CacheStats> {
        Arc::clone(&self.stats)
    }

    /// 后端类型。
    pub fn backend_kind(&self) -> &'static str {
        self.backend.kind()
    }

    /// 是否启用。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 连接 Redis，失败则**降级到内存后端**。
    ///
    /// ## ⚠️ 这是一个**明确的降级**，不是静默兜底
    ///
    /// 打 `warn!` 并在返回值里标出实际用的后端类型，
    /// 让调用方能如实告诉用户「你现在用的是本地内存缓存」。
    ///
    /// ## 为什么允许降级而不是直接失败
    ///
    /// 缓存不可用时，Agent 依然能正常工作（只是慢、贵）。
    /// 而「Redis 挂了 → 服务起不来」是把辅助组件变成了硬依赖。
    pub async fn connect_with_fallback(url: &str, default_ttl: u64) -> Self {
        match crate::cache::backend::RedisCache::connect(url).await {
            Ok(redis) => {
                info!(url, backend = "redis", "缓存后端：Redis");
                CacheService::new(Arc::new(redis), default_ttl)
            }
            Err(e) => {
                warn!(
                    url,
                    error = %e,
                    backend = "local",
                    "⚠️ Redis 不可用 —— 降级到内存缓存（多 worker 下命中率会显著下降）"
                );
                CacheService::new(
                    Arc::new(crate::cache::backend::LocalCache::new()),
                    default_ttl,
                )
            }
        }
    }

    /// 读一条 LLM 响应缓存。
    ///
    /// 返回 `None` 表示未命中（或被跳过/降级）。
    pub async fn get_response(
        &self,
        key: &CacheKey,
        control: &CacheControl,
    ) -> Option<CachedResponse> {
        if !self.enabled {
            return None;
        }

        if control.skip_read() {
            self.stats.read_skipped.fetch_add(1, Ordering::Relaxed);
            debug!(key = %key.redis_key(), "no-cache：跳过读缓存");
            return None;
        }

        let redis_key = key.redis_key();

        match self.backend.get(&redis_key).await {
            Ok(Some(value)) => {
                // ⚠️ `s-maxage` 检查必须在**后端返回之后** ——
                // Redis 的 TTL 管的是「还剩多久过期」，
                // 「有多老」只有我们自己存的时间戳知道。
                if control.is_too_old(value.stored_at_ms, now_ms()) {
                    self.stats.stale_rejected.fetch_add(1, Ordering::Relaxed);
                    self.stats.misses.fetch_add(1, Ordering::Relaxed);
                    debug!(
                        key = %redis_key,
                        stored_at_ms = value.stored_at_ms,
                        "缓存太老（超出 s-maxage）—— 当作未命中"
                    );
                    return None;
                }

                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .saved_tokens
                    .fetch_add(value.usage.total_tokens, Ordering::Relaxed);

                debug!(
                    key = %redis_key,
                    desc = %key.describe(),
                    saved_tokens = value.usage.total_tokens,
                    "缓存命中"
                );

                Some(value)
            }
            Ok(None) => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                debug!(key = %redis_key, desc = %key.describe(), "缓存未命中");
                None
            }
            Err(e) => {
                // ⚠️ 降级：读失败当未命中，**不重试**。
                //
                // 缓存在关键路径之外，重试只会让请求更慢。
                self.stats.degraded_reads.fetch_add(1, Ordering::Relaxed);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                warn!(key = %redis_key, error = %e, "缓存读失败 —— 降级为未命中（不重试）");
                None
            }
        }
    }

    /// 写一条 LLM 响应缓存。
    pub async fn set_response(
        &self,
        key: &CacheKey,
        value: &CachedResponse,
        control: &CacheControl,
    ) {
        if !self.enabled {
            return;
        }

        if control.skip_write() {
            self.stats.write_skipped.fetch_add(1, Ordering::Relaxed);
            debug!(key = %key.redis_key(), "no-store：跳过写缓存");
            return;
        }

        let ttl = control.ttl.unwrap_or(self.default_ttl);
        let redis_key = key.redis_key();

        match self.backend.set(&redis_key, value, ttl).await {
            Ok(()) => {
                self.stats.writes.fetch_add(1, Ordering::Relaxed);
                debug!(key = %redis_key, ttl, "缓存写入");
            }
            Err(e) => {
                // ⚠️ 降级：写失败只记日志，不影响请求。
                self.stats.degraded_writes.fetch_add(1, Ordering::Relaxed);
                warn!(key = %redis_key, error = %e, "缓存写失败 —— 降级（不影响本次请求）");
            }
        }
    }

    /// 设置命名空间。
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// 命名空间。
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// 工具结果的缓存键。
    ///
    /// 形态：`{namespace}:tool:{sha256(工具名 + 参数)前16位}`
    fn tool_key(&self, tool: &str, args: &serde_json::Value) -> String {
        use sha2::{Digest, Sha256};

        let payload = serde_json::to_string(&(tool, args)).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(payload.as_bytes());
        let digest: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        format!("{}:tool:{}", self.namespace, &digest[..16])
    }

    /// 这个工具**允许**被缓存吗？
    pub fn is_cacheable_tool(&self, tool: &str) -> bool {
        self.cacheable_tools.iter().any(|t| t == tool)
    }

    /// 读工具结果缓存。
    pub async fn get_tool_result(
        &self,
        tool: &str,
        args: &serde_json::Value,
    ) -> Option<CachedToolResult> {
        if !self.enabled || !self.is_cacheable_tool(tool) {
            return None;
        }

        let key = self.tool_key(tool, args);

        match self.backend.get(&key).await {
            Ok(Some(envelope)) => {
                // ⚠️ 工具结果的存储形状是「`CachedResponse` 的外壳 + `text` 里放工具结果的 JSON」。
                //
                // 之所以要这么绕，是因为后端 trait 只有**一个**序列化形状
                // （见 `backend.rs` 的说明）。读回来后要从 `text` 里再解一层。
                //
                // ⚠️ 这一层解析失败也只当未命中 —— 缓存数据坏了不该让请求失败。
                match serde_json::from_str::<CachedToolResult>(&envelope.text) {
                    Ok(result) => {
                        self.stats.hits.fetch_add(1, Ordering::Relaxed);
                        debug!(tool = %tool, key = %key, "工具结果缓存命中");
                        Some(result)
                    }
                    Err(e) => {
                        self.stats.misses.fetch_add(1, Ordering::Relaxed);
                        warn!(tool = %tool, error = %e, "工具结果缓存解析失败 —— 当作未命中");
                        None
                    }
                }
            }
            Ok(None) => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
            Err(e) => {
                self.stats.degraded_reads.fetch_add(1, Ordering::Relaxed);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                warn!(tool = %tool, error = %e, "工具结果缓存读失败 —— 降级");
                None
            }
        }
    }

    /// 写工具结果缓存。
    pub async fn set_tool_result(
        &self,
        tool: &str,
        args: &serde_json::Value,
        result: &CachedToolResult,
    ) {
        if !self.enabled || !self.is_cacheable_tool(tool) {
            return;
        }

        let key = self.tool_key(tool, args);

        // 工具结果复用 `CachedResponse` 的存储通道 ——
        // 把工具结果**塞进** `text` 字段是不行的（会丢结构化数据），
        // 所以这里把整个工具结果 JSON 存进 `text`。
        //
        // ⚠️ 这是个折中：Redis 里存了两层 JSON。
        //    好处是后端 trait 只有一个序列化形状，实现简单。
        let envelope = CachedResponse {
            text: serde_json::to_string(result).unwrap_or_default(),
            tool_calls: Vec::new(),
            reasoning: None,
            usage: Default::default(),
            model: format!("tool:{tool}"),
            stored_at_ms: now_ms(),
            key_desc: format!("tool={tool}"),
        };

        match self.backend.set(&key, &envelope, self.tool_ttl).await {
            Ok(()) => {
                self.stats.writes.fetch_add(1, Ordering::Relaxed);
                debug!(tool = %tool, key = %key, "工具结果缓存写入");
            }
            Err(e) => {
                self.stats.degraded_writes.fetch_add(1, Ordering::Relaxed);
                warn!(tool = %tool, error = %e, "工具结果缓存写失败 —— 降级");
            }
        }
    }

    /// 列出命名空间下的条目（排查用）。
    ///
    /// ## ⚠️ 只返回**摘要**
    ///
    /// 缓存里存的是模型的完整输出，可能包含敏感内容。
    /// 排查时人眼看个开头就够了。
    pub async fn list_entries(&self, namespace: &str) -> Result<Vec<CacheEntry>, String> {
        self.backend.list_entries(namespace).await
    }

    /// 删一条缓存。
    pub async fn delete(&self, key: &CacheKey) -> bool {
        self.backend
            .delete(&key.redis_key())
            .await
            .unwrap_or_else(|e| {
                warn!(error = %e, "缓存删除失败");
                false
            })
    }

    /// 清空某个命名空间。
    pub async fn clear_namespace(&self, namespace: &str) -> Result<usize, String> {
        self.backend.clear_namespace(namespace).await
    }

    /// 统计某个命名空间的键数量。
    pub async fn count_namespace(&self, namespace: &str) -> Result<usize, String> {
        self.backend.count_namespace(namespace).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::backend::LocalCache;
    use crate::cache::key::CacheKey;
    use genai::chat::{ChatMessage, MessageContent};

    fn key(text: &str) -> CacheKey {
        CacheKey::new(
            "agent-cache",
            "deepseek-flash",
            &[ChatMessage::user(MessageContent::from_text(
                text.to_string(),
            ))],
            "digest",
            "auto",
            0.0,
            2048,
            true,
        )
    }

    fn response(text: &str) -> CachedResponse {
        CachedResponse {
            tool_calls: Vec::new(),
            text: text.to_string(),
            reasoning: None,
            usage: crate::cache::backend::CachedUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
            model: "deepseek-flash".to_string(),
            stored_at_ms: now_ms(),
            key_desc: "test".to_string(),
        }
    }

    fn service() -> CacheService {
        CacheService::new(Arc::new(LocalCache::new()), 60)
    }

    #[tokio::test]
    async fn misses_on_first_read_then_hits() {
        let service = service();
        let k = key("问题");

        assert!(
            service
                .get_response(&k, &CacheControl::default())
                .await
                .is_none()
        );
        service
            .set_response(&k, &response("答案"), &CacheControl::default())
            .await;

        let got = service.get_response(&k, &CacheControl::default()).await;
        assert_eq!(got.unwrap().text, "答案");

        let stats = service.stats().snapshot();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.writes, 1);
        assert!((stats.hit_rate - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn no_cache_skips_read_but_still_writes() {
        let service = service();
        let k = key("问题");

        service
            .set_response(&k, &response("答案"), &CacheControl::default())
            .await;

        // no-cache：不用缓存读。
        assert!(
            service
                .get_response(&k, &CacheControl::bypass_read())
                .await
                .is_none()
        );
        // 但缓存**确实写进去了**。
        assert!(
            service
                .get_response(&k, &CacheControl::default())
                .await
                .is_some()
        );
    }

    #[tokio::test]
    async fn no_store_skips_write_but_still_reads() {
        let service = service();
        let k = key("问题");

        service
            .set_response(&k, &response("答案"), &CacheControl::read_only())
            .await;

        // 没写进去 → 读不到。
        assert!(
            service
                .get_response(&k, &CacheControl::default())
                .await
                .is_none()
        );

        let stats = service.stats().snapshot();
        assert_eq!(stats.write_skipped, 1);
        assert_eq!(stats.writes, 0);
    }

    #[tokio::test]
    async fn disabled_service_never_caches() {
        let service = CacheService::disabled(Arc::new(LocalCache::new()));
        let k = key("问题");

        service
            .set_response(&k, &response("答案"), &CacheControl::default())
            .await;

        assert!(
            service
                .get_response(&k, &CacheControl::default())
                .await
                .is_none()
        );
        let stats = service.stats().snapshot();
        assert_eq!(stats.writes, 0);
        assert_eq!(stats.queries, 0);
    }

    #[tokio::test]
    async fn smaxage_rejects_stale_entry() {
        let service = service();
        let k = key("问题");

        // 造一条「100 秒前写入」的缓存。
        let mut old = response("旧答案");
        old.stored_at_ms = now_ms() - 100_000;

        service
            .set_response(&k, &old, &CacheControl::default())
            .await;

        // smaxage=10 → 太老 → 当未命中。
        let control = CacheControl::normal().with_smaxage(10);
        assert!(service.get_response(&k, &control).await.is_none());

        let stats = service.stats().snapshot();
        assert_eq!(stats.stale_rejected, 1);
    }

    #[tokio::test]
    async fn saves_tokens_on_hit() {
        let service = service();
        let k = key("问题");
        service
            .set_response(&k, &response("答案"), &CacheControl::default())
            .await;
        service.get_response(&k, &CacheControl::default()).await;

        assert_eq!(service.stats().saved_tokens(), 30);
    }

    #[tokio::test]
    async fn tool_results_only_cache_whitelisted_tools() {
        let service = service().with_cacheable_tools(vec!["calculate".to_string()]);
        let args = serde_json::json!({"expression": "1+1"});

        assert!(service.is_cacheable_tool("calculate"));
        assert!(!service.is_cacheable_tool("get_weather"));

        let result = CachedToolResult {
            tool: "calculate".to_string(),
            observation: "2".to_string(),
            data: None,
            stored_at_ms: now_ms(),
        };
        service.set_tool_result("calculate", &args, &result).await;

        let got = service.get_tool_result("calculate", &args).await;
        assert_eq!(got.unwrap().observation, "2");
    }

    #[tokio::test]
    async fn non_cacheable_tool_is_never_stored() {
        let service = service().with_cacheable_tools(vec!["calculate".to_string()]);
        let args = serde_json::json!({"location": "北京"});

        let result = CachedToolResult {
            tool: "get_weather".to_string(),
            observation: "晴 25 度".to_string(),
            data: None,
            stored_at_ms: now_ms(),
        };
        service.set_tool_result("get_weather", &args, &result).await;

        assert!(
            service
                .get_tool_result("get_weather", &args)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn different_arguments_do_not_share_tool_cache() {
        let service = service().with_cacheable_tools(vec!["calculate".to_string()]);
        let result = CachedToolResult {
            tool: "calculate".to_string(),
            observation: "2".to_string(),
            data: None,
            stored_at_ms: now_ms(),
        };

        service
            .set_tool_result(
                "calculate",
                &serde_json::json!({"expression": "1+1"}),
                &result,
            )
            .await;

        assert!(
            service
                .get_tool_result("calculate", &serde_json::json!({"expression": "2+2"}))
                .await
                .is_none()
        );
    }
}
