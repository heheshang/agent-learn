//! 缓存后端 —— Redis / 内存两种实现 + 统一的 [`CacheBackend`] trait。
//!
//! ## 为什么要有 trait
//!
//! LiteLLM 的 `Cache(type=...)` 支持 `redis` / `local` / `disk` / `s3` / `gcs`。
//! 本项目实现两种：**redis** 和 **local（内存）**。
//!
//! 内存版不是「玩具」—— 它是**对照组**：
//!
//! LiteLLM 文档明确说：
//!
//! > An in-memory cache lives inside one worker process, so a proxy running four
//! > workers keeps four separate caches and the hit rate drops roughly by the
//! > worker count.
//!
//! 所以本项目要能**实测**这件事：内存缓存 vs Redis 缓存在多 worker 下命中率差多少。
//!
//! ## ⚠️ Redis 挂掉必须降级，不能拖垮 Agent
//!
//! 这是本项目最重要的一条设计约束。
//!
//! 缓存是**优化**，不是**功能**。缓存不可用时正确的行为是：
//! **打一条 `warn!` 然后当 miss 处理** —— 请求照常走模型。
//!
//! 反过来做（缓存不可用就报错）会让一个**辅助组件**的故障
//! 变成**整个服务的不可用**。这是典型的「缓存依赖倒倒置」。
//!
//! 参考本仓库的通用约定：「降级必须打 `warn!`（静默降级等于埋雷）」。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agent_logging::{debug, warn};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 缓存里存的值。
///
/// ## ⚠️ 为什么不直接存 `ChatResponse`
///
/// `genai::ChatResponse` 是第三方类型，它的 serde 表现是
/// `{"content":{"parts":[...]},"model_iden":{...}}` ——
///
/// 1. **焊死了第三方内部结构**：升级 `genai` 可能让全部历史缓存无法反序列化。
/// 2. **存了用不上的字段**：`model_iden` / `provider_model_iden` /
///    `stop_reason` / `captured_raw_body` 对「把响应喂回 Agent 循环」毫无用处，
///    却让每条缓存变大。
/// 3. **反序列化时可能失败**：`ModelIden` 的 `AdapterKind` 枚举如果加了新变体，
///    老缓存里的字符串就认不出来了。
///
/// 所以本项目定义**自己的** [`CachedResponse`] ——
/// 17-agent-memory 已经踩过这个坑（「事实用自己的类型，不直接存 `genai::ChatMessage`」）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    /// 模型输出的文本。
    pub text: String,

    /// 模型输出的**工具调用**（中间轮才有）。
    ///
    /// ## ⚠️ 为什么需要这个字段
    ///
    /// 原本只缓存「纯文本的最终答案」那一轮 —— 于是**中间轮永远真调模型**。
    ///
    /// 而这恰恰是 agentic 链路命中不了的**根本原因**（不只是 `call_id`）：
    ///
    /// ```text
    /// 第 1 步（模型决定调工具）：不写缓存 → 每次都真调
    ///   → 模型每次的措辞不同（"我查一下知识库" vs "让我检索相关资料"）
    ///   → 第 2 步的历史不同 → 第 2 步的键不同 → 必然 miss
    /// ```
    ///
    /// 也就是说：**中间轮的不稳定性会顺着消息历史污染后面所有轮次**。
    ///
    /// 所以要让它能命中，就必须**连中间轮一起缓存** ——
    /// 而中间轮的响应里含 `tool_calls`，不存它们就还原不出来。
    ///
    /// ⚠️ 存的是**本项目自己的结构**（不是 `genai::ToolCall`），
    /// 理由同 `CachedResponse` 本身：不把第三方类型的内部结构焊死在缓存格式里。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<CachedToolCall>,

    /// 思考内容（推理模型才有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    /// token 用量。
    pub usage: CachedUsage,

    /// 模型名（便于排查「这个缓存是哪个模型产的」）。
    pub model: String,

    /// 写入时间戳（毫秒）—— 用于 `s-maxage` 语义。
    ///
    /// ## 为什么必须自己存
    ///
    /// LiteLLM 的 `s-maxage` 语义是
    /// 「只接受**用户定义时间范围内**的缓存」。
    /// 要判断「这个缓存有多老」，就必须知道它的写入时间。
    ///
    /// Redis 的 `GET` **不返回写入时间**（要 `OBJECT IDLETIME`，
    /// 而那测的是「空闲时间」不是「写入时间」，语义不同）。
    /// 所以时间戳只能**自己存进 value**。
    #[serde(default)]
    pub stored_at_ms: u64,

    /// 缓存键的构成摘要（审计用）。
    #[serde(default)]
    pub key_desc: String,
}

/// token 用量（只存需要的三个字段）。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CachedUsage {
    /// 输入 token。
    pub prompt_tokens: u64,
    /// 输出 token。
    pub completion_tokens: u64,
    /// 总 token。
    pub total_tokens: u64,
}

/// 缓存里的工具调用。
///
/// ## 为什么要有自己的类型
///
/// 同 [`CachedResponse`] 的理由：不把 `genai::ToolCall` 的 serde 表现
/// （`{"call_id":..,"fn_name":..,"fn_arguments":..}`）焊死在缓存格式里。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedToolCall {
    /// 调用 ID。
    pub call_id: String,

    /// 工具名。
    pub name: String,

    /// 参数。
    pub arguments: Value,
}

/// 缓存条目摘要。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// 缓存键。
    pub key: String,

    /// 值的摘要（前若干字符）。
    pub preview: String,

    /// TTL（秒，-1 表示永不过期，-2 表示键不存在）。
    pub ttl: i64,
}

/// 缓存后端的统一接口。
///
/// ## 为什么是同步的（而不是 async）
///
/// `redis` crate 的 `MultiplexedConnection` 的方法本身是 async 的，
/// 但本 trait 用 **`Pin<Box<dyn Future>>`**（和 07 的 `LlmFuture`、
/// 14 的 `ToolFuture` 同一种手法），这样：
///
/// - trait 是 `dyn` 兼容的（`async fn` 在 trait 里不是 dyn 兼容）
/// - 内存实现不需要任何异步运行时
pub trait CacheBackend: Send + Sync + std::fmt::Debug {
    /// 后端类型名（`redis` / `local`）。
    fn kind(&self) -> &'static str;

    /// 读缓存。
    ///
    /// ## ⚠️ 返回 `Result` 而不是 `Option`
    ///
    /// 因为「读失败」和「未命中」是**两件不同的事**：
    ///
    /// - 未命中 → 正常走模型，**不算错误**
    /// - 读失败 → Redis 挂了，要打 `warn!` 并**降级**
    ///
    /// 如果返回 `Option`，这两种情况会被混在一起，
    /// 「Redis 挂了」就会变成**静默的**性能退化 ——
    /// 而静默降级等于埋雷（本仓库的硬约定）。
    fn get<'a>(&'a self, key: &'a str) -> CacheFuture<'a, Result<Option<CachedResponse>, String>>;

    /// 写缓存（带 TTL 秒数）。
    fn set<'a>(
        &'a self,
        key: &'a str,
        value: &'a CachedResponse,
        ttl_seconds: u64,
    ) -> CacheFuture<'a, Result<(), String>>;

    /// 删一个键。
    fn delete<'a>(&'a self, key: &'a str) -> CacheFuture<'a, Result<bool, String>>;

    /// 删某个命名空间下的**所有**键。
    ///
    /// ## ⚠️ 必须用 `SCAN` 不能用 `KEYS`
    ///
    /// `KEYS pattern` 会**阻塞整个 Redis 实例**直到扫完全部键 ——
    /// 在生产上这是能压垮服务的操作。
    /// `SCAN` 是**增量**的，每次只返回一部分，可以安全地在生产用。
    fn clear_namespace<'a>(&'a self, namespace: &'a str) -> CacheFuture<'a, Result<usize, String>>;

    /// 统计某个命名空间下的键数量。
    fn count_namespace<'a>(&'a self, namespace: &'a str) -> CacheFuture<'a, Result<usize, String>>;

    /// 列出某个命名空间下的条目（**摘要**，不含完整内容）。
    ///
    /// ## ⚠️ 只返回摘要的理由
    ///
    /// 缓存里存的是模型的完整输出 —— 可能包含用户的隐私数据。
    /// 排查时人眼看一下开头就够了，没必要把全文吐到 API 响应里。
    fn list_entries<'a>(
        &'a self,
        namespace: &'a str,
    ) -> CacheFuture<'a, Result<Vec<CacheEntry>, String>>;
}

/// 缓存操作的 future。
pub type CacheFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// 生成值摘要（取前 120 个字符）。
fn preview_of(text: &str) -> String {
    let trimmed: String = text.chars().take(120).collect();
    if text.chars().count() > 120 {
        format!("{trimmed}…")
    } else {
        trimmed
    }
}

// ─────────────────────────────────────────────────────────────────────
// 内存后端
// ─────────────────────────────────────────────────────────────────────

/// 内存缓存后端。
///
/// ## 它服务什么目的
///
/// 1. **对照组** —— 和 Redis 后端对比命中率与延迟。
/// 2. **测试替身** —— 单元测试零外部依赖。
/// 3. **降级兜底** —— Redis 连不上时可以退到这里。
///
/// ## ⚠️ 内存缓存的已知局限（LiteLLM 文档明确指出）
///
/// 缓存活在一个 worker 进程里。4 个 worker = 4 份缓存，
/// 命中率大约降到 1/4。这不是实现缺陷，是**架构事实**。
#[derive(Debug, Default)]
pub struct LocalCache {
    entries: Mutex<HashMap<String, LocalEntry>>,
}

/// 内存里的一条缓存。
#[derive(Debug, Clone)]
struct LocalEntry {
    /// 值。
    value: CachedResponse,
    /// 过期时刻。
    expires_at: Option<Instant>,
}

impl LocalCache {
    /// 创建空缓存。
    pub fn new() -> Self {
        Self::default()
    }
}

impl CacheBackend for LocalCache {
    fn kind(&self) -> &'static str {
        "local"
    }

    fn get<'a>(&'a self, key: &'a str) -> CacheFuture<'a, Result<Option<CachedResponse>, String>> {
        Box::pin(async move {
            let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());

            let Some(entry) = guard.get(key) else {
                return Ok(None);
            };

            // ⚠️ 惰性过期：读到的时候才判断过期。
            //
            // 后台定时清理需要额外任务，而这里判过期只是顺手的事。
            // 内存里残留的过期条目会等下一次访问时被清掉 —— 代价是内存占用，
            // 但**不会返回过期数据**。
            if let Some(expires_at) = entry.expires_at
                && Instant::now() >= expires_at
            {
                guard.remove(key);
                return Ok(None);
            }

            Ok(Some(entry.value.clone()))
        })
    }

    fn set<'a>(
        &'a self,
        key: &'a str,
        value: &'a CachedResponse,
        ttl_seconds: u64,
    ) -> CacheFuture<'a, Result<(), String>> {
        Box::pin(async move {
            let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            guard.insert(
                key.to_string(),
                LocalEntry {
                    value: value.clone(),
                    expires_at: Some(Instant::now() + Duration::from_secs(ttl_seconds)),
                },
            );
            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> CacheFuture<'a, Result<bool, String>> {
        Box::pin(async move {
            let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            Ok(guard.remove(key).is_some())
        })
    }

    fn clear_namespace<'a>(&'a self, namespace: &'a str) -> CacheFuture<'a, Result<usize, String>> {
        Box::pin(async move {
            let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let prefix = format!("{namespace}:");
            let before = guard.len();
            guard.retain(|k, _| !k.starts_with(&prefix));
            Ok(before - guard.len())
        })
    }

    fn count_namespace<'a>(&'a self, namespace: &'a str) -> CacheFuture<'a, Result<usize, String>> {
        Box::pin(async move {
            let guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let prefix = format!("{namespace}:");
            Ok(guard.keys().filter(|k| k.starts_with(&prefix)).count())
        })
    }

    fn list_entries<'a>(
        &'a self,
        namespace: &'a str,
    ) -> CacheFuture<'a, Result<Vec<CacheEntry>, String>> {
        Box::pin(async move {
            let guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let prefix = format!("{namespace}:");

            let mut entries: Vec<CacheEntry> = guard
                .iter()
                .filter(|(k, _)| k.starts_with(&prefix))
                .map(|(k, entry)| {
                    let ttl = entry
                        .expires_at
                        .map(|at| at.saturating_duration_since(Instant::now()).as_secs() as i64)
                        .unwrap_or(-1);

                    CacheEntry {
                        key: k.clone(),
                        preview: preview_of(&entry.value.text),
                        ttl,
                    }
                })
                .collect();

            // 按键排序 —— 保证输出稳定（HashMap 迭代顺序不确定）。
            entries.sort_by(|a, b| a.key.cmp(&b.key));
            Ok(entries)
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Redis 后端
// ─────────────────────────────────────────────────────────────────────

/// Redis 缓存后端。
#[derive(Clone)]
pub struct RedisCache {
    conn: Arc<tokio::sync::Mutex<redis::aio::MultiplexedConnection>>,
}

impl std::fmt::Debug for RedisCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisCache").finish_non_exhaustive()
    }
}

impl RedisCache {
    /// 连接 Redis。
    ///
    /// ## ⚠️ 这里的错误是**启动期错误**
    ///
    /// 连不上 Redis 时**启动就失败**，而不是「启动成功但每次请求都失败」。
    ///
    /// 理由：如果允许启动时连不上，那么：
    /// - 健康检查会报 `ok`（误导调用方）
    /// - 每次请求都要走一遍「连接失败 → warn → 当 miss」的路径
    /// - 日志会被淹没
    ///
    /// 想要「Redis 挂了就降级」的语义，用
    /// [`crate::cache::CacheService::connect_with_fallback`] ——
    /// 它会先试 Redis，失败则**明确地**退到内存后端并打 `warn!`。
    pub async fn connect(url: &str) -> Result<Self, String> {
        let client = redis::Client::open(url).map_err(|e| format!("Redis URL 无效: {e}"))?;

        let conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| format!("连接 Redis 失败: {e}"))?;

        // 连上之后立刻 PING 一次 ——
        // `get_multiplexed_async_connection` 只保证 TCP 握手成功，
        // 不保证服务端真的能响应命令（比如连到了错误端口上的其它服务）。
        let mut probe = conn.clone();
        redis::cmd("PING")
            .query_async::<String>(&mut probe)
            .await
            .map_err(|e| format!("Redis PING 失败: {e}"))?;

        debug!(url, "Redis 连接成功");
        Ok(Self {
            conn: Arc::new(tokio::sync::Mutex::new(conn)),
        })
    }
}

impl CacheBackend for RedisCache {
    fn kind(&self) -> &'static str {
        "redis"
    }

    fn get<'a>(&'a self, key: &'a str) -> CacheFuture<'a, Result<Option<CachedResponse>, String>> {
        Box::pin(async move {
            let mut conn = self.conn.lock().await;

            let raw: Option<String> = redis::cmd("GET")
                .arg(key)
                .query_async(&mut *conn)
                .await
                .map_err(|e| format!("Redis GET 失败: {e}"))?;

            let Some(raw) = raw else {
                return Ok(None);
            };

            // ⚠️ 反序列化失败**不能**当成致命错误 ——
            // 它只说明这条缓存坏了（格式升级 / 数据损坏），
            // 当成 miss 重新算就行。
            match serde_json::from_str::<CachedResponse>(&raw) {
                Ok(value) => Ok(Some(value)),
                Err(e) => {
                    warn!(key = %key, error = %e, "缓存反序列化失败 —— 当作未命中");
                    Ok(None)
                }
            }
        })
    }

    fn set<'a>(
        &'a self,
        key: &'a str,
        value: &'a CachedResponse,
        ttl_seconds: u64,
    ) -> CacheFuture<'a, Result<(), String>> {
        Box::pin(async move {
            let payload =
                serde_json::to_string(value).map_err(|e| format!("缓存序列化失败: {e}"))?;

            let mut conn = self.conn.lock().await;

            // ⚠️ 用 `SET ... EX` 而不是 `SET` + `EXPIRE` 两条命令 ——
            // 两条命令之间如果进程挂了，key 会**永久存在**（没有过期时间）。
            // 一条命令是原子的。
            let _: () = redis::cmd("SET")
                .arg(key)
                .arg(payload)
                .arg("EX")
                .arg(ttl_seconds)
                .query_async(&mut *conn)
                .await
                .map_err(|e| format!("Redis SET 失败: {e}"))?;

            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> CacheFuture<'a, Result<bool, String>> {
        Box::pin(async move {
            let mut conn = self.conn.lock().await;
            let removed: usize = redis::cmd("DEL")
                .arg(key)
                .query_async(&mut *conn)
                .await
                .map_err(|e| format!("Redis DEL 失败: {e}"))?;
            Ok(removed > 0)
        })
    }

    fn clear_namespace<'a>(&'a self, namespace: &'a str) -> CacheFuture<'a, Result<usize, String>> {
        Box::pin(async move {
            let pattern = format!("{namespace}:*");
            let mut conn = self.conn.lock().await;

            // ⚠️ 用 `SCAN` 游标**增量**删除 —— 不用 `KEYS`（会阻塞整个实例）。
            let mut cursor = 0u64;
            let mut removed = 0usize;

            loop {
                let (next, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                    .arg(cursor)
                    .arg("MATCH")
                    .arg(&pattern)
                    .arg("COUNT")
                    .arg(200)
                    .query_async(&mut *conn)
                    .await
                    .map_err(|e| format!("Redis SCAN 失败: {e}"))?;

                if !keys.is_empty() {
                    let mut args: Vec<&str> = Vec::with_capacity(keys.len());
                    for k in &keys {
                        args.push(k.as_str());
                    }
                    let deleted: usize = redis::cmd("DEL")
                        .arg(&args)
                        .query_async(&mut *conn)
                        .await
                        .map_err(|e| format!("Redis DEL 失败: {e}"))?;
                    removed += deleted;
                }

                cursor = next;
                if cursor == 0 {
                    break;
                }
            }

            Ok(removed)
        })
    }

    fn count_namespace<'a>(&'a self, namespace: &'a str) -> CacheFuture<'a, Result<usize, String>> {
        Box::pin(async move {
            let pattern = format!("{namespace}:*");
            let mut conn = self.conn.lock().await;

            let mut cursor = 0u64;
            let mut count = 0usize;

            loop {
                let (next, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                    .arg(cursor)
                    .arg("MATCH")
                    .arg(&pattern)
                    .arg("COUNT")
                    .arg(200)
                    .query_async(&mut *conn)
                    .await
                    .map_err(|e| format!("Redis SCAN 失败: {e}"))?;

                count += keys.len();

                cursor = next;
                if cursor == 0 {
                    break;
                }
            }

            Ok(count)
        })
    }

    fn list_entries<'a>(
        &'a self,
        namespace: &'a str,
    ) -> CacheFuture<'a, Result<Vec<CacheEntry>, String>> {
        Box::pin(async move {
            let pattern = format!("{namespace}:*");
            let mut conn = self.conn.lock().await;

            let mut cursor = 0u64;
            let mut entries: Vec<CacheEntry> = Vec::new();

            loop {
                let (next, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                    .arg(cursor)
                    .arg("MATCH")
                    .arg(&pattern)
                    .arg("COUNT")
                    .arg(200)
                    .query_async(&mut *conn)
                    .await
                    .map_err(|e| format!("Redis SCAN 失败: {e}"))?;

                for key in keys {
                    // TTL 单独查一次 —— `GET` 不返回它。
                    let ttl: i64 = redis::cmd("TTL")
                        .arg(&key)
                        .query_async(&mut *conn)
                        .await
                        .unwrap_or(-2);

                    let raw: Option<String> = redis::cmd("GET")
                        .arg(&key)
                        .query_async(&mut *conn)
                        .await
                        .unwrap_or(None);

                    let preview = match raw {
                        Some(text) => match serde_json::from_str::<CachedResponse>(&text) {
                            Ok(value) => preview_of(&value.text),
                            Err(_) => preview_of(&text),
                        },
                        None => String::new(),
                    };

                    entries.push(CacheEntry { key, preview, ttl });
                }

                cursor = next;
                if cursor == 0 {
                    break;
                }
            }

            // 按键排序 —— 保证输出稳定。
            entries.sort_by(|a, b| a.key.cmp(&b.key));
            Ok(entries)
        })
    }
}

/// 缓存里存的一段「工具结果」。
///
/// ## 为什么工具结果也要缓存
///
/// LiteLLM 只缓存 LLM 响应。但本项目的工具分两类：
///
/// - `search_knowledge` / `calculate` —— **纯函数**，同样的输入永远同样的输出。
///   缓存它们是**零风险**的。
/// - `get_weather` —— **实时数据**，缓存它意味着答案可能过时。
///
/// 所以本项目把工具结果缓存做成**可选的、按工具逐个开关**的（见配置 `cacheable_tools`），
/// 默认**只缓存白名单里的工具**。
///
/// 这也回答了 README 里那个问题：
/// 「exact-match cache 会不会给出过时答案？」—— 会，取决于你缓存了什么。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedToolResult {
    /// 工具名。
    pub tool: String,

    /// 观察文本。
    pub observation: String,

    /// 结构化数据。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,

    /// 写入时间戳（毫秒）。
    #[serde(default)]
    pub stored_at_ms: u64,
}

/// 取当前时间戳（毫秒）。
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> CachedResponse {
        CachedResponse {
            tool_calls: Vec::new(),
            text: "你好".to_string(),
            reasoning: None,
            usage: CachedUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            model: "deepseek-flash".to_string(),
            stored_at_ms: now_ms(),
            key_desc: "ns=test".to_string(),
        }
    }

    #[tokio::test]
    async fn local_cache_stores_and_reads() {
        let cache = LocalCache::new();
        assert!(cache.get("k").await.unwrap().is_none());

        cache.set("k", &sample(), 60).await.unwrap();
        let got = cache.get("k").await.unwrap().unwrap();
        assert_eq!(got.text, "你好");
        assert_eq!(got.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn local_cache_expires_entries() {
        let cache = LocalCache::new();
        // TTL = 0 秒 → 立即过期。
        cache.set("k", &sample(), 0).await.unwrap();
        assert!(cache.get("k").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn local_cache_deletes() {
        let cache = LocalCache::new();
        cache.set("k", &sample(), 60).await.unwrap();
        assert!(cache.delete("k").await.unwrap());
        assert!(!cache.delete("k").await.unwrap());
    }

    #[tokio::test]
    async fn local_cache_clears_namespace_only() {
        let cache = LocalCache::new();
        cache.set("ns1:a", &sample(), 60).await.unwrap();
        cache.set("ns1:b", &sample(), 60).await.unwrap();
        cache.set("ns2:a", &sample(), 60).await.unwrap();

        assert_eq!(cache.count_namespace("ns1").await.unwrap(), 2);
        assert_eq!(cache.clear_namespace("ns1").await.unwrap(), 2);
        assert_eq!(cache.count_namespace("ns1").await.unwrap(), 0);
        // 另一个命名空间不受影响。
        assert_eq!(cache.count_namespace("ns2").await.unwrap(), 1);
    }

    #[tokio::test]
    async fn local_cache_lists_entries_sorted() {
        let cache = LocalCache::new();
        cache.set("ns:b", &sample(), 60).await.unwrap();
        cache.set("ns:a", &sample(), 60).await.unwrap();

        let entries = cache.list_entries("ns").await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].key < entries[1].key);
        assert_eq!(entries[0].preview, "你好");
    }

    #[tokio::test]
    async fn local_cache_reports_kind() {
        assert_eq!(LocalCache::new().kind(), "local");
    }

    #[test]
    fn cached_response_roundtrips_through_json() {
        let original = sample();
        let json = serde_json::to_string(&original).unwrap();
        let back: CachedResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.text, original.text);
        assert_eq!(back.model, original.model);
        assert_eq!(back.stored_at_ms, original.stored_at_ms);
    }

    #[test]
    fn cached_tool_result_roundtrips() {
        let original = CachedToolResult {
            tool: "calculate".to_string(),
            observation: "42".to_string(),
            data: Some(json!({"result": 42})),
            stored_at_ms: now_ms(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let back: CachedToolResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool, "calculate");
        assert_eq!(back.data.unwrap()["result"], json!(42));
    }

    #[test]
    fn preview_truncates_long_text() {
        let long = "x".repeat(200);
        let preview = preview_of(&long);
        assert_eq!(preview.chars().count(), 121); // 120 + 省略号
    }
}
