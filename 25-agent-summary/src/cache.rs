//! **缓存层** —— 内容寻址缓存键 + 缓存控制参数。
//!
//! # 这个模块汇总了哪些历史能力
//!
//! | 能力 | 来自 | 在这里的形态 |
//! |---|---|---|
//! | 内容寻址缓存键（SHA-256） | 21-agent-cache | [`CacheKey`] |
//! | 缓存控制四参数 | 21-agent-cache | [`CacheControl`] |
//!
//! # 三条硬约束
//!
//! ## ① 缓存键必须包含**所有影响输出的东西**
//!
//! 漏一个字段就在两个本该不同的请求间返回错答，
//! 而且症状是「答案看起来很合理」，极难排查。
//!
//! ## ② `no-cache` 管读、`no-store` 管写
//!
//! 名字有**误导性** —— 但语义是明确的。
//!
//! ## ③ `ttl` 管存储、`s-maxage` 管读取
//!
//! 两者是**正交维度**：TTL 是「活多久」，s-maxage 是「多久算过期」。

use serde::{Deserialize, Serialize};

/// 缓存键的版本号。
///
/// ⚠️ 改键的构造逻辑时**必须** bump 版本 ——
/// 否则旧键会被当成新键的合法命中，返回**过期格式**的缓存。
pub const CACHE_KEY_VERSION: &str = "v1";

/// 一条规范化后的消息。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedMessage {
    /// 角色。
    pub role: String,

    /// 内容。
    pub content: String,
}

/// 一个规范化后的工具调用。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedToolCall {
    /// 调用 ID。
    pub id: String,

    /// 工具名。
    pub name: String,

    /// 参数（**规范化后的 JSON 字符串**）。
    pub arguments: String,
}

/// 内容寻址缓存键。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheKey {
    /// 版本。
    pub version: String,

    /// 命名空间。
    pub namespace: String,

    /// 模型名。
    pub model: String,

    /// 消息历史。
    pub messages: Vec<NormalizedMessage>,

    /// 工具定义的摘要。
    pub tools_digest: String,

    /// 工具选择模式。
    pub tool_choice: String,

    /// 温度。
    pub temperature: f64,

    /// 最大 token。
    pub max_tokens: u32,

    /// 是否关闭思考。
    pub disable_reasoning: bool,
}

impl CacheKey {
    /// 构造一个缓存键。
    ///
    /// ## ⚠️ 9 个字段，**一个都不能少**
    ///
    /// 这是硬编码的 —— 加字段时**必须**同步加到这里，
    /// 否则会返回错答。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        namespace: impl Into<String>,
        model: impl Into<String>,
        messages: &[NormalizedMessage],
        tools_digest: impl Into<String>,
        tool_choice: impl Into<String>,
        temperature: f64,
        max_tokens: u32,
        disable_reasoning: bool,
    ) -> Self {
        Self {
            version: CACHE_KEY_VERSION.to_string(),
            namespace: namespace.into(),
            model: model.into(),
            messages: messages.to_vec(),
            tools_digest: tools_digest.into(),
            tool_choice: tool_choice.into(),
            temperature,
            max_tokens,
            disable_reasoning,
        }
    }

    /// ★ 内容的 SHA-256 十六进制摘要。
    ///
    /// ## ⚠️ 序列化必须**确定**
    ///
    /// `serde_json::to_string` 对一个 struct（字段顺序固定）是确定的。
    /// 但如果中间塞了 `HashMap`，顺序就不确定了 ——
    /// 所以这里所有集合都是 `Vec`。
    ///
    /// ## ⚠️ 不能用 `DefaultHasher`
    ///
    /// 它**不保证跨进程稳定**且 seed 随机 ——
    /// 重启一次所有键全变，缓存永远命中不了。
    pub fn digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let json = serde_json::to_string(self).expect("缓存键序列化不会失败");
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Redis 形态的键。
    pub fn redis_key(&self) -> String {
        let digest = self.digest();
        format!("{}:{}:{}", self.namespace, self.version, &digest[..16])
    }

    /// 人类可读的摘要（只展示「哪些维度不同」，不展示完整内容 ——
    /// 完整内容可能含隐私数据）。
    pub fn describe(&self) -> String {
        format!(
            "ns={} model={} msgs={} tools={} choice={} temp={} max_tokens={} reasoning={}",
            self.namespace,
            self.model,
            self.messages.len(),
            &self.tools_digest[..self.tools_digest.len().min(8)],
            self.tool_choice,
            self.temperature,
            self.max_tokens,
            self.disable_reasoning
        )
    }
}

/// 单次请求的缓存控制。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CacheControl {
    /// `no-cache` —— 这次**不要用**缓存（但仍然**写**）。
    pub no_cache: bool,

    /// `no-store` —— 这次**不要存**（但仍然**读**）。
    pub no_store: bool,

    /// TTL（秒）—— 管**存储**。
    pub ttl: Option<u64>,

    /// `s-maxage`（秒）—— 管**读取**。
    pub s_maxage: Option<u64>,
}

impl Default for CacheControl {
    fn default() -> Self {
        Self::normal()
    }
}

impl CacheControl {
    /// 默认（全部启用）。
    pub fn normal() -> Self {
        Self {
            no_cache: false,
            no_store: false,
            ttl: None,
            s_maxage: None,
        }
    }

    /// 跳过读。
    pub fn bypass_read() -> Self {
        Self {
            no_cache: true,
            ..Self::normal()
        }
    }

    /// 跳过写。
    pub fn read_only() -> Self {
        Self {
            no_store: true,
            ..Self::normal()
        }
    }

    /// 设 TTL。
    pub fn with_ttl(mut self, seconds: u64) -> Self {
        self.ttl = Some(seconds);
        self
    }

    /// 设 s-maxage。
    pub fn with_smaxage(mut self, seconds: u64) -> Self {
        self.s_maxage = Some(seconds);
        self
    }

    /// 是否跳过读。
    pub fn skip_read(&self) -> bool {
        self.no_cache
    }

    /// 是否跳过写。
    pub fn skip_write(&self) -> bool {
        self.no_store
    }

    /// ★ 缓存条目是否「太老」（超过 `s-maxage`）。
    ///
    /// ## ⚠️ 必须自己存写入时间戳
    ///
    /// Redis 的 `GET` **不返回写入时间**，
    /// `OBJECT IDLETIME` 测的是「**空闲**时间」不是「**写入**时间」——
    /// 语义不同。TTL 与「有多老」不等价。
    pub fn is_too_old(&self, stored_at_ms: u64, now_ms: u64) -> bool {
        match self.s_maxage {
            Some(max_age) => now_ms.saturating_sub(stored_at_ms) > max_age * 1000,
            None => false,
        }
    }
}

/// 工具定义的摘要。
///
/// ## ⚠️ 工具摘要必须**按名字排序**
///
/// 工具的**顺序**不影响模型行为，但影响键 ——
/// 不排序的话，同一组工具换个顺序就 miss。
pub fn tools_digest(specs: &[ToolSpecLite]) -> String {
    use sha2::{Digest, Sha256};
    let mut sorted: Vec<&ToolSpecLite> = specs.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut hasher = Sha256::new();
    for spec in sorted {
        hasher.update(spec.name.as_bytes());
        hasher.update(b"\x00");
        hasher.update(spec.description.as_bytes());
        hasher.update(b"\x00");
        hasher.update(spec.schema.as_bytes());
        hasher.update(b"\x00");
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// 一个轻量的工具定义（只为算摘要）。
#[derive(Debug, Clone)]
pub struct ToolSpecLite {
    /// 工具名。
    pub name: String,

    /// 描述。
    pub description: String,

    /// Schema（**规范化的 JSON 字符串** —— 键要排序）。
    pub schema: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs() -> Vec<NormalizedMessage> {
        vec![NormalizedMessage {
            role: "user".into(),
            content: "问题".into(),
        }]
    }

    fn key() -> CacheKey {
        CacheKey::new(
            "ns",
            "model",
            &msgs(),
            "toolsdigest",
            "auto",
            0.0,
            1024,
            true,
        )
    }

    #[test]
    fn digest_is_stable_across_calls() {
        assert_eq!(key().digest(), key().digest());
    }

    #[test]
    fn digest_is_sensitive_to_every_field() {
        let base = key().digest();

        let mut k = key();
        k.namespace = "other".into();
        assert_ne!(k.digest(), base);

        let mut k = key();
        k.model = "other".into();
        assert_ne!(k.digest(), base);

        let mut k = key();
        k.tools_digest = "other".into();
        assert_ne!(k.digest(), base);

        let mut k = key();
        k.tool_choice = "none".into();
        assert_ne!(k.digest(), base);

        let mut k = key();
        k.temperature = 0.7;
        assert_ne!(k.digest(), base);

        let mut k = key();
        k.max_tokens = 512;
        assert_ne!(k.digest(), base);

        let mut k = key();
        k.disable_reasoning = false;
        assert_ne!(k.digest(), base);

        let mut k = key();
        k.messages.push(NormalizedMessage {
            role: "assistant".into(),
            content: "回答".into(),
        });
        assert_ne!(k.digest(), base);
    }

    #[test]
    fn redis_key_has_three_segments() {
        let rk = key().redis_key();
        assert_eq!(rk.split(':').count(), 3);
        assert!(rk.starts_with("ns:"));
    }

    #[test]
    fn no_cache_skips_read_not_write() {
        let c = CacheControl::bypass_read();
        assert!(c.skip_read());
        assert!(!c.skip_write());
    }

    #[test]
    fn no_store_skips_write_not_read() {
        let c = CacheControl::read_only();
        assert!(!c.skip_read());
        assert!(c.skip_write());
    }

    #[test]
    fn smaxage_is_independent_of_ttl() {
        let c = CacheControl::normal().with_ttl(3600).with_smaxage(60);
        assert_eq!(c.ttl, Some(3600));
        assert_eq!(c.s_maxage, Some(60));
        // 存了 90 秒 → 超过 s-maxage → 太老。
        assert!(c.is_too_old(0, 90_000));
        // 存了 30 秒 → 还新鲜。
        assert!(!c.is_too_old(0, 30_000));
    }

    #[test]
    fn no_smaxage_means_never_too_old() {
        let c = CacheControl::normal();
        assert!(!c.is_too_old(0, 999_999));
    }

    #[test]
    fn tools_digest_ignores_tool_order() {
        let a = ToolSpecLite {
            name: "a".into(),
            description: "d1".into(),
            schema: "{}".into(),
        };
        let b = ToolSpecLite {
            name: "b".into(),
            description: "d2".into(),
            schema: "{}".into(),
        };
        assert_eq!(
            tools_digest(&[a.clone(), b.clone()]),
            tools_digest(&[b, a]),
            "工具顺序不应影响摘要"
        );
    }
}
