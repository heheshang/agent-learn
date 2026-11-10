//! 全局配置。
//!
//! 环境变量前缀 `CACHE_`（21-agent-cache）。
//!
//! 前缀约定（与全仓一致）：
//! 06 用 `RAG_`、07 用 `QUERY_`、13 用 `AGENT_`、14 用 `TOOL_`、
//! 15 用 `SQL_`、16 用 `FC_`、17 用 `MEM_`、18 用 `ERR_`、19 用 `SUM_`、
//! 20 用 `PERF_`、**21 用 `CACHE_`**。
//!
//! ## ⚠️ Redis 环境变量用 `CACHE_REDIS_*` 而不是裸 `REDIS_*`
//!
//! LiteLLM 用裸 `REDIS_HOST` / `REDIS_PORT`。
//!
//! 本项目**故意不用**裸名 —— 因为一台机器上可能跑多个项目，
//! 每个项目的 Redis 地址、数据库号、缓存 TTL 都可能不同。
//! 用 `REDIS_URL` 这种全局名，任何一个项目的 `.env` 都会影响其它项目。
//!
//! 这也是 LiteLLM 文档自己承认的取舍
//! （「Use REDIS_* environment variables as the primary mechanism」是给
//! 单项目部署用的；多项目共存时必须加前缀）。

use crate::cache::key::DEFAULT_NAMESPACE;

/// Redis 配置。
#[derive(Debug, Clone)]
pub struct RedisConfig {
    /// Redis URL。
    pub url: String,

    /// 缓存命名空间。
    ///
    /// ## 为什么命名空间要能配
    ///
    /// 1. **多环境隔离** —— dev / staging / prod 用不同命名空间，
    ///    避免 staging 的脏数据影响 prod。
    /// 2. **整块失效** —— `clear_namespace` 一次删掉一整个命名空间。
    pub namespace: String,

    /// 连不上 Redis 时是否**直接失败**。
    ///
    /// ## ⚠️ 默认 `false`（降级到内存缓存）
    ///
    /// 理由见 [`crate::cache::CacheService::connect_with_fallback`]：
    /// 缓存是优化不是功能，Redis 挂了不该让 Agent 起不来。
    ///
    /// 但在**生产**环境可能希望「Redis 挂了就是挂了」——
    /// 因为降级到内存缓存意味着多 worker 下命中率暴跌，
    /// 而用户以为自己在用 Redis。
    /// 这种情况设 `CACHE_REDIS_REQUIRED=1`。
    pub required: bool,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            namespace: DEFAULT_NAMESPACE.to_string(),
            required: false,
        }
    }
}

/// 缓存策略配置。
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// 是否启用缓存。
    pub enabled: bool,

    /// 默认 TTL（秒）。
    ///
    /// ## 默认 600 秒的取舍
    ///
    /// - 太短（如 30 秒）→ 命中率低，几乎没省到钱
    /// - 太长（如 1 天）→ 答案可能严重过时
    ///
    /// 10 分钟是个折中：**足以覆盖「同一个问题被问多次」的真实场景**
    /// （比如同一个 FAQ 被反复问），又不会让答案过时到不可接受。
    pub default_ttl: u64,

    /// 工具结果缓存的 TTL（秒）。
    ///
    /// 默认比 `default_ttl` **短** —— 因为工具结果是「更底层的数据」，
    /// 它过期得应该更快。
    pub tool_ttl: u64,

    /// 哪些工具的结果允许被缓存。
    ///
    /// ## ⚠️ 默认值是**显式白名单**，不含 `get_weather`
    ///
    /// 这是本项目最重要的一条默认值设计。
    ///
    /// 理由：天气是实时数据。缓存它 =「10 分钟前的天气」。
    /// 而 `search_knowledge` / `calculate` 是**纯函数** ——
    /// 同样的输入永远同样的输出，缓存零风险。
    ///
    /// ⚠️ 不用「默认全部 + 黑名单」：
    /// 新加的工具会自动进缓存，而作者可能没意识到它的时效性。
    /// **白名单**强迫作者做一次显式判断。
    pub cacheable_tools: Vec<String>,

    /// 是否缓存工具结果。
    pub cache_tool_results: bool,

    /// 是否启用**确定性 `call_id`**。
    ///
    /// ## ⚠️ 为什么默认**关闭**
    ///
    /// 开启后，模型返回的随机 `call_id` 会被重写成内容派生的确定性值，
    /// 这让 **agentic 链路**也能命中精确匹配缓存。
    ///
    /// 但它**改写了发给模型的请求**，所以不该偷偷生效 ——
    /// 做成显式开关，才能做**对照实验**（唯一变量是这一个）。
    ///
    /// 详见 [`crate::cache::call_id`]。
    pub normalize_call_id: bool,

    /// 是否缓存**中间轮**（含 `tool_calls` 的轮次）。
    ///
    /// ## ⚠️ 这是 agentic 链路能否命中的**决定性开关**
    ///
    /// 关掉时，只有最后一轮（纯文本答案）写缓存，于是：
    ///
    /// ```text
    /// 第 1 步不写缓存 → 每次都真调模型
    ///   → 模型每次的措辞不同 → 第 2 步的历史不同 → 必然 miss
    /// ```
    ///
    /// **中间轮的不稳定性会顺着消息历史污染后面所有轮次**。
    ///
    /// 打开后命中率大幅上升，代价是「命中中间轮 = 跳过工具执行」。
    pub cache_intermediate_rounds: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_ttl: 600,
            tool_ttl: 60,
            // ⚠️ 天气工具**故意不在**白名单里。
            cacheable_tools: vec!["search_knowledge".to_string(), "calculate".to_string()],
            cache_tool_results: true,
            // ⚠️ 默认**关闭** —— 它改写了发给模型的请求。
            normalize_call_id: false,
            // ⚠️ 默认**关闭** —— 命中中间轮 = 跳过工具执行（结果被冻结）。
            cache_intermediate_rounds: false,
        }
    }
}

/// Agent 循环配置。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 最大步数。
    pub max_steps: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self { max_steps: 5 }
    }
}

/// 提示词配置。
#[derive(Debug, Clone)]
pub struct PromptConfig {
    /// 是否强调严格遵守 Schema。
    pub strict: bool,

    /// 工具选择策略。
    pub tool_choice: agent_16::llm::ToolChoice,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            strict: true,
            tool_choice: agent_16::llm::ToolChoice::Auto,
        }
    }
}

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 模型名。
    pub model: String,

    /// 生成温度。
    pub temperature: f64,

    /// 生成 max tokens。
    pub max_tokens: u32,

    /// 是否关闭推理模型的思考模式。
    pub disable_reasoning: bool,

    /// Agent 循环配置。
    pub agent: AgentConfig,

    /// 提示词配置。
    pub prompt: PromptConfig,

    /// 缓存配置。
    pub cache: CacheConfig,

    /// Redis 配置。
    pub redis: RedisConfig,

    /// 知识库目录。
    pub data_dir: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3130".to_string(),
            model: "deepseek-flash".to_string(),
            temperature: 0.0,
            max_tokens: 2048,
            disable_reasoning: true,
            agent: AgentConfig::default(),
            prompt: PromptConfig::default(),
            cache: CacheConfig::default(),
            redis: RedisConfig::default(),
            data_dir: "data".to_string(),
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `CACHE_BIND_ADDR` | 主服务监听地址 | `127.0.0.1:3130` |
    /// | `CACHE_MODEL` | 模型名 | `deepseek-flash` |
    /// | `CACHE_TEMPERATURE` | 温度 | `0.0` |
    /// | `CACHE_MAX_TOKENS` | max tokens | `2048` |
    /// | `CACHE_MAX_STEPS` | 最大步数 | `5` |
    /// | `CACHE_STRICT` | 强调严格遵守 Schema | `true` |
    /// | `CACHE_TOOL_CHOICE` | `auto`/`none`/`required`/`forced:工具名` | `auto` |
    /// | `CACHE_REASONING` | 是否开思考模式 | `false`（关） |
    /// | `CACHE_ENABLED` | 是否启用缓存 | `true` |
    /// | `CACHE_TTL` | 默认 TTL（秒） | `600` |
    /// | `CACHE_TOOL_TTL` | 工具结果 TTL（秒） | `60` |
    /// | `CACHE_TOOL_RESULTS` | 是否缓存工具结果 | `true` |
    /// | `CACHE_NORMALIZE_CALL_ID` | 是否重写为**确定性 `call_id`** | `false` |
    /// | `CACHE_INTERMEDIATE_ROUNDS` | 是否缓存**中间轮**（agentic 命中的关键） | `false` |
    /// | `CACHE_CACHEABLE_TOOLS` | 工具白名单（逗号分隔） | `search_knowledge,calculate` |
    /// | `CACHE_REDIS_URL` | Redis URL | `redis://127.0.0.1:6379` |
    /// | `CACHE_REDIS_NAMESPACE` | 缓存命名空间 | `agent-cache` |
    /// | `CACHE_REDIS_REQUIRED` | Redis 连不上是否直接失败 | `false` |
    /// | `CACHE_DATA_DIR` | 知识库目录 | `data` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("CACHE_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("CACHE_MODEL") {
            config.model = v;
        }
        if let Some(v) = env_f64("CACHE_TEMPERATURE") {
            config.temperature = v;
        }
        if let Some(v) = env_u32("CACHE_MAX_TOKENS") {
            config.max_tokens = v;
        }
        if let Some(v) = env_usize("CACHE_MAX_STEPS") {
            config.agent.max_steps = v.clamp(1, 30);
        }
        if let Some(v) = env_bool("CACHE_STRICT") {
            config.prompt.strict = v;
        }
        if let Some(v) = env_str("CACHE_TOOL_CHOICE")
            && let Some(m) = agent_16::llm::parse_tool_choice(&v)
        {
            config.prompt.tool_choice = m;
        }
        if let Some(v) = env_bool("CACHE_REASONING") {
            config.disable_reasoning = !v;
        }
        if let Some(v) = env_bool("CACHE_ENABLED") {
            config.cache.enabled = v;
        }
        if let Some(v) = env_u64("CACHE_TTL") {
            config.cache.default_ttl = v;
        }
        if let Some(v) = env_u64("CACHE_TOOL_TTL") {
            config.cache.tool_ttl = v;
        }
        if let Some(v) = env_bool("CACHE_TOOL_RESULTS") {
            config.cache.cache_tool_results = v;
        }
        if let Some(v) = env_bool("CACHE_NORMALIZE_CALL_ID") {
            config.cache.normalize_call_id = v;
        }
        if let Some(v) = env_bool("CACHE_INTERMEDIATE_ROUNDS") {
            config.cache.cache_intermediate_rounds = v;
        }
        if let Some(v) = env_str("CACHE_CACHEABLE_TOOLS") {
            config.cache.cacheable_tools = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if let Some(v) = env_str("CACHE_REDIS_URL") {
            config.redis.url = v;
        }
        if let Some(v) = env_str("CACHE_REDIS_NAMESPACE") {
            config.redis.namespace = v;
        }
        if let Some(v) = env_bool("CACHE_REDIS_REQUIRED") {
            config.redis.required = v;
        }
        if let Some(v) = env_str("CACHE_DATA_DIR") {
            config.data_dir = v;
        }

        config
    }

    /// 配置摘要。
    pub fn summary(&self) -> String {
        format!(
            "监听地址  : {}\n\
             模型      : {} temp={} max_tokens={}\n\
             思考模式  : {}\n\
             最大步数  : {}\n\
             缓存      : {}（TTL {}s，工具结果 {}）\n\
             工具白名单: {}\n\
             Redis     : {}\n\
             命名空间  : {}\n\
             Redis必需 : {}",
            self.bind_addr,
            self.model,
            self.temperature,
            self.max_tokens,
            if self.disable_reasoning { "关" } else { "开" },
            self.agent.max_steps,
            if self.cache.enabled { "开" } else { "关" },
            self.cache.default_ttl,
            if self.cache.cache_tool_results {
                "开"
            } else {
                "关"
            },
            if self.cache.cacheable_tools.is_empty() {
                "（空）".to_string()
            } else {
                self.cache.cacheable_tools.join(", ")
            },
            self.redis.url,
            self.redis.namespace,
            if self.redis.required {
                "是"
            } else {
                "否（降级到内存）"
            },
        )
    }
}

/// 读一个非空环境变量。
fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// 读一个 `usize` 环境变量。
fn env_usize(key: &str) -> Option<usize> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `u32` 环境变量。
fn env_u32(key: &str) -> Option<u32> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `u64` 环境变量。
fn env_u64(key: &str) -> Option<u64> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `f64` 环境变量。
fn env_f64(key: &str) -> Option<f64> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个布尔环境变量（`1/true/yes/on` 为真）。
fn env_bool(key: &str) -> Option<bool> {
    env_str(key).map(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_port_is_3130() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3130");
    }

    #[test]
    fn weather_tool_is_not_cacheable_by_default() {
        // ⚠️ 这是一条**设计断言**：天气是实时数据，默认不缓存。
        let config = AppConfig::default();
        assert!(
            !config
                .cache
                .cacheable_tools
                .contains(&"get_weather".to_string())
        );
        assert!(
            config
                .cache
                .cacheable_tools
                .contains(&"calculate".to_string())
        );
    }

    #[test]
    fn default_ttl_is_ten_minutes() {
        assert_eq!(AppConfig::default().cache.default_ttl, 600);
    }

    #[test]
    fn tool_ttl_is_shorter_than_default_ttl() {
        // 工具结果过期得应该更快。
        let config = AppConfig::default();
        assert!(config.cache.tool_ttl < config.cache.default_ttl);
    }

    #[test]
    fn redis_is_optional_by_default() {
        assert!(!AppConfig::default().redis.required);
    }

    #[test]
    fn summary_mentions_key_settings() {
        let summary = AppConfig::default().summary();
        assert!(summary.contains("3130"));
        assert!(summary.contains("agent-cache"));
        assert!(summary.contains("redis://127.0.0.1:6379"));
    }
}
