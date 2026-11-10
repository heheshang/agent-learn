//! 全局配置。
//!
//! 环境变量前缀用 `ASYNC_`（Async），
//! 避免和其它项目撞名（13 用 `AGENT_`，14 用 `TOOL_`，15 用 `SQL_`，
//! 16 用 `FC_`，17 用 `MEM_`，18 用 `ERR_`，19 用 `SUM_`，21 用 `CACHE_`）。

use std::time::Duration;

use agent_16::llm::ToolChoice;

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 模型名。
    pub model: String,

    /// 生成温度。
    ///
    /// 默认 0 —— 对照实验要可复现。
    pub temperature: f64,

    /// 生成 max tokens。
    pub max_tokens: u32,

    /// 是否关闭推理模型的「思考」模式。
    pub disable_reasoning: bool,

    /// Agent 最大步数。
    pub max_steps: usize,

    /// 工具选择策略。
    pub tool_choice: ToolChoice,

    /// 工具的**额外**重试次数（总次数 = 1 + max_retries）。
    pub max_retries: usize,

    /// 重试退避的基准延迟（毫秒）。
    pub base_delay_ms: u64,

    /// 重试退避的上限（毫秒）。
    pub max_delay_ms: u64,

    /// ⚠️ **并发上限**（LangChain 的 `max_concurrency` 对应物）。
    ///
    /// 这是本项目最核心的一个旋钮：
    /// 同时最多跑多少个 I/O 密集型任务。
    ///
    /// ## 为什么要限流
    ///
    /// 「全都并发」听起来总是更好，但：
    /// - 上游会限流（429）
    /// - 本机 socket / fd 会耗尽
    /// - 内存里同时挂着 N 份响应体
    ///
    /// 所以默认值取 **4** —— 足够体现并发收益，又不会打爆上游。
    pub max_concurrency: usize,

    /// 单个工具调用的超时（秒）。
    pub tool_timeout_secs: u64,

    /// 整个请求的超时（秒）。
    pub request_timeout_secs: u64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3140".to_string(),
            model: "deepseek-flash".to_string(),
            temperature: 0.0,
            max_tokens: 4096,
            disable_reasoning: true,
            max_steps: 6,
            tool_choice: ToolChoice::Auto,
            max_retries: 2,
            base_delay_ms: 50,
            max_delay_ms: 2000,
            max_concurrency: 4,
            tool_timeout_secs: 10,
            request_timeout_secs: 120,
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `ASYNC_BIND_ADDR` | 主服务监听地址 | `127.0.0.1:3140` |
    /// | `ASYNC_MODEL` | 模型名 | `deepseek-flash` |
    /// | `ASYNC_TEMPERATURE` | 温度 | `0.0` |
    /// | `ASYNC_MAX_TOKENS` | max tokens | `4096` |
    /// | `ASYNC_REASONING` | 是否开推理模型的思考模式 | `false`（即关思考） |
    /// | `ASYNC_MAX_STEPS` | Agent 最大步数 | `6` |
    /// | `ASYNC_TOOL_CHOICE` | `auto`/`none`/`required`/`forced:工具名` | `auto` |
    /// | `ASYNC_MAX_RETRIES` | 工具额外重试次数 | `2` |
    /// | `ASYNC_BASE_DELAY_MS` | 退避基准延迟（毫秒） | `50` |
    /// | `ASYNC_MAX_DELAY_MS` | 退避上限（毫秒） | `2000` |
    /// | `ASYNC_MAX_CONCURRENCY` | **并发上限** | `4` |
    /// | `ASYNC_TOOL_TIMEOUT_SECS` | 单个工具调用超时（秒） | `10` |
    /// | `ASYNC_REQUEST_TIMEOUT_SECS` | 整个请求超时（秒） | `120` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("ASYNC_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("ASYNC_MODEL") {
            config.model = v;
        }
        if let Some(v) = env_f64("ASYNC_TEMPERATURE") {
            config.temperature = v;
        }
        if let Some(v) = env_u32("ASYNC_MAX_TOKENS") {
            config.max_tokens = v;
        }
        if let Some(v) = env_bool("ASYNC_REASONING") {
            // ⚠️ 语义翻转：`ASYNC_REASONING=1` = 开思考。
            config.disable_reasoning = !v;
        }
        if let Some(v) = env_usize("ASYNC_MAX_STEPS") {
            config.max_steps = v.clamp(1, 30);
        }
        if let Some(v) = env_str("ASYNC_TOOL_CHOICE")
            && let Some(m) = agent_16::llm::parse_tool_choice(&v)
        {
            config.tool_choice = m;
        }
        if let Some(v) = env_usize("ASYNC_MAX_RETRIES") {
            config.max_retries = v.min(10);
        }
        if let Some(v) = env_u64("ASYNC_BASE_DELAY_MS") {
            config.base_delay_ms = v;
        }
        if let Some(v) = env_u64("ASYNC_MAX_DELAY_MS") {
            config.max_delay_ms = v;
        }
        if let Some(v) = env_usize("ASYNC_MAX_CONCURRENCY") {
            // ⚠️ 上限 1 —— 0 会让 `buffer_unordered(0)` 永远不 poll，
            //    表现为「请求挂住不返回」。
            config.max_concurrency = v.clamp(1, 64);
        }
        if let Some(v) = env_u64("ASYNC_TOOL_TIMEOUT_SECS") {
            config.tool_timeout_secs = v.clamp(1, 120);
        }
        if let Some(v) = env_u64("ASYNC_REQUEST_TIMEOUT_SECS") {
            config.request_timeout_secs = v.clamp(1, 600);
        }
        config
    }

    /// 单个工具调用的超时。
    pub fn tool_timeout(&self) -> Duration {
        Duration::from_secs(self.tool_timeout_secs)
    }

    /// 整个请求的超时。
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs)
    }

    /// 配置摘要（启动时打印）。
    pub fn summary(&self) -> String {
        format!(
            "监听地址  : {}\n\
             模型      : {} temp={} max_tokens={}\n\
             思考模式  : {}\n\
             最大步数  : {}\n\
             工具选择  : {}\n\
             工具重试  : 额外 {} 次（退避 {} ms ~ {} ms）\n\
             并发上限  : {} ★\n\
             工具超时  : {} s\n\
             请求超时  : {} s",
            self.bind_addr,
            self.model,
            self.temperature,
            self.max_tokens,
            if self.disable_reasoning { "关" } else { "开" },
            self.max_steps,
            agent_16::llm::tool_choice_name(&self.tool_choice),
            self.max_retries,
            self.base_delay_ms,
            self.max_delay_ms,
            self.max_concurrency,
            self.tool_timeout_secs,
            self.request_timeout_secs,
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
    fn default_port_is_3140() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3140");
    }

    #[test]
    fn default_is_deterministic() {
        // ⚠️ 对照实验要可复现 —— 默认温度必须是 0。
        assert_eq!(AppConfig::default().temperature, 0.0);
    }

    #[test]
    fn default_concurrency_is_four() {
        assert_eq!(AppConfig::default().max_concurrency, 4);
    }

    #[test]
    fn summary_mentions_key_settings() {
        let summary = AppConfig::default().summary();
        assert!(summary.contains("3140"));
        assert!(summary.contains("deepseek-flash"));
        assert!(summary.contains("并发上限"));
    }
}
