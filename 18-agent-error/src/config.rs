//! 全局配置。
//!
//! 环境变量前缀用 `ERR_`（Error），
//! 避免和其它项目撞名（13 用 `AGENT_`，14 用 `TOOL_`，
//! 15 用 `SQL_`，16 用 `FC_`，17 用 `MEM_`）。

use crate::llm::ToolChoiceMode;

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 模型名。
    pub model: String,

    /// 生成温度。
    ///
    /// 错误处理场景要**可复现** —— 温度高会让工具调用序列飘忽，
    /// 对照实验的差值就说不清了。
    pub temperature: f64,

    /// 生成 max tokens。
    pub max_tokens: u32,

    /// 是否关闭推理模型的「思考」模式。
    ///
    /// ## 默认 `true` 的理由（沿用 13/16/17 的结论）
    ///
    /// 思考 token 与正文**共用** `max_tokens` 预算；
    /// 且开思考后 DeepSeek 要求回传 `reasoning_content`，
    /// 否则下一轮 400。关掉就少一个必须正确实现的约束。
    pub disable_reasoning: bool,

    /// Agent 最大步数。
    pub max_steps: usize,

    /// 工具选择策略。
    pub tool_choice: ToolChoiceMode,

    /// 工具的**额外**重试次数（总次数 = 1 + max_retries）。
    pub max_retries: usize,

    /// 重试退避的基准延迟（毫秒）。
    pub base_delay_ms: u64,

    /// 重试退避的上限（毫秒）。
    pub max_delay_ms: u64,

    /// `flaky_fetch` 工具前几次失败（用于演示可重试错误）。
    pub flaky_fail_times: usize,

    /// 是否给支付工具配置 API key（`None` = 不配置 → 致命错误）。
    pub pay_api_key: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3110".to_string(),
            model: "deepseek-flash".to_string(),
            temperature: 0.0,
            max_tokens: 2048,
            disable_reasoning: true,
            max_steps: 5,
            tool_choice: ToolChoiceMode::Auto,
            max_retries: 2,
            base_delay_ms: 50,
            max_delay_ms: 2000,
            flaky_fail_times: 2,
            pay_api_key: None,
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `ERR_BIND_ADDR` | 主服务监听地址 | `127.0.0.1:3110` |
    /// | `ERR_MODEL` | 模型名 | `deepseek-flash` |
    /// | `ERR_TEMPERATURE` | 温度 | `0.0` |
    /// | `ERR_MAX_TOKENS` | max tokens | `2048` |
    /// | `ERR_REASONING` | 是否开推理模型的思考模式 | `false`（即关思考） |
    /// | `ERR_MAX_STEPS` | Agent 最大步数 | `5` |
    /// | `ERR_TOOL_CHOICE` | `auto`/`none`/`required`/`forced:工具名` | `auto` |
    /// | `ERR_MAX_RETRIES` | 工具额外重试次数 | `2` |
    /// | `ERR_BASE_DELAY_MS` | 退避基准延迟（毫秒） | `50` |
    /// | `ERR_MAX_DELAY_MS` | 退避上限（毫秒） | `2000` |
    /// | `ERR_FLAKY_FAIL_TIMES` | flaky 工具前几次失败 | `2` |
    /// | `ERR_PAY_API_KEY` | 支付工具的 API key（不设 → 致命错误） | 无 |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("ERR_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("ERR_MODEL") {
            config.model = v;
        }
        if let Some(v) = env_f64("ERR_TEMPERATURE") {
            config.temperature = v;
        }
        if let Some(v) = env_u32("ERR_MAX_TOKENS") {
            config.max_tokens = v;
        }
        if let Some(v) = env_bool("ERR_REASONING") {
            // ⚠️ 语义翻转：`ERR_REASONING=1` = 开思考。
            config.disable_reasoning = !v;
        }
        if let Some(v) = env_usize("ERR_MAX_STEPS") {
            config.max_steps = v.clamp(1, 30);
        }
        if let Some(v) = env_str("ERR_TOOL_CHOICE")
            && let Some(m) = ToolChoiceMode::parse(&v)
        {
            config.tool_choice = m;
        }
        if let Some(v) = env_usize("ERR_MAX_RETRIES") {
            config.max_retries = v.min(10);
        }
        if let Some(v) = env_u64("ERR_BASE_DELAY_MS") {
            config.base_delay_ms = v;
        }
        if let Some(v) = env_u64("ERR_MAX_DELAY_MS") {
            config.max_delay_ms = v;
        }
        if let Some(v) = env_usize("ERR_FLAKY_FAIL_TIMES") {
            config.flaky_fail_times = v.min(20);
        }
        if let Some(v) = env_str("ERR_PAY_API_KEY") {
            config.pay_api_key = Some(v);
        }

        config
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
             flaky 失败: 前 {} 次\n\
             支付密钥  : {}",
            self.bind_addr,
            self.model,
            self.temperature,
            self.max_tokens,
            if self.disable_reasoning { "关" } else { "开" },
            self.max_steps,
            self.tool_choice.name(),
            self.max_retries,
            self.base_delay_ms,
            self.max_delay_ms,
            self.flaky_fail_times,
            if self.pay_api_key.is_some() {
                "已配置"
            } else {
                "未配置（将触发致命错误）"
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
    fn default_port_is_3110() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3110");
    }

    #[test]
    fn default_is_deterministic() {
        assert_eq!(AppConfig::default().temperature, 0.0);
    }

    #[test]
    fn reasoning_is_disabled_by_default() {
        assert!(AppConfig::default().disable_reasoning);
    }

    #[test]
    fn pay_api_key_is_absent_by_default() {
        // ⚠️ 默认**不**配置密钥 —— 这样才能演示 Fatal 错误。
        assert!(AppConfig::default().pay_api_key.is_none());
    }

    #[test]
    fn summary_mentions_key_settings() {
        let summary = AppConfig::default().summary();
        assert!(summary.contains("3110"));
        assert!(summary.contains("deepseek-flash"));
        assert!(summary.contains("auto"));
    }
}
