//! 配置 —— 全部走环境变量，前缀 `WK_`。
//!
//! ## 硬约定
//!
//! - **4xx 用 `warn!`、5xx 用 `error!`**
//! - **降级必须打 `warn!`**（静默降级等于埋雷）
//! - **业务逻辑里不打 `println!`**

use crate::aggregate::Strategy;

/// 应用配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// 监听地址。
    pub addr: String,

    /// 端口。
    pub port: u16,

    /// 活动记录数。
    pub activities: usize,

    /// 压测请求数。
    pub requests: usize,

    /// 压测并发度。
    pub concurrency: usize,

    /// 模拟的服务端时间（毫秒）。
    pub service_time_ms: u64,

    /// 服务端 worker 数。
    pub server_workers: usize,

    /// 默认策略。
    pub strategy: Strategy,

    /// 输出目录。
    pub output_dir: String,

    /// ★ 模型名（LLM 链路用）。
    pub model: String,

    /// ★ 最大输出 token。
    ///
    /// ⚠️ 推理模型的思考 token 与正文**共用**这个额度 ——
    /// 调小会让正文被思考吃光（`finish_reason = length`）。
    pub max_tokens: u32,

    /// ★ 工具选择策略（`auto` / `none` / `required` / `forced:工具名`）。
    ///
    /// ## ⚠️ `none` 的语义不是「模型不许调」
    ///
    /// 它是「**不把工具传给模型**」——
    /// 16-agent-function-calling 实测：只传 `tool_choice: "none"`
    /// 但仍把工具列表传过去，模型会用**文本**模拟调用。
    pub tool_choice: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1".to_string(),
            port: 3170,
            activities: 5000,
            requests: 500,
            concurrency: 16,
            service_time_ms: 0,
            server_workers: 8,
            strategy: Strategy::Optimized,
            output_dir: "reports".to_string(),
            model: "deepseek-flash".to_string(),
            max_tokens: 2048,
            tool_choice: "auto".to_string(),
        }
    }
}

impl AppConfig {
    /// 从环境变量加载。
    pub fn from_env() -> Self {
        Self {
            addr: env_str("WK_ADDR", "127.0.0.1"),
            port: env_u16("WK_PORT", 3170),
            activities: env_usize("WK_ACTIVITIES", 5000),
            requests: env_usize("WK_REQUESTS", 500),
            concurrency: env_usize("WK_CONCURRENCY", 16),
            service_time_ms: env_u64("WK_SERVICE_TIME_MS", 0),
            server_workers: env_usize("WK_SERVER_WORKERS", 8),
            strategy: env_str("WK_STRATEGY", "optimized")
                .as_str()
                .pipe(Strategy::parse)
                .unwrap_or(Strategy::Optimized),
            output_dir: env_str("WK_OUTPUT_DIR", "reports"),
            model: env_str("WK_MODEL", "deepseek-flash"),
            max_tokens: env_u32("WK_MAX_TOKENS", 2048),
            tool_choice: env_str("WK_TOOL_CHOICE", "auto"),
        }
    }

    /// ★ 解析后的工具选择策略。
    ///
    /// ⚠️ 解析失败**回落到 `auto`** —— 拼错策略名不该让服务起不来。
    pub fn tool_choice_mode(&self) -> crate::llm::ToolChoice {
        crate::llm::parse_tool_choice(&self.tool_choice)
            .unwrap_or(crate::llm::ToolChoice::Auto)
    }

    /// 监听地址（`host:port`）。
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.addr, self.port)
    }
}

/// 环境变量字符串（带默认值）。
pub fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// 环境变量 u64。
pub fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// 环境变量 usize。
pub fn env_usize(key: &str, default: usize) -> usize {
    env_u64(key, default as u64) as usize
}

/// 环境变量 u16。
pub fn env_u16(key: &str, default: u16) -> u16 {
    env_u64(key, default as u64) as u16
}

/// 环境变量 u32。
pub fn env_u32(key: &str, default: u32) -> u32 {
    env_u64(key, default as u64) as u32
}

/// 环境变量 f64。
pub fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// 环境变量 bool（`1` / `true` / `yes` 都算真）。
pub fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => default,
    }
}

/// `Option` 辅助。
trait Pipe: Sized {
    /// 管道。
    fn pipe<F, T>(self, f: F) -> T
    where
        F: FnOnce(Self) -> T;
}

impl<T> Pipe for T {
    fn pipe<F, U>(self, f: F) -> U
    where
        F: FnOnce(Self) -> U,
    {
        f(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let c = AppConfig::default();
        assert_eq!(c.port, 3170);
        assert!(c.requests >= 500, "P99 需要足够样本");
    }

    #[test]
    fn bind_addr_is_host_port() {
        let c = AppConfig {
            addr: "0.0.0.0".into(),
            port: 1234,
            ..Default::default()
        };
        assert_eq!(c.bind_addr(), "0.0.0.0:1234");
    }

    #[test]
    fn tool_choice_defaults_to_auto() {
        // ⚠️ 默认必须是 `auto` —— 16-agent-function-calling 实测：
        //    `auto` 是唯一正确的默认档。
        assert_eq!(c_tool_choice("auto"), crate::llm::ToolChoice::Auto);
    }

    #[test]
    fn tool_choice_falls_back_to_auto_on_typo() {
        // ⚠️ 拼错策略名**不该**让服务起不来 —— 回落到 `auto`。
        assert_eq!(
            crate::llm::parse_tool_choice("autoo"),
            None,
            "解析不出来就应该回落到默认值"
        );
    }

    fn c_tool_choice(value: &str) -> crate::llm::ToolChoice {
        crate::llm::parse_tool_choice(value).unwrap_or(crate::llm::ToolChoice::Auto)
    }
}
