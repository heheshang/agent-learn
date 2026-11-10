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
        }
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
}
