//! 全局配置。
//!
//! 环境变量前缀用 `VLLM_`（vLLM），
//! 避开其它项目（13 用 `AGENT_`，14 用 `TOOL_`，15 用 `SQL_`，
//! 16 用 `FC_`，17 用 `MEM_`，18 用 `ERR_`，19 用 `SUM_`，
//! 21 用 `CACHE_`，22 用 `ASYNC_`，23 用 `BATCH_`）。

use std::time::Duration;

use crate::client::ClientConfig;
use crate::metrics::Slo;
use crate::scheduler::EngineConfig;
use crate::workload::WorkloadConfig;

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// HTTP 监听地址（本项目自己的服务）。
    pub bind_addr: String,

    /// 引擎配置（对应 vLLM 的启动参数）。
    pub engine: EngineConfig,

    /// 负载配置。
    pub workload: WorkloadConfig,

    /// 客户端配置。
    pub client: ClientConfig,

    /// goodput 的 SLO 约束。
    pub slo: Slo,

    /// 对照实验的轮数。
    pub rounds: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3160".to_string(),
            engine: EngineConfig::default(),
            workload: WorkloadConfig::default(),
            client: ClientConfig::default(),
            slo: Slo::default(),
            // ⚠️ 默认 3 轮 —— 单轮数字会被抖动污染。
            rounds: 3,
        }
    }
}

impl AppConfig {
    /// 从环境变量加载。
    ///
    /// 变量清单见各子模块的 `from_env`（`EngineConfig` / `WorkloadConfig` / `ClientConfig`）。
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Some(v) = env_str("VLLM_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_usize("VLLM_ROUNDS") {
            config.rounds = v.clamp(1, 20);
        }
        if let (Some(ttft), Some(tpot)) = (env_f64("VLLM_SLO_TTFT_MS"), env_f64("VLLM_SLO_TPOT_MS"))
        {
            config.slo = Slo::new(ttft, tpot);
        }

        config.engine = EngineConfig::from_env();
        config.workload = WorkloadConfig::from_env();
        config.client = ClientConfig::from_env();
        config
    }

    /// 请求超时。
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(600)
    }

    /// 配置摘要（启动时打印）。
    pub fn summary(&self) -> String {
        format!(
            "监听地址    : {}\n\
             模型服务    : {}\n\
             并发上限    : {}（max_num_seqs）\n\
             批预算      : {} token（max_num_batched_tokens）\n\
             chunked 预填充: {}\n\
             请求条数    : {}　到达模型：{}\n\
             长度分布    : {}（输入 {}~{}，输出 {}~{}）\n\
             对照轮数    : {}　SLO：TTFT<={}ms，TPOT<={}ms",
            self.bind_addr,
            self.client.base_url,
            self.engine.max_num_seqs,
            self.engine.max_num_batched_tokens,
            if self.engine.enable_chunked_prefill {
                "开启"
            } else {
                "关闭"
            },
            self.workload.num_requests,
            self.workload.arrival_model.label(),
            self.workload.length_distribution.label(),
            self.workload.min_input_tokens,
            self.workload.max_input_tokens,
            self.workload.min_output_tokens,
            self.workload.max_output_tokens,
            self.rounds,
            self.slo.ttft_ms,
            self.slo.tpot_ms,
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

/// 读一个 `f64` 环境变量。
fn env_f64(key: &str) -> Option<f64> {
    env_str(key).and_then(|v| v.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_port_is_3160() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3160");
    }

    #[test]
    fn default_rounds_is_three() {
        // ⚠️ 单轮数字会被抖动污染 —— 默认必须多轮。
        assert_eq!(AppConfig::default().rounds, 3);
    }

    #[test]
    fn summary_mentions_key_settings() {
        let summary = AppConfig::default().summary();
        assert!(summary.contains("3160"));
        assert!(summary.contains("max_num_seqs"));
        assert!(summary.contains("对照轮数"));
    }
}
