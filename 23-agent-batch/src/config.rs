//! 全局配置。
//!
//! 环境变量前缀用 `BATCH_`（Batch），
//! 避免和其它项目撞名（13 用 `AGENT_`，14 用 `TOOL_`，15 用 `SQL_`，
//! 16 用 `FC_`，17 用 `MEM_`，18 用 `ERR_`，19 用 `SUM_`，
//! 21 用 `CACHE_`，22 用 `ASYNC_`）。

use std::time::Duration;

use crate::batching::BatchConfig;

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 嵌入模型名（fastembed 的模型枚举名）。
    pub embed_model: String,

    /// 重排模型名。
    pub rerank_model: String,

    /// 批处理配置。
    pub batch: BatchConfig,

    /// 对照实验的样本条数。
    pub sample_size: usize,

    /// 对照实验重复轮数（取中位数，降低抖动影响）。
    pub rounds: usize,

    /// 是否启用远程对照（需要 Ollama 服务）。
    pub enable_remote: bool,

    /// 远程 Ollama 地址。
    pub ollama_endpoint: String,

    /// 远程 Ollama 模型。
    pub ollama_model: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3150".to_string(),
            embed_model: "multilingual-e5-small".to_string(),
            rerank_model: "bge-reranker-base".to_string(),
            batch: BatchConfig::default(),
            sample_size: 64,
            // ⚠️ 默认 3 轮 —— 单轮的数字会被抖动污染，
            //    多轮取中位数才是可复现的结论。
            rounds: 3,
            enable_remote: false,
            ollama_endpoint: "http://127.0.0.1:11434".to_string(),
            ollama_model: "qwen3-embedding:0.6b".to_string(),
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `BATCH_BIND_ADDR` | 主服务监听地址 | `127.0.0.1:3150` |
    /// | `BATCH_EMBED_MODEL` | 嵌入模型 | `multilingual-e5-small` |
    /// | `BATCH_RERANK_MODEL` | 重排模型 | `bge-reranker-base` |
    /// | `BATCH_SIZE` | 批大小 | `32` |
    /// | `BATCH_BUCKET_SIZE` | 分桶批大小 | `32` |
    /// | `BATCH_MAX_WAIT_MS` | 最大等待时间（毫秒） | `50` |
    /// | `BATCH_MAX_QUEUE_LEN` | 队列上限 | `1024` |
    /// | `BATCH_MAX_CONCURRENCY` | 客户端并行上限 | `8` |
    /// | `BATCH_SAMPLE_SIZE` | 对照实验样本条数 | `64` |
    /// | `BATCH_ROUNDS` | 对照实验轮数 | `3` |
    /// | `BATCH_REMOTE` | 是否启用远程对照 | `false` |
    /// | `BATCH_OLLAMA_ENDPOINT` | Ollama 地址 | `http://127.0.0.1:11434` |
    /// | `BATCH_OLLAMA_MODEL` | Ollama 模型 | `qwen3-embedding:0.6b` |
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Some(v) = env_str("BATCH_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("BATCH_EMBED_MODEL") {
            config.embed_model = v;
        }
        if let Some(v) = env_str("BATCH_RERANK_MODEL") {
            config.rerank_model = v;
        }
        if let Some(v) = env_usize("BATCH_SAMPLE_SIZE") {
            // ⚠️ clamp 到 ≥ 1 —— 0 条样本没法做对照。
            config.sample_size = v.clamp(1, 10_000);
        }
        if let Some(v) = env_usize("BATCH_ROUNDS") {
            config.rounds = v.clamp(1, 20);
        }
        if let Some(v) = env_bool("BATCH_REMOTE") {
            config.enable_remote = v;
        }
        if let Some(v) = env_str("BATCH_OLLAMA_ENDPOINT") {
            config.ollama_endpoint = v;
        }
        if let Some(v) = env_str("BATCH_OLLAMA_MODEL") {
            config.ollama_model = v;
        }

        config.batch = BatchConfig::from_env();
        config
    }

    /// 请求超时。
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(600)
    }

    /// 配置摘要（启动时打印）。
    pub fn summary(&self) -> String {
        format!(
            "监听地址  : {}\n\
             嵌入模型  : {}\n\
             重排模型  : {}\n\
             批大小    : {}（分桶批大小 {}）\n\
             最大等待  : {} ms\n\
             队列上限  : {}\n\
             样本条数  : {}　对照轮数 {}",
            self.bind_addr,
            self.embed_model,
            self.rerank_model,
            self.batch.batch_size,
            self.batch.bucket_batch_size,
            self.batch.max_wait_ms,
            self.batch.max_queue_len,
            self.sample_size,
            self.rounds,
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
    fn default_port_is_3150() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3150");
    }

    #[test]
    fn default_rounds_is_three() {
        // ⚠️ 单轮数字会被抖动污染 —— 默认必须多轮。
        assert_eq!(AppConfig::default().rounds, 3);
    }

    #[test]
    fn summary_mentions_key_settings() {
        let summary = AppConfig::default().summary();
        assert!(summary.contains("3150"));
        assert!(summary.contains("批大小"));
        assert!(summary.contains("对照轮数"));
    }
}
