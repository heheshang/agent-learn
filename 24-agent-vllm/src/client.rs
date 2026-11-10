//! vLLM 客户端 —— 打 OpenAI 兼容接口。
//!
//! # 为什么不用 `genai` / `rig`
//!
//! 本仓前面项目（03/07/13…）都用 `genai` 调 DeepSeek。
//! 但本项目的目标是**压测** —— 需要精确控制「什么时候发、并发多少、
//! 发多少条」，而 `genai` 抽象层会挡在中间，且它主要面向「对话式」调用。
//!
//! vLLM 提供的是 **OpenAI 兼容接口**，所以直接用 `reqwest` 打
//! `/v1/chat/completions` 最直接，也和 vLLM 官方
//! `vllm bench serve --backend openai-chat` 的口径一致。
//!
//! # ⚠️ 三个必须注意的细节
//!
//! ## 1. `max_tokens` 必须显式给
//!
//! 不给的话 vLLM 会用模型默认的 `max_model_len`，
//! 可能生成几百个 token —— 压测时间失控。
//! 本项目**按负载配置**逐请求指定 `max_tokens`。
//!
//! ## 2. 要显式关掉采样随机性
//!
//! `temperature: 0` 保证同一批请求的输出长度**可复现**，
//! 否则「token 吞吐」的分母会飘。
//!
//! ## 3. ★ 客户端并发**不等于**服务端批处理
//!
//! 客户端一次发 N 个 HTTP 请求，服务端把它们**合成一个 forward**
//! —— 这是 vLLM 连续批处理的效果。
//! 但如果客户端并发只有 1，服务端就**永远攒不出批**。
//!
//! 所以本项目把「客户端并发度」作为**一个显式的实验变量**：
//! 1 → 8 → 32 → 64，用来观察「并发度 → 服务端批大小 → 吞吐」的传导。

use std::time::{Duration, Instant};

use agent_logging::warn;
use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};

/// 一条压测请求。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchRequest {
    /// 请求 id。
    pub id: usize,

    /// prompt 文本。
    pub prompt: String,

    /// 期望生成的最大 token 数。
    pub max_tokens: usize,
}

/// 一条压测结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchResponse {
    /// 请求 id。
    pub id: usize,

    /// 完整响应文本。
    pub text: String,

    /// prompt 的 token 数（服务端回报）。
    pub prompt_tokens: usize,

    /// 生成的 token 数（服务端回报）。
    pub completion_tokens: usize,

    /// 端到端耗时（毫秒，客户端侧）。
    pub elapsed_ms: f64,

    /// 是否成功。
    pub ok: bool,

    /// 失败原因。
    pub error: Option<String>,
}

impl BenchResponse {
    /// 生成 token / 秒。
    pub fn output_token_throughput(&self) -> f64 {
        if self.elapsed_ms <= 0.0 {
            return 0.0;
        }
        self.completion_tokens as f64 / (self.elapsed_ms / 1000.0)
    }
}

/// vLLM 客户端配置。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientConfig {
    /// 服务地址（不含 `/v1`）。
    pub base_url: String,

    /// 模型名（vLLM 用 `--served-model-name` 指定）。
    pub model: String,

    /// API key（vLLM 不开鉴权时任意填 `"EMPTY"`）。
    pub api_key: String,

    /// 单请求超时。
    pub timeout: Duration,

    /// 客户端并发度（**实验变量**）。
    pub concurrency: usize,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8000".to_string(),
            model: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            api_key: "EMPTY".to_string(),
            timeout: Duration::from_secs(120),
            concurrency: 32,
        }
    }
}

impl ClientConfig {
    /// 从环境变量加载。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `VLLM_BASE_URL` | 服务地址 | `http://127.0.0.1:8000` |
    /// | `VLLM_MODEL` | 模型名 | `Qwen/Qwen2.5-1.5B-Instruct` |
    /// | `VLLM_API_KEY` | API key | `EMPTY` |
    /// | `VLLM_TIMEOUT_SECS` | 单请求超时（秒） | `120` |
    /// | `VLLM_CONCURRENCY` | 客户端并发度 | `32` |
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Some(v) = env_str("VLLM_BASE_URL") {
            config.base_url = v;
        }
        if let Some(v) = env_str("VLLM_MODEL") {
            config.model = v;
        }
        if let Some(v) = env_str("VLLM_API_KEY") {
            config.api_key = v;
        }
        if let Some(v) = env_usize("VLLM_TIMEOUT_SECS") {
            config.timeout = Duration::from_secs(v.clamp(1, 3600) as u64);
        }
        if let Some(v) = env_usize("VLLM_CONCURRENCY") {
            // ⚠️ clamp 到 ≥ 1 —— 0 会让「一个请求都发不出去」。
            config.concurrency = v.clamp(1, 4096);
        }
        config
    }
}

/// vLLM 的 `/v1/chat/completions` 响应（只取需要的字段）。
#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: usize,
    #[serde(default)]
    completion_tokens: usize,
}

/// vLLM 客户端。
#[derive(Debug, Clone)]
pub struct VllmClient {
    http: reqwest::Client,
    config: ClientConfig,
}

impl VllmClient {
    /// 新建客户端。
    pub fn new(config: ClientConfig) -> AppResult<Self> {
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| AppError::Internal(format!("构造 HTTP 客户端失败: {e}")))?;

        Ok(Self { http, config })
    }

    /// 配置。
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// ★ 健康检查 —— `GET /v1/models`。
    ///
    /// 用来在压测前确认「服务真的起来了」。
    pub async fn health(&self) -> AppResult<Vec<String>> {
        let url = format!("{}/v1/models", self.config.base_url.trim_end_matches('/'));

        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .send()
            .await
            .map_err(|e| AppError::Upstream(format!("连不上 vLLM（{url}）: {e}")))?;

        if !resp.status().is_success() {
            return Err(AppError::Upstream(format!(
                "vLLM 返回 {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            )));
        }

        let body: ModelsResponse = resp
            .json()
            .await
            .map_err(|e| AppError::Upstream(format!("解析 /v1/models 失败: {e}")))?;

        Ok(body.data.into_iter().map(|m| m.id).collect())
    }

    /// ★ 发一条 chat completion 请求。
    pub async fn chat(&self, request: &BenchRequest) -> BenchResponse {
        let started = Instant::now();

        let payload = serde_json::json!({
            "model": self.config.model,
            "messages": [{
                "role": "user",
                "content": request.prompt,
            }],
            "max_tokens": request.max_tokens,
            // ⚠️ 温度 0 → 输出长度可复现 → token 吞吐的分母不飘。
            "temperature": 0.0,
            "stream": false,
        });

        let result = self
            .http
            .post(format!(
                "{}/v1/chat/completions",
                self.config.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.config.api_key)
            .json(&payload)
            .send()
            .await;

        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<ChatCompletionResponse>().await {
                    Ok(body) => {
                        let text = body
                            .choices
                            .first()
                            .map(|c| c.message.content.clone())
                            .unwrap_or_default();
                        let usage = body.usage.unwrap_or_default();

                        // ⚠️ 服务端没回报 usage 时，按「返回的文本」估算
                        //    （英文约 4 字符/token，中文约 1.5 字符/token）。
                        let completion_tokens = if usage.completion_tokens > 0 {
                            usage.completion_tokens
                        } else {
                            estimate_tokens(&text)
                        };
                        let prompt_tokens = if usage.prompt_tokens > 0 {
                            usage.prompt_tokens
                        } else {
                            estimate_tokens(&request.prompt)
                        };

                        BenchResponse {
                            id: request.id,
                            text,
                            prompt_tokens,
                            completion_tokens,
                            elapsed_ms,
                            ok: true,
                            error: None,
                        }
                    }
                    Err(e) => {
                        warn!(request_id = request.id, error = %e, "解析 vLLM 响应失败");
                        BenchResponse {
                            id: request.id,
                            text: String::new(),
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            elapsed_ms,
                            ok: false,
                            error: Some(format!("解析响应失败: {e}")),
                        }
                    }
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                warn!(request_id = request.id, status = %status, "vLLM 返回错误状态");
                BenchResponse {
                    id: request.id,
                    text: String::new(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    elapsed_ms,
                    ok: false,
                    error: Some(format!("HTTP {status}: {}", body.trim())),
                }
            }
            Err(e) => {
                warn!(request_id = request.id, error = %e, "请求 vLLM 失败");
                BenchResponse {
                    id: request.id,
                    text: String::new(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    elapsed_ms,
                    ok: false,
                    error: Some(e.to_string()),
                }
            }
        }
    }
}

/// 估算 token 数（服务端没回报 usage 时的兜底）。
///
/// ## ⚠️ 密度取值
///
/// | 字符类型 | 密度 | 依据 |
/// |---|---|---|
/// | CJK（中日韩） | ~1.5 字符/token | 中文单字常占 1~2 个 token |
/// | 拉丁字母 | ~4 字符/token | BPE 常见词长 |
///
/// ⚠️ **1 个汉字 ≈ 1 个 token** 是**错的** ——
/// 实测中文的 token/字比通常在 1.0~2.0 之间，
/// 用「1 字 1 token」会把中文长度**低估**约 30%。
/// 这个误差只影响「服务端没回 usage 时的兜底估算」，
/// 不影响正确性（vLLM 正常会回报 `usage`）。
pub fn estimate_tokens(text: &str) -> usize {
    let mut cjk = 0usize;
    let mut other = 0usize;
    for ch in text.chars() {
        let code = ch as u32;
        if (0x4E00..=0x9FFF).contains(&code) || (0x3040..=0x30FF).contains(&code) {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    // CJK：ceil(cjk / 1.5) = ceil(cjk * 2 / 3)
    // 其它：ceil(other / 4)
    (cjk * 2).div_ceil(3) + other.div_ceil(4)
}

/// `/v1/models` 的响应。
#[derive(Debug, Clone, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelInfo {
    id: String,
}

/// 读一个 `usize` 环境变量。
fn env_usize(key: &str) -> Option<usize> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个非空环境变量。
fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_base_url_is_localhost_8000() {
        let c = ClientConfig::default();
        assert_eq!(c.base_url, "http://127.0.0.1:8000");
        assert_eq!(c.concurrency, 32);
    }

    #[test]
    fn estimate_tokens_counts_cjk_denser_than_latin() {
        // 中文密度更高 → 同样字符数，中文的 token 数更多。
        assert!(estimate_tokens("中文中文中文中文") > estimate_tokens("abcdabcdabcdabcd"));
    }

    #[test]
    fn estimate_tokens_of_empty_is_zero() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn response_throughput_is_zero_for_zero_elapsed() {
        let r = BenchResponse {
            id: 0,
            text: String::new(),
            prompt_tokens: 0,
            completion_tokens: 100,
            elapsed_ms: 0.0,
            ok: true,
            error: None,
        };
        assert_eq!(r.output_token_throughput(), 0.0);
    }

    #[test]
    fn response_throughput_computes_correctly() {
        let r = BenchResponse {
            id: 0,
            text: String::new(),
            prompt_tokens: 0,
            completion_tokens: 100,
            elapsed_ms: 1000.0,
            ok: true,
            error: None,
        };
        assert!((r.output_token_throughput() - 100.0).abs() < 1e-9);
    }
}
