//! API 的请求 / 响应模型。

use serde::{Deserialize, Serialize};

use crate::metrics::Slo;
use crate::scheduler::BatchingMode;
use crate::workload::{ArrivalModel, LengthDistribution};

/// 跑一次模拟的请求。
#[derive(Debug, Clone, Deserialize)]
pub struct SimulateRequest {
    /// 批处理模式。
    #[serde(default = "default_mode")]
    pub mode: BatchingMode,

    /// 请求条数。
    #[serde(default = "default_num_requests")]
    pub num_requests: usize,

    /// 到达模型。
    #[serde(default = "default_arrival_model")]
    pub arrival_model: ArrivalModel,

    /// 平均到达速率（请求/秒）。
    #[serde(default = "default_request_rate")]
    pub request_rate: f64,

    /// 长度分布。
    #[serde(default = "default_length_dist")]
    pub length_distribution: LengthDistribution,

    /// 最短输入长度。
    #[serde(default = "default_min_input")]
    pub min_input_tokens: usize,

    /// 最长输入长度。
    #[serde(default = "default_max_input")]
    pub max_input_tokens: usize,

    /// 最短输出长度。
    #[serde(default = "default_min_output")]
    pub min_output_tokens: usize,

    /// 最长输出长度。
    #[serde(default = "default_max_output")]
    pub max_output_tokens: usize,

    /// 服务端同时处理的序列数上限。
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: usize,

    /// 单个 forward 的 token 预算。
    #[serde(default = "default_max_batched_tokens")]
    pub max_num_batched_tokens: usize,

    /// 是否启用 chunked prefill。
    #[serde(default = "default_true")]
    pub enable_chunked_prefill: bool,

    /// 随机种子。
    #[serde(default = "default_seed")]
    pub seed: u64,
}

/// 跑一组对照实验的请求。
#[derive(Debug, Clone, Deserialize)]
pub struct BenchRequest {
    /// 跑哪几组（`a` / `b` / `c`，空 = 全跑）。
    #[serde(default)]
    pub experiments: Vec<String>,

    /// 请求条数。
    #[serde(default = "default_num_requests")]
    pub num_requests: usize,

    /// 到达模型。
    #[serde(default = "default_arrival_model")]
    pub arrival_model: ArrivalModel,

    /// 平均到达速率（请求/秒）。
    #[serde(default = "default_request_rate")]
    pub request_rate: f64,

    /// 长度分布。
    #[serde(default = "default_length_dist")]
    pub length_distribution: LengthDistribution,

    /// 最短输入长度。
    #[serde(default = "default_min_input")]
    pub min_input_tokens: usize,

    /// 最长输入长度。
    #[serde(default = "default_max_input")]
    pub max_input_tokens: usize,

    /// 最短输出长度。
    #[serde(default = "default_min_output")]
    pub min_output_tokens: usize,

    /// 最长输出长度。
    #[serde(default = "default_max_output")]
    pub max_output_tokens: usize,

    /// 每档跑几轮。
    #[serde(default = "default_rounds")]
    pub rounds: usize,

    /// goodput 的 TTFT 上限（毫秒）。
    #[serde(default = "default_slo_ttft")]
    pub slo_ttft_ms: f64,

    /// goodput 的 TPOT 上限（毫秒）。
    #[serde(default = "default_slo_tpot")]
    pub slo_tpot_ms: f64,
}

impl BenchRequest {
    /// 转成 SLO。
    pub fn slo(&self) -> Slo {
        Slo::new(self.slo_ttft_ms, self.slo_tpot_ms)
    }

    /// 是否要跑某组实验。
    ///
    /// ⚠️ 空列表 = 全跑。
    pub fn wants(&self, key: &str) -> bool {
        self.experiments.is_empty() || self.experiments.iter().any(|e| e.eq_ignore_ascii_case(key))
    }
}

/// 统一的错误响应。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// 稳定错误码。
    pub code: String,

    /// 错误信息。
    pub message: String,
}

/// 服务信息。
#[derive(Debug, Clone, Serialize)]
pub struct InfoResponse {
    /// 服务名。
    pub service: String,

    /// 学习主题。
    pub topic: String,

    /// 参考文档。
    pub reference: String,

    /// 替代方案对比。
    pub alternatives: Vec<String>,

    /// 可用的批处理模式。
    pub modes: Vec<ModeInfo>,

    /// 环境变量清单。
    pub env_vars: Vec<EnvVarInfo>,
}

/// 一种批处理模式的说明。
#[derive(Debug, Clone, Serialize)]
pub struct ModeInfo {
    /// 机器名（`static` / `continuous`）。
    pub name: String,

    /// 中文标签。
    pub label: String,

    /// 说明。
    pub description: String,
}

/// 一个环境变量的说明。
#[derive(Debug, Clone, Serialize)]
pub struct EnvVarInfo {
    /// 变量名。
    pub name: String,

    /// 说明。
    pub description: String,

    /// 默认值。
    pub default: String,
}

/// 默认批处理模式。
fn default_mode() -> BatchingMode {
    BatchingMode::Continuous
}

/// 默认请求条数。
fn default_num_requests() -> usize {
    100
}

/// 默认到达模型。
fn default_arrival_model() -> ArrivalModel {
    ArrivalModel::Poisson
}

/// 默认到达速率。
fn default_request_rate() -> f64 {
    20.0
}

/// 默认长度分布。
fn default_length_dist() -> LengthDistribution {
    LengthDistribution::Lognormal
}

/// 默认最短输入长度。
fn default_min_input() -> usize {
    32
}

/// 默认最长输入长度。
fn default_max_input() -> usize {
    512
}

/// 默认最短输出长度。
fn default_min_output() -> usize {
    16
}

/// 默认最长输出长度。
fn default_max_output() -> usize {
    256
}

/// 默认并发上限。
fn default_max_num_seqs() -> usize {
    16
}

/// 默认批预算。
fn default_max_batched_tokens() -> usize {
    2048
}

/// 默认种子。
fn default_seed() -> u64 {
    42
}

/// 默认轮数。
fn default_rounds() -> usize {
    3
}

/// 默认 SLO TTFT。
fn default_slo_ttft() -> f64 {
    500.0
}

/// 默认 SLO TPOT。
fn default_slo_tpot() -> f64 {
    50.0
}

/// 默认 true。
fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_request_defaults_to_running_all_experiments() {
        let req = BenchRequest {
            experiments: vec![],
            num_requests: default_num_requests(),
            arrival_model: default_arrival_model(),
            request_rate: default_request_rate(),
            length_distribution: default_length_dist(),
            min_input_tokens: default_min_input(),
            max_input_tokens: default_max_input(),
            min_output_tokens: default_min_output(),
            max_output_tokens: default_max_output(),
            rounds: default_rounds(),
            slo_ttft_ms: default_slo_ttft(),
            slo_tpot_ms: default_slo_tpot(),
        };
        assert!(req.wants("a"));
        assert!(req.wants("b"));
        assert!(req.wants("c"));
    }

    #[test]
    fn bench_request_can_select_specific_experiments() {
        let mut req = BenchRequest {
            experiments: vec!["a".into()],
            num_requests: default_num_requests(),
            arrival_model: default_arrival_model(),
            request_rate: default_request_rate(),
            length_distribution: default_length_dist(),
            min_input_tokens: default_min_input(),
            max_input_tokens: default_max_input(),
            min_output_tokens: default_min_output(),
            max_output_tokens: default_max_output(),
            rounds: default_rounds(),
            slo_ttft_ms: default_slo_ttft(),
            slo_tpot_ms: default_slo_tpot(),
        };
        assert!(req.wants("A"));
        assert!(!req.wants("b"));

        req.experiments = vec!["B".into()];
        assert!(!req.wants("a"));
        assert!(req.wants("B"));
    }

    #[test]
    fn simulate_request_deserializes_with_defaults() {
        let req: SimulateRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(req.mode, BatchingMode::Continuous);
        assert_eq!(req.num_requests, 100);
        assert!(req.enable_chunked_prefill);
    }
}
