//! 负载生成 —— 构造可复现的请求流。
//!
//! # 三种到达模型
//!
//! | 模型 | 到达间隔 | 对应现实场景 |
//! |---|---|---|
//! | **突发**（burst） | 全部在 t=0 到达 | 离线批量灌库、压测上限 |
//! | **固定速率**（constant） | 固定间隔 | 稳定生产流量 |
//! | **泊松**（poisson） | 指数分布间隔 | **最接近真实用户流量** |
//!
//! ## ⚠️ 为什么泊松模型最真实
//!
//! 真实用户行为是**独立到达**的 —— 每个用户什么时候来互不相关，
//! 于是到达间隔服从**指数分布**（泊松过程的定义）。
//!
//! 这不是学术洁癖，它有实际后果：
//!
//! - **突发**会测出「队列堆积」的最坏情况，TTFT 尖刺很夸张
//! - **固定速率**太平滑，测不出排队效应
//! - **泊松**既有平均速率，又有随机性，**尾延迟**最真实
//!
//! vLLM 官方 `vllm bench serve` 的 `--request-rate` 就是**泊松**，
//! `--burstiness` 控制方差（gamma 分布）。
//!
//! # ⚠️ 随机性必须可复现
//!
//! 每次跑出来的负载不一样，对照实验就**无法归因** ——
//! 「差值到底是来自变量还是来自随机抖动」。
//!
//! 所以本模块用**确定性伪随机数**（xorshift + 固定种子），
//! 同一种子 → 同一份负载 → 对照实验**变量单一**。

use serde::{Deserialize, Serialize};

use crate::scheduler::{Arrival, RequestSpec};

/// 到达模型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArrivalModel {
    /// 全部在 t=0 到达（压吞吐上限）。
    Burst,

    /// 固定间隔到达。
    Constant,

    /// 指数分布间隔（**最接近真实流量**）。
    Poisson,
}

impl ArrivalModel {
    /// 人类可读名。
    pub fn label(&self) -> &'static str {
        match self {
            ArrivalModel::Burst => "突发",
            ArrivalModel::Constant => "固定速率",
            ArrivalModel::Poisson => "泊松",
        }
    }

    /// 解析（从字符串）。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "burst" => Some(ArrivalModel::Burst),
            "constant" => Some(ArrivalModel::Constant),
            "poisson" => Some(ArrivalModel::Poisson),
            _ => None,
        }
    }
}

/// 输入长度分布。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LengthDistribution {
    /// 所有请求一样长。
    Fixed,

    /// 长度在 `[min, max]` 上**均匀**分布。
    Uniform,

    /// **对数正态**分布（真实场景的输入长度分布）。
    ///
    /// ⚠️ 为什么真实流量是对数正态：大多数请求很短，
    /// 少数请求很长。均匀分布会**低估**长尾带来的影响。
    Lognormal,
}

impl LengthDistribution {
    /// 人类可读名。
    pub fn label(&self) -> &'static str {
        match self {
            LengthDistribution::Fixed => "固定长度",
            LengthDistribution::Uniform => "均匀分布",
            LengthDistribution::Lognormal => "对数正态",
        }
    }

    /// 解析（从字符串）。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "fixed" => Some(LengthDistribution::Fixed),
            "uniform" => Some(LengthDistribution::Uniform),
            "lognormal" => Some(LengthDistribution::Lognormal),
            _ => None,
        }
    }
}

/// 负载配置。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorkloadConfig {
    /// 请求条数。
    pub num_requests: usize,

    /// 到达模型。
    pub arrival_model: ArrivalModel,

    /// 平均到达速率（请求/秒）—— 只对 `Constant` / `Poisson` 有效。
    pub request_rate: f64,

    /// 输入长度分布。
    pub length_distribution: LengthDistribution,

    /// 最短输入长度（token）。
    pub min_input_tokens: usize,

    /// 最长输入长度（token）。
    pub max_input_tokens: usize,

    /// 最短输出长度（token）。
    pub min_output_tokens: usize,

    /// 最长输出长度（token）。
    pub max_output_tokens: usize,

    /// 随机种子（**保证可复现**）。
    pub seed: u64,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            num_requests: 100,
            arrival_model: ArrivalModel::Poisson,
            request_rate: 20.0,
            length_distribution: LengthDistribution::Lognormal,
            min_input_tokens: 32,
            max_input_tokens: 512,
            min_output_tokens: 16,
            max_output_tokens: 256,
            seed: 42,
        }
    }
}

impl WorkloadConfig {
    /// 从环境变量加载。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `VLLM_NUM_REQUESTS` | 请求条数 | `100` |
    /// | `VLLM_ARRIVAL_MODEL` | `burst`/`constant`/`poisson` | `poisson` |
    /// | `VLLM_REQUEST_RATE` | 平均到达速率（请求/秒） | `20` |
    /// | `VLLM_LENGTH_DIST` | `fixed`/`uniform`/`lognormal` | `lognormal` |
    /// | `VLLM_MIN_INPUT_TOKENS` | 最短输入长度 | `32` |
    /// | `VLLM_MAX_INPUT_TOKENS` | 最长输入长度 | `512` |
    /// | `VLLM_MIN_OUTPUT_TOKENS` | 最短输出长度 | `16` |
    /// | `VLLM_MAX_OUTPUT_TOKENS` | 最长输出长度 | `256` |
    /// | `VLLM_SEED` | 随机种子 | `42` |
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Some(v) = env_usize("VLLM_NUM_REQUESTS") {
            config.num_requests = v.clamp(1, 1_000_000);
        }
        if let Some(v) = env_str("VLLM_ARRIVAL_MODEL")
            && let Some(model) = ArrivalModel::parse(v.as_str())
        {
            config.arrival_model = model;
        }
        if let Some(v) = env_f64("VLLM_REQUEST_RATE") {
            config.request_rate = v.clamp(0.001, 1_000_000.0);
        }
        if let Some(v) = env_str("VLLM_LENGTH_DIST")
            && let Some(dist) = LengthDistribution::parse(v.as_str())
        {
            config.length_distribution = dist;
        }
        if let Some(v) = env_usize("VLLM_MIN_INPUT_TOKENS") {
            config.min_input_tokens = v.max(1);
        }
        if let Some(v) = env_usize("VLLM_MAX_INPUT_TOKENS") {
            config.max_input_tokens = v.max(1);
        }
        if let Some(v) = env_usize("VLLM_MIN_OUTPUT_TOKENS") {
            config.min_output_tokens = v.max(1);
        }
        if let Some(v) = env_usize("VLLM_MAX_OUTPUT_TOKENS") {
            config.max_output_tokens = v.max(1);
        }
        if let Some(v) = env_u64("VLLM_SEED") {
            config.seed = v;
        }

        // ⚠️ min > max 会让区间反转 —— 必须夹紧。
        if config.min_input_tokens > config.max_input_tokens {
            config.min_input_tokens = config.max_input_tokens;
        }
        if config.min_output_tokens > config.max_output_tokens {
            config.min_output_tokens = config.max_output_tokens;
        }

        config
    }
}

/// 确定性伪随机数生成器（xorshift64*）。
///
/// ## 为什么不用 `rand`
///
/// ① **可复现**：同样的种子 → 同样的序列，跨版本稳定
/// ② **零依赖**：不往 workspace 引入新的依赖
/// ③ **够用**：生成负载只需要「均匀分布的随机数」
///
/// ⚠️ 注意：`rand` 的 `SmallRng` / `StdRng` 在**版本升级时可能改变算法**，
/// 而这里的随机性直接决定负载形状 —— 必须锁死。
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// 新建（给定种子）。
    ///
    /// ⚠️ 种子 0 会让 xorshift 永远输出 0 —— 必须避开。
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    /// 下一个 `u64`。
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// 下一个 `[0, 1)` 的浮点数（53 位精度）。
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// 下一个 `[low, high]` 的**整数**。
    ///
    /// ⚠️ `high` 是**闭区间**上界。
    pub fn range_usize(&mut self, low: usize, high: usize) -> usize {
        if high <= low {
            return low;
        }
        let span = (high - low + 1) as u64;
        low + (self.next_u64() % span) as usize
    }
}

/// ★ 生成一份负载。
///
/// ## 关键性质
///
/// 1. **确定性** —— 同 `seed` → 同一份负载
/// 2. **按到达时刻升序** —— 模拟器依赖这个前提
/// 3. **泊松间隔用 `-ln(U)`** —— 指数分布的标准反变换法
pub fn generate(config: &WorkloadConfig) -> Vec<Arrival> {
    let mut rng = Rng::new(config.seed);
    let mut arrivals = Vec::with_capacity(config.num_requests);

    let mean_interval_ms = if config.request_rate > 0.0 {
        1000.0 / config.request_rate
    } else {
        0.0
    };

    let mut now_ms = 0.0f64;

    for i in 0..config.num_requests {
        // 1. 到达时刻。
        match config.arrival_model {
            ArrivalModel::Burst => {
                now_ms = 0.0;
            }
            ArrivalModel::Constant => {
                if i > 0 {
                    now_ms += mean_interval_ms;
                }
            }
            ArrivalModel::Poisson => {
                if i > 0 {
                    // ⚠️ 指数分布的反变换：间隔 = -ln(U) / λ。
                    //    λ = request_rate（每秒），换算成毫秒。
                    let u = rng.next_f64().max(f64::MIN_POSITIVE);
                    now_ms += -u.ln() * mean_interval_ms;
                }
            }
        }

        // 2. 输入长度。
        let input_tokens = sample_length(
            &mut rng,
            config.length_distribution,
            config.min_input_tokens,
            config.max_input_tokens,
        );

        // 3. 输出长度。
        let output_tokens = sample_length(
            &mut rng,
            config.length_distribution,
            config.min_output_tokens,
            config.max_output_tokens,
        );

        arrivals.push(Arrival {
            at_ms: now_ms,
            spec: RequestSpec {
                id: i,
                prompt_tokens: input_tokens,
                output_tokens,
            },
        });
    }

    arrivals
}

/// 按分布采样一个长度。
fn sample_length(rng: &mut Rng, dist: LengthDistribution, min: usize, max: usize) -> usize {
    if max <= min {
        return min;
    }
    match dist {
        LengthDistribution::Fixed => min,
        LengthDistribution::Uniform => rng.range_usize(min, max),
        LengthDistribution::Lognormal => {
            // ⚠️ 对数正态：median = exp(mu)，mu = ln(median)。
            //    几何平均值取区间中点，然后 clamp 到 [min, max]。
            let median = ((min as f64) * (max as f64)).sqrt();
            let mu = median.ln();
            // 标准差取「区间宽度的一半」对应的比例。
            let sigma = ((max as f64) / (min as f64)).ln().abs() / 4.0;
            let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
            let u2 = rng.next_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            let value = (mu + sigma * z).exp().round();
            let value = value.clamp(min as f64, max as f64);
            value as usize
        }
    }
}

/// 读一个 `usize` 环境变量。
fn env_usize(key: &str) -> Option<usize> {
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

/// 读一个非空环境变量。
fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic_for_the_same_seed() {
        let config = WorkloadConfig::default();
        let a = generate(&config);
        let b = generate(&config);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.at_ms - y.at_ms).abs() < 1e-9);
            assert_eq!(x.spec.prompt_tokens, y.spec.prompt_tokens);
            assert_eq!(x.spec.output_tokens, y.spec.output_tokens);
        }
    }

    #[test]
    fn different_seeds_give_different_workloads() {
        let mut a = WorkloadConfig::default();
        let mut b = WorkloadConfig::default();
        a.seed = 1;
        b.seed = 2;

        let wa = generate(&a);
        let wb = generate(&b);
        // 泊松模型的到达时刻应该不同。
        assert!(
            wa.iter()
                .zip(wb.iter())
                .any(|(x, y)| (x.at_ms - y.at_ms).abs() > 1e-6)
        );
    }

    #[test]
    fn burst_arrivals_are_all_at_zero() {
        let config = WorkloadConfig {
            arrival_model: ArrivalModel::Burst,
            num_requests: 50,
            ..Default::default()
        };

        let arrivals = generate(&config);
        assert_eq!(arrivals.len(), 50);
        assert!(arrivals.iter().all(|a| a.at_ms == 0.0));
    }

    #[test]
    fn constant_arrivals_have_equal_intervals() {
        let config = WorkloadConfig {
            arrival_model: ArrivalModel::Constant,
            request_rate: 10.0,
            num_requests: 5,
            ..Default::default()
        };

        let arrivals = generate(&config);
        // 10 req/s → 间隔 100ms
        assert!((arrivals[1].at_ms - arrivals[0].at_ms - 100.0).abs() < 1e-6);
        assert!((arrivals[4].at_ms - arrivals[3].at_ms - 100.0).abs() < 1e-6);
    }

    #[test]
    fn poisson_arrivals_are_sorted_and_non_decreasing() {
        let config = WorkloadConfig {
            arrival_model: ArrivalModel::Poisson,
            num_requests: 200,
            ..Default::default()
        };

        let arrivals = generate(&config);
        for w in arrivals.windows(2) {
            assert!(w[1].at_ms >= w[0].at_ms, "到达时刻必须非递减");
        }
    }

    #[test]
    fn poisson_intervals_vary() {
        // ⚠️ 泊松的间隔应该**不等** —— 全相等就退化成固定速率了。
        let config = WorkloadConfig {
            arrival_model: ArrivalModel::Poisson,
            num_requests: 50,
            ..Default::default()
        };

        let arrivals = generate(&config);
        let intervals: Vec<f64> = arrivals
            .windows(2)
            .map(|w| w[1].at_ms - w[0].at_ms)
            .collect();
        let min = intervals.iter().cloned().fold(f64::MAX, f64::min);
        let max = intervals.iter().cloned().fold(f64::MIN, f64::max);
        assert!(max > min * 1.5, "泊松间隔应有明显差异：{min} ~ {max}");
    }

    #[test]
    fn fixed_length_distribution_gives_identical_lengths() {
        let config = WorkloadConfig {
            length_distribution: LengthDistribution::Fixed,
            num_requests: 20,
            ..Default::default()
        };

        let arrivals = generate(&config);
        let first = arrivals[0].spec.prompt_tokens;
        assert!(arrivals.iter().all(|a| a.spec.prompt_tokens == first));
    }

    #[test]
    fn lengths_stay_within_bounds() {
        let config = WorkloadConfig {
            length_distribution: LengthDistribution::Lognormal,
            min_input_tokens: 10,
            max_input_tokens: 100,
            min_output_tokens: 5,
            max_output_tokens: 50,
            num_requests: 500,
            ..Default::default()
        };

        for a in generate(&config) {
            assert!(
                (10..=100).contains(&a.spec.prompt_tokens),
                "输入长度越界: {}",
                a.spec.prompt_tokens
            );
            assert!(
                (5..=50).contains(&a.spec.output_tokens),
                "输出长度越界: {}",
                a.spec.output_tokens
            );
        }
    }

    #[test]
    fn rng_is_reproducible() {
        let mut a = Rng::new(12345);
        let mut b = Rng::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn rng_seed_zero_does_not_collapse() {
        let mut rng = Rng::new(0);
        let first = rng.next_u64();
        let second = rng.next_u64();
        assert_ne!(first, second, "种子 0 时 xorshift 会退化成常数");
        assert_ne!(first, 0);
    }

    #[test]
    fn rng_range_is_inclusive_and_bounded() {
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let v = rng.range_usize(3, 7);
            assert!((3..=7).contains(&v));
        }
    }

    #[test]
    fn rng_range_with_equal_bounds_returns_that_bound() {
        let mut rng = Rng::new(7);
        assert_eq!(rng.range_usize(5, 5), 5);
    }

    #[test]
    fn lognormal_has_a_long_tail() {
        // ⚠️ 对数正态的关键特征：大部分短，少数长。
        let config = WorkloadConfig {
            length_distribution: LengthDistribution::Lognormal,
            min_input_tokens: 8,
            max_input_tokens: 1024,
            num_requests: 1000,
            ..Default::default()
        };

        let lengths: Vec<usize> = generate(&config)
            .iter()
            .map(|a| a.spec.prompt_tokens)
            .collect();
        let mean = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
        let max = lengths.iter().copied().max().unwrap();

        // 最大值应明显高于均值（长尾）。
        assert!(max as f64 > mean * 2.0, "长尾不明显：mean={mean} max={max}");
    }
}
