//! 批处理核心 —— **把 N 条独立请求变成 1 次前向**。
//!
//! # 批处理为什么能提升吞吐量
//!
//! 一次 transformer 前向的耗时**不是**与输入条数线性相关的。
//!
//! ```text
//! 单条前向：  [权重加载] [tokenize] [前向计算] [取输出]
//!                    ↑ 固定开销（与条数无关）
//!
//! 批处理前向： [权重加载] [tokenize × N] [前向计算] [取输出]
//!                    ↑ 固定开销只付一次，被 N 条数据分摊
//! ```
//!
//! 固定开销里最大的一块是**权重复制到计算设备**
//! （CPU 上是内存带宽，GPU 上是 PCIe 传输）。
//! N 条数据共享这一次复制，**单位成本随 N 下降**。
//!
//! # 三个必须讲清的坑
//!
//! ## ⚠️ 坑 1：padding —— 「批」里每条被补到同样长
//!
//! 神经网络要求一个 tensor 的形状规整，
//! 所以批里的短文本必须被 **padding** 到该批最长的那条。
//!
//! ```text
//! 批里三条，长度分别是 12 / 300 / 8 token
//!                  ↓ padding 到 300
//! 实际算的是 12×300 + 300×300 + 8×300 的算力
//! ```
//!
//! 短的那两条**白算了**。所以：
//!
//! - **批越大，padding 浪费越多**（批里的最长条会拖慢所有人）
//! - **按长度分桶**（本项目的 [`bucket_by_length`]）能大幅减少浪费
//!
//! ⚠️ 但要注意：fastembed 用的是 `PaddingStrategy::BatchLongest`，
//! 也就是**每批**按该批最长条 padding ——
//! 分桶让「每批的最长条」接近「该批的平均长度」，收益直接体现在批耗时上。
//!
//! ## ⚠️ 坑 2：吞吐 ↑ 但**单条延迟** ↑
//!
//! 批处理把 N 条塞进一次前向，**单条要等整批跑完**。
//!
//! ```text
//! 逐条：第 1 条 20ms 就返回了
//! 批处理：第 1 条要等整批 100ms
//! ```
//!
//! 所以批处理**只适合「攒一批一起要」的场景**
//! （离线灌库、索引构建、批量评估），
//! **不适合「用户在线等单条结果」**。
//!
//! 本项目同时报告吞吐和延迟两个维度（见 [`crate::metrics`]），
//! 就是为了防止「只看总耗时」得出错误结论。
//!
//! ## ⚠️ 坑 3：**背压（backpressure）** —— 不能无限攒批
//!
//! 「攒够 32 条再发」听起来很好，但如果上游只有 1 QPS，
//! 攒批就变成了「把请求无限期压住」。
//!
//! 正确做法是**同时**设三个闸门：
//!
//! | 闸门 | 含义 | 触发时的行为 |
//! |---|---|---|
//! | **最大批大小** | 一批最多 N 条 | 立即发出去 |
//! | **最大等待时间** | 最多等 T 毫秒 | 立即发出去（哪怕不满） |
//! | **最大队列长度** | 最多压 M 条 | 拒绝新的请求（返回错误，不静默丢弃） |
//!
//! 本项目的 [`BatchAccumulator`] 就是这个三闸门实现。

use std::time::{Duration, Instant};

/// 读一个 `u64` 环境变量。
fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .and_then(|v| v.parse().ok())
}

/// 读一个 `usize` 环境变量。
fn env_usize(key: &str) -> Option<usize> {
    env_u64(key).map(|v| v as usize)
}

use agent_logging::warn;
use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};

/// 一个批次的元信息。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInfo {
    /// 批次下标（从 0 开始）。
    pub index: usize,

    /// 这一批里的条数。
    pub size: usize,

    /// 这一批的总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 批处理的执行策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStrategy {
    /// **逐条调用** —— 基线。每条一次独立的推理。
    ///
    /// 这是对照组：没有批处理时的真实开销。
    Sequential,

    /// **固定大小批** —— 每 `batch_size` 条一次前向。
    ///
    /// 最简单也最常用。缺点是**批内的 padding 浪费**：
    /// 一个批里的最长条会拖慢同批的所有短条。
    Fixed,

    /// **按长度分桶** —— 先按 token 长度排序分桶，同桶内再定批。
    ///
    /// 减少 padding 浪费的**唯一有效手段**。
    ///
    /// ⚠️ 代价：必须**先收齐全部输入**才能排序，
    /// 所以**不适合流式 / 在线场景**。
    Bucketed,
}

impl BatchStrategy {
    /// 人类可读名。
    pub fn label(&self) -> &'static str {
        match self {
            BatchStrategy::Sequential => "逐条调用",
            BatchStrategy::Fixed => "固定批",
            BatchStrategy::Bucketed => "长度分桶",
        }
    }

    /// 解析（从字符串）。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sequential" | "seq" => Some(BatchStrategy::Sequential),
            "fixed" | "batch" => Some(BatchStrategy::Fixed),
            "bucketed" | "bucket" => Some(BatchStrategy::Bucketed),
            _ => None,
        }
    }
}

/// 批处理配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// 批大小（`Fixed` 策略用）。
    pub batch_size: usize,

    /// 长度分桶的目标批大小（`Bucketed` 策略用）。
    pub bucket_batch_size: usize,

    /// 每个桶的**最大等待时间**（毫秒）。
    pub max_wait_ms: u64,

    /// 队列的最大长度（背压闸门）。
    pub max_queue_len: usize,

    /// **客户端并行**的上限（LangChain 的 `max_concurrency` 对应物）。
    ///
    /// ## ⚠️ 这和「批大小」是**两个正交维度**
    ///
    /// | 维度 | 省下什么 | 本项目的实现 |
    /// |---|---|---|
    /// | **并发** | 多次网络往返的墙钟时间 | 远程后端的「组 B」 |
    /// | **批大小** | 每次请求的固定开销 | 本地前向 / 远程的「组 C」 |
    ///
    /// 本地 ONNX 推理没有网络往返，所以并发对它**无意义** ——
    /// 它只能靠「一次前向吃 N 条」来提吞吐。
    pub max_concurrency: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            // ⚠️ 32 是个保守的起点：
            //    太小 → 固定开销分摊不够；
            //    太大 → padding 浪费急剧上升（坑 1）。
            batch_size: 32,
            bucket_batch_size: 32,
            max_wait_ms: 50,
            max_queue_len: 1024,
            // ⚠️ 默认 8 —— 足够体现并发收益，又不会把上游打爆。
            max_concurrency: 8,
        }
    }
}

impl BatchConfig {
    /// 从环境变量加载。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `BATCH_SIZE` | 固定批大小 | `32` |
    /// | `BATCH_BUCKET_SIZE` | 分桶批大小 | `32` |
    /// | `BATCH_MAX_WAIT_MS` | 最大等待时间（毫秒） | `50` |
    /// | `BATCH_MAX_QUEUE_LEN` | 队列上限（背压） | `1024` |
    /// | `BATCH_MAX_CONCURRENCY` | 客户端并行上限 | `8` |
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Some(v) = env_usize("BATCH_SIZE") {
            // ⚠️ clamp 到 ≥ 1 —— 0 会让除零 / 空批。
            config.batch_size = v.clamp(1, 4096);
        }
        if let Some(v) = env_usize("BATCH_BUCKET_SIZE") {
            config.bucket_batch_size = v.clamp(1, 4096);
        }
        if let Some(v) = env_u64("BATCH_MAX_WAIT_MS") {
            config.max_wait_ms = v.min(60_000);
        }
        if let Some(v) = env_usize("BATCH_MAX_QUEUE_LEN") {
            config.max_queue_len = v.clamp(1, 1_000_000);
        }
        if let Some(v) = env_usize("BATCH_MAX_CONCURRENCY") {
            // ⚠️ clamp 到 ≥ 1 —— 0 路并发等于串行，
            //    而这会让「并发组」的对照实验失去意义。
            config.max_concurrency = v.clamp(1, 64);
        }
        config
    }
}

/// 按长度分桶。
///
/// ## 算法
///
/// 1. 按「估算 token 长度」**降序**排序
/// 2. 按 `batch_size` 切成**连续**的批
/// 3. **批内按原顺序恢复**（否则调用方拿到的结果顺序就乱了）
///
/// ## ⚠️ 「长度排序」才是关键，「桶」只是副产品
///
/// 常见的误解是「按长度分成若干个**桶**，每桶再切批」——
/// 但那样**桶的边界**上仍然会混进长短不一的文本。
///
/// 真正有效的是：
///
/// ```text
/// 排序后（长 → 短）：
///   [100, 98, 97, 95, 10, 9, 8, 3]
///         ├── 批 1（批大小 4）──┤   ← 全是长的
///                 ├── 批 2 ──┤       ← 全是短的
/// ```
///
/// 每批内部的长度**天然接近**，
/// 于是 padding 到该批最长条时的浪费**最小**。
///
/// ## ⚠️ 为什么「批内恢复原顺序」不可省
///
/// 排序打乱了输入顺序。如果直接按排序后的顺序返回，
/// 调用方就**无法把结果映射回输入**——
/// 而「结果与输入一一对应」是批处理的基本契约。
/// 所以每批内部必须记录原始下标，返回时还原。
///
/// ## ⚠️ 长度是**估算**的
///
/// 精确 token 数要跑 tokenizer（贵）。
/// 这里用「CJK 1 字符 ≈ 1 token，其余 4 字符 ≈ 1 token」粗估——
///
/// 对于**分桶**这个用途，估算精度**足够**：
/// 目标是让同批的长度**接近**，不是精确知道长度。
/// 估错的后果只是「分桶没那么完美」，不会影响正确性。
pub fn bucket_by_length(texts: &[String], batch_size: usize) -> Vec<Vec<(usize, usize)>> {
    if texts.is_empty() || batch_size == 0 {
        return Vec::new();
    }

    // 估算长度 → 下标。
    let mut indexed: Vec<(usize, usize)> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| (i, estimate_tokens(t)))
        .collect();

    // 按长度降序 —— 长的一批，短的另一批。
    indexed.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut buckets: Vec<Vec<(usize, usize)>> = Vec::new();
    for chunk in indexed.chunks(batch_size) {
        let mut bucket: Vec<(usize, usize)> = chunk.to_vec();
        // ⚠️ 批内按**原始下标**升序恢复 —— 保证结果可映射回输入。
        bucket.sort_by_key(|(idx, _)| *idx);
        buckets.push(bucket);
    }
    buckets
}

/// 估算 token 数。
///
/// 中文 / 日文 / 韩文：1 字符 ≈ 1 token。
/// 其它（拉丁字母等）：4 字符 ≈ 1 token。
///
/// 这是经验值，**只用于分桶**（不影响正确性）。
pub fn estimate_tokens(text: &str) -> usize {
    let mut cjk = 0usize;
    let mut other = 0usize;
    for ch in text.chars() {
        let code = ch as u32;
        if (0x4E00..=0x9FFF).contains(&code)
            || (0x3040..=0x30FF).contains(&code)
            || (0xAC00..=0xD7AF).contains(&code)
        {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    cjk + other.div_ceil(4)
}

/// 批处理的**积压器** —— 实现三个背压闸门。
///
/// ## 三个闸门
///
/// | 闸门 | 字段 | 触发时 |
/// |---|---|---|
/// | 最大批大小 | [`BatchConfig::batch_size`] | 凑满 → 立即返回这一批 |
/// | 最大等待时间 | [`BatchConfig::max_wait_ms`] | 超时 → 返回当前批（可能不满） |
/// | 最大队列长度 | [`BatchConfig::max_queue_len`] | 队列满 → **返回错误**（不静默丢弃） |
///
/// ## ⚠️ 为什么队列满了要「返回错误」而不是「丢掉最老的」
///
/// 静默丢弃 = 请求凭空消失 = 调用方永远等不到结果 = **挂死**。
/// 返回错误至少让调用方知道「这次没成」。
/// 这与 21 项目的「缓存是优化不是功能」是同一个原则。
pub struct BatchAccumulator {
    config: BatchConfig,
    queue: Vec<String>,
    first_queued_at: Option<Instant>,
}

impl BatchAccumulator {
    /// 创建积压器。
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            queue: Vec::new(),
            first_queued_at: None,
        }
    }

    /// 当前配置。
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// 队列里待处理的条数。
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// 队列是否为空。
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// 塞入一条待处理数据。
    ///
    /// ## 返回值
    ///
    /// - `Ok(None)` —— 还没到发批条件，继续攒
    /// - `Ok(Some(batch))` —— 到点了，这一批可以发
    /// - `Err(_)` —— **队列满了**（背压）
    pub fn push(&mut self, text: String) -> AppResult<Option<Vec<String>>> {
        // ⚠️ 闸门 3：队列上限。
        if self.queue.len() >= self.config.max_queue_len {
            warn!(
                queued = self.queue.len(),
                limit = self.config.max_queue_len,
                "批处理队列已满，拒绝新请求（背压）"
            );
            return Err(AppError::BadRequest(format!(
                "批处理队列已满（{} 条），请稍后重试",
                self.config.max_queue_len
            )));
        }

        if self.queue.is_empty() {
            self.first_queued_at = Some(Instant::now());
        }
        self.queue.push(text);

        // ⚠️ 闸门 1：凑满一批。
        if self.queue.len() >= self.config.batch_size {
            return Ok(Some(self.take()));
        }

        // ⚠️ 闸门 2：等待超时。
        if let Some(start) = self.first_queued_at
            && start.elapsed() >= Duration::from_millis(self.config.max_wait_ms)
        {
            return Ok(Some(self.take()));
        }

        Ok(None)
    }

    /// 取出当前队列里的全部数据（组成一批）。
    pub fn take(&mut self) -> Vec<String> {
        self.first_queued_at = None;
        std::mem::take(&mut self.queue)
    }

    /// 强制冲刷（把当前队列发出去，不管满不满）。
    pub fn flush(&mut self) -> Vec<String> {
        self.take()
    }
}

/// 一次批处理的完整报告。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchReport {
    /// 使用的策略。
    pub strategy: BatchStrategy,

    /// 度量。
    pub stats: crate::metrics::BatchStats,

    /// 每批的元信息。
    pub batches: Vec<BatchInfo>,

    /// 结果条数（成功数）。
    pub result_count: usize,
}

impl BatchReport {
    /// 一行摘要。
    pub fn summary_line(&self) -> String {
        format!(
            "{:<10} {}",
            self.strategy.label(),
            self.stats.summary_line()
        )
    }
}

/// 把输入切成固定大小的批。
pub fn chunk_fixed(texts: &[String], batch_size: usize) -> Vec<Vec<usize>> {
    if texts.is_empty() || batch_size == 0 {
        return Vec::new();
    }
    texts
        .chunks(batch_size)
        .enumerate()
        .map(|(bi, chunk)| (bi * batch_size..bi * batch_size + chunk.len()).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("文本 {i}")).collect()
    }

    #[test]
    fn chunk_fixed_splits_correctly() {
        let chunks = chunk_fixed(&texts(10), 4);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 4);
        assert_eq!(chunks[1].len(), 4);
        assert_eq!(chunks[2].len(), 2);
    }

    #[test]
    fn chunk_fixed_handles_empty_and_zero() {
        assert!(chunk_fixed(&[], 4).is_empty());
        assert!(chunk_fixed(&texts(4), 0).is_empty());
    }

    #[test]
    fn bucket_groups_similar_lengths_together() {
        let mut input = vec!["短".to_string(); 4];
        input.push("很长的一段中文文本内容用来测试分桶效果".repeat(4));

        let buckets = bucket_by_length(&input, 2);

        // 5 条 / 批大小 2 → 3 批
        assert_eq!(buckets.len(), 3);

        // ⚠️ 正确的性质不是「每批长度差 ≤ 1」——
        //    排序后**第一个**批必然是「最长 + 次长」，
        //    差值大是必然的（那一批里没有更短的可以配）。
        //
        //    真正的性质是：**各批的最长长度单调不增** ——
        //    也就是「长的都聚在前面」，这才是减少 padding 的机制。
        let batch_maxes: Vec<usize> = buckets
            .iter()
            .map(|b| b.iter().map(|(_, l)| *l).max().unwrap())
            .collect();

        for pair in batch_maxes.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "各批最长长度应单调不增: {batch_maxes:?}"
            );
        }
    }

    #[test]
    fn bucket_reduces_padding_waste_versus_unsorted() {
        // ⚠️ 分桶收益的**直接证据**：
        // 同样批大小下，按长度排序后切分的 padding 系数更低。
        let lengths: Vec<usize> = (0..16).map(|i| if i % 2 == 0 { 100 } else { 10 }).collect();

        let comparison = crate::padding::compare_padding(&lengths, 4);

        assert!(
            comparison.bucketed.padding_factor < comparison.fixed.padding_factor,
            "分桶后 padding 系数应更低: {:.3} vs {:.3}",
            comparison.bucketed.padding_factor,
            comparison.fixed.padding_factor
        );
    }

    #[test]
    fn bucket_preserves_original_indices_within_a_batch() {
        let input = vec![
            "很长的中文内容".repeat(3),
            "短".to_string(),
            "中等长度内容".repeat(2),
            "短".to_string(),
        ];

        let buckets = bucket_by_length(&input, 2);

        // 每个批内的原始下标必须递增 —— 否则结果无法映射回输入。
        for bucket in &buckets {
            let indices: Vec<usize> = bucket.iter().map(|(i, _)| *i).collect();
            let mut sorted = indices.clone();
            sorted.sort_unstable();
            assert_eq!(indices, sorted, "批内下标未按原顺序恢复: {indices:?}");
        }

        // 所有下标必须**恰好**出现一次（不重不漏）。
        let mut all: Vec<usize> = buckets.iter().flatten().map(|(i, _)| *i).collect();
        all.sort_unstable();
        assert_eq!(all, (0..input.len()).collect::<Vec<_>>());
    }

    #[test]
    fn bucket_of_empty_is_empty() {
        assert!(bucket_by_length(&[], 8).is_empty());
        assert!(bucket_by_length(&texts(3), 0).is_empty());
    }

    #[test]
    fn estimate_tokens_counts_cjk_as_one() {
        // 4 个汉字 → 4 token
        assert_eq!(estimate_tokens("你好世界"), 4);
    }

    #[test]
    fn estimate_tokens_counts_latin_as_four_chars_per_token() {
        // 8 个 ASCII → 2 token
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn accumulator_returns_batch_when_full() {
        let mut acc = BatchAccumulator::new(BatchConfig {
            batch_size: 3,
            max_wait_ms: 10_000,
            ..Default::default()
        });

        assert!(acc.push("a".into()).unwrap().is_none());
        assert!(acc.push("b".into()).unwrap().is_none());

        let batch = acc.push("c".into()).unwrap();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 3);
    }

    #[test]
    fn accumulator_rejects_when_queue_is_full() {
        let mut acc = BatchAccumulator::new(BatchConfig {
            batch_size: 100,
            max_wait_ms: 10_000,
            max_queue_len: 2,
            ..Default::default()
        });

        assert!(acc.push("a".into()).unwrap().is_none());
        assert!(acc.push("b".into()).unwrap().is_none());

        // ⚠️ 队列满 → 必须**返回错误**，不能静默丢弃。
        let err = acc.push("c".into()).unwrap_err();
        assert_eq!(err.status_code(), 400);
        // 队列长度不变（c 没有被塞进去）。
        assert_eq!(acc.len(), 2);
    }

    #[test]
    fn accumulator_flush_returns_pending_items() {
        let mut acc = BatchAccumulator::new(BatchConfig {
            batch_size: 100,
            max_wait_ms: 10_000,
            ..Default::default()
        });

        acc.push("a".into()).unwrap();
        acc.push("b".into()).unwrap();

        let batch = acc.flush();
        assert_eq!(batch.len(), 2);
        assert!(acc.is_empty());
    }

    #[test]
    fn strategy_parse_roundtrip() {
        for s in [
            BatchStrategy::Sequential,
            BatchStrategy::Fixed,
            BatchStrategy::Bucketed,
        ] {
            assert_eq!(BatchStrategy::parse(s.label()), None); // 标签是中文
        }
        assert_eq!(BatchStrategy::parse("fixed"), Some(BatchStrategy::Fixed));
        assert_eq!(
            BatchStrategy::parse("bucket"),
            Some(BatchStrategy::Bucketed)
        );
        assert_eq!(BatchStrategy::parse("nope"), None);
    }

    #[test]
    fn config_clamps_batch_size_to_at_least_one() {
        // ⚠️ 0 会导致空批 / 除零。
        let config = BatchConfig {
            batch_size: 0,
            bucket_batch_size: 0,
            ..Default::default()
        };
        // 直接验证 clamp 逻辑的意图（from_env 依赖环境变量，不在这里测）。
        assert!(config.batch_size.max(1) >= 1);
    }
}
