//! vLLM 调度器的**离散事件模拟** —— 连续批处理 vs 静态批处理。
//!
//! # 为什么要有这个模块
//!
//! 本项目的目标是「用 docker 跑 vLLM 并测吞吐量」。
//! 但 vLLM 官方镜像**只支持 Linux + NVIDIA GPU**，
//! 而本仓库的开发机是 **Apple M1 Max（ARM64 macOS，无 CUDA）** ——
//! 所以「真 vLLM」这条轨在本机**跑不起来**。
//!
//! 如果只写 docker 脚本，本机就只能「看文档跑不了」，
//! 项目变成**不可验证**的。
//!
//! 所以本项目分两轨：
//!
//! | 轨 | 跑在哪 | 能验证什么 |
//! |---|---|---|
//! | **真 vLLM 轨** | 有 NVIDIA GPU 的 Linux 机器（docker） | 真实吞吐、真实 TTFT |
//! | **模拟轨**（本模块） | **任何机器**（纯计算、零依赖） | 调度算法的**机制**与**相对关系** |
//!
//! ⚠️ **模拟轨不能替代真 vLLM 的数字** ——
//! 真实吞吐取决于 GPU 型号、显存带宽、量化方式、kernel 实现。
//! 模拟轨的价值是：**把「批处理为什么有效」这件事变成可复现的实验**，
//! 而不是去猜「连续批处理能快几倍」。
//!
//! # 模拟的核心模型
//!
//! ## 一次前向（forward）的耗时
//!
//! ```text
//! 一次前向耗时 = max(权重加载耗时, 总 token 数 / 饱和算力)
//!                        ↑                    ↑
//!               与批大小**无关**        与批大小**成正比**
//! ```
//!
//! 这是批处理能提速的**根本原因**：
//! GPU 一次前向要先把权重从显存搬到计算单元，
//! 这一步的耗时**与批里有几条数据无关** ——
//! N 条数据共享它，单位成本随 N 下降。
//!
//! ⚠️ 但「与批大小无关」只在**小批**时成立：
//! 批大了之后算力饱和，耗时变成与 token 数成正比。
//! 上式的 `max` 正好表达了这个拐点。
//!
//! ## 静态批处理 vs 连续批处理
//!
//! | | 静态批处理（static batching） | 连续批处理（continuous batching） |
//! |---|---|---|
//! | 组成方式 | **凑够一批**才开始 | **有位置就加入** |
//! | 批的成员 | 固定，跑完这一批才换下一批 | **动态**，谁先完成谁腾位置 |
//! | 新请求等待 | 等**整个批**跑完 | 等**一个位置**空出来 |
//! | 短请求 | 被同批长请求拖着 | 跑完立刻腾位置 |
//!
//! vLLM 用的是**连续批处理**（continuous batching），
//! 这也是它相对 HF `generate` 最大的吞吐优势来源。
//!
//! ## 调度优先级：decode 优先于 prefill
//!
//! 一个 forward 里，如果既有 decode（生成一个 token）又有 prefill（处理 prompt），
//! 先给谁？**先给 decode**。
//!
//! 理由：decode 的请求**已经在服务中**了（用户正在等），
//! 而 prefill 的请求**还没开始**（多等一会儿无感）。
//! 所以预算是「先满足 decode，剩余的给 prefill」。
//!
//! ## ⚠️ chunked prefill 为什么重要
//!
//! 一条 4000 token 的 prompt 如果**一次算完**（不切块），
//! 这一个 forward 就要几秒 —— 期间**所有正在 decode 的请求全部卡住**
//! （head-of-line blocking，用户看到的是 TTFT 尖刺）。
//!
//! **chunked prefill** 把长 prompt 切成若干块，
//! 每块和其它请求的 decode **混在同一个 forward 里**，
//! 于是「长 prompt 的预填充」被摊薄到多个 forward，
//! decode 请求的停顿变成**许多次很小**的停顿。
//!
//! 本模块用 `enable_chunked_prefill` 开关对比这两种行为。

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// 批处理模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchingMode {
    /// **静态批处理** —— 凑够一批再跑，跑完才换下一批。
    ///
    /// 对照组：HF `generate` / TGI 老版本的默认行为。
    Static,

    /// **连续批处理** —— 有空位就加入，谁先完成谁腾位置。
    ///
    /// vLLM 的默认行为。
    Continuous,
}

impl BatchingMode {
    /// 人类可读名。
    pub fn label(&self) -> &'static str {
        match self {
            BatchingMode::Static => "静态批处理",
            BatchingMode::Continuous => "连续批处理",
        }
    }

    /// 解析（从字符串）。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "static" => Some(BatchingMode::Static),
            "continuous" => Some(BatchingMode::Continuous),
            _ => None,
        }
    }
}

/// 推理引擎的**硬件/调度参数**。
///
/// 全部对应 vLLM 的同名命令行参数。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EngineConfig {
    /// ★ 同时处理的序列数上限 → vLLM 的 `--max-num-seqs`。
    ///
    /// 这是**最关键**的调参旋钮：
    /// 调大 → 批更满 → 吞吐更高，但**单请求延迟变差**、显存占用上升。
    pub max_num_seqs: usize,

    /// ★ 单个 forward 里最多处理多少 token → vLLM 的 `--max-num-batched-tokens`。
    ///
    /// 它限制了一个 forward 的**算力预算**。
    /// 调小 → 每个 forward 更短 → 长 prompt 不会「霸占」整个 forward，
    /// 但 decode 请求的每次前进也更慢。
    pub max_num_batched_tokens: usize,

    /// ★ 是否启用 chunked prefill → vLLM 的 `--enable-chunked-prefill`。
    ///
    /// 启用后长 prompt 被切块，与 decode 混在同一个 forward 里。
    pub enable_chunked_prefill: bool,

    /// **权重加载耗时**（毫秒）—— 每次 forward 都要付一次。
    ///
    /// 这是批处理收益的来源：批越大，这笔固定成本被摊得越薄。
    /// 在真实 GPU 上它约等于「把权重从显存搬到计算单元」的时间。
    pub weight_load_ms: f64,

    /// **饱和算力**（token/秒）—— 算力打满时的聚合吞吐。
    ///
    /// 超过这个速率后，再加批也只是线性变慢。
    pub compute_tokens_per_sec: f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 16,
            max_num_batched_tokens: 2048,
            enable_chunked_prefill: true,
            // ⚠️ 这两个数是「桌面级 GPU」的量级估计，
            //    不是任何具体 GPU 的实测值 —— 模拟轨的价值在**相对关系**。
            weight_load_ms: 5.0,
            compute_tokens_per_sec: 20_000.0,
        }
    }
}

impl EngineConfig {
    /// 从环境变量加载。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `VLLM_MAX_NUM_SEQS` | 同时处理的序列数上限 | `16` |
    /// | `VLLM_MAX_BATCHED_TOKENS` | 单 forward 的 token 预算 | `2048` |
    /// | `VLLM_CHUNKED_PREFILL` | 是否启用 chunked prefill | `1` |
    /// | `VLLM_WEIGHT_LOAD_MS` | 每次 forward 的权重加载耗时 | `5.0` |
    /// | `VLLM_COMPUTE_TPS` | 饱和算力（token/秒） | `20000` |
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Some(v) = env_usize("VLLM_MAX_NUM_SEQS") {
            // ⚠️ clamp 到 ≥ 1 —— 0 会让「空批」永远无法调度。
            config.max_num_seqs = v.clamp(1, 4096);
        }
        if let Some(v) = env_usize("VLLM_MAX_BATCHED_TOKENS") {
            config.max_num_batched_tokens = v.clamp(1, 1_048_576);
        }
        if let Some(v) = env_bool("VLLM_CHUNKED_PREFILL") {
            config.enable_chunked_prefill = v;
        }
        if let Some(v) = env_f64("VLLM_WEIGHT_LOAD_MS") {
            config.weight_load_ms = v.clamp(0.0, 10_000.0);
        }
        if let Some(v) = env_f64("VLLM_COMPUTE_TPS") {
            config.compute_tokens_per_sec = v.clamp(1.0, 10_000_000.0);
        }
        config
    }

    /// 一次 forward 的耗时（毫秒）。
    ///
    /// ⚠️ 这是整个模拟的**核心公式**：
    /// `max(权重加载, 算力时间)` —— 固定开销与可变开销取大者。
    fn step_duration_ms(&self, total_tokens: usize) -> f64 {
        if total_tokens == 0 {
            return 0.0;
        }
        let compute_ms = total_tokens as f64 / self.compute_tokens_per_sec * 1000.0;
        self.weight_load_ms.max(compute_ms)
    }
}

/// 一条请求的**规格**（不含时间）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestSpec {
    /// 请求 id。
    pub id: usize,

    /// prompt 的 token 数。
    pub prompt_tokens: usize,

    /// 期望生成的 token 数。
    pub output_tokens: usize,
}

/// 一条请求的**到达时刻**与规格。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arrival {
    /// 到达时刻（毫秒，从实验开始算）。
    pub at_ms: f64,

    /// 请求规格。
    pub spec: RequestSpec,
}

/// 一条请求的**执行结果**。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    /// 请求 id。
    pub id: usize,

    /// 到达时刻（毫秒）。
    pub arrived_ms: f64,

    /// 被**接纳**（进入批）的时刻（毫秒）。
    ///
    /// ⚠️ 连续批处理与静态批处理的核心差异就体现在这里：
    /// 静态批处理里新请求要等**整个批**跑完才能被接纳。
    pub admitted_ms: f64,

    /// **首 token 时延** TTFT（毫秒）—— 到达 → 拿到第一个输出 token。
    pub ttft_ms: f64,

    /// **端到端时延**（毫秒）—— 到达 → 全部生成完。
    pub e2e_ms: f64,

    /// 实际生成的 token 数。
    pub output_tokens: usize,

    /// 每个输出 token 的产出时刻（毫秒）。
    ///
    /// 长度 = `output_tokens`（第一个是 TTFT 时刻）。
    pub token_times_ms: Vec<f64>,
}

impl Outcome {
    /// **每 token 时延** TPOT（毫秒）—— 首 token 之后的平均间隔。
    ///
    /// ⚠️ 这是「解码阶段」的延迟指标，
    /// 与 TTFT（「等待阶段」的延迟）**必须分开看**：
    /// 连续批处理通常**抬高 TTFT**（要排队等位置）但**改善 TPOT**
    /// （一旦跑起来，同批的请求互相分摊了固定开销）。
    pub fn tpot_ms(&self) -> f64 {
        if self.output_tokens <= 1 {
            return 0.0;
        }
        let span = self.e2e_ms - self.ttft_ms;
        span / (self.output_tokens - 1) as f64
    }

    /// 排队等待时间（毫秒）—— 到达 → 被接纳。
    pub fn queue_wait_ms(&self) -> f64 {
        (self.admitted_ms - self.arrived_ms).max(0.0)
    }
}

/// 一次模拟的完整结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationResult {
    /// 用的模式。
    pub mode: BatchingMode,

    /// 用的引擎配置。
    pub config: EngineConfig,

    /// 每条请求的结果。
    pub outcomes: Vec<Outcome>,

    /// 模拟总时长（毫秒）—— 从最早到达到最后完成。
    pub wall_ms: f64,
}

impl SimulationResult {
    /// 模拟执行过的 forward 次数。
    pub fn forwards(&self) -> usize {
        self.outcomes.len()
    }

    /// 汇总度量。
    pub fn metrics(&self) -> crate::metrics::ServeMetrics {
        crate::metrics::ServeMetrics::from_outcomes(&self.outcomes, self.wall_ms)
    }
}

/// 模拟器内部的一条**在途序列**。
#[derive(Debug, Clone)]
struct Seq {
    /// 对应 `RequestSpec` 的下标。
    spec_idx: usize,

    /// 还没算的 prompt token 数。
    prefill_left: usize,

    /// 已经生成的 token 数。
    generated: usize,

    /// 目标 token 数。
    target: usize,

    /// 是否已完成 prefill。
    prefill_done: bool,

    /// 已产出的 token 时刻。
    token_times: Vec<f64>,

    /// 被接纳的时刻。
    admitted_ms: f64,

    /// ★ 是否已完成（静态批处理里，**完成的成员仍然占着批的位置**）。
    ///
    /// ⚠️ 这是静态批处理与连续批处理的**语义分水岭**：
    ///
    /// - **连续批处理**：完成的序列**立刻**腾出位置 → 新请求马上能进
    /// - **静态批处理**：完成的序列**留在批里空转** → 新请求要等**整批**跑完
    ///
    /// 静态批处理「浪费」的那些槽位，正是它吞吐低的原因 ——
    /// 而这个浪费**就是**静态批处理的本质。
    finished: bool,
}

/// **运行一次模拟**。
///
/// ## 参数
///
/// - `config` —— 引擎配置
/// - `mode` —— 批处理模式
/// - `arrivals` —— 请求的到达序列（**必须按 `at_ms` 升序**）
///
/// ## ⚠️ 为什么是「离散事件」而不是「真实时间」
///
/// 模拟器**不 sleep**、不消耗墙钟时间 —— 它在内存里推进一个虚拟时钟。
/// 这样：
/// - 跑 10000 条请求只要几毫秒，测试可以零网络、零等待地跑
/// - 结果**完全确定性**（同样的输入 → 同样的输出），可断言
pub fn simulate(
    config: &EngineConfig,
    mode: BatchingMode,
    arrivals: &[Arrival],
) -> SimulationResult {
    let mut sim = Simulator::new(config, mode);
    sim.run(arrivals);
    sim.finish()
}

/// 模拟器。
struct Simulator<'a> {
    config: &'a EngineConfig,
    mode: BatchingMode,
    now_ms: f64,
    /// 还没到达的请求（按 `at_ms` 升序）。
    pending: VecDeque<usize>,
    /// 已到达但还没被接纳的请求下标。
    waiting: VecDeque<usize>,
    /// 在途序列。
    running: Vec<Seq>,
    /// 已完成的结果。
    done: Vec<Outcome>,
    /// 到达时刻（用于算总时长起点）。
    first_arrival_ms: f64,
    /// ⚠️ 步数上限 —— 防御「调度器卡住」导致的死循环。
    max_steps: usize,
}

impl<'a> Simulator<'a> {
    /// 新建模拟器。
    fn new(config: &'a EngineConfig, mode: BatchingMode) -> Self {
        Self {
            config,
            mode,
            now_ms: 0.0,
            pending: VecDeque::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            done: Vec::new(),
            first_arrival_ms: 0.0,
            max_steps: 10_000_000,
        }
    }

    /// 主循环。
    fn run(&mut self, arrivals: &[Arrival]) {
        if arrivals.is_empty() {
            return;
        }
        self.first_arrival_ms = arrivals[0].at_ms;
        self.now_ms = arrivals[0].at_ms;

        // ⚠️ 必须先按 at_ms 排序 —— 乱序会让「到达」的语义失效。
        let mut ordered: Vec<usize> = (0..arrivals.len()).collect();
        ordered.sort_by(|&a, &b| {
            arrivals[a]
                .at_ms
                .partial_cmp(&arrivals[b].at_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        self.pending = ordered.into_iter().collect();

        let mut steps = 0usize;
        while steps < self.max_steps {
            steps += 1;

            // 1. 把已到达的请求移进等待队列。
            while let Some(&idx) = self.pending.front() {
                if arrivals[idx].at_ms <= self.now_ms {
                    self.pending.pop_front();
                    self.waiting.push_back(idx);
                } else {
                    break;
                }
            }

            // 2. 没有任何在途请求且没有等待请求 → 直接跳到下一次到达。
            if self.running.is_empty() && self.waiting.is_empty() {
                if let Some(&next) = self.pending.front() {
                    self.now_ms = arrivals[next].at_ms;
                    continue;
                }
                break;
            }

            // 3. 规划这一个 forward。
            let plan = self.plan_step(arrivals);

            // ⚠️ 空计划说明「什么都做不了」（例如只差一个 token 但预算被占），
            //    此时**推进时间**避免死循环。
            if plan.is_empty() {
                // 静态模式下可能因为「批里所有序列都还在等 prefill 预算」而空转。
                // 强制让批里第一个还在 prefill 的序列继续（哪怕超预算）。
                if !self.force_progress(arrivals) {
                    // 真的做不了 → 跳过这一批等待的请求（不应发生）。
                    self.now_ms += 1.0;
                    if self.pending.is_empty() && self.waiting.is_empty() {
                        break;
                    }
                    continue;
                }
            }

            // 4. 执行 forward：推进时间、产出 token。
            self.execute_step(arrivals, &plan);
        }
    }

    /// 规划一个 forward：返回 `[(序列下标, 本步处理的 token 数)]`。
    fn plan_step(&mut self, arrivals: &[Arrival]) -> Vec<(usize, usize)> {
        match self.mode {
            BatchingMode::Static => self.plan_static(arrivals),
            BatchingMode::Continuous => self.plan_continuous(arrivals),
        }
    }

    /// ★ 静态批处理的规划。
    ///
    /// ## 关键差异
    ///
    /// 1. **批的成员固定** —— 只有当前批**全部完成**，才会换下一批
    /// 2. **锁步前进** —— 批里每个成员每步都要分到预算，
    ///    一个成员的长 prefill 会**饿死**其它成员的 decode
    /// 3. ★ **完成的成员留在批里空转** —— 它们**占着槽位**但不再消耗算力，
    ///    所以新请求要等**整批**跑完才能进
    ///
    /// ## ⚠️ 第 3 点是静态批处理吞吐低的**根本原因**
    ///
    /// 一个 8 条的批里，如果 7 条先跑完了，只有 1 条还在跑 ——
    /// 静态批处理里那 7 个槽位就**白白浪费**了，
    /// 期间新来的请求只能干等。
    /// 连续批处理正是为了消掉这个浪费而生的。
    fn plan_static(&mut self, arrivals: &[Arrival]) -> Vec<(usize, usize)> {
        // 批空了 → 从等待队列里凑一批。
        if self.running.is_empty() {
            let take = self.config.max_num_seqs.min(self.waiting.len());
            for _ in 0..take {
                if let Some(idx) = self.waiting.pop_front() {
                    self.admit(arrivals, idx);
                }
            }
        }

        // ⚠️ ★ 只有**整批都完成**了才清空批，换下一批。
        //    否则保留原批（完成的成员继续占位）。
        if !self.running.is_empty() && self.running.iter().all(|s| s.finished) {
            self.running.clear();
            // 立刻用等待队列里的请求填上下一批。
            let take = self.config.max_num_seqs.min(self.waiting.len());
            for _ in 0..take {
                if let Some(idx) = self.waiting.pop_front() {
                    self.admit(arrivals, idx);
                }
            }
            if self.running.is_empty() {
                return Vec::new();
            }
        }

        let mut plan: Vec<(usize, usize)> = Vec::new();
        let mut budget = self.config.max_num_batched_tokens;

        // ⚠️ 先给 prefill（按成员顺序）—— 这就是「长 prompt 饿死同批 decode」的来源。
        for i in 0..self.running.len() {
            if budget == 0 {
                break;
            }
            let seq = &self.running[i];
            if !seq.prefill_done && seq.prefill_left > 0 {
                let take = seq.prefill_left.min(budget);
                budget -= take;
                plan.push((i, take));
            }
        }

        // 剩下的预算给 decode（每个成员 1 个 token）。
        for i in 0..self.running.len() {
            if budget == 0 {
                break;
            }
            if self.running[i].prefill_done
                && !self.running[i].finished
                && self.running[i].generated < self.running[i].target
            {
                budget -= 1;
                plan.push((i, 1));
            }
        }

        plan
    }

    /// ★ 连续批处理的规划。
    ///
    /// ## 关键差异
    ///
    /// 1. **有空位就接纳** —— 不等凑满
    /// 2. **decode 优先** —— 先保证在途请求前进，再拿剩余预算做 prefill
    /// 3. **chunked prefill** —— 长 prompt 切块，与 decode 混在同一个 forward
    fn plan_continuous(&mut self, arrivals: &[Arrival]) -> Vec<(usize, usize)> {
        // 1. 接纳新请求（有空位就接纳，不等凑满）。
        //
        // ⚠️ 完成的序列在 `collect()` 里已经被**移除**了，
        //    所以 `running.len()` 就是「当前占用的槽位数」。
        while self.running.len() < self.config.max_num_seqs {
            let Some(idx) = self.waiting.pop_front() else {
                break;
            };
            self.admit(arrivals, idx);
        }

        let mut plan: Vec<(usize, usize)> = Vec::new();
        let mut budget = self.config.max_num_batched_tokens;

        // 2. ⚠️ 关闭 chunked prefill 时，prefill 与 decode **不能混在一个 forward**。
        //    有 prefill 要做 → 这个 forward 只做 prefill（decode 被阻塞 = head-of-line）。
        if !self.config.enable_chunked_prefill {
            let has_prefill = self
                .running
                .iter()
                .any(|s| !s.prefill_done && s.prefill_left > 0);

            if has_prefill {
                // 只调度 prefill。
                for i in 0..self.running.len() {
                    if budget == 0 {
                        break;
                    }
                    let seq = &self.running[i];
                    if !seq.prefill_done && seq.prefill_left > 0 {
                        let take = seq.prefill_left.min(budget);
                        budget -= take;
                        plan.push((i, take));
                    }
                }
                return plan;
            }
        }

        // 3. decode 优先 —— 每个在 decode 的序列各拿 1 个 token。
        for i in 0..self.running.len() {
            if budget == 0 {
                break;
            }
            if self.running[i].prefill_done && self.running[i].generated < self.running[i].target {
                budget -= 1;
                plan.push((i, 1));
            }
        }

        // 4. 剩余预算给 prefill（chunked）。
        for i in 0..self.running.len() {
            if budget == 0 {
                break;
            }
            let seq = &self.running[i];
            if !seq.prefill_done && seq.prefill_left > 0 {
                let take = seq.prefill_left.min(budget);
                budget -= take;
                plan.push((i, take));
            }
        }

        plan
    }

    /// 接纳一条请求。
    fn admit(&mut self, arrivals: &[Arrival], idx: usize) {
        let spec = arrivals[idx].spec;
        self.running.push(Seq {
            spec_idx: idx,
            prefill_left: spec.prompt_tokens,
            generated: 0,
            target: spec.output_tokens,
            prefill_done: spec.prompt_tokens == 0,
            token_times: Vec::new(),
            admitted_ms: self.now_ms,
            finished: false,
        });
    }

    /// 强制推进（避免空计划导致的死循环）。
    ///
    /// 场景：关闭 chunked prefill 且某条 prompt 长度超过整个预算 ——
    /// 它永远分不到 token。修法是**允许它超预算**（真 vLLM 会报错，
    /// 这里选择「跑得动」优先，README 里明确标注了这个简化）。
    fn force_progress(&mut self, _arrivals: &[Arrival]) -> bool {
        for i in 0..self.running.len() {
            if !self.running[i].prefill_done && self.running[i].prefill_left > 0 {
                let left = self.running[i].prefill_left;
                self.running[i].prefill_left = 0;
                self.running[i].prefill_done = true;
                self.execute_step_tokens(i, left);
                return true;
            }
        }
        false
    }

    /// 执行一个 forward。
    fn execute_step(&mut self, arrivals: &[Arrival], plan: &[(usize, usize)]) {
        let total_tokens: usize = plan.iter().map(|(_, t)| *t).sum();
        let step_ms = self.config.step_duration_ms(total_tokens);

        for &(seq_idx, tokens) in plan {
            self.execute_step_tokens(seq_idx, tokens);
        }

        self.now_ms += step_ms;

        // 收尾：把完成的序列转成结果。
        self.collect(arrivals);
    }

    /// 给某条序列分配 token 并推进它的状态。
    fn execute_step_tokens(&mut self, seq_idx: usize, tokens: usize) {
        let seq = &mut self.running[seq_idx];
        if !seq.prefill_done {
            // prefill：消耗 prompt token。
            seq.prefill_left = seq.prefill_left.saturating_sub(tokens);
            if seq.prefill_left == 0 {
                seq.prefill_done = true;
            }
            return;
        }

        // decode：每步产出 1 个 token（`tokens` 恒为 1）。
        if seq.generated < seq.target {
            seq.generated += 1;
            seq.token_times.push(self.now_ms);
        }
    }

    /// 收尾：把完成的序列标记 / 收集成结果。
    ///
    /// ## ⚠️ ★ 两种模式的差别就在这里
    ///
    /// | 模式 | 完成的序列 |
    /// |---|---|
    /// | **连续批处理** | **立刻移除** → 槽位腾出来，新请求马上能进 |
    /// | **静态批处理** | **保留在批里**（`finished = true`）→ **继续占着槽位**，新请求要等整批跑完 |
    ///
    /// 这个差别就是静态批处理吞吐低的**根本原因**：
    /// 批里 7 条跑完了、只剩 1 条在跑时，那 7 个槽位被白白占着。
    fn collect(&mut self, arrivals: &[Arrival]) {
        let now = self.now_ms;

        // ⚠️ 用 `while` 而不是 `for i in 0..len()` ——
        //    连续批处理里会 `remove(i)`，删元素会让 `len()` 变化，
        //    `for` 循环的下标会**越界**。
        let mut i = 0;
        while i < self.running.len() {
            let seq = &self.running[i];
            if seq.prefill_done && seq.generated >= seq.target && !seq.finished {
                let spec = arrivals[seq.spec_idx].spec;
                let arrived_ms = arrivals[seq.spec_idx].at_ms;
                let token_times = seq.token_times.clone();
                let ttft_ms = token_times.first().copied().unwrap_or(now);

                self.done.push(Outcome {
                    id: spec.id,
                    arrived_ms,
                    admitted_ms: seq.admitted_ms,
                    ttft_ms: ttft_ms - arrived_ms,
                    e2e_ms: now - arrived_ms,
                    output_tokens: seq.generated,
                    token_times_ms: token_times,
                });

                match self.mode {
                    // ★ 连续批处理：立刻腾位置。
                    BatchingMode::Continuous => {
                        self.running.remove(i);
                        // ⚠️ 不加 i —— 移除后下一个元素**前移**到了 i。
                        continue;
                    }
                    // ★ 静态批处理：保留占位（但不再消耗算力）。
                    BatchingMode::Static => {
                        self.running[i].finished = true;
                    }
                }
            }
            i += 1;
        }
    }

    /// 收尾，组装结果。
    fn finish(self) -> SimulationResult {
        let wall_ms = self
            .done
            .iter()
            .map(|o| o.arrived_ms + o.e2e_ms)
            .fold(f64::MIN, f64::max);
        let wall_ms = if wall_ms == f64::MIN {
            0.0
        } else {
            (wall_ms - self.first_arrival_ms).max(0.0)
        };

        let mut outcomes = self.done;
        outcomes.sort_by_key(|o| o.id);

        SimulationResult {
            mode: self.mode,
            config: *self.config,
            outcomes,
            wall_ms,
        }
    }
}

/// 读一个 `usize` 环境变量。
fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .and_then(|v| v.parse().ok())
}

/// 读一个 `f64` 环境变量。
fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .and_then(|v| v.parse().ok())
}

/// 读一个布尔环境变量（`1/true/yes/on` 为真）。
fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrival(id: usize, at: f64, prompt: usize, output: usize) -> Arrival {
        Arrival {
            at_ms: at,
            spec: RequestSpec {
                id,
                prompt_tokens: prompt,
                output_tokens: output,
            },
        }
    }

    fn config() -> EngineConfig {
        EngineConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: 2048,
            enable_chunked_prefill: true,
            weight_load_ms: 5.0,
            compute_tokens_per_sec: 20_000.0,
        }
    }

    #[test]
    fn step_duration_is_dominated_by_weight_load_for_small_batches() {
        let c = config();
        // 1 个 token：算力时间 0.05ms，权重加载 5ms → 取大者。
        assert!((c.step_duration_ms(1) - 5.0).abs() < 1e-9);
        // 1000 个 token：算力时间 50ms > 5ms → 取算力。
        assert!((c.step_duration_ms(1000) - 50.0).abs() < 1e-9);
    }

    #[test]
    fn larger_batches_amortize_the_weight_load_cost() {
        let c = config();
        // 一个 forward 里塞 32 个 token 的成本 == 1 个 token 的成本
        // → 固定开销被摊薄了 32 倍。
        assert_eq!(c.step_duration_ms(1), c.step_duration_ms(32));
    }

    #[test]
    fn every_request_completes_in_both_modes() {
        let arrivals: Vec<Arrival> = (0..20)
            .map(|i| arrival(i, i as f64 * 10.0, 32, 8))
            .collect();

        for mode in [BatchingMode::Static, BatchingMode::Continuous] {
            let result = simulate(&config(), mode, &arrivals);
            assert_eq!(result.outcomes.len(), 20, "{mode:?} 有请求没完成");
            for o in &result.outcomes {
                assert_eq!(o.output_tokens, 8);
                assert!(o.ttft_ms >= 0.0);
                assert!(o.e2e_ms >= o.ttft_ms);
            }
        }
    }

    #[test]
    fn continuous_batching_has_lower_queue_wait_than_static() {
        // ⚠️ ★ 关键：**同一个批里必须有长短混杂的请求**。
        //
        //    静态批处理里，短请求跑完后**仍占着槽位**（空转），
        //    要等同批的长请求跑完才能换下一批 ——
        //    这段时间里新来的请求**全被挡住**。
        //
        //    如果所有请求长度相同，两种模式**完全等价**
        //    （大家一起跑完，谁先完成都一样）——
        //    这时测不出差异是**正确**的，不是 bug。
        let arrivals: Vec<Arrival> = (0..16)
            .map(|i| {
                // 奇偶交替：短请求 / 长请求混在同一批里。
                if i % 2 == 0 {
                    arrival(i, 0.0, 32, 4)
                } else {
                    arrival(i, 0.0, 512, 64)
                }
            })
            .collect();

        let static_result = simulate(&config(), BatchingMode::Static, &arrivals);
        let cont_result = simulate(&config(), BatchingMode::Continuous, &arrivals);

        let static_wait: f64 = static_result
            .outcomes
            .iter()
            .map(|o| o.queue_wait_ms())
            .sum();
        let cont_wait: f64 = cont_result.outcomes.iter().map(|o| o.queue_wait_ms()).sum();

        // ⚠️ 静态批处理：新请求要等**整个批**跑完才能被接纳。
        assert!(
            cont_wait < static_wait,
            "连续批处理的排队等待应更少：{cont_wait} vs {static_wait}"
        );
    }

    #[test]
    fn continuous_batching_has_higher_throughput_on_mixed_lengths() {
        // ⚠️ 长短混杂的负载 —— 这是连续批处理发挥优势的场景。
        let arrivals: Vec<Arrival> = (0..16)
            .map(|i| {
                if i % 2 == 0 {
                    arrival(i, 0.0, 32, 4)
                } else {
                    arrival(i, 0.0, 512, 64)
                }
            })
            .collect();

        let static_result = simulate(&config(), BatchingMode::Static, &arrivals);
        let cont_result = simulate(&config(), BatchingMode::Continuous, &arrivals);

        let sm = static_result.metrics();
        let cm = cont_result.metrics();

        assert!(
            cm.output_token_throughput_per_sec > sm.output_token_throughput_per_sec,
            "连续批处理 token 吞吐应更高：{} vs {}",
            cm.output_token_throughput_per_sec,
            sm.output_token_throughput_per_sec
        );
    }

    #[test]
    fn uniform_workload_makes_both_modes_equivalent() {
        // ⚠️ **对照组** —— 长度完全一致时，两种模式**应该**几乎等价。
        //
        //    这条测试锁住的是「差异来自长度参差」这个前提：
        //    如果哪天改了调度器导致两者**永远**有差异，
        //    说明我们对「静态批处理什么时候没用」的判断变了。
        let arrivals: Vec<Arrival> = (0..8).map(|i| arrival(i, 0.0, 64, 16)).collect();

        let static_result = simulate(&config(), BatchingMode::Static, &arrivals);
        let cont_result = simulate(&config(), BatchingMode::Continuous, &arrivals);

        let s: f64 = static_result.outcomes.iter().map(|o| o.e2e_ms).sum();
        let c: f64 = cont_result.outcomes.iter().map(|o| o.e2e_ms).sum();

        // 允许 1% 的误差 —— 浮点与调度细节会带来微小差异。
        assert!(
            (s - c).abs() / s.max(1.0) < 0.01,
            "长度一致时两种模式应几乎等价：{s} vs {c}"
        );
    }

    #[test]
    fn chunked_prefill_reduces_the_tail_latency_spike() {
        // 一条超长 prompt（4000 token）+ 若干短请求。
        let mut arrivals = vec![arrival(0, 0.0, 4000, 16)];
        for i in 1..8 {
            arrivals.push(arrival(i, 1.0, 16, 16));
        }

        let mut with_chunk = config();
        with_chunk.enable_chunked_prefill = true;
        let mut without_chunk = config();
        without_chunk.enable_chunked_prefill = false;

        let a = simulate(&with_chunk, BatchingMode::Continuous, &arrivals);
        let b = simulate(&without_chunk, BatchingMode::Continuous, &arrivals);

        // ⚠️ 关闭 chunked prefill → 长 prompt 霸占 forward →
        //    短请求的 TTFT 明显变差。
        let short_ttft_a: f64 = a.outcomes[1..].iter().map(|o| o.ttft_ms).sum();
        let short_ttft_b: f64 = b.outcomes[1..].iter().map(|o| o.ttft_ms).sum();
        assert!(
            short_ttft_a < short_ttft_b,
            "chunked prefill 应降低短请求的 TTFT 合计：{short_ttft_a} vs {short_ttft_b}"
        );
    }

    #[test]
    fn max_num_seqs_limits_concurrency() {
        let mut c = config();
        c.max_num_seqs = 2;
        let arrivals: Vec<Arrival> = (0..6).map(|i| arrival(i, 0.0, 16, 8)).collect();
        let result = simulate(&c, BatchingMode::Continuous, &arrivals);
        assert_eq!(result.outcomes.len(), 6);
        // 限流后总时长应该更长（并发被压到 2）。
        let mut wide = config();
        wide.max_num_seqs = 16;
        let wide_result = simulate(&wide, BatchingMode::Continuous, &arrivals);
        assert!(result.wall_ms > wide_result.wall_ms);
    }

    #[test]
    fn simulation_is_deterministic() {
        let arrivals: Vec<Arrival> = (0..50)
            .map(|i| arrival(i, i as f64 * 3.0, 20 + i, 5 + i % 7))
            .collect();
        let a = simulate(&config(), BatchingMode::Continuous, &arrivals);
        let b = simulate(&config(), BatchingMode::Continuous, &arrivals);
        assert_eq!(a.outcomes, b.outcomes);
        assert!((a.wall_ms - b.wall_ms).abs() < 1e-9);
    }

    #[test]
    fn empty_workload_produces_empty_result() {
        let result = simulate(&config(), BatchingMode::Continuous, &[]);
        assert!(result.outcomes.is_empty());
        assert_eq!(result.wall_ms, 0.0);
    }

    #[test]
    fn zero_output_tokens_still_completes() {
        let arrivals = vec![arrival(0, 0.0, 32, 0)];
        let result = simulate(&config(), BatchingMode::Continuous, &arrivals);
        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(result.outcomes[0].output_tokens, 0);
    }

    #[test]
    fn tpot_is_zero_for_single_token_output() {
        let o = Outcome {
            id: 0,
            arrived_ms: 0.0,
            admitted_ms: 0.0,
            ttft_ms: 10.0,
            e2e_ms: 10.0,
            output_tokens: 1,
            token_times_ms: vec![10.0],
        };
        assert_eq!(o.tpot_ms(), 0.0);
    }
}
