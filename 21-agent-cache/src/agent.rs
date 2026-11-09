//! 带缓存的 Agent 循环 —— 本项目的核心。
//!
//! ## 与 16-agent-function-calling 的关系
//!
//! 16 的循环是「**每轮都调模型**」。本项目在**每轮调模型之前**插入一次缓存查询：
//!
//! ```text
//! 16（无缓存）
//!   ┌─▶ 调模型 ─▶ 有 tool_calls？ ─▶ 执行工具 ─▶ 回执 ─▶ 回到开头
//!
//! 21（有缓存）
//!   ┌─▶ 算键 ─▶ 查缓存 ─▶ 命中？ ─是─▶ 用缓存响应（不调模型）──┐
//!   │                    └─否─▶ 调模型 ─▶ 写缓存 ────────────┤
//!   └────────────────────────────────────────────────────────┘
//! ```
//!
//! ## ⚠️ 最重要的一个设计决策：**缓存粒度**
//!
//! 有两种粒度可选：
//!
//! | 粒度 | 缓存什么 | 命中后跳过什么 | 风险 |
//! |---|---|---|---|
//! | **轮级**（本项目采用） | 一次 LLM 调用的响应 | 只跳过**这次模型调用** | 工具仍会执行 → 数据仍新鲜 |
//! | **会话级** | 整个 Agent 运行的最终答案 | 跳过**整条链路**（含所有工具调用） | 工具结果被冻结 → 可能过时 |
//!
//! 本项目选**轮级**，理由：
//!
//! 1. **工具仍然执行** —— 缓存只省了「模型推理的钱和时间」，
//!    没有省「工具调用的副作用」。
//! 2. **键天然正确** —— 轮级的键 = (模型, 消息历史, 工具, 参数)，
//!    这正好是「这次调用的全部输入」。
//!    会话级的键 = (问题)，那问题里的隐含上下文（时间、用户状态）就丢了。
//! 3. **可解释** —— 「第 2 轮命中」是有意义的；
//!    「整个回答命中」很难说清是「模型调用命中」还是「工具调用命中」。
//!
//! ## ⚠️ 但轮级缓存有一个**曾经被误判**的问题
//!
//! 中间轮（带 `tool_calls` 的那轮）的键里包含 `call_id` ——
//! 而 `call_id` 是模型**每次生成都不同**的。
//!
//! 所以：**带工具调用的那一轮几乎永远命中不了**。
//!
//! 这不是 bug，是「exact-match cache」的本质（LiteLLM 文档原话：
//! 「any change to the conversation is a miss」）。
//!
//! ### ⚠️⚠️ 但本项目实测发现：**这个解释是错的**
//!
//! 最初的结论是「带工具的链路整体命中率很低」，
//! 理由是「键里含 `call_id`，而去掉它会破坏回执对账（16 的坑 1）」。
//!
//! 实测推翻了这个解释：
//!
//! 1. 实现了「确定性 `call_id`」（重写成内容派生的哈希）→ **仍然不命中**。
//! 2. 探针定位到**第二个**变化源：**模型调工具时附带的自然语言**
//!    （"我查一下知识库" vs "让我检索相关资料"）——
//!    它在 assistant 消息**正文**里，同样进消息历史、进缓存键。
//! 3. 而它之所以致命，是因为代码里「**中间轮不写缓存**」这条规则：
//!
//! ```text
//! 中间轮不写缓存 → 每次都真调模型 → 模型每次措辞不同
//!   → 喂给下一轮的历史不同 → 下一轮的键必然不同
//! ```
//!
//! > **中间轮的不稳定性会顺着消息历史污染后面所有轮次。**
//!
//! 修法：**中间轮也写缓存**（`with_cache_intermediate_rounds(true)`）。
//! 命中中间轮 → assistant 文本被**完整还原**（措辞不再变）
//! → 下一轮的历史逐字节相同 → 也命中。
//!
//! **实测差值**（真实 DeepSeek，同一个问题跑两次）：
//!
//! | 档 | 第二次命中 | 第二次耗时 | 第二次 token |
//! |---|---|---|---|
//! | 只缓存最后一轮 | 0 | 4465 ms | 2926 |
//! | + 缓存中间轮 | **2** | **6 ms** | **0** |
//!
//! ⚠️ 代价：命中中间轮 = **跳过工具执行**（工具结果被冻结）。
//! 所以它是个**开关**（`CACHE_INTERMEDIATE_ROUNDS`），默认关闭。
//!
//! 详见 README「核心产出 第二节」与 `cache::call_id`。

use std::sync::Arc;
use std::time::Instant;

use agent_logging::{debug, info, warn};
use genai::chat::{ChatMessage, ChatResponse, ToolCall, ToolResponse};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cache::{
    CacheControl, CacheKey, CacheService, CachedResponse, CachedToolCall, CachedToolResult,
    CachedUsage, assign_deterministic_call_ids, tools_digest,
};
use crate::config::AppConfig;
use crate::error::{AppError, AppResult};

/// 一次 LLM 调用的来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallSource {
    /// 真正调了模型。
    Live,

    /// 命中了缓存，没调模型。
    CacheHit,
}

/// 一步的执行记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// 第几步（从 1 开始）。
    pub step: usize,

    /// 模型返回的**文本**部分。
    pub text: String,

    /// 思考内容。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    /// 本步发出的工具调用。
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,

    /// 本步是否直接给出了最终答案。
    pub final_answer: bool,

    /// 本步的模型响应来自缓存还是真实调用。
    pub source: CallSource,

    /// 本步的缓存键（便于排查「为什么没命中」）。
    pub cache_key: String,
}

/// 一次工具调用的记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// 调用 ID。
    pub call_id: String,
    /// 工具名。
    pub name: String,
    /// 参数。
    pub arguments: Value,
    /// 工具结果（给模型看的文本）。
    pub observation: String,
    /// 是否执行成功。
    pub ok: bool,
    /// 结构化结果。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    /// 工具结果是否来自缓存。
    #[serde(default)]
    pub cached: bool,
}

/// 一次 Agent 运行的完整结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRun {
    /// 原始问题。
    pub question: String,

    /// 最终答案。
    pub answer: String,

    /// 每一步的记录。
    pub steps: Vec<StepRecord>,

    /// 工具调用总次数。
    pub tool_calls: usize,

    /// 用到的工具名（去重，按首次出现顺序）。
    pub tools_used: Vec<String>,

    /// 是否因为达到步数上限而中止。
    pub hit_max_steps: bool,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,

    /// 使用的模型名。
    pub model: String,

    /// token 用量。
    pub usage: TokenUsage,

    /// 本次运行命中缓存的轮数。
    pub cache_hits: usize,

    /// 本次运行真实调模型的轮数。
    pub cache_misses: usize,
}

/// 累计 token 用量。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// 输入 token。
    pub prompt_tokens: u64,
    /// 输出 token。
    pub completion_tokens: u64,
    /// 总 token。
    pub total_tokens: u64,
}

impl TokenUsage {
    /// 累加一步的用量。
    fn add(&mut self, response: &ChatResponse) {
        let prompt = response.usage.prompt_tokens.unwrap_or(0) as u64;
        let completion = response.usage.completion_tokens.unwrap_or(0) as u64;
        self.prompt_tokens += prompt;
        self.completion_tokens += completion;
        self.total_tokens += response
            .usage
            .total_tokens
            .unwrap_or((prompt + completion) as i32) as u64;
    }
}

/// 带缓存的 Agent。
pub struct CachedAgent {
    llm: Arc<dyn agent_16::llm::FunctionLlm>,
    registry: agent_16::tool::ToolRegistry,
    cache: Arc<CacheService>,
    config: AppConfig,
    /// 工具结果的缓存 TTL。
    tool_ttl: u64,
    /// 是否启用**确定性 `call_id`**。
    ///
    /// ## ⚠️ 为什么这是一个**开关**而不是默认行为
    ///
    /// 开启后，模型返回的随机 `call_id` 会被**重写**成内容派生的确定性 ID。
    /// 这让 agentic 链路的精确匹配缓存**真正能命中**。
    ///
    /// 但它有代价，所以默认**关闭**：
    ///
    /// 1. **改写了发给模型的 prompt** —— 模型看到的 `call_id` 不再是
    ///    provider 生成的原始值。实践中 provider 不关心这个（只做配对），
    ///    但这毕竟是对请求内容的改动。
    /// 2. **需要额外的对账保证** —— assistant 消息和 tool 回执必须
    ///    **一起**重写，否则会触发 16 的坑 1（少一条回执 → 400）。
    /// 3. **改变了语义** —— 同一个问题在两次运行里用的是同一个 ID，
    ///    而 provider 的自动缓存（prompt caching）可能按 ID 做前缀分片。
    ///
    /// 所以它是**可对照的实验选项**，而不是「偷偷改掉的行为」。
    normalize_call_id: bool,

    /// 是否缓存**中间轮**（含 `tool_calls` 的那些轮次）。
    ///
    /// ## ⚠️ 这是 agentic 链路能否命中的**决定性开关**
    ///
    /// 关掉时（只有最后一轮写缓存）：
    ///
    /// ```text
    /// 第 1 步不写缓存 → 每次都真调模型
    ///   → 模型每次的措辞不同（"我查一下知识库" vs "让我检索相关资料"）
    ///   → 第 2 步的历史不同 → 第 2 步的键不同 → 必然 miss
    /// ```
    ///
    /// 也就是说，**中间轮的不稳定性会顺着消息历史污染后面所有轮次**。
    ///
    /// 打开时，第 1 步命中缓存 → assistant 文本被**完整还原**（措辞不再变）
    /// → 第 2 步的历史逐字节相同 → 第 2 步也命中。
    ///
    /// ⚠️ 代价：命中中间轮 = **跳过了工具执行**（工具结果被冻结）。
    /// 工具结果自己还有一层缓存兜底，但时效性会下降。
    cache_intermediate_rounds: bool,
}

impl std::fmt::Debug for CachedAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedAgent")
            .field("tools", &self.registry.names())
            .field("model", &self.config.model)
            .field("cache", &self.cache)
            .finish()
    }
}

impl CachedAgent {
    /// 创建 Agent。
    pub fn new(
        llm: Arc<dyn agent_16::llm::FunctionLlm>,
        registry: agent_16::tool::ToolRegistry,
        cache: Arc<CacheService>,
        config: AppConfig,
    ) -> Self {
        Self {
            llm,
            registry,
            cache,
            config,
            tool_ttl: 60,
            normalize_call_id: false,
            cache_intermediate_rounds: false,
        }
    }

    /// 设置工具结果缓存 TTL。
    pub fn with_tool_ttl(mut self, seconds: u64) -> Self {
        self.tool_ttl = seconds;
        self
    }

    /// ★ 启用**确定性 `call_id`**（让 agentic 链路也能命中精确匹配缓存）。
    ///
    /// ## 它做了什么
    ///
    /// 模型每次返回的工具调用 ID 都是随机的（`call_abc123`），
    /// 于是「同一个问题跑两次」的缓存键完全不同 → 必然 miss。
    ///
    /// 开启后，ID 被**重写**成内容派生的确定性值：
    ///
    /// ```text
    /// call_id = "call_" + sha256(步数 | 工具名 | 规范化参数 | 同参序号)[..16]
    /// ```
    ///
    /// ## ⚠️ 为什么是「重写」而不是「从键里剔除」
    ///
    /// 两种做法看起来等价，实际差别很大：
    ///
    /// | 做法 | 模型看到的 prompt | 键是否稳定 |
    /// |---|---|---|
    /// | 从**键**里剔除 `call_id` | ID 文本仍在变 | ✅ 稳定 |
    /// | **重写消息本身** | ID 文本也固定 | ✅ 稳定 |
    ///
    /// 重写更彻底：模型每次看到的请求**逐字节相同**，
    /// 这对 provider 的自动 prompt caching 也友好。
    ///
    /// ## ⚠️ 对账安全性
    ///
    /// 16 的坑 1：「`call_id` 是 tool 消息回执的对账依据，少一条回执就 400」。
    ///
    /// 本实现**同时**重写 assistant 消息里的 `tool_call_id`
    /// 和 tool 回执里的 `tool_call_id` —— 配对关系不变，
    /// 所以不会触发 400。
    ///
    /// 回归测试 `normalized_call_ids_still_pair_up_with_tool_receipts` 锁住这个语义。
    pub fn with_normalized_call_id(mut self, enabled: bool) -> Self {
        self.normalize_call_id = enabled;
        self
    }

    /// 是否启用了确定性 `call_id`。
    pub fn normalize_call_id(&self) -> bool {
        self.normalize_call_id
    }

    /// ★ 设置是否缓存**中间轮**（含 `tool_calls` 的轮次）。
    ///
    /// ## 它为什么是决定性的
    ///
    /// 关掉时，中间轮每次都真调模型，而模型每次的**措辞**不同 ——
    /// 这个差异会顺着消息历史污染后面所有轮次，让整条链路的键都不稳定。
    ///
    /// 打开后，第 1 步命中缓存 → assistant 文本被完整还原（措辞不再变）
    /// → 第 2 步的历史逐字节相同 → 第 2 步也命中。
    ///
    /// ⚠️ 代价：命中中间轮 = 跳过了工具执行（工具结果被冻结）。
    pub fn with_cache_intermediate_rounds(mut self, enabled: bool) -> Self {
        self.cache_intermediate_rounds = enabled;
        self
    }

    /// 是否缓存中间轮。
    pub fn cache_intermediate_rounds(&self) -> bool {
        self.cache_intermediate_rounds
    }

    /// 工具注册表。
    pub fn registry(&self) -> &agent_16::tool::ToolRegistry {
        &self.registry
    }

    /// 缓存服务。
    pub fn cache(&self) -> &Arc<CacheService> {
        &self.cache
    }

    /// 跑一次 Agent。
    ///
    /// ## 循环骨架
    ///
    /// ```text
    /// ┌────────────────────────────────────────────────────┐
    /// │ 1. 组装 messages（system + user）                 │
    /// │ 2. 算缓存键                                        │
    /// │ 3. 查缓存                                          │
    /// │    ├─ 命中 → 用缓存响应（source=CacheHit）          │
    /// │    └─ 未命中 → 调模型 → 写缓存（source=Live）       │
    /// │ 4. 响应里有 tool_calls？                           │
    /// │    ├─ 没有 → 最终答案，收工                        │
    /// │    └─ 有 → 逐个执行工具（先查工具结果缓存）        │
    /// │ 5. 为每个 call_id 生成恰好一条 tool 消息           │
    /// │ 6. 回到 1                                          │
    /// └────────────────────────────────────────────────────┘
    /// ```
    pub async fn run(&self, question: &str) -> AppResult<AgentRun> {
        self.run_with_control(question, &CacheControl::default())
            .await
    }

    /// 带缓存控制跑一次 Agent。
    pub async fn run_with_control(
        &self,
        question: &str,
        control: &CacheControl,
    ) -> AppResult<AgentRun> {
        let start = Instant::now();
        let question = question.trim();

        if question.is_empty() {
            return Err(AppError::BadRequest("问题不能为空".to_string()));
        }

        let specs = self.registry.specs();
        let system = agent_16::prompt::system_prompt(&specs, self.config.prompt.strict);
        let digest = tools_digest(&specs);

        let mut messages: Vec<ChatMessage> = vec![ChatMessage::user(question.to_string())];

        let mut steps: Vec<StepRecord> = Vec::new();
        let mut tool_calls = 0usize;
        let mut tools_used: Vec<String> = Vec::new();
        let mut usage = TokenUsage::default();
        let mut cache_hits = 0usize;
        let mut cache_misses = 0usize;

        for step_no in 1..=self.config.agent.max_steps {
            // ── 算键 ────────────────────────────────────────────
            //
            // ⚠️ 键必须包含**所有**影响输出的东西。
            //    漏一个字段 = 两个本该不同的请求返回同一个答案。
            let key = CacheKey::new(
                self.cache.namespace(),
                &self.config.model,
                &messages,
                &digest,
                self.config.prompt.tool_choice.name(),
                self.config.temperature,
                self.config.max_tokens,
                self.config.disable_reasoning,
            );
            let redis_key = key.redis_key();

            // ── 查缓存 ──────────────────────────────────────────
            let cached = self.cache.get_response(&key, control).await;

            let (response, source) = match cached {
                Some(cached_response) => {
                    cache_hits += 1;
                    debug!(
                        step = step_no,
                        key = %redis_key,
                        "★ 命中缓存 —— 不调模型"
                    );
                    (
                        cached_response_to_chat_response(&cached_response),
                        CallSource::CacheHit,
                    )
                }
                None => {
                    cache_misses += 1;
                    let response = self
                        .llm
                        .chat_with_tools(
                            &system,
                            &messages,
                            &specs,
                            self.config.prompt.tool_choice.clone(),
                        )
                        .await?;

                    // ── 写缓存 ──────────────────────────────────────
                    //
                    // ⚠️ **中间轮也要写**（这是让 agentic 链路能命中的关键）。
                    //
                    // 原来的规则是「只写纯文本的最终答案那一轮」，
                    // 理由是「中间轮的键含 `call_id`，几乎必然 miss」。
                    //
                    // 但这个理由**只说对了一半** —— 真正的伤害不是「多存一条没用的」，
                    // 而是：
                    //
                    // ```text
                    // 第 1 步不写缓存 → 每次都真调模型
                    //   → 模型每次的措辞不同（"我查一下知识库" vs "让我检索相关资料"）
                    //   → 第 2 步的历史不同 → 第 2 步的键不同 → 必然 miss
                    // ```
                    //
                    // 也就是说：**中间轮的不稳定性会顺着消息历史污染后面所有轮次**。
                    // 只缓存最后一轮，等于把 agentic 链路的命中率锁死在 0。
                    //
                    // ⚠️ 但这也带来一个**必须承认的代价**：
                    // 命中中间轮缓存 = **跳过了工具执行**。
                    // 工具结果会被冻结（工具结果自己还有一层缓存兜底）。
                    // 这是「轮级缓存 → 会话级缓存」的滑动 ——
                    // 详见 README「三档缓存语义」。
                    //
                    // ⚠️ 所以这是个**开关** —— 代价明确，应该由使用者显式决定。
                    let is_final_round = response.tool_calls().is_empty();
                    if is_final_round || self.cache_intermediate_rounds {
                        let snapshot = chat_response_to_cached(&response, &key);
                        self.cache.set_response(&key, &snapshot, control).await;
                    }

                    usage.add(&response);
                    (response, CallSource::Live)
                }
            };

            let text = response.first_text().unwrap_or_default().to_string();
            let reasoning = response.reasoning_content.clone();
            let mut calls: Vec<ToolCall> = response.into_tool_calls();

            // ── ★ 重写 `call_id`（可选）────────────────────────────
            //
            // ⚠️ **为什么在这里改，而不是在键计算时改**：
            //
            // 模型返回的 `call_id` 是随机的（`call_abc123`），
            // 它会同时出现在两处：
            //
            // 1. assistant 消息里的 `tool_call_id`
            // 2. tool 回执里的 `tool_call_id`
            //
            // 下一轮的消息历史由这两处组成 → 键里含随机 ID → 必然 miss。
            //
            // 而**只**改其中一处会破坏对账（16 的坑 1：少一条回执就 400）。
            //
            // 所以这里**一次性改 `calls` 里的 ID** ——
            // 下面的 `build_assistant_message` 和 `ToolResponse::from_tool_call`
            // 都是从 `calls` 取 ID 的，所以**两处自动同步**，
            // 不可能出现「改了一处忘了另一处」的不一致。
            if self.normalize_call_id && !calls.is_empty() {
                assign_deterministic_call_ids(step_no, &mut calls, |call| {
                    agent_16::agent::normalize_arguments(&call.fn_arguments)
                });
                debug!(
                    step = step_no,
                    ids = ?calls.iter().map(|c| c.call_id.clone()).collect::<Vec<_>>(),
                    "★ 已重写为确定性 call_id"
                );
            }

            // ── 没有工具调用 → 最终答案 ─────────────────────────
            if calls.is_empty() {
                steps.push(StepRecord {
                    step: step_no,
                    text: text.clone(),
                    reasoning: reasoning.clone(),
                    tool_calls: Vec::new(),
                    final_answer: true,
                    source,
                    cache_key: redis_key,
                });

                info!(
                    step = step_no,
                    source = ?source,
                    cache_hits,
                    cache_misses,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "Agent 给出最终答案"
                );

                return Ok(AgentRun {
                    question: question.to_string(),
                    answer: text,
                    steps,
                    tool_calls,
                    tools_used,
                    hit_max_steps: false,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    model: self.config.model.clone(),
                    usage,
                    cache_hits,
                    cache_misses,
                });
            }

            // ── 有工具调用 ─────────────────────────────────────
            messages.push(build_assistant_message(
                text.clone(),
                reasoning.clone(),
                &calls,
            ));

            let mut records = Vec::with_capacity(calls.len());
            let mut responses: Vec<ToolResponse> = Vec::with_capacity(calls.len());

            for call in &calls {
                let (outcome, cached_tool) = self.execute_call(call).await;

                if !outcome.ok {
                    warn!(
                        step = step_no,
                        tool = %call.fn_name,
                        observation = %outcome.observation,
                        "工具执行失败（错误已作为 tool 消息喂回模型）"
                    );
                }

                if !tools_used.contains(&call.fn_name) {
                    tools_used.push(call.fn_name.clone());
                }

                records.push(ToolCallRecord {
                    call_id: call.call_id.clone(),
                    name: call.fn_name.clone(),
                    arguments: call.fn_arguments.clone(),
                    observation: outcome.observation.clone(),
                    ok: outcome.ok,
                    data: outcome.data.clone(),
                    cached: cached_tool,
                });

                responses.push(ToolResponse::from_tool_call(call, outcome.observation));
                tool_calls += 1;
            }

            messages.push(ChatMessage::tool(
                genai::chat::MessageContent::from_tool_responses(responses),
            ));

            steps.push(StepRecord {
                step: step_no,
                text,
                reasoning,
                tool_calls: records,
                final_answer: false,
                source,
                cache_key: redis_key,
            });
        }

        // ── 达到步数上限 ────────────────────────────────────────
        warn!(
            max_steps = self.config.agent.max_steps,
            tool_calls, "达到步数上限 —— 降级返回"
        );

        let answer = steps
            .last()
            .and_then(|s| s.tool_calls.first())
            .map(|c| c.observation.clone())
            .unwrap_or_else(|| {
                format!(
                    "达到最大步数（{}）仍未得到最终答案。请缩小问题范围后重试。",
                    self.config.agent.max_steps
                )
            });

        Ok(AgentRun {
            question: question.to_string(),
            answer,
            steps,
            tool_calls,
            tools_used,
            hit_max_steps: true,
            elapsed_ms: start.elapsed().as_millis() as u64,
            model: self.config.model.clone(),
            usage,
            cache_hits,
            cache_misses,
        })
    }

    /// 执行一个工具调用（先查工具结果缓存）。
    ///
    /// 返回 `(结果, 是否来自缓存)`。
    async fn execute_call(&self, call: &ToolCall) -> (agent_16::tool::ToolOutcome, bool) {
        let args = agent_16::agent::normalize_arguments(&call.fn_arguments);

        // ── 查工具结果缓存 ────────────────────────────────────
        //
        // ⚠️ 只有**白名单里的工具**才会走到这里 ——
        //    `is_cacheable_tool` 对不在白名单的工具直接返回 false。
        if self.cache.is_cacheable_tool(&call.fn_name)
            && let Some(hit) = self.cache.get_tool_result(&call.fn_name, &args).await
        {
            debug!(
                tool = %call.fn_name,
                "★ 工具结果命中缓存 —— 跳过工具执行"
            );
            return (
                agent_16::tool::ToolOutcome {
                    observation: hit.observation.clone(),
                    ok: true,
                    tool: hit.tool.clone(),
                    data: hit.data.clone(),
                },
                true,
            );
        }

        let outcome = self
            .registry
            .call(&call.fn_name, &args, &agent_16::tool::ToolContext::empty())
            .await;

        // ── 写工具结果缓存 ────────────────────────────────────
        //
        // ⚠️ **只缓存成功的** —— 失败的工具结果（错误文本）
        //    缓存下来会让「模型自我纠正」的流程变成「反复看到同一个错误」。
        if outcome.ok && self.cache.is_cacheable_tool(&call.fn_name) {
            let result = CachedToolResult {
                tool: outcome.tool.clone(),
                observation: outcome.observation.clone(),
                data: outcome.data.clone(),
                stored_at_ms: crate::cache::now_ms(),
            };
            self.cache
                .set_tool_result(&call.fn_name, &args, &result)
                .await;
        }

        (outcome, false)
    }
}

/// 把 `ChatResponse` 转成缓存快照。
pub fn chat_response_to_cached(response: &ChatResponse, key: &CacheKey) -> CachedResponse {
    let prompt = response.usage.prompt_tokens.unwrap_or(0) as u64;
    let completion = response.usage.completion_tokens.unwrap_or(0) as u64;

    CachedResponse {
        text: response.first_text().unwrap_or_default().to_string(),
        // ⚠️ **中间轮也要存** —— 否则中间轮永远真调模型，
        // 而它的措辞差异会顺着消息历史污染后面所有轮次。
        tool_calls: response
            .tool_calls()
            .iter()
            .map(|call| CachedToolCall {
                call_id: call.call_id.clone(),
                name: call.fn_name.clone(),
                arguments: call.fn_arguments.clone(),
            })
            .collect(),
        reasoning: response.reasoning_content.clone(),
        usage: CachedUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: response
                .usage
                .total_tokens
                .unwrap_or((prompt + completion) as i32) as u64,
        },
        model: key.model.clone(),
        stored_at_ms: crate::cache::now_ms(),
        key_desc: key.describe(),
    }
}

/// 把缓存快照还原成 `ChatResponse`。
///
/// ## ⚠️ 为什么要还原成 `ChatResponse` 而不是自定义结构
///
/// 因为下游的 Agent 循环逻辑（提取文本、提取 tool_calls、读 usage）
/// 和 16 完全一样 —— **复用 16 的循环 = 少一处可能出错的地方**。
///
/// 缓存层对 Agent 循环是**透明**的：循环不知道也不关心
/// 这次响应是从缓存来的还是模型来的。
pub fn cached_response_to_chat_response(cached: &CachedResponse) -> ChatResponse {
    use genai::chat::{ContentPart, MessageContent};

    let mut parts: Vec<ContentPart> = Vec::new();

    if let Some(reasoning) = &cached.reasoning
        && !reasoning.is_empty()
    {
        parts.push(ContentPart::ReasoningContent(reasoning.clone()));
    }

    if !cached.text.is_empty() {
        parts.push(ContentPart::Text(cached.text.clone()));
    }

    // ⚠️ **必须还原 `tool_calls`** —— 否则中间轮命中缓存后，
    //    循环会以为「模型没有调工具」，直接拿空文本当最终答案。
    //
    //    顺序：**文本在前，工具调用在后**（和 `build_assistant_message` 一致）。
    for call in &cached.tool_calls {
        parts.push(ContentPart::ToolCall(genai::chat::ToolCall {
            call_id: call.call_id.clone(),
            fn_name: call.name.clone(),
            fn_arguments: call.arguments.clone(),
            thought_signatures: None,
        }));
    }

    ChatResponse {
        content: MessageContent::from_parts(parts),
        reasoning_content: cached.reasoning.clone(),
        model_iden: genai::ModelIden::new(genai::adapter::AdapterKind::OpenAI, &cached.model),
        provider_model_iden: genai::ModelIden::new(
            genai::adapter::AdapterKind::OpenAI,
            &cached.model,
        ),
        stop_reason: None,
        usage: genai::chat::Usage {
            prompt_tokens: Some(cached.usage.prompt_tokens as i32),
            completion_tokens: Some(cached.usage.completion_tokens as i32),
            prompt_tokens_details: None,
            completion_tokens_details: None,
            total_tokens: Some(cached.usage.total_tokens as i32),
        },
        captured_raw_body: None,
        response_id: None,
    }
}

/// 构造 assistant 消息（带工具调用 + 思考内容）。
///
/// 复用 16 的语义：**思考内容在前，工具调用在后**。
fn build_assistant_message(
    text: String,
    reasoning: Option<String>,
    calls: &[ToolCall],
) -> ChatMessage {
    use genai::chat::ContentPart;

    let mut parts: Vec<ContentPart> = Vec::new();

    if let Some(reasoning) = reasoning
        && !reasoning.is_empty()
    {
        parts.push(ContentPart::ReasoningContent(reasoning));
    }

    if !text.is_empty() {
        parts.push(ContentPart::Text(text));
    }

    for call in calls {
        parts.push(ContentPart::ToolCall(call.clone()));
    }

    ChatMessage::assistant(genai::chat::MessageContent::from_parts(parts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CacheService, LocalCache};
    use agent_16::llm::ScriptedStep;
    use agent_16::tool::ToolRegistry;
    use serde_json::json;

    fn config() -> AppConfig {
        AppConfig::default()
    }

    fn cache_service() -> Arc<CacheService> {
        Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60))
    }

    /// 一个总是返回同一段文本的假工具。
    struct EchoTool;

    impl agent_16::tool::FunctionTool for EchoTool {
        fn spec(&self) -> agent_16::tools::ToolSpec {
            agent_16::tools::ToolSpec::new(
                "echo",
                "回显文本",
                json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string" }
                    },
                    "required": ["text"]
                }),
            )
        }

        fn call<'a>(
            &'a self,
            args: &'a Value,
            _ctx: &'a agent_16::tool::ToolContext,
        ) -> agent_16::tool::ToolFuture<'a> {
            Box::pin(async move {
                let text = args
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                Ok(agent_16::tool::ToolOutcome::ok("echo", text))
            })
        }
    }

    /// 可重复的测试 LLM —— 按调用次数循环返回预设步骤。
    ///
    /// ## 为什么不用 16 的 `ScriptedFunctionLlm`
    ///
    /// 它的脚本是**消耗式**的（`steps.remove(0)`），
    /// 跑第二遍就没脚本了 —— 而缓存测试**必须**能重复跑同一个问题。
    ///
    /// 所以这里自己写一个「循环取模」的替身。
    struct LoopingLlm {
        steps: Vec<ScriptedStep>,
        calls: std::sync::Mutex<usize>,
    }

    impl LoopingLlm {
        fn new(steps: Vec<ScriptedStep>) -> Self {
            Self {
                steps,
                calls: std::sync::Mutex::new(0),
            }
        }

        fn call_count(&self) -> usize {
            *self.calls.lock().unwrap_or_else(|e| e.into_inner())
        }
    }

    impl agent_16::llm::FunctionLlm for LoopingLlm {
        fn chat_with_tools<'a>(
            &'a self,
            _system: &'a str,
            _messages: &'a [ChatMessage],
            _tools: &'a [agent_16::tools::ToolSpec],
            _tool_choice: agent_16::llm::ToolChoiceMode,
        ) -> agent_16::llm::FunctionLlmFuture<'a> {
            let step = {
                let mut calls = self.calls.lock().unwrap_or_else(|e| e.into_inner());
                let index = *calls % self.steps.len().max(1);
                *calls += 1;
                self.steps.get(index).cloned()
            };

            Box::pin(async move {
                let step = step
                    .ok_or_else(|| agent_16::AppError::Llm("测试 LLM 没有预设响应".to_string()))?;
                Ok(step.to_response())
            })
        }

        fn model_name(&self) -> &str {
            "looping"
        }
    }

    fn registry() -> ToolRegistry {
        ToolRegistry::new(vec![Arc::new(EchoTool)])
    }

    /// 模拟**真实模型**的行为：每次调用都返回**不同的 `call_id`**。
    ///
    /// ## 为什么需要这个替身
    ///
    /// 真实模型的 `call_id` 是随机生成的（`call_abc123`）。
    /// 而 `LoopingLlm` 每次返回同样的 `call_id` ——
    /// 那会让缓存键稳定，**掩盖**了本项目最核心的结论。
    ///
    /// 这个替身让「call_id 每轮都变」这件事在测试里**可见**。
    ///
    /// ## ⚠️ 它还模拟了真实模型的第二个特征：**措辞也变**
    ///
    /// 真实模型在调工具时**通常会附带一句自然语言**
    /// （"我查一下知识库" / "让我检索相关资料"）。
    /// 那句话会进 assistant 消息 → 进下一轮的消息历史 → 进缓存键。
    ///
    /// ⚠️ 如果测试里**不**模拟这个特征，就会得出错误结论：
    /// 「只重写 `call_id` 就够了」。实测证明**不够** ——
    /// 真正的主因是这个措辞差异。
    ///
    /// 所以这里用 `TextAndCalls` 带上**每次不同**的文本。
    struct VariedCallIdLlm {
        counter: std::sync::Mutex<u32>,
    }

    impl VariedCallIdLlm {
        fn new() -> Self {
            Self {
                counter: std::sync::Mutex::new(0),
            }
        }
    }

    impl agent_16::llm::FunctionLlm for VariedCallIdLlm {
        fn chat_with_tools<'a>(
            &'a self,
            _system: &'a str,
            messages: &'a [ChatMessage],
            _tools: &'a [agent_16::tools::ToolSpec],
            _tool_choice: agent_16::llm::ToolChoiceMode,
        ) -> agent_16::llm::FunctionLlmFuture<'a> {
            let n = {
                let mut counter = self.counter.lock().unwrap_or_else(|e| e.into_inner());
                *counter += 1;
                *counter
            };

            Box::pin(async move {
                // 只有「还没调过工具」的那一轮返回工具调用。
                let already_called = messages.iter().any(|m| !m.content.tool_calls().is_empty());

                if already_called {
                    return Ok(ScriptedStep::Text("最终答案".to_string()).to_response());
                }

                // ⚠️ **同时**模拟两个真实特征：
                // ① 随机的 `call_id`
                // ② 每次都不同的自由文本（真实模型的自然语言铺垫）
                Ok(ScriptedStep::TextAndCalls(
                    format!("让我检索一下相关资料（第 {n} 次尝试）"),
                    vec![agent_16::llm::ScriptedToolCall::new(
                        &format!("call_{n}"),
                        "echo",
                        json!({"text": "hi"}),
                    )],
                )
                .to_response())
            })
        }

        fn model_name(&self) -> &str {
            "varied-call-id"
        }
    }

    #[tokio::test]
    async fn second_identical_run_hits_the_cache() {
        // 第一次：真实调用。
        let llm = Arc::new(LoopingLlm::new(vec![ScriptedStep::Text(
            "缓存的答案".to_string(),
        )]));
        let cache = cache_service();
        let agent = CachedAgent::new(llm.clone(), registry(), cache.clone(), config());

        let first = agent.run("同一个问题").await.unwrap();
        assert_eq!(first.cache_hits, 0);
        assert_eq!(first.cache_misses, 1);
        assert_eq!(first.steps[0].source, CallSource::Live);

        // 第二次：命中缓存，**模型不会被调用第二次**。
        let second = agent.run("同一个问题").await.unwrap();
        assert_eq!(second.cache_hits, 1);
        assert_eq!(second.cache_misses, 0);
        assert_eq!(second.steps[0].source, CallSource::CacheHit);
        assert_eq!(second.answer, first.answer);

        // ⚠️ 关键断言：模型**只被调用了一次**。
        assert_eq!(llm.call_count(), 1);
    }

    #[tokio::test]
    async fn different_questions_do_not_share_the_cache() {
        let llm = Arc::new(LoopingLlm::new(vec![ScriptedStep::Text(
            "答案".to_string(),
        )]));
        let cache = cache_service();
        let agent = CachedAgent::new(llm.clone(), registry(), cache.clone(), config());

        agent.run("问题 A").await.unwrap();
        let second = agent.run("问题 B").await.unwrap();

        assert_eq!(second.cache_hits, 0);
        assert_eq!(llm.call_count(), 2);
    }

    #[tokio::test]
    async fn no_cache_forces_a_real_call_but_still_writes() {
        let llm = Arc::new(LoopingLlm::new(vec![ScriptedStep::Text(
            "答案".to_string(),
        )]));
        let cache = cache_service();
        let agent = CachedAgent::new(llm.clone(), registry(), cache.clone(), config());

        // 第一次：写缓存。
        agent.run("问题").await.unwrap();

        // 第二次：no-cache → 强制真调。
        let second = agent
            .run_with_control("问题", &CacheControl::bypass_read())
            .await
            .unwrap();
        assert_eq!(second.cache_hits, 0);
        assert_eq!(llm.call_count(), 2);

        // 第三次：缓存里**确实有**（因为 no-cache 也会写）。
        let third = agent.run("问题").await.unwrap();
        assert_eq!(third.cache_hits, 1);
        assert_eq!(llm.call_count(), 2);
    }

    #[tokio::test]
    async fn no_store_never_writes() {
        let llm = Arc::new(LoopingLlm::new(vec![ScriptedStep::Text(
            "答案".to_string(),
        )]));
        let cache = cache_service();
        let agent = CachedAgent::new(llm.clone(), registry(), cache.clone(), config());

        agent
            .run_with_control("问题", &CacheControl::read_only())
            .await
            .unwrap();

        let second = agent.run("问题").await.unwrap();
        assert_eq!(second.cache_hits, 0);
        assert_eq!(llm.call_count(), 2);
    }

    #[tokio::test]
    async fn disabled_cache_never_hits() {
        let llm = Arc::new(LoopingLlm::new(vec![ScriptedStep::Text(
            "答案".to_string(),
        )]));
        let cache = Arc::new(CacheService::disabled(Arc::new(LocalCache::new())));
        let agent = CachedAgent::new(llm.clone(), registry(), cache, config());

        agent.run("问题").await.unwrap();
        let second = agent.run("问题").await.unwrap();

        assert_eq!(second.cache_hits, 0);
        assert_eq!(llm.call_count(), 2);
    }

    #[tokio::test]
    async fn cached_response_roundtrips_through_chat_response() {
        let cached = CachedResponse {
            tool_calls: Vec::new(),
            text: "你好".to_string(),
            reasoning: Some("思考".to_string()),
            usage: CachedUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            model: "deepseek-flash".to_string(),
            stored_at_ms: 1,
            key_desc: "x".to_string(),
        };

        let response = cached_response_to_chat_response(&cached);
        assert_eq!(response.first_text(), Some("你好"));
        assert_eq!(response.reasoning_content.as_deref(), Some("思考"));
        assert_eq!(response.usage.total_tokens, Some(15));
    }

    #[tokio::test]
    async fn tool_results_are_cached_when_whitelisted() {
        let llm = Arc::new(LoopingLlm::new(vec![
            ScriptedStep::call("c1", "echo", json!({"text": "hi"})),
            ScriptedStep::Text("最终答案".to_string()),
        ]));

        let cache = Arc::new(
            CacheService::new(Arc::new(LocalCache::new()), 60)
                .with_cacheable_tools(vec!["echo".to_string()]),
        );

        let agent = CachedAgent::new(llm.clone(), registry(), cache.clone(), config());

        let first = agent.run("问题").await.unwrap();
        assert_eq!(first.tool_calls, 1);
        assert!(!first.steps[0].tool_calls[0].cached);

        // ⚠️ 第二次运行：**工具结果**命中了缓存（工具没被执行第二次）。
        let second = agent.run("问题").await.unwrap();
        assert!(second.steps[0].tool_calls[0].cached);
    }

    #[tokio::test]
    async fn without_intermediate_cache_the_agentic_chain_misses() {
        // ⚠️ 这是**档 1**（原始实现）的行为 —— 也是本项目原先记录的结论。
        //
        // 带工具的链路里：
        //
        // - 第 1 轮（模型决定调工具）的消息历史 = [user 问题]
        //   → 键是稳定的，**但默认不写这一轮的缓存**
        // - 第 2 轮（最终答案）的消息历史 = [user, assistant(tool_calls), tool]
        //   → 键里含 `call_id`，**每次运行都不同** → 必然 miss
        //
        // ⚠️ **但这个结论后来被推翻了** —— 真正的元凶不是 `call_id`，
        // 而是「中间轮不写缓存」导致**模型每次措辞不同**，
        // 顺着消息历史污染了后面所有轮次。
        //
        // 下一条测试 `caching_intermediate_rounds_makes_the_agentic_chain_hit`
        // 就是修正后的结论（档 2）。
        //
        // 这里用**不同的 call_id** 来精确模拟真实模型的行为。
        let llm = Arc::new(VariedCallIdLlm::new());

        let cache = Arc::new(
            CacheService::new(Arc::new(LocalCache::new()), 60)
                .with_cacheable_tools(vec!["echo".to_string()]),
        );

        let agent = CachedAgent::new(llm.clone(), registry(), cache, config());

        agent.run("问题").await.unwrap();
        let second = agent.run("问题").await.unwrap();

        // ⚠️ 最终答案那一轮**没有命中**（中间轮没缓存 → 措辞每次不同）。
        assert_eq!(second.cache_hits, 0);
        // 工具结果命中了（工具是纯函数）。
        assert!(second.steps[0].tool_calls[0].cached);
    }

    #[tokio::test]
    async fn caching_intermediate_rounds_makes_the_agentic_chain_hit() {
        // ⚠️ **这是修正后的核心结论**（档 2）。
        //
        // 只要**缓存中间轮**，agentic 链路就能 100% 命中 ——
        // **不需要**重写 `call_id`。
        //
        // 原因：
        //
        // ```text
        // 第 1 步命中缓存 → assistant 文本被完整还原（措辞不再变）
        //   → 第 2 步的历史逐字节相同 → 第 2 步也命中
        // ```
        let llm = Arc::new(VariedCallIdLlm::new());

        let cache = Arc::new(
            CacheService::new(Arc::new(LocalCache::new()), 60)
                .with_cacheable_tools(vec!["echo".to_string()]),
        );

        let agent = CachedAgent::new(llm.clone(), registry(), cache, config())
            .with_cache_intermediate_rounds(true);

        agent.run("问题").await.unwrap();
        let second = agent.run("问题").await.unwrap();

        // ⚠️ 两轮**全部命中**（档 1 时是 0）。
        assert_eq!(second.cache_hits, 2);
        assert_eq!(second.cache_misses, 0);
        // 工具结果也命中了。
        assert!(second.steps[0].tool_calls[0].cached);
    }

    /// 记录每次调用**看到的完整消息历史**的 LLM 替身。
    ///
    /// ## 为什么需要它
    ///
    /// 「重写 `call_id` 后回执是否仍然配对」这个问题，
    /// **必须检查真正发出去的请求**才能回答 ——
    /// 只看内部数据结构，测不到「发给 provider 的消息对不对」这个真正的风险点。
    struct RecordingLlm {
        /// 每次调用看到的消息历史。
        histories: std::sync::Mutex<Vec<Vec<ChatMessage>>>,
    }

    impl RecordingLlm {
        fn new() -> Self {
            Self {
                histories: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn histories(&self) -> Vec<Vec<ChatMessage>> {
            self.histories
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        }
    }

    impl agent_16::llm::FunctionLlm for RecordingLlm {
        fn chat_with_tools<'a>(
            &'a self,
            _system: &'a str,
            messages: &'a [ChatMessage],
            _tools: &'a [agent_16::tools::ToolSpec],
            _tool_choice: agent_16::llm::ToolChoiceMode,
        ) -> agent_16::llm::FunctionLlmFuture<'a> {
            self.histories
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(messages.to_vec());

            Box::pin(async move {
                let already_called = messages.iter().any(|m| !m.content.tool_calls().is_empty());
                if already_called {
                    return Ok(ScriptedStep::Text("最终答案".to_string()).to_response());
                }
                Ok(
                    ScriptedStep::call("call_random_9999", "echo", json!({"text": "hi"}))
                        .to_response(),
                )
            })
        }

        fn model_name(&self) -> &str {
            "recording"
        }
    }

    /// 抽取一条消息历史里的 `(assistant 的 call_id, tool 回执的 call_id)`。
    fn extract_call_id_pairs(messages: &[ChatMessage]) -> Vec<(String, String)> {
        let mut assistant_ids: Vec<String> = Vec::new();
        let mut receipt_ids: Vec<String> = Vec::new();

        for msg in messages {
            for call in msg.content.tool_calls() {
                assistant_ids.push(call.call_id.clone());
            }
            for part in msg.content.parts() {
                if let genai::chat::ContentPart::ToolResponse(resp) = part {
                    receipt_ids.push(resp.call_id.clone());
                }
            }
        }

        assistant_ids
            .into_iter()
            .zip(receipt_ids)
            .collect::<Vec<_>>()
    }

    #[tokio::test]
    async fn normalized_call_ids_still_pair_up_with_tool_receipts() {
        // ⚠️ **最关键的安全断言**。
        //
        // 16 的坑 1：「`call_id` 是 tool 消息回执的对账依据，少一条回执就 400」。
        //
        // 本实现重写了 `call_id` —— 所以必须验证：
        // **assistant 消息里的 ID 与 tool 回执里的 ID 仍然一一对应**。
        //
        // 如果对不上，发给 provider 的请求会畸形（400 或静默错配）。
        let llm = Arc::new(RecordingLlm::new());
        let cache = Arc::new(
            CacheService::new(Arc::new(LocalCache::new()), 60)
                .with_cacheable_tools(vec!["echo".to_string()]),
        );

        let agent = CachedAgent::new(llm.clone(), registry(), cache, config())
            .with_normalized_call_id(true);

        agent.run("问题").await.unwrap();

        let histories = llm.histories();
        assert!(histories.len() >= 2, "应该至少有两次模型调用");

        // 第二轮的消息历史里含 assistant(tool_calls) + tool(回执)。
        let second_round = &histories[1];
        let pairs = extract_call_id_pairs(second_round);
        assert_eq!(pairs.len(), 1, "应该有一对 (assistant, 回执)");

        // ⚠️ 核心断言：两边的 ID **完全相同**。
        let (assistant_id, receipt_id) = &pairs[0];
        assert_eq!(
            assistant_id, receipt_id,
            "重写 call_id 后，assistant 与 tool 回执必须仍然配对"
        );

        // ⚠️ 而且 ID 已经被**重写**成了确定性形式（不再是原始的随机值）。
        assert!(
            assistant_id.starts_with("call_") && !assistant_id.contains("random"),
            "call_id 应该被重写成确定性形式，实际是: {assistant_id}"
        );
    }

    #[tokio::test]
    async fn normalized_call_id_alone_does_not_fix_the_agentic_chain() {
        // ⚠️ **这是一个「修正后被推翻的假设」的回归测试**。
        //
        // 最初的假设是：「agentic 链路不命中，是因为 `call_id` 每次都变」。
        // 于是实现了「重写 `call_id`」。
        //
        // 但**实测证明这个假设是错的** —— 只重写 `call_id`（不开中间轮缓存），
        // agentic 链路**依然不命中**。
        //
        // 真正的元凶是「中间轮不写缓存 → 模型每次措辞不同 → 污染后续轮次」。
        // 见 `caching_intermediate_rounds_makes_the_agentic_chain_hit`。
        //
        // 所以这条测试锁住的是「**单独开 call_id 归一化不够**」这个事实 ——
        // 防止有人以为「重写 call_id 就解决了问题」而把中间轮缓存关掉。
        let llm = Arc::new(VariedCallIdLlm::new());

        let cache = Arc::new(
            CacheService::new(Arc::new(LocalCache::new()), 60)
                .with_cacheable_tools(vec!["echo".to_string()]),
        );

        let agent = CachedAgent::new(llm.clone(), registry(), cache, config())
            .with_normalized_call_id(true);

        agent.run("问题").await.unwrap();
        let second = agent.run("问题").await.unwrap();

        // ⚠️ 只开 call_id 归一化**不足以**让 agentic 链路命中。
        assert_eq!(
            second.cache_hits, 0,
            "单独开确定性 call_id 不够 —— 必须同时缓存中间轮"
        );
    }

    #[tokio::test]
    async fn normalized_call_id_and_intermediate_cache_together_hit_every_round() {
        // ⚠️ 档 3：两个开关都开。
        //
        // 与 `caching_intermediate_rounds_makes_the_agentic_chain_hit` 的差别是
        // 多了 `call_id` 归一化 —— 它的价值不在于命中率（档 2 已经 100%），
        // 而在于让**发给 provider 的请求逐字节相同**（利于其侧的前缀缓存）。
        let llm = Arc::new(VariedCallIdLlm::new());

        let cache = Arc::new(
            CacheService::new(Arc::new(LocalCache::new()), 60)
                .with_cacheable_tools(vec!["echo".to_string()]),
        );

        let agent = CachedAgent::new(llm.clone(), registry(), cache, config())
            .with_normalized_call_id(true)
            .with_cache_intermediate_rounds(true);

        agent.run("问题").await.unwrap();
        let second = agent.run("问题").await.unwrap();

        assert_eq!(second.cache_hits, 2);
        assert_eq!(second.cache_misses, 0);
    }

    #[tokio::test]
    async fn both_switches_are_disabled_by_default() {
        // ⚠️ 默认**全关** —— 两个开关都有明确代价，不该偷偷生效。
        //
        // ① `normalize_call_id` 改写了发给模型的请求；
        // ② `cache_intermediate_rounds` 让命中跳过工具执行（结果被冻结）。
        let llm = Arc::new(VariedCallIdLlm::new());
        let cache = Arc::new(
            CacheService::new(Arc::new(LocalCache::new()), 60)
                .with_cacheable_tools(vec!["echo".to_string()]),
        );

        let agent = CachedAgent::new(llm.clone(), registry(), cache, config());
        assert!(!agent.normalize_call_id());
        assert!(!agent.cache_intermediate_rounds());

        agent.run("问题").await.unwrap();
        let second = agent.run("问题").await.unwrap();

        // 与「关闭」时一致：agentic 链路不命中。
        assert_eq!(second.cache_hits, 0);
    }

    #[tokio::test]
    async fn pure_qa_chains_hit_the_cache() {
        // ⚠️ 与上一条对照：**不涉及工具**的纯问答能正常命中。
        //
        // 这就是「缓存到底有没有用」的回答：
        // **纯问答链路有收益，agentic 链路没有**。
        let llm = Arc::new(LoopingLlm::new(vec![ScriptedStep::Text(
            "缓存的答案".to_string(),
        )]));
        let cache = cache_service();
        let agent = CachedAgent::new(llm.clone(), registry(), cache, config());

        agent.run("纯问题").await.unwrap();
        let second = agent.run("纯问题").await.unwrap();

        assert_eq!(second.cache_hits, 1);
        assert_eq!(llm.call_count(), 1);
    }

    #[tokio::test]
    async fn tool_results_are_not_cached_when_not_whitelisted() {
        let llm = Arc::new(LoopingLlm::new(vec![
            ScriptedStep::call("c1", "echo", json!({"text": "hi"})),
            ScriptedStep::Text("最终答案".to_string()),
        ]));

        // 白名单是空的 → 工具结果不缓存。
        let cache = Arc::new(CacheService::new(Arc::new(LocalCache::new()), 60));
        let agent = CachedAgent::new(llm.clone(), registry(), cache, config());

        agent.run("问题").await.unwrap();
        let second = agent.run("问题").await.unwrap();

        assert!(!second.steps[0].tool_calls[0].cached);
    }

    #[tokio::test]
    async fn empty_question_is_rejected() {
        let llm = Arc::new(LoopingLlm::new(vec![ScriptedStep::Text("x".to_string())]));
        let agent = CachedAgent::new(llm, registry(), cache_service(), config());

        let err = agent.run("   ").await.unwrap_err();
        assert_eq!(err.status_code(), 400);
    }
}
