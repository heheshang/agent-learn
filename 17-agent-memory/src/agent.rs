//! 带记忆的 Agent —— 本项目的核心。
//!
//! ## 一次对话的完整流程
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │ 1. 从 checkpointer 载入线程状态（thread_id → 状态）         │
//! │    └─ 没有就新建空状态                                     │
//! │ 2. ★ 从长期记忆召回事实（按本轮问题检索）                  │
//! │ 3. 组装上下文：                                            │
//! │    system = Identity + Instructions + Memory（召回的事实）  │
//! │    messages = 历史（截断到 history_keep）                   │
//! │ 4. 调模型 → 得到回复                                       │
//! │ 5. 把「用户输入 + 模型回复」写回状态 → 存回 checkpointer    │
//! │ 6. ★ 抽取事实 → 存进长期记忆                               │
//! │    └─ 抽取失败 → warn 日志 + 降级（不影响本轮答案）         │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## 三个必须做对的顺序
//!
//! ### ① 召回在**调模型之前**
//!
//! 事实要进 system 提示，就必须先召回。
//! 反过来做（先问模型再召回）就没有意义了。
//!
//! ### ② 写回状态在**得到回复之后**
//!
//! 先存「用户输入」再调模型，会导致：
//! 模型失败时留下一条**没有回复**的用户消息 ——
//! 下次恢复时历史里就是一条孤儿消息。
//!
//! ### ③ 抽取在**写回状态之后**
//!
//! 抽取要基于**完整的这一轮**（用户说了什么 + 模型答了什么）。
//! 而且抽取是**尽力而为**的 —— 失败不能影响已经得到的答案。
//!
//! ## ⚠️ 线程隔离的验证
//!
//! 本模块把「线程隔离」做成了**可断言的**：
//! [`AgentRun`] 里带 `thread_id` 和 `history_messages`，
//! 测试可以断言「线程 B 的历史里没有线程 A 的内容」。

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use agent_logging::{debug, info, warn};

use crate::checkpoint::SharedCheckpointer;
use crate::error::{AppError, AppResult};
use crate::llm::ChatLlm;
use crate::longterm::{MemoryFact, MemoryRetriever, MemoryStore, SharedFactExtractor};
use crate::prompt::{self, PromptConfig};
use crate::state::{MemoryMessage, MemoryRole, ThreadId, ThreadState};

/// Agent 配置。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 提示词配置。
    pub prompt: PromptConfig,

    /// 是否在每轮之后抽取长期记忆。
    ///
    /// ## 默认 `true`
    ///
    /// 这是「记忆」和「聊天机器人」的分界线 ——
    /// 关掉它，本项目就退化成一个普通的对话机器人。
    ///
    /// 但它**每次都要调一次模型**（抽取）—— 所以也提供了开关，
    /// 方便做「有记忆 vs 没记忆」的对照实验。
    pub extract_memories: bool,

    /// 长期记忆的检索条数上限（覆盖 `PromptConfig::memory_top_k`）。
    pub memory_top_k: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            prompt: PromptConfig::default(),
            extract_memories: true,
            memory_top_k: 5,
        }
    }
}

/// 一次对话的结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRun {
    /// 线程 id。
    pub thread_id: String,

    /// 本轮问题。
    pub question: String,

    /// 模型回复。
    pub answer: String,

    /// 本轮召回了哪些长期记忆。
    #[serde(default)]
    pub recalled: Vec<MemoryFact>,

    /// 本轮抽取出了哪些新事实。
    #[serde(default)]
    pub extracted: Vec<MemoryFact>,

    /// 写回之后，线程里有多少条消息。
    pub history_messages: usize,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 带记忆的 Agent。
pub struct Agent {
    llm: Arc<dyn ChatLlm>,
    checkpointer: SharedCheckpointer,
    store: Arc<dyn MemoryStore>,
    retriever: Arc<dyn MemoryRetriever>,
    extractor: SharedFactExtractor,
    config: AgentConfig,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("model", &self.llm.model_name())
            .field("extract_memories", &self.config.extract_memories)
            .finish()
    }
}

impl Agent {
    /// 创建 Agent。
    ///
    /// ## ⚠️ 五个依赖**全部**从构造时注入
    ///
    /// 这是仓库里贯穿的一条原则（07 的 `Llm`、13 的 `Tool`、14 的 `ToolRegistry`）。
    ///
    /// 有了它，整个记忆层能**零网络确定性测试**：
    /// `ScriptedChatLlm` + `ScriptedExtractor` + 内存 checkpointer + 内存 store。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        llm: Arc<dyn ChatLlm>,
        checkpointer: SharedCheckpointer,
        store: Arc<dyn MemoryStore>,
        retriever: Arc<dyn MemoryRetriever>,
        extractor: SharedFactExtractor,
        config: AgentConfig,
    ) -> Self {
        Self {
            llm,
            checkpointer,
            store,
            retriever,
            extractor,
            config,
        }
    }

    /// 跑一轮对话。
    ///
    /// ## 参数
    ///
    /// - `thread_id` —— 会话线程（隔离的边界）
    /// - `owner` —— 长期记忆的归属（通常就是用户 id）
    /// - `question` —— 本轮输入
    pub async fn run(
        &self,
        thread_id: &ThreadId,
        owner: &str,
        question: &str,
    ) -> AppResult<AgentRun> {
        let start = Instant::now();
        let question = question.trim();

        if question.is_empty() {
            return Err(AppError::BadRequest("问题不能为空".to_string()));
        }

        // ── 1. 载入线程状态 ────────────────────────────────────
        //
        // ⚠️ 「线程不存在」是**正常状态**，不是错误。
        let mut state = self
            .checkpointer
            .load(thread_id)
            .await?
            .unwrap_or_else(|| ThreadState::new(thread_id.as_str()));

        // ── 2. 召回长期记忆 ────────────────────────────────────
        //
        // ⚠️ 召回用**本轮问题**做检索词 —— 不是用历史。
        //
        // 用历史做检索词会把「所有历史话题」都召回来，
        // 上下文被无关事实淹没。
        let recalled = if self.config.prompt.use_longterm {
            self.retriever
                .search(question, owner, self.config.memory_top_k)
                .await
                .unwrap_or_else(|e| {
                    // ⚠️ 召回失败**降级**，不中断对话 ——
                    //    「查不到记忆」不等于「不能聊天」。
                    warn!(thread = %thread_id, error = %e, "长期记忆召回失败 —— 降级为空");
                    Vec::new()
                })
        } else {
            Vec::new()
        };

        debug!(
            thread = %thread_id,
            owner = %owner,
            recalled = recalled.len(),
            history = state.message_count(),
            "已载入状态并召回记忆"
        );

        // ── 3. 组装上下文 ──────────────────────────────────────
        let system = prompt::system_prompt(&recalled, &self.config.prompt);

        // 用户的这轮输入**先**放进历史（模型要看到它）。
        state.merge_messages(vec![MemoryMessage::user(question)]);

        let history = prompt::build_messages(&state.messages, &self.config.prompt);

        // ── 4. 调模型 ──────────────────────────────────────────
        let answer = self.llm.chat_messages(&system, &history).await?;

        // ── 5. 把回复写回状态 → 存回 checkpointer ─────────────
        state.merge_messages(vec![MemoryMessage::assistant(&answer)]);
        state.version += 1;
        self.checkpointer.save(&state).await?;

        debug!(
            thread = %thread_id,
            messages = state.message_count(),
            version = state.version,
            "线程状态已写回"
        );

        // ── 6. 抽取长期记忆 ────────────────────────────────────
        //
        // ⚠️ 抽取是**尽力而为** ——
        //    失败只记 warn，不影响已经得到的答案。
        //
        //    这是本项目的一条硬原则：
        //    **「记住」失败不能让「回答」失败。**
        let extracted = if self.config.extract_memories {
            self.extract_and_store(owner, &state.messages).await
        } else {
            Vec::new()
        };

        let elapsed_ms = start.elapsed().as_millis() as u64;

        info!(
            thread = %thread_id,
            recalled = recalled.len(),
            extracted = extracted.len(),
            history = state.message_count(),
            elapsed_ms,
            "带记忆的对话完成"
        );

        Ok(AgentRun {
            thread_id: thread_id.to_string(),
            question: question.to_string(),
            answer,
            recalled,
            extracted,
            history_messages: state.message_count(),
            elapsed_ms,
        })
    }

    /// 抽取事实并存进长期记忆。
    ///
    /// ## ⚠️ 全程降级 —— 这一步的任何失败都不该影响主流程
    ///
    /// 失败的三种可能：
    ///
    /// 1. 模型调用失败（抽取器挂了）
    /// 2. JSON 解析失败（模型输出不是合法 JSON）
    /// 3. 存储失败（磁盘满了）
    ///
    /// 三种都只记 `warn`，返回空列表。
    ///
    /// ## 为什么抽取要基于**整个历史**而不是只基于本轮
    ///
    /// 因为「我叫 Bob」这句话可能是**十轮之前**说的 ——
    /// 用户现在问「我叫什么」，模型从历史里看到了，
    /// 但如果抽取器只看本轮，就抽不到任何事实。
    async fn extract_and_store(&self, owner: &str, messages: &[MemoryMessage]) -> Vec<MemoryFact> {
        // 只把「有信息量」的消息喂给抽取器 ——
        // 系统提示和工具回执里没有事实。
        let relevant: Vec<MemoryMessage> = messages
            .iter()
            .filter(|m| {
                matches!(m.role, MemoryRole::User | MemoryRole::Assistant)
                    && !m.content.trim().is_empty()
            })
            .cloned()
            .collect();

        if relevant.is_empty() {
            return Vec::new();
        }

        let extracted = match self.extractor.extract(&relevant, owner).await {
            Ok(facts) => facts,
            Err(e) => {
                warn!(owner = %owner, error = %e, "事实抽取失败 —— 降级（不影响本轮回答）");
                return Vec::new();
            }
        };

        if extracted.is_empty() {
            debug!(owner = %owner, "本轮没有抽出新事实");
            return Vec::new();
        }

        // 存进长期记忆。
        //
        // ⚠️ 单条存失败不影响其它条 —— 逐条处理。
        let mut stored = Vec::new();
        for fact in extracted {
            match self.store.put(fact.clone()).await {
                Ok(_) => stored.push(fact),
                Err(e) => {
                    warn!(
                        owner = %owner,
                        subject = %fact.subject,
                        predicate = %fact.predicate,
                        error = %e,
                        "事实入库失败 —— 跳过这一条"
                    );
                }
            }
        }

        debug!(owner = %owner, stored = stored.len(), "事实已入库");
        stored
    }

    /// 读一个线程的当前状态（不给它发消息）。
    ///
    /// 给「查看会话记录」的 HTTP 端点用。
    pub async fn peek(&self, thread_id: &ThreadId) -> AppResult<Option<ThreadState>> {
        self.checkpointer.load(thread_id).await
    }

    /// 列出所有已存在的线程 id。
    pub async fn list_threads(&self) -> AppResult<Vec<String>> {
        self.checkpointer.list_threads().await
    }

    /// 清空一个线程的短期记忆。
    pub async fn forget_thread(&self, thread_id: &ThreadId) -> AppResult<bool> {
        self.checkpointer.delete(thread_id).await
    }

    /// 清空一个 owner 的长期记忆。
    pub async fn forget_memories(&self, owner: &str) -> AppResult<usize> {
        self.store.clear(owner).await
    }

    /// 列出某 owner 的长期记忆。
    pub async fn list_memories(&self, owner: &str) -> AppResult<Vec<MemoryFact>> {
        self.store.list(owner).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::in_memory_checkpointer;
    use crate::llm::ScriptedChatLlm;
    use crate::longterm::{
        FactExtractor, NoopExtractor, ScriptedExtractor, in_memory_store, rank_facts,
    };

    /// 一个只做 owner 过滤的检索器（不依赖 BM25 的确定性足够）。
    struct OwnerRetriever {
        store: Arc<dyn MemoryStore>,
    }

    impl MemoryRetriever for OwnerRetriever {
        fn search<'a>(
            &'a self,
            query: &'a str,
            owner: &'a str,
            top_k: usize,
        ) -> crate::longterm::MemorySearchFuture<'a, Vec<MemoryFact>> {
            let store = self.store.clone();
            let query = query.to_string();
            let owner = owner.to_string();
            Box::pin(async move {
                let facts = store.list(&owner).await?;
                Ok(rank_facts(&query, &facts, top_k))
            })
        }
    }

    fn agent_with(
        llm: Arc<dyn ChatLlm>,
        extractor: SharedFactExtractor,
        store: Arc<dyn MemoryStore>,
    ) -> Agent {
        let checkpointer = in_memory_checkpointer();
        let retriever = Arc::new(OwnerRetriever {
            store: store.clone(),
        });
        Agent::new(
            llm,
            checkpointer,
            store,
            retriever,
            extractor,
            AgentConfig::default(),
        )
    }

    fn tid(name: &str) -> ThreadId {
        ThreadId::new(name).unwrap()
    }

    #[tokio::test]
    async fn answers_and_persists_the_thread() {
        let llm = Arc::new(ScriptedChatLlm::new(vec![
            "你好 Bob".to_string(),
            "你叫 Bob".to_string(),
        ]));
        let agent = agent_with(llm, Arc::new(NoopExtractor), in_memory_store());

        let run = agent.run(&tid("t1"), "alice", "我叫 Bob").await.unwrap();
        assert_eq!(run.answer, "你好 Bob");
        assert_eq!(run.history_messages, 2);

        // 第二轮 —— 历史里应该有第一轮。
        let run2 = agent.run(&tid("t1"), "alice", "我叫什么").await.unwrap();
        assert_eq!(run2.answer, "你叫 Bob");
        assert_eq!(run2.history_messages, 4);
    }

    #[tokio::test]
    async fn the_model_sees_the_history() {
        // ⚠️ 核心断言：模型确实收到了之前的历史。
        let llm = Arc::new(ScriptedChatLlm::always("答案"));
        let agent = agent_with(llm.clone(), Arc::new(NoopExtractor), in_memory_store());

        agent.run(&tid("t1"), "alice", "我叫 Bob").await.unwrap();
        agent.run(&tid("t1"), "alice", "我叫什么").await.unwrap();

        let calls = llm.calls();
        assert_eq!(calls.len(), 2);

        // 第二轮调用应该能看到「我叫 Bob」。
        let second = &calls[1].messages;
        assert!(second.iter().any(|m| m.content == "我叫 Bob"));
        assert!(second.iter().any(|m| m.content == "我叫什么"));
    }

    #[tokio::test]
    async fn threads_are_isolated() {
        // ⚠️ **线程隔离的核心测试**。
        let llm = Arc::new(ScriptedChatLlm::always("答案"));
        let agent = agent_with(llm.clone(), Arc::new(NoopExtractor), in_memory_store());

        agent
            .run(&tid("alice"), "alice", "我叫 Alice")
            .await
            .unwrap();
        agent.run(&tid("bob"), "bob", "我叫 Bob").await.unwrap();

        // 线程 bob 的模型调用里**不能**出现「我叫 Alice」。
        let calls = llm.calls();
        let bob_call = calls
            .iter()
            .find(|c| c.messages.iter().any(|m| m.content == "我叫 Bob"))
            .unwrap();
        assert!(
            !bob_call.messages.iter().any(|m| m.content == "我叫 Alice"),
            "线程 B 不应该看到线程 A 的消息"
        );

        // 两个线程各自的历史长度都是 2。
        assert_eq!(
            agent
                .peek(&tid("alice"))
                .await
                .unwrap()
                .unwrap()
                .message_count(),
            2
        );
        assert_eq!(
            agent
                .peek(&tid("bob"))
                .await
                .unwrap()
                .unwrap()
                .message_count(),
            2
        );
    }

    #[tokio::test]
    async fn longterm_memory_is_recalled_across_threads() {
        // ⚠️ **跨线程记忆的核心测试**：
        //    线程 1 告诉模型「我叫 Bob」，
        //    线程 2（完全不同的线程）应该能召回这条事实。
        let store = in_memory_store();
        store
            .put(MemoryFact::new("用户", "名字", "Bob", "alice"))
            .await
            .unwrap();

        let llm = Arc::new(ScriptedChatLlm::always("你叫 Bob"));
        let agent = agent_with(llm.clone(), Arc::new(NoopExtractor), store);

        let run = agent
            .run(&tid("fresh-thread"), "alice", "我叫什么名字")
            .await
            .unwrap();

        assert_eq!(run.recalled.len(), 1);
        assert_eq!(run.recalled[0].value, "Bob");

        // ⚠️ 事实必须进 **system 提示** ——
        //    否则模型看不到它，召回就白做了。
        let calls = llm.calls();
        assert!(calls[0].system.contains("Bob"));
        assert!(calls[0].system.contains("# Memory"));
    }

    #[tokio::test]
    async fn recalled_facts_are_not_written_into_the_history() {
        // ⚠️ 召回的事实**不进消息数组** ——
        //    否则每轮都会把事实重复写进历史，越滚越长。
        let store = in_memory_store();
        store
            .put(MemoryFact::new("用户", "名字", "Bob", "alice"))
            .await
            .unwrap();

        let llm = Arc::new(ScriptedChatLlm::always("答案"));
        let agent = agent_with(llm, Arc::new(NoopExtractor), store);

        let run = agent.run(&tid("t1"), "alice", "我叫什么").await.unwrap();

        // 历史里只有「用户输入 + 模型回复」两条，没有 fact。
        assert_eq!(run.history_messages, 2);

        let calls = agent.peek(&tid("t1")).await.unwrap().unwrap();
        assert!(calls.messages.iter().all(|m| m.role != MemoryRole::Fact));
    }

    #[tokio::test]
    async fn extracts_and_stores_new_facts() {
        let store = in_memory_store();
        let extractor =
            ScriptedExtractor::new(vec![vec![MemoryFact::new("用户", "名字", "Bob", "alice")]]);

        let llm = Arc::new(ScriptedChatLlm::always("你好 Bob"));
        let agent = agent_with(llm, Arc::new(extractor), store.clone());

        let run = agent.run(&tid("t1"), "alice", "我叫 Bob").await.unwrap();

        assert_eq!(run.extracted.len(), 1);
        assert_eq!(run.extracted[0].value, "Bob");

        // 事实应该真的进了长期记忆库。
        let facts = store.list("alice").await.unwrap();
        assert_eq!(facts.len(), 1);
    }

    #[tokio::test]
    async fn extraction_failure_does_not_break_the_answer() {
        // ⚠️ **「记住失败不能让回答失败」** —— 这是硬原则。
        struct FailingExtractor;

        impl FactExtractor for FailingExtractor {
            fn extract<'a>(
                &'a self,
                _messages: &'a [MemoryMessage],
                _owner: &'a str,
            ) -> crate::longterm::MemorySearchFuture<'a, Vec<MemoryFact>> {
                Box::pin(async move { Err(AppError::Llm("抽取器挂了".to_string())) })
            }
        }

        let llm = Arc::new(ScriptedChatLlm::always("答案"));
        let agent = agent_with(llm, Arc::new(FailingExtractor), in_memory_store());

        let run = agent.run(&tid("t1"), "alice", "我叫 Bob").await.unwrap();

        // 答案照常返回，抽取结果为空。
        assert_eq!(run.answer, "答案");
        assert!(run.extracted.is_empty());
    }

    #[tokio::test]
    async fn extraction_can_be_disabled() {
        // 对照实验：关掉抽取 = 「没有长期记忆的 agent」。
        let store = in_memory_store();
        let llm = Arc::new(ScriptedChatLlm::always("答案"));
        let checkpointer = in_memory_checkpointer();
        let retriever = Arc::new(OwnerRetriever {
            store: store.clone(),
        });

        let agent = Agent::new(
            llm,
            checkpointer,
            store.clone(),
            retriever,
            Arc::new(NoopExtractor),
            AgentConfig {
                extract_memories: false,
                ..Default::default()
            },
        );

        let run = agent.run(&tid("t1"), "alice", "我叫 Bob").await.unwrap();
        assert!(run.extracted.is_empty());
        assert!(store.list("alice").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn rejects_an_empty_question() {
        let llm = Arc::new(ScriptedChatLlm::always("答案"));
        let agent = agent_with(llm, Arc::new(NoopExtractor), in_memory_store());

        let err = agent.run(&tid("t1"), "alice", "   ").await.unwrap_err();
        assert_eq!(err.status_code(), 400);
    }

    #[tokio::test]
    async fn forgets_a_thread() {
        let llm = Arc::new(ScriptedChatLlm::always("答案"));
        let agent = agent_with(llm, Arc::new(NoopExtractor), in_memory_store());

        agent.run(&tid("t1"), "alice", "你好").await.unwrap();
        assert!(agent.peek(&tid("t1")).await.unwrap().is_some());

        assert!(agent.forget_thread(&tid("t1")).await.unwrap());
        assert!(agent.peek(&tid("t1")).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn forgets_longterm_memories() {
        let store = in_memory_store();
        store
            .put(MemoryFact::new("用户", "名字", "Bob", "alice"))
            .await
            .unwrap();

        let llm = Arc::new(ScriptedChatLlm::always("答案"));
        let agent = agent_with(llm, Arc::new(NoopExtractor), store);

        assert_eq!(agent.forget_memories("alice").await.unwrap(), 1);
        assert!(agent.list_memories("alice").await.unwrap().is_empty());
    }
}
