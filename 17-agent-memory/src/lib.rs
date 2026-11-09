//! # 17-agent-memory —— 手撕 **Agent Memory**（消息状态 + thread_id + checkpointer）
//!
//! 这个项目回答一个问题：**Agent 怎么记住用户？**
//!
//! 参考资料：
//!
//! - 文档：[LangChain Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
//!   —— `AgentState` / `thread_id` / `checkpointer` 三件套
//! - 工具：[Mem0](https://github.com/mem0ai/mem0)
//!   —— 「The Memory Layer for AI Agents」：抽取事实 → 存储 → 召回
//! - 工具：[ReMe / MemoryScope](https://github.com/agentscope-ai/ReMe)
//!   —— 「Memory as File, File as Memory」：记忆是**普通文件**
//!
//! ## 核心产出
//!
//! 一个**真正有记忆**的 Agent：
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │ 短期记忆（线程内）                                   │
//! │   消息状态 + thread_id + checkpointer                │
//! │   → 进程重启后，同一个 thread_id 还能接着聊           │
//! ├──────────────────────────────────────────────────────┤
//! │ 长期记忆（跨线程）                                   │
//! │   事实抽取 + 存储 + 召回（Mem0 口径）                │
//! │   → 新线程里，模型仍然记得「你叫 Bob」               │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! ## 两级记忆的分工
//!
//! | | 短期记忆 | 长期记忆 |
//! |---|---|---|
//! | 作用域 | **一个线程内** | **跨线程、跨会话** |
//! | 内容 | 原始对话消息 | **提炼出的事实** |
//! | 存取 | 全量读入上下文 | **按需召回**（关键词检索） |
//! | LangChain 对应 | `AgentState` + `checkpointer` | 长期记忆 store |
//!
//! ## 三个实测踩出来的坑（本项目最有价值的部分）
//!
//! | 坑 | 现象 | 修法 |
//! |---|---|---|
//! | **1** | 召回的事实被当成「用户说的话」写进历史 → 几轮后历史全是重复事实 | 事实只进 **system 提示**，**不进**消息数组 |
//! | **2** | 用户改口（「我以前叫 Bob」）后，新旧两条事实并存 → 召回时打架 | 同 `(owner, subject, predicate)` **合并**，旧值进 `versions` |
//! | **3** | 抽取用真实模型 → 每次结果不同，测试没法写 | 抽成 trait，测试用 `ScriptedExtractor`（零网络确定性） |
//!
//! ## 关键设计决策
//!
//! | 决策 | 理由 |
//! |---|---|
//! | **线程隔离在存储层**做，不在查询层过滤 | 「先取出全部再 filter」漏一次就是数据泄漏 |
//! | **事实用自己���类型**（不直接存 `genai::ChatMessage`） | 持久化格式不焊死在第三方类型的内部结构上 |
//! | **召回失败降级为空**，不中断对话 | 「查不到记忆」≠「不能聊天」 |
//! | **抽取失败降级**，不影响已得到的答案 | **「记住」失败不能让「回答」失败** |
//! | **写回状态在拿到回复之后** | 否则模型失败会留下一条孤儿用户消息 |
//!
//! ## 模块划分
//!
//! | 模块 | 职责 | 对应概念 |
//! |---|---|---|
//! | [`state`] | 消息 / 线程 / 角色模型 | `AgentState` / `thread_id` |
//! | [`checkpoint`] | 状态持久化（内存 / 文件） | `checkpointer` |
//! | [`longterm`] | 事实抽取 / 存储 / 检索 | Mem0 / ReMe |
//! | [`prompt`] | 提示词组装（身份 / 规则 / 记忆 / 历史） | `AgentState` → 模型 |
//! | [`agent`] | **带记忆的对话循环** | `agent.invoke()` |
//! | [`llm`] | LLM 抽象（含 `ScriptedChatLlm`） | Model Interface |
//! | [`api`] | HTTP 接口（含 **隔离 / 恢复对照**） | — |

pub mod agent;
pub mod api;
pub mod checkpoint;
pub mod clock;
pub mod config;
pub mod error;
pub mod llm;
pub mod longterm;
pub mod prompt;
pub mod state;

pub use agent::{Agent, AgentConfig, AgentRun};
pub use checkpoint::{
    CheckpointFuture, Checkpointer, FileCheckpointer, InMemoryCheckpointer, SharedCheckpointer,
    file_checkpointer, in_memory_checkpointer,
};
pub use config::{AppConfig, CheckpointerKind, MemoryStoreKind, StorageConfig};
pub use error::{AppError, AppResult};
pub use llm::{ChatLlm, ChatLlmFuture, GenaiChatLlm, ScriptedChatLlm};
pub use longterm::{
    Bm25MemoryRetriever, FactExtractor, InMemoryMemoryStore, LlmFactExtractor, MemoryFact,
    MemoryRetriever, MemorySearchFuture, MemoryStore, NoopExtractor, ScriptedExtractor, file_store,
    in_memory_store, rank_facts,
};
pub use prompt::PromptConfig;
pub use state::{MemoryMessage, MemoryRole, ThreadId, ThreadState};
