//! # 25-agent-summary —— **Agent 全链路汇总与压测台**
//!
//! 这个项目回答一个问题：
//! **「前面 24 个项目逐步长出来的能力，串起来之后跑得怎么样？」**
//!
//! ## 为什么要有这个项目
//!
//! 前面 24 个项目是**逐步演进**的：
//! 04 教会分块，08 教会混合检索，13 教会 ReAct，
//! 16 教会结构化工具调用，17 教会记忆，18 教会错误处理，
//! 21 教会缓存，22 教会并发，23 教会批处理，24 教会服务端调度……
//!
//! 但它们是**十一个独立的 crate**，各写各的 `main.rs`，
//! 没有任何一处能回答「**整个 Agent 系统一共有哪些零件？**」
//! 更没有一处能回答「**这些零件串起来，是谁在拖后腿？**」。
//!
//! 本项目就是那个「**汇总处**」。
//!
//! ## ⚠️ 本项目**不依赖**其它项目
//!
//! 它**自己实现**了历史项目的核心能力，形成一个自洽的统一体。
//! 这样做有三个理由：
//!
//! 1. **它是「演进」的终点** —— 不是「引用」，是「长出来」。
//!    每个模块都能独立读懂、独立运行、独立压测。
//! 2. **压测口径可控** —— 一旦依赖别人的 crate，
//!    对方改个内部实现就会让你的基线漂移，差值就不可归因了。
//! 3. **零外部依赖** —— 不引入 faiss / fastembed / Milvus / Redis，
//!    换机器就能跑。
//!
//! ## 七个模块
//!
//! | 模块 | 职责 | 汇总了哪些历史能力 |
//! |---|---|---|
//! | [`ingest`] | 数据入口 | 04 的分块、08 的分词 |
//! | [`retrieve`] | 检索 | 08 的 BM25 + RRF |
//! | [`agent`] | 推理 | 13 的 ReAct 解析 + 算术、17 的记忆 |
//! | [`tooling`] | 工具 | 14 的 Schema 校验、15 的 SQL 护栏、18 的错误分类 |
//! | [`cache`] | 缓存 | 21 的内容寻址键 |
//! | [`concurrency`] | 并发 | 22 的四档模式 |
//! | [`aggregate`] | **聚合（核心）** | 三档策略 |
//!
//! ## ★ P1：接进 `genai-agentprism` 的六项能力
//!
//! | 能力 | 落点 |
//! |---|---|
//! | Schema 消毒（`sanitize_json_schema`） | [`tool`] —— **不自己改写**，交给适配器 |
//! | 结构化输出（`JsonSpec` / `JsonMode`） | [`structured`] + [`llm`] |
//! | token 计量（`usage`） | [`llm::LlmUsage`] |
//! | 截断检测（`StopReason::is_max_tokens`） | [`llm::LlmReply::truncated`] |
//! | 思考内容回传（`ContentPart::ReasoningContent`） | [`agent_loop`] |
//! | 提示词缓存键（`with_prompt_cache_key`） | [`llm::LlmOptions::prompt_cache_key`] |
//!
//! ## ★ 三条协议
//!
//! | 协议 | 用途 |
//! |---|---|
//! | `react` | 文本协议（13 的路子）—— **要正则解析** |
//! | `function` | 结构化工具调用（16 的路子）—— 不用解析 |
//! | `structured` | 结构化输出（19 的路子）—— **结论可校验** |
//!
//! ## ★ 三档策略 —— 本项目最重要的产出
//!
//! | 变体 | 做了什么 | 验证什么 |
//! |---|---|---|
//! | `naive` | 每次全量扫描 + 每条记录的派生计算 | 基线 |
//! | `pre-agg` | 建索引时把派生计算**预聚合**并**整型化** | 算法收益 |
//! | `optimized` | `pre-agg` + 内容寻址**结果缓存** | 缓存收益 |
//!
//! ## ★ 四个最值得记住的结论
//!
//! ### ① 「减少扫描量」这个直觉优化，实测只值 1.00x
//!
//! 分段剖析显示：真正的开销不是「扫了多少条」，而是**每条记录的派生计算**
//! （占 **89%**：字符串 clone + 字符串 hash + 格式化 + `BTreeMap` 增量插入）。
//! 把派生计算**搬到建索引时并整型化**，才换来 **5.16x**。
//!
//! ### ② 只比两档 = 无法归因
//!
//! `naive` vs `optimized` 的差距被「命中缓存直接返回」完全主导，
//! 数字漂亮但**说不出是谁的功劳**。加了中间的 `pre-agg` 档才能拆开。
//!
//! ### ③ 归因到哪一层，取决于你在哪一层测
//!
//! 进程内测出「算法快了 5.16x」，HTTP 轨测出「QPS 只快了 1.03x」——
//! 这**不是矛盾**，而是 HTTP 栈开销（1~4ms）比被测逻辑（0.1ms）
//! 大了一个数量级。**两条轨不能混着比。**
//!
//! ### ④ 「测不出差异」有两种原因
//!
//! 「**实现错了**」和「**实验前提不成立**」——
//! 写测试前要先分清，否则会去改本来正确的代码。

pub mod agent;
pub mod agent_loop;
pub mod aggregate;
pub mod api;
pub mod bench;
pub mod cache;
pub mod concurrency;
pub mod config;
pub mod error;
pub mod ingest;
pub mod llm;
pub mod loadtest;
pub mod metrics;
pub mod model;
pub mod pipeline;
pub mod render;
pub mod retrieve;
pub mod structured;
pub mod table;
pub mod tool;
pub mod tooling;

pub use aggregate::{Aggregator, Strategy};
pub use model::{Activity, Digest, Window};
pub use pipeline::{Pipeline, RunOutcome};
