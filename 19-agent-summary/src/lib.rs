//! # 19-agent-summary —— 研究助手 Agent（RAG + Web 搜索 + 周报 / 项目构建）
//!
//! 这个项目回答一个问题：**Agent 怎么把「查资料」变成「一份有依据的文档」？**
//!
//! 参考资料：
//!
//! - 文档：[OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
//! - 文档：[DeepSeek Function Calling](https://api-docs.deepseek.com/guides/function_calling)
//! - 文档：[LangChain 中间件](https://docs.langchain.com/oss/python/langchain/middleware/custom)
//! - 工具：[Mem0](https://github.com/mem0ai/mem0) —— 长期记忆的事实抽取 / 召回
//!
//! ## 核心产出
//!
//! 一个**真的能出报告**的研究助手：
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │ 阶段 A：研究（Agent 循环，多轮工具调用）                │
//! │   kb_search（本地 BM25）                              │
//! │   memory（长期记忆）                                  │
//! │   web_search / web_fetch（互联网）                    │
//! │   calculate（手写算术）                               │
//! │   └─► 产出：证据集合（E1、E2…）                      │
//! ├──────────────────────────────────────────────────────┤
//! │ 阶段 B：成文（一次纯文本调用，不带工具）                 │
//! │   └─► 产出：结构化 JSON（每条结论标注证据编号）         │
//! ├──────────────────────────────────────────────────────┤
//! │ 阶段 C：校验（代码，不靠提示词）                        │
//! │   └─► 产出：grounding = 接地 / 部分接地 / 未接地        │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! ## 两个应用场景
//!
//! | 场景 | 端点 | 产物 |
//! |---|---|---|
//! | **周度总结** | `POST /api/weekly` | 概览 / 亮点 / 风险 / 下周计划 |
//! | **项目构建** | `POST /api/plan` | 目标 / 技术选型 / 模块 / 里程碑 / 风险 |
//!
//! ## ⚠️ 四个实测踩出来的坑
//!
//! | 坑 | 现象 | 修法 |
//! |---|---|---|
//! | **1** | 一个提示词同时要求「查资料」和「写报告」→ 模型**先写**，然后编引用编号 | 拆成两阶段：研究 → 成文。成文时输入**只有证据** |
//! | **2** | 模型输出的 JSON 外面包着 ` ```json ` 围栏，前言里还带花括号 | 逐字符扫描**配对括号**（跳过字符串里的括号），剥围栏 |
//! | **3** | 模型引用了不存在的证据编号（`E99`） | **代码里**校验：`grounding` 三态（接地 / 部分接地 / 未接地） |
//! | **4** | 搜索服务因网络环境而异 —— DuckDuckGo / Wikipedia / Brave **全部超时** | 搜索抽象成 trait，Bing + GitHub 两个真实后端 + Mock 后端 |
//!
//! ## 复用关系（继承了前面所有项目的坑）
//!
//! | 来源 | 复用什么 | 继承的坑 |
//! |---|---|---|
//! | [`agent_18`] | 错误分类中间件、`ErrorAwareTool` | 重试/降级/终止三态、「每个 call_id 一条回执」 |
//! | [`agent_16`] | `ToolSpec`、`ToolChoiceMode` | 「Schema 是对模型的承诺」、`tool_choice` 四档 |
//! | [`agent_17`] | 长期记忆（事实存储 + BM25 召回） | 「事实只进 system 提示，不进消息数组」 |
//! | [`agent_14`] | JSON Schema 子集校验器 | 「参数错了别发请求」 |
//! | [`agent_13`] | 手写算术求值器 | 「统计数字不走模型心算」 |
//! | [`rag_08`] | BM25 倒排索引 | 「BM25 用块正文，不含标题」 |
//! | [`rag_04`] | 分块配置 | 分块口径与向量索引一致 |
//!
//! ## 模块划分
//!
//! | 模块 | 职责 |
//! |---|---|
//! | [`search`] | Web 搜索 / 抓取（**可替换后端**） |
//! | [`tools`] | 五个工具（RAG / 搜索 / 抓取 / 记忆 / 计算） |
//! | [`evidence`] | 证据收集与编号 |
//! | [`agent`] | 研究阶段循环 |
//! | [`report`] | 结构化产物 + **接地校验** |
//! | [`pipeline`] | 研究 → 成文 → 校验 三阶段编排 |
//! | [`api`] | HTTP 接口 |

pub mod agent;
pub mod api;
pub mod config;
pub mod error;
pub mod evidence;
pub mod llm;
pub mod pipeline;
pub mod prompt;
pub mod report;
pub mod search;
pub mod tools;

pub use agent::{Agent, AgentConfig, AgentRun, StepRecord, ToolCallRecord};
pub use config::{AppConfig, SearchBackend};
pub use error::{AppError, AppResult, ErrorCode, ErrorKind, ToolError};
pub use evidence::{Evidence, EvidenceCollector, EvidenceKind};
pub use pipeline::{PlanPipeline, PlanResult, WeeklyPipeline, WeeklyResult};
pub use report::{Grounding, GroundingReport, ProjectPlan, WeeklySummary};
pub use search::{MockBackend, PageContent, SearchResult, WebBackend};
pub use tools::{MemoryTool, ToolSpec, default_registry};
