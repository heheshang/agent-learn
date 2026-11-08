//! # 13-agent-core —— 手撕 **Agent 核心概念**（ReAct + 工具 + 记忆 + 规划）
//!
//! 这个项目回答一个问题：**一个「自主 Agent」到底由什么构成，怎么跑起来？**
//!
//! 参考资料：
//!
//! - 博客：[LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
//!   （Lilian Weng, 2023-06-23）—— 三大组件：规划 / 记忆 / 工具使用
//! - 论文：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
//!   （Yao et al., ICLR 2023）—— 推理与行动交织的循环
//!
//! ## 三大组件的对应关系
//!
//! | 博客里的组件 | 本项目的实现 | 模块 |
//! |---|---|---|
//! | **工具使用** | `search`（BM25 检索）+ `calculate`（符号工具） | [`tools`] |
//! | **规划 / 自我反思** | ReAct 循环的 `Thought` 步骤 | [`agent`] / [`prompt`] |
//! | **短期记忆** | 循环轨迹（trajectory）—— 每一轮的思考与观察都在上下文里 | [`agent`] |
//! | **长期记忆** | 本项目的 `search` 工具背后的 BM25 索引 | [`tools`] |
//!
//! ## ReAct 循环
//!
//! ```text
//! ┌────────────────────────────────────────────────────────┐
//! │ 1. 组装提示词（格式 + 工具说明 + few-shot + 问题 + 轨迹） │
//! ├────────────────────────────────────────────────────────┤
//! │ 2. 调模型 → 一段文本                                   │
//! ├────────────────────────────────────────────────────────┤
//! │ 3. 解析：                                             │
//! │    ├─ `Final Answer:` → 收工                          │
//! │    ├─ `Action:` + `Action Input:` → 执行工具           │
//! │    └─ 都没有 → 容错当成答案                            │
//! ├────────────────────────────────────────────────────────┤
//! │ 4. 回到 1，直到 max_steps                              │
//! └────────────────────────────────────────────────────────┘
//! ```
//!
//! ## 三模式消融
//!
//! | 模式 | 工具 | 思考 | 验证什么 |
//! |---|---|---|---|
//! | `direct` | ❌ | ❌ | 模型参数里的知识够不够 |
//! | `act_only` | ✅ | ❌ | 「行动」本身值多少 |
//! | `react` | ✅ | ✅ | 「思考」又贡献了多少 |
//!
//! 三者的**变量是单一的** —— 这是复现 ReAct 论文结论的关键。
//!
//! ## 模块划分
//!
//! | 模块 | 职责 | 对应概念 |
//! |---|---|---|
//! | [`prompt`] | ReAct 提示词组装 + 三种模式变体 | ReAct Prompt |
//! | [`agent`] | ReAct 循环 + 输出解析 | ReAct Loop |
//! | [`tools`] | 工具定义 + 工具实现 + 参数解析 | Tool Use / MRKL |
//! | [`llm`] | LLM 抽象（复用 07 的 `Llm` trait） | Model Interface |
//! | [`config`] | 全局配置 | — |
//! | [`api`] | HTTP 接口（含**三模式对照**端点） | — |

pub mod agent;
pub mod api;
pub mod config;
pub mod error;
pub mod llm;
pub mod prompt;
pub mod tools;

pub use agent::{Agent, AgentRun, ParsedOutput, StepRecord, parse_react_output};
pub use config::{AgentConfig, AgentMode, AppConfig, PromptConfig, ToolConfig};
pub use error::{AgentError, AgentResult};
pub use llm::{GenaiLlm, Llm, LlmOptions, ScriptedLlm};
pub use prompt::{assemble, few_shot_for, system_prompt_for};
pub use tools::{CalculateTool, SearchTool, Tool, ToolSpec, build_index};
