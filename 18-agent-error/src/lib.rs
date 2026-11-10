//! # 18-agent-error —— 手撕 **Agent 错误处理**（middleware 驱动的可诊断工具失败）
//!
//! 这个项目回答一个问题：**Agent 里失败了，到底该怎么办？**
//!
//! 参考资料：
//!
//! - 文档：[Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
//!   —— `wrap_tool_call` 的语义：包住每次工具调用，
//!   决定「要不要调、调几次、失败了怎么办」
//!
//! ## 核心产出
//!
//! 一个**错误可诊断**的 Agent：
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │  工具失败                                             │
//! │    ↓                                                   │
//! │  分类：Transient / Business / Fatal                     │
//! │    ↓                                                   │
//! │  Transient → 指数退避重试 → 耗尽后降级                 │
//! │  Business  → 不重试，直接降级                          │
//! │  Fatal     → 不重试、不降级，终止                      │
//! │    ↓                                                   │
//! │  降级 = 变成一段**带错误码**的观察喂回模型              │
//! │         [E_INVALID_ARGUMENTS] 参数不合法：...。建议：... │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## 三个实测踩出来的坑（本项目最有价值的部分）
//!
//! | 坑 | 现象 | 修法 |
//! |---|---|---|
//! | **1** | 对所有错误一律重试 → 参数填错时白等 3 次 | 按**类别**决策：只有 Transient 重试 |
//! | **2** | 致命错误也降级 → 模型以为「工具坏了」，转而编答案 | Fatal **不降级**，直接终止 |
//! | **3** | 致命错误直接 `?` 传播 → **少一条 tool 回执**，下一轮 400 | 终止前仍要为它生成回执 |
//!
//! ## 关键设计决策
//!
//! | 决策 | 理由 |
//! |---|---|
//! | **重试和降级是两个正交维度** | 重试 = 再调一次同样的工具；降级 = 让模型换条路 |
//! | 错误分三类而非两类 | 「业务失败」和「可重试」都该降级，但只有前者不该重试 |
//! | 错误带**稳定错误码** + **可执行建议** | 文案会变，码不会；没有 hint 的 Business 错误是缺陷 |
//! | 致命错误也要产回执 | 少一条就 400（16 实测） |
//! | 模型调用失败**不**降级 | 「没有模型就没有答案」 |
//!
//! ## 模块划分
//!
//! | 模块 | 职责 | 对应概念 |
//! |---|---|---|
//! | [`error`] | **错误分类**（Transient/Business/Fatal）+ 稳定错误码 | — |
//! | [`tool`] | 声明失败类别的工具 + 注册表 | `ToolCallRequest` |
//! | [`middleware`] | **核心**：`wrap_tool_call` 的重试 / 降级 / 终止决策 | `wrap_tool_call` |
//! | [`agent`] | 把中间件接进结构化工具调用循环 | `create_agent` |
//! | [`llm`] | LLM 抽象（可预设失败的脚本替身） | Model Interface |
//! | [`prompt`] | 教模型**怎么读错误** | system prompt |
//! | [`table`] | 终端表格渲染（按**显示宽度**补空格） | — |
//! | [`api`] | HTTP 接口（含**错误分类速查表**） | — |

pub mod agent;
pub mod api;
pub mod config;
pub mod error;
pub mod llm;
pub mod middleware;
pub mod prompt;
pub mod table;
pub mod tool;
pub mod tools;

pub use agent::{Agent, AgentConfig, AgentRun, StepRecord, ToolCallRecord};
pub use config::AppConfig;
pub use error::{AppError, AppResult, ErrorCode, ErrorKind, ToolError};
pub use llm::{FunctionLlm, GenaiFunctionLlm, LlmStep, ScriptedLlm, ToolChoice};
pub use middleware::{
    ErrorHandlingMiddleware, MiddlewareOutcome, RegistryHandler, ToolCallRequest, ToolHandler,
};
pub use tool::{
    ErrorAwareTool, FLAKY_FETCH_TOOL_NAME, FlakyTool, GUARDED_PAY_TOOL_NAME, GuardedTool,
    STRICT_LOOKUP_TOOL_NAME, StrictTool, ToolContext, ToolFuture, ToolOutcome, ToolRegistry,
    demo_registry,
};
pub use tools::ToolSpec;
