//! # 14-agent-tool —— 手撕 **自定义工具开发**（LangChain Custom Tools + MCP）
//!
//! 这个项目回答一个问题：**怎么给 Agent 加工具，加工具时到底要考虑什么？**
//!
//! 参考资料：
//!
//! - 教程：[LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
//!   —— 自定义工具的契约、`args_schema`、运行时上下文、动态工具选择
//! - 参考：[Model Context Protocol](https://modelcontextprotocol.io/) /
//!   [Smithery](https://smithery.ai/) —— MCP 协议与服务器注册中心
//!
//! ## 核心产出
//!
//! 一个**查询天气的自定义工具**（`get_weather`），并集成到 Agent 里。
//!
//! 它和 13-agent-core 的 `search` / `calculate` 有本质区别：
//! **它必须等外部 HTTP 服务**。这个差异逼出了工具设计的全部要点。
//!
//! ## 工具的三个要素
//!
//! LangChain 教程说「**类型标注是必需的**，因为它们定义了工具的输入 Schema」，
//! 而 Schema 同时干两件事：
//!
//! 1. 告诉模型参数怎么填
//! 2. **本地校验**（模型填错时不发请求，直接告诉模型哪里错了）
//!
//! 本项目用 JSON Schema 子集实现第 2 点（[`schema`]）。
//!
//! ## 五个模块
//!
//! | 模块 | 职责 | 对应 LangChain |
//! |---|---|---|
//! | [`tool`] | 工具 trait / 注册表 / 参数校验 / 失败策略 | `@tool` / `Tool` / `handle_tool_errors` |
//! | [`schema`] | JSON Schema 子集校验器 | Pydantic `args_schema` |
//! | [`weather`] | **天气工具**（要联网） | 教程里的 `get_weather` |
//! | [`mcp`] | MCP 客户端（JSON-RPC 2.0） | `MCPAdapter` |
//! | [`mcp_server`] | 内置 MCP 服务器 | 一个真的 MCP 服务端 |
//!
//! ## 一条贯穿全项目的原则
//!
//! > **工具失败不是请求失败。**
//!
//! 工具报错时，把错误信息当成一条 `Observation` 喂回模型，
//! 模型读到「参数 `location` 必须是字符串」就能自己纠正。
//!
//! 只有用户**直接调工具**时，错误才变成 HTTP 响应 ——
//! 那时用户扮演了「模型」的角色。
//!
//! ## 工具来源可互换
//!
//! ```text
//! 本地工具：get_weather(location)  → 直接调 Open-Meteo
//! MCP 工具：get_weather(location)  → 经 JSON-RPC 转发到 MCP 服务端
//! ```
//!
//! 在 Agent 眼里**完全一样** —— 都只是一个 `ToolSpec`。
//! 「工具来自哪里」被彻底解耦了 —— 这正是 MCP 的核心价值。

pub mod agent;
pub mod api;
pub mod config;
pub mod error;
pub mod llm;
pub mod mcp;
pub mod mcp_server;
pub mod prompt;
pub mod schema;
pub mod tool;
pub mod weather;

pub use agent::{Agent, AgentRun, StepRecord, call_tool_directly};
pub use config::{AgentConfig, AppConfig, PromptConfig, ToolConfig, ToolMode};
pub use error::{AppError, AppResult, ToolError};
pub use llm::{GenaiLlm, Llm, LlmOptions, ScriptedLlm};
pub use mcp::{McpClient, McpToolDefinition};
pub use tool::{Tool, ToolContext, ToolFuture, ToolOutcome, ToolPolicy, ToolRegistry, ToolSpec};
pub use weather::{GET_WEATHER_TOOL_NAME, Units, WeatherTool, weather_schema};
