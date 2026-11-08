//! 全局配置。
//!
//! 所有「可调参数」集中在这里，来源有两档（后者覆盖前者）：
//!
//! ```text
//! 默认值  <  环境变量
//! ```
//!
//! 环境变量名带 `AGENT_` 前缀，避免和其它项目撞名
//! （06 用 `RAG_`，07 用 `QUERY_`，08 用 `HYBRID_`，10 用 `MILVUS_`）。
//!
//! ## 本项目特有的三组配置
//!
//! | 组 | 作用 | 关键项 |
//! |---|---|---|
//! | [`AgentConfig`] | Agent 循环本身 | `max_steps` / `max_observations` / `mode` |
//! | [`PromptConfig`] | ReAct 提示词 | `few_shot` / `final_answer_prompt` |
//! | [`ToolConfig`] | 工具行为 | `search_top_k` / `max_observation_chars` |
//!
//! ## 为什么 `max_steps` 必须有上限
//!
//! Agent 是「模型决定下一步」的循环 —— 如果模型一直要求调工具，
//! 就会变成烧钱的死循环（AutoGPT 最著名的失败模式）。
//! `max_steps` 是**兜底**，不是可选项。

use std::path::PathBuf;

/// Agent 的运行模式。
///
/// | 模式 | 行为 | 用途 |
/// |---|---|---|
/// | [`AgentMode::ReAct`] | 完整的 `Thought / Action / Action Input / Observation` 循环 | **默认**，对齐论文 |
/// | [`AgentMode::ActOnly`] | 去掉 `Thought:`，只做 `Action / Action Input` | **对照组** —— 验证「思考」到底有没有用 |
/// | [`AgentMode::Direct`] | 不调工具，直接让模型回答 | **对照组** —— 验证「工具」到底有没有用 |
///
/// 三个模式对应 ReAct 论文里的三个可消融的组件，
/// 让「推理」和「行动」各自的贡献可以被**实测**出来。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    /// 完整 ReAct 循环。
    ReAct,

    /// 只行动，不思考（去掉 `Thought:` 步骤）。
    ActOnly,

    /// 不调用工具，直接回答。
    Direct,
}

impl AgentMode {
    /// 模式名。
    pub fn name(self) -> &'static str {
        match self {
            AgentMode::ReAct => "react",
            AgentMode::ActOnly => "act_only",
            AgentMode::Direct => "direct",
        }
    }

    /// 从字符串解析（不区分大小写，支持连字符）。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "react" | "full" | "thought" => Some(AgentMode::ReAct),
            "act_only" | "act" | "no_thought" => Some(AgentMode::ActOnly),
            "direct" | "no_tool" | "baseline" => Some(AgentMode::Direct),
            _ => None,
        }
    }

    /// 所有模式（按「从基线到全量」排列）。
    pub fn all() -> Vec<Self> {
        vec![AgentMode::Direct, AgentMode::ActOnly, AgentMode::ReAct]
    }

    /// 是否使用工具。
    pub fn uses_tools(self) -> bool {
        !matches!(self, AgentMode::Direct)
    }

    /// 是否要求模型输出 `Thought:`。
    pub fn uses_thought(self) -> bool {
        matches!(self, AgentMode::ReAct)
    }
}

/// Agent 循环配置。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 运行模式。
    pub mode: AgentMode,

    /// 最大步数（一次「模型输出 + 工具执行」算一步）。
    ///
    /// ⚠️ **必须有上限**。模型可能反复调同一个工具
    /// （工具描述写得不好时很常见），没有上限就是烧钱死循环。
    ///
    /// 默认 6 步：够查两次资料 + 给出答案。
    pub max_steps: usize,

    /// 单次工具执行返回给模型的观察文本上限（字符数）。
    ///
    /// ## 为什么需要截断
    ///
    /// 工具返回的内容会**原样进上下文**。如果不截断，
    /// 一次检索返回 5000 字，两次就把上下文吃光了 ——
    /// 而模型的注意力会被稀释（「中间迷失」）。
    ///
    /// 截断时保留**头部**（开头通常是最相关的定义段）
    /// 并在尾部加一句省略提示。
    pub max_observation_chars: usize,

    /// `search` 工具默认返回几条。
    pub search_top_k: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            mode: AgentMode::ReAct,
            max_steps: 6,
            max_observation_chars: 1200,
            search_top_k: 3,
        }
    }
}

/// 提示词配置。
#[derive(Debug, Clone)]
pub struct PromptConfig {
    /// 是否在提示词里带 few-shot 示例。
    ///
    /// ## 为什么 few-shot 很重要
    ///
    /// ReAct 论文的一个核心发现是：**模型很容易漏掉 `Observation` 之后
    /// 的 `Thought`**，或者在没调工具时直接给答案。
    /// 一两个完整的轨迹示例能把这个行为**锚定**下来。
    ///
    /// 关掉它就能看出「模型有多依赖示例」——
    /// 这正是 `act_only` 对照组的一部分。
    pub few_shot: bool,

    /// 最终答案的引导语。
    ///
    /// 模型在给出答案前需要知道「什么时候该收尾」。
    /// 没有这句，它可能永远不停在工具循环里。
    pub final_answer_prompt: String,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            few_shot: true,
            final_answer_prompt:
                "如果你已经掌握了足够的信息，请以 `Final Answer:` 开头给出最终答案。\n".to_string(),
        }
    }
}

/// 工具配置。
#[derive(Debug, Clone)]
pub struct ToolConfig {
    /// `search` 工具的名字。
    ///
    /// ⚠️ 名字会**原样出现在提示词里** —— 模型照着这个名字调工具。
    /// 所以它必须和提示词里写的完全一致（这里统一由常量提供）。
    pub search_tool_name: String,

    /// `search` 工具的最大返回条数（模型可以传更小的值）。
    pub max_top_k: usize,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            search_tool_name: crate::tools::SEARCH_TOOL_NAME.to_string(),
            max_top_k: 5,
        }
    }
}

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// 知识库文档目录（`search` 工具的检索范围）。
    pub data_dir: PathBuf,

    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 模型名。
    pub model: String,

    /// 生成温度。
    ///
    /// Agent 循环需要**可复现** —— 温度太高会让模型的行为飘忽，
    /// 同一问题两次跑出完全不同的工具调用序列。
    /// 默认 0.0。
    pub temperature: f64,

    /// 生成 max tokens。
    ///
    /// ⚠️ 每个 step 都要重新发一次请求，历史越来越长 ——
    /// 这个值限制的是**单次响应**的长度（ReAct 的一步输出很短）。
    pub max_tokens: u32,

    /// Agent 循环配置。
    pub agent: AgentConfig,

    /// 提示词配置。
    pub prompt: PromptConfig,

    /// 工具配置。
    pub tool: ToolConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            data_dir: manifest.join("data"),
            bind_addr: "127.0.0.1:3060".to_string(),
            model: "deepseek-flash".to_string(),
            temperature: 0.0,
            max_tokens: 1024,
            agent: AgentConfig::default(),
            prompt: PromptConfig::default(),
            tool: ToolConfig::default(),
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置（未设置的项用默认值）。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `AGENT_DATA_DIR` | 知识库目录 | `13-agent-core/data` |
    /// | `AGENT_BIND_ADDR` | 监听地址 | `127.0.0.1:3060` |
    /// | `AGENT_MODE` | `react` / `act_only` / `direct` | `react` |
    /// | `AGENT_MAX_STEPS` | 最大步数 | `6` |
    /// | `AGENT_MAX_OBSERVATION_CHARS` | 观察文本截断长度 | `1200` |
    /// | `AGENT_SEARCH_TOP_K` | `search` 默认返回条数 | `3` |
    /// | `AGENT_MODEL` | 模型名 | `deepseek-flash` |
    /// | `AGENT_TEMPERATURE` | 温度 | `0.0` |
    /// | `AGENT_MAX_TOKENS` | max tokens | `1024` |
    /// | `AGENT_FEW_SHOT` | 是否带 few-shot 示例 | `true` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("AGENT_DATA_DIR") {
            config.data_dir = PathBuf::from(v);
        }
        if let Some(v) = env_str("AGENT_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("AGENT_MODE")
            && let Some(m) = AgentMode::parse(&v)
        {
            config.agent.mode = m;
        }
        if let Some(v) = env_usize("AGENT_MAX_STEPS") {
            config.agent.max_steps = v.clamp(1, 30);
        }
        if let Some(v) = env_usize("AGENT_MAX_OBSERVATION_CHARS") {
            config.agent.max_observation_chars = v.clamp(100, 20_000);
        }
        if let Some(v) = env_usize("AGENT_SEARCH_TOP_K") {
            config.agent.search_top_k = v.clamp(1, 10);
        }
        if let Some(v) = env_str("AGENT_MODEL") {
            config.model = v;
        }
        if let Some(v) = env_f64("AGENT_TEMPERATURE") {
            config.temperature = v;
        }
        if let Some(v) = env_u32("AGENT_MAX_TOKENS") {
            config.max_tokens = v;
        }
        if let Some(v) = env_bool("AGENT_FEW_SHOT") {
            config.prompt.few_shot = v;
        }

        config
    }

    /// 配置摘要（启动时打印）。
    pub fn summary(&self) -> String {
        format!(
            "语料目录  : {}\n\
             监听地址  : {}\n\
             模型      : {} temp={} max_tokens={}\n\
             模式      : {}\n\
             最大步数  : {}\n\
             观察截断  : {} 字符\n\
             search    : 默认 top_k={}（上限 {}）\n\
             few-shot  : {}",
            self.data_dir.display(),
            self.bind_addr,
            self.model,
            self.temperature,
            self.max_tokens,
            self.agent.mode.name(),
            self.agent.max_steps,
            self.agent.max_observation_chars,
            self.agent.search_top_k,
            self.tool.max_top_k,
            if self.prompt.few_shot { "开" } else { "关" },
        )
    }
}

/// 读一个非空环境变量。
fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// 读一个 `usize` 环境变量（解析失败就忽略，保留默认值）。
fn env_usize(key: &str) -> Option<usize> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `u32` 环境变量。
fn env_u32(key: &str) -> Option<u32> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `f64` 环境变量。
fn env_f64(key: &str) -> Option<f64> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个布尔环境变量（`1/true/yes/on` 为真）。
fn env_bool(key: &str) -> Option<bool> {
    env_str(key).map(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mode_names() {
        assert_eq!(AgentMode::parse("react"), Some(AgentMode::ReAct));
        assert_eq!(AgentMode::parse("Act-Only"), Some(AgentMode::ActOnly));
        assert_eq!(AgentMode::parse("direct"), Some(AgentMode::Direct));
        assert_eq!(AgentMode::parse("unknown"), None);
    }

    #[test]
    fn mode_round_trips() {
        for mode in AgentMode::all() {
            assert_eq!(AgentMode::parse(mode.name()), Some(mode));
        }
    }

    #[test]
    fn direct_mode_uses_no_tools_and_no_thought() {
        assert!(!AgentMode::Direct.uses_tools());
        assert!(!AgentMode::Direct.uses_thought());
    }

    #[test]
    fn act_only_uses_tools_but_no_thought() {
        assert!(AgentMode::ActOnly.uses_tools());
        assert!(!AgentMode::ActOnly.uses_thought());
    }

    #[test]
    fn react_uses_both() {
        assert!(AgentMode::ReAct.uses_tools());
        assert!(AgentMode::ReAct.uses_thought());
    }

    #[test]
    fn max_steps_has_a_hard_cap() {
        // 上限必须有 —— 否则模型可以无限循环烧钱。
        let config = AgentConfig::default();
        assert!(config.max_steps > 0);
        assert!(config.max_steps <= 30);
    }

    #[test]
    fn default_temperature_is_deterministic() {
        // Agent 循环需要可复现。
        assert_eq!(AppConfig::default().temperature, 0.0);
    }

    #[test]
    fn default_port_is_3060() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3060");
    }
}
