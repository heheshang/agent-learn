//! 全局配置。
//!
//! 所有「可调参数」集中在这里，来源有两档（后者覆盖前者）：
//!
//! ```text
//! 默认值  <  环境变量
//! ```
//!
//! 环境变量名带 `TOOL_` 前缀，避免和其它项目撞名
//! （06 用 `RAG_`，07 用 `QUERY_`，13 用 `AGENT_`）。
//!
//! ## 本项目特有的三组配置
//!
//! | 组 | 作用 | 关键项 |
//! |---|---|---|
//! | [`AgentConfig`] | Agent 循环 | `max_steps` / `mode` |
//! | [`ToolConfig`] | 工具行为 | `weather_default_days` / `mcp_endpoint` |
//! | [`PromptConfig`] | 提示词 | `few_shot` |

/// Agent 的运行模式。
///
/// | 模式 | 行为 | 用途 |
/// |---|---|---|
/// | [`ToolMode::ReAct`] | 有工具、有思考 | **默认** |
/// | [`ToolMode::ActOnly`] | 有工具、无思考 | **对照组** —— 验证「思考」的贡献 |
/// | [`ToolMode::Direct`] | 无工具 | **对照组** —— 验证「工具」的贡献 |
///
/// ## ⚠️ 序列化名必须和 [`ToolMode::name`] 一致
///
/// `#[serde(rename_all = "snake_case")]` 会把 `ReAct` 变成 `re_act` ——
/// 和 CLI、配置里用的 `react` **不一致**。
///
/// 这种不一致很坑：CLI 能跑的模式名，API 却收不了。
/// 所以这里显式 `rename`，让三处（CLI / 环境变量 / JSON）**完全对齐**。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ToolMode {
    /// 完整 ReAct 循环（有工具 + 有思考）。
    #[serde(rename = "react")]
    ReAct,
    /// 只行动，不思考。
    #[serde(rename = "act_only")]
    ActOnly,
    /// 不调用工具，直接回答。
    #[serde(rename = "direct")]
    Direct,
}

impl ToolMode {
    /// 模式名。
    pub fn name(self) -> &'static str {
        match self {
            ToolMode::ReAct => "react",
            ToolMode::ActOnly => "act_only",
            ToolMode::Direct => "direct",
        }
    }

    /// 从字符串解析。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "react" | "full" | "thought" => Some(ToolMode::ReAct),
            "act_only" | "act" | "no_thought" => Some(ToolMode::ActOnly),
            "direct" | "no_tool" | "baseline" => Some(ToolMode::Direct),
            _ => None,
        }
    }

    /// 所有模式。
    pub fn all() -> Vec<Self> {
        vec![ToolMode::Direct, ToolMode::ActOnly, ToolMode::ReAct]
    }

    /// 是否使用工具。
    pub fn uses_tools(self) -> bool {
        !matches!(self, ToolMode::Direct)
    }

    /// 是否要求模型输出 `Thought:`。
    pub fn uses_thought(self) -> bool {
        matches!(self, ToolMode::ReAct)
    }
}

/// Agent 循环配置。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 运行模式。
    pub mode: ToolMode,

    /// 最大步数。
    ///
    /// ⚠️ **必须有上限** —— 模型可能反复调同一个工具。
    pub max_steps: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            mode: ToolMode::ReAct,
            max_steps: 6,
        }
    }
}

/// 提示词配置。
#[derive(Debug, Clone)]
pub struct PromptConfig {
    /// 是否带 few-shot 示例。
    pub few_shot: bool,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self { few_shot: true }
    }
}

/// 工具配置。
#[derive(Debug, Clone)]
pub struct ToolConfig {
    /// 天气工具默认预报天数。
    pub weather_default_days: usize,

    /// 天气工具最大预报天数。
    pub weather_max_days: usize,

    /// MCP 服务端地址（内置服务器默认在 `127.0.0.1:3071`）。
    ///
    /// 空字符串 = 不启用 MCP。
    pub mcp_endpoint: String,

    /// 是否把 MCP 工具和本地工具合并到一起。
    pub enable_mcp: bool,

    /// 工具白名单（空 = 全部启用）。
    ///
    /// 对应 LangChain 的动态工具选择 ——
    /// 「工具太多会让模型过载并增加错误」。
    pub allowed_tools: Vec<String>,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            weather_default_days: 1,
            weather_max_days: 7,
            mcp_endpoint: "http://127.0.0.1:3071/mcp".to_string(),
            enable_mcp: false,
            allowed_tools: Vec::new(),
        }
    }
}

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 内置 MCP 服务器监听地址。
    pub mcp_bind_addr: String,

    /// 模型名。
    pub model: String,

    /// 生成温度。
    ///
    /// Agent 循环需要**可复现** —— 温度太高会让工具调用序列飘忽。
    pub temperature: f64,

    /// 生成 max tokens。
    ///
    /// ⚠️ 每个 step 都要重新发一次请求，历史越来越长 ——
    /// 这个值限制的是**单次响应**的长度。
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
        Self {
            bind_addr: "127.0.0.1:3070".to_string(),
            mcp_bind_addr: "127.0.0.1:3071".to_string(),
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
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `TOOL_BIND_ADDR` | 主服务监听地址 | `127.0.0.1:3070` |
    /// | `TOOL_MCP_BIND_ADDR` | 内置 MCP 服务器地址 | `127.0.0.1:3071` |
    /// | `TOOL_MODE` | `react` / `act_only` / `direct` | `react` |
    /// | `TOOL_MAX_STEPS` | 最大步数 | `6` |
    /// | `TOOL_MODEL` | 模型名 | `deepseek-flash` |
    /// | `TOOL_TEMPERATURE` | 温度 | `0.0` |
    /// | `TOOL_MAX_TOKENS` | max tokens | `1024` |
    /// | `TOOL_FEW_SHOT` | 是否带 few-shot 示例 | `true` |
    /// | `TOOL_ENABLE_MCP` | 是否连接 MCP 服务端 | `false` |
    /// | `TOOL_MCP_ENDPOINT` | MCP 服务端地址 | `http://127.0.0.1:3071/mcp` |
    /// | `TOOL_WEATHER_DAYS` | 天气默认预报天数 | `1` |
    /// | `TOOL_ALLOWED_TOOLS` | 工具白名单（逗号分隔） | 空（全部） |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("TOOL_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("TOOL_MCP_BIND_ADDR") {
            config.mcp_bind_addr = v;
        }
        if let Some(v) = env_str("TOOL_MODE")
            && let Some(m) = ToolMode::parse(&v)
        {
            config.agent.mode = m;
        }
        if let Some(v) = env_usize("TOOL_MAX_STEPS") {
            config.agent.max_steps = v.clamp(1, 30);
        }
        if let Some(v) = env_str("TOOL_MODEL") {
            config.model = v;
        }
        if let Some(v) = env_f64("TOOL_TEMPERATURE") {
            config.temperature = v;
        }
        if let Some(v) = env_u32("TOOL_MAX_TOKENS") {
            config.max_tokens = v;
        }
        if let Some(v) = env_bool("TOOL_FEW_SHOT") {
            config.prompt.few_shot = v;
        }
        if let Some(v) = env_bool("TOOL_ENABLE_MCP") {
            config.tool.enable_mcp = v;
        }
        if let Some(v) = env_str("TOOL_MCP_ENDPOINT") {
            config.tool.mcp_endpoint = v;
        }
        if let Some(v) = env_usize("TOOL_WEATHER_DAYS") {
            config.tool.weather_default_days = v.clamp(1, 7);
        }
        if let Some(v) = env_str("TOOL_ALLOWED_TOOLS") {
            config.tool.allowed_tools = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        config
    }

    /// 配置摘要（启动时打印）。
    pub fn summary(&self) -> String {
        let allowed = if self.tool.allowed_tools.is_empty() {
            "（全部）".to_string()
        } else {
            self.tool.allowed_tools.join(", ")
        };

        format!(
            "监听地址  : {}\n\
             MCP 地址  : {}（启用: {}）\n\
             模型      : {} temp={} max_tokens={}\n\
             模式      : {}\n\
             最大步数  : {}\n\
             天气天数  : 默认 {}（上限 {}）\n\
             工具白名单: {}",
            self.bind_addr,
            self.tool.mcp_endpoint,
            if self.tool.enable_mcp { "是" } else { "否" },
            self.model,
            self.temperature,
            self.max_tokens,
            self.agent.mode.name(),
            self.agent.max_steps,
            self.tool.weather_default_days,
            self.tool.weather_max_days,
            allowed,
        )
    }
}

/// 读一个非空环境变量。
fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// 读一个 `usize` 环境变量。
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
        assert_eq!(ToolMode::parse("react"), Some(ToolMode::ReAct));
        assert_eq!(ToolMode::parse("Act-Only"), Some(ToolMode::ActOnly));
        assert_eq!(ToolMode::parse("direct"), Some(ToolMode::Direct));
        assert_eq!(ToolMode::parse("unknown"), None);
    }

    #[test]
    fn mode_round_trips() {
        for mode in ToolMode::all() {
            assert_eq!(ToolMode::parse(mode.name()), Some(mode));
        }
    }

    #[test]
    fn serde_name_matches_cli_name() {
        // ⚠️ 回归测试：序列化名必须和 CLI 名**完全一致**。
        //
        // `#[serde(rename_all = "snake_case")]` 会把 `ReAct` 变成 `re_act`，
        // 于是 CLI 能跑的模式名，API 却收不了。
        for mode in ToolMode::all() {
            let json = serde_json::to_string(&mode).unwrap();
            let quoted = format!("\"{}\"", mode.name());
            assert_eq!(
                json,
                quoted,
                "模式 {} 的序列化名和 CLI 名不一致",
                mode.name()
            );

            // 反序列化也要能回来。
            let parsed: ToolMode = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, mode);
        }
    }

    #[test]
    fn direct_mode_uses_no_tools() {
        assert!(!ToolMode::Direct.uses_tools());
        assert!(!ToolMode::Direct.uses_thought());
        assert!(ToolMode::ActOnly.uses_tools());
        assert!(!ToolMode::ActOnly.uses_thought());
        assert!(ToolMode::ReAct.uses_tools());
        assert!(ToolMode::ReAct.uses_thought());
    }

    #[test]
    fn default_temperature_is_deterministic() {
        assert_eq!(AppConfig::default().temperature, 0.0);
    }

    #[test]
    fn default_ports_are_3070_and_3071() {
        let config = AppConfig::default();
        assert_eq!(config.bind_addr, "127.0.0.1:3070");
        assert_eq!(config.mcp_bind_addr, "127.0.0.1:3071");
    }

    #[test]
    fn max_steps_has_a_hard_cap() {
        assert!(AgentConfig::default().max_steps <= 30);
    }
}
