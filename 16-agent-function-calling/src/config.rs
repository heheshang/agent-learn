//! 全局配置。
//!
//! 所有「可调参数」集中在这里，来源有两档（后者覆盖前者）：
//!
//! ```text
//! 默认值  <  环境变量
//! ```
//!
//! 环境变量名带 `FC_` 前缀（Function Calling），
//! 避免和其它项目撞名（06 用 `RAG_`，13 用 `AGENT_`，14 用 `TOOL_`，15 用 `SQL_`）。

use crate::llm::ToolChoiceMode;

/// Agent 循环配置。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 最大步数。
    ///
    /// ⚠️ **必须有上限** —— 模型可能反复调同一个工具。
    ///
    /// Function Calling 的典型链路比 ReAct 短
    /// （不用「Thought/Action」文本解析，模型自己决定调不调），
    /// 所以 5 步足够覆盖「查一次 → 回答」。
    pub max_steps: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self { max_steps: 5 }
    }
}

/// 提示词配置。
#[derive(Debug, Clone)]
pub struct PromptConfig {
    /// 是否在提示词里强调「严格遵守 Schema」。
    ///
    /// ## 为什么这是个开关
    ///
    /// OpenAI 文档说 strict 模式能「确保函数调用可靠地遵守 schema」。
    /// 加上提示词里的一句话是**双保险**。
    ///
    /// 但它和 `strict` 一样**不是保证** ——
    /// DeepSeek 接受 `strict` 却不强制执行（实测）。
    /// 真正的兜底是**本地 Schema 校验**（在 `ToolRegistry` 里）。
    pub strict: bool,

    /// 工具选择策略。
    ///
    /// 四档对应 API 的 `tool_choice`：
    /// `auto` / `none` / `required` / `forced:工具名`。
    ///
    /// ## 为什么要暴露它
    ///
    /// `auto` 和 `none` 的对照能回答一个关键问题：
    /// 「**模型自己决定要不要用工具**」到底值多少。
    pub tool_choice: ToolChoiceMode,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            strict: true,
            tool_choice: ToolChoiceMode::Auto,
        }
    }
}

/// 工具配置。
#[derive(Debug, Clone)]
pub struct ToolConfig {
    /// 天气工具默认预报天数。
    pub weather_default_days: usize,

    /// 天气工具最大预报天数。
    pub weather_max_days: usize,

    /// 工具白名单（空 = 全部启用）。
    ///
    /// 对应 OpenAI 最佳实践里的「动态工具选择」：
    /// 「**保持初始可用函数的数量足够少**，以获得更高的准确率
    /// —— 建议一开始少于 20 个函数。」
    pub allowed_tools: Vec<String>,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            weather_default_days: 1,
            weather_max_days: 7,
            allowed_tools: Vec::new(),
        }
    }
}

/// 知识库目录（相对本项目目录）。
///
/// ## 为什么要有它
///
/// 13-agent-core 的知识库在 `13-agent-core/data/kb.md`。
/// 本项目**不复用**那个目录 —— 而是自带一份知识库，
/// 内容专门讲 Function Calling，这样 `search_knowledge` 工具
/// 问出来的答案和本项目的主题一致。
///
/// 路径是**相对**当前工作目录的，所以 `cargo run -p agent-16`
/// （在 workspace 根目录跑）会找不到 —— 这也是 Makefile 里
/// 用绝对路径 `export FC_DATA_DIR=...` 的原因。
pub fn default_data_dir() -> String {
    "data".to_string()
}

/// 应用总配置。
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// HTTP 监听地址。
    pub bind_addr: String,

    /// 模型名。
    pub model: String,

    /// 生成温度。
    ///
    /// Function Calling 需要**可复现** —— 温度太高会让工具调用序列飘忽。
    pub temperature: f64,

    /// 生成 max tokens。
    ///
    /// ⚠️ 每个 step 都要重新发一次请求，历史越来越长 ——
    /// 这个值限制的是**单次响应**的长度。
    pub max_tokens: u32,

    /// 是否关闭推理模型的「思考」模式。
    ///
    /// ## 默认 `true` 的理由（实测）
    ///
    /// 1. 推理模型的**思考 token 和正文共用 `max_tokens` 预算** ——
    ///    思考没结束正文就是空的。
    /// 2. 开思考模式后 DeepSeek **要求**回传 `reasoning_content`，
    ///    否则下一轮 400。关掉就少一个约束。
    ///
    /// 本项目的循环**已经正确实现了 reasoning 回传**，
    /// 所以设成 `false` 也能跑 —— 只是多一层约束。
    pub disable_reasoning: bool,

    /// Agent 循环配置。
    pub agent: AgentConfig,

    /// 提示词配置。
    pub prompt: PromptConfig,

    /// 工具配置。
    pub tool: ToolConfig,

    /// 知识库目录。
    ///
    /// `search_knowledge` 工具从这里建 BM25 索引。
    pub data_dir: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3090".to_string(),
            model: "deepseek-flash".to_string(),
            temperature: 0.0,
            max_tokens: 2048,
            disable_reasoning: true,
            agent: AgentConfig::default(),
            prompt: PromptConfig::default(),
            tool: ToolConfig::default(),
            data_dir: default_data_dir(),
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `FC_BIND_ADDR` | 主服务监听地址 | `127.0.0.1:3090` |
    /// | `FC_MODEL` | 模型名 | `deepseek-flash` |
    /// | `FC_TEMPERATURE` | 温度 | `0.0` |
    /// | `FC_MAX_TOKENS` | max tokens | `2048` |
    /// | `FC_MAX_STEPS` | 最大步数 | `5` |
    /// | `FC_STRICT` | 提示词里强调严格遵守 Schema | `true` |
    /// | `FC_TOOL_CHOICE` | `auto`/`none`/`required`/`forced:工具名` | `auto` |
    /// | `FC_REASONING` | 是否开推理模型的思考模式 | `false`（即关思考） |
    /// | `FC_WEATHER_DAYS` | 天气默认预报天数 | `1` |
    /// | `FC_ALLOWED_TOOLS` | 工具白名单（逗号分隔） | 空（全部） |
    /// | `FC_DATA_DIR` | 知识库目录 | `data` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("FC_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("FC_MODEL") {
            config.model = v;
        }
        if let Some(v) = env_f64("FC_TEMPERATURE") {
            config.temperature = v;
        }
        if let Some(v) = env_u32("FC_MAX_TOKENS") {
            config.max_tokens = v;
        }
        if let Some(v) = env_usize("FC_MAX_STEPS") {
            config.agent.max_steps = v.clamp(1, 30);
        }
        if let Some(v) = env_bool("FC_STRICT") {
            config.prompt.strict = v;
        }
        if let Some(v) = env_str("FC_TOOL_CHOICE")
            && let Some(m) = ToolChoiceMode::parse(&v)
        {
            config.prompt.tool_choice = m;
        }
        if let Some(v) = env_bool("FC_REASONING") {
            // ⚠️ 语义翻转：`FC_REASONING=1` = 开思考。
            config.disable_reasoning = !v;
        }
        if let Some(v) = env_usize("FC_WEATHER_DAYS") {
            config.tool.weather_default_days = v.clamp(1, 7);
        }
        if let Some(v) = env_str("FC_ALLOWED_TOOLS") {
            config.tool.allowed_tools = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if let Some(v) = env_str("FC_DATA_DIR") {
            config.data_dir = v;
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
             模型      : {} temp={} max_tokens={}\n\
             思考模式  : {}\n\
             最大步数  : {}\n\
             严格 Schema: {}\n\
             工具选择  : {}\n\
             天气天数  : 默认 {}（上限 {}）\n\
             工具白名单: {}",
            self.bind_addr,
            self.model,
            self.temperature,
            self.max_tokens,
            if self.disable_reasoning { "关" } else { "开" },
            self.agent.max_steps,
            if self.prompt.strict { "是" } else { "否" },
            self.prompt.tool_choice.name(),
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
    fn default_temperature_is_deterministic() {
        assert_eq!(AppConfig::default().temperature, 0.0);
    }

    #[test]
    fn default_port_is_3090() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3090");
    }

    #[test]
    fn reasoning_is_disabled_by_default() {
        // ⚠️ 默认关思考 —— 思考 token 会吃光 max_tokens 预算。
        assert!(AppConfig::default().disable_reasoning);
    }

    #[test]
    fn max_steps_has_a_hard_cap() {
        assert!(AgentConfig::default().max_steps <= 30);
    }

    #[test]
    fn default_max_steps_covers_the_typical_chain() {
        // 查一次工具 → 回答 = 2 步；给 5 步留了余量。
        assert!(AgentConfig::default().max_steps >= 3);
    }

    #[test]
    fn summary_mentions_key_settings() {
        let summary = AppConfig::default().summary();
        assert!(summary.contains("3090"));
        assert!(summary.contains("auto"));
        assert!(summary.contains("deepseek-flash"));
    }

    #[test]
    fn parses_tool_choice_from_env_value() {
        assert_eq!(
            ToolChoiceMode::parse("forced:get_weather"),
            Some(ToolChoiceMode::Forced("get_weather".to_string()))
        );
    }
}
