//! 全局配置。
//!
//! 环境变量前缀用 `SUM_`（Summary），
//! 避免和其它项目撞名（13 用 `AGENT_`，14 用 `TOOL_`，15 用 `SQL_`，
//! 16 用 `FC_`，17 用 `MEM_`，18 用 `ERR_`）。

use std::path::PathBuf;

use agent_16::llm::ToolChoiceMode;

/// Web 搜索后端。
///
/// ## 为什么有多个后端
///
/// 搜索服务的可用性**因网络环境而异** —— 有的网络能连 Google，
/// 有的只能连 Bing，还有的是完全隔离的内网。
///
/// 所以本项目把「搜索」抽象成 trait，
/// 让上层 Agent 逻辑**不依赖任何一家搜索服务**。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchBackend {
    /// Bing 网页版（解析 HTML 结果页）。
    ///
    /// 实测在受限网络里**唯一稳定可用**的通用搜索引擎。
    Bing,

    /// GitHub 搜索 API（只搜仓库 / 代码）。
    ///
    /// 技术调研场景下往往比通用搜索引擎更准。
    Github,

    /// 不用网络搜索（只靠本地知识库）。
    Offline,
}

impl SearchBackend {
    /// 解析环境变量值。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bing" => Some(Self::Bing),
            "github" => Some(Self::Github),
            "offline" | "none" => Some(Self::Offline),
            _ => None,
        }
    }

    /// 名称。
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bing => "bing",
            Self::Github => "github",
            Self::Offline => "offline",
        }
    }
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
    /// 周报场景要**可复现** —— 温度高会让同一个输入产出不同的总结，
    /// 对照实验的差值就说不清了。
    pub temperature: f64,

    /// 生成 max tokens。
    pub max_tokens: u32,

    /// 是否关闭推理模型的「思考」模式。
    pub disable_reasoning: bool,

    /// Agent 最大步数。
    pub max_steps: usize,

    /// 工具选择策略。
    pub tool_choice: ToolChoiceMode,

    /// 工具的**额外**重试次数（总次数 = 1 + max_retries）。
    pub max_retries: usize,

    /// 重试退避的基准延迟（毫秒）。
    pub base_delay_ms: u64,

    /// 重试退避的上限（毫秒）。
    pub max_delay_ms: u64,

    /// 本地知识库目录（RAG 侧的语料）。
    pub kb_dir: PathBuf,

    /// Web 搜索后端。
    pub search_backend: SearchBackend,

    /// 单次搜索返回的最大结果数。
    pub search_top_k: usize,

    /// Web 搜索 / 抓取的超时（秒）。
    pub web_timeout_secs: u64,

    /// GitHub 搜索用的 token（可选）。
    ///
    /// 没有 token 也能用（匿名请求），只是限流更严。
    pub github_token: Option<String>,

    /// 周报里每类内容最多保留多少条证据。
    pub max_evidence_per_section: usize,

    /// 周报输出目录。
    pub output_dir: PathBuf,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3120".to_string(),
            model: "deepseek-flash".to_string(),
            temperature: 0.0,
            max_tokens: 4096,
            disable_reasoning: true,
            max_steps: 6,
            tool_choice: ToolChoiceMode::Auto,
            max_retries: 2,
            base_delay_ms: 50,
            max_delay_ms: 2000,
            kb_dir: PathBuf::from("data/kb"),
            search_backend: SearchBackend::Bing,
            search_top_k: 5,
            web_timeout_secs: 15,
            github_token: None,
            max_evidence_per_section: 8,
            output_dir: PathBuf::from("data/reports"),
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `SUM_BIND_ADDR` | 主服务监听地址 | `127.0.0.1:3120` |
    /// | `SUM_MODEL` | 模型名 | `deepseek-flash` |
    /// | `SUM_TEMPERATURE` | 温度 | `0.0` |
    /// | `SUM_MAX_TOKENS` | max tokens | `4096` |
    /// | `SUM_REASONING` | 是否开推理模型的思考模式 | `false`（即关思考） |
    /// | `SUM_MAX_STEPS` | Agent 最大步数 | `6` |
    /// | `SUM_TOOL_CHOICE` | `auto`/`none`/`required`/`forced:工具名` | `auto` |
    /// | `SUM_MAX_RETRIES` | 工具额外重试次数 | `2` |
    /// | `SUM_BASE_DELAY_MS` | 退避基准延迟（毫秒） | `50` |
    /// | `SUM_MAX_DELAY_MS` | 退避上限（毫秒） | `2000` |
    /// | `SUM_KB_DIR` | 本地知识库目录 | `data/kb` |
    /// | `SUM_SEARCH_BACKEND` | `bing`/`github`/`offline` | `bing` |
    /// | `SUM_SEARCH_TOP_K` | 搜索返回条数上限 | `5` |
    /// | `SUM_WEB_TIMEOUT_SECS` | Web 请求超时（秒） | `15` |
    /// | `SUM_GITHUB_TOKEN` | GitHub token（可选） | 无 |
    /// | `SUM_MAX_EVIDENCE` | 每类内容最多保留的证据条数 | `8` |
    /// | `SUM_OUTPUT_DIR` | 周报输出目录 | `data/reports` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("SUM_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("SUM_MODEL") {
            config.model = v;
        }
        if let Some(v) = env_f64("SUM_TEMPERATURE") {
            config.temperature = v;
        }
        if let Some(v) = env_u32("SUM_MAX_TOKENS") {
            config.max_tokens = v;
        }
        if let Some(v) = env_bool("SUM_REASONING") {
            // ⚠️ 语义翻转：`SUM_REASONING=1` = 开思考。
            config.disable_reasoning = !v;
        }
        if let Some(v) = env_usize("SUM_MAX_STEPS") {
            config.max_steps = v.clamp(1, 30);
        }
        if let Some(v) = env_str("SUM_TOOL_CHOICE")
            && let Some(m) = ToolChoiceMode::parse(&v)
        {
            config.tool_choice = m;
        }
        if let Some(v) = env_usize("SUM_MAX_RETRIES") {
            config.max_retries = v.min(10);
        }
        if let Some(v) = env_u64("SUM_BASE_DELAY_MS") {
            config.base_delay_ms = v;
        }
        if let Some(v) = env_u64("SUM_MAX_DELAY_MS") {
            config.max_delay_ms = v;
        }
        if let Some(v) = env_str("SUM_KB_DIR") {
            config.kb_dir = PathBuf::from(v);
        }
        if let Some(v) = env_str("SUM_SEARCH_BACKEND")
            && let Some(b) = SearchBackend::parse(&v)
        {
            config.search_backend = b;
        }
        if let Some(v) = env_usize("SUM_SEARCH_TOP_K") {
            config.search_top_k = v.clamp(1, 20);
        }
        if let Some(v) = env_u64("SUM_WEB_TIMEOUT_SECS") {
            config.web_timeout_secs = v.clamp(1, 120);
        }
        if let Some(v) = env_str("SUM_GITHUB_TOKEN") {
            config.github_token = Some(v);
        }
        if let Some(v) = env_usize("SUM_MAX_EVIDENCE") {
            config.max_evidence_per_section = v.clamp(1, 50);
        }
        if let Some(v) = env_str("SUM_OUTPUT_DIR") {
            config.output_dir = PathBuf::from(v);
        }

        config
    }

    /// 配置摘要（启动时打印）。
    pub fn summary(&self) -> String {
        format!(
            "监听地址  : {}\n\
             模型      : {} temp={} max_tokens={}\n\
             思考模式  : {}\n\
             最大步数  : {}\n\
             工具选择  : {}\n\
             工具重试  : 额外 {} 次（退避 {} ms ~ {} ms）\n\
             知识库    : {}\n\
             搜索后端  : {}（top_k={}，超时 {} s）\n\
             输出目录  : {}",
            self.bind_addr,
            self.model,
            self.temperature,
            self.max_tokens,
            if self.disable_reasoning { "关" } else { "开" },
            self.max_steps,
            self.tool_choice.name(),
            self.max_retries,
            self.base_delay_ms,
            self.max_delay_ms,
            self.kb_dir.display(),
            self.search_backend.name(),
            self.search_top_k,
            self.web_timeout_secs,
            self.output_dir.display(),
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

/// 读一个 `u64` 环境变量。
fn env_u64(key: &str) -> Option<u64> {
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
    fn default_port_is_3120() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3120");
    }

    #[test]
    fn default_is_deterministic() {
        // ⚠️ 周报要可复现 —— 默认温度必须是 0。
        assert_eq!(AppConfig::default().temperature, 0.0);
    }

    #[test]
    fn default_backend_is_bing() {
        assert_eq!(AppConfig::default().search_backend, SearchBackend::Bing);
    }

    #[test]
    fn parses_backends() {
        assert_eq!(SearchBackend::parse("BING"), Some(SearchBackend::Bing));
        assert_eq!(
            SearchBackend::parse("offline"),
            Some(SearchBackend::Offline)
        );
        assert_eq!(SearchBackend::parse("nope"), None);
    }

    #[test]
    fn summary_mentions_key_settings() {
        let summary = AppConfig::default().summary();
        assert!(summary.contains("3120"));
        assert!(summary.contains("deepseek-flash"));
        assert!(summary.contains("bing"));
    }
}
