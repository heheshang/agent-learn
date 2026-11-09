//! 全局配置。
//!
//! 所有「可调参数」集中在这里，来源有两档（后者覆盖前者）：
//!
//! ```text
//! 默认值  <  环境变量
//! ```
//!
//! 环境变量名带 `MEM_` 前缀（Memory），
//! 避免和其它项目撞名（06 用 `RAG_`，13 用 `AGENT_`，14 用 `TOOL_`，
//! 15 用 `SQL_`，16 用 `FC_`）。

use crate::longterm::MemoryFact;
use crate::prompt::PromptConfig;

/// 存储配置。
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// checkpointer 的存储方式。
    pub checkpointer: CheckpointerKind,

    /// 长期记忆的存储方式。
    pub memory_store: MemoryStoreKind,

    /// 存储根目录（文件模式下用）。
    ///
    /// ## ⚠️ 相对路径的问题
    ///
    /// `cargo run -p agent-17` 在 workspace 根目录跑时 cwd 是根目录，
    /// `data` 会指向**根目录的** `data`（不存在）。
    /// 所以 Makefile 里用绝对路径 `export MEM_DATA_DIR=...`。
    pub data_dir: String,
}

impl StorageConfig {
    /// 线程状态（短期记忆）的目录。
    ///
    /// ## ⚠️ 为什么短期记忆和长期记忆要**分目录**
    ///
    /// 两者都用 `.json` 后缀，如果放在同一个目录里，
    /// `FileCheckpointer::list_threads` 会把**记忆文件也当成线程**列出来
    /// —— `GET /api/threads` 会返回一堆 `{hash}_{owner}.json` 的垃圾 id。
    ///
    /// 这是一个**静默**的错：接口不报错，但列表是脏的。
    /// 分目录是最省事的修法（比加前缀后缀更不容易漏）。
    pub fn threads_dir(&self) -> String {
        format!("{}/threads", self.data_dir.trim_end_matches('/'))
    }

    /// 长期记忆的目录。
    pub fn memories_dir(&self) -> String {
        format!("{}/memories", self.data_dir.trim_end_matches('/'))
    }
}

/// checkpointer 的存储方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointerKind {
    /// 内存（进程重启就没了）。
    Memory,
    /// 文件（进程重启后还能恢复）。
    File,
}

impl CheckpointerKind {
    /// 模式名。
    pub fn name(&self) -> &'static str {
        match self {
            CheckpointerKind::Memory => "memory",
            CheckpointerKind::File => "file",
        }
    }

    /// 从字符串解析。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "memory" | "mem" | "in-memory" => Some(CheckpointerKind::Memory),
            "file" | "disk" | "filesystem" => Some(CheckpointerKind::File),
            _ => None,
        }
    }
}

/// 长期记忆的存储方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStoreKind {
    /// 内存。
    Memory,
    /// 文件。
    File,
}

impl MemoryStoreKind {
    /// 模式名。
    pub fn name(&self) -> &'static str {
        match self {
            MemoryStoreKind::Memory => "memory",
            MemoryStoreKind::File => "file",
        }
    }

    /// 从字符串解析。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "memory" | "mem" | "in-memory" => Some(MemoryStoreKind::Memory),
            "file" | "disk" | "filesystem" => Some(MemoryStoreKind::File),
            _ => None,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            checkpointer: CheckpointerKind::File,
            memory_store: MemoryStoreKind::File,
            data_dir: "data/memory".to_string(),
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
    /// 记忆场景要**可复现** —— 温度高会让抽取出的事实飘忽。
    pub temperature: f64,

    /// 生成 max tokens。
    pub max_tokens: u32,

    /// 是否关闭推理模型的「思考」模式。
    ///
    /// ## 默认 `true` 的理由（沿用 07/13/16 的结论）
    ///
    /// 推理模型的**思考 token 与正文共用 `max_tokens` 预算** ——
    /// 思考没结束正文就是空的。
    ///
    /// 记忆项目对思考的收益也最低：抽取要的是结构化短输出，
    /// 「记住我的名字」不需要推理。
    pub disable_reasoning: bool,

    /// 存储配置。
    pub storage: StorageConfig,

    /// 提示词配置。
    pub prompt: PromptConfig,

    /// 是否启用长期记忆抽取。
    pub extract_memories: bool,

    /// 长期记忆召回条数上限。
    pub memory_top_k: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3100".to_string(),
            model: "deepseek-flash".to_string(),
            temperature: 0.0,
            max_tokens: 2048,
            disable_reasoning: true,
            storage: StorageConfig::default(),
            prompt: PromptConfig::default(),
            extract_memories: true,
            memory_top_k: 5,
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `MEM_BIND_ADDR` | 主服务监听地址 | `127.0.0.1:3100` |
    /// | `MEM_MODEL` | 模型名 | `deepseek-flash` |
    /// | `MEM_TEMPERATURE` | 温度 | `0.0` |
    /// | `MEM_MAX_TOKENS` | max tokens | `2048` |
    /// | `MEM_REASONING` | 是否开推理模型的思考模式 | `false`（即关思考） |
    /// | `MEM_CHECKPOINTER` | `memory` / `file` | `file` |
    /// | `MEM_MEMORY_STORE` | `memory` / `file` | `file` |
    /// | `MEM_DATA_DIR` | 存储根目录 | `data/memory` |
    /// | `MEM_MEMORY_TOP_K` | 长期记忆召回条数 | `5` |
    /// | `MEM_HISTORY_KEEP` | 历史保留条数（`0` = 全部） | `20` |
    /// | `MEM_EXTRACT` | 是否启用长期记忆抽取 | `true` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("MEM_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("MEM_MODEL") {
            config.model = v;
        }
        if let Some(v) = env_f64("MEM_TEMPERATURE") {
            config.temperature = v;
        }
        if let Some(v) = env_u32("MEM_MAX_TOKENS") {
            config.max_tokens = v;
        }
        if let Some(v) = env_bool("MEM_REASONING") {
            // ⚠️ 语义翻转：`MEM_REASONING=1` = 开思考。
            config.disable_reasoning = !v;
        }
        if let Some(v) = env_str("MEM_CHECKPOINTER")
            && let Some(kind) = CheckpointerKind::parse(&v)
        {
            config.storage.checkpointer = kind;
        }
        if let Some(v) = env_str("MEM_MEMORY_STORE")
            && let Some(kind) = MemoryStoreKind::parse(&v)
        {
            config.storage.memory_store = kind;
        }
        if let Some(v) = env_str("MEM_DATA_DIR") {
            config.storage.data_dir = v;
        }
        if let Some(v) = env_usize("MEM_MEMORY_TOP_K") {
            config.memory_top_k = v.clamp(1, 50);
        }
        if let Some(v) = env_usize("MEM_HISTORY_KEEP") {
            config.prompt.history_keep = v;
        }
        if let Some(v) = env_bool("MEM_EXTRACT") {
            config.extract_memories = v;
        }

        // 提示词里的 top_k 要和顶层保持一致 —— 两处都可能被单独设置。
        config.prompt.memory_top_k = config.memory_top_k;

        config
    }

    /// 配置摘要（启动时打印）。
    pub fn summary(&self) -> String {
        format!(
            "监听地址  : {}\n\
             模型      : {} temp={} max_tokens={}\n\
             思考模式  : {}\n\
             存储      : checkpointer={} memory_store={}\n\
             存储目录  : {}\n\
             记忆抽取  : {}\n\
             召回条数  : {}\n\
             历史保留  : {}",
            self.bind_addr,
            self.model,
            self.temperature,
            self.max_tokens,
            if self.disable_reasoning { "关" } else { "开" },
            self.storage.checkpointer.name(),
            self.storage.memory_store.name(),
            self.storage.data_dir,
            if self.extract_memories { "开" } else { "关" },
            self.memory_top_k,
            if self.prompt.history_keep == 0 {
                "全部".to_string()
            } else {
                self.prompt.history_keep.to_string()
            },
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

/// 构造一条示例事实（给 CLI 演示用）。
pub fn demo_fact(owner: &str) -> MemoryFact {
    MemoryFact::new("用户", "名字", "Bob", owner).with_importance(0.9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_port_is_3100() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3100");
    }

    #[test]
    fn default_is_deterministic() {
        assert_eq!(AppConfig::default().temperature, 0.0);
    }

    #[test]
    fn reasoning_is_disabled_by_default() {
        // ⚠️ 默认关思考 —— 思考 token 会吃光 max_tokens 预算。
        assert!(AppConfig::default().disable_reasoning);
    }

    #[test]
    fn default_storage_is_file_based() {
        // ⚠️ 默认必须是 file —— 内存版进程一重启就全没了，
        //    那不叫「持久化」。
        let config = AppConfig::default();
        assert_eq!(config.storage.checkpointer, CheckpointerKind::File);
        assert_eq!(config.storage.memory_store, MemoryStoreKind::File);
    }

    #[test]
    fn parses_storage_kinds() {
        assert_eq!(
            CheckpointerKind::parse("memory"),
            Some(CheckpointerKind::Memory)
        );
        assert_eq!(
            CheckpointerKind::parse("FILE"),
            Some(CheckpointerKind::File)
        );
        assert_eq!(CheckpointerKind::parse("nope"), None);

        assert_eq!(MemoryStoreKind::parse("file"), Some(MemoryStoreKind::File));
        assert_eq!(MemoryStoreKind::parse("nope"), None);
    }

    #[test]
    fn summary_mentions_key_settings() {
        let summary = AppConfig::default().summary();
        assert!(summary.contains("3100"));
        assert!(summary.contains("deepseek-flash"));
        assert!(summary.contains("file"));
    }

    #[test]
    fn demo_fact_is_scoped_to_the_owner() {
        let fact = demo_fact("alice");
        assert_eq!(fact.owner, "alice");
        assert_eq!(fact.subject, "用户");
    }
}
