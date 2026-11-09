//! 全局配置。
//!
//! 所有「可调参数」集中在这里，来源有两档（后者覆盖前者）：
//!
//! ```text
//! 默认值  <  环境变量
//! ```
//!
//! 环境变量名带 `SQL_` 前缀，避免和其它项目撞名
//! （06 用 `RAG_`，07 用 `QUERY_`，13 用 `AGENT_`，14 用 `TOOL_`）。

/// Agent 循环配置。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 最大步数。
    ///
    /// ⚠️ **必须有上限** —— 模型可能反复调同一个工具。
    ///
    /// SQL Agent 的典型链路是
    /// 「list_tables → schema → checker → query」，
    /// 所以至少要给 6 步（4 步 + 收尾）。
    pub max_steps: usize,

    /// 是否**强制先查库再作答**。
    ///
    /// ## ⚠️ 为什么这个开关是必需的（实测踩坑）
    ///
    /// 实测发现模型经常**只查 schema 就直接给答案** —— 不执行任何
    /// `sql_db_query`，然后编一个看起来很合理的数字。
    ///
    /// 一次真实运行（`deepseek-flash`，temp=0）：
    ///
    /// ```text
    /// 步骤 1: sql_db_list_tables  → 7 张表
    /// 步骤 2: sql_db_schema      → Track / Genre 的 DDL + 3 行样例
    /// 步骤 3: Final Answer: 摇滚，平均 393599.5 毫秒
    /// ```
    ///
    /// 而真实 SQL 的答案是**金属 391064ms** —— 模型**根本没查库**，
    /// 数字是编的。样例数据里只有 3 行，不可能算出平均值。
    ///
    /// ## 为什么这很严重
    ///
    /// 提示词里写了「答案里的每个数字都必须来自查询结果」，
    /// 但**提示词是软的**。模型完全可以不听。
    ///
    /// 如果放任不管，本项目最核心的三样东西全部形同虚设：
    ///
    /// - 护栏拦住的 SQL 是 0 条（因为压根没发 SQL）
    /// - 只读连接从未被触发
    /// - 安全审计里 `executed_sql` 永远为空
    ///
    /// ## 修法：把「软约束」变成「硬约束」
    ///
    /// 模型给出 `Final Answer` 时，如果**从未成功执行过查询**，
    /// 就把这条答案**退回**成一条 Observation 喂回模型，
    /// 要求它先查库再作答。
    ///
    /// 这和 13/14 的「工具失败不终止循环」是同一个思路 ——
    /// 用**机制**兜住模型的不确定性，而不是靠祈祷。
    pub require_query: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 10,
            require_query: true,
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

/// 数据库配置。
#[derive(Debug, Clone)]
pub struct DbConfig {
    /// 数据库文件路径。
    pub path: String,

    /// 种子 SQL 路径（用于首次生成数据库）。
    pub seed_path: String,

    /// 单次查询最多返回多少行。
    pub max_rows: usize,

    /// 是否在数据库文件缺失时**自动**用种子 SQL 生成。
    ///
    /// ## 为什么默认 `true`
    ///
    /// 学习项目的首要目标是「clone 下来就能跑」。
    /// 让用户先跑 `make seed` 是额外的摩擦。
    ///
    /// 但生产环境应该是 `false` ——
    /// 数据库文件缺失时**静默重建**是危险行为
    /// （可能连到了错误的库，或者用旧数据覆盖了新数据）。
    pub auto_seed: bool,
}

impl Default for DbConfig {
    fn default() -> Self {
        Self {
            path: "data/chinook.db".to_string(),
            seed_path: "data/seed.sql".to_string(),
            max_rows: 100,
            auto_seed: true,
        }
    }
}

/// 检查器配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckerMode {
    /// 纯本地静态检查（默认，零成本、确定性）。
    Static,

    /// 调模型做语义检查（对齐 LangChain 教程，多一次模型调用）。
    Llm,
}

impl CheckerMode {
    /// 模式名。
    pub fn name(self) -> &'static str {
        match self {
            CheckerMode::Static => "static",
            CheckerMode::Llm => "llm",
        }
    }

    /// 从字符串解析。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "static" | "local" | "off" => Some(CheckerMode::Static),
            "llm" | "model" | "on" => Some(CheckerMode::Llm),
            _ => None,
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
    /// SQL Agent 需要**可复现** —— 温度太高会让查询序列飘忽。
    pub temperature: f64,

    /// 生成 max tokens。
    ///
    /// ⚠️ 每个 step 都要重新发一次请求，历史越来越长 ——
    /// 这个值限制的是**单次响应**的长度。
    ///
    /// SQL Agent 每步输出的是 `Thought / Action / Action Input`，
    /// 而 `Action Input` 里可能嵌着一条长 SQL —— 所以要给够。
    pub max_tokens: u32,

    /// Agent 循环配置。
    pub agent: AgentConfig,

    /// 提示词配置。
    pub prompt: PromptConfig,

    /// 数据库配置。
    pub db: DbConfig,

    /// 检查器模式。
    pub checker: CheckerMode,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3080".to_string(),
            model: "deepseek-flash".to_string(),
            temperature: 0.0,
            max_tokens: 2048,
            agent: AgentConfig::default(),
            prompt: PromptConfig::default(),
            db: DbConfig::default(),
            checker: CheckerMode::Static,
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `SQL_BIND_ADDR` | 主服务监听地址 | `127.0.0.1:3080` |
    /// | `SQL_MODEL` | 模型名 | `deepseek-flash` |
    /// | `SQL_TEMPERATURE` | 温度 | `0.0` |
    /// | `SQL_MAX_TOKENS` | max tokens | `2048` |
    /// | `SQL_MAX_STEPS` | 最大步数 | `10` |
    /// | `SQL_REQUIRE_QUERY` | 必须先查库才允许作答 | `true` |
    /// | `SQL_FEW_SHOT` | 是否带 few-shot 示例 | `true` |
    /// | `SQL_DB_PATH` | 数据库文件路径 | `data/chinook.db` |
    /// | `SQL_SEED_PATH` | 种子 SQL 路径 | `data/seed.sql` |
    /// | `SQL_MAX_ROWS` | 单次查询最多返回行数 | `100` |
    /// | `SQL_AUTO_SEED` | 库缺失时自动生成 | `true` |
    /// | `SQL_CHECKER` | `static` / `llm` | `static` |
    pub fn from_env() -> Self {
        let mut config = AppConfig::default();

        if let Some(v) = env_str("SQL_BIND_ADDR") {
            config.bind_addr = v;
        }
        if let Some(v) = env_str("SQL_MODEL") {
            config.model = v;
        }
        if let Some(v) = env_f64("SQL_TEMPERATURE") {
            config.temperature = v;
        }
        if let Some(v) = env_u32("SQL_MAX_TOKENS") {
            config.max_tokens = v;
        }
        if let Some(v) = env_usize("SQL_MAX_STEPS") {
            config.agent.max_steps = v.clamp(1, 30);
        }
        if let Some(v) = env_bool("SQL_FEW_SHOT") {
            config.prompt.few_shot = v;
        }
        if let Some(v) = env_str("SQL_DB_PATH") {
            config.db.path = v;
        }
        if let Some(v) = env_str("SQL_SEED_PATH") {
            config.db.seed_path = v;
        }
        if let Some(v) = env_usize("SQL_MAX_ROWS") {
            config.db.max_rows = v.clamp(1, 1000);
        }
        if let Some(v) = env_bool("SQL_AUTO_SEED") {
            config.db.auto_seed = v;
        }
        if let Some(v) = env_bool("SQL_REQUIRE_QUERY") {
            config.agent.require_query = v;
        }
        if let Some(v) = env_str("SQL_CHECKER")
            && let Some(m) = CheckerMode::parse(&v)
        {
            config.checker = m;
        }

        config
    }

    /// 配置摘要（启动时打印）。
    pub fn summary(&self) -> String {
        format!(
            "监听地址  : {}\n\
             模型      : {} temp={} max_tokens={}\n\
             最大步数  : {}\n\
             必须查库  : {}\n\
             数据库    : {}\n\
             种子 SQL  : {}\n\
             自动建库  : {}\n\
             最大行数  : {}\n\
             检查器    : {}\n\
             few-shot  : {}",
            self.bind_addr,
            self.model,
            self.temperature,
            self.max_tokens,
            self.agent.max_steps,
            if self.agent.require_query {
                "是"
            } else {
                "否"
            },
            self.db.path,
            self.db.seed_path,
            if self.db.auto_seed { "是" } else { "否" },
            self.db.max_rows,
            self.checker.name(),
            if self.prompt.few_shot { "是" } else { "否" },
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
    fn parses_checker_modes() {
        assert_eq!(CheckerMode::parse("static"), Some(CheckerMode::Static));
        assert_eq!(CheckerMode::parse("LLM"), Some(CheckerMode::Llm));
        assert_eq!(CheckerMode::parse("nope"), None);
    }

    #[test]
    fn default_temperature_is_deterministic() {
        assert_eq!(AppConfig::default().temperature, 0.0);
    }

    #[test]
    fn default_port_is_3080() {
        assert_eq!(AppConfig::default().bind_addr, "127.0.0.1:3080");
    }

    #[test]
    fn max_steps_has_a_hard_cap() {
        assert!(AgentConfig::default().max_steps <= 30);
    }

    #[test]
    fn default_max_steps_fits_the_discovery_chain() {
        // list_tables → schema → checker → query → 收尾 = 5 步，
        // 默认 10 步留了余量。
        assert!(AgentConfig::default().max_steps >= 6);
    }

    #[test]
    fn default_checker_is_static() {
        // ⚠️ 默认必须是静态检查器 —— 零成本、确定性。
        assert_eq!(AppConfig::default().checker, CheckerMode::Static);
    }

    #[test]
    fn summary_mentions_key_settings() {
        let summary = AppConfig::default().summary();
        assert!(summary.contains("3080"));
        assert!(summary.contains("data/chinook.db"));
        assert!(summary.contains("static"));
    }
}
