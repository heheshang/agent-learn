//! 三个具体的工具 —— 覆盖 Function Calling 的三类典型场景。
//!
//! | 工具 | 类型 | 特点 |
//! |---|---|---|
//! | [`WeatherTool`] | **联网**（异步） | 真·外部依赖，会失败，需要重试语义 |
//! | [`KnowledgeTool`] | **本地检索**（同步） | 零延迟、确定性、离线可用 |
//! | [`CalculateTool`] | **纯计算**（同步） | 零依赖、结果精确 |
//!
//! ## 为什么要有三种不同性质的工具
//!
//! OpenAI 文档说工具「可以是任何东西」——但实际上工具的**失败模式**各不相同，
//! 而这直接决定了 Agent 循环该怎么处理失败：
//!
//! - **联网工具**：上游可能超时、限流、返回 5xx
//! - **检索工具**：可能「查不到」（这不是错误，是结果）
//! - **计算工具**：只有「表达式非法」这一种失败
//!
//! ## 同步工具怎么包成 async
//!
//! 14 的 `Tool` trait 是 **async** 的（因为要等 HTTP）。
//! 13 的 `SearchTool` / `CalculateTool` 是**同步**的。
//!
//! 本项目的 `FunctionTool` trait 统一是 async，
//! 所以这里用两个**适配器**把同步工具包起来 ——
//! 包在 `async move` 里立即完成，不会真的阻塞。
//!
//! ⚠️ 但**不能**把 CPU 密集的同步逻辑直接塞进 async 块 ——
//! 那会卡住 tokio 的工作线程。
//! 本项目的检索和计算都是毫秒级的，所以没问题。

use agent_13::Tool as _;
use agent_14::Tool as _;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::error::ToolError;
use crate::tool::{FunctionTool, ToolContext, ToolFuture, ToolOutcome};
use crate::tools::ToolSpec;

// ─────────────────────────────────────────────────────────────────────
// ① 天气工具（联网）
// ─────────────────────────────────────────────────────────────────────

/// `get_weather` 工具名。
pub const GET_WEATHER_TOOL_NAME: &str = "get_weather";

/// 天气工具的参数 Schema。
///
/// ## `additionalProperties: false` 的意义
///
/// 模型经常会**自作主张**地加参数（`{"location": "北京", "unit": "C"}`）。
/// 拒绝未知参数能让它**立刻**看到「`unit` 不是合法参数」，
/// 而不是悄悄忽略一个拼错的参数导致查错地方。
///
/// ## 为什么 `days` 用 `enum` 而不是自由整数
///
/// OpenAI 最佳实践里明确说：
/// 「**使用枚举和对象结构来防止无效状态**」。
/// `enum` 就是「用类型系统消灭非法状态」在 Schema 层的体现。
///
/// ## ⚠️ 参数名必须和**实现**严格一致（本项目实测踩过的坑）
///
/// 本项目第一版的 Schema 写的是 `city`，而底层实现读的是 `location`。
/// 结果是模型陷入了**拉锯战**：
///
/// ```text
/// 第 1 步：get_weather({"location":"北京"})
///         → 错误：`city` 是必填参数，但没有被提供；`location` 是未知参数
/// 第 2 步：get_weather({"city":"北京"})
///         → 错误：`location` 不能为空
/// 第 3 步：get_weather({"city":"北京","location":"北京"})
///         → 错误：`location` 是未知参数
/// ```
///
/// 模型试了三种组合才放弃 —— **烧掉了 8 次工具调用和 5 步预算**。
///
/// 教训：**Schema 是「对模型的承诺」**。
/// 承诺了 `city` 就必须真的读 `city`，否则模型的每一次自我纠正
/// 都在撞一堵看不见的墙。
///
/// 本项目最终**统一用 `location`** —— 与底层实现对齐。
pub fn weather_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "城市名，例如 \"北京\"、\"上海\"、\"Tokyo\"。必须是城市名，不是省或国家。"
            },
            "days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 7,
                "default": 1,
                "description": "预报天数。1 表示只看当前实况，2~7 表示今天起未来几天的预报。"
            }
        },
        "required": ["location"],
        "additionalProperties": false
    })
}

/// 天气工具 —— 包 14 的 `WeatherTool`，转成本项目的 `FunctionTool`。
///
/// ## 为什么要包一层而不是直接用 14 的
///
/// 14 的 `WeatherTool` 实现的是 14 的 `Tool` trait（返回 `ToolOutcome`，
/// 错误用 `Err`），而本项目的 trait 形状不同（`ToolSpec` 多一个 `strict`）。
///
/// 包一层的好处是：**14 修了什么，这里自动继承**。
pub struct WeatherTool {
    inner: Arc<agent_14::weather::WeatherTool>,
}

impl std::fmt::Debug for WeatherTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeatherTool").finish_non_exhaustive()
    }
}

impl WeatherTool {
    /// 创建天气工具（连接真实的 Open-Meteo 服务）。
    pub fn new(default_days: usize, max_days: usize) -> Self {
        Self {
            inner: Arc::new(agent_14::weather::WeatherTool::new(default_days, max_days)),
        }
    }
}

impl FunctionTool for WeatherTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            GET_WEATHER_TOOL_NAME,
            "查询某个城市的实时天气和未来几天的预报。\
             需要实时的天气信息（温度、湿度、风力、降水）时必须调用这个工具，\
             不要凭记忆回答天气问题。如果工具返回错误，请根据错误提示修正参数后重试。",
            weather_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            // 14 的 `ToolContext` 和本项目的字段一致，直接构造。
            let inner_ctx = agent_14::tool::ToolContext {
                user_id: ctx.user_id.clone(),
                timeout_ms: 0,
            };

            match self.inner.call(args, &inner_ctx).await {
                Ok(outcome) => Ok(ToolOutcome::ok_with_data(
                    outcome.tool,
                    outcome.observation,
                    outcome.data.unwrap_or(Value::Null),
                )),
                Err(err) => Err(err),
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ② 知识库检索工具（本地）
// ─────────────────────────────────────────────────────────────────────

/// `search_knowledge` 工具名。
pub const SEARCH_KNOWLEDGE_TOOL_NAME: &str = "search_knowledge";

/// 知识库检索工具的参数 Schema。
pub fn knowledge_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词。用知识库里可能出现的词，而不是用户的口语原话。"
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
                "description": "返回的片段数量，默认 3。"
            }
        },
        "required": ["query"],
        "additionalProperties": false
    })
}

/// 知识库检索工具 —— 包 13 的 `SearchTool`（BM25）。
///
/// ## 为什么要复用 13 的
///
/// 13 的 `SearchTool` 是**同步**的（纯内存操作），
/// 但它已经把「观察文本要截断」「命中要渲染成可读格式」这些细节处理好了。
/// 重写一遍毫无意义。
pub struct KnowledgeTool {
    inner: Arc<agent_13::tools::SearchTool>,
}

impl std::fmt::Debug for KnowledgeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnowledgeTool").finish_non_exhaustive()
    }
}

impl KnowledgeTool {
    /// 创建知识库工具。
    pub fn new(index: Arc<rag_08::bm25::Bm25Index>) -> Self {
        Self {
            inner: Arc::new(agent_13::tools::SearchTool::new(index, 3, 5, 2000)),
        }
    }
}

impl FunctionTool for KnowledgeTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            SEARCH_KNOWLEDGE_TOOL_NAME,
            "在本地知识库中做关键词检索，返回最相关的文档片段。\
             用于回答关于 RAG、Agent、ReAct、Function Calling、记忆与工具使用的事实性问题。\
             检索不到时不要编造，工具会明确告诉你「没有检索到相关资料」。",
            knowledge_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            // 13 的工具是同步的 —— 直接在 async 块里跑。
            //
            // ⚠️ BM25 检索是毫秒级的纯内存操作，
            //    放进 async 块不会卡住 tokio 工作线程。
            let outcome = self.inner.run(args);

            Ok(ToolOutcome::ok(
                SEARCH_KNOWLEDGE_TOOL_NAME,
                outcome.observation,
            ))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ③ 计算工具（纯计算）
// ─────────────────────────────────────────────────────────────────────

/// `calculate` 工具名。
pub const CALCULATE_TOOL_NAME: &str = "calculate";

/// 计算工具的参数 Schema。
pub fn calculate_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "算术表达式，例如 (12 + 34) * 2。支持 + - * / 和括号。"
            }
        },
        "required": ["expression"],
        "additionalProperties": false
    })
}

/// 计算工具 —— 包 13 的 `CalculateTool`。
///
/// ## 为什么必须有这个工具
///
/// MRKL 论文的实验结论：LLM（7B）**很难可靠地提取出算术运算的正确参数**。
/// 口头表述的算术题比写明式的算术题难得多。
/// 所以「把算数交给工具」不是锦上添花，是必要的。
pub struct CalculateTool {
    inner: agent_13::tools::CalculateTool,
}

impl std::fmt::Debug for CalculateTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalculateTool").finish()
    }
}

impl Default for CalculateTool {
    fn default() -> Self {
        Self {
            inner: agent_13::tools::CalculateTool,
        }
    }
}

impl FunctionTool for CalculateTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            CALCULATE_TOOL_NAME,
            "计算一个算术表达式并返回结果。支持 + - * / 和括号。\
             涉及具体数值计算时必须调用这个工具，不要自己心算。",
            calculate_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let outcome = self.inner.run(args);

            if outcome.ok {
                Ok(ToolOutcome::ok(CALCULATE_TOOL_NAME, outcome.observation))
            } else {
                Err(ToolError::execution(
                    CALCULATE_TOOL_NAME,
                    outcome.observation,
                ))
            }
        })
    }
}

/// 构建默认的工具注册表。
///
/// ## 三个工具对应三种「失败语义」
///
/// - `get_weather` —— 上游可能挂（会失败，要重试）
/// - `search_knowledge` —— 查不到是**正常结果**，不是失败
/// - `calculate` —— 表达式非法才是失败
pub fn default_registry(index: Arc<rag_08::bm25::Bm25Index>) -> ToolRegistry {
    ToolRegistry::new(vec![
        Arc::new(WeatherTool::new(1, 7)),
        Arc::new(KnowledgeTool::new(index)),
        Arc::new(CalculateTool::default()),
    ])
}

/// 便捷别名。
pub use crate::tool::ToolRegistry;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolContext;

    #[test]
    fn weather_schema_has_the_right_shape() {
        // ⚠️ 天气工具**不开** strict（因为 `days` 有默认值，
        //    strict 会要求模型每次都填它）。
        //    所以只检查「形状」，不检查 strict 的 required 约束。
        let schema = weather_schema();
        assert!(crate::tools::check_schema_shape(&schema).is_ok());
    }

    #[test]
    fn knowledge_schema_has_the_right_shape() {
        let schema = knowledge_schema();
        assert!(crate::tools::check_schema_shape(&schema).is_ok());
    }

    #[test]
    fn calculate_schema_has_the_right_shape() {
        let schema = calculate_schema();
        assert!(crate::tools::check_schema_shape(&schema).is_ok());
    }

    #[test]
    fn optional_parameters_use_defaults_not_strict() {
        // ⚠️ 用 `default` 表达可选参数的工具**不能开 strict** ——
        //    strict 会要求模型每次都填这些字段。
        //
        //    这里显式断言这一点：三个工具都不开 strict。
        for spec in default_registry(Arc::new(rag_08::bm25::Bm25Index::new())).specs() {
            assert!(
                !spec.strict,
                "工具 {} 不该开 strict（它有带默认值的可选参数）",
                spec.name
            );
        }
    }

    #[test]
    fn tool_names_are_valid() {
        for name in [
            GET_WEATHER_TOOL_NAME,
            SEARCH_KNOWLEDGE_TOOL_NAME,
            CALCULATE_TOOL_NAME,
        ] {
            assert!(
                crate::tools::is_valid_tool_name(name),
                "工具名 `{name}` 不合法"
            );
        }
    }

    #[tokio::test]
    async fn calculate_tool_computes() {
        let tool = CalculateTool::default();
        let outcome = tool
            .call(
                &json!({ "expression": "(12 + 34) * 2" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap();

        assert!(outcome.ok);
        assert!(outcome.observation.contains("92"));
    }

    #[tokio::test]
    async fn calculate_tool_rejects_bad_expression() {
        let tool = CalculateTool::default();
        let err = tool
            .call(&json!({ "expression": "1 +" }), &ToolContext::empty())
            .await
            .unwrap_err();

        assert!(err.to_string().contains("calculate"));
    }

    #[tokio::test]
    async fn knowledge_tool_reports_empty_result_as_success() {
        // ⚠️ 「查不到」是**正常结果**，不是失败 ——
        //    模型需要知道「确实没有」而不是「工具坏了」。
        let index = rag_08::bm25::Bm25Index::new();
        let tool = KnowledgeTool::new(Arc::new(index));

        let outcome = tool
            .call(&json!({ "query": "不存在的主题" }), &ToolContext::empty())
            .await
            .unwrap();

        assert!(outcome.ok);
        assert!(outcome.observation.contains("没有检索到"));
    }

    #[test]
    fn default_registry_has_three_tools() {
        let index = rag_08::bm25::Bm25Index::new();
        let registry = default_registry(Arc::new(index));

        assert_eq!(registry.len(), 3);
        assert!(
            registry
                .names()
                .contains(&GET_WEATHER_TOOL_NAME.to_string())
        );
        assert!(
            registry
                .names()
                .contains(&SEARCH_KNOWLEDGE_TOOL_NAME.to_string())
        );
        assert!(registry.names().contains(&CALCULATE_TOOL_NAME.to_string()));
    }
}
