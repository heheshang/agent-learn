//! 工具层（Tool Use）—— Agent 三大组件里的「工具使用」。
//!
//! ## 为什么工具是 Agent 的关键
//!
//! 博客里说得很直接：**工具使用是人类区别于其它物种的标志性能力**。
//! 模型通过调用外部 API 拿到权重里没有的信息 —— 实时信息、
//! 代码执行能力、私有数据源。
//!
//! MRKL 系统的实验结论更具体：把 LLM 当**路由器**，
//! 把请求分发给「专家模块」（可以是神经网络，也可以是符号工具），
//! 当外部符号工具足够可靠时，
//! **「知道何时以及如何使用工具」比工具本身更关键**。
//!
//! ## 本项目的两个工具
//!
//! | 工具 | 类型 | 对应 |
//! |---|---|---|
//! | [`search`] | **稀疏检索**（BM25 倒排索引） | LangChain 示例里的 `Wikipedia` |
//! | [`calculate`] | **符号工具**（手写算术求值器） | MRKL 论文里的 `Calculator` |
//!
//! 两个工具正好覆盖两类专家模块：
//! 「查资料」和「算数」。
//!
//! ## 为什么 `search` 用 BM25 而不是向量检索
//!
//! 因为这个项目的主角是 **ReAct 的动作空间**，不是检索。
//!
//! 检索质量不是重点 —— 重点是「模型能不能自己决定查什么、查几次」。
//! BM25 的好处是：
//!
//! 1. **零模型依赖** —— 不需要下载嵌入模型，`cargo run` 立刻能跑
//! 2. **完全确定性** —— 同一查询永远同一结果，测试可复现
//! 3. **关键词精确命中** —— 查「ReAct」这种专有名词时比向量更准
//!
//! 想要向量检索的话，仓库里 05~12 已经有完整实现 —— 换掉 `search`
//! 的实现即可，**Agent 循环一行都不用改**。
//!
//! ## 工具失败要能被模型看见
//!
//! 工具执行失败时，我们**不返回错误**，而是把错误信息当作一条
//! `Observation` 喂回模型 —— 模型读到「参数不合法」就能自己纠正。
//!
//! 这就是 [`AgentError::is_degradable`](crate::error::AgentError::is_degradable)
//! 只对工具错误返回 `true` 的原因。

use agent_logging::{debug, warn};
use std::sync::Arc;

use rag_08::bm25::{Bm25Document, Bm25Hit, Bm25Index};
use serde_json::{Value, json};

use crate::error::AgentResult;

/// `search` 工具的名字。
///
/// ⚠️ 这个名字会**原样出现在提示词里**，模型照着它调工具。
/// 提示词里写的就是这个常量，所以两边永远一致。
pub const SEARCH_TOOL_NAME: &str = "search";

/// `calculate` 工具的名字。
pub const CALCULATE_TOOL_NAME: &str = "calculate";

/// 工具的定义（名字 + 描述 + JSON Schema）。
///
/// 这三项是**工具的「契约」**：
///
/// - **名字** —— 模型用它来调用
/// - **描述** —— 模型靠它判断「什么时候该调这个工具」
/// - **Schema** —— 既告诉模型参数怎么填，也用于**本地校验**
///
/// 博客里那句「工具的描述写得不好时，模型会反复调同一个工具」
/// 就是描述质量问题的直接后果 —— 描述是 Agent 行为的一部分，
/// 不是文档。
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// 工具名。
    pub name: String,

    /// 工具描述。
    pub description: String,

    /// JSON Schema（参数定义）。
    pub schema: Value,
}

impl ToolSpec {
    /// 构造一个工具定义。
    pub fn new(name: impl Into<String>, description: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
        }
    }
}

/// `search` 工具的 Schema。
pub fn search_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词。用知识库里可能出现的词，而不是用户的口语原话。"
            },
            "top_k": {
                "type": "integer",
                "description": "返回的片段数量，默认 3。"
            }
        },
        "required": ["query"]
    })
}

/// `calculate` 工具的 Schema。
pub fn calculate_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "算术表达式，例如 (12 + 34) * 2。支持 + - * / 和括号。"
            }
        },
        "required": ["expression"]
    })
}

/// 工具执行的结果。
#[derive(Debug, Clone)]
pub struct ToolOutcome {
    /// 给模型看的文本（会变成 `Observation:`）。
    pub observation: String,

    /// 是否成功（失败时 `observation` 是错误描述）。
    pub ok: bool,

    /// 工具名（用于日志）。
    pub tool: String,
}

impl ToolOutcome {
    /// 成功的结果。
    pub fn ok(tool: impl Into<String>, observation: impl Into<String>) -> Self {
        Self {
            observation: observation.into(),
            ok: true,
            tool: tool.into(),
        }
    }

    /// 失败的结果（**不是错误** —— 错误文本会喂回模型让它自我纠正）。
    pub fn failed(tool: impl Into<String>, observation: impl Into<String>) -> Self {
        Self {
            observation: observation.into(),
            ok: false,
            tool: tool.into(),
        }
    }
}

/// 工具 trait。
///
/// 抽成 trait 的意义和仓库里其它层一样：
/// 上层 Agent 循环只依赖 `Box<dyn Tool>`，
/// 于是「本地 BM25 检索」和「远程 API」可以互换，
/// 测试时也能塞进一个**确定性的假工具**。
pub trait Tool: Send + Sync {
    /// 工具定义。
    fn spec(&self) -> ToolSpec;

    /// 执行工具。
    fn run(&self, args: &Value) -> ToolOutcome;
}

/// `search` 工具 —— 在本地知识库里做关键词检索。
pub struct SearchTool {
    index: Arc<Bm25Index>,
    default_top_k: usize,
    max_top_k: usize,
    max_observation_chars: usize,
}

impl std::fmt::Debug for SearchTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchTool")
            .field("documents", &self.index.len())
            .field("default_top_k", &self.default_top_k)
            .finish()
    }
}

impl SearchTool {
    /// 创建 `search` 工具。
    pub fn new(
        index: Arc<Bm25Index>,
        default_top_k: usize,
        max_top_k: usize,
        max_observation_chars: usize,
    ) -> Self {
        Self {
            index,
            default_top_k: default_top_k.max(1),
            max_top_k: max_top_k.max(1),
            max_observation_chars: max_observation_chars.max(100),
        }
    }

    /// 索引里的文档数。
    pub fn document_count(&self) -> usize {
        self.index.len()
    }

    /// 把命中块渲染成给模型看的观察文本。
    ///
    /// ## 格式为什么是 `[1] # 标题 > 子标题\n正文`
    ///
    /// 和 06~12 的 RAG 上下文格式保持一致 ——
    /// 这样同一个知识库在「固定流水线」和「Agent」两种模式下
    /// 看到的文本长得一样，对比实验才有意义。
    fn render(&self, hits: &[Bm25Hit]) -> String {
        if hits.is_empty() {
            return "没有检索到相关资料。".to_string();
        }

        let mut out = String::new();
        for (i, hit) in hits.iter().enumerate() {
            let head = if hit.heading_path.is_empty() {
                String::new()
            } else {
                format!("# {}\n", hit.heading_path.join(" > "))
            };
            out.push_str(&format!("[{}] {}{}\n\n", i + 1, head, hit.text));
        }
        out.trim_end().to_string()
    }
}

impl Tool for SearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            SEARCH_TOOL_NAME,
            "在本地知识库中做关键词检索，返回最相关的文档片段。\
             用于回答关于 RAG、Agent、ReAct、记忆与工具使用的事实性问题。",
            search_schema(),
        )
    }

    fn run(&self, args: &Value) -> ToolOutcome {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();

        if query.is_empty() {
            warn!("search 工具缺少 query 参数");
            return ToolOutcome::failed(
                SEARCH_TOOL_NAME,
                "错误：缺少 query 参数。请用一句话给出你想检索的关键词。",
            );
        }

        let top_k = args
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.default_top_k)
            .clamp(1, self.max_top_k);

        let hits = self.index.search(&query, top_k);

        // ⚠️ 观察文本要**截断** —— 它会原样进上下文，
        // 不截断的话两三次检索就把上下文吃光了。
        let observation = self.render(&hits);
        let observation = truncate(&observation, self.max_observation_chars);

        debug!(query = %query, top_k, hit_count = hits.len(), "search 工具检索完成");

        ToolOutcome::ok(SEARCH_TOOL_NAME, observation)
    }
}

/// `calculate` 工具 —— 手写算术求值器。
///
/// ## 为什么手写而不是用 `eval`
///
/// 1. **安全** —— `eval` 能执行任意代码
/// 2. **零依赖** —— 不需要引入表达式解析库
/// 3. **可控的错误** —— 除零、括号不匹配都有明确报错
///
/// MRKL 论文里的实验正好说明了这类符号工具的价值：
/// LLM（7B）**很难可靠地提取出算术运算的正确参数** ——
/// 口头表述的算术题比写明式的算术题难得多。
/// 所以「把算数交给工具」不是锦上添花，是必要的。
pub struct CalculateTool;

impl std::fmt::Debug for CalculateTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CalculateTool")
    }
}

impl Tool for CalculateTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            CALCULATE_TOOL_NAME,
            "计算一个算术表达式并返回结果。支持 + - * / 和括号。\
             涉及具体数值计算时必须调用这个工具，不要自己心算。",
            calculate_schema(),
        )
    }

    fn run(&self, args: &Value) -> ToolOutcome {
        let expression = args
            .get("expression")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();

        if expression.is_empty() {
            warn!("calculate 工具缺少 expression 参数");
            return ToolOutcome::failed(CALCULATE_TOOL_NAME, "错误：缺少 expression 参数。");
        }

        match calculate(&expression) {
            Ok(value) => {
                debug!(expression = %expression, value, "calculate 工具计算完成");
                ToolOutcome::ok(CALCULATE_TOOL_NAME, format!("{expression} = {value}"))
            }
            Err(msg) => {
                warn!(expression = %expression, error = %msg, "calculate 工具计算失败");
                ToolOutcome::failed(
                    CALCULATE_TOOL_NAME,
                    format!("错误：无法计算 `{expression}` —— {msg}"),
                )
            }
        }
    }
}

/// 算术求值器 —— 只支持 `+ - * /` 和括号。
///
/// 手写递归下降解析器，优先级：`()` > `* /` > `+ -`。
pub fn calculate(expression: &str) -> Result<f64, String> {
    let mut parser = Parser {
        chars: expression.chars().collect(),
        pos: 0,
    };
    let value = parser.parse_expr()?;
    parser.skip_whitespace();
    if parser.pos != parser.chars.len() {
        return Err(format!(
            "表达式尾部有多余字符: {}",
            expression.chars().skip(parser.pos).collect::<String>()
        ));
    }
    Ok(value)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn skip_whitespace(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_whitespace();
        self.chars.get(self.pos).copied()
    }

    /// expr := term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<f64, String> {
        let mut left = self.parse_term()?;
        loop {
            match self.peek() {
                Some('+') => {
                    self.pos += 1;
                    left += self.parse_term()?;
                }
                Some('-') => {
                    self.pos += 1;
                    left -= self.parse_term()?;
                }
                _ => return Ok(left),
            }
        }
    }

    /// term := factor (('*' | '/') factor)*
    fn parse_term(&mut self) -> Result<f64, String> {
        let mut left = self.parse_factor()?;
        loop {
            match self.peek() {
                Some('*') => {
                    self.pos += 1;
                    left *= self.parse_factor()?;
                }
                Some('/') => {
                    self.pos += 1;
                    let right = self.parse_factor()?;
                    if right == 0.0 {
                        return Err("除数为零".to_string());
                    }
                    left /= right;
                }
                _ => return Ok(left),
            }
        }
    }

    /// factor := number | '(' expr ')' | '-' factor
    fn parse_factor(&mut self) -> Result<f64, String> {
        match self.peek() {
            Some('(') => {
                self.pos += 1;
                let value = self.parse_expr()?;
                if self.peek() != Some(')') {
                    return Err("缺少右括号".to_string());
                }
                self.pos += 1;
                Ok(value)
            }
            Some('-') => {
                self.pos += 1;
                Ok(-self.parse_factor()?)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => {
                let start = self.pos;
                while self
                    .chars
                    .get(self.pos)
                    .is_some_and(|c| c.is_ascii_digit() || *c == '.')
                {
                    self.pos += 1;
                }
                let text: String = self.chars[start..self.pos].iter().collect();
                text.parse::<f64>()
                    .map_err(|e| format!("无法解析数字 {text}: {e}"))
            }
            Some(c) => Err(format!("不支持的字符: {c}")),
            None => Err("表达式意外结束".to_string()),
        }
    }
}

/// 截断文本到指定字符数，尾部加省略提示。
///
/// ## 为什么截断**头部**而不是尾部
///
/// 检索结果里最关键的信息通常在**开头**（定义、结论）。
/// 截尾会丢掉结论，截头会丢掉定义 —— 对「回答事实性问题」来说
/// 丢掉结论更致命。
pub fn truncate(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }

    let kept: String = text.chars().take(max_chars).collect();
    format!("{kept}\n…（内容过长，已截断，共 {total} 字符）")
}

/// 从知识库目录构建 BM25 索引。
///
/// 复用 08-rag 的 `build_bm25` —— 它用和向量索引**完全相同**的分块配置，
/// 但**不需要嵌入模型**（分块是纯 CPU 的字符串操作）。
pub fn build_index(data_dir: &std::path::Path) -> AgentResult<Bm25Index> {
    let splitter_config = rag_04::splitter::SplitterConfig::default();
    rag_08::index::build_bm25(data_dir, &splitter_config)
        .map_err(|e| crate::error::AgentError::Internal(format!("构建知识库索引失败: {e}")))
}

/// 用一批文档直接建索引（测试用）。
pub fn index_from_documents(documents: Vec<Bm25Document>) -> Bm25Index {
    Bm25Index::build(&documents)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, text: &str) -> Bm25Document {
        Bm25Document {
            id: id.to_string(),
            source: "kb.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: text.to_string(),
        }
    }

    fn search_tool() -> SearchTool {
        let index = index_from_documents(vec![
            doc("a#0", "ReAct 把推理和行动交织在一起"),
            doc("b#0", "BM25 是稀疏检索的打分函数"),
        ]);
        SearchTool::new(Arc::new(index), 2, 5, 1200)
    }

    #[test]
    fn search_tool_returns_matching_snippet() {
        let tool = search_tool();
        let outcome = tool.run(&json!({ "query": "ReAct" }));

        assert!(outcome.ok);
        assert!(outcome.observation.contains("ReAct"));
    }

    #[test]
    fn search_tool_reports_missing_query_without_erroring() {
        // 关键：工具失败**不是错误**，而是可观察的失败 ——
        // 模型读到「缺少 query 参数」能自我纠正。
        let tool = search_tool();
        let outcome = tool.run(&json!({}));

        assert!(!outcome.ok);
        assert!(outcome.observation.contains("缺少 query"));
    }

    #[test]
    fn search_tool_respects_top_k() {
        let tool = search_tool();
        let outcome = tool.run(&json!({ "query": "检索 推理", "top_k": 1 }));

        // 只会返回 1 个片段 → 只出现一个 `[1]` 标记。
        assert_eq!(outcome.observation.matches("[").count(), 1);
    }

    #[test]
    fn search_tool_truncates_long_observations() {
        let index = index_from_documents(vec![doc("a#0", &"长文本".repeat(500))]);
        let tool = SearchTool::new(Arc::new(index), 1, 5, 100);

        let outcome = tool.run(&json!({ "query": "长文本" }));
        assert!(outcome.observation.contains("已截断"));
        assert!(outcome.observation.chars().count() < 300);
    }

    #[test]
    fn calculate_tool_evaluates_expressions() {
        let tool = CalculateTool;
        let outcome = tool.run(&json!({ "expression": "(12 + 34) * 2" }));

        assert!(outcome.ok);
        assert!(outcome.observation.contains("92"));
    }

    #[test]
    fn calculate_tool_reports_errors_as_observations() {
        let tool = CalculateTool;
        let outcome = tool.run(&json!({ "expression": "1 / 0" }));

        assert!(!outcome.ok);
        assert!(outcome.observation.contains("除数为零"));
    }

    #[test]
    fn tool_specs_have_required_fields() {
        assert_eq!(search_schema()["required"], json!(["query"]));
        assert_eq!(calculate_schema()["required"], json!(["expression"]));

        let search_spec = search_tool().spec();
        assert_eq!(search_spec.name, SEARCH_TOOL_NAME);
        assert!(!search_spec.description.is_empty());
    }

    #[test]
    fn evaluates_basic_expressions() {
        assert_eq!(calculate("1 + 2").unwrap(), 3.0);
        assert_eq!(calculate("(12 + 34) * 2").unwrap(), 92.0);
        assert_eq!(calculate("10 / 4").unwrap(), 2.5);
        assert_eq!(calculate("-3 + 1").unwrap(), -2.0);
    }

    #[test]
    fn rejects_invalid_expressions() {
        assert!(calculate("1 +").is_err());
        assert!(calculate("(1 + 2").is_err());
        assert!(calculate("1 / 0").is_err());
        assert!(calculate("a + b").is_err());
    }

    #[test]
    fn truncate_keeps_head() {
        let text = "abcdefghij";
        let out = truncate(text, 4);
        assert!(out.starts_with("abcd"));
        assert!(out.contains("已截断"));
    }

    #[test]
    fn truncate_is_noop_when_short_enough() {
        assert_eq!(truncate("abc", 10), "abc");
    }
}
