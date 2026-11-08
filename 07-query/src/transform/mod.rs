//! 查询改写策略（Query Transformations）—— 本项目的核心。
//!
//! ## 改写要解决什么
//!
//! 向量检索有个结构性弱点：**它比的是「语义相似」，不是「关键词匹配」**。
//! 所以「用户的问法」和「文档的写法」只要措辞不同，就可能落在向量空间的
//! 不同位置 —— 明明语义相关，检索却召回不到。
//!
//! 改写就是在中间加一层翻译：**用户问法 → 检索式**。
//!
//! ## 四种策略的取舍
//!
//! | 策略 | 改写什么 | LLM 调用数 | 检索次数 | 适合什么 |
//! |---|---|---|---|---|
//! | [`Strategy::HyDE`] | 假想答案文档 | 1 | 1 | 原问题太短/太口语，文档是陈述句 |
//! | [`Strategy::MultiQuery`] | N 个等价问法 | 1 | N | 单个问法召不全，需多角度覆盖 |
//! | [`Strategy::Rewrite`] | 1 个更紧凑的检索式 | 1 | 1 | 问题是多轮对话的指代（「它」「那个」） |
//! | [`Strategy::Decompose`] | N 个独立子问题 | 1 | N | 复合问题（「A 和 B 的区别」） |
//!
//! ## 共同的失败模式
//!
//! 三种策略（除 Rewrite 外）都依赖 LLM 输出**多个**结果，
//! 而 LLM 的输出格式是**不受控**的：
//!
//! ```text
//! 1. 问法一
//! 2. 问法二
//! 3. 问法三
//! ```
//!
//! 也可能是：
//!
//! ```text
//! 问法一
//! 问法二
//! ```
//!
//! 还可能带编号、破折号、markdown 列表、解释文字……
//!
//! 所以 [`parse_numbered_lines`] 是这个模块**最关键的一段代码** ——
//! 它必须容忍 LLM 的所有「小毛病」，只提取出干净的问法列表。
//! 这段逻辑是完全确定的，所以能被单元测试充分覆盖。

pub mod decompose;
pub mod hyde;
pub mod multi_query;
pub mod prompts;
pub mod rewrite;

use serde::{Deserialize, Serialize};

use crate::error::{QueryError, QueryResult};

pub use decompose::DecomposeTransformer;
pub use hyde::HydeTransformer;
pub use multi_query::MultiQueryTransformer;
pub use prompts::{DECOMPOSE_PROMPT, HYDE_PROMPT, MULTI_QUERY_PROMPT, REWRITE_PROMPT};
pub use rewrite::RewriteTransformer;

/// 改写策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// 不改写（原问题直接检索）—— **对照组**。
    ///
    /// 它的存在不是为了「用」，而是为了**基准**。
    /// 没有 `None` 这一档，任何改写策略的收益都无法量化。
    None,

    /// HyDE（Hypothetical Document Embeddings）。
    Hyde,

    /// Multi-Query（多问法 + 融合）。
    MultiQuery,

    /// Rewrite（改写成紧凑检索式）。
    Rewrite,

    /// Decompose（复合问题拆成子问题）。
    Decompose,
}

impl Strategy {
    /// 策略名（用于日志 / API 响应）。
    pub fn name(self) -> &'static str {
        match self {
            Strategy::None => "none",
            Strategy::Hyde => "hyde",
            Strategy::MultiQuery => "multi_query",
            Strategy::Rewrite => "rewrite",
            Strategy::Decompose => "decompose",
        }
    }

    /// 是否会产生**多个**检索查询。
    pub fn is_multi(self) -> bool {
        matches!(self, Strategy::MultiQuery | Strategy::Decompose)
    }

    /// 从字符串解析（不区分大小写，支持连字符）。
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "none" | "naive" | "baseline" => Some(Strategy::None),
            "hyde" => Some(Strategy::Hyde),
            "multi_query" | "multiquery" => Some(Strategy::MultiQuery),
            "rewrite" | "query_rewrite" => Some(Strategy::Rewrite),
            "decompose" | "decomposition" => Some(Strategy::Decompose),
            _ => None,
        }
    }

    /// 所有策略（`None` 在最前，因为它是对照组）。
    pub fn all() -> Vec<Self> {
        vec![
            Strategy::None,
            Strategy::Hyde,
            Strategy::MultiQuery,
            Strategy::Rewrite,
            Strategy::Decompose,
        ]
    }
}

/// 一条改写后的查询。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformedQuery {
    /// 改写后的文本。
    pub text: String,

    /// 这条查询来自哪个策略（同一个流水线可能组合多个策略）。
    pub strategy: Strategy,

    /// 它是原问题的第几个变体（`None` 策略和单条改写都是 `Some(0)`）。
    pub variant: Option<usize>,

    /// 改写备注（比如 HyDE 的「假想文档」标记）。
    pub note: Option<String>,
}

impl TransformedQuery {
    /// 构造一条改写查询。
    pub fn new(text: impl Into<String>, strategy: Strategy) -> Self {
        Self {
            text: text.into(),
            strategy,
            variant: None,
            note: None,
        }
    }

    /// 一行摘要。
    pub fn summary(&self) -> String {
        let variant = match self.variant {
            Some(v) => format!(" #{}", v + 1),
            None => String::new(),
        };
        format!("[{}{}] {}", self.strategy.name(), variant, self.text)
    }
}

/// 查询改写器的异步调用 future。
///
/// 和 [`crate::llm::LlmFuture`] 同一个理由 —— trait 要 dyn 兼容，
/// 就不能用 `async fn` 或 RPITIT，必须显式装箱。
pub type TransformFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = QueryResult<Vec<TransformedQuery>>> + Send + 'a>,
>;

/// 查询改写器 trait。
///
/// 和 [`crate::llm::Llm`] 一样用装箱 future —— 这样
/// `QueryTransformer` 可以放进 `Box<dyn QueryTransformer>`，
/// 在运行时按配置切换策略。
pub trait QueryTransformer: Send + Sync {
    /// 策略标识。
    fn strategy(&self) -> Strategy;

    /// 改写：原问题 → N 条检索查询。
    ///
    /// 返回**至少一条**查询 —— 即使降级，也要返回原问题，
    /// 让调用方永远能继续往下走。
    fn transform<'a>(&'a self, question: &'a str) -> TransformFuture<'a>;
}

/// 把 LLM 输出解析成干净的列表。
///
/// ## 要处理的格式问题
///
/// LLM 可能返回：
///
/// ```text
/// 1. 问法一
/// 2. 问法二
/// ```
///
/// ```text
/// - 问法一
/// - 问法二
/// ```
///
/// ```text
/// 问法一
/// 问法二
/// ```
///
/// ```text
/// 1) 问法一
/// 2) 问法二
/// ```
///
/// 甚至可能带前缀解释（"Here are the queries:"）或 markdown 标题。
///
/// ## 解析规则
///
/// 1. 按行切分
/// 2. 去掉行首的编号 / 项目符号 / 引用标记
/// 3. 丢掉空行和「纯标点」行
/// 4. 丢掉明显的元话语行（`Here are...` / `以下是...` / `Sure, ...`）
/// 5. 丢掉过长的行（模型跑题了）
///
/// 第 4、5 条是**启发式** —— 因为 LLM 就是会跑题。
/// 它们的代价是：偶尔会误删一条正常查询。
/// 这个代价是**可接受**的 —— 少一条查询只是少一点召回，
/// 而漏掉一条「Here's the answer: ...」会污染整个结果集。
pub fn parse_numbered_lines(text: &str, max_items: usize) -> Vec<String> {
    parse_lines(text, max_items, MAX_QUERY_LINE_LEN)
}

/// 列表项的最大长度。
///
/// 超过这个长度的行大概率是模型跑题写了段解释，不是查询本身。
const MAX_QUERY_LINE_LEN: usize = 200;

/// 单条改写结果的最大长度。
///
/// 比列表项宽松 —— HyDE 的输出是一整段「假想文档」，
/// 提示词明确要求 150~250 字。
const MAX_DOCUMENT_LINE_LEN: usize = 600;

/// 通用行解析。
///
/// - `max_items`：最多返回几条
/// - `max_len`：单行最大长度（超长行被丢弃）
///
/// 为什么要把长度上限做成参数：
///
/// 「列表项」和「单条文档」的长度期望完全不同：
///
/// - Multi-Query 的每一项是一个**问句** —— 一句话，几十个字。超长 = 跑题
/// - HyDE 的输出是**一整段文档** —— 提示词明确要求 150~250 字
///
/// 如果对 HyDE 用同一个 200 字上限，它的输出会**整段被丢掉**，
/// 导致改写永远解析不出内容 → 永远降级成原问题 → **HyDE 静默失效**。
///
/// 这个 bug 很隐蔽：代码不报错，测试也过（测试用的是短文本），
/// 但线上 HyDE 一直在原地踏步。
fn parse_lines(text: &str, max_items: usize, max_len: usize) -> Vec<String> {
    let mut items = Vec::new();

    for raw_line in text.lines() {
        if items.len() >= max_items {
            break;
        }

        let line = raw_line.trim();

        if line.is_empty() {
            continue;
        }

        // 去掉行首的编号、项目符号、引用标记。
        let cleaned = strip_list_marker(line);

        if cleaned.is_empty() {
            continue;
        }

        // 丢掉元话语行（"Here are the queries:" 之类）。
        if is_meta_line(&cleaned) {
            continue;
        }

        // 丢掉过长的行 —— 模型跑题写了一段解释。
        if cleaned.chars().count() > max_len {
            continue;
        }

        items.push(cleaned);
    }

    items
}

/// 去掉行首的列表标记。
///
/// 支持 `1.` `1)` `-` `*` `•` `>` `#` 以及它们的组合（markdown 的 `1. ` 已经覆盖 `* ` / `- `）。
///
/// ## 为什么 `-` 和 `*` 要求后面跟空白
///
/// markdown 里 `**加粗**` 是**行首的两个星号**，不是列表标记。
/// 如果无条件剥 `*`，`**问法一**` 会变成 `问法一**` —— 反而破坏了原文。
///
/// markdown 的列表语法规定 `*` / `-` 后面必须跟空格（否则会被当成强调语法），
/// 所以「后面跟空白」正好能区分这两种情况。
fn strip_list_marker(line: &str) -> String {
    let mut s = line.trim();

    // 反复剥离：处理 `1. - 问法` 这种「编号 + 项目符号」的组合。
    loop {
        let before = s.len();

        // 数字编号：`1.` `1)` `12.` `12)`
        let digits_end = s
            .char_indices()
            .find(|(_, c)| !c.is_ascii_digit())
            .map(|(i, _)| i)
            .unwrap_or(s.len());

        if digits_end > 0 && digits_end < s.len() {
            let rest = &s[digits_end..];
            if rest.starts_with('.') || rest.starts_with(')') || rest.starts_with('、') {
                s = rest[1..].trim();
            }
        }

        // 项目符号。
        //
        // `-` `*` `+` 要求后面跟空白（区分 markdown 强调语法）；
        // 其它符号（`•` `>` `#` `|` `·`）无条件剥离。
        for bullet in ['-', '*', '+'] {
            if let Some(rest) = s.strip_prefix(bullet)
                && rest.starts_with(char::is_whitespace)
            {
                s = rest.trim();
                break;
            }
        }

        for bullet in ['•', '>', '#', '|', '·'] {
            if let Some(stripped) = s.strip_prefix(bullet) {
                s = stripped.trim();
                break;
            }
        }

        if s.len() == before {
            break;
        }
    }

    s.trim().to_string()
}

/// 判断一行是不是「元话语」（模型的客套话 / 解释，不是查询本身）。
///
/// 这类行必须丢掉 —— 它们不是查询，但长度上完全像查询，
/// 混进结果会让检索查到一个完全无关的东西。
fn is_meta_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();

    const META_PREFIXES: [&str; 12] = [
        "here are",
        "here is",
        "here's",
        "the following",
        "below are",
        "following queries",
        "query 1:",
        "queries:",
        "note:",
        "explanation:",
        "answer:",
        "sure,",
    ];

    const META_CHINESE: [&str; 8] = [
        "以下是",
        "下面是",
        "以下是几个",
        "如下",
        "解释",
        "说明",
        "注意",
        "答案",
    ];

    if META_PREFIXES.iter().any(|p| lower.starts_with(p)) {
        return true;
    }

    if META_CHINESE.iter().any(|p| line.starts_with(p)) {
        return true;
    }

    false
}

/// 把 LLM 输出解析成**单条**改写结果。
///
/// 用于 Rewrite / HyDE —— 这两种策略只需要一条结果。
///
/// 做法：取第一条「像查询」的行。如果 LLM 返回了带编号的多行，
/// 只取第一条（HyDE 场景下 LLM 通常返回一整段，所以这里还有
/// 「多行合并」的兜底逻辑 —— 见 [`join_paragraph`]）。
///
/// ⚠️ 长度上限用 [`MAX_DOCUMENT_LINE_LEN`] 而不是列表项的上限 ——
/// HyDE 的输出是一整段（150~250 字），用列表项的 200 字上限会把它整段丢掉，
/// 导致 HyDE 永远降级成原问题（静默失效）。
pub fn parse_single(text: &str) -> Option<String> {
    let lines = parse_lines(text, 16, MAX_DOCUMENT_LINE_LEN);

    if lines.is_empty() {
        return None;
    }

    // 如果第一行看起来是完整的（长度够长、没有明显的列表残缺），直接用。
    // 否则把多行拼起来 —— HyDE 的输出常常是一整段。
    Some(strip_wrapping_quotes(&join_paragraph(&lines)))
}

/// 去掉包裹整段结果的首尾引号。
///
/// LLM 经常把结果写成 `"改写后的检索式"` 或 `「改写后的检索式」`。
/// 引号对向量检索是纯噪声 —— 去掉能减少一点无意义的 token，
/// 也避免引号被当成内容的一部分。
///
/// 只处理「首尾成对」的情况，不会动句子中间的引号。
fn strip_wrapping_quotes(text: &str) -> String {
    let t = text.trim();
    let pairs = [
        ('"', '"'),
        ('\'', '\''),
        ('「', '」'),
        ('『', '』'),
        ('“', '”'),
        ('‘', '’'),
    ];

    for (open, close) in pairs {
        if t.len() >= 2 && t.starts_with(open) && t.ends_with(close) {
            let inner = t[open.len_utf8()..t.len() - close.len_utf8()].trim();
            // 引号里不能是空的。
            if !inner.is_empty() {
                return inner.to_string();
            }
        }
    }

    t.to_string()
}

/// 把多行拼成一段（用空格连接，保留段落感）。
fn join_paragraph(lines: &[String]) -> String {
    lines
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

/// 构造一个「不改写」的转换器（对照组）。
///
/// 它不调 LLM，直接返回原问题。
pub struct NoopTransformer;

impl QueryTransformer for NoopTransformer {
    fn strategy(&self) -> Strategy {
        Strategy::None
    }

    fn transform<'a>(&'a self, question: &'a str) -> TransformFuture<'a> {
        Box::pin(async move { Ok(vec![TransformedQuery::new(question.trim(), Strategy::None)]) })
    }
}

/// 统一的「带降级的改写」包装。
///
/// ## 为什么需要它
///
/// LLM 调用会失败（网络、限流、鉴权）。而**改写失败不应该让整个请求失败** ——
/// 退回原问题继续检索，召回差一点，但答案仍然能给出。
///
/// 所以这里的约定是：
///
/// - LLM 调用失败 → 记一条 warning，**返回原问题**
/// - LLM 调用成功但输出解析不出东西 → 同样降级
/// - 只有「问题本身是空的」才算 BadRequest
///
/// 这条降级路径让「改写」成为一个**纯粹的增强**，而不是新的故障点。
pub struct FallibleTransformer {
    inner: Box<dyn QueryTransformer>,
    /// 降级时是否记录日志（用 `tracing` 风格的回调，避免引入日志依赖）。
    on_degrade: Option<Box<dyn Fn(String) + Send + Sync>>,
}

impl FallibleTransformer {
    /// 包装一个转换器（不记录降级日志）。
    pub fn new(inner: Box<dyn QueryTransformer>) -> Self {
        Self {
            inner,
            on_degrade: None,
        }
    }

    /// 包装一个转换器，降级时调用回调。
    pub fn with_degrade_handler(
        inner: Box<dyn QueryTransformer>,
        on_degrade: Box<dyn Fn(String) + Send + Sync>,
    ) -> Self {
        Self {
            inner,
            on_degrade: Some(on_degrade),
        }
    }

    /// 降级回调。
    pub fn degrade_handler(&self) -> Option<&(dyn Fn(String) + Send + Sync)> {
        self.on_degrade.as_deref()
    }
}

impl QueryTransformer for FallibleTransformer {
    fn strategy(&self) -> Strategy {
        self.inner.strategy()
    }

    fn transform<'a>(&'a self, question: &'a str) -> TransformFuture<'a> {
        Box::pin(async move {
            let question = question.trim();
            if question.is_empty() {
                return Err(QueryError::BadRequest("问题不能为空".to_string()));
            }

            match self.inner.transform(question).await {
                Ok(queries) if !queries.is_empty() => Ok(queries),
                Ok(_) => {
                    if let Some(handler) = &self.on_degrade {
                        handler("改写返回空结果，降级为原问题".to_string());
                    }
                    Ok(vec![TransformedQuery::new(question, self.inner.strategy())])
                }
                Err(e) => {
                    if let Some(handler) = &self.on_degrade {
                        handler(format!("改写失败（{e}），降级为原问题"));
                    }
                    Ok(vec![TransformedQuery::new(question, self.inner.strategy())])
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numbered_list() {
        let out = parse_numbered_lines("1. 问法一\n2. 问法二\n3. 问法三", 10);
        assert_eq!(out, vec!["问法一", "问法二", "问法三"]);
    }

    #[test]
    fn parses_bullet_list() {
        let out = parse_numbered_lines("- 问法一\n* 问法二\n• 问法三", 10);
        assert_eq!(out, vec!["问法一", "问法二", "问法三"]);
    }

    #[test]
    fn parses_bare_lines() {
        let out = parse_numbered_lines("问法一\n问法二", 10);
        assert_eq!(out, vec!["问法一", "问法二"]);
    }

    #[test]
    fn parses_parenthesis_numbering() {
        let out = parse_numbered_lines("1) 问法一\n2) 问法二", 10);
        assert_eq!(out, vec!["问法一", "问法二"]);
    }

    #[test]
    fn strips_combined_markers() {
        let out = parse_numbered_lines("1. - 问法一\n2. * 问法二", 10);
        assert_eq!(out, vec!["问法一", "问法二"]);
    }

    #[test]
    fn skips_meta_lines() {
        let text = "Here are the queries:\n1. 问法一\n2. 问法二";
        let out = parse_numbered_lines(text, 10);
        assert_eq!(out, vec!["问法一", "问法二"]);
    }

    #[test]
    fn skips_chinese_meta_lines() {
        let text = "以下是几个不同角度的问法：\n1. 问法一\n2. 问法二";
        let out = parse_numbered_lines(text, 10);
        assert_eq!(out, vec!["问法一", "问法二"]);
    }

    #[test]
    fn drops_overlong_lines() {
        let long = "啊".repeat(300);
        let text = format!("1. 问法一\n2. {long}");
        let out = parse_numbered_lines(&text, 10);
        assert_eq!(out, vec!["问法一"]);
    }

    #[test]
    fn respects_max_items() {
        let text = "1. 一\n2. 二\n3. 三\n4. 四";
        assert_eq!(parse_numbered_lines(text, 2).len(), 2);
    }

    #[test]
    fn keeps_question_marks() {
        // 关键：不能把问号当噪声去掉 —— 问法本身就是问句。
        let out = parse_numbered_lines("1. 什么是 RAG？", 10);
        assert_eq!(out, vec!["什么是 RAG？"]);
    }

    #[test]
    fn does_not_strip_middle_asterisk() {
        // `**问法**` 是 markdown 加粗，不是列表标记。
        let out = parse_numbered_lines("**问法一**", 10);
        assert_eq!(out, vec!["**问法一**"]);
    }

    #[test]
    fn parse_single_joins_multiple_lines() {
        let out = parse_single("第一行内容\n第二行内容").unwrap();
        assert_eq!(out, "第一行内容 第二行内容");
    }

    /// 回归测试：HyDE 的输出是一整段（150~250 字），
    /// 不能因为「列表项 200 字上限」被整段丢掉。
    ///
    /// 这个 bug 很隐蔽：代码不报错、测试全绿，但线上 HyDE 一直在原地踏步
    /// —— 每次都降级成原问题，HyDE 等于没启用。
    #[test]
    fn parse_single_accepts_a_long_paragraph() {
        let paragraph = "模型的知识截止日期问题可以通过检索增强生成来解决。".repeat(12);
        assert!(paragraph.chars().count() > 200, "测试数据应该超过 200 字");

        let out = parse_single(&paragraph).expect("长段落应该能被解析出来");
        assert!(out.contains("检索增强生成"));
    }

    /// 列表项仍然要守住 200 字上限（模型跑题的长行必须丢掉）。
    #[test]
    fn numbered_lines_still_drop_overlong_items() {
        let long = "啊".repeat(300);
        let text = format!("1. 问法一\n2. {long}");
        let out = parse_numbered_lines(&text, 10);
        assert_eq!(out, vec!["问法一"]);
    }

    #[test]
    fn parse_single_returns_none_on_empty() {
        assert!(parse_single("   \n\n  ").is_none());
    }

    #[test]
    fn strategy_parsing_is_lenient() {
        assert_eq!(Strategy::parse("hyde"), Some(Strategy::Hyde));
        assert_eq!(Strategy::parse("MULTI-QUERY"), Some(Strategy::MultiQuery));
        assert_eq!(Strategy::parse("multi_query"), Some(Strategy::MultiQuery));
        assert_eq!(Strategy::parse("naive"), Some(Strategy::None));
        assert_eq!(Strategy::parse("unknown"), None);
    }

    #[tokio::test]
    async fn noop_transformer_returns_original() {
        let t = NoopTransformer;
        let out = t.transform("  原问题  ").await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "原问题");
        assert_eq!(out[0].strategy, Strategy::None);
    }

    #[tokio::test]
    async fn fallible_degrades_to_original_on_error() {
        struct Failing;
        impl QueryTransformer for Failing {
            fn strategy(&self) -> Strategy {
                Strategy::Hyde
            }
            fn transform<'a>(&'a self, _q: &'a str) -> TransformFuture<'a> {
                Box::pin(async move { Err(QueryError::Transformation("模型挂了".to_string())) })
            }
        }

        let notes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = notes.clone();

        let t = FallibleTransformer::with_degrade_handler(
            Box::new(Failing),
            Box::new(move |msg: String| sink.lock().unwrap().push(msg)),
        );

        let out = t.transform("原问题").await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "原问题");
        assert_eq!(notes.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn fallible_rejects_empty_question() {
        let t = FallibleTransformer::new(Box::new(NoopTransformer));
        let err = t.transform("   ").await.unwrap_err();
        assert_eq!(err.status_code(), 400);
    }
}
