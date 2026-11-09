//! 工具层 —— 把「能力」包装成「模型可调用的工具」。
//!
//! # 五个工具
//!
//! | 工具 | 能力 | 来源 |
//! |---|---|---|
//! | [`KbSearchTool`] | 本地知识库检索（RAG） | 08-rag 的 BM25 |
//! | [`WebSearchTool`] | Web 搜索 | [`crate::search`] |
//! | [`WebFetchTool`] | 抓取网页正文 | [`crate::search`] |
//! | [`MemoryTool`] | 长期记忆读写 | 17-agent-memory |
//! | [`CalculateTool`] | 手写算术 | 13-agent-core |
//!
//! # 为什么工具要**声明失败类别**
//!
//! 直接复用 18 的 [`ErrorAwareTool`] ——
//! 因为中间件要按错误类别决定「重试 / 降级 / 终止」。
//!
//! 举几个本项目里的例子：
//!
//! | 失败 | 类别 | 理由 |
//! |---|---|---|
//! | 搜索服务 5xx / 连接超时 | **Transient** | 重试可能成功 |
//! | 查询词为空 | **Business** | 改参数才行，重试没用 |
//! | 链接 404 | **Business** | 换个链接 |
//! | 本地索引没建 | **Fatal** | 环境没配好，重试和降级都是徒劳 |

use std::sync::Arc;

use agent_18::tool::{ErrorAwareTool, ToolContext, ToolFuture, ToolOutcome};
use agent_logging::{debug, warn};
use serde_json::{Value, json};

use crate::error::{ErrorCode, ToolError};
use crate::search::{SearchResult, WebBackend, decode_entities};

/// 复用 16 的 `ToolSpec` —— 工具定义的契约（名字 / 描述 / Schema / strict）。
///
/// ## 为什么直接复用
///
/// `ToolSpec` 是「工具对模型的承诺」——
/// Schema 写什么，模型就会以为有什么。
/// 16 实测踩过这个坑（Schema 写 `city` 而实现读 `location`，
/// 模型试了 3 种组合才放弃，烧掉 8 次工具调用）。
///
/// 所以**只允许一份定义**，避免两个版本的 Schema 不一致。
pub use agent_16::tools::ToolSpec;

/// 构造一个「带结构化数据」的工具结果。
///
/// ## ⚠️ 为什么要把 `data` **编码进 observation**
///
/// 18 的 `MiddlewareOutcome` 只保留 `observation` 文本，
/// `ToolOutcome.data` 会被丢掉。
/// 而本项目的**证据就是从 `data` 里提取的**。
///
/// 所以这里把 `data` 序列化成 JSON，追加成 observation 的**最后一行**：
///
/// ```text
/// 一些给人看的文本
/// @@DATA:{"hits":[...]}
/// ```
///
/// 循环会在两处处理它：
///
/// - 收集证据时解回 `Value`
/// - 发给模型前**剥掉**（那坨 JSON 对模型没用）
///
/// ## 为什么追加在**最后**
///
/// 因为有些工具会**截断** observation（防上下文爆炸）——
/// 截断会把尾部切掉，数据行必须在校验之后才追加。
pub fn outcome_with_data(tool: &str, visible: String, data: &Value) -> ToolOutcome {
    let data_line = format!(
        "{}{}",
        crate::agent::DATA_PREFIX,
        serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string())
    );

    ToolOutcome::ok_with_data(tool, format!("{visible}\n{data_line}"), data.clone())
}

// ─────────────────────────────────────────────────────────────────────
// ① RAG：本地知识库检索
// ─────────────────────────────────────────────────────────────────────

/// `kb_search` 工具名。
pub const KB_SEARCH_TOOL_NAME: &str = "kb_search";

/// 本地知识库检索工具（RAG 的「召回」侧）。
///
/// ## 与 08/12 的关系
///
/// 08 实现了「dense + BM25 混合检索 + 重排」，
/// 12 实现了「服务端混合检索 + 交叉编码器重排 + 上下文压缩」。
///
/// 但那些都需要**向量库**（Milvus）或**本地模型**（fastembed 交叉编码器）——
/// 在一个「研究助手」里，检索的定位不一样：
///
/// > **研究助手的第一检索源永远是本地知识库**，
/// > 因为它零延迟、零成本、结果稳定可复现。
///
/// 所以本项目只做 BM25（复用 08 的 [`rag_08::bm25::Bm25Index`]），
/// 不引向量库 —— 让整个项目**零外部服务依赖**。
#[derive(Debug)]
pub struct KbSearchTool {
    index: Arc<rag_08::bm25::Bm25Index>,
    default_top_k: usize,
    max_top_k: usize,
    max_observation_chars: usize,
}

impl KbSearchTool {
    /// 创建工具。
    pub fn new(
        index: Arc<rag_08::bm25::Bm25Index>,
        default_top_k: usize,
        max_top_k: usize,
        max_observation_chars: usize,
    ) -> Self {
        Self {
            index,
            default_top_k: default_top_k.max(1),
            max_top_k: max_top_k.max(1),
            max_observation_chars: max_observation_chars.max(200),
        }
    }

    /// 索引里的文档数。
    pub fn document_count(&self) -> usize {
        self.index.len()
    }

    /// 把命中块渲染成给模型看的观察文本。
    ///
    /// ## 格式：`[1] 文件名 > 标题路径\n正文`
    ///
    /// 和 06~12 的 RAG 上下文格式保持一致 ——
    /// 同一个知识库在「固定流水线」和「Agent」两种模式下
    /// 看到的文本长得一样，对比实验才有意义。
    fn render(&self, hits: &[rag_08::bm25::Bm25Hit]) -> String {
        if hits.is_empty() {
            return "本地知识库里没有检索到相关资料。".to_string();
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

/// `kb_search` 的参数 Schema。
pub fn kb_search_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词或问题。用自然语言描述你要找的内容。"
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 4,
                "description": "返回的片段数量。"
            }
        },
        "required": ["query"]
    })
}

impl ErrorAwareTool for KbSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            KB_SEARCH_TOOL_NAME,
            "在**本地知识库**中做关键词检索，返回最相关的文档片段。\
             这是最可靠、最快的信息源 —— 优先检索它，再考虑联网搜索。\
             适合查项目自身的文档、历史决策、术语定义。",
            kb_search_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();

            if query.is_empty() {
                // ⚠️ 参数错 = Business —— 改参数才行，重试没用。
                return Err(
                    ToolError::invalid_args(KB_SEARCH_TOOL_NAME, "缺少 query 参数")
                        .with_hint("用一句话描述你想检索的内容，例如「错误处理的降级策略」。"),
                );
            }

            let top_k = args
                .get("top_k")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(self.default_top_k)
                .clamp(1, self.max_top_k);

            let hits = self.index.search(&query, top_k);

            if hits.is_empty() {
                return Err(ToolError::new(
                    ErrorCode::NotFound,
                    KB_SEARCH_TOOL_NAME,
                    format!("知识库里没有匹配「{query}」的内容"),
                )
                .with_hint("换个关键词，或改用 web_search 查外部资料。"));
            }

            let observation =
                crate::search::truncate_chars(&self.render(&hits), self.max_observation_chars);

            debug!(query = %query, top_k, hits = hits.len(), "知识库检索完成");

            Ok(outcome_with_data(
                KB_SEARCH_TOOL_NAME,
                observation,
                &json!({
                    "query": query,
                    "hits": hits.iter().map(|h| json!({
                        "source": h.source,
                        "heading_path": h.heading_path,
                        "score": h.score,
                        // ⚠️ 块正文必须带上 ——
                        //    证据层靠它区分「来源位置」和「证据内容」。
                        //    少了它，每条证据的正文都等于标题路径。
                        "text": h.text,
                    })).collect::<Vec<_>>(),
                }),
            ))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ② Web 搜索
// ─────────────────────────────────────────────────────────────────────

/// `web_search` 工具名。
pub const WEB_SEARCH_TOOL_NAME: &str = "web_search";

/// Web 搜索工具。
///
/// ## ⚠️ 与 `kb_search` 的分工
///
/// - `kb_search` —— **内部**知识，稳定、免费、快
/// - `web_search` —— **外部**信息，时效性强、但可能失败
///
/// 提示词里明确写了这个优先级 —— 因为「先查本地再查网」
/// 既省钱又降低幻觉。
#[derive(Debug)]
pub struct WebSearchTool {
    backend: Arc<dyn WebBackend>,
    default_top_k: usize,
    max_top_k: usize,
    max_observation_chars: usize,
}

impl WebSearchTool {
    /// 创建工具。
    pub fn new(
        backend: Arc<dyn WebBackend>,
        default_top_k: usize,
        max_top_k: usize,
        max_observation_chars: usize,
    ) -> Self {
        Self {
            backend,
            default_top_k: default_top_k.max(1),
            max_top_k: max_top_k.max(1),
            max_observation_chars: max_observation_chars.max(200),
        }
    }

    /// 后端名。
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}

/// `web_search` 的参数 Schema。
pub fn web_search_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词。用具体的名词短语，不要用整句问句。"
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
                "description": "返回的结果条数。"
            }
        },
        "required": ["query"]
    })
}

impl ErrorAwareTool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            WEB_SEARCH_TOOL_NAME,
            "在**互联网**上搜索，返回标题、链接和摘要。\
             用于获取最新信息、项目文档、第三方库用法等**本地知识库没有**的内容。\
             搜索结果只是线索 —— 重要结论请再用 `web_fetch` 打开链接看正文。",
            web_search_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();

            if query.is_empty() {
                return Err(
                    ToolError::invalid_args(WEB_SEARCH_TOOL_NAME, "缺少 query 参数")
                        .with_hint("传入要搜索的关键词，例如「tokio 任务取消」。"),
                );
            }

            let top_k = args
                .get("top_k")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(self.default_top_k)
                .clamp(1, self.max_top_k);

            let results = self.backend.search(&query, top_k).await?;

            // ⚠️ 摘要里的 HTML 实体**必须**解码 ——
            //    实测 Bing 会返回 `&ensp;` `&#0183;`，
            //    不解码的话模型看到的是一堆「乱码」。
            //
            //    ⚠️ 只解码文本，**不碰 URL** ——
            //    URL 里的 `&` 是查询参数分隔符，解码会破坏链接。
            let results: Vec<SearchResult> = results
                .into_iter()
                .map(|mut r| {
                    r.title = decode_entities(&r.title);
                    r.snippet = decode_entities(&r.snippet);
                    r
                })
                .collect();

            let mut observation = String::new();
            for (i, r) in results.iter().enumerate() {
                observation.push_str(&format!("[{}] {}\n", i + 1, r.title));
                observation.push_str(&format!("    {}\n", r.url));
                if !r.snippet.is_empty() {
                    observation.push_str(&format!("    {}\n", r.snippet));
                }
                observation.push('\n');
            }

            let observation =
                crate::search::truncate_chars(&observation, self.max_observation_chars);

            debug!(query = %query, results = results.len(), "Web 搜索完成");

            Ok(outcome_with_data(
                WEB_SEARCH_TOOL_NAME,
                observation,
                &json!({
                    "query": query,
                    "results": results,
                }),
            ))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ③ 网页正文抓取
// ─────────────────────────────────────────────────────────────────────

/// `web_fetch` 工具名。
pub const WEB_FETCH_TOOL_NAME: &str = "web_fetch";

/// 抓取网页正文的工具。
///
/// ## 为什么需要它（而不是只搜索）
///
/// 搜索结果的**摘要只有一句话**，经常不足以支撑一个结论。
/// 要写「这个库的 1.81 版本改了什么」，
/// 必须**打开原文看正文**。
///
/// 这也是「研究助手」和「搜索引擎客户端」的区别。
#[derive(Debug)]
pub struct WebFetchTool {
    backend: Arc<dyn WebBackend>,
    max_chars: usize,
}

impl WebFetchTool {
    /// 创建工具。
    pub fn new(backend: Arc<dyn WebBackend>, max_chars: usize) -> Self {
        Self {
            backend,
            max_chars: max_chars.max(500),
        }
    }
}

/// `web_fetch` 的参数 Schema。
pub fn web_fetch_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "url": {
                "type": "string",
                "description": "要抓取的网页完整链接（http/https 开头）。"
            }
        },
        "required": ["url"]
    })
}

impl ErrorAwareTool for WebFetchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            WEB_FETCH_TOOL_NAME,
            "打开一个网页并返回它的**正文**（已去掉导航、广告和脚本）。\
             搜索结果的摘要通常太短，\
             需要引用具体条款、版本号、API 用法时请务必打开原文。",
            web_fetch_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let url = args
                .get("url")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();

            if url.is_empty() {
                return Err(
                    ToolError::invalid_args(WEB_FETCH_TOOL_NAME, "缺少 url 参数")
                        .with_hint("传入完整链接，例如 \"https://doc.rust-lang.org/book/\"。"),
                );
            }

            // ⚠️ 只允许 http/https —— 挡掉 `file://` / `data:` 这类本地读取。
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Err(ToolError::invalid_args(
                    WEB_FETCH_TOOL_NAME,
                    format!("只支持 http/https 链接，收到的是 `{url}`"),
                )
                .with_hint("把链接改成 http:// 或 https:// 开头的完整地址。"));
            }

            let page = self.backend.fetch(&url, self.max_chars).await?;

            debug!(url = %url, chars = page.text.chars().count(), "网页正文抓取完成");

            Ok(outcome_with_data(
                WEB_FETCH_TOOL_NAME,
                page.text.clone(),
                &json!({
                    "url": page.url,
                    "chars": page.text.chars().count(),
                    // ⚠️ 正文也要带上 —— 证据层靠它生成证据正文。
                    "text": page.text,
                }),
            ))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ④ 长期记忆
// ─────────────────────────────────────────────────────────────────────

/// `memory` 工具名。
pub const MEMORY_TOOL_NAME: &str = "memory";

/// 长期记忆工具（读写一体）。
///
/// ## 为什么一个工具两个动作
///
/// OpenAI 的最佳实践里有一条「**合并工具优于碎片化工具**」——
/// 工具太多会让模型选择困难（16 实测：`required` 档 7 次调用里 5 次是硬凑的）。
///
/// 所以 `memory` 用一个 `action` 参数区分「记」和「查」，
/// 保持工具总数在低位。
///
/// ## 与 17 的关系
///
/// 复用 17 的 `MemoryStore` / `MemoryRetriever` ——
/// 包括「同 (owner, subject, predicate) 合并、旧值进 versions」的语义。
pub struct MemoryTool {
    store: Arc<dyn agent_17::longterm::MemoryStore>,
    retriever: Arc<dyn agent_17::longterm::MemoryRetriever>,
    owner: String,
    default_top_k: usize,
    max_top_k: usize,
}

impl std::fmt::Debug for MemoryTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ⚠️ `dyn MemoryStore` 没有 `Debug` —— 所以不能 derive。
        f.debug_struct("MemoryTool")
            .field("owner", &self.owner)
            .field("default_top_k", &self.default_top_k)
            .finish_non_exhaustive()
    }
}

impl MemoryTool {
    /// 创建工具。
    pub fn new(
        store: Arc<dyn agent_17::longterm::MemoryStore>,
        retriever: Arc<dyn agent_17::longterm::MemoryRetriever>,
        owner: impl Into<String>,
        default_top_k: usize,
        max_top_k: usize,
    ) -> Self {
        Self {
            store,
            retriever,
            owner: owner.into(),
            default_top_k: default_top_k.max(1),
            max_top_k: max_top_k.max(1),
        }
    }

    /// 工具的 owner（按 owner 分区）。
    pub fn owner(&self) -> &str {
        &self.owner
    }
}

/// `memory` 的参数 Schema。
pub fn memory_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember", "recall"],
                "description": "remember = 写入一条事实；recall = 检索已有事实。"
            },
            "subject": {
                "type": "string",
                "description": "事实的主语，例如「用户」「项目 A」。action=remember 时必填。"
            },
            "predicate": {
                "type": "string",
                "description": "事实的属性，例如「技术栈偏好」。action=remember 时必填。"
            },
            "value": {
                "type": "string",
                "description": "事实的值。action=remember 时必填。"
            },
            "query": {
                "type": "string",
                "description": "检索关键词。action=recall 时必填。"
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
                "description": "召回条数。action=recall 时可选。"
            }
        },
        "required": ["action"]
    })
}

impl ErrorAwareTool for MemoryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            MEMORY_TOOL_NAME,
            "读写**长期记忆**（跨会话、跨线程）。\n\
             · action=remember：记下一条事实（subject + predicate + value），\
               相同 (主语, 属性) 再次写入会**覆盖**旧值（旧值进历史）。\n\
             · action=recall：按关键词召回与当前任务相关的事实。\n\
             适合记录用户的偏好、项目背景、已确认的结论。",
            memory_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let action = args
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();

            match action.as_str() {
                "remember" => {
                    let subject = args
                        .get("subject")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim();
                    let predicate = args
                        .get("predicate")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim();
                    let value = args
                        .get("value")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim();

                    if subject.is_empty() || predicate.is_empty() || value.is_empty() {
                        return Err(ToolError::invalid_args(
                            MEMORY_TOOL_NAME,
                            "remember 需要 subject、predicate、value 三个字段",
                        )
                        .with_hint("例如 subject=\"用户\" predicate=\"技术栈偏好\" value=\"Rust + axum\"。"));
                    }

                    let fact =
                        agent_17::longterm::MemoryFact::new(subject, predicate, value, &self.owner);

                    let merged = self.store.put(fact.clone()).await.map_err(|e| {
                        ToolError::new(
                            ErrorCode::InternalFault,
                            MEMORY_TOOL_NAME,
                            format!("写入记忆失败: {e}"),
                        )
                    })?;

                    let observation = if merged {
                        format!("已更新记忆：{}", fact.render())
                    } else {
                        format!("已记住：{}", fact.render())
                    };

                    Ok(outcome_with_data(
                        MEMORY_TOOL_NAME,
                        observation,
                        &json!({ "id": fact.id, "merged": merged }),
                    ))
                }
                "recall" => {
                    let query = args
                        .get("query")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();

                    if query.is_empty() {
                        return Err(ToolError::invalid_args(
                            MEMORY_TOOL_NAME,
                            "recall 需要 query 参数",
                        )
                        .with_hint("传入要检索的关键词，例如「技术栈」。"));
                    }

                    let top_k = args
                        .get("top_k")
                        .and_then(Value::as_u64)
                        .map(|v| v as usize)
                        .unwrap_or(self.default_top_k)
                        .clamp(1, self.max_top_k);

                    let facts = self
                        .retriever
                        .search(&query, &self.owner, top_k)
                        .await
                        .map_err(|e| {
                            ToolError::new(
                                ErrorCode::InternalFault,
                                MEMORY_TOOL_NAME,
                                format!("召回记忆失败: {e}"),
                            )
                        })?;

                    if facts.is_empty() {
                        return Err(ToolError::new(
                            ErrorCode::NotFound,
                            MEMORY_TOOL_NAME,
                            format!("记忆里没有与「{query}」相关的内容"),
                        )
                        .with_hint("换个关键词，或先 remember 记下必要的信息。"));
                    }

                    let observation = facts
                        .iter()
                        .map(|f| format!("- {}", f.render()))
                        .collect::<Vec<_>>()
                        .join("\n");

                    Ok(outcome_with_data(
                        MEMORY_TOOL_NAME,
                        observation,
                        &json!({
                            "count": facts.len(),
                            "facts": facts.iter().map(|f| json!({
                                "subject": f.subject,
                                "predicate": f.predicate,
                                "value": f.value,
                            })).collect::<Vec<_>>(),
                        }),
                    ))
                }
                "" => Err(
                    ToolError::invalid_args(MEMORY_TOOL_NAME, "缺少 action 参数")
                        .with_hint("action 只能是 \"remember\" 或 \"recall\"。"),
                ),
                other => Err(ToolError::invalid_args(
                    MEMORY_TOOL_NAME,
                    format!("未知的 action：`{other}`"),
                )
                .with_hint("action 只能是 \"remember\" 或 \"recall\"。")),
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ⑤ 算术
// ─────────────────────────────────────────────────────────────────────

/// `calculate` 工具名。
pub const CALCULATE_TOOL_NAME: &str = "calculate";

/// 手写算术求值器工具。
///
/// ## 为什么复用 13 的而不是重写
///
/// 13 的 `calculate` 是**手写**的表达式求值器（不用 `eval`，
/// 因为 `eval` 能执行任意代码）。
/// 重写等于把「不用 eval」这个决策重新做一遍。
#[derive(Debug)]
pub struct CalculateTool;

/// `calculate` 的参数 Schema。
pub fn calculate_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "expression": {
                "type": "string",
                "description": "算术表达式，例如 \"(120 - 45) / 3\"。支持 + - * / % 和括号。"
            }
        },
        "required": ["expression"]
    })
}

impl ErrorAwareTool for CalculateTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            CALCULATE_TOOL_NAME,
            "计算一个算术表达式。\
             周报里的统计数字（求和、平均、百分比）必须用它算，\
             不要自己心算 —— 心算容易错，而数字错了整篇周报就不可信。",
            calculate_schema(),
        )
    }

    fn call<'a>(&'a self, args: &'a Value, _ctx: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let expression = args
                .get("expression")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();

            if expression.is_empty() {
                return Err(
                    ToolError::invalid_args(CALCULATE_TOOL_NAME, "缺少 expression 参数")
                        .with_hint("例如 \"(120 - 45) / 3\"。"),
                );
            }

            match agent_13::tools::calculate(&expression) {
                Ok(value) => Ok(outcome_with_data(
                    CALCULATE_TOOL_NAME,
                    format!("{expression} = {value}"),
                    &json!({ "expression": expression, "value": value }),
                )),
                Err(e) => Err(ToolError::invalid_args(
                    CALCULATE_TOOL_NAME,
                    format!("表达式无法计算：{e}"),
                )
                .with_hint("检查括号和运算符，例如 \"(120 - 45) / 3\"。")),
            }
        })
    }
}

/// 构建默认工具注册表。
///
/// ## 工具数量的取舍
///
/// OpenAI 最佳实践：「保持初始可用函数的数量足够少 —— 建议一开始少于 20 个」。
/// 本项目**只有 5 个** —— 因为每个工具都会让模型的选择空间变大。
///
/// ## ⚠️ `max_evidence_per_section` 在这里**不用**
///
/// 它是**报告生成阶段**的参数（每类证据渲染几条进提示词），
/// 不是工具层的 —— 工具层不知道「报告」这回事。
///
/// 保留它是为了让调用方**一处配置**（`AppConfig`），
/// 不用在两个地方各传一遍。
pub fn default_registry(
    kb: Arc<rag_08::bm25::Bm25Index>,
    backend: Arc<dyn WebBackend>,
    memory: MemoryTool,
    search_top_k: usize,
    _max_evidence_per_section: usize,
) -> agent_18::tool::ToolRegistry {
    agent_18::tool::ToolRegistry::new(vec![
        Arc::new(KbSearchTool::new(kb, 4, 10, 4000)),
        Arc::new(WebSearchTool::new(backend.clone(), search_top_k, 10, 4000)),
        Arc::new(WebFetchTool::new(backend, 6000)),
        Arc::new(memory),
        Arc::new(CalculateTool),
    ])
}

/// 让 `SearchBackend` 的导入不被判定为未使用。
#[allow(dead_code)]
fn _backend_marker(backend: crate::config::SearchBackend) -> crate::config::SearchBackend {
    backend
}

/// 让 `warn` 的导入不被判定为未使用。
#[allow(dead_code)]
fn _warn_marker() {
    warn!("marker");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_08::bm25::{Bm25Document, Bm25Index};

    fn index() -> Arc<Bm25Index> {
        let docs = vec![
            Bm25Document {
                id: "1".to_string(),
                source: "notes.md".to_string(),
                heading_path: vec!["架构".to_string()],
                text: "错误处理的降级策略是把失败转成观察喂回模型".to_string(),
            },
            Bm25Document {
                id: "2".to_string(),
                source: "notes.md".to_string(),
                heading_path: vec!["检索".to_string()],
                text: "BM25 是关键词检索算法，不理解语义".to_string(),
            },
        ];
        Arc::new(Bm25Index::build(&docs))
    }

    fn kb_tool() -> KbSearchTool {
        KbSearchTool::new(index(), 4, 10, 4000)
    }

    #[tokio::test]
    async fn kb_search_finds_relevant_chunk() {
        let tool = kb_tool();

        let outcome = tool
            .call(&json!({ "query": "降级策略" }), &ToolContext::empty())
            .await
            .unwrap();

        assert!(outcome.ok);
        assert!(outcome.observation.contains("降级"));
    }

    #[tokio::test]
    async fn kb_search_rejects_missing_query() {
        let tool = kb_tool();

        let err = tool
            .call(&json!({}), &ToolContext::empty())
            .await
            .unwrap_err();

        // ⚠️ 参数错 = Business —— 不该重试。
        assert_eq!(err.kind, agent_18::error::ErrorKind::Business);
        assert!(!err.retryable());
    }

    #[tokio::test]
    async fn kb_search_reports_not_found_as_business() {
        let tool = kb_tool();

        // ⚠️ 查询词必须和语料**完全没有共同字符** ——
        //    08 的中文分词是「unigram + bigram」，
        //    所以「量子计算」里的「算」会命中「算法」，
        //    从而返回一堆不相关的片段。
        //
        //    这不是 bug，是中文关键词检索的固有特性
        //    （要解决得靠重排 / 向量检索，那是 08/12 的活）。
        let err = tool
            .call(&json!({ "query": "xyzzy" }), &ToolContext::empty())
            .await
            .unwrap_err();

        assert_eq!(err.code, ErrorCode::NotFound);
        assert_eq!(err.kind, agent_18::error::ErrorKind::Business);
    }

    #[tokio::test]
    async fn web_search_uses_backend() {
        let backend = Arc::new(crate::search::MockBackend::new(vec![
            crate::search::SearchResult {
                title: "Rust 官网".to_string(),
                url: "https://rust-lang.org".to_string(),
                snippet: "官方站点".to_string(),
            },
        ]));
        let tool = WebSearchTool::new(backend, 5, 10, 4000);

        let outcome = tool
            .call(&json!({ "query": "rust" }), &ToolContext::empty())
            .await
            .unwrap();

        assert!(outcome.observation.contains("rust-lang.org"));
    }

    #[tokio::test]
    async fn web_search_decodes_entities_in_snippets() {
        // ⚠️ **回归测试**：实测踩过的坑。
        //
        // Bing 的摘要里有 `&ensp;` `&#0183;` —— 不解码的话
        // 模型看到的是一堆乱码。
        let backend = Arc::new(crate::search::MockBackend::new(vec![
            crate::search::SearchResult {
                title: "标题 &amp; 更多".to_string(),
                url: "https://example.com/a?b=1&c=2".to_string(),
                snippet: "3 天之前&ensp;&#0183;&ensp;正文".to_string(),
            },
        ]));
        let tool = WebSearchTool::new(backend, 5, 10, 4000);

        let outcome = tool
            .call(&json!({ "query": "rust" }), &ToolContext::empty())
            .await
            .unwrap();

        assert!(outcome.observation.contains("3 天之前"));
        assert!(!outcome.observation.contains("&ensp;"));
        assert!(!outcome.observation.contains("&#0183;"));
        assert!(outcome.observation.contains("标题 & 更多"));

        // ⚠️ URL **不能**被解码 ——
        //    里面的 `&` 是查询参数分隔符，解码会破坏链接。
        assert!(
            outcome
                .observation
                .contains("https://example.com/a?b=1&c=2")
        );
    }

    #[tokio::test]
    async fn web_fetch_rejects_non_http_urls() {
        let backend = Arc::new(crate::search::MockBackend::new(Vec::new()));
        let tool = WebFetchTool::new(backend, 1000);

        let err = tool
            .call(
                &json!({ "url": "file:///etc/passwd" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap_err();

        assert_eq!(err.kind, agent_18::error::ErrorKind::Business);
    }

    #[tokio::test]
    async fn web_fetch_reads_page() {
        let backend =
            Arc::new(crate::search::MockBackend::new(Vec::new()).with_page("这是正文内容"));
        let tool = WebFetchTool::new(backend, 1000);

        let outcome = tool
            .call(
                &json!({ "url": "https://example.com" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap();

        assert!(outcome.observation.contains("这是正文内容"));
    }

    #[tokio::test]
    async fn calculate_evaluates_expression() {
        let tool = CalculateTool;

        let outcome = tool
            .call(
                &json!({ "expression": "(120 - 45) / 3" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap();

        assert!(outcome.observation.contains("25"));
    }

    #[tokio::test]
    async fn calculate_reports_bad_expression() {
        let tool = CalculateTool;

        let err = tool
            .call(&json!({ "expression": "1 +" }), &ToolContext::empty())
            .await
            .unwrap_err();

        assert_eq!(err.kind, agent_18::error::ErrorKind::Business);
    }

    #[tokio::test]
    async fn memory_tool_rejects_unknown_action() {
        let store: Arc<dyn agent_17::longterm::MemoryStore> =
            Arc::new(agent_17::longterm::InMemoryMemoryStore::new());
        let retriever: Arc<dyn agent_17::longterm::MemoryRetriever> =
            Arc::new(agent_17::longterm::Bm25MemoryRetriever::new(store.clone()));
        let tool = MemoryTool::new(store, retriever, "tester", 5, 20);

        let err = tool
            .call(&json!({ "action": "forget" }), &ToolContext::empty())
            .await
            .unwrap_err();

        assert_eq!(err.kind, agent_18::error::ErrorKind::Business);
    }

    #[tokio::test]
    async fn memory_tool_remember_and_recall() {
        let store: Arc<dyn agent_17::longterm::MemoryStore> =
            Arc::new(agent_17::longterm::InMemoryMemoryStore::new());
        let retriever: Arc<dyn agent_17::longterm::MemoryRetriever> =
            Arc::new(agent_17::longterm::Bm25MemoryRetriever::new(store.clone()));
        let tool = MemoryTool::new(store, retriever, "tester", 5, 20);

        tool.call(
            &json!({ "action": "remember", "subject": "用户", "predicate": "技术栈", "value": "Rust" }),
            &ToolContext::empty(),
        )
        .await
        .unwrap();

        let outcome = tool
            .call(
                &json!({ "action": "recall", "query": "技术栈" }),
                &ToolContext::empty(),
            )
            .await
            .unwrap();

        assert!(outcome.observation.contains("Rust"));
    }
}
