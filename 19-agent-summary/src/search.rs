//! Web 搜索与网页抓取 —— **可替换的后端**。
//!
//! # 为什么搜索要抽象成 trait
//!
//! 搜索服务的可用性**因网络环境而异**。本项目的开发环境实测：
//!
//! | 端点 | 结果 |
//! |---|---|
//! | `html.duckduckgo.com` | ❌ 连接超时 |
//! | `api.duckduckgo.com` | ❌ 连接超时 |
//! | `zh.wikipedia.org` | ❌ 连接超时 |
//! | `api.search.brave.com` | ❌ 连接超时 |
//! | `searx.be` | ❌ 连接超时 |
//! | `www.mojeek.com` | ⚠️ HTTP 200 但正文是「403 Forbidden —— 检测到自动化查询」 |
//! | **`www.bing.com`** | ✅ HTTP 200，10 条结果，结构稳定 |
//! | **`api.github.com`** | ✅ HTTP 200，JSON 格式，最稳 |
//! | **`crates.io`** | ✅ HTTP 200，JSON 格式 |
//!
//! 也就是说：**通用网页搜索在这个网络里只有 Bing 可靠**，
//! 而**技术类搜索有 GitHub / crates.io 这类 JSON API 更稳**。
//!
//! 所以本项目：
//!
//! 1. 把搜索抽象成 [`WebBackend`] trait —— 上层逻辑不依赖任何一家
//! 2. 内置 [`BingBackend`]（通用搜索）与 [`GithubBackend`]（技术搜索）
//! 3. 内置 [`MockBackend`] —— **零网络确定性测试**用
//!
//! # ⚠️ 抓取网页的三个坑
//!
//! ## 坑 1：HTML 不是文本
//!
//! 网页正文里混着 `<script>` / `<style>` / 导航栏 / 页脚。
//! 直接剥标签会把一堆 JS 变量塞进上下文 —— 既浪费 token 又干扰模型。
//!
//! ## 坑 2：`<a>` 标签里的文字会**粘成一片**
//!
//! `<a>第一个</a><a>第二个</a>` 直接剥标签会变成「第一个第二个」，
//! 读起来像一句话。需要在**块级标签**处插入换行、**内联标签**处插入空格。
//!
//! ## 坑 3：HTML 实体
//!
//! `&amp;` `&lt;` `&#39;` `&nbsp;` 不解码的话，
//! 模型会看到一堆 `&amp;` —— 看起来像乱码，实际是「&」。

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use agent_logging::{debug, info};

use crate::error::{ErrorCode, ToolError};

/// 一条搜索结果。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    /// 标题。
    pub title: String,

    /// 链接。
    pub url: String,

    /// 摘要（搜索服务给的短描述）。
    pub snippet: String,
}

/// 抓取到的网页正文。
#[derive(Debug, Clone)]
pub struct PageContent {
    /// 最终 URL（跟随重定向后）。
    pub url: String,

    /// 抽取出来的正文（已去掉脚本 / 样式 / 导航）。
    pub text: String,
}

/// 搜索 / 抓取的 future。
pub type WebFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, ToolError>> + Send + 'a>>;

/// Web 后端。
///
/// ## 与 18 的 `ErrorAwareTool` 的关系
///
/// 这是**能力层**，不是工具层。
/// 工具（`web_search` / `web_fetch`）调用它，
/// 并把它的错误**分类**（Transient / Business / Fatal）后交给中间件。
pub trait WebBackend: Send + Sync + std::fmt::Debug {
    /// 搜索。
    fn search<'a>(&'a self, query: &'a str, top_k: usize) -> WebFuture<'a, Vec<SearchResult>>;

    /// 抓取网页正文。
    fn fetch<'a>(&'a self, url: &'a str, max_chars: usize) -> WebFuture<'a, PageContent>;

    /// 后端名（用于日志与 API 展示）。
    fn name(&self) -> &str;
}

// ─────────────────────────────────────────────────────────────────────
// Bing 后端
// ─────────────────────────────────────────────────────────────────────

/// Bing 网页搜索后端。
#[derive(Debug)]
pub struct BingBackend {
    client: reqwest::Client,
    base_url: String,
}

impl BingBackend {
    /// 创建后端。
    pub fn new(timeout_secs: u64) -> Result<Self, ToolError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| {
                ToolError::new(
                    ErrorCode::InternalFault,
                    "bing",
                    format!("构造 HTTP 客户端失败: {e}"),
                )
            })?;

        Ok(Self {
            client,
            base_url: "https://www.bing.com/search".to_string(),
        })
    }

    /// 用固定前缀构造（测试用）。
    pub fn with_base_url(client: reqwest::Client, base_url: impl Into<String>) -> Self {
        Self {
            client,
            base_url: base_url.into(),
        }
    }
}

/// 浏览器 User-Agent。
///
/// ⚠️ 搜索引擎会对**没有 UA** 的请求做特殊处理（返回精简版页面或直接拒绝）。
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

impl WebBackend for BingBackend {
    fn search<'a>(&'a self, query: &'a str, top_k: usize) -> WebFuture<'a, Vec<SearchResult>> {
        Box::pin(async move {
            let url = format!(
                "{}?q={}&count={}",
                self.base_url,
                url_encode(query),
                (top_k * 2).clamp(1, 50)
            );

            let html = self.client.get(&url).send().await.map_err(|e| {
                // ⚠️ 网络层失败 = **Transient** —— 重试可能成功。
                ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_search",
                    format!("搜索请求失败: {e}"),
                )
                .with_hint("这是网络临时故障，可以重试或换个搜索后端。")
            })?;

            let status = html.status();
            if !status.is_success() {
                return Err(ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_search",
                    format!("搜索服务返回 HTTP {status}"),
                )
                .with_hint("稍后重试，或切换 SUM_SEARCH_BACKEND。"));
            }

            let body = html.text().await.map_err(|e| {
                ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_search",
                    format!("读取搜索结果失败: {e}"),
                )
            })?;

            let results = parse_bing_results(&body, top_k);

            if results.is_empty() {
                // ⚠️ 「搜不到」和「搜错了」是两回事。
                // 解析不出结果通常是**页面结构变了** —— 那不是模型的错，
                // 也不是 Business（改参数没用），但重试同一次搜索也没意义。
                // 归为 Business：调用方应该换后端或改查询词。
                return Err(ToolError::new(
                    ErrorCode::NotFound,
                    "web_search",
                    "没有解析到搜索结果（搜索服务可能返回了验证页，或查询过于宽泛）",
                )
                .with_hint("换个关键词，或改用 SUM_SEARCH_BACKEND=github。"));
            }

            debug!(query = %query, results = results.len(), "Bing 搜索完成");
            Ok(results)
        })
    }

    fn fetch<'a>(&'a self, url: &'a str, max_chars: usize) -> WebFuture<'a, PageContent> {
        Box::pin(async move {
            let response = self.client.get(url).send().await.map_err(|e| {
                ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_fetch",
                    format!("抓取网页失败: {e}"),
                )
                .with_hint("网络临时故障，可以重试。")
            })?;

            let status = response.status();
            let final_url = response.url().to_string();

            if !status.is_success() {
                return Err(ToolError::new(
                    ErrorCode::NotFound,
                    "web_fetch",
                    format!("网页返回 HTTP {status}"),
                )
                .with_hint("换一个可访问的链接。"));
            }

            let html = response.text().await.map_err(|e| {
                ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_fetch",
                    format!("读取网页正文失败: {e}"),
                )
            })?;

            let text = extract_main_text(&html);

            if text.trim().is_empty() {
                return Err(ToolError::new(
                    ErrorCode::NotFound,
                    "web_fetch",
                    "网页里没有提取到正文",
                )
                .with_hint("换一个人类可读的页面（避免纯 JS 渲染的站点）。"));
            }

            Ok(PageContent {
                url: final_url,
                text: truncate_chars(&text, max_chars),
            })
        })
    }

    fn name(&self) -> &str {
        "bing"
    }
}

// ─────────────────────────────────────────────────────────────────────
// GitHub 后端
// ─────────────────────────────────────────────────────────────────────

/// GitHub 搜索后端。
///
/// ## 定位
///
/// 它**不是**通用网页搜索 —— 它搜的是**仓库**。
/// 但对「技术选型」「找参考实现」这类问题，
/// 仓库搜索的结果质量往往比通用网页搜索高得多。
#[derive(Debug)]
pub struct GithubBackend {
    client: reqwest::Client,
    token: Option<String>,
}

impl GithubBackend {
    /// 创建后端。
    pub fn new(timeout_secs: u64, token: Option<String>) -> Result<Self, ToolError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| {
                ToolError::new(
                    ErrorCode::InternalFault,
                    "github",
                    format!("构造 HTTP 客户端失败: {e}"),
                )
            })?;

        Ok(Self { client, token })
    }
}

impl WebBackend for GithubBackend {
    fn search<'a>(&'a self, query: &'a str, top_k: usize) -> WebFuture<'a, Vec<SearchResult>> {
        Box::pin(async move {
            let mut request = self.client.get(format!(
                "https://api.github.com/search/repositories?q={}&per_page={}&sort=stars",
                url_encode(query),
                top_k.clamp(1, 30)
            ));
            request = request.header("Accept", "application/vnd.github+json");
            if let Some(token) = &self.token {
                request = request.header("Authorization", format!("Bearer {token}"));
            }

            let response = request.send().await.map_err(|e| {
                ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_search",
                    format!("GitHub 搜索请求失败: {e}"),
                )
            })?;

            let status = response.status();
            if !status.is_success() {
                return Err(ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_search",
                    format!("GitHub 搜索返回 HTTP {status}"),
                )
                .with_hint("稍后重试（匿名请求限流较严）。"));
            }

            let body: serde_json::Value = response.json().await.map_err(|e| {
                ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_search",
                    format!("解析 GitHub 响应失败: {e}"),
                )
            })?;

            let items = body
                .get("items")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let results: Vec<SearchResult> = items
                .iter()
                .filter_map(|item| {
                    let full_name = item.get("full_name")?.as_str()?;
                    let html_url = item.get("html_url")?.as_str()?;
                    let description = item
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let stars = item
                        .get("stargazers_count")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);

                    Some(SearchResult {
                        title: format!("{full_name}（⭐ {stars}）"),
                        url: html_url.to_string(),
                        snippet: description,
                    })
                })
                .collect();

            if results.is_empty() {
                return Err(ToolError::new(
                    ErrorCode::NotFound,
                    "web_search",
                    "GitHub 没有匹配的仓库",
                )
                .with_hint("换个关键词，或减少限定词。"));
            }

            Ok(results)
        })
    }

    fn fetch<'a>(&'a self, url: &'a str, max_chars: usize) -> WebFuture<'a, PageContent> {
        Box::pin(async move {
            let response = self.client.get(url).send().await.map_err(|e| {
                ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_fetch",
                    format!("抓取网页失败: {e}"),
                )
            })?;

            let status = response.status();
            if !status.is_success() {
                return Err(ToolError::new(
                    ErrorCode::NotFound,
                    "web_fetch",
                    format!("网页返回 HTTP {status}"),
                ));
            }

            let html = response.text().await.map_err(|e| {
                ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_fetch",
                    format!("读取网页正文失败: {e}"),
                )
            })?;

            Ok(PageContent {
                url: url.to_string(),
                text: truncate_chars(&extract_main_text(&html), max_chars),
            })
        })
    }

    fn name(&self) -> &str {
        "github"
    }
}

// ─────────────────────────────────────────────────────────────────────
// Mock 后端（测试 / 离线）
// ─────────────────────────────────────────────────────────────────────

/// 离线后端 —— 返回预设结果。
///
/// ## 用途
///
/// 1. **零网络确定性测试**（本仓库一贯的做法）
/// 2. `SUM_SEARCH_BACKEND=offline` 时**故意**让 Web 工具失败，
///    用来演示「搜索不可用时降级到本地知识库」
#[derive(Debug)]
pub struct MockBackend {
    results: Vec<SearchResult>,
    page: String,
    /// 搜索是否应该失败（用来测错误路径）。
    fail_search: bool,
}

impl MockBackend {
    /// 创建一个返回固定搜索结果的离线后端。
    pub fn new(results: Vec<SearchResult>) -> Self {
        Self {
            results,
            page: "这是离线后端提供的固定页面内容。".to_string(),
            fail_search: false,
        }
    }

    /// 设置固定的页面内容。
    pub fn with_page(mut self, page: impl Into<String>) -> Self {
        self.page = page.into();
        self
    }

    /// 让搜索**总是失败**（测降级路径）。
    pub fn failing(mut self) -> Self {
        self.fail_search = true;
        self
    }
}

impl WebBackend for MockBackend {
    fn search<'a>(&'a self, _query: &'a str, top_k: usize) -> WebFuture<'a, Vec<SearchResult>> {
        Box::pin(async move {
            if self.fail_search {
                return Err(ToolError::new(
                    ErrorCode::UpstreamUnavailable,
                    "web_search",
                    "离线后端：搜索被禁用",
                )
                .with_hint("这是配置决定的失败，不是网络问题。"));
            }
            Ok(self.results.iter().take(top_k).cloned().collect())
        })
    }

    fn fetch<'a>(&'a self, url: &'a str, max_chars: usize) -> WebFuture<'a, PageContent> {
        Box::pin(async move {
            Ok(PageContent {
                url: url.to_string(),
                text: truncate_chars(&self.page, max_chars),
            })
        })
    }

    fn name(&self) -> &str {
        "mock"
    }
}

// ─────────────────────────────────────────────────────────────────────
// HTML 解析
// ─────────────────────────────────────────────────────────────────────

/// 解析 Bing 搜索结果页。
///
/// ## 页面结构（实测）
///
/// ```text
/// <li class="b_algo" ...>
///   <div class="b_tpcn"> ... <div class="tptt">runoob.com</div> ... </div>
///   <h2><a href="https://www.runoob.com/rust/rust-tutorial.html">Rust 教程 | 菜鸟教程</a></h2>
///   <div class="b_caption"><p class="b_lineclamp2">...摘要...</p></div>
/// </li>
/// ```
///
/// ## ⚠️ 为什么不用正则
///
/// 正则解析 HTML 是**长期维护灾难** —— 页面一改版就全废。
/// 这里用「**找标记 + 截取 + 就近配对**」的朴素方法：
/// 它对结构小改有容忍度，而且**失败了能看出是哪一步失败**。
///
/// 抽取三样东西：
/// - `h2` 里的 `<a href=...>` → 链接
/// - `h2` 里的 `<a>` 文本 → 标题
/// - `b_caption` 后的第一个 `<p>` → 摘要
pub fn parse_bing_results(html: &str, top_k: usize) -> Vec<SearchResult> {
    let mut results = Vec::new();

    for block in html.split("<li class=\"b_algo\"").skip(1) {
        // 只取到下一个结果块为止 —— 否则后面的内容会污染这一块。
        let block = match block.find("</li>") {
            Some(end) => &block[..end],
            None => block,
        };

        let Some(h2) = find_tag_block(block, "<h2", "</h2>") else {
            continue;
        };

        let Some(href) = extract_href(h2) else {
            continue;
        };

        let title = match find_tag_block(h2, "<a", "</a>") {
            Some(anchor) => strip_tags(anchor),
            None => continue,
        };

        if title.trim().is_empty() || href.trim().is_empty() {
            continue;
        }

        let snippet = find_tag_block(block, "<p", "</p>")
            .map(strip_tags)
            .unwrap_or_default();

        results.push(SearchResult {
            title: title.trim().to_string(),
            url: href.trim().to_string(),
            snippet: snippet.trim().to_string(),
        });

        if results.len() >= top_k {
            break;
        }
    }

    results
}

/// 抽取网页正文。
///
/// ## 处理步骤
///
/// 1. 删掉 `<script>` / `<style>` / `<noscript>` / `<!-- -->` 注释
/// 2. 删掉 `<nav>` / `<header>` / `<footer>` / `<aside>`（导航噪声）
/// 3. `<br>` / `</p>` / `</div>` 等块级标签 → 换行
/// 4. 其余标签 → 剥掉，块级标签之间补空格
/// 5. 解码 HTML 实体
/// 6. 压缩连续空白
pub fn extract_main_text(html: &str) -> String {
    let mut text = html.to_string();

    // ① 整块删除（脚本 / 样式 / 注释 / 导航）。
    for tag in [
        "script", "style", "noscript", "nav", "header", "footer", "aside", "svg",
    ] {
        text = remove_element(&text, tag);
    }
    text = remove_html_comments(&text);

    // ② 块级标签 → 换行。
    for tag in [
        "br",
        "p",
        "div",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "pre",
        "blockquote",
        "table",
    ] {
        text = text.replace(&format!("<{tag}"), "\n<{tag}");
        text = text.replace(&format!("</{tag}>"), "\n</{tag}>");
    }

    // ③ 剥标签。
    //
    // ⚠️ 但**行内标签**（a / span / b / i / code / em / strong）
    //    剥掉后要把两侧**用空格隔开** ——
    //    否则 `<a>第一个</a><a>第二个</a>` 会粘成「第一个第二个」，
    //    读起来像一句话，模型会误解语义。
    text = strip_tags_with_spacing(&text);

    // ④ 解码实体。
    text = decode_entities(&text);

    // ⑤ 压缩空白：连续空行压成一行，连续空格压成一个。
    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    for line in text.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
            out.push('\n');
        } else {
            blank_run = 0;
            out.push_str(&line);
            out.push('\n');
        }
    }

    out.trim().to_string()
}

/// 剥掉所有标签（保留文本）。
pub fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut depth = 0usize;

    for ch in html.chars() {
        match ch {
            '<' => depth += 1,
            '>' => {
                // ⚠️ 可能出现 `>` 在文本里的情况 —— 不能让 depth 变负。
                depth = depth.saturating_sub(1);
            }
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }

    out
}

/// 剥掉标签，但**行内标签**两侧补空格。
///
/// ## 为什么行内标签要特殊处理
///
/// ```text
/// <a>第一个</a><a>第二个</a>
/// ```
///
/// 直接剥标签会得到「第一个第二个」—— 读起来像一个词组，
/// 而原文其实是**两个并列的链接**。
///
/// 块级标签（`div` / `p` / `li`…）在前面已经转成换行了，
/// 所以这里只需要处理行内的。
fn strip_tags_with_spacing(html: &str) -> String {
    /// 行内标签 —— 两侧要补空格。
    const INLINE: [&str; 8] = ["a", "span", "b", "i", "em", "strong", "code", "small"];

    let mut out = String::with_capacity(html.len());
    let mut rest = html;

    loop {
        let Some(start) = rest.find('<') else {
            out.push_str(rest);
            break;
        };

        // 标签之前的文本 —— 原样保留。
        out.push_str(&rest[..start]);
        rest = &rest[start..];

        let Some(end) = rest.find('>') else {
            // 没有闭合的 `>` —— 当成文本。
            out.push_str(rest);
            break;
        };

        let tag_text = &rest[1..end];
        let tag_name: String = tag_text
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();

        if INLINE.contains(&tag_name.as_str()) {
            // ⚠️ 行内标签 —— 在标签**前面**补一个空格。
            //
            // 为什么是「前面」而不是「后面」：
            // 这样 `<a>第一个</a><a>第二个</a>` 会变成
            // 「 第一个 第二个 」—— 两个词被空格隔开。
            out.push(' ');
        }

        rest = &rest[end + 1..];
    }

    out
}

/// 解码常见 HTML 实体。
///
/// ## ⚠️ 为什么不能只解固定的那几个
///
/// 实测 Bing 的摘要里会出现 `&ensp;`（窄空格）和 `&#0183;`（数字实体）。
/// 只解 `&amp;` `&lt;` 那几个的话，
/// 模型会看到一堆 `&ensp;&#0183;` —— 看起来像乱码，实际是「·」和「⋮」。
///
/// 所以这里做三件事：
///
/// 1. 具名实体（固定表）
/// 2. **数字实体**（`&#123;` / `&#x1F600;`）—— 按码点还原
/// 3. 剩下的原样保留（不认识的实体留着比替换成空更好）
pub fn decode_entities(text: &str) -> String {
    let text = text
        .replace("&nbsp;", " ")
        .replace("&ensp;", " ")
        .replace("&emsp;", " ")
        .replace("&thinsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&mdash;", "—")
        .replace("&ndash;", "–")
        .replace("&hellip;", "…")
        .replace("&middot;", "·")
        .replace("&times;", "×")
        .replace("&copy;", "©")
        .replace("&reg;", "®")
        .replace("&trade;", "™")
        .replace("&rarr;", "→")
        .replace("&larr;", "←")
        .replace("&harr;", "↔")
        .replace("&bull;", "•")
        .replace("&laquo;", "«")
        .replace("&raquo;", "»")
        .replace("&deg;", "°")
        .replace("&plusmn;", "±")
        .replace("&frac12;", "½")
        .replace("&sup2;", "²")
        .replace("&sup3;", "³");

    decode_numeric_entities(&text)
}

/// 解码数字实体（`&#123;` 和 `&#x1F600;`）。
///
/// ## ⚠️ 为什么要自己解
///
/// `char::from_u32` 就够了 —— 但**必须过滤掉无效码点**：
/// `&#0;` 和 `&#xD800;`（代理区）都不是合法字符，
/// 直接转换会得到「无字符」或 panic。
fn decode_numeric_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    loop {
        let Some(start) = rest.find("&#") else {
            out.push_str(rest);
            break;
        };

        out.push_str(&rest[..start]);
        rest = &rest[start..];

        let Some(end) = rest.find(';') else {
            // 没有分号 —— 原样保留。
            out.push_str(rest);
            break;
        };

        let body = &rest[2..end];
        let code = if let Some(hex) = body.strip_prefix(['x', 'X']) {
            u32::from_str_radix(hex, 16).ok()
        } else {
            body.parse::<u32>().ok()
        };

        match code.and_then(char::from_u32) {
            // ⚠️ 控制字符（NUL、代理区…）不能放进正文 ——
            //    NUL 会污染日志和 JSON，代理区根本不是合法字符。
            //    所以只接受**可见**字符（以及 tab / 换行）。
            Some(c) if !c.is_control() || c == '\t' || c == '\n' || c == '\r' => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            Some(_) => {
                // ⚠️ 控制字符 —— 换成空格而不是丢掉内容。
                out.push(' ');
                rest = &rest[end + 1..];
            }
            None => {
                // ⚠️ 无效码点 —— 原样保留（不吞掉内容）。
                out.push_str(&rest[..=end]);
                rest = &rest[end + 1..];
            }
        }
    }

    out
}

/// 找出某个标签的**内容**（含开闭标签）。
fn find_tag_block<'a>(html: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = html.find(open)?;
    let end = html[start..].find(close)? + start + close.len();
    Some(&html[start..end])
}

/// 从一段 HTML 里抽出第一个 `href`。
fn extract_href(html: &str) -> Option<String> {
    let idx = html.find("href=\"")?;
    let rest = &html[idx + 6..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// 删除某个元素（含内容）。
fn remove_element(html: &str, tag: &str) -> String {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");

    let mut out = String::with_capacity(html.len());
    let mut rest = html;

    loop {
        let Some(start) = rest.find(&open) else {
            out.push_str(rest);
            break;
        };

        out.push_str(&rest[..start]);
        rest = &rest[start..];

        match rest.find(&close) {
            Some(end) => {
                rest = &rest[end + close.len()..];
            }
            None => break,
        }
    }

    out
}

/// 删除 HTML 注释。
fn remove_html_comments(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut rest = html;

    loop {
        let Some(start) = rest.find("<!--") else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..start]);
        match rest[start..].find("-->") {
            Some(end) => rest = &rest[start + end + 3..],
            None => break,
        }
    }

    out
}

/// URL 编码（只编码必要字符，避免引入 url crate）。
pub fn url_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// 按**字符数**截断（保留开头）。
pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 || text.chars().count() <= max_chars {
        return text.to_string();
    }
    let kept: String = text.chars().take(max_chars).collect();
    format!("{kept}\n…（内容过长，已截断）")
}

/// 构造一个后端（按配置）。
pub fn build_backend(config: &crate::config::AppConfig) -> Result<Box<dyn WebBackend>, ToolError> {
    match config.search_backend {
        crate::config::SearchBackend::Bing => {
            Ok(Box::new(BingBackend::new(config.web_timeout_secs)?))
        }
        crate::config::SearchBackend::Github => Ok(Box::new(GithubBackend::new(
            config.web_timeout_secs,
            config.github_token.clone(),
        )?)),
        crate::config::SearchBackend::Offline => {
            info!("使用离线后端 —— Web 搜索工具会被禁用");
            Ok(Box::new(MockBackend::new(Vec::new()).failing()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一小段 Bing 结果页样本（结构取自实测）。
    const BING_SAMPLE: &str = r#"
<html><body>
<li class="b_algo" data-id iid=SERP.1>
<div class="b_tpcn"><div class="tptt">rust-lang.org</div></div>
<h2><a target="_blank" href="https://rust-lang.org/" h="ID=SERP.1.2"><strong>Rust</strong> Programming Language</a></h2>
<div class="b_caption"><p class="b_lineclamp2">Rust is <strong>blazingly fast</strong> and memory-efficient.</p></div>
</li>
<li class="b_algo" data-id iid=SERP.2>
<h2><a href="https://doc.rust-lang.org/book/">The Rust Programming Language book</a></h2>
<div class="b_caption"><p>Learn Rust here.</p></div>
</li>
</body></html>
"#;

    #[test]
    fn parses_bing_results() {
        let results = parse_bing_results(BING_SAMPLE, 5);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].url, "https://rust-lang.org/");
        assert!(results[0].title.contains("Rust"));
        assert!(results[0].title.contains("Programming Language"));
        assert!(results[0].snippet.contains("blazingly fast"));
        assert_eq!(results[1].url, "https://doc.rust-lang.org/book/");
    }

    #[test]
    fn respects_top_k() {
        let results = parse_bing_results(BING_SAMPLE, 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn empty_html_yields_no_results() {
        assert!(parse_bing_results("<html></html>", 5).is_empty());
    }

    #[test]
    fn extracts_main_text_and_drops_scripts() {
        let html = r#"
            <html><head><script>var x = 1; alert("hi");</script>
            <style>body { color: red; }</style></head>
            <body><nav>导航 首页 关于</nav>
            <h1>标题</h1><p>第一段。</p><p>第二段。</p>
            <footer>页脚</footer></body></html>
        "#;

        let text = extract_main_text(html);

        assert!(text.contains("标题"));
        assert!(text.contains("第一段"));
        assert!(text.contains("第二段"));
        // ⚠️ 脚本 / 样式 / 导航 / 页脚都不该出现。
        assert!(!text.contains("alert"));
        assert!(!text.contains("color: red"));
        assert!(!text.contains("导航"));
        assert!(!text.contains("页脚"));
    }

    #[test]
    fn decodes_entities() {
        let text = extract_main_text("<p>A &amp; B &lt;tag&gt; &#39;quoted&#39;</p>");
        assert!(text.contains("A & B"));
        assert!(text.contains("<tag>"));
        assert!(text.contains("'quoted'"));
    }

    #[test]
    fn decodes_numeric_entities() {
        // ⚠️ 实测 Bing 摘要里会出现这些。
        let text = decode_entities("3 天之前&ensp;&#0183;&ensp;Rust 很快");
        assert!(text.contains("3 天之前"));
        assert!(text.contains("Rust 很快"));
        assert!(!text.contains("&ensp;"));
        assert!(!text.contains("&#0183;"));
    }

    #[test]
    fn decodes_hex_entities() {
        assert_eq!(decode_entities("&#x1F600;"), "😀");
        assert_eq!(decode_entities("&#65;"), "A");
    }

    #[test]
    fn keeps_invalid_numeric_entities() {
        // ⚠️ 无效码点不能被吞掉。
        assert_eq!(decode_entities("&#xD800;"), "&#xD800;");
    }

    #[test]
    fn control_characters_become_spaces() {
        // ⚠️ NUL 会污染日志和 JSON —— 必须换成空格。
        assert_eq!(decode_entities("a&#0;b"), "a b");
    }

    #[test]
    fn inline_tags_get_separated() {
        // ⚠️ `<a>第一个</a><a>第二个</a>` 不能粘成「第一个第二个」。
        let text = extract_main_text("<p><a>第一个</a><a>第二个</a></p>");
        assert!(text.contains("第一个"));
        assert!(text.contains("第二个"));
        assert!(!text.contains("第一个第二个"));
    }

    #[test]
    fn url_encodes_spaces_and_reserves() {
        assert_eq!(url_encode("rust async"), "rust%20async");
        assert_eq!(url_encode("a-b_c.d~e"), "a-b_c.d~e");
        assert_eq!(url_encode("a&b"), "a%26b");
    }

    #[test]
    fn truncates_by_chars() {
        let text = truncate_chars("中文内容测试", 4);
        assert!(text.starts_with("中文内容"));
        assert!(text.contains("已截断"));
    }

    #[tokio::test]
    async fn mock_backend_returns_preset_results() {
        let backend = MockBackend::new(vec![SearchResult {
            title: "标题".to_string(),
            url: "https://example.com".to_string(),
            snippet: "摘要".to_string(),
        }]);

        let results = backend.search("query", 5).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "标题");
    }

    #[tokio::test]
    async fn mock_backend_can_fail() {
        let backend = MockBackend::new(Vec::new()).failing();

        let err = backend.search("query", 5).await.unwrap_err();

        assert_eq!(err.code, ErrorCode::UpstreamUnavailable);
        assert!(err.retryable());
    }

    #[tokio::test]
    async fn mock_backend_fetches_fixed_page() {
        let backend = MockBackend::new(Vec::new()).with_page("页面内容");

        let page = backend.fetch("https://example.com", 100).await.unwrap();

        assert_eq!(page.text, "页面内容");
    }
}
