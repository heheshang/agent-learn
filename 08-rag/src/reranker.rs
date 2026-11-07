//! 重排（Rerank）—— 用 cross-encoder 给候选集做精排。
//!
//! ## 为什么召回之后还要重排
//!
//! 向量检索用的是**双编码器（bi-encoder）**：问题和文档**分别**编码，
//! 编码时彼此看不见，只能比向量距离。
//!
//! ```text
//! 问题 ──encode──> q        d ──encode──> 文档
//!                        ↘ 比距离 ↙
//! ```
//!
//! 这带来两个后果：
//!
//! 1. **快** —— 文档向量可以离线算好，查询时只做一次向量搜索
//! 2. **粗** —— 问题和文档之间**没有任何交互**，模型无法判断
//!    「这个词在这个问题语境下是什么意思」
//!
//! 重排模型用的是 **cross-encoder**：把问题和文档**拼在一起**送进模型。
//!
//! ```text
//! 问题 + 文档 ──encode──> 相关性分数
//! ```
//!
//! 它能看到交互，所以更准 —— 但每个候选都要跑一次模型，慢得多。
//!
//! 所以标准做法是**两阶段**：
//!
//! ```text
//! 召回（快、粗）──> top-20~50 ──> 重排（慢、精）──> top-3~5
//! ```
//!
//! ## 教程里的 Cohere Rerank
//!
//! [LlamaIndex 的 Cohere Rerank 例子](https://developers.llamaindex.ai/python/examples/node_postprocessor/coherererank/)
//! 演示的正是这个流程：
//!
//! ```python
//! query_engine = index.as_query_engine(
//!     similarity_top_k=10,          # ← 先召回 10 条
//!     node_postprocessors=[cohere_rerank],  # ← 再精排成 2 条
//! )
//! ```
//!
//! 教程里那个例子特别有说服力：**只取 top-2 时，答案是幻觉的**（
//! 召回了不相关的段落，模型硬编了一个答案）；
//! 而 **top-10 召回 + rerank 到 top-2**，答案才正确。
//!
//! 这正好说明：**召回阶段的排序是不够准的，重排才是精度的来源**。
//!
//! ## 本项目的两个后端
//!
//! | 后端 | 实现 | 需要 key | 说明 |
//! |---|---|---|---|
//! | [`LocalCrossEncoder`] | fastembed 的 ONNX cross-encoder | ❌ | 默认。`bge-reranker-base` 中英双语 |
//! | [`CohereReranker`] | Cohere `/v1/rerank` HTTP API | ✅ | 对齐教程 |
//!
//! 两者实现同一个 [`Reranker`] trait，所以上层完全不感知差异。

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{RagError, RagResult};

/// 一条重排结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankedHit {
    /// 重排后的排名（0 = 最相关）。
    pub rank: usize,

    /// 块 ID。
    pub id: String,

    /// 重排分数（越大越相关）。
    ///
    /// ⚠️ **不同 reranker 的分数不可比**：
    ///
    /// - cross-encoder 输出的是 **logit**（可以是任意实数，正负都有）
    /// - Cohere 输出的是 **relevance_score**（`[0, 1]` 区间的概率）
    ///
    /// 所以这个分数只能用于**同一次重排内部**排序，
    /// 不能拿来做阈值（跨后端阈值不一样），也不能和向量相似度比大小。
    pub score: f32,

    /// 原始候选里的下标（用于回溯原始块）。
    pub original_index: usize,

    /// 来源文件名。
    pub source: String,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 块正文。
    pub text: String,
}

impl RerankedHit {
    /// 引用标记（1 开始）。
    pub fn citation(&self) -> String {
        format!("[{}]", self.rank + 1)
    }

    /// 渲染成上下文片段（格式和 06/07 保持一致）。
    pub fn to_context(&self) -> String {
        let head = if self.heading_path.is_empty() {
            String::new()
        } else {
            format!("# {}\n", self.heading_path.join(" > "))
        };
        format!("[{}] {}{}", self.rank + 1, head, self.text)
    }
}

/// 重排器 trait。
///
/// 之所以要抽成 trait：本地模型和远程 API 的**失败模式完全不同**
/// （本地是「模型没下载」，远程是「key 无效 / 网络超时」），
/// 而且测试时需要能替换成**确定性的替身**。
///
/// `Debug` 是超 trait —— 这样 `Arc<dyn Reranker>` 本身也是 `Debug`，
/// 调试时可以直接打印整个流水线状态。
pub trait Reranker: Send + Sync + std::fmt::Debug {
    /// 重排：给定查询和候选块，返回按相关度降序排列的结果。
    ///
    /// # 参数
    ///
    /// - `query`：用户查询
    /// - `documents`：候选块（按召回顺序）
    /// - `top_n`：保留多少条
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [RerankDocument],
        top_n: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RagResult<Vec<RerankedHit>>> + Send + 'a>>;

    /// 重排器的名字（用于日志和响应体）。
    fn name(&self) -> &str;
}

/// 送进 reranker 的候选块。
///
/// 单独抽出来是因为 reranker **只需要文本** ——
/// ID、来源这些元数据在重排**之后**才用得上。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankDocument {
    /// 块 ID。
    pub id: String,
    /// 来源文件名。
    pub source: String,
    /// 标题路径。
    pub heading_path: Vec<String>,
    /// 块正文。
    pub text: String,
}

impl RerankDocument {
    /// 构造候选块。
    pub fn new(
        id: impl Into<String>,
        source: impl Into<String>,
        heading_path: Vec<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            source: source.into(),
            heading_path,
            text: text.into(),
        }
    }
}

/// 本地 cross-encoder —— 用 fastembed 的 ONNX reranker。
///
/// 默认模型 `BAAI/bge-reranker-base`：中英双语，约 1.1GB，首次使用自动下载。
///
/// ## 为什么默认用它而不是 Cohere
///
/// 1. **不需要 API key** —— `cargo run` 就能跑通全流程
/// 2. **没有网络延迟和限流** —— 可以放心地在对比实验里反复调参
/// 3. **离线可用**
///
/// 但如果你要复现教程里的效果，把 `RERANK_BACKEND=cohere` 打开就行。
pub struct LocalCrossEncoder {
    /// 模型句柄放在 `Arc` 里。
    ///
    /// 为什么需要 `Arc`：重排推理是 **CPU 密集的同步调用**，
    /// 必须放到 `spawn_blocking` 的阻塞线程池里跑。
    /// 而 `spawn_blocking` 要求闭包是 `'static` —— 不能借用 `&self`。
    ///
    /// 所以这里克隆一份 `Arc` 塞进闭包，而不是借用 `self`。
    model: Arc<fastembed::TextRerank>,
    model_name: String,
}

impl std::fmt::Debug for LocalCrossEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalCrossEncoder")
            .field("model", &self.model_name)
            .finish()
    }
}

impl LocalCrossEncoder {
    /// 创建重排器（会下载模型，首次较慢）。
    pub fn new(model: RerankerModel, cache_dir: Option<std::path::PathBuf>) -> RagResult<Self> {
        let mut opts = fastembed::RerankInitOptions::new(model.to_fastembed())
            .with_show_download_progress(true);

        if let Some(dir) = cache_dir {
            opts = opts.with_cache_dir(dir);
        }

        let handle = fastembed::TextRerank::try_new(opts)
            .map_err(|e| RagError::Rerank(format!("重排模型初始化失败（首次使用需要联网下载模型）: {e}")))?;

        Ok(Self {
            model: Arc::new(handle),
            model_name: model.display_name().to_string(),
        })
    }
}

impl Reranker for LocalCrossEncoder {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [RerankDocument],
        top_n: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RagResult<Vec<RerankedHit>>> + Send + 'a>>
    {
        Box::pin(async move {
            if documents.is_empty() || top_n == 0 {
                return Ok(Vec::new());
            }

            // 送给模型的文本 = 标题路径 + 正文。
            //
            // ⚠️ 带上标题路径很重要：同一个词在「数据库索引」和
            // 「向量索引」章节里含义完全不同，cross-encoder 能看到
            // 标题就能判断得更准。
            let texts: Vec<String> = documents
                .iter()
                .map(|d| {
                    if d.heading_path.is_empty() {
                        d.text.clone()
                    } else {
                        format!("# {}\n{}", d.heading_path.join(" > "), d.text)
                    }
                })
                .collect();

            // ONNX 推理是 CPU 密集的同步调用 —— 必须放到阻塞线程池，
            // 否则会卡住 tokio 的工作线程。
            //
            // ⚠️ 闭包必须 `'static`（`spawn_blocking` 的要求），
            // 所以这里克隆 `Arc` 而不是借用 `&self`。
            let model = Arc::clone(&self.model);
            let query = query.to_string();
            let texts = texts.clone();

            let result = tokio::task::spawn_blocking(move || {
                model
                    .rerank(query, texts, false, None)
                    .map_err(|e| RagError::Rerank(format!("重排推理失败: {e}")))
            })
            .await
            .map_err(|e| RagError::Internal(format!("重排任务失败: {e}")))??;

            Ok(take_top(result, documents, top_n))
        })
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}

/// Cohere Rerank —— 调远程 API。
///
/// 对应教程里的 `CohereRerank`。
///
/// ## 请求格式
///
/// ```http
/// POST https://api.cohere.com/v1/rerank
/// Authorization: Bearer $COHERE_API_KEY
/// Content-Type: application/json
///
/// {
///   "model": "rerank-multilingual-v3.0",
///   "query": "...",
///   "documents": ["...", "..."],
///   "top_n": 3
/// }
/// ```
///
/// ## 响应格式
///
/// ```json
/// { "results": [ { "index": 1, "relevance_score": 0.98 }, ... ] }
/// ```
///
/// `index` 是**原始候选数组里的下标** —— 所以响应要按 `index` 映射回去。
pub struct CohereReranker {
    client: reqwest::Client,
    api_key: String,
    model: String,
    endpoint: String,
}

impl std::fmt::Debug for CohereReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereReranker")
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

/// Cohere 的重排请求体。
#[derive(Debug, Serialize)]
struct CohereRerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: Vec<&'a str>,
    top_n: usize,
}

/// Cohere 的重排响应体。
#[derive(Debug, Deserialize)]
struct CohereRerankResponse {
    results: Vec<CohereRerankItem>,
}

/// 响应里的一条结果。
#[derive(Debug, Deserialize)]
struct CohereRerankItem {
    index: usize,
    relevance_score: f32,
}

impl CohereReranker {
    /// 创建 Cohere 重排器。
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            endpoint: "https://api.cohere.com/v1/rerank".to_string(),
        }
    }

    /// 覆盖 API 地址（自建代理 / 测试用）。
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

impl Reranker for CohereReranker {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [RerankDocument],
        top_n: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RagResult<Vec<RerankedHit>>> + Send + 'a>>
    {
        Box::pin(async move {
            if documents.is_empty() || top_n == 0 {
                return Ok(Vec::new());
            }

            // ⚠️ 这里只传**正文**给 Cohere —— 和 BM25 索引的取舍一致：
            // 标题信息通过 `heading_path` 字段单独保留，在结果里回填。
            //
            // （本地 cross-encoder 传的是「标题 + 正文」，
            //  因为它的 tokenizer 截断到 512 token，标题能帮它定位语境。）
            let texts: Vec<&str> = documents.iter().map(|d| d.text.as_str()).collect();

            let payload = CohereRerankRequest {
                model: &self.model,
                query,
                documents: texts,
                top_n: top_n.min(documents.len()),
            };

            let response = self
                .client
                .post(&self.endpoint)
                .bearer_auth(&self.api_key)
                .json(&payload)
                .send()
                .await
                .map_err(|e| RagError::Rerank(format!("调用 Cohere rerank 失败: {e}")))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(RagError::Rerank(format!(
                    "Cohere rerank 返回错误状态 {status}: {}",
                    body.chars().take(200).collect::<String>()
                )));
            }

            let parsed: CohereRerankResponse = response
                .json()
                .await
                .map_err(|e| RagError::Rerank(format!("解析 Cohere rerank 响应失败: {e}")))?;

            let mut hits: Vec<RerankedHit> = parsed
                .results
                .into_iter()
                .filter_map(|item| {
                    let doc = documents.get(item.index)?;
                    Some(RerankedHit {
                        rank: 0,
                        id: doc.id.clone(),
                        score: item.relevance_score,
                        original_index: item.index,
                        source: doc.source.clone(),
                        heading_path: doc.heading_path.clone(),
                        text: doc.text.clone(),
                    })
                })
                .collect();

            // 按分数降序 —— Cohere 理论上已经排好了，但别依赖它的顺序。
            hits.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.original_index.cmp(&b.original_index))
            });
            hits.truncate(top_n);

            for (i, hit) in hits.iter_mut().enumerate() {
                hit.rank = i;
            }

            Ok(hits)
        })
    }

    fn name(&self) -> &str {
        &self.model
    }
}

/// 把 fastembed 的重排结果转换成统一的 [`RerankedHit`]。
fn take_top(
    results: Vec<fastembed::RerankResult>,
    documents: &[RerankDocument],
    top_n: usize,
) -> Vec<RerankedHit> {
    let mut hits: Vec<RerankedHit> = results
        .into_iter()
        .filter_map(|r| {
            let doc = documents.get(r.index)?;
            Some(RerankedHit {
                rank: 0,
                id: doc.id.clone(),
                score: r.score,
                original_index: r.index,
                source: doc.source.clone(),
                heading_path: doc.heading_path.clone(),
                text: doc.text.clone(),
            })
        })
        .collect();

    hits.truncate(top_n);
    for (i, hit) in hits.iter_mut().enumerate() {
        hit.rank = i;
    }
    hits
}

/// 重排模型选择。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankerModel {
    /// `BAAI/bge-reranker-base` —— 中英双语，**默认**。
    BgeRerankerBase,
    /// `rozgo/bge-reranker-v2-m3` —— 多语言，更大更准。
    BgeRerankerV2M3,
    /// `jinaai/jina-reranker-v1-turbo-en` —— 英文。
    JinaRerankerV1TurboEn,
}

impl RerankerModel {
    /// 转换成 fastembed 的模型枚举。
    pub fn to_fastembed(self) -> fastembed::RerankerModel {
        match self {
            RerankerModel::BgeRerankerBase => fastembed::RerankerModel::BGERerankerBase,
            RerankerModel::BgeRerankerV2M3 => fastembed::RerankerModel::BGERerankerV2M3,
            RerankerModel::JinaRerankerV1TurboEn => {
                fastembed::RerankerModel::JINARerankerV1TurboEn
            }
        }
    }

    /// 人类可读名。
    pub fn display_name(self) -> &'static str {
        match self {
            RerankerModel::BgeRerankerBase => "BAAI/bge-reranker-base",
            RerankerModel::BgeRerankerV2M3 => "rozgo/bge-reranker-v2-m3",
            RerankerModel::JinaRerankerV1TurboEn => "jinaai/jina-reranker-v1-turbo-en",
        }
    }

    /// 从环境变量解析。
    pub fn from_env_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "bge" | "bge_reranker_base" | "base" => Some(RerankerModel::BgeRerankerBase),
            "m3" | "bge_reranker_v2_m3" | "v2_m3" => Some(RerankerModel::BgeRerankerV2M3),
            "jina" | "jina_turbo" => Some(RerankerModel::JinaRerankerV1TurboEn),
            _ => None,
        }
    }
}

/// 重排后端选择。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankBackend {
    /// 本地 cross-encoder（fastembed）—— **默认**，不需要 key。
    Local,
    /// Cohere 远程 API —— 对齐教程。
    Cohere,
}

impl RerankBackend {
    /// 从环境变量解析：`local` / `cohere`。
    pub fn from_env_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "local" | "cross_encoder" | "cross-encoder" => Some(RerankBackend::Local),
            "cohere" => Some(RerankBackend::Cohere),
            _ => None,
        }
    }

    /// 名字。
    pub fn name(self) -> &'static str {
        match self {
            RerankBackend::Local => "local",
            RerankBackend::Cohere => "cohere",
        }
    }
}

/// 按后端创建重排器。
///
/// `cohere_api_key` 为空时，指定 `RerankBackend::Cohere` 会报错 ——
/// 明确失败好过运行到一半才挂。
pub fn make_reranker(
    backend: RerankBackend,
    model: RerankerModel,
    cache_dir: Option<std::path::PathBuf>,
    cohere_api_key: Option<&str>,
    cohere_model: &str,
) -> RagResult<Arc<dyn Reranker>> {
    match backend {
        RerankBackend::Local => Ok(Arc::new(LocalCrossEncoder::new(model, cache_dir)?)),
        RerankBackend::Cohere => {
            let key = cohere_api_key
                .filter(|k| !k.trim().is_empty())
                .ok_or_else(|| {
                    RagError::BadRequest(
                        "使用 Cohere rerank 需要设置 COHERE_API_KEY（或 RERANK_BACKEND=local）"
                            .to_string(),
                    )
                })?;
            Ok(Arc::new(CohereReranker::new(key, cohere_model)))
        }
    }
}

/// 测试用的确定性重排器 —— 按关键词匹配打分。
///
/// 存在的意义和 07 的 `ScriptedLlm` 一样：
/// 让「重排之后下游逻辑对不对」能被**确定性地**测试，
/// 而不依赖真实模型的输出。
#[derive(Debug)]
pub struct KeywordReranker {
    keyword: String,
}

impl KeywordReranker {
    /// 创建一个重排器：包含 `keyword` 的文档得高分。
    pub fn new(keyword: impl Into<String>) -> Self {
        Self {
            keyword: keyword.into().to_lowercase(),
        }
    }
}

impl Reranker for KeywordReranker {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [RerankDocument],
        top_n: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RagResult<Vec<RerankedHit>>> + Send + 'a>>
    {
        Box::pin(async move {
            let mut hits: Vec<RerankedHit> = documents
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    // 打分：查询词和关键词都出现在文档里 → 分数高。
                    let text = d.text.to_lowercase();
                    let mut score = 0.0f32;
                    if text.contains(&self.keyword) {
                        score += 2.0;
                    }
                    if query.to_lowercase().contains(&self.keyword)
                        && text.contains(&self.keyword)
                    {
                        score += 1.0;
                    }
                    RerankedHit {
                        rank: 0,
                        id: d.id.clone(),
                        score,
                        original_index: i,
                        source: d.source.clone(),
                        heading_path: d.heading_path.clone(),
                        text: d.text.clone(),
                    }
                })
                .collect();

            hits.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.original_index.cmp(&b.original_index))
            });
            hits.truncate(top_n);

            for (i, hit) in hits.iter_mut().enumerate() {
                hit.rank = i;
            }
            Ok(hits)
        })
    }

    fn name(&self) -> &str {
        "keyword"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, text: &str) -> RerankDocument {
        RerankDocument::new(id, "test.md", vec!["标题".to_string()], text)
    }

    #[tokio::test]
    async fn keyword_reranker_promotes_matching_documents() {
        let reranker = KeywordReranker::new("光合作用");
        let docs = vec![
            doc("a", "数据库索引的查找复杂度是 O(log n)"),
            doc("b", "光合作用分为光反应和暗反应"),
        ];

        let hits = reranker.rerank("光合作用", &docs, 2).await.unwrap();

        assert_eq!(hits[0].id, "b");
        assert!(hits[0].score > hits[1].score);
    }

    #[tokio::test]
    async fn keyword_reranker_respects_top_n() {
        let reranker = KeywordReranker::new("x");
        let docs = vec![doc("a", "x"), doc("b", "x"), doc("c", "x")];

        let hits = reranker.rerank("x", &docs, 2).await.unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[tokio::test]
    async fn empty_documents_produce_empty_result() {
        let reranker = KeywordReranker::new("x");
        let hits = reranker.rerank("query", &[], 5).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn ranks_are_sequential() {
        let reranker = KeywordReranker::new("x");
        let docs = vec![doc("a", "x"), doc("b", "y")];

        let hits = reranker.rerank("x", &docs, 5).await.unwrap();
        assert_eq!(hits[0].rank, 0);
        assert_eq!(hits[1].rank, 1);
        assert_eq!(hits[0].citation(), "[1]");
    }

    #[tokio::test]
    async fn original_index_is_preserved() {
        let reranker = KeywordReranker::new("target");
        let docs = vec![doc("a", "no"), doc("b", "no"), doc("c", "target")];

        let hits = reranker.rerank("target", &docs, 5).await.unwrap();
        assert_eq!(hits[0].id, "c");
        assert_eq!(hits[0].original_index, 2);
    }

    #[tokio::test]
    async fn context_format_matches_previous_projects() {
        let hit = RerankedHit {
            rank: 0,
            id: "a.md#0".to_string(),
            score: 0.5,
            original_index: 0,
            source: "a.md".to_string(),
            heading_path: vec!["标题".to_string()],
            text: "正文".to_string(),
        };

        let ctx = hit.to_context();
        assert!(ctx.contains("[1]"));
        assert!(ctx.contains("# 标题"));
        assert!(ctx.contains("正文"));
    }

    #[test]
    fn parses_backend_names() {
        assert_eq!(RerankBackend::from_env_value("local"), Some(RerankBackend::Local));
        assert_eq!(RerankBackend::from_env_value("Cohere"), Some(RerankBackend::Cohere));
        assert_eq!(RerankBackend::from_env_value("unknown"), None);
    }

    #[test]
    fn parses_model_names() {
        assert_eq!(
            RerankerModel::from_env_value("bge"),
            Some(RerankerModel::BgeRerankerBase)
        );
        assert_eq!(
            RerankerModel::from_env_value("m3"),
            Some(RerankerModel::BgeRerankerV2M3)
        );
        assert_eq!(RerankerModel::from_env_value("unknown"), None);
    }

    #[test]
    fn cohere_backend_requires_api_key() {
        let err = make_reranker(
            RerankBackend::Cohere,
            RerankerModel::BgeRerankerBase,
            None,
            None,
            "rerank-multilingual-v3.0",
        )
        .unwrap_err();

        assert_eq!(err.status_code(), 400);
        assert!(err.to_string().contains("COHERE_API_KEY"));
    }
}
