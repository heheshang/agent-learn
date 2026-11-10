//! 测试替身 —— 零网络、零模型的确定性替身。
//!
//! ## 为什么要有这个模块
//!
//! 本仓库的硬约定：**单元测试必须零网络、零模型、确定性**。
//! 真实模型（fastembed ONNX）要下载 470MB~1.1GB 才能跑，
//! 塞进单元测试会让 `cargo test` 变成网络依赖。
//!
//! 所以：
//!
//! | 层 | 真实实现 | 测试替身 |
//! |---|---|---|
//! | 嵌入 | [`BatchEmbedder`] + fastembed | [`ScriptedEmbedder`] |
//! | 重排 | [`BatchReranker`] + fastembed | [`ScriptedReranker`] |
//!
//! 替身不是「假装成功」，而是**记录调用形状**的探针：
//! 它能告诉你「这次前向喂进去了几条」——
//! 这正是验证「批处理真的批了」的关键。
//!
//! ⚠️ 替身必须模拟真实行为的**全部影响源**，
//! 不只是你正在修的那一个（21 项目的教训）：
//! 本项目的替身模拟了「批量调用」和「单条调用」两种形状。

use std::sync::Mutex;
use std::time::Duration;

use rag_05::embed::EmbedError;
use rag_08::reranker::{RerankDocument, RerankedHit, Reranker};

use crate::error::{AppError, AppResult};

/// 确定性的嵌入替身。
///
/// ## 它做什么
///
/// - 向量的第 0 个分量 = 文本的字符数（便于断言「结果对回了原输入」）
/// - 记录每次调用喂进去多少条（便于断言「真的批了」）
/// - 可选地给每次调用加固定延迟（便于模拟耗时）
pub struct ScriptedEmbedder {
    dimension: usize,
    /// 每次 `embed_documents` 喂进去的条数。
    pub calls: Mutex<Vec<usize>>,
    /// 每次调用的额外延迟。
    pub delay: Duration,
}

impl ScriptedEmbedder {
    /// 创建（默认 4 维，无延迟）。
    pub fn new() -> Self {
        Self {
            dimension: 4,
            calls: Mutex::new(Vec::new()),
            delay: Duration::ZERO,
        }
    }

    /// 设置维度。
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension.max(1);
        self
    }

    /// 设置每次调用的延迟。
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// 取每次调用喂进去的条数。
    pub fn call_sizes(&self) -> Vec<usize> {
        self.calls.lock().unwrap().clone()
    }
}

impl Default for ScriptedEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl rag_05::embed::Embedder for ScriptedEmbedder {
    fn model_name(&self) -> &str {
        "scripted-embedder"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.calls.lock().unwrap().push(texts.len());
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0; self.dimension];
                v[0] = t.chars().count() as f32;
                v[1] = 1.0;
                v
            })
            .collect())
    }
}

/// 确定性的重排替身。
///
/// ## 它做什么
///
/// 按「文档里是否包含查询词」打分 —— 简单但**确定性**，
/// 而且能区分「哪条文档更相关」。
pub struct ScriptedReranker {
    /// 每次调用喂进去的候选数。
    pub calls: Mutex<Vec<usize>>,
    /// 每次调用的额外延迟。
    pub delay: Duration,
}

impl std::fmt::Debug for ScriptedReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedReranker")
            .field("delay", &self.delay)
            .finish()
    }
}

impl ScriptedReranker {
    /// 创建。
    pub fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            delay: Duration::ZERO,
        }
    }

    /// 设置延迟。
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// 直接算分（不经过 trait）。
    pub fn rank(&self, query: &str, documents: &[RerankDocument]) -> Vec<RerankedHit> {
        let mut hits: Vec<RerankedHit> = documents
            .iter()
            .enumerate()
            .map(|(index, doc)| {
                let score = if doc.text.contains(query) { 1.0 } else { 0.1 };
                RerankedHit {
                    rank: 0,
                    id: doc.id.clone(),
                    score,
                    original_index: index,
                    source: doc.source.clone(),
                    heading_path: doc.heading_path.clone(),
                    text: doc.text.clone(),
                }
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.original_index.cmp(&b.original_index))
        });

        for (i, hit) in hits.iter_mut().enumerate() {
            hit.rank = i;
        }
        hits
    }
}

impl Default for ScriptedReranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for ScriptedReranker {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [RerankDocument],
        top_n: usize,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = rag_08::error::RagResult<Vec<RerankedHit>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            self.calls.lock().unwrap().push(documents.len());
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            let mut hits = self.rank(query, documents);
            hits.truncate(top_n);
            Ok(hits)
        })
    }

    fn name(&self) -> &str {
        "scripted-reranker"
    }
}

/// fastembed 的模型缓存目录（**绝对路径**）。
///
/// ⚠️ 必须用绝对路径 —— fastembed 的默认缓存目录是相对路径，
/// 会跟随 CWD 变化，导致模型被重复下载。
pub fn cache_dir() -> std::path::PathBuf {
    rag_05::embed::cache_dir()
}

/// 造一批测试语料。
pub fn corpus(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("这是第 {i} 条测试文本。{}", "内容".repeat((i % 8) + 1)))
        .collect()
}

/// 造一批测试候选块。
pub fn documents(n: usize) -> Vec<RerankDocument> {
    (0..n)
        .map(|i| {
            RerankDocument::new(
                format!("doc-{i}"),
                "test.md",
                vec![format!("标题 {i}")],
                format!("这是第 {i} 条候选。{}", "内容".repeat((i % 8) + 1)),
            )
        })
        .collect()
}

/// 校验一条重排结果的分数序列是否一致。
///
/// 用于「批处理不改变结果」的不变量断言。
pub fn assert_same_scores(a: &[RerankedHit], b: &[RerankedHit]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("结果条数不同: {} vs {}", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.id != y.id {
            return Err(format!("第 {i} 条 ID 不同: {} vs {}", x.id, y.id));
        }
        if (x.score - y.score).abs() > 1e-6 {
            return Err(format!("第 {i} 条分数不同: {} vs {}", x.score, y.score));
        }
    }
    Ok(())
}

/// 把错误转成 `AppResult` 的辅助函数（测试里用）。
pub fn to_app_result<T>(v: Result<T, AppError>) -> AppResult<T> {
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_embedder_records_call_sizes() {
        let embedder = ScriptedEmbedder::new();
        let corpus = super::corpus(3);

        rag_05::embed::Embedder::embed_documents(&embedder, &corpus).unwrap();
        rag_05::embed::Embedder::embed_documents(&embedder, &corpus[..1]).unwrap();

        assert_eq!(embedder.call_sizes(), vec![3, 1]);
    }

    #[test]
    fn scripted_embedder_first_component_is_length() {
        let embedder = ScriptedEmbedder::new();
        let corpus = super::corpus(2);

        let vectors = rag_05::embed::Embedder::embed_documents(&embedder, &corpus).unwrap();
        assert_eq!(vectors[0][0] as usize, corpus[0].chars().count());
    }

    #[test]
    fn scripted_reranker_scores_matching_docs_higher() {
        let reranker = ScriptedReranker::new();
        let docs = vec![
            RerankDocument::new("a", "s.md", vec![], "这里有目标关键词"),
            RerankDocument::new("b", "s.md", vec![], "无关内容"),
        ];

        let hits = reranker.rank("目标关键词", &docs);
        assert_eq!(hits[0].id, "a");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn assert_same_scores_detects_difference() {
        let a = vec![RerankedHit {
            rank: 0,
            id: "x".into(),
            score: 1.0,
            original_index: 0,
            source: "s".into(),
            heading_path: vec![],
            text: "t".into(),
        }];
        let mut b = a.clone();
        b[0].score = 0.5;

        assert!(assert_same_scores(&a, &b).is_err());
        assert!(assert_same_scores(&a, &a).is_ok());
    }
}
