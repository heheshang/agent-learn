//! 向量化层（Embeddings）—— 把文本变成向量。
//!
//! 这是 RAG 流水线的第三步：
//!
//! ```text
//! 块 ──Embedder──> 向量
//! ```
//!
//! ## Embedding 是什么
//!
//! Embedding 模型把一段文本映射成固定长度的浮点向量（`f32`），
//! 使得**语义相近的文本，向量也相近**。检索时把问题也变成向量，
//! 用余弦相似度找最像的向量，就找到了语义相关的块。
//!
//! ## 三个关键概念
//!
//! ### 1. 归一化（normalization）
//!
//! 本项目所有向量都做 L2 归一化（模长 = 1）。此时：
//!
//! - 余弦相似度 == 内积（dot product）
//!
//! 这让 FAISS 可以用 `MetricType::InnerProduct` 索引 —— 比 L2 更快，
//! 而且余弦相似度天然就是「越大越相似」，不需要转换方向。
//!
//! ### 2. 前缀（prefix）
//!
//! 很多嵌入模型区分「查询」和「文档」两种角色：
//!
//! - E5 系：`query: ` / `passage: `
//! - BGE 系：查询加指令前缀，文档不加
//!
//! 用错前缀会明显拉低召回率。`Embedder` 把这件事封装进 trait，
//! 上层不需要知道用的是哪个模型。
//!
//! ### 3. 维度（dimension）
//!
//! 维度由模型决定，写死在索引里。**换了模型就必须重建索引** ——
//! 维度对不上，FAISS 会直接报错或静默返回垃圾结果。

use agent_logging::{debug, info, warn};
use std::path::PathBuf;

use rag_04::splitter::Chunk;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// 向量化失败。
#[derive(Debug, Error)]
pub enum EmbedError {
    /// 嵌入模型初始化失败（通常是没网、模型没下载成功）。
    #[error("嵌入模型初始化失败: {0}")]
    Init(String),

    /// 推理失败。
    #[error("嵌入推理失败: {0}")]
    Inference(String),

    /// 维度不一致 —— 换了模型但索引没重建。
    #[error("向量维度不一致: 期望 {expected}, 实际 {actual}")]
    DimensionMismatch {
        /// 索引期望的维度。
        expected: usize,
        /// 实际拿到的维度。
        actual: usize,
    },
}

/// 一条向量化的记录：块 + 它的向量 + 元数据。
///
/// 元数据在这里存，而不是等到检索时才去查原文 —— 因为
/// FAISS 只返回整数 ID，召回之后你还需要知道「这条结果来自哪个文件、
/// 哪个标题路径、原文是什么」。这些必须和向量存在一起。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    /// 稳定 ID：`{文件名}#{块序号}`，用于 upsert 和引用。
    pub id: String,

    /// 来源文件名。
    pub source: String,

    /// 块序号。
    pub index: usize,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 块正文。
    pub text: String,

    /// 该块的向量（L2 归一化后）。
    pub vector: Vec<f32>,
}

impl EmbeddedChunk {
    /// 一行摘要。
    #[allow(dead_code)]
    pub fn summary(&self) -> String {
        let head = if self.heading_path.is_empty() {
            "(无标题)".to_string()
        } else {
            self.heading_path.join(" > ")
        };
        let preview: String = self.text.chars().take(40).collect();
        format!(
            "{} [{}] dim={} | {}",
            self.id,
            head,
            self.vector.len(),
            preview.replace('\n', " ")
        )
    }
}

/// 嵌入器 trait。
///
/// 存在这个 trait 是为了：上层流水线只依赖 `Box<dyn Embedder>`，
/// 于是「本地 ONNX 模型」和「远程 API（OpenAI / DeepSeek embedding）」
/// 可以互换 —— 换模型不用改存储层和检索层。
pub trait Embedder: Send + Sync {
    /// 模型名（写进索引元数据，用于检测「换了模型没重建索引」）。
    fn model_name(&self) -> &str;

    /// 向量维度。
    fn dimension(&self) -> usize;

    /// 是否需要给「文档」加前缀（E5 系需要，BGE 系不需要）。
    fn needs_document_prefix(&self) -> bool {
        false
    }

    /// 是否需要给「查询」加前缀。
    fn needs_query_prefix(&self) -> bool {
        false
    }

    /// 批量嵌入「文档」（块的 `embedded_text`）。
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// 嵌入单个「查询」。
    ///
    /// 默认复用 `embed_documents`，模型有查询前缀时会被覆盖。
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut prefixed = text.to_string();
        if self.needs_query_prefix() {
            prefixed = format!("query: {prefixed}");
        }
        let mut vecs = self.embed_documents(std::slice::from_ref(&prefixed))?;
        vecs.pop()
            .ok_or_else(|| EmbedError::Inference("嵌入返回空结果".to_string()))
    }

    /// 把块批量向量化。
    fn embed_chunks(&self, chunks: &[Chunk]) -> Result<Vec<EmbeddedChunk>, EmbedError> {
        let texts: Vec<String> = chunks
            .iter()
            .map(|c| {
                if self.needs_document_prefix() {
                    format!("passage: {}", c.embedded_text)
                } else {
                    c.embedded_text.clone()
                }
            })
            .collect();

        debug!(
            model = %self.model_name(),
            chunk_count = chunks.len(),
            dimension = self.dimension(),
            document_prefix = self.needs_document_prefix(),
            "开始向量化"
        );

        let vectors = self.embed_documents(&texts)?;

        if vectors.len() != chunks.len() {
            warn!(
                input = chunks.len(),
                output = vectors.len(),
                "嵌入数量不匹配"
            );
            return Err(EmbedError::Inference(format!(
                "嵌入数量不匹配: 输入 {} 条, 输出 {} 条",
                chunks.len(),
                vectors.len()
            )));
        }

        let dim = self.dimension();
        let out = chunks
            .iter()
            .zip(vectors)
            .map(|(chunk, vector)| {
                if vector.len() != dim {
                    return Err(EmbedError::DimensionMismatch {
                        expected: dim,
                        actual: vector.len(),
                    });
                }
                Ok(EmbeddedChunk {
                    id: chunk_id(&chunk.source, chunk.index),
                    source: chunk.source.clone(),
                    index: chunk.index,
                    heading_path: chunk.heading_path.clone(),
                    text: chunk.text.clone(),
                    vector,
                })
            })
            .collect::<Result<Vec<_>, EmbedError>>()?;

        info!(
            model = %self.model_name(),
            chunk_count = out.len(),
            dimension = dim,
            "向量化完成"
        );

        Ok(out)
    }
}

/// 生成稳定 ID。
pub fn chunk_id(source: &str, index: usize) -> String {
    format!("{source}#{index}")
}

/// 本地嵌入器 —— 用 fastembed 跑 ONNX 版的 sentence-transformers 模型。
///
/// 对应教程里的 **Sentence Transformers**。
///
/// - 无需 Python、无需 API key
/// - 模型首次使用时自动从 HuggingFace 下载到本地缓存
/// - 输出已经过 L2 归一化，所以余弦相似度 == 内积
pub struct LocalEmbedder {
    model: fastembed::TextEmbedding,
    model_name: String,
    dimension: usize,
    query_prefix: Option<&'static str>,
    document_prefix: Option<&'static str>,
}

impl std::fmt::Debug for LocalEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalEmbedder")
            .field("model", &self.model_name)
            .field("dimension", &self.dimension)
            .finish()
    }
}

/// 可选的本地模型。
///
/// | 模型 | 维度 | 语言 | 说明 |
/// |---|---|---|---|
/// | `MiniLM` | 384 | 英文为主 | 最快，体积最小 |
/// | `MultilingualE5Small` | 384 | 多语言 | **本项目默认**，中英文都能用 |
/// | `BGESmallZH` | 512 | 中文 | 中文检索效果更好 |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// `sentence-transformers/all-MiniLM-L6-v2`
    MiniLM,
    /// `intfloat/multilingual-e5-small`
    MultilingualE5Small,
    /// `BAAI/bge-small-zh-v1.5`
    BGESmallZH,
}

/// 模型信息：维度、查询前缀、文档前缀。
fn model_spec(model: Model) -> (usize, Option<&'static str>, Option<&'static str>) {
    match model {
        // E5 系：查询和文档都要加前缀。
        Model::MultilingualE5Small => (384, Some("query: "), Some("passage: ")),
        // BGE 系：文档不加前缀（加了反而降分），查询可不加。
        Model::BGESmallZH => (512, None, None),
        // MiniLM 是纯对称模型，两边都不加。
        Model::MiniLM => (384, None, None),
    }
}

/// 把我们的模型枚举映射到 fastembed 的枚举。
fn to_fastembed(model: Model) -> fastembed::EmbeddingModel {
    match model {
        Model::MiniLM => fastembed::EmbeddingModel::AllMiniLML6V2,
        Model::MultilingualE5Small => fastembed::EmbeddingModel::MultilingualE5Small,
        Model::BGESmallZH => fastembed::EmbeddingModel::BGESmallZHV15,
    }
}

/// 模型的人类可读名。
fn display_name(model: Model) -> &'static str {
    match model {
        Model::MiniLM => "sentence-transformers/all-MiniLM-L6-v2",
        Model::MultilingualE5Small => "intfloat/multilingual-e5-small",
        Model::BGESmallZH => "BAAI/bge-small-zh-v1.5",
    }
}

impl LocalEmbedder {
    /// 创建嵌入器（会下载模型，首次较慢）。
    ///
    /// ## 缓存目录
    ///
    /// **务必传 [`cache_dir`] 返回的绝对路径**，不要传 `None`。
    ///
    /// fastembed 的默认缓存目录是**相对路径** `.fastembed_cache`，
    /// 相对于**当前工作目录**（CWD）。这在 Cargo workspace 里是个经典坑：
    ///
    /// ```text
    /// cargo run  -p rag-05   → CWD = workspace 根 → 缓存到 <root>/.fastembed_cache  ✅
    /// cd 05-rag && make search → CWD = 05-rag/    → 缓存到 05-rag/.fastembed_cache  ❌ 重新下载 470MB！
    /// ```
    ///
    /// 更糟的是：一旦本地缓存没命中而网络又不通（比如 HuggingFace 被墙），
    /// 就会直接报 `Failed to retrieve onnx/model.onnx` 而不是给出「缓存路径不对」的提示。
    pub fn new(model: Model, cache_dir: Option<PathBuf>) -> Result<Self, EmbedError> {
        info!(model = %display_name(model), "开始初始化嵌入模型（首次会下载模型）");

        let mut opts =
            fastembed::InitOptions::new(to_fastembed(model)).with_show_download_progress(true);

        if let Some(dir) = cache_dir {
            opts = opts.with_cache_dir(dir);
        }

        let model_handle = fastembed::TextEmbedding::try_new(opts).map_err(|e| {
            warn!(model = %display_name(model), error = %e, "嵌入模型初始化失败");
            EmbedError::Init(e.to_string())
        })?;

        let (dimension, query_prefix, document_prefix) = model_spec(model);

        info!(
            model = %display_name(model),
            dimension,
            query_prefix = query_prefix.unwrap_or("-"),
            document_prefix = document_prefix.unwrap_or("-"),
            "嵌入模型就绪"
        );

        Ok(Self {
            model: model_handle,
            model_name: display_name(model).to_string(),
            dimension,
            query_prefix,
            document_prefix,
        })
    }
}

impl Embedder for LocalEmbedder {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn needs_document_prefix(&self) -> bool {
        self.document_prefix.is_some()
    }

    fn needs_query_prefix(&self) -> bool {
        self.query_prefix.is_some()
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.model.embed(texts.to_vec(), None).map_err(|e| {
            warn!(count = texts.len(), error = %e, "嵌入推理失败");
            EmbedError::Inference(e.to_string())
        })
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut prefixed = text.to_string();
        if let Some(prefix) = self.query_prefix {
            prefixed = format!("{prefix}{prefixed}");
        }
        let mut vecs = self.embed_documents(std::slice::from_ref(&prefixed))?;
        vecs.pop()
            .ok_or_else(|| EmbedError::Inference("嵌入返回空结果".to_string()))
    }
}

/// L2 归一化。
///
/// 归一化后模长为 1，于是：
/// - 余弦相似度 == 内积
/// - 可以用 FAISS 的 `InnerProduct` 索引
#[allow(dead_code)]
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// 余弦相似度（不要求输入已归一化）。
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 0.0;
    }
    dot / (na * nb)
}

/// 嵌入模型的缓存目录（**绝对路径**）。
///
/// 1. `FASTEMBED_CACHE_DIR` 环境变量（用户显式指定，最高优先级）
/// 2. workspace 根目录下的 `.fastembed_cache`
///
/// 为什么必须是绝对路径：fastembed 的默认缓存目录是**相对路径** `.fastembed_cache`，
/// 相对 CWD 解析。在 workspace 里 CWD 会随运行方式变化
/// （`cargo run -p rag-05` 在根目录，`cd 05-rag && make search` 在子目录），
/// 一旦解析到另一个目录，模型缓存就miss，会重新下载 470MB；
/// 网络不通时更是直接报 `Failed to retrieve onnx/model.onnx`。
///
/// 与 06/07/08/10/12 的 `cache_dir()` 保持一致。
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("FASTEMBED_CACHE_DIR")
        && !dir.is_empty()
    {
        return PathBuf::from(dir);
    }

    // `CARGO_MANIFEST_DIR` = <workspace>/05-rag，往上一级就是 workspace 根。
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join(".fastembed_cache"))
        .unwrap_or_else(|| PathBuf::from(".fastembed_cache"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_makes_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm was {norm}");
    }

    #[test]
    fn l2_normalize_handles_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_opposite_vectors_is_negative_one() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn chunk_id_is_stable() {
        assert_eq!(chunk_id("a.md", 3), "a.md#3");
    }

    #[test]
    fn cache_dir_is_absolute() {
        // 必须是绝对路径，否则从不同 CWD 运行会解析到不同的缓存目录。
        assert!(cache_dir().is_absolute());
    }
}
