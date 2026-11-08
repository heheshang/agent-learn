//! 装配层 —— 把各层组件连成一个可运行的流水线。
//!
//! ## 装配顺序
//!
//! ```text
//! 语料 ──> 分块 ──> 向量化 ──> Milvus upsert
//!                                │
//! 语料 ──> 分块 ──> BM25 索引 ────┤
//!                                │
//!           嵌入器 + 重排器 + 改写器 + 压缩器 + LLM
//!                                │
//!                                └──> SummaryPipeline
//! ```
//!
//! ## 为什么索引建在 Milvus 而不是本地文件
//!
//! 这是 05 → 10 的演进：
//!
//! - 05-rag：索引在**本地文件**（`index.faiss`），`save` / `load` 有意义
//! - 10-rag-milvus：索引在**远程服务**里，`save` / `load` 是 no-op
//!
//! 「重启服务要重新灌一遍数据」不是缺陷，而是生产向量库的常态 ——
//! 数据本来就在服务端，客户端只负责「确保它在」。
//!
//! ## ⚠️ collection 名的隔离
//!
//! 本项目和 10-rag-milvus 共用同一个 Milvus 实例，但 collection 名不同
//! （`rag_summary_chunks` vs `rag_chunks`）—— 否则两个项目的索引
//! 会互相覆盖，而且因为 embedding 模型一样，撞起来极难发现。

use agent_logging::{info, stage_guard};
use std::sync::Arc;

use rag_05::embed::{Embedder, LocalEmbedder, Model};
use rag_07::llm::{GenaiLlm, Llm, LlmOptions};
use rag_07::transform::{FallibleTransformer, MultiQueryTransformer, QueryTransformer};
use rag_08::reranker::{Reranker, make_reranker};
use rag_10::milvus_store::MilvusStore;
use rag_10::schema::records_from_chunks;

use crate::compress::ContextCompressor;
use crate::config::{AppConfig, model_dimension, model_name};
use crate::error::{RagError, RagResult};
use crate::pipeline::SummaryPipeline;
use crate::retriever::{HybridRetriever, build_bm25_index};

/// 索引构建结果。
#[derive(Debug, Clone)]
pub struct IndexReport {
    /// 灌进去多少条。
    pub chunks: usize,

    /// 服务端确认的条数。
    pub server_count: usize,

    /// collection 名。
    pub collection: String,

    /// 向量维度。
    pub dimension: usize,

    /// 嵌入模型名。
    pub embedding_model: String,

    /// 客户端 BM25 索引的文档数。
    pub bm25_docs: usize,

    /// 是否新建了 collection。
    pub created: bool,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

impl IndexReport {
    /// 一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "{} | {} 块（服务端 {}）| dim={} | BM25 {} 块 | {} | {}ms",
            self.collection,
            self.chunks,
            self.server_count,
            self.dimension,
            self.bm25_docs,
            if self.created { "新建" } else { "已存在" },
            self.elapsed_ms
        )
    }
}

/// 缓存目录（绝对路径）。
///
/// 与 05~11 保持一致：优先 `FASTEMBED_CACHE_DIR`，
/// 否则用 workspace 根目录下的 `.fastembed_cache`。
///
/// 绝对路径是必须的 —— 否则 `cargo run` 和 `cargo test`
/// 的工作目录不同，会重复下载模型。
pub fn cache_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("FASTEMBED_CACHE_DIR")
        && !dir.is_empty()
    {
        return std::path::PathBuf::from(dir);
    }

    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join(".fastembed_cache"))
        .unwrap_or_else(|| std::path::PathBuf::from(".fastembed_cache"))
}

/// 创建嵌入器。
pub fn make_embedder(model: Model) -> RagResult<LocalEmbedder> {
    LocalEmbedder::new(model, Some(cache_dir())).map_err(|e| RagError::Internal(e.to_string()))
}

/// 确保 Milvus 索引就绪：加载语料 → 向量化 → 写入服务端。
pub async fn ensure_index(config: &AppConfig) -> RagResult<IndexReport> {
    let start = std::time::Instant::now();

    let embedder = make_embedder(config.embedding_model)?;
    let dimension = model_dimension(config.embedding_model);

    let index_config = rag_05::pipeline::IndexConfig {
        data_dir: config.data_dir.clone(),
        splitter: config.splitter.clone(),
        ..Default::default()
    };

    let build = rag_05::pipeline::build_chunks(&embedder, &index_config)
        .map_err(|e| RagError::Internal(format!("构建索引失败: {e}")))?;

    let mut store = MilvusStore::connect(
        &config.milvus,
        dimension,
        model_name(config.embedding_model),
    )
    .await
    .map_err(|e| RagError::NotReady(format!("连接 Milvus 失败: {e}")))?;

    let created = store
        .ensure_collection(&config.milvus)
        .await
        .map_err(|e| RagError::Retrieval(format!("准备 collection 失败: {e}")))?;

    let records = records_from_chunks(&build.chunks);
    let inserted = store
        .upsert_records(&records)
        .await
        .map_err(|e| RagError::Retrieval(format!("写入 Milvus 失败: {e}")))?;

    let server_count = store
        .count()
        .await
        .map_err(|e| RagError::Retrieval(format!("统计条数失败: {e}")))?;

    // 客户端 BM25 索引（不落盘，每次启动重建 —— 和 08-rag 同一个理由）。
    let bm25 = build_bm25_index(&config.data_dir, &config.splitter)?;

    info!(
        collection = %config.milvus.collection,
        chunks = inserted as usize,
        server_count,
        bm25_docs = bm25.len(),
        created,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "索引就绪"
    );

    Ok(IndexReport {
        chunks: inserted as usize,
        server_count,
        collection: config.milvus.collection.clone(),
        dimension,
        embedding_model: model_name(config.embedding_model).to_string(),
        bm25_docs: bm25.len(),
        created,
        elapsed_ms: start.elapsed().as_millis() as u64,
    })
}

/// 装配完整的流水线。
///
/// 返回 `(流水线, 索引报告)` —— 索引报告给 `/` 端点展示用
/// （「索引：未建立」对运维没用）。
pub async fn bootstrap(config: &AppConfig) -> RagResult<(SummaryPipeline, IndexReport)> {
    let _guard = stage_guard("装配摘要式 RAG 流水线");
    let index_report = ensure_index(config).await?;
    tracing_report(&index_report);

    let embedder = make_embedder(config.embedding_model)?;

    let store = MilvusStore::connect(
        &config.milvus,
        model_dimension(config.embedding_model),
        model_name(config.embedding_model),
    )
    .await
    .map_err(|e| RagError::NotReady(format!("连接 Milvus 失败: {e}")))?;

    let bm25 = build_bm25_index(&config.data_dir, &config.splitter)?;

    let retriever = Arc::new(HybridRetriever::new(
        Arc::new(store),
        Arc::new(embedder) as Arc<dyn Embedder>,
        Arc::new(bm25),
    )) as Arc<dyn crate::retriever::Retriever>;

    // 重排器。
    let reranker: Option<Arc<dyn Reranker>> = if config.rerank.enabled {
        Some(
            make_reranker(
                config.rerank.backend,
                config.rerank.model,
                Some(cache_dir()),
                config.rerank.cohere_api_key.as_deref(),
                &config.rerank.cohere_model,
            )
            .map_err(|e| RagError::Internal(format!("创建重排器失败: {e}")))?,
        )
    } else {
        None
    };

    // LLM。
    let llm = Arc::new(GenaiLlm::new(
        &config.generation_model,
        LlmOptions {
            temperature: config.generation_temperature,
            max_tokens: config.generation_max_tokens,
            // ⚠️ 生成侧**保留思考** —— 摘要质量需要推理。
            disable_reasoning: false,
        },
    )) as Arc<dyn Llm>;

    // 查询改写器（多问法）。
    let transformer: Option<Arc<dyn QueryTransformer>> = if config.rewrite.enabled {
        let llm_for_rewrite = Arc::clone(&llm);
        Some(Arc::new(FallibleTransformer::new(Box::new(
            MultiQueryTransformer::new(llm_for_rewrite)
                .with_num_queries(config.rewrite.num_queries),
        ))))
    } else {
        None
    };

    let compressor = Arc::new(ContextCompressor::new(config.compress.clone())?);

    let pipeline = SummaryPipeline::new(
        retriever,
        reranker,
        compressor,
        transformer,
        llm,
        config.clone(),
    );

    Ok((pipeline, index_report))
}

/// 打印索引报告。
fn tracing_report(report: &IndexReport) {
    eprintln!("  索引        : {}", report.summary());
    info!(
        collection = %report.collection,
        chunks = report.chunks,
        server_count = report.server_count,
        bm25_docs = report.bm25_docs,
        created = report.created,
        "索引报告"
    );
}

/// 检查后端是否可达（健康检查）。
pub async fn check_health(config: &AppConfig) -> RagResult<String> {
    let store = MilvusStore::connect(
        &config.milvus,
        model_dimension(config.embedding_model),
        model_name(config.embedding_model),
    )
    .await
    .map_err(|e| RagError::NotReady(format!("连接 Milvus 失败: {e}")))?;

    let count = store
        .count()
        .await
        .map_err(|e| RagError::Retrieval(format!("统计条数失败: {e}")))?;

    Ok(format!(
        "milvus healthy collection={} count={}",
        config.milvus.collection, count
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_is_absolute() {
        assert!(cache_dir().is_absolute());
    }

    #[test]
    fn index_report_summary_mentions_collection() {
        let report = IndexReport {
            chunks: 10,
            server_count: 10,
            collection: "c".to_string(),
            dimension: 384,
            embedding_model: "e5".to_string(),
            bm25_docs: 10,
            created: true,
            elapsed_ms: 5,
        };
        assert!(report.summary().contains("c"));
        assert!(report.summary().contains("新建"));
    }
}
