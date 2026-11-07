//! Chroma 存储 —— 带元数据过滤的向量数据库。
//!
//! 对应教程里的 **Chroma**。
//!
//! ## Chroma 是什么
//!
//! 一个「开箱即用」的向量数据库：HTTP 接口 + 持久化目录，
//! 支持向量检索 + 元数据过滤 + 集合（collection）管理。
//! Python 生态用得最多（LangChain / LlamaIndex 默认集成它）。
//!
//! ## 和 FAISS 的本质差异
//!
//! | | FAISS | Chroma |
//! |---|---|---|
//! | 定位 | **库** —— 链接进你的进程 | **服务** —— 独立进程，HTTP 访问 |
//! | 元数据 | 在 Rust 侧，要自己 join | **在数据库里**，能服务端过滤 |
//! | 过滤 | 检索回来自己过滤（浪费） | 检索时就在服务端过滤（省 I/O） |
//! | 并发 | 进程内，天然独占 | 多客户端共享 |
//!
//! **元数据过滤是 Chroma 相对 FAISS 最大的价值**。
//! 比如「只搜 2024 年的文档」：
//!
//! - FAISS：先取 top-1000，再在 Rust 里过滤掉不满足条件的 —— 可能剩 3 条
//! - Chroma：`where` 条件直接下推到存储层 —— 只扫满足条件的向量
//!
//! ## 距离语义必须对齐
//!
//! 这是接入 Chroma 时最容易踩的坑：
//!
//! | 空间 | Chroma 返回的 `distances` | 方向 |
//! |---|---|---|
//! | `l2`（默认） | 平方 L2 距离 | 越小越相似 |
//! | `ip` | 内积 | 越大越相似 |
//! | `cosine` | `1 - 余弦相似度` | 越小越相似 |
//!
//! FAISS 用的是 `InnerProduct`（越大越相似）。
//! 本项目把 Chroma 集合建成 `cosine` 空间，再把距离换算回
//! 「越大越相似」的相似度，这样两个 store 的 `SearchHit.score`
//! 语义完全一致，可以直接对比。
//!
//! ## 需要的外部服务
//!
//! Chroma 服务端是 Python 实现的，本项目通过 HTTP 访问它：
//!
//! ```bash
//! uv venv .venv && uv pip install --python .venv/bin/python chromadb
//! .venv/bin/chroma run --path ./chroma-data --port 8765
//! ```
//!
//! 服务没起时，`chroma` 相关的 demo 会优雅降级（打印提示后跳过），
//! 不影响其它 demo 运行。

use std::path::Path;

use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{ChromaCollection, CollectionEntries, GetOptions, QueryOptions};
use serde_json::json;

use crate::embed::EmbeddedChunk;
use crate::store::{SearchHit, StoreError, VectorStore};

/// Chroma 向量存储。
pub struct ChromaStore {
    /// Chroma 客户端。
    client: ChromaClient,
    /// 集合句柄。
    collection: ChromaCollection,
    /// 集合名。
    collection_name: String,
    /// 模型名。
    model_name: String,
    /// 向量维度。
    dimension: usize,
    /// 本地缓存的条数（Chroma 的真实条数要 await 才能拿到）。
    cached_count: usize,
}

impl std::fmt::Debug for ChromaStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChromaStore")
            .field("collection", &self.collection_name)
            .field("model", &self.model_name)
            .field("dimension", &self.dimension)
            .field("count", &self.cached_count)
            .finish()
    }
}

impl ChromaStore {
    /// 连接（或创建）一个 Chroma 集合。
    ///
    /// `endpoint` 形如 `http://localhost:8765`。
    ///
    /// 集合会被建成 `cosine` 空间 —— 理由见模块文档的「距离语义必须对齐」。
    pub async fn connect(
        endpoint: &str,
        collection_name: &str,
        dimension: usize,
        model_name: &str,
    ) -> Result<Self, StoreError> {
        let client = ChromaClient::new(ChromaClientOptions {
            url: Some(endpoint.to_string()),
            ..Default::default()
        })
        .await
        .map_err(|e| StoreError::Chroma(e.to_string()))?;

        // 先用 HTTP 把集合建成 cosine 空间。
        // `chromadb` crate 的 `create_collection` 还不支持 `configuration`
        // 参数，没法指定距离空间，只能自己发请求。
        ensure_collection_space(endpoint, collection_name, "cosine").await?;

        // 再用 crate 的正规 API 拿集合句柄 ——
        // 这样 `collection.api` 才绑定了客户端，后续 add / query 才能工作。
        let collection = client
            .get_or_create_collection(collection_name, None)
            .await
            .map_err(|e| StoreError::Chroma(e.to_string()))?;

        // 用服务端返回的真实条数初始化缓存。
        let count = collection
            .count()
            .await
            .map_err(|e| StoreError::Chroma(e.to_string()))?;

        Ok(Self {
            client,
            collection,
            collection_name: collection_name.to_string(),
            model_name: model_name.to_string(),
            dimension,
            cached_count: count,
        })
    }

    /// 集合名。
    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }

    /// 服务端返回的真实条数。
    pub async fn count(&self) -> Result<usize, StoreError> {
        self.collection
            .count()
            .await
            .map_err(|e| StoreError::Chroma(e.to_string()))
    }

    /// 带元数据过滤的检索。
    ///
    /// `filter` 是 Chroma 的 where 表达式，例如：
    ///
    /// ```json
    /// { "source": { "$eq": "rag_guide.md" } }
    /// ```
    ///
    /// 过滤在**服务端**完成 —— 这是 Chroma 相对 FAISS 的核心优势。
    pub async fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<SearchHit>, StoreError> {
        if query.len() != self.dimension {
            return Err(StoreError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let result = self
            .collection
            .query(
                QueryOptions {
                    query_embeddings: Some(vec![query.to_vec()]),
                    n_results: Some(k),
                    where_metadata: filter,
                    include: Some(vec!["metadatas", "documents", "distances"]),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| StoreError::Chroma(e.to_string()))?;

        Ok(hits_from_query_result(&result))
    }

    /// 删除整个集合。
    pub async fn drop_collection(&self) -> Result<(), StoreError> {
        self.client
            .delete_collection(&self.collection_name)
            .await
            .map_err(|e| StoreError::Chroma(e.to_string()))
    }
}

/// 确保集合存在，并且使用指定的距离空间。
///
/// 直接打 Chroma 的 HTTP API —— `chromadb` crate 的 `create_collection`
/// 还不支持 `configuration` 参数。
///
/// `get_or_create: true` 是幂等的：集合已存在时不会覆盖已有配置
/// （实测确认），所以可以放心重复调用。
async fn ensure_collection_space(
    endpoint: &str,
    name: &str,
    space: &str,
) -> Result<(), StoreError> {
    // 解析 tenant —— 默认是 `default_tenant`，
    // 但 Chroma 允许配置成别的，所以先问一次。
    let tenant = resolve_tenant(endpoint).await?;

    let url = format!("{endpoint}/api/v2/tenants/{tenant}/databases/default_database/collections");

    let body = json!({
        "name": name,
        "get_or_create": true,
        "configuration": {
            "hnsw": { "space": space }
        }
    });

    let resp = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| StoreError::Chroma(format!("无法连接 Chroma 服务 {endpoint}: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(StoreError::Chroma(format!(
            "创建集合 `{name}` 失败（HTTP {status}）: {text}"
        )));
    }

    Ok(())
}

/// 问 Chroma 服务当前的 tenant 是什么。
async fn resolve_tenant(endpoint: &str) -> Result<String, StoreError> {
    let resp = reqwest::get(format!("{endpoint}/api/v2/auth/identity"))
        .await
        .map_err(|e| StoreError::Chroma(format!("无法连接 Chroma 服务 {endpoint}: {e}")))?;

    if !resp.status().is_success() {
        // identity 接口不可用时退回默认值。
        return Ok("default_tenant".to_string());
    }

    let identity: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| StoreError::Chroma(format!("解析 identity 失败: {e}")))?;

    Ok(identity
        .get("tenant")
        .and_then(|v| v.as_str())
        .filter(|t| *t != "*")
        .unwrap_or("default_tenant")
        .to_string())
}

/// 把 Chroma 的 query 结果转成统一的 `SearchHit`。
///
/// **距离 → 相似度的转换**：Chroma 的 cosine 空间返回
/// `1 - 余弦相似度`，这里换算回「越大越相似」，和 FAISS 对齐。
fn hits_from_query_result(result: &chromadb::collection::QueryResult) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let ids = result.ids.first().cloned().unwrap_or_default();
    let distances = result
        .distances
        .as_ref()
        .and_then(|d| d.first().cloned())
        .unwrap_or_default();
    let metadatas = result.metadatas.as_ref().and_then(|m| m.first().cloned());

    for (i, id) in ids.iter().enumerate() {
        let meta = metadatas
            .as_ref()
            .and_then(|m| m.get(i))
            .and_then(|m| m.as_ref());
        hits.push(SearchHit {
            id: id.clone(),
            score: 1.0 - distances.get(i).copied().unwrap_or(0.0),
            source: meta
                .and_then(|m| m.get("source"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            heading_path: meta
                .and_then(|m| m.get("heading_path"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            text: meta
                .and_then(|m| m.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        });
    }

    hits
}

impl VectorStore for ChromaStore {
    fn name(&self) -> &str {
        "chroma"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.cached_count
    }

    async fn add(&mut self, chunks: &[EmbeddedChunk]) -> Result<(), StoreError> {
        if chunks.is_empty() {
            return Ok(());
        }

        for c in chunks {
            if c.vector.len() != self.dimension {
                return Err(StoreError::DimensionMismatch {
                    expected: self.dimension,
                    actual: c.vector.len(),
                });
            }
        }

        let ids: Vec<&str> = chunks.iter().map(|c| c.id.as_str()).collect();
        let embeddings: Vec<Vec<f32>> = chunks.iter().map(|c| c.vector.clone()).collect();
        let documents: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let metadatas: Vec<serde_json::Map<String, serde_json::Value>> = chunks
            .iter()
            .map(|c| {
                json!({
                    "source": c.source,
                    "index": c.index,
                    "heading_path": c.heading_path,
                    "text": c.text,
                })
                .as_object()
                .cloned()
                .unwrap_or_default()
            })
            .collect();

        // 用 upsert 而不是 add —— 重复写入同一个 ID 时覆盖，
        // 不会报「ID 已存在」。
        self.collection
            .upsert(
                CollectionEntries {
                    ids,
                    embeddings: Some(embeddings),
                    metadatas: Some(metadatas),
                    documents: Some(documents),
                },
                None,
            )
            .await
            .map_err(|e| StoreError::Chroma(e.to_string()))?;

        // 用服务端返回的真实条数刷新缓存。
        self.cached_count = self
            .collection
            .count()
            .await
            .map_err(|e| StoreError::Chroma(e.to_string()))?;

        Ok(())
    }

    async fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchHit>, StoreError> {
        self.search_filtered(query, k, None).await
    }

    async fn save(&self, _path: &Path) -> Result<(), StoreError> {
        // Chroma 自己管持久化 —— 服务端写盘，客户端不需要 save。
        Ok(())
    }

    async fn load(&mut self, _path: &Path) -> Result<(), StoreError> {
        // 同上：Chroma 的数据在服务端，客户端不需要 load。
        Ok(())
    }
}

/// 读取集合里的原始记录（不做检索）。
#[allow(dead_code)]
pub async fn peek_all(store: &ChromaStore, limit: usize) -> Result<Vec<SearchHit>, StoreError> {
    let result = store
        .collection
        .get(GetOptions {
            limit: Some(limit),
            include: Some(vec!["metadatas".to_string(), "documents".to_string()]),
            ..Default::default()
        })
        .await
        .map_err(|e| StoreError::Chroma(e.to_string()))?;

    let mut hits = Vec::new();
    for (i, id) in result.ids.iter().enumerate() {
        let meta = result
            .metadatas
            .as_ref()
            .and_then(|m| m.get(i))
            .and_then(|m| m.as_ref());
        hits.push(SearchHit {
            id: id.clone(),
            score: 0.0,
            source: meta
                .and_then(|m| m.get("source"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            heading_path: meta
                .and_then(|m| m.get("heading_path"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            text: meta
                .and_then(|m| m.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        });
    }

    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_is_converted_to_similarity() {
        // Chroma cosine 空间：distance = 1 - cos
        // 转换后 score 应该等于余弦相似度。
        let result = chromadb::collection::QueryResult {
            ids: vec![vec!["a".to_string(), "b".to_string()]],
            distances: Some(vec![vec![0.0, 0.25]]),
            metadatas: None,
            documents: None,
            embeddings: None,
        };

        let hits = hits_from_query_result(&result);
        assert_eq!(hits.len(), 2);
        assert!((hits[0].score - 1.0).abs() < 1e-6, "完全相同 → 相似度 1");
        assert!(
            (hits[1].score - 0.75).abs() < 1e-6,
            "距离 0.25 → 相似度 0.75"
        );
    }
}
