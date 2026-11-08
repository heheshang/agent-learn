//! Qdrant 存储 —— Milvus 的横向对照对象。
//!
//! ## 为什么要有这个模块
//!
//! 选向量数据库不能靠「谁的名气大」。靠谱的做法是：
//! **同一份语料、同一套嵌入模型、同一批查询**，分别灌进两个库，
//! 用同一套指标对比。
//!
//! Qdrant 和 Milvus 是最直接的对手 —— 同样 gRPC、同样 HNSW、
//! 同样支持 payload 过滤 + 混合检索。但差异也真实存在：
//!
//! | | Milvus | Qdrant |
//! |---|---|---|
//! | 架构 | 分布式（etcd + 对象存储 + 计算节点） | 单体服务（Rust 写的） |
//! | 部署复杂度 | 三个容器 | 一个容器 |
//! | 全文检索 | 服务端 BM25 函数（稀疏向量） | payload text index + BM25 |
//! | 混合检索 | `hybrid_search` + RRF/Weighted rerank | `prefetch` + `Fusion::Rrf` |
//! | 分片 | 内建（shard / partition / resource group） | 分片键 + 多副本 |
//! | 内存模型 | segment + 索引加载到 QueryNode | 全量进内存（可 mmap） |
//!
//! 本模块实现同一个 [`VectorStore`] trait，于是两边的对比
//! 可以在**同一套代码**里跑 —— 这才是公平的对照。
//!
//! ## Qdrant 的 payload 设计
//!
//! Qdrant 没有「列」的概念 —— 每个 point 有一份 JSON payload：
//!
//! ```json
//! {
//!   "chunk_id": "kb.md#3",
//!   "source": "kb.md",
//!   "heading_path": ["RAG", "分块"],
//!   "text": "块正文……"
//! }
//! ```
//!
//! 过滤就是在这份 payload 上做条件匹配，检索时**服务端执行**。

use std::path::Path;

use agent_logging::{debug, info, warn};
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Distance, FieldType, Filter, PointId, PointStruct,
    SearchParams, SparseVector, TextIndexParamsBuilder, TokenizerType, Value, VectorParamsBuilder,
    Vectors,
};
use qdrant_client::{Payload, Qdrant, QdrantBuilder};
use rag_05::embed::EmbeddedChunk;
use rag_05::store::{SearchHit, StoreError, VectorStore};
use sha1::{Digest, Sha1};

use crate::config::{MetricKind, QdrantConfig};
use crate::schema::ChunkRecord;

/// payload 里的字段名 —— 与 Milvus 侧保持一致，方便对照。
const PAYLOAD_CHUNK_ID: &str = "chunk_id";
const PAYLOAD_SOURCE: &str = "source";
const PAYLOAD_HEADING: &str = "heading_path";
const PAYLOAD_TEXT: &str = "text";

/// Qdrant 向量存储。
pub struct QdrantStore {
    /// SDK 客户端。
    client: Qdrant,
    /// collection 名。
    collection: String,
    /// 向量字段名。
    vector_field: String,
    /// 嵌入模型名。
    model_name: String,
    /// 向量维度。
    dimension: usize,
    /// 距离度量。
    metric: MetricKind,
    /// 条数缓存。
    cached_count: usize,
}

impl std::fmt::Debug for QdrantStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantStore")
            .field("collection", &self.collection)
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("count", &self.cached_count)
            .finish()
    }
}

impl QdrantStore {
    /// 连接 Qdrant。
    pub async fn connect(
        config: &QdrantConfig,
        dimension: usize,
        model_name: &str,
    ) -> Result<Self, StoreError> {
        let mut builder: QdrantBuilder = Qdrant::from_url(&config.url);

        if let Some(key) = config.api_key.clone() {
            builder = builder.api_key(key);
        }

        let client = builder
            .build()
            .map_err(|e| StoreError::Qdrant(format!("连接 Qdrant 失败: {e}")))?;

        info!(
            url = %config.url,
            collection = %config.collection,
            dimension,
            "已连接 Qdrant"
        );

        Ok(Self {
            client,
            collection: config.collection.clone(),
            vector_field: config.vector_field.clone(),
            model_name: model_name.to_string(),
            dimension,
            metric: config.metric,
            cached_count: 0,
        })
    }

    /// collection 名。
    pub fn collection_name(&self) -> &str {
        &self.collection
    }

    /// SDK 客户端。
    pub fn client(&self) -> &Qdrant {
        &self.client
    }

    /// 向量字段名。
    ///
    /// 本项目用 Qdrant 的**默认（匿名）向量**，所以检索时不需要 `using`；
    /// 这个字段保留下来是为了与 Milvus 侧的 schema 概念对齐。
    pub fn vector_field(&self) -> &str {
        &self.vector_field
    }

    /// 服务端真实条数。
    pub async fn count(&self) -> Result<usize, StoreError> {
        let response = self
            .client
            .count(qdrant_client::qdrant::CountPointsBuilder::new(
                &self.collection,
            ))
            .await
            .map_err(|e| StoreError::Qdrant(e.to_string()))?;

        Ok(response.result.map(|r| r.count).unwrap_or(0) as usize)
    }

    /// 确保 collection 存在：不存在就建，并给 `text` 字段建全文索引。
    pub async fn ensure_collection(&mut self, config: &QdrantConfig) -> Result<bool, StoreError> {
        let exists = self
            .client
            .collection_exists(&self.collection)
            .await
            .map_err(|e| StoreError::Qdrant(e.to_string()))?;

        let created = !exists;

        if created {
            info!(collection = %self.collection, "collection 不存在，开始创建");
            let request = CreateCollectionBuilder::new(&self.collection).vectors_config(
                VectorParamsBuilder::new(self.dimension as u64, to_qdrant_distance(config.metric)),
            );

            self.client
                .create_collection(request)
                .await
                .map_err(|e| StoreError::Qdrant(e.to_string()))?;

            // 建全文索引 —— Qdrant 的 BM25 靠它。
            // 字段已存在索引时服务端会报错，这里忽略（幂等）。
            let index_request = qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(
                &self.collection,
                PAYLOAD_TEXT,
                FieldType::Text,
            )
            .field_index_params(
                TextIndexParamsBuilder::new(TokenizerType::Multilingual).bm25_scoring(),
            );

            if let Err(e) = self.client.create_field_index(index_request).await {
                // 索引已存在不是致命错误 —— 记录但继续。
                warn!(collection = %self.collection, error = %e, "创建 text 字段索引失败（可能已存在）");
            }
        }

        self.cached_count = self.count().await.unwrap_or(0);

        info!(
            collection = %self.collection,
            created,
            cached_count = self.cached_count,
            "collection 已就绪"
        );

        Ok(created)
    }

    /// 删除整个 collection。
    pub async fn drop_collection(&self) -> Result<(), StoreError> {
        self.client
            .delete_collection(&self.collection)
            .await
            .map_err(|e| StoreError::Qdrant(e.to_string()))?;
        Ok(())
    }

    /// 写入一批记录。
    pub async fn upsert_records(&self, records: &[ChunkRecord]) -> Result<usize, StoreError> {
        if records.is_empty() {
            return Ok(0);
        }

        let points: Vec<PointStruct> = records
            .iter()
            .map(|record| {
                let mut payload = Payload::new();
                payload.insert(PAYLOAD_CHUNK_ID, record.chunk_id.clone());
                payload.insert(PAYLOAD_SOURCE, record.source.clone());
                payload.insert(
                    PAYLOAD_HEADING,
                    Value::from(serde_json::json!(record.heading_path)),
                );
                payload.insert(PAYLOAD_TEXT, record.text.clone());

                PointStruct::new(
                    // Qdrant 的 point id 只有两种形态：**无符号整数** 或 **UUID**。
                    // 我们的 chunk_id（`kb.md#0`）既不是整数也不是合法 UUID ——
                    // 所以从 chunk_id 派生一个**稳定的 UUID**。
                    //
                    // 「稳定」很重要：派生算法固定，所以同一份 chunk_id
                    // 永远得到同一个 UUID —— 重复灌库时 upsert 会覆盖而不是新增。
                    PointId::from(stable_uuid(&record.chunk_id)),
                    Vectors::from(record.vector.clone()),
                    payload,
                )
            })
            .collect();

        let request =
            qdrant_client::qdrant::UpsertPointsBuilder::new(&self.collection, points).wait(true);

        self.client
            .upsert_points(request)
            .await
            .map_err(|e| StoreError::Qdrant(e.to_string()))?;

        info!(
            collection = %self.collection,
            records = records.len(),
            "Qdrant upsert 完成"
        );
        Ok(records.len())
    }

    /// 纯向量检索。
    pub async fn search_dense(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchHit>, StoreError> {
        let mut builder = qdrant_client::qdrant::QueryPointsBuilder::new(&self.collection)
            .query(qdrant_client::qdrant::VectorInput::from(query.to_vec()))
            .limit(k as u64)
            .with_payload(true);

        if let Some(expr) = filter {
            builder = builder.filter(expr);
        }

        let response = self
            .client
            .query(builder)
            .await
            .map_err(|e| StoreError::Qdrant(e.to_string()))?;

        let hits = hits_from_response(&response, self.metric);
        debug!(collection = %self.collection, k, hit_count = hits.len(), "Qdrant 稠密检索完成");
        Ok(hits)
    }

    /// 关键词检索（payload 全文索引 + BM25）。
    ///
    /// Qdrant 1.20 支持在 payload 字段上建 **text index**，
    /// 然后用 `TextQuery` 走 BM25 打分。
    pub async fn search_text(
        &self,
        query_text: &str,
        k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchHit>, StoreError> {
        let mut builder = qdrant_client::qdrant::QueryPointsBuilder::new(&self.collection)
            .query(qdrant_client::qdrant::TextQueryBuilder::new(query_text.to_string()).build())
            .using(PAYLOAD_TEXT)
            .limit(k as u64)
            .with_payload(true);

        if let Some(expr) = filter {
            builder = builder.filter(expr);
        }

        let response = self
            .client
            .query(builder)
            .await
            .map_err(|e| StoreError::Qdrant(e.to_string()))?;

        let hits = hits_from_response(&response, self.metric);
        debug!(
            collection = %self.collection,
            k,
            hit_count = hits.len(),
            "Qdrant 全文（BM25）检索完成"
        );
        Ok(hits)
    }
}

impl VectorStore for QdrantStore {
    fn name(&self) -> &str {
        "qdrant"
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
        let records: Vec<ChunkRecord> = chunks.iter().map(crate::schema::chunk_to_record).collect();

        let written = self.upsert_records(&records).await?;
        self.cached_count = self.count().await.unwrap_or(self.cached_count + written);

        Ok(())
    }

    async fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchHit>, StoreError> {
        if query.len() != self.dimension {
            return Err(StoreError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        self.search_dense(query, k, None).await
    }

    async fn save(&self, _path: &Path) -> Result<(), StoreError> {
        // Qdrant 自己管持久化。
        Ok(())
    }

    async fn load(&mut self, _path: &Path) -> Result<(), StoreError> {
        // 同上。
        Ok(())
    }
}

/// 从业务 ID 派生一个**稳定的 UUID**。
///
/// Qdrant 的 point id 只接受无符号整数或 UUID，chunk_id（`kb.md#0`）两者都不是。
///
/// 这里用 **UUIDv5**（SHA-1 命名空间）—— 它是**确定性**的：
/// 同样的输入永远得到同样的 UUID。这正是 upsert 需要���性质：
/// 重复灌同一份语料不会产生重复 point。
///
/// 命名空间用 RFC 4122 的 DNS 命名空间，再自己套一层，
/// 避免和别的系统派生的 UUID 撞车。
fn stable_uuid(chunk_id: &str) -> String {
    // 固定命名空间（自选的随机值，只用于本项目）。
    const NAMESPACE: [u8; 16] = [
        0x6b, 0xa7, 0xb8, 0x14, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30,
        0xc8,
    ];

    // UUIDv5 = SHA-1(namespace || name)，取前 16 字节并打上版本/变体位。
    let mut hasher = Sha1::new();
    hasher.update(NAMESPACE);
    hasher.update(chunk_id.as_bytes());
    let digest = hasher.finalize();

    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    // version 5
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    // variant = RFC 4122
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    )
}

/// 把配置里的度量映射成 Qdrant 的枚举。
fn to_qdrant_distance(kind: MetricKind) -> Distance {
    match kind {
        MetricKind::Cosine => Distance::Cosine,
        MetricKind::Ip => Distance::Dot,
        MetricKind::L2 => Distance::Euclid,
    }
}

/// 把 Qdrant 的检索结果转成统一的 `SearchHit`。
fn hits_from_response(
    response: &qdrant_client::qdrant::QueryResponse,
    metric: MetricKind,
) -> Vec<SearchHit> {
    let mut hits = Vec::new();

    // `QueryResponse` 的结果在 `result` 字段里（不是 `points`）。
    for point in &response.result {
        let payload = &point.payload;

        let id = payload_str(payload, PAYLOAD_CHUNK_ID).unwrap_or_else(|| {
            point
                .id
                .as_ref()
                .map(point_id_to_string)
                .unwrap_or_default()
        });
        let source = payload_str(payload, PAYLOAD_SOURCE).unwrap_or_default();
        let text = payload_str(payload, PAYLOAD_TEXT).unwrap_or_default();
        let heading_path = payload_str_list(payload, PAYLOAD_HEADING);

        hits.push(SearchHit {
            id,
            score: normalize_score(point.score, metric),
            source,
            heading_path,
            text,
        });
    }

    hits
}

/// 把 `PointId` 变成字符串（数值 ID 转十进制，UUID 直接取字符串）。
fn point_id_to_string(id: &PointId) -> String {
    match id.point_id_options.as_ref() {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => n.to_string(),
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s)) => s.clone(),
        None => String::new(),
    }
}

/// 从 payload 里取一个字符串字段。
///
/// `Value` 是一层 protobuf oneof 包装 —— 要先拆 `kind` 再看具体类型。
fn payload_str(payload: &std::collections::HashMap<String, Value>, key: &str) -> Option<String> {
    match payload.get(key)?.kind.as_ref()? {
        qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
        _ => None,
    }
}

/// 从 payload 里取一个字符串数组字段。
fn payload_str_list(payload: &std::collections::HashMap<String, Value>, key: &str) -> Vec<String> {
    let Some(value) = payload.get(key) else {
        return Vec::new();
    };
    match value.kind.as_ref() {
        Some(qdrant_client::qdrant::value::Kind::ListValue(list)) => list
            .values
            .iter()
            .filter_map(|v| match v.kind.as_ref() {
                Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// 把原始分数换算成「越大越相似」。
///
/// Qdrant 的 `Dot` / `Cosine` 返回越大越相似的值；
/// `Euclid` 返回**平方**欧氏距离，越小越相似。
fn normalize_score(raw: f32, metric: MetricKind) -> f32 {
    match metric {
        MetricKind::Cosine | MetricKind::Ip => raw,
        MetricKind::L2 => 1.0 - raw / 2.0,
    }
}

/// 构造一个按来源文件过滤的条件。
pub fn source_filter(source: &str) -> Filter {
    Filter::must([Condition::matches("source", source.to_string())])
}

/// 构造一个按 chunk id 前缀过滤的条件。
pub fn id_prefix_filter(prefix: &str) -> Filter {
    Filter::must([Condition::matches("chunk_id", prefix.to_string())])
}

/// 稀疏向量（预留：Qdrant 的 sparse vector 用于 BM25 / SPLADE）。
pub fn sparse_vector(pairs: &[(u32, f32)]) -> SparseVector {
    let (indices, values): (Vec<u32>, Vec<f32>) = pairs.iter().copied().unzip();
    qdrant_client::qdrant::SparseVectorBuilder::new(indices, values).build()
}

/// 构造检索参数（HNSW 的 ef）。
pub fn search_params(hnsw_ef: u32, exact: bool) -> SearchParams {
    SearchParams {
        hnsw_ef: Some(hnsw_ef as u64),
        exact: Some(exact),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_distance_is_converted_to_similarity() {
        assert!((normalize_score(0.0, MetricKind::L2) - 1.0).abs() < 1e-6);
        assert!(normalize_score(2.0, MetricKind::L2).abs() < 1e-6);
    }

    #[test]
    fn cosine_keeps_raw_score() {
        assert!((normalize_score(0.9, MetricKind::Cosine) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn distance_mapping() {
        assert_eq!(to_qdrant_distance(MetricKind::Cosine), Distance::Cosine);
        assert_eq!(to_qdrant_distance(MetricKind::Ip), Distance::Dot);
        assert_eq!(to_qdrant_distance(MetricKind::L2), Distance::Euclid);
    }

    #[test]
    fn payload_string_is_extracted() {
        let mut payload = std::collections::HashMap::new();
        payload.insert(
            PAYLOAD_CHUNK_ID.to_string(),
            Value::from("kb.md#1".to_string()),
        );
        assert_eq!(
            payload_str(&payload, PAYLOAD_CHUNK_ID).as_deref(),
            Some("kb.md#1")
        );
        assert!(payload_str(&payload, "missing").is_none());
    }

    #[test]
    fn payload_list_is_extracted() {
        let mut payload = std::collections::HashMap::new();
        payload.insert(
            PAYLOAD_HEADING.to_string(),
            Value::from(serde_json::json!(["RAG", "分块"])),
        );
        assert_eq!(payload_str_list(&payload, PAYLOAD_HEADING).len(), 2);
    }

    #[test]
    fn stable_uuid_is_deterministic_and_valid() {
        let a = stable_uuid("kb.md#0");
        let b = stable_uuid("kb.md#0");
        let c = stable_uuid("kb.md#1");

        // 确定性 —— 同样输入永远同样输出（upsert 的前提）。
        assert_eq!(a, b);
        // 不同输入不同输出。
        assert_ne!(a, c);
        // 格式合法：8-4-4-4-12 的十六进制。
        assert_eq!(a.len(), 36);
        assert_eq!(a.chars().filter(|c| *c == '-').count(), 4);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
        // version 5 + RFC 4122 variant
        assert_eq!(a.as_bytes()[14], b'5');
        assert!(matches!(a.as_bytes()[19], b'8' | b'9' | b'a' | b'b'));
    }
}
