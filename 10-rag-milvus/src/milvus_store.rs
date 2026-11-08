//! Milvus 存储 —— 生产级向量数据库。
//!
//! 这个模块是本项目的核心：**把 Milvus 实现成 05-rag 的 [`VectorStore`] trait**。
//!
//! ## 为什么这层封装有价值
//!
//! 05-rag 定义了 `VectorStore`：
//!
//! ```text
//! add / search / len / save / load
//! ```
//!
//! FAISS（进程内）、Chroma（HTTP 服务）都实现了它。
//! 现在 Milvus 也实现了 —— 于是「用 Milvus 换掉 Chroma」这件事
//! 变成一次**改一行配置**的操作，而不是一次重写。
//!
//! ## Milvus 与前两代存储的关键差异
//!
//! | | FAISS | Chroma | **Milvus** |
//! |---|---|---|---|
//! | 形态 | 进程内 C++ 库 | HTTP 服务 | **分布式服务（etcd + MinIO + 计算节点）** |
//! | 元数据 | Rust 侧 sidecar | 库内 JSON | **库内标量字段 + JSON 字段** |
//! | 过滤 | 检索后自己滤 | 服务端过滤 | **服务端过滤 + segment pruning** |
//! | 索引 | 进程内建 | 无（暴力） | **服务端建索引（HNSW/IVF/DISKANN/SCANN）** |
//! | 全文 | 无 | 无 | **服务端 BM25 函数（稀疏向量）** |
//! | 混合检索 | 自己写（08） | 无 | **服务端 hybrid_search + RRF/Weighted** |
//! | 持久化 | 一个文件 | 一个目录 | **对象存储 + segment + WAL** |
//!
//! ## 距离语义必须对齐
//!
//! 这是接入任何向量库最容易踩的坑：
//!
//! | 空间 | 返回值 | 方向 |
//! |---|---|---|
//! | `COSINE` | 余弦相似度 | **越大越相似** |
//! | `IP` | 内积 | 越大越相似 |
//! | `L2` | **平方** L2 距离 | 越小越相似 |
//!
//! 05-rag 的 `SearchHit.score` 统一是「越大越相似」。
//! 我们的向量已 L2 归一化，所以：
//!
//! - `COSINE` → 直接就是余弦相似度，**无需转换** ✅
//! - `IP` → 归一化后等于余弦相似度，**无需转换** ✅
//! - `L2` → 需要 `1 - d/2` 换算（`d = 2 - 2cos`）
//!
//! 本项目默认用 `COSINE`，并把 `L2` 的换算逻辑也实现好了。
//!
//! ## Schema 设计
//!
//! ```text
//! pk       Int64             主键（自增序号）
//! chunk_id VarChar           业务 ID —— `{文件名}#{序号}`
//! source   VarChar           来源文件
//! heading  JSON              标题路径（数组）
//! text     VarChar           原文（BM25 的输入）
//! vector   FloatVector       稠密向量
//! sparse   SparseFloatVector BM25 稀疏向量（服务端生成，不可直接写）
//! ```
//!
//! `sparse` 字段是**服务端函数输出** —— Milvus 会在写入时
//! 自动对 `text` 做分词、算 BM25 权重、生成稀疏向量。
//! 客户端**不需要**（也不能）写这个字段。

use std::collections::HashMap;
use std::path::Path;

use agent_logging::{debug, info};
use milvus::v2::client::ClientV2;
use milvus::v2::request::collection::{
    CreateCollectionRequest, DropCollectionRequest, GetCollectionStatsRequest,
    HasCollectionRequest, LoadCollectionRequest,
};
use milvus::v2::request::dml::{InsertRequest, UpsertRequest};
use milvus::v2::request::dql::{
    HybridSearchRequest, QueryRequest, SearchRequest, SubSearchRequest,
};
use milvus::v2::request::index::CreateIndexRequest;
use milvus::v2::response::dql::SearchResponse;
use milvus::v2::types::{
    CollectionSchema, ConnectConfig, ConsistencyLevel, DataType, FieldData, FieldSchema, Function,
    FunctionType, IndexParam, IndexType, MetricType, RRFRerank, SearchVectors,
};
use rag_05::embed::EmbeddedChunk;
use rag_05::store::{SearchHit, StoreError, VectorStore};

use crate::config::{MetricKind, MilvusConfig};
use crate::schema::ChunkRecord;

/// Milvus 的字段名常量集合。
///
/// 从配置构造，所以 schema 改了名字不会漏改某处。
#[derive(Debug, Clone)]
pub struct MilvusSchema {
    /// 主键字段名。
    pub primary: String,
    /// 业务 ID 字段名。
    pub id_field: String,
    /// 来源文件字段名。
    pub source_field: String,
    /// 标题路径字段名。
    pub heading_field: String,
    /// 原文字段名。
    pub text_field: String,
    /// 稠密向量字段名。
    pub vector_field: String,
    /// BM25 稀疏向量字段名。
    pub sparse_field: String,
}

impl MilvusSchema {
    /// 从配置构造。
    pub fn from_config(config: &MilvusConfig) -> Self {
        Self {
            primary: config.primary_field.clone(),
            id_field: config.id_field.clone(),
            source_field: "source".to_string(),
            heading_field: config.heading_field.clone(),
            text_field: config.text_field.clone(),
            vector_field: config.vector_field.clone(),
            sparse_field: config.sparse_field.clone(),
        }
    }

    /// 需要在检索结果里返回的标量字段。
    fn output_fields(&self) -> Vec<String> {
        vec![
            self.id_field.clone(),
            self.source_field.clone(),
            self.heading_field.clone(),
            self.text_field.clone(),
        ]
    }
}

/// Milvus 向量存储。
pub struct MilvusStore {
    /// SDK 客户端。
    client: ClientV2,
    /// collection 名。
    collection: String,
    /// 数据库名。
    database: String,
    /// 字段名。
    schema: MilvusSchema,
    /// 嵌入模型名。
    model_name: String,
    /// 向量维度。
    dimension: usize,
    /// 距离度量。
    metric: MetricKind,
    /// 服务端条数缓存。
    cached_count: usize,
}

impl std::fmt::Debug for MilvusStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MilvusStore")
            .field("collection", &self.collection)
            .field("database", &self.database)
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("count", &self.cached_count)
            .finish()
    }
}

impl MilvusStore {
    /// 连接 Milvus（不创建 collection）。
    ///
    /// `config.token` 的格式是 `用户名:密码` —— SDK 内部会做 base64。
    pub async fn connect(
        config: &MilvusConfig,
        dimension: usize,
        model_name: &str,
    ) -> Result<Self, StoreError> {
        let connect = ConnectConfig::new()
            .uri(config.uri.clone())
            .token(config.token.clone())
            .database(config.database.clone());

        let client = ClientV2::new(&connect)
            .await
            .map_err(|e| StoreError::Milvus(format!("连接 Milvus 失败: {e}")))?;

        info!(
            uri = %config.uri,
            database = %config.database,
            collection = %config.collection,
            dimension,
            "已连接 Milvus"
        );

        Ok(Self {
            client,
            collection: config.collection.clone(),
            database: config.database.clone(),
            schema: MilvusSchema::from_config(config),
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

    /// 数据库名。
    pub fn database_name(&self) -> &str {
        &self.database
    }

    /// 字段名。
    pub fn schema(&self) -> &MilvusSchema {
        &self.schema
    }

    /// SDK 客户端（供混合检索等高级操作使用）。
    pub fn client(&self) -> &ClientV2 {
        &self.client
    }

    /// 服务端真实条数。
    ///
    /// 用 `count(*)` 查询而不是本地缓存 —— 服务端才是事实来源。
    pub async fn count(&self) -> Result<usize, StoreError> {
        let request = QueryRequest::builder()
            .collection_name(self.collection.clone())
            .output_fields(["count(*)"])
            .consistency_level(ConsistencyLevel::Strong)
            .build()
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let response = self
            .client
            .query(request)
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let mut rows = response
            .results()
            .rows()
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let row = rows
            .next()
            .ok_or_else(|| StoreError::Milvus("count 查询没有返回任何行".to_string()))?;

        let count = row
            .get_i64("count(*)")
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        Ok(count.max(0) as usize)
    }

    /// collection 的服务端统计（行数 + segment 数等）。
    pub async fn stats(&self) -> Result<usize, StoreError> {
        let request = GetCollectionStatsRequest::builder()
            .collection_name(self.collection.clone())
            .build()
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let response = self
            .client
            .get_collection_stats(request)
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        Ok(response.row_count().unwrap_or(0).max(0) as usize)
    }

    /// 确保 collection 存在：不存在就建，建完建索引并加载。
    ///
    /// **幂等** —— 重复调用不会报错，已存在时只补建缺失的索引。
    ///
    /// ## 顺序很重要
    ///
    /// ```text
    /// create_collection (schema)
    ///      ↓
    /// create_index      ← 索引建在字段上
    ///      ↓
    /// load_collection   ← 把 segment + 索引加载进内存，不 load 搜不了
    /// ```
    pub async fn ensure_collection(&mut self, config: &MilvusConfig) -> Result<bool, StoreError> {
        let exists = self
            .client
            .has_collection(
                HasCollectionRequest::builder()
                    .collection_name(self.collection.clone())
                    .build()
                    .map_err(|e| StoreError::Milvus(e.to_string()))?,
            )
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?
            .exists();

        let created = !exists;

        if created {
            info!(collection = %self.collection, "collection 不存在，开始创建");
            self.create_collection(config).await?;
        } else {
            // 已存在时，索引可能因为中途失败而缺失 —— 补建。
            debug!(collection = %self.collection, "collection 已存在，补建缺失索引");
            self.ensure_indexes(config).await?;
        }

        // load：把数据与索引载入 QueryNode 内存。
        // 不 load 的话 search 会报「collection not loaded」。
        self.client
            .load_collection(
                LoadCollectionRequest::builder()
                    .collection_name(self.collection.clone())
                    .sync(true)
                    .build()
                    .map_err(|e| StoreError::Milvus(e.to_string()))?,
            )
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        self.cached_count = self.count().await.unwrap_or(0);

        info!(
            collection = %self.collection,
            created,
            cached_count = self.cached_count,
            "collection 已就绪（已 load）"
        );

        Ok(created)
    }

    /// 创建 collection（schema + 索引）。
    async fn create_collection(&self, config: &MilvusConfig) -> Result<(), StoreError> {
        let schema = self.build_schema(config);

        let request = CreateCollectionRequest::builder()
            .collection_name(self.collection.clone())
            .schema(schema)
            .build()
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        self.client
            .create_collection(request)
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        self.ensure_indexes(config).await
    }

    /// 构造 collection schema。
    ///
    /// ⚠️ 关键点：`text` 字段**只能声明一次**。
    /// BM25 需要它开启分析器 —— 所以启用 BM25 时直接用
    /// 「带分析器的 text 字段」这一份声明，而不是加两份同名字段
    /// （同名字段会让 Milvus 直接报错）。
    fn build_schema(&self, config: &MilvusConfig) -> CollectionSchema {
        let text_field = if config.enable_bm25 {
            // BM25 的输入字段：开启分析器（服务端分词）。
            FieldSchema::new()
                .name(self.schema.text_field.clone())
                .description("块正文（启用分析器，供 BM25 使用）")
                .data_type(DataType::VarChar)
                .max_length(65535)
                .enable_analyzer(true)
        } else {
            FieldSchema::new()
                .name(self.schema.text_field.clone())
                .description("块正文")
                .data_type(DataType::VarChar)
                .max_length(65535)
        };

        let mut schema = CollectionSchema::new()
            // 主键：整数序号。Milvus 不允许 auto_id —— 否则我们自己
            // 没法用稳定的 chunk_id 做 upsert。
            .add_field(
                FieldSchema::new()
                    .name(self.schema.primary.clone())
                    .description("自增主键")
                    .data_type(DataType::Int64)
                    .primary_key(true)
                    .auto_id(false),
            )
            // 业务 ID：真正的身份标识。
            .add_field(
                FieldSchema::new()
                    .name(self.schema.id_field.clone())
                    .description("业务 ID: {文件名}#{序号}")
                    .data_type(DataType::VarChar)
                    .max_length(512),
            )
            .add_field(
                FieldSchema::new()
                    .name(self.schema.source_field.clone())
                    .description("来源文件")
                    .data_type(DataType::VarChar)
                    .max_length(512),
            )
            // 标题路径用 JSON 存 —— 它是变长数组。
            .add_field(
                FieldSchema::new()
                    .name(self.schema.heading_field.clone())
                    .description("标题路径")
                    .data_type(DataType::Json),
            )
            .add_field(text_field)
            .add_field(
                FieldSchema::new()
                    .name(self.schema.vector_field.clone())
                    .description("稠密向量")
                    .data_type(DataType::FloatVector)
                    .dimension(self.dimension as u32),
            );

        if config.enable_bm25 {
            schema = schema
                // BM25 稀疏向量字段：**函数输出**，客户端不写。
                .add_field(
                    FieldSchema::new()
                        .name(self.schema.sparse_field.clone())
                        .description("BM25 稀疏向量（服务端生成）")
                        .data_type(DataType::SparseFloatVector),
                )
                // 服务端函数：text → sparse（BM25 权重）。
                .add_function(
                    Function::new()
                        .name("bm25")
                        .function_type(FunctionType::Bm25)
                        .input_fields([self.schema.text_field.clone()])
                        .output_fields([self.schema.sparse_field.clone()]),
                );
        }

        schema
    }

    /// 建索引（稠密向量 + 稀疏向量）。
    async fn ensure_indexes(&self, config: &MilvusConfig) -> Result<(), StoreError> {
        let mut params = vec![
            IndexParam::new()
                .field_name(self.schema.vector_field.clone())
                .index_type(to_milvus_index_type(config.index_type))
                .metric_type(to_milvus_metric(config.metric)),
        ];

        if config.enable_bm25 {
            params.push(
                IndexParam::new()
                    .field_name(self.schema.sparse_field.clone())
                    .index_type(IndexType::SparseInvertedIndex)
                    .metric_type(MetricType::Bm25),
            );
        }

        self.client
            .create_index(
                CreateIndexRequest::builder()
                    .collection_name(self.collection.clone())
                    .index_params(params)
                    .sync(true)
                    .build()
                    .map_err(|e| StoreError::Milvus(e.to_string()))?,
            )
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        Ok(())
    }

    /// 删除整个 collection。
    pub async fn drop_collection(&self) -> Result<(), StoreError> {
        self.client
            .drop_collection(
                DropCollectionRequest::builder()
                    .collection_name(self.collection.clone())
                    .build()
                    .map_err(|e| StoreError::Milvus(e.to_string()))?,
            )
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?;
        Ok(())
    }

    /// 纯向量检索。
    pub async fn search_dense(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&str>,
    ) -> Result<Vec<SearchHit>, StoreError> {
        let mut builder = SearchRequest::builder()
            .collection_name(self.collection.clone())
            .vector_field(self.schema.vector_field.clone())
            .vectors(SearchVectors::Float(vec![query.to_vec()]))
            .output_fields(self.schema.output_fields())
            .limit(k as i64)
            // ⚠️ 必须用 Strong（SDK 映射成 guarantee_ts = 0，服务端按「最新提交」读）。
            // 用 Bounded 会被 SDK 映射成字面时间戳 2，读到的是远古快照 —— 表现为
            // 「刚灌完库却只搜到 1~2 条」。
            .consistency_level(ConsistencyLevel::Strong);

        if let Some(expr) = filter {
            builder = builder.filter(expr.to_string());
        }

        let response = self
            .client
            .search(
                builder
                    .build()
                    .map_err(|e| StoreError::Milvus(e.to_string()))?,
            )
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let hits = hits_from_search(&response, self.metric, &self.schema)?;
        debug!(
            collection = %self.collection,
            k,
            hit_count = hits.len(),
            filter,
            "Milvus 稠密检索完成"
        );
        Ok(hits)
    }

    /// 服务端混合检索：稠密向量 + BM25 全文，用 RRF 融合。
    ///
    /// 这是 08-rag「自己在 Rust 里写 BM25 + RRF」的**服务端版本** ——
    /// 区别在于：分词、倒排索引、权重计算、融合排序全在 Milvus 里做，
    /// 一次 RPC 出结果。
    ///
    /// `query_text` 是用于 BM25 的查询串。
    pub async fn search_hybrid(
        &self,
        query: &[f32],
        query_text: &str,
        k: usize,
        per_path_k: usize,
        filter: Option<&str>,
    ) -> Result<Vec<SearchHit>, StoreError> {
        let mut dense = SubSearchRequest::builder()
            .vector_field(self.schema.vector_field.clone())
            .vectors(SearchVectors::Float(vec![query.to_vec()]))
            .limit(per_path_k as i64);

        if let Some(expr) = filter {
            dense = dense.filter(expr.to_string());
        }

        // `sub_requests` 要的是**已构建**的 `SubSearchRequest`，不是 builder。
        let mut sub_requests = vec![
            dense
                .build()
                .map_err(|e| StoreError::Milvus(e.to_string()))?,
        ];

        if self.has_bm25() {
            let mut sparse = SubSearchRequest::builder()
                .vector_field(self.schema.sparse_field.clone())
                // EmbeddedText：告诉服务端「用 BM25 函数把这个文本转成稀疏向量」。
                .vectors(SearchVectors::EmbeddedText(vec![query_text.to_string()]))
                .limit(per_path_k as i64);

            if let Some(expr) = filter {
                sparse = sparse.filter(expr.to_string());
            }

            sub_requests.push(
                sparse
                    .build()
                    .map_err(|e| StoreError::Milvus(e.to_string()))?,
            );
        }

        let request = HybridSearchRequest::builder()
            .collection_name(self.collection.clone())
            .sub_requests(sub_requests)
            // RRF：只看排名不看分数 —— 稠密和 BM25 的分数本来就不可比。
            .rerank(RRFRerank::new().k(60))
            .output_fields(self.schema.output_fields())
            .limit(k as i64)
            .consistency_level(ConsistencyLevel::Strong)
            .build()
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let response = self
            .client
            .hybrid_search(request)
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let hits = hits_from_search(&response, self.metric, &self.schema)?;
        info!(
            collection = %self.collection,
            k,
            per_path_k,
            bm25 = self.has_bm25(),
            hit_count = hits.len(),
            "Milvus 服务端混合检索完成"
        );
        Ok(hits)
    }

    /// 是否启用了服务端 BM25。
    pub fn has_bm25(&self) -> bool {
        !self.schema.sparse_field.is_empty()
    }

    /// 把记录批量写入（**upsert 语义**）。
    ///
    /// ## 主键怎么来
    ///
    /// 主键**不是**自增序号，而是从 `chunk_id` 稳定派生的 Int64
    /// （见 [`crate::schema::stable_pk`]）。这样重复灌同一份语料时，
    /// 服务端会**覆盖**旧行而不是新增 —— 这才是 upsert 该有的行为。
    ///
    /// 如果主键用自增序号，每次重跑都会新增一份数据，
    /// `count(*)` 会一路膨胀，而检索结果里会混进同一段文本的多份副本。
    pub async fn upsert_records(&self, records: &[ChunkRecord]) -> Result<i64, StoreError> {
        if records.is_empty() {
            return Ok(0);
        }

        let pks: Vec<i64> = records
            .iter()
            .map(|r| crate::schema::stable_pk(&r.chunk_id))
            .collect();
        let ids: Vec<String> = records.iter().map(|r| r.chunk_id.clone()).collect();
        let sources: Vec<String> = records.iter().map(|r| r.source.clone()).collect();
        let headings: Vec<serde_json::Value> = records
            .iter()
            .map(|r| serde_json::json!(r.heading_path))
            .collect();
        let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
        let vectors: Vec<Vec<f32>> = records.iter().map(|r| r.vector.clone()).collect();

        // 用「列式」接口（FieldData）而不是「行式」（JSON rows）：
        // 列式省掉了 SDK 侧的 JSON → 列的转换，服务端也更好对齐。
        let columns = vec![
            FieldData::Int64 {
                name: self.schema.primary.clone(),
                values: pks,
            },
            FieldData::VarChar {
                name: self.schema.id_field.clone(),
                values: ids,
            },
            FieldData::VarChar {
                name: self.schema.source_field.clone(),
                values: sources,
            },
            FieldData::Json {
                name: self.schema.heading_field.clone(),
                values: headings,
            },
            FieldData::VarChar {
                name: self.schema.text_field.clone(),
                values: texts,
            },
            FieldData::FloatVector {
                name: self.schema.vector_field.clone(),
                values: vectors,
            },
        ];

        let request = UpsertRequest::builder()
            .insert(
                InsertRequest::builder()
                    .collection_name(self.collection.clone())
                    .columns(columns)
                    .build()
                    .map_err(|e| StoreError::Milvus(e.to_string()))?,
            )
            .build()
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let response = self
            .client
            .upsert(request)
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let count = response.upsert_count();
        info!(
            collection = %self.collection,
            records = records.len(),
            upsert_count = count,
            "Milvus upsert 完成"
        );
        Ok(count)
    }

    /// 查询某个业务 ID 对应的记录（用于校验写入是否生效）。
    ///
    /// ⚠️ Milvus 的 `get` 只认**主键** —— 我们的主键是整数 `pk`，
    /// 不是字符串 `chunk_id`。所以这里走 `query` + 过滤表达式。
    pub async fn get_by_chunk_id(&self, chunk_id: &str) -> Result<Option<ChunkRecord>, StoreError> {
        let request = QueryRequest::builder()
            .collection_name(self.collection.clone())
            .filter(format!(
                "{} == \"{}\"",
                self.schema.id_field,
                chunk_id.replace('"', "\\\"")
            ))
            .output_fields(self.schema.output_fields())
            .limit(1)
            .consistency_level(ConsistencyLevel::Strong)
            .build()
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let response = self
            .client
            .query(request)
            .await
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        let mut rows = response
            .results()
            .rows()
            .map_err(|e| StoreError::Milvus(e.to_string()))?;

        // 只取第一行 —— 过滤表达式已经保证最多命中一条。
        if let Some(row) = rows.next() {
            let id = row
                .get_str(&self.schema.id_field)
                .map_err(|e| StoreError::Milvus(e.to_string()))?
                .to_string();
            let source = row
                .get_str(&self.schema.source_field)
                .unwrap_or_default()
                .to_string();
            let text = row
                .get_str(&self.schema.text_field)
                .unwrap_or_default()
                .to_string();
            let heading = row
                .get_json(&self.schema.heading_field)
                .ok()
                .and_then(|v| {
                    v.as_array().map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect::<Vec<_>>()
                    })
                })
                .unwrap_or_default();

            return Ok(Some(ChunkRecord {
                chunk_id: id,
                source,
                index: 0,
                heading_path: heading,
                text,
                vector: Vec::new(),
            }));
        }

        Ok(None)
    }
}

impl VectorStore for MilvusStore {
    fn name(&self) -> &str {
        "milvus"
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

        // 主键由 chunk_id 稳定派生 —— 重复 add 是 upsert，不是 insert。
        let inserted = self.upsert_records(&records).await?;
        self.cached_count = self
            .count()
            .await
            .unwrap_or(self.cached_count + inserted as usize);

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
        // Milvus 自己管持久化 —— 数据在对象存储，客户端不需要 save。
        Ok(())
    }

    async fn load(&mut self, _path: &Path) -> Result<(), StoreError> {
        // 同上：数据在服务端，客户端不需要 load。
        Ok(())
    }
}

/// 把配置里的索引类型映射成 SDK 的枚举。
fn to_milvus_index_type(kind: crate::config::IndexKind) -> IndexType {
    match kind {
        crate::config::IndexKind::Flat => IndexType::Flat,
        crate::config::IndexKind::Hnsw => IndexType::Hnsw,
        crate::config::IndexKind::IvfFlat => IndexType::IvfFlat,
        crate::config::IndexKind::DiskAnn => IndexType::DiskAnn,
        crate::config::IndexKind::Scann => IndexType::Scann,
        crate::config::IndexKind::AutoIndex => IndexType::AutoIndex,
    }
}

/// 把配置里的度量映射成 SDK 的枚举。
fn to_milvus_metric(kind: MetricKind) -> MetricType {
    match kind {
        MetricKind::Cosine => MetricType::Cosine,
        MetricKind::Ip => MetricType::Ip,
        MetricKind::L2 => MetricType::L2,
    }
}

/// 把 Milvus 的检索结果转成统一的 `SearchHit`。
///
/// **分数方向必须对齐** —— `SearchHit.score` 是「越大越相似」。
fn hits_from_search(
    response: &SearchResponse,
    metric: MetricKind,
    schema: &MilvusSchema,
) -> Result<Vec<SearchHit>, StoreError> {
    let mut hits = Vec::new();

    for result in response.results().iter() {
        let scores = result.get_scores();
        for (index, row) in result
            .rows()
            .map_err(|e| StoreError::Milvus(e.to_string()))?
            .enumerate()
        {
            let id = row
                .get_str(&schema.id_field)
                .map_err(|e| StoreError::Milvus(e.to_string()))?
                .to_string();
            let source = row
                .get_str(&schema.source_field)
                .unwrap_or_default()
                .to_string();
            let text = row
                .get_str(&schema.text_field)
                .unwrap_or_default()
                .to_string();
            let heading_path = row
                .get_json(&schema.heading_field)
                .ok()
                .and_then(|v| {
                    v.as_array().map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect::<Vec<_>>()
                    })
                })
                .unwrap_or_default();

            let raw_score = scores.get(index).copied().unwrap_or(0.0);
            let score = normalize_score(raw_score, metric);

            hits.push(SearchHit {
                id,
                score,
                source,
                heading_path,
                text,
            });
        }
    }

    Ok(hits)
}

/// 把原始分数换算成「越大越相似」。
///
/// - `COSINE` / `IP`：本来就是越大越相似，直接用
/// - `L2`：Milvus 返回**平方** L2 距离，越小越相似。
///   归一化向量下 `d² = 2 - 2cos`，所以 `cos = 1 - d²/2`。
fn normalize_score(raw: f32, metric: MetricKind) -> f32 {
    match metric {
        MetricKind::Cosine | MetricKind::Ip => raw,
        MetricKind::L2 => 1.0 - raw / 2.0,
    }
}

/// 构造一个用于「服务端过滤」的表达式（按来源文件过滤）。
///
/// Milvus 的过滤表达式语法：字段名 + 操作符 + 值。
pub fn source_filter(source: &str) -> String {
    format!("source == \"{}\"", source.replace('"', "\\\""))
}

/// 构造一个按 chunk id 前缀过滤的表达式。
pub fn id_prefix_filter(prefix: &str) -> String {
    format!("chunk_id like \"{}%\"", prefix.replace('"', "\\\""))
}

/// HNSW 的构建参数（`M` / `efConstruction`）。
///
/// Milvus 的 HNSW 默认 `M=30, efConstruction=200` —— 对学习环境偏重，
/// 这里调小一点，建索引更快。
pub fn hnsw_extra_params() -> HashMap<String, String> {
    HashMap::from([
        ("M".to_string(), "16".to_string()),
        ("efConstruction".to_string(), "200".to_string()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_distance_is_converted_to_similarity() {
        // 归一化向量：完全相同 → d²=0 → cos=1
        assert!((normalize_score(0.0, MetricKind::L2) - 1.0).abs() < 1e-6);
        // 正交 → d²=2 → cos=0
        assert!(normalize_score(2.0, MetricKind::L2).abs() < 1e-6);
    }

    #[test]
    fn cosine_and_ip_keep_raw_score() {
        assert!((normalize_score(0.87, MetricKind::Cosine) - 0.87).abs() < 1e-6);
        assert!((normalize_score(0.87, MetricKind::Ip) - 0.87).abs() < 1e-6);
    }

    #[test]
    fn source_filter_escapes_quotes() {
        assert_eq!(source_filter("a.md"), "source == \"a.md\"");
        assert_eq!(source_filter("a\"b.md"), "source == \"a\\\"b.md\"");
    }

    #[test]
    fn id_prefix_filter_uses_like() {
        assert_eq!(id_prefix_filter("kb"), "chunk_id like \"kb%\"");
    }
}
