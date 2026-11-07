//! FAISS 存储 —— 进程内向量索引。
//!
//! 对应教程里的 **FAISS Intro**。
//!
//! ## FAISS 是什么
//!
//! Facebook AI Similarity Search：一个 C++ 的向量检索库。
//! 它的核心思想是**近似最近邻（ANN）** ——
//! 用一点精度损失，换来数量级的速度提升。
//!
//! ## 索引类型（本项目实现了三种）
//!
//! | 类型 | 描述串 | 原理 | 何时用 |
//! |---|---|---|---|
//! | 暴力检索 | `Flat` | 跟每个向量比一遍 | 数据量小（<1万），要 100% 精确 |
//! | 倒排文件 | `IVF{nlist}` | 先聚类，查询只搜几个簇 | 十万级以上 |
//! | 分层图 | `HNSW{M}` | 多层可导航小世界图 | 大规模、要求高召回 |
//!
//! ### Flat —— 精确但线性
//!
//! 所有向量平铺在内存里，查询时逐个算距离。
//! 复杂度 O(N)，但**没有精度损失**。作为教学基线最合适：
//! 你看到的检索结果就是理论最优，不会被近似算法干扰。
//!
//! ### IVF —— 用聚类剪枝
//!
//! 先用 k-means 把向量空间分成 `nlist` 个簇。
//! 查询时只搜其中 `nprobe` 个簇，剩下的一律不看。
//!
//! - `nlist` 越大 → 簇越细 → 越准但越慢
//! - `nprobe` 越大 → 搜的簇越多 → 越准但越慢
//!
//! **注意**：训练集至少要 `39 * nlist` 个点，否则 k-means 会警告
//! 且聚类质量很差（本项目实测过这个警告）。
//!
//! ### HNSW —— 图索引
//!
//! 把向量组织成多层图，查询时从顶层入口点贪心往下走。
//! 单查询是 O(log N)，且召回率可以调到很高（靠 `efSearch`）。
//!
//! ## 关键实现细节
//!
//! ### 为什么用 InnerProduct 而不是 L2
//!
//! 因为向量已经 L2 归一化了：余弦相似度 == 内积。
//! 归一化后两个向量的 L2 距离是 `sqrt(2 - 2*cos)`，
//! 是余弦相似度的单调变换 —— 排序完全一致，但内积直接给出相似度值，
//! 不用再转换。
//!
//! ### ID 映射用 `IDMap,` 前缀而不是 Rust 侧包装
//!
//! FAISS 默认给每个向量分配**顺序 ID**（0, 1, 2...）。
//! 我们想把 ID 换成稳定的整数（`{文件名}#{序号}` 哈希而来），
//! 方便引用和去重。
//!
//! 关键点：映射必须**存在 FAISS 索引内部**，才能跟着
//! `write_index` / `read_index` 一起持久化。
//! 如果在 Rust 侧用 `IdMap` 包装，序列化时只会写出被包装的裸索引，
//! ID 映射丢失 —— 加载后所有 label 都对不上记录。
//!
//! 所以用工厂描述串 `"IDMap,Flat"` 直接创建带映射的索引。
//!
//! ### 持久化要存两个文件
//!
//! FAISS 的 `write_index` 只写向量索引，**不写元数据**。
//! 所以本项目同时写一个 sidecar JSON（`*.meta.json`）存
//! `EmbeddedChunk` 的元数据部分。两边靠**相同的顺序**对齐。

use std::collections::HashMap;
use std::path::Path;

use faiss::index::IndexImpl;
use faiss::index::io::{read_index, write_index};
use faiss::{Idx, Index, MetricType, index_factory};
use serde::{Deserialize, Serialize};

use crate::embed::EmbeddedChunk;
use crate::store::{SearchHit, StoreError, VectorStore, flatten};

/// FAISS 索引类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKind {
    /// 暴力精确检索（Flat）。
    Flat,
    /// 倒排文件（IVF），参数是簇数 `nlist`。
    Ivf(u32),
    /// 分层图（HNSW），参数是每层的连接数 `M`。
    Hnsw(u32),
}

impl IndexKind {
    /// FAISS 的索引描述串（不含 `IDMap,` 前缀）。
    pub fn description(&self) -> String {
        match self {
            IndexKind::Flat => "Flat".to_string(),
            // IVF 的工厂语法是 `IVF{nlist},Flat` —— 必须带 `,Flat` 后缀，
            // 只写 `IVF4` 会被 FAISS 解析失败。
            IndexKind::Ivf(nlist) => format!("IVF{nlist},Flat"),
            IndexKind::Hnsw(m) => format!("HNSW{m}"),
        }
    }

    /// 带 ID 映射的完整描述串 —— 这才是真正交给 FAISS 的。
    ///
    /// `IDMap,` 前缀让索引自带 ID 映射，从而能随索引一起持久化。
    pub fn factory_description(&self) -> String {
        format!("IDMap,{}", self.description())
    }

    /// 从描述串解析。
    pub fn from_description(desc: &str) -> Option<Self> {
        // 兼容带不带 `IDMap,` 前缀两种写法。
        let desc = desc.strip_prefix("IDMap,").unwrap_or(desc);
        // IVF 的描述串是 `IVF{nlist},Flat` —— 去掉后缀再解析。
        let desc = desc.strip_suffix(",Flat").unwrap_or(desc);
        match desc {
            "Flat" => Some(IndexKind::Flat),
            other => {
                if let Some(rest) = other.strip_prefix("IVF") {
                    rest.parse().ok().map(IndexKind::Ivf)
                } else if let Some(rest) = other.strip_prefix("HNSW") {
                    rest.parse().ok().map(IndexKind::Hnsw)
                } else {
                    None
                }
            }
        }
    }
}

/// FAISS 索引的 sidecar 元数据。
///
/// FAISS 自己不知道我们的块长什么样，所以要单独存一份。
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexMeta {
    /// 模型名 —— 加载时检查，不一致就报错（避免静默返回垃圾）。
    model_name: String,
    /// 索引类型描述串。
    index_kind: String,
    /// 向量维度。
    dimension: usize,
    /// 记录顺序，与 FAISS 索引里的向量顺序一一对应。
    records: Vec<MetaRecord>,
}

/// 单条记录的元数据（向量本身存在 FAISS 里）。
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetaRecord {
    /// 内部的整数 label —— 与 FAISS 里存的 ID 一致。
    label: u64,
    id: String,
    source: String,
    index: usize,
    heading_path: Vec<String>,
    text: String,
}

/// FAISS 向量存储。
pub struct FaissStore {
    /// FAISS 索引（自带 ID 映射）。
    index: IndexImpl,
    /// 索引类型。
    kind: IndexKind,
    /// 模型名。
    model_name: String,
    /// 向量维度。
    dimension: usize,
    /// 字符串 ID → 内部 label。
    ///
    /// 用于去重：同一个 ID 重复写入时跳过。
    id_to_label: HashMap<String, u64>,
    /// label → 记录。检索时用它把 FAISS 返回的 label 换回元数据。
    label_to_record: HashMap<u64, MetaRecord>,
    /// 下一个可用的 label。
    next_label: u64,
}

impl std::fmt::Debug for FaissStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaissStore")
            .field("kind", &self.kind)
            .field("model", &self.model_name)
            .field("dimension", &self.dimension)
            .field("size", &self.label_to_record.len())
            .finish()
    }
}

impl FaissStore {
    /// 创建一个空索引。
    ///
    /// `dimension` 必须和嵌入模型一致 —— FAISS 的索引在创建时
    /// 就把维度写死了，之后传错维度会直接崩在底层。
    pub fn new(kind: IndexKind, dimension: usize, model_name: &str) -> Result<Self, StoreError> {
        let index = index_factory(
            dimension as u32,
            kind.factory_description(),
            MetricType::InnerProduct,
        )
        .map_err(|e| StoreError::Faiss(e.to_string()))?;

        Ok(Self {
            index,
            kind,
            model_name: model_name.to_string(),
            dimension,
            id_to_label: HashMap::new(),
            label_to_record: HashMap::new(),
            next_label: 0,
        })
    }

    /// 返回索引类型的描述串。
    pub fn index_description(&self) -> String {
        self.kind.description()
    }
}

impl VectorStore for FaissStore {
    fn name(&self) -> &str {
        "faiss"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.label_to_record.len()
    }

    async fn add(&mut self, chunks: &[EmbeddedChunk]) -> Result<(), StoreError> {
        if chunks.is_empty() {
            return Ok(());
        }

        // 维度检查 —— 这是最常见的「换了模型没重建索引」错误。
        for c in chunks {
            if c.vector.len() != self.dimension {
                return Err(StoreError::DimensionMismatch {
                    expected: self.dimension,
                    actual: c.vector.len(),
                });
            }
        }

        // 已存在的 ID → 跳过（upsert 语义），不重复写入。
        let mut fresh: Vec<&EmbeddedChunk> = Vec::new();
        let mut fresh_ids: Vec<Idx> = Vec::new();
        for c in chunks {
            if self.id_to_label.contains_key(&c.id) {
                continue;
            }
            let label = self.next_label;
            self.next_label += 1;

            self.id_to_label.insert(c.id.clone(), label);
            self.label_to_record.insert(
                label,
                MetaRecord {
                    label,
                    id: c.id.clone(),
                    source: c.source.clone(),
                    index: c.index,
                    heading_path: c.heading_path.clone(),
                    text: c.text.clone(),
                },
            );

            fresh.push(c);
            fresh_ids.push(Idx::new(label));
        }

        if fresh.is_empty() {
            return Ok(());
        }

        let vectors = flatten(&fresh.iter().map(|c| c.vector.clone()).collect::<Vec<_>>());

        // IVF 这类需要训练的索引，必须先 train 再 add。
        // 训练数据就是待写入的向量本身 —— 对教学规模（几百到几千条）
        // 足够 k-means 用了。
        if !self.index.is_trained() {
            self.index
                .train(&vectors)
                .map_err(|e| StoreError::Faiss(format!("索引训练失败: {e}")))?;
        }

        self.index
            .add_with_ids(&vectors, &fresh_ids)
            .map_err(|e| StoreError::Faiss(e.to_string()))?;

        Ok(())
    }

    async fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchHit>, StoreError> {
        // 顺序很重要：维度不匹配是**编程错误**，
        // 要比「索引为空」更早报出来 —— 否则空索引会掩盖真正的 bug。
        if query.len() != self.dimension {
            return Err(StoreError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        if self.label_to_record.is_empty() {
            return Err(StoreError::EmptyIndex);
        }

        let k = k.min(self.label_to_record.len());
        let result = self
            .index
            .search(query, k)
            .map_err(|e| StoreError::Faiss(e.to_string()))?;

        let distances = result.distances;
        let labels = result.labels;

        let mut hits = Vec::with_capacity(k);
        for (i, &label) in labels.iter().enumerate() {
            let label = label.get().unwrap_or(u64::MAX);
            let Some(rec) = self.label_to_record.get(&label) else {
                continue;
            };
            hits.push(SearchHit {
                id: rec.id.clone(),
                score: distances[i],
                source: rec.source.clone(),
                heading_path: rec.heading_path.clone(),
                text: rec.text.clone(),
            });
        }

        Ok(hits)
    }

    async fn save(&self, path: &Path) -> Result<(), StoreError> {
        // FAISS 索引文件。
        // `write_index` 的参数是 `AsRef<str>` 而不是 `AsRef<Path>` ——
        // 非 UTF-8 路径要显式报错，而不是静默丢数据。
        let path_str = path
            .to_str()
            .ok_or_else(|| StoreError::Faiss(format!("路径不是合法 UTF-8: {}", path.display())))?;
        write_index(&self.index, path_str).map_err(|e| StoreError::Faiss(e.to_string()))?;

        // sidecar 元数据。
        let meta = IndexMeta {
            model_name: self.model_name.clone(),
            index_kind: self.kind.description(),
            dimension: self.dimension,
            records: self.label_to_record.values().cloned().collect(),
        };
        let meta_path = meta_path_for(path);
        std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;

        Ok(())
    }

    async fn load(&mut self, path: &Path) -> Result<(), StoreError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| StoreError::Faiss(format!("路径不是合法 UTF-8: {}", path.display())))?;
        let index = read_index(path_str).map_err(|e| StoreError::Faiss(e.to_string()))?;

        let meta_path = meta_path_for(path);
        let raw = std::fs::read(&meta_path)?;
        let meta: IndexMeta = serde_json::from_slice(&raw)?;

        // 关键检查：模型不一致说明索引是旧模型建的，必须重建。
        if meta.model_name != self.model_name {
            return Err(StoreError::Faiss(format!(
                "索引是用模型 `{}` 建的，当前模型是 `{}`，必须重建索引",
                meta.model_name, self.model_name
            )));
        }
        if meta.dimension != self.dimension {
            return Err(StoreError::DimensionMismatch {
                expected: self.dimension,
                actual: meta.dimension,
            });
        }

        // 重建两张映射表。
        let mut id_to_label = HashMap::new();
        let mut label_to_record = HashMap::new();
        let mut next_label = 0u64;
        for rec in &meta.records {
            id_to_label.insert(rec.id.clone(), rec.label);
            next_label = next_label.max(rec.label + 1);
            label_to_record.insert(rec.label, rec.clone());
        }

        self.index = index;
        self.kind = IndexKind::from_description(&meta.index_kind).unwrap_or(self.kind);
        self.id_to_label = id_to_label;
        self.label_to_record = label_to_record;
        self.next_label = next_label;

        Ok(())
    }
}

/// sidecar 元数据文件的路径。
fn meta_path_for(index_path: &Path) -> std::path::PathBuf {
    let mut s = index_path.as_os_str().to_os_string();
    s.push(".meta.json");
    std::path::PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, vector: Vec<f32>) -> EmbeddedChunk {
        EmbeddedChunk {
            id: id.to_string(),
            source: "test.md".into(),
            index: 0,
            heading_path: vec!["标题".into()],
            text: format!("文本 {id}"),
            vector,
        }
    }

    #[tokio::test]
    async fn index_kind_roundtrips_through_description() {
        for kind in [IndexKind::Flat, IndexKind::Ivf(64), IndexKind::Hnsw(32)] {
            let desc = kind.description();
            assert_eq!(IndexKind::from_description(&desc), Some(kind));
            // 带前缀的写法也要能解析回来。
            assert_eq!(
                IndexKind::from_description(&kind.factory_description()),
                Some(kind)
            );
        }
    }

    #[tokio::test]
    async fn flat_index_finds_exact_match() {
        let mut store = FaissStore::new(IndexKind::Flat, 3, "test-model").unwrap();
        store
            .add(&[
                chunk("a", vec![1.0, 0.0, 0.0]),
                chunk("b", vec![0.0, 1.0, 0.0]),
                chunk("c", vec![0.0, 0.0, 1.0]),
            ])
            .await
            .unwrap();

        let hits = store.search(&[0.99, 0.01, 0.0], 2).await.unwrap();
        assert_eq!(hits[0].id, "a");
        assert!(hits[0].score > hits[1].score);
    }

    #[tokio::test]
    async fn dimension_mismatch_is_rejected() {
        let mut store = FaissStore::new(IndexKind::Flat, 3, "test-model").unwrap();
        let err = store.add(&[chunk("a", vec![1.0, 0.0])]).await.unwrap_err();
        assert!(matches!(err, StoreError::DimensionMismatch { .. }));

        let err = store.search(&[1.0, 0.0], 1).await.unwrap_err();
        assert!(matches!(err, StoreError::DimensionMismatch { .. }));
    }

    #[tokio::test]
    async fn searching_empty_index_errors() {
        let mut store = FaissStore::new(IndexKind::Flat, 3, "test-model").unwrap();
        assert!(matches!(
            store.search(&[1.0, 0.0, 0.0], 1).await.unwrap_err(),
            StoreError::EmptyIndex
        ));
    }

    #[tokio::test]
    async fn duplicate_ids_are_not_added_twice() {
        let mut store = FaissStore::new(IndexKind::Flat, 3, "test-model").unwrap();
        store.add(&[chunk("a", vec![1.0, 0.0, 0.0])]).await.unwrap();
        store.add(&[chunk("a", vec![1.0, 0.0, 0.0])]).await.unwrap();
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join("rag05-faiss-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("idx.faiss");

        let mut store = FaissStore::new(IndexKind::Flat, 3, "test-model").unwrap();
        store
            .add(&[
                chunk("a", vec![1.0, 0.0, 0.0]),
                chunk("b", vec![0.0, 1.0, 0.0]),
            ])
            .await
            .unwrap();
        store.save(&path).await.unwrap();

        let mut loaded = FaissStore::new(IndexKind::Flat, 3, "test-model").unwrap();
        loaded.load(&path).await.unwrap();

        assert_eq!(loaded.len(), 2);
        let hits = loaded.search(&[1.0, 0.0, 0.0], 1).await.unwrap();
        assert_eq!(hits[0].id, "a");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn loading_index_from_other_model_is_rejected() {
        let dir = std::env::temp_dir().join("rag05-faiss-modelmismatch");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("idx.faiss");

        let mut store = FaissStore::new(IndexKind::Flat, 3, "model-a").unwrap();
        store.add(&[chunk("a", vec![1.0, 0.0, 0.0])]).await.unwrap();
        store.save(&path).await.unwrap();

        let mut other = FaissStore::new(IndexKind::Flat, 3, "model-b").unwrap();
        assert!(other.load(&path).await.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn many_records_keep_correct_label_mapping() {
        // 曾经的 bug：ID → label 的映射用 HashMap 存，
        // 遍历顺序不确定，导致检索结果和记录错位。
        // 这个测试用 50 条**方向各不相同**的向量验证映射是一一对应的。
        let mut store = FaissStore::new(IndexKind::Flat, 4, "test-model").unwrap();

        let chunks: Vec<EmbeddedChunk> = (0..50)
            .map(|i| {
                // 第 0 维放主导值，保证 50 个向量两两方向不同
                // （同方向的向量余弦相似度都是 1，无法区分谁是谁）。
                let mut v = vec![0.0f32; 4];
                v[0] = 1.0;
                v[1] = i as f32 / 50.0;
                // ⚠️ 必须先 L2 归一化：索引用的是 InnerProduct，
                // 未归一化时内积会被模长主导 —— 排序变成「谁的模长大谁赢」，
                // 而不是「谁的方向最接近」。这正是生产流水线上
                // 所有向量都要归一化的原因。
                crate::embed::l2_normalize(&mut v);
                chunk(&format!("c{i}"), v)
            })
            .collect();
        store.add(&chunks).await.unwrap();

        for (i, c) in chunks.iter().enumerate() {
            let hits = store.search(&c.vector, 1).await.unwrap();
            assert_eq!(hits[0].id, c.id, "第 {i} 条检索到了错误的记录");
        }
    }

    #[tokio::test]
    async fn unnormalized_vectors_rank_by_magnitude_not_direction() {
        // 反向验证上面那条性质：未归一化时，内积排序会被模长主导。
        // 这个测试记录了「为什么必须归一化」这个事实。
        let mut store = FaissStore::new(IndexKind::Flat, 2, "test-model").unwrap();

        // 三个向量：
        // - short：模长 1，方向 [1, 0.01]（和查询完全同向）
        // - long：模长 10，方向 [1, 0.02]（和查询几乎同向，但模长大 10 倍）
        // - other：模长 1，方向正交（和查询差 90 度）
        let a = vec![1.0f32, 0.01];
        let b = vec![10.0f32, 0.2];
        let other = vec![0.0f32, 1.0];
        store
            .add(&[
                chunk("short", a.clone()),
                chunk("long", b.clone()),
                chunk("other", other.clone()),
            ])
            .await
            .unwrap();

        // 查询和「short」「long」同向。
        let query = vec![1.0f32, 0.01];

        // 未归一化时：「long」的内积 ≈ 10，「short」≈ 1 —— 模长主导排序。
        let hits = store.search(&query, 3).await.unwrap();
        assert_eq!(hits[0].id, "long", "未归一化时内积被模长主导");

        // 归一化后：方向说了算。「short」和查询完全同向，得分 1.0；
        // 「long」方向略有偏差，得分略低 —— 模长不再影响排序。
        let mut q = query.clone();
        crate::embed::l2_normalize(&mut q);
        let mut a_n = a.clone();
        let mut b_n = b.clone();
        let mut other_n = other.clone();
        crate::embed::l2_normalize(&mut a_n);
        crate::embed::l2_normalize(&mut b_n);
        crate::embed::l2_normalize(&mut other_n);

        let mut store2 = FaissStore::new(IndexKind::Flat, 2, "test-model").unwrap();
        store2
            .add(&[
                chunk("short", a_n),
                chunk("long", b_n),
                chunk("other", other_n),
            ])
            .await
            .unwrap();

        let hits = store2.search(&q, 3).await.unwrap();
        assert_eq!(hits[0].id, "short", "归一化后按方向排序");

        let by_id = |id: &str| hits.iter().find(|h| h.id == id).unwrap();
        // 「long」和「short」方向接近，归一化后得分也接近 1。
        assert!(
            by_id("long").score > 0.99,
            "同方向的向量归一化后得分应接近 1: {}",
            by_id("long").score
        );
        // 正交的向量得分接近 0。
        // （`other` = [0, 1] 与查询 [1, 0.01] 并非严格正交，
        //  所以得分是 ~0.01 而不是 0 —— 但依然远低于同向的 ~1.0。）
        assert!(
            by_id("other").score.abs() < 0.05,
            "垂直方向的向量得分应接近 0: {}",
            by_id("other").score
        );
    }
}
