//! 语料块 → 向量库记录的映射。
//!
//! 05-rag 的 [`EmbeddedChunk`] 是
//! **Rust 侧**的结构 —— 它假设索引和元数据都在同一个进程里。
//!
//! 但生产向量库是**独立服务**：向量在 Milvus 那边，
//! 我们这边只剩「哪些 id 对应哪些原文」这件事。
//!
//! ## 为什么要做这层映射
//!
//! 因为 Milvus / Qdrant 的主键必须是**标量**（整数或字符串），
//! 而我们的 chunk id 是 `{文件名}#{序号}` —— 字符串，恰好能用。
//!
//! 但 Milvus 的主键还有个额外约束：**不能开启 auto_id**，
//! 否则服务端会自己生成主键，我们就无法用稳定 id 做 upsert 了。
//!
//! 所以 schema 设计成：
//!
//! ```text
//! pk         Int64      主键（自增序号，稳定）
//! chunk_id   VarChar    业务 ID（`{文件名}#{序号}`）—— 真正的身份
//! source     VarChar    来源文件
//! heading    JSON       标题路径（数组）
//! text       VarChar    原文
//! vector     FloatVector  稠密向量
//! sparse     SparseFloatVector  BM25 稀疏向量（服务端生成）
//! ```
//!
//! 为什么同时要 `pk` 和 `chunk_id`？
//!
//! - `pk` 是 Milvus 的主键 —— 必须存在，且类型受限
//! - `chunk_id` 是**业务身份** —— 换库、重建索引后仍然稳定
//!
//! 这跟数据库里「自增 ID」和「业务主键」的区别是一样的。

use rag_05::embed::EmbeddedChunk;
use serde::{Deserialize, Serialize};
use sha1::Digest;

/// 一条落库的语料块记录。
///
/// 这是在「Rust 世界」和「向量库世界」之间的中转结构：
/// 写入时转成 Milvus 的 `FieldData` / Qdrant 的 `PointStruct`，
/// 读出时转回 05-rag 的 [`SearchHit`](rag_05::store::SearchHit)。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    /// 稳定业务 ID：`{文件名}#{序号}`。
    pub chunk_id: String,

    /// 来源文件名。
    pub source: String,

    /// 块在文档中的序号。
    pub index: usize,

    /// 标题路径。
    pub heading_path: Vec<String>,

    /// 块正文。
    pub text: String,

    /// 稠密向量。
    pub vector: Vec<f32>,
}

impl ChunkRecord {
    /// 一行摘要。
    pub fn summary(&self) -> String {
        let head = if self.heading_path.is_empty() {
            "(无标题)".to_string()
        } else {
            self.heading_path.join(" > ")
        };
        let preview: String = self.text.chars().take(40).collect();
        format!(
            "{} [{}] dim={} | {}",
            self.chunk_id,
            head,
            self.vector.len(),
            preview.replace('\n', " ")
        )
    }
}

/// 从 05-rag 的 `EmbeddedChunk` 转成可落库的记录。
pub fn chunk_to_record(chunk: &EmbeddedChunk) -> ChunkRecord {
    ChunkRecord {
        chunk_id: chunk.id.clone(),
        source: chunk.source.clone(),
        index: chunk.index,
        heading_path: chunk.heading_path.clone(),
        text: chunk.text.clone(),
        vector: chunk.vector.clone(),
    }
}

/// 批量转换。
pub fn records_from_chunks(chunks: &[EmbeddedChunk]) -> Vec<ChunkRecord> {
    chunks.iter().map(chunk_to_record).collect()
}

/// 从业务 ID 派生一个**稳定的 Int64 主键**。
///
/// Milvus 的主键是 Int64，而我们的业务 ID 是字符串
/// （`kb.md#3`）。如果每次灌库都从 0 开始自增，
/// 同一个 `chunk_id` 每次都会拿到**不同的主键** ——
/// 于是 upsert 退化成 insert，库里堆出一份份重复数据
/// （实测跑 5 次 demo，`count(*)` 就是 55 = 11 × 5）。
///
/// 解决办法：主键也用稳定派生。
///
/// - 输入：`chunk_id` 字符串
/// - 输出：SHA-1 的前 8 字节，取正 63 位（保证非负，避开符号位）
///
/// 性质：
///
/// - **确定性** —— 同一个 `chunk_id` 永远得到同一个主键
/// - **幂等** —— 重复灌库时 upsert 覆盖而不是新增
/// - **无冲突** —— 63 位空间，语料规模到 10^18 都不会撞
///
/// 这和 Qdrant 侧的 [`stable_uuid`](crate::qdrant_store) 是同一个思路：
/// **向量库的主键必须由业务 ID 派生，而不是由写入顺序决定**。
pub fn stable_pk(chunk_id: &str) -> i64 {
    let digest = sha1::Sha1::digest(chunk_id.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);

    // 只保留 63 位 —— 最高位置 0，保证结果非负。
    (u64::from_be_bytes(bytes) >> 1) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> EmbeddedChunk {
        EmbeddedChunk {
            id: "kb.md#3".to_string(),
            source: "kb.md".to_string(),
            index: 3,
            heading_path: vec!["RAG".to_string(), "分块".to_string()],
            text: "正文".to_string(),
            vector: vec![0.1, 0.2, 0.3],
        }
    }

    #[test]
    fn record_keeps_business_id() {
        let record = chunk_to_record(&sample());
        assert_eq!(record.chunk_id, "kb.md#3");
        assert_eq!(record.index, 3);
        assert_eq!(record.heading_path.len(), 2);
    }

    #[test]
    fn batch_conversion_preserves_order() {
        let chunks = vec![sample(), sample()];
        let records = records_from_chunks(&chunks);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].chunk_id, "kb.md#3");
    }

    #[test]
    fn stable_pk_is_deterministic_and_non_negative() {
        let a = stable_pk("kb.md#3");
        let b = stable_pk("kb.md#3");
        let c = stable_pk("kb.md#4");

        // 确定性 —— upsert 的前提。
        assert_eq!(a, b);
        // 不同 chunk 得到不同主键。
        assert_ne!(a, c);
        // 非负 —— 避免符号位带来的意外。
        assert!(a >= 0);
        assert!(c >= 0);
    }
}
