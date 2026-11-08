//! 评估集（Evaluation Dataset）—— 评估的「地面真相」。
//!
//! ## 一条评估样本长什么样
//!
//! ```json
//! {
//!   "id": "bm25-formula",
//!   "question": "BM25 打分公式是什么",
//!   "reference_answer": "BM25 由 IDF、词频饱和和长度归一化三部分组成……",
//!   "reference_contexts": ["BM25 是最经典的稀疏检索算法……"],
//!   "relevant_chunk_ids": ["kb.md#12"]
//! }
//! ```
//!
//! 四个字段各有用处：
//!
//! | 字段 | 喂给哪个指标 | 缺了会怎样 |
//! |---|---|---|
//! | `question` | 全部 | 没得评 |
//! | `reference_answer` | answer relevancy、token F1、correctness | 只能评检索侧 |
//! | `reference_contexts` | context precision / recall / relevancy | 只能评答案侧 |
//! | `relevant_chunk_ids` | recall@k、nDCG、MRR | 检索指标无法计算 |
//!
//! ## ⚠️ 为什么 `relevant_chunk_ids` 是这个项目里最难的部分
//!
//! 检索指标（recall@k / nDCG / MRR）需要知道「**哪些块是相关的**」。
//!
//! 这不是人工标起来就完事 —— 它有一个**结构性依赖**：
//!
//! ```text
//! 块 ID = "{文件名}#{块序号}"
//!        ▲           ▲
//!        │           └── 由「分块配置」决定！
//!        └── 来自文档加载
//! ```
//!
//! 块序号是**分块的产物** —— 改一下 `chunk_size`，同一个段落的块序号就变了，
//! 评估集里所有 `relevant_chunk_ids` 全部失效。
//!
//! 所以本模块提供两种写法：
//!
//! 1. **直接写块 ID**（快，但要保证分块配置不变）
//! 2. **写「相关章节标题」**（稳）—— 运行时按当前分块配置翻译成块 ID
//!
//! 第 2 种是这个项目的核心机制，见 [`EvalCase::resolve_relevant_ids`]。

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{EvalError, EvalResult};

/// 一条评估样本。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase {
    /// 样本 ID（唯一）。
    pub id: String,

    /// 用户问题。
    pub question: String,

    /// 标准答案（人工写好的「理想答案」）。
    ///
    /// ⚠️ 它**不是**唯一正确答案 —— 只是评估的参照物。
    /// 所以 `answer relevancy` 用的不是「答案和标准答案一样吗」
    /// （那是 exact match，太严格），而是「答案和**问题**相关吗」。
    #[serde(default)]
    pub reference_answer: String,

    /// 标准上下文（人工判断「理想情况下应该检索到什么」）。
    #[serde(default)]
    pub reference_contexts: Vec<String>,

    /// 相关块的 ID（`{文件名}#{序号}`）。
    #[serde(default)]
    pub relevant_chunk_ids: Vec<String>,

    /// 相关块的**章节标题**。
    ///
    /// ⚠️ 和 `relevant_chunk_ids` 二选一（或同时给，取并集）。
    ///
    /// 为什么需要它：块 ID 依赖分块配置，改一下 `chunk_size` 就全失效。
    /// 章节标题则**和分块配置无关** —— 运行时按当前分块重新翻译即可。
    #[serde(default)]
    pub relevant_sections: Vec<String>,
}

impl EvalCase {
    /// 构造一条评估样本。
    pub fn new(id: impl Into<String>, question: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            question: question.into(),
            reference_answer: String::new(),
            reference_contexts: Vec::new(),
            relevant_chunk_ids: Vec::new(),
            relevant_sections: Vec::new(),
        }
    }

    /// 带上标准答案。
    pub fn with_answer(mut self, answer: impl Into<String>) -> Self {
        self.reference_answer = answer.into();
        self
    }

    /// 带上标准上下文。
    pub fn with_contexts(mut self, contexts: Vec<String>) -> Self {
        self.reference_contexts = contexts;
        self
    }

    /// 带上相关块 ID。
    pub fn with_relevant_ids(mut self, ids: Vec<String>) -> Self {
        self.relevant_chunk_ids = ids;
        self
    }

    /// 带上相关章节标题。
    pub fn with_sections(mut self, sections: Vec<String>) -> Self {
        self.relevant_sections = sections;
        self
    }

    /// 把「相关章节标题」翻译成「相关块 ID」。
    ///
    /// ## 翻译规则
    ///
    /// 一个块如果它的**标题路径里包含**任一相关章节标题，就算相关。
    ///
    /// 为什么用「包含」而不是「完全相等」：
    /// 知识库的标题路径是 `["检索增强生成", "文档分块"]` 这样的层级，
    /// 写「文档分块」应该能命中这个块。
    ///
    /// ## 为什么这比直接写块 ID 更好
    ///
    /// 块 ID 是分块的**产物**，标题是文档的**事实**。
    /// 用标题标注，改 `chunk_size` 不会让评估集失效。
    ///
    /// ## 返回值
    ///
    /// 翻译结果的**并集**：直接写的 ID + 标题翻译出来的 ID。
    pub fn resolve_relevant_ids(&self, chunk_index: &ChunkIndex) -> Vec<String> {
        let mut resolved: Vec<String> = self.relevant_chunk_ids.clone();

        for section in &self.relevant_sections {
            for id in chunk_index.ids_by_section(section) {
                if !resolved.contains(&id) {
                    resolved.push(id);
                }
            }
        }

        resolved
    }
}

/// 评估集。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDataset {
    /// 评估集名称（报告标题用）。
    #[serde(default = "default_name")]
    pub name: String,

    /// 评估样本。
    pub cases: Vec<EvalCase>,
}

fn default_name() -> String {
    "rag-eval".to_string()
}

impl EvalDataset {
    /// 构造评估集。
    pub fn new(name: impl Into<String>, cases: Vec<EvalCase>) -> Self {
        Self {
            name: name.into(),
            cases,
        }
    }

    /// 样本数。
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// 空评估集。
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }

    /// 按 ID 找样本。
    pub fn find(&self, id: &str) -> Option<&EvalCase> {
        self.cases.iter().find(|c| c.id == id)
    }

    /// 从 JSON 文件加载。
    ///
    /// 支持两种格式：
    ///
    /// ```json
    /// { "name": "kb", "cases": [ ... ] }
    /// ```
    ///
    /// 或直接是数组：
    ///
    /// ```json
    /// [ { "id": "...", ... } ]
    /// ```
    pub fn load(path: &Path) -> EvalResult<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| EvalError::Dataset(format!("读取评估集失败 {}: {e}", path.display())))?;

        let trimmed = raw.trim_start();
        if trimmed.starts_with('[') {
            let cases: Vec<EvalCase> = serde_json::from_str(&raw)
                .map_err(|e| EvalError::Dataset(format!("解析评估集失败: {e}")))?;
            Ok(Self::new(default_name(), cases))
        } else {
            serde_json::from_str(&raw)
                .map_err(|e| EvalError::Dataset(format!("解析评估集失败: {e}")))
        }
    }

    /// 保存到 JSON 文件。
    pub fn save(&self, path: &Path) -> EvalResult<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// 校验评估集（ID 唯一、问题非空）。
    ///
    /// ## 为什么需要显式校验
    ///
    /// 一个重复的 ID 会让**报告里的样本对不上号** —— 你看到某条低分，
    /// 但不知道是哪个问题产生的。这比没有报告更糟。
    pub fn validate(&self) -> EvalResult<()> {
        if self.cases.is_empty() {
            return Err(EvalError::Dataset("评估集为空".to_string()));
        }

        let mut seen: Vec<&str> = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            if case.question.trim().is_empty() {
                return Err(EvalError::Dataset(format!("样本 {} 的问题为空", case.id)));
            }
            if seen.contains(&case.id.as_str()) {
                return Err(EvalError::Dataset(format!("样本 ID 重复: {}", case.id)));
            }
            seen.push(&case.id);
        }

        Ok(())
    }
}

/// 块索引 —— 「章节标题 → 块 ID」的映射。
///
/// ## 为什么需要它
///
/// 检索指标需要「相关块 ID」来算 recall@k / nDCG / MRR。
/// 而块 ID 依赖分块配置 —— 所以要在**运行时按当前分块**重新建立映射。
///
/// ```text
/// 文档 ──Splitter──> 块（含标题路径）
///                        │
///                        ├──> id_by_section:  "文档分块" -> ["kb.md#1", "kb.md#2"]
///                        └──> id_by_id:       "kb.md#1"  -> "文档分块"
/// ```
pub struct ChunkIndex {
    /// 章节标题（规范化后）→ 块 ID 列表。
    ids_by_section: HashMap<String, Vec<String>>,

    /// 块 ID → 所属章节标题（规范化后）。
    section_by_id: HashMap<String, String>,
}

impl std::fmt::Debug for ChunkIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkIndex")
            .field("sections", &self.ids_by_section.len())
            .field("chunks", &self.section_by_id.len())
            .finish()
    }
}

impl ChunkIndex {
    /// 建索引。
    ///
    /// `chunks` 是 `(块 ID, 标题路径)` 的列表。
    pub fn build(chunks: &[(String, Vec<String>)]) -> Self {
        let mut ids_by_section: HashMap<String, Vec<String>> = HashMap::new();
        let mut section_by_id: HashMap<String, String> = HashMap::new();

        for (id, heading_path) in chunks {
            for heading in heading_path {
                let key = normalize_section(heading);
                ids_by_section.entry(key).or_default().push(id.clone());
            }
            // 用**最深的标题**作为该块的主章节 —— 命中它比命中父标题更精确。
            if let Some(last) = heading_path.last() {
                section_by_id.insert(id.clone(), normalize_section(last));
            }
        }

        Self {
            ids_by_section,
            section_by_id,
        }
    }

    /// 某个章节标题下的所有块 ID。
    ///
    /// 找不到就返回空 —— **不报错**。
    ///
    /// 理由：标注里的章节名可能和知识库不完全一致（手抖多打一个字）。
    /// 这种情况应该表现为「这个样本没有相关块」，
    /// 由调用方决定怎么处理（跳过该样本，或报告 0 分），
    /// 而不是整个评估挂掉。
    pub fn ids_by_section(&self, section: &str) -> Vec<String> {
        let key = normalize_section(section);
        self.ids_by_section.get(&key).cloned().unwrap_or_default()
    }

    /// 块 ID 所属的章节标题。
    pub fn section_by_id(&self, id: &str) -> Option<&str> {
        self.section_by_id.get(id).map(|s| s.as_str())
    }

    /// 索引里的块总数。
    pub fn chunk_count(&self) -> usize {
        self.section_by_id.len()
    }

    /// 索引里的章节总数。
    pub fn section_count(&self) -> usize {
        self.ids_by_section.len()
    }
}

/// 规范化章节标题 —— 用于大小写和空白差异的匹配。
fn normalize_section(section: &str) -> String {
    section.trim().to_lowercase().replace(' ', "")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_index() -> ChunkIndex {
        ChunkIndex::build(&[
            ("kb.md#0".to_string(), vec!["检索增强生成".to_string()]),
            (
                "kb.md#1".to_string(),
                vec!["文档分块".to_string(), "块大小".to_string()],
            ),
            (
                "kb.md#2".to_string(),
                vec!["文档分块".to_string(), "重叠区域".to_string()],
            ),
        ])
    }

    #[test]
    fn section_lookup_returns_matching_ids() {
        let index = chunk_index();
        let ids = index.ids_by_section("文档分块");

        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"kb.md#1".to_string()));
        assert!(ids.contains(&"kb.md#2".to_string()));
    }

    #[test]
    fn section_lookup_is_case_and_space_insensitive() {
        let index = chunk_index();
        assert_eq!(index.ids_by_section(" 文档分块 ").len(), 2);
    }

    #[test]
    fn unknown_section_returns_empty_not_error() {
        let index = chunk_index();
        assert!(index.ids_by_section("不存在的章节").is_empty());
    }

    #[test]
    fn resolves_sections_and_ids_as_union() {
        let case = EvalCase::new("c1", "问题")
            .with_relevant_ids(vec!["kb.md#0".to_string()])
            .with_sections(vec!["文档分块".to_string()]);

        let ids = case.resolve_relevant_ids(&chunk_index());

        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"kb.md#0".to_string()));
        assert!(ids.contains(&"kb.md#1".to_string()));
    }

    #[test]
    fn dataset_validation_rejects_duplicates() {
        let dataset = EvalDataset::new(
            "test",
            vec![EvalCase::new("a", "q1"), EvalCase::new("a", "q2")],
        );

        let err = dataset.validate().unwrap_err();
        assert!(err.to_string().contains("重复"));
    }

    #[test]
    fn dataset_validation_rejects_empty_question() {
        let dataset = EvalDataset::new("test", vec![EvalCase::new("a", "  ")]);
        assert!(dataset.validate().is_err());
    }

    #[test]
    fn dataset_validation_rejects_empty_dataset() {
        assert!(EvalDataset::new("test", Vec::new()).validate().is_err());
    }

    #[test]
    fn dataset_round_trips_through_json() {
        let dataset = EvalDataset::new("kb", vec![EvalCase::new("c1", "问题").with_answer("答案")]);

        let json = serde_json::to_string(&dataset).unwrap();
        let parsed: EvalDataset = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "kb");
        assert_eq!(parsed.cases[0].reference_answer, "答案");
    }

    #[test]
    fn dataset_loads_from_bare_array() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cases.json");
        std::fs::write(&path, r#"[{"id":"a","question":"q"}]"#).unwrap();

        let dataset = EvalDataset::load(&path).unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.cases[0].id, "a");
    }
}
