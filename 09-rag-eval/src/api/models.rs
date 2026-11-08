//! 请求 / 响应模型 —— API 契约。
//!
//! 单独放一个文件，是为了让「服务能接什么、返回什么」一眼可见。

use serde::{Deserialize, Serialize};

use crate::pipeline::EvalVariant;
use crate::report::ComparisonReport;

/// 评估请求体。
///
/// ```json
/// {
///   "dataset": "data/eval_cases.json",
///   "variants": ["vector", "hybrid"],
///   "metrics": ["token_f1", "faithfulness"],
///   "case_ids": ["bm25-formula"]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct EvalRequest {
    /// 评估集路径（可选，默认用配置里的）。
    #[serde(default)]
    pub dataset: Option<String>,

    /// 要评估的变体（可选，默认全部）。
    #[serde(default)]
    pub variants: Option<Vec<String>>,

    /// 要计算的指标（可选，默认全部）。
    #[serde(default)]
    pub metrics: Option<Vec<String>>,

    /// 只评估这些样本 ID（可选，默认全部）。
    #[serde(default)]
    pub case_ids: Option<Vec<String>>,
}

/// 单个变体的评估结果。
#[derive(Debug, Clone, Serialize)]
pub struct VariantReport {
    /// 变体名。
    pub variant: String,

    /// 评估报告。
    pub report: crate::report::EvalReport,

    /// 生成失败的样本数。
    pub failed_cases: usize,
}

/// 评估响应体。
#[derive(Debug, Clone, Serialize)]
pub struct EvalResponse {
    /// 评估集名称。
    pub dataset: String,

    /// 每个变体的报告。
    pub variants: Vec<VariantReport>,

    /// 对比报告（第一个变体 vs 最后一个变体）。
    ///
    /// ⚠️ 只有在跑了**多个**变体时才有值。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comparison: Option<ComparisonReport>,

    /// 总耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 快速评估请求体 —— 问一个问题，立刻拿到分数。
///
/// ## 为什么需要这个端点
///
/// 因为调试单个指标时，「跑完整评估集 → 看报告 → 找那条样本」的路径太慢。
/// 这个端点让**一条样本 + 指定指标**在几秒内出结果。
///
/// ⚠️ 但要清楚：单样本分数**不能**用来做优化决策 ——
/// 一个样本的波动可能很大。做决策要看**整个评估集**的聚合分数。
#[derive(Debug, Clone, Deserialize)]
pub struct QuickEvalRequest {
    /// 问题。
    pub question: String,

    /// 标准答案（可选）。
    #[serde(default)]
    pub reference_answer: Option<String>,

    /// 相关章节标题（可选）。
    #[serde(default)]
    pub relevant_sections: Option<Vec<String>>,

    /// 检索模式（可选，默认 hybrid）。
    #[serde(default)]
    pub mode: Option<String>,

    /// 要计算的指标（可选，默认全部）。
    #[serde(default)]
    pub metrics: Option<Vec<String>>,
}

/// 单个指标的快速评估结果。
#[derive(Debug, Clone, Serialize)]
pub struct QuickMetricResult {
    /// 指标名。
    pub name: String,

    /// 得分。
    pub value: f32,

    /// 判定理由。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,

    /// 是否达到阈值。
    pub passed: bool,
}

/// 快速评估响应体。
#[derive(Debug, Clone, Serialize)]
pub struct QuickEvalResponse {
    /// 问题。
    pub question: String,

    /// 生成的答案。
    pub answer: String,

    /// 检索到的块 ID。
    pub retrieved_ids: Vec<String>,

    /// 各指标得分。
    pub metrics: Vec<QuickMetricResult>,

    /// 耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// 评估集信息响应体。
#[derive(Debug, Clone, Serialize)]
pub struct DatasetInfo {
    /// 评估集名称。
    pub name: String,

    /// 样本数。
    pub case_count: usize,

    /// 样本摘要。
    pub cases: Vec<DatasetCaseInfo>,

    /// 块索引统计。
    pub chunk_index: ChunkIndexStats,
}

/// 单个样本的摘要。
#[derive(Debug, Clone, Serialize)]
pub struct DatasetCaseInfo {
    /// 样本 ID。
    pub id: String,

    /// 问题。
    pub question: String,

    /// 有没有标准答案。
    pub has_reference_answer: bool,

    /// 有没有相关块标注。
    pub has_relevant: bool,

    /// 解析出的相关块 ID 数量。
    pub resolved_relevant_count: usize,
}

/// 块索引统计。
#[derive(Debug, Clone, Serialize)]
pub struct ChunkIndexStats {
    /// 块总数。
    pub chunk_count: usize,

    /// 章节总数。
    pub section_count: usize,
}

/// 错误响应体。
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// 错误类别（HTTP 状态码的规范名）。
    pub error: String,

    /// 错误详情。
    pub message: String,
}

/// 解析检索模式（缺省 hybrid）。
pub fn resolve_mode(raw: Option<&String>) -> Result<Option<rag_08::hybrid::SearchMode>, String> {
    match raw {
        None => Ok(None),
        Some(value) => rag_08::hybrid::SearchMode::parse(value)
            .map(Some)
            .ok_or_else(|| format!("未知的检索模式: {value}（可选: vector / bm25 / hybrid）")),
    }
}

/// 解析变体列表。
pub fn resolve_variants(raw: Option<&Vec<String>>) -> Result<Vec<EvalVariant>, String> {
    match raw {
        None => Ok(EvalVariant::all()),
        Some(list) => {
            let mut parsed = Vec::with_capacity(list.len());
            for name in list {
                parsed.push(EvalVariant::parse(name).ok_or_else(|| format!("未知的变体: {name}"))?);
            }
            if parsed.is_empty() {
                Ok(EvalVariant::all())
            } else {
                Ok(parsed)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_variants_from_strings() {
        let variants =
            resolve_variants(Some(&vec!["vector".to_string(), "hybrid".to_string()])).unwrap();
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0], EvalVariant::Vector);
    }

    #[test]
    fn rejects_unknown_variant() {
        let err = resolve_variants(Some(&vec!["不存在".to_string()])).unwrap_err();
        assert!(err.contains("不存在"));
    }

    #[test]
    fn empty_variant_list_falls_back_to_all() {
        let variants = resolve_variants(Some(&Vec::new())).unwrap();
        assert_eq!(variants.len(), EvalVariant::all().len());
    }

    #[test]
    fn resolves_search_mode() {
        assert_eq!(
            resolve_mode(Some(&"hybrid".to_string())).unwrap(),
            Some(rag_08::hybrid::SearchMode::Hybrid)
        );
        assert!(resolve_mode(Some(&"unknown".to_string())).is_err());
        assert_eq!(resolve_mode(None).unwrap(), None);
    }
}
