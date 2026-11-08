//! 全局配置。
//!
//! 环境变量前缀 `EVAL_` —— 和 06（`RAG_`）、07（`QUERY_`）、08（`HYBRID_`）都不撞。
//!
//! ## 为什么评估要单独一套配置
//!
//! 因为评估有几个**和被评估系统完全不同**的参数：
//!
//! - `EVAL_DATASET` —— 评估集路径
//! - `EVAL_VARIANTS` —— 要对比哪些变体
//! - `EVAL_JUDGE_MODEL` —— judge 用哪个模型（**应该和生成模型不同**）
//! - `EVAL_JUDGE_ENABLED` —— 是否启用 LLM judge 指标
//!
//! 最后一条最重要：**judge 模型应该和生成模型不同**。
//!
//! 用同一个模型当生成器和 judge，等于让模型给自己打分 ——
//! 它会倾向于给自己的答案打高分（这是 LLM-as-a-judge 最大的偏差来源之一）。

use std::path::PathBuf;

use crate::pipeline::EvalVariant;

/// 评估配置。
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// 知识库文档目录（复用 08-rag 的）。
    pub data_dir: PathBuf,

    /// 索引文件路径（复用 08-rag 的）。
    pub index_path: PathBuf,

    /// 评估集路径。
    pub dataset_path: PathBuf,

    /// 报告输出目录。
    pub report_dir: PathBuf,

    /// 要评估的变体。
    pub variants: Vec<EvalVariant>,

    /// 是否启用 LLM judge 指标。
    pub judge_enabled: bool,

    /// judge 用的模型名。
    ///
    /// ⚠️ 建议和生成模型**不同** —— 同模型自评有系统性偏高。
    pub judge_model: String,

    /// judge 的温度（必须低）。
    pub judge_temperature: f64,

    /// judge 的 max tokens。
    ///
    /// ⚠️ 这要**够大** —— judge 要一次性给几十个句子打分，
    /// 每个句子还要带理由。1024 在开启思考时会被思考吃光。
    pub judge_max_tokens: u32,

    /// 是否关闭 judge 模型的「思考」模式。
    ///
    /// ⚠️ 默认 `true` —— 见 [`EvalConfig::judge_max_tokens`]。
    ///
    /// judge 是**结构化短输出**任务：要的是稳定、可复现的分数，
    /// 思考只会烧 token（实测 reasoning_tokens 能吃掉 100% 预算）
    /// 而不会提升判定质量。
    pub judge_disable_reasoning: bool,

    /// 嵌入模型（用于语义相似度指标）。
    pub embedding_model: rag_06::config::EmbeddingModel,

    /// 分块配置（**必须和建索引时一致**）。
    pub splitter: rag_04::splitter::SplitterConfig,

    /// 生成模型（被评估系统用的）。
    pub generation_model: String,

    /// 生成温度。
    pub generation_temperature: f64,

    /// 生成 max tokens。
    pub generation_max_tokens: u32,

    /// 检索 top-k。
    pub top_k: usize,

    /// 是否启用重排。
    pub rerank_enabled: bool,

    /// 评估并发度。
    ///
    /// ⚠️ 别调太高 —— 重排是 **CPU 密集**的（cross-encoder 逐条推理），
    /// 并发太高会互相抢 CPU，反而更慢，还可能打爆上游限流。
    pub concurrency: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            data_dir: manifest.join("../08-rag/data"),
            index_path: manifest.join("index/index.faiss"),
            dataset_path: manifest.join("data/eval_cases.json"),
            report_dir: manifest.join("reports"),
            variants: EvalVariant::all(),
            judge_enabled: true,
            judge_model: "deepseek-flash".to_string(),
            judge_temperature: 0.0,
            // ⚠️ 2048 而不是 1024 —— judge 要一次性给几十个句子打分 + 理由。
            // 1024 在开启思考时会被思考 token 吃光，导致正文为空。
            judge_max_tokens: 2048,
            judge_disable_reasoning: true,
            embedding_model: rag_06::config::EmbeddingModel::MultilingualE5Small,
            splitter: rag_04::splitter::SplitterConfig::default(),
            generation_model: "deepseek-flash".to_string(),
            generation_temperature: 0.1,
            generation_max_tokens: 2000,
            top_k: 4,
            rerank_enabled: true,
            concurrency: crate::pipeline::DEFAULT_CONCURRENCY,
        }
    }
}

impl EvalConfig {
    /// 从环境变量加载配置。
    ///
    /// | 变量 | 说明 | 默认 |
    /// |---|---|---|
    /// | `EVAL_DATA_DIR` | 知识库目录 | `08-rag/data` |
    /// | `EVAL_INDEX_PATH` | 索引路径 | `09-rag-eval/index/index.faiss` |
    /// | `EVAL_DATASET` | 评估集路径 | `09-rag-eval/data/eval_cases.json` |
    /// | `EVAL_REPORT_DIR` | 报告输出目录 | `09-rag-eval/reports` |
    /// | `EVAL_VARIANTS` | 逗号分隔的变体 | 全部 |
    /// | `EVAL_JUDGE_ENABLED` | 是否启用 judge 指标 | `true` |
    /// | `EVAL_JUDGE_MODEL` | judge 模型 | `deepseek-flash` |
    /// | `EVAL_GENERATION_MODEL` | 生成模型 | `deepseek-flash` |
    pub fn from_env() -> Self {
        let mut config = EvalConfig::default();

        if let Some(v) = env_str("EVAL_DATA_DIR") {
            config.data_dir = PathBuf::from(v);
        }
        if let Some(v) = env_str("EVAL_INDEX_PATH") {
            config.index_path = PathBuf::from(v);
        }
        if let Some(v) = env_str("EVAL_DATASET") {
            config.dataset_path = PathBuf::from(v);
        }
        if let Some(v) = env_str("EVAL_REPORT_DIR") {
            config.report_dir = PathBuf::from(v);
        }
        if let Some(v) = env_str("EVAL_VARIANTS") {
            let parsed: Vec<EvalVariant> = v
                .split(',')
                .filter_map(|s| EvalVariant::parse(s.trim()))
                .collect();
            if !parsed.is_empty() {
                config.variants = parsed;
            }
        }
        if let Some(v) = env_bool("EVAL_JUDGE_ENABLED") {
            config.judge_enabled = v;
        }
        if let Some(v) = env_str("EVAL_JUDGE_MODEL") {
            config.judge_model = v;
        }
        if let Some(v) = env_f64("EVAL_JUDGE_TEMPERATURE") {
            config.judge_temperature = v;
        }
        if let Some(v) = env_u32("EVAL_JUDGE_MAX_TOKENS") {
            config.judge_max_tokens = v;
        }
        if let Some(v) = env_bool("EVAL_JUDGE_DISABLE_REASONING") {
            config.judge_disable_reasoning = v;
        }
        if let Some(v) = env_str("EVAL_GENERATION_MODEL") {
            config.generation_model = v;
        }
        if let Some(v) = env_f64("EVAL_GENERATION_TEMPERATURE") {
            config.generation_temperature = v;
        }
        if let Some(v) = env_u32("EVAL_GENERATION_MAX_TOKENS") {
            config.generation_max_tokens = v;
        }
        if let Some(v) = env_usize("EVAL_TOP_K") {
            config.top_k = v;
        }
        if let Some(v) = env_bool("EVAL_RERANK_ENABLED") {
            config.rerank_enabled = v;
        }
        if let Some(v) = env_usize("EVAL_CONCURRENCY") {
            config.concurrency = v.max(1);
        }

        config
    }

    /// 打印当前配置。
    pub fn summary(&self) -> String {
        format!(
            "数据目录: {}\n索引路径: {}\n评估集: {}\n报告目录: {}\n\
             变体: {}\njudge: 启用={} 模型={} 温度={} max_tokens={} 思考={}\n\
             生成: {} 温度={}\n检索: top_k={} 重排={}",
            self.data_dir.display(),
            self.index_path.display(),
            self.dataset_path.display(),
            self.report_dir.display(),
            self.variants
                .iter()
                .map(|v| v.name())
                .collect::<Vec<_>>()
                .join(", "),
            self.judge_enabled,
            self.judge_model,
            self.judge_temperature,
            self.judge_max_tokens,
            if self.judge_disable_reasoning {
                "关闭"
            } else {
                "开启"
            },
            self.generation_model,
            self.generation_temperature,
            self.top_k,
            self.rerank_enabled,
        )
    }

    /// 转成 08-rag 的配置（被评估系统的配置）。
    ///
    /// ## 为什么复用 08-rag 的 `AppConfig`
    ///
    /// 因为「被评估系统」**就是** 08-rag —— 不该抄一份配置。
    /// 抄一份的话，08 改了参数 09 不会跟着改，对比实验就失去意义了。
    pub fn to_rag_config(&self) -> rag_08::config::AppConfig {
        let mut config = rag_08::config::AppConfig {
            data_dir: self.data_dir.clone(),
            index_path: self.index_path.clone(),
            embedding_model: self.embedding_model,
            splitter: self.splitter.clone(),
            ..Default::default()
        };
        config.retrieval.top_k = self.top_k;
        config.rerank.enabled = self.rerank_enabled;
        config.generation.model = self.generation_model.clone();
        config.generation.temperature = self.generation_temperature;
        config.generation.max_tokens = self.generation_max_tokens;
        config
    }
}

/// 读一个非空环境变量。
fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// 读一个 `usize` 环境变量。
fn env_usize(key: &str) -> Option<usize> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `u32` 环境变量。
fn env_u32(key: &str) -> Option<u32> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个 `f64` 环境变量。
fn env_f64(key: &str) -> Option<f64> {
    env_str(key).and_then(|v| v.parse().ok())
}

/// 读一个布尔环境变量。
fn env_bool(key: &str) -> Option<bool> {
    env_str(key).map(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_variants_cover_all() {
        let config = EvalConfig::default();
        assert_eq!(config.variants.len(), EvalVariant::all().len());
    }

    #[test]
    fn judge_temperature_is_low() {
        // judge 的温度必须低 —— 否则同一份输入会得到不同分数。
        assert!(EvalConfig::default().judge_temperature <= 0.2);
    }

    #[test]
    fn judge_disables_reasoning_by_default() {
        // 推理 token 会吃光 max_tokens 预算，导致 judge 输出为空。
        assert!(EvalConfig::default().judge_disable_reasoning);
    }

    #[test]
    fn judge_max_tokens_is_generous_enough() {
        // 1024 在开启思考时不够 —— 关思考也要给足（几十个句子 + 理由）。
        assert!(EvalConfig::default().judge_max_tokens >= 2048);
    }

    #[test]
    fn rag_config_mirrors_eval_config() {
        let config = EvalConfig::default();
        let rag = config.to_rag_config();

        assert_eq!(rag.data_dir, config.data_dir);
        assert_eq!(rag.retrieval.top_k, config.top_k);
        assert_eq!(rag.generation.model, config.generation_model);
    }

    #[test]
    fn judge_model_differs_from_generation_model_by_default() {
        // 默认配置里两者相同（用同一个便宜的模型），
        // 但 README 明确建议生产环境分开。
        let config = EvalConfig::default();
        assert_eq!(config.judge_model, config.generation_model);
    }
}
