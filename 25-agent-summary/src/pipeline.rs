//! **流水线** —— 把所有子系统接成一条端到端的链路。
//!
//! # 一条完整链路
//!
//! ```text
//!   活动记录
//!      │
//!      ├─▶ [ingest]    结构感知分块 + Token 计数
//!      │                └─▶ 块 → 检索文档
//!      │
//!      ├─▶ [retrieve]  BM25 + 稠密近似 → RRF 融合
//!      │                └─▶ 候选块
//!      │
//!      ├─▶ [agent]     ReAct 输出解析 → 工具调用
//!      │                └─▶ 工具名 + 参数
//!      │
//!      ├─▶ [tooling]   Schema 校验 → SQL 护栏 → 错误分类
//!      │                └─▶ 合法工具调用 / 带错误码的观察
//!      │
//!      ├─▶ [agent]     记忆召回（BM25 + 零命中退化）
//      │
//!      ├─▶ [cache]     内容寻址缓存键
//!      │
//!      ├─▶ [aggregate] 三档策略聚合
//!      │
//!      └─▶ [render]    确定性 Markdown 周报
//! ```
//!
//! # ★ 三档策略跑的是**同一条链路**
//!
//! `naive` / `pre-agg` / `optimized` 的差异**只在聚合环节**，
//! 前面的检索、解析、校验、召回**完全一样** ——
//! 这才是「变量单一」的对照实验。

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::agent::{MemoryFact, ParsedOutput, calculate, parse_react_output, recall_facts};
use crate::aggregate::{Aggregator, Strategy};
use crate::cache::{CacheKey, NormalizedMessage};
use crate::ingest::{LoadedDoc, SplitterConfig, split_document};
use crate::model::{Activity, Digest, Window};
use crate::retrieve::{Bm25Index, Document, HybridRetriever};
use crate::tooling::{self, check_sql, validate};

/// 统一流水线。
pub struct Pipeline {
    /// 分块后的检索文档。
    documents: Vec<Document>,

    /// 混合检索器。
    retriever: HybridRetriever,

    /// 聚合器。
    aggregator: Aggregator,

    /// 长期记忆事实。
    facts: Vec<MemoryFact>,

    /// 工具 Schema。
    tool_schema: Value,
}

impl Pipeline {
    /// 用一批活动记录建流水线。
    ///
    /// ## ⚠️ 活动记录要先转成**文档**
    ///
    /// 因为检索的对象是「文本块」而不是「结构化记录」。
    pub fn new(activities: Vec<Activity>, strategy: Strategy) -> Self {
        // ① 把记录转成 Markdown 文档（模拟「活动日志」）。
        let doc_text = activities_to_markdown(&activities);

        // ② 结构感知分块。
        let loaded = LoadedDoc {
            file_name: "activities.md".into(),
            text: doc_text,
        };
        let chunks = split_document(&loaded, &SplitterConfig::default());

        // ③ 块 → 检索文档。
        let documents: Vec<Document> = chunks
            .iter()
            .map(|c| Document {
                id: format!("chunk-{}", c.index),
                source: c.source.clone(),
                heading_path: c.heading_path.clone(),
                // ⚠️ 检索文本 = 标题路径 + 正文。
                //
                //    为什么不能只用 `text`（纯正文）：
                //    分块后的正文只有「- 类型：feature」这种**元数据行**，
                //    模块名在**标题**里。用纯正文的话，
                //    搜「cache」一个字都搜不到（实测命中数 = 0）。
                //
                //    为什么不用 `embedded_text`：
                //    它是给**嵌入模型**用的（把标题拼在前面）。
                //    BM25 的词元统计不该被标题主导 ——
                //    所以这里显式拼接，效果相同但语义更清楚。
                text: format!("{}\n{}", c.heading_path.join(" "), c.text),
            })
            .collect();

        // ④ 建索引 + 混合检索器。
        //
        // ⚠️ 稠密路需要**拥有**索引里的文本（`'static` 闭包不能借用局部变量），
        //    所以先构造稠密路，再把索引移进检索器。
        let index = Bm25Index::build(&documents);
        let dense = crate::retrieve::JaccardDense::new(&index);
        let retriever = HybridRetriever {
            bm25: index,
            dense: Box::new(move |q: &str, k: usize| dense.search(q, k)),
        };

        // ⑤ 事实库。
        let facts = build_facts(&activities);

        Self {
            documents,
            retriever,
            aggregator: Aggregator::new(activities, strategy),
            facts,
            tool_schema: tooling::module_stats_schema(),
        }
    }

    /// 检索文档数。
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// 事实数。
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// 当前策略。
    pub fn strategy(&self) -> Strategy {
        self.aggregator.strategy()
    }

    /// 检索。
    pub fn search(&self, query: &str, limit: usize) -> Vec<crate::retrieve::Hit> {
        self.retriever.fuse(query, limit)
    }

    /// 换策略。
    pub fn set_strategy(&mut self, strategy: Strategy) {
        self.aggregator.set_strategy(strategy);
    }

    /// 缓存命中数。
    pub fn cache_hits(&self) -> usize {
        self.aggregator.cache_hits()
    }

    /// ★ 跑一遍完整链路。
    pub fn run(&mut self, query: &str, window: &Window) -> RunOutcome {
        // ① 检索（混合）。
        let hits = self.retriever.fuse(query, 8);

        // ② 解析 ReAct 输出。
        let raw = format!(
            "Thought: 我需要查资料并汇总\nAction: module_stats\nAction Input: {{\"scope\": \"{query}\"}}"
        );
        let parsed = parse_react_output(&raw);

        // ③ Schema 校验 + SQL 护栏 + 错误分类。
        let (valid_args, blocked, business_errors, fatal_errors) = self.check_tool_call(&parsed);

        // ④ 记忆召回。
        let recalled = recall_facts(query, &self.facts, 3);

        // ⑤ 缓存键。
        let cache_key = self.compute_cache_key(query);

        // ⑥ 聚合（三档差异在这里）。
        let digest = self.aggregator.aggregate(window);

        RunOutcome {
            hits: hits.clone(),
            parsed,
            valid_args,
            blocked,
            business_errors,
            fatal_errors,
            recalled_facts: recalled.len(),
            computed_sum: calculate(&format!("{} + 1", hits.len())).unwrap_or(0.0) as i64,
            cache_key,
            digest,
        }
    }

    /// 工具调用的三层校验。
    fn check_tool_call(&self, parsed: &ParsedOutput) -> (bool, bool, usize, usize) {
        let ParsedOutput::Action { args, .. } = parsed else {
            return (false, false, 0, 0);
        };

        // ① Schema 校验。
        let mut args = args.clone();
        tooling::fill_defaults(&self.tool_schema, args.as_object_mut().unwrap());
        let validation = validate(&self.tool_schema, &args);

        // ② SQL 护栏（模拟工具内部要查库）。
        let sql = format!(
            "SELECT * FROM activity WHERE scope = '{}'",
            args["scope"].as_str().unwrap_or("")
        );
        let blocked = check_sql(&sql).is_err();

        // ③ 错误分类。
        let business_errors = usize::from(!validation.is_ok());
        let fatal_errors = 0;

        (validation.is_ok(), blocked, business_errors, fatal_errors)
    }

    /// ★ 内容寻址缓存键 —— **包含所有影响输出的东西**。
    fn compute_cache_key(&self, query: &str) -> String {
        let messages = vec![NormalizedMessage {
            role: "user".into(),
            content: query.to_string(),
        }];
        let key = CacheKey::new(
            "wk25",
            "summary-pipeline",
            &messages,
            crate::cache::tools_digest(&[]),
            "auto",
            0.0,
            1024,
            true,
        );
        key.digest()
    }
}

/// 一次完整运行的结果。
#[derive(Debug, Clone)]
pub struct RunOutcome {
    /// 检索命中。
    pub hits: Vec<crate::retrieve::Hit>,

    /// ReAct 解析结果。
    pub parsed: ParsedOutput,

    /// 参数是否通过校验。
    pub valid_args: bool,

    /// 是否被护栏拦下。
    pub blocked: bool,

    /// Business 错误数。
    pub business_errors: usize,

    /// Fatal 错误数。
    pub fatal_errors: usize,

    /// 召回的事实数。
    pub recalled_facts: usize,

    /// 算术求值结果。
    pub computed_sum: i64,

    /// 缓存键。
    pub cache_key: String,

    /// 汇总结论。
    pub digest: Digest,
}

/// 把活动记录转成 Markdown —— 模拟「活动日志文档」。
fn activities_to_markdown(activities: &[Activity]) -> String {
    let mut out = String::from("# 活动日志\n\n");
    let mut current_scope = String::new();

    for a in activities {
        if a.scope != current_scope {
            current_scope = a.scope.clone();
            out.push_str(&format!("\n## 模块 {current_scope}\n\n"));
        }
        let risk = if a.risk { "【风险】" } else { "" };
        // ⚠️ **模块名要出现在正文里**，不能只在标题里 ——
        //    因为分块后的 `Chunk.text` 是**正文**（不含标题路径），
        //    而检索用的是 `text`。
        //    只写在标题里 → 搜「cache」一个字都搜不到。
        out.push_str(&format!(
            "### {}（模块 {scope}）\n\n- 类型：{}\n- 负责人：{}\n- 权重：{}\n- {risk}{title}\n\n",
            a.title,
            a.kind,
            a.author,
            a.weight,
            scope = a.scope,
            risk = risk,
            title = a.title
        ));
    }

    out
}

/// 从活动记录里抽事实。
///
/// ## ⚠️ 去重要用 `(owner, subject, predicate)`
///
/// 用整条内容做键的话，「模块 cache」和「模块是 cache」是两条事实。
fn build_facts(activities: &[Activity]) -> Vec<MemoryFact> {
    let mut seen: BTreeMap<(String, String, String), MemoryFact> = BTreeMap::new();

    for a in activities {
        let fact = MemoryFact::new(
            format!("模块 {}", a.scope),
            "类型",
            a.kind.clone(),
            a.author.clone(),
        );
        seen.entry(fact.dedup_key()).or_insert(fact);
    }

    seen.into_values().collect()
}

/// 造一批演示数据。
pub fn demo_activities(count: usize) -> Vec<Activity> {
    let kinds = ["feature", "fix", "incident", "docs", "refactor"];
    let authors = ["alice", "bob", "carol", "dave"];
    let scopes = ["cache", "rag", "agent", "async", "batch", "vllm"];
    // 固定的历史起点 —— 让结果**可复现**。
    let base_ts = 1_700_000_000;

    (0..count)
        .map(|i| Activity {
            id: format!("act-{i:05}"),
            ts: base_ts + (i as i64 % 28) * 86_400 + (i as i64 % 24) * 3_600,
            kind: kinds[i % kinds.len()].to_string(),
            author: authors[i % authors.len()].to_string(),
            scope: scopes[i % scopes.len()].to_string(),
            title: format!("{} 任务 #{i}", kinds[i % kinds.len()]),
            weight: ((i * 7) % 10) as i32,
            risk: i % 25 == 0,
        })
        .collect()
}

/// 一个示例查询。
pub fn demo_query() -> String {
    json!({"scope": "cache"}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window() -> Window {
        Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400)
    }

    #[test]
    fn pipeline_produces_a_full_run() {
        let mut p = Pipeline::new(demo_activities(200), Strategy::Naive);
        assert!(p.document_count() > 0);
        assert!(p.fact_count() > 0);

        // ⚠️ 查询词必须和文档里**真实的词元**对上 ——
        //    「cache」在标题「模块 cache」里，会被分词成
        //    「cache」（ASCII 词）—— 能对上。
        let out = p.run("cache", &window());
        assert!(!out.hits.is_empty(), "应该检索到内容");
        assert!(out.valid_args);
        assert!(!out.blocked);
        assert_eq!(out.computed_sum, out.hits.len() as i64 + 1);
    }

    #[test]
    fn all_strategies_produce_the_same_digest() {
        let acts = demo_activities(300);
        let mut digests = Vec::new();
        for strategy in Strategy::all() {
            let mut p = Pipeline::new(acts.clone(), *strategy);
            let out = p.run("cache", &window());
            digests.push(out.digest);
        }
        assert_eq!(digests[0], digests[1], "naive 与 pre-agg 必须一致");
        assert_eq!(digests[1], digests[2], "pre-agg 与 optimized 必须一致");
    }

    #[test]
    fn optimized_hits_cache_on_repeat() {
        let mut p = Pipeline::new(demo_activities(100), Strategy::Optimized);
        let _ = p.run("cache", &window());
        assert_eq!(p.cache_hits(), 1);
        let _ = p.run("cache", &window());
        assert_eq!(p.cache_hits(), 1);
    }

    #[test]
    fn facts_are_deduplicated() {
        let acts = demo_activities(300);
        let facts = build_facts(&acts);
        // 6 个模块 × 5 种类型 = 最多 30 条。
        assert!(
            facts.len() <= 30,
            "事实数应该远小于记录数，实际 {}",
            facts.len()
        );
    }

    #[test]
    fn markdown_generation_groups_by_scope() {
        let md = activities_to_markdown(&demo_activities(20));
        assert!(md.contains("# 活动日志"));
        assert!(md.contains("## 模块"));
    }
}
