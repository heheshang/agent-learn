//! 工作负载：Rust 侧的「靶子」。
//!
//! ## 和 Python 侧的关系
//!
//! Python 靶子（`profiler/src/agent_perf_profiler/`）和本模块做**同一件事**：
//! 改写 → 检索 → 工具调用 → 生成。
//!
//! 但**只有一个阶段有对比价值**：检索（BM25）。
//! 因为 Python 侧的瓶颈（naive 版）主要就在 BM25 打分里，
//! 而两边的语料、分词口径、BM25 参数**必须一致**，
//! 耗时差值才能归因到「语言与实现」而不是「算法不同」。
//!
//! ⚠️ 本仓库 08-rag 的 BM25 用的是**中文 bigram + 英文按词**的分词，
//!    和 Python 侧 `tokenizer.py` 的口径一致（有意对齐）——
//!    这样两边的检索命中数应该相同，可以互相印证。

use std::time::Instant;

use rag_08::bm25::{Bm25Document, Bm25Index};

/// 基线文档（与 Python 侧 `WorkloadSpec.documents` **完全一致**）。
///
/// ⚠️ 两边语料必须逐字相同，否则 BM25 分数不同、命中不同，
///    耗时对比就掺进了「工作量不同」这个无关变量。
pub const DOCUMENTS: [&str; 10] = [
    "性能剖析是定位程序热点的方法，常见工具有 cProfile、py-spy 与 Scalene。",
    "cProfile 是 Python 标准库提供的确定性统计剖析器，输出调用次数与累计耗时。",
    "py-spy 是采样型剖析器，它读取进程内存采样栈帧，无需修改被测代码。",
    "Scalene 同时统计 CPU 时间与内存分配，并区分 Python 代码与原生库的时间。",
    "BM25 是经典的相关性排序算法，在短文档检索上表现稳定。",
    "大语言模型推理延迟主要由网络往返与 token 生成速度决定。",
    "查询改写通过多路召回提升召回率，但会带来额外的模型调用开销。",
    "ReAct 循环让模型交替进行思考与行动，通常需要多轮工具调用。",
    "上下文压缩可以在生成前裁剪无关内容，从而降低 token 消耗。",
    "混合检索结合稠密向量与稀疏关键词两路召回，再用重排模型融合。",
];

/// 查询列表（与 Python 侧一致）。
pub const QUERIES: [&str; 5] = [
    "Agent 系统的性能瓶颈通常出现在哪里",
    "如何降低 LLM 调用的响应时间",
    "BM25 检索在大规模语料上会退化吗",
    "Python 性能剖析工具有哪些",
    "查询改写会带来多少额外开销",
];

/// 语料扩增倍数（与 Python 侧一致）。
pub const CORPUS_REPEAT: usize = 4;

/// 计时时的**外层重复倍数**。
///
/// ⚠️ 为什么需要它（实测踩过的坑）：
/// 一轮工作负载（40 个文档块 × 5 个查询）在 Rust 里**不到 1 毫秒**，
/// `as_millis()` 全部取整成 0 —— 报告里全是 0，看不出任何东西。
///
/// 计时必须用**微秒**（`as_micros`）才有分辨率。
/// 这里再用外层重复把耗时放大到**毫秒级**，
/// 这样「中位数」这个统计量才有意义（不然全是 0，取中位数也是 0）。
pub const TIMING_OUTER_REPEATS: usize = 50;

/// 一次工作负载跑完后的结果。
#[derive(Debug, Clone)]
pub struct WorkloadOutcome {
    /// 命中的文档 ID。
    pub hits: Vec<String>,
    /// 总耗时（**微秒**）。
    ///
    /// ⚠️ 用微秒而不是毫秒：Rust 侧一轮工作负载不到 1 毫秒，
    ///    `as_millis()` 会全部取整成 0，报告里就全是 0。
    pub elapsed_us: u64,
    /// 检索轮次。
    pub search_rounds: usize,
}

/// Rust 侧的工作负载。
pub struct RustWorkload {
    /// BM25 索引（**建一次**，这本身就是「优化后」的做法）。
    index: Bm25Index,
}

impl Default for RustWorkload {
    fn default() -> Self {
        Self::new()
    }
}

impl RustWorkload {
    /// 建索引并准备好。
    pub fn new() -> Self {
        let mut documents: Vec<Bm25Document> = Vec::new();
        for repeat in 0..CORPUS_REPEAT {
            for (doc_id, text) in DOCUMENTS.iter().enumerate() {
                documents.push(Bm25Document {
                    id: format!("doc-{repeat}-{doc_id}"),
                    source: format!("corpus-{repeat}.md"),
                    heading_path: vec![format!("第{repeat}版"), format!("段落{doc_id}")],
                    text: format!("{text}（版本{repeat}-{doc_id}）"),
                });
            }
        }
        Self {
            index: Bm25Index::build(&documents),
        }
    }

    /// 跑一轮检索。
    ///
    /// ⚠️ 这里**故意不做**「每次查询都重建索引」这种低效操作 ——
    ///    Rust 侧扮演的是「优化后」的角色，
    ///    与 Python 侧 optimized 版对齐。
    pub fn run_once(&self, query: &str) -> Vec<rag_08::bm25::Bm25Hit> {
        self.index.search(query, 3)
    }

    /// 跑完整工作负载（全部查询 × 重复次数）。
    pub fn run(&self, repeats: usize) -> WorkloadOutcome {
        let start = Instant::now();
        let mut hits: Vec<String> = Vec::new();
        let mut search_rounds = 0usize;

        for _ in 0..repeats.max(1) {
            for query in QUERIES {
                for hit in self.run_once(query) {
                    hits.push(hit.id);
                }
                search_rounds += 1;
            }
        }

        WorkloadOutcome {
            hits,
            elapsed_us: start.elapsed().as_micros() as u64,
            search_rounds,
        }
    }

    /// 语料规模（文档块数）。
    pub fn corpus_size(&self) -> usize {
        CORPUS_REPEAT * DOCUMENTS.len()
    }
}
