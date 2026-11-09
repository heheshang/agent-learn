"""工作负载规格：把「跑多少、跑什么」从实现里剥出来。

为什么要单独抽一层：
    剖析实验要求「**除实现外一切相同**」。如果调用次数散落在
    naive.py / optimized.py 两份代码里，两边很容易差一点点，
    最后的耗时差值就分不清是「优化有效」还是「工作量不同」。

    所以调用次数、查询语料、文档语料统一定义在这里，
    两套实现共用**同一个** `WorkloadSpec` 实例。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkloadSpec:
    """一次剖析工作负载的完整定义（不可变）。

    字段
    ----
    queries:
        用户提问列表。每个提问会触发一次完整的 Agent 流程。
    documents:
        知识库文档。检索阶段会在这些文档里做 BM25 打分。
    corpus_repeat:
        语料**扩增倍数**。真实 RAG 的知识库有几千个块，
        而本项目的基线文档只有 10 条 —— 规模太小的话，
        「线性扫描全量文档」这个瓶颈根本不会显现，
        两版的耗时会几乎一样（**实测踩过**：初版 naive 与 optimized
        耗时差 < 5%，根本看不出差别，瓶颈白埋了）。

        扩增方式：每条文档复制 N 份，末尾加不同后缀
        （保证每份内容不完全相同，避免 BM25 分数全部相同）。
    rewrite_rounds:
        每个提问的**查询改写轮数**。真实系统里改写常做 1~3 轮，
        每轮是一次 LLM 调用 —— 这是典型的「看起来很快，
        实际占了大部分延迟」的阶段。
    tool_calls:
        每个提问的**工具调用次数**。ReAct 循环每轮可能调多个工具。
    chunk_size:
        文档切块粒度。影响索引构建与检索的规模。
    """

    queries: tuple[str, ...] = (
        "Agent 系统的性能瓶颈通常出现在哪里",
        "如何降低 LLM 调用的响应时间",
        "BM25 检索在大规模语料上会退化吗",
        "Python 性能剖析工具有哪些",
        "查询改写会带来多少额外开销",
    )

    documents: tuple[str, ...] = (
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
    )

    corpus_repeat: int = 4
    rewrite_rounds: int = 3
    tool_calls: int = 4
    chunk_size: int = 120

    @property
    def query_count(self) -> int:
        """提问总数（剖析时的「样本量」）。"""
        return len(self.queries)

    @property
    def corpus(self) -> tuple[str, ...]:
        """扩增后的完整语料（naive 与 optimized **共用**这一份）。

        ⚠️ 必须共用：否则两版的索引规模可能不同，
        耗时差值里就混进了「工作量不同」这个无关变量。
        """
        expanded: list[str] = []
        for repeat in range(max(self.corpus_repeat, 1)):
            for doc_id, doc in enumerate(self.documents):
                expanded.append(f"{doc}（版本{repeat}-{doc_id}）")
        return tuple(expanded)

    @property
    def total_llm_calls(self) -> int:
        """理论上的 LLM 调用总次数。

        每个提问：改写 ``rewrite_rounds`` 次 + **最终生成 1 次**。

        ⚠️ **工具调用不计入 LLM 调用** —— 它们是 tool call，不是模型调用。
        早期版本把 ``tool_calls`` 也算进这个公式，
        导致推算值（40）与实际值（20）对不上，测试直接暴露了它。

        这个数字在**报告**里用于把「总耗时」换算成「单次调用耗时」，
        便于和真实系统的 P50 / P99 对齐。
        """
        per_query = self.rewrite_rounds + 1
        return per_query * self.query_count


@dataclass
class WorkloadResult:
    """一次工作负载跑完后的结果（用于断言与报告）。"""

    answers: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    retrieved_docs: int = 0

    def summary(self) -> str:
        """一行摘要（打印到终端）。"""
        return (
            f"问题数={len(self.answers)} llm调用={self.llm_calls} "
            f"工具调用={self.tool_calls} 检索命中={self.retrieved_docs}"
        )
