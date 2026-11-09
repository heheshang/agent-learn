"""测试：锁住「优化前后行为等价」这个核心前提。

⚠️ 整个剖析实验的有效性**完全依赖**一条性质：
   naive 版与 optimized 版产出的答案**逐字相同**。

   如果它们不等价，耗时差值里就混进了「正确性」这个无关变量 ——
   「optimized 更快」可能只是因为它少干了活或算错了。

   所以这条性质必须有测试兜底，不是「看一眼差不多」。
"""

from __future__ import annotations

import dataclasses

from agent_perf_profiler.naive import NaiveAgent
from agent_perf_profiler.optimized import OptimizedAgent
from agent_perf_profiler.tokenizer import tokenize
from agent_perf_profiler.workload import WorkloadSpec


def _spec(**kwargs: object) -> WorkloadSpec:
    """构造一个小的测试用规格（语料小，跑得快）。"""
    base = WorkloadSpec()
    return dataclasses.replace(base, corpus_repeat=2, **kwargs)  # type: ignore[arg-type]


def test_two_versions_produce_identical_answers() -> None:
    """⭐ 核心性质：两版答案逐字相同。

    这是整个对比实验成立的前提。
    """
    spec = _spec()
    naive_result = NaiveAgent(spec).run()
    optimized_result = OptimizedAgent(spec).run()

    assert naive_result.answers == optimized_result.answers
    assert naive_result.citations == optimized_result.citations


def test_two_versions_do_same_amount_of_work() -> None:
    """两版的 LLM 调用次数、工具调用次数、检索命中数必须一致。

    ⚠️ 如果次数不同，耗时差值就不能归因到「实现优化」。
    """
    spec = _spec()
    naive_result = NaiveAgent(spec).run()
    optimized_result = OptimizedAgent(spec).run()

    assert naive_result.llm_calls == optimized_result.llm_calls
    assert naive_result.tool_calls == optimized_result.tool_calls
    assert naive_result.retrieved_docs == optimized_result.retrieved_docs
    # 命中数必须 > 0，否则检索阶段根本没被测到
    assert naive_result.retrieved_docs > 0


def test_retrieval_actually_finds_relevant_docs() -> None:
    """检索必须真的能命中（中文 bigram 分词生效的标志）。

    ⚠️ **实测踩过的坑**：初版分词把整句中文当成一个 token，
    查询与文档的词元交集为空 → 命中数 = 0 → 检索阶段的 CPU 开销几乎为 0
    → 「检索是瓶颈」这个结论根本测不出来。
    """
    spec = _spec()
    agent = NaiveAgent(spec)
    hits = agent._search("性能剖析 工具有哪些")
    assert hits, "检索必须有命中，否则瓶颈根本没被触发"


def test_tokenizer_handles_chinese_with_bigram() -> None:
    """中文必须切成 bigram，而不是整句一个 token。"""
    tokens = tokenize("性能剖析")
    assert "性能" in tokens
    assert "能剖" in tokens
    assert "剖析" in tokens
    # 整句不该是一个 token
    assert "性能剖析" not in tokens


def test_tokenizer_splits_english_and_punctuation() -> None:
    """英文按词切，标点独立成 token。"""
    tokens = tokenize("cProfile、py-spy 与 Scalene")
    assert "cprofile" in tokens
    assert "py" in tokens
    assert "spy" in tokens
    assert "scalene" in tokens
    assert "、" in tokens


def test_tokenizer_is_deterministic() -> None:
    """分词必须确定性 —— 两版共用它做等价性对比。"""
    text = "Agent 系统的性能瓶颈通常出现在哪里"
    assert tokenize(text) == tokenize(text)


def test_stopwords_are_removed() -> None:
    """停用词被丢弃。"""
    tokens = tokenize("这是 的 一个 测试")
    assert "的" not in tokens


def test_optimized_uses_inverted_index() -> None:
    """optimized 版必须建了倒排索引（而不是每次全量扫描）。"""
    spec = _spec()
    agent = OptimizedAgent(spec)
    assert agent._postings, "倒排索引不能为空"
    assert agent._idf, "idf 必须预计算"


def test_optimized_tokenizer_cache_returns_same_result() -> None:
    """缓存命中时结果必须与首次计算完全一致。"""
    spec = _spec()
    agent = OptimizedAgent(spec)
    text = "性能剖析 工具"
    first = agent.optimized_tokenize(text)
    second = agent.optimized_tokenize(text)
    assert first == second


def test_corpus_is_shared_and_expanded() -> None:
    """扩增语料必须被两版共用（否则工作量不同）。"""
    spec = WorkloadSpec(corpus_repeat=3)
    expected = len(spec.documents) * 3
    assert len(spec.corpus) == expected
    # 两版拿到的是同一份
    assert NaiveAgent(spec).spec.corpus == OptimizedAgent(spec).spec.corpus


def test_workload_spec_counts_llm_calls_consistently() -> None:
    """LLM 调用总次数的推算要与实际跑出来的对得上。"""
    spec = WorkloadSpec(corpus_repeat=1)
    result = NaiveAgent(spec).run()
    assert result.llm_calls == spec.total_llm_calls
