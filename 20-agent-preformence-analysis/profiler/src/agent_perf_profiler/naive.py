"""naive 版实现：**故意低效**，用来制造可剖析的性能瓶颈。

这个模块里的每个「低效」都不是随意的，它们对应真实 Agent 系统里
被 profiler 抓到过的典型问题。

| 瓶颈 | 现实场景 |
|---|---|
| `naive_rewrite_query` 里的字符串拼接 | 改写结果用 `+` 拼字符串，消息多时 O(n²) 拷贝 |
| `naive_tokenize` 里的逐字符扫描 + 每次重新 lower | BM25 分词没缓存，每次检索都重扫全文 |
| `naive_bm25_score` 里的线性扫描 + 每次算 idf | 检索打分每次都重算 idf，没有预计算 |
| `naive_generate` 里的逐 token 拼接 | 生成时 `answer += token`，字符串反复拷贝 |
| `naive_tool_call` 里的字符串切分取参数 | 工具参数用字符串切分解析，没有结构化 |

⚠️ 这些低效**在功能上全部正确** —— 返回的答案与 optimized 版一致。
   这很重要：剖析对比的是**耗时**，不是行为。
   如果 naive 跑出来的答案是错的，那差值就掺进了「正确性」这个无关变量。
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from .tokenizer import tokenize
from .workload import WorkloadResult, WorkloadSpec

# ── 模拟 LLM 的网络往返延迟 ──────────────────────────────────────────
#
# ⚠️ 这个 sleep 是**故意留的 IO 等待**，它制造一个关键的教学点：
#
#   cProfile 统计的是「函数被占用的墙钟时间」，**包含 IO 等待**。
#   所以 cProfile 报告里 `time.sleep` 会是第一名 ——
#   但它不是「CPU 瓶颈」，而是「IO 瓶颈」。
#
#   py-spy 默认**排除空闲线程**（idle 检测），所以 sleep 期间几乎不采样，
#   它的报告更接近「谁在烧 CPU」。
#   Scalene 则把 CPU 时间和「等待」分开统计。
#
#   三个工具对同一个程序的「看法」不同 —— 这正是本项目要讲的核心。
_LLM_LATENCY_SEC = 0.002


class NaiveAgent:
    """低效版 Agent（剖析靶子）。

    流程：改写 → 检索 → 工具调用 → 生成。
    与 [`OptimizedAgent`][agent_perf_profiler.optimized.OptimizedAgent]
    **共用同一个** `WorkloadSpec`，保证工作量严格一致。
    """

    def __init__(self, spec: WorkloadSpec) -> None:
        self.spec = spec
        self._doc_tokens: list[list[str]] = []
        self._build_corpus()

    # ── 索引构建 ─────────────────────────────────────────────────────

    def _build_corpus(self) -> None:
        """把文档切成块并分词。

        naive 之处：**分词结果不缓存**，且每次都重新算一遍。
        真实系统里 BM25 索引应该只建一次，这里却在每次检索时重建（见 `_search`）。
        """
        for doc in self.spec.corpus:
            for start in range(0, len(doc), self.spec.chunk_size):
                chunk = doc[start : start + self.spec.chunk_size]
                self._doc_tokens.append(self.naive_tokenize(chunk))

    def naive_tokenize(self, text: str) -> list[str]:
        """朴素分词：**每次都重新算**，结果不缓存。

        真实问题：真实 BM25 分词器（jieba / bigram 切分）会对每个词调用
        `str.lower()` 和 `strip()`，如果**没有缓存**，
        在高 QPS 检索服务里会占掉可观 CPU。

        ⚠️ 分词**算法**与 optimized 版完全一致（共用 `tokenizer.tokenize`），
        唯一的差别是「算完之后要不要缓存」——
        变量单一，耗时差值才能归因到缓存。
        """
        return tokenize(text)

    # ── 查询改写（LLM 阶段）────────────────────────────────────────

    def naive_rewrite_query(self, query: str, round_index: int) -> str:
        """第 N 轮查询改写。

        naive 之处：用 `+` 反复拼字符串构造「改写历史」。
        真实问题：提示词模板里把历史轮次用 `+` 串起来，
        轮次一多就是 O(n²) 的字符串拷贝。

        另外注意这里的 `sleep`：它模拟 LLM 网络往返。
        """
        history = ""
        for i in range(round_index + 1):
            history = history + f"[改写{i}] {query} 的第{i}轮表述\n"
        time.sleep(_LLM_LATENCY_SEC)
        return history + query

    # ── 检索（BM25）─────────────────────────────────────────────────

    def _search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        """BM25 检索。

        naive 之处（三个叠加）：
        1. 每次检索都**重新分词**查询（应该缓存）；
        2. 打分时**每次都重算 idf**（应该预计算）；
        3. 用**线性扫描**遍历所有文档找 top-k（应该用堆）。
        """
        query_tokens = self.naive_tokenize(query)
        scored: list[tuple[int, float]] = []
        for doc_id, doc_tokens in enumerate(self._doc_tokens):
            score = self.naive_bm25_score(query_tokens, doc_tokens)
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def naive_bm25_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """BM25 打分。

        naive 之处：每次都现算 idf（`log` 每文档每词一次），
        而且对重复的查询词不合并词频。
        """
        if not query_tokens or not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        avg_len = sum(len(d) for d in self._doc_tokens) / max(len(self._doc_tokens), 1)

        score = 0.0
        for term in query_tokens:
            # idf 每次现算，没有缓存
            df = sum(1 for d in self._doc_tokens if term in d)
            if df == 0:
                continue
            idf = math.log(1 + (len(self._doc_tokens) - df + 0.5) / (df + 0.5))
            tf = 0
            for t in doc_tokens:
                if t == term:
                    tf += 1
            if tf == 0:
                continue
            score += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * doc_len / avg_len))
        return score

    # ── 工具调用 ────────────────────────────────────────────────────

    def naive_tool_call(self, tool_name: str, args: str) -> str:
        """执行一次工具调用。

        naive 之处：用字符串切分解析参数，而不是结构化 dict。
        真实问题：早期 Agent 实现常把工具参数拼成
        `"key=value,key2=value2"`，再用 `split` 解析 ——
        既慢又容易在值里含分隔符时出错。
        """
        result = ""
        for pair in args.split(","):
            if "=" not in pair:
                continue
            key, _, value = pair.partition("=")
            result = result + f"{tool_name}:{key.strip()}={value.strip()};"
        time.sleep(_LLM_LATENCY_SEC * 0.5)
        return result

    # ── JSON 编解码 ────────────────────────────────────────────────

    def naive_json_dumps(self, payload: dict[str, Any]) -> str:
        """JSON 序列化。

        naive 之处：`ensure_ascii=False` 且开启了 `sort_keys` ——
        后者会让每次序列化都对键排序（O(k log k)），
        而键的顺序在实践中是稳定的，排序纯属浪费。
        """
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def naive_json_loads(self, text: str) -> dict[str, Any]:
        """JSON 反序列化（naive 版会重复调用）。"""
        return json.loads(text)

    # ── 生成 ────────────────────────────────────────────────────────

    def naive_generate(self, question: str, context: list[str]) -> str:
        """生成最终答案。

        naive 之处：用 `answer += token` 逐 token 拼接（O(n²) 拷贝），
        而不是 `''.join(tokens)`。
        """
        tokens: list[str] = []
        for chunk in context:
            tokens.append(chunk[:40])
            tokens.append("；")
        tokens.append(f"关于「{question}」，")
        tokens.append("根据检索到的资料可以得出上述结论。")

        answer = ""
        for token in tokens:
            answer = answer + token
        time.sleep(_LLM_LATENCY_SEC)
        return answer

    # ── 主流程 ──────────────────────────────────────────────────────

    def run(self) -> WorkloadResult:
        """跑完整工作负载。"""
        result = WorkloadResult()
        for query in self.spec.queries:
            rewritten = query
            for round_index in range(self.spec.rewrite_rounds):
                rewritten = self.naive_rewrite_query(rewritten, round_index)
                result.llm_calls += 1

            hits = self._search(rewritten)
            result.retrieved_docs += len(hits)

            corpus = self.spec.corpus
            context = [corpus[doc_id % len(corpus)] for doc_id, _ in hits]

            for call_index in range(self.spec.tool_calls):
                args = f"query={rewritten[:20]},index={call_index}"
                self.naive_tool_call("search", args)
                result.tool_calls += 1

            payload = {"question": query, "context": context}
            text = self.naive_json_dumps(payload)
            payload = self.naive_json_loads(text)

            answer = self.naive_generate(payload["question"], payload["context"])
            result.llm_calls += 1
            result.answers.append(answer)
            result.citations.extend(f"doc-{doc_id}" for doc_id, _ in hits)
        return result


def naive_workload(spec: WorkloadSpec | None = None) -> WorkloadResult:
    """跑一次 naive 工作负载（便捷入口）。"""
    spec = spec or WorkloadSpec()
    return NaiveAgent(spec).run()