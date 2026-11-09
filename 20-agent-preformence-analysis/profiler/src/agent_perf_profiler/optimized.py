"""optimized 版实现：与 naive 版**行为完全一致**，但消掉了已知的低效点。

这个模块是剖析实验的**对照组**。它遵守三条铁律：

1. **行为等价**：返回的答案与 naive 版逐字相同。
   差值因此只能来自「耗时」，不含「正确性」这个无关变量。
2. **变量单一**：与 naive 版共用同一个 `WorkloadSpec`，
   LLM 调用次数、工具调用次数、IO 等待时长全部一致。
3. **每个优化点都能说清代价**：优化不是免费的，
   预计算占内存、缓存占内存 —— 报告里必须交代这些代价。

优化点与代价对照表：

| 优化 | 省掉的 | 代价 |
|---|---|---|
| 预计算 idf + 倒排索引 | 每次检索重算 log | 索引常驻内存 |
| 分词缓存 | 重复 lower/strip | 缓存占内存（LRU 限制） |
| `join` 代替 `+=` | O(n²) 字符串拷贝 | 无 |
| `heapq.nlargest` 代替全排序 | O(n log n) 排序 | 无 |
| 去掉 `sort_keys` | O(k log k) 键排序 | 无（但键序需稳定） |
| 工具参数改结构化 dict | 字符串切分 | 无 |
"""

from __future__ import annotations

import heapq
import json
import math
import time
from typing import Any

from .tokenizer import tokenize
from .workload import WorkloadResult, WorkloadSpec

# 与 naive 版**完全相同**的 IO 等待。
#
# ⚠️ 这一行是整个对比实验成立的前提：
#   如果 optimized 版的 sleep 更短，耗时差值里就混进了「IO 等待变少」，
#   而那不是优化的功劳 —— IO 等待是外部约束，优化不了。
#   所以两边的 sleep 必须一模一样。
_LLM_LATENCY_SEC = 0.002


class OptimizedAgent:
    """优化版 Agent（剖析对照组）。

    与 [`NaiveAgent`][agent_perf_profiler.naive.NaiveAgent]
    共用同一个 `WorkloadSpec`，行为一致、耗时更低。
    """

    def __init__(self, spec: WorkloadSpec) -> None:
        self.spec = spec
        # 倒排索引：term -> [(doc_id, tf), ...]
        self._postings: dict[str, list[tuple[int, int]]] = {}
        # 文档长度
        self._doc_len: list[int] = []
        # 预计算的 idf
        self._idf: dict[str, float] = {}
        # 平均文档长度
        self._avg_len: float = 0.0
        # 分词缓存：LRU 上限，防止无界增长
        self._token_cache: dict[str, list[str]] = {}
        self._token_cache_limit = 512
        self._build_index()

    # ── 索引构建 ─────────────────────────────────────────────────────

    def _build_index(self) -> None:
        """**一次性**建好倒排索引 + 预计算 idf。

        这是 naive 版最大的浪费：naive 每次检索都重建分词并重算 idf。
        真实系统里 BM25 索引应该**建一次、复用无数次** ——
        这也是为什么向量库的索引构建是离线任务。
        """
        doc_count = 0
        for doc in self.spec.corpus:
            for start in range(0, len(doc), self.spec.chunk_size):
                chunk = doc[start : start + self.spec.chunk_size]
                tokens = self.optimized_tokenize(chunk)
                doc_id = doc_count
                doc_count += 1
                self._doc_len.append(len(tokens))

                tf: dict[str, int] = {}
                for token in tokens:
                    tf[token] = tf.get(token, 0) + 1
                for term, freq in tf.items():
                    self._postings.setdefault(term, []).append((doc_id, freq))

        self._avg_len = (sum(self._doc_len) / max(len(self._doc_len), 1)) or 1.0

        # 预计算 idf —— naive 版是每次打分现算 log
        for term, posting in self._postings.items():
            df = len(posting)
            self._idf[term] = math.log(
                1 + (len(self._doc_len) - df + 0.5) / (df + 0.5)
            )

    def optimized_tokenize(self, text: str) -> list[str]:
        """分词 + **缓存**。

        缓存是「空间换时间」的典型。naive 版每次都重新算，
        这里对同一个字符串只算一次。

        ⚠️ 分词**算法**与 naive 版完全一致（共用 `tokenizer.tokenize`），
        唯一的差别是缓存 —— 变量单一，耗时差值才能归因到缓存。
        """
        cached = self._token_cache.get(text)
        if cached is not None:
            return cached

        result = tokenize(text)

        # 简单的容量控制：超过上限就清空（避免无界增长）
        if len(self._token_cache) >= self._token_cache_limit:
            self._token_cache.clear()
        self._token_cache[text] = result
        return result

    # ── 查询改写（LLM 阶段）────────────────────────────────────────

    def optimized_rewrite_query(self, query: str, round_index: int) -> str:
        """第 N 轮查询改写。

        优化：用 **list + join** 代替 `+` 反复拼接。
        输出内容与 naive 版**逐字相同**（注意换行位置一致）。
        """
        parts: list[str] = []
        for i in range(round_index + 1):
            parts.append(f"[改写{i}] {query} 的第{i}轮表述\n")
        parts.append(query)
        time.sleep(_LLM_LATENCY_SEC)
        return "".join(parts)

    # ── 检索（BM25）─────────────────────────────────────────────────

    def _search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        """BM25 检索（走倒排索引）。

        优化点：
        1. 只遍历**查询词命中的倒排链**，不再扫全量文档；
        2. idf 已预计算；
        3. 用 `heapq.nlargest` 取 top-k，不再对全部命中排序。
        """
        query_tokens = self.optimized_tokenize(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = {}
        for term in query_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self._postings.get(term, ()):  # 只命中才遍历
                doc_len = self._doc_len[doc_id]
                denom = tf + 1.5 * (0.25 + 0.75 * doc_len / self._avg_len)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * 2.5) / denom

        # top-k 用堆，避免 O(n log n) 全排序
        return heapq.nlargest(k, scores.items(), key=lambda item: item[1])

    # ── 工具调用 ────────────────────────────────────────────────────

    def optimized_tool_call(self, tool_name: str, args: dict[str, Any]) -> str:
        """执行一次工具调用。

        优化：参数是**结构化 dict**，不需要字符串切分。
        输出格式与 naive 版一致。
        """
        result = ""
        for key, value in args.items():
            result = result + f"{tool_name}:{key.strip()}={str(value).strip()};"
        time.sleep(_LLM_LATENCY_SEC * 0.5)
        return result

    # ── JSON 编解码 ────────────────────────────────────────────────

    def optimized_json_dumps(self, payload: dict[str, Any]) -> str:
        """JSON 序列化。

        优化：去掉 `sort_keys`（键序本来稳定）。
        """
        return json.dumps(payload, ensure_ascii=False)

    def optimized_json_loads(self, text: str) -> dict[str, Any]:
        """JSON 反序列化。"""
        return json.loads(text)

    # ── 生成 ────────────────────────────────────────────────────────

    def optimized_generate(self, question: str, context: list[str]) -> str:
        """生成最终答案。

        优化：用 `''.join` 代替 `+=`。
        输出内容与 naive 版**逐字相同**。
        """
        tokens: list[str] = []
        for chunk in context:
            tokens.append(chunk[:40])
            tokens.append("；")
        tokens.append(f"关于「{question}」，")
        tokens.append("根据检索到的资料可以得出上述结论。")

        time.sleep(_LLM_LATENCY_SEC)
        return "".join(tokens)

    # ── 主流程 ──────────────────────────────────────────────────────

    def run(self) -> WorkloadResult:
        """跑完整工作负载（与 naive 版流程一一对应）。"""
        result = WorkloadResult()
        for query in self.spec.queries:
            rewritten = query
            for round_index in range(self.spec.rewrite_rounds):
                rewritten = self.optimized_rewrite_query(rewritten, round_index)
                result.llm_calls += 1

            hits = self._search(rewritten)
            result.retrieved_docs += len(hits)

            corpus = self.spec.corpus
            context = [corpus[doc_id % len(corpus)] for doc_id, _ in hits]

            for call_index in range(self.spec.tool_calls):
                args = {"query": rewritten[:20], "index": call_index}
                self.optimized_tool_call("search", args)
                result.tool_calls += 1

            payload = {"question": query, "context": context}
            text = self.optimized_json_dumps(payload)
            payload = self.optimized_json_loads(text)

            answer = self.optimized_generate(payload["question"], payload["context"])
            result.llm_calls += 1
            result.answers.append(answer)
            result.citations.extend(f"doc-{doc_id}" for doc_id, _ in hits)
        return result


def optimized_workload(spec: WorkloadSpec | None = None) -> WorkloadResult:
    """跑一次 optimized 工作负载（便捷入口）。"""
    spec = spec or WorkloadSpec()
    return OptimizedAgent(spec).run()