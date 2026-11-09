"""靶子包：一个**故意有性能瓶颈**的 Python Agent 系统。

这个包是剖析实验的**被测对象**。它模拟一个真实的 RAG + Agent 问答系统：

    用户提问
      → 查询改写（多次 LLM 调用）
      → BM25 检索（本地倒排索引）
      → 工具调用（ReAct 循环）
      → 生成最终答案

设计原则（决定了 profile 是否「干净」）：

1. **零第三方依赖**。不用 requests / numpy / pydantic ——
   否则 cProfile 会把库的 import 与内部调用也统计进来，
   热点被淹没。LLM 调用、BM25、JSON 编解码全部手写。
2. **瓶颈是「已知且可解释」的**。每个热点函数都对应一个真实场景
   （见各函数 docstring），剖析结果应当能**一一对应**回去。
3. **有 naive 版与 optimized 版两套实现**，变量单一：
   只有实现不同，其余（数据、调用次数、输入）完全一致。
   这样「优化前后」的差值才能归因到实现本身。
"""

from .naive import NaiveAgent, naive_workload
from .optimized import OptimizedAgent, optimized_workload
from .workload import WorkloadSpec

__all__ = [
    "NaiveAgent",
    "naive_workload",
    "OptimizedAgent",
    "optimized_workload",
    "WorkloadSpec",
]
