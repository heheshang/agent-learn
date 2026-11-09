"""分词算法（CJK 二元切分）—— naive 与 optimized **共用**。

## 为什么要单独一个模块

分词算法**必须两版完全一致**，否则检索结果会不同，
耗时差值里就混进了「召回不同」这个无关变量。
所以算法抽到公共模块，两版只是「要不要缓存」的差别。

## 为什么要 bigram（二元切分）

⚠️ **这是实测踩出来的坑**：初版用「连续字符算一个词」的方式分词，
中文整句会被切成**一个超长 token**：

    "Agent 系统的性能瓶颈通常出现在哪里"
    → ['agent 系统的性能瓶颈通常出现在哪里']   # 整句一个 token！

结果：
1. 查询 token 与文档 token **完全没有交集** → 检索命中数 = 0；
2. 检索退化成「恒返回空」→ 检索阶段的 CPU 开销几乎为 0 →
   **想测的「检索是瓶颈」根本测不出来**。

真实的中文检索系统（jieba / HanLP / Lucene CJK）都会做词切分或 bigram。
这里手写 bigram：把连续的中文字符切成**相邻两字的组合**，
既不需要词典，又能让「性能」「能瓶」「瓶颈」这类词被匹配上。
"""

from __future__ import annotations

# 停用词表（真实 BM25 都会维护这么一张表）
STOPWORDS = frozenset(
    {
        "的", "了", "和", "是", "在", "有", "与", "或", "如何", "哪些", "多少",
        "a", "an", "the", "is", "are", "of", "to", "and", "or", "in", "on",
    }
)


def _is_cjk(ch: str) -> bool:
    """判断是否为 CJK 字符（含中日韩统一表意文字与扩展 A）。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def tokenize(text: str) -> list[str]:
    """CJK bigram + 英文按空白/标点切分。

    规则：
    - 连续 CJK 字符 → **相邻两字**的 bigram（单个字也保留）；
    - 连续英文/数字 → 作为一个词（小写化）；
    - 标点 → 作为独立 token 保留（`cProfile` 这类词会被 `、` 断开）；
    - 停用词丢弃。

    ⚠️ 输出必须是**确定的**（同输入同输出），
    因为两版要用它做等价性对比。
    """
    tokens: list[str] = []
    cjk_run: list[str] = []
    word_run: list[str] = []

    def flush_cjk() -> None:
        if not cjk_run:
            return
        if len(cjk_run) == 1:
            tokens.append(cjk_run[0])
        else:
            for i in range(len(cjk_run) - 1):
                tokens.append(cjk_run[i] + cjk_run[i + 1])
        cjk_run.clear()

    def flush_word() -> None:
        if not word_run:
            return
        word = "".join(word_run).lower()
        if word:
            tokens.append(word)
        word_run.clear()

    for ch in text:
        if _is_cjk(ch):
            flush_word()
            cjk_run.append(ch)
        elif ch.isalnum():
            flush_cjk()
            word_run.append(ch)
        elif ch.isspace():
            flush_cjk()
            flush_word()
        else:
            # 标点：断开两侧，标点本身作为独立 token
            flush_cjk()
            flush_word()
            tokens.append(ch)

    flush_cjk()
    flush_word()

    return [t for t in tokens if t and t not in STOPWORDS]
