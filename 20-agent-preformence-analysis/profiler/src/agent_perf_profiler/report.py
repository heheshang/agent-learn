"""报告：把剖析结果整理成**可对比、可断言**的报告。

## 核心设计：指标要同时给出「名称 + 绝对耗时 + 占比」

早先的版本只给「名称 + 占比」，看不出「这个热点到底吃掉了多少秒」。
一个占 87% 但总耗时只有 0.1s 的热点，和一个占 10% 但耗了 2s 的热点，
**优化价值完全不同**。所以两个维度都必须写出来。

## 第二个核心设计：跨工具**归并**后再对比

三个工具的命名风格完全不同：

| 工具 | 命名 |
|---|---|
| cProfile | `naive_bm25_score` |
| py-spy | `NaiveAgent.naive_bm25_score` |
| Scalene | `naive.py:NaiveAgent.naive_bm25_score` |

不归一化的话，同一个热点会被**拆成三行**，读者根本看不出「三个工具指向同一处」。

归并后可以回答真正有价值的问题：
**「哪些热点被多个工具同时指向？」** —— 这就是交叉验证。

## 第三个核心设计：不要把 IO 等待当成 CPU 热点

`time.sleep` 在 cProfile 里排第一，但它是 **IO 等待**不是 CPU 消耗。
报告里必须**标注**出来，否则读者会去优化一个根本优化不了的东西。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .profiler import (
    TOOL_CPROFILE,
    TOOL_PYSPY,
    TOOL_SCALENE,
    Metric,
    _normalize_key,
    summarize_cprofile,
    summarize_scalene,
    summarize_scalene_io_wait,
    summarize_speedscope,
)

#: 行号漂移容差：cProfile 与 py-spy 报的同一函数可能差几行。
LINE_TOLERANCE = 3

#: 工具 → `MergedRow` 上的字段名。
_FIELD_BY_TOOL: dict[str, str] = {
    TOOL_CPROFILE: "cprofile",
    TOOL_PYSPY: "pyspy",
    TOOL_SCALENE: "scalene",
}


def _line_delta(a: str, b: str) -> int:
    """两个位置的行号差；无法解析时返回极大值（表示「不可比」）。

    ⚠️ 位置格式可能带文件前缀（`naive.py:135`），
    也可能没有（`~:0`、`<frozen ...>:757`）——
    只取最后一段冒号后的数字。

    ## ⚠️⚠️ 关键坑：无源文件的位置**必须判为不可比**

    - `~:0` → 内建函数（C 实现的 `time.sleep`），没有 Python 源码行
    - `<frozen importlib._bootstrap_external>:757` → 冻结模块，不是项目代码

    如果只按「行号差 ≤ 3」判定归一化，`~:0`（行号 0）和
    `naive.py:2`（行号 2）会被**误并**成同一个热点 ——
    一个内建函数和一个项目函数就此混为一谈。

    所以：**只要有一方不是真实源码位置，直接判不可比**。
    """
    def parse(loc: str) -> int | None:
        if not loc or ":" not in loc:
            return None
        head, _, tail = loc.rpartition(":")
        if not tail.isdigit():
            return None
        # ⚠️ 文件部分必须是真实源码文件（.py），否则不可比
        if not head.endswith(".py"):
            return None
        return int(tail)

    line_a = parse(a)
    line_b = parse(b)
    if line_a is None or line_b is None:
        return 10**9
    return abs(line_a - line_b)


@dataclass
class ToolReport:
    """单个工具的剖析结果。"""

    tool: str
    ok: bool
    metrics: list[Metric] = field(default_factory=list)
    unit: str = "seconds"
    note: str = ""
    error: str = ""

    @property
    def total_seconds(self) -> float:
        """该工具统计到的总耗时（秒）。"""
        return sum(m.seconds for m in self.metrics)

    def to_markdown(self, top: int = 10) -> str:
        """渲染成 Markdown 表格。

        ⚠️ 表格里**必须**同时给出「指标名称」和「绝对耗时」和「占比」——
           只给占比会让读者误判优化价值。
        """
        if not self.ok:
            return f"### {self.tool}\n\n⚠️ 剖析失败：{self.error}\n"

        lines = [
            f"### {self.tool}",
            "",
            f"单位：{self.unit}；{self.note}",
            "",
            "| 排名 | 指标名称 | 位置 | 耗时(秒) | 占比(%) | 类别 |",
            "|---|---|---|---|---|---|",
        ]
        for rank, m in enumerate(self.metrics[:top], start=1):
            category = "IO 等待" if m.is_io_wait else ("C 扩展" if m.is_c_native else "CPU")
            lines.append(
                f"| {rank} | `{m.name}` | `{m.location}` | "
                f"{m.seconds:.4f} | {m.percent:.2f} | {category} |"
            )
        if not self.metrics:
            lines.append("| — | （无样本） | — | — | — | — |")
        lines.append("")
        return "\n".join(lines)


@dataclass
class MergedRow:
    """跨工具归并后的一行。"""

    key: str
    display: str
    location: str
    cprofile: Metric | None = None
    pyspy: Metric | None = None
    scalene: Metric | None = None

    @property
    def tools_hit(self) -> int:
        """有几个工具指向这一行。"""
        return sum(1 for m in (self.cprofile, self.pyspy, self.scalene) if m is not None)

    @property
    def max_seconds(self) -> float:
        """各工具报告的最大绝对耗时（秒）。"""
        values = [
            m.seconds for m in (self.cprofile, self.pyspy, self.scalene) if m is not None
        ]
        return max(values) if values else 0.0


def _loc_key(metric: Metric) -> str:
    """位置键：`文件:行号`。

    ⚠️ 只在**有真实源码位置**时使用。
    `<built-in method time.sleep>` 的位置是 `~:0`（`~` 表示「无文件」），
    这种必须退回函数名键，否则所有内建函数会被并成一行。
    """
    loc = metric.location or ""
    if not loc or loc.startswith("~") or loc.startswith("<"):
        return ""
    return loc


def merge_reports(reports: dict[str, ToolReport], top: int = 10) -> list[MergedRow]:
    """把三个工具的结果**归并**成按「被命中工具数 + 最大耗时」排序的行。

    ## ⚠️ 归并的两级策略

    1. **首选按位置**（`文件:行号`）：cProfile 与 py-spy 报的是同一行，
       就是同一处代码 —— 这是最可靠的信号。
    2. **位置缺失时按函数名**：内建函数（`time.sleep`）没有源码位置，
       只能用名字归并。

    ## ⚠️ 行号漂移的坑

    cProfile 与 py-spy 的行号可能差几行（`@staticmethod`、装饰器、
    生成器表达式的 `lineno` 指向定义行 vs 执行行），
    所以同一函数名且行号相差 ≤ `LINE_TOLERANCE` 时**也认为是同一处**。

    若不处理，两个工具会把同一个热点拆成两行，
    「命中工具数」这个交叉验证信号就废了。
    """
    buckets: dict[str, MergedRow] = {}

    def bucket_for(metric: Metric, tool: str) -> MergedRow:
        # 位置键优先；没有位置就退回名字键
        key = _loc_key(metric) or f"name:{_normalize_key(metric.name)}"

        # 行号漂移的二次归并：找同名的行，若行号相差 ≤ 容差则并入
        if key.startswith("name:") is False:
            for other_key, other in buckets.items():
                if other_key.startswith("name:"):
                    continue
                if other.display != metric.name:
                    continue
                if _line_delta(other.location, metric.location) <= LINE_TOLERANCE:
                    key = other_key
                    break

        if key not in buckets:
            buckets[key] = MergedRow(
                key=key, display=metric.name, location=metric.location
            )
        row = buckets[key]
        # 位置取最短的非空（`naive.py:135` 比 `/abs/path/naive.py:135` 好读）
        if metric.location and (
            not row.location or len(metric.location) < len(row.location)
        ):
            row.location = metric.location
        setattr(row, _FIELD_BY_TOOL[tool], metric)
        return row

    for tool, field_name in _FIELD_BY_TOOL.items():
        report = reports.get(tool)
        if report is None or not report.ok:
            continue
        for m in report.metrics:
            bucket_for(m, tool)

    rows = list(buckets.values())
    rows.sort(key=lambda r: (-r.tools_hit, -r.max_seconds))
    return rows[:top]


def render_comparison(reports: dict[str, ToolReport]) -> str:
    """跨工具归并对比表（核心产出）。

    每行是一个**热点**，三列分别是各工具报告的**绝对耗时（秒）**与**占比**，
    最后一列是「几个工具都指向它」。

    ## ⚠️ 必须写清楚：三个工具的秒数**不可直接横向比较**

    三个工具是**分别在三次独立运行**中采集的，各自的测量口径不同：

    | 工具 | 采集方式 | 自身开销 | 是否含 IO 等待 |
    |---|---|---|---|
    | cProfile | 逐函数调用加钩子 | **大**（每次调用都要记账） | 含 |
    | py-spy | 定时采样（200Hz） | 小 | **不含**（排除空闲线程） |
    | Scalene | 定时采样 + 系统调用计数 | 中 | 单列（`n_sys_percent`） |

    所以 cProfile 报 0.71s、py-spy 报 1.00s **不代表谁错了** ——
    cProfile 的钩子开销算进了它自己的 `tottime`，而 py-spy 的采样窗口更长。

    真正可靠的读法是「**相对占比**」与「**命中工具数**」：
    前者说明「谁占大头」，后者说明「几个工具独立指向同一处」。
    """
    rows = merge_reports(reports, top=10)

    lines = [
        "## 跨工具归并对比（同一份程序，三次独立运行）",
        "",
        "★ 排序：先按「被几个工具同时指向」降序，再按「最大耗时」降序。",
        "",
        "> ⚠️ **三列秒数不可直接横向比较**：三个工具各自独立运行，",
        "> 采集机制与自身开销不同（cProfile 逐调用加钩子 → 开销大且含 IO 等待；",
        "> py-spy 定时采样 → 开销小但排除空闲线程）。",
        "> 可靠读法是**相对占比**与**命中工具数**。",
        "",
        "| # | 指标名称 | 位置 | cProfile | py-spy | Scalene | 命中 |",
        "|---|---|---|---|---|---|---|",
    ]

    for index, row in enumerate(rows, start=1):
        cells = []
        for metric in (row.cprofile, row.pyspy, row.scalene):
            if metric is None:
                cells.append("—")
            else:
                cells.append(f"{metric.seconds:.3f}s ({metric.percent:.1f}%)")
        lines.append(
            f"| {index} | `{row.display}` | `{row.location}` | "
            f"{cells[0]} | {cells[1]} | {cells[2]} | {row.tools_hit} |"
        )

    if not rows:
        lines.append("| — | （无数据） | — | — | — | — | — |")

    lines.append("")
    return "\n".join(lines)


def render_metric_names(reports: dict[str, ToolReport]) -> str:
    """各工具的 Top 指标名称 + 绝对耗时 + 占比（分工具明细）。

    这张表回答「每个工具各自认为谁是热点」。
    """
    parts = ["## 各工具 Top 指标（含绝对耗时与占比）", ""]
    for tool in (TOOL_CPROFILE, TOOL_PYSPY, TOOL_SCALENE):
        report = reports.get(tool)
        if report is None:
            continue
        parts.append(report.to_markdown(top=8))
    return "\n".join(parts)


def render_io_wait(reports: dict[str, ToolReport], io_metrics: list[Metric]) -> str:
    """IO 等待专项（这是本项目的核心认知）。

    ⚠️ 必须单独列出来 —— 因为 **IO 等待不是 CPU 瓶颈，优化不了**。
    cProfile 报告里 `time.sleep` 排第一很容易让人误以为是性能 bug，
    实际上那是模拟的 LLM 网络往返。
    """
    if not io_metrics:
        return ""

    lines = [
        "## IO 等待专项（Scalene 的 `n_sys_percent`）",
        "",
        "> ⚠️ 这些是**等系统调用 / 等网络**的时间，**不是 CPU 消耗**，优化不了。",
        "> cProfile 报告里 `time.sleep` 排第一，py-spy 完全看不到 —— 三者不矛盾。",
        "",
        "| 指标名称 | 位置 | 等待(秒) | 占比(%) |",
        "|---|---|---|---|",
    ]
    for m in io_metrics:
        lines.append(
            f"| `{m.name}` | `{m.location}` | {m.seconds:.4f} | {m.percent:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_reports(reports_dir: Path, top: int = 12) -> dict[str, ToolReport]:
    """读取三个工具的产物，生成可对比的报告。"""
    result: dict[str, ToolReport] = {}

    # cProfile
    prof_path = reports_dir / "cprofile.prof"
    if prof_path.exists():
        try:
            metrics = summarize_cprofile(prof_path, top=top)
            result[TOOL_CPROFILE] = ToolReport(
                tool=TOOL_CPROFILE,
                ok=True,
                metrics=metrics,
                unit="秒",
                note="**含 IO 等待**（`time.sleep` 会计入）",
            )
        except Exception as exc:  # noqa: BLE001 — 报告失败不应中断整体流程
            result[TOOL_CPROFILE] = ToolReport(
                tool=TOOL_CPROFILE, ok=False, error=str(exc)
            )
    else:
        result[TOOL_CPROFILE] = ToolReport(
            tool=TOOL_CPROFILE, ok=False, error="产物不存在"
        )

    # py-spy
    spy_path = reports_dir / "pyspy.speedscope.json"
    if spy_path.exists():
        try:
            metrics = summarize_speedscope(spy_path, top=top)
            result[TOOL_PYSPY] = ToolReport(
                tool=TOOL_PYSPY,
                ok=True,
                metrics=metrics,
                unit="秒",
                note="**不含 IO 等待**（默认排除空闲线程）",
            )
        except Exception as exc:  # noqa: BLE001
            result[TOOL_PYSPY] = ToolReport(
                tool=TOOL_PYSPY, ok=False, error=str(exc)
            )
    else:
        result[TOOL_PYSPY] = ToolReport(
            tool=TOOL_PYSPY, ok=False, error="产物不存在"
        )

    # Scalene
    scalene_path = reports_dir / "scalene.json"
    if scalene_path.exists():
        try:
            metrics = summarize_scalene(scalene_path, top=top)
            result[TOOL_SCALENE] = ToolReport(
                tool=TOOL_SCALENE,
                ok=True,
                metrics=metrics,
                unit="秒",
                note="**仅 Python 代码**，不含 IO 等待与 C 扩展",
            )
        except Exception as exc:  # noqa: BLE001
            result[TOOL_SCALENE] = ToolReport(
                tool=TOOL_SCALENE, ok=False, error=str(exc)
            )
    else:
        result[TOOL_SCALENE] = ToolReport(
            tool=TOOL_SCALENE, ok=False, error="产物不存在"
        )

    return result


def build_io_wait_metrics(reports_dir: Path, top: int = 8) -> list[Metric]:
    """读取 Scalene 的 IO 等待指标（若产物存在）。"""
    scalene_path = reports_dir / "scalene.json"
    if not scalene_path.exists():
        return []
    try:
        return summarize_scalene_io_wait(scalene_path, top=top)
    except Exception:  # noqa: BLE001
        return []


def write_report(
    reports: dict[str, ToolReport],
    out_path: Path,
    title: str,
    io_metrics: list[Metric] | None = None,
) -> None:
    """把报告写到 Markdown 文件。"""
    content = "\n".join(
        [
            f"# {title}",
            "",
            "> 由 `agent_perf_profiler.report` 自动生成。",
            "",
            render_comparison(reports),
            render_metric_names(reports),
            render_io_wait(reports, io_metrics or []),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def load_raw_json(path: Path) -> dict:
    """读取工具产物原始 JSON（调试用）。"""
    return json.loads(path.read_text(encoding="utf-8"))
