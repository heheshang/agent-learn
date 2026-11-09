"""测试：报告层的**绝对耗时 + 占比 + 指标名称 + 跨工具归并**。

## 为什么这组测试重要

早先的报告只给「名称 + 占比」，有两个隐性缺陷：

1. **看不出绝对耗时** —— 占 87% 但只花 0.1s 的热点，
   和占 10% 但花 2s 的热点，优化价值完全不同。
2. **跨工具没归并** —— 三个工具的命名风格完全不同，
   不归一化的话同一个热点会被拆成三行，
   「几个工具共同指向」这个交叉验证信号就废了。

这组测试把这两条性质锁死。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_perf_profiler.profiler import Metric, _normalize_key
from agent_perf_profiler.report import (
    LINE_TOLERANCE,
    MergedRow,
    ToolReport,
    _line_delta,
    merge_reports,
    render_comparison,
    render_metric_names,
)


def _metric(
    name: str,
    seconds: float,
    percent: float = 0.0,
    location: str = "",
    is_io_wait: bool = False,
) -> Metric:
    return Metric(
        name=name,
        location=location,
        seconds=seconds,
        percent=percent,
        is_io_wait=is_io_wait,
    )


# ── 指标名称归一化 ────────────────────────────────────────────────


def test_normalize_key_strips_class_qualifier() -> None:
    """类名限定要剥掉：`NaiveAgent.naive_bm25_score` → `naive_bm25_score`。"""
    assert _normalize_key("NaiveAgent.naive_bm25_score") == "naive_bm25_score"


def test_normalize_key_strips_file_prefix() -> None:
    """文件名前缀要剥掉：`naive.py:NaiveAgent.foo` → `foo`。"""
    assert _normalize_key("naive.py:NaiveAgent.foo") == "foo"


def test_normalize_key_strips_locals_qualifier() -> None:
    """`<locals>` 限定要剥掉，只留最内层名字。"""
    assert (
        _normalize_key("NaiveAgent.naive_bm25_score.<locals>.<genexpr>")
        == "<genexpr>"
    )


# ── 行号漂移 ──────────────────────────────────────────────────────


def test_line_delta_parses_line_numbers() -> None:
    """行号差要从 `文件:行号` 里解析出来。"""
    assert _line_delta("naive.py:135", "naive.py:138") == 3


def test_line_delta_returns_huge_for_unparsable() -> None:
    """无法解析时返回极大值，表示「不可比」。"""
    assert _line_delta("~:0", "<frozen x>:757") == 10**9


def test_line_delta_rejects_non_source_locations() -> None:
    """⚠️⚠️ 无源文件的位置必须判为不可比。

    `~:0`（内建函数）行号是 0，`naive.py:2` 行号是 2 ——
    行号差只有 2，若只按行号判定就会被误并成一个热点。
    一个 C 实现的 `time.sleep` 和一个项目函数混为一谈，
    整张对比表的可信度就没了。
    """
    assert _line_delta("~:0", "naive.py:2") == 10**9
    assert _line_delta("<frozen importlib>:757", "naive.py:757") == 10**9


def test_line_delta_compares_same_source_file() -> None:
    """真实源码文件之间才能比行号。"""
    assert _line_delta("naive.py:135", "naive.py:136") == 1


def test_line_tolerance_is_small() -> None:
    """容差不能太大，否则会把不同函数并成一行。"""
    assert LINE_TOLERANCE <= 5


# ── 跨工具归并 ────────────────────────────────────────────────────


def test_merge_by_location_unifies_same_line() -> None:
    """⭐ 同一行 = 同一处代码，跨工具必须并成一行。

    cProfile 报 `naive.py:135`，py-spy 也报 `naive.py:135` ——
    这两列必须是同一行，否则「命中工具数」毫无意义。
    """
    reports = {
        "cprofile": ToolReport(
            tool="cprofile",
            ok=True,
            metrics=[_metric("<genexpr>", 0.7, 50.0, "naive.py:135")],
        ),
        "py-spy": ToolReport(
            tool="py-spy",
            ok=True,
            metrics=[_metric("<genexpr>", 1.0, 89.0, "naive.py:135")],
        ),
    }
    rows = merge_reports(reports, top=10)
    assert len(rows) == 1
    assert rows[0].tools_hit == 2


def test_merge_keeps_different_locations_separate() -> None:
    """⚠️ 不同行的同名函数不能误并（`naive.py:130` vs `naive.py:135`）。"""
    reports = {
        "cprofile": ToolReport(
            tool="cprofile",
            ok=True,
            metrics=[
                _metric("<genexpr>", 0.1, 5.0, "naive.py:130"),
                _metric("<genexpr>", 0.7, 50.0, "naive.py:135"),
            ],
        ),
    }
    rows = merge_reports(reports, top=10)
    assert len(rows) == 2


def test_merge_tolerates_small_line_drift() -> None:
    """行号漂移（≤ 容差）要能并成一行。

    真实场景：cProfile 与 py-spy 对同一函数报的行号可能差几行。
    """
    reports = {
        "cprofile": ToolReport(
            tool="cprofile",
            ok=True,
            metrics=[_metric("naive_bm25_score", 0.05, 3.0, "naive.py:120")],
        ),
        "py-spy": ToolReport(
            tool="py-spy",
            ok=True,
            metrics=[_metric("naive_bm25_score", 0.04, 3.5, "naive.py:122")],
        ),
    }
    rows = merge_reports(reports, top=10)
    assert len(rows) == 1
    assert rows[0].tools_hit == 2


def test_merge_does_not_unify_distant_lines() -> None:
    """⚠️ 行号差太大不能并 —— 那是两个不同的热点。"""
    reports = {
        "cprofile": ToolReport(
            tool="cprofile",
            ok=True,
            metrics=[_metric("foo", 0.05, 3.0, "naive.py:120")],
        ),
        "py-spy": ToolReport(
            tool="py-spy",
            ok=True,
            metrics=[_metric("foo", 0.04, 3.5, "naive.py:200")],
        ),
    }
    rows = merge_reports(reports, top=10)
    assert len(rows) == 2


def test_merge_falls_back_to_name_for_builtins() -> None:
    """内建函数没有源码位置，要按名字归并。"""
    reports = {
        "cprofile": ToolReport(
            tool="cprofile",
            ok=True,
            metrics=[
                _metric("<built-in method time.sleep>", 0.58, 41.0, "~:0", True)
            ],
        ),
        "py-spy": ToolReport(
            tool="py-spy",
            ok=True,
            metrics=[_metric("<built-in method time.sleep>", 0.5, 30.0, "~:0", True)],
        ),
    }
    rows = merge_reports(reports, top=10)
    assert len(rows) == 1
    assert rows[0].tools_hit == 2


def test_merge_sorts_by_hit_count_first() -> None:
    """⭐ 排序先看「被几个工具命中」，再看耗时。

    被多个工具共同指向的热点置信度最高，
    必须排在「某个工具口径下碰巧第一」的前面。
    """
    reports = {
        "cprofile": ToolReport(
            tool="cprofile",
            ok=True,
            metrics=[
                _metric("hot_single", 5.0, 60.0, "a.py:10"),
                _metric("hot_shared", 0.1, 1.0, "a.py:20"),
            ],
        ),
        "py-spy": ToolReport(
            tool="py-spy",
            ok=True,
            metrics=[_metric("hot_shared", 0.1, 1.0, "a.py:20")],
        ),
        "scalene": ToolReport(
            tool="scalene",
            ok=True,
            metrics=[_metric("hot_shared", 0.1, 1.0, "a.py:20")],
        ),
    }
    rows = merge_reports(reports, top=10)
    assert rows[0].display == "hot_shared"
    assert rows[0].tools_hit == 3


def test_max_seconds_reflects_largest_measurement() -> None:
    """`max_seconds` 取各工具报告的最大值。"""
    row = MergedRow(
        key="k",
        display="k",
        location="a.py:1",
        cprofile=_metric("k", 0.1),
        pyspy=_metric("k", 0.9),
    )
    assert row.max_seconds == pytest.approx(0.9)


# ── 渲染 ──────────────────────────────────────────────────────────


def test_comparison_table_contains_name_seconds_and_percent() -> None:
    """⭐ 表格必须同时含「指标名称 + 绝对耗时 + 占比」。"""
    reports = {
        "cprofile": ToolReport(
            tool="cprofile",
            ok=True,
            metrics=[_metric("<genexpr>", 0.7, 50.9, "naive.py:135")],
        ),
    }
    text = render_comparison(reports)
    assert "<genexpr>" in text
    assert "naive.py:135" in text
    assert "0.700s" in text
    assert "50.9%" in text


def test_comparison_table_warns_about_incomparable_units() -> None:
    """必须写清楚「三列秒数不可横向比较」。

    三个工具是三次独立运行，采集机制不同，
    不加这句警告读者会误读成「py-spy 测得更准」。
    """
    reports = {"cprofile": ToolReport(tool="cprofile", ok=True, metrics=[])}
    text = render_comparison(reports)
    assert "不可直接横向比较" in text


def test_metric_names_section_lists_each_tool() -> None:
    """分工具明细里要出现每个工具自己的表头。"""
    reports = {
        "cprofile": ToolReport(
            tool="cprofile", ok=True, metrics=[_metric("f", 1.0, 10.0, "a.py:1")]
        ),
        "py-spy": ToolReport(
            tool="py-spy", ok=True, metrics=[_metric("f", 1.0, 10.0, "a.py:1")]
        ),
    }
    text = render_metric_names(reports)
    assert "### cprofile" in text
    assert "### py-spy" in text


def test_failed_tool_renders_error_not_crash() -> None:
    """工具失败时报告要显示错误，而不是抛异常。"""
    reports = {"cprofile": ToolReport(tool="cprofile", ok=False, error="产物不存在")}
    text = render_comparison(reports)
    assert "⚠️" in text
