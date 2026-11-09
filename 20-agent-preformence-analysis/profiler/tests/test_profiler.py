"""测试：剖析驱动的正确性。

这组测试锁的是「工具调用方式」本身 ——
因为本项目踩过的坑几乎全在这一层：
CLI 语法变了、输出路径对不上、speedscope 结构读错了。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_perf_profiler.profiler import (
    TOOL_CPROFILE,
    TOOL_PYSPY,
    TOOL_SCALENE,
    TargetSpec,
    profile_with,
    summarize_cprofile,
    summarize_scalene,
    summarize_speedscope,
)
from agent_perf_profiler.report import build_reports, render_comparison

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_DIR / "reports"


def test_target_spec_module_args() -> None:
    """三个工具共用同一份目标程序参数。"""
    spec = TargetSpec(variant="naive", reps=2)
    args = spec.module_args()
    assert args[:3] == ["-m", "agent_perf_profiler.target", "naive"]
    assert "--reps" in args
    assert "2" in args
    assert "--no-tail-sleep" not in args


def test_target_spec_tail_sleep_can_be_disabled() -> None:
    """tail sleep 可以关掉（只在自己测采样率时用）。"""
    spec = TargetSpec(variant="optimized", reps=1, tail_sleep=False)
    assert "--no-tail-sleep" in spec.module_args()


def test_scalene_cli_uses_run_subcommand() -> None:
    """⭐ Scalene 2.x 的 CLI 是 `scalene run`，不是 `scalene <file>`。

    照旧教程抄命令，scalene 会打帮助文本然后退出 ——
    看起来像「装错了」，其实是语法变了。
    """
    source = (PROJECT_DIR / "src" / "agent_perf_profiler" / "profiler.py").read_text(
        encoding="utf-8"
    )
    assert '"run",' in source
    # 旧写法 `scalene <file>` 不应出现
    assert "scalene\", \n" not in source


def test_unknown_tool_raises() -> None:
    """未知工具必须报错，而不是静默跳过。"""
    import pytest

    with pytest.raises(ValueError):
        profile_with("nonexistent", PROJECT_DIR, PROJECT_DIR, TargetSpec())


def test_speedscope_parser_reads_shared_frames() -> None:
    """⭐ speedscope 的帧表在**顶层** `shared.frames`，不在 `profiles[0].frames`。

    照常见示例读 `profiles[0]["frames"]` 会直接 KeyError。
    """
    fake = {
        "shared": {
            "frames": [
                {"name": "hot_loop", "file": "x.py", "line": 1},
                {"name": "caller", "file": "x.py", "line": 2},
            ]
        },
        "profiles": [
            {
                "type": "sampled",
                # 两个采样：栈顶分别是 caller（帧1）与 hot_loop（帧0）
                "samples": [[0, 1], [0]],
                "weights": [1, 1],
            }
        ],
    }
    path = PROJECT_DIR / "reports" / "_test_speedscope.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fake), encoding="utf-8")
    try:
        rows = summarize_speedscope(path, top=10)
    finally:
        path.unlink(missing_ok=True)

    names = {m.name for m in rows}
    assert names == {"hot_loop", "caller"}
    # 只统计叶子帧：hot_loop 出现 1 次，caller 出现 1 次
    total = sum(m.percent for m in rows)
    assert abs(total - 100.0) < 1e-6


def test_speedscope_parser_converts_weights_to_seconds() -> None:
    """⭐ 采样权重必须换算成**绝对秒数**。

    早先只统计采样次数占比，看不出「这个热点吃掉了多少秒」。
    """
    fake = {
        "shared": {"frames": [{"name": "hot_loop", "file": "x.py", "line": 7}]},
        "profiles": [
            {
                "type": "sampled",
                "samples": [[0], [0], [0]],
                "weights": [0.5, 0.5, 0.5],
                "unit": "seconds",
                "endValue": 1.5,
            }
        ],
    }
    path = PROJECT_DIR / "reports" / "_test_speedscope3.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fake), encoding="utf-8")
    try:
        rows = summarize_speedscope(path, top=10)
    finally:
        path.unlink(missing_ok=True)

    assert rows[0].name == "hot_loop"
    # 3 次采样 × 0.5 秒权重 = 1.5 秒
    assert rows[0].seconds == pytest.approx(1.5)
    assert rows[0].percent == pytest.approx(100.0)
    assert rows[0].location == "x.py:7"


def test_speedscope_parser_handles_missing_frames_gracefully() -> None:
    """帧索引越界时不能崩溃（应回退到 `?`）。"""
    fake = {
        "shared": {"frames": [{"name": "only", "file": "x.py", "line": 1}]},
        "profiles": [{"type": "sampled", "samples": [[99]], "weights": [1]}],
    }
    path = PROJECT_DIR / "reports" / "_test_speedscope2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fake), encoding="utf-8")
    try:
        rows = summarize_speedscope(path, top=10)
    finally:
        path.unlink(missing_ok=True)

    assert rows[0].name == "?"


def test_reports_dir_has_expected_tools() -> None:
    """构建报告时三个工具的键都要在（即使失败也要有占位）。"""
    reports = build_reports(REPORTS_DIR / "nonexistent")
    assert set(reports) == {TOOL_CPROFILE, TOOL_PYSPY, TOOL_SCALENE}


def test_render_comparison_produces_table() -> None:
    """对比表必须能渲染出 Markdown 表格。"""
    reports = build_reports(REPORTS_DIR / "nonexistent")
    markdown = render_comparison(reports)
    assert "|" in markdown
    assert "cProfile" in markdown


def test_cprofile_summarizer_sorts_by_tottime() -> None:
    """⭐ cProfile 必须按 **tottime**（自耗时）排序，不能按 cumtime。

    cumtime 包含子函数，顶层框架帧（`<module>` → `run_module` → ...）
    会显示成同一段时间，霸占整个 Top 5，真正的热点被挤掉。
    """
    prof_path = REPORTS_DIR / "naive" / "cprofile.prof"
    if not prof_path.exists():
        return  # 没有产物就跳过（CI 里不跑剖析）

    rows = summarize_cprofile(prof_path, top=10)
    values = [m.seconds for m in rows]
    assert values == sorted(values, reverse=True)
    # 顶层框架帧不应该出现在靠前的位置
    top_names = [m.name for m in rows[:5]]
    assert not any("<module>" in name for name in top_names)


def test_cprofile_summarizer_computes_percent() -> None:
    """⭐ cProfile 产物里没有百分比字段，必须自己算出来。"""
    prof_path = REPORTS_DIR / "naive" / "cprofile.prof"
    if not prof_path.exists():
        return

    rows = summarize_cprofile(prof_path, top=10)
    assert all(m.percent > 0 for m in rows if m.seconds > 0)


def test_cprofile_marks_io_wait() -> None:
    """⚠️ `time.sleep` 必须被标成 IO 等待，否则读者会去优化它。

    实际名字是 `<built-in method time.sleep>`，
    只判 `name == "sleep"` 会漏掉。
    """
    prof_path = REPORTS_DIR / "naive" / "cprofile.prof"
    if not prof_path.exists():
        return

    rows = summarize_cprofile(prof_path, top=30)
    sleep_rows = [m for m in rows if "sleep" in m.name]
    assert sleep_rows, "应当能读到 time.sleep"
    assert all(m.is_io_wait for m in sleep_rows)


def test_scalene_summarizer_reads_cpu_percent() -> None:
    """Scalene 解析要读 `n_cpu_percent_python`。"""
    scalene_path = REPORTS_DIR / "naive" / "scalene.json"
    if not scalene_path.exists():
        return

    rows = summarize_scalene(scalene_path, top=10)
    values = [m.percent for m in rows]
    assert values == sorted(values, reverse=True)
    assert all(value >= 0 for value in values)


def test_scalene_summarizer_converts_percent_to_seconds() -> None:
    """⭐ Scalene 的占比必须换算成绝对秒数。"""
    scalene_path = REPORTS_DIR / "naive" / "scalene.json"
    if not scalene_path.exists():
        return

    rows = summarize_scalene(scalene_path, top=10)
    assert all(m.seconds > 0 for m in rows)
    # 换算必须单调：占比高的，秒数也高
    assert [m.seconds for m in rows] == sorted(
        [m.seconds for m in rows], reverse=True
    )
