"""剖析实验的统一入口。

用法（在**项目内 venv** 里跑）：

    # 剖析 naive 版
    uv run python -m agent_perf_profiler.cli profile naive

    # 剖析 optimized 版
    uv run python -m agent_perf_profiler.cli profile optimized

    # 三个工具全跑一遍 + 生成对比报告
    uv run python -m agent_perf_profiler.cli compare

    # 只看报告
    uv run python -m agent_perf_profiler.cli report

⚠️ 所有命令都在**项目内 venv** 里执行，不碰本机 Python 环境。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .profiler import ALL_TOOLS, TargetSpec, profile_with
from .report import (
    build_io_wait_metrics,
    build_reports,
    render_comparison,
    render_io_wait,
    render_metric_names,
    write_report,
)

# 项目根目录（`profiler/`）
PROJECT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_DIR / "reports"


def cmd_profile(args: argparse.Namespace) -> int:
    """剖析单个版本。"""
    out_dir = Path(args.out) if args.out else REPORTS_DIR / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = TargetSpec(variant=args.variant, reps=args.reps)

    tools = [args.tool] if args.tool else list(ALL_TOOLS)
    failures = 0
    for tool in tools:
        print(f"==> 剖析 {tool} …", file=sys.stderr)
        result = profile_with(tool, PROJECT_DIR, out_dir, spec)
        status = "✓" if result.ok else "✗"
        print(f"    {status} {tool} → {result.output_path}", file=sys.stderr)
        if not result.ok:
            failures += 1
            print(f"      {result.message[:400]}", file=sys.stderr)

    if failures:
        print(f"\n⚠️ {failures} 个工具剖析失败（见上方输出）", file=sys.stderr)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """跑 naive + optimized 两个版本，生成三工具对比报告。

    这是本项目的主命令：它同时产出了
    ① 三个工具对同一程序的各自看法；
    ② 优化前后的耗时差值。
    """
    reports_dir = REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    for variant in ("naive", "optimized"):
        print(f"\n==> 剖析 {variant} …", file=sys.stderr)
        out_dir = reports_dir / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        spec = TargetSpec(variant=variant, reps=args.reps)
        for tool in ALL_TOOLS:
            result = profile_with(tool, PROJECT_DIR, out_dir, spec)
            status = "✓" if result.ok else "✗"
            print(f"    {status} {tool}", file=sys.stderr)
            if not result.ok:
                print(f"      {result.message[:300]}", file=sys.stderr)

        # 单独测一次「纯净耗时」（不挂剖析器），作为耗时基线
        timings[variant] = _measure(variant, args.reps)

    reports = build_reports(reports_dir / "naive")
    io_metrics = build_io_wait_metrics(reports_dir / "naive")
    comparison = render_comparison(reports)

    naive_t = timings["naive"]
    opt_t = timings["optimized"]
    speedup = (naive_t / opt_t) if opt_t else 0.0

    summary = [
        "",
        "=" * 72,
        "优化前后耗时对比（纯净运行，不挂剖析器）",
        "=" * 72,
        f"naive     : {naive_t:.4f}s",
        f"optimized : {opt_t:.4f}s",
        f"加速比    : {speedup:.2f}x（节省 {naive_t - opt_t:.4f}s）",
        "=" * 72,
        "",
        comparison,
    ]
    print("\n".join(summary), file=sys.stderr)

    out_md = reports_dir / "report.md"
    write_report(reports, out_md, "性能剖析报告（naive 版）", io_metrics=io_metrics)
    print(f"\n报告已写入 {out_md}", file=sys.stderr)
    return 0


def _measure(variant: str, reps: int) -> float:
    """测一次纯净耗时（不挂剖析器）。

    ⚠️ **必须单独测**：剖析器本身有开销（cProfile 尤其明显，
    它会给每次函数调用加钩子）。拿「挂了剖析器的耗时」做对比，
    测的是剖析器的开销，不是程序的耗时。
    """
    import subprocess

    cmd = [
        str(PROJECT_DIR / ".venv" / "bin" / "python"),
        "-m",
        "agent_perf_profiler.target",
        variant,
        "--reps",
        str(reps),
        "--no-tail-sleep",
    ]
    start = time.perf_counter()
    subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    return time.perf_counter() - start


def cmd_report(args: argparse.Namespace) -> int:
    """打印已生成的报告。"""
    reports = build_reports(REPORTS_DIR / args.variant)
    io_metrics = build_io_wait_metrics(REPORTS_DIR / args.variant)
    print(render_comparison(reports))
    print(render_metric_names(reports))
    if io_metrics:
        print(render_io_wait(reports, io_metrics))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_perf_profiler.cli",
        description="性能剖析实验台：cProfile / py-spy / Scalene",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_profile = sub.add_parser("profile", help="剖析单个版本")
    p_profile.add_argument("variant", choices=("naive", "optimized"))
    p_profile.add_argument("--tool", choices=ALL_TOOLS, help="只跑指定工具")
    p_profile.add_argument("--out", help="输出目录")
    p_profile.add_argument("--reps", type=int, default=1, help="每版重复次数")
    p_profile.set_defaults(func=cmd_profile)

    p_compare = sub.add_parser("compare", help="跑两版 + 生成对比报告")
    p_compare.add_argument("--reps", type=int, default=1, help="每版重复次数")
    p_compare.set_defaults(func=cmd_compare)

    p_report = sub.add_parser("report", help="打印已生成的报告")
    p_report.add_argument("--variant", choices=("naive", "optimized"), default="naive")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())