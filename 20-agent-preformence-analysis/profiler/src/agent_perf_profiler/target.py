"""剖析靶子的可执行入口。

用法（在**项目内 venv** 里跑）：

    uv run python -m agent_perf_profiler.target naive
    uv run python -m agent_perf_profiler.target optimized

它做三件事：
1. 跑一遍工作负载（naive 或 optimized）；
2. 打印一行结果摘要（供人工核对「两版行为一致」）；
3. **故意留一点存活时间**，让采样器（py-spy）能抓到足够样本。

⚠️ 第 3 点很重要：py-spy 是**采样器**，如果进程秒退，它可能只采到几帧。
   程序末尾的 sleep 是为了让 profile 有足够样本量 —— 否则会得到
   「Samples: 0」的假象，看起来像工具坏了，其实是被测程序跑太快。

被剖析的代码路径要足够长，所以默认把工作负载跑 `reps` 遍。
"""

from __future__ import annotations

import argparse
import sys
import time

from .naive import NaiveAgent
from .optimized import OptimizedAgent
from .workload import WorkloadSpec

# 采样器需要足够的运行时间才能采到有意义的样本
_TAIL_SLEEP_SEC = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_perf_profiler.target",
        description="性能剖析靶子：跑一遍（低效 / 优化）Agent 工作负载",
    )
    parser.add_argument(
        "variant",
        choices=("naive", "optimized"),
        help="跑哪个版本：naive=故意低效，optimized=优化后",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="工作负载重复次数（让采样器采到足够样本；默认 3）",
    )
    parser.add_argument(
        "--no-tail-sleep",
        action="store_true",
        help="末尾不 sleep（只在自己手动测采样率时用）",
    )
    args = parser.parse_args(argv)

    spec = WorkloadSpec()
    agent_cls = NaiveAgent if args.variant == "naive" else OptimizedAgent
    agent = agent_cls(spec)

    start = time.perf_counter()
    result = None
    for _ in range(max(args.reps, 1)):
        result = agent.run()
    elapsed = time.perf_counter() - start

    assert result is not None
    print(f"[{args.variant}] {result.summary()}", file=sys.stderr)
    print(f"[{args.variant}] elapsed={elapsed:.4f}s", file=sys.stderr)

    if not args.no_tail_sleep:
        time.sleep(_TAIL_SLEEP_SEC)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())