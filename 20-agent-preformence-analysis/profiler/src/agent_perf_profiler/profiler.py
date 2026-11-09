"""剖析驱动：把三个工具的调用封装成统一接口。

## 为什么要有这一层

三个工具的命令行接口完全不同，输出格式也完全不同：

| 工具 | 调用方式 | 输出 |
|---|---|---|
| cProfile | `python -m cProfile -o out.prof script.py` | 二进制 `.prof` |
| py-spy | `py-spy record --pid <PID> -o out.json --format speedscope` | speedscope JSON |
| Scalene | `scalene run -o out.json script.py` | Scalene 私有 JSON |

如果让 README / Makefile 直接写这三条命令，那么：
- 换工具要改多处；
- 路径管理（产物落在哪）会散落；
- 「跑一次剖析并汇总」这个动作无法脚本化。

所以这里统一成 `profile_with(tool, target, out_dir)`。

## 一个实测踩出来的坑：Scalene 的 CLI 在 2.x 变了

旧教程写的是 `scalene myscript.py`，2.x 之后**必须**是 `scalene run myscript.py`。
照旧教程抄命令，scalene 会打出帮助文本然后退出 —— 看起来像「装错了」，
其实是**参数语法变了**。

另一个坑：Scalene 完整模式（默认，含内存追踪）在本机 Python 3.9 下会被
SIGKILL（退出码 137），换到 3.12 + `--cpu-only` 组合才稳定。
项目固定用 **Python 3.12** 并在脚本里显式传参，就是为了绕开这两个坑。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ── 剖析工具清单 ──────────────────────────────────────────────────────

TOOL_CPROFILE = "cprofile"
TOOL_PYSPY = "py-spy"
TOOL_SCALENE = "scalene"

ALL_TOOLS = (TOOL_CPROFILE, TOOL_PYSPY, TOOL_SCALENE)


@dataclass
class Metric:
    """一条剖析指标：**名称 + 位置 + 绝对耗时 + 占比**。

    ## 为什么要有这个结构

    早先各工具只返回一个 `(名称, 数值)` 元组，且数值语义还不一样
    （cProfile 是秒、py-spy 是百分比、Scalene 是 CPU 百分比）。
    报告因此只能并排展示三列**无法直接比较**的数字。

    统一成 `Metric` 之后，每个指标都同时具备：
    - `seconds`：**绝对耗时（秒）** —— 「这个热点到底吃掉了多少时间」
    - `percent`：**占比（%）** —— 「它占总时间的多少」
    - `name` / `location`：**指标名称与位置** —— 「是谁、在哪」

    ⚠️ **两个维度缺一不可**：
    一个占 87% 但总耗时只有 0.1s 的热点，和一个占 10% 但耗了 2s 的热点，
    优化价值完全不同。只看占比会漏掉后者，只看绝对值会被小而多的项淹没。

    ## 跨工具归并的坑

    三个工具的函数名格式完全不同（`naive_bm25_score` /
    `NaiveAgent.naive_bm25_score` / `naive.py:naive_bm25_score`），
    归并前必须先归一化，否则**同一个热点会被拆成三行**。
    """

    #: 指标名称（函数名 / 方法名）。
    name: str

    #: 位置（`文件:行号`），便于直接跳到源码。
    location: str = ""

    #: **绝对耗时（秒）**。
    seconds: float = 0.0

    #: **占比（%）**。
    percent: float = 0.0

    #: 原始单位（speedscope 是 `seconds`，其它工具统一为秒）。
    unit: str = "seconds"

    #: 是否是 **IO 等待**（不是 CPU 消耗）。
    is_io_wait: bool = False

    #: 是否是 **C 扩展 / 原生库**时间。
    is_c_native: bool = False

    @property
    def display_name(self) -> str:
        """展示用的名称（含位置，便于定位源码）。"""
        return f"{self.name} @ {self.location}" if self.location else self.name


def _normalize_key(name: str) -> str:
    """把不同工具的函数名归一化成可比较的 key。

    ⚠️ 三个工具的命名风格：
    - cProfile：`naive_bm25_score` / `<genexpr>`
    - py-spy：  `NaiveAgent.naive_bm25_score` / `NaiveAgent.naive_bm25_score.<locals>.<genexpr>`
    - Scalene： `naive.py:NaiveAgent.naive_bm25_score`

    归一化规则：去掉类名限定与 `<locals>` 限定，保留**最内层**的名字。

    ## ⚠️ 归并键必须带上「宿主函数」，否则会误并

    光按最内层名字归并，`naive.py:135` 的 `<genexpr>` 会和
    另一个 `<genexpr>`（`naive.py:130`）**并成一行** ——
    但它们是两处完全不同的代码。

    而 py-spy 的 `NaiveAgent.naive_bm25_score.<locals>.<genexpr>`
    与 cProfile 的 `<genexpr>`（同在 `naive.py:135`）**确实是同一个**。

    解法：**位置优先，名字兜底**。
    - 有位置时用 `文件:行号`（cProfile 与 py-spy 在同一行就是同一处）
    - 没位置时才退回函数名（`<built-in method time.sleep>` 这类没有源码位置）

    ⚠️ 但 cProfile 与 py-spy 的行号可能差几行（装饰器/包装层），
    所以还得**同时**保留一个「宽松匹配」：函数名相同且行号相差 ≤ 3 行时
    也认为是同一处 —— 这在 `merge_reports` 里做二次归并。
    """
    key = name.strip()

    # 去掉文件名前缀：`naive.py:xxx` → `xxx`
    if ":" in key:
        head, _, tail = key.rpartition(":")
        if "/" in head or head.endswith(".py"):
            key = tail

    # 去掉类名限定与 `<locals>` 限定：`A.b.<locals>.<genexpr>` → `<genexpr>`
    if "." in key:
        key = key.split(".")[-1]

    return key


@dataclass
class ProfileResult:
    """一次剖析的结果摘要。"""

    tool: str
    ok: bool
    output_path: str
    message: str = ""
    exit_code: int = 0


def venv_python(project_dir: Path) -> Path:
    """返回项目内 venv 的 python 解释器路径。

    ⚠️ **绝不用 `sys.executable` 之外的解释器**：
    剖析必须跑在项目自己的 venv 里，否则装的 py-spy / scalene 版本不对，
    而且会污染本机 Python 环境。
    """
    candidate = project_dir / ".venv" / "bin" / "python"
    if not candidate.exists():
        raise FileNotFoundError(
            f"找不到 venv 解释器 {candidate}；先执行 `uv venv` + `uv sync`"
        )
    return candidate


def venv_bin(project_dir: Path, tool: str) -> Path:
    """返回 venv 里某个工具的可执行路径。"""
    candidate = project_dir / ".venv" / "bin" / tool
    if not candidate.exists():
        raise FileNotFoundError(
            f"找不到 {tool}（{candidate}）；先执行 `uv sync` 安装开发依赖"
        )
    return candidate


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
    """执行命令并返回 (退出码, 合并输出)。"""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _module_env(project_dir: Path) -> dict[str, str]:
    """构造带 ``PYTHONPATH`` 的环境变量。

    ⚠️ 为什么必须显式设 PYTHONPATH：
    - 容器里跑的是**裸解释器**（`python3 -m ...`），没有 venv 的 site-packages，
      所以 `src/` 必须在 PYTHONPATH 里；
    - 宿主机用 venv 时，项目本身是 editable 安装的，但显式设 PYTHONPATH
      能保证「换一种跑法结果一致」——剖析结论不依赖运行方式。
    """
    env = dict(os.environ)
    src = str(project_dir / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


# ── 被剖析程序的调用方式 ─────────────────────────────────────────────
#
# ⚠️ 统一用 `-m agent_perf_profiler.target`（**模块模式**）而不是脚本路径：
#   模块模式下 `sys.path[0]` 是当前工作目录，行为更接近真实服务
#   （而不是「直接跑一个 .py 文件」的特殊模式）。
#
#   变体（naive / optimized）与重复次数通过参数传入，
#   三个工具**共用同一组参数** —— 这是对比实验成立的前提。
_TARGET_MODULE = "agent_perf_profiler.target"


@dataclass(frozen=True)
class TargetSpec:
    """被剖析程序的规格（三个工具共用同一份）。"""

    variant: str = "naive"
    reps: int = 1
    tail_sleep: bool = True

    def module_args(self) -> list[str]:
        """转成命令行参数。

        ⚠️ `tail_sleep` 默认**开**：给采样器留采样窗口，
        否则 py-spy 可能只采到几帧（实测 Samples: 2）。
        """
        args = ["-m", _TARGET_MODULE, self.variant, "--reps", str(self.reps)]
        if not self.tail_sleep:
            args.append("--no-tail-sleep")
        return args


# ── cProfile ─────────────────────────────────────────────────────────


def run_cprofile(
    project_dir: Path, out_dir: Path, spec: TargetSpec
) -> ProfileResult:
    """用 cProfile 剖析（标准库，零额外依赖）。

    cProfile 是**确定性统计剖析器**：它用 `sys.setprofile` 钩住每一次
    函数调用与返回，所以**每次调用都被计数**（不像采样器可能漏采）。

    代价：钩子本身有开销，且会**改变被测程序的时序** ——
    所以它测出来的绝对时间**偏大**，适合找「谁被调用得最多」，
    不适合测「真实耗时」。

    另外：`time.sleep` 在 cProfile 里是**墙钟时间**，
    它会把 IO 等待算进 `tottime` —— 这是最常见的误读点。
    """
    out_path = out_dir / "cprofile.prof"
    out_dir.mkdir(parents=True, exist_ok=True)

    code, output = _run(
        [
            str(venv_python(project_dir)),
            "-m",
            "cProfile",
            "-o",
            str(out_path),
            *spec.module_args(),
        ],
        cwd=project_dir,
        env=_module_env(project_dir),
    )
    ok = code == 0 and out_path.exists()
    return ProfileResult(
        tool=TOOL_CPROFILE,
        ok=ok,
        output_path=str(out_path),
        message=output.strip(),
        exit_code=code,
    )


# ── py-spy ───────────────────────────────────────────────────────────


def run_py_spy(project_dir: Path, out_dir: Path, spec: TargetSpec) -> ProfileResult:
    """用 py-spy 剖析（采样型）。

    ⚠️ **macOS 上必须进容器**，原因见 `docker/Dockerfile`：
    py-spy 靠 `task_for_pid` 读目标进程内存，macOS 的 SIP 不允许
    普通用户 attach 别的进程（官方文档：「OSX always requires running as root」）。
    重签名（`codesign` 加 `get-task-allow`）**也绕不过去**（实测过）。

    Linux 下 py-spy attach 自己 spawn 的子进程不需要 root，
    只需要容器给 `SYS_PTRACE` 能力。

    py-spy 是**采样器**：每 N 毫秒抓一次栈。
    优点是开销极低、不改代码、可 attach 线上进程；
    缺点是**短函数可能一次都没被采到**，
    且默认**排除空闲线程**（所以 IO 等待不会出现在报告里）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pyspy.speedscope.json"

    # 容器里写的是 `/work/<相对路径>` —— 必须和宿主机产物路径一一对应，
    # 否则报告解析会找不到文件（实测踩过：写到了 /work/ 根目录，
    # 而宿主机在 reports/naive/ 下，py-spy 报「成功」但产物找不到）。
    try:
        rel_out = out_path.resolve().relative_to(project_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"输出目录 {out_dir} 必须在项目目录 {project_dir} 之内"
        ) from exc

    cmd = [
        "docker",
        "run",
        "--rm",
        "--cap-add",
        "SYS_PTRACE",
        "--security-opt",
        "seccomp=unconfined",
        "-v",
        f"{project_dir}:/work",
        "-w",
        "/work",
        "-e",
        "PYTHONPATH=/work/src",
        "agent-perf-tools",
        "py-spy",
        "record",
        "--rate",
        "200",
        "--format",
        "speedscope",
        "-o",
        "/work/" + rel_out.as_posix(),
        "--",
        "python3",
        *spec.module_args(),
    ]
    code, output = _run(cmd, cwd=project_dir)
    ok = code == 0 and out_path.exists()
    return ProfileResult(
        tool=TOOL_PYSPY,
        ok=ok,
        output_path=str(out_path),
        message=output.strip(),
        exit_code=code,
    )


# ── Scalene ──────────────────────────────────────────────────────────


def run_scalene(project_dir: Path, out_dir: Path, spec: TargetSpec) -> ProfileResult:
    """用 Scalene 剖析（CPU + 内存 + 异步）。

    ⚠️ Scalene 2.x 的 CLI 是 `scalene run`（**不是** `scalene <file>`）。

    Scalene 的独有价值：它把 **Python 时间**和**原生库时间**分开
    （`n_python_fraction`），并且统计**内存分配**与**异步任务的并发度**。
    cProfile 和 py-spy 都做不到后者。

    这里显式传 `--cpu-only`：完整模式（含内存追踪）在部分 Python 版本上
    会被 SIGKILL（实测 Python 3.9 退出码 137），3.12 下也出现过。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scalene.json"

    cmd = [
        str(venv_bin(project_dir, "scalene")),
        "run",
        "--cpu-only",
        "-o",
        str(out_path),
        *spec.module_args(),
    ]
    code, output = _run(cmd, cwd=project_dir, env=_module_env(project_dir))
    ok = code == 0 and out_path.exists()
    return ProfileResult(
        tool=TOOL_SCALENE,
        ok=ok,
        output_path=str(out_path),
        message=output.strip(),
        exit_code=code,
    )


# ── 统一入口 ─────────────────────────────────────────────────────────

_RUNNERS = {
    TOOL_CPROFILE: run_cprofile,
    TOOL_PYSPY: run_py_spy,
    TOOL_SCALENE: run_scalene,
}


def profile_with(
    tool: str, project_dir: Path, out_dir: Path, spec: TargetSpec
) -> ProfileResult:
    """用指定工具剖析目标程序。"""
    if tool not in _RUNNERS:
        raise ValueError(f"未知剖析工具 {tool}；可选：{', '.join(ALL_TOOLS)}")
    return _RUNNERS[tool](project_dir, out_dir, spec)


# ── 报告解析 ─────────────────────────────────────────────────────────


def summarize_cprofile(prof_path: Path, top: int = 15) -> list["Metric"]:
    """解析 cProfile 产物，输出自耗时（秒）指标列表。

    用 `pstats` 而不是 `snakeviz`/`gprof2dot`：
    `pstats` 是标准库，零依赖，而且能直接读 `.prof` 二进制。

    ## 为什么排的是 `tottime` 而不是 `cumtime`（实测踩过的坑）

    `cumtime`（累计耗时）**包含子函数**，所以顶层框架帧
    （`<module>` → `run_module` → `_run_code` → `main`）全都显示 1.53 秒 ——
    它们是**同一段时间**，不是各自的耗时。排出来的 Top 5 全是噪音，
    真正的热点被挤掉了。

    `tottime`（自耗时）是「这个函数自己花掉的时间，不含子函数」，
    排出来的才是真正的热点。

    ⚠️ 另外 `time.sleep` 会在 tottime 里出现（0.58s）——
       它代表 **IO 等待**而非 CPU 消耗，这是 cProfile 最常见的误读点。
    """
    import pstats

    stats = pstats.Stats(str(prof_path))
    rows: list[Metric] = []
    for func, (_cc, _nc, tottime, _ct, _callers) in stats.stats.items():
        filename, lineno, name = func
        rows.append(
            Metric(
                name=name,
                location=f"{Path(filename).name}:{lineno}",
                seconds=tottime,
                # ⚠️ 名字实际是 `<built-in method time.sleep>`，
                #    不是 `sleep` —— 只判 `name == "sleep"` 会漏掉。
                is_io_wait="sleep" in name,
            )
        )
    rows.sort(key=lambda m: m.seconds, reverse=True)

    # ⭐ 补上占比。
    #
    #   cProfile 的产物里**没有**百分比字段（只有绝对秒数），
    #   占比要自己算 —— 分母用「所有函数自耗时之和」。
    #
    #   ⚠️ 分母**包含** IO 等待：
    #      排除掉 `time.sleep` 会让 CPU 热点占比虚高（分母被压缩），
    #      读者会以为「这个函数占了 50% 的 CPU」，
    #      实际上整个进程只有 1.4s、它只占 0.7s。
    #      IO 等待单独标成「IO 等待」类别，占比照实计入总时间。
    total = sum(m.seconds for m in rows)
    for m in rows:
        m.percent = (m.seconds * 100.0 / total) if total else 0.0

    return rows[:top]


def summarize_speedscope(path: Path, top: int = 15) -> list["Metric"]:
    """解析 speedscope 格式（py-spy 产出），输出采样耗时（秒）指标列表。

    ## speedscope 的数据结构（实测踩过的坑）

    帧表**不在** `profiles[0].frames` 里，而在**顶层** `shared.frames`：

    ```
    {
      "shared": { "frames": [ {"name":..., "file":..., "line":...}, ... ] },
      "profiles": [ { "type":"sampled", "samples": [[帧索引...], ...],
                     "weights":[0.005, ...], "unit":"seconds" } ]
    }
    ```

    ⚠️ 照着网上常见的 speedscope 示例写 `profiles[0]["frames"]` 会直接
    `KeyError: 'frames'` —— 因为那是「每个 profile 各自带一份帧表」的
    变体，py-spy 用的是**共享帧表**的变体。

    `samples` 是**栈帧列表**的列表，每个元素从栈底到栈顶。

    ## ⭐ 怎么把占比换算成**绝对秒数**

    speedscope 的每个采样配一个 `weights` 元素（本项目是 0.005 = 1/采样率），
    它表示「这个采样代表多长时间」。所以：

    ```text
    某帧的耗时 = Σ 该帧作为叶子帧出现时的 weight
    ```

    ⚠️ 早先只统计「采样次数占比」，看不出「这个热点到底吃掉了多少秒」——
       一个占 87% 但总耗时只有 0.1s 的热点，和一个占 10% 但耗了 2s 的热点，
       优化价值完全不同。**占比 + 绝对时间**两个维度都要。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles") or []
    if not profiles:
        return []

    # 帧表：优先顶层 shared.frames，退化到 profile 内的 frames
    shared = data.get("shared") or {}
    frame_list = shared.get("frames") or profiles[0].get("frames") or []
    names = [f.get("name", "?") for f in frame_list]
    locations = [
        f"{Path(f.get('file', '')).name}:{f.get('line', '')}" for f in frame_list
    ]

    profile = profiles[0]
    samples = profile.get("samples") or []
    weights = profile.get("weights") or []

    # 每个采样对应的时长；weights 缺失时退化为「均分总时长」
    total_seconds = float(profile.get("endValue") or 0.0)
    unit = profile.get("unit") or "seconds"

    seconds_by_frame: dict[str, float] = {}
    seconds_by_location: dict[str, str] = {}

    for index, stack in enumerate(samples):
        # ⚠️ 只统计**叶子帧**（栈顶 = 真正在执行的函数）。
        #
        #   speedscope 的每个采样是一整条栈（栈底 → 栈顶）。
        #   如果把所有帧都展开计数，得到的不是「谁在执行」，
        #   而是「谁在调用链上」—— 一个被调用 33 万次的函数
        #   会因为出现在 33 万条栈里而霸榜。
        #
        #   正确做法：只取栈顶那一帧。
        if not stack:
            continue
        frame_index = stack[-1]
        name = names[frame_index] if frame_index < len(names) else "?"
        if frame_index < len(locations):
            seconds_by_location[name] = locations[frame_index]

        if index < len(weights):
            weight = float(weights[index])
        elif total_seconds and samples:
            weight = total_seconds / len(samples)
        else:
            weight = 0.0
        seconds_by_frame[name] = seconds_by_frame.get(name, 0.0) + weight

    total_weight = sum(seconds_by_frame.values())
    rows = [
        Metric(
            name=name,
            location=seconds_by_location.get(name, ""),
            seconds=seconds,
            percent=(seconds * 100.0 / total_weight) if total_weight else 0.0,
            unit=unit,
        )
        for name, seconds in seconds_by_frame.items()
    ]
    rows.sort(key=lambda m: m.seconds, reverse=True)
    return rows[:top]


def summarize_scalene(path: Path, top: int = 15) -> list["Metric"]:
    """解析 Scalene 产物，输出 CPU 耗时（秒）指标列表。

    Scalene 的 JSON 结构：
    `files` → 每个文件 → `functions` → 每个函数 → `n_cpu_percent_python`。

    ⚠️ 只有 `n_cpu_percent_python` 才是「Python 代码自身」的 CPU 占比；
    `n_cpu_percent_c` 是原生库（C 扩展）的时间。

    ## ⭐ 怎么把占比换算成**绝对秒数**

    顶层有 `elapsed_time_sec`（整个剖析过程的墙钟时长），
    所以：

    ```text
    某函数的 CPU 秒数 = elapsed_time_sec × n_cpu_percent_python / 100
    ```

    ## ⭐ `n_sys_percent` 是**另一个维度的指标**

    Scalene 独有：它把「系统调用/IO 等待」单独统计为 `n_sys_percent`。
    在本项目里 `time.sleep`（模拟 LLM 网络往返）就落在这一项 ——
    **这正是 cProfile 里 `time.sleep` 排第一、而 py-spy 看不到的那部分**。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    elapsed = float(data.get("elapsed_time_sec") or 0.0)

    rows: list[Metric] = []
    for file_path, file_info in (data.get("files") or {}).items():
        file_name = Path(file_path).name
        for func in file_info.get("functions") or []:
            name = func.get("line") or func.get("name") or "?"
            pct = float(func.get("n_cpu_percent_python") or 0.0)
            if pct <= 0:
                continue
            rows.append(
                Metric(
                    name=name,
                    location=f"{file_name}:{func.get('lineno', '')}",
                    seconds=elapsed * pct / 100.0,
                    percent=pct,
                    is_c_native=False,
                )
            )
    rows.sort(key=lambda m: m.seconds, reverse=True)
    return rows[:top]


def summarize_scalene_io_wait(path: Path, top: int = 10) -> list["Metric"]:
    """解析 Scalene 的 **IO 等待（系统调用）** 指标。

    ⚠️ 这是 Scalene 独有的价值：cProfile 把 IO 等待混在 `time.sleep` 里
    （看不出是哪个函数在等），py-spy 直接把空闲线程排除掉。
    只有 Scalene 会明确告诉你「有 X% 的时间在等系统调用」。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    elapsed = float(data.get("elapsed_time_sec") or 0.0)

    rows: list[Metric] = []
    for file_path, file_info in (data.get("files") or {}).items():
        file_name = Path(file_path).name
        for func in file_info.get("functions") or []:
            name = func.get("line") or func.get("name") or "?"
            pct = float(func.get("n_sys_percent") or 0.0)
            if pct <= 0:
                continue
            rows.append(
                Metric(
                    name=name,
                    location=f"{file_name}:{func.get('lineno', '')}",
                    seconds=elapsed * pct / 100.0,
                    percent=pct,
                    is_io_wait=True,
                )
            )
    rows.sort(key=lambda m: m.seconds, reverse=True)
    return rows[:top]
