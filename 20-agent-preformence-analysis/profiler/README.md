# profiler —— Python 侧性能剖析实验台

> 本目录是 20-agent-preformence-analysis 的 **Python 侧**。
> Rust 侧（`pprof` + `criterion`）在上一级目录。

---

## ⚠️ 环境约定（重要）

**所有 Python 操作都在本目录的 `.venv` 里执行，绝不碰本机 Python 环境。**

```bash
# 创建项目内虚拟环境（uv）
uv venv --python 3.12 .venv

# 安装依赖（uv）
uv sync
```

依赖分两组：

| 组 | 内容 | 用途 |
|---|---|---|
| `dependencies` | **空**（零第三方运行时依赖） | 被剖析的靶子代码要「干净」 |
| `dependency-groups.dev` | `py-spy==0.4.3`、`scalene==2.4.0` | 剖析工具 |

### 为什么运行时依赖是空的

剖析实验的**核心变量是「被剖析代码本身」**。
如果靶子用了 requests / numpy / pydantic，
cProfile 会把库的 import 与内部调用也统计进来，热点被淹没。

所以靶子里的 LLM 调用、BM25 检索、JSON 编解码**全部手写** ——
这既是本仓库「手撕」的风格，也让 profile 干净到能一眼看出热点。

### 为什么固定 Python 3.12

Scalene 的完整模式（含内存追踪）在 Python 3.9 上会 **SIGKILL（退出码 137）**，
3.12 下稳定。所以 `.python-version` 锁 `3.12`。

---

## 目录结构

```text
profiler/
├── pyproject.toml            # uv 管理，零运行时依赖
├── .python-version           # 固定 3.12
├── docker/Dockerfile         # py-spy 专用镜像
├── src/agent_perf_profiler/
│   ├── workload.py           # 工作负载规格（两版共用 ★）
│   ├── tokenizer.py          # CJK bigram 分词（两版共用 ★）
│   ├── naive.py              # 低效版（剖析靶子）
│   ├── optimized.py          # 优化版（对照组）
│   ├── target.py             # 被剖析程序入口
│   ├── profiler.py           # 三工具驱动 + 产物解析
│   ├── report.py             # 对比报告
│   └── cli.py                # 命令行
└── tests/                    # 21 项测试
```

★ = **两版共用**：这是「行为等价」的前提，差值才能归因到实现。

---

## 快速上手

```bash
# 剖析 naive 版（三个工具全跑）
.venv/bin/python -m agent_perf_profiler.cli profile naive

# 只跑一个工具
.venv/bin/python -m agent_perf_profiler.cli profile naive --tool scalene

# 两版对比 + 优化前后差值（主命令）
.venv/bin/python -m agent_perf_profiler.cli compare

# 打印已生成的报告
.venv/bin/python -m agent_perf_profiler.cli report
```

产物落在 `reports/`：

```text
reports/
├── naive/
│   ├── cprofile.prof              # cProfile 二进制（pstats 读）
│   ├── pyspy.speedscope.json      # py-spy 采样（speedscope 格式）
│   └── scalene.json               # Scalene 私有 JSON
├── optimized/                     # 同上
└── report.md                      # 自动生成的对比报告
```

---

## ⚠️ 五个必知的坑

### 1. py-spy 在 macOS 上恒需 root → 必须进容器

官方文档：「OSX always requires running as root」。
它靠 `task_for_pid` 读目标进程内存，macOS 的 SIP 不允许普通用户 attach 别的进程。

**重签名（`codesign` 加 `get-task-allow`）绕不过去** —— 实测过，
给 py-spy 自己和目标解释器都签了，仍然报 requires root。

所以 py-spy 一律在 Docker 里跑：

```bash
docker run --rm \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$PWD:/work" -w /work -e PYTHONPATH=/work/src \
  agent-perf-tools \
  py-spy record --rate 200 --format speedscope \
  -o /work/reports/naive/pyspy.speedscope.json \
  -- python3 -m agent_perf_profiler.target naive --reps 1
```

### 2. Scalene 2.x 的 CLI 是 `scalene run`

旧教程写 `scalene prog.py`，2.x 之后必须是 `scalene run prog.py`。
照旧教程抄命令，scalene 会打帮助文本然后退出 —— 看起来像「装错了」。

### 3. speedscope 的帧表在顶层 `shared.frames`

不是 `profiles[0]["frames"]`（那是另一种变体）。
照常见示例读会直接 `KeyError: 'frames'`。

### 4. 只统计叶子帧

speedscope 的每个采样是**一整条栈**。把所有帧展开计数，
统计的是「谁在调用链上」而不是「谁在执行」——
一个被调用 33 万次的函数会因此霸榜。只取 `stack[-1]`。

### 5. 采样器需要运行时间

程序秒退时 py-spy 只能采到几帧（实测 `Samples: 2`），
报告「看起来像工具坏了」，其实只是样本不够。
`target.py` 末尾留了 0.5s 采样窗口。

---

## 实测数据

### 三工具并排对比

| # | cProfile（自耗时 秒） | py-spy（采样 %） | Scalene（CPU %，Python 部分） |
|---|---|---|---|
| 1 | `naive.py:<genexpr>` 0.87 | `naive_bm25_score.<locals>.<genexpr>` 84.36 | `naive.py:naive_bm25_score` 34.13 |
| 2 | `~:<built-in method time.sleep>` 0.58 | `main` 5.35 | — |
| 3 | `naive.py:naive_bm25_score` 0.04 | `naive_bm25_score` 4.53 | — |

**cProfile 第 2 名的 `time.sleep` 不是 CPU 瓶颈，是 IO 等待。**
py-spy 完全看不到它（默认排除空闲线程）—— 两者都没错，量的不是同一件事。

### 优化前后

| | 耗时 |
|---|---|
| naive | 1.0113s |
| optimized | 0.1207s |
| **加速比** | **8.38x** |

---

## 两版的差异在哪

| 优化点 | naive | optimized | 代价 |
|---|---|---|---|
| BM25 索引 | 每次检索现算 idf + 全量线性扫描 | **建一次倒排索引** + 预计算 idf | 索引常驻内存 |
| 分词 | 每次重算 | **缓存**（LRU） | 缓存占内存 |
| 字符串拼接 | `history = history + ...`（O(n²)） | `''.join(parts)` | 无 |
| top-k | 全排序 | `heapq.nlargest` | 无 |
| JSON | `sort_keys=True`（每次排序键） | 去掉 | 无 |
| 工具参数 | 字符串切分 | 结构化 dict | 无 |

⚠️ **两版的 IO 等待（`time.sleep`）完全相同** ——
这是对比实验成立的前提。如果 optimized 的 sleep 更短，
差值里就混进了「IO 等待变少」，而那不是优化的功劳。

---

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
```

21 项测试，重点是**行为等价性**：

| 测试 | 锁住什么 |
|---|---|
| `test_two_versions_produce_identical_answers` | ⭐ 两版答案**逐字相同** |
| `test_two_versions_do_same_amount_of_work` | 调用次数一致 |
| `test_retrieval_actually_finds_relevant_docs` | 检索必须命中（否则瓶颈没被触发） |
| `test_tokenizer_handles_chinese_with_bigram` | 中文 bigram 生效 |
| `test_speedscope_parser_reads_shared_frames` | 帧表位置 |
| `test_cprofile_summarizer_sorts_by_tottime` | 排 tottime 而非 cumtime |
