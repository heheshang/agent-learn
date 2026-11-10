# agent-guide-python

AI Agent 开发工程师学习路线图（工程落地版）的 **Python 实现**。

本项目是 Rust 侧 [`agent-learn`](../) 的**一对一对照实现**：
00~24 共 25 个子包在目录划分、公开 API、依赖图上与 Rust 侧的 crate **严格对应**。

## 为什么要有这个项目

同一套设计换语言重写，最能暴露两件事：

- **语言特性**各自贡献了什么（Rust 的所有权/生命周期 vs Python 的动态特性与 GIL）
- **框架约定**各自隐藏了什么（axum 的 extractor vs FastAPI 的 Depends）

目录划分与依赖图必须一一对应，**否则差异无法归因** —— 差异到底是「语言不同」
还是「设计不同」，就说不清了。

## 当前状态：骨架阶段

25 个子包的目录结构、`pyproject.toml`、占位 `README.md` 与 `__init__.py` 已就位，
**实现尚未开始**。骨架由 `scripts/gen_scaffold.py` 生成，可重复执行。

| 目录 | 包名 | 对应 Rust 侧 | 说明 |
|---|---|---|---|
| `00-logging` | `agent_logging` | `00-logging` | 结构化日志设施（工作区共享） |
| `01-axum` | `axum_01` | `01-axum` | HTTP 服务基础（FastAPI 路由 / 异步 I/O / Pydantic） |
| `02-rig` | `rig_02` | `02-rig` | Agent 框架抽象 |
| `03-genai` | `genai_03` | `03-genai` | 模型调用与工具协议（OpenAI 兼容） |
| `04-rag` ~ `12-rag-summary` | `rag_04` ~ `rag_12` | 同名 | RAG 全链路 |
| `13-agent-core` ~ `22-agent-async` | `agent_13` ~ `agent_22` | 同名 | Agent 核心概念 / 工具 / SQL / Function Calling / Memory / 错误处理 / 缓存 / 并发 |
| `23-agent-batch` | `agent_23` | `23-agent-batch` | 批处理优化（客户端侧） |
| `24-agent-vllm` | `agent_24` | `24-agent-vllm` | 推理引擎调度（服务端侧） |

完整清单见 [`.packages.json`](.packages.json)。

## 快速上手

```bash
cd agent-guide-python      # ⚠️ 必须在项目目录里跑，下面命令用的是相对路径

make setup      # ★ 幂等：建 venv + 装依赖（用 uv，绝不动本机 Python 环境）
make test       # 跑全部测试（零网络）
make lint       # ruff 零警告检查
make check      # 格式化 + lint + 测试
make help       # 查看所有目标
```

> ⚠️ `uv venv` 在 venv 已存在时会直接报错退出，所以 `make setup` 用的是
> `uv sync`（它自己会处理已存在的 venv），**第二次跑不会挂**。

## 环境约定

- **Python 3.12**（见 `.python-version`）
- **环境变量前缀 `AGENTPY_`**（Rust 侧各项目用 `RAG_` / `QUERY_` / `AGENT_` 等分前缀）
- **零第三方运行时依赖**起步 —— 与 Rust 侧一致，先手写再引入框架
- 依赖版本统一在根 `pyproject.toml` 管理，成员包用 `{ workspace = true }` 引用

## 目录

```text
agent-guide-python/
├── pyproject.toml          # workspace 根：成员列表 + 依赖版本唯一来源
├── uv.lock                 # 锁文件（入库）
├── Makefile                # 统一入口
├── scripts/
│   └── gen_scaffold.py     # 生成 25 个子包骨架
├── src/
│   └── agent_guide_python/ # 根包
├── 00-logging/             # ┐
├── 01-axum/                # │
├── ...                     # ├ 25 个成员包，结构与 Rust 侧一一对应
└── 24-agent-vllm/          # ┘
```

## 约定

- **代码注释与文档一律中文**
- 每个子包的 `README.md` 以 Rust 侧同名项目的 README 为**对齐基准**：
  模块划分、公开 API、坑位说明、实测数据都要一一对应
- 坑位要写清楚**「为什么踩」和「怎么修」**，不能只写结论
