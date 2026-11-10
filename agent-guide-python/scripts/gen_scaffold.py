"""生成 25 个 Python 子包的骨架（一次性脚本）。

⚠️ 这个脚本只负责**造目录结构和占位文件**，
   真正的实现由后续逐个包补齐（对应 Rust 侧 25 个 crate）。

用法：uv run scripts/gen_scaffold.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (目录名, 包名, 项目名, 中文描述)
PACKAGES: list[tuple[str, str, str, str]] = [
    ("00-logging", "agent_logging", "agent-logging-py", "结构化日志设施（工作区共享）"),
    ("01-axum", "axum_01", "axum-01-py", "HTTP 服务基础（FastAPI 路由 / 异步 I/O / Pydantic）"),
    ("02-rig", "rig_02", "rig-02-py", "Agent 框架抽象（角色 / 中间件 / 模型调用 / 工具）"),
    ("03-genai", "genai_03", "genai-03-py", "模型调用与工具协议（OpenAI 兼容协议）"),
    ("04-rag", "rag_04", "rag-04-py", "文档加载与文本分块"),
    ("05-rag", "rag_05", "rag-05-py", "向量化与向量存储"),
    ("06-rag", "rag_06", "rag-06-py", "端到端 RAG 服务"),
    ("07-query", "rag_07", "rag-07-py", "查询改写与融合检索"),
    ("08-rag", "rag_08", "rag-08-py", "混合检索（BM25 + 向量）与重排"),
    ("09-rag-eval", "rag_09", "rag-09-py", "RAG 评估体系"),
    ("10-rag-milvus", "rag_10", "rag-10-py", "生产级向量数据库（Milvus / Qdrant）"),
    ("11-rag-advance", "rag_11", "rag-11-py", "复杂文档解析（表格 / 图片 / 结构感知分块）"),
    ("12-rag-summary", "rag_12", "rag-12-py", "摘要式 RAG 服务（五档消融阶梯）"),
    ("13-agent-core", "agent_13", "agent-13-py", "Agent 核心（ReAct 循环 / 三模式消融）"),
    ("14-agent-tool", "agent_14", "agent-14-py", "自定义工具开发与 MCP"),
    ("15-agent-sql", "agent_15", "agent-15-py", "SQL Agent（自然语言转 SQL + 三层护栏）"),
    ("16-agent-function-calling", "agent_16", "agent-16-py", "结构化 Function Calling"),
    ("17-agent-memory", "agent_17", "agent-17-py", "Agent Memory（短期状态 + 长期事实）"),
    ("18-agent-error", "agent_18", "agent-18-py", "Agent 错误处理（重试 / 降级 / 终止）"),
    ("19-agent-summary", "agent_19", "agent-19-py", "研究助手 Agent（研究 → 成文 → 接地校验）"),
    (
        "20-agent-preformence-analysis",
        "agent_20",
        "agent-20-py",
        "性能瓶颈分析（cProfile / py-spy / Scalene）",
    ),
    ("21-agent-cache", "agent_21", "agent-21-py", "缓存优化（Redis 内容寻址缓存）"),
    ("22-agent-async", "agent_22", "agent-22-py", "异步处理（并发执行 / 限流 / 线程池轨道）"),
    (
        "23-agent-batch",
        "agent_23",
        "agent-23-py",
        "批处理优化（批量 embedding / padding / 批重排）",
    ),
    ("24-agent-vllm", "agent_24", "agent-24-py", "推理引擎调度（vLLM / goodput / 调度器）"),
]

# 每个包依赖哪些前序包（对应 Rust 侧的 crate 依赖）
# ⚠️ 键是**目录名**，值是**前序包的项目名**（不是 Python 模块名）。
#    Python 的项目名可以带连字符（PEP 508 允许），模块名用下划线。
DEPS: dict[str, list[str]] = {
    "00-logging": [],
    "01-axum": [],
    "02-rig": ["agent-logging-py"],
    "03-genai": ["agent-logging-py"],
    "04-rag": ["agent-logging-py"],
    "05-rag": ["agent-logging-py", "rag-04-py"],
    "06-rag": ["agent-logging-py", "rag-04-py", "rag-05-py", "genai-03-py"],
    "07-query": ["agent-logging-py", "rag-06-py", "genai-03-py"],
    "08-rag": ["agent-logging-py", "rag-06-py", "genai-03-py"],
    "09-rag-eval": ["agent-logging-py", "rag-08-py", "genai-03-py"],
    "10-rag-milvus": ["agent-logging-py", "rag-04-py", "genai-03-py"],
    "11-rag-advance": ["agent-logging-py", "rag-04-py"],
    "12-rag-summary": ["agent-logging-py", "rag-04-py", "rag-05-py", "rag-10-py", "genai-03-py"],
    "13-agent-core": ["agent-logging-py", "rag-08-py", "genai-03-py"],
    "14-agent-tool": ["agent-logging-py", "agent-13-py", "genai-03-py"],
    "15-agent-sql": ["agent-logging-py", "agent-13-py", "genai-03-py"],
    "16-agent-function-calling": ["agent-logging-py", "agent-14-py", "genai-03-py"],
    "17-agent-memory": ["agent-logging-py", "agent-13-py", "rag-08-py"],
    "18-agent-error": ["agent-logging-py", "agent-16-py", "agent-14-py"],
    "19-agent-summary": ["agent-logging-py", "agent-18-py", "agent-17-py", "agent-13-py"],
    "20-agent-preformence-analysis": ["agent-logging-py", "rag-08-py"],
    "21-agent-cache": ["agent-logging-py", "agent-16-py", "agent-13-py"],
    "22-agent-async": ["agent-logging-py", "agent-18-py", "agent-16-py"],
    "23-agent-batch": ["agent-logging-py", "rag-08-py", "agent-16-py"],
    "24-agent-vllm": ["agent-logging-py", "agent-23-py"],
}


def package_dir(pkg: str) -> str:
    return f"{pkg.replace('-', '_')}"


def main() -> None:
    members: list[str] = []
    for dirname, pkg, project, desc in PACKAGES:
        members.append(dirname)
        d = ROOT / dirname
        (d / "src" / pkg).mkdir(parents=True, exist_ok=True)
        (d / "tests").mkdir(parents=True, exist_ok=True)

        dep_lines = "\n".join(f'    "{dep}>=0.1.0",' for dep in DEPS[dirname])
        pyproject = f'''[project]
name = "{project}"
version = "0.1.0"
description = "{desc}"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
{dep_lines}
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.25",
    "ruff>=0.9",
]

[build-system]
requires = ["uv_build>=0.12.13,<0.13.0"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "{pkg}"
module-root = "src"

[tool.pytest.ini_options]
asyncio_mode = "auto"
filterwarnings = ["ignore::DeprecationWarning"]
'''
        (d / "pyproject.toml").write_text(pyproject, encoding="utf-8")

        init = f'''"""{desc}。

对应 Rust 侧 ``{dirname}``。
"""

from __future__ import annotations

__version__ = "0.1.0"
'''
        (d / "src" / pkg / "__init__.py").write_text(init, encoding="utf-8")

        readme = f"""# {dirname}（{desc}）

对应 Rust 侧 crate `{dirname}`。

> 本文件在实现过程中持续补充。Rust 侧的 README 是**对齐基准**：
> 模块划分、公开 API、坑位说明、实测数据都要与 Rust 侧一一对应。
"""
        (d / "README.md").write_text(readme, encoding="utf-8")

    workspace_members = ",\n".join(f'    "{m}"' for m in members)
    (ROOT / ".packages.json").write_text(
        json.dumps([[d, p, pr] for d, p, pr, _ in PACKAGES], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已生成 {len(members)} 个子包骨架")
    print(f"workspace 成员列表已准备：\n{workspace_members}")


if __name__ == "__main__":
    main()
