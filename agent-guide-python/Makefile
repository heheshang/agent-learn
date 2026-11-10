# ============================================================================
# agent-guide-python —— AI Agent 开发工程师学习路线图（工程落地版）Python 实现
#
# 快速上手：
#   make help       查看所有目标
#   make setup      ★ 幂等：建 venv + 装依赖
#   make test       跑全部测试（零网络）
#   make lint       ruff 零警告检查
#
# ⚠️ 当前处于**骨架阶段**：25 个子包（00~24）的目录结构、pyproject.toml、
#    占位 README 与 __init__.py 已就位，但**实现尚未开始**。
#    因此 demo / bench / serve 这些「跑起来才有意义」的目标暂未开放，
#    等对应子包实现落地后再逐个补上（与 Rust 侧同样的推进节奏）。
# ============================================================================

UV ?= uv
PY := $(UV) run

.PHONY: help setup setup-dev build test lint fmt fmt-check check clean \
        scaffold docker-build

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── 环境准备 ──────────────────────────────────────────────────────────
#
# ⚠️ **必须幂等**：`uv sync` 在 venv 已存在时会自己处理，不会报错。
#     （早期版本用 `uv venv` 会因为目录已存在而直接退出。）
setup: ## ★ 建 venv + 装全部依赖（含 dev 组）
	$(UV) sync --all-groups

setup-dev: ## 只装开发依赖
	$(UV) sync --all-groups --group dev

build: ## 构建 wheel（验证包结构正确）
	$(UV) build

# ── 骨架 ──────────────────────────────────────────────────────────────

scaffold: ## 重新生成 25 个子包的骨架（目录 / pyproject / README / __init__）
	$(PY) scripts/gen_scaffold.py

# ── 质量 ──────────────────────────────────────────────────────────────

test: ## 跑全部测试（零网络、零 API Key）
	$(PY) pytest -q

lint: ## ruff 零警告检查
	$(PY) ruff check .

fmt: ## 格式化
	$(PY) ruff format .

fmt-check: ## 检查格式
	$(PY) ruff format --check .

check: fmt lint test ## 格式化 + lint + 测试

clean: ## 清理构建产物与缓存
	rm -rf dist build .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# ── Docker ────────────────────────────────────────────────────────────

docker-build: ## 构建镜像
	docker build -t agent-guide-python:latest .
