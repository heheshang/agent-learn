# ============================================================================
# 20-agent-preformence-analysis —— 性能瓶颈分析
#
# 核心卖点：**三个剖析工具对同一份程序给出三张不同的名单**，
#           以及「优化前后」的可复现差值。
#
# 快速上手：
#   make help         查看所有目标
#   make setup        ★ 首次准备（uv venv + uv sync + 构建镜像）
#   make compare      ★ 三工具对比 + 优化前后差值（Python 侧）
#   make profile      ★ 只剖析 naive 版（产出报告）
#   make bench        Rust 侧：只测时
#   make pprof        Rust 侧：采样出火焰图
#   make test         两侧测试全跑
#
# ⚠️ 所有 Python 操作都在**项目目录内的 venv** 里执行，
#    不碰本机 Python 环境（见 profiler/README.md 的说明）。
# ============================================================================

BIN := agent-20
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROFILER := $(ROOT)/profiler
PY := $(PROFILER)/.venv/bin/python
UV := uv
IMAGE := agent-perf-tools

.PHONY: help build run bench pprof compare tools test test-rust test-python lint fmt clean \
        setup venv sync image docker-build profile profile-optimized report \
        py-venv py-sync py-test py-lint py-format

# ── 常用命令 ─────────────────────────────────────────────────────────
help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## ★ 跑一次（bench + 工具对应表）
	cargo run -p $(BIN) -- bench

bench: ## ★ Rust 侧：只测时（多次取中位数）
	cargo run -p $(BIN) -- bench

pprof: ## ★ Rust 侧：采样剖析，产出火焰图
	cargo run -p $(BIN) -- pprof

compare: ## ★ Rust 侧：计时 + 采样 + 出报告
	cargo run -p $(BIN) -- compare

tools: ## 打印 Python 三工具 ↔ Rust 对应物
	cargo run -p $(BIN) -- tools

# ── Python 侧 ─────────────────────────────────────────────────────────
#
# ⚠️ 全部走项目内的 venv（$(PROFILER)/.venv），
#    **不 install 到本机 Python**。

setup: venv sync docker-build ## ★ 首次准备：venv + 依赖 + 镜像
	@echo "✓ 准备完成。试试：make py-compare"

venv: ## 创建项目内虚拟环境（uv venv，幂等：已存在则跳过）
	@if [ -d "$(PROFILER)/.venv" ]; then \
		echo "✓ venv 已存在，跳过创建（要重建：rm -rf $(PROFILER)/.venv && make venv）"; \
	else \
		cd $(PROFILER) && $(UV) venv --python 3.12 .venv; \
	fi

sync: ## 安装依赖（uv sync）
	cd $(PROFILER) && $(UV) sync

py-venv: venv ## 创建项目内虚拟环境
py-sync: sync ## 安装依赖
py-test: ## 跑 Python 侧测试
	cd $(PROFILER) && .venv/bin/python -m pytest tests/ -q

py-lint: ## Python 侧基础检查（编译所有模块）
	cd $(PROFILER) && .venv/bin/python -m compileall -q src/ tests/

py-format: ## Python 侧格式化检查（用 ruff，需自行安装）
	cd $(PROFILER) && .venv/bin/python -m ruff check src/ tests/ 2>/dev/null || \
		echo "（未安装 ruff，跳过）"

profile: ## ★ Python 侧：剖析 naive 版（三工具全跑）
	cd $(PROFILER) && .venv/bin/python -m agent_perf_profiler.cli profile naive

profile-optimized: ## Python 侧：剖析 optimized 版
	cd $(PROFILER) && .venv/bin/python -m agent_perf_profiler.cli profile optimized

py-compare: ## ★ Python 侧：三工具对比 + 优化前后差值（出 Markdown 报告）
	cd $(PROFILER) && .venv/bin/python -m agent_perf_profiler.cli compare

report: ## 打印已生成的 Python 侧报告
	cd $(PROFILER) && .venv/bin/python -m agent_perf_profiler.cli report

# ── Docker（py-spy 必需）──────────────────────────────────────────────
#
# ⚠️ py-spy 在 macOS 上恒需 root，**只能进容器跑**（详见 docker/Dockerfile）。
docker-build: ## 构建 py-spy 镜像
	docker build -t $(IMAGE) -f $(PROFILER)/docker/Dockerfile $(PROFILER)

image: docker-build ## 构建 py-spy 镜像

# ── 质量 ──────────────────────────────────────────────────────────────
test: test-rust test-python ## 跑两侧测试

test-rust: ## 跑 Rust 侧测试
	cargo test -p $(BIN)

test-python: py-test ## 跑 Python 侧测试

lint: ## clippy 零警告检查
	cargo clippy -p $(BIN) --all-targets -- -D warnings

fmt: ## 格式化
	cargo fmt -p $(BIN)

check: fmt lint test ## 格式化 + lint + 测试

clean: ## 清理构建产物
	cargo clean -p $(BIN)
	rm -rf $(PROFILER)/reports $(PROFILER)/.pytest_cache $(ROOT)/perf-out
	find $(PROFILER)/src $(PROFILER)/tests -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
