# ============================================================================
# 19-agent-summary —— 研究助手 Agent（RAG + Web 搜索 + 周报 / 项目构建）
#
# 核心卖点：**研究 → 成文 → 接地校验** 三阶段流水线。
#
# 快速上手：
#   make help          查看所有目标
#   make tools         ★ 工具清单（不调模型、不联网）
#   make compare       ★ 接地校验对照（不调模型、不联网）
#   make search        ★ 直接调 Web 搜索（真的联网）
#   make fetch         ★ 直接调网页抓取（真的联网）
#   make research      跑一次研究阶段（看证据链）
#   make weekly        ★ 跑一次周度总结
#   make plan          ★ 跑一次项目构建方案
#   make serve         启动 HTTP 服务（:3120）
# ============================================================================

BIN := agent-19
PORT := 3120
ADDR := 127.0.0.1:$(PORT)

# 本目录的绝对路径 —— 让数据路径与 cwd 无关。
#
# ⚠️ 必须用绝对路径：`cargo run -p agent-19` 在 workspace 根目录跑时
#    cwd 是根目录，`data/kb` 会找不到。
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
export SUM_KB_DIR := $(ROOT)/data/kb
export SUM_OUTPUT_DIR := $(ROOT)/data/reports

# 默认参数（`make weekly WEEKLY="..."` 可覆盖）。
#
# ⚠️ 默认问题**故意**和知识库里的示例问题错开 ——
#    否则模型会直接抄答案，演示就失真了。
QUESTION ?= 本项目的错误处理是怎么设计的？降级策略是什么？
WEEKLY ?= 错误处理与工具层的进展
REQUIREMENT ?= 用 Rust 构建一个能查资料、能出周报的研究助手
SEARCH_QUERY ?= rust async runtime tokio
FETCH_URL ?= https://doc.rust-lang.org/book/

.PHONY: help build run tools compare search fetch research weekly plan serve all \
        test lint fmt clean check port-check \
        api-root api-research api-weekly api-plan api-call api-evidence

# ── 常用命令 ──────────────────────────────────────────────────────────
help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## ★ 跑一次研究阶段（只拿证据，不成文）
	cargo run -p $(BIN) -- research "$(QUESTION)"

tools: ## ★ 打印工具清单（不调模型、不联网）
	cargo run -p $(BIN) -- tools

compare: ## ★ 接地校验对照（不调模型、不联网）
	cargo run -p $(BIN) -- compare

search: ## ★ 直接调 Web 搜索工具（真的联网）SEARCH_QUERY="..."
	cargo run -p $(BIN) -- search "$(SEARCH_QUERY)"

fetch: ## ★ 直接调网页抓取工具（真的联网）FETCH_URL="..."
	cargo run -p $(BIN) -- fetch "$(FETCH_URL)"

weekly: ## ★ 跑一次周度总结（研究 + 成文 + 校验）
	cargo run -p $(BIN) -- weekly "$(WEEKLY)"

plan: ## ★ 跑一次项目构建方案（研究 + 成文 + 校验）
	cargo run -p $(BIN) -- plan "$(REQUIREMENT)"

serve: ## 启动 HTTP 服务（127.0.0.1:3120）
	SUM_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

# ⚠️ 「Address already in use」是每个 HTTP 项目最常见的启动失败。
#
#    根因通常是**上一次的服务还在跑**（Ctrl+C 有时没杀干净，
#    或者用 `&` 后台启动后忘了停）。
#
#    这个目标把「诊断 + 清理」变成一条命令。
port-check: ## ★ 诊断端口占用（Address already in use 时跑这个）
	@echo "检查端口 $(PORT) …"
	@PIDS=$$(lsof -nP -iTCP:$(PORT) -sTCP:LISTEN -t 2>/dev/null); \
	if [ -z "$$PIDS" ]; then \
		echo "✓ 端口 $(PORT) 空闲，可以 make serve"; \
	else \
		echo "⚠️ 端口 $(PORT) 被以下进程占用："; \
		for p in $$PIDS; do ps -o pid,lstart,command -p $$p | tail -1; done; \
		echo ""; \
		echo "确认是自己的残留进程后，停掉它："; \
		echo "    kill $$PIDS"; \
		echo "或一键清理："; \
		echo "    make port-clean"; \
	fi

port-clean: ## ★ 停掉占用端口 $(PORT) 的残留进程
	@PIDS=$$(lsof -nP -iTCP:$(PORT) -sTCP:LISTEN -t 2>/dev/null); \
	if [ -z "$$PIDS" ]; then \
		echo "✓ 端口 $(PORT) 本来就是空闲的"; \
	else \
		kill $$PIDS 2>/dev/null; sleep 1; \
		if lsof -nP -iTCP:$(PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
			kill -9 $$PIDS 2>/dev/null; sleep 1; \
		fi; \
		if lsof -nP -iTCP:$(PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
			echo "❌ 仍然被占用，请手动检查"; exit 1; \
		else \
			echo "✓ 端口 $(PORT) 已释放"; \
		fi; \
	fi

all: ## 全部跑一遍（tools → compare → research）
	cargo run -p $(BIN) -- all

# ── HTTP 端点 ─────────────────────────────────────────────────────────
#
# 需先 `make serve`。
api-root: ## 用 HTTP 端点看服务信息 + 工具清单
	curl -s http://$(ADDR)/ | python3 -m json.tool --no-ensure-ascii

api-evidence: ## 用 HTTP 端点看证据来源契约
	curl -s http://$(ADDR)/api/evidence | python3 -m json.tool --no-ensure-ascii

api-research: ## 用 HTTP 端点跑一次研究阶段
	curl -s -X POST http://$(ADDR)/api/research \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool --no-ensure-ascii

api-weekly: ## ★ 用 HTTP 端点生成周报
	curl -s -X POST http://$(ADDR)/api/weekly \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(WEEKLY)\"}" | python3 -m json.tool --no-ensure-ascii

api-plan: ## ★ 用 HTTP 端点生成项目方案
	curl -s -X POST http://$(ADDR)/api/plan \
		-H 'Content-Type: application/json' \
		-d "{\"requirement\": \"$(REQUIREMENT)\"}" | python3 -m json.tool --no-ensure-ascii

api-call: ## ★ 用 HTTP 端点直接调工具（不经模型）
	curl -s -X POST http://$(ADDR)/api/tools/call \
		-H 'Content-Type: application/json' \
		-d '{"tool": "kb_search", "args": {"query": "错误处理 降级"}}' \
		| python3 scripts/strip_data.py

# ── 质量 ──────────────────────────────────────────────────────────────
test: ## 跑全部测试（零网络、零 token）
	cargo test -p $(BIN)

lint: ## clippy 零警告检查
	cargo clippy -p $(BIN) --all-targets -- -D warnings

fmt: ## 格式化
	cargo fmt -p $(BIN)

check: fmt lint test ## 格式化 + lint + 测试

clean: ## 清理构建产物
	cargo clean -p $(BIN)
