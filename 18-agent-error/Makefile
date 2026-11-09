# ============================================================================
# 18-agent-error —— 手撕 Agent 错误处理（middleware 驱动的可诊断工具失败）
#
# 核心卖点：**按错误类别决策** ——
#           Transient 重试、Business 直接降级、Fatal 直接终止。
#
# 快速上手：
#   make help          查看所有目标
#   make table         ★ 打印错误分类速查表（不调模型）
#   make compare       ★ 三类错误对照（不调模型）
#   make call          ★ 直接调工具（看错误码）
#   make run           跑一次 Agent
#   make serve         启动 HTTP 服务（:3110）
# ============================================================================

BIN := agent-18
PORT := 3110
ADDR := 127.0.0.1:$(PORT)

# 默认问题（`make run QUESTION="..."` 可覆盖）。
#
# ⚠️ 必须是「**会触发工具失败**」的问题 ——
#    这样才能观察到「重试 → 降级」的完整过程。
QUESTION ?= 帮我拉取一下 report/2026 的数据

.PHONY: help build run trace compare call tools table serve all \
        test lint fmt clean check api-root api-run api-call api-table

# ── 常用命令 ──────────────────────────────────────────────────────────
help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## ★ 跑一次 Agent（QUESTION="你的问题" 可覆盖）
	cargo run -p $(BIN) -- run "$(QUESTION)"

compare: ## ★ 三类错误对照（Transient / Business / Fatal）
	cargo run -p $(BIN) -- compare

call: ## ★ 直接调工具 TOOL="strict_lookup" ARGS='{"id":"nope"}'
	cargo run -p $(BIN) -- call "$(or $(TOOL),strict_lookup)" '$(or $(ARGS),{"id":"nope"})'

tools: ## 打印工具清单 + Schema（不调模型）
	cargo run -p $(BIN) -- tools

table: ## ★ 打印错误分类速查表（不调模型）
	cargo run -p $(BIN) -- table

serve: ## 启动 HTTP 服务（127.0.0.1:3110）
	ERR_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

all: ## 全部跑一遍（tools → table → compare）
	cargo run -p $(BIN) -- all

# ── HTTP 端点 ─────────────────────────────────────────────────────────
#
# 需先 `make serve`。
api-root: ## 用 HTTP 端点看服务信息 + 工具清单
	curl -s http://$(ADDR)/ | python3 -m json.tool --no-ensure-ascii

api-run: ## 用 HTTP 端点跑一次 Agent
	curl -s -X POST http://$(ADDR)/api/run \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool --no-ensure-ascii

api-call: ## ★ 用 HTTP 端点直接调工具（看错误码 / 状态码）
	curl -s -X POST http://$(ADDR)/api/tools/call \
		-H 'Content-Type: application/json' \
		-d '{"tool": "strict_lookup", "args": {"id": "nope"}}' | python3 -m json.tool --no-ensure-ascii

api-table: ## ★ 用 HTTP 端点查错误分类速查表
	curl -s http://$(ADDR)/api/errors/table | python3 -m json.tool --no-ensure-ascii

# ── 质量 ──────────────────────────────────────────────────────────────
test: ## 跑全部测试
	cargo test -p $(BIN)

lint: ## clippy 零警告检查
	cargo clippy -p $(BIN) --all-targets -- -D warnings

fmt: ## 格式化
	cargo fmt -p $(BIN)

check: fmt lint test ## 格式化 + lint + 测试

clean: ## 清理构建产物
	cargo clean -p $(BIN)
