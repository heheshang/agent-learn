# ============================================================================
# 16-agent-function-calling —— 手撕 Function Calling（结构化工具调用）
#
# 核心卖点：**走 OpenAI 的结构化协议**（tools 字段 + tool_calls），
#           而不是 13~15 的 ReAct 文本协议。
#
# 快速上手：
#   make help          查看所有目标
#   make tools         ★ 打印工具清单 + Schema（不调模型）
#   make call          ★ 直接调工具（不经模型）
#   make run           跑一次 Agent
#   make compare       ★ tool_choice 四档对照
#   make serve         启动 HTTP 服务（:3090）
# ============================================================================

BIN := agent-16
PORT := 3090
ADDR := 127.0.0.1:$(PORT)

# 本目录的绝对路径 —— 让知识库路径与 cwd 无关。
#
# ⚠️ 必须用绝对路径：`cargo run -p agent-16` 在 workspace 根目录跑时
#    cwd 是根目录，`data/` 会找不到。
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
export FC_DATA_DIR := $(ROOT)/data

# 默认问题（`make run QUESTION="..."` 可覆盖）。
#
# ⚠️ 必须是「模型不可能凭空知道」的问题 ——
#    这样「用了工具」和「没用工具」的差值才能归因到工具，
#    而不是「模型本来就知道」。
QUESTION ?= 北京和上海现在天气怎么样？

.PHONY: help build run trace compare call tools serve all \
        test lint fmt clean check api-root api-tools api-call api-run api-compare

# ── 常用命令 ──────────────────────────────────────────────────────────
help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## ★ 跑一次 Agent（QUESTION="你的问题" 可覆盖）
	cargo run -p $(BIN) -- run "$(QUESTION)"

trace: ## ★ 打印完整结构化轨迹（含思考内容）
	cargo run -p $(BIN) -- trace "$(QUESTION)"

compare: ## ★ tool_choice 四档对照（auto / none / required / forced）
	cargo run -p $(BIN) -- compare "$(QUESTION)"

call: ## ★ 直接调工具（不经模型）TOOL="get_weather" ARGS='{"city":"北京"}'
	cargo run -p $(BIN) -- call "$(or $(TOOL),get_weather)" '$(or $(ARGS),{"location":"北京"})'

tools: ## ★ 打印工具清单 + Schema（不调模型）
	cargo run -p $(BIN) -- tools

serve: ## 启动 HTTP 服务（127.0.0.1:3090）
	FC_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

all: ## 全部跑一遍（tools → call → run）
	cargo run -p $(BIN) -- all

# ── HTTP 端点 ─────────────────────────────────────────────────────────
#
# 需先 `make serve`。
api-root: ## 用 HTTP 端点看服务信息 + 工具清单
	curl -s http://$(ADDR)/ | python3 -m json.tool --no-ensure-ascii

api-tools: ## 用 HTTP 端点查工具清单
	curl -s http://$(ADDR)/api/tools | python3 -m json.tool --no-ensure-ascii

api-call: ## ★ 用 HTTP 端点直接调工具（不经模型）
	curl -s -X POST http://$(ADDR)/api/tools/call \
		-H 'Content-Type: application/json' \
		-d '{"tool": "get_weather", "args": {"location": "北京"}}' \
		| python3 -m json.tool --no-ensure-ascii

api-run: ## 用 HTTP 端点跑一次 Agent
	curl -s -X POST http://$(ADDR)/api/run \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool --no-ensure-ascii

api-compare: ## ★ 用 HTTP 端点跑 tool_choice 四档对照
	curl -s -X POST http://$(ADDR)/api/compare \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool --no-ensure-ascii

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
