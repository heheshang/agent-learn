# ============================================================================
# 22-agent-async —— 手撕异步处理（并发工具执行 + 限流 + 线程池轨道）
#
# 核心卖点：**Agent 一轮里的多个工具调用是并发执行的**，
#           并且能**量化**「并发到底快多少」。
#
# 快速上手：
#   make help          查看所有目标
#   make tools         ★ 打印工具清单 + 三条执行轨道
#   make modes         ★ ★ 四种并发模式说明
#   make bench         ★ ★ ★ 并发对照实验（模拟负载，稳）
#   make bench-real    ★ ★ ★ 并发对照实验（真实天气接口）
#   make run           跑一次 Agent（并发工具执行）
#   make serve         启动 HTTP 服务（:3140）
# ============================================================================

BIN := agent-22
PORT := 3140
ADDR := 127.0.0.1:$(PORT)

# 对照实验的并发上限。
#
# ⚠️ 必须 ≥ 1 —— `buffer_unordered(0)` 会让请求**永远挂住**。
CONCURRENCY ?= 4

# 对照实验的城市列表（逗号分隔）。
CITIES ?= 北京,上海,广州,深圳

.PHONY: help build run bench bench-real call tools modes serve all \
        test lint fmt clean check \
        api-root api-modes api-run api-bench api-call

# ── 常用命令 ──────────────────────────────────────────────────────────
help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## ★ 跑一次 Agent（并发工具执行）
	cargo run -p $(BIN) -- run "北京和上海现在天气怎么样？请分别说明。"

bench: ## ★ ★ ★ 并发对照实验（模拟负载，稳）
	ASYNC_MAX_CONCURRENCY=$(CONCURRENCY) cargo run -p $(BIN) -- bench

bench-real: ## ★ ★ ★ 并发对照实验（真实天气接口）
	ASYNC_MAX_CONCURRENCY=$(CONCURRENCY) cargo run -p $(BIN) -- bench-real

call: ## ★ 直接调工具 TOOL="get_weather" ARGS='{"location":"北京"}'
	cargo run -p $(BIN) -- call "$(or $(TOOL),get_weather)" '$(or $(ARGS),{"location":"北京"})'

tools: ## ★ 打印工具清单 + 三条执行轨道
	cargo run -p $(BIN) -- tools

modes: ## ★ ★ 四种并发模式说明
	cargo run -p $(BIN) -- modes

serve: ## 启动 HTTP 服务（127.0.0.1:3140）
	ASYNC_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

all: ## 全部跑一遍（tools → modes → bench）
	cargo run -p $(BIN) -- all

# ── HTTP 端点 ─────────────────────────────────────────────────────────
#
# 需先 `make serve`。
api-root: ## 服务信息 + 工具清单
	curl -s http://$(ADDR)/ | python3 -m json.tool --no-ensure-ascii

api-modes: ## 四种并发模式说明
	curl -s http://$(ADDR)/api/modes | python3 -m json.tool --no-ensure-ascii

api-run: ## ★ 跑一次 Agent
	curl -s -X POST http://$(ADDR)/api/run \
		-H 'Content-Type: application/json' \
		-d '{"question":"北京和上海现在天气怎么样？请分别说明。"}' \
		| python3 -m json.tool --no-ensure-ascii

api-bench: ## ★ ★ ★ 并发对照实验
	curl -s -X POST http://$(ADDR)/api/bench \
		-H 'Content-Type: application/json' \
		-d "{\"cities\":[\"$(subst ,, \",\",\"$(CITIES))\"],\"max_concurrency\":$(CONCURRENCY),\"simulated\":true}" \
		| python3 -m json.tool --no-ensure-ascii

api-call: ## 直接调工具
	curl -s -X POST http://$(ADDR)/api/tools/call \
		-H 'Content-Type: application/json' \
		-d '{"tool":"get_weather","args":{"location":"北京"}}' \
		| python3 -m json.tool --no-ensure-ascii

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
