# ============================================================================
# 21-agent-cache —— 手撕 Redis 缓存（缓存 LLM 响应）
#
# 核心卖点：**缓存层对 Agent 循环透明** —— 命中就不调模型，
#           而且缓存挂了**不能**拖垮 Agent（降级到内存 / 当未命中）。
#
# 快速上手：
#   make help          查看所有目标
#   make tools         ★ 打印工具清单（看哪些工具可缓存）
#   make run2          ★ ★ 连跑两次同一个问题（演示缓存命中）
#   make compare       ★ 缓存开/关对照实验
#   make stats         缓存统计（命中率）
#   make keys          列出缓存键
#   make serve         启动 HTTP 服务（:3130）
# ============================================================================

BIN := agent-21
PORT := 3130
ADDR := 127.0.0.1:$(PORT)

# 本目录的绝对路径 —— 让知识库路径与 cwd 无关。
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
export CACHE_DATA_DIR := $(ROOT)/data

# Redis 连接（用 docker 起一个本地实例）。
export CACHE_REDIS_URL ?= redis://127.0.0.1:6379

# 默认问题。
#
# ⚠️ 必须是「**不需要工具**」的纯问答 ——
#    带工具的链路里 `call_id` 每轮都变，缓存几乎必然 miss
#    （见 README「exact-match cache 对 agentic traffic 会失灵」）。
QUESTION ?= 什么是缓存击穿（cache stampede）？请简要解释。

# 演示「agentic 链路必然 miss」的问题（会触发知识库检索）。
#
# 单独拆一个目标，而不是把默认问题改坏 ——
# 默认问题要能演示「命中」，这个问题要能演示「失灵」，两者是**对照**。
AGENTIC_QUESTION ?= 什么是缓存击穿（cache stampede）？请简要解释,如何避免缓存击穿.

.PHONY: help build run run2 run2-agentic compare semantics call tools stats keys clear serve all \
        test lint fmt clean check \
        redis-up redis-down redis-logs \
        api-root api-tools api-run api-run2 api-stats api-keys api-clear

# ── 常用命令 ──────────────────────────────────────────────────────────
help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## 跑一次 Agent
	cargo run -p $(BIN) -- run "$(QUESTION)"

run2: ## ★ ★ 连跑两次同一个问题（第二次应当命中缓存）
	cargo run -p $(BIN) -- run2 "$(QUESTION)"

run2-agentic: ## ★ 对照演示：带工具的问题在**默认配置**下第二次必然 miss
	@echo "⚠️ 本目标演示的是「默认配置下，agentic 链路的精确匹配缓存失灵」。"
	@echo "   原因不是 call_id，而是「中间轮不写缓存」→ 模型每次措辞不同 → 污染后续轮次。"
	@echo "   ★ 想看修复后的 100% 命中，请跑：make semantics"
	cargo run -p $(BIN) -- run2 "$(AGENTIC_QUESTION)"

semantics: ## ★ ★ 三档缓存语义对照（agentic 链路能否命中的定量实验）
	cargo run -p $(BIN) -- semantics "$(AGENTIC_QUESTION)"

compare: ## ★ 缓存开 / 关对照实验
	cargo run -p $(BIN) -- compare "$(QUESTION)"

call: ## ★ 直接调工具（不经模型）TOOL="calculate" ARGS='{"expression":"(12+34)*2"}'
	cargo run -p $(BIN) -- call "$(or $(TOOL),calculate)" '$(or $(ARGS),{"expression":"(12+34)*2"})'

tools: ## ★ 打印工具清单（看哪些工具可缓存）
	cargo run -p $(BIN) -- tools

stats: ## 打印缓存统计（命中率）
	cargo run -p $(BIN) -- stats

keys: ## 列出缓存键 + TTL
	cargo run -p $(BIN) -- keys

clear: ## 清空缓存命名空间
	cargo run -p $(BIN) -- clear

serve: ## 启动 HTTP 服务（127.0.0.1:3130）
	CACHE_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

all: ## 全部跑一遍（tools → call → run2 → stats）
	cargo run -p $(BIN) -- all

# ── HTTP 端点 ─────────────────────────────────────────────────────────
#
# 需先 `make serve`。
api-root: ## 服务信息 + 工具清单 + 缓存状态
	curl -s http://$(ADDR)/ | python3 -m json.tool --no-ensure-ascii

api-tools: ## 工具清单
	curl -s http://$(ADDR)/api/tools | python3 -m json.tool --no-ensure-ascii

api-run: ## ★ 跑一次 Agent
	curl -s -X POST http://$(ADDR)/api/run \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool --no-ensure-ascii

api-run2: ## ★ ★ 连跑两次（第二次命中缓存）
	curl -s -X POST http://$(ADDR)/api/run \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" > /dev/null
	curl -s -X POST http://$(ADDR)/api/run \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool --no-ensure-ascii

api-stats: ## ★ 缓存统计（命中率）
	curl -s http://$(ADDR)/api/cache/stats | python3 -m json.tool --no-ensure-ascii

api-keys: ## 列出缓存键
	curl -s http://$(ADDR)/api/cache/entries | python3 -m json.tool --no-ensure-ascii

api-clear: ## 清空缓存命名空间
	curl -s -X DELETE http://$(ADDR)/api/cache | python3 -m json.tool --no-ensure-ascii

# ── Redis 容器 ────────────────────────────────────────────────────────
#
# ⚠️ Redis 用**独立容器 + 独立端口**，不复用 Milvus 那套。
#    缓存是可选组件，挂了要能降级 —— 所以它必须能独立起停。
REDIS_NAME := agent-learn-redis
REDIS_PORT := 6379

redis-up: ## 启动 Redis（docker）
	@if [ -n "$$(docker ps -q -f name=^/$(REDIS_NAME)$$)" ]; then \
		echo "Redis 已在运行"; \
	else \
		docker run -d --name $(REDIS_NAME) -p $(REDIS_PORT):6379 redis:7-alpine && \
		sleep 2 && echo "Redis 已启动（127.0.0.1:$(REDIS_PORT)）"; \
	fi

redis-down: ## 停止并删除 Redis 容器
	@docker rm -f $(REDIS_NAME) 2>/dev/null || echo "Redis 未运行"

redis-logs: ## 查看 Redis 日志
	docker logs --tail 50 $(REDIS_NAME)

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
