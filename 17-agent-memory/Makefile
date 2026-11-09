# ============================================================================
# 17-agent-memory —— 手撕 Agent Memory（消息状态 + thread_id + checkpointer）
#
# 核心卖点：**两级记忆**
#           短期记忆（线程内，checkpointer 持久化）
#           长期记忆（跨线程，事实抽取 + 召回）
#
# 快速上手：
#   make help          查看所有目标
#   make chat          ★ 跑一次带记忆的对话（两轮：告诉名字 → 问名字）
#   make compare       ★ 线程隔离 / 恢复对照
#   make threads       列出所有线程
#   make memories      列出长期记忆
#   make serve         启动 HTTP 服务（:3100）
# ============================================================================

BIN := agent-17
PORT := 3100
ADDR := 127.0.0.1:$(PORT)

# 本目录的绝对路径 —— 让存储路径与 cwd 无关。
#
# ⚠️ 必须用绝对路径：`cargo run -p agent-17` 在 workspace 根目录跑时
#    cwd 是根目录，`data/` 会找不到。
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
export MEM_DATA_DIR := $(ROOT)/data/memory

# 默认问题（`make chat QUESTION="..."` 可覆盖）。
#
# ⚠️ 必须是「模型不可能凭空知道」的问题 ——
#    这样「答对了」才一定是「记忆生效了」。
QUESTION ?= 我叫 Bob

.PHONY: help build run chat compare threads memories history serve all \
        test lint fmt clean check api-root api-chat api-threads api-memories api-compare

# ── 常用命令 ──────────────────────────────────────────────────────────
help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## ★ 跑一次带记忆的对话（两轮）
	cargo run -p $(BIN) -- chat "$(QUESTION)"

chat: run ## ★ 同 run（别名）

compare: ## ★ 线程隔离 / 恢复对照
	cargo run -p $(BIN) -- compare

threads: ## 列出所有线程
	cargo run -p $(BIN) -- threads

memories: ## 列出长期记忆
	cargo run -p $(BIN) -- memories

history: ## 查看某个线程的历史（THREAD="demo-thread"）
	cargo run -p $(BIN) -- history "$(or $(THREAD),demo-thread)"

serve: ## 启动 HTTP 服务（127.0.0.1:3100）
	MEM_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

all: ## 全部跑一遍（compare → threads → memories）
	cargo run -p $(BIN) -- all

# ── HTTP 端点 ─────────────────────────────────────────────────────────
#
# 需先 `make serve`。
api-root: ## 用 HTTP 端点看服务信息 + 配置
	curl -s http://$(ADDR)/ | python3 -m json.tool --no-ensure-ascii

api-chat: ## ★ 用 HTTP 端点跑一次带记忆的对话
	curl -s -X POST http://$(ADDR)/api/chat \
		-H 'Content-Type: application/json' \
		-d "{\"thread_id\": \"demo-thread\", \"owner\": \"default\", \"question\": \"$(QUESTION)\"}" \
		| python3 -m json.tool --no-ensure-ascii

api-threads: ## 用 HTTP 端点列出所有线程
	curl -s http://$(ADDR)/api/threads | python3 -m json.tool --no-ensure-ascii

api-memories: ## 用 HTTP 端点列出长期记忆
	curl -s 'http://$(ADDR)/api/memories?owner=default' | python3 -m json.tool --no-ensure-ascii

api-compare: ## ★ 用 HTTP 端点跑线程隔离 / 恢复对照
	curl -s -X POST http://$(ADDR)/api/compare \
		-H 'Content-Type: application/json' \
		-d '{"owner": "default", "threads": ["thread-a", "thread-b"], "question": "我叫什么名字？"}' \
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
