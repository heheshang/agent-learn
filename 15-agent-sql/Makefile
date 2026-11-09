# ============================================================================
# 15-agent-sql —— 手撕 SQL Agent（LangChain SQL Agent）
#
# 核心卖点：**安全护栏 + 只读连接 + 行数上限** 三层防线。
#
# 快速上手：
#   make help          查看所有目标
#   make guard         ★ 只跑护栏（不调模型、不碰数据库）
#   make call          ★ 直接调工具（不经模型）
#   make run           跑一次 Agent
#   make serve         启动 HTTP 服务（:3080）
# ============================================================================

BIN := agent-15
PORT := 3080
ADDR := 127.0.0.1:$(PORT)

# 本目录的绝对路径 —— 让数据库路径与 cwd 无关。
#
# ⚠️ 必须用绝对路径：`cargo run -p agent-15` 在 workspace 根目录跑时
#    cwd 是根目录，`data/chinook.db` 会找不到。
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
export SQL_DB_PATH := $(ROOT)/data/chinook.db
export SQL_SEED_PATH := $(ROOT)/data/seed.sql

# 默认问题（`make run QUESTION="..."` 可覆盖）。
#
# ⚠️ 故意不等于 few-shot 示例里的问题（「有哪些流派」）——
#    否则模型会直接抄示例的答案，演示就失真了。
QUESTION ?= 哪个流派的曲目平均时长最长？

# 默认要检查的 SQL。
SQL ?= DELETE FROM Track

.PHONY: help build run trace raw prompt call guard schema seed serve all \
        test lint fmt clean check api-root api-tools api-schema api-call api-run api-guard

# ── 常用命令 ──────────────────────────────────────────────────────────
help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## ★ 跑一次 Agent（ReAct 模式）（QUESTION="你的问题" 可覆盖）
	cargo run -p $(BIN) -- run "$(QUESTION)"

trace: ## ★ 打印完整 ReAct 轨迹
	cargo run -p $(BIN) -- trace "$(QUESTION)"

raw: ## 跑一次 Agent 并打印模型原始输出（排查解析问题）
	SQL_SHOW_RAW=1 cargo run -p $(BIN) -- run "$(QUESTION)"

prompt: ## 跑一次 Agent 并打印**完整提示词**（排查模型行为）
	SQL_SHOW_PROMPT=1 cargo run -p $(BIN) -- run "$(QUESTION)"

call: ## ★ 直接调用工具（不经模型）SQL="SELECT ..."
	cargo run -p $(BIN) -- call sql_db_query '{"query":"SELECT Name FROM Genre LIMIT 5"}'

guard: ## ★ 只跑护栏（不执行）SQL="DELETE FROM Track"
	cargo run -p $(BIN) -- guard "$(SQL)"

schema: ## 数据库概览（不调模型）
	cargo run -p $(BIN) -- schema

seed: ## 用 data/seed.sql 生成种子数据库
	cargo run -p $(BIN) -- seed

serve: ## 启动 HTTP 服务（127.0.0.1:3080）
	SQL_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

all: ## 全部跑一遍（schema → guard → call → run）
	cargo run -p $(BIN) -- all

# ── HTTP 端点 ─────────────────────────────────────────────────────────
#
# 需先 `make serve`。
api-root: ## 用 HTTP 端点看服务信息 + 工具清单
	curl -s http://$(ADDR)/ | python3 -m json.tool --no-ensure-ascii

api-tools: ## 用 HTTP 端点查工具清单
	curl -s http://$(ADDR)/api/tools | python3 -m json.tool --no-ensure-ascii

api-schema: ## 用 HTTP 端点查数据库概览
	curl -s http://$(ADDR)/api/schema | python3 -m json.tool --no-ensure-ascii

api-call: ## ★ 用 HTTP 端点直接调工具（不经模型）
	curl -s -X POST http://$(ADDR)/api/tools/call \
		-H 'Content-Type: application/json' \
		-d '{"tool": "sql_db_query", "args": {"query": "SELECT Name FROM Genre LIMIT 5"}}' \
		| python3 -m json.tool --no-ensure-ascii

api-run: ## 用 HTTP 端点跑一次 Agent
	curl -s -X POST http://$(ADDR)/api/run \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool --no-ensure-ascii

api-guard: ## ★ 用 HTTP 端点只跑护栏（不执行）
	curl -s -X POST http://$(ADDR)/api/guard \
		-H 'Content-Type: application/json' \
		-d "{\"query\": \"$(SQL)\"}" | python3 -m json.tool --no-ensure-ascii

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
