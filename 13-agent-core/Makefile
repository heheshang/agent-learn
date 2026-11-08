.PHONY: help build run compare trace tools parse serve test lint fmt clean check

BIN := agent-13
PORT := 3060
ADDR := 127.0.0.1:$(PORT)

# 默认问题（`make run QUESTION="..."` 可覆盖）。
#
# ⚠️ 故意不等于 few-shot 示例里的问题（MRKL 那条）——
# 否则模型会直接抄示例答案，消融实验就失真了。
QUESTION ?= Agent 的三大组件是什么？

# ── 常用命令 ────────────────────────────────────────────────
help: ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## ★ 跑一次 Agent（ReAct 模式）（QUESTION="你的问题" 可覆盖）
	cargo run -p $(BIN) -- run "$(QUESTION)"

compare: ## ★ 三模式消融对照（direct / act_only / react）
	cargo run -p $(BIN) -- compare "$(QUESTION)"

trace: ## ★ 打印完整 ReAct 轨迹
	cargo run -p $(BIN) -- trace "$(QUESTION)"

raw: ## 跑一次 Agent 并打印模型原始输出（排查解析问题）
	AGENT_SHOW_RAW=1 cargo run -p $(BIN) -- run "$(QUESTION)"

tools: ## 工具说明 + 工具实测（不调模型）
	cargo run -p $(BIN) -- tools

parse: ## 只测试输出解析器（不调模型）
	cargo run -p $(BIN) -- parse

serve: ## 启动 HTTP 服务（127.0.0.1:3060）
	AGENT_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

api-run: ## 用 HTTP 端点跑一次 Agent
	curl -s -X POST http://$(ADDR)/api/run \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool

api-compare: ## ★ 用 HTTP 端点跑三模式对照
	curl -s -X POST http://$(ADDR)/api/compare \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool

# ── 质量 ────────────────────────────────────────────────────
test: ## 跑全部测试
	cargo test -p $(BIN)

lint: ## clippy 零警告检查
	cargo clippy -p $(BIN) --all-targets -- -D warnings

fmt: ## 格式化
	cargo fmt -p $(BIN)

check: fmt lint test ## 格式化 + lint + 测试

clean: ## 清理构建产物
	cargo clean -p $(BIN)
