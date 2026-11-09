.PHONY: help build run compare trace call tools parse mcp serve test lint fmt clean check api-tools api-call api-run api-compare

BIN := agent-14
PORT := 3070
MCP_PORT := 3071
ADDR := 127.0.0.1:$(PORT)
MCP_ADDR := 127.0.0.1:$(MCP_PORT)

# 默认问题（`make run QUESTION="..."` 可覆盖）。
#
# ⚠️ 故意不等于 few-shot 示例里的问题（上海那条）——
#    否则模型会直接抄示例答案，消融实验就失真了。
QUESTION ?= 北京现在天气怎么样？

# 默认调用的工具参数。
LOCATION ?= 北京

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
	TOOL_SHOW_RAW=1 cargo run -p $(BIN) -- run "$(QUESTION)"

prompt: ## 跑一次 Agent 并打印**完整提示词**（排查模型行为）
	TOOL_SHOW_PROMPT=1 cargo run -p $(BIN) -- run "$(QUESTION)"

call: ## ★ 直接调用工具（不经模型）LOCATION="上海"
	cargo run -p $(BIN) -- call get_weather '{"location":"$(LOCATION)","days":3}'

tools: ## 工具说明 + 参数校验演示（不调模型）
	cargo run -p $(BIN) -- tools

parse: ## 只测试输出解析器（不调模型）
	cargo run -p $(BIN) -- parse

mcp: ## 启动内置 MCP 服务器（127.0.0.1:3071）
	TOOL_MCP_BIND_ADDR=$(MCP_ADDR) cargo run -p $(BIN) -- mcp

serve: ## 启动 HTTP 服务（127.0.0.1:3070）
	TOOL_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

api-tools: ## 用 HTTP 端点查工具清单
	curl -s http://$(ADDR)/api/tools | python3 -m json.tool --no-ensure-ascii

api-call: ## ★ 用 HTTP 端点直接调工具
	curl -s -X POST http://$(ADDR)/api/tools/call \
		-H 'Content-Type: application/json' \
		-d "{\"tool\": \"get_weather\", \"args\": {\"location\": \"$(LOCATION)\"}}" | python3 -m json.tool --no-ensure-ascii

api-run: ## 用 HTTP 端点跑一次 Agent
	curl -s -X POST http://$(ADDR)/api/run \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool --no-ensure-ascii

api-compare: ## ★ 用 HTTP 端点跑三模式对照
	curl -s -X POST http://$(ADDR)/api/compare \
		-H 'Content-Type: application/json' \
		-d "{\"question\": \"$(QUESTION)\"}" | python3 -m json.tool --no-ensure-ascii

# ── MCP 客户端直连演示 ────────────────────────────────────────
#
# 需先起 MCP 服务器：`make mcp`
mcp-list: ## 用 MCP 协议直连内置服务器（tools/list）
	curl -s -X POST http://$(MCP_ADDR)/mcp \
		-H 'Content-Type: application/json' \
		-d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' -i | head -8
	@echo ""
	curl -s -X POST http://$(MCP_ADDR)/mcp \
		-H 'Content-Type: application/json' \
		-d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | python3 -m json.tool --no-ensure-ascii

mcp-call: ## 用 MCP 协议调内置服务器的 get_weather
	curl -s -X POST http://$(MCP_ADDR)/mcp \
		-H 'Content-Type: application/json' \
		-d "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"get_weather\",\"arguments\":{\"location\":\"$(LOCATION)\"}}}" | python3 -m json.tool --no-ensure-ascii

mcp-agent: ## ★ 让 Agent **通过 MCP** 调工具（需先起 MCP 服务器）
	TOOL_ENABLE_MCP=1 TOOL_MCP_ENDPOINT=http://$(MCP_ADDR)/mcp \
		cargo run -p $(BIN) -- run "$(QUESTION)"

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
