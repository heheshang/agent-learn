# ============================================================================
# 03-genai —— 多厂商 LLM 调用 + 工具 + Agent 循环（genai / rust-genai-agent）
#
# 用法：
#   make help          查看所有目标
#   make agent         Agent 自动工具循环
#   make hooks         工具钩子（middleware）
#
# ⚠️ 需要可用的模型 API key（DEEPSEEK_API_KEY 或对应厂商的 key）
# ============================================================================

PKG      := genai-03
CARGO   ?= cargo

.DEFAULT_GOAL := help
.PHONY: help build run test check fmt fmt-check lint clean \
        chat tools agent hooks http state all

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	$(CARGO) build -p $(PKG)

run: build ## 运行（默认 all demo）
	$(CARGO) run -p $(PKG)

test: ## 跑测试
	$(CARGO) test -p $(PKG)

check: ## 编译检查
	$(CARGO) check -p $(PKG)

fmt: ## 格式化
	$(CARGO) fmt -p $(PKG)

fmt-check: ## 检查格式
	$(CARGO) fmt -p $(PKG) -- --check

lint: ## clippy 静态检查
	$(CARGO) clippy -p $(PKG) --all-targets -- -D warnings

clean: ## 清理构建产物
	$(CARGO) clean -p $(PKG)

## ── Demo ────────────────────────────────────────────────────────────────

chat: build ## 模型层：单轮 / 多轮 / 流式 / JSON
	$(CARGO) run -p $(PKG) -- chat

tools: build ## 模型层：原始工具调用（手工执行 + 回填）
	$(CARGO) run -p $(PKG) -- tools

agent: build ## Agent 层：自动工具循环
	$(CARGO) run -p $(PKG) -- agent

hooks: build ## Agent 层：工具钩子（middleware）
	$(CARGO) run -p $(PKG) -- hooks

http: build ## Agent 层：HTTP 边界 middleware
	$(CARGO) run -p $(PKG) -- http

state: build ## Agent 层：状态累积与重置
	$(CARGO) run -p $(PKG) -- state

all: build ## 跑全部 demo
	$(CARGO) run -p $(PKG) -- all
