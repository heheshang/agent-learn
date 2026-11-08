.PHONY: help build demo compare eval serve test lint fmt clean check reindex

BIN := rag-12
PORT := 3050
ADDR := 127.0.0.1:$(PORT)

# 默认问题（`make demo QUESTION="..."` 可覆盖）。
QUESTION ?= 混合检索和重排分别解决什么问题

# ── 常用命令 ────────────────────────────────────────────────
help: ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

demo: ## ★ 命令行问答（QUESTION="你的问题" 可覆盖）
	cargo run -p $(BIN) -- demo "$(QUESTION)"

compare: ## ★ 同题消融对照（阶梯各档）
	cargo run -p $(BIN) -- compare "$(QUESTION)"

eval: ## ★ 跑评估集，产出消融报告
	cargo run -p $(BIN) -- eval

eval-limit: LIMIT=3 ## 跑评估集（只跑前 LIMIT 条，快速冒烟）
	SUMMARY_EVAL_LIMIT=$(LIMIT) cargo run -p $(BIN) -- eval

serve: ## 启动 HTTP 服务（127.0.0.1:3050）
	SUMMARY_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

reindex: ## 重建 Milvus 索引
	curl -s -X POST http://$(ADDR)/api/reindex | python3 -m json.tool

# ── 质量 ────────────────────────────────────────────────────
test: ## 跑全部测试
	cargo test -p $(BIN)

test-backend: ## 跑测试并强制要求 Milvus 可达
	RAG12_REQUIRE_BACKEND=1 cargo test -p $(BIN)

lint: ## clippy 零警告检查
	cargo clippy -p $(BIN) --all-targets -- -D warnings

fmt: ## 格式化
	cargo fmt -p $(BIN)

check: fmt lint test ## 格式化 + lint + 测试

clean: ## 清理构建产物
	cargo clean -p $(BIN)
