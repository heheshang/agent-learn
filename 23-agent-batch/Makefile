# ============================================================================
# 23-agent-batch —— 手撕批处理优化（Batching）
#
# 核心卖点：**Embedding 和 Reranker 的批处理**，
#           以及「批处理到底快多少」的**可复现对照实验**。
#
# 快速上手：
#   make help          查看所有目标
#   make modes         ★ 打印三档批策略说明
#   make bench         ★ ★ ★ 嵌入批处理对照实验（本地 ONNX）
#   make bench-rerank  ★ ★ ★ 重排批处理对照实验（本地 ONNX）
#   make sweep         ★ ★ ★ 批大小扫描（找最优批大小）
#   make bench-remote  ★ ★ 远程 HTTP 对照实验（需 Ollama）
#   make run           批量嵌入一条
#   make serve         启动 HTTP 服务（:3150）
# ============================================================================

BIN := agent-23
PORT := 3150
ADDR := 127.0.0.1:$(PORT)

# 对照实验的样本条数。
#
# ⚠️ 必须**够大**才能看出差异 ——
#    20 项目的教训：「语料太小瓶颈显现不出来」。
SAMPLE_SIZE ?= 64

# 对照实验轮数（取中位数，降低抖动影响）。
ROUNDS ?= 3

# 批大小扫描的档位。
BATCH_SIZES ?= 1,2,4,8,16,32,64,128

# 远程对照实验的批大小与客户端并发上限。
REMOTE_BATCH ?= 32
CONCURRENCY ?= 8

.PHONY: help build run modes bench bench-rerank sweep bench-remote \
        serve all test lint fmt clean check \
        api-root api-embed api-rerank api-bench api-bench-rerank api-sweep api-search

# ── 常用命令 ──────────────────────────────────────────────────────────
help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## ★ 批量嵌入一条
	cargo run -p $(BIN) -- embed "批处理优化能提升多少吞吐？"

modes: ## ★ 打印三档批策略说明
	cargo run -p $(BIN) -- modes

bench: ## ★ ★ ★ 嵌入批处理对照实验（本地 ONNX）
	BATCH_SAMPLE_SIZE=$(SAMPLE_SIZE) BATCH_ROUNDS=$(ROUNDS) \
		cargo run --release -p $(BIN) -- bench

bench-rerank: ## ★ ★ ★ 重排批处理对照实验（本地 ONNX）
	BATCH_SAMPLE_SIZE=$(SAMPLE_SIZE) BATCH_ROUNDS=$(ROUNDS) \
		cargo run --release -p $(BIN) -- bench-rerank

sweep: ## ★ ★ ★ 批大小扫描（找最优批大小）
	BATCH_SAMPLE_SIZE=$(SAMPLE_SIZE) BATCH_ROUNDS=$(ROUNDS) \
		cargo run --release -p $(BIN) -- sweep

bench-remote: ## ★ ★ 远程 HTTP 对照实验（需 Ollama 在跑）
	BATCH_SAMPLE_SIZE=$(SAMPLE_SIZE) BATCH_MAX_CONCURRENCY=$(CONCURRENCY) \
		cargo run --release -p $(BIN) -- bench-remote

serve: ## 启动 HTTP 服务（127.0.0.1:3150）
	BATCH_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

all: ## 全部跑一遍（modes → bench → bench-rerank）
	cargo run -p $(BIN) -- modes
	$(MAKE) bench
	$(MAKE) bench-rerank

# ── HTTP 端点 ─────────────────────────────────────────────────────────
#
# ⚠️ 需先 `make serve`。
#
# ⚠️ 用 `python3 -m json.tool --no-ensure-ascii` ——
#    `json.tool` 的 `ensure_ascii` 默认是 True，
#    会把中文转义成 `\uXXXX`（人眼不可读）。
api-root: ## 服务信息 + 批策略清单
	curl -s http://$(ADDR)/ | python3 -m json.tool --no-ensure-ascii

api-embed: ## ★ 批量嵌入
	curl -s -X POST http://$(ADDR)/api/embed \
		-H 'Content-Type: application/json' \
		-d '{"texts":["第一条文本","第二条文本","第三条文本"],"strategy":"fixed"}' \
		| python3 -m json.tool --no-ensure-ascii

api-rerank: ## ★ 批量重排
	curl -s -X POST http://$(ADDR)/api/rerank \
		-H 'Content-Type: application/json' \
		-d '{"query":"批处理","documents":[{"id":"d1","text":"批处理优化"},{"id":"d2","text":"完全无关的内容"}],"top_n":2}' \
		| python3 -m json.tool --no-ensure-ascii

api-bench: ## ★ ★ ★ 嵌入对照实验
	curl -s -X POST http://$(ADDR)/api/bench/embed \
		-H 'Content-Type: application/json' \
		-d "{\"sample_size\":$(SAMPLE_SIZE),\"rounds\":$(ROUNDS)}" \
		| python3 -m json.tool --no-ensure-ascii

api-bench-rerank: ## ★ ★ ★ 重排对照实验
	curl -s -X POST http://$(ADDR)/api/bench/rerank \
		-H 'Content-Type: application/json' \
		-d "{\"sample_size\":$(SAMPLE_SIZE),\"rounds\":$(ROUNDS)}" \
		| python3 -m json.tool --no-ensure-ascii

api-sweep: ## ★ ★ ★ 批大小扫描
	curl -s -X POST http://$(ADDR)/api/bench/sweep \
		-H 'Content-Type: application/json' \
		-d "{\"sample_size\":$(SAMPLE_SIZE),\"rounds\":$(ROUNDS),\"batch_sizes\":[$(subst ,, ,$(BATCH_SIZES))]}" \
		| python3 -m json.tool --no-ensure-ascii

api-search: ## ★ 批量检索（召回 + 批量重排）
	curl -s -X POST http://$(ADDR)/api/search \
		-H 'Content-Type: application/json' \
		-d '{"query":"批处理如何提升吞吐量","top_k":20,"top_n":3,"strategy":"bucketed"}' \
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
