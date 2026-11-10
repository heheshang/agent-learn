# ============================================================================
# 24-agent-vllm —— 手撕批处理优化（vLLM 连续批处理）
#
# 核心卖点：**把「批处理为什么有效」变成可复现的实验**，
#           并且给出**能直接拿去用的 vLLM 部署与压测流程**。
#
# 快速上手：
#   make help          查看所有目标
#   make modes         ★ 打印两种批处理模式的说明
#   make bench         ★ ★ ★ 三组对照实验（模拟轨，本机可跑）
#   make simulate      ★ 用调度器跑一次负载
#   make mock          启动 mock vLLM（:8000，OpenAI 兼容）
#   make docker-up     ★ 启动真 vLLM 容器（需要 NVIDIA GPU + Linux）
#   make bench-vllm    ★ ★ ★ ★ 打真 vLLM 测吞吐
#   make serve         启动 HTTP 服务（:3160）
# ============================================================================

BIN := agent-24
PORT := 3160
ADDR := 127.0.0.1:$(PORT)

# 真 vLLM 服务（docker 里的 8000 端口）。
VLLM_PORT ?= 8000
VLLM_ADDR := 127.0.0.1:$(VLLM_PORT)

# ── 对照实验参数 ──────────────────────────────────────────────────────
#
# ⚠️ 请求条数要**够大** —— 条数太少时两种批处理模式的差异会被抖动淹没。
NUM_REQUESTS ?= 200

# 到达模型：burst（突发）/ constant（固定速率）/ poisson（泊松）。
ARRIVAL_MODEL ?= burst

# 长度分布：fixed / uniform / lognormal。
#
# ⚠️ ★ **uniform 或 lognormal 才能测出静态 vs 连续批处理的差异** ——
#    长度完全一致时两种模式**完全等价**（测不出差异是正确行为）。
LENGTH_DIST ?= uniform

# 客户端并发度（★ 压测真 vLLM 时的关键变量）。
CONCURRENCY ?= 32

# 服务端并发上限（对应 vLLM 的 --max-num-seqs）。
MAX_NUM_SEQS ?= 16

.PHONY: help build run bench bench-mode bench-seqs bench-chunked simulate \
        modes serve mock health bench-vllm docker-up docker-down docker-logs \
        api-root api-modes api-env api-simulate api-bench \
        test lint fmt clean check all

# ── 常用命令 ──────────────────────────────────────────────────────────
help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

run: ## ★ 跑一次模拟（两种批处理模式对照）
	cargo run -p $(BIN) -- simulate

modes: ## ★ 打印两种批处理模式的说明
	cargo run -p $(BIN) -- modes

bench: ## ★ ★ ★ 三组对照实验（模拟轨，本机可跑）
	VLLM_NUM_REQUESTS=$(NUM_REQUESTS) \
	VLLM_ARRIVAL_MODEL=$(ARRIVAL_MODEL) \
	VLLM_LENGTH_DIST=$(LENGTH_DIST) \
	VLLM_MAX_NUM_SEQS=$(MAX_NUM_SEQS) \
	cargo run -p $(BIN) -- bench

bench-mode: ## ★ 实验 A：静态批处理 vs 连续批处理
	VLLM_NUM_REQUESTS=$(NUM_REQUESTS) \
	VLLM_ARRIVAL_MODEL=$(ARRIVAL_MODEL) \
	VLLM_LENGTH_DIST=$(LENGTH_DIST) \
	cargo run -p $(BIN) -- bench-mode

bench-seqs: ## ★ 实验 B：max_num_seqs 扫描（找甜点）
	VLLM_NUM_REQUESTS=$(NUM_REQUESTS) \
	VLLM_ARRIVAL_MODEL=$(ARRIVAL_MODEL) \
	VLLM_LENGTH_DIST=$(LENGTH_DIST) \
	cargo run -p $(BIN) -- bench-seqs

bench-chunked: ## ★ 实验 C：chunked prefill 对照
	VLLM_NUM_REQUESTS=$(NUM_REQUESTS) \
	VLLM_ARRIVAL_MODEL=$(ARRIVAL_MODEL) \
	VLLM_LENGTH_DIST=$(LENGTH_DIST) \
	cargo run -p $(BIN) -- bench-chunked

simulate: ## ★ 用调度器跑一次负载
	cargo run -p $(BIN) -- simulate

serve: ## 启动 HTTP 服务（127.0.0.1:3160）
	VLLM_BIND_ADDR=$(ADDR) cargo run -p $(BIN) -- serve

mock: ## ★ 启动 mock vLLM（127.0.0.1:8000，OpenAI 兼容）
	VLLM_BASE_URL=http://$(VLLM_ADDR) cargo run -p $(BIN) -- mock

health: ## ★ 检查 vLLM 服务是否可达
	VLLM_BASE_URL=http://$(VLLM_ADDR) cargo run -p $(BIN) -- health

bench-vllm: ## ★ ★ ★ ★ 打真 vLLM 测吞吐
	VLLM_BASE_URL=http://$(VLLM_ADDR) \
	VLLM_NUM_REQUESTS=$(NUM_REQUESTS) \
	VLLM_CONCURRENCY=$(CONCURRENCY) \
	cargo run -p $(BIN) -- bench-vllm

# ── Docker（真 vLLM，需要 NVIDIA GPU + Linux）────────────────────────
docker-up: ## ★ 启动真 vLLM 容器（需要 NVIDIA GPU）
	VLLM_MAX_NUM_SEQS=$(MAX_NUM_SEQS) cargo run -p $(BIN) -- docker-up

docker-down: ## 停掉 vLLM 容器
	cargo run -p $(BIN) -- docker-down

docker-logs: ## 看 vLLM 容器日志（等模型下载）
	docker compose -f docker/docker-compose.yml logs -f

# ── HTTP 端点 ─────────────────────────────────────────────────────────
#
# 需先 `make serve`。
api-root: ## 服务信息
	curl -s http://$(ADDR)/ | python3 -m json.tool --no-ensure-ascii

api-modes: ## ★ 批处理模式说明
	curl -s http://$(ADDR)/api/modes | python3 -m json.tool --no-ensure-ascii

api-env: ## 当前生效的配置
	curl -s http://$(ADDR)/api/env | python3 -m json.tool --no-ensure-ascii

api-simulate: ## ★ 用调度器跑一次负载
	curl -s -X POST http://$(ADDR)/api/simulate \
		-H 'Content-Type: application/json' \
		-d '{"num_requests":100,"arrival_model":"burst","length_distribution":"uniform","min_input_tokens":32,"max_input_tokens":512,"min_output_tokens":16,"max_output_tokens":128}' \
		| python3 -m json.tool --no-ensure-ascii

api-bench: ## ★ ★ ★ 三组对照实验
	curl -s -X POST http://$(ADDR)/api/bench \
		-H 'Content-Type: application/json' \
		-d "{\"num_requests\":$(NUM_REQUESTS),\"arrival_model\":\"$(ARRIVAL_MODEL)\",\"length_distribution\":\"$(LENGTH_DIST)\",\"rounds\":3}" \
		| python3 -m json.tool --no-ensure-ascii

# ── 质量 ──────────────────────────────────────────────────────────────
test: ## 跑全部测试（82 项）
	cargo test -p $(BIN)

lint: ## clippy 零警告检查
	cargo clippy -p $(BIN) --all-targets -- -D warnings

fmt: ## 格式化
	cargo fmt -p $(BIN)

check: fmt lint test ## 格式化 + lint + 测试

clean: ## 清理构建产物
	cargo clean -p $(BIN)

all: ## 全部跑一遍（modes → bench）
	cargo run -p $(BIN) -- modes
	cargo run -p $(BIN) -- bench
