# 25-agent-summary —— Agent 全链路汇总与压测台
#
# ⚠️ 本项目**不依赖**其它项目 —— 能力全部自己实现。
#    依赖版本只在根 Cargo.toml 的 [workspace.dependencies] 里管理。

.PHONY: help build run test lint fmt clean serve bench bench-http profile sweep cache-sweep report agent compare

help: ## 显示所有可用目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译（release）
	@echo "⚠️ 25 用 release 构建 —— debug 构建慢到淹没优化前后的差值"
	cargo build --release -p agent-25

run: build ## 运行帮助
	./target/release/agent-25 help

test: ## 跑测试（零网络零模型，全确定性）
	cargo test -p agent-25

lint: ## clippy（-D warnings 零警告门槛）
	cargo clippy -p agent-25 --all-targets -- -D warnings

fmt: ## 格式化
	cargo fmt -p agent-25

clean: ## 清理
	cargo clean -p agent-25

serve: build ## 启动 HTTP 服务（:3170）
	@echo "服务地址：http://127.0.0.1:3170"
	@echo "  GET  /health"
	@echo "  GET  /modules          列出汇总了哪些能力"
	@echo "  GET  /summary?start=&end=  生成周报"
	@echo "  POST /summary          生成周报（JSON body）"
	@echo "  POST /search           混合检索"
	@echo "  POST /strategy         切换策略（naive / pre-agg / optimized）"
	@echo "  POST /bench            三档对照实验"
	@echo "  POST /agent            跑一次 Agent（protocol: function / react / structured）"
	./target/release/agent-25 serve

bench: build ## ★ 三档对照实验（进程内轨 —— 用来归因）
	./target/release/agent-25 bench

bench-http: build ## ★ HTTP 轨（打真实服务 —— 外部视角验证）
	@echo "⚠️ 需要先另开一个终端跑 'make serve'"
	./target/release/agent-25 bench-http

profile: build ## ★ 分段剖析 —— 指出时间花在哪
	./target/release/agent-25 profile

sweep: build ## ★ 并发度扫描
	./target/release/agent-25 sweep

cache-sweep: build ## ★ 缓存命中率扫描（缓存收益的唯一自变量是命中率）
	./target/release/agent-25 cache-sweep

report: build ## 生成周报 Markdown（写到 reports/weekly-summary.md）
	./target/release/agent-25 report

agent: build ## ★ 真调模型跑一次 Agent 循环（需要 DEEPSEEK_API_KEY）
	@echo "⚠️ 需要 DEEPSEEK_API_KEY —— 没有会明确报错，不会静默降级"
	./target/release/agent-25 agent "$(or $(Q),cache 模块上周做了哪些事？有没有风险项？)"

compare: build ## ★ 协议对照 —— ReAct vs Function Calling vs 结构化输出（需要 DEEPSEEK_API_KEY）
	@echo "⚠️ 需要 DEEPSEEK_API_KEY —— 没有会明确报错，不会静默降级"
	./target/release/agent-25 compare "$(or $(Q),cache 模块上周做了哪些事？有没有风险项？)"
