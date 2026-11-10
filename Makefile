# ============================================================================
# agent-learn —— workspace 级统一入口
#
# 用法：
#   make help          查看所有目标
#   make build         编译全部
#   make test          跑全部测试
#   make lint          clippy 全量检查
#   make rag           RAG 主线一链跑（04 → 05 → 06）
#   make serve-07      启动 07-query 服务
# ============================================================================

CARGO := cargo
PKGS  := axum-01 rig-02 genai-03 rag-04 rag-05 rag-06 rag-07 rag-08 rag-09 rag-10 rag-11 rag-12 agent-13 agent-14 agent-15 agent-16 agent-17 agent-18 agent-19 agent-20 agent-21 agent-22 agent-23 agent-24 agent-25

.DEFAULT_GOAL := help
.PHONY: help build run test check fmt fmt-check lint clean logs \
        01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
        axum rig genai rag agent \
        serve-06 serve-07 serve-08 serve-09 serve-10 serve-11 serve-12 serve-13 serve-14 serve-15 serve-16 serve-17 serve-18 serve-19 serve-21 serve-22 serve-23 serve-24 serve-25 \
        demo-04 demo-05 demo-06 demo-07 demo-08 demo-09 demo-10 demo-11 demo-12 demo-13 demo-14 demo-15 demo-16 demo-17 demo-18 demo-19 demo-21 demo-22 demo-23 demo-24 demo-25 \
        compare-07 compare-08 compare-09 compare-10 compare-11 compare-12 compare-13 compare-14 compare-16 compare-17 compare-18 compare-19 compare-21 compare-22 \
        semantics-21 \
        call-14 guard-15 \
        eval-12 \
        profile-20 pprof-20 bench-20 setup-20 py-compare \
        redis-up-21 redis-down-21 \
        bench-22 bench-real-22 modes-22 \
        bench-23 bench-rerank-23 sweep-23 bench-remote-23 \
        bench-24 bench-mode-24 bench-seqs-24 bench-chunked-24 modes-24 \
        mock-24 health-24 bench-vllm-24 vllm-up-24 vllm-down-24 \
        bench-25 bench-a-25 bench-b-25 bench-c-25 loadtest-25 locust-25 jmeter-25 profile-25 \
        milvus-up milvus-down milvus-destroy \
        check-mermaid check-toc \
        prop prop-08 prop-13 prop-14 prop-15 prop-21 \
        fuzz-build fuzz-run fuzz-list \
        deps faiss chroma

help: ## 显示本文件的所有目标
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## ── Workspace 通用 ─────────────────────────────────────────────────────

build: ## 编译全部
	$(CARGO) build --workspace

run: ## 运行（未指定则跑 04-rag 的 all demo）
	$(CARGO) run -p rag-04

test: ## 跑全部测试
	$(CARGO) test --workspace

check: ## 编译检查
	$(CARGO) check --workspace

fmt: ## 格式化
	$(CARGO) fmt --all

fmt-check: ## 检查格式
	$(CARGO) fmt --all -- --check

lint: ## clippy 全量检查（零警告门槛）
	$(CARGO) clippy --workspace --all-targets -- -D warnings

logs: ## 打印日志过滤示例（用 RUST_LOG 控制各项目日志级别）
	@echo "日志约定：全仓库统一走 00-logging（agent-logging）"
	@echo ""
	@echo "  全局 info        ：RUST_LOG=info"
	@echo "  单项目 debug     ：RUST_LOG=debug,rag_06=debug"
	@echo "  单模块 trace     ：RUST_LOG=info,rag_12::retriever=trace"
	@echo "  只看告警         ：RUST_LOG=warn"
	@echo ""
	@echo "  例：RUST_LOG=debug make serve-12"

clean: ## 清理构建产物
	$(CARGO) clean

## ── 单项目快捷入口（等价于进入子目录 make xxx）─────────────────────────

01: ## 01-axum：启动 HTTP 服务
	$(MAKE) -C 01-axum run

02: ## 02-rig：跑全部 demo
	$(MAKE) -C 02-rig run

03: ## 03-genai：跑全部 demo
	$(MAKE) -C 03-genai run

04: ## 04-rag：加载 + 分割全流程
	$(MAKE) -C 04-rag all

05: ## 05-rag：向量化 + 存储全流程
	$(MAKE) -C 05-rag all

06: ## 06-rag：端到端 demo
	$(MAKE) -C 06-rag demo

07: ## 07-query：端到端 demo
	$(MAKE) -C 07-query demo

08: ## 08-rag：端到端 demo
	$(MAKE) -C 08-rag demo

09: ## 09-rag-eval：端到端 demo
	$(MAKE) -C 09-rag-eval demo

10: ## 10-rag-milvus：端到端 demo
	$(MAKE) -C 10-rag-milvus demo

11: ## 11-rag-advance：端到端 demo
	$(MAKE) -C 11-rag-advance demo

12: ## 12-rag-summary：端到端 demo
	$(MAKE) -C 12-rag-summary demo

13: ## 13-agent-core：跑一次 Agent（ReAct）
	$(MAKE) -C 13-agent-core run

14: ## 14-agent-tool：直接调天气工具（不经模型）
	$(MAKE) -C 14-agent-tool call

15: ## 15-agent-sql：只跑 SQL 护栏（不执行、不调模型）★
	$(MAKE) -C 15-agent-sql guard

16: ## 16-agent-function-calling：打印工具清单 + Schema（不调模型）★
	$(MAKE) -C 16-agent-function-calling tools

17: ## 17-agent-memory：★ 线程隔离 / 恢复对照
	$(MAKE) -C 17-agent-memory compare

18: ## 18-agent-error：★ 三类错误对照（Transient/Business/Fatal）
	$(MAKE) -C 18-agent-error compare

19: ## 19-agent-summary：★ 接地校验对照（接地/部分接地/未接地）
	$(MAKE) -C 19-agent-summary compare

20: ## 20-agent-preformence-analysis：★ 三工具对比 + 优化前后差值
	$(MAKE) -C 20-agent-preformence-analysis py-compare

21: ## 21-agent-cache：★ 缓存命中演示（连跑两次同一问题）
	$(MAKE) -C 21-agent-cache run2

22: ## 22-agent-async：★ ★ ★ 并发对照实验（模拟负载）
	$(MAKE) -C 22-agent-async bench

23: ## 23-agent-batch：★ ★ ★ 批处理对照实验（嵌入 + 重排）
	$(MAKE) -C 23-agent-batch bench

24: ## 24-agent-vllm：★ ★ ★ 批处理对照实验（连续批处理调度器）
	$(MAKE) -C 24-agent-vllm bench

25: ## 25-agent-summary：★ ★ ★ 三档策略对照 + 分段剖析（Agent 全链路汇总）
	$(MAKE) -C 25-agent-summary bench

axum: 01 ## 别名
rig: 02 ## 别名
genai: 03 ## 别名
rag: 04 ## 别名
agent: 13 ## 别名

## ── 服务 ────────────────────────────────────────────────────────────────

serve-06: ## 启动 06-rag 服务（:3000）
	$(MAKE) -C 06-rag serve

serve-07: ## 启动 07-query 服务（:3010）
	$(MAKE) -C 07-query serve

serve-08: ## 启动 08-rag 服务（:3020）
	$(MAKE) -C 08-rag serve

serve-09: ## 启动 09-rag-eval 服务（:3030）
	$(MAKE) -C 09-rag-eval serve

serve-10: ## 启动 10-rag-milvus 服务（:3030）
	$(MAKE) -C 10-rag-milvus serve

serve-11: ## 启动 11-rag-advance 服务（:3040）
	$(MAKE) -C 11-rag-advance serve

serve-12: ## 启动 12-rag-summary 服务（:3050）
	$(MAKE) -C 12-rag-summary serve

serve-13: ## 启动 13-agent-core 服务（:3060）
	$(MAKE) -C 13-agent-core serve

serve-14: ## 启动 14-agent-tool 服务（:3070）
	$(MAKE) -C 14-agent-tool serve

serve-15: ## 启动 15-agent-sql 服务（:3080）
	$(MAKE) -C 15-agent-sql serve

serve-16: ## 启动 16-agent-function-calling 服务（:3090）
	$(MAKE) -C 16-agent-function-calling serve

serve-17: ## 启动 17-agent-memory 服务（:3100）
	$(MAKE) -C 17-agent-memory serve

serve-18: ## 启动 18-agent-error 服务（:3110）
	$(MAKE) -C 18-agent-error serve

serve-19: ## 启动 19-agent-summary 服务（:3120）
	$(MAKE) -C 19-agent-summary serve

serve-21: ## 启动 21-agent-cache 服务（:3130）
	$(MAKE) -C 21-agent-cache serve

serve-22: ## 启动 22-agent-async 服务（:3140）
	$(MAKE) -C 22-agent-async serve

serve-23: ## 启动 23-agent-batch 服务（:3150）
	$(MAKE) -C 23-agent-batch serve

serve-24: ## 启动 24-agent-vllm 服务（:3160）
	$(MAKE) -C 24-agent-vllm serve

serve-25: ## 启动 25-agent-summary 服务（:3170）
	$(MAKE) -C 25-agent-summary serve

mcp-14: ## 启动 14-agent-tool 内置 MCP 服务器（:3071）
	$(MAKE) -C 14-agent-tool mcp

## ── Demo ────────────────────────────────────────────────────────────────

demo-04: ## 04-rag：加载 + 分割
	$(MAKE) -C 04-rag all

demo-05: ## 05-rag：向量化 + 存储
	$(MAKE) -C 05-rag all

demo-06: ## 06-rag：Naive RAG 问答
	$(MAKE) -C 06-rag demo

demo-07: ## 07-query：查询改写
	$(MAKE) -C 07-query demo

demo-08: ## 08-rag：混合检索 + 重排
	$(MAKE) -C 08-rag demo

demo-09: ## 09-rag-eval：评估 + 对比
	$(MAKE) -C 09-rag-eval demo

demo-10: ## 10-rag-milvus：灌数据 + 检索
	$(MAKE) -C 10-rag-milvus demo

demo-11: ## 11-rag-advance：解析复杂文档
	$(MAKE) -C 11-rag-advance demo

demo-12: ## 12-rag-summary：摘要式问答
	$(MAKE) -C 12-rag-summary demo

demo-13: ## 13-agent-core：跑一次 Agent（ReAct）
	$(MAKE) -C 13-agent-core run

demo-14: ## 14-agent-tool：直接调天气工具
	$(MAKE) -C 14-agent-tool call

demo-15: ## 15-agent-sql：跑一次 SQL Agent（ReAct）★
	$(MAKE) -C 15-agent-sql run

demo-16: ## 16-agent-function-calling：跑一次 Function Calling Agent ★
	$(MAKE) -C 16-agent-function-calling run

demo-17: ## 17-agent-memory：跑一次带记忆的对话（两轮）★
	$(MAKE) -C 17-agent-memory chat

demo-18: ## 18-agent-error：跑一次带错误处理的 Agent ★
	$(MAKE) -C 18-agent-error run

demo-19: ## 19-agent-summary：跑一次周度总结（研究 + 成文 + 校验）★
	$(MAKE) -C 19-agent-summary weekly

demo-19-plan: ## 19-agent-summary：跑一次项目构建方案 ★
	$(MAKE) -C 19-agent-summary plan

search-19: ## 19-agent-summary：直接调 Web 搜索工具（不经模型）★
	$(MAKE) -C 19-agent-summary search

demo-21: ## 21-agent-cache：★ 连跑两次同一问题（演示缓存命中）
	$(MAKE) -C 21-agent-cache run2

demo-22: ## 22-agent-async：★ ★ ★ 并发对照实验（模拟负载）
	$(MAKE) -C 22-agent-async bench

demo-23: ## 23-agent-batch：★ ★ ★ 批处理对照实验（嵌入 + 重排）
	$(MAKE) -C 23-agent-batch bench

demo-24: ## 24-agent-vllm：★ ★ ★ 批处理对照实验（连续批处理调度器）
	$(MAKE) -C 24-agent-vllm bench

demo-25: ## 25-agent-summary：★ ★ ★ 三档策略对照 + 分段剖析
	$(MAKE) -C 25-agent-summary bench

call-14: ## 14-agent-tool：直接调天气工具（不经模型）★
	$(MAKE) -C 14-agent-tool call

guard-15: ## 15-agent-sql：只跑 SQL 护栏（不执行、不调模型）★
	$(MAKE) -C 15-agent-sql guard

compare-07: ## 07-query：多策略改写对比 ★
	$(MAKE) -C 07-query compare

compare-08: ## 08-rag：多模式检索对比 ★
	$(MAKE) -C 08-rag compare

compare-09: ## 09-rag-eval：优化前后对比 ★
	$(MAKE) -C 09-rag-eval compare

compare-10: ## 10-rag-milvus：Milvus vs Qdrant 同题对照 ★
	$(MAKE) -C 10-rag-milvus compare

compare-11: ## 11-rag-advance：几何还原 vs 压平对照 ★
	$(MAKE) -C 11-rag-advance compare

compare-12: ## 12-rag-summary：五档消融阶梯同题对照 ★
	$(MAKE) -C 12-rag-summary compare

compare-13: ## 13-agent-core：三模式消融对照（direct/act_only/react）★
	$(MAKE) -C 13-agent-core compare

compare-14: ## 14-agent-tool：三模式消融对照（工具贡献量化）★
	$(MAKE) -C 14-agent-tool compare

compare-16: ## 16-agent-function-calling：tool_choice 四档对照 ★
	$(MAKE) -C 16-agent-function-calling compare

compare-17: ## 17-agent-memory：线程隔离 / 恢复 / 跨线程共享三向对照 ★
	$(MAKE) -C 17-agent-memory compare

compare-18: ## 18-agent-error：三类错误对照（Transient/Business/Fatal）★
	$(MAKE) -C 18-agent-error compare

compare-19: ## 19-agent-summary：接地校验对照（接地/部分接地/未接地）★
	$(MAKE) -C 19-agent-summary compare

compare-21: ## 21-agent-cache：★ 缓存开 / 关对照实验
	$(MAKE) -C 21-agent-cache compare

semantics-21: ## 21-agent-cache：★ ★ 三档缓存语义对照（agentic 链路能否命中的定量实验）
	$(MAKE) -C 21-agent-cache semantics

# ── 22-agent-async 专属 ───────────────────────────────────────────────
#
# ⚠️ 对照实验分两种负载：
#    - bench-22      模拟 I/O（sleep）—— 稳定、可复现，差值干净
#    - bench-real-22 真实天气接口 —— 能看到真实网络下的抖动
#
#    两者都要跑：模拟负载给出「干净的加速比」，
#    真实负载暴露「限流组可能反而更慢」这个反直觉现象。

bench-22: ## 22-agent-async：★ ★ ★ 并发对照实验（模拟负载，稳）★
	$(MAKE) -C 22-agent-async bench

bench-real-22: ## 22-agent-async：★ ★ ★ 并发对照实验（真实天气接口）
	$(MAKE) -C 22-agent-async bench-real

modes-22: ## 22-agent-async：★ ★ 四种并发模式说明
	$(MAKE) -C 22-agent-async modes

# ── 23-agent-batch 专属 ───────────────────────────────────────────────
#
# ⚠️ 本地 ONNX 对照实验**必须用 release 构建** ——
#    debug 构建的推理慢到会淹没批处理的收益差异。

bench-23: ## 23-agent-batch：★ ★ ★ 嵌入批处理对照实验（本地 ONNX）
	$(MAKE) -C 23-agent-batch bench

bench-rerank-23: ## 23-agent-batch：★ ★ ★ 重排批处理对照实验（本地 ONNX）
	$(MAKE) -C 23-agent-batch bench-rerank

sweep-23: ## 23-agent-batch：★ ★ ★ 批大小扫描（找最优批大小）
	$(MAKE) -C 23-agent-batch sweep

bench-remote-23: ## 23-agent-batch：★ ★ 远程 HTTP 对照实验（需 Ollama）
	$(MAKE) -C 23-agent-batch bench-remote

# ── 24-agent-vllm 专属 ───────────────────────────────────────────────
#
# ⚠️ 「真 vLLM」轨需要 **NVIDIA GPU + Linux** ——
#    vLLM 官方镜像只支持 CUDA，macOS（哪怕是 Apple Silicon）跑不了。
#    没有 GPU 时用 mock 轨（`make mock-24`）验证客户端与指标口径。

bench-24: ## 24-agent-vllm：★ ★ ★ 三组批处理对照实验（模拟轨，本机可跑）
	$(MAKE) -C 24-agent-vllm bench

bench-mode-24: ## 24-agent-vllm：★ 实验 A 静态 vs 连续批处理
	$(MAKE) -C 24-agent-vllm bench-mode

bench-seqs-24: ## 24-agent-vllm：★ 实验 B max_num_seqs 扫描（找甜点）
	$(MAKE) -C 24-agent-vllm bench-seqs

bench-chunked-24: ## 24-agent-vllm：★ 实验 C chunked prefill 对照
	$(MAKE) -C 24-agent-vllm bench-chunked

modes-24: ## 24-agent-vllm：★ 两种批处理模式的说明
	$(MAKE) -C 24-agent-vllm modes

# ── 25-agent-summary 专属 ─────────────────────────────────────────────
#
# ⚠️ 25 **不依赖**其它项目 —— 能力全部自己实现。
#    三档策略的对照实验是本项目的核心产出。

bench-25: ## 25-agent-summary：★ 三档策略对照（进程内轨 —— 用来归因）
	$(MAKE) -C 25-agent-summary bench

bench-http-25: ## 25-agent-summary：★ HTTP 轨（外部视角验证，需先 make serve-25）
	$(MAKE) -C 25-agent-summary bench-http

profile-25: ## 25-agent-summary：★ 分段剖析 —— 指出时间花在哪
	$(MAKE) -C 25-agent-summary profile

sweep-25: ## 25-agent-summary：★ 并发度扫描
	$(MAKE) -C 25-agent-summary sweep

cache-sweep-25: ## 25-agent-summary：★ 缓存命中率扫描（缓存收益的唯一自变量是命中率）
	$(MAKE) -C 25-agent-summary cache-sweep

report-25: ## 25-agent-summary：生成周报 Markdown
	$(MAKE) -C 25-agent-summary report

mock-24: ## 24-agent-vllm：启动 mock vLLM（:8000，OpenAI 兼容）
	$(MAKE) -C 24-agent-vllm mock

health-24: ## 24-agent-vllm：检查 vLLM 服务是否可达
	$(MAKE) -C 24-agent-vllm health

bench-vllm-24: ## 24-agent-vllm：★ ★ ★ ★ 打真 vLLM 测吞吐
	$(MAKE) -C 24-agent-vllm bench-vllm

vllm-up-24: ## 24-agent-vllm：启动真 vLLM 容器（需 NVIDIA GPU）
	$(MAKE) -C 24-agent-vllm docker-up

vllm-down-24: ## 24-agent-vllm：停掉 vLLM 容器
	$(MAKE) -C 24-agent-vllm docker-down

# ── 21-agent-cache 专属 ───────────────────────────────────────────────
#
# ⚠️ Redis 用**独立容器 + 独立端口**，不复用 Milvus 那套。
#    缓存是可选组件，挂了要能降级 —— 所以它必须能独立起停。
redis-up-21: ## 启动 Redis 容器（21-agent-cache 用）
	$(MAKE) -C 21-agent-cache redis-up

redis-down-21: ## 停止 Redis 容器
	$(MAKE) -C 21-agent-cache redis-down

eval-12: ## 12-rag-summary：跑评估集出消融报告 ★
	$(MAKE) -C 12-rag-summary eval

## ── 20 性能剖析 ─────────────────────────────────────────────────────

profile-20: ## 20-agent-preformence-analysis：剖析 naive 版（三工具全跑）★
	$(MAKE) -C 20-agent-preformence-analysis profile

py-compare: ## 20-agent-preformence-analysis：三工具对比 + 优化前后差值 ★
	$(MAKE) -C 20-agent-preformence-analysis py-compare

bench-20: ## 20-agent-preformence-analysis：Rust 侧计时（多次取中位数）
	$(MAKE) -C 20-agent-preformence-analysis bench

pprof-20: ## 20-agent-preformence-analysis：Rust 侧采样出火焰图 ★
	$(MAKE) -C 20-agent-preformence-analysis pprof

setup-20: ## 20-agent-preformence-analysis：首次准备（uv venv + uv sync + 镜像）★
	$(MAKE) -C 20-agent-preformence-analysis setup

## ── Milvus 容器 ────────────────────────────────────────────────────────

milvus-up: ## 启动 Milvus standalone（etcd + minio + milvus）
	$(MAKE) -C 10-rag-milvus up

milvus-down: ## 停止 Milvus（保留数据卷）
	$(MAKE) -C 10-rag-milvus down

milvus-destroy: ## 停止并删除 Milvus 数据卷
	$(MAKE) -C 10-rag-milvus down-v

## ── 系统依赖 ────────────────────────────────────────────────────────────

check-mermaid: ## 校验所有 README 的 mermaid 图能正常渲染
	@bash scripts/check-mermaid.sh

check-toc: ## 校验所有 README 的目录锚点链接有效
	@python3 scripts/check-toc-links.py

## ── 属性测试 / 模糊测试 ───────────────────────────────────────────────
#
# 两套机制，**互补**（详见 fuzz/README.md）：
#
#   属性测试（proptest）  —— 语义层面的不变量，输入按语义生成
#   模糊测试（libFuzzer） —— 字节层面的畸形输入，任意字节序列
#
# ⚠️ 属性测试跑在**稳定工具链**上（跟主 workspace 一致），
#    所以可以进 CI；模糊测试需要 **nightly**（libFuzzer 要 `-Z sanitizer`），
#    是**可选**的深度检查。

prop: ## 跑全部属性测试（proptest，稳定工具链）
	$(CARGO) test --workspace --test properties

prop-08: ## 跑 08-rag 分词器的属性测试
	$(CARGO) test -p rag-08 --test properties

prop-13: ## 跑 13-agent-core ReAct 解析器的属性测试
	$(CARGO) test -p agent-13 --test properties

prop-14: ## 跑 14-agent-tool Schema 校验器的属性测试
	$(CARGO) test -p agent-14 --test properties

prop-15: ## 跑 15-agent-sql SQL 护栏的属性测试
	$(CARGO) test -p agent-15 --test properties

prop-21: ## 跑 21-agent-cache 缓存键的属性测试
	$(CARGO) test -p agent-21 --test properties

fuzz-build: ## 编译全部模糊测试目标（需要 nightly）
	@cargo +nightly fuzz build

fuzz-run: ## 跑全部模糊测试目标各 30 秒（需要 nightly）
	@for t in react_parser sql_guard schema_validator cache_key tokenizer; do \
		echo "==> fuzzing $$t"; \
		cargo +nightly fuzz run $$t -- -max_total_time=30 || exit 1; \
	done

fuzz-list: ## 列出全部模糊测试目标
	@cargo +nightly fuzz list

deps: ## 检查系统依赖
	@echo "==> FAISS (05/06/07/08/09 需要)"; $(MAKE) -C 05-rag deps

faiss: ## 安装 FAISS（macOS）
	brew install faiss

chroma: ## 启动 Chroma 服务（05-rag 的 chroma demo 需要）
	docker run --rm -p 8765:8000 -v chroma-data:/chroma/chroma chromadb/chroma:latest
