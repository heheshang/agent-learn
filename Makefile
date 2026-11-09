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
PKGS  := axum-01 rig-02 genai-03 rag-04 rag-05 rag-06 rag-07 rag-08 rag-09 rag-10 rag-11 rag-12 agent-13 agent-14 agent-15 agent-16 agent-17 agent-18

.DEFAULT_GOAL := help
.PHONY: help build run test check fmt fmt-check lint clean logs \
        01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 \
        axum rig genai rag agent \
        serve-06 serve-07 serve-08 serve-09 serve-10 serve-11 serve-12 serve-13 serve-14 serve-15 serve-16 serve-17 serve-18 \
        demo-04 demo-05 demo-06 demo-07 demo-08 demo-09 demo-10 demo-11 demo-12 demo-13 demo-14 demo-15 demo-16 demo-17 demo-18 \
        compare-07 compare-08 compare-09 compare-10 compare-11 compare-12 compare-13 compare-14 compare-16 compare-17 compare-18 \
        call-14 guard-15 \
        eval-12 \
        milvus-up milvus-down milvus-destroy \
        check-mermaid check-toc \
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

eval-12: ## 12-rag-summary：跑评估集出消融报告 ★
	$(MAKE) -C 12-rag-summary eval

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

deps: ## 检查系统依赖
	@echo "==> FAISS (05/06/07/08/09 需要)"; $(MAKE) -C 05-rag deps

faiss: ## 安装 FAISS（macOS）
	brew install faiss

chroma: ## 启动 Chroma 服务（05-rag 的 chroma demo 需要）
	docker run --rm -p 8765:8000 -v chroma-data:/chroma/chroma chromadb/chroma:latest
