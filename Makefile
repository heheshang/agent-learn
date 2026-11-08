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
PKGS  := axum-01 rig-02 genai-03 rag-04 rag-05 rag-06 rag-07 rag-08 rag-09 rag-10 rag-11 rag-12

.DEFAULT_GOAL := help
.PHONY: help build run test check fmt fmt-check lint clean \
        01 02 03 04 05 06 07 08 09 10 11 12 \
        axum rig genai rag \
        serve-06 serve-07 serve-08 serve-09 serve-10 serve-11 serve-12 \
        demo-04 demo-05 demo-06 demo-07 demo-08 demo-09 demo-10 demo-11 demo-12 \
        compare-07 compare-08 compare-09 compare-10 compare-11 compare-12 \
        eval-12 \
        milvus-up milvus-down milvus-destroy \
        check-mermaid \
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

axum: 01 ## 别名
rig: 02 ## 别名
genai: 03 ## 别名
rag: 04 ## 别名

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

deps: ## 检查系统依赖
	@echo "==> FAISS (05/06/07/08/09 需要)"; $(MAKE) -C 05-rag deps

faiss: ## 安装 FAISS（macOS）
	brew install faiss

chroma: ## 启动 Chroma 服务（05-rag 的 chroma demo 需要）
	docker run --rm -p 8765:8000 -v chroma-data:/chroma/chroma chromadb/chroma:latest
