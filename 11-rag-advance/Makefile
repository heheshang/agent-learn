.PHONY: help build demo compare chunk serve test lint fmt clean sample-pdf sample-image

BIN := rag-11
DATA_DIR := data
ASSETS_DIR := assets

# ── 常用命令 ────────────────────────────────────────────────
help: ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 编译
	cargo build -p $(BIN)

demo: ## 解析示例 PDF，展示结构化结果
	cargo run -p $(BIN) -- demo

compare: ## ★ 对照实验：几何还原 vs 压平
	cargo run -p $(BIN) -- compare

chunk: ## 解析并分块
	cargo run -p $(BIN) -- chunk

serve: ## 启动 HTTP 服务（127.0.0.1:3040）
	cargo run -p $(BIN) -- serve

# ── 质量 ────────────────────────────────────────────────────
test: ## 跑全部测试
	cargo test -p $(BIN)

lint: ## clippy 零警告检查
	cargo clippy -p $(BIN) --all-targets -- -D warnings

fmt: ## 格式化
	cargo fmt -p $(BIN)

check: fmt lint test ## 格式化 + lint + 测试

# ── 样例数据 ────────────────────────────────────────────────
sample-pdf: ## 造一份含表格 + 矢量图的 PDF
	python3 $(DATA_DIR)/make_complex_pdf.py $(DATA_DIR)/complex_layout.pdf

sample-image: ## 造一份含内嵌位图的 PDF
	python3 $(DATA_DIR)/make_pdf_with_image.py $(DATA_DIR)/with_image.pdf

clean: ## 清理构建产物和生成的图片
	cargo clean -p $(BIN)
	rm -rf $(ASSETS_DIR)
