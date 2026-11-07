# agent-learn

从零手撕 AI Agent 与 RAG 全链路的 Rust 学习项目。**不依赖 LangChain / LlamaIndex
这类框架** —— 每一层都自己实现，把「框架帮你藏起来的东西」摊开来看。

```text
01-axum   HTTP 服务基础（路由 / 处理器 / 状态 / 错误）
02-rig    Rust 的 AI 编排框架（rig）
03-genai  多厂商 LLM 调用 + 工具 + Agent 循环（genai）
04-rag    RAG Part 1：文档加载 + 文本分块
05-rag    RAG Part 2：向量化 + 向量存储
06-rag    RAG Part 3：完整 Naive RAG 系统 + HTTP 问答 API + 自主 Agent
07-query  RAG Part 4：查询改写（HyDE / Multi-Query / Rewrite / Decompose）+ RRF 融合
08-rag    RAG Part 5：混合检索（BM25 + 向量）+ 重排（cross-encoder / Cohere）
```

## 全局架构全景

```mermaid
graph TB
    subgraph L0["基础设施层"]
        AX["01-axum<br/>HTTP 路由 / 状态 / 错误"]
        RG["02-rig<br/>Agent 编排框架"]
        GN["03-genai<br/>多厂商 LLM 客户端"]
    end

    subgraph L1["RAG Part 1：加载 + 分割"]
        R4["04-rag<br/>Loader → Splitter → Chunk"]
    end

    subgraph L2["RAG Part 2：向量化 + 存储"]
        R5["05-rag<br/>Embedder → VectorStore → Index"]
    end

    subgraph L3["RAG Part 3：Naive RAG"]
        R6["06-rag<br/>检索 → Prompt → 生成 → API"]
    end

    subgraph L4["RAG Part 4：查询改写"]
        R7["07-query<br/>改写 → 多路检索 → RRF 融合"]
    end

    subgraph L5["RAG Part 5：混合检索 + 重排"]
        R8["08-rag<br/>BM25 + 向量 → RRF → cross-encoder 重排"]
    end

    R4 --> R5 --> R6 --> R7 --> R8
    AX -.HTTP 层复用到.-> R6 & R7 & R8
    GN -.LLM 客户端.-> R6 & R7 & R8
    RG -.Agent 模式.-> R6

    style R6 fill:#e8f5e9
    style R8 fill:#f3e5f5
```

### RAG 主线数据流（离线索引 + 在线查询）

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant API as HTTP API
    participant TR as 改写层 07-query
    participant HY as 混合检索 08-rag
    participant RR as 重排层 08-rag
    participant LLM as LLM
    participant SP as 切分 04-rag
    participant EM as 嵌入 05-rag
    participant IX as 索引 FAISS

    rect rgb(255, 243, 224)
        Note over SP,IX: 离线：索引阶段
        SP->>EM: 块 → 向量
        EM->>IX: 写入向量索引
    end

    rect rgb(232, 245, 233)
        Note over U,LLM: 在线：查询阶段
        U->>API: POST /api/ask
        API->>TR: 改写问题（HyDE / Multi-Query / ...）
        TR-->>API: N 条检索式
        API->>HY: 并发双路召回（BM25 + 向量）
        HY-->>API: RRF 融合后的候选集
        API->>RR: cross-encoder 精排 top-n
        RR-->>API: top-n 命中块
        API->>LLM: Prompt + 上下文 + 引用编号
        LLM-->>API: 带引用的答案
        API-->>U: JSON answer + citations
    end
```

### 增强手段的演进阶梯

```mermaid
graph TB
    subgraph S0["06-rag · Naive 基线"]
        A1[原样检索]
        A2[只有向量]
        A3[无重排]
        A4[无拒答阈值]
    end
    subgraph S1["07-query · 解决 问法"]
        B1[查询改写<br/>HyDE / Multi-Query / Rewrite / Decompose]
        B2[多路检索 + RRF 融合]
    end
    subgraph S2["08-rag · 解决 检索精度"]
        C1[BM25 关键词路]
        C2[混合 + RRF]
        C3[cross-encoder 重排]
        C4[三个对照组可量化]
    end

    S0 -->|人话 ≠ 文档语言| S1
    S1 -->|向量对精确词弱| S2

    style S0 fill:#ffebee
    style S1 fill:#e8f5e9
    style S2 fill:#f3e5f5
```

### 思维导图：整个仓库的知识地图

```mermaid
mindmap
  root((agent-learn<br/>手撕 AI Agent 与 RAG))
    基础设施
      01-axum
        路由
        提取器
        共享状态
        统一错误
      02-rig
        Model 调用
        AgentBuilder
        Tool 四问
        ToolContext
        HttpMiddleware
      03-genai
        genai 模型层
        rust-genai-agent
        事件流
        钩子
    RAG Part 1
      04-rag
        Loader
          Markdown
          PdfText
          Docling
        Splitter
          Recursive
          Markdown
          StructureAware
        Chunk 标题路径
    RAG Part 2
      05-rag
        Embedder
          fastembed
          L2 归一化
          前缀约定
        VectorStore
          FAISS Flat IVF HNSW
          Chroma
        持久化
    RAG Part 3
      06-rag
        Retriever
        Prompt 组装
        Generator
        Agent
        HTTP API
    RAG Part 4
      07-query
        HyDE
        Multi-Query
        Rewrite
        Decompose
        RRF / MaxScore
        降级链
    RAG Part 5
      08-rag
        tokenize CJK bigram
        BM25
        混合检索
        cross-encoder 重排
        三对照组
    核心认知
      Loader 决定拿到什么
      Splitter 决定能检索什么
      Embedding 决定什么算相关
      VectorStore 决定候选池
      改写是翻译层
      重排是精度来源
      收益靠差值证明
```

## RAG 主线

```text
04-rag                        05-rag                       06-rag
加载 → 分块          ─────→  向量化 → 存储        ─────→  检索 → 生成 → API
 Loader   Splitter             Embedder  VectorStore        Retriever  Generator
                                                         │
                                                         ▼
                                                     07-query
                                              查询改写 → 多路检索 → 融合
                                              Transformer  MultiQuery  RRF
                                                         │
                                                         ▼
                                                      08-rag
                                              双路检索 → RRF 融合 → 重排
                                              BM25  Hybrid  Reranker
```

| 模块 | 解决的问题 | 核心认知 |
|---|---|---|
| [04-rag](04-rag/) | 文档 → 块 | **Loader 决定你拿到什么文本，Splitter 决定模型能检索到什么** |
| [05-rag](05-rag/) | 块 → 可检索的索引 | **Embedding 决定「什么算相关」，VectorStore 决定「在多少候选里找」** |
| [06-rag](06-rag/) | 问题 → 带引用的答案 | **Naive RAG 的链路本身就是基线，每个增强手段都要用实测差值证明它值不值** |
| [07-query](07-query/) | 问法 → 检索式 | **人话和文档不是同一种语言，改写就是中间的翻译层；收益要用对比矩阵量化** |
| [08-rag](08-rag/) | 召回不准 | **向量和关键词的失败模式正好互补；召回只是粗筛，重排才是精度的来源** |

## 快速开始

```bash
# 依赖（macOS）
brew install faiss          # FAISS 的 C++ 库，05/06/07/08-rag 需要

# Makefile 统一入口（根目录）
make help                   # 查看所有目标
make build                  # 编译全部
make test                   # 跑全部测试
make lint                   # clippy 零警告
make 04                     # 04-rag：加载 + 分割
make 05                     # 05-rag：向量化 + 存储
make demo-06                # 06-rag：端到端问答
make compare-07             # 07-query：多策略改写对比 ★
make compare-08             # 08-rag：多模式检索对比 ★
make serve-08               # 启动 08-rag 服务（:3020）

# 各模块独立运行
cargo run -p axum-01                    # HTTP 服务
cargo run -p rag-04 -- pipeline         # 加载 + 分割
cargo run -p rag-05 -- search           # 向量检索
export DEEPSEEK_API_KEY=sk-...          # 06/07/08-rag 的生成环节需要
cargo run -p rag-06 -- demo             # 完整 RAG 问答
cargo run -p rag-07 -- compare          # 多策略对比（改写收益量化）
cargo run -p rag-08 -- compare          # 多模式对比（混合检索 + 重排收益量化）

# 测试
cargo test --workspace
```

每个子项目都有自己的 `Makefile`，也可以单独进入使用：

```bash
cd 08-rag && make help
```

## 依赖版本管理

所有成员 crate 的依赖版本在根 `Cargo.toml` 的 `[workspace.dependencies]` 里统一定义，
成员用 `dep = { workspace = true }` 引用 —— 版本只出现一处，避免模块之间版本漂移。

⚠️ 两个已知的版本陷阱（详见各模块 README）：

- `fastembed` 锁 4.x —— 与 `rig` 依赖的 `ort` 精确版本冲突
- `genai` 指向 AgentPrism fork —— 和上游 `genai 0.7` 的类型不互通
