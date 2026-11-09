# 04-rag — RAG Part 1：加载与分割

> 目标：掌握不同格式文档（PDF、MD）的加载和文本分块策略。
> 对应 LlamaIndex 的 **Loaders** 概念，以及 Unstructured.io / MinerU / Docling 三类工具生态。

## 这一部分解决什么

RAG 流水线的头两步：

```text
文件 ──Loader──> 文本 ──Splitter──> 块
```

后面（嵌入、向量库、检索、重排、生成）在 Part 2。

**核心认知：Loader 决定你拿到什么文本，Splitter 决定模型能检索到什么。**

大多数 RAG 效果差不是模型的问题，而是这两步没做对：

- 用平铺文本抽取处理 PDF → 标题结构全丢 → 块边界全是乱的
- 按字符数硬切 → 切碎句子和表格 → 块本身语义就不完整
- 不加标题路径 → 「什么是 RAG」在「概述」和「深入」两节检索结果无法区分

## 快速开始

```bash
make load       # 加载：对比三种 loader 的输出
make split      # 分割：对比三种 splitter 的输出
make compare    # 对比：量化三种策略的差异
make pipeline   # 全流程：加载 → 分割 → 统计
make test       # 单元测试
```

> 也可直接 `cargo run -- load` 等，`make` 只是包了一层。

## 项目结构

```text
04-rag/
├── data/
│   ├── rag_guide.md        # Markdown 测试文档（多级标题、代码块）
│   ├── rag_notes.pdf       # PDF 测试文档（手写生成的规范 PDF 文本层）
│   ├── rag_notes.txt       # PDF 的源文本
│   ├── make_pdf.py         # PDF 生成脚本（无第三方依赖）
│   └── fetch_layout_model.sh  # 下载 Docling 版面模型（可选）
├── layout-demo/            # 独立 crate：Docling 完整 ML 管线演示
│   └── src/main.rs
├── src/
│   ├── loader.rs           # 加载层：DocumentLoader trait + 3 个实现
│   ├── splitter.rs         # 分割层：3 种策略 + Chunk 结构
│   └── main.rs             # 4 个 demo
└── Cargo.toml
```

---

## 架构全景

```mermaid
graph TB
    subgraph Files["输入文件"]
        F1[rag_guide.md]
        F2[rag_notes.pdf]
    end

    subgraph LoaderLayer["① 加载层 loader.rs"]
        TRAIT["DocumentLoader trait<br/>supported_extensions / load / supports"]
        ML["MarkdownLoader<br/>直接读，保留 # 标题"]
        PL["PdfTextLoader<br/>pdf-extract，快但丢结构"]
        DL["DoclingLoader<br/>结构感知 → Markdown"]
        PICK["loader_for 路径<br/>按扩展名自动选"]
    end

    subgraph Doc["LoadedDoc 统一输出"]
        LD["source / file_name / format<br/>text / page_count / loader"]
    end

    subgraph SplitterLayer["② 分割层 splitter.rs"]
        STRAT["Splitter trait<br/>name / split"]
        RS["RecursiveSplitter<br/>段落→句子→词 递归降级"]
        MS["MarkdownSplitter<br/>CommonMark 块级元素"]
        SAS["StructureAwareSplitter ★<br/>沿标题层级，块不跨标题"]
        TC["TokenCounter<br/>tiktoken cl100k_base 真实计数"]
    end

    subgraph ChunkOut["Chunk 统一输出"]
        CH["source / index / heading_path<br/>text / embedded_text<br/>token_count / char_count"]
    end

    subgraph Part2["Part 2（05-rag）"]
        EMB["Embedder → 向量"]
        VS["VectorStore → 索引"]
    end

    F1 --> ML
    F2 --> PL
    F2 --> DL
    PICK -.按扩展名.-> ML & PL & DL
    TRAIT -.实现.-> ML & PL & DL
    ML & PL & DL --> LD
    LD --> STRAT
    STRAT -.实现.-> RS & MS & SAS
    TC -.提供 token 预算.-> RS & MS & SAS
    RS & MS & SAS --> CH
    CH --> EMB --> VS

    style SAS fill:#e8f5e9
    style LD fill:#e1f5fe
    style CH fill:#e1f5fe
```

### 加载层的时序图（以 PDF 为例）

```mermaid
sequenceDiagram
    autonumber
    participant M as main.rs
    participant P as loader_for
    participant D as DoclingLoader
    participant DL as docling crate
    participant S as Splitter
    participant TC as TokenCounter

    M->>P: loader_for("rag_notes.pdf")
    P-->>M: Box dyn DocumentLoader = DoclingLoader
    M->>D: load(path)
    D->>DL: SourceDocument::from_file(path)
    DL-->>D: SourceDocument
    D->>DL: DocumentConverter::new().text_layer_only(true).convert(src)
    DL-->>D: DoclingDocument（带层级）
    D->>D: export_to_markdown()
    D-->>M: LoadedDoc { text: "# 标题\n...", loader: "docling" }
    M->>S: split(&doc, &counter)
    S->>TC: count(text) 真实 token 数
    TC-->>S: token 数
    S-->>M: Vec<Chunk>（含 heading_path / embedded_text）
```

### 三种 Splitter 的切分行为对比

```mermaid
graph TB
    subgraph Doc["文档结构"]
        D1["# 一级标题"]
        D2["段落 A"]
        D3["## 二级标题"]
        D4["段落 B"]
        D5["### 三级标题"]
        D6["段落 C"]
    end

    subgraph RS["RecursiveSplitter"]
        R1["块 0：# 一级标题 + 段落 A"]
        R2["块 1：## 二级标题 + 段落 B"]
        R3["块 2：### 三级标题 + 段落 C"]
    end

    subgraph MS["MarkdownSplitter"]
        M1["块 0：# 一级标题"]
        M2["块 1：段落 A"]
        M3["块 2：## 二级标题"]
        M4["块 3：段落 B"]
        M5["块 4：### 三级标题"]
        M6["块 5：段落 C"]
    end

    subgraph SA["StructureAwareSplitter ★"]
        S1["块 0：路径 [一级标题]"]
        S2["块 1：路径 [一级标题, 二级标题]"]
        S3["块 2：路径 [一级标题, 二级标题, 三级标题]"]
    end

    style SA fill:#e8f5e9
    style RS fill:#ffebee
```

> 关键差异：`RecursiveSplitter` 的块**可能跨标题**（语义污染）；`StructureAwareSplitter` 保证**块永不跨标题**，且标题路径覆盖率 100%。

### 知识地图：加载与分割

```mermaid
graph LR
    N1(("04-rag<br/>加载 + 分割"))
    subgraph N2["加载层 Loaders"]
        subgraph N3["DocumentLoader trait"]
            N4["supported_extensions"]
            N3 --> N4
            N5["load"]
            N3 --> N5
            N6["supports"]
            N3 --> N6
        end
        N2 --> N3
        subgraph N7["MarkdownLoader"]
            N8["直接读"]
            N7 --> N8
            N9["天然带结构"]
            N7 --> N9
        end
        N2 --> N7
        subgraph N10["PdfTextLoader"]
            N11["pdf-extract"]
            N10 --> N11
            N12["快 零依赖"]
            N10 --> N12
            N13["丢结构"]
            N10 --> N13
        end
        N2 --> N10
        subgraph N14["DoclingLoader"]
            N15["text_layer_only"]
            N14 --> N15
            N16["export_to_markdown"]
            N14 --> N16
            N17["结构感知"]
            N14 --> N17
        end
        N2 --> N14
        N18["loader_for 按扩展名"]
        N2 --> N18
        N19["load_directory 排序保证稳定"]
        N2 --> N19
    end
    N1 --> N2
    subgraph N20["分割层 Splitters"]
        N21["Splitter trait"]
        N20 --> N21
        subgraph N22["RecursiveSplitter"]
            N23["段落 换行 句子 词 字符"]
            N22 --> N23
            N24["可能跨标题"]
            N22 --> N24
        end
        N20 --> N22
        subgraph N25["MarkdownSplitter"]
            N26["CommonMark 块级元素"]
            N25 --> N26
        end
        N20 --> N25
        subgraph N27["StructureAwareSplitter"]
            N28["沿标题层级"]
            N27 --> N28
            N29["永不跨标题"]
            N27 --> N29
            N30["section 内滑窗"]
            N27 --> N30
        end
        N20 --> N27
        subgraph N31["SplitterConfig"]
            N32["chunk_size 300"]
            N31 --> N32
            N33["chunk_overlap 30"]
            N31 --> N33
            N34["add_heading_context"]
            N31 --> N34
        end
        N20 --> N31
        subgraph N35["TokenCounter"]
            N36["tiktoken cl100k_base"]
            N35 --> N36
            N37["真实 token 不是估算"]
            N35 --> N37
        end
        N20 --> N35
    end
    N1 --> N20
    subgraph N38["Chunk 结构"]
        N39["heading_path 标题路径"]
        N38 --> N39
        N40["text 正文"]
        N38 --> N40
        N41["embedded_text 带前缀 送去做嵌入"]
        N38 --> N41
        N42["token_count / char_count"]
        N38 --> N42
    end
    N1 --> N38
    subgraph N43["核心认知"]
        N44["Loader 决定拿到什么文本"]
        N43 --> N44
        N45["Splitter 决定能检索到什么"]
        N43 --> N45
        N46["标题路径是提升召回最有效的单技巧"]
        N43 --> N46
    end
    N1 --> N43
```

---

## 思维导图：加载与分割体系

```mermaid
mindmap
  root((04-rag))
    加载层 loader.rs
      DocumentLoader trait
        supported_extensions
        load
        supports
      MarkdownLoader
        保留 # 标题
      PdfTextLoader
        pdf-extract
        快但丢结构
      DoclingLoader
        结构感知
        转 Markdown
      loader_for
        按扩展名自动选
    分割层 splitter.rs
      Splitter trait
        name
        split
      RecursiveSplitter
        段落→句子→词
        递归降级
        填满 chunk_size
      MarkdownSplitter
        CommonMark 块级
      StructureAwareSplitter
        沿标题层级
        块不跨标题
        标题路径
      TokenCounter
        tiktoken cl100k_base
        真实 token 计数
    输出契约
      LoadedDoc
        source file_name format
        text page_count loader
      Chunk
        source index heading_path
        text embedded_text
        token_count char_count
    核心认知
      Loader 决定拿到什么
      Splitter 决定能检索什么
      标题路径是检索的护城河
      token 计数不能靠字符估算
      块 ID 是分块的产物
```

---

## 分割流程：从文档到块

```mermaid
flowchart TB
    F["原始文件<br/>.md / .pdf / .docx"] --> PICK{"按扩展名选 loader"}
    PICK --> ML["MarkdownLoader"]
    PICK --> PL["PdfTextLoader"]
    PICK --> DL["DoclingLoader"]

    ML --> LD["LoadedDoc<br/>text + format + loader"]
    PL --> LD
    DL --> LD

    LD --> SPL{"选 splitter"}
    SPL -->|默认| RS["RecursiveSplitter<br/>段落→换行→句子→词<br/>递归降级到字符"]
    SPL -->|markdown| MS["MarkdownSplitter<br/>按 CommonMark 块元素"]
    SPL -->|结构感知| SAS["StructureAwareSplitter ★<br/>沿标题层级切<br/>块不跨标题"]

    RS --> TC["TokenCounter<br/>tiktoken 真实计数"]
    MS --> TC
    SAS --> TC

    TC --> HP{"heading_path<br/>从字节偏移反查"}
    HP --> CH["Chunk<br/>id = {文件名}#{序号}<br/>text + embedded_text<br/>heading_path + token_count"]
    CH --> OUT["进入 05-rag<br/>向量化"]

    style SAS fill:#e8f5e9
    style CH fill:#fff3e0
    style OUT fill:#e3f2fd
```

⚠️ **注意 `heading_path` 是从字节偏移反查出来的** ——
不管用哪个 splitter，块在文档中的位置都能映射回它所属的标题层级。
这是后面 09-rag-eval 能做「章节 → 块 ID」映射的基础。

---

## 代码调用详细流程图（函数级）

> 节点写的是**真实函数名**，边上标的是**调用点行号**。全部对着源码。

### ★ 加载链路：目录 → 文档

```mermaid
flowchart TD
    MAIN["main.rs:42<br/>main()"] --> LD["main.rs:74<br/>run_load_demo(dir)"]
    MAIN --> SP["main.rs:132<br/>run_split_demo(dir)"]
    MAIN --> PIPE["main.rs:261<br/>run_pipeline_demo(dir)"]

    LD --> LODIR["loader.rs:330<br/>load_directory(dir)"]
    LODIR --> SORT["★ 排序保证加载顺序稳定<br/>（否则每次 run 结果可能不同）"]
    SORT --> PICK["loader.rs:308<br/>loader_for(path)"]
    PICK --> ML["loader.rs:141<br/>MarkdownLoader"]
    PICK --> PL["loader.rs:187<br/>PdfTextLoader"]
    PICK --> DL["loader.rs:253<br/>DoclingLoader"]
    ML --> SUP["loader.rs:118<br/>DocumentLoader::supports(path)"]
    PL --> SUP
    DL --> SUP
    SUP --> LOAD["loader.rs:118<br/>DocumentLoader::load(path)"]
    LOAD --> DOC["LoadedDoc<br/>text + format + loader"]

    PIPE --> LODIR
    PIPE --> SPLIT["splitter.rs:350<br/>StructureAwareSplitter::split(doc, counter)"]
    SP --> LODIR
    SP --> SPLIT

    style LODIR fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style SORT fill:#ffe6e6,stroke:#d94a4a,stroke-width:2px
    style PICK fill:#fff4e6,stroke:#e8a33d
```

### ★ 分割链路：文档 → 块

**`heading_path` 是从字节偏移反查出来的** —— 不管用哪个 splitter，块的位置都能映射回标题层级。

```mermaid
flowchart TD
    DOC["LoadedDoc"] --> SPL{"选 splitter"}
    SPL -->|"默认"| RS["splitter.rs:232<br/>RecursiveSplitter::split()"]
    SPL -->|"markdown"| MS["splitter.rs:302<br/>MarkdownSplitter::split()"]
    SPL -->|"结构感知"| SAS["★ splitter.rs:350<br/>StructureAwareSplitter::split()"]

    RS --> PARSE["递归降级：<br/>段落 → 换行 → 句子 → 词 → 字符"]
    MS --> MCP["按 CommonMark 块元素切"]
    SAS --> SECT["docling::chunker::parse_sections_with_stack()"]

    SECT --> FITSIZE{"section_tokens <= chunk_size?"}
    FITSIZE -->|"是"| BUILD["splitter.rs:157<br/>build_chunk(doc, index, heading_path, body, counter, add_heading_context)"]
    FITSIZE -->|"否"| SPLIT2["再切分"]

    PARSE --> BUILD
    MCP --> BUILD

    BUILD --> TC["splitter.rs:119<br/>TokenCounter::count()<br/>tiktoken 真实计数"]
    TC --> HP["splitter.rs:477<br/>heading_path_at(text, byte_offset)<br/>★ 从字节偏移反查标题层级"]
    HP --> CH["Chunk<br/>id = {文件名}#{序号}<br/>text + embedded_text<br/>heading_path + token_count"]
    CH --> OUT["进入 05-rag 向量化"]

    style SAS fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    style HP fill:#ffe6e6,stroke:#d94a4a,stroke-width:2px
    style TC fill:#fff4e6,stroke:#e8a33d
```

### 三个 splitter 的差异：切点选在哪

```mermaid
flowchart TB
    subgraph RS["RecursiveSplitter（默认）"]
        R1["段落"] --> R2["换行"] --> R3["句子"] --> R4["词"] --> R5["字符"]
        R5 -.->|"都切不动了<br/>只能硬切"| NOTE["⚠️ 切点落在 token 预算上<br/>不是字符数上"]
    end

    subgraph MS["MarkdownSplitter"]
        M1["按 CommonMark 块元素<br/>（标题/列表/代码块）"]
    end

    subgraph SAS["StructureAwareSplitter"]
        S1["沿标题层级切<br/>★ 块不跨标题"]
    end

    style R5 fill:#fff4e6,stroke:#e8a33d
    style S1 fill:#e8f5e9,stroke:#4caf50
    style NOTE fill:#ffe6e6,stroke:#d94a4a
```

### `heading_path` 为什么重要

**这是 09-rag-eval 能做「章节 → 块 ID」映射的唯一依据。**

```mermaid
flowchart LR
    OFFSET["块在文档中的字节偏移"] --> LOOKUP["splitter.rs:477<br/>heading_path_at(text, byte_offset)"]
    LOOKUP --> PATH["heading_path<br/>例：RAG / 分块"]
    PATH --> CHUNK["写进 Chunk.heading_path"]
    CHUNK --> EVAL["09-rag-eval<br/>ChunkIndex：章节 → 块 ID 映射<br/>→ 支撑 ContextRecallById 指标"]

    style LOOKUP fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style EVAL fill:#e8f5e9,stroke:#4caf50
```

## 一、加载层（Loaders）

### 抽象

```rust
pub trait DocumentLoader: Send + Sync {
    fn supported_extensions(&self) -> &[&str];
    fn load(&self, path: &Path) -> Result<LoadedDoc, LoaderError>;
}
```

存在这个 trait 是为了：**上层流水线只依赖 `Box<dyn DocumentLoader>`**，
于是「本地 PDF loader」和「远程 API loader」（Unstructured / MinerU）可以互换。

`LoadedDoc` 是统一的输出格式：正文 + 元数据（来源、格式、页数、loader 名）。
后续所有 splitter 只认这个结构。

### 三种实现

| loader | 适用 | 结构保留 | 依赖 | 速度 |
|---|---|---|---|---|
| `MarkdownLoader` | `.md` / `.txt` | ✅ 原生 | 无 | 即时 |
| `PdfTextLoader` | `.pdf` | ❌ **全丢** | `pdf-extract` | 快 |
| `DoclingLoader` | `.pdf` / `.html` | ⚠️ 见下 | `docling` | 中等 |

#### `PdfTextLoader` —— 快，但丢结构

纯 Rust 的 PDF 文本层抽取。输出是**扁平的行**：

```text
Retrieval-Augmented Generation Engineering Notes
Section 1. Why chunking matters
Retrieval quality is dominated by chunk granularity. Chunks that are too large...
```

标题和正文完全混在一起，没有任何标记。分块器只能按字符数硬切。
**这就是为什么需要结构感知的 loader。**

#### `DoclingLoader` —— 结构感知

Docling 是 Python 版 docling 的 Rust 移植。它把 PDF 解析成带层级的
中间表示（`DoclingDocument`），再导出成 Markdown。

**⚠️ 真实限制（重要）**：Rust 版 docling 的 PDF 后端分两档：

| | `pdf-text`（本项目默认） | `pdf`（完整 ML 管线） |
|---|---|---|
| 文字内容 | ✅ | ✅ |
| 标题 → `##` | ❌ | ✅ |
| 表格结构 | ❌ | ✅ |
| 扫描件 OCR | ❌ | ✅（需另下 OCR 模型） |
| 额外依赖 | 无 | ONNX runtime |
| 运行期模型 | 无 | `.models/layout_heron_int8.onnx`（~68MB） |

主 crate 默认用 `pdf-text`（零额外依赖，`cargo run` 就能跑）。
完整 ML 管线放在独立 crate `layout-demo/` 里演示 —— 原因见「已知问题」。

`cargo run -- load` 的实际输出对比：

```text
[PdfTextLoader]  [pdf-extract] rag_notes.pdf (1785 字符, 38 行), 1 页
[DoclingLoader]  [docling]     rag_notes.pdf (1784 字符, 23 行)
```

两者文字内容一致，差别在结构 —— Docling 版少了 PDF 固有的空行和换行残留。

#### 完整 ML 管线：`layout-demo/`

```bash
# 1. 下载版面模型（~68MB）
./04-rag/data/fetch_layout_model.sh

# 2. 运行（从仓库根目录跑，模型放在 ./.models/）
cargo run --manifest-path 04-rag/layout-demo/Cargo.toml
```

实测输出：

```text
--- 识别到的标题 ---
  ## Retrieval-Augmented Generation Engineering Notes
  ## Section 1. Why chunking matters
  ## Section 2. Chunk size guidance
  ## Section 3. Structure aware chunking
  ## Section 4. Metadata and citations
  ## Section 5. Evaluation pitfalls
```

**这就是结构感知分块能在 PDF 上生效的前提** —— 标题没被还原成 `##`，
分块器就无从知道章节边界在哪。

### 工具生态的对应关系

题目提到的三个工具，在 Rust 生态里的对应：

| 工具 | 形态 | Rust 对应 | 本项目 |
|---|---|---|---|
| **Docling** | 本地 Python 库 / 服务 | `docling` crate（原生 Rust 移植） | ✅ 已集成 |
| **Unstructured.io** | 云端 SaaS API | `unstructured-client` crate（HTTP 客户端） | 见下 |
| **MinerU** | 云端 API / 本地 CLI | `mineru` crate（Rust 客户端） | 见下 |

**Unstructured.io / MinerU 是远程服务**，需要 API key 和网络。
本项目选择 Docling 做本地集成，因为：

1. 无需 API key，`cargo run` 就能跑
2. 无网络依赖，教育场景可复现
3. 同样的目标（结构感知），本地就能达到

要接入远程服务，只要实现 `DocumentLoader` trait 即可 —— 架构上完全兼容：

```rust
// 伪代码：Unstructured 适配器
impl DocumentLoader for UnstructuredLoader {
    fn load(&self, path: &Path) -> Result<LoadedDoc, LoaderError> {
        let elements = self.client.partition(path)?;   // 远程 API
        // elements 里每个元素都带 category（Title / NarrativeText / Table）
        // category == "Title" 的可以映射成 "# " 前缀 → 复用结构感知分块
        Ok(LoadedDoc { text: elements_to_markdown(elements), .. })
    }
}
```

关键洞察：**Unstructured 的 `Element.category` 提供了标题标记，
映射成 Markdown 之后，同一套结构感知分块器就能直接复用。**
这就是为什么 loader 层要输出统一格式。

## 二、分割层（Splitters）

### 三种策略

| 策略 | 切分依据 | 块会跨标题吗 | 适用 |
|---|---|---|---|
| `RecursiveSplitter` | 递归降级：段落→行→句→词 | ⚠️ 会 | 无结构文本 |
| `MarkdownSplitter` | CommonMark 块级元素 | ⚠️ 会 | Markdown |
| `StructureAwareSplitter` | 标题层级（section 边界） | ❌ **绝不** | 有结构的文档 |

### `RecursiveSplitter`

对应 LlamaIndex `SentenceSplitter` / LangChain `RecursiveCharacterTextSplitter`。

切分点优先级（从粗到细），每一级都尽量多塞，填满 token 预算才停：

```text
段落(\n\n) → 换行(\n) → 句子 → 词 → 字形簇 → 字符
```

**关键实现细节**：容量单位用的是**真实 BPE tokenizer**，不是「字符数 ÷ 4」估算。

```rust
let config = text_splitter::ChunkConfig::new(chunk_size)
    .with_sizer(counter.bpe())      // ← CoreBPE 实现了 ChunkSizer trait
    .with_overlap(chunk_overlap);
```

为什么重要：英文约 4 字符/token，中文约 1.5 字符/token，代码更碎。
用估算会让中文块的实际 token 数超标 2 倍以上，直接撑爆上下文预算。

（`with_overlap` 在 overlap ≥ capacity 时会返回 `Err` —— 代码里做了降级，
退回「无重叠」而不是 panic。重叠只是优化，不该让整条流水线挂掉。）

### `StructureAwareSplitter` —— 推荐

1. 用 docling 的 section 解析器把 Markdown 拆成「标题路径 + 正文词」
2. 沿 section 边界切，**块永不跨标题**
3. section 内部超长时，再按 token 预算滑动窗口切（带重叠）

**与递归分块的关键差异**：递归分块可能把「语义分块」这一节的最后一段
和「文档结构感知分块」这一节的开头拼在一起 —— 语义完全混乱。
结构感知分块不会。

### 标题路径 —— 提升召回最有效的单个技巧

每个块记录它所在的完整标题路径：

```rust
pub struct Chunk {
    pub heading_path: Vec<String>,   // ["RAG 检索增强生成", "文本分块策略", "语义分块"]
    pub text: String,                // 正文
    pub embedded_text: String,       // ← 送去做嵌入的其实是这个
    pub token_count: usize,
    // ...
}
```

`embedded_text` = 标题路径 + 空行 + 正文：

```text
# RAG 检索增强生成 > 为什么需要 RAG

大语言模型存在几个根本限制。首先是知识截止日期，模型的训练数据有截止时间……
```

**为什么要这样**：语义完全相同的句子，在不同章节里含义不同。
把层级语境注入向量，检索才能区分。

### 实测对比

`cargo run -- compare` 的输出（`rag_guide.md`，chunk_size=120 token）：

```text
策略                     块数   平均 token   最大 token      标题路径覆盖率
------------------------------------------------------------
recursive              17        111        148          94%
markdown               19         99        143          94%
structure-aware        18        100        139         100%
```

- **标题路径覆盖率**：块里有多少带上了「所在章节」信息
- structure-aware 是 100%（按标题切的，块必然在某个标题下）
- recursive / markdown 是 94%（有的块跨了标题，只能归属到其中一个）

### 重叠（overlap）

相邻块保留一部分重叠，避免关键句正好落在切分线上。

- 重叠太小 → 关键句被切断
- 重叠太大 → 成本翻倍，还引入重复内容干扰检索
- 经验值：块大小的 10%~20%

本项目默认 300 token / 30 token 重叠。

## 三、几个容易踩的坑

**1. 用字符数估算 token 数**
中文和代码里误差极大。必须用真实 tokenizer。

**2. 忘记加标题路径**
语义相同的句子在不同章节含义完全不同。

**3. 表格被拍平成一行**
表格结构丢失后，检索基本无效。要保留表格需要 Docling 的完整 `pdf` feature
或 Unstructured 的 `Table` element 类型。

**4. 只做召回不做重排**
向量检索召回 20 条，再用交叉编码器重排取 top 3 —— 效果提升明显。（Part 2）

**5. 测试集泄漏**
用测试问题评估时，如果文档内容本身包含了答案，指标会虚高。（Part 2）

## 四、已知问题

### 完整 ML 管线（Docling 版面分析）不能放进主 workspace

`layout-demo/` 必须是一个**脱离 workspace 的独立 crate**，原因：

```text
docling (完整 ML 管线)  →  ort ^2.0.0-rc.14
rig → fastembed 4.9.1   →  ort =2.0.0-rc.9   ← 精确锁定
```

两者的 `ort` 版本 semver 不兼容。Cargo 的依赖解析只能二选一，
放在同一个 workspace 里会让**整个 workspace 都无法解析**（连 `cargo build` 都过不去）。

实测过几种绕法，都不可行：

| 尝试 | 结果 |
|---|---|
| 升级 `ort` 到 rc.15 | ❌ fastembed 锁死 `=2.0.0-rc.9`，不让升 |
| 换 fastembed 版本 | ❌ rig-fastembed 锁 `^4.5`，最新的 5.x/7.x 也用 rc.9 之前的 ort |
| 降级 docling 到 2.20.0 | ❌ 依然要 `ort ^2.0.0-rc.13` |
| 去掉 rig 的 fastembed | ❌ fastembed 是 `rig` 的可选依赖，但 `rig-agent` 默认链仍会带上 |

所以 `layout-demo/Cargo.toml` 里写了空的 `[workspace]` 表，
让它脱离父 workspace 独立解析依赖。

### PDF 测试文档是自己生成的

`data/rag_notes.pdf` 由 `data/make_pdf.py` 手写 PDF 语法生成，不用第三方库。

原因：macOS 自带的 `cupsfilter` 也能生成 PDF，但它把每行文字拆成多个独立的
show-text 操作，行与行的 y 坐标差很小，PDF 文本层后端在还原阅读顺序时
会把相邻行**交错**在一起 —— 看起来像 loader 的 bug，其实是 PDF 本身的
编码方式决定的。手写 PDF 保证每行一个完整的 `Tj` 操作、坐标严格递增，
阅读顺序可还原。

## 五、依赖清单

```toml
docling = { version = "2.67", default-features = false, features = ["chunking", "pdf-text"] }
pdf-extract = "0.12"
pulldown-cmark = "0.13"
text-splitter = { version = "0.33", features = ["markdown", "tiktoken-rs"] }
tiktoken-rs = "0.12"
```

`text-splitter` 的 `tiktoken-rs` feature 很关键 —— 它让 `CoreBPE` 实现了
`ChunkSizer` trait，从而能做**精确的 token 级切分**，而不是字符数近似。

## 六、Part 2 预告

接下来要做的：

- 嵌入模型（DeepSeek / OpenAI embedding API）
- 向量存储（内存版 / LanceDB / Qdrant）
- 相似度检索 + 交叉编码器重排
- 把块拼成 prompt，调用 LLM 生成带引用的答案

## 日志

全仓库统一走 `00-logging`（`agent-logging`）。

```bash
RUST_LOG=info cargo run -p rag-04 -- pipeline
RUST_LOG=debug,rag_04=trace cargo run -p rag-04 -- pipeline
```

| 事件 | 级别 | 关键字段 |
|---|---|---|
| 加载文档 | `INFO` | `loader` / `file_count` / `char_count` |
| 切分 | `INFO` | `splitter` / `chunk_count` / `elapsed_ms` |
| 单个 Loader / Splitter 的内部过程 | `DEBUG` | `file` / `chunk_size` |
| 加载失败 / 切分失败 | `ERROR` | `file` / `error` |

---
## 技术参考速查

### 三种 Loader 对照表

| | MarkdownLoader | PdfTextLoader | DoclingLoader |
|---|---|---|---|
| 原理 | 直接读文本 | `pdf-extract` 抽纯文本 | 结构感知 → 转 Markdown |
| 保留标题 | ✅ | ❌ | ✅ |
| 保留表格结构 | ⚠️ 部分 | ❌ | ✅ |
| 保留页码 | ❌ | ✅ | ✅ |
| 速度 | 最快 | 快 | 慢 |
| 依赖 | 无 | `pdf-extract` | `docling` |

⚠️ **PdfTextLoader 丢结构**是它的固有限制 —— PDF 里没有「标题」这个概念，
只有字号和字重。`pdf-extract` 只给你一坨文本流，
**所以 PDF 场景应该用 DoclingLoader**，否则标题路径整个丢失。

### 三种 Splitter 对照表

| | RecursiveSplitter | MarkdownSplitter | StructureAwareSplitter |
|---|---|---|---|
| 切分依据 | 段落→换行→句子→词→字符 | CommonMark 块级元素 | **标题层级** |
| 块会跨标题吗 | ⚠️ 可能 | ⚠️ 可能 | ❌ **绝不** |
| `heading_path` | 从偏移反查 | 从偏移反查 | 直接来自 section |
| 语义完整性 | 一般 | 一般 | **最好** |
| 块大小均匀度 | **最好** | 一般 | 可能不均（大章节） |

### `text-splitter` 的切分点优先级

```text
段落 (\n\n)  →  换行 (\n)  →  句子  →  词  →  字形簇  →  字符
   粗                                                        细
```

每一级都尽量多塞，**填满 `chunk_size` 个 token 才停**。
只有当前级别塞不下时，才降级到下一级。

⚠️ 所以「段落」是最优先的切分点 —— 段落边界天然是语义边界。
这解释了为什么 `RecursiveSplitter` 虽简单，效果却不差。

### Token 计数：为什么必须用真的 tokenizer

```rust
// ❌ 常见错误：字符数 / 4 估算 token
let est_tokens = text.len() / 4;

// ✅ 正确：用真实的 BPE tokenizer
let count = tiktoken::get_encoding("cl100k_base")?.encode(text).len();
```

⚠️ **中英文差异极大**：中文一个字符 ≈ 1~2 个 token，
而英文一个字符 ≈ 0.25 个 token。用「字符数 / 4」估算中文会**严重低估**，
导致块实际远超预算，模型上下文被撑爆。

本项目把 `tiktoken` 的 `cl100k_base` 编码器作为**容量单位**喂给 splitter ——
切分点在 **token 预算**上选，而不是字符数上。

### `Chunk` 字段的用途对照

| 字段 | 谁在用 | 用途 |
|---|---|---|
| `source` | 05/06 | 过滤（只搜某个文件） |
| `index` | 05/06 | 生成块 ID `{文件名}#{序号}` |
| `heading_path` | 05~09 | **给检索加上下文**、给评估集做章节映射 |
| `text` | 05 | 送去做嵌入 |
| `embedded_text` | 05 | **带标题前缀的**送去做嵌入（见下） |
| `token_count` | 04/05 | 预算控制 |
| `char_count` | 04 | 统计展示 |

### `embedded_text` vs `text`：为什么要两个

```text
text:          "BM25 由 IDF、词频饱和、长度归一化组成。"
embedded_text: "RAG > 混合检索 > BM25 打分算法\n\nBM25 由 IDF、词频饱和..."
               └────────────── 标题路径前缀 ──────────────┘
```

因为**脱离上下文的块语义不完整** ——「长度归一化」这四个字，
单独看不知道在讲什么；加上标题路径后，嵌入向量才能把它编码到
「BM25 打分算法」这个语义空间里。

⚠️ 注意：**只有 `embedded_text` 送去嵌入**，`text` 是原文，
用于展示和生成答案。混用会导致检索到的内容看起来「莫名其妙」。

### 结构感知分割的权衡

```text
StructureAwareSplitter 的优点：
  ✅ 块不跨标题 → 语义完整
  ✅ heading_path 天然准确

代价：
  ⚠️ 块大小可能不均（一个大章节 → 一个巨大的块）
  ⚠️ 需要超大 chunk_size 才能容纳大章节
```

本项目的做法是：**章节超预算时才细分**（按句子降级），
保证「不跨标题」这个硬约束优先于「块大小均匀」这个软目标。

### 块 ID 的脆弱性（09-rag-eval 的教训）

```text
块 ID = "{文件名}#{块序号}"
                  ▲
                  └── 由分块配置决定！
```

改一下 `chunk_size`，同一个段落的块序号就变了 ——
**所有基于块 ID 的标注全部失效，召回率恒为 0，而且没有任何报错**。

这就是 09-rag-eval 改用 `relevant_sections`（章节标题）标注的原因：
**标题是文档的事实，块 ID 是分块的产物**。

---

## 七、Makefile

```bash
make help        # 查看所有目标
make load        # 对比三种 loader 的输出
make split       # 对比三种 splitter 的输出
make compare     # 量化不同策略的差异
make pipeline    # 全流程：加载 → 分割 → 统计
make all         # 全部 demo
make build       # 编译
make test        # 跑单元测试
make lint        # clippy 零警告
make fmt         # 格式化
make clean       # 清理产物
```
