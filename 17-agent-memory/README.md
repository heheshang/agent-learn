# 17-agent-memory —— 手撕 Agent Memory（消息状态 + thread_id + checkpointer）

> **主题**：LangChain [Short-term Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) + [Mem0](https://github.com/mem0ai/mem0)
> **工具**：[ReMe / MemoryScope](https://github.com/agentscope-ai/ReMe) ——「Memory as File, File as Memory」
> **后端**：DeepSeek API（`deepseek-flash`）
> **端口**：3100

这个项目回答一个问题：**Agent 到底怎么记住用户？**

13/14/15/16 四个项目已经把「Agent 怎么**做事**」讲完了（ReAct 循环、工具开发、SQL Agent、Function Calling）。
但它们都缺一块：**每一轮对话之间，那个 Agent 什么都不记得**。

本项目补上这块 —— 并且不是简单地「把消息存起来」，而是分成**两级**：

| | 短期记忆 | 长期记忆 |
|---|---|---|
| 作用域 | **一个线程内** | **跨线程、跨会话** |
| 内容 | 原始对话消息 | **提炼出的事实**（「用户名字是 Bob」） |
| 存取 | 全量读入上下文 | **按需召回**（关键词检索） |
| LangChain 对应 | `AgentState` + `checkpointer` | 长期记忆 store |

**核心产出**：一个真正有记忆的 Agent + **五个实测踩出来的坑** +
一组「线程隔离 / 恢复 / 跨线程共享」的三向对照。

---

## 目录

- [一、两级记忆的分工](#一两级记忆的分工)
- [二、架构全景](#二架构全景)
- [三、核心流程图](#三核心流程图)
- [四、代码调用详细流程图（函数级）](#四代码调用详细流程图函数级)
- [五、时序图](#五时序图)
- [六、五个实测踩出来的坑](#六五个实测踩出来的坑)
- [七、线程隔离 / 恢复 / 共享对照](#七线程隔离-恢复-共享对照)
- [八、提示词的三段式结构](#八提示词的三段式结构)
- [九、事实的幂等合并](#九事实的幂等合并)
- [十、知识地图](#十知识地图)
- [十一、运行方式](#十一运行方式)
- [十二、HTTP API](#十二http-api)
- [十三、测试](#十三测试)
- [十四、技术参考速查](#十四技术参考速查)

---

## 一、两级记忆的分工

### 短期记忆 = 线程内的消息状态

对应 LangChain 的三件套：

| LangChain 概念 | 本项目的对应物 | 代码位置 |
|---|---|---|
| `AgentState.messages` | `ThreadState.messages` | `state.rs:341` |
| `configurable.thread_id` | `ThreadId` | `state.rs:479` |
| `checkpointer` | `Checkpointer` trait | `checkpoint.rs:58` |

### 长期记忆 = 跨线程的事实库

对应 Mem0 的 `memory.add()` / `memory.search()`：

| Mem0 概念 | 本项目的对应物 | 代码位置 |
|---|---|---|
| LLM 事实抽取器 | `FactExtractor` trait | `longterm.rs:204` |
| 向量库 + 过滤器 | `MemoryStore` trait | `longterm.rs:219` |
| `memory.search()` | `MemoryRetriever` trait | `longterm.rs:264` |
| 「只增不覆盖」 | `(owner, subject, predicate)` **合并** + `versions` | `longterm.rs:342` |

### 为什么必须分成两级

因为它们的**性质**完全不同：

```text
短期记忆（原始消息）              长期记忆（提炼事实）
├── 有「谁说的」语义              ├── 没有「谁说的」语义
├── 每次对话都在变                ├── 跨会话稳定
├── 可能几十条                    ├── 通常几条（top_k 限制）
└── 全量进上下文                  └── 按需召回
```

混在一起有两个后果：

1. 模型分不清「Bob 是我现在告诉你的，还是你记得的」
2. 每次召回都把事实重复写进历史 → 历史越来越臃肿

---

## 二、架构全景

```mermaid
graph TB
    subgraph CLI["命令行入口"]
        MAIN["main.rs:47<br/>main()<br/>chat / threads / memories / compare / serve"]
    end

    subgraph API["HTTP API 层（axum）"]
        ROUTES["routes.rs:34<br/>router()"]
        HANDLERS["handlers.rs:61<br/>chat() / compare()"]
        MODELS["models.rs<br/>请求/响应契约"]
        EXTRACT["extract.rs<br/>统一错误提取器"]
    end

    subgraph CORE["核心层"]
        AGENT["agent.rs:170<br/>★ Agent::run()<br/>带记忆的对话循环"]
        STATE["state.rs<br/>消息 / 线程 / 角色模型"]
        PROMPT["prompt.rs:86<br/>system_prompt()<br/>三段式组装"]
        LLM["llm.rs:65<br/>ChatLlm trait<br/>带历史的对话"]
    end

    subgraph SHORT["短期记忆（线程内）"]
        CP["checkpoint.rs:58<br/>Checkpointer trait"]
        CPMEM["checkpoint.rs:98<br/>InMemoryCheckpointer"]
        CPFILE["checkpoint.rs:246<br/>FileCheckpointer<br/>原子写 + 落盘"]
    end

    subgraph LONG["长期记忆（跨线程）"]
        EXT["longterm.rs:204<br/>FactExtractor trait"]
        STORE["longterm.rs:219<br/>MemoryStore trait"]
        RET["longterm.rs:264<br/>MemoryRetriever trait"]
        RANK["longterm.rs:805<br/>rank_facts()<br/>BM25 + 零命中兜底"]
    end

    subgraph REUSE["复用 08-rag"]
        BM25["rag_08::bm25::Bm25Index"]
        TOK["rag_08::tokenize::tokenize()"]
    end

    subgraph EXT2["外部"]
        DS["DeepSeek API"]
    end

    MAIN --> CORE
    MAIN --> API
    ROUTES --> HANDLERS
    HANDLERS --> MODELS
    HANDLERS --> AGENT
    EXTRACT --> MODELS

    AGENT --> STATE
    AGENT --> PROMPT
    AGENT --> LLM
    AGENT --> CP
    AGENT --> RET
    AGENT --> EXT
    AGENT --> STORE

    CP --> CPMEM
    CP --> CPFILE
    RET --> RANK
    RANK --> BM25
    BM25 --> TOK
    EXT --> LLM
    LLM --> DS

    style AGENT fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style CP fill:#fff4e6,stroke:#e8a33d
    style STORE fill:#fff4e6,stroke:#e8a33d
    style RANK fill:#ffe6e6,stroke:#d94a4a
```

### 模块职责

| 模块 | 行数 | 职责 | 对应概念 |
|---|---|---|---|
| `state.rs` | 697 | 消息 / 线程 / 角色模型 | `AgentState` / `thread_id` |
| `checkpoint.rs` | 567 | 状态持久化（内存 / 文件） | `checkpointer` |
| `longterm.rs` | 1187 | 事实抽取 / 存储 / 检索 | Mem0 / ReMe |
| `prompt.rs` | 455 | 提示词组装（身份 / 规则 / 记忆 / 历史） | `AgentState` → 模型 |
| `agent.rs` | 650 | **带记忆的对话循环** | `agent.invoke()` |
| `llm.rs` | 305 | LLM 抽象（含 `ScriptedChatLlm`） | Model Interface |
| `api/` | 403 | HTTP 接口（含 **隔离 / 恢复对照**） | — |

---

## 三、核心流程图

```mermaid
flowchart TD
    START["用户提问<br/>thread_id + owner + question"] --> LOAD["① checkpointer.load(thread_id)<br/>没有就新建空状态"]
    LOAD --> RECALL["② retriever.search(question, owner, top_k)<br/>★ 用本轮问题做检索词"]
    RECALL --> DEGRADE1{"召回失败?"}
    DEGRADE1 -->|"是"| EMPTY["降级为空<br/>「查不到记忆」≠「不能聊天」"]
    DEGRADE1 -->|"否"| FACTS["得到召回的事实"]
    EMPTY --> SYS
    FACTS --> SYS["③ system_prompt(facts)<br/>Identity + Instructions + Memory"]
    SYS --> MERGE["merge_messages(用户输入)"]
    MERGE --> TRIM["build_messages(history)<br/>截断到 history_keep"]
    TRIM --> CHAT["④ llm.chat_messages(system, history)"]
    CHAT --> WRITE["⑤ merge_messages(助手回复)<br/>version += 1<br/>checkpointer.save(state)"]
    WRITE --> EXTRACT["⑥ extractor.extract(整个历史)<br/>★ 尽力而为"]
    EXTRACT --> DEGRADE2{"抽取失败?"}
    DEGRADE2 -->|"是"| EMPTY2["warn + 降级<br/>「记住失败」≠「回答失败」"]
    DEGRADE2 -->|"否"| PUT["store.put(fact)<br/>★ 同 key 合并，旧值进 versions"]
    EMPTY2 --> DONE
    PUT --> DONE(["返回 AgentRun<br/>answer / recalled / extracted"])

    style WRITE fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style DEGRADE1 fill:#fff4e6,stroke:#e8a33d
    style DEGRADE2 fill:#fff4e6,stroke:#e8a33d
    style EMPTY2 fill:#fff4e6,stroke:#e8a33d
```

### 三个必须做对的顺序

| 顺序 | 为什么 |
|---|---|
| ① **召回在调模型之前** | 事实要进 system 提示，就必须先召回。反过来做没有意义。 |
| ② **写回状态在拿到回复之后** | 先存「用户输入」再调模型，模型失败就会留下一条**孤儿用户消息**。 |
| ③ **抽取在写回状态之后** | 抽取要基于**完整的这一轮**；而且抽取失败不能影响已经得到的答案。 |

---

## 四、代码调用详细流程图（函数级）

> 每一张图都对着源码，节点写的是**真实函数名**，边上标的是**调用点行号**。
> 图例：`文件.rs:行号` = 该函数的定义位置（少数标注实现步骤的行会直接指向那行代码）；
> `— 函数名` = 从哪个函数调进去。
> ★ 标出「容易写错」的位置。

### 4.1 入口分发：一条命令是怎么走到 Agent 的

```mermaid
flowchart TD
    MAIN["main.rs:47<br/>main()"] --> ARG["main.rs:86<br/>arg_or_default()"]
    MAIN -->|"demo == 'chat'"| CHATD["main.rs:195<br/>chat_demo()"]
    MAIN -->|"demo == 'compare'"| CMPD["main.rs:332<br/>compare_demo()"]
    MAIN -->|"demo == 'threads'"| THRD["main.rs:288<br/>threads_demo()"]
    MAIN -->|"demo == 'memories'"| MEMD["main.rs:305<br/>memories_demo()"]
    MAIN -->|"demo == 'serve'"| SERVE["main.rs:440<br/>serve()"]

    CHATD --> STO["main.rs:104<br/>build_storage()"]
    CMPD --> STO
    SERVE --> STO
    SERVE --> LLM["main.rs:120<br/>build_llm()"]
    CHATD --> LLM
    CMPD --> LLM
    SERVE --> AGENT["main.rs:168<br/>build_agent()"]
    CHATD --> AGENT
    CMPD --> AGENT

    STO --> CPM["checkpoint.rs:404<br/>file_checkpointer()"]
    STO --> STOREF["longterm.rs:1059<br/>file_store()"]
    STO --> CPMEM["checkpoint.rs:399<br/>in_memory_checkpointer()"]
    STO --> STOREM["longterm.rs:1054<br/>in_memory_store()"]

    LLM --> GENAI["llm.rs:95<br/>GenaiChatLlm::new()"]
    AGENT --> RET["main.rs:141<br/>build_retriever()"]
    AGENT --> NEW["agent.rs:145<br/>Agent::new()"]
    SERVE --> ROUTER["api/routes.rs:34<br/>router()"]
    ROUTER --> HANDLER["api/handlers.rs:61<br/>chat()"]

    style NEW fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style ROUTER fill:#e8f4ff,stroke:#4a90d9
```

### 4.2 ★ 核心：`Agent::run()` 的完整调用链

这是整个项目最重要的一张图。**红色节点是「容易写错」的地方**。

```mermaid
flowchart TD
    RUN["agent.rs:170<br/>Agent::run(thread_id, owner, question)"] --> LOAD["checkpoint.rs:169<br/>checkpointer.load()"]
    RUN --> RECALL["longterm.rs:744<br/>retriever.search(question, owner, top_k)"]
    RUN --> SYS["prompt.rs:86<br/>system_prompt(recalled, config)"]
    RUN --> MERGE["state.rs:396<br/>ThreadState::merge_messages()"]
    RUN --> TRIM["prompt.rs:136<br/>build_messages()"]
    RUN --> CHAT["llm.rs:107<br/>chat_messages()"]
    RUN --> SAVE["checkpoint.rs:182<br/>checkpointer.save()"]
    RUN --> EXTRACT["agent.rs:295<br/>extract_and_store()"]

    LOAD --> LOADF{"线程存在?"}
    LOADF -->|"否"| NEWSTATE["state.rs:374<br/>ThreadState::new()"]
    LOADF -->|"是"| KEEP["沿用已有历史"]

    RECALL --> RANK["longterm.rs:805<br/>rank_facts()"]
    RANK --> ZERO{"BM25 零命中?"}
    ZERO -->|"是"| FALLBACK["★ longterm.rs:805<br/>rank_facts() 零命中兜底<br/>退化为按 importance 取前 k 条"]
    ZERO -->|"否"| HIT["按 BM25 分数排序"]

    FALLBACK --> SYS
    HIT --> SYS
    KEEP --> MERGE
    NEWSTATE --> MERGE

    MERGE --> TRIM
    TRIM --> CHAT
    CHAT --> API["DeepSeek API<br/>system + messages"]

    API --> SAVE
    SAVE --> EXTRACT

    EXTRACT --> PROMPT2["prompt.rs:179<br/>extraction_prompt()"]
    PROMPT2 --> LLM2["llm.rs:107<br/>chat_messages()"]
    LLM2 --> PARSE["prompt.rs:229<br/>parse_extraction()"]
    PARSE --> PUT["longterm.rs:342<br/>store.put(fact)"]
    PUT --> MERGE2["★ 同 key 合并<br/>旧值进 versions"]

    style RUN fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style SAVE fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style FALLBACK fill:#ffe6e6,stroke:#d94a4a,stroke-width:2px
    style MERGE2 fill:#ffe6e6,stroke:#d94a4a
```

### 4.3 ★ 状态写回：为什么顺序不能反

**先写回再调模型 → 留下孤儿消息**；**先调模型再写回 → 失败时历史不完整**。
正确做法是「调模型 → 拿到回复 → 一次性写回两条」。

```mermaid
flowchart TD
    A["state.rs:396<br/>ThreadState::merge_messages(用户输入)"] --> B["prompt.rs:136<br/>build_messages(history)"]
    B --> C["llm.rs:107<br/>GenaiChatLlm::chat_messages()"]
    C --> D{"模型调用成功?"}
    D -->|"否"| ERR["❌ 直接返回错误<br/>★ 用户消息**没有**写回<br/>下次恢复时历史是干净的"]
    D -->|"是"| E["state.rs:396<br/>ThreadState::merge_messages(助手回复)"]
    E --> F["state.rs:341<br/>ThreadState.version"]
    F --> G["checkpoint.rs:327<br/>FileCheckpointer::save(state)"]
    G --> H["agent.rs:295<br/>Agent::extract_and_store()"]

    style D fill:#ffe6e6,stroke:#d94a4a
    style E fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style H fill:#fff4e6,stroke:#e8a33d
```

### 4.4 事实抽取：失败降级 vs 成功合并

```mermaid
flowchart TD
    A["agent.rs:295<br/>Agent::extract_and_store(owner, messages)"] --> B{"筛出 user/assistant<br/>且内容非空?"}
    B -->|"否"| EMPTY["返回空列表<br/>不是错误"]
    B -->|"是"| C["longterm.rs:1025<br/>LlmFactExtractor::extract()"]
    C --> D["prompt.rs:179<br/>extraction_prompt()"]
    D --> E["llm.rs:107<br/>chat_messages()"]
    E --> F["prompt.rs:229<br/>parse_extraction()"]
    F --> G{"解析出事实?"}
    G -->|"否"| EMPTY
    G -->|"是"| H["longterm.rs:342<br/>store.put(fact)"]
    H --> I{"同 (owner,subject,predicate)?"}
    I -->|"是"| J["★ 合并<br/>旧 value 进 versions<br/>新 value 成为当前值"]
    I -->|"否"| K["追加到 owner 桶"]
    J --> L["返回抽取到的事实"]
    K --> L

    E -->|"模型挂了"| M["★ agent.rs:295<br/>warn + 降级为空<br/>不影响已得到的答案"]
    H -->|"存储失败"| N["★ agent.rs:295<br/>warn + 跳过这一条<br/>逐条处理，互不影响"]
    M --> O(["返回"])
    N --> O

    style J fill:#ffe6e6,stroke:#d94a4a
    style M fill:#fff4e6,stroke:#e8a33d
    style N fill:#fff4e6,stroke:#e8a33d
```

### 4.5 `FileCheckpointer` 的原子写

```mermaid
flowchart TD
    A["checkpoint.rs:327<br/>save(state)"] --> B["checkpoint.rs:327<br/>FileCheckpointer::save()<br/>serde_json::to_string_pretty()"]
    B --> C["checkpoint.rs:340<br/>写临时文件 {thread}.json.tmp<br/>★ 原子写的关键"]
    C --> D{"写入成功?"}
    D -->|"否"| E["❌ 错误传播<br/>★ 旧文件**完好无损**"]
    D -->|"是"| F["checkpoint.rs:342<br/>rename(tmp → 正式文件)<br/>★ 要么新的，要么旧的"]
    F --> G["✅ 原子替换<br/>要么是新的，要么是旧的"]

    style C fill:#fff4e6,stroke:#e8a33d
    style F fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
```

---

## 五、时序图

### 5.1 完整的一轮对话（含长期记忆召回）

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant H as handlers.rs
    participant A as Agent::run
    participant CP as Checkpointer
    participant R as MemoryRetriever
    participant S as MemoryStore
    participant L as LLM
    participant E as FactExtractor

    U->>H: POST /api/chat {thread_id, owner, question}
    H->>H: 校验 thread_id（★ 入口挡住路径穿越）
    H->>A: run(thread_id, owner, question)

    A->>CP: load(thread_id)
    CP-->>A: Option<ThreadState>
    Note over CP: None = 正常状态，不是错误

    A->>R: search(question, owner, top_k)
    R->>S: list(owner)
    S-->>R: Vec<MemoryFact>
    R->>R: rank_facts() BM25 排序
    R-->>A: 召回的事实

    A->>A: system_prompt(事实) ← 事实进 system
    A->>A: merge_messages(用户输入)
    A->>A: build_messages() 截断历史
    A->>L: chat_messages(system, history)
    L-->>A: 答案

    A->>CP: save(state) ← ★ 回复到手后才写回
    A->>E: extract(整个历史, owner)
    E->>L: chat_messages(抽取提示词)
    L-->>E: JSON 数组
    E->>S: put(fact) ← 同 key 合并
    S-->>E: 是否合并

    A-->>H: AgentRun
    H-->>U: {answer, recalled, extracted, history_messages}
```

### 5.2 抽取失败 → 降级，不影响答案

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant A as Agent::run
    participant L as LLM
    participant E as FactExtractor
    participant S as MemoryStore

    U->>A: run(thread_id, owner, "我叫 Bob")
    A->>L: chat_messages(system, history)
    L-->>A: "你好 Bob"
    Note over A: ★ 先拿到答案

    A->>E: extract(历史, owner)
    E->>L: chat_messages(抽取提示词)
    L--xE: 网络错误
    E-->>A: Err(LLM)
    Note over A: ★ 记 warn，降级为空
    A->>S: （不调用）
    A-->>U: answer = "你好 Bob"，extracted = []
```

---

## 六、五个实测踩出来的坑

### 坑 1：召回的事实被当成「用户说的话」写进历史

**现象**：几轮之后历史里全是重复事实（「用户的名字：Bob」出现 5 次），上下文被垃圾淹没。

**根因**：把召回的事实当成一条普通消息塞进消息数组。

**修法**：事实**只进 system 提示**（`# Memory` 段），**不进**消息数组。

```mermaid
flowchart LR
    subgraph BAD["❌ 错误做法"]
        B1["召回事实"] --> B2["作为 MemoryMessage 塞进消息数组"]
        B2 --> B3["写回 checkpointer"]
        B3 --> B4["下一轮又召回一次<br/>→ 历史里全是重复事实"]
    end

    subgraph GOOD["✅ 正确做法"]
        G1["召回事实"] --> G2["渲染成 system 提示的 # Memory 段"]
        G2 --> G3["消息数组里**只有** user/assistant"]
        G3 --> G4["历史干净，条数 = 轮数 × 2"]
    end

    style BAD fill:#ffe6e6,stroke:#d94a4a
    style GOOD fill:#e8f4ff,stroke:#4a90d9
```

**回归测试**：`recalled_facts_never_enter_the_message_array`（集成测试）

---

### 坑 2：用户改口后，新旧两条事实并存 → 召回时打架

**现象**：用户先说「我叫 Bob」，后说「我改叫 Robert」。记忆库里有两条事实，召回时两条都进 system 提示，模型不知道该用哪个。

**根因**：纯追加式存储 —— Mem0 强调「Memories accumulate; nothing is overwritten」，但纯追加会留下矛盾。

**修法**：**同 `(owner, subject, predicate)` 合并**，旧值进 `versions`，新值成为当前值。

```text
用户 / 名字 / Bob
   ↓ 用户改口
用户 / 名字 / Robert     ← 当前值
   versions: ["Bob"]    ← 旧值不丢，但不再是当前值
```

**为什么去重键不是整条内容**：因为「用户叫 Bob」和「用户名字是 Bob」是**同一件事**。用整条内容做键，模型换个说法就产生重复事实。

**回归测试**：`a_changed_fact_is_merged_not_duplicated`（集成测试）

---

### 坑 3：BM25 零命中 → 模型彻底失忆

**现象**：事实明明存了「用户 / 名字 / Bob」，用户在新线程问「**我叫什么**」，模型答不出来。

**根因**：BM25 是**纯词面**匹配，匹配不了「语义等价」。

```text
查询词元：我 叫 什 么 我叫 叫什 什么
事实词元：用户 名字 bob 用户 户名 名字
交集    ：∅
```

这正是**向量检索**能解决、BM25 解决不了的场景。

**修法**：零命中时**退化为「按 importance 取前 k 条」**。

- 一个 owner 的事实通常只有几条到几十条，全部塞进上下文并不昂贵
- 模型自己会挑 —— 它看到「用户的名字：Bob」就知道这是答案
- 比「什么都不给」好一个量级

**回归测试**：`zero_hit_falls_back_to_importance_ordering`（单元测试）

---

### 坑 4：短期记忆和长期记忆共用一个目录 → 线程列表被污染

**现象**：`GET /api/threads` 返回一堆 `{hash}_{owner}.json` 的垃圾 id。

**根因**：两者都用 `.json` 后缀，放在同一个目录里，`FileCheckpointer::list_threads` 扫描目录时把**记忆文件也当成线程**了。

这是一个**静默**的错 —— 接口不报错，但列表是脏的。

**修法**：两级记忆**分目录**。

```text
data/memory/
├── threads/     ← 短期记忆：{thread_id}.json
└── memories/    ← 长期记忆：{hash}_{owner}.json
```

**回归测试**：`memory_files_are_not_listed_as_threads`（集成测试）

---

### 坑 5：`compare` 端点的隔离判据算错

**现象**：隔离判据写成 `(线程数 + 1) * 2`，随线程数变化 —— 两个线程时算出 6，实际是 4，判据**永远为 false**。

```text
实际：每线程 2 轮 × 2 条消息 = 4 条
错误：(2 + 1) * 2 = 6 条   ← 永远对不上
```

**修法**：轮数是**固定的 2**（代码里明确跑了第一轮 + 第二轮），判据写死 `2 * 2 = 4`。

**顺带修的第二个 bug**：`longterm_shared` 原来用**第一轮**的召回判定，但第一轮时记忆库还是空的（事实是这一轮结束之后才抽取入库的）→ 永远为 false。改成用**第二轮**判定。

**回归测试**：`compare_endpoint_reports_isolation_and_resumption`（集成测试）

---

## 七、线程隔离 / 恢复 / 共享对照

`/api/compare` 和 `make compare` 跑的是同一套逻辑，做**三向对照**：

| 对照项 | 判据 | 证明什么 |
|---|---|---|
| `threads_isolated` | 每个线程的历史**恰好** 4 条（2 轮 × 2 条） | 短期记忆互不串台 |
| `thread_resumed` | 同一线程第二轮的历史**比第一轮多** | checkpointer 真的存下来了 |
| `longterm_shared` | **第二轮**每个线程都召回了 ≥1 条事实 | 长期记忆跨线程共享 |

### 为什么隔离要「实际检查」而不是「声称隔离」

隔离是一个**否定性**性质 ——「不该出现的东西没出现」。
光看响应体看不出来，必须**读回状态**逐条比对。

```mermaid
flowchart TD
    A["/api/compare"] --> B["第一轮：每个线程各聊一次<br/>记录 history_messages"]
    B --> C["第二轮：每个线程再聊一次<br/>记录 history_messages + recalled"]
    C --> D{"每个线程历史 == 4?"}
    D -->|"是"| ISO["✅ threads_isolated"]
    D -->|"否"| NOTISO["❌ 可能串台了"]
    C --> E{"第二轮历史 > 第一轮?"}
    E -->|"是"| RES["✅ thread_resumed"]
    E -->|"否"| NOTRES["❌ 没存下来"]
    C --> F{"第二轮每线程都召回了事实?"}
    F -->|"是"| SHARED["✅ longterm_shared"]
    F -->|"否"| NOTSHARED["❌ 长期记忆没生效"]

    style ISO fill:#e8f4ff,stroke:#4a90d9
    style RES fill:#e8f4ff,stroke:#4a90d9
    style SHARED fill:#e8f4ff,stroke:#4a90d9
```

### 实测输出

```text
── 线程 A 第 1 轮 ──
问题: 我叫 Bob
历史条数: 2
  抽取: 用户的名字：Bob

── 线程 B 第 1 轮 ──
问题: 我住在杭州
历史条数: 2
  抽取: 用户的居住城市：杭州

── 线程 A 第 2 轮（同一线程）──
问题: 我叫什么名字？
历史条数: 4
召回的记忆:
  · 用户的名字：Bob

── 线程 C（全新线程）第 1 轮 ──
问题: 我叫什么名字？
历史条数: 2
召回的记忆（跨线程!）:
  · 用户的名字：Bob

── 结论 ──
线程隔离    : ✅（线程 A 4 条 / 线程 B 2 条）
线程恢复    : ✅（2 → 4 条）
长期记忆共享: ✅（新线程召回了 1 条事实）
```

---

## 八、提示词的三段式结构

```mermaid
flowchart TD
    subgraph SP["system 提示（每次重新拼）"]
        ID["# Identity<br/>你是谁"]
        IN["# Instructions<br/>行为规则"]
        MEM["★ # Memory<br/>召回的事实<br/>「可能已经过时」"]
    end

    subgraph MSGS["消息数组（历史）"]
        U1["user: 我叫 Bob"]
        A1["assistant: 你好 Bob"]
        U2["user: 我叫什么名字？"]
    end

    ID --> IN --> MEM
    MEM -.->|"事实**不**进这里"| MSGS

    style MEM fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
```

### 关键决策

| 决策 | 理由 |
|---|---|
| 长期记忆和对话历史**分成两段** | 模型分不清「Bob 是你现在说的还是你记得的」 |
| 事实段措辞是「**可能已经过时**」 | 用户可能改口，不写这句模型会用旧记忆纠正用户 |
| 指令写「冲突时**以当前说法为准**」 | 同上 |
| 事实**不进**消息数组 | 见坑 1 |
| 事实段**每次重新拼** | 召回结果每轮可能不同 |

### 记忆抽取的提示词

```text
# Task

从下面的对话里提取**关于用户**的原子事实。

# Rules

- 只提取**事实**，不要提取闲聊、客套、情绪。
- 每条事实是「主语 + 谓语 + 值」三段。
- 谓语要短且稳定（如「名字」「偏好」「职业」「语言」）。
- **没有事实就返回空数组** `[]` —— 不要为了凑数编造。

# Output Format

严格的 JSON 数组，不要 markdown 代码块，不要任何解释文字：

[{"subject": "用户", "predicate": "名字", "value": "Bob", "importance": 0.8}]
```

⚠️ 输出格式**必须**是严格的 JSON 数组 —— 模型很爱用 markdown 代码块包裹，所以解析器要做容错（`extract_json_array` 找第一个 `[` 到配对的 `]`）。

---

## 九、事实的幂等合并

```mermaid
stateDiagram-v2
    [*] --> 不存在
    不存在 --> 当前值: put(用户/名字/Bob)
    当前值 --> 当前值: put(用户/名字/Bob)<br/>值相同 → 不动
    当前值 --> 当前值: put(用户/名字/Robert)<br/>★ 合并：旧值进 versions
    当前值 --> [*]: delete(id)
```

### `MemoryFact` 的字段

| 字段 | 作用 |
|---|---|
| `subject` / `predicate` / `value` | 事实的三段式结构（便于去重和矛盾检测） |
| `owner` | 归属（= Mem0 的 `user_id`），**存储层强制分区** |
| `importance` | 重要度（0~1），召回时的次级排序键 |
| `versions` | 历史值（旧信息不丢，但不再是当前值） |
| `dedup_key()` | `(owner, subject, predicate)` —— 去重键 |

### 为什么 owner 分区在**存储层**做

和线程隔离同一个道理：「先取出全部再 filter」漏一次就是数据泄漏。
`MemoryStore::list(owner)` 直接按 owner 取，物理上不可能串台。

---

## 十、知识地图

```mermaid
graph LR
    subgraph CORE["核心概念"]
        N1["Agent Memory<br/>两级记忆"]
        N2["thread_id<br/>短期记忆的边界"]
        N3["checkpointer<br/>状态持久化"]
        N4["AgentState<br/>消息状态"]
        N5["trim_messages<br/>历史截断"]
    end

    subgraph LONG["长期记忆（Mem0）"]
        N6["FactExtractor<br/>LLM 抽事实"]
        N7["MemoryStore<br/>按 owner 分区"]
        N8["MemoryRetriever<br/>按需召回"]
        N9["事实合并<br/>versions"]
    end

    subgraph FILE["ReMe：Memory as File"]
        N10["文件即记忆<br/>可直接 cat / 编辑 / 备份"]
        N11["索引可重建<br/>不是真相来源"]
    end

    subgraph PIT["踩过的坑"]
        N12["事实只进 system<br/>不进消息数组"]
        N13["BM25 零命中<br/>→ 退化召回"]
        N14["两级记忆分目录"]
        N15["合并而非追加"]
    end

    N1 --> N2 --> N3 --> N4 --> N5
    N1 --> N6 --> N7 --> N8 --> N9
    N1 --> N10 --> N11
    N12 -.-> N8
    N13 -.-> N8
    N14 -.-> N3
    N15 -.-> N9

    style N1 fill:#e8f4ff,stroke:#4a90d9,stroke-width:2px
    style N12 fill:#ffe6e6,stroke:#d94a4a
    style N13 fill:#ffe6e6,stroke:#d94a4a
    style N14 fill:#ffe6e6,stroke:#d94a4a
    style N15 fill:#ffe6e6,stroke:#d94a4a
```

---

## 十一、运行方式

### 快速开始

```bash
# 1. 设置 API key
export DEEPSEEK_API_KEY=sk-xxx

# 2. ★ 跑一次带记忆的对话（两轮：告诉名字 → 问名字）
make chat

# 3. ★ 线程隔离 / 恢复对照
make compare

# 4. 列出所有线程 / 长期记忆
make threads
make memories

# 5. 启动 HTTP 服务
make serve
```

### 全部目标

| 目标 | 说明 |
|---|---|
| `make help` | 显示所有目标 |
| `make build` | 编译 |
| `make chat` | ★ 跑一次带记忆的对话 |
| `make compare` | ★ 线程隔离 / 恢复对照 |
| `make threads` | 列出所有线程 |
| `make memories` | 列出长期记忆 |
| `make history THREAD=xxx` | 查看某个线程的历史 |
| `make serve` | 启动 HTTP 服务（:3100） |
| `make test` | 跑全部测试 |
| `make lint` | clippy 零警告检查 |
| `make fmt` | 格式化 |
| `make check` | fmt + lint + test |

### 环境变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `MEM_BIND_ADDR` | 监听地址 | `127.0.0.1:3100` |
| `MEM_MODEL` | 模型名 | `deepseek-flash` |
| `MEM_TEMPERATURE` | 温度 | `0.0` |
| `MEM_MAX_TOKENS` | max tokens | `2048` |
| `MEM_REASONING` | 是否开思考模式（`1` = 开） | `0`（关） |
| `MEM_CHECKPOINTER` | `memory` / `file` | `file` |
| `MEM_MEMORY_STORE` | `memory` / `file` | `file` |
| `MEM_DATA_DIR` | 存储根目录 | `data/memory` |
| `MEM_MEMORY_TOP_K` | 长期记忆召回条数 | `5` |
| `MEM_HISTORY_KEEP` | 历史保留条数（`0` = 全部） | `20` |
| `MEM_EXTRACT` | 是否启用长期记忆抽取 | `1` |

⚠️ 温度默认 **0.0** —— 记忆场景要**可复现**，温度高会让抽取出的事实飘忽。

---

## 十二、HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/` | 服务信息 + 配置 |
| POST | `/api/chat` | **带记忆的对话**（核心端点） |
| GET | `/api/threads` | 列出所有线程 |
| GET | `/api/threads/{id}` | 查看某个线程的消息历史 |
| DELETE | `/api/threads/{id}` | 清空某个线程的短期记忆 |
| GET | `/api/memories?owner=` | 列出某 owner 的长期记忆 |
| DELETE | `/api/memories?owner=` | 清空某 owner 的长期记忆 |
| POST | `/api/compare` | ★ **线程隔离 / 恢复对照** |

### 示例

```bash
# 跑一次带记忆的对话
curl -s -X POST localhost:3100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"thread_id": "t1", "owner": "alice", "question": "我叫 Bob"}' \
  | python3 -m json.tool

# 三向对照
curl -s -X POST localhost:3100/api/compare \
  -H 'Content-Type: application/json' \
  -d '{"owner": "alice", "threads": ["thread-a", "thread-b"], "question": "我叫什么名字？"}' \
  | python3 -m json.tool
```

### 状态码约定

| 状态码 | 含义 |
|---|---|
| 200 | 成功 |
| 400 | 参数不合法（thread_id 非法 / 问题为空 / 请求体解析失败） |
| 404 | 线程不存在 |
| 502 | 模型调用失败 |
| 500 | 内部错误 |

---

## 十三、测试

```bash
make test        # 单元 84 + 集成 21
```

### 被锁住的关键行为

| 行为 | 测试 |
|---|---|
| 线程隔离 | `threads_do_not_leak_into_each_other` |
| 线程恢复 | `a_thread_resumes_with_its_previous_history` |
| 进程重启后恢复 | `the_state_survives_a_checkpointer_restart` |
| 跨线程长期记忆 | `a_fact_is_extracted_and_recalled_across_threads` |
| 事实不进消息数组 | `recalled_facts_never_enter_the_message_array` |
| 事实合并而非追加 | `a_changed_fact_is_merged_not_duplicated` |
| owner 隔离 | `owners_are_isolated_in_the_long_term_store` |
| 抽取失败不影响答案 | `a_failing_extractor_does_not_break_the_answer` |
| BM25 零命中兜底 | `zero_hit_falls_back_to_importance_ordering` |
| 两级记忆分目录 | `memory_files_are_not_listed_as_threads` |
| 损坏文件容错 | `a_corrupted_thread_file_is_tolerated` |
| `thread_id` 路径穿越防护 | `chat_endpoint_rejects_an_invalid_thread_id` |
| 隔离判据正确性 | `compare_endpoint_reports_isolation_and_resumption` |

### 为什么全部零网络

整个记忆层的五个依赖**全部从构造时注入**：

```text
ChatLlm      → ScriptedChatLlm（记录每次调用看到的完整历史）
Checkpointer → InMemoryCheckpointer / FileCheckpointer
MemoryStore  → InMemoryMemoryStore / FileMemoryStore
MemoryRetriever → 线性打分检索器
FactExtractor → ScriptedExtractor / NoopExtractor
```

所以测试可以**断言「模型看到了第几轮的什么」**，而不只是断言「返回了什么」。

---

## 十四、技术参考速查

### 核心概念

| 概念 | 本项目 | LangChain | Mem0 |
|---|---|---|---|
| 短期记忆 | `ThreadState` | `AgentState` | — |
| 线程隔离 | `ThreadId` + 分键存储 | `configurable.thread_id` | — |
| 状态持久化 | `Checkpointer` | `Checkpointer` | — |
| 历史截断 | `build_messages` | `trim_messages` | — |
| 事实抽取 | `FactExtractor` | — | `memory.add()` 的 LLM 抽取 |
| 事实存储 | `MemoryStore` | 长期记忆 store | 向量库 + 过滤器 |
| 事实召回 | `MemoryRetriever` | — | `memory.search()` |
| 幂等合并 | `versions` | — | `nothing is overwritten` |
| 记忆即文件 | `FileMemoryStore` | — | ReMe |

### 仓库内的复用关系

| 复用了 | 来自 | 为什么 |
|---|---|---|
| `ChatMessage` / `ContentPart` / `ToolResponse` | `genai` | 消息类型统一 |
| `ToolError` | `agent_16::error` | 错误分类（400 vs 500）已想清楚 |
| `Bm25Index` / `Bm25Document` / `tokenize` | `rag_08` | 中英混排分词 + BM25 打分 |
| `trace_layer` | `agent_logging` | 统一请求追踪 |

### 本项目特有的约定

| 约定 | 说明 |
|---|---|
| 事实用**自己的类型** | 不直接存 `genai::ChatMessage` —— 持久化格式不焊死在第三方类型的内部结构上 |
| `MemoryRole::Fact` | 显式表达「这是召回的背景，不是对话」 |
| `Fact` 映射成 `ChatRole::System` | 事实是权威上下文，不是「某人在说话」 |
| 抽取基于**整个历史** | 「我叫 Bob」可能是十轮之前说的 |
| 抽取基于**整个历史**≠ 写入事实 | 写入走 `store.put()` 的幂等合并 |
| 抽取必须**关思考** | 思考 token 与正文共用 `max_tokens` 预算 |
| 解析失败**不是错误** | 「抽不出事实」是正常结果，返回空列表 |

### 环境变量前缀

| 前缀 | 项目 |
|---|---|
| `RAG_` | 06-rag |
| `QUERY_` | 07-query |
| `AGENT_` | 13-agent-core |
| `TOOL_` | 14-agent-tool |
| `SQL_` | 15-agent-sql |
| `FC_` | 16-agent-function-calling |
| `MEM_` | **17-agent-memory** |

### 相关项目

| 项目 | 主题 |
|---|---|
| [13-agent-core](../13-agent-core/) | ReAct 循环（文本协议） |
| [14-agent-tool](../14-agent-tool/) | 自定义工具开发 + MCP |
| [15-agent-sql](../15-agent-sql/) | SQL Agent + 三层安全防线 |
| [16-agent-function-calling](../16-agent-function-calling/) | 结构化 Function Calling |
| [08-rag](../08-rag/) | BM25 检索（长期记忆的召回底座） |
| [12-rag-summary](../12-rag-summary/) | 摘要式 RAG（向量检索的完整版） |
