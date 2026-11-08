# 向量数据库知识库

## 向量数据库是什么

向量数据库是一种专门存储、索引和检索**高维向量**的数据库。
它把文本、图片、音频等非结构化数据通过嵌入模型映射成向量，
然后用近似最近邻（ANN）算法找出语义最相近的条目。

与普通关系型数据库的核心区别在于查询目标：
关系库问「哪些行满足 `WHERE id = 42`」，向量库问「哪些条目在语义上最接近这段文本」。

## 为什么需要专门的向量数据库

传统数据库做向量检索有三个硬伤：

- **索引不匹配**：B+ 树擅长的是范围查询和精确匹配，对高维空间的最近邻无能为力
- **维度灾难**：在高维空间里，距离度量会退化（所有点的距离趋于相等）
- **无法压缩**：向量是稠密的 float 数组，原始存储开销极大

向量数据库的解法是**专用索引结构**：HNSW 图、IVF 倒排、量化压缩、
磁盘友好的图索引 —— 每一项都是为了在「召回率 / 延迟 / 内存」三者间取舍。

## Milvus 的架构

Milvus 是分布式的向量数据库，核心组件分四类角色：

- **Proxy**：无状态入口，负责解析请求、路由到目标分片、归并多路结果
- **QueryNode**：只读节点，加载 segment 与索引到内存，执行向量检索
- **DataNode**：只写节点，接收写入、落 WAL、攒够后 flush 成 segment
- **Coordinator**：集群大脑，调度索引构建、触发 compaction、监控健康

数据分三层存放：

- **etcd**：collection schema、segment 元信息、别名 —— 强一致要求
- **对象存储（MinIO / S3）**：真正的向量数据、索引文件 —— 大且可重算
- **内存**：QueryNode 加载的 segment 与索引 —— 热数据

## Collection 与 Schema

Milvus 的 collection 相当于关系库的 table，但它的 schema 更接近「带类型的 JSON」：

- 每个字段有明确的数据类型（`Int64` / `VarChar` / `JSON` / `FloatVector` / `SparseFloatVector` …）
- 必须有一个主键字段，且**不能开启 auto_id**（除非你不需要稳定 ID）
- 向量字段必须声明维度
- 可以声明**服务端函数**（Function），比如把文本字段转成 BM25 稀疏向量

## 索引类型

| 类型 | 原理 | 适用规模 |
|---|---|---|
| FLAT | 暴力遍历，无索引 | < 1 万，要求 100% 精确 |
| HNSW | 多层可导航小世界图 | 通用首选，百万级 |
| IVF_FLAT | k-means 聚类 + 簇内暴力 | 十万级，内存紧张 |
| DISKANN | 磁盘图索引 | 十亿级 |
| SCANN | 量化扫描 + 重排 | 极高召回 |
| AUTOINDEX | 服务端自动选 | 不想操心时 |

HNSW 的两个关键参数：`M`（每层连接数，默认 30）和 `efConstruction`（建索引时的候选宽度，默认 200）。
查询时的 `ef` 越大召回越高、延迟越大。

## 距离度量

- **COSINE**：余弦相似度，越大越相似。与 L2 归一化后的内积等价
- **IP**：内积，越大越相似
- **L2**：欧氏距离（Milvus 返回平方值），越小越相似

归一化向量下 `d² = 2 - 2cos`，所以 L2 和余弦排序完全一致，只是方向相反。

## 服务端 BM25 与混合检索

Milvus 把 BM25 做成了**服务端函数**：

1. 文本字段开启分析器（`enable_analyzer`），启用分词
2. 一个 `FunctionType::Bm25` 的函数，输入文本字段，输出稀疏向量字段
3. 稀疏向量字段建 `SPARSE_INVERTED_INDEX` 索引，度量用 `BM25`
4. 检索时用 `SearchVectors::EmbeddedText` 传入原始查询串，服务端负责转成稀疏向量

于是「稠密向量 + 关键词」可以一次 `hybrid_search` 完成，
融合策略由服务端的 `RRFRerank`（只看排名）或 `WeightedRerank`（加权分数）完成。

这比 08-rag 那种「自己在 Rust 里写倒排索引 + RRF」的方式好在：
分词、倒排、权重、融合全在服务端，一次 RPC 出结果，且能利用 segment 级剪枝。

## 一致性级别

Milvus 提供多个一致性级别：

- **Strong**：读最新数据，需要等 WAL 刷盘 —— 最安全，最慢
- **Session**：保证同一 session 内的读写顺序一致（默认）
- **Bounded**：允许读到略微陈旧的数据 —— 检索场景通常够用，最快
- **Eventually**：最终一致

检索场景用 `Bounded` 是合理的：稍微陈旧几十毫秒不影响语义召回。

## Partition 与多租户

partition 是 collection 内的逻辑子集，用来隔离不同租户或业务线。
partition key 字段则由服务端自动路由 —— 写入时按 key 的哈希决定落到哪个 partition。

这比「每个租户一个 collection」更好：collection 数量少了，索引管理成本就低。

## 与 Qdrant 的对比

| | Milvus | Qdrant |
|---|---|---|
| 架构 | 分布式（etcd + 对象存储 + 计算节点） | 单体服务 |
| 部署 | 三个容器 | 一个容器 |
| 全文检索 | 服务端 BM25 函数（稀疏向量） | payload text index + BM25 |
| 混合检索 | hybrid_search + RRF/Weighted | prefetch + Fusion::Rrf |
| 内存模型 | segment 加载到 QueryNode | 内存 / mmap |
| 运维复杂度 | 高 | 低 |

选型建议：
数据量千万级以上、需要多租户隔离、团队有运维能力 → Milvus；
中小规模、想快速上线、运维资源有限 → Qdrant。

## 部署方式

standalone 三容器组合：etcd（元数据）+ MinIO（对象存储）+ Milvus（计算）。

生产环境还要考虑：

- **Milvus Operator**（Kubernetes）：自动分片、扩缩容、备份
- **Milvus Backup**：collection 级别的备份与恢复
- **监控**：Prometheus 抓取 `/metrics`，重点看 QueryNode 的内存和 segment 数
- **RBAC**：Milvus 2.5+ 支持用户/角色/权限，以及行级安全（RLS）
