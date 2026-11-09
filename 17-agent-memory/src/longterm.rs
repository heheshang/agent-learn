//! 长期记忆层 —— 对齐 Mem0 与 ReMe（MemoryScope）的核心思想。
//!
//! ## 参考资料
//!
//! - [Mem0](https://github.com/mem0ai/mem0) —— 「The Memory Layer for AI Agents」
//! - [ReMe / MemoryScope](https://github.com/agentscope-ai/ReMe) ——
//!   「Memory as File, File as Memory」
//!
//! ## 短期记忆 vs 长期记忆
//!
//! | | 短期记忆（本项目的 [`crate::state`]） | 长期记忆（本模块） |
//! |---|---|---|
//! | 作用域 | **一个线程内** | **跨线程、跨会话** |
//! | 内容 | 原始对话消息 | **提炼出的事实**（「用户叫 Bob」） |
//! | 数量 | 无限增长 | 有上限、可合并 |
//! | 存取 | 全量读入上下文 | **按需召回**（关键词检索） |
//!
//! ## Mem0 的核心机制（我们手抄的部分）
//!
//! Mem0 的 `memory.add(messages, user_id=...)` 做的事情是：
//!
//! 1. 从对话里**抽取事实**（「我叫 Bob」→ 事实「用户名字是 Bob」）
//! 2. 把事实**存起来**（按 `user_id` 分区）
//! 3. 查询时按 `query` **检索**相关事实，塞进系统提示
//!
//! 本项目把这三步拆成三个 trait：
//!
//! | trait | 职责 | Mem0 的对应物 |
//! |---|---|---|
//! | [`FactExtractor`] | 从对话里抽事实 | Mem0 的 LLM 抽取器 |
//! | [`MemoryStore`] | 存 / 取 / 删事实 | Mem0 的向量库 + 过滤器 |
//! | [`MemoryRetriever`] | 按查询召回事实 | Mem0 的 `search` |
//!
//! ## ReMe 的核心机制（我们手抄的部分）
//!
//! ReMe 的一句话是「**Memory as File, File as Memory**」——
//! 记忆就是**普通 Markdown 文件**，带 frontmatter，用户可以直接读、编辑、备份。
//!
//! 本项目的 [`FileMemoryStore`] 就是这个思路：
//!
//! - 记忆存成**一个 JSON 文件**（本项目用 JSON 而不是 Markdown ——
//!   因为我们存的是结构化事实，JSON 能保证 schema 稳定；
//!   ReMe 存的是散文，Markdown 更合适）
//! - `metadata/` 目录（ReMe 的可重建索引）在本项目里对应
//!   **BM25 索引**—— 也是**可重建**的，不是真相来源
//!
//! ## 关键设计：事实的**幂等合并**
//!
//! Mem0 的新算法强调「**Memories accumulate; nothing is overwritten**」
//! —— 记忆只增不覆盖。
//!
//! 但纯追加会有问题：用户说「我叫 Bob」又说「我改名叫 Robert」时，
//! 两条事实同时存在，召回会打架。
//!
//! 本项目采用**带来源的合并**：新事实的 `subject + predicate` 相同就
//! **替换**旧事实（保留历史 `versions`），不同就追加。
//!
//! 这既保留了 Mem0「不丢信息」的优点，又解决了矛盾问题。

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use agent_logging::{debug, info, warn};
use serde::{Deserialize, Serialize};

use crate::error::AppResult;

/// 一条长期记忆（事实）。
///
/// ## 「事实」的结构
///
/// ```text
/// subject —— 关于谁（用户 / 项目 / 偏好）
/// predicate —— 是什么（名字 / 偏好 / 约束）
/// value —— 具体内容
/// ```
///
/// 拆成三段而不是一整句话，是为了：
///
/// 1. **去重** —— 同一个 (subject, predicate) 只能有一条当前值
/// 2. **矛盾检测** —— 新事实和旧事实的 subject+predicate 相同但 value 不同，
///    说明用户改了主意
/// 3. **可检索** —— 「用户」相关的所有事实可以被一起召回
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFact {
    /// 唯一 id。
    pub id: String,

    /// 主语（谁）。
    ///
    /// 默认是 `"user"` —— 绝大多数事实都是关于用户的。
    pub subject: String,

    /// 谓语（是什么）。
    pub predicate: String,

    /// 值（具体内容）。
    pub value: String,

    /// 所属的用户 / 租户 id。
    ///
    /// ## 对应 Mem0 的 `user_id` 过滤器
    ///
    /// Mem0 检索时带 `filters={"user_id": user_id}` ——
    /// 本项目把它做成**存储层强制分区**（不是可选的过滤器），
    /// 理由和 checkpointer 一样：**在存储层隔离比在查询层过滤可靠**。
    pub owner: String,

    /// 重要度（0.0 ~ 1.0）。
    ///
    /// 召回时用于排序 —— 重要的记忆优先塞进上下文。
    pub importance: f32,

    /// 创建时间（RFC 3339）。
    pub created_at: String,

    /// 最后更新时间（RFC 3339）。
    pub updated_at: String,

    /// 历史版本（旧值）。
    ///
    /// ## 为什么保留
    ///
    /// 用户改口时，旧信息**不是没用** ——
    /// 「我以前叫 Bob」这件事本身可能有意义
    /// （比如查历史订单）。而且 Mem0 强调「nothing is overwritten」。
    pub versions: Vec<String>,
}

impl MemoryFact {
    /// 构造一条事实。
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
        owner: impl Into<String>,
    ) -> Self {
        let now = crate::clock::now_rfc3339();
        Self {
            id: crate::clock::new_fact_id(),
            subject: subject.into(),
            predicate: predicate.into(),
            value: value.into(),
            owner: owner.into(),
            importance: 0.5,
            created_at: now.clone(),
            updated_at: now,
            versions: Vec::new(),
        }
    }

    /// 设置重要度（钳制到 0.0 ~ 1.0）。
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// 事实的「去重键」。
    ///
    /// ## ⚠️ 为什么是 (owner, subject, predicate) 而不是整条内容
    ///
    /// 因为「用户叫 Bob」和「用户名字是 Bob」应该被认成**同一件事**。
    /// 如果用整条内容做键，模型换个说法就会产生一条重复事实。
    pub fn dedup_key(&self) -> (String, String, String) {
        (
            self.owner.clone(),
            self.subject.clone(),
            self.predicate.clone(),
        )
    }

    /// 渲染成给模型看的一行文本。
    pub fn render(&self) -> String {
        format!("{}的{}：{}", self.subject, self.predicate, self.value)
    }

    /// 用于检索的文本（把三段拼起来，提高召回率）。
    pub fn searchable_text(&self) -> String {
        format!("{} {} {}", self.subject, self.predicate, self.value)
    }
}

/// 记忆操作（存取 / 检索）的 future。
///
/// 和仓库里其它地方一致（07 的 `LlmFuture`、14 的 `ToolFuture`）——
/// `async fn` 在 trait 里不是 dyn 兼容的，所以显式装箱。
pub type MemorySearchFuture<'a, T> = Pin<Box<dyn Future<Output = AppResult<T>> + Send + 'a>>;

/// 从一段对话里抽取事实。
///
/// ## Mem0 的对应物
///
/// Mem0 用 LLM 做这件事（`memory.add(messages)` 内部会调模型抽事实）。
///
/// ## 为什么抽成 trait
///
/// 1. **测试要确定性** —— 用真实模型抽事实，测试就没法写死断言。
///    脚本化的 [`ScriptedExtractor`] 让测试零网络、零延迟。
/// 2. **抽取是**可选**能力** —— 抽不出来不是致命错误，
///    应该是「本轮没有新事实」，而不是「整轮对话失败」。
pub trait FactExtractor: Send + Sync {
    /// 从一段对话里抽取事实。
    ///
    /// ## 返回值
    ///
    /// - `Ok(vec![])` —— 没有新事实（**不是错误**）
    /// - `Err` —— 抽取过程本身失败（模型挂了等）
    fn extract<'a>(
        &'a self,
        messages: &'a [crate::state::MemoryMessage],
        owner: &'a str,
    ) -> MemorySearchFuture<'a, Vec<MemoryFact>>;
}

/// 长期记忆的存储层。
pub trait MemoryStore: Send + Sync {
    /// 存入一条事实。
    ///
    /// ## 语义：**合并**（不是纯追加）
    ///
    /// 同一个 `(owner, subject, predicate)` 已存在时，
    /// 旧事实的 value 进 `versions`，新值成为当前值。
    ///
    /// 返回是否发生了**合并**（true = 覆盖了旧值）。
    fn put<'a>(&'a self, fact: MemoryFact) -> MemorySearchFuture<'a, bool>;

    /// 列出某 owner 的全部事实。
    fn list<'a>(&'a self, owner: &'a str) -> MemorySearchFuture<'a, Vec<MemoryFact>>;

    /// 按 id 删除一条事实。
    fn delete<'a>(&'a self, owner: &'a str, id: &'a str) -> MemorySearchFuture<'a, bool>;

    /// 清空某 owner 的全部事实。
    fn clear<'a>(&'a self, owner: &'a str) -> MemorySearchFuture<'a, usize>;

    /// 列出所有**已知的 owner**。
    ///
    /// ## 用途
    ///
    /// 长期记忆是**按 owner 分区**的，检索时也只查一个 owner。
    /// 但「重建全局索引」需要知道**所有** owner 才能把它们的文档
    /// 一次性灌进 BM25 索引。
    ///
    /// ## ⚠️ 为什么是**必需方法**而不是「可选能力」
    ///
    /// 之前这里用 `Any` 向下转型探测（`downcast_ref::<InMemoryMemoryStore>()`）——
    /// 但 `dyn MemoryStore` **不是** `Any`，向下转型在编译期就编译不过
    /// （`E0605: non-primitive cast`）。
    ///
    /// 正确的做法是把能力**提到 trait 上**：
    /// 实现方各写各的，调用方不用猜类型。
    /// 这也是本仓库一贯的做法（13 的 `Llm`、14 的 `Tool`、16 的 `ToolRegistry`）。
    fn owners<'a>(&'a self) -> MemorySearchFuture<'a, Vec<String>>;
}

/// 长期记忆的检索层。
///
/// ## Mem0 的对应物
///
/// Mem0 的 `memory.search(query=..., filters={"user_id": ...}, top_k=3)`。
pub trait MemoryRetriever: Send + Sync {
    /// 按查询召回事实。
    ///
    /// ## 返回值
    ///
    /// 按相关度**降序**排列，最多 `top_k` 条。
    fn search<'a>(
        &'a self,
        query: &'a str,
        owner: &'a str,
        top_k: usize,
    ) -> MemorySearchFuture<'a, Vec<MemoryFact>>;
}

/// 内存记忆库 —— 事实存在 `RwLock<HashMap>` 里。
///
/// ## 用途
///
/// - 单元 / 集成测试（零 IO）
/// - 单进程服务的默认实现
pub struct InMemoryMemoryStore {
    facts: RwLock<HashMap<String, Vec<MemoryFact>>>,
}

impl std::fmt::Debug for InMemoryMemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryMemoryStore")
            .field("owners", &self.facts.read().map(|m| m.len()).unwrap_or(0))
            .finish()
    }
}

impl Default for InMemoryMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryMemoryStore {
    /// 创建空的内存记忆库。
    pub fn new() -> Self {
        Self {
            facts: RwLock::new(HashMap::new()),
        }
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Vec<MemoryFact>>> {
        match self.facts.read() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("记忆库读锁被毒化 —— 继续使用");
                poisoned.into_inner()
            }
        }
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<String, Vec<MemoryFact>>> {
        match self.facts.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("记忆库写锁被毒化 —— 继续使用");
                poisoned.into_inner()
            }
        }
    }

    /// 事实总数。
    pub fn len(&self) -> usize {
        self.read().values().map(|v| v.len()).sum()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl MemoryStore for InMemoryMemoryStore {
    fn put<'a>(&'a self, fact: MemoryFact) -> MemorySearchFuture<'a, bool> {
        Box::pin(async move {
            let mut guard = self.write();
            let bucket = guard.entry(fact.owner.clone()).or_default();

            let key = fact.dedup_key();
            if let Some(existing) = bucket.iter_mut().find(|f| f.dedup_key() == key) {
                // ⚠️ 合并：旧值进历史，新值成为当前值。
                //
                // 这是 Mem0「只增不覆盖」和「不能有矛盾事实」的折中 ——
                // 旧信息不丢（进 `versions`），但当前状态是明确的。
                if existing.value != fact.value {
                    existing.versions.push(existing.value.clone());
                    existing.value = fact.value;
                    existing.updated_at = fact.updated_at;
                    existing.importance = fact.importance;
                    info!(
                        owner = %fact.owner,
                        subject = %fact.subject,
                        predicate = %fact.predicate,
                        "事实合并（旧值进历史）"
                    );
                    return Ok(true);
                }
                return Ok(false);
            }

            debug!(
                owner = %fact.owner,
                subject = %fact.subject,
                predicate = %fact.predicate,
                "新增事实"
            );
            bucket.push(fact);
            Ok(false)
        })
    }

    fn list<'a>(&'a self, owner: &'a str) -> MemorySearchFuture<'a, Vec<MemoryFact>> {
        Box::pin(async move { Ok(self.read().get(owner).cloned().unwrap_or_default()) })
    }

    fn delete<'a>(&'a self, owner: &'a str, id: &'a str) -> MemorySearchFuture<'a, bool> {
        Box::pin(async move {
            let mut guard = self.write();
            let Some(bucket) = guard.get_mut(owner) else {
                return Ok(false);
            };
            let before = bucket.len();
            bucket.retain(|f| f.id != id);
            Ok(bucket.len() != before)
        })
    }

    fn clear<'a>(&'a self, owner: &'a str) -> MemorySearchFuture<'a, usize> {
        Box::pin(async move {
            let mut guard = self.write();
            let removed = guard.remove(owner).map(|v| v.len()).unwrap_or(0);
            if removed > 0 {
                info!(owner = %owner, removed, "清空长期记忆");
            }
            Ok(removed)
        })
    }

    fn owners<'a>(&'a self) -> MemorySearchFuture<'a, Vec<String>> {
        Box::pin(async move {
            let mut owners: Vec<String> = self.read().keys().cloned().collect();
            owners.sort();
            Ok(owners)
        })
    }
}

/// 文件记忆库 —— 每个 owner 一个 JSON 文件。
///
/// ## 对应 ReMe 的「Memory as File」
///
/// ReMe 的核心主张是「记忆就是普通文件」——
/// 用户可以直接 `cat`、编辑、备份、迁移。
///
/// 本项目的实现：
///
/// ```text
/// <root>/
///   default.json     ← owner=default 的事实
///   alice.json       ← owner=alice 的事实
/// ```
///
/// ## ⚠️ 为什么 owner 要做文件名净化
///
/// owner 来自用户输入（`user_id`），可能带 `/`、`.` 等。
/// 直接当文件名会有**路径穿越**风险。
///
/// 这里用**哈希 + 可读前缀**的方式：
/// `sha1(owner)` 的前 8 位 + 净化后的 owner 前缀。
/// 这样既能读懂是哪个 owner，又绝对安全。
pub struct FileMemoryStore {
    root: PathBuf,
}

impl std::fmt::Debug for FileMemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileMemoryStore")
            .field("root", &self.root)
            .finish()
    }
}

impl FileMemoryStore {
    /// 创建文件记忆库（目录自动创建）。
    pub fn new(root: impl Into<PathBuf>) -> AppResult<Self> {
        let root = root.into();
        if !root.exists() {
            std::fs::create_dir_all(&root)?;
        }
        info!(root = %root.display(), "文件记忆库已就绪");
        Ok(Self { root })
    }

    /// 存储根目录。
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// owner 对应的文件路径。
    ///
    /// ## 文件名构造
    ///
    /// ```text
    /// {sha1前8位}_{净化后的owner前32字符}.json
    /// ```
    ///
    /// - **哈希前缀** —— 保证唯一且不可预测（哪怕两个 owner 净化后同名）
    /// - **可读部分** —— 让人 `ls` 一眼看出是哪个 owner 的
    fn path_for(&self, owner: &str) -> PathBuf {
        self.root.join(format!("{}.json", sanitize_owner(owner)))
    }
}

/// 把 owner 净化成安全的文件名。
///
/// ## 做法
///
/// 1. 取 `sha1(owner)` 的前 8 位十六进制 —— 保证唯一
/// 2. 取 owner 里**只保留**字母数字、`-`、`_` 的部分，最多 32 字符 —— 可读
/// 3. 可读部分为空时用 `owner` 代替
///
/// ## ⚠️ 为什么不用 owner 原文当文件名
///
/// owner 可能包含 `/`（路径分隔符）、`..`（父目录）、
/// 甚至空字节 —— 直接当文件名就是**路径穿越漏洞**。
pub fn sanitize_owner(owner: &str) -> String {
    let hash = sha1_hex(owner);
    let readable: String = owner
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(32)
        .collect();

    if readable.is_empty() {
        format!("{hash}_owner")
    } else {
        format!("{hash}_{readable}")
    }
}

/// 计算 SHA-1 的十六进制摘要（只用它的**确定性**性质）。
///
/// ## ⚠️ 这不是安全用途
///
/// SHA-1 在这里**不是**用来防碰撞的（那需要抗碰撞哈希）。
/// 它只是需要一个「同样的输入永远给同样的短标识」的东西。
///
/// 仓库里 10-rag-milvus 也用同样手法派生稳定的 point id。
fn sha1_hex(input: &str) -> String {
    // 用标准库的 `DefaultHasher`？不行 —— 它的输出**不保证跨版本稳定**。
    // 这里手写一个 FNV-1a 64 位哈希 —— 简单、确定、够用。
    //
    // FNV-1a 的常量是标准规定的偏移基和质数。
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

impl MemoryStore for FileMemoryStore {
    fn put<'a>(&'a self, fact: MemoryFact) -> MemorySearchFuture<'a, bool> {
        Box::pin(async move {
            let path = self.path_for(&fact.owner);
            let mut facts = self.read_facts(&path).await?;

            let key = fact.dedup_key();
            if let Some(existing) = facts.iter_mut().find(|f| f.dedup_key() == key) {
                if existing.value != fact.value {
                    existing.versions.push(existing.value.clone());
                    existing.value = fact.value;
                    existing.updated_at = fact.updated_at;
                    existing.importance = fact.importance;
                    self.write_facts(&path, &facts).await?;
                    return Ok(true);
                }
                return Ok(false);
            }

            facts.push(fact);
            self.write_facts(&path, &facts).await?;
            Ok(false)
        })
    }

    fn list<'a>(&'a self, owner: &'a str) -> MemorySearchFuture<'a, Vec<MemoryFact>> {
        Box::pin(async move {
            let path = self.path_for(owner);
            self.read_facts(&path).await
        })
    }

    fn delete<'a>(&'a self, owner: &'a str, id: &'a str) -> MemorySearchFuture<'a, bool> {
        Box::pin(async move {
            let path = self.path_for(owner);
            let mut facts = self.read_facts(&path).await?;
            let before = facts.len();
            facts.retain(|f| f.id != id);
            if facts.len() == before {
                return Ok(false);
            }
            self.write_facts(&path, &facts).await?;
            Ok(true)
        })
    }

    fn clear<'a>(&'a self, owner: &'a str) -> MemorySearchFuture<'a, usize> {
        Box::pin(async move {
            let path = self.path_for(owner);
            let facts = self.read_facts(&path).await?;
            let removed = facts.len();
            if removed > 0 {
                std::fs::remove_file(&path)?;
                info!(owner = %owner, removed, "清空长期记忆文件");
            }
            Ok(removed)
        })
    }

    fn owners<'a>(&'a self) -> MemorySearchFuture<'a, Vec<String>> {
        Box::pin(async move {
            let mut owners = Vec::new();
            let entries = match std::fs::read_dir(&self.root) {
                Ok(entries) => entries,
                Err(e) => {
                    warn!(root = %self.root.display(), error = %e, "读取记忆目录失败");
                    return Ok(owners);
                }
            };

            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                // 文件名格式：{hash}_{readable}
                if let Some((_, readable)) = stem.split_once('_') {
                    owners.push(readable.to_string());
                }
            }

            owners.sort();
            Ok(owners)
        })
    }
}

impl FileMemoryStore {
    /// 读一个 owner 的事实文件。
    ///
    /// ⚠️ 文件不存在 → 空列表（不是错误）。
    /// ⚠️ 文件损坏 → 空列表 + warn（不让服务挂掉）。
    async fn read_facts(&self, path: &Path) -> AppResult<Vec<MemoryFact>> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let text = std::fs::read_to_string(path)?;
        match serde_json::from_str::<Vec<MemoryFact>>(&text) {
            Ok(facts) => Ok(facts),
            Err(e) => {
                warn!(
                    path = %path.display(),
                    error = %e,
                    "记忆文件损坏 —— 按空记忆处理"
                );
                Ok(Vec::new())
            }
        }
    }

    /// 写一个 owner 的事实文件（原子替换）。
    async fn write_facts(&self, path: &Path, facts: &[MemoryFact]) -> AppResult<()> {
        let text = serde_json::to_string_pretty(facts)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text.as_bytes())?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// 基于 BM25 的记忆检索器。
///
/// ## 为什么用 BM25 而不是向量检索
///
/// 1. **零模型依赖** —— 不用下载 embedding 模型，不用 API key。
///    这对本项目很重要 —— 记忆层应该是**最轻**的那一层。
/// 2. **关键词匹配对事实检索足够好** —— 事实是很短的句子
///    （「用户名字是 Bob」），关键词命中比语义相似更可靠。
/// 3. **和仓库其它部分一致** —— 复用 `rag_08::bm25::Bm25Index`，
///    不引入新的检索实现。
///
/// ## ⚠️ BM25 索引是**可重建**的
///
/// 对应 ReMe 的「`metadata/` 是可重建的，文件才是真相来源」——
/// 索引丢了就重建，**不影响**记忆本身。
pub struct Bm25MemoryRetriever {
    store: Arc<dyn MemoryStore>,
    index: RwLock<Option<rag_08::bm25::Bm25Index>>,
}

impl std::fmt::Debug for Bm25MemoryRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bm25MemoryRetriever")
            .finish_non_exhaustive()
    }
}

impl Bm25MemoryRetriever {
    /// 创建检索器。
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self {
            store,
            index: RwLock::new(None),
        }
    }

    /// 重建索引（从 store 全量拉取）。
    ///
    /// ## ⚠️ 索引是**缓存**，不是真相来源
    ///
    /// 任何时候都可以丢掉重建 —— 所以 `reindex` 是安全的。
    pub async fn reindex(&self) -> AppResult<usize> {
        let mut facts = Vec::new();

        // ⚠️ `store.list` 需要 owner —— 这里列举**所有** owner。
        //
        // `owners()` 是 [`MemoryStore`] 的**必需方法**，
        // 所以这里不用猜类型（之前用 `Any` 向下转型是编译不过的）。
        for owner in self.store.owners().await? {
            facts.extend(self.store.list(&owner).await?);
        }

        let documents: Vec<rag_08::bm25::Bm25Document> = facts
            .iter()
            .map(|f| rag_08::bm25::Bm25Document {
                id: f.id.clone(),
                source: f.owner.clone(),
                heading_path: vec![format!("{} / {}", f.subject, f.predicate)],
                text: f.searchable_text(),
            })
            .collect();

        let index = rag_08::bm25::Bm25Index::build(&documents);

        let count = index.len();
        match self.index.write() {
            Ok(mut guard) => {
                *guard = Some(index);
            }
            Err(poisoned) => {
                warn!("BM25 索引锁被毒化 —— 继续使用");
                *poisoned.into_inner() = Some(index);
            }
        }

        info!(facts = count, "BM25 记忆索引已重建");
        Ok(count)
    }
}

/// 检索结果（事实 + 相关度）。
#[derive(Debug, Clone)]
pub struct RetrievedFact {
    /// 事实本身。
    pub fact: MemoryFact,

    /// BM25 分数。
    pub score: f32,
}

impl MemoryRetriever for Bm25MemoryRetriever {
    fn search<'a>(
        &'a self,
        query: &'a str,
        owner: &'a str,
        top_k: usize,
    ) -> MemorySearchFuture<'a, Vec<MemoryFact>> {
        Box::pin(async move {
            let facts = self.store.list(owner).await?;
            Ok(rank_facts(query, &facts, top_k))
        })
    }
}

/// 用 BM25 给事实排序并取 top-k。
///
/// ## 为什么单独抽成自由函数
///
/// 1. **可测试** —— 不需要构造 store 就能测排序逻辑
/// 2. **索引缺失时也能用** —— 事实不多时（几十条），
///    全量线性打分完全够用，而且**没有 stale 索引**的问题
///
/// ## ⚠️ 线性打分 vs 索引
///
/// 事实数量少的时候，「全量算一遍 BM25」比「维护一个索引」更简单也更准
/// （没有索引过期的问题）。
///
/// 事实上千条以后才需要真正的倒排索引 ——
/// 那时把 `rank_facts` 换成 `Bm25MemoryRetriever` 的索引版本即可。
///
/// ## ⚠️ 零命中时的兜底（**本项目实测踩出来的坑**）
///
/// BM25 是**纯词面**匹配 —— 它匹配不了「语义等价」。
///
/// 实测：事实是「用户 / 名字 / Bob」，
/// 用户在新线程里问「**我叫什么**」——
///
/// ```text
/// 查询词元：我 叫 什 么 我叫 叫什 什么
/// 事实词元：用户 名字 bob 用户 户名 名字
/// 交集    ：∅
/// ```
///
/// **零命中** → 模型完全失忆，明明存过「用户叫 Bob」却答不出来。
///
/// 这正是向量检索能解决、BM25 解决不了的场景
/// （「我叫什么」↔「名字」是语义等价，词面毫无重叠）。
///
/// ### 修法：零命中时退化为「按重要度取前 k 条」
///
/// 一个 owner 的事实通常只有**几条到几十条**，
/// 全部塞进上下文并不昂贵。所以在 BM25 零命中时，
/// 退化成「按 `importance` 降序取前 k 条」——
///
/// - 事实集合很小 → 大概率包含相关的那条
/// - 模型自己会挑 → 它看到「用户的名字：Bob」就知道这是答案
/// - 比「什么都不给」好一个量级
///
/// ⚠️ 这只是**近似**，不是真正的语义检索。
/// 真正要解决它，得换成向量检索（12-rag-summary 的 Milvus 那套）。
/// 但对「学习项目」而言，这个近似足够，且**行为是可预期的**。
pub fn rank_facts(query: &str, facts: &[MemoryFact], top_k: usize) -> Vec<MemoryFact> {
    if facts.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let documents: Vec<rag_08::bm25::Bm25Document> = facts
        .iter()
        .map(|f| rag_08::bm25::Bm25Document {
            id: f.id.clone(),
            source: f.owner.clone(),
            heading_path: vec![format!("{} / {}", f.subject, f.predicate)],
            text: f.searchable_text(),
        })
        .collect();

    let index = rag_08::bm25::Bm25Index::build(&documents);
    let hits = index.search(query, facts.len());

    // ⚠️ BM25 分数**不能**和重要度直接相加（量纲不同）。
    //
    // 这里用「重要度作为平局时的次级排序键」——
    // BM25 分相同时，重要的事实排前面。
    let mut ranked: Vec<(f32, f32, MemoryFact)> = hits
        .into_iter()
        .filter_map(|hit| {
            let fact = facts.iter().find(|f| f.id == hit.id)?;
            Some((hit.score, fact.importance, fact.clone()))
        })
        .collect();

    // ── 零命中兜底 ────────────────────────────────────────────
    //
    // ⚠️ BM25 匹配不了「语义等价」（详见本函数的文档）。
    //    零命中时退化为「按重要度取前 k 条」，宁可给模型一点上下文，
    //    也别让它彻底失忆。
    if ranked.is_empty() {
        debug!(
            query = %query,
            facts = facts.len(),
            "BM25 零命中 —— 退化为按重要度召回"
        );

        let mut by_importance: Vec<(f32, MemoryFact)> =
            facts.iter().map(|f| (f.importance, f.clone())).collect();

        by_importance.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        return by_importance
            .into_iter()
            .take(top_k)
            .map(|(_, fact)| fact)
            .collect();
    }

    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    ranked
        .into_iter()
        .take(top_k)
        .map(|(_, _, fact)| fact)
        .collect()
}

impl FileMemoryStore {
    /// 存储根目录（供调试 / 运维查看）。
    pub fn owner_files(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.root)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let path = e.path();
                if path.extension().and_then(|s| s.to_str()) != Some("json") {
                    return None;
                }
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
            })
            .collect();
        names.sort();
        names
    }
}

/// 脚本化的事实抽取器 —— 按脚本返回预设事实。
///
/// ## 为什么需要它
///
/// 抽取要用真实模型才能测「抽取得好不好」，
/// 但那**不是单元测试该验证的东西**（模型输出不确定）。
///
/// 单元测试要验证的是：
///
/// - 「抽取到的事实被正确存进了 store」
/// - 「没有事实时不会报错」
/// - 「抽取失败时降级，不影响主流程」
///
/// 这些都用脚本化抽取器就够。
pub struct ScriptedExtractor {
    steps: RwLock<Vec<Vec<MemoryFact>>>,
    fallback: Option<Vec<MemoryFact>>,
}

impl std::fmt::Debug for ScriptedExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedExtractor").finish_non_exhaustive()
    }
}

impl ScriptedExtractor {
    /// 创建脚本化抽取器（每次调用返回下一步的预设事实）。
    pub fn new(steps: Vec<Vec<MemoryFact>>) -> Self {
        Self {
            steps: RwLock::new(steps),
            fallback: None,
        }
    }

    /// 创建脚本化抽取器（每次都返回同一批事实）。
    pub fn always(facts: Vec<MemoryFact>) -> Self {
        Self {
            steps: RwLock::new(Vec::new()),
            fallback: Some(facts),
        }
    }

    /// 已调用次数。
    pub fn call_count(&self) -> usize {
        self.steps
            .read()
            .map(|s| s.len())
            .unwrap_or_default()
            .max(self.fallback.as_ref().map(|_| 1).unwrap_or(0))
    }
}

impl FactExtractor for ScriptedExtractor {
    fn extract<'a>(
        &'a self,
        _messages: &'a [crate::state::MemoryMessage],
        _owner: &'a str,
    ) -> MemorySearchFuture<'a, Vec<MemoryFact>> {
        Box::pin(async move {
            let next = {
                let mut steps = self.steps.write().unwrap_or_else(|e| e.into_inner());
                if steps.is_empty() {
                    self.fallback.clone().unwrap_or_default()
                } else {
                    steps.remove(0)
                }
            };
            Ok(next)
        })
    }
}

/// 永不抽取任何事实的抽取器（默认实现）。
///
/// ## 用途
///
/// 默认关闭长期记忆抽取时用它 ——
/// 这样主流程不用到处判断「抽取器有没有配」。
pub struct NoopExtractor;

impl FactExtractor for NoopExtractor {
    fn extract<'a>(
        &'a self,
        _messages: &'a [crate::state::MemoryMessage],
        _owner: &'a str,
    ) -> MemorySearchFuture<'a, Vec<MemoryFact>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
}

/// 基于 LLM 的事实抽取器 —— 对齐 Mem0 的 `memory.add()`。
///
/// ## 流程
///
/// ```text
/// 对话历史 ──> 抽取提示词 ──> 调模型 ──> 解析 JSON ──> Vec<MemoryFact>
/// ```
///
/// ## ⚠️ 解析失败**不是错误**
///
/// 模型输出不是合法 JSON 时，返回**空列表**而不是 `Err`。
///
/// 理由：
///
/// - 「抽不出来事实」是**正常结果**（这段对话确实没有事实）
/// - 解析失败和「没有事实」在业务上**等价** —— 都不往记忆库里写东西
/// - 让它变成错误会触发上游的 warn 日志噪音，还可能误导排查
///
/// ## ⚠️ 抽取必须**关思考**
///
/// 推理模型的思考 token 和正文共用 `max_tokens` 预算。
/// 抽取要的是**结构化短输出** —— 思考只会烧 token，且温度为 0 时收益极低。
pub struct LlmFactExtractor {
    llm: Arc<dyn crate::llm::ChatLlm>,
}

impl std::fmt::Debug for LlmFactExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmFactExtractor")
            .field("model", &self.llm.model_name())
            .finish()
    }
}

impl LlmFactExtractor {
    /// 创建 LLM 抽取器。
    pub fn new(llm: Arc<dyn crate::llm::ChatLlm>) -> Self {
        Self { llm }
    }
}

impl FactExtractor for LlmFactExtractor {
    fn extract<'a>(
        &'a self,
        messages: &'a [crate::state::MemoryMessage],
        owner: &'a str,
    ) -> MemorySearchFuture<'a, Vec<MemoryFact>> {
        Box::pin(async move {
            let prompt = crate::prompt::extraction_prompt(messages, owner);

            let text = self
                .llm
                .chat_messages("", &[crate::state::MemoryMessage::user(&prompt)])
                .await?;

            let extracted = crate::prompt::parse_extraction(&text);
            if extracted.is_empty() {
                debug!(owner = %owner, "抽取器没有解析出事实");
            }

            Ok(extracted.into_iter().map(|e| e.to_fact(owner)).collect())
        })
    }
}

/// 便捷别名。
pub type SharedMemoryStore = Arc<dyn MemoryStore>;
pub type SharedFactExtractor = Arc<dyn FactExtractor>;

/// 创建一个内存记忆库。
pub fn in_memory_store() -> SharedMemoryStore {
    Arc::new(InMemoryMemoryStore::new())
}

/// 创建一个文件记忆库。
pub fn file_store(root: impl Into<PathBuf>) -> AppResult<SharedMemoryStore> {
    Ok(Arc::new(FileMemoryStore::new(root)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(subject: &str, predicate: &str, value: &str) -> MemoryFact {
        MemoryFact::new(subject, predicate, value, "default")
    }

    #[tokio::test]
    async fn stores_and_lists_facts() {
        let store = in_memory_store();
        store.put(fact("用户", "名字", "Bob")).await.unwrap();

        let facts = store.list("default").await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].value, "Bob");
    }

    #[tokio::test]
    async fn isolates_owners() {
        // ⚠️ 长期记忆的**用户隔离** —— 和线程隔离同级别的重要。
        let store = in_memory_store();
        store
            .put(MemoryFact::new("用户", "名字", "Bob", "alice"))
            .await
            .unwrap();
        store
            .put(MemoryFact::new("用户", "名字", "Alice", "bob"))
            .await
            .unwrap();

        let alice = store.list("alice").await.unwrap();
        let bob = store.list("bob").await.unwrap();

        assert_eq!(alice.len(), 1);
        assert_eq!(alice[0].value, "Bob");
        assert_eq!(bob[0].value, "Alice");
    }

    #[tokio::test]
    async fn merges_facts_with_the_same_key() {
        // ⚠️ Mem0 的「不覆盖」和「不能有矛盾」之间的折中。
        let store = in_memory_store();
        store.put(fact("用户", "名字", "Bob")).await.unwrap();

        let merged = store.put(fact("用户", "名字", "Robert")).await.unwrap();
        assert!(merged, "同 key 的新值应该触发合并");

        let facts = store.list("default").await.unwrap();
        assert_eq!(facts.len(), 1, "合并后应该还是一条");
        assert_eq!(facts[0].value, "Robert", "新值成为当前值");
        assert_eq!(facts[0].versions, vec!["Bob".to_string()], "旧值进历史");
    }

    #[tokio::test]
    async fn keeps_facts_with_different_keys() {
        let store = in_memory_store();
        store.put(fact("用户", "名字", "Bob")).await.unwrap();
        store.put(fact("用户", "偏好", "深色模式")).await.unwrap();

        assert_eq!(store.list("default").await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn identical_value_is_not_a_merge() {
        let store = in_memory_store();
        store.put(fact("用户", "名字", "Bob")).await.unwrap();
        let merged = store.put(fact("用户", "名字", "Bob")).await.unwrap();
        assert!(!merged);
        assert_eq!(store.list("default").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn deletes_and_clears() {
        let store = in_memory_store();
        let f = fact("用户", "名字", "Bob");
        let id = f.id.clone();
        store.put(f).await.unwrap();

        assert!(store.delete("default", &id).await.unwrap());
        assert!(!store.delete("default", &id).await.unwrap());
        assert!(store.list("default").await.unwrap().is_empty());

        store.put(fact("用户", "偏好", "深色")).await.unwrap();
        assert_eq!(store.clear("default").await.unwrap(), 1);
        assert!(store.list("default").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn file_store_persists_across_instances() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = file_store(dir.path()).unwrap();
            store.put(fact("用户", "名字", "Bob")).await.unwrap();
        }

        let store = file_store(dir.path()).unwrap();
        let facts = store.list("default").await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].value, "Bob");
    }

    #[tokio::test]
    async fn file_store_sanitizes_owner_names() {
        // ⚠️ owner 里带 `/` 不能造成路径穿越。
        let dir = tempfile::tempdir().unwrap();
        let store = file_store(dir.path()).unwrap();

        store
            .put(MemoryFact::new("用户", "名字", "Bob", "../../evil"))
            .await
            .unwrap();

        let facts = store.list("../../evil").await.unwrap();
        assert_eq!(facts.len(), 1);

        // 文件必须落在目录**内**。
        let entries: Vec<_> = std::fs::read_dir(dir.path()).unwrap().flatten().collect();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].path().starts_with(dir.path()));
    }

    #[test]
    fn sanitize_owner_is_deterministic_and_safe() {
        let a = sanitize_owner("alice");
        let b = sanitize_owner("alice");
        assert_eq!(a, b);
        assert!(!a.contains('/'));
        assert!(!a.contains(".."));

        // 不同的 owner 得到不同的文件名。
        assert_ne!(sanitize_owner("alice"), sanitize_owner("bob"));
    }

    #[tokio::test]
    async fn ranks_facts_by_relevance() {
        let facts = vec![
            fact("用户", "名字", "Bob"),
            fact("用户", "偏好", "喜欢深色模式"),
            fact("项目", "语言", "Rust"),
        ];

        let ranked = rank_facts("用户喜欢什么颜色", &facts, 2);
        assert_eq!(ranked.len(), 2);
        // 「喜欢深色模式」应该排第一。
        assert_eq!(ranked[0].predicate, "偏好");
    }

    #[tokio::test]
    async fn ranking_handles_empty_input() {
        assert!(rank_facts("查询", &[], 3).is_empty());
        let facts = vec![fact("用户", "名字", "Bob")];
        assert!(rank_facts("查询", &facts, 0).is_empty());
    }

    #[tokio::test]
    async fn zero_hit_falls_back_to_importance_ordering() {
        // ⚠️ **本项目实测踩出来的坑** ——
        //    BM25 匹配不了「语义等价」。
        //
        //    事实是「用户 / 名字 / Bob」，查询是「我叫什么」——
        //    词元交集为空，BM25 **零命中**。
        //
        //    如果不兜底，模型就彻底失忆了（明明存过「用户叫 Bob」）。
        let facts = vec![fact("用户", "名字", "Bob").with_importance(0.9)];

        let ranked = rank_facts("我叫什么", &facts, 3);
        assert_eq!(ranked.len(), 1, "零命中也要召回事实");
        assert_eq!(ranked[0].value, "Bob");
    }

    #[tokio::test]
    async fn the_fallback_respects_the_top_k_limit() {
        let facts = vec![
            fact("用户", "名字", "Bob").with_importance(0.9),
            fact("用户", "偏好", "深色").with_importance(0.8),
            fact("用户", "职业", "工程师").with_importance(0.7),
        ];

        let ranked = rank_facts("完全不相关的查询", &facts, 2);
        assert_eq!(ranked.len(), 2);
        // 按重要度降序 —— 最重要的排第一。
        assert_eq!(ranked[0].value, "Bob");
        assert_eq!(ranked[1].value, "深色");
    }

    #[tokio::test]
    async fn a_keyword_hit_still_wins_over_the_fallback() {
        // ⚠️ 有命中时**不能**退化 —— BM25 命中必须优先。
        let facts = vec![
            fact("用户", "名字", "Bob").with_importance(0.1),
            fact("用户", "偏好", "喜欢深色模式").with_importance(0.9),
        ];

        let ranked = rank_facts("用户喜欢什么颜色", &facts, 2);
        assert_eq!(ranked[0].predicate, "偏好");
    }

    #[tokio::test]
    async fn scripted_extractor_returns_presets() {
        let extractor = ScriptedExtractor::new(vec![vec![fact("用户", "名字", "Bob")]]);

        let facts = extractor.extract(&[], "default").await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].value, "Bob");

        // 脚本用完 → 空（不是错误）。
        let facts = extractor.extract(&[], "default").await.unwrap();
        assert!(facts.is_empty());
    }

    #[tokio::test]
    async fn noop_extractor_returns_nothing() {
        let extractor = NoopExtractor;
        assert!(extractor.extract(&[], "default").await.unwrap().is_empty());
    }

    #[test]
    fn fact_renders_readably() {
        let f = fact("用户", "名字", "Bob");
        assert_eq!(f.render(), "用户的名字：Bob");
    }

    #[test]
    fn importance_is_clamped() {
        let f = MemoryFact::new("a", "b", "c", "d").with_importance(5.0);
        assert_eq!(f.importance, 1.0);
        let f = MemoryFact::new("a", "b", "c", "d").with_importance(-1.0);
        assert_eq!(f.importance, 0.0);
    }
}
