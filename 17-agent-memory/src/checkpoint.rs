//! Checkpointer —— 状态持久化层。
//!
//! ## 对应 LangChain 的什么
//!
//! LangChain 的 short-term memory 文档里：
//!
//! > State is persisted to a database (or memory) using a checkpointer
//! > so the thread can be resumed at any time.
//!
//! 所以 `checkpointer` 就是「**线程状态 ↔ 存储**」的抽象。
//! 本项目定义 [`Checkpointer`] trait，然后给两个实现：
//!
//! | 实现 | 存储 | 用途 |
//! |---|---|---|
//! | [`InMemoryCheckpointer`] | `HashMap` | 测试、临时会话 |
//! | [`FileCheckpointer`] | JSON 文件（每个线程一个） | 进程重启后还能恢复 |
//!
//! ## 为什么状态要**异步**读写
//!
//! 文件 I/O 是阻塞操作。放进 async 函数里会卡住 tokio 工作线程。
//! 所以 trait 的方法都是 `async fn`。
//!
//! 但 `async fn` 在 trait 里**不是 dyn 兼容**的 ——
//! 而 `AppState` 里要存 `Arc<dyn Checkpointer>`。
//!
//! 解法和仓库里其它地方一致（07 的 `LlmFuture`、14 的 `ToolFuture`）：
//! **显式装箱** `Pin<Box<dyn Future>>`。
//!
//! ## ⚠️ 线程隔离的关键
//!
//! 每个实现**必须**按 `thread_id` 分键存储。
//! 读的时候直接按 key 取 —— **物理上不可能串台**。
//!
//! 这不是「实现细节」，而是**正确性的前提**：
//! 如果先取出全部再在内存里过滤，一次 `filter` 漏写就是数据泄漏。

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use agent_logging::{debug, info, warn};

use crate::error::{AppError, AppResult};
use crate::state::{ThreadId, ThreadState};

/// Checkpointer 操作的 future。
pub type CheckpointFuture<'a, T> = Pin<Box<dyn Future<Output = AppResult<T>> + Send + 'a>>;

/// 状态持久化层。
///
/// ## 三个方法对应三个语义
///
/// - [`Checkpointer::load`] —— 读一个线程的状态（**不存在**也要能返回）
/// - [`Checkpointer::save`] —— 写一个线程的状态（**覆盖**语义）
/// - [`Checkpointer::delete`] —— 删一个线程（清空记忆）
pub trait Checkpointer: Send + Sync {
    /// 读一个线程的状态。
    ///
    /// ## ⚠️ 返回 `None` 而不是报错
    ///
    /// 「线程还没建」是**正常状态**，不是错误 ——
    /// 用户第一次说话时，线程当然不存在。
    ///
    /// 如果这里报错，agent 就得在每个入口先「试探性创建」，
    /// 逻辑会绕一大圈。
    fn load<'a>(&'a self, thread_id: &'a ThreadId) -> CheckpointFuture<'a, Option<ThreadState>>;

    /// 保存一个线程的状态（**覆盖**语义）。
    ///
    /// ⚠️ 必须是**全量覆盖**，不是增量追加 ——
    /// 否则「删除一条消息」这种操作无法表达。
    fn save<'a>(&'a self, state: &'a ThreadState) -> CheckpointFuture<'a, ()>;

    /// 删除一个线程的全部状态。
    ///
    /// 返回是否真的删掉了（线程本来就不存在时返回 `false`）。
    fn delete<'a>(&'a self, thread_id: &'a ThreadId) -> CheckpointFuture<'a, bool>;

    /// 列出所有已存在的线程 id。
    ///
    /// 用于「会话列表」这类接口。
    fn list_threads<'a>(&'a self) -> CheckpointFuture<'a, Vec<String>>;
}

/// 内存 checkpointer —— 状态存在 `HashMap` 里。
///
/// ## 用途
///
/// - **单元 / 集成测试** —— 零 IO、零延迟
/// - 临时会话（服务重启后不保留）
///
/// ## ⚠️ 不是线程安全的「简单 HashMap」
///
/// 需要 `RwLock`（多读单写）—— 读操作（`load` / `list`）在 Agent 循环里
/// 每步都要做，写操作少得多。
pub struct InMemoryCheckpointer {
    threads: RwLock<HashMap<String, ThreadState>>,
}

impl std::fmt::Debug for InMemoryCheckpointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryCheckpointer")
            .field(
                "threads",
                &self.threads.read().map(|t| t.len()).unwrap_or(0),
            )
            .finish()
    }
}

impl Default for InMemoryCheckpointer {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryCheckpointer {
    /// 创建空的内存 checkpointer。
    pub fn new() -> Self {
        Self {
            threads: RwLock::new(HashMap::new()),
        }
    }

    /// 读锁（忽略 poisoning）。
    ///
    /// ## ⚠️ 为什么忽略 poisoning
    ///
    /// 某个线程 panic 会把锁标记为 poisoned，之后所有 `read()` 都返回 `Err`。
    /// 对记忆存储来说，锁中毒后数据大概率还是好的（我们只在锁内做
    /// HashMap 操作，没有跨 await 点），拿回锁继续用比「整个服务死掉」好。
    ///
    /// 但我们**记一条 warn 日志** —— 静默忽略 panic 是排查噩梦。
    fn read(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, ThreadState>> {
        match self.threads.read() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("内存 checkpointer 读锁被毒化 —— 继续使用");
                poisoned.into_inner()
            }
        }
    }

    /// 写锁（忽略 poisoning）。
    fn write(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<String, ThreadState>> {
        match self.threads.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("内存 checkpointer 写锁被毒化 —— 继续使用");
                poisoned.into_inner()
            }
        }
    }

    /// 线程数量。
    pub fn len(&self) -> usize {
        self.read().len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }
}

impl Checkpointer for InMemoryCheckpointer {
    fn load<'a>(&'a self, thread_id: &'a ThreadId) -> CheckpointFuture<'a, Option<ThreadState>> {
        Box::pin(async move {
            let state = self.read().get(thread_id.as_str()).cloned();
            debug!(
                thread = %thread_id,
                found = state.is_some(),
                messages = state.as_ref().map(|s| s.message_count()).unwrap_or(0),
                "读取线程状态"
            );
            Ok(state)
        })
    }

    fn save<'a>(&'a self, state: &'a ThreadState) -> CheckpointFuture<'a, ()> {
        Box::pin(async move {
            let mut guard = self.write();
            guard.insert(state.thread_id.clone(), state.clone());
            debug!(
                thread = %state.thread_id,
                messages = state.message_count(),
                version = state.version,
                "保存线程状态"
            );
            Ok(())
        })
    }

    fn delete<'a>(&'a self, thread_id: &'a ThreadId) -> CheckpointFuture<'a, bool> {
        Box::pin(async move {
            let removed = self.write().remove(thread_id.as_str()).is_some();
            info!(thread = %thread_id, removed, "删除线程状态");
            Ok(removed)
        })
    }

    fn list_threads<'a>(&'a self) -> CheckpointFuture<'a, Vec<String>> {
        Box::pin(async move {
            let mut ids: Vec<String> = self.read().keys().cloned().collect();
            // 排序 —— 让列表顺序**确定**（HashMap 的迭代顺序是随机的）。
            ids.sort();
            Ok(ids)
        })
    }
}

/// 文件 checkpointer —— 每个线程一个 JSON 文件。
///
/// ## 为什么要有「落盘」的版本
///
/// LangChain 文档里说：
///
/// > In production, use a checkpointer backed by a database.
///
/// 内存版**进程一重启就全没了** —— 那不叫「持久化」。
/// 这个版本用「一个线程一个文件」的方式实现了真正的恢复：
/// **进程重启后，同一个 `thread_id` 还能接着聊**。
///
/// ## 目录布局
///
/// ```text
/// <root>/
///   thread-a.json
///   thread-b.json
/// ```
///
/// ## ⚠️ 文件名就是 `thread_id`
///
/// 所以 `ThreadId::new` **必须**限制字符集（不允许 `/`、`.` 开头的 `..`）——
/// 否则会出现路径穿越。这是「校验必须在入口」的一个具体例子。
///
/// ## 写入策略
///
/// 写文件时先写**临时文件**，再 `rename` 覆盖 ——
///
/// ⚠️ 这是**原子替换**：要么写成功，要么保持原样。
/// 直接覆盖的话，进程在写到一半被杀会留下**半截 JSON**，
/// 下次读就彻底读不出来（而且是**永久性**的 —— 覆盖掉了好的版本）。
pub struct FileCheckpointer {
    root: PathBuf,
}

impl std::fmt::Debug for FileCheckpointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileCheckpointer")
            .field("root", &self.root)
            .finish()
    }
}

impl FileCheckpointer {
    /// 创建文件 checkpointer（目录会被自动创建）。
    pub fn new(root: impl Into<PathBuf>) -> AppResult<Self> {
        let root = root.into();
        if !root.exists() {
            std::fs::create_dir_all(&root)?;
        }
        info!(root = %root.display(), "文件 checkpointer 已就绪");
        Ok(Self { root })
    }

    /// 存储根目录。
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 某个线程的文件路径。
    ///
    /// ## ⚠️ 为什么用 `format!("{}.json", id)` 而不是直接拼
    ///
    /// `ThreadId` 已经限制了字符集（不允许 `/`），所以这里不会有路径穿越。
    /// 但加一层 `.json` 后缀能避免和目录里的其它文件撞名。
    fn path_for(&self, thread_id: &ThreadId) -> PathBuf {
        self.root.join(format!("{}.json", thread_id.as_str()))
    }
}

impl Checkpointer for FileCheckpointer {
    fn load<'a>(&'a self, thread_id: &'a ThreadId) -> CheckpointFuture<'a, Option<ThreadState>> {
        Box::pin(async move {
            let path = self.path_for(thread_id);

            if !path.exists() {
                debug!(thread = %thread_id, "线程状态文件不存在");
                return Ok(None);
            }

            let text = std::fs::read_to_string(&path)?;

            // ⚠️ 损坏的 JSON **不能让整个服务挂掉**。
            //
            // 如果这里直接 `?` 传播错误，一次磁盘写入中断就会让
            // 这个线程**永久**无法使用 —— 而它只是「丢了」，
            // 不该连累别的线程。
            //
            // 所以：读不出来 → 记 warn → 当作「没有状态」。
            // 用户的对话会从新线程开始，而不是收到一个 500。
            let state = match serde_json::from_str::<ThreadState>(&text) {
                Ok(state) => state,
                Err(e) => {
                    warn!(
                        thread = %thread_id,
                        path = %path.display(),
                        error = %e,
                        "线程状态文件损坏 —— 按「无状态」处理"
                    );
                    return Ok(None);
                }
            };

            debug!(
                thread = %thread_id,
                messages = state.message_count(),
                "读取线程状态文件"
            );
            Ok(Some(state))
        })
    }

    fn save<'a>(&'a self, state: &'a ThreadState) -> CheckpointFuture<'a, ()> {
        Box::pin(async move {
            let path = self.path_for(
                &ThreadId::new(state.thread_id.clone())
                    .map_err(|e| AppError::Internal(format!("线程 id 非法，无法保存: {e}")))?,
            );

            let text = serde_json::to_string_pretty(state)?;

            // ⚠️ 原子写：先写临时文件，再 rename 覆盖。
            //
            // 直接覆盖的话，中途崩溃会留下半截 JSON，
            // 而且**把好的旧版本也毁掉了**。
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, text.as_bytes())?;
            std::fs::rename(&tmp, &path)?;

            debug!(
                thread = %state.thread_id,
                messages = state.message_count(),
                version = state.version,
                path = %path.display(),
                "保存线程状态文件"
            );
            Ok(())
        })
    }

    fn delete<'a>(&'a self, thread_id: &'a ThreadId) -> CheckpointFuture<'a, bool> {
        Box::pin(async move {
            let path = self.path_for(thread_id);
            if !path.exists() {
                return Ok(false);
            }
            std::fs::remove_file(&path)?;
            info!(thread = %thread_id, "删除线程状态文件");
            Ok(true)
        })
    }

    fn list_threads<'a>(&'a self) -> CheckpointFuture<'a, Vec<String>> {
        Box::pin(async move {
            let mut ids = Vec::new();

            let entries = match std::fs::read_dir(&self.root) {
                Ok(entries) => entries,
                Err(e) => {
                    warn!(root = %self.root.display(), error = %e, "读取存储目录失败");
                    return Ok(ids);
                }
            };

            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    ids.push(stem.to_string());
                }
            }

            ids.sort();
            Ok(ids)
        })
    }
}

/// checkpointer 的共享句柄。
pub type SharedCheckpointer = Arc<dyn Checkpointer>;

/// 创建一个内存 checkpointer。
pub fn in_memory_checkpointer() -> SharedCheckpointer {
    Arc::new(InMemoryCheckpointer::new())
}

/// 创建一个文件 checkpointer。
pub fn file_checkpointer(root: impl Into<PathBuf>) -> AppResult<SharedCheckpointer> {
    Ok(Arc::new(FileCheckpointer::new(root)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::MemoryMessage;

    fn state_of(thread: &str) -> ThreadState {
        let mut state = ThreadState::new(thread);
        state.merge_messages(vec![MemoryMessage::user("你好")]);
        state
    }

    #[tokio::test]
    async fn in_memory_saves_and_loads() {
        let cp = in_memory_checkpointer();
        let tid = ThreadId::new("t1").unwrap();

        // 不存在的线程 → None（不是错误）。
        assert!(cp.load(&tid).await.unwrap().is_none());

        cp.save(&state_of("t1")).await.unwrap();

        let loaded = cp.load(&tid).await.unwrap().unwrap();
        assert_eq!(loaded.message_count(), 1);
        assert_eq!(loaded.messages[0].content, "你好");
    }

    #[tokio::test]
    async fn in_memory_isolates_threads() {
        // ⚠️ **线程隔离的核心测试**。
        let cp = in_memory_checkpointer();

        cp.save(&state_of("alice")).await.unwrap();
        cp.save(&state_of("bob")).await.unwrap();

        let alice = cp
            .load(&ThreadId::new("alice").unwrap())
            .await
            .unwrap()
            .unwrap();
        let bob = cp
            .load(&ThreadId::new("bob").unwrap())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(alice.thread_id, "alice");
        assert_eq!(bob.thread_id, "bob");
        // 各自的最后一条消息不能串台。
        assert_eq!(alice.messages[0].content, "你好");
        assert_eq!(bob.messages[0].content, "你好");
    }

    #[tokio::test]
    async fn in_memory_overwrites_on_save() {
        let cp = in_memory_checkpointer();
        let tid = ThreadId::new("t1").unwrap();

        let mut state = state_of("t1");
        cp.save(&state).await.unwrap();

        state.merge_messages(vec![MemoryMessage::assistant("回复")]);
        cp.save(&state).await.unwrap();

        let loaded = cp.load(&tid).await.unwrap().unwrap();
        assert_eq!(loaded.message_count(), 2);
    }

    #[tokio::test]
    async fn in_memory_deletes() {
        let cp = in_memory_checkpointer();
        let tid = ThreadId::new("t1").unwrap();

        cp.save(&state_of("t1")).await.unwrap();
        assert!(cp.delete(&tid).await.unwrap());
        // 删第二次 → 没有了。
        assert!(!cp.delete(&tid).await.unwrap());
        assert!(cp.load(&tid).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn in_memory_lists_threads_sorted() {
        let cp = in_memory_checkpointer();
        cp.save(&state_of("charlie")).await.unwrap();
        cp.save(&state_of("alice")).await.unwrap();

        let ids = cp.list_threads().await.unwrap();
        assert_eq!(ids, vec!["alice".to_string(), "charlie".to_string()]);
    }

    #[tokio::test]
    async fn file_checkpointer_persists_across_instances() {
        // ⚠️ **恢复能力的核心测试**：
        //    第二个实例（模拟进程重启）能读到第一个实例写的状态。
        let dir = tempfile::tempdir().unwrap();

        let tid = ThreadId::new("t1").unwrap();
        {
            let cp = file_checkpointer(dir.path()).unwrap();
            cp.save(&state_of("t1")).await.unwrap();
        }

        // 新的实例 —— 相当于进程重启了。
        let cp = file_checkpointer(dir.path()).unwrap();
        let loaded = cp.load(&tid).await.unwrap().unwrap();
        assert_eq!(loaded.message_count(), 1);
        assert_eq!(loaded.messages[0].content, "你好");
    }

    #[tokio::test]
    async fn file_checkpointer_isolates_threads() {
        let dir = tempfile::tempdir().unwrap();
        let cp = file_checkpointer(dir.path()).unwrap();

        let mut alice = state_of("alice");
        alice.merge_messages(vec![MemoryMessage::user("Alice 专属")]);
        cp.save(&alice).await.unwrap();
        cp.save(&state_of("bob")).await.unwrap();

        let a = cp
            .load(&ThreadId::new("alice").unwrap())
            .await
            .unwrap()
            .unwrap();
        let b = cp
            .load(&ThreadId::new("bob").unwrap())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(a.messages.len(), 2);
        assert_eq!(b.messages.len(), 1);
        assert!(!b.messages[0].content.contains("Alice"));
    }

    #[tokio::test]
    async fn file_checkpointer_tolerates_a_corrupted_file() {
        // ⚠️ 损坏的文件**不能**让整个服务挂掉 —— 按「无状态」处理。
        let dir = tempfile::tempdir().unwrap();
        let cp = file_checkpointer(dir.path()).unwrap();

        std::fs::write(dir.path().join("broken.json"), "{ 不是合法 JSON").unwrap();

        let loaded = cp.load(&ThreadId::new("broken").unwrap()).await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn file_checkpointer_deletes_and_lists() {
        let dir = tempfile::tempdir().unwrap();
        let cp = file_checkpointer(dir.path()).unwrap();

        cp.save(&state_of("a")).await.unwrap();
        cp.save(&state_of("b")).await.unwrap();

        assert_eq!(cp.list_threads().await.unwrap().len(), 2);

        assert!(cp.delete(&ThreadId::new("a").unwrap()).await.unwrap());
        assert_eq!(cp.list_threads().await.unwrap(), vec!["b".to_string()]);
    }

    #[tokio::test]
    async fn file_checkpointer_writes_atomically() {
        // 写完之后**不能**留下临时文件（否则 list_threads 会把它当状态）。
        let dir = tempfile::tempdir().unwrap();
        let cp = file_checkpointer(dir.path()).unwrap();
        cp.save(&state_of("a")).await.unwrap();

        let entries: Vec<_> = std::fs::read_dir(dir.path()).unwrap().flatten().collect();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].path().to_string_lossy().ends_with("a.json"));
    }
}
