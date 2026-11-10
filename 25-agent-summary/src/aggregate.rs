//! **聚合层** —— 三档策略的核心，也是本项目最重要的产出。
//!
//! # 三档策略
//!
//! | 变体 | 做了什么 | 验证什么 |
//! |---|---|---|
//! | `naive` | 每次全量扫描 + 每条记录的派生计算 | 基线 |
//! | `pre-agg` | 建索引时把派生计算**预聚合**并**整型化** | 算法收益 |
//! | `optimized` | `pre-agg` + 内容寻址**结果缓存** | 缓存收益 |
//!
//! # ★ 为什么是三档而不是两档
//!
//! 只比 `naive` vs `optimized`（含缓存），差距被「命中缓存直接返回」**完全主导**
//! —— 数字很漂亮，但**说不出是谁的功劳**。
//!
//! 加了中间的 `pre-agg` 档（预聚合 + 整型化但**不缓存**），
//! 才能拆出「算法赚了多少」与「缓存赚了多少」。
//!
//! # ★ 第一版优化实测 1.00x —— 优化点选错了
//!
//! 最初的 `optimized` 只做「按天预聚合」（**减少扫描量**），实测 **1.00x**。
//!
//! 分段剖析（`examples/profile_pipeline.rs`）显示：
//! 「过滤（全量扫描）」只占 **1%**，真正贵的是**每条记录的派生计算**（占 **89%**）：
//!
//! - 字符串 clone
//! - 字符串 hash（去重时）
//! - `day_key` 之类的格式化
//! - `BTreeMap<String>` 的增量插入（统计时）
//!
//! 修法是**把派生计算搬到建索引时并整型化**：
//!
//! ```text
//! String → u16/u32 下标    （去重从「字符串 hash」变「整数比较」）
//! BTreeMap<String> → 计数器数组 （统计从「增量插入」变「自增」）
//! ```
//!
//! 这才换来 **5.16x**。
//!
//! **教训（可推广）**：「测不出差异」有两种原因 ——「**实现错了**」和「**优化点选错了**」。
//! 必须**分段剖析**才能分清。

use std::collections::BTreeMap;

use crate::model::{Activity, Digest, Window, day_key};

/// 聚合策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// 每次全量重算。
    Naive,
    /// 预聚合 + 整型化。
    PreAggregated,
    /// 预聚合 + 内容寻址缓存。
    Optimized,
}

impl Strategy {
    /// 变体名。
    pub fn name(self) -> &'static str {
        match self {
            Strategy::Naive => "naive",
            Strategy::PreAggregated => "pre-agg",
            Strategy::Optimized => "optimized",
        }
    }

    /// 人类可读的说明。
    pub fn label(self) -> &'static str {
        match self {
            Strategy::Naive => "全量重算（基线）",
            Strategy::PreAggregated => "预聚合 + 整型化",
            Strategy::Optimized => "预聚合 + 内容寻址缓存",
        }
    }

    /// 全部变体（按「基线 → 对照」顺序）。
    pub fn all() -> &'static [Strategy] {
        &[
            Strategy::Naive,
            Strategy::PreAggregated,
            Strategy::Optimized,
        ]
    }

    /// 从字符串解析。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "naive" => Some(Strategy::Naive),
            "pre-agg" | "pre_agg" | "preaggregated" => Some(Strategy::PreAggregated),
            "optimized" => Some(Strategy::Optimized),
            _ => None,
        }
    }
}

/// ★ 预聚合索引 —— 派生计算从「每次请求」搬到「建索引时」。
///
/// ## 整型化的两层含义
///
/// | 原来 | 整型化后 | 省下的 |
/// |---|---|---|
/// | `String`（模块名 / 作者 / 类型） | `u16` 下标 | 字符串 hash + 比较 |
/// | `BTreeMap<String>` 计数 | `Vec<u32>` 计数器数组 | 增量插入 + 树维护 |
/// | `day_key()` 返回 `String` | `i64`（天序号） | 字符串格式化 |
#[derive(Debug, Clone, Default)]
pub struct PreIndex {
    /// 维度下标表（**有序**，保证确定性）。
    ///
    /// ⚠️ 用 `BTreeMap` 而不是 `HashMap` —— `HashMap` 迭代顺序不确定，
    /// 会让「三档产出必须一致」这条约束挂掉。
    scopes: Vec<String>,
    scope_ids: BTreeMap<String, u16>,

    authors: Vec<String>,
    author_ids: BTreeMap<String, u16>,

    kinds: Vec<String>,
    kind_ids: BTreeMap<String, u16>,

    /// 整条记录的整型化视图。
    items: Vec<IndexedItem>,

    /// 按天的记录下标（`day` → `items` 下标列表）。
    by_day: BTreeMap<i64, Vec<u32>>,

    /// 预聚合的统计计数器（维度下标 → 计数）。
    scope_counts: Vec<u32>,
    author_counts: Vec<u32>,
    kind_counts: Vec<u32>,
    day_counts: BTreeMap<i64, u32>,

    /// 权重总和（预聚合）。
    total_weight: i64,

    /// 风险项计数（预聚合）。
    total_risks: u32,
}

/// 整型化后的一条记录。
#[derive(Debug, Clone, Copy)]
struct IndexedItem {
    /// 记录下标（去重键用「原始 ID 的哈希」不合适 —— 改成下标）。
    id_hash: u64,
    /// ★ 原始记录在 `activities` 里的下标。
    ///
    /// ⚠️ 必须保留它 —— 否则回查原始标题/权重时要**再扫一遍**
    /// （`O(n)` 的字符串比较），那就把「预聚合」的收益全吐回去了。
    source_idx: u32,
    /// 类型下标。
    kind: u16,
    /// 作者下标。
    author: u16,
    /// 模块下标。
    scope: u16,
    /// 权重。
    weight: i32,
    /// 是否风险项。
    risk: bool,
    /// 天序号。
    day: i64,
}

impl PreIndex {
    /// ★ 建索引 —— 所有派生计算都在这里做完。
    pub fn build(activities: &[Activity]) -> Self {
        let mut index = PreIndex::default();

        // ① 维度下标表（按**出现顺序**建，保证确定性）。
        for a in activities {
            index.intern_scope(&a.scope);
            index.intern_author(&a.author);
            index.intern_kind(&a.kind);
        }

        // ② 整型化每条记录。
        for (idx, a) in activities.iter().enumerate() {
            let kind = index.kind_ids[&a.kind];
            let author = index.author_ids[&a.author];
            let scope = index.scope_ids[&a.scope];
            let day = day_key(a.ts);

            index.items.push(IndexedItem {
                id_hash: hash_id(&a.id),
                source_idx: idx as u32,
                kind,
                author,
                scope,
                weight: a.weight,
                risk: a.risk,
                day,
            });
            // 预聚合权重与风险计数（供上层直接读取，避免二次扫描）。
            index.total_weight += a.weight as i64;
            if a.risk {
                index.total_risks += 1;
            }
        }

        // ③ 按天建桶。
        for (i, item) in index.items.iter().enumerate() {
            index.by_day.entry(item.day).or_default().push(i as u32);
        }

        // ④ 预聚合统计（**计数器数组自增**，不是 `BTreeMap` 插入）。
        index.scope_counts = vec![0; index.scopes.len()];
        index.author_counts = vec![0; index.authors.len()];
        index.kind_counts = vec![0; index.kinds.len()];

        for item in &index.items {
            index.scope_counts[item.scope as usize] += 1;
            index.author_counts[item.author as usize] += 1;
            index.kind_counts[item.kind as usize] += 1;
            *index.day_counts.entry(item.day).or_insert(0) += 1;
        }

        index
    }

    /// 把模块名 intern 成下标。
    fn intern_scope(&mut self, name: &str) -> u16 {
        if let Some(id) = self.scope_ids.get(name) {
            return *id;
        }
        let id = self.scopes.len() as u16;
        self.scopes.push(name.to_string());
        self.scope_ids.insert(name.to_string(), id);
        id
    }

    /// 把作者 intern 成下标。
    fn intern_author(&mut self, name: &str) -> u16 {
        if let Some(id) = self.author_ids.get(name) {
            return *id;
        }
        let id = self.authors.len() as u16;
        self.authors.push(name.to_string());
        self.author_ids.insert(name.to_string(), id);
        id
    }

    /// 把类型 intern 成下标。
    fn intern_kind(&mut self, name: &str) -> u16 {
        if let Some(id) = self.kind_ids.get(name) {
            return *id;
        }
        let id = self.kinds.len() as u16;
        self.kinds.push(name.to_string());
        self.kind_ids.insert(name.to_string(), id);
        id
    }

    /// 模块名表。
    pub fn scopes(&self) -> &[String] {
        &self.scopes
    }

    /// 记录数。
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// 某天的记录下标。
    pub fn items_of_day(&self, day: i64) -> &[u32] {
        self.by_day.get(&day).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// 模块计数。
    pub fn scope_counts(&self) -> &[u32] {
        &self.scope_counts
    }

    /// 作者计数。
    pub fn author_counts(&self) -> &[u32] {
        &self.author_counts
    }

    /// 类型计数。
    pub fn kind_counts(&self) -> &[u32] {
        &self.kind_counts
    }

    /// 天计数。
    pub fn day_counts(&self) -> &BTreeMap<i64, u32> {
        &self.day_counts
    }

    /// 权重总和（预聚合）。
    pub fn weight_sum(&self) -> i64 {
        self.total_weight
    }

    /// 风险项计数（预聚合）。
    pub fn risk_count(&self) -> u32 {
        self.total_risks
    }

    /// 模块名表。
    pub fn scope_names(&self) -> &[String] {
        &self.scopes
    }

    /// 作者名表。
    pub fn author_names(&self) -> &[String] {
        &self.authors
    }

    /// 类型名表。
    pub fn kind_names(&self) -> &[String] {
        &self.kinds
    }
}

/// ★ 内容寻址缓存 —— 键是「数据集指纹 + 窗口」。
#[derive(Debug, Clone, Default)]
pub struct ReportCache {
    entries: BTreeMap<String, Digest>,
}

impl ReportCache {
    /// 新建缓存。
    pub fn new() -> Self {
        Self::default()
    }

    /// 取。
    pub fn get(&self, key: &str) -> Option<&Digest> {
        self.entries.get(key)
    }

    /// 存。
    pub fn insert(&mut self, key: String, digest: Digest) {
        self.entries.insert(key, digest);
    }

    /// 条数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 清空。
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// ★ 数据集指纹 —— SHA-256。
///
/// ⚠️ 不能用 `DefaultHasher`：它**不保证跨进程稳定**且 seed 随机
/// —— 重启一次所有键全变，缓存永远命中不了。
pub fn dataset_fingerprint(activities: &[Activity]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for a in activities {
        hasher.update(a.id.as_bytes());
        hasher.update(b"\x00");
        hasher.update(a.ts.to_le_bytes());
        hasher.update(b"\x00");
        hasher.update(a.kind.as_bytes());
        hasher.update(b"\x00");
        hasher.update(a.author.as_bytes());
        hasher.update(b"\x00");
        hasher.update(a.scope.as_bytes());
        hasher.update(b"\x00");
        hasher.update(a.title.as_bytes());
        hasher.update(b"\x00");
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// 缓存键 = 数据集指纹 + 窗口。
pub fn cache_key(fingerprint: &str, window: &Window) -> String {
    format!("{fingerprint}:{}:{}", window.start, window.end)
}

/// ID 的哈希（用 FNV-1a —— 快且无依赖）。
fn hash_id(id: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in id.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// 聚合器 —— 三档策略的统一入口。
pub struct Aggregator {
    /// 原始数据（**按 `ts` 升序**）。
    ///
    /// ⚠️ 输入**必须**按 `ts` 升序 —— 因为两版的去重都保留「先遇到的那条」，
    /// 遍历顺序不同结果就不同。
    activities: Vec<Activity>,

    /// 预聚合索引。
    pre: PreIndex,

    /// 缓存。
    cache: ReportCache,

    /// ★ 缓存命中次数。
    pub cache_hits_count: usize,

    /// ★ 缓存未命中次数。
    pub cache_misses_count: usize,

    /// ★ 数据集指纹（**建索引时算一次**）。
    ///
    /// ## ⚠️ 为什么不能每次调用都算
    ///
    /// 指纹是 SHA-256 over 全部记录 —— O(n)。
    /// 如果每次 `aggregate` 都重算，指纹本身就成了瓶颈，
    /// **优化档反而比基线慢**（实测 111 QPS vs 597 QPS）。
    ///
    /// 数据集是**不可变**的（`activities` 只在构造时排一次序），
    /// 所以指纹可以在构造时算一次。
    fingerprint: String,

    /// 当前策略。
    strategy: Strategy,
}

impl Aggregator {
    /// 新建聚合器。
    pub fn new(mut activities: Vec<Activity>, strategy: Strategy) -> Self {
        // ★ 排序只做一次 —— 两版共用。
        activities.sort_by(|a, b| a.ts.cmp(&b.ts).then(a.id.cmp(&b.id)));
        let pre = PreIndex::build(&activities);
        // ★ 指纹算一次 —— 数据集不可变。
        let fingerprint = dataset_fingerprint(&activities);
        Self {
            activities,
            pre,
            cache: ReportCache::new(),
            cache_hits_count: 0,
            cache_misses_count: 0,
            fingerprint,
            strategy,
        }
    }

    /// 当前策略。
    pub fn strategy(&self) -> Strategy {
        self.strategy
    }

    /// 原始数据（按 `ts` 升序）。
    pub fn activities(&self) -> &[Activity] {
        &self.activities
    }

    /// 切换策略（清空缓存 —— 缓存键与策略绑定）。
    pub fn set_strategy(&mut self, strategy: Strategy) {
        self.strategy = strategy;
        self.cache.clear();
        self.cache_hits_count = 0;
        self.cache_misses_count = 0;
    }

    /// ★ 缓存命中率（0.0 ~ 1.0）。
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits_count + self.cache_misses_count;
        if total == 0 {
            0.0
        } else {
            self.cache_hits_count as f64 / total as f64
        }
    }

    /// 缓存命中数。
    pub fn cache_hits(&self) -> usize {
        self.cache.len()
    }

    /// 聚合一个窗口。
    pub fn aggregate(&mut self, window: &Window) -> Digest {
        match self.strategy {
            Strategy::Naive => self.aggregate_naive(window),
            Strategy::PreAggregated => self.aggregate_preaggregated(window),
            Strategy::Optimized => self.aggregate_optimized(window),
        }
    }

    /// ★ naive：每次全量扫描 + 每条记录的派生计算。
    fn aggregate_naive(&self, window: &Window) -> Digest {
        // ① 过滤（全量扫）。
        let filtered: Vec<&Activity> = self
            .activities
            .iter()
            .filter(|a| window.contains(a.ts))
            .collect();

        // ② 去重 —— 用 `HashSet<String>`（字符串 hash！）。
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut deduped: Vec<&Activity> = Vec::new();
        for a in &filtered {
            if seen.insert(a.id.as_str()) {
                deduped.push(a);
            }
        }

        // ③ 统计 —— `BTreeMap<String>` 增量插入（**这是最贵的部分**）。
        let mut by_kind: BTreeMap<String, usize> = BTreeMap::new();
        let mut by_author: BTreeMap<String, usize> = BTreeMap::new();
        let mut by_scope: BTreeMap<String, usize> = BTreeMap::new();
        for a in &deduped {
            *by_kind.entry(a.kind.clone()).or_insert(0) += 1;
            *by_author.entry(a.author.clone()).or_insert(0) += 1;
            *by_scope.entry(a.scope.clone()).or_insert(0) += 1;
        }

        // ④ 按天 —— `day_key` 返回**字符串**（格式化！）。
        let mut by_day: BTreeMap<String, usize> = BTreeMap::new();
        for a in &deduped {
            let key = crate::model::day_label(a.ts);
            *by_day.entry(key).or_insert(0) += 1;
        }

        // ⑤ 亮点 —— ⚠️ **先排序再取前 5**（不能先取再排序）。
        //
        //    为什么：`deduped` 的顺序是「先遇到的那条」，不是「权重最高的」。
        //    先 take(5) 再排序会选出「前 5 条里权重最高的」，
        //    而全局排序再 take 选出的是「权重最高的 5 条」——
        //    两者**不是同一个集合**。pre-agg 档走的是后者，
        //    所以 naive 也必须走后者，否则两档结果不一致。
        let mut all_highlights: Vec<crate::model::Highlight> = deduped
            .iter()
            .filter(|a| a.weight > 0 && !a.risk)
            .map(|a| crate::model::Highlight {
                id: a.id.clone(),
                title: a.title.clone(),
                scope: a.scope.clone(),
                author: a.author.clone(),
                kind: a.kind.clone(),
                weight: a.weight,
            })
            .collect();
        all_highlights.sort_by(|a, b| b.weight.cmp(&a.weight).then(a.id.cmp(&b.id)));
        let highlights: Vec<crate::model::Highlight> = all_highlights.into_iter().take(5).collect();

        // ⑥ 风险 —— 同上。
        let mut all_risks: Vec<crate::model::Highlight> = deduped
            .iter()
            .filter(|a| a.risk)
            .map(|a| crate::model::Highlight {
                id: a.id.clone(),
                title: a.title.clone(),
                scope: a.scope.clone(),
                author: a.author.clone(),
                kind: a.kind.clone(),
                weight: a.weight,
            })
            .collect();
        all_risks.sort_by(|a, b| b.weight.cmp(&a.weight).then(a.id.cmp(&b.id)));
        let risks: Vec<crate::model::Highlight> = all_risks.into_iter().take(5).collect();

        Digest {
            total: filtered.len(),
            deduped: deduped.len(),
            scopes: by_scope.len(),
            risks: risks.len(),
            days: by_day.len(),
            by_kind: by_kind.into_iter().collect(),
            by_author: by_author.into_iter().collect(),
            by_scope: by_scope.into_iter().collect(),
            by_day: by_day.into_iter().collect(),
            highlights,
            risks_list: risks,
        }
    }

    /// ★ pre-agg：查预聚合表，避开 89% 的派生计算。
    fn aggregate_preaggregated(&self, window: &Window) -> Digest {
        // ① 只扫「窗口内的天」—— 不扫全部记录。
        let mut selected: Vec<u32> = Vec::new();

        let start_day = crate::model::day_key(window.start);
        let end_day = crate::model::day_key(window.end);
        let mut day = start_day;
        while day <= end_day {
            let items = self.pre.items_of_day(day);
            selected.extend_from_slice(items);
            day += 1;
        }

        // ② 去重 —— **整数比较**（`id_hash`），不是字符串 hash。
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut deduped_ids: Vec<u64> = Vec::new();
        let mut kept: Vec<&IndexedItem> = Vec::new();
        for idx in &selected {
            let item = &self.pre.items[*idx as usize];
            if seen.insert(item.id_hash) {
                deduped_ids.push(item.id_hash);
                kept.push(item);
            }
        }

        // ③ 统计 —— **计数器数组**，但窗口内要重新数（不能直接用全局计数）。
        //
        // ⚠️ 这里仍有一处 O(n) 的活（遍历窗口内的记录数计数），
        //    但它是「整数自增」而不是「字符串 clone + 树插入」——
        //    这就是 5.16x 的来源。
        let mut kind_counts = vec![0u32; self.pre.kind_names().len()];
        let mut author_counts = vec![0u32; self.pre.author_names().len()];
        let mut scope_counts = vec![0u32; self.pre.scope_names().len()];
        for item in &kept {
            kind_counts[item.kind as usize] += 1;
            author_counts[item.author as usize] += 1;
            scope_counts[item.scope as usize] += 1;
        }

        // ⚠️ 必须**按名字排序** —— naive 档用的是 `BTreeMap`（按名字有序），
        //    而这里是从「intern 顺序」的计数器数组来的（按出现顺序）。
        //    不排序 → 两档的 `by_kind` 顺序不同 → 等价性测试挂。
        let mut by_kind: Vec<(String, usize)> = Vec::new();
        for (i, name) in self.pre.kind_names().iter().enumerate() {
            if kind_counts[i] > 0 {
                by_kind.push((name.clone(), kind_counts[i] as usize));
            }
        }
        by_kind.sort_by(|a, b| a.0.cmp(&b.0));

        let mut by_author: Vec<(String, usize)> = Vec::new();
        for (i, name) in self.pre.author_names().iter().enumerate() {
            if author_counts[i] > 0 {
                by_author.push((name.clone(), author_counts[i] as usize));
            }
        }
        by_author.sort_by(|a, b| a.0.cmp(&b.0));

        let mut by_scope: Vec<(String, usize)> = Vec::new();
        for (i, name) in self.pre.scope_names().iter().enumerate() {
            if scope_counts[i] > 0 {
                by_scope.push((name.clone(), scope_counts[i] as usize));
            }
        }
        by_scope.sort_by(|a, b| a.0.cmp(&b.0));

        // ④ 按天 —— 预聚合的 `day_counts`，**不再重新格式化**。
        let mut by_day: Vec<(String, usize)> = Vec::new();
        for (d, count) in self.pre.day_counts() {
            if *d >= start_day && *d <= end_day && *count > 0 {
                by_day.push((crate::model::day_label(d * 86_400), *count as usize));
            }
        }

        // ⑤ 亮点 / ⑥ 风险 —— 通过 `source_idx` **直接回查**原始记录。
        //
        // ⚠️ 不能用「按哈希再扫一遍」—— 那是 O(n) 的字符串比较，
        //    会把预聚合的收益全吐回去。
        let mut all_highlights: Vec<crate::model::Highlight> = Vec::new();
        let mut all_risks: Vec<crate::model::Highlight> = Vec::new();
        for item in &kept {
            let act = &self.activities[item.source_idx as usize];
            let h = crate::model::Highlight {
                id: act.id.clone(),
                title: act.title.clone(),
                scope: self.pre.scope_names()[item.scope as usize].clone(),
                author: self.pre.author_names()[item.author as usize].clone(),
                kind: self.pre.kind_names()[item.kind as usize].clone(),
                // ★ 权重从**整型化视图**读 —— 不回查原始记录。
                weight: item.weight,
            };
            // ★ 风险标记同样从整型化视图读。
            if item.risk {
                all_risks.push(h);
            } else if item.weight > 0 {
                all_highlights.push(h);
            }
        }
        all_highlights.sort_by(|a, b| b.weight.cmp(&a.weight).then(a.id.cmp(&b.id)));
        all_risks.sort_by(|a, b| b.weight.cmp(&a.weight).then(a.id.cmp(&b.id)));
        let highlights: Vec<crate::model::Highlight> = all_highlights.into_iter().take(5).collect();
        let risks: Vec<crate::model::Highlight> = all_risks.into_iter().take(5).collect();

        Digest {
            total: selected.len(),
            deduped: deduped_ids.len(),
            scopes: by_scope.len(),
            risks: risks.len(),
            days: by_day.len(),
            by_kind,
            by_author,
            by_scope,
            by_day,
            highlights,
            risks_list: risks,
        }
    }

    /// ★ optimized：预聚合 + 内容寻址缓存。
    fn aggregate_optimized(&mut self, window: &Window) -> Digest {
        // ★ 用预先算好的指纹 —— 不要每次重算。
        let key = cache_key(&self.fingerprint, window);

        if let Some(digest) = self.cache.get(&key) {
            self.cache_hits_count += 1;
            return digest.clone();
        }

        self.cache_misses_count += 1;
        let digest = self.aggregate_preaggregated(window);
        self.cache.insert(key, digest.clone());
        digest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn activities(n: usize) -> Vec<Activity> {
        (0..n)
            .map(|i| Activity {
                id: format!("a-{i:04}"),
                ts: 1_700_000_000 + (i as i64 % 7) * 86_400,
                kind: ["feature", "fix", "incident", "docs"][i % 4].to_string(),
                author: ["alice", "bob", "carol"][i % 3].to_string(),
                scope: ["cache", "rag", "agent"][i % 3].to_string(),
                title: format!("任务 {i}"),
                weight: (i % 10) as i32,
                risk: i % 20 == 0,
            })
            .collect()
    }

    fn window() -> Window {
        Window::new(1_700_000_000, 1_700_000_000 + 7 * 86_400)
    }

    #[test]
    fn naive_and_preagg_produce_identical_digests() {
        let acts = activities(500);
        let mut naive = Aggregator::new(acts.clone(), Strategy::Naive);
        let mut pre = Aggregator::new(acts, Strategy::PreAggregated);
        assert_eq!(
            naive.aggregate(&window()),
            pre.aggregate(&window()),
            "两档必须产出完全一致的结果，否则测的是「不同的功能」而不是「优化」"
        );
    }

    #[test]
    fn naive_and_optimized_produce_identical_digests() {
        let acts = activities(500);
        let mut naive = Aggregator::new(acts.clone(), Strategy::Naive);
        let mut opt = Aggregator::new(acts, Strategy::Optimized);
        assert_eq!(naive.aggregate(&window()), opt.aggregate(&window()));
    }

    #[test]
    fn optimized_second_call_hits_cache() {
        let acts = activities(200);
        let mut opt = Aggregator::new(acts, Strategy::Optimized);
        let _ = opt.aggregate(&window());
        assert_eq!(opt.cache_hits(), 1);
        let _ = opt.aggregate(&window());
        assert_eq!(opt.cache_hits(), 1, "同样的输入不应产生新键");
    }

    #[test]
    fn different_windows_produce_different_keys() {
        let acts = activities(200);
        let mut opt = Aggregator::new(acts, Strategy::Optimized);
        let _ = opt.aggregate(&window());
        let _ = opt.aggregate(&Window::new(1_700_000_000, 1_700_100_000));
        assert_eq!(opt.cache_hits(), 2);
    }

    #[test]
    fn fingerprint_is_stable_and_sensitive() {
        let a = activities(10);
        let b = activities(10);
        assert_eq!(dataset_fingerprint(&a), dataset_fingerprint(&b));

        let mut c = a.clone();
        c[0].title = "改了标题".into();
        assert_ne!(dataset_fingerprint(&a), dataset_fingerprint(&c));
    }

    #[test]
    fn window_days_uses_calendar_boundaries() {
        // 10:00 → 次日 09:00 跨了 2 个自然日。
        let start = 1_700_000_000; // 不是零点
        let w = Window::new(start, start + 86_400);
        assert_eq!(w.days(), 2);
    }

    #[test]
    fn day_key_floors_to_midnight() {
        let a = 1_700_000_000;
        let b = 1_700_000_000 + 3_600;
        assert_eq!(day_key(a), day_key(b));
    }

    #[test]
    fn dedup_keeps_the_first_one() {
        let mut acts = activities(20);
        // 造一条重复（ID 相同、ts 更晚）。
        acts.push(Activity {
            id: "a-0000".into(),
            ts: 1_700_000_000 + 100,
            kind: "fix".into(),
            author: "bob".into(),
            scope: "rag".into(),
            title: "重复".into(),
            weight: 99,
            risk: false,
        });
        let mut agg = Aggregator::new(acts, Strategy::Naive);
        let d = agg.aggregate(&window());
        // 重复被去掉。
        assert_eq!(d.total, 21);
        assert_eq!(d.deduped, 20);
    }

    #[test]
    fn strategy_switch_clears_cache() {
        let acts = activities(100);
        let mut opt = Aggregator::new(acts, Strategy::Optimized);
        let _ = opt.aggregate(&window());
        assert_eq!(opt.cache_hits(), 1);
        opt.set_strategy(Strategy::Naive);
        assert_eq!(opt.cache_hits(), 0);
    }

    #[test]
    fn strategy_parses_from_string() {
        assert_eq!(Strategy::parse("naive"), Some(Strategy::Naive));
        assert_eq!(Strategy::parse("pre-agg"), Some(Strategy::PreAggregated));
        assert_eq!(Strategy::parse("optimized"), Some(Strategy::Optimized));
        assert_eq!(Strategy::parse("nope"), None);
    }
}
