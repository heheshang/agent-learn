//! 提示词组装 —— SQL Agent 的「行为契约」。
//!
//! ## 教程里的 system prompt
//!
//! ```python
//! system_prompt = """
//! You are an agent designed to interact with a SQL database.
//! Given an input question, create a syntactically correct {dialect} query to run,
//! then look at the results of the query and return the answer. Unless the user
//! specifies a specific number of examples they wish to obtain, always limit your
//! query to at most {top_k} results.
//!
//! You can order the results by a relevant column to return the most interesting
//! examples in the database. Never query for all the columns from a specific table,
//! only ask for the relevant columns given the question.
//!
//! You MUST double check your query before executing it. If you get an error while
//! executing a query, rewrite the query and try again.
//!
//! DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
//! database.
//!
//! To start you should ALWAYS look at the tables in the database to see what you
//! can query. Do NOT skip this step.
//!
//! Then you should query the schema of the most relevant tables.
//! """.format(dialect="sqlite", top_k=5)
//! ```
//!
//! ## 本项目在教程基础上**加强**的三处
//!
//! 1. **DML 禁令升级为「数据库是只读的」** ——
//!    因为护栏真的会拦，模型需要知道这一点（否则它会反复尝试写操作）。
//! 2. **明确「错误信息就是反馈」** ——
//!    教程的「rewrite the query and try again」很轻，
//!    本项目把它写成显式的工作流。
//! 3. **护栏拦截时的正确反应** ——
//!    「别改参数，改成只读查询」。

use agent_14::tool::ToolSpec;

/// SQL 方言。
///
/// 本项目只支持 SQLite —— 但把方言**显式化**是好习惯：
/// 换 Postgres 时只要改这一处（提示词里的 SQL 语法提示）。
pub const DIALECT: &str = "sqlite";

/// 默认返回行数上限。
pub const TOP_K: usize = 5;

/// SQL Agent 的 system prompt。
///
/// ## 为什么强调「先看表、再看 schema」
///
/// 教程说「To start you should ALWAYS look at the tables in the database
/// to see what you can query. Do NOT skip this step.」
///
/// 原因是**模型不知道表名**。中文问题（「哪个流派平均曲目最长」）
/// 里的「流派」在库里叫 `Genre` —— 模型只能靠 schema 才知道。
///
/// 跳过这一步的话，模型会**猜**表名，然后拿到
/// 「no such table: xxx」的错误，白白浪费一轮。
pub fn system_prompt() -> String {
    format!(
        concat!(
            "你是一个与 **{dialect}** 数据库交互的 Agent。\n",
            "\n",
            "给定一个用户问题，你要：\n",
            "1. 写一条语法正确的 {dialect} 查询语句；\n",
            "2. 查看查询结果；\n",
            "3. 基于结果给出答案。\n",
            "\n",
            "## 工作规则\n",
            "\n",
            "### 1. 必须先探查数据库\n",
            "开始时**总是**先调用 `sql_db_list_tables` 看看有哪些表 —— 不要跳过这一步。\n",
            "然后调用 `sql_db_schema` 查看与问题最相关的几张表的结构。\n",
            "你不知道表名和列名（用户说的是中文，库里是英文），\
             只能靠 schema 工具获取。\n",
            "\n",
            "### 2. 查询结果默认最多 {top_k} 条\n",
            "除非用户明确说了要几条，否则给查询加上 `LIMIT {top_k}`。\n",
            "可以按与问题相关的列排序（`ORDER BY`），让结果更有信息量。\n",
            "\n",
            "### 3. 只查相关的列\n",
            "**不要** `SELECT *`。只查与问题相关的列。\n",
            "\n",
            "### 4. 执行前必须检查\n",
            "在执行查询前，先用 `sql_db_query_checker` 检查一遍。\n",
            "如果检查器改写了查询，就用改写后的版本。\n",
            "\n",
            "### 5. 错误就是反馈 —— 读懂它，改对，再试\n",
            "如果查询执行失败，**先读懂错误信息**再改：\n",
            "- `no such column: X` → 列名错了，用 `sql_db_schema` 查正确的列名\n",
            "- `no such table: X` → 表名错了，用 `sql_db_list_tables` 查正确的表名\n",
            "- `syntax error` → 语法错了，改写查询\n",
            "- `database is locked` / `readonly` → 读操作失败，改成纯 SELECT\n",
            "改对之后**重试**，不要放弃也不要重复同样的错误查询。\n",
            "\n",
            "### 6. 数据库是**只读**的\n",
            "这个数据库**只允许** `SELECT` / `WITH` / `VALUES` / `EXPLAIN` 开头的只读查询。\n",
            "**禁止**任何写操作：`INSERT` / `UPDATE` / `DELETE` / `DROP` / `ALTER` / \
             `CREATE` / `PRAGMA` / `ATTACH` 等。\n",
            "也**禁止**多条语句（一条查询只能有一个分号结尾）。\n",
            "如果你写的 SQL 被安全护栏拦截了，那说明它试图写数据 —— \
             请改成只读的 SELECT 查询再试。\n",
            "\n",
            "### 7. 只根据查询结果回答\n",
            "答案里的每个数字都必须来自查询结果。\n",
            "不要自己算、不要从记忆里补。查询结果里没有的，就说明查不到。\n",
            "\n",
            "### 8. 用 `Final Answer:` 开头给出最终答案\n",
            "答案用自然语言表述，把关键数字说清楚。\n",
        ),
        dialect = DIALECT,
        top_k = TOP_K
    )
}

/// 渲染工具说明（含参数明细）。
///
/// 复用 14 的 [`agent_14::schema::render_schema`] —— 那是**同一套**参数渲染逻辑，
/// 不重复实现。
pub fn render_tool_descriptions(tools: &[ToolSpec]) -> String {
    let mut out = String::new();

    for tool in tools {
        out.push_str(&format!("- {}: {}\n", tool.name, tool.description));

        let params = tool.render_parameters();
        if !params.is_empty() {
            out.push_str("  参数：\n");
            for line in params.lines() {
                out.push_str(&format!("    {line}\n"));
            }
        }
    }

    out.trim_end().to_string()
}

/// 组装 Agent 循环的用户提示词。
///
/// ## ⚠️ 与 14-agent-tool 的**关键差异**
///
/// 14 的提示词结构是：
///
/// ```text
/// 格式说明 → 工具说明 → few-shot 示例 → 真实对话
/// ```
///
/// 本项目**完全一样** —— 因为 14 已经踩过两个坑：
///
/// 1. **真实对话必须和 few-shot 示例明确区分开**
///    （否则模型把上一轮轨迹误认成示例，然后自己编一个问题去查）
/// 2. **轨迹必须标注成「已完成的步骤」**
///    （否则模型以为那是可以照着续写的模板）
///
/// 这两个坑**重新踩一遍就是纯浪费**，所以照搬 14 的结构。
pub fn build_prompt(
    tools: &[ToolSpec],
    question: &str,
    trajectory: &str,
    few_shot: bool,
) -> String {
    let mut out = String::new();

    // ── 格式说明 ──────────────────────────────────────────────
    out.push_str("请用下面的格式回答问题。\n\n");
    out.push_str("Thought: 你必须先思考下一步要做什么，然后再行动\n");
    out.push_str("Action: 你要采取的行动\n");
    out.push_str("Action Input: 行动的参数（JSON 对象）\n");
    out.push_str("Final Answer: 最终答案（信息足够时给出）\n\n");

    // ⚠️⚠️ 「一次只输出一步」是**实测踩出来的**硬要求。
    //
    // 原始的格式说明里有一句「Thought / Action / Action Input / Observation
    // 可以重复多次」—— 那是在描述**整个对话**长什么样，
    // 但模型理解成了「我在一次回复里要把整个对话写完」。
    //
    // 后果非常严重（实测）：
    //
    // ```text
    // 步骤 3 的模型原始输出（一次写完了 4 步）：
    //   Thought: ...
    //   Action: sql_db_query_checker
    //   Action Input: {...}
    //   Observation: 列: Name | avg_ms          ← 模型自己写的！
    //   行数: 5
    //   ---
    //   摇滚 | 352000.0                        ← 模型自己编的！
    //   ...
    //   Final Answer: 平均时长最长的是摇滚，393600 毫秒   ← 编的
    // ```
    //
    // 它把**整条轨迹**（包括 Observation 和最终答案）一口气编了出来 ——
    // 数据库**一次都没被访问**，数字全是编的。
    //
    // 所以这里必须明确说：**你一次只输出一步，然后停下**。
    out.push_str(
        "**每次只输出一步**（`Thought` + `Action` + `Action Input`），\
                  然后立刻停下等待工具结果。\n",
    );
    out.push_str(
        "`Observation:` 是**工具返回**的内容，会出现在下一次对话里 —— \
                  那不是你写的，你**绝对不要自己编造** `Observation:` 行。\n\n",
    );

    // ── 可用工具 ──────────────────────────────────────────────
    if !tools.is_empty() {
        out.push_str("你可以使用这些工具：\n\n");
        out.push_str(&render_tool_descriptions(tools));
        out.push_str("\n\n");
        out.push_str(&format!(
            "`Action` 必须是下面这些工具之一：[{}]\n\n",
            tools
                .iter()
                .map(|t| t.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        out.push_str("每次只调用一个工具，然后根据 Observation 决定下一步。\n\n");
    } else {
        out.push_str("你没有可用的工具，请直接给出最终答案。\n\n");
    }

    // ── few-shot 示例 ─────────────────────────────────────────
    if few_shot && !tools.is_empty() {
        out.push_str(
            "下面是几个**示例**，它们演示工具怎么用\
             （注意：示例里的表名和问题都不是真的）：\n\n",
        );
        out.push_str(FEW_SHOT);
        out.push_str("\n\n");
    }

    // ── 真实问题 + 已有轨迹 ───────────────────────────────────
    //
    // ⚠️⚠️ 必须和示例明显区分开 —— 14-agent-tool 实测踩到的坑。
    out.push_str("========== 以下是真实对话（不是示例）==========\n\n");
    out.push_str(&format!("真实问题: {question}\n"));
    if !trajectory.trim().is_empty() {
        out.push_str("\n已完成的步骤：\n");
        out.push_str(trajectory.trim_end());
        out.push('\n');
    }
    out.push_str("\n现在请基于以上真实对话继续，**只输出下一步**（`Thought` + `Action` + `Action Input`），然后停下。");

    out
}

/// few-shot 示例。
///
/// ## ⚠️ 示例里的问题必须和真实提问**错开**
///
/// 这是 13/14 实测踩到的坑：示例问题和真实问题撞车时，
/// 模型会分不清「我刚查到的 Observation」和「示例里的」，
/// 于是照着示例编答案。
///
/// 这里的示例问的是「**有哪些流派**」（列表演示），
/// 而默认演示问题问的是「**平均时长**」（聚合 + JOIN 演示）—— 两者错开。
///
/// ## ⚠️⚠️ 示例里**只能有一步**，不能有 Observation / Final Answer
///
/// 这也是实测踩出来的坑（见 [`build_prompt`] 的说明）。
///
/// 原始版本的 few-shot 展示的是**完整轨迹**：
///
/// ```text
/// Thought: ...
/// Action: sql_db_list_tables
/// Action Input: {}
/// Observation: 可用数据表（5 张）: ...     ← 这一行是致命的
/// Thought: 看到了 Genre 表...
/// Action: sql_db_schema
/// ...
/// Final Answer: ...
/// ```
///
/// 模型学到的模式是「一次回复要把整个对话写完」，
/// 于是它**自己**写出了 `Observation:` 和编造的查询结果。
///
/// 所以现在的示例只有 `Thought` + `Action` + `Action Input` 三行，
/// 而且**用 `---` 分隔**不同的示例，明确表示「每个示例都是独立的一步」。
///
/// ## ⚠️ 示例 2 专门演示「被护栏拦截后怎么改」
///
/// 这是本项目最重要的一条行为模式。
pub const FEW_SHOT: &str = r#"下面每个示例都是**独立的一步**（`Thought` + `Action` + `Action Input`），
示例之间用 `---` 分隔。

⚠️ 注意：示例里**只有** `Thought` / `Action` / `Action Input` 三行 ——
没有工具返回的结果行，也没有最终答案行。
工具结果（Observation）是由**程序**返回给你的，只出现在下一次对话里；
轮到你回答时，你**只输出一步**，然后停下等待。

---
示例 1
Question: 数据库里都有哪些流派？
Thought: 用户问的是有哪些流派。我不知道表名，得先看数据库里有哪些表。
Action: sql_db_list_tables
Action Input: {}
---
示例 2
Question: 数据库里都有哪些流派？
Thought: 上一轮列出了表，我看到了 Genre 表。现在看它的结构，确认列名。
Action: sql_db_schema
Action Input: {"table_names": "Genre"}
---
示例 3
Question: 数据库里都有哪些流派？
Thought: 结构清楚了。查一下全部流派名称，执行前先检查一遍。
Action: sql_db_query_checker
Action Input: {"query": "SELECT Name FROM Genre LIMIT 5"}
---
示例 4
Question: 数据库里都有哪些流派？
Thought: 检查通过，现在执行这条只读查询。
Action: sql_db_query
Action Input: {"query": "SELECT Name FROM Genre LIMIT 5"}
---
示例 5
Question: 帮我删掉所有 2020 年之前的专辑。
Thought: 用户想要删除数据，但这个数据库是只读的。数据库只允许只读查询。
Action: sql_db_query
Action Input: {"query": "DELETE FROM Album WHERE ReleaseYear < 2020"}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::StaticChecker;
    use crate::db::Database;
    use crate::tools::{ListTablesTool, QueryCheckerTool, QueryTool, SchemaTool};
    use agent_14::tool::Tool;
    use std::sync::Arc;

    fn tools() -> Vec<ToolSpec> {
        let dir = tempfile::tempdir().unwrap();
        let seed = dir.path().join("seed.sql");
        std::fs::write(
            &seed,
            "CREATE TABLE Genre (GenreId INTEGER PRIMARY KEY, Name TEXT);",
        )
        .unwrap();
        let db_path = dir.path().join("t.db");
        crate::db::load_seed(&seed, &db_path).unwrap();
        let db = Arc::new(Database::open_readonly(&db_path).unwrap());

        vec![
            ListTablesTool::new(db.clone()).spec(),
            SchemaTool::new(db).spec(),
            QueryTool::new(Arc::new(Database::open_readonly(&db_path).unwrap())).spec(),
            QueryCheckerTool::new(Arc::new(StaticChecker::new())).spec(),
        ]
    }

    #[test]
    fn system_prompt_mentions_readonly() {
        let system = system_prompt();
        assert!(system.contains("只读"));
        assert!(system.contains("DELETE"));
        assert!(system.contains("LIMIT"));
    }

    #[test]
    fn system_prompt_requires_table_discovery_first() {
        let system = system_prompt();
        assert!(system.contains("sql_db_list_tables"));
        assert!(system.contains("sql_db_schema"));
    }

    #[test]
    fn system_prompt_explains_error_handling() {
        let system = system_prompt();
        assert!(system.contains("no such column"));
        assert!(system.contains("no such table"));
    }

    #[test]
    fn prompt_lists_tool_names() {
        let prompt = build_prompt(&tools(), "问题", "", true);
        assert!(prompt.contains("[sql_db_list_tables"));
        assert!(prompt.contains("sql_db_query"));
    }

    #[test]
    fn prompt_renders_parameters() {
        let prompt = build_prompt(&tools(), "问题", "", true);
        assert!(prompt.contains("table_names (string, 必填)"));
        assert!(prompt.contains("query (string, 必填)"));
    }

    #[test]
    fn prompt_separates_real_dialogue_from_examples() {
        // ⚠️ 回归测试（14 实测踩过的坑）。
        let prompt = build_prompt(&tools(), "哪个流派平均曲目最长？", "", true);

        assert!(prompt.contains("以下是真实对话（不是示例）"));
        assert!(prompt.contains("真实问题: 哪个流派平均曲目最长？"));
        assert!(!prompt.contains("Question: 哪个流派平均曲目最长？"));

        let example_pos = prompt.find("下面是几个**示例**").unwrap();
        let real_pos = prompt.find("以下是真实对话").unwrap();
        assert!(example_pos < real_pos);
    }

    #[test]
    fn trajectory_is_labelled_as_completed_steps() {
        let trajectory =
            "Thought: 查表\nAction: sql_db_list_tables\nAction Input: {}\nObservation: Genre";
        let prompt = build_prompt(&tools(), "问题", trajectory, true);
        assert!(prompt.contains("已完成的步骤："));
    }

    #[test]
    fn few_shot_demonstrates_every_step_of_the_chain() {
        assert!(FEW_SHOT.contains("Action: sql_db_list_tables"));
        assert!(FEW_SHOT.contains("Action: sql_db_schema"));
        assert!(FEW_SHOT.contains("Action: sql_db_query_checker"));
        assert!(FEW_SHOT.contains("Action: sql_db_query"));
    }

    #[test]
    fn few_shot_never_shows_an_observation_or_final_answer() {
        // ⚠️⚠️ 回归测试（实测踩过的坑）：
        //
        // few-shot 里**绝对不能**出现 `Observation:` 或 `Final Answer:`。
        //
        // 原因：模型会学到「一次回复要把整条对话写完」这个模式，
        // 然后**自己编造** Observation 和查询结果 ——
        // 数据库一次都没被访问，答案里的数字全是编的。
        //
        // 修法：few-shot 只展示「独立的一步」（Thought + Action + Action Input）。
        assert!(!FEW_SHOT.contains("Observation:"));
        assert!(!FEW_SHOT.contains("Final Answer:"));
    }

    #[test]
    fn few_shot_separates_examples_with_dashes() {
        // 示例之间必须有明确分隔 —— 否则模型分不清「这是新的一步」。
        assert!(FEW_SHOT.contains("---"));
        assert!(FEW_SHOT.contains("示例 1"));
        assert!(FEW_SHOT.contains("示例 5"));
    }

    #[test]
    fn prompt_tells_the_model_to_output_only_one_step() {
        // ⚠️ 回归测试：格式说明必须强调「一次只输出一步」。
        let prompt = build_prompt(&tools(), "问题", "", true);
        assert!(prompt.contains("每次只输出一步"));
        assert!(prompt.contains("绝对不要自己编造"));
    }

    #[test]
    fn few_shot_demonstrates_guard_rejection() {
        // ⚠️ 示例里必须演示「被护栏拦截后怎么改」——
        //    这是本项目最重要的一条行为模式。
        assert!(FEW_SHOT.contains("DELETE FROM Album"));
        assert!(FEW_SHOT.contains("只读"));
    }

    #[test]
    fn few_shot_examples_differ_from_the_default_question() {
        // ⚠️ 回归测试：示例问题不能和默认演示问题撞车。
        let default_question = "哪个流派的曲目平均时长最长？";
        assert!(!FEW_SHOT.contains(default_question));
    }
}
