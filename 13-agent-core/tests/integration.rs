//! 集成测试 —— 验证 ReAct 循环的**完整行为**。
//!
//! ## 为什么这些测试能用假 LLM
//!
//! Agent 循环是一个**状态机**：每一步的输出决定下一步的输入。
//! 有了 [`ScriptedLlm`]，整个循环可以在**零网络、零延迟、完全确定**的
//! 条件下被测试 ——
//! 包括那些靠真实模型根本触发不了的边界路径
//! （比如「模型连续 6 次要求调工具」）。
//!
//! ## 覆盖的场景
//!
//! | 场景 | 验证什么 |
//! |---|---|
//! | 单次工具调用后作答 | 主路径 |
//! | 多次工具调用 | 轨迹累积正确 |
//! | 工具失败 | 错误作为 Observation 喂回模型 |
//! | 未知工具 | 同上，且提示可用工具列表 |
//! | 达到步数上限 | 兜底再问一次，不报错 |
//! | 模型输出跑偏 | 容错当成答案 |
//! | 模式差异 | `direct` 不挂工具 |

use std::sync::Arc;

use agent_13::agent::Agent;
use agent_13::config::{AgentConfig, AgentMode, AppConfig};
use agent_13::error::AgentError;
use agent_13::llm::ScriptedLlm;
use agent_13::tools::{CalculateTool, SearchTool};
use rag_08::bm25::Bm25Document;

/// 建一个带假工具的 Agent（不碰知识库目录）。
fn agent_with(answers: Vec<String>, max_steps: usize) -> (Agent, Arc<ScriptedLlm>) {
    let documents = vec![Bm25Document {
        id: "kb.md#0".to_string(),
        source: "kb.md".to_string(),
        heading_path: vec!["ReAct 框架".to_string()],
        text: "Observation 是工具执行后返回给模型的结果。".to_string(),
    }];
    let index = agent_13::tools::index_from_documents(documents);

    let search = SearchTool::new(Arc::new(index), 3, 5, 1200);
    let calculate = CalculateTool;

    let llm = Arc::new(ScriptedLlm::new(answers));

    let mut config = AppConfig::default();
    config.agent.max_steps = max_steps;

    let agent = Agent::new(
        llm.clone(),
        vec![Arc::new(search), Arc::new(calculate)],
        config,
    );

    (agent, llm)
}

/// 构造一个只跑 `Direct` 模式的 Agent（不挂工具）。
fn direct_agent(answers: Vec<String>) -> (Agent, Arc<ScriptedLlm>) {
    let llm = Arc::new(ScriptedLlm::new(answers));
    let config = AppConfig::default();
    let agent = Agent::new(llm.clone(), Vec::new(), config);
    (agent, llm)
}

#[tokio::test]
async fn single_tool_call_then_final_answer() {
    let (agent, _llm) = agent_with(
        vec![
            "Thought: 需要查资料\nAction: search\nAction Input: {\"query\": \"Observation\"}"
                .to_string(),
            "Thought: 信息足够了\nFinal Answer: Observation 是工具返回的结果".to_string(),
        ],
        5,
    );

    let run = agent.run("Observation 是什么？").await.unwrap();

    assert_eq!(run.tool_calls, 1);
    assert_eq!(run.steps.len(), 2);
    assert!(run.answer.contains("工具返回的结果"));
    assert!(!run.hit_max_steps);
}

#[tokio::test]
async fn trajectory_accumulates_across_steps() {
    let (agent, _llm) = agent_with(
        vec![
            "Thought: 先查\nAction: search\nAction Input: {\"query\": \"a\"}".to_string(),
            "Thought: 再算\nAction: calculate\nAction Input: {\"expression\": \"1+1\"}".to_string(),
            "Final Answer: 完成".to_string(),
        ],
        5,
    );

    let run = agent.run("问题").await.unwrap();

    assert_eq!(run.tool_calls, 2);
    assert_eq!(run.steps.len(), 3);

    let trajectory = run.trajectory();
    assert!(trajectory.contains("Thought: 先查"));
    assert!(trajectory.contains("Action: search"));
    assert!(trajectory.contains("Action: calculate"));
    assert!(trajectory.contains("Observation:"));
    assert!(trajectory.contains("Final Answer: 完成"));
}

#[tokio::test]
async fn observation_is_fed_back_to_the_model() {
    let (agent, llm) = agent_with(
        vec![
            "Thought: 查\nAction: search\nAction Input: {\"query\": \"Observation\"}".to_string(),
            "Final Answer: 答案".to_string(),
        ],
        5,
    );

    agent.run("问题").await.unwrap();

    // 第二轮调用的 user 提示里必须包含第一轮的 Observation。
    let calls = llm.calls();
    assert_eq!(calls.len(), 2);
    assert!(
        calls[1].1.contains("Observation:"),
        "第二轮必须能看到 Observation"
    );
    assert!(
        calls[1].1.contains("工具执行后返回给模型的结果"),
        "Observation 必须带上工具返回的实际内容"
    );
}

#[tokio::test]
async fn tool_failure_is_reported_as_observation_not_error() {
    let (agent, llm) = agent_with(
        vec![
            // 缺少 query 参数 → 工具失败。
            "Thought: 查\nAction: search\nAction Input: {}".to_string(),
            "Final Answer: 我看到工具报错了".to_string(),
        ],
        5,
    );

    let run = agent.run("问题").await.unwrap();

    // 关键：整个运行**没有失败**。
    assert!(run.steps[0].tool_ok == Some(false));
    assert!(
        run.steps[0]
            .observation
            .as_ref()
            .unwrap()
            .contains("缺少 query")
    );

    // 错误文本必须进了下一轮的提示词。
    let calls = llm.calls();
    assert!(calls[1].1.contains("缺少 query"));
}

#[tokio::test]
async fn unknown_tool_lists_available_tools() {
    let (agent, llm) = agent_with(
        vec![
            "Thought: 用不存在的工具\nAction: nonexistent\nAction Input: {}".to_string(),
            "Final Answer: 好".to_string(),
        ],
        5,
    );

    let run = agent.run("问题").await.unwrap();

    assert!(run.steps[0].tool_ok == Some(false));
    let observation = run.steps[0].observation.as_ref().unwrap();
    assert!(observation.contains("未知工具"));
    // 必须告诉模型有哪些工具可用 —— 它要靠这个自我纠正。
    assert!(observation.contains("search"));
    assert!(observation.contains("calculate"));

    let calls = llm.calls();
    assert!(calls[1].1.contains("未知工具"));
}

#[tokio::test]
async fn hitting_max_steps_triggers_a_final_graceful_answer() {
    // 模型一直要求调工具，永远不给 Final Answer。
    let (agent, llm) = agent_with(
        vec![
            "Action: search\nAction Input: {\"query\": \"a\"}".to_string(),
            "Action: search\nAction Input: {\"query\": \"b\"}".to_string(),
            "Action: search\nAction Input: {\"query\": \"c\"}".to_string(),
        ],
        3,
    );

    let run = agent.run("问题").await.unwrap();

    // 达到上限 → 但**不报错**。
    assert!(run.hit_max_steps);
    assert_eq!(run.tool_calls, 3);
    assert!(!run.answer.is_empty());

    // 兜底会**再问一次** —— 共 4 次模型调用（3 步 + 1 次收尾）。
    assert_eq!(llm.call_count(), 4);
}

#[tokio::test]
async fn unstructured_output_is_tolerated_as_an_answer() {
    let (agent, _llm) = agent_with(vec!["我觉得答案是 42。".to_string()], 5);

    let run = agent.run("问题").await.unwrap();

    assert_eq!(run.tool_calls, 0);
    assert!(run.answer.contains("42"));
    // 但要记录下这是个异常输出。
    assert!(run.steps[0].parse_error.is_some());
}

#[tokio::test]
async fn direct_mode_never_calls_tools() {
    let (agent, llm) = direct_agent(vec!["Final Answer: 直接回答".to_string()]);

    let run = agent
        .run_with_mode("问题", AgentMode::Direct)
        .await
        .unwrap();

    assert_eq!(run.tool_calls, 0);
    assert_eq!(run.answer, "直接回答");

    // 提示词里不能出现工具。
    let calls = llm.calls();
    assert!(!calls[0].1.contains("Action:"));
    assert!(!calls[0].0.contains("search"));
}

#[tokio::test]
async fn act_only_mode_has_no_thought_in_prompt() {
    let (agent, llm) = agent_with(vec!["Final Answer: 答案".to_string()], 5);

    agent
        .run_with_mode("问题", AgentMode::ActOnly)
        .await
        .unwrap();

    let calls = llm.calls();
    assert!(!calls[0].1.contains("Thought:"));
    assert!(calls[0].1.contains("Action:"));
}

#[tokio::test]
async fn react_mode_has_thought_in_prompt() {
    let (agent, llm) = agent_with(vec!["Final Answer: 答案".to_string()], 5);

    agent.run_with_mode("问题", AgentMode::ReAct).await.unwrap();

    let calls = llm.calls();
    assert!(calls[0].1.contains("Thought:"));
}

#[tokio::test]
async fn empty_question_is_rejected() {
    let (agent, _llm) = agent_with(vec![], 5);

    let err = agent.run("   ").await.unwrap_err();
    assert!(matches!(err, AgentError::BadRequest(_)));
}

#[tokio::test]
async fn calculate_tool_is_reachable_through_the_loop() {
    let (agent, _llm) = agent_with(
        vec![
            "Thought: 算一下\nAction: calculate\nAction Input: {\"expression\": \"(12 + 34) * 2\"}"
                .to_string(),
            "Final Answer: 92".to_string(),
        ],
        5,
    );

    let run = agent.run("12+34 乘 2 等于多少").await.unwrap();

    assert_eq!(run.tool_calls, 1);
    let observation = run.steps[0].observation.as_ref().unwrap();
    assert!(observation.contains("92"));
}

#[tokio::test]
async fn agent_config_max_steps_is_respected() {
    // 5 步上限，模型一直要工具 → 最多 5 次工具调用 + 1 次收尾。
    let (agent, llm) = agent_with(
        vec![
            "Action: search\nAction Input: {\"query\": \"a\"}".to_string(),
            "Action: search\nAction Input: {\"query\": \"b\"}".to_string(),
            "Action: search\nAction Input: {\"query\": \"c\"}".to_string(),
            "Action: search\nAction Input: {\"query\": \"d\"}".to_string(),
            "Action: search\nAction Input: {\"query\": \"e\"}".to_string(),
        ],
        5,
    );

    let run = agent.run("问题").await.unwrap();

    assert!(run.hit_max_steps);
    assert_eq!(run.tool_calls, 5);
    assert_eq!(llm.call_count(), 6);
}

#[tokio::test]
async fn agent_config_defaults_are_sane() {
    let config = AgentConfig::default();
    assert!(config.max_steps > 0 && config.max_steps <= 30);
    assert!(config.max_observation_chars > 0);
}

#[tokio::test]
async fn final_graceful_call_failure_does_not_lose_the_steps() {
    // ⚠️ 回归测试。
    //
    // 早期版本在「达到步数上限 → 收尾再问一次」时用 `?` 直接传播错误，
    // 结果模型挂掉 → 整个 run 报错 → **前面几步收集的观察全丢了**。
    //
    // 但那些观察是有价值的（用户能看到 Agent 查了什么）。
    // 所以收尾失败必须**降级**，而不是让整个运行失败。
    let (agent, llm) = agent_with(
        vec![
            "Action: search\nAction Input: {\"query\": \"a\"}".to_string(),
            "Action: search\nAction Input: {\"query\": \"b\"}".to_string(),
        ],
        2,
    );

    // 脚本只有 2 个答案 —— 第 3 次（收尾）调用必然失败。
    let run = agent
        .run("问题")
        .await
        .expect("收尾失败不应该让整个运行失败");

    assert!(run.hit_max_steps);
    assert_eq!(run.tool_calls, 2);
    // 前面两步的记录**必须还在**。
    assert_eq!(run.steps.len(), 3);
    assert_eq!(run.steps[0].tool.as_deref(), Some("search"));
    assert_eq!(run.steps[1].tool.as_deref(), Some("search"));
    // 答案里要说明发生了什么。
    assert!(run.answer.contains("收尾请求失败"));
    assert_eq!(llm.call_count(), 3);
}
