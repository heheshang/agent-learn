//! 工具 —— 让模型能反过来调用你的代码。
//!
//! Rig 里一个工具要回答四个问题：
//!
//! 1. **叫什么** —— `Tool::NAME`，模型看到的名字。
//! 2. **做什么** —— `description()`，模型据此决定要不要调。
//! 3. **要什么参数** —— `parameters()`，一份 JSON Schema，模型照着填。
//! 4. **怎么执行** —— `call(&mut ToolContext, args)`，真正干活的地方。
//!
//! 两种写法：
//!
//! - `#[rig_tool]` 宏：从一个普通函数自动生成上面四样东西。
//! - 手写 `impl Tool`：需要自定义错误映射、执行注解（annotations）时用。
//!
//! `ToolContext` 是工具的「入参通道」：宿主可以把任意类型塞进去，工具里
//! 用 `get` / `require` 取出来。里面的东西**不会发给模型**。

use rig::prelude::ToolContext;
use rig::tool::{Tool, ToolErrorKind, ToolExecutionError};
use rig::{ContextValue, rig_tool};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

/// 用 `#[rig_tool]` 宏定义的工具：加两个整数。
///
/// 宏会生成：
/// - `AddTwo` 结构体（`Tool` 的实现体）
/// - `AddTwoParameters` 结构体（参数，反序列化 + JSON Schema）
/// - `ADD_TWO` 静态常量（工具本身）
#[rig_tool(
    description = "把两个整数相加",
    params(a = "第一个加数", b = "第二个加数")
)]
pub fn add_two(a: i64, b: i64) -> Result<i64, ToolExecutionError> {
    Ok(a + b)
}

/// 用 `#[rig_tool]` 宏定义的工具：按城市名查天气（这里返回假数据）。
#[rig_tool(
    name = "get_weather",
    description = "查询指定城市的当前天气",
    params(city = "城市名，例如 北京")
)]
pub fn get_weather(city: String) -> Result<String, ToolExecutionError> {
    Ok(format!("{city}：晴，25°C，东南风 2 级"))
}

/// 手写 `Tool` 实现的示例：一个需要宿主持有状态（计数器）的工具。
///
/// 注意工具实例是被 `Agent` 共享的，所以可变状态放在**实例**里，
/// 而不是 `ToolContext` 里（后者是每次调用独立的入参通道）。
#[derive(Clone, Default)]
pub struct CountCalls {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

/// 计数器的参数：空。
#[derive(Debug, Deserialize, Serialize)]
pub struct CountCallsParams {}

impl Tool for CountCalls {
    const NAME: &'static str = "count_calls";

    type Args = CountCallsParams;
    type Output = usize;
    type Error = Infallible;

    fn description(&self) -> String {
        "返回本工具被调用的次数".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let calls = self.calls.clone();
        async move { Ok(calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1) }
    }
}

/// 一份由宿主放进 `ToolContext`、只有工具能读到的数据。
///
/// 它实现了 `ContextValue`（用 `#[derive(ContextValue)]`），因此用类型本身
/// 作为槽位 key，不需要手写字符串 key。
#[derive(Debug, Clone, Serialize, Deserialize, ContextValue)]
#[context(key = "session.user")]
pub struct SessionUser {
    pub name: String,
    pub tier: String,
}

/// 一个读取 `ToolContext` 的工具：查当前登录用户是谁。
///
/// 注意工具签名里的 `&mut ToolContext` 参数：写的时候它叫 `_context`，
/// 表示「这个工具不需要上下文」；这里叫 `context`，就能读了。
///
/// `#[rig(context)]` 告诉宏：这个参数是上下文，不是模型要填的参数，
/// 因此它不会出现在 JSON Schema 里。
#[rig_tool(name = "whoami", description = "查询当前登录用户的名字和会员等级")]
pub fn whoami(#[rig(context)] context: &mut ToolContext) -> Result<String, ToolExecutionError> {
    let user = context
        .require::<SessionUser>()
        .map_err(|error| ToolExecutionError::new(ToolErrorKind::NotFound, error.to_string()))?;

    Ok(format!("{}（{} 会员）", user.name, user.tier))
}

/// 演示：工具如何从 `ToolContext` 读取宿主持有的状态。
///
/// `ToolContext` 里的值只在本进程内流转，**不会**出现在发给模型的请求里 ——
/// 模型必须通过调用工具才能间接知道它。
pub async fn context_demo() -> anyhow::Result<()> {
    let mut context = ToolContext::new();
    context.insert(SessionUser {
        name: "Rodriguez".to_string(),
        tier: "pro".to_string(),
    })?;

    let user = context.require::<SessionUser>()?;
    println!("=== 工具上下文 ===");
    println!("宿主注入的用户: {} ({})", user.name, user.tier);
    println!("说明: 这些值只存在于进程内，不发给模型。");
    Ok(())
}
