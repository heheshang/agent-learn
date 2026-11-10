//! 模糊测试目标 —— 14-agent-tool 的 **JSON Schema 子集校验器**。
//!
//! ## 这个目标在找什么
//!
//! 1. **panic** —— 校验器在 Agent 循环的关键路径上，
//!    它挂了整个工具调用就挂了。
//! 2. **不一致** —— 同一个输入两次校验，问题列表必须一样。
//!    不一致意味着校验结果不可复现，
//!    而 14 的错误信息是要**喂回模型让它纠正**的 ——
//!    不稳定的错误信息会让模型反复改不对。

#![no_main]

use libfuzzer_sys::fuzz_target;
use agent_14::schema::{fill_defaults, render_schema, validate};
use serde_json::Value;

fuzz_target!(|data: &[u8]| {
    // 把任意字节当成一段 JSON 文本尝试解析。
    //
    // ⚠️ 用 `from_slice` 而不是 `from_str` ——
    //    `from_str` 要求输入是合法 UTF-8，
    //    `from_slice` 只要求**字节是合法 UTF-8 序列**（对 JSON 更宽松）。
    let Ok(schema) = serde_json::from_slice::<Value>(data) else {
        return;
    };

    // 1. 校验器不能 panic。
    let validation = validate(&schema, &schema);

    // 2. 渲染不能 panic。
    let _ = render_schema(&schema);

    // 3. 校验必须确定（同输入同输出）。
    let again = validate(&schema, &schema);
    assert_eq!(
        validation.issues, again.issues,
        "校验结果必须确定（schema={schema:?}）"
    );

    // 4. 每个问题都有非空路径与非空描述。
    for issue in &validation.issues {
        assert!(
            !issue.path.is_empty(),
            "问题路径不能为空（schema={schema:?}）"
        );
        assert!(
            !issue.message.is_empty(),
            "问题描述不能为空（schema={schema:?}）"
        );
    }

    // 5. 填默认值不能改变「是否合法」。
    //
    // 流程是「填默认值 → 校验」，所以填默认值不该把非法变合法。
    if let Value::Object(obj) = &schema {
        let mut filled = obj.clone();
        fill_defaults(&schema, &mut filled);
        let before = validate(&schema, &Value::Object(obj.clone())).is_ok();
        let after = validate(&schema, &Value::Object(filled)).is_ok();
        assert_eq!(
            before, after,
            "填默认值不该改变合法性（schema={schema:?}）"
        );
    }
});
