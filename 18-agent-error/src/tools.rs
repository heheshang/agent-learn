//! 工具定义 —— 复用 16 的 `ToolSpec`。
//!
//! ## 为什么直接复用
//!
//! `ToolSpec` 是「工具的契约」：名字 / 描述 / 参数 Schema / strict 标志。
//! 它和错误处理**无关**，所以没必要重写第二遍 ——
//! 重写只会带来「两个版本的 Schema 校验规则不一致」这种隐患。
//!
//! 但本项目**不复用** 16 的 `ToolRegistry` ——
//! 因为它的失败语义太粗（没有 Transient / Business / Fatal 的区分），
//! 详见 [`crate::tool`]。

pub use agent_16::tools::{
    TOOL_NAME_MAX_LEN, ToolSpec, check_schema_shape, check_strict_compliance, is_valid_tool_name,
    strict_schema,
};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_spec_carries_the_strict_flag() {
        let spec = ToolSpec::new("t", "描述", json!({ "type": "object" }));
        assert!(!spec.strict);

        let spec = spec.with_strict(true);
        assert!(spec.strict);
    }

    #[test]
    fn schema_shape_checks_are_reusable() {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": { "a": { "type": "string" } },
            "required": ["a"]
        });
        assert!(check_schema_shape(&schema).is_ok());
    }
}
