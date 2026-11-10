//! 属性测试（property-based testing）—— 14-agent-tool 的 **JSON Schema 子集校验器**。
//!
//! ## 校验器为什么需要属性测试
//!
//! 校验器的输入是**模型填的工具参数** —— 完全不可信。
//! 而且校验器的输出**不是给人看的，是给模型看的** ——
//! 它会被渲染成 `Observation` 喂回模型，模型读完要能改对。
//!
//! 现有单元测试覆盖了「典型错误」（类型错、缺必填、超范围）。
//! 属性测试补的是**组合爆炸**：
//! 嵌套对象 + 数组 + 枚举 + 范围 + 未知字段同时出现时，
//! 校验器会怎么表现？
//!
//! ## 不变量清单
//!
//! | 不变量 | 含义 |
//! |---|---|
//! | `never_panics` | 任意 schema + 参数都不 panic |
//! | `valid_args_pass` | 合法参数必须通过（**不能误伤**） |
//! | `invalid_args_fail` | 类型错 / 缺必填必被拦（**不能漏放**） |
//! | `issue_paths_are_non_empty` | 每个问题都有可定位的路径 |
//! | `unknown_fields_rejected_when_closed` | `additionalProperties: false` 时未知字段必被拦 |
//! | `defaults_do_not_break_validation` | 填完默认值后校验结果一致 |

use agent_14::schema::{fill_defaults, render_schema, validate};
use proptest::prelude::*;
use serde_json::{Map, Value, json};

/// 一个「天气工具」的 Schema（14 的真实用例）。
fn weather_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "城市名"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            },
            "days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 7
            },
            "detail": {
                "type": "object",
                "properties": {
                    "raw": { "type": "boolean" }
                }
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["location"],
        "additionalProperties": false
    })
}

/// 生成任意 JSON 值（递归）。
fn json_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::from),
        any::<f64>().prop_map(Value::from),
        "[a-zA-Z0-9一-鿿 ]{0,20}".prop_map(Value::String),
    ];

    leaf.prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
            prop::collection::hash_map("[a-zA-Z_]{1,8}", inner, 0..4,)
                .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
}

/// 生成一个「合法的工具参数」（只填合法值）。
fn valid_args() -> impl Strategy<Value = Value> {
    (
        "[a-zA-Z一-鿿 ]{1,12}",
        prop::option::of(prop::sample::select(vec![
            Value::String("celsius".to_string()),
            Value::String("fahrenheit".to_string()),
        ])),
        prop::option::of(1i64..=7),
        prop::option::of(any::<bool>()),
        prop::option::of(prop::collection::vec("[a-z]{1,6}", 0..3)),
    )
        .prop_map(|(location, units, days, detail, tags)| {
            let mut obj = Map::new();
            obj.insert("location".to_string(), Value::String(location));
            if let Some(u) = units {
                obj.insert("units".to_string(), u);
            }
            if let Some(d) = days {
                obj.insert("days".to_string(), Value::from(d));
            }
            if let Some(d) = detail {
                obj.insert("detail".to_string(), json!({ "raw": d }));
            }
            if let Some(t) = tags {
                obj.insert(
                    "tags".to_string(),
                    Value::Array(t.into_iter().map(Value::String).collect()),
                );
            }
            Value::Object(obj)
        })
}

/// 生成一个「必填参数缺失或类型错误」的参数。
fn invalid_args() -> impl Strategy<Value = Value> {
    prop_oneof![
        // 缺必填
        prop::collection::hash_map("[a-zA-Z_]{1,8}", json_value(), 0..3)
            .prop_map(|m| Value::Object(m.into_iter().collect())),
        // location 类型错
        Just(json!({ "location": 42 })),
        Just(json!({ "location": ["a"] })),
        Just(json!({ "location": { "nested": 1 } })),
        // days 类型错（1.0 不是 integer）
        Just(json!({ "location": "北京", "days": 1.5 })),
        // days 超范围
        Just(json!({ "location": "北京", "days": 100 })),
        // units 不在枚举里
        Just(json!({ "location": "北京", "units": "kelvin" })),
        // 未知字段
        Just(json!({ "location": "北京", "unknown": 1 })),
    ]
}

/// 造一个「多了一个未知字段」的对象。
fn with_unknown_field(known: &str) -> Value {
    let mut obj = Map::new();
    obj.insert("location".to_string(), Value::String("北京".to_string()));
    obj.insert(known.to_string(), Value::String("x".to_string()));
    Value::Object(obj)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// ⭐ 任意 schema + 参数都不 panic。
    #[test]
    fn validator_never_panics(
        schema in json_value(),
        args in json_value(),
    ) {
        let _ = validate(&schema, &args);
        let _ = render_schema(&schema);
    }

    /// ⭐ **合法参数必须通过** —— 校验器不能误伤。
    ///
    /// 误伤的代价是「模型白白重写一次参数」。
    #[test]
    fn valid_args_pass(args in valid_args()) {
        let validation = validate(&weather_schema(), &args);
        prop_assert!(
            validation.is_ok(),
            "合法参数不该被拦：{} —— 问题：{}",
            args.to_string(),
            validation.message()
        );
    }

    /// ⭐ **非法参数必须被拦** —— 校验器不能漏放。
    ///
    /// 漏放的代价是「把错参数发给真实工具」（比如天数 100 天）。
    #[test]
    fn invalid_args_fail(args in invalid_args()) {
        let validation = validate(&weather_schema(), &args);
        prop_assert!(
            !validation.is_ok(),
            "非法参数必须被拦：{}",
            args.to_string()
        );
    }

    /// 每个问题都有**非空路径** —— 模型要靠它知道改哪个参数。
    #[test]
    fn issue_paths_are_non_empty(
        schema in json_value(),
        args in json_value(),
    ) {
        for issue in validate(&schema, &args).issues {
            prop_assert!(
                !issue.path.is_empty(),
                "问题路径不能为空：{:?}",
                issue
            );
            prop_assert!(
                !issue.message.is_empty(),
                "问题描述不能为空：{:?}",
                issue
            );
        }
    }

    /// `additionalProperties: false` 时未知字段必被拦。
    #[test]
    fn unknown_fields_are_rejected_when_closed(
        known in "[a-zA-Z_]{1,8}",
    ) {
        // 除非恰好撞上已知字段名。
        prop_assume!(known != "location");
        prop_assume!(known != "units");
        prop_assume!(known != "days");
        prop_assume!(known != "detail");
        prop_assume!(known != "tags");

        let validation = validate(&weather_schema(), &with_unknown_field(&known));

        prop_assert!(
            !validation.is_ok(),
            "未知字段 `{}` 必须被拦",
            known
        );
    }

    /// 未知字段被拦时，错误信息里要**点名那个字段**。
    #[test]
    fn unknown_field_is_named_in_the_message(
        known in "[a-zA-Z_]{3,8}",
    ) {
        prop_assume!(known != "location");
        prop_assume!(known != "units");
        prop_assume!(known != "days");
        prop_assume!(known != "detail");
        prop_assume!(known != "tags");

        let message = validate(&weather_schema(), &with_unknown_field(&known)).message();

        prop_assert!(
            message.contains(&known),
            "错误信息必须点名未知字段 `{}`，实际是：{}",
            known,
            message
        );
    }

    /// 填默认值**不改变**「是否合法」的判断。
    ///
    /// 流程是「填默认值 → 校验」，所以填默认值不能把非法变合法，
    /// 也不能把合法变非法。
    #[test]
    fn defaults_do_not_change_validity(
        args in valid_args(),
    ) {
        let schema = weather_schema();

        let before = validate(&schema, &args).is_ok();

        let mut obj = match args {
            Value::Object(map) => map,
            _ => unreachable!("valid_args 只产出对象"),
        };
        fill_defaults(&schema, &mut obj);
        let after = validate(&schema, &Value::Object(obj)).is_ok();

        prop_assert_eq!(before, after, "填默认值不该改变合法性");
    }

    /// 填默认值会把缺失的 `units` 补上 `celsius`。
    #[test]
    fn defaults_fill_missing_optional_fields(args in valid_args()) {
        let schema = weather_schema();
        let mut obj = match args {
            Value::Object(map) => map,
            _ => unreachable!(),
        };
        prop_assume!(!obj.contains_key("units"));

        fill_defaults(&schema, &mut obj);
        prop_assert_eq!(obj.get("units"), Some(&Value::String("celsius".to_string())));
    }

    /// 校验是**幂等**的 —— 同样的输入两次得到同样的问题列表。
    #[test]
    fn validation_is_deterministic(
        schema in json_value(),
        args in json_value(),
    ) {
        let first = validate(&schema, &args);
        let second = validate(&schema, &args);
        prop_assert_eq!(first.issues, second.issues, "校验结果必须稳定");
    }
}
