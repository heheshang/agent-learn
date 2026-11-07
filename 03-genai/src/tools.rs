//! 工具实现 —— 模型层和 agent 层共用。
//!
//! 模型层直接调函数；agent 层把它包成 `AgentTool`（带 JSON Schema），
//! 由 agent 循环自动执行。

use serde_json::json;

/// 工具内部错误。
#[derive(Debug)]
pub enum ToolError {
    /// 参数不合法。
    InvalidArguments(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArguments(msg) => write!(f, "参数不合法: {msg}"),
        }
    }
}

impl std::error::Error for ToolError {}

/// 一个极简算术求值器：只支持 + - * / 和括号。
///
/// 故意手写而不是用 eval，避免引入依赖、也避免安全风险。
pub fn calculate(expression: &str) -> Result<f64, ToolError> {
    let mut parser = Parser {
        chars: expression.chars().collect(),
        pos: 0,
    };
    let value = parser.parse_expr()?;
    parser.skip_whitespace();
    if parser.pos != parser.chars.len() {
        return Err(ToolError::InvalidArguments(format!(
            "表达式尾部有多余字符: {}",
            &expression[parser.pos.min(expression.len())..]
        )));
    }
    Ok(value)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn skip_whitespace(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_whitespace();
        self.chars.get(self.pos).copied()
    }

    /// expr := term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<f64, ToolError> {
        let mut left = self.parse_term()?;
        loop {
            match self.peek() {
                Some('+') => {
                    self.pos += 1;
                    left += self.parse_term()?;
                }
                Some('-') => {
                    self.pos += 1;
                    left -= self.parse_term()?;
                }
                _ => return Ok(left),
            }
        }
    }

    /// term := factor (('*' | '/') factor)*
    fn parse_term(&mut self) -> Result<f64, ToolError> {
        let mut left = self.parse_factor()?;
        loop {
            match self.peek() {
                Some('*') => {
                    self.pos += 1;
                    left *= self.parse_factor()?;
                }
                Some('/') => {
                    self.pos += 1;
                    let right = self.parse_factor()?;
                    if right == 0.0 {
                        return Err(ToolError::InvalidArguments("除数为零".into()));
                    }
                    left /= right;
                }
                _ => return Ok(left),
            }
        }
    }

    /// factor := number | '(' expr ')' | '-' factor
    fn parse_factor(&mut self) -> Result<f64, ToolError> {
        match self.peek() {
            Some('(') => {
                self.pos += 1;
                let value = self.parse_expr()?;
                if self.peek() != Some(')') {
                    return Err(ToolError::InvalidArguments("缺少右括号".into()));
                }
                self.pos += 1;
                Ok(value)
            }
            Some('-') => {
                self.pos += 1;
                Ok(-self.parse_factor()?)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => {
                let start = self.pos;
                while self
                    .chars
                    .get(self.pos)
                    .is_some_and(|c| c.is_ascii_digit() || *c == '.')
                {
                    self.pos += 1;
                }
                let text: String = self.chars[start..self.pos].iter().collect();
                text.parse::<f64>()
                    .map_err(|e| ToolError::InvalidArguments(format!("无法解析数字 {text}: {e}")))
            }
            Some(c) => Err(ToolError::InvalidArguments(format!("不支持的字符: {c}"))),
            None => Err(ToolError::InvalidArguments("表达式意外结束".into())),
        }
    }
}

/// 工具的 JSON Schema（模型层 `Tool` 和 agent 层 `ToolSpec` 共用）。
pub fn calculate_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "算术表达式，例如 (12 + 34) * 2"
            }
        },
        "required": ["expression"]
    })
}

#[cfg(test)]
mod tests {
    use super::calculate;

    #[test]
    fn evaluates_basic_expressions() {
        assert_eq!(calculate("1 + 2").unwrap(), 3.0);
        assert_eq!(calculate("(12 + 34) * 2").unwrap(), 92.0);
        assert_eq!(calculate("10 / 4").unwrap(), 2.5);
        assert_eq!(calculate("-3 + 1").unwrap(), -2.0);
    }

    #[test]
    fn rejects_invalid_expressions() {
        assert!(calculate("1 +").is_err());
        assert!(calculate("(1 + 2").is_err());
        assert!(calculate("1 / 0").is_err());
        assert!(calculate("a + b").is_err());
    }
}
