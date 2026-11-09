# 工具调用协议

## Function Calling

### 结构化协议 vs 文本协议

ReAct 文本协议靠正则解析 `Action:` 标记，结构化协议靠 API 的 `tools` 字段和返回的 `tool_calls` 数组。

结构化的优势是能精确知道「哪个 call_id 的哪个工具失败了」——错误处理才有意义。

### Schema 是对模型的承诺

Schema 写什么，模型就会以为有什么。如果 Schema 写 city 而实现读 location，模型会试三种组合才放弃。

实测代价：8 次工具调用 / 5 步 / 6999 token。统一成 location 后降到 2 次 / 2 步 / 2291 token。

### tool_choice 四档

auto 是唯一正确的默认档。none 不可靠——模型会用 `<toolcall>{"name":...}</toolcall>` 文本模拟调用，所以真正禁止用工具是不把 tools 传给模型。

required 会诱发凑数调用，让模型丧失判断力。它是强制工具调用的开关，不是更好的 auto。

## 消息历史

### 每个 tool_call 都要有回执

DeepSeek 原文：`insufficient tool messages following tool_calls message`。

少一条就是 400，且错误信息完全不提示少了哪条。工具失败也算回执。

### 思考内容要回传

开思考模式必须回传 reasoning_content，原文：`must be passed back to the API`。

assistant 消息里带上 reasoning 内容，且顺序在工具调用之前。

## 流式与并行

### parallel_tool_calls false 被忽略

实测传了仍返回 2 个调用。并行度只能靠应用层控制。

### strict true 被接受但不强制执行

改了 required 照样通过。所以本地 Schema 校验是必需品，strict 只能讨好 OpenAI。

### 有 default 的可选参数不能开 strict

strict 要求所有字段进 required，不认 default。OpenAI 的解法是把可选字段类型写成 `["string","null"]`。
