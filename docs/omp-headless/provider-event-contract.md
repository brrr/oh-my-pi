# omp-headless · provider↔loop 事件接口契约

> Contract-Version: 1（2026-07-25，WP-1.1a 钉死）
> Rust 权威实现：`crates/pi-ai/src/event.rs`（事件）+ `crates/pi-ai/src/message.rs`（消息模型）
> TS 对照源：`packages/ai/src/types.ts:900-923`（事件联合）/ `:712` `AssistantMessage` / `packages/ai/src/utils/event-stream.ts:147`（流容器）
> 变更纪律见 §9 —— **本契约的任何增删改（含 serde 名/tag 值）须先走宪章条款 2 plan 拍板，代码与文档同 commit 更新。**

## 1. 范围与地位

本契约钉死纯 Rust omp-headless 中 **provider 层（pi-ai）→ agent-loop 层（未来 pi-agent，WP-1.4）** 的事件接口。它是三条并行 track（provider / pi-core+tools / session）在 WP-1.4a 汇合的前提（R3：接口先斩后奏 → 1.3/1.4 连环返工）。

原则：**与 TS 版 1:1 语义拷贝**（U-omp-15）。JSON 形状（字段名 camelCase、tag 值）与 TS 序列化逐字节兼容，golden 对拍与 session 互读零转换。

口径更正：TS 源实数 **13 个事件变体**（此前文档口径"12"系口误）：start + text×3 + thinking×3 + image_end + toolcall×3 + done + error。

## 2. 事件总表（13 变体）

| # | tag（JSON `type`） | Rust 变体 | 载荷字段（JSON 名） | TS 源行 | 触发时机 |
|---|---|---|---|---|---|
| 1 | `start` | `Start` | `partial` | types.ts:901 | 首事件（错误短路路径除外，§7） |
| 2 | `text_start` | `TextStart` | `contentIndex`, `partial` | :902 | 文本块开始 |
| 3 | `text_delta` | `TextDelta` | `contentIndex`, `delta`, `partial` | :903 | 文本增量 |
| 4 | `text_end` | `TextEnd` | `contentIndex`, `content`, `partial` | :904 | 文本块完成（content=全文） |
| 5 | `thinking_start` | `ThinkingStart` | `contentIndex`, `partial` | :905 | 思考块开始 |
| 6 | `thinking_delta` | `ThinkingDelta` | `contentIndex`, `delta`, `partial` | :906 | 思考增量 |
| 7 | `thinking_end` | `ThinkingEnd` | `contentIndex`, `content`, `partial` | :907 | 思考块完成 |
| 8 | `image_end` | `ImageEnd` | `contentIndex`, `content`(ImageContent), `partial` | :908 | 图像块完成（无 start/delta） |
| 9 | `toolcall_start` | `ToolcallStart` | `contentIndex`, `partial` | :909 | 工具调用开始（id/name 已知，arguments 为 `{}`） |
| 10 | `toolcall_delta` | `ToolcallDelta` | `contentIndex`, `delta`(partial JSON 串), `partial` | :910 | 工具参数流式增量 |
| 11 | `toolcall_end` | `ToolcallEnd` | `contentIndex`, `toolCall`(ToolCall 全量), `partial` | :911 | 工具调用完成 |
| 12 | `done` | `Done` | `reason`("stop"\|"length"\|"toolUse"), `message` | :912-917 | 成功终止（恰一个终止事件） |
| 13 | `error` | `Error` | `reason`("aborted"\|"error"), `error` | :918-923 | 失败/中止终止 |

`redactedThinking` 与 `fallback` 内容块**没有专属事件**（TS 同），只出现在后续 `partial.content` 中。

## 3. partial 快照语义

- 每个增量事件带 `partial`：截至该事件的**累积不可变快照**（`Arc<AssistantMessage>`，serde 透明——JSON 与内联消息完全一致）。
- usage/stopReason 等元数据滚动在 `partial` 内，**无独立 usage 事件**。
- 消费者**不得**假设跨事件的对象同一性（Arc 共享是实现细节）；快照一旦发出即不再变。
- `done.message` / `error.error` 是终值；`error.error` 是携带 `errorMessage`/`errorStatus` 的 AssistantMessage（错误是值不是异常）。

## 4. AssistantMessage 字段表（WP-1.1a 填充面）

| Rust 字段 | JSON 名 | TS 行 | 1.1a | 说明 |
|---|---|---|---|---|
| （tag） | `role`="assistant" | :713 | ✅ | serde tag |
| `content` | `content` | :714 | ✅ | 6 种块：text/thinking/redactedThinking/fallback/image/toolCall（tag 值同 TS） |
| `api` | `api` | :722 | ✅ | 恒 `"anthropic-messages"` |
| `provider` | `provider` | :723 | ✅ | 配置的 provider id（如 `"deepseek"`） |
| `model` | `model` | :724 | ✅ | 请求 model（响应回显不覆盖；fallback-served 覆盖逻辑归后续 WP） |
| `usage` | `usage` | :736 | ✅ | §6 换算；cost 全 0（定价归 catalog 层，后续 WP） |
| `stop_reason` | `stopReason` | :737 | ✅ | §5 映射 |
| `response_id` | `responseId` | :727 | ✅ | 响应 `id` |
| `timestamp` / `duration` | 同名 | :756-757 | ✅ | 请求起点 ms / 时长 ms |
| `error_message` / `error_status` | `errorMessage`/`errorStatus` | :739/:743 | ✅ | 错误路径填充 |
| `stop_details` | `stopDetails` | :738 | ⏸ 1.1b | SSE `message_delta.stop_details` 才有 |
| `ttft` | `ttft` | :758 | ⏸ 1.1b | 首 token 时延需流式 |
| `context_snapshot` / `retry_recovery` / `upstream_provider` / `tool_call_abort_messages` / `error_id` / `disabled_features` / `provider_payload` | camelCase 同名 | :725-755 | ⏸ 后续 | `retryRecovery`/`providerPayload` 在 Rust 侧为 opaque `Value`（harness 专属结构不强类型化） |

其余角色（`UserMessage`/`DeveloperMessage`/`ToolResultMessage`/`Message` untagged by `role`）已同步定义为请求侧上下文最小面。

## 5. StopReason 映射表（wire → model）

照抄 `anthropic.ts:4303-4331` `mapStopReason`（Rust: `convert::map_stop_reason`）：

| wire | model | 备注 |
|---|---|---|
| `end_turn` | `stop` | |
| `max_tokens` | `length` | |
| `model_context_window_exceeded` | `length` | 内容有效仅截断 |
| `tool_use` | `toolUse` | |
| `stop_sequence` | `stop` | 正常完成 |
| `pause_turn` | `stop` | stop 足够 → 重提交 |
| `refusal` | `error` | |
| `sensitive` | `error` | 安全过滤 |
| 未知值 | `stop` | 服务端新增值先行，降级不失败（TS default 分支） |

## 6. Usage 换算表（wire → model）

照抄 `anthropic.ts:2155-2163` + `applyAnthropicUsageExtras`（:1590）（Rust: `convert::convert_usage`）：

| wire | model（camelCase） | 规则 |
|---|---|---|
| `input_tokens` | `input` | 缺省 0 |
| `output_tokens` | `output` | 缺省 0 |
| `cache_read_input_tokens` | `cacheRead` | 缺省 0 |
| `cache_creation_input_tokens` | `cacheWrite` | 缺省 0 |
| （求和） | `totalTokens` | input+output+cacheRead+cacheWrite |
| `cache_creation.{ephemeral_5m,1h}` | `cttl.{ephemeral5m,ephemeral1h}` | 仅 >0 的字段；全 0 → 无 cttl |
| `server_tool_use.{web_search,web_fetch}_requests` | `server.{webSearch,webFetch}` | 仅 >0；全 0 → 无 server |
| `iterations` | —— | 1.1a 不消费（server-side fallback 记账，后续 WP） |
| —— | `cost.*` | 全 0（定价归 catalog） |

## 7. 事件序不变量 + 非流式合成序列规范

不变量（挂 L3 测试 `tests/events.rs`）：

1. 首事件必为 `start`——**除**纯错误路径（请求层失败）：仅发一个 `error`；
2. 终止事件**恰一个**（`done` xor `error`），终止后 push 为 no-op；
3. `contentIndex` 单调不减；同块 `*_start` → `*_delta`* → `*_end` 成对有序；
4. `partial.content` 长度单调不减；终值 == `done.message`。

非流式合成序列（`convert::emit_nonstream_events`，1.1a 唯一生产者）：

- `start`（content 空）→ 逐块：text/thinking 发 `*_start`（空壳块入 partial）→ 单发一个全量 `*_delta` → `*_end`；toolcall 三连（start 时 arguments=`{}`，delta=完整 JSON 串）；image 仅 `image_end`；redactedThinking/fallback 不发事件仅入 partial → `done`。
- 所有 partial 携带终值的 usage/stop 元数据（非流式无中间态可言，文档化为契约行为）。
- 1.1b 真 SSE 接管后，多 delta 细粒度化，但事件种类/字段/不变量不变。

## 8. 错误映射

| AiError | 事件 | stopReason | 附加 |
|---|---|---|---|
| `Api{status,..}` | `error(reason="error")` | `error` | `errorMessage`="<status> <body>"、`errorStatus` |
| `Connection`/`ConnectionTimeout`/`Decode`/`Auth` | `error(reason="error")` | `error` | `errorMessage` |
| `Aborted`（1.1b 接线） | `error(reason="aborted")` | `aborted` | |

## 9. 变更纪律

1. 事件枚举 / 消息模型 / serde 名 / tag 值的**任何**增删改 → 先走宪章条款 2 plan（U-X 拍板），后动代码；
2. 代码与本文档**同 commit** 更新，commit 带 `OMP-WP:` trailer；
3. `Contract-Version` 递增并在下表登记：

| 版本 | 日期 | 变更 | WP |
|---|---|---|---|
| 1 | 2026-07-25 | 初版钉死（13 变体 + 消息模型 + 映射表 + 合成序列规范） | 1.1a |

## 附录 A · DeepSeek Anthropic 兼容端点注意点（L2 实测 2026-07-25）

- base `https://api.deepseek.com/anthropic`，全 URL `…/anthropic/v1/messages`；base_url 归一化只剥尾部 `/v1`，不伤 `/anthropic` 段；
- 可用模型 `deepseek-v4-pro` / `deepseek-v4-flash`（`deepseek-chat` 已下线，400 明示）；flash 默认输出 thinking 块（signature=消息 id），max_tokens 需给足否则 `max_tokens` 截断在 thinking 块内；
- 认证 `x-api-key` 生效；`anthropic-version` 头照发无害；`?beta=true` query **不发**（官方端点差异，client flag 预留）；
- usage 含私有字段 `service_tier`（忽略）、缺 `cache_creation`/`server_tool_use`（Option 容忍）；
- 已知延期：tool schema 清洗（`anthropic.ts:3909` `normalizeAnthropicToolSchemaNode`）归 WP-1.2；SSE/取消贯穿/背压条款归 WP-1.1b。
