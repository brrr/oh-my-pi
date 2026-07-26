# omp-headless · provider↔loop 事件接口契约

> Contract-Version: 1.2（2026-07-26，WP-1.6a：接口本体（13 变体）不变；附录 B 三项流式简化升定案——流建立后重试 A1 / idle watchdog A2 / spliced 去重 A3，§7 补硬化语义）
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
| `stop_details` | `stopDetails` | :738 | ✅ 1.1b | SSE `message_delta.stop_details`（error 类 stop 时填充） |
| `ttft` | `ttft` | :758 | ✅ 1.1b | 首 content_block_start 时刻记录 |
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
- **WP-1.1b 起 `Client::stream()` 为真 SSE 路径**（`sse.rs` 帧解析 + `builder.rs` 状态机）：delta 细粒度化、partial 渐进（usage 从 message_start 起有值，message_delta 到达时覆写 output 等字段并重算 total）、ttft/duration 真实记录；`emit_nonstream_events` 仍是 `complete()` 消费者与错误短路路径的合成器。事件种类/字段/不变量不变。
- 流式补充语义：SSE `error` 帧 / 传输层断连 / 取消（`stream_with_cancel`）→ 终止 `error` 事件，**已累积内容保留在 error.error.content**；`message_stop` 之前断流 → dangling 块补发 `*_end` 后按当前 stop_reason 收终（TS anomaly 语义）；`message_start` 都没到 → error("stream ended before message_start")。
- **WP-1.6a 硬化语义**（`stream_runner::drive_stream` 包住 `sse.rs` + `builder.rs`，附录 B A1/A2/A3 定案）：head 到达后按需**重开重试**（首内容前）+ **双看门狗**（first-event / idle），并对 spliced 重连**去重重放**。前置 `start` 事件全程只发一次（跨重试保持）；一次 head-open 失败仍走错误短路合成路径（`emit_nonstream_events`，请求从未成流）；上述不变量（首 `start`、恰一终止、contentIndex 单调、partial 单调）在硬化路径下继续成立。
- 背压条款：流容器为无界队列（TS 对齐），生产者永不阻塞；SSE 逐帧生产的内存水位 = 未消费事件 × Arc 快照（O(指针)），风险与 TS 同级。真背压（有界 + 生产者暂停）仍留后续 WP 硬化议题（不在 1.6a 段 1 范围）。

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
| 1.1 | 2026-07-26 | 接口本体不变；§7 补 SSE 流式细则与背压条款；新增附录 B（流式简化清单）；`stream_with_cancel` 取消语义 | 1.1b |
| 1.2 | 2026-07-26 | 接口本体（13 变体）不变；附录 B 三项流式简化升**定案**——A1 流建立后首内容前重试（预算 10 + 退避）/ A2 双看门狗（first-event 可重试 + idle 终止）/ A3 spliced envelope 去重重放；新增 `stream_runner` 驱动 + 断流混沌 fixture 回归资产（`crates/pi-ai/tests/fixtures/sse_chaos_*.sse`）；§7 补硬化语义 | 1.6a |

## 附录 A · DeepSeek Anthropic 兼容端点注意点（L2 实测 2026-07-25）

- base `https://api.deepseek.com/anthropic`，全 URL `…/anthropic/v1/messages`；base_url 归一化只剥尾部 `/v1`，不伤 `/anthropic` 段；
- 可用模型 `deepseek-v4-pro` / `deepseek-v4-flash`（`deepseek-chat` 已下线，400 明示）；flash 默认输出 thinking 块（signature=消息 id），max_tokens 需给足否则 `max_tokens` 截断在 thinking 块内；
- 认证 `x-api-key` 生效；`anthropic-version` 头照发无害；`?beta=true` query **不发**（官方端点差异，client flag 预留）；
- usage 含私有字段 `service_tier`（忽略）、缺 `cache_creation`/`server_tool_use`（Option 容忍）；
- 已知延期：tool schema 清洗（`anthropic.ts:3909` `normalizeAnthropicToolSchemaNode`）归 WP-1.2。
- 流式实测（2026-07-26）：兼容层 SSE 帧完整（message_start/content_block_*/message_delta/message_stop + ping + signature_delta，signature=消息 id 经 signature_delta 单帧下发）；真实 transcript 入 `crates/pi-ai/tests/fixtures/sse_deepseek_v4_flash.sse` 锁形。

## 附录 B · WP-1.1b 流式简化清单（vs TS anthropic.ts，均为刻意取舍）

| 项 | TS 行为 | Rust 1.1b | 归属 |
|---|---|---|---|
| thinking envelope unwrap | `unwrapAnthropicThinkingEnvelope`（Harmony 泄漏清洗） | 不做 | 按需归 1.6 |
| 工具参数流式解析 | throttled 增量 parse 进 arguments | arguments 保持 `{}` 至块关闭（partial JSON 仍经 toolcall_delta 下发）；关闭时一次 parse，失败落 `{__parseError,__rawJson}`（与 TS 兜底同形） | 定案 |
| JSON repair | `parseJsonWithRepair` | 仅 serde 严格 parse + 兜底形状 | 按需归 1.6 |
| server-side fallback | opted-in 时采纳 fallback 模型/成本 | fallback 块一律忽略（= TS 未 opt-in 分支） | 定案（barm 场景不用该 beta） |
| spliced envelope 重连 | 去重重放 | **去重重放**：重复 `message_start` 置 spliced 标志（`builder.rs` `saw_spliced_envelope`）；此后对已 `content_block_stop` 关闭过的 index（`closed_block_indexes`）的 replay `content_block_start` 静默丢弃，不产重复事件；未关闭 index 的重复 open 仍按原 anomaly 跳过（anthropic.ts:2141-2203/:2399） | **定案 A3（1.2）** |
| 流建立后 provider 重试 | 首内容前可重试 | **首内容前可重试**：head 到达后、首个 `content_block_start`（`firstTokenTime`）**前**的传输错误 / 流早断（message_start 未到）/ first-event 超时 → 重开新请求，预算 `PROVIDER_MAX_RETRIES=10`，退避 `min(0.5·2^n, 8s)·(1−25% jitter)`（`calculateAnthropicRetryDelayMs`）；首内容后失败仍直接 `error` 事件（replay-unsafe）。落点 `stream_runner::drive_stream`（anthropic.ts:2028-2621） | **定案 A1（1.2）** |
| idle watchdog / first-event timeout | StreamTimeoutError 双看门狗 | **双看门狗**：first-event（首帧前，默认 `max(100s, idle)`）+ inter-event idle（帧间，默认 120s），env `PI_STREAM_FIRST_EVENT_TIMEOUT_MS` / `PI_STREAM_IDLE_TIMEOUT_MS`（`0` 关）。超时产 `error` 事件，文本对齐 TS `StreamTimeoutError`（"…while waiting for the first event" / "…while waiting for the next event"）。first-event 超时按 A1 可重试；**idle 超时终止不重试**（TS `isLocalIdleTimeout`）。简化：看门狗守 chunk 级到达而非 TS 的 event 级计时器（慢-而-活的字节涓流不区分为进展） | **定案 A2（1.2）** |
| cost 计算 | calculateCost | 全 0 | 归 catalog 层 WP |
