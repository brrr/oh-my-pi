//! `AgentEvent` → ACP `SessionUpdate` 映射 + prompt turn 的 `StopReason` 归结。
//!
//! 纯函数、无副作用、可单测——把 pi-agent 的 [`AgentEvent`] 流翻译成 ACP
//! `session/update` 的 update 负载，宿主（[`crate::server`]）再包 `sessionId`
//! 发 `SessionNotification`。逐行对照 TS `acp-event-mapper.ts` /
//! `acp-agent.ts#resolveStopReason`（packages/coding-agent/src/modes/acp/）。
//!
//! ## 映射表（与 TS mapper 对齐点）
//! | `AgentEvent` | ACP `SessionUpdate` | 对照 |
//! |---|---|---|
//! | `message_update` + `text_delta` | `agent_message_chunk`(text=delta) | mapAssistantMessageUpdate |
//! | `message_update` + `thinking_delta` | `agent_thought_chunk`(text=delta) | 同上 |
//! | `tool_execution_start` | `tool_call`(id/title/kind/rawInput) | buildToolCallStartUpdate |
//! | `tool_execution_update` | `tool_call_update`(status=`in_progress` + `rawOutput`) | `tool_execution_update` 分支 |
//! | `tool_execution_end` | `tool_call_update`(status completed/failed + content) | `tool_execution_end` 分支 |
//! | `message_end`（无 delta 时） | `agent_message_chunk`（终值文本兜底） | mapAssistantMessageEnd |
//! | 其余（turn/message start·end、agent start·end 等） | —（不映射） | mapper default `[]` |
//!
//! ## `message_end` 文本兜底（WP-1.6e，F8）
//! [`MapProgress`] 在 [`map_event`] 之上叠一层 per-message 的
//! `text_emitted` 去重态，照抄 TS `acp-event-mapper.ts` 的
//! `progress.textEmitted`（:283-352）：一条 assistant message 期间**发过**任何
//! `text_delta` → `message_end` 不再补发；**没发过** delta（非流式 provider 一
//! 次性返终值消息的情形）→ 从终值 message 抽文本补一发
//! `agent_message_chunk`，保证宿主拿得到完整文本。`DeepSeek` 流式恒有渐进
//! delta，故此兜底在现网 golden 里**不触发**（golden 事件形状不变）；它只为非流
//! 式 provider 兜底。宿主（[`crate::server`]）持一个 `MapProgress` 跨整个
//! prompt turn 复用，`message_start` 处自动重置去重态。
//! - `plan`/`todo` update **不产**（无来源，defer；见 [`crate`] 顶注）。
//! - `buildToolTitle` 只取 intent / path·command·pattern·query 主语，未移植
//!   command-tool（bash）整行文本与 eval 特判——最小可读标题，golden 自洽即可。
//! - tool 内容抽取（diff / 多模态）未移植，`tool_call_update` 只带 rawOutput +
//!   单条文本内容块（driver 只读 status，内容仅装饰）。
//! - **F6 tool_execution_update（WP-1.6e）**：映射面（`AgentEvent` →
//!   `tool_call_update` 的 `in_progress` 态）已打通并单测，但 pi-agent
//!   **尚不发射** 该事件——发射需 `Tool::execute` 加 `on_update` 流式回调（Tool
//!   trait 接口变更），属 WP-1.4b defer（`pi_tools::Tool` doc +
//!   `pi_agent::execute` 顶注均登记）。本 WP 只补映射侧零接口改动的事件面，产出
//!   侧留 defer。

use agent_client_protocol::schema::v1::{
	ContentBlock, ContentChunk, SessionUpdate, StopReason as AcpStopReason, ToolCall,
	ToolCallContent, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use pi_agent::AgentEvent;
use pi_ai::{
	AssistantMessageEvent,
	message::{Message, StopReason},
};
use regex::Regex;
use serde_json::Value;

/// 工具名 → ACP `ToolKind`（TS `mapToolKind`，acp-event-mapper.ts）。
///
/// xd:// 设备派发特判（write→execute）未移植：本 WP 六工具无 internal-url 面。
#[must_use]
pub fn map_tool_kind(tool_name: &str) -> ToolKind {
	match tool_name {
		"read" => ToolKind::Read,
		"write" | "edit" => ToolKind::Edit,
		"delete" => ToolKind::Delete,
		"move" => ToolKind::Move,
		"bash" | "shell" | "exec" | "eval" => ToolKind::Execute,
		"grep" | "glob" | "ast_grep" => ToolKind::Search,
		"web_search" => ToolKind::Fetch,
		"todo" => ToolKind::Think,
		_ => ToolKind::Other,
	}
}

/// 人类可读工具标题（TS `buildToolTitle` 最小面：intent 优先，其次主语）。
fn tool_title(tool_name: &str, args: &Value, intent: Option<&str>) -> String {
	if let Some(text) = intent {
		let trimmed = text.trim();
		if !trimmed.is_empty() {
			return trimmed.to_owned();
		}
	}
	for key in ["path", "command", "pattern", "query"] {
		if let Some(subject) = args.get(key).and_then(Value::as_str) {
			return format!("{tool_name}: {subject}");
		}
	}
	tool_name.to_owned()
}

/// 从工具结果里抽一条展示文本（`tool_call_update.content` 用；无则 `None`）。
fn result_text(result: &Value) -> Option<String> {
	match result {
		Value::String(text) => Some(text.clone()),
		Value::Object(map) => ["content", "output", "text", "llmContent", "message"]
			.into_iter()
			.find_map(|key| match map.get(key) {
				Some(Value::String(text)) => Some(text.clone()),
				_ => None,
			}),
		_ => None,
	}
}

/// 把一个 [`AgentEvent`] 映射为零或一条 ACP `SessionUpdate`。
///
/// 返回 `Vec` 而非 `Option` 是给未来一事件多 update（如 TS todo→plan 伴随）
/// 留形；本 WP 每事件至多一条。
#[must_use]
pub fn map_event(event: &AgentEvent) -> Vec<SessionUpdate> {
	match event {
		AgentEvent::MessageUpdate { assistant_message_event, .. } => {
			map_message_update(assistant_message_event)
		},
		AgentEvent::ToolExecutionStart { tool_call_id, tool_name, args, intent } => {
			let title = tool_title(tool_name, args, intent.as_deref());
			vec![SessionUpdate::ToolCall(
				ToolCall::new(tool_call_id.clone(), title)
					.kind(map_tool_kind(tool_name))
					.status(ToolCallStatus::Pending)
					.raw_input(args.clone()),
			)]
		},
		AgentEvent::ToolExecutionUpdate { tool_call_id, partial_result, .. } => {
			// F6：in-flight 工具的流式部分结果 → tool_call_update(in_progress)。
			// TS `tool_execution_update` 分支（acp-event-mapper.ts:198-216）：
			// status=in_progress，rawOutput=partialResult，有文本则附内容块。
			// pi-agent 尚不发射此事件（Tool on_update 回调 defer），映射面先备好。
			let mut fields = ToolCallUpdateFields::new()
				.status(ToolCallStatus::InProgress)
				.raw_output(partial_result.clone());
			if let Some(text) = result_text(partial_result) {
				fields = fields.content(vec![ToolCallContent::from(text)]);
			}
			vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(tool_call_id.clone(), fields))]
		},
		AgentEvent::ToolExecutionEnd { tool_call_id, result, is_error, .. } => {
			let status = if is_error.unwrap_or(false) {
				ToolCallStatus::Failed
			} else {
				ToolCallStatus::Completed
			};
			let mut fields = ToolCallUpdateFields::new()
				.status(status)
				.raw_output(result.clone());
			if let Some(text) = result_text(result) {
				fields = fields.content(vec![ToolCallContent::from(text)]);
			}
			vec![SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(tool_call_id.clone(), fields))]
		},
		// turn/message start·end、agent start·end：不映射（`message_end` 文本兜底
		// 由 [`MapProgress`] 承接，见下）。
		_ => vec![],
	}
}

/// 跨事件的映射进度态（TS `MessageProgress`：只跟踪 `text_emitted` 去重位）。
///
/// [`map_event`] 是纯 per-event 函数，无法自己判断 `message_end` 该不该补发。
/// 宿主持一个 `MapProgress` 跨整个 prompt turn 复用，把每个 [`AgentEvent`] 交给
/// [`MapProgress::map`]：常规事件透传给 [`map_event`]，另加两处 F8 逻辑——
/// `text_delta`（非空）置位 `text_emitted`，`message_start` 重置去重位，
/// `message_end` 在**未发过** delta 时从终值消息抽文本补一发。
#[derive(Debug, Default)]
pub struct MapProgress {
	/// 当前 assistant message 期间是否已发过 `agent_message_chunk`（text）。
	text_emitted: bool,
}

impl MapProgress {
	/// 新建一个初始（未发文本）的进度态。
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// 映射一个 [`AgentEvent`]，维护 `text_emitted` 去重位（F8）。
	#[must_use]
	pub fn map(&mut self, event: &AgentEvent) -> Vec<SessionUpdate> {
		match event {
			// message 边界：重置去重位，让下一条 message 独立判定兜底。
			AgentEvent::MessageStart { .. } => {
				self.text_emitted = false;
				Vec::new()
			},
			AgentEvent::MessageUpdate { assistant_message_event, .. } => {
				let updates = map_message_update(assistant_message_event);
				// 发过任何 agent_message_chunk（text）即置位，抑制 message_end 兜底。
				if updates
					.iter()
					.any(|u| matches!(u, SessionUpdate::AgentMessageChunk(_)))
				{
					self.text_emitted = true;
				}
				updates
			},
			AgentEvent::MessageEnd { message } => self.message_end_fallback(message),
			other => map_event(other),
		}
	}

	/// `message_end` 文本兜底（TS `mapAssistantMessageEnd`,
	/// acp-event-mapper.ts:336-359）：本条消息发过 delta（`text_emitted`）→
	/// 不补； 否则从终值 assistant 消息抽文本，非空则补发一条
	/// `agent_message_chunk` 并置 位。非 assistant / 无文本 →
	/// 空。收尾重置去重位（下条 message 重新判定）。
	fn message_end_fallback(&mut self, message: &Message) -> Vec<SessionUpdate> {
		if self.text_emitted {
			self.text_emitted = false;
			return Vec::new();
		}
		let text = assistant_message_text(message);
		self.text_emitted = false;
		if text.is_empty() {
			return Vec::new();
		}
		self.text_emitted = true;
		vec![SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::from(text)))]
	}
}

/// 抽 assistant 消息里所有 text block 拼成一段（TS
/// `extractAssistantMessageText`；非 assistant 返空串）。thinking / toolCall /
/// image 块不计入——兜底只补可见正文。
fn assistant_message_text(message: &Message) -> String {
	let Message::Assistant(assistant) = message else {
		return String::new();
	};
	let mut text = String::new();
	for block in &assistant.content {
		if let pi_ai::message::AssistantContent::Text(content) = block {
			text.push_str(&content.text);
		}
	}
	text
}

/// `message_update` 内嵌 provider 事件 → text/thought chunk（其余变体不产）。
fn map_message_update(event: &AssistantMessageEvent) -> Vec<SessionUpdate> {
	match event {
		AssistantMessageEvent::TextDelta { delta, .. } if !delta.is_empty() => {
			vec![SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::from(
				delta.clone(),
			)))]
		},
		AssistantMessageEvent::ThinkingDelta { delta, .. } if !delta.is_empty() => {
			vec![SessionUpdate::AgentThoughtChunk(ContentChunk::new(ContentBlock::from(
				delta.clone(),
			)))]
		},
		_ => vec![],
	}
}

/// prompt turn 结束时把 loop 终态归结为 ACP `StopReason`。
///
/// TS `acp-agent.ts#resolveStopReason`（:1423）逐条移植：
/// - `cancelRequested` → `Cancelled`（`session/cancel` 语义，最高优先）；
/// - 否则看**最后一条 assistant** 的 pi-ai `stop_reason`：
///   `aborted`→`Cancelled` / `length`→`MaxTokens` / `error`→（errorMessage 命中
///   content-filter·refusal 正则）`Refusal` 否则 `EndTurn` /
///   其余（`stop`/`tool_use`/无）→`EndTurn`。
///
/// **结论（error 落点）**：ACP 契约无 `error` stop reason；error turn 落
/// `Refusal`（内容过滤/拒绝）或 `EndTurn`，`aborted` 落 `Cancelled`。
#[must_use]
pub fn resolve_stop_reason(messages: &[Message], cancel_requested: bool) -> AcpStopReason {
	if cancel_requested {
		return AcpStopReason::Cancelled;
	}
	let last_assistant = messages.iter().rev().find_map(|message| match message {
		Message::Assistant(assistant) => Some(assistant.as_ref()),
		_ => None,
	});
	let Some(assistant) = last_assistant else {
		return AcpStopReason::EndTurn;
	};
	match &assistant.stop_reason {
		StopReason::Aborted => AcpStopReason::Cancelled,
		StopReason::Length => AcpStopReason::MaxTokens,
		StopReason::Error => {
			let error_message = assistant.error_message.as_deref().unwrap_or_default();
			if is_refusal(error_message) {
				AcpStopReason::Refusal
			} else {
				AcpStopReason::EndTurn
			}
		},
		_ => AcpStopReason::EndTurn,
	}
}

/// error turn 的 errorMessage 是否为内容过滤 / 拒绝（TS `#resolveStopReason`
/// 正则 `/content[_ ]?filter|refus(al|ed)/i`）。
fn is_refusal(text: &str) -> bool {
	use std::sync::OnceLock;
	static RE: OnceLock<Regex> = OnceLock::new();
	RE.get_or_init(|| Regex::new(r"(?i)content[_ ]?filter|refus(al|ed)").expect("valid regex"))
		.is_match(text)
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use pi_ai::message::{AssistantContent, AssistantMessage, Usage};

	use super::*;

	/// 最小 assistant message 夹具（`map_event` 只读
	/// `stop_reason`/`error_message`）。
	fn assistant(stop_reason: StopReason, error_message: Option<&str>) -> AssistantMessage {
		AssistantMessage {
			content: Vec::new(),
			api: String::new(),
			provider: String::new(),
			model: "test".to_owned(),
			context_snapshot: None,
			retry_recovery: None,
			response_id: None,
			upstream_provider: None,
			usage: Usage::default(),
			stop_reason,
			stop_details: None,
			error_message: error_message.map(ToOwned::to_owned),
			tool_call_abort_messages: None,
			error_status: None,
			error_id: None,
			disabled_features: None,
			provider_payload: None,
			timestamp: 0,
			duration: None,
			ttft: None,
		}
	}

	fn text_delta(delta: &str) -> AgentEvent {
		let partial = Arc::new(assistant(StopReason::Stop, None));
		AgentEvent::MessageUpdate {
			message:                 Message::Assistant(Box::new(assistant(StopReason::Stop, None))),
			assistant_message_event: AssistantMessageEvent::TextDelta {
				content_index: 0,
				delta: delta.to_owned(),
				partial,
			},
		}
	}

	fn thinking_delta(delta: &str) -> AgentEvent {
		let partial = Arc::new(assistant(StopReason::Stop, None));
		AgentEvent::MessageUpdate {
			message:                 Message::Assistant(Box::new(assistant(StopReason::Stop, None))),
			assistant_message_event: AssistantMessageEvent::ThinkingDelta {
				content_index: 0,
				delta: delta.to_owned(),
				partial,
			},
		}
	}

	/// update 变体判别串（断言序列用）。
	fn variant(update: &SessionUpdate) -> &'static str {
		match update {
			SessionUpdate::AgentMessageChunk(_) => "agent_message_chunk",
			SessionUpdate::AgentThoughtChunk(_) => "agent_thought_chunk",
			SessionUpdate::ToolCall(_) => "tool_call",
			SessionUpdate::ToolCallUpdate(_) => "tool_call_update",
			_ => "other",
		}
	}

	#[test]
	fn text_delta_maps_to_agent_message_chunk() {
		let updates = map_event(&text_delta("hello"));
		assert_eq!(updates.len(), 1);
		match &updates[0] {
			SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
				ContentBlock::Text(text) => assert_eq!(text.text, "hello"),
				other => panic!("期望 text content, 实得 {other:?}"),
			},
			other => panic!("期望 agent_message_chunk, 实得 {other:?}"),
		}
	}

	#[test]
	fn thinking_delta_maps_to_agent_thought_chunk() {
		let updates = map_event(&thinking_delta("pondering"));
		assert_eq!(updates.len(), 1);
		assert_eq!(variant(&updates[0]), "agent_thought_chunk");
	}

	#[test]
	fn empty_delta_produces_no_update() {
		assert!(map_event(&text_delta("")).is_empty());
		assert!(map_event(&thinking_delta("")).is_empty());
	}

	#[test]
	fn tool_execution_start_maps_to_tool_call_with_kind_and_title() {
		let event = AgentEvent::ToolExecutionStart {
			tool_call_id: "call-1".to_owned(),
			tool_name:    "write".to_owned(),
			args:         serde_json::json!({ "path": "hello.txt", "content": "x" }),
			intent:       None,
		};
		let updates = map_event(&event);
		assert_eq!(updates.len(), 1);
		match &updates[0] {
			SessionUpdate::ToolCall(call) => {
				assert_eq!(call.tool_call_id.0.as_ref(), "call-1");
				assert_eq!(call.title, "write: hello.txt");
				assert_eq!(call.kind, ToolKind::Edit);
				assert_eq!(call.status, ToolCallStatus::Pending);
				assert_eq!(
					call.raw_input.as_ref().and_then(|v| v.get("path")),
					Some(&serde_json::json!("hello.txt"))
				);
			},
			other => panic!("期望 tool_call, 实得 {other:?}"),
		}
	}

	#[test]
	fn tool_execution_start_intent_overrides_title() {
		let event = AgentEvent::ToolExecutionStart {
			tool_call_id: "c".to_owned(),
			tool_name:    "grep".to_owned(),
			args:         serde_json::json!({ "pattern": "foo" }),
			intent:       Some("  search for foo  ".to_owned()),
		};
		match &map_event(&event)[0] {
			SessionUpdate::ToolCall(call) => {
				assert_eq!(call.title, "search for foo");
				assert_eq!(call.kind, ToolKind::Search);
			},
			other => panic!("期望 tool_call, 实得 {other:?}"),
		}
	}

	#[test]
	fn tool_execution_end_success_maps_to_completed_update() {
		let event = AgentEvent::ToolExecutionEnd {
			tool_call_id: "call-1".to_owned(),
			tool_name:    "read".to_owned(),
			result:       serde_json::json!({ "content": "file body" }),
			is_error:     Some(false),
		};
		match &map_event(&event)[0] {
			SessionUpdate::ToolCallUpdate(update) => {
				assert_eq!(update.tool_call_id.0.as_ref(), "call-1");
				assert_eq!(update.fields.status, Some(ToolCallStatus::Completed));
				assert!(update.fields.content.as_ref().is_some_and(|c| c.len() == 1));
			},
			other => panic!("期望 tool_call_update, 实得 {other:?}"),
		}
	}

	#[test]
	fn tool_execution_update_maps_to_in_progress_update() {
		// F6：tool_execution_update → tool_call_update(in_progress) + rawOutput。
		let event = AgentEvent::ToolExecutionUpdate {
			tool_call_id:   "call-9".to_owned(),
			tool_name:      "bash".to_owned(),
			args:           serde_json::json!({ "command": "sleep 1" }),
			partial_result: serde_json::json!({ "output": "partial line\n" }),
		};
		match &map_event(&event)[0] {
			SessionUpdate::ToolCallUpdate(update) => {
				assert_eq!(update.tool_call_id.0.as_ref(), "call-9");
				assert_eq!(update.fields.status, Some(ToolCallStatus::InProgress));
				assert!(update.fields.content.as_ref().is_some_and(|c| c.len() == 1));
			},
			other => panic!("期望 tool_call_update(in_progress), 实得 {other:?}"),
		}
	}

	#[test]
	fn tool_execution_end_error_maps_to_failed_update() {
		let event = AgentEvent::ToolExecutionEnd {
			tool_call_id: "call-2".to_owned(),
			tool_name:    "bash".to_owned(),
			result:       serde_json::json!("boom"),
			is_error:     Some(true),
		};
		match &map_event(&event)[0] {
			SessionUpdate::ToolCallUpdate(update) => {
				assert_eq!(update.fields.status, Some(ToolCallStatus::Failed));
			},
			other => panic!("期望 tool_call_update, 实得 {other:?}"),
		}
	}

	#[test]
	fn lifecycle_events_are_not_mapped() {
		assert!(map_event(&AgentEvent::AgentStart).is_empty());
		assert!(map_event(&AgentEvent::TurnStart).is_empty());
		assert!(
			map_event(&AgentEvent::MessageStart {
				message: Message::Assistant(Box::new(assistant(StopReason::Stop, None))),
			})
			.is_empty()
		);
		assert!(map_event(&AgentEvent::AgentEnd { messages: Vec::new() }).is_empty());
	}

	#[test]
	fn map_tool_kind_table() {
		assert_eq!(map_tool_kind("read"), ToolKind::Read);
		assert_eq!(map_tool_kind("edit"), ToolKind::Edit);
		assert_eq!(map_tool_kind("glob"), ToolKind::Search);
		assert_eq!(map_tool_kind("web_search"), ToolKind::Fetch);
		assert_eq!(map_tool_kind("todo"), ToolKind::Think);
		assert_eq!(map_tool_kind("unknown_tool"), ToolKind::Other);
	}

	#[test]
	fn stop_reason_normal_end_turn() {
		let messages = vec![Message::Assistant(Box::new(assistant(StopReason::Stop, None)))];
		assert_eq!(resolve_stop_reason(&messages, false), AcpStopReason::EndTurn);
	}

	#[test]
	fn stop_reason_cancel_requested_wins() {
		let messages = vec![Message::Assistant(Box::new(assistant(StopReason::Stop, None)))];
		assert_eq!(resolve_stop_reason(&messages, true), AcpStopReason::Cancelled);
	}

	#[test]
	fn stop_reason_aborted_maps_to_cancelled() {
		let messages = vec![Message::Assistant(Box::new(assistant(StopReason::Aborted, None)))];
		assert_eq!(resolve_stop_reason(&messages, false), AcpStopReason::Cancelled);
	}

	#[test]
	fn stop_reason_length_maps_to_max_tokens() {
		let messages = vec![Message::Assistant(Box::new(assistant(StopReason::Length, None)))];
		assert_eq!(resolve_stop_reason(&messages, false), AcpStopReason::MaxTokens);
	}

	#[test]
	fn stop_reason_error_refusal_vs_end_turn() {
		let refusal = vec![Message::Assistant(Box::new(assistant(
			StopReason::Error,
			Some("request triggered content_filter"),
		)))];
		assert_eq!(resolve_stop_reason(&refusal, false), AcpStopReason::Refusal);

		let plain = vec![Message::Assistant(Box::new(assistant(
			StopReason::Error,
			Some("stream read error"),
		)))];
		assert_eq!(resolve_stop_reason(&plain, false), AcpStopReason::EndTurn);
	}

	#[test]
	fn stop_reason_empty_defaults_end_turn() {
		assert_eq!(resolve_stop_reason(&[], false), AcpStopReason::EndTurn);
	}

	/// 全序列：fake `AgentEvent` 序 → update 变体序断言（映射层端到端）。
	#[test]
	fn full_event_sequence_maps_to_expected_update_stream() {
		let events = [
			AgentEvent::AgentStart,
			AgentEvent::TurnStart,
			text_delta("hel"),
			text_delta("lo"),
			AgentEvent::ToolExecutionStart {
				tool_call_id: "c1".to_owned(),
				tool_name:    "write".to_owned(),
				args:         serde_json::json!({ "path": "hello.txt" }),
				intent:       None,
			},
			AgentEvent::ToolExecutionEnd {
				tool_call_id: "c1".to_owned(),
				tool_name:    "write".to_owned(),
				result:       serde_json::json!({ "content": "ok" }),
				is_error:     Some(false),
			},
			text_delta("done"),
			AgentEvent::AgentEnd { messages: Vec::new() },
		];
		let variants: Vec<&str> = events
			.iter()
			.flat_map(map_event)
			.map(|update| variant(&update))
			.collect();
		assert_eq!(variants, [
			"agent_message_chunk",
			"agent_message_chunk",
			"tool_call",
			"tool_call_update",
			"agent_message_chunk",
		]);
	}

	// ── F8: message_end 文本兜底（MapProgress）────────────────────────────

	/// 带文本内容块的 assistant message（终值消息，兜底抽文本用）。
	fn assistant_with_text(text: &str) -> Message {
		let mut msg = assistant(StopReason::Stop, None);
		msg.content = vec![AssistantContent::Text(pi_ai::message::TextContent {
			text:           text.to_owned(),
			text_signature: None,
		})];
		Message::Assistant(Box::new(msg))
	}

	fn message_end(text: &str) -> AgentEvent {
		AgentEvent::MessageEnd { message: assistant_with_text(text) }
	}

	fn chunk_text(update: &SessionUpdate) -> &str {
		match update {
			SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
				ContentBlock::Text(text) => &text.text,
				other => panic!("期望 text content, 实得 {other:?}"),
			},
			other => panic!("期望 agent_message_chunk, 实得 {other:?}"),
		}
	}

	#[test]
	fn message_end_fallback_emits_when_no_delta() {
		// 非流式 provider：一路无 text_delta → message_end 从终值补发全文。
		let mut p = MapProgress::new();
		assert!(
			p.map(&AgentEvent::MessageStart { message: assistant_with_text("") })
				.is_empty()
		);
		let updates = p.map(&message_end("full answer body"));
		assert_eq!(updates.len(), 1, "无 delta 时 message_end 补发一条 chunk");
		assert_eq!(chunk_text(&updates[0]), "full answer body");
	}

	#[test]
	fn message_end_fallback_suppressed_after_delta() {
		// 流式 provider（DeepSeek）：发过 delta → message_end 不再补（去重）。
		let mut p = MapProgress::new();
		let _ = p.map(&AgentEvent::MessageStart { message: assistant_with_text("hello") });
		let d = p.map(&text_delta("hello"));
		assert_eq!(d.len(), 1, "delta 正常透传");
		let updates = p.map(&message_end("hello"));
		assert!(updates.is_empty(), "已发过 delta，message_end 兜底被抑制");
	}

	#[test]
	fn message_end_fallback_empty_text_no_emit() {
		// 终值消息无可见正文（如纯 tool_use 轮）→ 不补发空 chunk。
		let mut p = MapProgress::new();
		let updates = p.map(&message_end(""));
		assert!(updates.is_empty(), "空文本不补发");
	}

	#[test]
	fn message_start_resets_dedup_across_messages() {
		// 多条 assistant message：第一条发过 delta，第二条非流式 → 各自独立判定。
		let mut p = MapProgress::new();
		let _ = p.map(&AgentEvent::MessageStart { message: assistant_with_text("a") });
		let _ = p.map(&text_delta("a"));
		assert!(p.map(&message_end("a")).is_empty(), "msg1 发过 delta 不补");
		// msg2：无 delta，message_start 已重置去重位 → 兜底应触发。
		let _ = p.map(&AgentEvent::MessageStart { message: assistant_with_text("b") });
		let updates = p.map(&message_end("second body"));
		assert_eq!(updates.len(), 1, "msg2 无 delta，重置后兜底触发");
		assert_eq!(chunk_text(&updates[0]), "second body");
	}
}
