//! Serial single-tool execution + tool-result coercion + synthetic
//! placeholders.
//!
//! Port of the tool-execution core of `agent-loop.ts`, folded to the WP-1.4a
//! face:
//! - [`execute_tool_calls`] runs the assistant turn's tool calls **strictly
//!   serially** in content order. The TS shared/exclusive concurrency scheduler
//!   (`agent-loop.ts:2210-2238`) is deferred to WP-1.4b — every tool runs
//!   sequentially here, which is a safe subset of "shared" (side-effecting
//!   tools never overlap).
//! - [`coerce_tool_result`] mirrors `coerceToolResult` (:267-330): since a Rust
//!   [`ToolResult`] is already a typed `Vec<UserContentBlock>`, the only
//!   surviving regularization is the empty-error guard (an `is_error` result
//!   with no substantive content is backfilled with `"Tool failed with no
//!   output."`, which Anthropic requires).
//! - [`create_aborted_tool_result`] / [`create_synthetic_tool_result_message`]
//!   mirror `createAbortedToolResult` (:2371) /
//!   `createSyntheticToolResultMessage` (:2342): they keep the
//!   `tool_use`/`tool_result` pairing the Anthropic API mandates for error /
//!   aborted / length / skipped turns.
//!
//! Deferred from the TS execute path (see `lib.rs` defer list): intent tracing,
//! `beforeToolCall`/`afterToolCall` hooks, per-tool interruptible/IRC signals,
//! telemetry spans, and the streaming `tool_execution_update` emission.

use std::{
	collections::BTreeSet,
	time::{SystemTime, UNIX_EPOCH},
};

use pi_ai::{
	TextContent, UserContentBlock,
	message::{AssistantContent, AssistantMessage, ToolCall, ToolResultMessage},
};
use pi_shell::cancel::CancelToken;
use pi_tools::{DynTool, ToolResult};
use serde_json::{Map, Value, json};

use crate::event::{AgentEvent, AgentEventSink};

/// `EMPTY_ERROR_TOOL_RESULT_TEXT` (agent-loop.ts:257).
const EMPTY_ERROR_TOOL_RESULT_TEXT: &str = "Tool failed with no output.";

/// The assistant-side termination state that produced a synthetic tool result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticReason {
	Aborted,
	Error,
	Skipped,
	Length,
}

impl SyntheticReason {
	/// `SyntheticToolResultDetails.source` string (agent-loop.ts:2322).
	const fn source(self) -> &'static str {
		match self {
			Self::Aborted => "assistant_stop_aborted",
			Self::Error => "assistant_stop_error",
			Self::Length => "assistant_stop_length",
			Self::Skipped => "assistant_stop_skipped",
		}
	}

	/// Human-facing message body (agent-loop.ts:2347-2354).
	const fn message(self) -> &'static str {
		match self {
			Self::Aborted => "Tool execution was aborted",
			Self::Length => {
				"Tool call was not executed because the assistant hit its output token limit \
				 (stop_reason: length) before the arguments could complete; the recorded arguments are \
				 truncated and unsafe to run. Do NOT retry by re-emitting the same large payload — \
				 split the work into several smaller tool calls"
			},
			Self::Skipped => "Tool call was not executed because the assistant ended its turn",
			Self::Error => {
				"Tool call was not executed because the provider stream ended with an error before the \
				 tool could run"
			},
		}
	}
}

/// Unix time in milliseconds.
pub(crate) fn now_millis() -> i64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map_or(0, |elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
}

/// The tool-call content blocks of an assistant message, in wire order.
pub(crate) fn tool_calls_of(message: &AssistantMessage) -> Vec<ToolCall> {
	message
		.content
		.iter()
		.filter_map(|block| match block {
			AssistantContent::ToolCall(call) => Some(call.clone()),
			_ => None,
		})
		.collect()
}

/// `hasSubstantiveToolResultContent` (agent-loop.ts:259).
fn has_substantive_content(content: &[UserContentBlock]) -> bool {
	content.iter().any(|block| match block {
		UserContentBlock::Image(_) => true,
		UserContentBlock::Text(text) => !text.text.trim().is_empty(),
	})
}

/// `coerceToolResult` (agent-loop.ts:267-330), typed subset. A Rust
/// [`ToolResult`] already carries validated `UserContentBlock`s, so the only
/// surviving transform is the empty-error backfill.
#[must_use]
pub fn coerce_tool_result(mut result: ToolResult) -> ToolResult {
	if result.is_error && !has_substantive_content(&result.content) {
		result.content = vec![UserContentBlock::Text(TextContent {
			text:           EMPTY_ERROR_TOOL_RESULT_TEXT.to_owned(),
			text_signature: None,
		})];
	}
	// Errors are never useless (agent-loop.ts:326).
	if result.is_error {
		result.useless = false;
	}
	result
}

/// Serialize a [`ToolResult`] into the TS `AgentToolResult` JSON shape
/// (`{ content, details, isError?, useless? }`) for the `result` /
/// `partialResult` fields of tool-execution events.
fn tool_result_to_value(result: &ToolResult) -> Value {
	let mut map = Map::new();
	map.insert(
		"content".to_owned(),
		serde_json::to_value(&result.content).unwrap_or(Value::Array(vec![])),
	);
	map.insert("details".to_owned(), result.details.clone().unwrap_or_else(|| json!({})));
	if result.is_error {
		map.insert("isError".to_owned(), Value::Bool(true));
	} else if result.useless {
		map.insert("useless".to_owned(), Value::Bool(true));
	}
	Value::Object(map)
}

/// Build the `toolResult` message from a coerced result.
fn tool_result_message(call: &ToolCall, result: &ToolResult, is_error: bool) -> ToolResultMessage {
	ToolResultMessage {
		tool_call_id: call.id.clone(),
		tool_name: call.name.clone(),
		content: result.content.clone(),
		details: result.details.clone(),
		is_error,
		attribution: None,
		pruned_at: None,
		useless: (result.useless && !is_error).then_some(true),
		timestamp: now_millis(),
	}
}

/// Emit the `tool_execution_end` + `message_start`/`message_end` triple for a
/// finished tool (start is pushed by the caller before this point).
fn emit_tool_end(
	sink: &AgentEventSink,
	call: &ToolCall,
	result: &ToolResult,
	is_error: bool,
) -> ToolResultMessage {
	sink.push(AgentEvent::ToolExecutionEnd {
		tool_call_id: call.id.clone(),
		tool_name:    call.name.clone(),
		result:       tool_result_to_value(result),
		is_error:     Some(is_error),
	});
	let message = tool_result_message(call, result, is_error);
	sink.push(AgentEvent::MessageStart {
		message: pi_ai::message::Message::ToolResult(message.clone()),
	});
	sink.push(AgentEvent::MessageEnd {
		message: pi_ai::message::Message::ToolResult(message.clone()),
	});
	message
}

/// Push the `tool_execution_start` event for a call.
fn emit_tool_start(sink: &AgentEventSink, call: &ToolCall, args: &Value) {
	sink.push(AgentEvent::ToolExecutionStart {
		tool_call_id: call.id.clone(),
		tool_name:    call.name.clone(),
		args:         args.clone(),
		intent:       call.intent.clone(),
	});
}

/// Minimal argument validation: required object fields must be present. Mirrors
/// the failure-surfacing semantics of `validateToolArguments` (a missing
/// required field yields an `is_error` result rather than crashing the loop),
/// without the full JSON-Schema validator (deferred).
fn validate_args(schema: &Value, args: &Value) -> Result<(), String> {
	let Some(required) = schema.get("required").and_then(Value::as_array) else {
		return Ok(());
	};
	let obj = args.as_object();
	let missing: Vec<String> = required
		.iter()
		.filter_map(Value::as_str)
		.filter(|key| !obj.is_some_and(|o| o.contains_key(*key)))
		.map(ToOwned::to_owned)
		.collect();
	if missing.is_empty() {
		Ok(())
	} else {
		Err(format!("Invalid arguments: missing required field(s): {}", missing.join(", ")))
	}
}

/// Build a synthetic `toolResult` message for a call the assistant emitted but
/// that was never invoked locally. `createSyntheticToolResultMessage`
/// (agent-loop.ts:2342).
#[must_use]
pub fn create_synthetic_tool_result_message(
	call: &ToolCall,
	reason: SyntheticReason,
	error_message: Option<&str>,
) -> ToolResultMessage {
	let body = reason.message();
	let text = match error_message {
		Some(err) => format!("{body}: {err}"),
		None => format!("{body}."),
	};
	let mut details = Map::new();
	details.insert("__synthetic".to_owned(), Value::Bool(true));
	details.insert("source".to_owned(), Value::String(reason.source().to_owned()));
	details.insert("executed".to_owned(), Value::Bool(false));
	if reason == SyntheticReason::Error
		&& let Some(err) = error_message
	{
		details.insert("upstreamError".to_owned(), Value::String(err.to_owned()));
	}
	ToolResultMessage {
		tool_call_id: call.id.clone(),
		tool_name:    call.name.clone(),
		content:      vec![UserContentBlock::Text(TextContent { text, text_signature: None })],
		details:      Some(Value::Object(details)),
		is_error:     true,
		attribution:  None,
		pruned_at:    None,
		useless:      None,
		timestamp:    now_millis(),
	}
}

/// Create AND emit a synthetic placeholder result (keeps the `tool_use` /
/// `tool_result` pairing). `createAbortedToolResult` (agent-loop.ts:2371).
pub fn create_aborted_tool_result(
	sink: &AgentEventSink,
	call: &ToolCall,
	reason: SyntheticReason,
	error_message: Option<&str>,
) -> ToolResultMessage {
	let message = create_synthetic_tool_result_message(call, reason, error_message);
	let result = ToolResult {
		content:  message.content.clone(),
		details:  message.details.clone(),
		is_error: true,
		useless:  false,
	};
	emit_tool_start(sink, call, &call.arguments);
	sink.push(AgentEvent::ToolExecutionEnd {
		tool_call_id: call.id.clone(),
		tool_name:    call.name.clone(),
		result:       tool_result_to_value(&result),
		is_error:     Some(true),
	});
	sink.push(AgentEvent::MessageStart {
		message: pi_ai::message::Message::ToolResult(message.clone()),
	});
	sink.push(AgentEvent::MessageEnd {
		message: pi_ai::message::Message::ToolResult(message.clone()),
	});
	message
}

/// Run every tool call of `message` serially, in content order.
///
/// For each call: resolve the tool by name → validate args → push
/// `tool_execution_start` → `tool.execute` → coerce → emit
/// `tool_execution_end` + the `toolResult` message events. An unknown tool
/// name, a validation miss, or a thrown [`pi_tools::ToolError`] each yield an
/// `is_error` result and continue the batch (a failing tool never breaks the
/// chain). A tool call reached after the token is aborted gets a skipped
/// placeholder instead of executing.
pub async fn execute_tool_calls(
	tools: &[Box<dyn DynTool>],
	message: &AssistantMessage,
	ct: &CancelToken,
	sink: &AgentEventSink,
) -> Vec<ToolResultMessage> {
	let calls = tool_calls_of(message);
	let mut results = Vec::with_capacity(calls.len());
	for call in &calls {
		// Already aborted → skipped placeholder, do not execute (:2411 semantics).
		if ct.aborted() {
			results.push(create_aborted_tool_result(
				sink,
				call,
				SyntheticReason::Skipped,
				Some("run was aborted before the tool could start"),
			));
			continue;
		}

		let Some(tool) = tools.iter().find(|t| t.name() == call.name) else {
			// Unknown tool → isError result (:1989 `Tool <name> not found`).
			emit_tool_start(sink, call, &call.arguments);
			let result = coerce_tool_result(ToolResult {
				content:  vec![UserContentBlock::Text(TextContent {
					text:           format!("Tool {} not found", call.name),
					text_signature: None,
				})],
				details:  None,
				is_error: true,
				useless:  false,
			});
			results.push(emit_tool_end(sink, call, &result, true));
			continue;
		};

		if let Err(err) = validate_args(&tool.input_schema(), &call.arguments) {
			emit_tool_start(sink, call, &call.arguments);
			let result = coerce_tool_result(ToolResult {
				content:  vec![UserContentBlock::Text(TextContent {
					text:           err,
					text_signature: None,
				})],
				details:  None,
				is_error: true,
				useless:  false,
			});
			results.push(emit_tool_end(sink, call, &result, true));
			continue;
		}

		emit_tool_start(sink, call, &call.arguments);
		let (result, is_error) = match tool.execute(&call.id, call.arguments.clone(), ct).await {
			Ok(raw) => {
				let coerced = coerce_tool_result(raw);
				let is_error = coerced.is_error;
				(coerced, is_error)
			},
			Err(err) => {
				// A thrown ToolError becomes an isError result (:2120-2127).
				let result = coerce_tool_result(ToolResult {
					content:  vec![UserContentBlock::Text(TextContent {
						text:           err.to_string(),
						text_signature: None,
					})],
					details:  None,
					is_error: true,
					useless:  false,
				});
				(result, true)
			},
		};
		results.push(emit_tool_end(sink, call, &result, is_error));
	}
	results
}

/// The distinct tool names available (used by transient-error recovery).
pub(crate) fn available_tool_names(tools: &[Box<dyn DynTool>]) -> BTreeSet<String> {
	tools.iter().map(|t| t.name().to_owned()).collect()
}
