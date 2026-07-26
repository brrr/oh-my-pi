//! Shared/exclusive tool scheduling + abort propagation + tool-result coercion
//! + synthetic placeholders.
//!
//! Port of the tool-execution core of `agent-loop.ts` (`executeToolCalls`
//! :1786-2434), carrying the WP-1.4b concurrency face that WP-1.4a folded to a
//! serial subset:
//! - [`execute_tool_calls`] schedules the assistant turn's tool calls by their
//!   [`Concurrency`] class, mirroring the TS promise-chain scheduler
//!   (:2210-2238). [`Concurrency::Shared`] calls run concurrently with each
//!   other; a [`Concurrency::Exclusive`] call is a barrier — it waits for every
//!   prior in-flight call, runs alone, then releases the calls after it. The TS
//!   chaining (`start = exclusive ? Promise.all([lastExclusive,
//!   ...sharedTasks]) : lastExclusive`) is equivalent to running the batch as
//!   ordered segments: a maximal run of consecutive shared calls executes as
//!   one concurrent group, and each exclusive call is a singleton barrier
//!   between groups. The concurrency is **cooperative** (a [`FuturesUnordered`]
//!   driven on the loop task, no `tokio::spawn`), which matches the TS
//!   single-threaded Promise model exactly — overlap happens at `await` points,
//!   and a tool that never yields runs to completion in content order.
//! - Results are collected in **completion order** (TS `emittedToolResults`
//!   push order, :1948): the returned `Vec` — which the loop backfills into
//!   `ctx.messages` verbatim (agent-loop.ts:1074-1085) — and the
//!   `tool_execution_end` events both follow real completion order, not
//!   initiation order. The tail sweep (:2262-2272) appends a skipped result for
//!   every un-run record, in record order, after the completed ones.
//! - Panic isolation mirrors `Promise.allSettled` (:2254): a tool that panics
//!   is caught (`catch_unwind`) and turned into an `is_error` result rather
//!   than unwinding the whole batch.
//! - Abort propagation (:2023-2185): a call reached under an already-aborted
//!   token, or cut off mid-flight before it completed, yields a clean aborted
//!   result; a call that **completed** keeps its real result even if the token
//!   aborts at cleanup (`completedToolExecution`, :2170-2185).
//! - Mid-batch steering (`checkSteering` :1868): before an exclusive barrier
//!   runs, [`SteeringPeek`] is polled non-consumingly; a queued steer stops the
//!   batch — in-flight shared calls finish, but the exclusive and every record
//!   after it are paired with a skipped result ([`create_skipped_tool_result`],
//!   `createSkippedToolResult` :2411) and not executed. Shared-only batches
//!   have no barrier, so they are not interrupted mid-flight (read-only tools
//!   are safe to complete); the steer is drained at the batch boundary by the
//!   loop. The TS dual-signal `interruptible` split (:1858, a peer-IRC signal
//!   that only aborts interruptible waits) is not modeled — one [`CancelToken`]
//!   face.
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
//! `beforeToolCall`/`afterToolCall` hooks, the dual-signal `interruptible`/IRC
//! split, the process-wide pause gate, `SoftToolRequirement`, telemetry spans,
//! and the streaming `tool_execution_update` emission. The dynamic
//! `concurrency(args)` resolver (TS `bash` pty→exclusive) is deferred at the
//! [`Concurrency`] type (a tool resolves to one static class here).

use std::{
	collections::BTreeSet,
	panic::AssertUnwindSafe,
	time::{SystemTime, UNIX_EPOCH},
};

use futures::{FutureExt, StreamExt, stream::FuturesUnordered};
use pi_ai::{
	TextContent, UserContentBlock,
	message::{AssistantContent, AssistantMessage, ToolCall, ToolResultMessage},
};
use pi_shell::cancel::CancelToken;
use pi_tools::{Concurrency, DynTool, ToolResult};
use serde_json::{Map, Value, json};

use crate::event::{AgentEvent, AgentEventSink};

/// Non-consuming steering peek (`config.hasSteeringMessages`).
///
/// `checkSteering` (agent-loop.ts:1868): returns `true` when a steering message
/// is queued; used mid-batch to interrupt before an exclusive barrier.
pub type SteeringPeek<'a> = Option<&'a (dyn Fn() -> bool + Send + Sync)>;

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

/// Build a plain (non-synthetic) `is_error` [`ToolResult`] from a single text
/// block (used for unknown-tool / validation / abort / panic outcomes).
fn error_result(text: impl Into<String>) -> ToolResult {
	ToolResult {
		content:  vec![UserContentBlock::Text(TextContent {
			text:           text.into(),
			text_signature: None,
		})],
		details:  None,
		is_error: true,
		useless:  false,
	}
}

/// `createToolSignalAbortedResult` (agent-loop.ts:2404): the plain `is_error`
/// result for a tool cut off by an aborted run token (no `__synthetic` detail —
/// distinct from [`create_aborted_tool_result`]).
fn tool_signal_aborted_result() -> ToolResult {
	error_result("Tool was not executed because the run was aborted.")
}

/// `createSkippedToolResult` (agent-loop.ts:2411): the plain `is_error` result
/// for a tool skipped because a steering message is queued. Emits the full
/// `tool_execution_start`/`_end` + `message_start`/`_end` pairing and returns
/// the `toolResult` message.
fn create_skipped_tool_result(sink: &AgentEventSink, call: &ToolCall) -> ToolResultMessage {
	let result = coerce_tool_result(error_result(
		"Skipped due to queued user message. Do not count this skipped result as completed work or \
		 verification. After the queued message is handled on the next step, retry the skipped tool \
		 if it is still needed.",
	));
	emit_tool_start(sink, call, &call.arguments);
	emit_tool_end(sink, call, &result, true)
}

/// Extract a human-readable message from a caught panic payload.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
	payload.downcast_ref::<&'static str>().map_or_else(
		|| {
			payload
				.downcast_ref::<String>()
				.cloned()
				.unwrap_or_else(|| "tool panicked".to_owned())
		},
		|s| (*s).to_owned(),
	)
}

/// Run one resolved tool call to completion and emit its lifecycle events.
///
/// Mirrors the per-call `runTool` body (agent-loop.ts:1953-2189): resolve /
/// validate → abort pre-check → `tool_execution_start` → execute (with panic
/// isolation) → coerce → post-abort adjustment → `tool_execution_end` + the
/// `toolResult` message. Unknown tool, validation miss, thrown
/// [`pi_tools::ToolError`], and panic each become an `is_error` result (the
/// batch never breaks). A call reached under an aborted token, or cut off
/// mid-flight before it completed, yields a clean aborted result; a call that
/// completed keeps its real result even if the token aborted at cleanup.
async fn run_tool(
	tool: Option<&dyn DynTool>,
	call: &ToolCall,
	ct: &CancelToken,
	sink: &AgentEventSink,
) -> ToolResultMessage {
	// Unknown tool → isError result (agent-loop.ts:1986 `Tool <name> not found`).
	let Some(tool) = tool else {
		emit_tool_start(sink, call, &call.arguments);
		let result = coerce_tool_result(error_result(format!("Tool {} not found", call.name)));
		return emit_tool_end(sink, call, &result, true);
	};

	// Missing required arg → isError result (validateToolArguments, :1989).
	if let Err(err) = validate_args(&tool.input_schema(), &call.arguments) {
		emit_tool_start(sink, call, &call.arguments);
		let result = coerce_tool_result(error_result(err));
		return emit_tool_end(sink, call, &result, true);
	}

	// Aborted before start → clean aborted result, do not execute (:2023-2032).
	if ct.aborted() {
		emit_tool_start(sink, call, &call.arguments);
		let result = coerce_tool_result(tool_signal_aborted_result());
		return emit_tool_end(sink, call, &result, true);
	}

	emit_tool_start(sink, call, &call.arguments);
	// `completedToolExecution` (:2103): true only when `execute` resolves
	// normally. A thrown `ToolError` or a panic leaves it false.
	let (result, is_error, completed) =
		match AssertUnwindSafe(tool.execute(&call.id, call.arguments.clone(), ct))
			.catch_unwind()
			.await
		{
			Ok(Ok(raw)) => {
				let coerced = coerce_tool_result(raw);
				let is_error = coerced.is_error;
				(coerced, is_error, true)
			},
			// A thrown ToolError becomes an isError result (:2120-2127).
			Ok(Err(err)) => (coerce_tool_result(error_result(err.to_string())), true, false),
			// Promise.allSettled parity (:2254): a panic is caught, not propagated.
			Err(payload) => (
				coerce_tool_result(error_result(format!("Tool panicked: {}", panic_text(&*payload)))),
				true,
				false,
			),
		};

	// completedToolExecution (:2170-2185): a tool that finished keeps its real
	// result even if the token aborted at cleanup; a tool cut off before
	// completing under an aborted token reports the clean aborted result.
	let (result, is_error) = if ct.aborted() && !completed {
		(coerce_tool_result(tool_signal_aborted_result()), true)
	} else {
		(result, is_error)
	};
	emit_tool_end(sink, call, &result, is_error)
}

/// A resolved tool call plus its scheduling class.
struct Record<'a> {
	call:        ToolCall,
	tool:        Option<&'a dyn DynTool>,
	concurrency: Concurrency,
}

/// Drive an in-flight group of shared tool calls concurrently to completion,
/// appending their results in **completion order** and marking each record run.
async fn flush_shared(
	records: &[Record<'_>],
	batch: &[usize],
	ct: &CancelToken,
	sink: &AgentEventSink,
	results: &mut Vec<ToolResultMessage>,
	ran: &mut [bool],
) {
	if batch.is_empty() {
		return;
	}
	let mut inflight = FuturesUnordered::new();
	for &i in batch {
		let record = &records[i];
		inflight.push(async move { (i, run_tool(record.tool, &record.call, ct, sink).await) });
	}
	while let Some((i, message)) = inflight.next().await {
		ran[i] = true;
		results.push(message);
	}
}

/// Schedule the assistant turn's tool calls by [`Concurrency`] class.
///
/// Shared calls run concurrently; each exclusive call is a barrier that waits
/// for every prior in-flight call, runs alone, then releases the rest — the
/// TS promise-chain scheduler (agent-loop.ts:2210-2238), executed as ordered
/// segments (shared groups run concurrently, exclusives are singleton
/// barriers). Results are collected in completion order (`emittedToolResults`,
/// :1948); the tail sweep (:2262-2272) appends a skipped result for any un-run
/// record. `has_steering`, when queued, interrupts the batch **before an
/// exclusive barrier** (`checkSteering`, :1868): in-flight shared calls finish,
/// and the exclusive plus every record after it get a skipped result.
pub async fn execute_tool_calls(
	tools: &[Box<dyn DynTool>],
	message: &AssistantMessage,
	ct: &CancelToken,
	sink: &AgentEventSink,
	has_steering: SteeringPeek<'_>,
) -> Vec<ToolResultMessage> {
	let records: Vec<Record> = tool_calls_of(message)
		.into_iter()
		.map(|call| {
			// Match on `name`; an unknown tool resolves to `None` (concurrency
			// `shared`, agent-loop.ts:2226 `?? "shared"`).
			let tool = tools
				.iter()
				.find(|t| t.name() == call.name)
				.map(AsRef::as_ref);
			let concurrency = tool.map_or(Concurrency::Shared, DynTool::concurrency);
			Record { call, tool, concurrency }
		})
		.collect();

	let mut results: Vec<ToolResultMessage> = Vec::with_capacity(records.len());
	let mut ran = vec![false; records.len()];
	let mut shared_batch: Vec<usize> = Vec::new();
	let mut interrupted = false;

	for idx in 0..records.len() {
		match records[idx].concurrency {
			Concurrency::Shared => shared_batch.push(idx),
			Concurrency::Exclusive => {
				// Barrier: let the in-flight shared group finish first.
				flush_shared(&records, &shared_batch, ct, sink, &mut results, &mut ran).await;
				shared_batch.clear();
				// checkSteering (:1868): a queued steer stops the batch here — the
				// exclusive and every record after it are tail-swept as skipped.
				if has_steering.is_some_and(|peek| peek()) {
					interrupted = true;
					break;
				}
				let message = run_tool(records[idx].tool, &records[idx].call, ct, sink).await;
				ran[idx] = true;
				results.push(message);
			},
		}
	}
	if !interrupted {
		flush_shared(&records, &shared_batch, ct, sink, &mut results, &mut ran).await;
	}

	// Tail sweep (:2262-2272): pair every un-run record with a skipped result so
	// the tool_use/tool_result pairing the provider API mandates is preserved.
	for (i, record) in records.iter().enumerate() {
		if !ran[i] {
			results.push(create_skipped_tool_result(sink, &record.call));
		}
	}
	results
}

/// The distinct tool names available (used by transient-error recovery).
pub(crate) fn available_tool_names(tools: &[Box<dyn DynTool>]) -> BTreeSet<String> {
	tools.iter().map(|t| t.name().to_owned()).collect()
}
