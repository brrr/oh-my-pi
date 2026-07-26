//! L1 state-machine tests for the WP-1.4a agent loop.
//!
//! Drives the loop with a scripted fake provider (`AgentConfig::stream_fn`) so
//! every stop-reason branch, the serial multi-tool path, error/abort handling,
//! and the event-sequence invariants are exercised without a network.

use std::{
	collections::{BTreeSet, VecDeque},
	sync::{Arc, Mutex},
};

use pi_agent::{AgentConfig, AgentContext, AgentEvent, agent_loop};
use pi_ai::{
	AssistantMessage, AssistantMessageEvent, TextContent,
	convert::emit_nonstream_events,
	event::{DoneReason, ErrorReason},
	message::{AssistantContent, Message, StopReason, ToolCall, Usage, UserContent, UserMessage},
	stream::AssistantMessageEventStream,
};
use pi_shell::cancel::{AbortReason, CancelToken};
use pi_tools::{Tool, ToolError, ToolResult};
use serde_json::{Value, json};

// ─── Builders ───────────────────────────────────────────────────────────────

fn assistant(content: Vec<AssistantContent>, stop: StopReason) -> AssistantMessage {
	AssistantMessage {
		content,
		api: "test".into(),
		provider: "test".into(),
		model: "fake".into(),
		context_snapshot: None,
		retry_recovery: None,
		response_id: None,
		upstream_provider: None,
		usage: Usage::default(),
		stop_reason: stop,
		stop_details: None,
		error_message: None,
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

fn text(s: &str) -> AssistantContent {
	AssistantContent::Text(TextContent { text: s.into(), text_signature: None })
}

fn tool_call(id: &str, name: &str, args: Value) -> AssistantContent {
	AssistantContent::ToolCall(ToolCall {
		id:                id.into(),
		name:              name.into(),
		arguments:         args,
		thought_signature: None,
		intent:            None,
		raw_block:         None,
		custom_wire_name:  None,
	})
}

fn user(s: &str) -> Message {
	Message::User(UserMessage {
		content:          UserContent::Text(s.into()),
		synthetic:        None,
		steering:         None,
		attribution:      None,
		provider_payload: None,
		timestamp:        0,
	})
}

/// A `stream_fn` that pops one scripted event vector per model call.
fn scripted(scripts: Vec<Vec<AssistantMessageEvent>>) -> pi_agent::StreamFn {
	let queue = Arc::new(Mutex::new(VecDeque::from(scripts)));
	Box::new(move |_ctx, _ct| {
		let events = queue.lock().unwrap().pop_front().unwrap_or_default();
		let (sink, stream) = AssistantMessageEventStream::channel();
		for event in events {
			assert!(sink.try_push(event));
		}
		stream
	})
}

/// A `stream_fn` that ignores the token and replays a synthesized event stream
/// for `message` (via the pi-ai non-stream synthesizer) on the first call, then
/// a terminating `stop` turn on subsequent calls.
fn scripted_msg(first: AssistantMessage) -> pi_agent::StreamFn {
	let arc = Arc::new(first);
	let stop = assistant(vec![text("done")], StopReason::Stop);
	scripted(vec![emit_nonstream_events(&arc), emit_nonstream_events(&std::sync::Arc::new(stop))])
}

async fn collect(mut stream: pi_agent::AgentEventStream) -> Vec<AgentEvent> {
	let mut events = Vec::new();
	while let Some(event) = stream.next().await {
		events.push(event);
	}
	events
}

fn ctx(tools: Vec<Box<dyn pi_tools::DynTool>>) -> AgentContext {
	AgentContext { system_prompt: vec!["sys".into()], messages: vec![], tools }
}

// ─── Recording tool
// ───────────────────────────────────────────────────────────

struct RecordTool {
	name: &'static str,
	log:  Arc<Mutex<Vec<String>>>,
	fail: bool,
}

impl Tool for RecordTool {
	fn name(&self) -> &'static str {
		self.name
	}

	fn description(&self) -> &'static str {
		"records invocation order"
	}

	fn input_schema(&self) -> Value {
		json!({ "type": "object", "properties": {} })
	}

	async fn execute(
		&self,
		id: &str,
		_args: Value,
		_ct: &CancelToken,
	) -> Result<ToolResult, ToolError> {
		self.log.lock().unwrap().push(format!("{}:{id}", self.name));
		if self.fail {
			Err(ToolError::new(format!("{} boom", self.name)))
		} else {
			Ok(ToolResult::text(format!("{} ran", self.name)))
		}
	}
}

// ─── Invariants
// ───────────────────────────────────────────────────────────────

/// Structural invariants every well-formed run must satisfy.
fn check_invariants(events: &[AgentEvent]) -> Result<(), String> {
	if !matches!(events.first(), Some(AgentEvent::AgentStart)) {
		return Err("first event is not agent_start".into());
	}
	if !matches!(events.last(), Some(AgentEvent::AgentEnd { .. })) {
		return Err("last event is not agent_end".into());
	}
	let starts = events
		.iter()
		.filter(|e| matches!(e, AgentEvent::AgentStart))
		.count();
	let ends = events
		.iter()
		.filter(|e| matches!(e, AgentEvent::AgentEnd { .. }))
		.count();
	if starts != 1 || ends != 1 {
		return Err(format!("agent_start/agent_end not unique: {starts}/{ends}"));
	}
	let turn_starts = events
		.iter()
		.filter(|e| matches!(e, AgentEvent::TurnStart))
		.count();
	let turn_ends = events
		.iter()
		.filter(|e| matches!(e, AgentEvent::TurnEnd { .. }))
		.count();
	if turn_starts != turn_ends {
		return Err(format!("turn_start/turn_end mismatch: {turn_starts}/{turn_ends}"));
	}

	// Every tool_execution_start id has a matching tool_execution_end.
	let mut start_ids = BTreeSet::new();
	let mut end_ids = BTreeSet::new();
	for event in events {
		match event {
			AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
				start_ids.insert(tool_call_id.clone());
			},
			AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
				end_ids.insert(tool_call_id.clone());
			},
			_ => {},
		}
	}
	if start_ids != end_ids {
		return Err(format!("tool exec start/end ids differ: {start_ids:?} vs {end_ids:?}"));
	}

	// message_update partial content length is monotonic within a message
	// (reset at each message_start).
	let mut last_len = 0usize;
	for event in events {
		match event {
			AgentEvent::MessageStart { .. } => last_len = 0,
			AgentEvent::MessageUpdate { message, .. } => {
				let len = match message {
					Message::Assistant(a) => a.content.len(),
					_ => 0,
				};
				if len < last_len {
					return Err(format!("message_update content length regressed: {len} < {last_len}"));
				}
				last_len = len;
			},
			_ => {},
		}
	}
	Ok(())
}

fn count(events: &[AgentEvent], f: impl Fn(&AgentEvent) -> bool) -> usize {
	events.iter().filter(|e| f(e)).count()
}

fn final_messages(events: &[AgentEvent]) -> Vec<Message> {
	match events.last() {
		Some(AgentEvent::AgentEnd { messages }) => messages.clone(),
		_ => vec![],
	}
}

// ─── stopReason branches
// ──────────────────────────────────────────────────────

#[tokio::test]
async fn stop_without_tools_ends_run() {
	let msg = assistant(vec![text("hello")], StopReason::Stop);
	let config = AgentConfig::new("fake", 1024)
		.with_stream_fn(scripted(vec![emit_nonstream_events(&std::sync::Arc::new(msg))]));
	let events =
		collect(agent_loop(vec![user("hi")], ctx(vec![]), config, CancelToken::default())).await;

	check_invariants(&events).unwrap();
	assert_eq!(count(&events, |e| matches!(e, AgentEvent::TurnEnd { .. })), 1);
	assert_eq!(count(&events, |e| matches!(e, AgentEvent::ToolExecutionStart { .. })), 0);
}

#[tokio::test]
async fn stop_with_tool_calls_executes_then_continues() {
	// `stop` (end_turn) carrying a tool call still runs the tool (adaptive Opus).
	let log = Arc::new(Mutex::new(vec![]));
	let tool = Box::new(RecordTool { name: "t", log: log.clone(), fail: false });
	let first = assistant(vec![tool_call("c1", "t", json!({}))], StopReason::Stop);
	let config = AgentConfig::new("fake", 1024).with_stream_fn(scripted_msg(first));
	let events =
		collect(agent_loop(vec![user("hi")], ctx(vec![tool]), config, CancelToken::default())).await;

	check_invariants(&events).unwrap();
	assert_eq!(*log.lock().unwrap(), vec!["t:c1".to_string()]);
	assert_eq!(count(&events, |e| matches!(e, AgentEvent::ToolExecutionEnd { .. })), 1);
	// Two turns: the tool turn + the terminating stop turn.
	assert_eq!(count(&events, |e| matches!(e, AgentEvent::TurnEnd { .. })), 2);
}

#[tokio::test]
async fn tool_use_stop_runs_tool() {
	let log = Arc::new(Mutex::new(vec![]));
	let tool = Box::new(RecordTool { name: "t", log: log.clone(), fail: false });
	let first = assistant(vec![tool_call("c1", "t", json!({}))], StopReason::ToolUse);
	let config = AgentConfig::new("fake", 1024).with_stream_fn(scripted_msg(first));
	let events =
		collect(agent_loop(vec![user("hi")], ctx(vec![tool]), config, CancelToken::default())).await;

	check_invariants(&events).unwrap();
	assert_eq!(*log.lock().unwrap(), vec!["t:c1".to_string()]);
}

#[tokio::test]
async fn length_stop_pairs_placeholder_without_executing() {
	let log = Arc::new(Mutex::new(vec![]));
	let tool = Box::new(RecordTool { name: "t", log: log.clone(), fail: false });
	// length: truncated tool_use is paired with a synthetic result, NOT run.
	let first = assistant(vec![tool_call("c1", "t", json!({}))], StopReason::Length);
	let config = AgentConfig::new("fake", 1024).with_stream_fn(scripted_msg(first));
	let events =
		collect(agent_loop(vec![user("hi")], ctx(vec![tool]), config, CancelToken::default())).await;

	check_invariants(&events).unwrap();
	assert!(log.lock().unwrap().is_empty(), "tool must not execute on length");
	// A synthetic placeholder result was still emitted (pairing preserved).
	assert_eq!(count(&events, |e| matches!(e, AgentEvent::ToolExecutionEnd { .. })), 1);
	// The synthetic result is a toolResult message flagged is_error.
	let has_synth = final_messages(&events)
		.iter()
		.any(|m| matches!(m, Message::ToolResult(r) if r.is_error));
	assert!(has_synth, "expected a synthetic error toolResult");
}

#[tokio::test]
async fn error_stop_ends_run_with_placeholder_pairing() {
	// Manually scripted: a completed tool call then a terminal error carrying it.
	let done_msg = assistant(vec![tool_call("c1", "t", json!({}))], StopReason::Error);
	let events_script = vec![
		AssistantMessageEvent::Start {
			partial: std::sync::Arc::new(assistant(vec![], StopReason::Error)),
		},
		AssistantMessageEvent::ToolcallStart {
			content_index: 0,
			partial:       std::sync::Arc::new(assistant(vec![], StopReason::Error)),
		},
		AssistantMessageEvent::ToolcallEnd {
			content_index: 0,
			tool_call:     ToolCall {
				id:                "c1".into(),
				name:              "t".into(),
				arguments:         json!({}),
				thought_signature: None,
				intent:            None,
				raw_block:         None,
				custom_wire_name:  None,
			},
			partial:       std::sync::Arc::new(done_msg.clone()),
		},
		AssistantMessageEvent::Error {
			reason: ErrorReason::Error,
			error:  std::sync::Arc::new(done_msg),
		},
	];
	let log = Arc::new(Mutex::new(vec![]));
	let tool = Box::new(RecordTool { name: "t", log: log.clone(), fail: false });
	let config = AgentConfig::new("fake", 1024).with_stream_fn(scripted(vec![events_script]));
	let events =
		collect(agent_loop(vec![user("hi")], ctx(vec![tool]), config, CancelToken::default())).await;

	check_invariants(&events).unwrap();
	assert!(log.lock().unwrap().is_empty(), "error turn must not execute tools");
	// The completed tool call was retained and paired with a placeholder.
	assert_eq!(count(&events, |e| matches!(e, AgentEvent::ToolExecutionEnd { .. })), 1);
	assert_eq!(count(&events, |e| matches!(e, AgentEvent::TurnEnd { .. })), 1);
}

#[tokio::test]
async fn aborted_stop_pairs_placeholder() {
	let done_msg = assistant(vec![tool_call("c1", "t", json!({}))], StopReason::Aborted);
	let events_script = vec![
		AssistantMessageEvent::Start {
			partial: std::sync::Arc::new(assistant(vec![], StopReason::Aborted)),
		},
		AssistantMessageEvent::ToolcallEnd {
			content_index: 0,
			tool_call:     ToolCall {
				id:                "c1".into(),
				name:              "t".into(),
				arguments:         json!({}),
				thought_signature: None,
				intent:            None,
				raw_block:         None,
				custom_wire_name:  None,
			},
			partial:       std::sync::Arc::new(done_msg.clone()),
		},
		AssistantMessageEvent::Error {
			reason: ErrorReason::Aborted,
			error:  std::sync::Arc::new(done_msg),
		},
	];
	let config = AgentConfig::new("fake", 1024).with_stream_fn(scripted(vec![events_script]));
	let events =
		collect(agent_loop(vec![user("hi")], ctx(vec![]), config, CancelToken::default())).await;

	check_invariants(&events).unwrap();
	let synth = final_messages(&events)
		.iter()
		.any(|m| matches!(m, Message::ToolResult(r) if r.is_error));
	assert!(synth, "aborted turn with a completed tool call must pair a placeholder");
}

// ─── Serial multi-tool
// ────────────────────────────────────────────────────────

#[tokio::test]
async fn multiple_tool_calls_run_serially_in_order() {
	let log = Arc::new(Mutex::new(vec![]));
	let tools: Vec<Box<dyn pi_tools::DynTool>> = vec![
		Box::new(RecordTool { name: "a", log: log.clone(), fail: false }),
		Box::new(RecordTool { name: "b", log: log.clone(), fail: false }),
	];
	let first = assistant(
		vec![tool_call("c1", "a", json!({})), tool_call("c2", "b", json!({}))],
		StopReason::ToolUse,
	);
	let config = AgentConfig::new("fake", 1024).with_stream_fn(scripted_msg(first));
	let events =
		collect(agent_loop(vec![user("hi")], ctx(tools), config, CancelToken::default())).await;

	check_invariants(&events).unwrap();
	assert_eq!(*log.lock().unwrap(), vec!["a:c1".to_string(), "b:c2".to_string()]);
	assert_eq!(count(&events, |e| matches!(e, AgentEvent::ToolExecutionEnd { .. })), 2);
}

#[tokio::test]
async fn tool_error_does_not_break_the_chain() {
	let log = Arc::new(Mutex::new(vec![]));
	let tools: Vec<Box<dyn pi_tools::DynTool>> = vec![
		Box::new(RecordTool { name: "a", log: log.clone(), fail: true }),
		Box::new(RecordTool { name: "b", log: log.clone(), fail: false }),
	];
	let first = assistant(
		vec![tool_call("c1", "a", json!({})), tool_call("c2", "b", json!({}))],
		StopReason::ToolUse,
	);
	let config = AgentConfig::new("fake", 1024).with_stream_fn(scripted_msg(first));
	let events =
		collect(agent_loop(vec![user("hi")], ctx(tools), config, CancelToken::default())).await;

	check_invariants(&events).unwrap();
	// Both tools ran (the first errored but the batch continued).
	assert_eq!(*log.lock().unwrap(), vec!["a:c1".to_string(), "b:c2".to_string()]);
	let end_errs: Vec<Option<bool>> = events
		.iter()
		.filter_map(|e| match e {
			AgentEvent::ToolExecutionEnd { tool_call_id, is_error, .. } if tool_call_id == "c1" => {
				Some(*is_error)
			},
			_ => None,
		})
		.collect();
	assert_eq!(end_errs, vec![Some(true)], "the failing tool produced an is_error result");
}

#[tokio::test]
async fn unknown_tool_name_yields_error_result() {
	let first = assistant(vec![tool_call("c1", "ghost", json!({}))], StopReason::ToolUse);
	let config = AgentConfig::new("fake", 1024).with_stream_fn(scripted_msg(first));
	let events =
		collect(agent_loop(vec![user("hi")], ctx(vec![]), config, CancelToken::default())).await;

	check_invariants(&events).unwrap();
	let err_result = events.iter().any(|e| {
		matches!(
			e,
			AgentEvent::ToolExecutionEnd { tool_call_id, is_error: Some(true), .. } if tool_call_id == "c1"
		)
	});
	assert!(err_result, "unknown tool must yield an is_error result");
}

#[tokio::test]
async fn missing_required_arg_yields_error_result() {
	struct NeedsArg;
	impl Tool for NeedsArg {
		fn name(&self) -> &'static str {
			"needs"
		}

		fn description(&self) -> &'static str {
			"requires x"
		}

		fn input_schema(&self) -> Value {
			json!({ "type": "object", "properties": { "x": { "type": "string" } }, "required": ["x"] })
		}

		async fn execute(
			&self,
			_id: &str,
			_args: Value,
			_ct: &CancelToken,
		) -> Result<ToolResult, ToolError> {
			panic!("must not execute with missing required arg");
		}
	}
	let first = assistant(vec![tool_call("c1", "needs", json!({}))], StopReason::ToolUse);
	let config = AgentConfig::new("fake", 1024).with_stream_fn(scripted_msg(first));
	let events = collect(agent_loop(
		vec![user("hi")],
		ctx(vec![Box::new(NeedsArg)]),
		config,
		CancelToken::default(),
	))
	.await;

	check_invariants(&events).unwrap();
	let err = events.iter().any(|e| {
		matches!(
			e,
			AgentEvent::ToolExecutionEnd { tool_call_id, is_error: Some(true), .. } if tool_call_id == "c1"
		)
	});
	assert!(err, "missing required arg must yield an is_error result");
}

// ─── Cancellation
// ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn pre_aborted_token_short_circuits_to_aborted() {
	let mut ct = CancelToken::default();
	let tok = ct.emplace_abort_token();
	tok.abort(AbortReason::User);
	let first = assistant(vec![tool_call("c1", "t", json!({}))], StopReason::ToolUse);
	let config = AgentConfig::new("fake", 1024).with_stream_fn(scripted_msg(first));
	let events = collect(agent_loop(vec![user("hi")], ctx(vec![]), config, ct)).await;

	check_invariants(&events).unwrap();
	// The run ended aborted: the final assistant message is aborted, one turn.
	let aborted = final_messages(&events)
		.iter()
		.any(|m| matches!(m, Message::Assistant(a) if a.stop_reason == StopReason::Aborted));
	assert!(aborted, "expected an aborted assistant message");
	assert_eq!(count(&events, |e| matches!(e, AgentEvent::TurnEnd { .. })), 1);
}

// ─── Serde shape
// ──────────────────────────────────────────────────────────────

#[test]
fn agent_event_serializes_camel_case_tagged() {
	let ev = AgentEvent::ToolExecutionStart {
		tool_call_id: "c1".into(),
		tool_name:    "t".into(),
		args:         json!({ "k": 1 }),
		intent:       None,
	};
	let v = serde_json::to_value(&ev).unwrap();
	assert_eq!(v["type"], "tool_execution_start");
	assert_eq!(v["toolCallId"], "c1");
	assert_eq!(v["toolName"], "t");
	assert!(v.get("intent").is_none(), "None intent must be omitted");

	let end = AgentEvent::ToolExecutionEnd {
		tool_call_id: "c1".into(),
		tool_name:    "t".into(),
		result:       json!({ "content": [] }),
		is_error:     Some(true),
	};
	let v = serde_json::to_value(&end).unwrap();
	assert_eq!(v["type"], "tool_execution_end");
	assert_eq!(v["isError"], true);

	let turn = AgentEvent::TurnEnd { message: user("x"), tool_results: vec![] };
	let v = serde_json::to_value(&turn).unwrap();
	assert_eq!(v["type"], "turn_end");
	assert!(v.get("toolResults").is_some(), "tool_results serializes as toolResults");
}

#[test]
fn done_and_error_reason_shapes() {
	assert_eq!(serde_json::to_value(DoneReason::ToolUse).unwrap(), json!("toolUse"));
	assert_eq!(serde_json::to_value(ErrorReason::Aborted).unwrap(), json!("aborted"));
}
