//! Event-interface tests: synthesized non-streaming sequences, stream-container
//! semantics, event JSON shape parity with TS (`types.ts:900`), and the
//! stop-reason / usage conversion tables.

use std::sync::Arc;

use pi_ai::{
	AiError,
	convert::{
		RequestMeta, convert_response, convert_usage, emit_nonstream_events, map_stop_reason,
	},
	event::{AssistantMessageEvent, DoneReason, ErrorReason},
	message::{AssistantContent, AssistantMessage, StopReason},
	stream::AssistantMessageEventStream,
	wire::{KnownStopReason, ResponseMessage, WireStopReason},
};

fn meta() -> RequestMeta {
	RequestMeta {
		api:       "anthropic-messages".into(),
		provider:  "anthropic".into(),
		model:     "claude-sonnet-5".into(),
		timestamp: 1_753_430_000_000,
		duration:  Some(1234),
	}
}

fn text_message() -> Arc<AssistantMessage> {
	let response: ResponseMessage = serde_json::from_str(
		&std::fs::read_to_string(format!(
			"{}/tests/fixtures/response_text.json",
			env!("CARGO_MANIFEST_DIR")
		))
		.unwrap(),
	)
	.unwrap();
	Arc::new(convert_response(&response, &meta()))
}

fn tool_message() -> Arc<AssistantMessage> {
	let response: ResponseMessage = serde_json::from_str(
		&std::fs::read_to_string(format!(
			"{}/tests/fixtures/response_tool_use.json",
			env!("CARGO_MANIFEST_DIR")
		))
		.unwrap(),
	)
	.unwrap();
	Arc::new(convert_response(&response, &meta()))
}

fn kinds(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
	events
		.iter()
		.map(|event| match event {
			AssistantMessageEvent::Start { .. } => "start",
			AssistantMessageEvent::TextStart { .. } => "text_start",
			AssistantMessageEvent::TextDelta { .. } => "text_delta",
			AssistantMessageEvent::TextEnd { .. } => "text_end",
			AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
			AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
			AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
			AssistantMessageEvent::ImageEnd { .. } => "image_end",
			AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
			AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
			AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
			AssistantMessageEvent::Done { .. } => "done",
			AssistantMessageEvent::Error { .. } => "error",
		})
		.collect()
}

#[test]
fn text_sequence_matches_contract() {
	let message = text_message();
	let events = emit_nonstream_events(&message);
	assert_eq!(kinds(&events), ["start", "text_start", "text_delta", "text_end", "done"]);
	// start has empty content, partials accumulate, done carries the final.
	let AssistantMessageEvent::Start { partial } = &events[0] else {
		unreachable!()
	};
	assert!(partial.content.is_empty());
	let AssistantMessageEvent::TextDelta { delta, partial, .. } = &events[2] else {
		unreachable!()
	};
	assert_eq!(delta, "Hello! How can I help you today?");
	assert_eq!(partial.content.len(), 1);
	let AssistantMessageEvent::Done { reason, message: done } = &events[4] else {
		unreachable!()
	};
	assert_eq!(*reason, DoneReason::Stop);
	assert_eq!(done.as_ref(), message.as_ref());
	// Usage metadata is present on every partial.
	assert_eq!(partial.usage.total_tokens, message.usage.total_tokens);
}

#[test]
fn toolcall_sequence_matches_contract() {
	let message = tool_message();
	let events = emit_nonstream_events(&message);
	assert_eq!(kinds(&events), [
		"start",
		"text_start",
		"text_delta",
		"text_end",
		"toolcall_start",
		"toolcall_delta",
		"toolcall_end",
		"done"
	]);
	let AssistantMessageEvent::ToolcallStart { content_index, partial } = &events[4] else {
		unreachable!()
	};
	assert_eq!(*content_index, 1);
	let AssistantContent::ToolCall(shell) = &partial.content[1] else {
		unreachable!()
	};
	assert_eq!(shell.name, "get_weather");
	assert_eq!(shell.arguments, serde_json::json!({}));
	let AssistantMessageEvent::ToolcallDelta { delta, .. } = &events[5] else {
		unreachable!()
	};
	let parsed: serde_json::Value = serde_json::from_str(delta).unwrap();
	assert_eq!(parsed["location"], "San Francisco, CA");
	let AssistantMessageEvent::ToolcallEnd { tool_call, .. } = &events[6] else {
		unreachable!()
	};
	assert_eq!(tool_call.id, "toolu_01A09q90qw90lq917835lq9");
	let AssistantMessageEvent::Done { reason, .. } = &events[7] else {
		unreachable!()
	};
	assert_eq!(*reason, DoneReason::ToolUse);
}

#[test]
fn partial_content_grows_monotonically() {
	let events = emit_nonstream_events(&tool_message());
	let mut last_len = 0usize;
	for event in &events {
		let partial = match event {
			AssistantMessageEvent::Done { message, .. } => message,
			AssistantMessageEvent::Error { error, .. } => error,
			AssistantMessageEvent::Start { partial }
			| AssistantMessageEvent::TextStart { partial, .. }
			| AssistantMessageEvent::TextDelta { partial, .. }
			| AssistantMessageEvent::TextEnd { partial, .. }
			| AssistantMessageEvent::ThinkingStart { partial, .. }
			| AssistantMessageEvent::ThinkingDelta { partial, .. }
			| AssistantMessageEvent::ThinkingEnd { partial, .. }
			| AssistantMessageEvent::ImageEnd { partial, .. }
			| AssistantMessageEvent::ToolcallStart { partial, .. }
			| AssistantMessageEvent::ToolcallDelta { partial, .. }
			| AssistantMessageEvent::ToolcallEnd { partial, .. } => partial,
		};
		assert!(partial.content.len() >= last_len, "content shrank");
		last_len = partial.content.len();
	}
}

#[test]
fn error_message_emits_single_error_event() {
	let error = AiError::Api {
		status:     429,
		message:    "429 rate limited".into(),
		body:       None,
		request_id: None,
	};
	let message = Arc::new(pi_ai::convert::error_to_message(&error, &meta()));
	assert_eq!(message.stop_reason, StopReason::Error);
	assert_eq!(message.error_status, Some(429));
	let events = emit_nonstream_events(&message);
	assert_eq!(kinds(&events), ["error"]);
	let AssistantMessageEvent::Error { reason, .. } = &events[0] else {
		unreachable!()
	};
	assert_eq!(*reason, ErrorReason::Error);
}

#[test]
fn event_json_shape_matches_ts() {
	let events = emit_nonstream_events(&text_message());
	let delta = serde_json::to_value(&events[2]).unwrap();
	assert_eq!(delta["type"], "text_delta");
	assert_eq!(delta["contentIndex"], 0);
	assert!(delta["delta"].is_string());
	assert_eq!(delta["partial"]["role"], "assistant");
	assert_eq!(delta["partial"]["stopReason"], "stop");
	assert_eq!(delta["partial"]["usage"]["cacheRead"], 4);
	assert_eq!(delta["partial"]["usage"]["cttl"]["ephemeral5m"], 8);
	let done = serde_json::to_value(&events[4]).unwrap();
	assert_eq!(done["type"], "done");
	assert_eq!(done["reason"], "stop");
	assert_eq!(done["message"]["content"][0]["type"], "text");
	let tool_events = emit_nonstream_events(&tool_message());
	let end = serde_json::to_value(&tool_events[6]).unwrap();
	assert_eq!(end["type"], "toolcall_end");
	assert_eq!(end["toolCall"]["type"], "toolCall");
	assert_eq!(end["toolCall"]["arguments"]["unit"], "celsius");
	let done = serde_json::to_value(&tool_events[7]).unwrap();
	assert_eq!(done["reason"], "toolUse");
}

#[test]
fn map_stop_reason_full_table() {
	use KnownStopReason as K;
	let cases = [
		(K::EndTurn, StopReason::Stop),
		(K::MaxTokens, StopReason::Length),
		(K::ModelContextWindowExceeded, StopReason::Length),
		(K::ToolUse, StopReason::ToolUse),
		(K::Refusal, StopReason::Error),
		(K::Sensitive, StopReason::Error),
		(K::PauseTurn, StopReason::Stop),
		(K::StopSequence, StopReason::Stop),
	];
	for (wire, expected) in cases {
		assert_eq!(map_stop_reason(&WireStopReason::Known(wire)), expected);
	}
	assert_eq!(map_stop_reason(&WireStopReason::Other("grape".into())), StopReason::Stop);
}

#[test]
fn convert_usage_table() {
	let wire: pi_ai::wire::WireUsage = serde_json::from_str(
		r#"{"input_tokens":10,"output_tokens":20,"cache_read_input_tokens":30,
		    "cache_creation_input_tokens":40,
		    "cache_creation":{"ephemeral_5m_input_tokens":40,"ephemeral_1h_input_tokens":0},
		    "server_tool_use":{"web_search_requests":2}}"#,
	)
	.unwrap();
	let usage = convert_usage(&wire);
	assert_eq!(
		(usage.input, usage.output, usage.cache_read, usage.cache_write, usage.total_tokens),
		(10, 20, 30, 40, 100)
	);
	let cttl = usage.cttl.unwrap();
	assert_eq!(cttl.ephemeral_5m, Some(40));
	assert_eq!(cttl.ephemeral_1h, None);
	assert_eq!(usage.server.unwrap().web_search, Some(2));
	assert_eq!(usage.cost.total, 0.0);
	// All-zero extras collapse to None.
	let empty = convert_usage(&serde_json::from_str(r#"{"cache_creation":{}}"#).unwrap());
	assert!(empty.cttl.is_none() && empty.server.is_none());
}

#[tokio::test]
async fn stream_delivers_terminal_and_result() {
	let message = text_message();
	let (sink, mut stream) = AssistantMessageEventStream::channel();
	let events = emit_nonstream_events(&message);
	let expected = events.len();
	for event in events {
		sink.push(event);
	}
	// Push after terminal is a no-op.
	sink.push(AssistantMessageEvent::Start { partial: Arc::clone(&message) });
	let mut seen = Vec::new();
	while let Some(event) = stream.next().await {
		let terminal = event.is_terminal();
		seen.push(event);
		if terminal {
			break;
		}
	}
	assert_eq!(seen.len(), expected);
	assert!(seen.last().unwrap().is_terminal());
}

#[tokio::test]
async fn stream_result_without_iteration() {
	let message = text_message();
	let (sink, stream) = AssistantMessageEventStream::channel();
	for event in emit_nonstream_events(&message) {
		sink.push(event);
	}
	drop(sink);
	let result = stream.result().await.unwrap();
	assert_eq!(result.as_ref(), message.as_ref());
}

#[tokio::test]
async fn stream_without_terminal_errors() {
	let message = text_message();
	let (sink, stream) = AssistantMessageEventStream::channel();
	sink.push(AssistantMessageEvent::Start { partial: Arc::clone(&message) });
	drop(sink);
	assert!(matches!(stream.result().await, Err(AiError::StreamEnded)));
}
