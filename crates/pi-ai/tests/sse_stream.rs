//! SSE pipeline tests: parser → `parse_message_event` → `StreamingBuilder`.
//!
//! The happy path replays a real captured `DeepSeek` transcript
//! (`fixtures/sse_deepseek_v4_flash.sse`); the edge cases use synthetic
//! transcripts (malformed frames, unknown blocks, refusal, truncation).

use std::sync::Arc;

use pi_ai::{
	builder::StreamingBuilder,
	convert::RequestMeta,
	event::{AssistantMessageEvent, DoneReason, ErrorReason},
	message::{AssistantContent, StopReason},
	sse::{SseParser, parse_message_event, stream_error_message},
	wire::RawMessageStreamEvent,
};

fn meta() -> RequestMeta {
	RequestMeta {
		api:       "anthropic-messages".into(),
		provider:  "deepseek".into(),
		model:     "deepseek-v4-flash".into(),
		timestamp: 1_753_500_000_000,
		duration:  None,
	}
}

/// Feed a transcript through the parser in deliberately awkward chunk sizes,
/// run the builder, and return (all contract events incl. terminal, message).
fn replay(
	transcript: &str,
	chunk: usize,
) -> (Vec<AssistantMessageEvent>, Arc<pi_ai::AssistantMessage>) {
	let mut parser = SseParser::new();
	let mut builder = StreamingBuilder::new(&meta());
	let mut events = vec![AssistantMessageEvent::Start { partial: builder.snapshot() }];
	let bytes = transcript.as_bytes();
	for piece in bytes.chunks(chunk) {
		for frame in parser.push(piece) {
			assert_ne!(frame.event.as_deref(), Some("error"), "unexpected error frame");
			let Some(raw) = parse_message_event(&frame) else {
				continue;
			};
			events.extend(builder.on_event(raw));
		}
	}
	let (message, terminal) = builder.finish();
	events.extend(terminal);
	(events, message)
}

const fn kind(event: &AssistantMessageEvent) -> &'static str {
	match event {
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
	}
}

#[test]
fn deepseek_transcript_replays_to_contract_sequence() {
	let transcript = std::fs::read_to_string(format!(
		"{}/tests/fixtures/sse_deepseek_v4_flash.sse",
		env!("CARGO_MANIFEST_DIR")
	))
	.unwrap();
	for chunk in [1, 7, 4096] {
		let (events, message) = replay(&transcript, chunk);
		let kinds: Vec<_> = events.iter().map(kind).collect();
		assert_eq!(kinds[0], "start");
		assert_eq!(kinds[1], "thinking_start");
		assert_eq!(kinds[kinds.len() - 1], "done");
		assert_eq!(kinds.iter().filter(|k| **k == "thinking_delta").count(), 27);
		assert_eq!(kinds.iter().filter(|k| **k == "text_delta").count(), 2);
		assert!(kinds.contains(&"thinking_end") && kinds.contains(&"text_end"));
		// Final message shape.
		assert_eq!(message.stop_reason, StopReason::Stop);
		let AssistantContent::Thinking(thinking) = &message.content[0] else {
			panic!("expected thinking block");
		};
		assert!(thinking.thinking.contains("pong"));
		assert!(thinking.thinking_signature.is_some(), "signature_delta must accumulate");
		let AssistantContent::Text(text) = &message.content[1] else {
			panic!("expected text")
		};
		assert_eq!(text.text, "pong");
		assert_eq!(message.usage.output, 30, "message_delta usage must overwrite");
		assert_eq!(message.usage.total_tokens, 40);
		assert!(message.response_id.is_some());
		// partial snapshots accumulate monotonically
		let mut last = 0usize;
		for event in &events {
			if let AssistantMessageEvent::TextDelta { partial, .. }
			| AssistantMessageEvent::ThinkingDelta { partial, .. } = event
			{
				assert!(partial.content.len() >= last);
				last = partial.content.len();
			}
		}
	}
}

fn frames(lines: &[(&str, &str)]) -> String {
	lines.iter().fold(String::new(), |mut out, (event, data)| {
		use std::fmt::Write;
		let _ = writeln!(out, "event: {event}\ndata: {data}\n");
		out
	})
}

#[test]
fn tool_use_arguments_accumulate_and_parse() {
	let transcript = frames(&[
		(
			"message_start",
			r#"{"type":"message_start","message":{"id":"m1","usage":{"input_tokens":5,"output_tokens":0}}}"#,
		),
		(
			"content_block_start",
			r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"get_weather","input":{}}}"#,
		),
		(
			"content_block_delta",
			r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}"#,
		),
		(
			"content_block_delta",
			r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}"#,
		),
		("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
		(
			"message_delta",
			r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}"#,
		),
		("message_stop", r#"{"type":"message_stop"}"#),
	]);
	let (events, message) = replay(&transcript, 11);
	let kinds: Vec<_> = events.iter().map(kind).collect();
	assert_eq!(kinds, [
		"start",
		"toolcall_start",
		"toolcall_delta",
		"toolcall_delta",
		"toolcall_end",
		"done"
	]);
	let AssistantMessageEvent::Done { reason, .. } = events.last().unwrap() else {
		unreachable!()
	};
	assert_eq!(*reason, DoneReason::ToolUse);
	let AssistantContent::ToolCall(call) = &message.content[0] else {
		panic!()
	};
	assert_eq!(call.arguments, serde_json::json!({"location": "Paris"}));
	// toolcall_end carries the parsed arguments too.
	let AssistantMessageEvent::ToolcallEnd { tool_call, .. } = &events[4] else {
		unreachable!()
	};
	assert_eq!(tool_call.arguments["location"], "Paris");
}

#[test]
fn malformed_and_unknown_frames_are_skipped() {
	let transcript = frames(&[
		("message_start", r#"{"type":"message_start","message":{"id":"m1","usage":{}}}"#),
		("content_block_delta", "{not json"),
		("bogus_event", r#"{"type":"bogus_event"}"#),
		(
			"content_block_start",
			r#"{"type":"content_block_start","index":0,"content_block":{"type":"banana","x":1}}"#,
		),
		(
			"content_block_delta",
			r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ignored"}}"#,
		),
		("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
		(
			"content_block_start",
			r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
		),
		(
			"content_block_delta",
			r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ok"}}"#,
		),
		("content_block_stop", r#"{"type":"content_block_stop","index":1}"#),
		("message_stop", r#"{"type":"message_stop"}"#),
	]);
	let (events, message) = replay(&transcript, 4096);
	let kinds: Vec<_> = events.iter().map(kind).collect();
	assert_eq!(kinds, ["start", "text_start", "text_delta", "text_end", "done"]);
	assert_eq!(message.content.len(), 1, "unknown block must not enter content");
}

#[test]
fn refusal_stop_reason_becomes_error_turn() {
	let transcript = frames(&[
		("message_start", r#"{"type":"message_start","message":{"id":"m1","usage":{}}}"#),
		(
			"message_delta",
			r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","category":"safety","explanation":"nope"}},"usage":{}}"#,
		),
		("message_stop", r#"{"type":"message_stop"}"#),
	]);
	let (events, message) = replay(&transcript, 4096);
	let AssistantMessageEvent::Error { reason, error } = events.last().unwrap() else {
		panic!("expected error terminal, got {:?}", events.last());
	};
	assert_eq!(*reason, ErrorReason::Error);
	assert_eq!(error.error_message.as_deref(), Some("Refusal (safety): nope"));
	assert_eq!(message.stop_reason, StopReason::Error);
	assert!(message.stop_details.is_some());
}

#[test]
fn truncated_stream_finalizes_dangling_block() {
	let transcript = frames(&[
		("message_start", r#"{"type":"message_start","message":{"id":"m1","usage":{}}}"#),
		(
			"content_block_start",
			r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
		),
		(
			"content_block_delta",
			r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#,
		),
		// stream drops here: no content_block_stop / message_delta / message_stop
	]);
	let (events, message) = replay(&transcript, 4096);
	let kinds: Vec<_> = events.iter().map(kind).collect();
	assert_eq!(kinds, ["start", "text_start", "text_delta", "text_end", "done"]);
	let AssistantContent::Text(text) = &message.content[0] else {
		panic!()
	};
	assert_eq!(text.text, "partial");
}

#[test]
fn stream_without_message_start_is_error() {
	let mut builder = StreamingBuilder::new(&meta());
	let _ = builder.on_event(RawMessageStreamEvent::MessageStop);
	let (message, terminal) = builder.finish();
	assert_eq!(message.stop_reason, StopReason::Error);
	assert!(matches!(
		terminal.last(),
		Some(AssistantMessageEvent::Error { reason: ErrorReason::Error, .. })
	));
	assert_eq!(message.error_message.as_deref(), Some("stream ended before message_start"));
}

#[test]
fn invalid_tool_json_falls_back_to_error_shape() {
	let transcript = frames(&[
		("message_start", r#"{"type":"message_start","message":{"id":"m1","usage":{}}}"#),
		(
			"content_block_start",
			r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"f","input":{}}}"#,
		),
		(
			"content_block_delta",
			r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"broken\": "}}"#,
		),
		("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
		("message_stop", r#"{"type":"message_stop"}"#),
	]);
	let (_, message) = replay(&transcript, 4096);
	let AssistantContent::ToolCall(call) = &message.content[0] else {
		panic!()
	};
	assert!(call.arguments["__parseError"].is_string());
	assert_eq!(call.arguments["__rawJson"], "{\"broken\": ");
}

#[test]
fn builder_fail_aborted_keeps_partial_content() {
	let mut builder = StreamingBuilder::new(&meta());
	let _ = builder.on_event(
		serde_json::from_str(
			r#"{"type":"message_start","message":{"id":"m1","usage":{"input_tokens":3}}}"#,
		)
		.unwrap(),
	);
	let _ = builder.on_event(
		serde_json::from_str(
			r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"hi"}}"#,
		)
		.unwrap(),
	);
	let (message, events) = builder.fail(StopReason::Aborted, "Request was aborted.".into());
	assert!(matches!(
		events.last(),
		Some(AssistantMessageEvent::Error { reason: ErrorReason::Aborted, .. })
	));
	assert_eq!(message.stop_reason, StopReason::Aborted);
	assert_eq!(message.content.len(), 1, "accumulated content survives abort");
}

#[test]
fn stream_error_message_formats_envelope() {
	assert_eq!(
		stream_error_message(
			r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
		),
		"Anthropic stream error (overloaded_error): Overloaded"
	);
	assert_eq!(stream_error_message("plain junk"), "plain junk");
}
