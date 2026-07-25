//! Response-parsing fixtures: official-doc shapes, tolerance paths (unknown
//! block tags / stop reasons / private fields), the error envelope, and an SSE
//! smoke pass over all six raw stream event kinds (types only in WP-1.1a).

use pi_ai::wire::{
	ContentBlockDelta, ErrorEnvelope, KnownStopReason, RawMessageStreamEvent, ResponseBlock,
	ResponseContentBlock, ResponseMessage, WireStopReason,
};

fn load(name: &str) -> String {
	let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
	std::fs::read_to_string(path).expect("read fixture")
}

fn known(block: &ResponseBlock) -> &ResponseContentBlock {
	match block {
		ResponseBlock::Known(inner) => inner,
		ResponseBlock::Unknown(value) => panic!("expected known block, got {value}"),
	}
}

#[test]
fn parses_text_response() {
	let message: ResponseMessage = serde_json::from_str(&load("response_text.json")).unwrap();
	assert_eq!(message.id, "msg_01XFDUDYJgAACzvnptvVoYEL");
	assert_eq!(message.model.as_deref(), Some("claude-sonnet-5"));
	assert_eq!(message.stop_reason, Some(WireStopReason::Known(KnownStopReason::EndTurn)));
	assert_eq!(message.stop_sequence, None);
	let ResponseContentBlock::Text { text } = known(&message.content[0]) else {
		panic!("expected text block");
	};
	assert_eq!(text, "Hello! How can I help you today?");
	assert_eq!(message.usage.input_tokens, Some(12));
	assert_eq!(message.usage.output_tokens, Some(9));
	assert_eq!(message.usage.cache_read_input_tokens, Some(4));
	assert_eq!(message.usage.cache_creation_input_tokens, Some(8));
	let cttl = message.usage.cache_creation.unwrap();
	assert_eq!(cttl.ephemeral_5m_input_tokens, Some(8));
	assert_eq!(cttl.ephemeral_1h_input_tokens, Some(0));
}

#[test]
fn parses_tool_use_response() {
	let message: ResponseMessage = serde_json::from_str(&load("response_tool_use.json")).unwrap();
	assert_eq!(message.stop_reason, Some(WireStopReason::Known(KnownStopReason::ToolUse)));
	assert_eq!(message.content.len(), 2);
	let ResponseContentBlock::ToolUse { id, name, input } = known(&message.content[1]) else {
		panic!("expected tool_use block");
	};
	assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
	assert_eq!(name, "get_weather");
	assert_eq!(input.as_ref().unwrap()["location"], "San Francisco, CA");
}

#[test]
fn parses_thinking_response() {
	let message: ResponseMessage = serde_json::from_str(&load("response_thinking.json")).unwrap();
	let ResponseContentBlock::Thinking { thinking, signature } = known(&message.content[0]) else {
		panic!("expected thinking block");
	};
	assert!(thinking.starts_with("Let me reason"));
	assert!(signature.is_some());
	let ResponseContentBlock::RedactedThinking { data } = known(&message.content[1]) else {
		panic!("expected redacted_thinking block");
	};
	assert!(!data.is_empty());
	let ResponseContentBlock::Text { text } = known(&message.content[2]) else {
		panic!("expected text block");
	};
	assert_eq!(text, "The answer is 42.");
}

#[test]
fn parses_error_envelope() {
	let envelope: ErrorEnvelope = serde_json::from_str(&load("response_error.json")).unwrap();
	assert_eq!(envelope.error.error_type, "invalid_request_error");
	assert_eq!(envelope.error.message, "max_tokens: Field required");
}

#[test]
fn tolerates_unknown_variants() {
	let message: ResponseMessage =
		serde_json::from_str(&load("response_unknown_variants.json")).unwrap();
	// Unknown block tag survives verbatim instead of failing the parse.
	let ResponseBlock::Unknown(value) = &message.content[0] else {
		panic!("expected unknown block");
	};
	assert_eq!(value["type"], "banana");
	assert_eq!(value["payload"]["nested"][2], 3);
	// Sibling known block still parses.
	let ResponseContentBlock::Text { text } = known(&message.content[1]) else {
		panic!("expected text block");
	};
	assert_eq!(text, "Still parsed.");
	// Unknown stop reason degrades to Other, private usage fields are ignored.
	assert_eq!(message.stop_reason, Some(WireStopReason::Other("grape".into())));
	assert_eq!(message.usage.input_tokens, Some(1));
}

/// Real body captured from the `DeepSeek` Anthropic-compatible endpoint
/// (`examples/complete.rs`, 2026-07-25): thinking model, signature = message
/// id, private `service_tier` usage field, no cache/server sections.
#[test]
fn parses_deepseek_v4_flash_response() {
	let message: ResponseMessage =
		serde_json::from_str(&load("response_deepseek_v4_flash.json")).unwrap();
	assert_eq!(message.model.as_deref(), Some("deepseek-v4-flash"));
	assert_eq!(message.stop_reason, Some(WireStopReason::Known(KnownStopReason::EndTurn)));
	let ResponseContentBlock::Thinking { signature, .. } = known(&message.content[0]) else {
		panic!("expected thinking block");
	};
	assert_eq!(signature.as_deref(), Some(message.id.as_str()));
	let ResponseContentBlock::Text { text } = known(&message.content[1]) else {
		panic!("expected text block");
	};
	assert_eq!(text, "pong");
	assert_eq!(message.usage.output_tokens, Some(23));
	assert!(message.usage.cache_creation.is_none());
}

#[test]
fn parses_all_sse_event_kinds() {
	let frames = [
		r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}"#,
		r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
		r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
		r#"{"type":"content_block_stop","index":0}"#,
		r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}"#,
		r#"{"type":"message_stop"}"#,
	];
	let events: Vec<RawMessageStreamEvent> = frames
		.iter()
		.map(|frame| serde_json::from_str(frame).unwrap())
		.collect();
	assert!(matches!(events[0], RawMessageStreamEvent::MessageStart { .. }));
	let RawMessageStreamEvent::ContentBlockDelta { index, delta } = &events[2] else {
		panic!("expected content_block_delta");
	};
	assert_eq!(*index, 0);
	assert_eq!(*delta, ContentBlockDelta::TextDelta { text: "Hello".into() });
	let RawMessageStreamEvent::MessageDelta { delta, usage } = &events[4] else {
		panic!("expected message_delta");
	};
	assert_eq!(delta.stop_reason, Some(WireStopReason::Known(KnownStopReason::EndTurn)));
	assert_eq!(usage.output_tokens, Some(15));
	assert!(matches!(events[5], RawMessageStreamEvent::MessageStop));
}
