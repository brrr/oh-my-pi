//! WP-1.1a L2 probe: one real non-streaming completion against an
//! Anthropic-compatible endpoint (`DeepSeek` by default), through both the
//! direct `complete_message` path and the synthesized event-stream path.
//!
//! ```sh
//! cargo run -p pi-ai --example complete
//! # overrides: OMP_AI_BASE_URL / OMP_AI_MODEL / OMP_AI_AUTH_ENTRY / ANTHROPIC_API_KEY
//! ```
//!
//! Exits non-zero unless every assertion holds; the raw wire body goes to
//! stderr so a redacted copy can be committed as a parse fixture.

use pi_ai::{
	auth::{AnthropicAuthConfig, resolve_api_key},
	client::Client,
	event::AssistantMessageEvent,
	message::{AssistantContent, StopReason},
	wire::{MessageContent, MessageCreateParams, MessageParam, Role},
};

fn env_or(name: &str, default: &str) -> String {
	std::env::var(name)
		.ok()
		.filter(|value| !value.is_empty())
		.unwrap_or_else(|| default.into())
}

fn params(model: String) -> MessageCreateParams {
	MessageCreateParams {
		model,
		messages: vec![MessageParam {
			role:    Role::User,
			content: MessageContent::Text("Reply with exactly: pong".into()),
		}],
		max_tokens: 512,
		system: None,
		temperature: None,
		top_p: None,
		top_k: None,
		stop_sequences: None,
		stream: None,
		tools: None,
		tool_choice: None,
		metadata: None,
		thinking: None,
		output_config: None,
		speed: None,
		context_management: None,
		fallbacks: None,
	}
}

fn first_text(content: &[AssistantContent]) -> Option<&str> {
	content.iter().find_map(|block| match block {
		AssistantContent::Text(text) => Some(text.text.as_str()),
		_ => None,
	})
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
	let base_url = env_or("OMP_AI_BASE_URL", "https://api.deepseek.com/anthropic");
	let model = env_or("OMP_AI_MODEL", "deepseek-v4-flash");
	let auth_entry = env_or("OMP_AI_AUTH_ENTRY", "deepseek");
	let api_key = resolve_api_key(None, Some(&auth_entry)).expect("resolve API key");
	let client = Client::new(AnthropicAuthConfig::new(api_key, Some(&base_url)), auth_entry.clone());
	let request = params(model.clone());

	// Path 1: raw wire body (stderr, for fixture backfill) + converted message.
	let raw = client
		.complete(&request)
		.await
		.expect("complete: wire call failed");
	eprintln!("--- raw wire body ---");
	eprintln!("{}", serde_json::to_string_pretty(&raw).expect("serialize raw body"));
	let message = client
		.complete_message(&request)
		.await
		.expect("complete_message failed");
	assert_eq!(message.stop_reason, StopReason::Stop, "expected stop, got {message:?}");
	let text = first_text(&message.content).expect("no text block in response");
	assert!(!text.trim().is_empty(), "empty text block");
	assert!(message.usage.input > 0 && message.usage.output > 0, "usage not populated");
	assert_eq!(message.api, "anthropic-messages");
	assert_eq!(message.provider, auth_entry);
	assert_eq!(message.model, model);
	assert!(message.response_id.is_some());
	println!("complete_message ok: text={text:?}");
	println!(
		"usage: input={} output={} total={}",
		message.usage.input, message.usage.output, message.usage.total_tokens
	);

	// Path 2: synthesized event stream over a second real call.
	let mut stream = client.stream(&request);
	let mut sequence = Vec::new();
	let mut terminal = None;
	while let Some(event) = stream.next().await {
		sequence.push(match &event {
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
		});
		if event.is_terminal() {
			terminal = event.terminal_message().cloned();
			break;
		}
	}
	println!("event sequence: {}", sequence.join(" -> "));
	assert_eq!(sequence.first().copied(), Some("start"), "first event must be start");
	assert_eq!(sequence.last().copied(), Some("done"), "terminal event must be done");
	let terminal = terminal.expect("terminal message missing");
	assert_eq!(terminal.stop_reason, StopReason::Stop);
	let stream_text = first_text(&terminal.content).expect("no text block in stream result");
	println!("stream ok: text={stream_text:?}");
	println!("L2 PASS: non-streaming completion + synthesized event stream against {base_url}");
}
