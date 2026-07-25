//! WP-1.1b L2 probe: real SSE streaming against an Anthropic-compatible
//! endpoint (`DeepSeek` by default) — progressive deltas on the happy path,
//! then a mid-stream cancellation on a second call.
//!
//! ```sh
//! cargo run -p pi-ai --example stream
//! # overrides: OMP_AI_BASE_URL / OMP_AI_MODEL / OMP_AI_AUTH_ENTRY / ANTHROPIC_API_KEY
//! ```

use pi_ai::{
	auth::{AnthropicAuthConfig, resolve_api_key},
	client::Client,
	event::{AssistantMessageEvent, ErrorReason},
	message::{AssistantContent, StopReason},
	wire::{MessageContent, MessageCreateParams, MessageParam, Role},
};
use tokio_util::sync::CancellationToken;

fn env_or(name: &str, default: &str) -> String {
	std::env::var(name)
		.ok()
		.filter(|value| !value.is_empty())
		.unwrap_or_else(|| default.into())
}

fn params(model: String, prompt: &str, max_tokens: u64) -> MessageCreateParams {
	MessageCreateParams {
		model,
		messages: vec![MessageParam {
			role:    Role::User,
			content: MessageContent::Text(prompt.into()),
		}],
		max_tokens,
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

#[tokio::main(flavor = "current_thread")]
async fn main() {
	let base_url = env_or("OMP_AI_BASE_URL", "https://api.deepseek.com/anthropic");
	let model = env_or("OMP_AI_MODEL", "deepseek-v4-flash");
	let auth_entry = env_or("OMP_AI_AUTH_ENTRY", "deepseek");
	let api_key = resolve_api_key(None, Some(&auth_entry)).expect("resolve API key");
	let client = Client::new(AnthropicAuthConfig::new(api_key, Some(&base_url)), auth_entry);

	// Path 1: full streaming turn with progressive deltas.
	let mut stream = client.stream(&params(model.clone(), "Reply with exactly: pong", 512));
	let mut counts: std::collections::BTreeMap<&'static str, u32> =
		std::collections::BTreeMap::new();
	let mut first = None;
	let mut terminal = None;
	while let Some(event) = stream.next().await {
		let kind = match &event {
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
		};
		first.get_or_insert(kind);
		*counts.entry(kind).or_default() += 1;
		if event.is_terminal() {
			terminal = event.terminal_message().cloned();
			break;
		}
	}
	println!("event counts: {counts:?}");
	assert_eq!(first, Some("start"), "first event must be start");
	assert_eq!(counts.get("done"), Some(&1), "terminal must be done, got {counts:?}");
	let delta_total = counts.get("text_delta").copied().unwrap_or(0)
		+ counts.get("thinking_delta").copied().unwrap_or(0);
	assert!(delta_total > 1, "expected progressive deltas, got {delta_total}");
	let message = terminal.expect("terminal message");
	assert_eq!(message.stop_reason, StopReason::Stop);
	let text = message
		.content
		.iter()
		.find_map(|block| match block {
			AssistantContent::Text(text) => Some(text.text.as_str()),
			_ => None,
		})
		.expect("text block");
	assert!(!text.trim().is_empty());
	assert!(message.ttft.is_some(), "ttft must be recorded");
	assert!(message.usage.output > 0, "usage must be merged from message_delta");
	println!(
		"stream ok: text={text:?} ttft={}ms duration={}ms usage.output={}",
		message.ttft.unwrap_or(0),
		message.duration.unwrap_or(0),
		message.usage.output
	);

	// Path 2: cancel mid-stream after the first delta arrives.
	let cancel = CancellationToken::new();
	let mut stream = client.stream_with_cancel(
		&params(model, "Write a 300-word story about a lighthouse.", 2048),
		cancel.clone(),
	);
	let mut saw_delta = false;
	let mut aborted = false;
	while let Some(event) = stream.next().await {
		match &event {
			AssistantMessageEvent::TextDelta { .. } | AssistantMessageEvent::ThinkingDelta { .. }
				if !saw_delta =>
			{
				saw_delta = true;
				cancel.cancel();
			},
			AssistantMessageEvent::Error { reason, error } => {
				assert_eq!(*reason, ErrorReason::Aborted, "expected aborted, got {error:?}");
				assert_eq!(error.stop_reason, StopReason::Aborted);
				aborted = true;
				break;
			},
			AssistantMessageEvent::Done { .. } => {
				panic!("stream completed before cancellation took effect");
			},
			_ => {},
		}
	}
	assert!(saw_delta && aborted, "cancellation path did not run");
	println!("cancel ok: aborted after first delta");
	println!("L2 PASS: SSE streaming + mid-stream cancellation against {base_url}");
}
