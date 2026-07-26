//! Streaming state machine: raw wire events in, contract events out.
//!
//! Minimal-surface port of the SSE consumption loop in
//! `packages/ai/src/providers/anthropic.ts:2140-2500` plus
//! `finalizeStreamBlock` (:1987). Anomalies (duplicate/unopened indices, kind
//! mismatches, unknown block kinds) are skipped without failing the turn, like
//! the TS `reportAnthropicEnvelopeAnomaly` paths.
//!
//! WP-1.6 段 2A (contract appendix B, 升定案 A4 + A5):
//! - **A4 JSON repair.** A tool-call block closing runs
//!   [`parse_json_with_repair`] (strict `serde_json` fast path → relaxed
//!   recovery); only when repair also fails does the `{__parseError,
//!   __rawJson}` fallback shape apply (TS `finalizeStreamBlock` toolCall path,
//!   anthropic.ts:1997-2024).
//! - **A5 thinking-envelope unwrap.** A `thinking` block closing strips any
//!   `<thinking>…</thinking>` envelope the model leaked into the block text and
//!   clears the (now-stale) signature (TS `unwrapAnthropicThinkingEnvelope`,
//!   anthropic.ts:1558-1566, applied at :1990-1995).
//!
//! Still not ported (contract appendix B): mid-stream throttled argument
//! parsing (`arguments` stays `{}` until the block closes; the raw partial JSON
//! still flows via `toolcall_delta` — F2 定案), Harmony-leak detection (A7,
//! `DeepSeek` has no Harmony — kept deferred), server-side fallback model
//! adoption (fallback blocks are ignored — the TS behavior when the beta is not
//! opted in), and cost calculation.
//!
//! WP-1.6 (contract appendix B,升定案 A3): spliced-envelope replay dedup — a
//! transparent reconnect can splice a second envelope (fresh `message_start`)
//! onto the same stream; blocks this stream already closed must not reopen and
//! duplicate their content. Ports the TS `sawSplicedEnvelope` +
//! `closedBlockIndexes` guard (anthropic.ts:2141-2203, :2399).

use std::{
	collections::{HashMap, HashSet},
	sync::Arc,
};

use crate::{
	convert::{RequestMeta, convert_usage, map_stop_reason},
	event::{AssistantMessageEvent, DoneReason, ErrorReason},
	json_repair::parse_json_with_repair,
	message::{
		AssistantContent, AssistantMessage, RedactedThinkingContent, StopReason, TextContent,
		ThinkingContent, ToolCall, Usage,
	},
	wire::{ContentBlockDelta, RawMessageStreamEvent, ResponseBlock, ResponseContentBlock},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
	Text,
	Thinking,
	RedactedThinking,
	ToolCall,
	Ignored,
}

struct OpenBlock {
	content_index: usize,
	kind:          BlockKind,
	partial_json:  String,
}

pub struct StreamingBuilder {
	working:              AssistantMessage,
	open_blocks:          HashMap<u64, OpenBlock>,
	saw_message_start:    bool,
	saw_terminal:         bool,
	saw_message_stop:     bool,
	/// A duplicate `message_start` was observed — a transparent reconnect
	/// spliced a second envelope onto this stream (TS `sawSplicedEnvelope`).
	saw_spliced_envelope: bool,
	/// Wire indexes already closed by a `content_block_stop`. Under a spliced
	/// envelope, a replayed `content_block_start` for one of these is dropped
	/// so the reconnect cannot duplicate finished content (TS
	/// `closedBlockIndexes`).
	closed_block_indexes: HashSet<u64>,
}

impl StreamingBuilder {
	/// Fresh builder around an empty message stamped with request metadata.
	/// The caller emits the leading `start` event itself (the TS provider
	/// pushes `start` before the first byte arrives).
	#[must_use]
	pub fn new(meta: &RequestMeta) -> Self {
		Self {
			working:              empty_message(meta),
			open_blocks:          HashMap::new(),
			saw_message_start:    false,
			saw_terminal:         false,
			saw_message_stop:     false,
			saw_spliced_envelope: false,
			closed_block_indexes: HashSet::new(),
		}
	}

	/// Immutable snapshot of the accumulated message.
	#[must_use]
	pub fn snapshot(&self) -> Arc<AssistantMessage> {
		Arc::new(self.working.clone())
	}

	/// Whether a `message_start` envelope has been observed. The retry driver
	/// keys "stream ended before content" retriability off this.
	#[must_use]
	pub const fn saw_message_start(&self) -> bool {
		self.saw_message_start
	}

	/// Record time-to-first-token once (later calls are no-ops).
	pub const fn set_ttft_once(&mut self, ttft_ms: u64) {
		if self.working.ttft.is_none() {
			self.working.ttft = Some(ttft_ms);
		}
	}

	/// Record the total request duration (call right before finishing).
	pub const fn set_duration(&mut self, duration_ms: u64) {
		self.working.duration = Some(duration_ms);
	}

	/// Apply one raw wire event; returns the contract events it produces.
	pub fn on_event(&mut self, event: RawMessageStreamEvent) -> Vec<AssistantMessageEvent> {
		match event {
			RawMessageStreamEvent::MessageStart { message } => {
				if self.saw_message_start {
					// A transparent reconnect spliced a second envelope onto the
					// stream: keep the original message, but arm the replay guard so
					// re-sent blocks for already-closed indexes are dropped
					// (anthropic.ts:2141-2149).
					self.saw_spliced_envelope = true;
					return Vec::new();
				}
				self.saw_message_start = true;
				self.working.response_id = Some(message.id);
				self.working.usage = convert_usage(&message.usage);
				Vec::new()
			},
			RawMessageStreamEvent::ContentBlockStart { index, content_block } => {
				self.on_block_start(index, &content_block)
			},
			RawMessageStreamEvent::ContentBlockDelta { index, delta } => {
				self.on_block_delta(index, delta)
			},
			RawMessageStreamEvent::ContentBlockStop { index } => {
				self
					.open_blocks
					.remove(&index)
					.map_or_else(Vec::new, |open| {
						// Record the close so a spliced replay of this index is dropped
						// (anthropic.ts:2399).
						self.closed_block_indexes.insert(index);
						self.finalize_block(&open)
					})
			},
			RawMessageStreamEvent::MessageDelta { delta, usage } => {
				if self.saw_terminal {
					return Vec::new();
				}
				if let Some(raw) = &delta.stop_reason {
					self.working.stop_reason = map_stop_reason(raw);
					self.saw_terminal = true;
					if self.working.stop_reason == StopReason::Error {
						self.working.stop_details.clone_from(&delta.stop_details);
						self.working.error_message = Some(error_stop_message(&delta));
					}
				}
				merge_delta_usage(&mut self.working.usage, &usage);
				Vec::new()
			},
			RawMessageStreamEvent::MessageStop => {
				self.saw_terminal = true;
				self.saw_message_stop = true;
				Vec::new()
			},
		}
	}

	fn on_block_start(&mut self, index: u64, block: &ResponseBlock) -> Vec<AssistantMessageEvent> {
		if self.saw_terminal || self.open_blocks.contains_key(&index) {
			return Vec::new();
		}
		if self.saw_spliced_envelope && self.closed_block_indexes.contains(&index) {
			// A spliced envelope is replaying an index this stream already closed;
			// consume its events silently so finished content is not duplicated
			// (anthropic.ts:2194-2203).
			self.open_blocks.insert(index, ignored_block());
			return Vec::new();
		}
		let known = match block {
			ResponseBlock::Known(known) => known,
			ResponseBlock::Unknown(_) => {
				self.open_blocks.insert(index, ignored_block());
				return Vec::new();
			},
		};
		let (content, kind, initial_delta) = match known {
			ResponseContentBlock::Text { text } => (
				AssistantContent::Text(TextContent {
					text:           text.clone(),
					text_signature: None,
				}),
				BlockKind::Text,
				(!text.is_empty()).then(|| text.clone()),
			),
			ResponseContentBlock::Thinking { thinking, signature } => (
				AssistantContent::Thinking(ThinkingContent {
					thinking:           thinking.clone(),
					thinking_signature: signature.clone().filter(|sig| !sig.is_empty()),
					item_id:            None,
				}),
				BlockKind::Thinking,
				(!thinking.is_empty()).then(|| thinking.clone()),
			),
			ResponseContentBlock::RedactedThinking { data } => (
				AssistantContent::RedactedThinking(RedactedThinkingContent { data: data.clone() }),
				BlockKind::RedactedThinking,
				None,
			),
			ResponseContentBlock::ToolUse { id, name, input } => (
				AssistantContent::ToolCall(ToolCall {
					id:                id.clone(),
					name:              name.clone(),
					arguments:         input.clone().unwrap_or_else(|| serde_json::json!({})),
					thought_signature: None,
					intent:            None,
					raw_block:         None,
					custom_wire_name:  None,
				}),
				BlockKind::ToolCall,
				None,
			),
			// Not opted into the server-side-fallback beta → drop the marker
			// entirely (TS unopted-in branch, anthropic.ts:2214-2222).
			ResponseContentBlock::Fallback { .. } => {
				self.open_blocks.insert(index, ignored_block());
				return Vec::new();
			},
		};
		self.working.content.push(content);
		let content_index = self.working.content.len() - 1;
		self.open_blocks.insert(index, OpenBlock {
			content_index,
			kind,
			partial_json: String::new(),
		});
		let mut events = Vec::new();
		let partial = self.snapshot();
		match kind {
			BlockKind::Text => {
				events.push(AssistantMessageEvent::TextStart {
					content_index,
					partial: Arc::clone(&partial),
				});
				if let Some(delta) = initial_delta {
					events.push(AssistantMessageEvent::TextDelta { content_index, delta, partial });
				}
			},
			BlockKind::Thinking => {
				events.push(AssistantMessageEvent::ThinkingStart {
					content_index,
					partial: Arc::clone(&partial),
				});
				if let Some(delta) = initial_delta {
					events.push(AssistantMessageEvent::ThinkingDelta { content_index, delta, partial });
				}
			},
			BlockKind::ToolCall => {
				events.push(AssistantMessageEvent::ToolcallStart { content_index, partial });
			},
			BlockKind::RedactedThinking | BlockKind::Ignored => {},
		}
		events
	}

	fn on_block_delta(
		&mut self,
		index: u64,
		delta: ContentBlockDelta,
	) -> Vec<AssistantMessageEvent> {
		if self.saw_terminal {
			return Vec::new();
		}
		let Some(open) = self.open_blocks.get_mut(&index) else {
			return Vec::new(); // unopened index — anomaly, skip
		};
		if open.kind == BlockKind::Ignored {
			return Vec::new();
		}
		let content_index = open.content_index;
		match (delta, &mut self.working.content[content_index]) {
			(ContentBlockDelta::TextDelta { text }, AssistantContent::Text(block)) => {
				block.text.push_str(&text);
				let partial = self.snapshot();
				vec![AssistantMessageEvent::TextDelta { content_index, delta: text, partial }]
			},
			(ContentBlockDelta::ThinkingDelta { thinking }, AssistantContent::Thinking(block)) => {
				block.thinking.push_str(&thinking);
				let partial = self.snapshot();
				vec![AssistantMessageEvent::ThinkingDelta { content_index, delta: thinking, partial }]
			},
			(ContentBlockDelta::SignatureDelta { signature }, AssistantContent::Thinking(block)) => {
				block
					.thinking_signature
					.get_or_insert_with(String::new)
					.push_str(&signature);
				Vec::new() // signature accumulates silently (TS parity)
			},
			(ContentBlockDelta::InputJsonDelta { partial_json }, AssistantContent::ToolCall(_)) => {
				open.partial_json.push_str(&partial_json);
				let partial = self.snapshot();
				vec![AssistantMessageEvent::ToolcallDelta {
					content_index,
					delta: partial_json,
					partial,
				}]
			},
			// Delta kind does not match the open block — anomaly, skip.
			_ => Vec::new(),
		}
	}

	/// `finalizeStreamBlock` (anthropic.ts:1987).
	fn finalize_block(&mut self, open: &OpenBlock) -> Vec<AssistantMessageEvent> {
		let content_index = open.content_index;
		match open.kind {
			BlockKind::Text => {
				let AssistantContent::Text(block) = &self.working.content[content_index] else {
					return Vec::new();
				};
				let content = block.text.clone();
				vec![AssistantMessageEvent::TextEnd {
					content_index,
					content,
					partial: self.snapshot(),
				}]
			},
			BlockKind::Thinking => {
				// A5: strip a leaked `<thinking>…</thinking>` envelope and drop the
				// now-stale signature before finalizing (anthropic.ts:1990-1995).
				let content = {
					let AssistantContent::Thinking(block) = &mut self.working.content[content_index]
					else {
						return Vec::new();
					};
					if let Some(unwrapped) = unwrap_thinking_envelope(&block.thinking) {
						block.thinking = unwrapped;
						block.thinking_signature = None;
					}
					block.thinking.clone()
				};
				vec![AssistantMessageEvent::ThinkingEnd {
					content_index,
					content,
					partial: self.snapshot(),
				}]
			},
			BlockKind::ToolCall => {
				let final_json = open.partial_json.clone();
				let AssistantContent::ToolCall(block) = &mut self.working.content[content_index] else {
					return Vec::new();
				};
				if !final_json.is_empty() {
					// A4: strict parse → relaxed repair; only a repair failure falls to
					// the TS `{__parseError, __rawJson}` shape (anthropic.ts:1997-2024).
					// The builder never runs a mid-stream throttled parse (F2), so no
					// partially-recovered arguments exist to preserve — the fallback is
					// reached exactly when repair yields nothing.
					block.arguments = match parse_json_with_repair(&final_json) {
						Ok(value) => value,
						Err(repair_error) => serde_json::json!({
							"__parseError": repair_error,
							"__rawJson": truncate_for_error(&final_json),
						}),
					};
				}
				let tool_call = block.clone();
				vec![AssistantMessageEvent::ToolcallEnd {
					content_index,
					tool_call,
					partial: self.snapshot(),
				}]
			},
			BlockKind::RedactedThinking | BlockKind::Ignored => Vec::new(),
		}
	}

	/// Finish the stream: close dangling blocks, then emit the terminal event.
	///
	/// A stream that ended before `message_start` becomes an `error` turn
	/// (TS `AnthropicStreamEnvelopeError`, anthropic.ts:2478).
	#[must_use]
	pub fn finish(mut self) -> (Arc<AssistantMessage>, Vec<AssistantMessageEvent>) {
		let mut events = Vec::new();
		let mut dangling: Vec<OpenBlock> = self.open_blocks.drain().map(|(_, open)| open).collect();
		dangling.sort_by_key(|open| open.content_index);
		for open in dangling {
			events.extend(self.finalize_block(&open));
		}
		if !self.saw_message_start {
			self.working.stop_reason = StopReason::Error;
			self.working.error_message = Some("stream ended before message_start".to_string());
		}
		let message = Arc::new(self.working);
		events.push(terminal_event(&message));
		(message, events)
	}

	/// Abort the stream with an out-of-band failure (in-stream `error` frame,
	/// transport drop, cancellation). Accumulated content stays on the message.
	#[must_use]
	pub fn fail(
		mut self,
		stop_reason: StopReason,
		error_message: String,
	) -> (Arc<AssistantMessage>, Vec<AssistantMessageEvent>) {
		self.working.stop_reason = stop_reason;
		self.working.error_message = Some(error_message);
		let message = Arc::new(self.working);
		(Arc::clone(&message), vec![terminal_event(&message)])
	}
}

const fn ignored_block() -> OpenBlock {
	OpenBlock {
		content_index: usize::MAX,
		kind:          BlockKind::Ignored,
		partial_json:  String::new(),
	}
}

fn empty_message(meta: &RequestMeta) -> AssistantMessage {
	AssistantMessage {
		content: Vec::new(),
		api: meta.api.clone(),
		provider: meta.provider.clone(),
		model: meta.model.clone(),
		context_snapshot: None,
		retry_recovery: None,
		response_id: None,
		upstream_provider: None,
		usage: Usage::default(),
		stop_reason: StopReason::Stop,
		stop_details: None,
		error_message: None,
		tool_call_abort_messages: None,
		error_status: None,
		error_id: None,
		disabled_features: None,
		provider_payload: None,
		timestamp: meta.timestamp,
		duration: meta.duration,
		ttft: None,
	}
}

fn terminal_event(message: &Arc<AssistantMessage>) -> AssistantMessageEvent {
	match message.stop_reason {
		StopReason::Stop => {
			AssistantMessageEvent::Done { reason: DoneReason::Stop, message: Arc::clone(message) }
		},
		StopReason::Length => {
			AssistantMessageEvent::Done { reason: DoneReason::Length, message: Arc::clone(message) }
		},
		StopReason::ToolUse => {
			AssistantMessageEvent::Done { reason: DoneReason::ToolUse, message: Arc::clone(message) }
		},
		StopReason::Error => {
			AssistantMessageEvent::Error { reason: ErrorReason::Error, error: Arc::clone(message) }
		},
		StopReason::Aborted => {
			AssistantMessageEvent::Error { reason: ErrorReason::Aborted, error: Arc::clone(message) }
		},
	}
}

/// Error-class `message_delta` → human message (anthropic.ts:2416-2436).
fn error_stop_message(delta: &crate::wire::MessageDelta) -> String {
	if let Some(details) = &delta.stop_details
		&& details.detail_type == "refusal"
	{
		let label = details
			.category
			.as_ref()
			.map_or_else(|| "Refusal".to_string(), |category| format!("Refusal ({category})"));
		return details
			.explanation
			.as_ref()
			.map(|explanation| explanation.trim())
			.filter(|explanation| !explanation.is_empty())
			.map_or_else(|| label.clone(), |explanation| format!("{label}: {explanation}"));
	}
	match &delta.stop_reason {
		Some(crate::wire::WireStopReason::Known(crate::wire::KnownStopReason::Refusal)) => {
			"Refusal (no details provided)".to_string()
		},
		Some(crate::wire::WireStopReason::Known(crate::wire::KnownStopReason::Sensitive)) => {
			"Content flagged by safety filters".to_string()
		},
		other => format!("Anthropic stream ended with stop_reason: {other:?}"),
	}
}

/// `message_delta.usage` merge (anthropic.ts:2437-2452): only fields present
/// overwrite, extras re-applied, total recomputed.
fn merge_delta_usage(usage: &mut Usage, delta: &crate::wire::WireUsage) {
	if let Some(input) = delta.input_tokens {
		usage.input = input;
	}
	if let Some(output) = delta.output_tokens {
		usage.output = output;
	}
	if let Some(cache_read) = delta.cache_read_input_tokens {
		usage.cache_read = cache_read;
	}
	if let Some(cache_write) = delta.cache_creation_input_tokens {
		usage.cache_write = cache_write;
	}
	let extras = convert_usage(delta);
	if extras.cttl.is_some() {
		usage.cttl = extras.cttl;
	}
	if extras.server.is_some() {
		usage.server = extras.server;
	}
	usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
}

fn truncate_for_error(json: &str) -> String {
	const MAX_LEN: usize = 512;
	if json.len() <= MAX_LEN {
		return json.to_string();
	}
	let mut cut = MAX_LEN;
	while !json.is_char_boundary(cut) {
		cut -= 1;
	}
	format!("{}… [truncated {} chars]", &json[..cut], json.len() - cut)
}

const THINKING_ENVELOPE_OPEN: &str = "<thinking>";
const THINKING_ENVELOPE_CLOSE: &str = "</thinking>";

/// Strip nested `<thinking>…</thinking>` envelopes a model may leak into a
/// thinking block's text, returning `Some(unwrapped)` only when at least one
/// layer was removed (`unwrapAnthropicThinkingEnvelope`, anthropic.ts:1558).
fn unwrap_thinking_envelope(text: &str) -> Option<String> {
	let mut current = text.trim().to_string();
	let mut stripped = false;
	while current.len() >= THINKING_ENVELOPE_OPEN.len() + THINKING_ENVELOPE_CLOSE.len()
		&& current.starts_with(THINKING_ENVELOPE_OPEN)
		&& current.ends_with(THINKING_ENVELOPE_CLOSE)
	{
		let inner =
			&current[THINKING_ENVELOPE_OPEN.len()..current.len() - THINKING_ENVELOPE_CLOSE.len()];
		current = inner.trim().to_string();
		stripped = true;
	}
	stripped.then_some(current)
}

#[cfg(test)]
mod tests {
	use super::{StreamingBuilder, unwrap_thinking_envelope};
	use crate::{
		convert::RequestMeta, event::AssistantMessageEvent, message::AssistantContent,
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

	fn wire(json: &str) -> RawMessageStreamEvent {
		serde_json::from_str(json).unwrap_or_else(|error| panic!("parse wire event {json}: {error}"))
	}

	const MESSAGE_START: &str = r#"{"type":"message_start","message":{"id":"msg-x","usage":{"input_tokens":1,"output_tokens":0}}}"#;

	// ---- A5: thinking-envelope unwrap ----------------------------------------

	#[test]
	fn unwrap_strips_single_envelope() {
		assert_eq!(unwrap_thinking_envelope("<thinking>hi</thinking>"), Some("hi".to_string()));
	}

	#[test]
	fn unwrap_strips_nested_envelopes_and_trims() {
		assert_eq!(
			unwrap_thinking_envelope("<thinking> <thinking> deep </thinking> </thinking>"),
			Some("deep".to_string())
		);
	}

	#[test]
	fn unwrap_returns_none_without_envelope() {
		assert_eq!(unwrap_thinking_envelope("plain reasoning"), None);
		// A lone open tag is not a wrapped envelope.
		assert_eq!(unwrap_thinking_envelope("<thinking>unterminated"), None);
	}

	#[test]
	fn thinking_end_unwraps_and_clears_signature() {
		let mut builder = StreamingBuilder::new(&meta());
		builder.on_event(wire(MESSAGE_START));
		builder.on_event(wire(
			r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"<thinking>real</thinking>","signature":"sig-abc"}}"#,
		));
		let events = builder.on_event(wire(r#"{"type":"content_block_stop","index":0}"#));
		let end = events
			.iter()
			.find(|event| matches!(event, AssistantMessageEvent::ThinkingEnd { .. }))
			.expect("thinking_end emitted");
		let AssistantMessageEvent::ThinkingEnd { content, .. } = end else {
			unreachable!()
		};
		assert_eq!(content, "real", "envelope stripped from thinking_end content");

		let (message, _) = builder.finish();
		match &message.content[0] {
			AssistantContent::Thinking(block) => {
				assert_eq!(block.thinking, "real");
				assert!(block.thinking_signature.is_none(), "stale signature cleared");
			},
			other => panic!("expected thinking block, got {other:?}"),
		}
	}

	// ---- A4: JSON repair at toolcall_end -------------------------------------

	fn drive_toolcall(partial_json: &str) -> serde_json::Value {
		let mut builder = StreamingBuilder::new(&meta());
		builder.on_event(wire(MESSAGE_START));
		builder.on_event(wire(
			r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call-1","name":"do_thing"}}"#,
		));
		let delta = serde_json::json!({
			"type": "content_block_delta",
			"index": 0,
			"delta": {"type": "input_json_delta", "partial_json": partial_json},
		});
		builder.on_event(serde_json::from_value(delta).unwrap());
		let events = builder.on_event(wire(r#"{"type":"content_block_stop","index":0}"#));
		events
			.iter()
			.find_map(|event| match event {
				AssistantMessageEvent::ToolcallEnd { tool_call, .. } => {
					Some(tool_call.arguments.clone())
				},
				_ => None,
			})
			.expect("toolcall_end emitted")
	}

	#[test]
	fn toolcall_end_repairs_malformed_json() {
		// Trailing comma + single quotes: strict parse fails, repair succeeds.
		let args = drive_toolcall("{'path': 'a.rs',}");
		assert_eq!(args, serde_json::json!({"path":"a.rs"}));
	}

	#[test]
	fn toolcall_end_clean_json_needs_no_repair() {
		let args = drive_toolcall(r#"{"path":"a.rs","n":3}"#);
		assert_eq!(args, serde_json::json!({"path":"a.rs","n":3}));
	}

	#[test]
	fn toolcall_end_unrepairable_json_falls_back() {
		// Truncated buffer: neither strict nor repair recovers → fallback shape.
		let args = drive_toolcall(r#"{"path": "untermin"#);
		assert!(args.get("__parseError").is_some(), "got {args:?}");
		assert_eq!(args["__rawJson"], serde_json::json!(r#"{"path": "untermin"#));
	}
}
