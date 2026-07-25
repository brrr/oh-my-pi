//! Streaming state machine: raw wire events in, contract events out.
//!
//! Minimal-surface port of the SSE consumption loop in
//! `packages/ai/src/providers/anthropic.ts:2140-2500` plus
//! `finalizeStreamBlock` (:1987). Anomalies (duplicate/unopened indices, kind
//! mismatches, unknown block kinds) are skipped without failing the turn, like
//! the TS `reportAnthropicEnvelopeAnomaly` paths.
//!
//! Deliberately not ported in WP-1.1b (contract appendix B): thinking-envelope
//! unwrap, mid-stream throttled argument parsing (`arguments` stays `{}` until
//! the block closes; the raw partial JSON still flows via `toolcall_delta`),
//! JSON repair beyond the TS `__parseError`/`__rawJson` fallback shape,
//! server-side fallback model adoption (fallback blocks are ignored — the TS
//! behavior when the beta is not opted in), spliced-envelope replay, and cost
//! calculation.

use std::{collections::HashMap, sync::Arc};

use crate::{
	convert::{RequestMeta, convert_usage, map_stop_reason},
	event::{AssistantMessageEvent, DoneReason, ErrorReason},
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
	working:           AssistantMessage,
	open_blocks:       HashMap<u64, OpenBlock>,
	saw_message_start: bool,
	saw_terminal:      bool,
	saw_message_stop:  bool,
}

impl StreamingBuilder {
	/// Fresh builder around an empty message stamped with request metadata.
	/// The caller emits the leading `start` event itself (the TS provider
	/// pushes `start` before the first byte arrives).
	#[must_use]
	pub fn new(meta: &RequestMeta) -> Self {
		Self {
			working:           empty_message(meta),
			open_blocks:       HashMap::new(),
			saw_message_start: false,
			saw_terminal:      false,
			saw_message_stop:  false,
		}
	}

	/// Immutable snapshot of the accumulated message.
	#[must_use]
	pub fn snapshot(&self) -> Arc<AssistantMessage> {
		Arc::new(self.working.clone())
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
					return Vec::new(); // duplicate envelope — anomaly, skip
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
			RawMessageStreamEvent::ContentBlockStop { index } => self
				.open_blocks
				.remove(&index)
				.map_or_else(Vec::new, |open| self.finalize_block(&open)),
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
				let AssistantContent::Thinking(block) = &self.working.content[content_index] else {
					return Vec::new();
				};
				let content = block.thinking.clone();
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
					block.arguments = match serde_json::from_str(&final_json) {
						Ok(value) => value,
						// TS fallback shape when nothing was recovered
						// (anthropic.ts:2013-2022).
						Err(parse_error) => serde_json::json!({
							"__parseError": parse_error.to_string(),
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
