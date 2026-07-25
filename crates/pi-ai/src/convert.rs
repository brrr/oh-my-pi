//! Wire → model conversions and the non-streaming event synthesizer.
//!
//! `map_stop_reason` mirrors `mapStopReason` (anthropic.ts:4303-4331) and
//! `convert_usage` mirrors the `message_start` usage handling plus
//! `applyAnthropicUsageExtras` (anthropic.ts:1590 / :2155-2163). The event
//! synthesizer's sequence spec lives in the contract
//! (`docs/omp-headless/provider-event-contract.md` §7).

use std::sync::Arc;

use crate::{
	AiError,
	event::{AssistantMessageEvent, DoneReason, ErrorReason},
	message::{
		AssistantContent, AssistantMessage, CttlUsage, FallbackContent, RedactedThinkingContent,
		ServerUsage, StopReason, TextContent, ThinkingContent, ToolCall, Usage,
	},
	wire::{
		KnownStopReason, ResponseBlock, ResponseContentBlock, ResponseMessage, WireStopReason,
		WireUsage,
	},
};

/// Static request context stamped onto every converted [`AssistantMessage`].
#[derive(Debug, Clone)]
pub struct RequestMeta {
	/// Wire family id, e.g. `"anthropic-messages"`.
	pub api:       String,
	/// Configured provider id, e.g. `"anthropic"` or `"deepseek"`.
	pub provider:  String,
	/// Requested model id (kept even when the response echoes a variant).
	pub model:     String,
	/// Request start, Unix milliseconds.
	pub timestamp: i64,
	/// Request duration in milliseconds, when measured.
	pub duration:  Option<u64>,
}

/// `mapStopReason` (anthropic.ts:4303-4331). Unknown reasons degrade to
/// `Stop` instead of failing a fully received turn.
#[must_use]
pub const fn map_stop_reason(reason: &WireStopReason) -> StopReason {
	match reason {
		WireStopReason::Known(known) => match known {
			KnownStopReason::EndTurn | KnownStopReason::PauseTurn | KnownStopReason::StopSequence => {
				StopReason::Stop
			},
			KnownStopReason::MaxTokens | KnownStopReason::ModelContextWindowExceeded => {
				StopReason::Length
			},
			KnownStopReason::ToolUse => StopReason::ToolUse,
			KnownStopReason::Refusal | KnownStopReason::Sensitive => StopReason::Error,
		},
		WireStopReason::Other(_) => StopReason::Stop,
	}
}

/// Wire usage → harness usage (anthropic.ts:2155-2163 +
/// `applyAnthropicUsageExtras` :1590).
///
/// Cost stays zero in WP-1.1a (pricing is a catalog concern); `iterations`
/// (server-side fallback accounting) is not consumed yet.
#[must_use]
pub fn convert_usage(wire: &WireUsage) -> Usage {
	let input = wire.input_tokens.unwrap_or(0);
	let output = wire.output_tokens.unwrap_or(0);
	let cache_read = wire.cache_read_input_tokens.unwrap_or(0);
	let cache_write = wire.cache_creation_input_tokens.unwrap_or(0);
	let cttl = wire.cache_creation.and_then(|cc| {
		let five = cc.ephemeral_5m_input_tokens.unwrap_or(0);
		let hour = cc.ephemeral_1h_input_tokens.unwrap_or(0);
		(five > 0 || hour > 0).then(|| CttlUsage {
			ephemeral_5m: (five > 0).then_some(five),
			ephemeral_1h: (hour > 0).then_some(hour),
		})
	});
	let server = wire.server_tool_use.and_then(|st| {
		let search = st.web_search_requests.unwrap_or(0);
		let fetch = st.web_fetch_requests.unwrap_or(0);
		(search > 0 || fetch > 0).then(|| ServerUsage {
			web_search: (search > 0).then_some(search),
			web_fetch:  (fetch > 0).then_some(fetch),
		})
	});
	Usage {
		input,
		output,
		cache_read,
		cache_write,
		total_tokens: input + output + cache_read + cache_write,
		cttl,
		server,
		..Usage::default()
	}
}

fn convert_block(block: &ResponseContentBlock) -> AssistantContent {
	match block {
		ResponseContentBlock::Text { text } => {
			AssistantContent::Text(TextContent { text: text.clone(), text_signature: None })
		},
		ResponseContentBlock::Thinking { thinking, signature } => {
			AssistantContent::Thinking(ThinkingContent {
				thinking:           thinking.clone(),
				thinking_signature: signature.clone(),
				item_id:            None,
			})
		},
		ResponseContentBlock::RedactedThinking { data } => {
			AssistantContent::RedactedThinking(RedactedThinkingContent { data: data.clone() })
		},
		ResponseContentBlock::ToolUse { id, name, input } => AssistantContent::ToolCall(ToolCall {
			id:                id.clone(),
			name:              name.clone(),
			arguments:         input.clone().unwrap_or_else(|| serde_json::json!({})),
			thought_signature: None,
			intent:            None,
			raw_block:         None,
			custom_wire_name:  None,
		}),
		ResponseContentBlock::Fallback { from, to } => {
			AssistantContent::Fallback(FallbackContent { from: from.clone(), to: to.clone() })
		},
	}
}

/// Convert a non-streaming 200 body into the harness message. Unknown content
/// blocks are dropped (the TS layer likewise only maps block kinds it knows).
#[must_use]
pub fn convert_response(response: &ResponseMessage, meta: &RequestMeta) -> AssistantMessage {
	let content = response
		.content
		.iter()
		.filter_map(|block| match block {
			ResponseBlock::Known(known) => Some(convert_block(known)),
			ResponseBlock::Unknown(_) => None,
		})
		.collect();
	AssistantMessage {
		content,
		api: meta.api.clone(),
		provider: meta.provider.clone(),
		model: meta.model.clone(),
		context_snapshot: None,
		retry_recovery: None,
		response_id: Some(response.id.clone()),
		upstream_provider: None,
		usage: convert_usage(&response.usage),
		stop_reason: response
			.stop_reason
			.as_ref()
			.map_or(StopReason::Stop, map_stop_reason),
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

/// Wrap a request failure into the error-shaped [`AssistantMessage`] every TS
/// provider produces in its catch block (`errorMessage` + `errorStatus`).
#[must_use]
pub fn error_to_message(error: &AiError, meta: &RequestMeta) -> AssistantMessage {
	let stop_reason = if matches!(error, AiError::Aborted) {
		StopReason::Aborted
	} else {
		StopReason::Error
	};
	let error_status = match error {
		AiError::Api { status, .. } => Some(*status),
		_ => None,
	};
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
		stop_reason,
		stop_details: None,
		error_message: Some(error.to_string()),
		tool_call_abort_messages: None,
		error_status,
		error_id: None,
		disabled_features: None,
		provider_payload: None,
		timestamp: meta.timestamp,
		duration: meta.duration,
		ttft: None,
	}
}

/// Synthesize the event sequence for an already-complete message.
///
/// Contract §7: `start`, then per content block its `*_start`/`*_delta`/
/// `*_end` triple (single full-payload delta), then the terminal event.
/// Redacted-thinking and fallback blocks emit no events but appear in
/// subsequent `partial` snapshots; every `partial` carries the final
/// usage/stop metadata.
#[must_use]
pub fn emit_nonstream_events(message: &Arc<AssistantMessage>) -> Vec<AssistantMessageEvent> {
	let done_reason = match message.stop_reason {
		StopReason::Stop => DoneReason::Stop,
		StopReason::Length => DoneReason::Length,
		StopReason::ToolUse => DoneReason::ToolUse,
		StopReason::Error => {
			return vec![AssistantMessageEvent::Error {
				reason: ErrorReason::Error,
				error:  Arc::clone(message),
			}];
		},
		StopReason::Aborted => {
			return vec![AssistantMessageEvent::Error {
				reason: ErrorReason::Aborted,
				error:  Arc::clone(message),
			}];
		},
	};

	let mut working = AssistantMessage { content: Vec::new(), ..(**message).clone() };
	let mut events = Vec::new();
	events.push(AssistantMessageEvent::Start { partial: Arc::new(working.clone()) });

	for (content_index, block) in message.content.iter().enumerate() {
		match block {
			AssistantContent::Text(text) => {
				working.content.push(AssistantContent::Text(TextContent {
					text:           String::new(),
					text_signature: None,
				}));
				events.push(AssistantMessageEvent::TextStart {
					content_index,
					partial: Arc::new(working.clone()),
				});
				working.content[content_index] = block.clone();
				let partial = Arc::new(working.clone());
				events.push(AssistantMessageEvent::TextDelta {
					content_index,
					delta: text.text.clone(),
					partial: Arc::clone(&partial),
				});
				events.push(AssistantMessageEvent::TextEnd {
					content_index,
					content: text.text.clone(),
					partial,
				});
			},
			AssistantContent::Thinking(thinking) => {
				working
					.content
					.push(AssistantContent::Thinking(ThinkingContent {
						thinking:           String::new(),
						thinking_signature: None,
						item_id:            None,
					}));
				events.push(AssistantMessageEvent::ThinkingStart {
					content_index,
					partial: Arc::new(working.clone()),
				});
				working.content[content_index] = block.clone();
				let partial = Arc::new(working.clone());
				events.push(AssistantMessageEvent::ThinkingDelta {
					content_index,
					delta: thinking.thinking.clone(),
					partial: Arc::clone(&partial),
				});
				events.push(AssistantMessageEvent::ThinkingEnd {
					content_index,
					content: thinking.thinking.clone(),
					partial,
				});
			},
			AssistantContent::ToolCall(call) => {
				working.content.push(AssistantContent::ToolCall(ToolCall {
					arguments: serde_json::json!({}),
					..call.clone()
				}));
				events.push(AssistantMessageEvent::ToolcallStart {
					content_index,
					partial: Arc::new(working.clone()),
				});
				working.content[content_index] = block.clone();
				let partial = Arc::new(working.clone());
				events.push(AssistantMessageEvent::ToolcallDelta {
					content_index,
					delta: call.arguments.to_string(),
					partial: Arc::clone(&partial),
				});
				events.push(AssistantMessageEvent::ToolcallEnd {
					content_index,
					tool_call: call.clone(),
					partial,
				});
			},
			AssistantContent::Image(image) => {
				working.content.push(block.clone());
				events.push(AssistantMessageEvent::ImageEnd {
					content_index,
					content: image.clone(),
					partial: Arc::new(working.clone()),
				});
			},
			AssistantContent::RedactedThinking(_) | AssistantContent::Fallback(_) => {
				working.content.push(block.clone());
			},
		}
	}

	events.push(AssistantMessageEvent::Done { reason: done_reason, message: Arc::clone(message) });
	events
}
