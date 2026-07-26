//! The agent-loop streaming state machine.
//!
//! Folded port of `agentLoop` / `runLoopBody` / `streamAssistantResponse`
//! (`packages/agent/src/agent-loop.ts`). The WP-1.4a face keeps the essential
//! control flow — drive the provider, branch on the five stop reasons, execute
//! tools serially, keep the `tool_use`/`tool_result` pairing — and folds away
//! everything the module doc's defer list names (steering / asides / follow-ups
//! / pause gate / soft-tool requirements / Harmony-leak retries / in-band
//! dialects / telemetry / deadline / hooks). The TS outer steering `while` is
//! folded to a single pass; an inner `max_steps` cap (default 64) guards
//! against a tool-loop that never yields.
//!
//! Rust `convertToLlm` is the identity: `AgentContext.messages` is already
//! `Vec<pi_ai::Message>`, so there is no host custom-message layer to
//! transform. The TS `convertToLlm` / `transformContext` /
//! `transformProviderContext` hooks live in the host layer and are deferred
//! here.

use std::sync::OnceLock;

use pi_ai::{
	AssistantMessage, AssistantMessageEvent,
	message::{Message, StopReason, ToolCall},
	normalize_anthropic_tool_schema,
	stream::AssistantMessageEventStream,
	wire::StopDetails,
};
use pi_shell::cancel::CancelToken;
use pi_tools::DynTool;
use regex::Regex;

use crate::{
	event::{AgentEvent, AgentEventSink, AgentEventStream},
	execute::{
		SyntheticReason, available_tool_names, create_aborted_tool_result, execute_tool_calls,
		now_millis, tool_calls_of,
	},
};

/// `STREAM_INTERRUPTED_AFTER_CONTENT_STOP_DETAIL` (agent-loop.ts:79).
pub const STREAM_INTERRUPTED_AFTER_CONTENT: &str = "stream_interrupted_after_content";

/// Default cap on consecutive tool-execution turns.
///
/// Not a TS constant — a WP-1.4a safety valve replacing the steering/deadline
/// machinery that bounds the TS loop, so a model that emits tool calls forever
/// cannot spin.
pub const DEFAULT_MAX_STEPS: usize = 64;

/// A tool spec as handed to the provider.
///
/// Name + description + the **normalized** `input_schema`. `pi-tools` returns
/// the raw (un-normalized) schema and the pi-ai client passes `input_schema`
/// through opaquely, so normalization happens exactly once, here, when the loop
/// assembles [`LlmContext`].
#[derive(Debug, Clone)]
pub struct WireTool {
	pub name:         String,
	pub description:  String,
	pub input_schema: serde_json::Value,
}

/// The LLM-facing context assembled per model call and handed to the
/// [`StreamFn`]. Rust counterpart of the TS `Context` at the provider boundary.
pub struct LlmContext {
	pub system_prompt: Vec<String>,
	pub messages:      Vec<Message>,
	pub tools:         Vec<WireTool>,
}

/// Provider-call injection point — TS `config.streamFn`.
///
/// Tests inject scripted event sequences; the real path (see
/// `provider::client_stream_fn`) wraps a pi-ai `Client`. Takes the
/// [`CancelToken`] so a real provider stream can be cancelled mid-flight via
/// `Client::stream_with_cancel` (a documented extension of the WP's literal
/// `Fn(Context)` signature — cancellation is otherwise unreachable from the
/// closure).
pub type StreamFn =
	Box<dyn Fn(LlmContext, CancelToken) -> AssistantMessageEventStream + Send + Sync>;

/// Loop input context. `convertToLlm` is identity, so `messages` is already the
/// provider message list; `tools` is the heterogeneous registry.
pub struct AgentContext {
	pub system_prompt: Vec<String>,
	pub messages:      Vec<Message>,
	pub tools:         Vec<Box<dyn DynTool>>,
}

/// Loop configuration. The provider knobs the loop threads into each request
/// plus the [`StreamFn`] injection point.
pub struct AgentConfig {
	pub model:       String,
	pub max_tokens:  u64,
	pub temperature: Option<f64>,
	/// Consecutive tool-turn cap (defaults to [`DEFAULT_MAX_STEPS`]).
	pub max_steps:   usize,
	/// Provider-call injection point. `None` fails every turn with an error
	/// message (a misconfiguration, surfaced rather than panicking).
	pub stream_fn:   Option<StreamFn>,
}

impl AgentConfig {
	/// A config with the default step cap and no `stream_fn`.
	#[must_use]
	pub fn new(model: impl Into<String>, max_tokens: u64) -> Self {
		Self {
			model: model.into(),
			max_tokens,
			temperature: None,
			max_steps: DEFAULT_MAX_STEPS,
			stream_fn: None,
		}
	}

	/// Set the provider-call injection point.
	#[must_use]
	pub fn with_stream_fn(mut self, stream_fn: StreamFn) -> Self {
		self.stream_fn = Some(stream_fn);
		self
	}
}

/// Start an agent loop with a new prompt. `agentLoop` (agent-loop.ts:336).
///
/// Pushes `agent_start` / `turn_start` / the prompt
/// `message_start`+`message_end` pairs, then runs the loop body on a spawned
/// task and returns the stream.
#[must_use]
pub fn agent_loop(
	prompts: Vec<Message>,
	context: AgentContext,
	config: AgentConfig,
	ct: CancelToken,
) -> AgentEventStream {
	let (sink, stream) = AgentEventStream::channel();
	tokio::spawn(async move {
		let mut new_messages: Vec<Message> = prompts.clone();
		let mut ctx = context;
		ctx.messages.extend(prompts.clone());

		sink.push(AgentEvent::AgentStart);
		sink.push(AgentEvent::TurnStart);
		for prompt in &prompts {
			sink.push(AgentEvent::MessageStart { message: prompt.clone() });
			sink.push(AgentEvent::MessageEnd { message: prompt.clone() });
		}

		run_loop_body(&mut ctx, &mut new_messages, &config, &ct, &sink).await;
	});
	stream
}

/// Continue an agent loop from existing context. `agentLoopContinue`
/// (agent-loop.ts:377). The last message must be a `user`/`toolResult` (not
/// `assistant`), else the provider rejects the request.
///
/// # Errors
///
/// Returns `Err` when the context is empty or ends on an assistant message.
pub fn agent_loop_continue(
	context: AgentContext,
	config: AgentConfig,
	ct: CancelToken,
) -> Result<AgentEventStream, String> {
	if context.messages.is_empty() {
		return Err("Cannot continue: no messages in context".to_owned());
	}
	if matches!(context.messages.last(), Some(Message::Assistant(_))) {
		return Err("Cannot continue from message role: assistant".to_owned());
	}
	let (sink, stream) = AgentEventStream::channel();
	tokio::spawn(async move {
		let mut new_messages: Vec<Message> = Vec::new();
		let mut ctx = context;
		sink.push(AgentEvent::AgentStart);
		sink.push(AgentEvent::TurnStart);
		run_loop_body(&mut ctx, &mut new_messages, &config, &ct, &sink).await;
	});
	Ok(stream)
}

/// The shared loop body. `runLoopBody` (agent-loop.ts:758-1192), folded.
async fn run_loop_body(
	ctx: &mut AgentContext,
	new_messages: &mut Vec<Message>,
	config: &AgentConfig,
	ct: &CancelToken,
	sink: &AgentEventSink,
) {
	let mut has_more = true;
	let mut first_turn = true;
	let mut steps = 0usize;

	while has_more {
		if steps >= config.max_steps {
			break;
		}
		steps += 1;

		if first_turn {
			first_turn = false;
		} else {
			sink.push(AgentEvent::TurnStart);
		}

		let message = stream_assistant_response(ctx, config, ct, sink).await;
		new_messages.push(Message::Assistant(Box::new(message.clone())));

		// error / aborted: pair residual tool calls with placeholders, end.
		if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
			let reason = if message.stop_reason == StopReason::Aborted {
				SyntheticReason::Aborted
			} else {
				SyntheticReason::Error
			};
			let mut tool_results = Vec::new();
			for call in &tool_calls_of(&message) {
				let result =
					create_aborted_tool_result(sink, call, reason, message.error_message.as_deref());
				ctx.messages.push(Message::ToolResult(result.clone()));
				new_messages.push(Message::ToolResult(result.clone()));
				tool_results.push(result);
			}
			sink.push(AgentEvent::TurnEnd {
				message: Message::Assistant(Box::new(message)),
				tool_results,
			});
			break;
		}

		let calls = tool_calls_of(&message);
		// `toolUse` and `stop` both continue when tool calls are present
		// (agent-loop.ts:1017): adaptive Opus emits tool calls under end_turn.
		let runnable = matches!(message.stop_reason, StopReason::ToolUse | StopReason::Stop);
		has_more = runnable && !calls.is_empty();

		let mut tool_results = Vec::new();
		if has_more {
			let results = execute_tool_calls(&ctx.tools, &message, ct, sink).await;
			for result in &results {
				ctx.messages.push(Message::ToolResult(result.clone()));
				new_messages.push(Message::ToolResult(result.clone()));
			}
			tool_results = results;
		} else if !calls.is_empty() {
			// Non-runnable stop (`length` truncation) left tool_use blocks:
			// pair each with a placeholder, do NOT execute (agent-loop.ts:1086).
			let reason = if message.stop_reason == StopReason::Length {
				SyntheticReason::Length
			} else {
				SyntheticReason::Skipped
			};
			for call in &calls {
				let result = create_aborted_tool_result(sink, call, reason, None);
				ctx.messages.push(Message::ToolResult(result.clone()));
				new_messages.push(Message::ToolResult(result.clone()));
				tool_results.push(result);
			}
			// A truncated turn with placeholders still continues so the model
			// can retry with smaller calls (agent-loop.ts:1102).
			if message.stop_reason == StopReason::Length && !tool_results.is_empty() {
				has_more = true;
			}
		}

		sink.push(AgentEvent::TurnEnd {
			message: Message::Assistant(Box::new(message)),
			tool_results,
		});
	}

	sink.push(AgentEvent::AgentEnd { messages: new_messages.clone() });
}

/// The `partial` snapshot carried by an incremental (non-terminal) event.
fn event_partial(event: &AssistantMessageEvent) -> Option<&AssistantMessage> {
	match event {
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
		| AssistantMessageEvent::ToolcallEnd { partial, .. } => Some(partial),
		AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => None,
	}
}

/// Stream one assistant response. `streamAssistantResponse`
/// (agent-loop.ts:1215), minimal face: build the LLM context, drive the
/// provider event stream, map its 13 variants onto agent events
/// (`start`→`message_start`, increments→ `message_update`,
/// terminal→`message_end`), and finalize via [`retain_completed_tool_calls`] +
/// [`recover_transient_error_tool_turn`]. An abort observed on `ct` mid-stream
/// yields an aborted message.
async fn stream_assistant_response(
	ctx: &mut AgentContext,
	config: &AgentConfig,
	ct: &CancelToken,
	sink: &AgentEventSink,
) -> AssistantMessage {
	let tools: Vec<WireTool> = ctx
		.tools
		.iter()
		.map(|tool| WireTool {
			name:         tool.name().to_owned(),
			description:  tool.description().to_owned(),
			input_schema: normalize_anthropic_tool_schema(&tool.input_schema()),
		})
		.collect();
	let tool_names = available_tool_names(&ctx.tools);

	let Some(stream_fn) = config.stream_fn.as_ref() else {
		let message = error_assistant_message(
			config,
			StopReason::Error,
			"no stream_fn configured on AgentConfig",
		);
		ctx.messages
			.push(Message::Assistant(Box::new(message.clone())));
		sink
			.push(AgentEvent::MessageStart { message: Message::Assistant(Box::new(message.clone())) });
		sink.push(AgentEvent::MessageEnd { message: Message::Assistant(Box::new(message.clone())) });
		return message;
	};

	let llm_context = LlmContext {
		system_prompt: ctx.system_prompt.clone(),
		messages: ctx.messages.clone(),
		tools,
	};
	let mut response = stream_fn(llm_context, ct.clone());

	let mut partial: Option<AssistantMessage> = None;
	let mut added_partial = false;
	let mut completed_tool_call_ids: std::collections::BTreeSet<String> =
		std::collections::BTreeSet::new();

	loop {
		let event = tokio::select! {
			biased;
			() = wait_abort(ct) => {
				return finish_aborted(partial, added_partial, &completed_tool_call_ids, ctx, config, sink, ct);
			},
			ev = response.next() => ev,
		};
		let Some(event) = event else {
			break;
		};

		match &event {
			AssistantMessageEvent::Done { message, .. }
			| AssistantMessageEvent::Error { error: message, .. } => {
				let mut final_message = (**message).clone();
				final_message = retain_completed_tool_calls(final_message, &completed_tool_call_ids);
				final_message = recover_transient_error_tool_turn(final_message, &tool_names);

				if added_partial {
					if let Some(last) = ctx.messages.last_mut() {
						*last = Message::Assistant(Box::new(final_message.clone()));
					}
				} else {
					ctx.messages
						.push(Message::Assistant(Box::new(final_message.clone())));
					sink.push(AgentEvent::MessageStart {
						message: Message::Assistant(Box::new(final_message.clone())),
					});
				}
				sink.push(AgentEvent::MessageEnd {
					message: Message::Assistant(Box::new(final_message.clone())),
				});
				return final_message;
			},
			AssistantMessageEvent::Start { partial: p } => {
				let snapshot = (**p).clone();
				partial = Some(snapshot.clone());
				if added_partial {
					if let Some(last) = ctx.messages.last_mut() {
						*last = Message::Assistant(Box::new(snapshot.clone()));
					}
					completed_tool_call_ids.clear();
					sink.push(AgentEvent::MessageUpdate {
						message:                 Message::Assistant(Box::new(snapshot)),
						assistant_message_event: event.clone(),
					});
				} else {
					ctx.messages
						.push(Message::Assistant(Box::new(snapshot.clone())));
					added_partial = true;
					sink.push(AgentEvent::MessageStart {
						message: Message::Assistant(Box::new(snapshot)),
					});
				}
			},
			_ => {
				if let AssistantMessageEvent::ToolcallEnd { tool_call, .. } = &event {
					completed_tool_call_ids.insert(tool_call.id.clone());
				}
				if let Some(p) = event_partial(&event) {
					let snapshot = p.clone();
					partial = Some(snapshot.clone());
					if added_partial && let Some(last) = ctx.messages.last_mut() {
						*last = Message::Assistant(Box::new(snapshot.clone()));
					}
					sink.push(AgentEvent::MessageUpdate {
						message:                 Message::Assistant(Box::new(snapshot)),
						assistant_message_event: event.clone(),
					});
				}
			},
		}
	}

	// Stream ended without a terminal event: settle on the last partial, or a
	// bare error message if nothing streamed.
	let final_message = partial.unwrap_or_else(|| {
		error_assistant_message(config, StopReason::Error, "stream ended without a final result")
	});
	if added_partial {
		if let Some(last) = ctx.messages.last_mut() {
			*last = Message::Assistant(Box::new(final_message.clone()));
		}
	} else {
		ctx.messages
			.push(Message::Assistant(Box::new(final_message.clone())));
		sink.push(AgentEvent::MessageStart {
			message: Message::Assistant(Box::new(final_message.clone())),
		});
	}
	sink.push(AgentEvent::MessageEnd {
		message: Message::Assistant(Box::new(final_message.clone())),
	});
	final_message
}

/// A future that resolves only when `ct` is aborted (pends forever otherwise).
async fn wait_abort(ct: &CancelToken) {
	ct.wait().await;
}

/// Build the aborted assistant message and commit it.
/// `emitAbortedAssistantMessage` (agent-loop.ts:1727), minimal.
fn finish_aborted(
	partial: Option<AssistantMessage>,
	added_partial: bool,
	completed_tool_call_ids: &std::collections::BTreeSet<String>,
	ctx: &mut AgentContext,
	config: &AgentConfig,
	sink: &AgentEventSink,
	_ct: &CancelToken,
) -> AssistantMessage {
	let base = partial.map_or_else(
		|| error_assistant_message(config, StopReason::Aborted, "Request was aborted"),
		|mut message| {
			message.stop_reason = StopReason::Aborted;
			message.error_message = Some("Request was aborted".to_owned());
			message
		},
	);
	let aborted = retain_completed_tool_calls(base, completed_tool_call_ids);
	if added_partial {
		if let Some(last) = ctx.messages.last_mut() {
			*last = Message::Assistant(Box::new(aborted.clone()));
		}
	} else {
		ctx.messages
			.push(Message::Assistant(Box::new(aborted.clone())));
		sink
			.push(AgentEvent::MessageStart { message: Message::Assistant(Box::new(aborted.clone())) });
	}
	sink.push(AgentEvent::MessageEnd { message: Message::Assistant(Box::new(aborted.clone())) });
	aborted
}

/// Construct a minimal error/aborted assistant message with the loop's model.
fn error_assistant_message(
	config: &AgentConfig,
	stop_reason: StopReason,
	error_message: &str,
) -> AssistantMessage {
	AssistantMessage {
		content: Vec::new(),
		api: String::new(),
		provider: String::new(),
		model: config.model.clone(),
		context_snapshot: None,
		retry_recovery: None,
		response_id: None,
		upstream_provider: None,
		usage: pi_ai::message::Usage::default(),
		stop_reason,
		stop_details: None,
		error_message: Some(error_message.to_owned()),
		tool_call_abort_messages: None,
		error_status: None,
		error_id: None,
		disabled_features: None,
		provider_payload: None,
		timestamp: now_millis(),
		duration: None,
		ttft: None,
	}
}

/// `retainCompletedToolCalls` (agent-loop.ts:1598).
///
/// On an error/aborted turn, drop `tool_call` blocks that never reached
/// `toolcall_end` (partial args are unsafe to keep) and tag `stopDetails` so
/// the truncation is recorded.
#[must_use]
pub fn retain_completed_tool_calls(
	mut message: AssistantMessage,
	completed_tool_call_ids: &std::collections::BTreeSet<String>,
) -> AssistantMessage {
	if !matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
		return message;
	}
	let mut dropped = false;
	message.content.retain(|block| {
		if let pi_ai::message::AssistantContent::ToolCall(call) = block {
			let keep = completed_tool_call_ids.contains(&call.id);
			if !keep {
				dropped = true;
			}
			keep
		} else {
			true
		}
	});
	if !dropped {
		return message;
	}
	message.stop_details = Some(interrupted_stop_details(
		message.stop_details.as_ref(),
		message.error_message.as_deref(),
	));
	message
}

/// `recoverTransientErrorToolTurn` (agent-loop.ts:1625).
///
/// An `error` turn whose tool calls are all known tools AND whose error text is
/// a transient stream-read/parse failure is promoted back to `toolUse` so the
/// completed calls are not lost. Refusal/sensitive turns are never recovered.
#[must_use]
pub fn recover_transient_error_tool_turn(
	mut message: AssistantMessage,
	tool_names: &std::collections::BTreeSet<String>,
) -> AssistantMessage {
	if message.stop_reason != StopReason::Error {
		return message;
	}
	let tool_calls: Vec<&ToolCall> = message
		.content
		.iter()
		.filter_map(|block| match block {
			pi_ai::message::AssistantContent::ToolCall(call) => Some(call),
			_ => None,
		})
		.collect();
	if tool_calls.is_empty() {
		return message;
	}
	let detail_type = message
		.stop_details
		.as_ref()
		.map(|d| d.detail_type.as_str());
	let detail_category = message
		.stop_details
		.as_ref()
		.and_then(|d| d.category.as_deref());
	if matches!(detail_type, Some("refusal" | "sensitive"))
		|| matches!(detail_category, Some("refusal" | "sensitive"))
	{
		return message;
	}
	if !tool_calls
		.iter()
		.all(|call| tool_names.contains(&call.name))
	{
		return message;
	}
	let error_text = message.error_message.clone().unwrap_or_default();
	let explanation = message
		.stop_details
		.as_ref()
		.and_then(|d| d.explanation.clone())
		.unwrap_or_default();
	let combined = format!("{error_text}\n{explanation}");
	let transient = is_stream_read_error(&combined)
		|| is_transient_stream_parse_error(&error_text)
		|| is_transient_stream_parse_error(&explanation);
	if !transient {
		return message;
	}
	let stop_details =
		interrupted_stop_details(message.stop_details.as_ref(), message.error_message.as_deref());
	message.stop_reason = StopReason::ToolUse;
	message.stop_details = Some(stop_details);
	message.error_message = None;
	message.error_id = None;
	message.error_status = None;
	message
}

/// The `stream_interrupted_after_content` stop-details marker, preserving an
/// existing marker or wrapping the prior details' type/explanation.
fn interrupted_stop_details(
	existing: Option<&StopDetails>,
	error_message: Option<&str>,
) -> StopDetails {
	if let Some(details) = existing
		&& details.detail_type == STREAM_INTERRUPTED_AFTER_CONTENT
	{
		return details.clone();
	}
	StopDetails {
		detail_type: STREAM_INTERRUPTED_AFTER_CONTENT.to_owned(),
		category:    existing.map(|d| d.detail_type.clone()),
		explanation: existing
			.and_then(|d| d.explanation.clone())
			.or_else(|| error_message.map(ToOwned::to_owned)),
	}
}

fn is_stream_read_error(text: &str) -> bool {
	static RE: OnceLock<Regex> = OnceLock::new();
	RE.get_or_init(|| Regex::new(r"(?i)stream[_ -]?read[_ -]?error").expect("valid regex"))
		.is_match(text)
}

/// `isTransientStreamParseError` for string diagnostics (flags.ts:508/520).
fn is_transient_stream_parse_error(text: &str) -> bool {
	static RE: OnceLock<Regex> = OnceLock::new();
	RE.get_or_init(|| {
		Regex::new(
			r"(?i)(?:json parse error:\s*(?:unterminated string|unexpected end of json input|unexpected end of data|unexpected eof|end of file|eof while parsing|truncated)|json\.parse:\s*(?:unterminated string|unexpected end of data)|unexpected end of json input|unexpected eof|eof while parsing)",
		)
		.expect("valid regex")
	})
	.is_match(text)
}
