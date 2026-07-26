//! `AgentEvent` + the agent-side event stream container.
//!
//! [`AgentEvent`] is a 1:1 serde mirror of the TS `AgentEvent` union
//! (`packages/agent/src/types.ts:740-761`, 11 variants). The `type` tag is
//! `snake_case` and every field is `camelCase`, byte-identical to the TS
//! serialization so a future golden corpus can diff the two implementations.
//!
//! Two families of fields present on the TS union are intentionally omitted
//! here (they belong to WP-1.4a's defer list — see `lib.rs`):
//! - `agent_end.telemetry` / `agent_end.coverage` — telemetry rollup;
//! - `tool_execution_update` is defined for shape parity but **never emitted**
//!   by this WP (streaming `on_update` callbacks are deferred to WP-1.4b).
//!
//! [`AgentEventStream`] copies the pi-ai [`AssistantMessageEventStream`]
//! channel design (`crates/pi-ai/src/stream.rs`): an unbounded queue whose
//! terminal event is [`AgentEvent::AgentEnd`], with `result()` resolving to the
//! `AgentEnd.messages` payload (the run's new messages).

use std::sync::{Arc, OnceLock};

use pi_ai::{
	AssistantMessageEvent,
	message::{Message, ToolResultMessage},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

/// `AgentEvent` (types.ts:740-761). Emitted by the loop for UI/host consumers.
///
/// This WP emits 8 of the 11 variants; [`AgentEvent::ToolExecutionUpdate`] is
/// present for wire-shape parity but never produced (defer WP-1.4b).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", rename_all_fields = "camelCase")]
#[allow(
	clippy::large_enum_variant,
	reason = "variants mirror the TS AgentEvent union 1:1; boxing fields would obscure the wire \
	          shape and events are short-lived"
)]
pub enum AgentEvent {
	/// Agent lifecycle start.
	AgentStart,
	/// Terminal event: carries the run's accumulated new messages.
	AgentEnd { messages: Vec<Message> },
	/// A turn (one assistant response + its tool calls/results) begins.
	TurnStart,
	/// A turn ends, carrying the assistant message and any tool results.
	TurnEnd { message: Message, tool_results: Vec<ToolResultMessage> },
	/// A message (user / assistant / toolResult) is added to history.
	MessageStart { message: Message },
	/// Streaming update for the in-flight assistant message; embeds the raw
	/// provider [`AssistantMessageEvent`] that produced it.
	MessageUpdate {
		message:                 Message,
		assistant_message_event: AssistantMessageEvent,
	},
	/// A message is finalized.
	MessageEnd { message: Message },
	/// A tool invocation begins.
	ToolExecutionStart {
		tool_call_id: String,
		tool_name:    String,
		args:         Value,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		intent:       Option<String>,
	},
	/// Streaming partial result for an in-flight tool. **Never emitted in
	/// WP-1.4a** (shape parity only; defer WP-1.4b).
	ToolExecutionUpdate {
		tool_call_id:   String,
		tool_name:      String,
		args:           Value,
		partial_result: Value,
	},
	/// A tool invocation ends.
	ToolExecutionEnd {
		tool_call_id: String,
		tool_name:    String,
		result:       Value,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		is_error:     Option<bool>,
	},
}

impl AgentEvent {
	/// The terminal `agent_end` messages payload, if this is the terminal event.
	#[must_use]
	pub const fn terminal_messages(&self) -> Option<&Vec<Message>> {
		match self {
			Self::AgentEnd { messages } => Some(messages),
			_ => None,
		}
	}
}

type SharedResult = Arc<OnceLock<Vec<Message>>>;

/// Producer side of an [`AgentEventStream`]. Dropping it ends the stream.
pub struct AgentEventSink {
	tx:     mpsc::UnboundedSender<AgentEvent>,
	result: SharedResult,
}

impl AgentEventSink {
	/// Push an event. After the terminal `agent_end` has been pushed this is a
	/// no-op (mirrors pi-ai `EventSink::push`).
	pub fn push(&self, event: AgentEvent) {
		if self.result.get().is_some() {
			return;
		}
		if let Some(messages) = event.terminal_messages() {
			let _ = self.result.set(messages.clone());
		}
		let _ = self.tx.send(event);
	}
}

/// Consumer side: an async `AgentEvent` sequence plus a terminal-result
/// accessor resolving to the run's new messages.
pub struct AgentEventStream {
	rx:     mpsc::UnboundedReceiver<AgentEvent>,
	result: SharedResult,
}

impl AgentEventStream {
	/// Create a connected sink/stream pair.
	#[must_use]
	pub fn channel() -> (AgentEventSink, Self) {
		let (tx, rx) = mpsc::unbounded_channel();
		let result: SharedResult = Arc::default();
		(AgentEventSink { tx, result: Arc::clone(&result) }, Self { rx, result })
	}

	/// Next event, `None` once the stream is exhausted.
	pub async fn next(&mut self) -> Option<AgentEvent> {
		self.rx.recv().await
	}

	/// Drain remaining events and return the terminal `agent_end` messages.
	///
	/// Returns an empty `Vec` if the producer went away without a terminal
	/// event (a failed run that never reached `agent_end`).
	pub async fn result(mut self) -> Vec<Message> {
		while self.result.get().is_none() {
			if self.rx.recv().await.is_none() {
				break;
			}
		}
		self.result.get().cloned().unwrap_or_default()
	}
}
