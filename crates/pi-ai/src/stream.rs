//! Event stream container — Rust counterpart of `AssistantMessageEventStream`
//! (`packages/ai/src/utils/event-stream.ts:147`).
//!
//! Semantics preserved from TS:
//! - unbounded queue: the producer never blocks on a slow consumer;
//! - a terminal event (`done`/`error`) settles the final result AND is still
//!   delivered to the iterator;
//! - pushes after the terminal event are silently dropped;
//! - `result()` resolves to the terminal [`AssistantMessage`] for both `done`
//!   and `error` (TS resolves the promise in both cases — an error turn is a
//!   value, not an exception).
//!
//! WP-1.1a only feeds this from the non-streaming synthesizer
//! ([`crate::convert::emit_nonstream_events`]); WP-1.1b swaps in true SSE
//! production behind the same surface.

use std::sync::{Arc, OnceLock};

use tokio::sync::mpsc;

use crate::{AiError, event::AssistantMessageEvent, message::AssistantMessage};

type SharedResult = Arc<OnceLock<Arc<AssistantMessage>>>;

/// Producer side. Dropping the sink ends the stream (a stream that ends
/// without a terminal event makes `result()` return an error, mirroring the TS
/// "Stream ended without a final result" rejection).
pub struct EventSink {
	tx:     mpsc::UnboundedSender<AssistantMessageEvent>,
	result: SharedResult,
}

impl EventSink {
	/// Push an event. After a terminal event has been pushed this is a no-op,
	/// matching `EventStream.push` (event-stream.ts:39).
	pub fn push(&self, event: AssistantMessageEvent) {
		if self.result.get().is_some() {
			return;
		}
		if let Some(message) = event.terminal_message() {
			let _ = self.result.set(Arc::clone(message));
		}
		let _ = self.tx.send(event);
	}
}

/// Consumer side: an async event sequence plus a terminal-result accessor.
pub struct AssistantMessageEventStream {
	rx:     mpsc::UnboundedReceiver<AssistantMessageEvent>,
	result: SharedResult,
}

impl AssistantMessageEventStream {
	/// Create a connected sink/stream pair.
	#[must_use]
	pub fn channel() -> (EventSink, Self) {
		let (tx, rx) = mpsc::unbounded_channel();
		let result: SharedResult = Arc::default();
		(EventSink { tx, result: Arc::clone(&result) }, Self { rx, result })
	}

	/// Next event, `None` once the stream is exhausted (terminal event
	/// consumed and sink dropped).
	pub async fn next(&mut self) -> Option<AssistantMessageEvent> {
		self.rx.recv().await
	}

	/// Drain remaining events and return the terminal message (`done.message`
	/// or `error.error`).
	///
	/// # Errors
	///
	/// [`AiError::StreamEnded`] when the producer went away without pushing a
	/// terminal event.
	pub async fn result(mut self) -> Result<Arc<AssistantMessage>, AiError> {
		while self.result.get().is_none() {
			if self.rx.recv().await.is_none() {
				break;
			}
		}
		self.result.get().cloned().ok_or(AiError::StreamEnded)
	}
}
