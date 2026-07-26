//! Event stream container — Rust counterpart of `AssistantMessageEventStream`
//! (`packages/ai/src/utils/event-stream.ts:147`).
//!
//! Semantics preserved from TS:
//! - a terminal event (`done`/`error`) settles the final result AND is still
//!   delivered to the iterator;
//! - pushes after the terminal event are silently dropped;
//! - `result()` resolves to the terminal [`AssistantMessage`] for both `done`
//!   and `error` (TS resolves the promise in both cases — an error turn is a
//!   value, not an exception).
//!
//! WP-1.6 段 2A (contract appendix B, 升定案 A6 真背压): the TS `EventStream`
//! keeps an **unbounded** `queue` array whose `push` never blocks the producer,
//! so a stalled consumer grows memory without bound. This container uses a
//! **bounded** channel ([`EVENT_CHANNEL_CAPACITY`] = 1024) instead: [`push`]
//! becomes `async` and suspends the producer once the queue is full, so a slow
//! consumer applies real backpressure to the SSE parse loop rather than
//! ballooning memory. The capacity is a Rust-specific value (TS has no explicit
//! bound); the terminal result is still recorded synchronously *before* the
//! send, so a consumer awaiting only [`result`] keeps draining and never
//! deadlocks against a full queue.
//!
//! [`push`]: EventSink::push
//! [`result`]: AssistantMessageEventStream::result

use std::sync::{Arc, OnceLock};

use tokio::sync::mpsc;

use crate::{AiError, event::AssistantMessageEvent, message::AssistantMessage};

type SharedResult = Arc<OnceLock<Arc<AssistantMessage>>>;

/// Bounded event-queue capacity (Rust-specific — the TS container is
/// unbounded).
///
/// On a full queue the producer's [`EventSink::push`] awaits until the consumer
/// drains, giving genuine backpressure without unbounded memory growth.
pub const EVENT_CHANNEL_CAPACITY: usize = 1024;

/// Producer side. Dropping the sink ends the stream (a stream that ends
/// without a terminal event makes `result()` return an error, mirroring the TS
/// "Stream ended without a final result" rejection).
pub struct EventSink {
	tx:     mpsc::Sender<AssistantMessageEvent>,
	result: SharedResult,
}

impl EventSink {
	/// Push an event, awaiting when the bounded queue is full (backpressure).
	/// After a terminal event has been pushed this is a no-op, matching
	/// `EventStream.push` (event-stream.ts:39). The terminal result is recorded
	/// before the send so a `result()`-only consumer still unblocks a full
	/// queue.
	pub async fn push(&self, event: AssistantMessageEvent) {
		if self.result.get().is_some() {
			return;
		}
		if let Some(message) = event.terminal_message() {
			let _ = self.result.set(Arc::clone(message));
		}
		let _ = self.tx.send(event).await;
	}
}

/// Consumer side: an async event sequence plus a terminal-result accessor.
pub struct AssistantMessageEventStream {
	rx:     mpsc::Receiver<AssistantMessageEvent>,
	result: SharedResult,
}

impl AssistantMessageEventStream {
	/// Create a connected sink/stream pair.
	#[must_use]
	pub fn channel() -> (EventSink, Self) {
		let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
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

#[cfg(test)]
mod tests {
	use std::{sync::Arc, time::Duration};

	use tokio::time::timeout;

	use super::{AssistantMessageEventStream, EVENT_CHANNEL_CAPACITY};
	use crate::{
		builder::StreamingBuilder, convert::RequestMeta, event::AssistantMessageEvent,
		message::AssistantMessage,
	};

	fn partial() -> Arc<AssistantMessage> {
		StreamingBuilder::new(&RequestMeta {
			api:       "anthropic-messages".into(),
			provider:  "deepseek".into(),
			model:     "m".into(),
			timestamp: 0,
			duration:  None,
		})
		.snapshot()
	}

	fn event(index: usize) -> AssistantMessageEvent {
		AssistantMessageEvent::TextStart { content_index: index, partial: partial() }
	}

	#[tokio::test]
	async fn push_applies_backpressure_when_full() {
		let (sink, mut stream) = AssistantMessageEventStream::channel();
		// Fill the bounded queue exactly to capacity (no consumer draining yet).
		for index in 0..EVENT_CHANNEL_CAPACITY {
			sink.push(event(index)).await;
		}
		// The next push must NOT resolve while the queue is full — real
		// backpressure, unlike the unbounded TS container.
		let mut blocked = Box::pin(sink.push(event(EVENT_CHANNEL_CAPACITY)));
		assert!(
			timeout(Duration::from_millis(50), &mut blocked)
				.await
				.is_err(),
			"push resolved despite a full queue — backpressure absent"
		);
		// Draining one event frees exactly one slot; the blocked push completes.
		assert!(stream.next().await.is_some());
		timeout(Duration::from_millis(500), &mut blocked)
			.await
			.expect("push must resolve once the consumer drains");
	}
}
