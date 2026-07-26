//! Stream-hardening driver: retry-before-first-content + dual idle watchdog.
//!
//! WP-1.6 (contract appendix B,升定案 A1 + A2). The WP-1.1b path opened the SSE
//! body once and turned any transport drop or stall into a terminal `error`
//! event. This module ports the two hardening layers the TS provider wraps its
//! SSE consumption in (`anthropic.ts:2028-2621`):
//!
//! - **A1 — retry before first content.** A transport failure (or a stream that
//!   ends before `message_start`, or a first-event watchdog timeout) *before
//!   any content event has been pushed downstream* re-opens a fresh request, up
//!   to `PROVIDER_MAX_RETRIES` (TS = 10) with capped-exponential backoff
//!   `min(0.5·2^n, 8s)·(1 − 25% jitter)` (`calculateAnthropicRetryDelayMs`,
//!   anthropic-client.ts:117). Once a content block has opened the turn is
//!   replay-unsafe (`firstTokenTime` set), so a later failure surfaces as a
//!   terminal `error` — never a silent retry that would duplicate content.
//! - **A2 — dual watchdog.** A *first-event* timeout guards the wait for the
//!   first SSE frame; an *inter-event idle* timeout guards each subsequent
//!   read. Both surface as `StreamTimeoutError`-shaped messages (TS strings
//!   verbatim). The first-event timeout is retriable (pre-content); the idle
//!   timeout is terminal and never retried (TS `isLocalIdleTimeout` bars
//!   provider retry).
//!
//! The driver is generic over a [`StreamOpener`]/[`ChunkStream`] pair so the
//! retry and watchdog logic is exercised by scripted transports in
//! `tests/sse_chaos.rs` (the断流混沌 fixtures) without a live network.

use std::{
	future::Future,
	time::{Duration, Instant},
};

use tokio_util::sync::CancellationToken;

use crate::{
	AiError,
	builder::StreamingBuilder,
	convert::{RequestMeta, emit_nonstream_events, error_to_message},
	event::AssistantMessageEvent,
	message::StopReason,
	sse::{SseParser, parse_message_event, stream_error_message},
	stream::EventSink,
	wire::RawMessageStreamEvent,
};

/// TS `PROVIDER_MAX_RETRIES` (anthropic.ts:1512): the streaming retry budget
/// the provider loop owns (the SDK's own per-request retry is pinned to 0 so
/// the two do not multiply).
pub const DEFAULT_MAX_STREAM_RETRIES: u32 = 10;

/// TS `DEFAULT_STREAM_FIRST_EVENT_TIMEOUT_MS` (idle-iterator.ts:5).
const DEFAULT_FIRST_EVENT_TIMEOUT: Duration = Duration::from_secs(100);
/// TS `DEFAULT_STREAM_IDLE_TIMEOUT_MS` (idle-iterator.ts:4).
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_mins(2);

/// TS `firstEventTimeoutAbortError` message (anthropic.ts:2032-2034).
pub const FIRST_EVENT_TIMEOUT_MSG: &str =
	"Anthropic stream timed out while waiting for the first event";
/// TS `idleTimeoutAbortError` message (anthropic.ts:2035-2037).
pub const IDLE_TIMEOUT_MSG: &str = "Anthropic stream stalled while waiting for the next event";

/// TS `INITIAL_RETRY_DELAY_S` / `MAX_RETRY_DELAY_S`
/// (anthropic-client.ts:36-37).
const INITIAL_RETRY_DELAY_SECS: f64 = 0.5;
const MAX_RETRY_DELAY_SECS: f64 = 8.0;

/// One connected byte-chunk source for a single stream attempt. The real
/// implementation wraps `reqwest::Response::chunk`; tests replay a scripted
/// transcript with injected stalls, tears, and drops.
pub trait ChunkStream: Send {
	/// Next byte chunk (`Ok(None)` = clean end of body), or a transport error
	/// message. Never called again after it yields `Ok(None)` or `Err`.
	fn next_chunk(&mut self) -> impl Future<Output = Result<Option<Vec<u8>>, String>> + Send;
}

/// Opens a fresh stream attempt. Called once per retry; head-level retries (the
/// non-streaming `open_with_retry` policy) live inside the concrete opener.
pub trait StreamOpener: Send + Sync {
	type Stream: ChunkStream;

	fn open(&self) -> impl Future<Output = Result<Self::Stream, AiError>> + Send;
}

/// Resolved watchdog deadlines. `None` on a field disables that guard (TS: env
/// knob set to `0`).
#[derive(Debug, Clone, Copy)]
pub struct WatchdogConfig {
	first_event: Option<Duration>,
	idle:        Option<Duration>,
}

impl WatchdogConfig {
	/// TS resolution (`getStreamIdleTimeoutMs` / `getStreamFirstEventTimeoutMs`,
	/// idle-iterator.ts:27-62): env override → default; the first-event budget
	/// is floored at the idle budget so a slow first token is never undercut.
	/// `PI_STREAM_IDLE_TIMEOUT_MS` (alias `PI_OPENAI_STREAM_IDLE_TIMEOUT_MS`)
	/// and `PI_STREAM_FIRST_EVENT_TIMEOUT_MS`; `0`/≤0 disables the guard.
	#[must_use]
	pub fn from_env() -> Self {
		let idle = resolve_timeout(
			env_first(&["PI_STREAM_IDLE_TIMEOUT_MS", "PI_OPENAI_STREAM_IDLE_TIMEOUT_MS"]),
			DEFAULT_IDLE_TIMEOUT,
		);
		let first_fallback =
			idle.map_or(DEFAULT_FIRST_EVENT_TIMEOUT, |idle| idle.max(DEFAULT_FIRST_EVENT_TIMEOUT));
		let first_event =
			resolve_timeout(env_first(&["PI_STREAM_FIRST_EVENT_TIMEOUT_MS"]), first_fallback);
		Self { first_event, idle }
	}

	/// Explicit deadlines for tests (`None` disables a guard).
	#[must_use]
	pub const fn new(first_event: Option<Duration>, idle: Option<Duration>) -> Self {
		Self { first_event, idle }
	}

	/// Deadline for the next read given whether a frame has already arrived.
	const fn deadline(&self, saw_first_frame: bool) -> Option<Duration> {
		if saw_first_frame {
			self.idle
		} else {
			self.first_event
		}
	}
}

fn env_first(names: &[&str]) -> Option<String> {
	names
		.iter()
		.find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
}

/// TS `normalizeIdleTimeoutMs` (idle-iterator.ts:9): non-finite → fallback,
/// ≤0 → disabled, else truncate to ms.
fn resolve_timeout(raw: Option<String>, fallback: Duration) -> Option<Duration> {
	let Some(raw) = raw else {
		return Some(fallback);
	};
	let Ok(parsed) = raw.trim().parse::<f64>() else {
		return Some(fallback);
	};
	if !parsed.is_finite() {
		return Some(fallback);
	}
	if parsed <= 0.0 {
		return None;
	}
	Some(Duration::from_millis(parsed.trunc() as u64))
}

/// TS `calculateAnthropicRetryDelayMs(attempt)` (anthropic-client.ts:117):
/// capped exponential backoff with 25% jitter. `attempt` is 0-based.
fn retry_delay(attempt: u32) -> Duration {
	let base =
		(INITIAL_RETRY_DELAY_SECS * f64::from(1u32 << attempt.min(8))).min(MAX_RETRY_DELAY_SECS);
	Duration::from_secs_f64(base * fastrand::f64().mul_add(-0.25, 1.0))
}

fn elapsed_ms(started: Instant) -> u64 {
	u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

async fn push_all(sink: &EventSink, events: Vec<AssistantMessageEvent>) {
	for event in events {
		sink.push(event).await;
	}
}

/// Push events, but let a cancellation preempt a producer suspended on the
/// bounded queue (A6 backpressure can block `push` when the consumer stalls;
/// the abort token must still win). Returns `false` when cancelled before all
/// events were delivered — the caller then finishes as an aborted turn.
async fn push_all_cancelable(
	sink: &EventSink,
	events: Vec<AssistantMessageEvent>,
	cancel: &CancellationToken,
) -> bool {
	for event in events {
		tokio::select! {
			biased;
			() = cancel.cancelled() => return false,
			() = sink.push(event) => {},
		}
	}
	true
}

/// How a single stream attempt ended.
enum AttemptEnd {
	/// Body closed cleanly (`Ok(None)`); `builder.finish()` decides the
	/// terminal.
	Closed,
	/// In-stream `error` frame — terminal `error` turn with this message.
	ErrorFrame(String),
	/// Caller cancelled mid-stream.
	Cancelled,
	/// Transport read failure.
	Transport(String),
	/// First-event watchdog fired (retriable pre-content).
	FirstEventTimeout,
	/// Inter-event idle watchdog fired (terminal, never retried).
	IdleTimeout,
}

/// Drive one streaming turn to completion, pushing contract events into `sink`.
///
/// Pushes the single leading `start` once (TS pushes before the first byte),
/// then loops: open → consume → on a retriable pre-content failure re-open with
/// backoff, else finish/fail. Head-open failure is a standard error turn (the
/// request never became a stream), matching the WP-1.1b behavior.
pub async fn drive_stream<O: StreamOpener>(
	opener: &O,
	sink: &EventSink,
	cancel: &CancellationToken,
	meta: &RequestMeta,
	watchdog: WatchdogConfig,
	max_retries: u32,
) {
	let started = Instant::now();
	// The leading `start` is pushed exactly once and survives retries (a
	// pre-content retry has emitted no content events, so this stays coherent).
	sink
		.push(AssistantMessageEvent::Start { partial: StreamingBuilder::new(meta).snapshot() })
		.await;

	let mut attempt: u32 = 0;
	loop {
		let mut stream = match opener.open().await {
			Ok(stream) => stream,
			Err(error) => {
				// The request never became a stream: standard error turn (no retry
				// here — the opener already applied the head-level retry policy).
				let message = std::sync::Arc::new(error_to_message(&error, meta));
				push_all(sink, emit_nonstream_events(&message)).await;
				return;
			},
		};

		let mut ctx = AttemptCtx::new(meta);
		let end = consume_attempt(&mut stream, &mut ctx, sink, cancel, watchdog, started).await;
		let AttemptCtx { mut builder, saw_first_content, .. } = ctx;

		match end {
			AttemptEnd::Closed => {
				// A clean end before `message_start` is an envelope error the TS loop
				// retries (up to the budget); after content it is a normal finish.
				if !builder.saw_message_start() && attempt < max_retries {
					attempt += 1;
					sleep_backoff(attempt - 1, cancel).await;
					continue;
				}
				builder.set_duration(elapsed_ms(started));
				push_all(sink, builder.finish().1).await;
				return;
			},
			AttemptEnd::Transport(message) => {
				if !saw_first_content && attempt < max_retries {
					attempt += 1;
					sleep_backoff(attempt - 1, cancel).await;
					continue;
				}
				builder.set_duration(elapsed_ms(started));
				let reason = format!("Connection error while streaming: {message}");
				push_all(sink, builder.fail(StopReason::Error, reason).1).await;
				return;
			},
			AttemptEnd::FirstEventTimeout => {
				if !saw_first_content && attempt < max_retries {
					attempt += 1;
					sleep_backoff(attempt - 1, cancel).await;
					continue;
				}
				builder.set_duration(elapsed_ms(started));
				push_all(
					sink,
					builder
						.fail(StopReason::Error, FIRST_EVENT_TIMEOUT_MSG.into())
						.1,
				)
				.await;
				return;
			},
			// Idle timeout is terminal — TS `isLocalIdleTimeout` bars retry so an
			// active-but-stalled stream fails loudly instead of looping.
			AttemptEnd::IdleTimeout => {
				builder.set_duration(elapsed_ms(started));
				push_all(sink, builder.fail(StopReason::Error, IDLE_TIMEOUT_MSG.into()).1).await;
				return;
			},
			AttemptEnd::ErrorFrame(message) => {
				builder.set_duration(elapsed_ms(started));
				push_all(sink, builder.fail(StopReason::Error, message).1).await;
				return;
			},
			AttemptEnd::Cancelled => {
				builder.set_duration(elapsed_ms(started));
				push_all(
					sink,
					builder
						.fail(StopReason::Aborted, AiError::Aborted.to_string())
						.1,
				)
				.await;
				return;
			},
		}
	}
}

/// Per-attempt mutable state: a fresh parser + builder + content flag, rebuilt
/// on each retry so a re-opened connection starts clean.
struct AttemptCtx {
	parser:            SseParser,
	builder:           StreamingBuilder,
	saw_first_content: bool,
}

impl AttemptCtx {
	fn new(meta: &RequestMeta) -> Self {
		Self {
			parser:            SseParser::new(),
			builder:           StreamingBuilder::new(meta),
			saw_first_content: false,
		}
	}
}

/// Sleep the backoff for `attempt` unless cancelled (a cancel during backoff
/// short-circuits to the next loop turn, which the caller resolves).
async fn sleep_backoff(attempt: u32, cancel: &CancellationToken) {
	tokio::select! {
		biased;
		() = cancel.cancelled() => {},
		() = tokio::time::sleep(retry_delay(attempt)) => {},
	}
}

/// Consume one attempt: read chunks under the watchdog, parse frames, feed the
/// builder, push events. Content events are only pushed after a block opens, so
/// a pre-content return value is guaranteed replay-safe for the retry loop.
async fn consume_attempt<S: ChunkStream>(
	stream: &mut S,
	ctx: &mut AttemptCtx,
	sink: &EventSink,
	cancel: &CancellationToken,
	watchdog: WatchdogConfig,
	started: Instant,
) -> AttemptEnd {
	let mut saw_first_frame = false;
	loop {
		let deadline = watchdog.deadline(saw_first_frame);
		let read = tokio::select! {
			biased;
			() = cancel.cancelled() => return AttemptEnd::Cancelled,
			read = read_chunk(stream, deadline) => read,
		};
		let bytes = match read {
			ReadResult::Chunk(Some(bytes)) => bytes,
			ReadResult::Chunk(None) => return AttemptEnd::Closed,
			ReadResult::Transport(message) => return AttemptEnd::Transport(message),
			ReadResult::TimedOut => {
				return if saw_first_frame {
					AttemptEnd::IdleTimeout
				} else {
					AttemptEnd::FirstEventTimeout
				};
			},
		};
		for frame in ctx.parser.push(&bytes) {
			saw_first_frame = true;
			if frame.event.as_deref() == Some("error") {
				return AttemptEnd::ErrorFrame(stream_error_message(&frame.data));
			}
			let Some(raw) = parse_message_event(&frame) else {
				continue;
			};
			if !ctx.saw_first_content && matches!(raw, RawMessageStreamEvent::ContentBlockStart { .. })
			{
				ctx.saw_first_content = true;
				ctx.builder.set_ttft_once(elapsed_ms(started));
			}
			if !push_all_cancelable(sink, ctx.builder.on_event(raw), cancel).await {
				return AttemptEnd::Cancelled;
			}
		}
	}
}

enum ReadResult {
	Chunk(Option<Vec<u8>>),
	Transport(String),
	TimedOut,
}

/// Read one chunk, optionally under a watchdog deadline. The deadline guards
/// the wait for *bytes* (a stalled network delivers none), which is the failure
/// mode the TS event-level watchdog exists to catch; a slow-but-live trickle is
/// not distinguished from steady progress (documented simplification vs the TS
/// per-event timer).
async fn read_chunk<S: ChunkStream>(stream: &mut S, deadline: Option<Duration>) -> ReadResult {
	let Some(deadline) = deadline else {
		return match stream.next_chunk().await {
			Ok(chunk) => ReadResult::Chunk(chunk),
			Err(message) => ReadResult::Transport(message),
		};
	};
	match tokio::time::timeout(deadline, stream.next_chunk()).await {
		Ok(Ok(chunk)) => ReadResult::Chunk(chunk),
		Ok(Err(message)) => ReadResult::Transport(message),
		Err(_elapsed) => ReadResult::TimedOut,
	}
}
