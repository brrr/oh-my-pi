//! WP-1.6 断流混沌 tests: the stream-hardening driver (`stream_runner`) under
//! transport drops, stalls, torn chunks, malformed frames, and spliced
//! reconnects. Each test drives [`drive_stream`] with a scripted transport so
//! the retry (A1), watchdog (A2), and spliced-dedup (A3) logic is exercised
//! without a live network. The `.sse` fixtures in `fixtures/sse_chaos_*.sse`
//! are the long-term regression assets these assertions pin.

use std::{collections::VecDeque, sync::Mutex, time::Duration};

use pi_ai::{
	AiError,
	convert::RequestMeta,
	event::{AssistantMessageEvent, ErrorReason},
	message::{AssistantContent, AssistantMessage, StopReason},
	stream::AssistantMessageEventStream,
	stream_runner::{
		ChunkStream, FIRST_EVENT_TIMEOUT_MSG, IDLE_TIMEOUT_MSG, StreamOpener, WatchdogConfig,
		drive_stream,
	},
};
use tokio_util::sync::CancellationToken;

// ---- scripted transport ----------------------------------------------------

/// One action a scripted stream performs on a `next_chunk()` call.
enum Step {
	/// Yield these bytes.
	Bytes(Vec<u8>),
	/// Sleep, then process the following step (models a stalled connection).
	Stall(Duration),
	/// Transport read error.
	Fail(String),
	/// Clean end of body (`Ok(None)`).
	End,
}

struct ScriptedStream {
	steps: VecDeque<Step>,
}

impl ChunkStream for ScriptedStream {
	async fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, String> {
		loop {
			match self.steps.pop_front() {
				Some(Step::Stall(delay)) => {
					tokio::time::sleep(delay).await;
				},
				Some(Step::Bytes(bytes)) => return Ok(Some(bytes)),
				Some(Step::Fail(message)) => return Err(message),
				Some(Step::End) | None => return Ok(None),
			}
		}
	}
}

/// One scripted `open()` outcome.
enum Attempt {
	Stream(Vec<Step>),
	OpenFail(AiError),
}

struct ScriptedOpener {
	attempts: Mutex<VecDeque<Attempt>>,
}

impl ScriptedOpener {
	fn single(steps: Vec<Step>) -> Self {
		Self { attempts: Mutex::new(VecDeque::from([Attempt::Stream(steps)])) }
	}

	fn sequence(attempts: Vec<Attempt>) -> Self {
		Self { attempts: Mutex::new(attempts.into()) }
	}
}

impl StreamOpener for ScriptedOpener {
	type Stream = ScriptedStream;

	async fn open(&self) -> Result<ScriptedStream, AiError> {
		let next = self
			.attempts
			.lock()
			.unwrap()
			.pop_front()
			.expect("ScriptedOpener: open() called more times than scripted (unexpected retry)");
		match next {
			Attempt::Stream(steps) => Ok(ScriptedStream { steps: steps.into() }),
			Attempt::OpenFail(error) => Err(error),
		}
	}
}

// ---- harness ---------------------------------------------------------------

fn meta() -> RequestMeta {
	RequestMeta {
		api:       "anthropic-messages".into(),
		provider:  "deepseek".into(),
		model:     "deepseek-v4-flash".into(),
		timestamp: 1_753_500_000_000,
		duration:  None,
	}
}

fn fixture(name: &str) -> String {
	std::fs::read_to_string(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR")))
		.unwrap_or_else(|error| panic!("read fixture {name}: {error}"))
}

/// A small, self-contained happy-path transcript used as a "retry succeeds"
/// tail.
fn good_transcript() -> String {
	r#"event: message_start
data: {"type":"message_start","message":{"id":"retry-ok","usage":{"input_tokens":3,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"pong"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

"#
	.to_string()
}

/// Deliver a whole transcript in one chunk, then close.
fn one_chunk(transcript: &str) -> Vec<Step> {
	vec![Step::Bytes(transcript.as_bytes().to_vec()), Step::End]
}

/// Run the driver to completion and collect every event (including terminal).
async fn drive(
	opener: ScriptedOpener,
	watchdog: WatchdogConfig,
	max_retries: u32,
) -> Vec<AssistantMessageEvent> {
	let cancel = CancellationToken::new();
	let (sink, mut stream) = AssistantMessageEventStream::channel();
	let meta = meta();
	drive_stream(&opener, &sink, &cancel, &meta, watchdog, max_retries).await;
	drop(sink);
	let mut events = Vec::new();
	while let Some(event) = stream.next().await {
		events.push(event);
	}
	events
}

const fn kind(event: &AssistantMessageEvent) -> &'static str {
	match event {
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
	}
}

fn terminal(events: &[AssistantMessageEvent]) -> &AssistantMessage {
	events
		.last()
		.and_then(AssistantMessageEvent::terminal_message)
		.expect("last event must be terminal")
}

fn text_of(message: &AssistantMessage, index: usize) -> &str {
	match &message.content[index] {
		AssistantContent::Text(block) => &block.text,
		other => panic!("content[{index}] is not text: {other:?}"),
	}
}

const fn watchdog_off() -> WatchdogConfig {
	WatchdogConfig::new(None, None)
}

// ---- A3: spliced-envelope dedup --------------------------------------------

#[tokio::test]
async fn spliced_message_start_replay_is_deduped() {
	let events = drive(
		ScriptedOpener::single(one_chunk(&fixture("sse_chaos_spliced_message_start.sse"))),
		watchdog_off(),
		0,
	)
	.await;
	let kinds: Vec<_> = events.iter().map(kind).collect();
	// Exactly one text block round: the spliced replay of index 0 is dropped.
	assert_eq!(kinds.iter().filter(|k| **k == "text_start").count(), 1, "{kinds:?}");
	assert_eq!(kinds.iter().filter(|k| **k == "text_end").count(), 1, "{kinds:?}");
	assert_eq!(kinds.last().copied(), Some("done"));
	let message = terminal(&events);
	assert_eq!(message.content.len(), 1, "replayed block must not duplicate content");
	assert_eq!(text_of(message, 0), "pong");
	// The original envelope's response id is retained, not the spliced one.
	assert_eq!(message.response_id.as_deref(), Some("chaos-splice-first"));
}

// ---- E1: malformed frame recovery ------------------------------------------

#[tokio::test]
async fn malformed_frame_is_skipped_and_stream_recovers() {
	let events = drive(
		ScriptedOpener::single(one_chunk(&fixture("sse_chaos_malformed_recover.sse"))),
		watchdog_off(),
		0,
	)
	.await;
	let message = terminal(&events);
	assert_eq!(message.stop_reason, StopReason::Stop);
	assert_eq!(message.content.len(), 1, "garbage frame must not create a block");
	assert_eq!(text_of(message, 0), "ok");
}

// ---- E1: transport drop before message_stop --------------------------------

#[tokio::test]
async fn drop_before_message_stop_finalizes_gracefully() {
	let events = drive(
		ScriptedOpener::single(one_chunk(&fixture("sse_chaos_drop_before_message_stop.sse"))),
		watchdog_off(),
		0,
	)
	.await;
	let message = terminal(&events);
	// message_delta already carried the stop reason, so the missing message_stop
	// degrades to a clean `done`.
	assert_eq!(kind(events.last().unwrap()), "done");
	assert_eq!(message.stop_reason, StopReason::Stop);
	assert_eq!(text_of(message, 0), "pong");
	assert_eq!(message.usage.output, 7);
}

// ---- E1: half-frame / chunk-boundary tearing -------------------------------

#[tokio::test]
async fn torn_chunk_boundaries_reassemble() {
	let transcript = fixture("sse_chaos_half_frame.sse");
	// One byte per chunk: every frame, line, and UTF-8 sequence is torn.
	let steps: Vec<Step> = transcript.bytes().map(|b| Step::Bytes(vec![b])).collect();
	let events = drive(ScriptedOpener::single(steps), watchdog_off(), 0).await;
	let message = terminal(&events);
	assert_eq!(message.stop_reason, StopReason::Stop);
	assert_eq!(text_of(message, 0), "the quick brown fox jumps over the lazy dog");
}

// ---- A1: retry before first content ----------------------------------------

#[tokio::test]
async fn stream_ended_before_message_start_errors_without_retry_budget() {
	let events = drive(
		ScriptedOpener::single(one_chunk(&fixture("sse_chaos_no_message_start.sse"))),
		watchdog_off(),
		0,
	)
	.await;
	let message = terminal(&events);
	assert_eq!(kind(events.last().unwrap()), "error");
	assert_eq!(message.stop_reason, StopReason::Error);
	assert_eq!(message.error_message.as_deref(), Some("stream ended before message_start"));
}

#[tokio::test]
async fn retry_recovers_after_pre_content_close() {
	// Attempt 1 ends before message_start (retriable); attempt 2 succeeds.
	let opener = ScriptedOpener::sequence(vec![
		Attempt::Stream(one_chunk(&fixture("sse_chaos_no_message_start.sse"))),
		Attempt::Stream(one_chunk(&good_transcript())),
	]);
	let events = drive(opener, watchdog_off(), 3).await;
	let kinds: Vec<_> = events.iter().map(kind).collect();
	assert_eq!(
		kinds.iter().filter(|k| **k == "start").count(),
		1,
		"start pushed once across retries"
	);
	assert_eq!(kinds.last().copied(), Some("done"));
	assert_eq!(text_of(terminal(&events), 0), "pong");
}

#[tokio::test]
async fn retry_recovers_after_transport_error_before_content() {
	// message_start arrived but no content block opened → still replay-safe.
	let mut attempt1 = one_chunk(
		r#"event: message_start
data: {"type":"message_start","message":{"id":"partial","usage":{"input_tokens":1,"output_tokens":0}}}

"#,
	);
	attempt1.pop(); // drop the trailing End
	attempt1.push(Step::Fail("connection reset".into()));
	let opener = ScriptedOpener::sequence(vec![
		Attempt::Stream(attempt1),
		Attempt::Stream(one_chunk(&good_transcript())),
	]);
	let events = drive(opener, watchdog_off(), 3).await;
	assert_eq!(kind(events.last().unwrap()), "done");
	assert_eq!(text_of(terminal(&events), 0), "pong");
}

#[tokio::test]
async fn transport_error_after_first_content_is_terminal_not_retried() {
	// A single scripted attempt: if the driver retried, open() would panic.
	let mut steps = one_chunk(
		r#"event: message_start
data: {"type":"message_start","message":{"id":"mid","usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"par"}}

"#,
	);
	steps.pop();
	steps.push(Step::Fail("connection reset".into()));
	let events = drive(ScriptedOpener::single(steps), watchdog_off(), 5).await;
	let message = terminal(&events);
	assert_eq!(kind(events.last().unwrap()), "error");
	assert_eq!(message.stop_reason, StopReason::Error);
	assert!(
		message
			.error_message
			.as_deref()
			.unwrap()
			.starts_with("Connection error while streaming"),
		"got {:?}",
		message.error_message
	);
	// Accumulated content survives the terminal error.
	assert_eq!(text_of(message, 0), "par");
}

#[tokio::test]
async fn head_open_failure_is_a_standard_error_turn_without_retry() {
	// The request never became a stream. A single scripted attempt with a
	// generous retry budget proves the driver does not retry the open path here
	// (the head-level retry policy lives inside the concrete opener).
	let opener = ScriptedOpener::sequence(vec![Attempt::OpenFail(AiError::ConnectionTimeout)]);
	let events = drive(opener, watchdog_off(), 5).await;
	let message = terminal(&events);
	assert_eq!(kind(events.last().unwrap()), "error");
	assert_eq!(message.stop_reason, StopReason::Error);
	assert!(message.error_message.is_some());
}

// ---- A2: dual watchdog -----------------------------------------------------

#[tokio::test]
async fn first_event_timeout_is_retriable() {
	// Attempt 1 stalls before the first frame (first-event watchdog); retry wins.
	let opener = ScriptedOpener::sequence(vec![
		Attempt::Stream(vec![Step::Stall(Duration::from_hours(1))]),
		Attempt::Stream(one_chunk(&good_transcript())),
	]);
	let watchdog =
		WatchdogConfig::new(Some(Duration::from_millis(20)), Some(Duration::from_hours(1)));
	let events = drive(opener, watchdog, 2).await;
	assert_eq!(kind(events.last().unwrap()), "done");
	assert_eq!(text_of(terminal(&events), 0), "pong");
}

#[tokio::test]
async fn first_event_timeout_terminal_when_budget_exhausted() {
	let opener = ScriptedOpener::single(vec![Step::Stall(Duration::from_hours(1))]);
	let watchdog =
		WatchdogConfig::new(Some(Duration::from_millis(20)), Some(Duration::from_hours(1)));
	let events = drive(opener, watchdog, 0).await;
	let message = terminal(&events);
	assert_eq!(kind(events.last().unwrap()), "error");
	assert_eq!(message.error_message.as_deref(), Some(FIRST_EVENT_TIMEOUT_MSG));
}

#[tokio::test]
async fn idle_timeout_is_terminal_and_never_retried() {
	// Frames arrive, then the connection stalls. A single scripted attempt plus a
	// generous retry budget proves the idle watchdog does NOT retry (open() would
	// panic on a second call).
	let mut steps = one_chunk(&fixture("sse_chaos_idle_gap.sse"));
	steps.pop(); // drop End
	steps.push(Step::Stall(Duration::from_hours(1)));
	let watchdog =
		WatchdogConfig::new(Some(Duration::from_hours(1)), Some(Duration::from_millis(20)));
	let events = drive(ScriptedOpener::single(steps), watchdog, 5).await;
	let message = terminal(&events);
	assert_eq!(kind(events.last().unwrap()), "error");
	assert_eq!(message.error_message.as_deref(), Some(IDLE_TIMEOUT_MSG));
	// Partial thinking content accumulated before the stall survives.
	match &message.content[0] {
		AssistantContent::Thinking(block) => assert_eq!(block.thinking, "hmm"),
		other => panic!("expected thinking block, got {other:?}"),
	}
}

// ---- D3/F1: cancellation under the hardened path ---------------------------

#[tokio::test]
async fn cancel_mid_stream_under_watchdog_emits_aborted() {
	let mut steps = one_chunk(&fixture("sse_chaos_idle_gap.sse"));
	steps.pop();
	steps.push(Step::Stall(Duration::from_hours(1)));
	let opener = ScriptedOpener::single(steps);
	let cancel = CancellationToken::new();
	let (sink, mut stream) = AssistantMessageEventStream::channel();
	let meta = meta();
	let watchdog = WatchdogConfig::new(Some(Duration::from_secs(30)), Some(Duration::from_secs(30)));
	let driver = {
		let cancel = cancel.clone();
		tokio::spawn(async move {
			drive_stream(&opener, &sink, &cancel, &meta, watchdog, 3).await;
		})
	};

	let mut aborted = false;
	let mut saw_delta = false;
	while let Some(event) = stream.next().await {
		match &event {
			AssistantMessageEvent::ThinkingDelta { .. } if !saw_delta => {
				saw_delta = true;
				cancel.cancel();
			},
			AssistantMessageEvent::Error { reason, error } => {
				assert_eq!(*reason, ErrorReason::Aborted);
				assert_eq!(error.stop_reason, StopReason::Aborted);
				// Accumulated thinking content is preserved on the aborted turn.
				match &error.content[0] {
					AssistantContent::Thinking(block) => assert_eq!(block.thinking, "hmm"),
					other => panic!("expected thinking block, got {other:?}"),
				}
				aborted = true;
				break;
			},
			AssistantMessageEvent::Done { .. } => panic!("stream completed before cancellation"),
			_ => {},
		}
	}
	driver.await.unwrap();
	assert!(
		saw_delta && aborted,
		"cancellation path did not run (saw_delta={saw_delta}, aborted={aborted})"
	);
}
