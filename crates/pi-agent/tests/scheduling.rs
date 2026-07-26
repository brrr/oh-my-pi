//! L1 tests for the WP-1.4b tool scheduler: shared/exclusive concurrency,
//! completion-order collection, panic isolation, abort propagation, and the
//! mid-batch steering interrupt.
//!
//! Most cases drive [`pi_agent::execute_tool_calls`] directly with
//! timeline-recording tools whose execution is gated by a
//! [`tokio::sync::Barrier`] (to force true overlap) or a
//! [`tokio::sync::Notify`] (to control completion order / abort timing) — a
//! serial scheduler physically cannot pass an N-party barrier, so the
//! concurrency assertions are deterministic, not timing-flaky.
//! The steering-injection cases drive the full [`pi_agent::agent_loop`].

use std::{
	sync::{
		Arc, Mutex,
		atomic::{AtomicUsize, Ordering},
	},
	time::Duration,
};

use pi_agent::{AgentConfig, AgentContext, AgentEvent, agent_loop, execute_tool_calls};
use pi_ai::{
	AssistantMessage, TextContent,
	convert::emit_nonstream_events,
	event::DoneReason,
	message::{AssistantContent, Message, StopReason, ToolCall, Usage, UserContent, UserMessage},
	stream::AssistantMessageEventStream,
};
use pi_shell::cancel::{AbortReason, CancelToken};
use pi_tools::{Concurrency, DynTool, Tool, ToolError, ToolResult};
use serde_json::{Value, json};
use tokio::sync::{Barrier, Notify};

// ─── Builders ────────────────────────────────────────────────────────────────

fn assistant(content: Vec<AssistantContent>, stop: StopReason) -> AssistantMessage {
	AssistantMessage {
		content,
		api: "test".into(),
		provider: "test".into(),
		model: "fake".into(),
		context_snapshot: None,
		retry_recovery: None,
		response_id: None,
		upstream_provider: None,
		usage: Usage::default(),
		stop_reason: stop,
		stop_details: None,
		error_message: None,
		tool_call_abort_messages: None,
		error_status: None,
		error_id: None,
		disabled_features: None,
		provider_payload: None,
		timestamp: 0,
		duration: None,
		ttft: None,
	}
}

fn tool_call(id: &str, name: &str) -> AssistantContent {
	AssistantContent::ToolCall(ToolCall {
		id:                id.into(),
		name:              name.into(),
		arguments:         json!({}),
		thought_signature: None,
		intent:            None,
		raw_block:         None,
		custom_wire_name:  None,
	})
}

/// An assistant message whose `content` is exactly the given tool calls.
fn tool_turn(calls: &[(&str, &str)]) -> AssistantMessage {
	assistant(calls.iter().map(|(id, name)| tool_call(id, name)).collect(), StopReason::ToolUse)
}

/// A `pi-agent` event sink whose stream is discarded (these tests assert on
/// tool timelines + returned results, not on the event feed).
fn sink() -> pi_agent::AgentEventSink {
	let (sink, _stream) = pi_agent::AgentEventStream::channel();
	sink
}

/// Behavior a [`TimedTool`] performs mid-execution.
#[derive(Clone, Default)]
enum Behavior {
	/// Return `Ok` immediately (completes on first poll).
	#[default]
	Instant,
	/// Rendezvous on the shared barrier, then return `Ok` (forces overlap).
	Barrier(Arc<Barrier>),
	/// Await the notify, then return `Ok` (completion-order control).
	Hold(Arc<Notify>),
	/// Await the run token, then return `Err` (cut off by abort; never completes
	/// normally).
	BlockUntilAbort,
	/// `panic!` on first poll.
	Panic,
	/// Return `Err(ToolError)` immediately (a thrown error, not an abort).
	Fail,
}

/// A tool that records `name:start` / `name:end` into a shared timeline and
/// whose mid-execution behavior + scheduling class are configurable.
struct TimedTool {
	name:        &'static str,
	concurrency: Concurrency,
	timeline:    Arc<Mutex<Vec<String>>>,
	behavior:    Behavior,
}

impl TimedTool {
	fn boxed(
		name: &'static str,
		concurrency: Concurrency,
		timeline: &Arc<Mutex<Vec<String>>>,
		behavior: Behavior,
	) -> Box<dyn DynTool> {
		Box::new(Self { name, concurrency, timeline: Arc::clone(timeline), behavior })
	}

	fn mark(&self, phase: &str) {
		self
			.timeline
			.lock()
			.unwrap()
			.push(format!("{}:{phase}", self.name));
	}
}

impl Tool for TimedTool {
	fn name(&self) -> &'static str {
		self.name
	}

	fn description(&self) -> &'static str {
		"timed"
	}

	fn input_schema(&self) -> Value {
		json!({ "type": "object", "properties": {} })
	}

	fn concurrency(&self) -> Concurrency {
		self.concurrency
	}

	async fn execute(
		&self,
		_id: &str,
		_args: Value,
		ct: &CancelToken,
	) -> Result<ToolResult, ToolError> {
		self.mark("start");
		match &self.behavior {
			Behavior::Instant => {},
			Behavior::Barrier(b) => {
				b.wait().await;
			},
			Behavior::Hold(n) => {
				n.notified().await;
			},
			Behavior::BlockUntilAbort => {
				ct.wait().await;
				self.mark("cut");
				return Err(ToolError::new(format!("{} interrupted", self.name)));
			},
			Behavior::Panic => panic!("{} exploded", self.name),
			Behavior::Fail => {
				self.mark("end");
				return Err(ToolError::new(format!("{} failed", self.name)));
			},
		}
		self.mark("end");
		Ok(ToolResult::text(format!("{} ran", self.name)))
	}
}

// ─── Timeline / result helpers ───────────────────────────────────────────────

fn timeline_of(t: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
	t.lock().unwrap().clone()
}

fn pos(timeline: &[String], token: &str) -> usize {
	timeline
		.iter()
		.position(|e| e == token)
		.unwrap_or_else(|| panic!("timeline missing {token}: {timeline:?}"))
}

/// L3 scheduling invariant: an exclusive tool's `[start,end]` window must not
/// be straddled by any other tool's token — the barrier is not穿越able.
fn assert_exclusive_isolated(timeline: &[String], exclusive: &str) {
	let start = pos(timeline, &format!("{exclusive}:start"));
	let end = pos(timeline, &format!("{exclusive}:end"));
	for (i, token) in timeline.iter().enumerate() {
		if i > start && i < end {
			assert!(
				token.starts_with(&format!("{exclusive}:")),
				"exclusive {exclusive} window straddled by {token}: {timeline:?}"
			);
		}
	}
}

fn result_names(results: &[pi_ai::message::ToolResultMessage]) -> Vec<String> {
	results.iter().map(|r| r.tool_name.clone()).collect()
}

fn result_text(r: &pi_ai::message::ToolResultMessage) -> String {
	let mut text = String::new();
	for block in &r.content {
		if let pi_ai::UserContentBlock::Text(t) = block {
			text.push_str(&t.text);
		}
	}
	text
}

/// L3: every emitted `tool_use` has exactly one paired `tool_result`.
fn assert_result_pairing(
	message: &AssistantMessage,
	results: &[pi_ai::message::ToolResultMessage],
) {
	let call_ids: Vec<&str> = message
		.content
		.iter()
		.filter_map(|c| match c {
			AssistantContent::ToolCall(c) => Some(c.id.as_str()),
			_ => None,
		})
		.collect();
	let result_ids: Vec<&str> = results.iter().map(|r| r.tool_call_id.as_str()).collect();
	assert_eq!(call_ids.len(), result_ids.len(), "tool_use/tool_result count mismatch");
	for id in &call_ids {
		assert!(result_ids.contains(id), "tool_use {id} has no tool_result: {result_ids:?}");
	}
}

// ─── 1. shared parallelism truly overlaps ────────────────────────────────────

#[tokio::test]
async fn shared_tools_run_concurrently() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	let barrier = Arc::new(Barrier::new(2));
	let tools = vec![
		TimedTool::boxed("a", Concurrency::Shared, &timeline, Behavior::Barrier(barrier.clone())),
		TimedTool::boxed("b", Concurrency::Shared, &timeline, Behavior::Barrier(barrier.clone())),
	];
	let message = tool_turn(&[("c1", "a"), ("c2", "b")]);
	let ct = CancelToken::default();

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), None).await;

	let tl = timeline_of(&timeline);
	// A 2-party barrier only releases if BOTH tools reached it → both started
	// before either finished. A serial scheduler would deadlock here.
	assert!(pos(&tl, "a:start") < pos(&tl, "a:end"));
	assert!(pos(&tl, "b:start") < pos(&tl, "b:end"));
	assert!(pos(&tl, "a:start") < pos(&tl, "b:end"), "a started after b ended → not concurrent");
	assert!(pos(&tl, "b:start") < pos(&tl, "a:end"), "b started after a ended → not concurrent");
	assert_result_pairing(&message, &results);
}

// ─── 2. exclusive is a barrier after preceding shared ────────────────────────

#[tokio::test]
async fn exclusive_waits_for_preceding_shared() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	let barrier = Arc::new(Barrier::new(2));
	let tools = vec![
		TimedTool::boxed("a", Concurrency::Shared, &timeline, Behavior::Barrier(barrier.clone())),
		TimedTool::boxed("b", Concurrency::Shared, &timeline, Behavior::Barrier(barrier.clone())),
		TimedTool::boxed("x", Concurrency::Exclusive, &timeline, Behavior::Instant),
	];
	let message = tool_turn(&[("c1", "a"), ("c2", "b"), ("c3", "x")]);
	let ct = CancelToken::default();

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), None).await;

	let tl = timeline_of(&timeline);
	// The exclusive tool starts only after BOTH shared tools ended.
	assert!(pos(&tl, "x:start") > pos(&tl, "a:end"), "exclusive ran before a finished");
	assert!(pos(&tl, "x:start") > pos(&tl, "b:end"), "exclusive ran before b finished");
	assert_exclusive_isolated(&tl, "x");
	assert_result_pairing(&message, &results);
}

// ─── 3. shared after an exclusive waits for it ───────────────────────────────

#[tokio::test]
async fn shared_after_exclusive_waits_for_it() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	let barrier = Arc::new(Barrier::new(2));
	let tools = vec![
		TimedTool::boxed("x", Concurrency::Exclusive, &timeline, Behavior::Instant),
		TimedTool::boxed("a", Concurrency::Shared, &timeline, Behavior::Barrier(barrier.clone())),
		TimedTool::boxed("b", Concurrency::Shared, &timeline, Behavior::Barrier(barrier.clone())),
	];
	let message = tool_turn(&[("c1", "x"), ("c2", "a"), ("c3", "b")]);
	let ct = CancelToken::default();

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), None).await;

	let tl = timeline_of(&timeline);
	assert!(pos(&tl, "a:start") > pos(&tl, "x:end"), "shared a ran before exclusive finished");
	assert!(pos(&tl, "b:start") > pos(&tl, "x:end"), "shared b ran before exclusive finished");
	// a & b still overlap each other (2-party barrier released).
	assert!(pos(&tl, "a:start") < pos(&tl, "b:end"));
	assert!(pos(&tl, "b:start") < pos(&tl, "a:end"));
	assert_exclusive_isolated(&tl, "x");
	assert_result_pairing(&message, &results);
}

// ─── 4. completion-order collection ≠ initiation order ───────────────────────

#[tokio::test]
async fn results_collected_in_completion_order() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	let release_slow = Arc::new(Notify::new());
	// Content order is [slow, fast]; `slow` holds until released, `fast`
	// completes immediately → completion order is [fast, slow].
	let tools = vec![
		TimedTool::boxed(
			"slow",
			Concurrency::Shared,
			&timeline,
			Behavior::Hold(release_slow.clone()),
		),
		TimedTool::boxed("fast", Concurrency::Shared, &timeline, Behavior::Instant),
	];
	let message = tool_turn(&[("c1", "slow"), ("c2", "fast")]);
	let ct = CancelToken::default();

	// Release `slow` once `fast` has ended (poll the timeline from a sibling task).
	let watcher = {
		let timeline = Arc::clone(&timeline);
		let release_slow = Arc::clone(&release_slow);
		tokio::spawn(async move {
			loop {
				if timeline_of(&timeline).iter().any(|e| e == "fast:end") {
					release_slow.notify_one();
					break;
				}
				tokio::task::yield_now().await;
			}
		})
	};

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), None).await;
	watcher.await.unwrap();

	// Returned Vec (which the loop backfills into ctx verbatim) is completion
	// order.
	assert_eq!(result_names(&results), vec!["fast", "slow"], "results must be completion order");
	// Content order was [slow, fast]; proving the two differ.
	assert_eq!(message.content.len(), 2);
	assert_result_pairing(&message, &results);
}

// ─── 5. a single panicking tool does not sink the batch ──────────────────────

#[tokio::test]
async fn tool_panic_does_not_sink_batch() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	let tools = vec![
		TimedTool::boxed("boom", Concurrency::Shared, &timeline, Behavior::Panic),
		TimedTool::boxed("ok", Concurrency::Shared, &timeline, Behavior::Instant),
	];
	let message = tool_turn(&[("c1", "boom"), ("c2", "ok")]);
	let ct = CancelToken::default();

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), None).await;

	assert_result_pairing(&message, &results);
	let boom = results.iter().find(|r| r.tool_name == "boom").unwrap();
	assert!(boom.is_error, "panicking tool must yield an is_error result");
	assert!(result_text(boom).contains("panicked"), "panic surfaced: {}", result_text(boom));
	let ok = results.iter().find(|r| r.tool_name == "ok").unwrap();
	assert!(!ok.is_error, "sibling tool survived the panic");
}

// ─── 6. abort mid-batch: completed kept, cut-off reports aborted ─────────────

#[tokio::test]
async fn abort_keeps_completed_and_cuts_off_inflight() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	let tools = vec![
		TimedTool::boxed("done", Concurrency::Shared, &timeline, Behavior::Instant),
		TimedTool::boxed("hung", Concurrency::Shared, &timeline, Behavior::BlockUntilAbort),
	];
	let message = tool_turn(&[("c1", "done"), ("c2", "hung")]);
	let mut ct = CancelToken::default();
	let tok = ct.emplace_abort_token();

	// Abort once `done` has completed but `hung` is still blocked.
	let watcher = {
		let timeline = Arc::clone(&timeline);
		tokio::spawn(async move {
			loop {
				if timeline_of(&timeline).iter().any(|e| e == "done:end") {
					tok.abort(AbortReason::User);
					break;
				}
				tokio::task::yield_now().await;
			}
		})
	};

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), None).await;
	watcher.await.unwrap();

	assert_result_pairing(&message, &results);
	let done = results.iter().find(|r| r.tool_name == "done").unwrap();
	assert!(!done.is_error, "completed tool keeps its real result");
	assert_eq!(result_text(done), "done ran");
	let hung = results.iter().find(|r| r.tool_name == "hung").unwrap();
	assert!(hung.is_error, "cut-off tool reports an error");
	assert!(result_text(hung).contains("aborted"), "cut-off reports aborted: {}", result_text(hung));
}

// ─── 7. mid-batch steering interrupt before an exclusive barrier ─────────────

#[tokio::test]
async fn steering_interrupts_before_exclusive() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	let tools = vec![
		TimedTool::boxed("a", Concurrency::Shared, &timeline, Behavior::Instant),
		TimedTool::boxed("x", Concurrency::Exclusive, &timeline, Behavior::Instant),
		TimedTool::boxed("b", Concurrency::Shared, &timeline, Behavior::Instant),
	];
	let message = tool_turn(&[("c1", "a"), ("c2", "x"), ("c3", "b")]);
	let ct = CancelToken::default();
	let steering: &(dyn Fn() -> bool + Send + Sync) = &|| true;

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), Some(steering)).await;

	// The shared prefix `a` ran; the exclusive `x` and everything after are
	// skipped (paired, not executed).
	let tl = timeline_of(&timeline);
	assert!(tl.iter().any(|e| e == "a:end"), "shared prefix should complete: {tl:?}");
	assert!(!tl.iter().any(|e| e == "x:start"), "exclusive must not start after steer: {tl:?}");
	assert!(!tl.iter().any(|e| e == "b:start"), "trailing shared must not start: {tl:?}");

	assert_result_pairing(&message, &results);
	let a = results.iter().find(|r| r.tool_name == "a").unwrap();
	assert!(!a.is_error);
	for skipped in results
		.iter()
		.filter(|r| r.tool_name == "x" || r.tool_name == "b")
	{
		assert!(skipped.is_error, "{} must be skipped", skipped.tool_name);
		assert!(result_text(skipped).contains("Skipped due to queued"), "{}", result_text(skipped));
	}
}

// ─── 8. a queued steer does NOT interrupt a shared-only batch ────────────────

#[tokio::test]
async fn steering_does_not_interrupt_shared_only_batch() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	let barrier = Arc::new(Barrier::new(2));
	let tools = vec![
		TimedTool::boxed("a", Concurrency::Shared, &timeline, Behavior::Barrier(barrier.clone())),
		TimedTool::boxed("b", Concurrency::Shared, &timeline, Behavior::Barrier(barrier.clone())),
	];
	let message = tool_turn(&[("c1", "a"), ("c2", "b")]);
	let ct = CancelToken::default();
	let steering: &(dyn Fn() -> bool + Send + Sync) = &|| true;

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), Some(steering)).await;

	// No exclusive barrier ⇒ no interruption point; both shared tools run.
	assert_result_pairing(&message, &results);
	assert!(results.iter().all(|r| !r.is_error), "shared-only batch runs to completion");
}

// ─── 9. tail-sweep pairs every un-run call under a hard steer
// ─────────────────

#[tokio::test]
async fn tail_sweep_pairs_all_skipped() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	// Leading exclusive → interrupt fires immediately, nothing runs.
	let tools = vec![
		TimedTool::boxed("x", Concurrency::Exclusive, &timeline, Behavior::Instant),
		TimedTool::boxed("y", Concurrency::Exclusive, &timeline, Behavior::Instant),
	];
	let message = tool_turn(&[("c1", "x"), ("c2", "y")]);
	let ct = CancelToken::default();
	let steering: &(dyn Fn() -> bool + Send + Sync) = &|| true;

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), Some(steering)).await;

	assert!(timeline_of(&timeline).is_empty(), "no tool should have started");
	assert_result_pairing(&message, &results);
	assert!(results.iter().all(|r| r.is_error), "every call paired with a skipped result");
}

// ─── 10. an exclusive tool that errors does not break the batch ──────────────

#[tokio::test]
async fn exclusive_error_does_not_break_batch() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	let tools = vec![
		TimedTool::boxed("x", Concurrency::Exclusive, &timeline, Behavior::Fail),
		TimedTool::boxed("a", Concurrency::Shared, &timeline, Behavior::Instant),
	];
	let message = tool_turn(&[("c1", "x"), ("c2", "a")]);
	let ct = CancelToken::default();

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), None).await;

	assert_result_pairing(&message, &results);
	let x = results.iter().find(|r| r.tool_name == "x").unwrap();
	assert!(x.is_error, "failed exclusive yields is_error");
	let a = results.iter().find(|r| r.tool_name == "a").unwrap();
	assert!(!a.is_error, "shared after a failed exclusive still runs");
	assert!(pos(&timeline_of(&timeline), "a:start") > pos(&timeline_of(&timeline), "x:end"));
}

// ─── Loop-level steering injection ───────────────────────────────────────────

fn user(s: &str) -> Message {
	Message::User(UserMessage {
		content:          UserContent::Text(s.into()),
		synthetic:        None,
		steering:         None,
		attribution:      None,
		provider_payload: None,
		timestamp:        0,
	})
}

/// A `stream_fn` that replays one scripted assistant message per model call.
fn scripted(messages: Vec<AssistantMessage>) -> pi_agent::StreamFn {
	let queue = Arc::new(Mutex::new(std::collections::VecDeque::from(messages)));
	Box::new(move |_ctx, _ct| {
		let next = queue
			.lock()
			.unwrap()
			.pop_front()
			.unwrap_or_else(|| assistant(vec![], StopReason::Stop));
		let (sink, stream) = AssistantMessageEventStream::channel();
		for event in emit_nonstream_events(&Arc::new(next)) {
			assert!(sink.try_push(event));
		}
		stream
	})
}

async fn collect(mut stream: pi_agent::AgentEventStream) -> Vec<AgentEvent> {
	let mut events = Vec::new();
	while let Some(event) = stream.next().await {
		events.push(event);
	}
	events
}

fn count(events: &[AgentEvent], f: impl Fn(&AgentEvent) -> bool) -> usize {
	events.iter().filter(|e| f(e)).count()
}

fn text(s: &str) -> AssistantContent {
	AssistantContent::Text(TextContent { text: s.into(), text_signature: None })
}

// ─── 11. steer queued at a stop boundary forces another turn ─────────────────

#[tokio::test]
async fn steering_after_batch_resumes_the_loop() {
	// Turn 1 stops with no tool calls; a steer is queued exactly once, so the
	// outer loop injects it and runs a second turn instead of ending.
	let calls = Arc::new(AtomicUsize::new(0));
	let get_calls = Arc::clone(&calls);
	let config = AgentConfig::new("fake", 1024)
		.with_stream_fn(scripted(vec![
			assistant(vec![text("first")], StopReason::Stop),
			assistant(vec![text("second")], StopReason::Stop),
		]))
		.with_steering(
			Box::new(move || {
				// Drain #0 is the loop-start poll (empty); the steer lands at the
				// post-batch drain (#1), forcing the inner loop to run turn 2.
				if get_calls.fetch_add(1, Ordering::SeqCst) == 1 {
					vec![user("keep going")]
				} else {
					vec![]
				}
			}),
			Box::new(|| false),
		);
	let ctx = AgentContext {
		system_prompt: vec!["sys".into()],
		messages:      vec![],
		tools:         vec![],
	};

	let events = collect(agent_loop(vec![user("hi")], ctx, config, CancelToken::default())).await;

	// Two turns ran (the steer forced the second), and the steer message is in
	// history.
	assert_eq!(
		count(&events, |e| matches!(e, AgentEvent::TurnEnd { .. })),
		2,
		"steer forced a 2nd turn"
	);
	let messages = match events.last() {
		Some(AgentEvent::AgentEnd { messages }) => messages.clone(),
		_ => panic!("no agent_end"),
	};
	assert!(
		messages.iter().any(|m| matches!(m, Message::User(u)
			if matches!(&u.content, UserContent::Text(t) if t == "keep going"))),
		"steering message injected into history"
	);
}

// ─── 12. no steering ⇒ the outer loop breaks (1.4a behavior unchanged) ───────

#[tokio::test]
async fn no_steering_breaks_after_single_turn() {
	let config = AgentConfig::new("fake", 1024)
		.with_stream_fn(scripted(vec![assistant(vec![text("done")], StopReason::Stop)]));
	let ctx = AgentContext {
		system_prompt: vec!["sys".into()],
		messages:      vec![],
		tools:         vec![],
	};

	let events = collect(agent_loop(vec![user("hi")], ctx, config, CancelToken::default())).await;

	assert_eq!(
		count(&events, |e| matches!(e, AgentEvent::TurnEnd { .. })),
		1,
		"no steer ⇒ one turn"
	);
	assert!(matches!(events.last(), Some(AgentEvent::AgentEnd { .. })));
	let _ = DoneReason::Stop; // keep the import honest across pi-ai revisions
}

// ─── 13. serial equivalence: non-yielding shared tools keep content order ────

#[tokio::test]
async fn non_yielding_shared_tools_preserve_content_order() {
	let timeline = Arc::new(Mutex::new(Vec::new()));
	let tools = vec![
		TimedTool::boxed("a", Concurrency::Shared, &timeline, Behavior::Instant),
		TimedTool::boxed("b", Concurrency::Shared, &timeline, Behavior::Instant),
	];
	let message = tool_turn(&[("c1", "a"), ("c2", "b")]);
	let ct = CancelToken::default();

	let results = execute_tool_calls(&tools, &message, &ct, &sink(), None).await;

	// Tools that never await complete on first poll in push order → content order.
	assert_eq!(result_names(&results), vec!["a", "b"]);
	let tl = timeline_of(&timeline);
	assert_eq!(tl, vec!["a:start", "a:end", "b:start", "b:end"]);
	let _ = Duration::from_millis(0);
}
