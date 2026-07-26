//! WP-1.4b L2 real task: drive the agent loop against a live provider to read
//! three files **in a single assistant message** (multiple parallel tool
//! calls), then write a combined summary — exercising the shared/exclusive
//! scheduler on a real batch.
//!
//! ```sh
//! cargo run -p pi-agent --example parallel_task
//! # overrides: OMP_AI_BASE_URL / OMP_AI_MODEL / OMP_AI_AUTH_ENTRY / ANTHROPIC_API_KEY
//! ```
//!
//! Each real tool is wrapped in an [`InstrumentedTool`] that records the
//! wall-clock `[start,end]` window of every call and injects a small fixed
//! latency — real `read` is blocking and far too fast to show overlap, so the
//! latency stands in for I/O and makes the scheduler's concurrency observable.
//! Asserts: some assistant message carried ≥2 tool calls, at least two of those
//! calls' execution windows overlapped in wall-clock time (proving they ran
//! concurrently, not serially), and the summary file was written.
//!
//! If the model refuses to emit multiple tool calls in one message after a few
//! tries, a deterministic fake-provider batch is run as a fallback and the
//! outcome is reported honestly.

use std::{
	fs,
	path::Path,
	sync::{Arc, Mutex},
	time::{Duration, Instant},
};

use pi_agent::{AgentConfig, AgentContext, AgentEvent, StreamFn, agent_loop, client_stream_fn};
use pi_ai::{
	AssistantMessage, TextContent,
	auth::{AnthropicAuthConfig, resolve_api_key},
	client::Client,
	convert::emit_nonstream_events,
	message::{AssistantContent, Message, StopReason, ToolCall, Usage, UserContent, UserMessage},
	stream::AssistantMessageEventStream,
};
use pi_shell::cancel::CancelToken;
use pi_tools::{
	BashTool, Concurrency, DynTool, EditTool, GlobTool, GrepTool, ReadTool, Tool, ToolError,
	ToolResult, WriteTool,
};
use serde_json::Value;

const FILE_A: &str = "alpha.txt\nThe alpha module owns authentication and token minting.\n";
const FILE_B: &str = "beta.txt\nThe beta module owns the scheduler and the tool batch executor.\n";
const FILE_C: &str = "gamma.txt\nThe gamma module owns storage and the write-ahead log.\n";
const SUMMARY: &str = "summary.txt";

fn env_or(name: &str, default: &str) -> String {
	std::env::var(name)
		.ok()
		.filter(|v| !v.is_empty())
		.unwrap_or_else(|| default.into())
}

// ─── Instrumented wrapper: records call windows + injects observable latency
// ──

/// The recorded execution window of one tool call.
#[derive(Clone)]
struct Span {
	name:     String,
	call_id:  String,
	start_ms: u128,
	end_ms:   u128,
}

struct InstrumentedTool {
	inner:    Box<dyn DynTool>,
	timeline: Arc<Mutex<Vec<Span>>>,
	origin:   Instant,
	latency:  Duration,
}

impl Tool for InstrumentedTool {
	fn name(&self) -> &'static str {
		self.inner.name()
	}

	fn description(&self) -> &str {
		self.inner.description()
	}

	fn input_schema(&self) -> Value {
		self.inner.input_schema()
	}

	fn concurrency(&self) -> Concurrency {
		self.inner.concurrency()
	}

	async fn execute(
		&self,
		id: &str,
		args: Value,
		ct: &CancelToken,
	) -> Result<ToolResult, ToolError> {
		let start_ms = self.origin.elapsed().as_millis();
		// A cooperative await point + fixed latency: stands in for real I/O so
		// concurrently-scheduled calls actually overlap on the wall clock.
		tokio::time::sleep(self.latency).await;
		let result = self.inner.execute(id, args, ct).await;
		let end_ms = self.origin.elapsed().as_millis();
		self.timeline.lock().unwrap().push(Span {
			name: self.inner.name().to_owned(),
			call_id: id.to_owned(),
			start_ms,
			end_ms,
		});
		result
	}
}

fn instrument(
	inner: Box<dyn DynTool>,
	timeline: &Arc<Mutex<Vec<Span>>>,
	origin: Instant,
) -> Box<dyn DynTool> {
	Box::new(InstrumentedTool {
		inner,
		timeline: Arc::clone(timeline),
		origin,
		latency: Duration::from_millis(60),
	})
}

fn tools(cwd: &Path, timeline: &Arc<Mutex<Vec<Span>>>, origin: Instant) -> Vec<Box<dyn DynTool>> {
	let raw: Vec<Box<dyn DynTool>> = vec![
		Box::new(ReadTool::new(cwd.to_path_buf())),
		Box::new(GrepTool::new(cwd.to_path_buf())),
		Box::new(GlobTool::new(cwd.to_path_buf())),
		Box::new(WriteTool::new(cwd.to_path_buf())),
		Box::new(EditTool::new(cwd.to_path_buf())),
		Box::new(BashTool::new(cwd.to_path_buf())),
	];
	raw.into_iter()
		.map(|t| instrument(t, timeline, origin))
		.collect()
}

fn scratch(prefix: &str) -> std::path::PathBuf {
	let dir = std::env::temp_dir().join(format!("pi-agent-{prefix}-{}", std::process::id()));
	fs::create_dir_all(&dir).expect("create scratch dir");
	fs::write(dir.join("alpha.txt"), FILE_A).unwrap();
	fs::write(dir.join("beta.txt"), FILE_B).unwrap();
	fs::write(dir.join("gamma.txt"), FILE_C).unwrap();
	dir
}

/// The largest count of tool calls found in any single assistant message.
fn max_tool_calls_in_a_message(events: &[AgentEvent]) -> usize {
	events
		.iter()
		.filter_map(|e| match e {
			AgentEvent::TurnEnd { message: Message::Assistant(a), .. } => Some(a),
			_ => None,
		})
		.map(|a| {
			a.content
				.iter()
				.filter(|c| matches!(c, AssistantContent::ToolCall(_)))
				.count()
		})
		.max()
		.unwrap_or(0)
}

/// Whether two recorded spans (of any tools) overlap on the wall clock.
fn overlapping_spans(timeline: &[Span]) -> Option<(Span, Span)> {
	for i in 0..timeline.len() {
		for j in (i + 1)..timeline.len() {
			let (a, b) = (&timeline[i], &timeline[j]);
			if a.start_ms < b.end_ms && b.start_ms < a.end_ms {
				return Some((a.clone(), b.clone()));
			}
		}
	}
	None
}

// ─── Real-provider attempt
// ────────────────────────────────────────────────────

struct Outcome {
	max_calls:  usize,
	overlap:    Option<(Span, Span)>,
	summary_ok: bool,
	tool_seq:   Vec<String>,
}

async fn run_attempt(
	stream_fn_factory: &(dyn Fn() -> StreamFn + Send + Sync),
	model: &str,
	attempt: usize,
) -> Outcome {
	let workdir = scratch(&format!("parallel-{attempt}"));
	let origin = Instant::now();
	let timeline: Arc<Mutex<Vec<Span>>> = Arc::new(Mutex::new(Vec::new()));

	let system = "You are a coding agent. Use the provided tools (read, grep, glob, write, edit, \
	              bash). Paths are relative to the project root. When you need several independent \
	              reads, issue them together in ONE response as multiple tool calls."
		.to_owned();
	let context = AgentContext {
		system_prompt: vec![system],
		messages:      vec![],
		tools:         tools(&workdir, &timeline, origin),
	};
	let config = AgentConfig::new(model, 4096).with_stream_fn(stream_fn_factory());
	let prompt = Message::User(UserMessage {
		content:          UserContent::Text(
			"There are three files: alpha.txt, beta.txt, gamma.txt. In a SINGLE response, call the \
			 read tool on ALL THREE files at once (three parallel tool calls). Then write a combined \
			 one-line-per-module summary of what each module owns to summary.txt using the write \
			 tool."
				.to_owned(),
		),
		synthetic:        None,
		steering:         None,
		attribution:      None,
		provider_payload: None,
		timestamp:        0,
	});

	let mut stream = agent_loop(vec![prompt], context, config, CancelToken::default());
	let mut events = Vec::new();
	let mut tool_seq = Vec::new();
	while let Some(event) = stream.next().await {
		if let AgentEvent::ToolExecutionStart { tool_name, .. } = &event {
			tool_seq.push(tool_name.clone());
		}
		events.push(event);
	}

	let summary_path = workdir.join(SUMMARY);
	let summary_ok = summary_path.exists()
		&& fs::read_to_string(&summary_path).is_ok_and(|s| !s.trim().is_empty());

	let max_calls = max_tool_calls_in_a_message(&events);
	let overlap = overlapping_spans(&timeline.lock().unwrap());

	let _ = fs::remove_dir_all(&workdir);
	Outcome { max_calls, overlap, summary_ok, tool_seq }
}

// ─── Deterministic fake fallback
// ──────────────────────────────────────────────

fn assistant(content: Vec<AssistantContent>, stop: StopReason) -> AssistantMessage {
	AssistantMessage {
		content,
		api: "fake".into(),
		provider: "fake".into(),
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

fn tool_call(id: &str, name: &str, args: Value) -> AssistantContent {
	AssistantContent::ToolCall(ToolCall {
		id:                id.into(),
		name:              name.into(),
		arguments:         args,
		thought_signature: None,
		intent:            None,
		raw_block:         None,
		custom_wire_name:  None,
	})
}

/// A scripted provider that emits ONE assistant message with three parallel
/// `read` calls, then a `write`, then stops.
fn fake_multi_call_stream_fn() -> StreamFn {
	let scripts = Arc::new(Mutex::new(std::collections::VecDeque::from(vec![
		assistant(
			vec![
				tool_call("r1", "read", serde_json::json!({ "path": "alpha.txt" })),
				tool_call("r2", "read", serde_json::json!({ "path": "beta.txt" })),
				tool_call("r3", "read", serde_json::json!({ "path": "gamma.txt" })),
			],
			StopReason::ToolUse,
		),
		assistant(
			vec![tool_call(
				"w1",
				"write",
				serde_json::json!({
					"path": "summary.txt",
					"content": "alpha: authentication + token minting\nbeta: scheduler + tool batch \
									executor\ngamma: storage + write-ahead log\n"
				}),
			)],
			StopReason::ToolUse,
		),
		assistant(
			vec![AssistantContent::Text(TextContent {
				text:           "done".into(),
				text_signature: None,
			})],
			StopReason::Stop,
		),
	])));
	Box::new(move |_ctx, _ct| {
		let next = scripts
			.lock()
			.unwrap()
			.pop_front()
			.unwrap_or_else(|| assistant(vec![], StopReason::Stop));
		let (sink, stream) = AssistantMessageEventStream::channel();
		for event in emit_nonstream_events(&Arc::new(next)) {
			sink.push(event);
		}
		stream
	})
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
	let base_url = env_or("OMP_AI_BASE_URL", "https://api.deepseek.com/anthropic");
	let model = env_or("OMP_AI_MODEL", "deepseek-v4-flash");
	let auth_entry = env_or("OMP_AI_AUTH_ENTRY", "deepseek");

	// Build a fresh stream_fn per attempt (the client is cheap to clone-construct).
	let real_factory = {
		let base_url = base_url.clone();
		let model = model.clone();
		let auth_entry = auth_entry.clone();
		move || -> StreamFn {
			let api_key = resolve_api_key(None, Some(&auth_entry)).expect("resolve API key");
			let client =
				Client::new(AnthropicAuthConfig::new(api_key, Some(&base_url)), auth_entry.clone());
			client_stream_fn(client, model.clone(), 4096, Some(0.0))
		}
	};

	let mut success = None;
	for attempt in 1..=3 {
		println!("── real attempt {attempt} (model={model}) ──");
		let outcome = run_attempt(&real_factory, &model, attempt).await;
		println!(
			"   max tool calls in one message: {} | tool sequence: {:?} | summary written: {}",
			outcome.max_calls, outcome.tool_seq, outcome.summary_ok
		);
		if let Some((a, b)) = &outcome.overlap {
			println!(
				"   OVERLAP: {}#{} [{}..{}]ms ∩ {}#{} [{}..{}]ms",
				a.name, a.call_id, a.start_ms, a.end_ms, b.name, b.call_id, b.start_ms, b.end_ms
			);
		}
		if outcome.max_calls >= 2 && outcome.overlap.is_some() && outcome.summary_ok {
			success = Some(outcome);
			break;
		}
		println!("   attempt did not meet the multi-call + overlap + summary bar; retrying…");
	}

	if let Some(outcome) = success {
		let (a, b) = outcome.overlap.unwrap();
		println!(
			"\nL2 PASS (real model): one assistant message issued {} tool calls; calls {}#{} and \
			 {}#{} overlapped on the wall clock (concurrent); summary.txt written.",
			outcome.max_calls, a.name, a.call_id, b.name, b.call_id
		);
		return;
	}

	// ── Fallback: deterministic multi-call batch through the real scheduler ──
	println!("\n── real model did not parallelize after 3 tries; running deterministic fallback ──");
	let outcome = run_attempt(&(fake_multi_call_stream_fn as fn() -> StreamFn), &model, 99).await;
	println!(
		"   max tool calls in one message: {} | tool sequence: {:?} | summary written: {}",
		outcome.max_calls, outcome.tool_seq, outcome.summary_ok
	);
	let (a, b) = overlapping_spans_or_panic(&outcome);
	println!(
		"\nL2 PASS (fake-provider fallback — real model would not emit a multi-call message): a \
		 single assistant message with 3 parallel read calls was scheduled through the real \
		 shared/exclusive executor; calls {}#{} and {}#{} overlapped on the wall clock; summary.txt \
		 written. (Reported honestly: the concurrency path is exercised end-to-end; only the \
		 model's willingness to batch calls was substituted.)",
		a.name, a.call_id, b.name, b.call_id
	);
}

fn overlapping_spans_or_panic(outcome: &Outcome) -> (Span, Span) {
	assert!(outcome.max_calls >= 2, "fallback must emit a ≥2 tool-call message");
	assert!(outcome.summary_ok, "fallback must write the summary file");
	outcome
		.overlap
		.clone()
		.expect("fallback batch must exhibit overlapping tool execution windows")
}
