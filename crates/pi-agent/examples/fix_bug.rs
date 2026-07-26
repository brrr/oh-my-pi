//! WP-1.4a L2 real task: drive the agent loop against a live provider to fix a
//! bug in a scratch Python file, then verify the fix by running it via the
//! `bash` tool.
//!
//! ```sh
//! cargo run -p pi-agent --example fix_bug
//! # overrides: OMP_AI_BASE_URL / OMP_AI_MODEL / OMP_AI_AUTH_ENTRY / ANTHROPIC_API_KEY
//! ```
//!
//! Requires network + a `DeepSeek` key (resolved from the opencode auth store,
//! entry `deepseek`, like the pi-ai `stream` example). Asserts: the loop runs
//! to completion, `buggy.py` was really modified, a `bash` tool result shows
//! the passing marker, and at least two `tool_execution_end` events fired.

use std::{fs, path::Path};

use pi_agent::{AgentConfig, AgentContext, AgentEvent, agent_loop, client_stream_fn};
use pi_ai::{
	auth::{AnthropicAuthConfig, resolve_api_key},
	client::Client,
	message::{Message, UserContent, UserMessage},
};
use pi_shell::cancel::CancelToken;
use pi_tools::{BashTool, DynTool, EditTool, GlobTool, GrepTool, ReadTool, WriteTool};

const BUGGY_PY: &str = r#"def add(a, b):
    # BUG: subtraction instead of addition
    return a - b


if __name__ == "__main__":
    assert add(2, 3) == 5, f"add(2, 3) should be 5, got {add(2, 3)}"
    assert add(10, 20) == 30, f"add(10, 20) should be 30, got {add(10, 20)}"
    print("ALL TESTS PASSED")
"#;

const PASS_MARKER: &str = "ALL TESTS PASSED";

fn env_or(name: &str, default: &str) -> String {
	std::env::var(name)
		.ok()
		.filter(|value| !value.is_empty())
		.unwrap_or_else(|| default.into())
}

fn tools(cwd: &Path) -> Vec<Box<dyn DynTool>> {
	vec![
		Box::new(ReadTool::new(cwd.to_path_buf())),
		Box::new(WriteTool::new(cwd.to_path_buf())),
		Box::new(EditTool::new(cwd.to_path_buf())),
		Box::new(BashTool::new(cwd.to_path_buf())),
		Box::new(GrepTool::new(cwd.to_path_buf())),
		Box::new(GlobTool::new(cwd.to_path_buf())),
	]
}

/// Extract the text of any content block (assistant text / tool result) for
/// logs.
fn result_text(value: &serde_json::Value) -> String {
	value
		.get("content")
		.and_then(|c| c.as_array())
		.map(|blocks| {
			blocks
				.iter()
				.filter_map(|b| b.get("text").and_then(|t| t.as_str()))
				.collect::<Vec<_>>()
				.join("")
		})
		.unwrap_or_default()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
	// ── scratch workdir ──────────────────────────────────────────────────────
	let workdir = std::env::temp_dir().join(format!("pi-agent-fixbug-{}", std::process::id()));
	fs::create_dir_all(&workdir).expect("create scratch dir");
	let buggy = workdir.join("buggy.py");
	fs::write(&buggy, BUGGY_PY).expect("write buggy.py");
	let original = fs::read_to_string(&buggy).expect("read buggy.py");
	println!("scratch workdir: {}", workdir.display());

	// ── provider ─────────────────────────────────────────────────────────────
	let base_url = env_or("OMP_AI_BASE_URL", "https://api.deepseek.com/anthropic");
	let model = env_or("OMP_AI_MODEL", "deepseek-v4-flash");
	let auth_entry = env_or("OMP_AI_AUTH_ENTRY", "deepseek");
	let api_key = resolve_api_key(None, Some(&auth_entry)).expect("resolve API key");
	let client = Client::new(AnthropicAuthConfig::new(api_key, Some(&base_url)), auth_entry);
	let stream_fn = client_stream_fn(client, model.clone(), 4096, Some(0.0));

	// ── loop config ──────────────────────────────────────────────────────────
	let system = "You are a coding agent working in a project directory. Use the provided tools \
	              (read, write, edit, bash, grep, glob) to complete the task. Paths are relative \
	              to the project root. When done, stop."
		.to_owned();
	let context = AgentContext {
		system_prompt: vec![system],
		messages:      vec![],
		tools:         tools(&workdir),
	};
	let config = AgentConfig::new(model, 4096).with_stream_fn(stream_fn);
	let prompt = Message::User(UserMessage {
		content:          UserContent::Text(
			"Read buggy.py, fix the bug in the add() function, then run it with bash (`python3 \
			 buggy.py`) to verify it prints ALL TESTS PASSED."
				.to_owned(),
		),
		synthetic:        None,
		steering:         None,
		attribution:      None,
		provider_payload: None,
		timestamp:        0,
	});

	// ── run + observe ────────────────────────────────────────────────────────
	let mut stream = agent_loop(vec![prompt], context, config, CancelToken::default());
	let mut tool_end_count = 0usize;
	let mut bash_pass_seen = false;
	let mut tool_calls: Vec<String> = Vec::new();
	while let Some(event) = stream.next().await {
		match &event {
			AgentEvent::ToolExecutionStart { tool_name, args, .. } => {
				let brief = serde_json::to_string(args).unwrap_or_default();
				let brief = if brief.len() > 120 {
					format!("{}…", &brief[..120])
				} else {
					brief
				};
				println!("→ tool call: {tool_name} {brief}");
				tool_calls.push(tool_name.clone());
			},
			AgentEvent::ToolExecutionEnd { tool_name, result, is_error, .. } => {
				tool_end_count += 1;
				let text = result_text(result);
				if tool_name == "bash" && text.contains(PASS_MARKER) {
					bash_pass_seen = true;
				}
				let head: String = text.lines().take(3).collect::<Vec<_>>().join(" ⏎ ");
				println!("← tool end: {tool_name} (isError={is_error:?}) {head}");
			},
			AgentEvent::TurnEnd { .. } => println!("— turn end —"),
			_ => {},
		}
	}

	// ── assertions ───────────────────────────────────────────────────────────
	let final_src = fs::read_to_string(&buggy).expect("read buggy.py after run");
	println!("\n=== tool call sequence: {tool_calls:?} ===");
	println!("=== tool_execution_end count: {tool_end_count} ===");

	assert!(final_src != original, "buggy.py was not modified by the agent");
	assert!(tool_end_count >= 2, "expected >= 2 tool_execution_end events, got {tool_end_count}");
	assert!(bash_pass_seen, "expected a bash tool result containing '{PASS_MARKER}'");

	// Independently re-run the fixed file to confirm it truly passes.
	let output = std::process::Command::new("python3")
		.arg(&buggy)
		.output()
		.expect("run python3 buggy.py");
	let stdout = String::from_utf8_lossy(&output.stdout);
	assert!(output.status.success(), "fixed buggy.py did not exit 0: {output:?}");
	assert!(stdout.contains(PASS_MARKER), "fixed buggy.py output missing marker: {stdout}");

	// Cleanup best-effort.
	let _ = fs::remove_dir_all(&workdir);
	println!("\nL2 PASS: bug fixed + verified via bash + independent re-run ({PASS_MARKER}).");
}
