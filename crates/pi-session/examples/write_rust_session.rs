//! Rust→TS parity fixture writer (WP-1.3, U-omp-30). Drives [`SessionWriter`]
//! to write a session (three messages + a `model_change` + a `compaction`) into
//! the directory given as `argv[1]`, then dumps the Rust-side read-only message
//! view (`load_session_messages`) next to it.
//! `scripts/verify-session-rust-to-ts.ts` loads the same JSONL through the TS
//! `loadSessionMessagesReadOnly` and asserts the two message arrays match.
//!
//! Prints one line of JSON to stdout: `{"jsonl": "...", "expected": "..."}`.

use std::path::Path;

use pi_session::{SessionWriter, load_session_messages, message_from_json};
use serde_json::json;

fn main() -> anyhow::Result<()> {
	let dir = std::env::args()
		.nth(1)
		.expect("usage: write_rust_session <out-dir>");
	let dir = Path::new(&dir);

	let assistant = |text: &str, tool: Option<(&str, &str)>| {
		let mut content = vec![json!({ "type": "text", "text": text })];
		if let Some((id, name)) = tool {
			content.push(json!({ "type": "toolCall", "id": id, "name": name, "arguments": {} }));
		}
		message_from_json(json!({
			"role": "assistant",
			"content": content,
			"api": "anthropic",
			"provider": "anthropic",
			"model": "claude-sonnet-4",
			"usage": {
				"input": 7, "output": 3, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 10,
				"cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 }
			},
			"stopReason": if tool.is_some() { "toolUse" } else { "stop" },
			"timestamp": 1_704_067_200_000_i64
		}))
		.expect("assistant message")
	};
	let user = |text: &str| {
		message_from_json(
			json!({ "role": "user", "content": text, "timestamp": 1_704_067_200_000_i64 }),
		)
		.expect("user message")
	};

	let mut writer = SessionWriter::create("/omp/fixture", dir)?;
	writer.append_message(user("what is 2+2?"))?;
	writer.append_message(assistant("let me compute", Some(("call_a", "calc"))))?;
	writer.append_model_change("anthropic/claude-opus-4", None)?;
	let kept = writer.append_message(user("and 3+3?"))?;
	writer.append_compaction("summary of the arithmetic so far", Some("arith"), &kept, 128)?;
	writer.flush()?;

	let jsonl = writer.path().to_path_buf();
	let expected = load_session_messages(&jsonl)?;
	let expected_path = dir.join("rust-session.expected.json");
	std::fs::write(&expected_path, format!("{}\n", serde_json::to_string_pretty(&expected)?))?;

	println!(
		"{}",
		json!({ "jsonl": jsonl.to_string_lossy(), "expected": expected_path.to_string_lossy() })
	);
	Ok(())
}
