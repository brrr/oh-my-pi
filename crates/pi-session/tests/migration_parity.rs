//! WP-1.6 (B4) v1→v3 migration parity: load each real v1 session fixture
//! (`tests/fixtures/v1-*.jsonl`, copied from the TS repo), migrate it in memory
//! (v1→v2 id/parentId + `firstKeptEntryIndex`→`firstKeptEntryId`, v2→v3),
//! rebuild the read-only message view with `build_session_context`, and assert
//! it is consistent with the `*.messages.json` dumped from the TS
//! `loadSessionMessagesReadOnly` (`scripts/gen-migration-goldens.ts`).
//!
//! The assertion is on a migration-focused *skeleton* — the ordered
//! `(role, text, tool-identity)` of every message — rather than byte parity of
//! the full message object. Migration correctness is exactly what that skeleton
//! captures (tree linearization, compaction kept-range, entry ordering, content
//! integrity); the full object would additionally re-check pi-ai field
//! normalization (e.g. a legacy `usage` without `totalTokens`), which is a
//! message-model concern, not a migration one. Entry ids differ per run (TS
//! random, Rust counter) and never appear in the skeleton.
//!
//! Both sides are filtered to the message roles pi-session's core `Message`
//! models. `before-compaction` carries 3 `bashExecution` messages — a
//! coding-agent *extension* message role (messages.ts:568), not one of pi-ai's
//! four core roles — which the loader drops to opaque unknown entries. That is
//! a message-model boundary, not a migration defect; a `sanity` assertion pins
//! that the only TS messages outside the modeled set are those known extension
//! roles, so a real dropped user/assistant turn still fails the test.

use std::path::{Path, PathBuf};

use pi_session::load_session_messages;
use serde_json::Value;

fn fixtures_dir() -> PathBuf {
	Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// One message reduced to what migration must preserve: role, joined text, and
/// tool identity (assistant tool-call names, toolResult call id).
#[derive(Debug, PartialEq, Eq)]
struct Skeleton {
	role: String,
	text: String,
	tool: String,
}

fn join_text(content: &Value) -> String {
	match content {
		Value::String(s) => s.clone(),
		Value::Array(blocks) => blocks
			.iter()
			.filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
			.filter_map(|b| b.get("text").and_then(Value::as_str))
			.collect::<Vec<_>>()
			.join("\u{1e}"),
		_ => String::new(),
	}
}

fn tool_identity(msg: &Value) -> String {
	let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
	match role {
		"assistant" => msg
			.get("content")
			.and_then(Value::as_array)
			.map(|blocks| {
				blocks
					.iter()
					.filter(|b| b.get("type").and_then(Value::as_str) == Some("toolCall"))
					.filter_map(|b| b.get("name").and_then(Value::as_str))
					.collect::<Vec<_>>()
					.join(",")
			})
			.unwrap_or_default(),
		"toolResult" => msg
			.get("toolCallId")
			.and_then(Value::as_str)
			.unwrap_or("")
			.to_string(),
		_ => String::new(),
	}
}

fn skeleton(msg: &Value) -> Skeleton {
	Skeleton {
		role: msg
			.get("role")
			.and_then(Value::as_str)
			.unwrap_or("")
			.to_string(),
		text: msg.get("content").map(join_text).unwrap_or_default(),
		tool: tool_identity(msg),
	}
}

/// Roles pi-session's core `Message` + context synthetic messages model.
const MODELED_ROLES: &[&str] = &[
	"user",
	"developer",
	"assistant",
	"toolResult",
	"custom",
	"compactionSummary",
	"branchSummary",
];

/// Coding-agent extension message roles pi-session's core `Message` does NOT
/// model (dropped to unknown entries). Anything outside modeled ∪
/// known-extension is an unexpected drop and fails the sanity check.
const KNOWN_EXTENSION_ROLES: &[&str] = &["bashExecution"];

fn role_of(msg: &Value) -> &str {
	msg.get("role").and_then(Value::as_str).unwrap_or("")
}

fn skeletons(messages: &Value) -> Vec<Skeleton> {
	messages
		.as_array()
		.expect("messages is an array")
		.iter()
		.filter(|m| MODELED_ROLES.contains(&role_of(m)))
		.map(skeleton)
		.collect()
}

fn assert_migration_parity(name: &str) {
	let dir = fixtures_dir();
	let jsonl = dir.join(format!("{name}.jsonl"));
	let expected_path = dir.join(format!("{name}.messages.json"));

	let messages =
		load_session_messages(&jsonl).unwrap_or_else(|e| panic!("load {}: {e:#}", jsonl.display()));
	let actual = serde_json::to_value(&messages).expect("serialize rust messages");
	let expected: Value = serde_json::from_str(
		&std::fs::read_to_string(&expected_path)
			.unwrap_or_else(|e| panic!("read {}: {e}", expected_path.display())),
	)
	.expect("parse expected json");

	// Sanity: every TS message outside the modeled set must be a known extension
	// role, so filtering can't mask a genuinely dropped user/assistant turn.
	for m in expected.as_array().expect("ts messages array") {
		let role = role_of(m);
		assert!(
			MODELED_ROLES.contains(&role) || KNOWN_EXTENSION_ROLES.contains(&role),
			"unexpected unmodeled TS message role {role:?} in {name}",
		);
	}

	let rust = skeletons(&actual);
	let ts = skeletons(&expected);
	assert_eq!(
		rust.len(),
		ts.len(),
		"message count mismatch for {name}: rust {} vs ts {}",
		rust.len(),
		ts.len(),
	);
	for (i, (r, t)) in rust.iter().zip(&ts).enumerate() {
		assert_eq!(r, t, "message #{i} skeleton mismatch for fixture {name}");
	}
}

#[test]
fn v1_before_compaction_migrates_to_ts_parity() {
	assert_migration_parity("v1-before-compaction");
}

#[test]
fn v1_large_session_migrates_to_ts_parity() {
	assert_migration_parity("v1-large-session");
}
