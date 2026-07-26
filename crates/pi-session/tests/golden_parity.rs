//! TS→Rust golden parity (WP-1.3, U-omp-30): load each session JSONL fixture
//! written by the real TypeScript `SessionManager`
//! (`scripts/gen-session-goldens.ts`), rebuild the read-only message view with
//! `build_session_context`, and assert it equals the `*.messages.json`
//! expectation dumped from the TS `loadSessionMessagesReadOnly`.
//!
//! Comparison is on parsed `serde_json::Value` (semantic equality), which is
//! robust to key ordering across the JS `JSON.stringify` and Rust `serde_json`
//! serializers — a deliberate, stronger check than raw byte compare (whitespace
//! / key-order noise cannot mask a real difference, and no real difference can
//! hide behind ordering). Numbers compare by numeric value so a JSON number has
//! no spurious int-vs-float identity (`0` == `0.0`): JSON itself draws no such
//! distinction, but Rust's `f64` cost fields render `0.0` where the JS number
//! renders `0`.

use std::path::{Path, PathBuf};

use pi_session::load_session_messages;
use serde_json::Value;

fn goldens_dir() -> PathBuf {
	Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens")
}

/// Structural JSON equality with numbers compared by value (`0` == `0.0`).
fn json_eq(a: &Value, b: &Value) -> bool {
	match (a, b) {
		(Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
			(Some(x), Some(y)) => x == y,
			_ => x == y,
		},
		(Value::Array(x), Value::Array(y)) => {
			x.len() == y.len() && x.iter().zip(y).all(|(a, b)| json_eq(a, b))
		},
		(Value::Object(x), Value::Object(y)) => {
			x.len() == y.len()
				&& x
					.iter()
					.all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
		},
		_ => a == b,
	}
}

/// Load a fixture's JSONL, rebuild the context, and compare against its dumped
/// TS message expectation.
fn assert_fixture_parity(name: &str) {
	let dir = goldens_dir();
	let jsonl = dir.join(format!("{name}.jsonl"));
	let expected_path = dir.join(format!("{name}.messages.json"));

	let messages =
		load_session_messages(&jsonl).unwrap_or_else(|e| panic!("load {}: {e:#}", jsonl.display()));
	let actual: Value = serde_json::to_value(&messages).expect("serialize rust messages");

	let expected_str = std::fs::read_to_string(&expected_path)
		.unwrap_or_else(|e| panic!("read {}: {e}", expected_path.display()));
	let expected: Value = serde_json::from_str(&expected_str).expect("parse expected json");

	assert!(
		json_eq(&actual, &expected),
		"message parity mismatch for fixture {name}\n--- rust ---\n{}\n--- ts ---\n{}",
		serde_json::to_string_pretty(&actual).unwrap(),
		serde_json::to_string_pretty(&expected).unwrap(),
	);
}

#[test]
fn basic_conversation_parity() {
	assert_fixture_parity("basic");
}

#[test]
fn compaction_and_settings_parity() {
	assert_fixture_parity("compaction");
}

#[test]
fn unknown_types_and_fork_parity() {
	assert_fixture_parity("unknown-and-fork");
}
