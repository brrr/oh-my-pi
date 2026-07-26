//! WP-1.6 session-lifecycle features exercised through the public load path:
//! blob dereference (B5), the ≥8 MiB streaming loader (B6), and the context
//! rebuild's `branchSummary` synthesis + `retryRecovery` skip (B7).

use base64::Engine as _;
use pi_ai::message::{Message, UserContent, UserContentBlock};
use pi_session::{
	ContextMessage, STREAM_LOAD_THRESHOLD_BYTES, build_session_context,
	load_entries_from_file_with_blobs, parse_session_content, parse_session_content_with_blobs,
};
use serde_json::{Value, json};

const HEADER: &str = r#"{"type":"session","version":3,"id":"s1","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#;

fn tmp_dir(tag: &str) -> std::path::PathBuf {
	let dir = std::env::temp_dir().join(format!("pi-session-{tag}-{}", std::process::id()));
	std::fs::create_dir_all(&dir).unwrap();
	dir
}

// ── B5: blob dereference through the full loader ─────────────────────────────

#[test]
fn b5_blob_ref_in_image_block_resolves_to_inline_base64() {
	let blobs = tmp_dir("b5-present");
	let bytes = b"\x89PNG\r\n\x1a\nfake-image-bytes";
	std::fs::write(blobs.join("hash01"), bytes).unwrap();

	let user = json!({"type":"message","id":"a","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z",
		"message":{"role":"user","timestamp":1,
			"content":[{"type":"image","data":"blob:sha256:hash01","mimeType":"image/png"}]}});
	let content = format!("{HEADER}\n{user}\n");

	let loaded = parse_session_content_with_blobs(&content, &blobs).unwrap();
	let ctx = build_session_context(&loaded.entries, None);
	let data = first_image_data(&ctx.messages);
	assert_eq!(data, base64::engine::general_purpose::STANDARD.encode(bytes));

	std::fs::remove_dir_all(&blobs).ok();
}

#[test]
fn b5_missing_blob_keeps_ref_verbatim() {
	let blobs = tmp_dir("b5-missing"); // empty dir → blob not found
	let user = json!({"type":"message","id":"a","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z",
		"message":{"role":"user","timestamp":1,
			"content":[{"type":"image","data":"blob:sha256:absent","mimeType":"image/png"}]}});
	let content = format!("{HEADER}\n{user}\n");

	let loaded = parse_session_content_with_blobs(&content, &blobs).unwrap();
	let ctx = build_session_context(&loaded.entries, None);
	assert_eq!(first_image_data(&ctx.messages), "blob:sha256:absent");

	std::fs::remove_dir_all(&blobs).ok();
}

fn first_image_data(messages: &[ContextMessage]) -> String {
	for m in messages {
		if let ContextMessage::Standard(Message::User(u)) = m
			&& let UserContent::Blocks(blocks) = &u.content
		{
			for b in blocks {
				if let UserContentBlock::Image(img) = b {
					return img.data.clone();
				}
			}
		}
	}
	panic!("no image block found in rebuilt messages");
}

// ── B6: ≥8 MiB streaming loader ──────────────────────────────────────────────

#[test]
fn b6_streaming_loads_all_entries_of_a_large_file() {
	let dir = tmp_dir("b6-stream");
	let path = dir.join("large.jsonl");
	let blobs = dir.join("blobs"); // absent → no blob work
	let count = 12_000usize;

	// Title slot + header + `count` user messages, each padded so the file
	// crosses the 8 MiB threshold and takes the streaming path.
	let mut body = pi_session::serialize_title_slot("streamed", None, "2026-01-01T00:00:00.000Z");
	body.push_str(HEADER);
	body.push('\n');
	let pad = "x".repeat(700);
	for i in 0..count {
		let entry = json!({"type":"message","id":format!("m{i}"),
			"parentId": if i == 0 { Value::Null } else { json!(format!("m{}", i - 1)) },
			"timestamp":"2026-01-01T00:00:00.000Z",
			"message":{"role":"user","timestamp":1,"content":format!("msg-{i}-{pad}")}});
		body.push_str(&entry.to_string());
		body.push('\n');
	}
	// A malformed line the lenient parser must skip.
	body.push_str("not-json-garbage\n");
	std::fs::write(&path, &body).unwrap();

	let size = std::fs::metadata(&path).unwrap().len();
	assert!(
		size >= STREAM_LOAD_THRESHOLD_BYTES,
		"fixture must exceed the {STREAM_LOAD_THRESHOLD_BYTES}-byte streaming threshold (was {size})",
	);

	let loaded = load_entries_from_file_with_blobs(&path, &blobs).unwrap();
	assert_eq!(loaded.entries.len(), count, "every entry parsed via streaming");
	assert_eq!(loaded.title_slot.as_ref().map(|s| s.title.as_str()), Some("streamed"));

	// First/last content survived streaming intact.
	assert!(text_of(&loaded.entries[0]).starts_with("msg-0-"));
	assert!(text_of(&loaded.entries[count - 1]).starts_with(&format!("msg-{}-", count - 1)));

	std::fs::remove_dir_all(&dir).ok();
}

fn text_of(entry: &pi_session::SessionEntry) -> String {
	let v = serde_json::to_value(entry).unwrap();
	v["message"]["content"].as_str().unwrap_or("").to_string()
}

// ── B7: branchSummary synthesis + retryRecovery skip ─────────────────────────

#[test]
fn b7_branch_summary_entry_synthesizes_a_branch_summary_message() {
	let user = entry("a1", None, r#"{"role":"user","content":"hi","timestamp":1}"#);
	let asst = entry(
		"a2",
		Some("a1"),
		r#"{"role":"assistant","content":[],"api":"x","provider":"p","model":"m","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"stop","timestamp":1}"#,
	);
	let branch = r#"{"type":"branch_summary","id":"b1","parentId":"a2","timestamp":"2026-01-01T00:00:00.000Z","fromId":"x9","summary":"forked-here"}"#;
	let content = format!("{HEADER}\n{user}\n{asst}\n{branch}\n");

	let loaded = parse_session_content(&content).unwrap();
	let ctx = build_session_context(&loaded.entries, None);
	assert_eq!(ctx.messages.len(), 3, "user + assistant + branchSummary");
	match &ctx.messages[2] {
		ContextMessage::BranchSummary(b) => {
			assert_eq!(b.summary, "forked-here");
			assert_eq!(b.from_id, "x9");
		},
		other => panic!("expected branchSummary, got {other:?}"),
	}
}

#[test]
fn b7_empty_summary_branch_summary_emits_nothing() {
	let user = entry("a1", None, r#"{"role":"user","content":"hi","timestamp":1}"#);
	let branch = r#"{"type":"branch_summary","id":"b1","parentId":"a1","timestamp":"2026-01-01T00:00:00.000Z","fromId":"x9","summary":""}"#;
	let content = format!("{HEADER}\n{user}\n{branch}\n");

	let loaded = parse_session_content(&content).unwrap();
	let ctx = build_session_context(&loaded.entries, None);
	assert_eq!(ctx.messages.len(), 1, "empty-summary branch_summary is skipped");
}

#[test]
fn b7_recovered_assistant_turn_is_skipped() {
	let user = entry("a1", None, r#"{"role":"user","content":"hi","timestamp":1}"#);
	let recovered = entry(
		"a2",
		Some("a1"),
		r#"{"role":"assistant","content":[{"type":"text","text":"lost"}],"api":"x","provider":"p","model":"m","retryRecovery":{"status":"recovered"},"usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"stop","timestamp":1}"#,
	);
	let good = entry(
		"a3",
		Some("a2"),
		r#"{"role":"assistant","content":[{"type":"text","text":"kept"}],"api":"x","provider":"p","model":"m","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"stop","timestamp":1}"#,
	);
	let content = format!("{HEADER}\n{user}\n{recovered}\n{good}\n");

	let loaded = parse_session_content(&content).unwrap();
	let ctx = build_session_context(&loaded.entries, None);
	assert_eq!(ctx.messages.len(), 2, "recovered assistant turn dropped");
	// The kept assistant is the 'kept' text, not 'lost'.
	let texts = assistant_texts(&ctx.messages);
	assert_eq!(texts, vec!["kept".to_string()]);
}

fn entry(id: &str, parent: Option<&str>, message_json: &str) -> String {
	let parent = parent.map_or_else(|| "null".to_string(), |p| format!("\"{p}\""));
	format!(
		r#"{{"type":"message","id":"{id}","parentId":{parent},"timestamp":"2026-01-01T00:00:00.000Z","message":{message_json}}}"#
	)
}

fn assistant_texts(messages: &[ContextMessage]) -> Vec<String> {
	let mut out = Vec::new();
	for m in messages {
		if let ContextMessage::Standard(Message::Assistant(a)) = m {
			for block in &a.content {
				let v = serde_json::to_value(block).unwrap();
				if v.get("type").and_then(Value::as_str) == Some("text")
					&& let Some(t) = v.get("text").and_then(Value::as_str)
				{
					out.push(t.to_string());
				}
			}
		}
	}
	out
}
