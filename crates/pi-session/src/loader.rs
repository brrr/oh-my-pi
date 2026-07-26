//! Session file reader. Rust mirror of
//! `packages/coding-agent/src/session/session-loader.ts` + the leaf-selection
//! of `session-context.ts`.
//!
//! Pipeline (faithful to `loadSessionMessagesReadOnly`, all stages operate on
//! raw JSON `Value`s before the strong-typed parse, exactly like the TS
//! `FileEntry[]` mutation chain):
//!
//! - Read the file — whole-file for small sessions, streaming line-by-line once
//!   the size reaches [`STREAM_LOAD_THRESHOLD_BYTES`] (≥8 MiB) so the file is
//!   never fully buffered (B6).
//! - Peel the optional fixed-width title slot, lenient-parse the JSONL body
//!   (malformed lines skipped).
//! - Elide superseded compactions — runs *before* migration, so a v1 file whose
//!   entries have no ids yet gets no elision (its branch-id set is empty),
//!   byte-for-byte matching `loadEntriesFromFile`.
//! - Migrate `< 3` → v3 (B4): v1→v2 linearizes the entry tree (id/parentId and
//!   `firstKeptEntryIndex`→`firstKeptEntryId`), v2→v3 renames a `hookMessage`
//!   role to `custom`.
//! - Resolve `blob:sha256:…` refs against the blob store (B5): image data
//!   blocks / `images` / `image_url` / `image_generation_call` results are read
//!   back inline; a missing blob keeps the ref verbatim and warns (never
//!   panics).
//! - Validate the header (`type == "session"`, string `id`) and strong-parse
//!   the remaining entries.
//!
//! Deferred (registered here): transcript mode (see `context.rs`).

use std::{
	collections::BTreeSet,
	fs,
	io::{BufRead, BufReader},
	path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::{
	context::{ContextMessage, SessionContext, build_session_context},
	entries::{SessionEntry, SessionHeader},
	title_slot::{TitleSlot, parse_title_slot_line},
};

const ELIDED_COMPACTION_SUMMARY: &str =
	"[Superseded compaction summary elided during session load]";
const ELIDED_COMPACTION_SHORT_SUMMARY: &str = "Superseded compaction elided";

/// Files at or above this size take the streaming (line-by-line) load path so
/// the whole file is never held in memory (`STREAM_LOAD_THRESHOLD_BYTES`).
pub const STREAM_LOAD_THRESHOLD_BYTES: u64 = 8 * 1024 * 1024;

/// A loaded session: the header, the folded title slot (if any), and the
/// logical entry list (header excluded).
#[derive(Debug, Clone)]
pub struct LoadedSession {
	pub header:     SessionHeader,
	pub title_slot: Option<TitleSlot>,
	pub entries:    Vec<SessionEntry>,
}

/// Content-addressed blob store directory: `$HOME/.omp/agent/blobs` (mirror of
/// `getBlobsDir()`; the XDG data-home variant is out of scope for the headless
/// default).
pub fn default_blobs_dir() -> PathBuf {
	std::env::var_os("HOME")
		.map(PathBuf::from)
		.unwrap_or_default()
		.join(".omp")
		.join("agent")
		.join("blobs")
}

/// Load and validate a session file, migrating any `< 3` version in memory and
/// resolving blob refs against the default blob store.
pub fn load_entries_from_file(path: impl AsRef<Path>) -> Result<LoadedSession> {
	load_entries_from_file_with_blobs(path, &default_blobs_dir())
}

/// [`load_entries_from_file`] with an explicit blob store directory (for
/// tests).
pub fn load_entries_from_file_with_blobs(
	path: impl AsRef<Path>,
	blobs_dir: &Path,
) -> Result<LoadedSession> {
	let path = path.as_ref();
	let size = fs::metadata(path)
		.with_context(|| format!("stat session file {}", path.display()))?
		.len();
	let (title_slot, values) = if size >= STREAM_LOAD_THRESHOLD_BYTES {
		read_values_streaming(path)?
	} else {
		let content = fs::read_to_string(path)
			.with_context(|| format!("reading session file {}", path.display()))?;
		split_and_parse(&content)
	};
	finalize(values, title_slot, blobs_dir)
}

/// Parse a full physical session body (title slot + JSONL). Exposed for tests
/// that construct content in memory; uses the default blob store.
pub fn parse_session_content(content: &str) -> Result<LoadedSession> {
	let (title_slot, values) = split_and_parse(content);
	finalize(values, title_slot, &default_blobs_dir())
}

/// [`parse_session_content`] with an explicit blob store directory (for tests).
pub fn parse_session_content_with_blobs(content: &str, blobs_dir: &Path) -> Result<LoadedSession> {
	let (title_slot, values) = split_and_parse(content);
	finalize(values, title_slot, blobs_dir)
}

/// Read-only message view: load the file and rebuild the context along the
/// persisted leaf (last entry). Mirror of `loadSessionMessagesReadOnly`.
pub fn load_session_messages(path: impl AsRef<Path>) -> Result<Vec<ContextMessage>> {
	Ok(load_session_context(path)?.messages)
}

/// Load the file and return the full rebuilt [`SessionContext`].
pub fn load_session_context(path: impl AsRef<Path>) -> Result<SessionContext> {
	let loaded = load_entries_from_file(path)?;
	Ok(build_session_context(&loaded.entries, None))
}

// ── Reading ──────────────────────────────────────────────────────────────────

/// Whole-file path: peel the title slot from the first physical line, then
/// lenient-parse every remaining non-blank line.
fn split_and_parse(content: &str) -> (Option<TitleSlot>, Vec<Value>) {
	let raw: Vec<&str> = content.split('\n').collect();
	let (title_slot, body_start) = match raw.first() {
		Some(first) => match parse_title_slot_line(first.trim()) {
			Some(slot) => (Some(slot), 1),
			None => (None, 0),
		},
		None => (None, 0),
	};
	let mut values = Vec::new();
	for line in &raw[body_start..] {
		push_lenient(line, &mut values);
	}
	(title_slot, values)
}

/// Streaming path (≥8 MiB): read the file line by line so the whole body is
/// never buffered. The first physical line is peeled as an optional title slot;
/// a non-slot first line is a real entry left for the parser.
fn read_values_streaming(path: &Path) -> Result<(Option<TitleSlot>, Vec<Value>)> {
	let file = fs::File::open(path)
		.with_context(|| format!("opening session file for streaming {}", path.display()))?;
	let mut reader = BufReader::new(file);
	let mut values = Vec::new();
	let mut title_slot = None;

	let mut first = String::new();
	let read = reader
		.read_line(&mut first)
		.context("reading first session line")?;
	if read > 0 {
		let trimmed = first.trim();
		match parse_title_slot_line(trimmed) {
			Some(slot) => title_slot = Some(slot),
			None => push_lenient(trimmed, &mut values),
		}
	}

	for line in reader.lines() {
		let line = line.context("reading session line")?;
		push_lenient(&line, &mut values);
	}
	Ok((title_slot, values))
}

/// Push a lenient-parsed JSON line: blanks and malformed lines are skipped.
fn push_lenient(line: &str, out: &mut Vec<Value>) {
	let trimmed = line.trim();
	if trimmed.is_empty() {
		return;
	}
	if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
		out.push(value);
	}
}

// ── Finalize (elide → migrate → blob → strong parse) ─────────────────────────

fn finalize(
	mut values: Vec<Value>,
	title_slot: Option<TitleSlot>,
	blobs_dir: &Path,
) -> Result<LoadedSession> {
	if values.is_empty() {
		bail!("session file has no entries");
	}

	// Order matters (mirrors `loadEntriesFromFile` →
	// `loadSessionMessagesReadOnly`): elide runs on the raw, pre-migration entries
	// — a v1 file has no ids yet so its active-branch set is empty and nothing is
	// elided.
	elide_superseded_compaction_entries(&mut values);
	migrate_to_current_version(&mut values);
	resolve_blob_refs_in_entries(&mut values, blobs_dir);

	// The first logical entry is the header.
	let header_value = values.remove(0);
	let header: SessionHeader =
		serde_json::from_value(header_value).context("parsing session header")?;
	if header.kind != "session" {
		bail!("first entry is not a session header (type = {:?})", header.kind);
	}

	let mut entries: Vec<SessionEntry> = Vec::with_capacity(values.len());
	for value in values {
		// Untagged `SessionEntry` never fails: an unmodeled type falls to `Unknown`.
		let entry: SessionEntry = serde_json::from_value(value).context("parsing session entry")?;
		entries.push(entry);
	}

	Ok(LoadedSession { header, title_slot, entries })
}

// ── Superseded-compaction elision (Value level, pre-migration) ───────────────

fn value_field<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
	v.get(key).and_then(Value::as_str)
}

/// Ids on the active branch: leaf→root from the last value.
fn collect_active_branch_ids(values: &[Value]) -> BTreeSet<String> {
	let by_id: std::collections::BTreeMap<&str, usize> = values
		.iter()
		.enumerate()
		.filter_map(|(i, v)| value_field(v, "id").map(|id| (id, i)))
		.collect();
	let mut branch: BTreeSet<String> = BTreeSet::new();
	let mut cursor = values.last();
	while let Some(value) = cursor {
		let Some(id) = value_field(value, "id") else {
			break;
		};
		if branch.contains(id) {
			break;
		}
		branch.insert(id.to_string());
		cursor = value_field(value, "parentId").and_then(|p| by_id.get(p).map(|&i| &values[i]));
	}
	branch
}

/// Replace the summary of every active-branch compaction except the newest with
/// the elision placeholder (session-loader.ts:93). Mutates in place.
fn elide_superseded_compaction_entries(values: &mut [Value]) {
	let branch_ids = collect_active_branch_ids(values);
	let mut previous_compaction: Option<usize> = None;
	for i in 0..values.len() {
		if value_field(&values[i], "type") != Some("compaction") {
			continue;
		}
		let on_branch = value_field(&values[i], "id").is_some_and(|id| branch_ids.contains(id));
		if !on_branch {
			continue;
		}
		if let Some(prev) = previous_compaction {
			elide_compaction_summary(&mut values[prev]);
		}
		previous_compaction = Some(i);
	}
}

fn elide_compaction_summary(entry: &mut Value) {
	let Some(obj) = entry.as_object_mut() else {
		return;
	};
	let already = obj.get("summary").and_then(Value::as_str) == Some(ELIDED_COMPACTION_SUMMARY)
		&& obj.get("shortSummary").and_then(Value::as_str) == Some(ELIDED_COMPACTION_SHORT_SUMMARY)
		&& obj.get("preserveData").is_none();
	if already {
		return;
	}
	obj.insert("summary".into(), json!(ELIDED_COMPACTION_SUMMARY));
	obj.insert("shortSummary".into(), json!(ELIDED_COMPACTION_SHORT_SUMMARY));
	obj.remove("preserveData");
}

// ── Version migration (B4) ───────────────────────────────────────────────────

/// Bring `< 3` sessions to v3 in place. Mirror of
/// `session-migrations.ts::migrateToCurrentVersion`.
fn migrate_to_current_version(values: &mut [Value]) -> bool {
	let version = values
		.iter()
		.find(|v| value_field(v, "type") == Some("session"))
		.and_then(|v| v.get("version").and_then(Value::as_u64))
		.unwrap_or(1) as u32;
	if version >= crate::entries::CURRENT_SESSION_VERSION {
		return false;
	}
	if version < 2 {
		migrate_v1_to_v2(values);
	}
	if version < 3 {
		migrate_v2_to_v3(values);
	}
	true
}

/// v1 → v2: attach the id/parentId linear tree and convert a compaction's
/// `firstKeptEntryIndex` (absolute index into the entry array, header at 0) to
/// `firstKeptEntryId`. Entry ids are deterministic 8-hex from a counter — same
/// shape as `generateId`'s `randomUUID().slice(-8)`, stable for tests.
fn migrate_v1_to_v2(values: &mut [Value]) {
	let mut prev: Option<String> = None;
	let mut counter: u64 = 0;
	let mut index_to_id: Vec<Option<String>> = vec![None; values.len()];

	for (i, entry) in values.iter_mut().enumerate() {
		if value_field(entry, "type") == Some("session") {
			if let Some(obj) = entry.as_object_mut() {
				obj.insert("version".into(), json!(2));
			}
			continue;
		}
		let id = format!("{:08x}", counter & 0xffff_ffff);
		counter += 1;
		if let Some(obj) = entry.as_object_mut() {
			obj.insert("id".into(), json!(id));
			obj.insert("parentId".into(), prev.as_ref().map_or(Value::Null, |p| json!(p)));
		}
		index_to_id[i] = Some(id.clone());
		prev = Some(id);
	}

	for entry in values.iter_mut() {
		if value_field(entry, "type") != Some("compaction") {
			continue;
		}
		let Some(idx) = entry.get("firstKeptEntryIndex").and_then(Value::as_u64) else {
			continue;
		};
		let target = index_to_id.get(idx as usize).and_then(Clone::clone);
		if let Some(obj) = entry.as_object_mut() {
			if let Some(tid) = target {
				obj.insert("firstKeptEntryId".into(), json!(tid));
			}
			obj.remove("firstKeptEntryIndex");
		}
	}
}

/// v2 → v3: rename a message entry's `hookMessage` role to `custom`.
fn migrate_v2_to_v3(values: &mut [Value]) {
	for entry in values.iter_mut() {
		match value_field(entry, "type") {
			Some("session") => {
				if let Some(obj) = entry.as_object_mut() {
					obj.insert("version".into(), json!(3));
				}
			},
			Some("message") => {
				if let Some(msg) = entry.get_mut("message")
					&& msg.get("role").and_then(Value::as_str) == Some("hookMessage")
					&& let Some(obj) = msg.as_object_mut()
				{
					obj.insert("role".into(), json!("custom"));
				}
			},
			_ => {},
		}
	}
}

// ── Blob dereference (B5) ────────────────────────────────────────────────────

const BLOB_PREFIX: &str = "blob:sha256:";

fn is_blob_ref(s: &str) -> bool {
	s.starts_with(BLOB_PREFIX)
}

fn parse_blob_ref(s: &str) -> Option<&str> {
	s.strip_prefix(BLOB_PREFIX)
}

/// `{type:"image", data:"…"}` — an image content block carrying inline data.
fn is_image_block(v: &Value) -> bool {
	value_field(v, "type") == Some("image") && v.get("data").and_then(Value::as_str).is_some()
}

/// Read a blob back to base64 (mirror of `resolveImageData`). Missing blob →
/// keep the ref + warn.
fn resolve_image_data(blobs_dir: &Path, data: &str) -> String {
	resolve_blob(blobs_dir, data, false)
}

/// Read a blob back to its original UTF-8 string (mirror of
/// `resolveImageDataUrl`). Missing blob → keep the ref + warn.
fn resolve_image_data_url(blobs_dir: &Path, data: &str) -> String {
	resolve_blob(blobs_dir, data, true)
}

fn resolve_blob(blobs_dir: &Path, data: &str, as_utf8: bool) -> String {
	let Some(hash) = parse_blob_ref(data) else {
		return data.to_string();
	};
	match fs::read(blobs_dir.join(hash)) {
		Ok(bytes) if as_utf8 => String::from_utf8_lossy(&bytes).into_owned(),
		Ok(bytes) => {
			use base64::Engine as _;
			base64::engine::general_purpose::STANDARD.encode(bytes)
		},
		Err(_) => {
			eprintln!("[pi-session] blob not found for reference {hash}; keeping ref verbatim");
			data.to_string()
		},
	}
}

/// Resolve blob refs across every non-header entry. Mirror of
/// `resolveBlobRefsInEntries` + `resolvePersistedBlobRefs`.
fn resolve_blob_refs_in_entries(values: &mut [Value], blobs_dir: &Path) {
	for entry in values.iter_mut() {
		if value_field(entry, "type") == Some("session") {
			continue;
		}
		resolve_persisted_blob_refs(entry, blobs_dir, None);
	}
}

fn resolve_persisted_blob_refs(value: &mut Value, blobs_dir: &Path, key: Option<&str>) {
	// Image data payload in a resolve position: `content` image block or `images`.
	if should_resolve_image_payload(value, key) {
		if let Some(data) = value.get("data").and_then(Value::as_str) {
			let resolved = resolve_image_data(blobs_dir, data);
			if let Some(obj) = value.as_object_mut() {
				obj.insert("data".into(), json!(resolved));
			}
		}
		return;
	}

	match value {
		Value::Array(items) => {
			for item in items.iter_mut() {
				resolve_persisted_blob_refs(item, blobs_dir, key);
			}
		},
		Value::Object(_) => {
			// image_generation_call result → base64.
			if value_field(value, "type") == Some("image_generation_call")
				&& let Some(result) = value.get("result").and_then(Value::as_str)
				&& is_blob_ref(result)
			{
				let resolved = resolve_image_data(blobs_dir, result);
				if let Some(obj) = value.as_object_mut() {
					obj.insert("result".into(), json!(resolved));
				}
			}
			// image_url provider transport field → original data URL.
			if let Some(url) = value.get("image_url").and_then(Value::as_str)
				&& is_blob_ref(url)
			{
				let resolved = resolve_image_data_url(blobs_dir, url);
				if let Some(obj) = value.as_object_mut() {
					obj.insert("image_url".into(), json!(resolved));
				}
			}
			let child_keys: Vec<String> = value
				.as_object()
				.map(|o| o.keys().cloned().collect())
				.unwrap_or_default();
			for child_key in child_keys {
				if let Some(child) = value.get_mut(&child_key) {
					resolve_persisted_blob_refs(child, blobs_dir, Some(&child_key));
				}
			}
		},
		_ => {},
	}
}

fn should_resolve_image_payload(value: &Value, key: Option<&str>) -> bool {
	let Some(data) = value.get("data").and_then(Value::as_str) else {
		return false;
	};
	if !is_blob_ref(data) {
		return false;
	}
	let is_image_mime = value
		.get("mimeType")
		.and_then(Value::as_str)
		.is_some_and(|m| m.to_ascii_lowercase().starts_with("image/"));
	let is_payload = is_image_block(value) || is_image_mime;
	if !is_payload {
		return false;
	}
	(key == Some("content") && is_image_block(value)) || key == Some("images")
}

#[cfg(test)]
mod tests {
	use super::*;

	fn typ(v: &Value) -> Option<&str> {
		v.get("type").and_then(Value::as_str)
	}

	#[test]
	fn v1_to_v2_assigns_id_parent_chain_and_first_kept_id() {
		// header + user + assistant + compaction(firstKeptEntryIndex=2) + user.
		let mut values = vec![
			json!({"type":"session","id":"s","timestamp":"t","cwd":"/c"}),
			json!({"type":"message","message":{"role":"user","content":"a"}}),
			json!({"type":"message","message":{"role":"assistant","content":[]}}),
			json!({"type":"compaction","summary":"s","firstKeptEntryIndex":2,"tokensBefore":1}),
			json!({"type":"message","message":{"role":"user","content":"b"}}),
		];
		assert!(migrate_to_current_version(&mut values));

		// Header bumped to v3, no id assigned.
		assert_eq!(values[0]["version"], json!(3));
		assert!(values[0].get("id").is_some_and(|v| v == "s"));

		// Deterministic 8-hex ids from a 0 counter, linear parentId chain.
		assert_eq!(values[1]["id"], json!("00000000"));
		assert_eq!(values[1]["parentId"], Value::Null);
		assert_eq!(values[2]["id"], json!("00000001"));
		assert_eq!(values[2]["parentId"], json!("00000000"));
		assert_eq!(values[3]["parentId"], json!("00000001"));
		assert_eq!(values[4]["id"], json!("00000003"));

		// firstKeptEntryIndex(2) → the id of values[2], and the index is removed.
		assert_eq!(values[3]["firstKeptEntryId"], json!("00000001"));
		assert!(values[3].get("firstKeptEntryIndex").is_none());
	}

	#[test]
	fn v2_to_v3_renames_hook_message_role_to_custom() {
		let mut values = vec![
			json!({"type":"session","version":2,"id":"s","timestamp":"t","cwd":"/c"}),
			json!({"type":"message","id":"a","parentId":null,"timestamp":"t",
				"message":{"role":"hookMessage","content":"hi"}}),
			json!({"type":"message","id":"b","parentId":"a","timestamp":"t",
				"message":{"role":"user","content":"u"}}),
		];
		assert!(migrate_to_current_version(&mut values));
		assert_eq!(values[0]["version"], json!(3));
		assert_eq!(values[1]["message"]["role"], json!("custom"));
		assert_eq!(values[2]["message"]["role"], json!("user"));
	}

	#[test]
	fn v3_is_a_no_op() {
		let mut values = vec![
			json!({"type":"session","version":3,"id":"s","timestamp":"t","cwd":"/c"}),
			json!({"type":"message","id":"a","parentId":null,"timestamp":"t",
				"message":{"role":"user","content":"u"}}),
		];
		let before = values.clone();
		assert!(!migrate_to_current_version(&mut values));
		assert_eq!(values, before);
	}

	#[test]
	fn elide_runs_before_migration_so_v1_is_untouched() {
		// Two v1 compactions (no ids): elide must skip them (empty branch set),
		// leaving both summaries intact — matches loadEntriesFromFile order.
		let mut values = vec![
			json!({"type":"session","id":"s","timestamp":"t","cwd":"/c"}),
			json!({"type":"compaction","summary":"first","firstKeptEntryIndex":0,"tokensBefore":1}),
			json!({"type":"compaction","summary":"second","firstKeptEntryIndex":0,"tokensBefore":1}),
		];
		elide_superseded_compaction_entries(&mut values);
		assert_eq!(values[1]["summary"], json!("first"));
		assert_eq!(values[2]["summary"], json!("second"));
	}

	#[test]
	fn elide_supersedes_earlier_v3_compaction_on_branch() {
		let mut values = vec![
			json!({"type":"session","id":"s","timestamp":"t","cwd":"/c"}),
			json!({"type":"compaction","id":"c1","parentId":null,"summary":"first","tokensBefore":1}),
			json!({"type":"compaction","id":"c2","parentId":"c1","summary":"second","tokensBefore":1}),
		];
		elide_superseded_compaction_entries(&mut values);
		assert_eq!(values[1]["summary"], json!(ELIDED_COMPACTION_SUMMARY));
		assert_eq!(values[2]["summary"], json!("second"));
	}

	#[test]
	fn blob_ref_resolves_to_base64_and_missing_ref_is_kept() {
		use base64::Engine as _;
		let dir = std::env::temp_dir().join(format!("pi-session-blob-unit-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let bytes = b"\x89PNG\r\n\x1a\nhello-blob";
		std::fs::write(dir.join("abc123"), bytes).unwrap();

		let mut present = json!({"type":"message","message":{"role":"user",
			"content":[{"type":"image","data":"blob:sha256:abc123","mimeType":"image/png"}]}});
		resolve_persisted_blob_refs(&mut present, &dir, None);
		let expected = base64::engine::general_purpose::STANDARD.encode(bytes);
		assert_eq!(present["message"]["content"][0]["data"], json!(expected));

		let mut missing = json!({"type":"message","message":{"role":"user",
			"content":[{"type":"image","data":"blob:sha256:nope","mimeType":"image/png"}]}});
		resolve_persisted_blob_refs(&mut missing, &dir, None);
		assert_eq!(missing["message"]["content"][0]["data"], json!("blob:sha256:nope"));

		let _ = std::fs::remove_dir_all(&dir);
	}

	#[test]
	fn streaming_and_wholefile_agree_via_public_parse() {
		// A small in-memory body goes through the whole-file path; assert the
		// header/entry split is stable (the ≥8MiB streaming path is covered by the
		// integration test).
		let content = format!(
			"{}\n{}\n{}\n",
			json!({"type":"session","version":3,"id":"s","timestamp":"t","cwd":"/c"}),
			json!({"type":"message","id":"a","parentId":null,"timestamp":"t","message":{"role":"user","content":"hi","timestamp":1}}),
			"garbage-not-json",
		);
		let loaded = parse_session_content(&content).unwrap();
		assert_eq!(loaded.header.id, "s");
		assert_eq!(loaded.entries.len(), 1, "malformed line skipped");
		assert_eq!(typ(&serde_json::to_value(&loaded.entries[0]).unwrap()), Some("message"));
	}
}
