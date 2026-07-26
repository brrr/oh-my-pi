//! Append-only session writer. Rust mirror of the minimal write face of
//! `packages/coding-agent/src/session/session-manager.ts`.
//!
//! `create` mints the session (title slot + v3 header), then each `append_*`
//! call attaches a fresh entry as a child of the current leaf and advances the
//! leaf (`#freshEntryFields`), serializing one `stringify(entry) + "\n"` line.
//! Writes are strictly append-only: existing lines are never rewritten, so any
//! unknown/opaque entry a resumed file already carried is preserved
//! byte-for-byte simply by never touching it.
//!
//! Deferred vs the TS manager: the lazy "no file until first assistant message"
//! gate, blob externalization, superseded-compaction rewrite, forking, and
//! title mutation. Entry ids are generated as collision-checked 8-hex strings
//! (the TS `generateId` shape `crypto.randomUUID().slice(-8)`) from a
//! per-writer counter rather than random UUID slices — same shape,
//! deterministic for tests.

use std::{
	collections::BTreeSet,
	fs::{File, OpenOptions},
	io::Write,
	path::{Path, PathBuf},
	time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};

use crate::{
	entries::{
		CURRENT_SESSION_VERSION, CompactionEntry, KnownEntry, Message, MessageEntry,
		ModelChangeEntry, SessionHeader,
	},
	loader::load_entries_from_file,
	time::unix_ms_to_iso,
};

/// An open, append-only session file.
pub struct SessionWriter {
	file:       File,
	path:       PathBuf,
	ids:        BTreeSet<String>,
	leaf:       Option<String>,
	id_counter: u64,
}

fn now_iso() -> String {
	let ms = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map_or(0, |d| d.as_millis() as i64);
	unix_ms_to_iso(ms)
}

/// `fileSafeTimestamp` (session-manager.ts:90): `iso.replace(/[:.]/g, "-")`.
fn file_safe_timestamp(iso: &str) -> String {
	iso.replace([':', '.'], "-")
}

/// A uuid-v7-shaped session id derived from the mint instant. The loader only
/// requires a string id; the shape mirrors `Bun.randomUUIDv7()` for realism.
fn mint_session_id(ms: i64) -> String {
	let ms = (ms as u64) & 0xffff_ffff_ffff;
	format!("{:08x}-{:04x}-7000-8000-{:012x}", (ms >> 16) & 0xffff_ffff, ms & 0xffff, ms)
}

impl SessionWriter {
	/// Create a fresh persisted session under `dir` and write the title slot +
	/// v3 header.
	pub fn create(cwd: &str, dir: impl AsRef<Path>) -> Result<Self> {
		let dir = dir.as_ref();
		std::fs::create_dir_all(dir)
			.with_context(|| format!("creating session dir {}", dir.display()))?;
		let ms = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.map_or(0, |d| d.as_millis() as i64);
		let timestamp = unix_ms_to_iso(ms);
		let session_id = mint_session_id(ms);
		let path = dir.join(format!("{}_{}.jsonl", file_safe_timestamp(&timestamp), session_id));

		let header = SessionHeader {
			kind: "session".into(),
			version: Some(CURRENT_SESSION_VERSION),
			id: session_id,
			title: None,
			title_source: None,
			timestamp: timestamp.clone(),
			cwd: cwd.to_string(),
			parent_session: None,
			provider_prompt_cache_key: None,
			extra: serde_json::Map::new(),
		};

		let mut body = crate::title_slot::serialize_title_slot("", None, &timestamp);
		body.push_str(&serde_json::to_string(&header).context("serializing header")?);
		body.push('\n');

		let mut file = OpenOptions::new()
			.create(true)
			.truncate(true)
			.write(true)
			.open(&path)
			.with_context(|| format!("creating session file {}", path.display()))?;
		file
			.write_all(body.as_bytes())
			.context("writing session header")?;

		Ok(Self { file, path, ids: BTreeSet::new(), leaf: None, id_counter: 0 })
	}

	/// Resume appending to an existing session file (leaf continues from the
	/// last entry).
	pub fn resume(path: impl AsRef<Path>) -> Result<Self> {
		let path = path.as_ref().to_path_buf();
		let loaded = load_entries_from_file(&path)?;
		let ids: BTreeSet<String> = loaded
			.entries
			.iter()
			.filter_map(|e| e.id().map(str::to_string))
			.collect();
		let leaf = loaded
			.entries
			.last()
			.and_then(|e| e.id().map(str::to_string));
		let file = OpenOptions::new()
			.append(true)
			.open(&path)
			.with_context(|| format!("opening session file for append {}", path.display()))?;
		Ok(Self { file, path, ids, leaf, id_counter: 0 })
	}

	/// The session file path.
	pub fn path(&self) -> &Path {
		&self.path
	}

	/// The current leaf entry id (`None` before any entry is appended).
	pub fn leaf_id(&self) -> Option<&str> {
		self.leaf.as_deref()
	}

	/// Fresh `{ id, parentId, timestamp }` for a new child of the current leaf.
	fn fresh_entry_fields(&mut self) -> (String, Option<String>, String) {
		let id = loop {
			let candidate = format!("{:08x}", self.id_counter & 0xffff_ffff);
			self.id_counter += 1;
			if !self.ids.contains(&candidate) {
				break candidate;
			}
		};
		(id, self.leaf.clone(), now_iso())
	}

	fn record(&mut self, entry: &KnownEntry, id: &str) -> Result<()> {
		let mut line = serde_json::to_string(entry).context("serializing entry")?;
		line.push('\n');
		self
			.file
			.write_all(line.as_bytes())
			.context("appending entry")?;
		self.ids.insert(id.to_string());
		self.leaf = Some(id.to_string());
		Ok(())
	}

	/// Append a message as a child of the current leaf; returns the entry id.
	pub fn append_message(&mut self, message: Message) -> Result<String> {
		let (id, parent_id, timestamp) = self.fresh_entry_fields();
		let entry =
			KnownEntry::Message(MessageEntry { id: id.clone(), parent_id, timestamp, message });
		self.record(&entry, &id)?;
		Ok(id)
	}

	/// Append a `model_change` entry; `role` defaults to `"default"` when
	/// `None`.
	pub fn append_model_change(&mut self, model: &str, role: Option<&str>) -> Result<String> {
		let (id, parent_id, timestamp) = self.fresh_entry_fields();
		let entry = KnownEntry::ModelChange(ModelChangeEntry {
			id: id.clone(),
			parent_id,
			timestamp,
			model: model.to_string(),
			role: role.map(str::to_string),
		});
		self.record(&entry, &id)?;
		Ok(id)
	}

	/// Append a `compaction` entry.
	pub fn append_compaction(
		&mut self,
		summary: &str,
		short_summary: Option<&str>,
		first_kept_entry_id: &str,
		tokens_before: i64,
	) -> Result<String> {
		let (id, parent_id, timestamp) = self.fresh_entry_fields();
		let entry = KnownEntry::Compaction(CompactionEntry {
			id: id.clone(),
			parent_id,
			timestamp,
			summary: summary.to_string(),
			short_summary: short_summary.map(str::to_string),
			first_kept_entry_id: first_kept_entry_id.to_string(),
			tokens_before,
			details: None,
			preserve_data: None,
			from_extension: None,
			warning: None,
		});
		self.record(&entry, &id)?;
		Ok(id)
	}

	/// Flush buffered writes to the OS.
	pub fn flush(&mut self) -> Result<()> {
		self.file.flush().context("flushing session file")
	}
}
