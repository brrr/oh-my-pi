//! Session file reader. Rust mirror of the v3 path of
//! `packages/coding-agent/src/session/session-loader.ts` + the leaf-selection
//! of `session-context.ts`.
//!
//! Read the whole file, peel the optional fixed-width title slot, lenient-parse
//! the JSONL body (malformed lines skipped), validate the header (`type ==
//! "session"`, string `id`, **version ≥ 3** — anything lower is a hard error;
//! migration is deferred per U-omp-29), then apply superseded-compaction
//! elision in memory.
//!
//! Deferred (registered here): version `< 3` migration, `blob:sha256:…`
//! dereference (refs are left verbatim), and the ≥8 MiB streaming loader (the
//! whole file is read at once).

use std::{collections::BTreeSet, fs, path::Path};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::{
	context::{ContextMessage, SessionContext, build_session_context},
	entries::{CURRENT_SESSION_VERSION, KnownEntry, SessionEntry, SessionHeader},
	title_slot::{TitleSlot, parse_title_slot_line},
};

const ELIDED_COMPACTION_SUMMARY: &str =
	"[Superseded compaction summary elided during session load]";
const ELIDED_COMPACTION_SHORT_SUMMARY: &str = "Superseded compaction elided";

/// A loaded session: the header, the folded title slot (if any), and the
/// logical entry list (header excluded).
#[derive(Debug, Clone)]
pub struct LoadedSession {
	pub header:     SessionHeader,
	pub title_slot: Option<TitleSlot>,
	pub entries:    Vec<SessionEntry>,
}

/// Load and validate a v3 session file.
pub fn load_entries_from_file(path: impl AsRef<Path>) -> Result<LoadedSession> {
	let path = path.as_ref();
	let content = fs::read_to_string(path)
		.with_context(|| format!("reading session file {}", path.display()))?;
	parse_session_content(&content)
}

/// Parse a full physical session body (title slot + JSONL). Exposed for tests
/// that construct content in memory.
pub fn parse_session_content(content: &str) -> Result<LoadedSession> {
	let raw: Vec<&str> = content.split('\n').collect();

	// Peel the optional fixed-width title slot from the first physical line.
	let (title_slot, body_start) = match raw.first() {
		Some(first) => match parse_title_slot_line(first.trim()) {
			Some(slot) => (Some(slot), 1),
			None => (None, 0),
		},
		None => (None, 0),
	};

	// Lenient JSONL parse: skip blank lines and any line that is not valid JSON.
	let mut values: Vec<Value> = Vec::new();
	for line in &raw[body_start..] {
		let trimmed = line.trim();
		if trimmed.is_empty() {
			continue;
		}
		if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
			values.push(value);
		}
	}

	if values.is_empty() {
		bail!("session file has no entries");
	}

	// The first logical entry is the header.
	let header_value = values.remove(0);
	let header: SessionHeader =
		serde_json::from_value(header_value).context("parsing session header")?;
	if header.kind != "session" {
		bail!("first entry is not a session header (type = {:?})", header.kind);
	}
	let version = header.version.unwrap_or(1);
	if version < CURRENT_SESSION_VERSION {
		bail!(
			"unsupported session version {version}: only v{CURRENT_SESSION_VERSION} is supported \
			 (migration deferred, U-omp-29)"
		);
	}

	let mut entries: Vec<SessionEntry> = Vec::with_capacity(values.len());
	for value in values {
		// Untagged `SessionEntry` never fails: an unmodeled type falls to `Unknown`.
		let entry: SessionEntry = serde_json::from_value(value).context("parsing session entry")?;
		entries.push(entry);
	}

	elide_superseded_compaction_entries(&mut entries);

	Ok(LoadedSession { header, title_slot, entries })
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

/// Ids on the active branch: leaf→root from the last entry.
fn collect_active_branch_ids(entries: &[SessionEntry]) -> BTreeSet<String> {
	let by_id: std::collections::BTreeMap<&str, usize> = entries
		.iter()
		.enumerate()
		.filter_map(|(i, e)| e.id().map(|id| (id, i)))
		.collect();
	let mut branch: BTreeSet<String> = BTreeSet::new();
	let mut cursor = entries.last();
	while let Some(entry) = cursor {
		let Some(id) = entry.id() else { break };
		if branch.contains(id) {
			break;
		}
		branch.insert(id.to_string());
		cursor = entry
			.parent_id()
			.and_then(|p| by_id.get(p).map(|&i| &entries[i]));
	}
	branch
}

/// Replace the summary of every active-branch compaction except the newest with
/// the elision placeholder (session-loader.ts:93). Mutates in place.
fn elide_superseded_compaction_entries(entries: &mut [SessionEntry]) {
	let branch_ids = collect_active_branch_ids(entries);
	let mut previous_compaction: Option<usize> = None;
	for i in 0..entries.len() {
		if entries[i].entry_type() != "compaction" {
			continue;
		}
		let on_branch = entries[i].id().is_some_and(|id| branch_ids.contains(id));
		if !on_branch {
			continue;
		}
		if let Some(prev) = previous_compaction {
			elide_compaction_summary(&mut entries[prev]);
		}
		previous_compaction = Some(i);
	}
}

fn elide_compaction_summary(entry: &mut SessionEntry) {
	let SessionEntry::Known(KnownEntry::Compaction(comp)) = entry else {
		return;
	};
	let already = comp.summary == ELIDED_COMPACTION_SUMMARY
		&& comp.short_summary.as_deref() == Some(ELIDED_COMPACTION_SHORT_SUMMARY)
		&& comp.preserve_data.is_none();
	if already {
		return;
	}
	comp.summary = ELIDED_COMPACTION_SUMMARY.to_string();
	comp.short_summary = Some(ELIDED_COMPACTION_SHORT_SUMMARY.to_string());
	comp.preserve_data = None;
}
