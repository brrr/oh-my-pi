//! `pi-session` — pure-Rust v3 session journal store for the headless omp
//! control plane (WP-1.3).
//!
//! Reads and writes the append-only JSONL session format defined by
//! `packages/coding-agent/src/session/` and rebuilds the read-only LLM message
//! view (`build_session_context`) with byte-parity against the TS
//! `loadSessionMessagesReadOnly`. The message model itself is reused from
//! [`pi_ai::message`] (already 1:1 with the TS types) rather than redefined.
//!
//! # Scope (v3 only + read/write both directions)
//!
//! In scope: the fixed-width [`title_slot`], the strongly-typed [`entries`]
//! (with opaque passthrough for every other type), the v3 [`loader`] +
//! superseded-compaction elision, the non-transcript [`context`] rebuild, and
//! the append-only [`writer`].
//!
//! **Deferred** (registered in each module's doc, not silently dropped):
//! version `< 3` migration, `blob:sha256:…` dereference (refs left verbatim),
//! the ≥8 MiB streaming loader, transcript mode, provider remote-compaction
//! replacement history, `retryRecovery` skip, and `branchSummary` message
//! synthesis. A `compaction` hook exists only as the loader's elision pass;
//! live compaction generation is a later WP.

pub mod context;
pub mod entries;
pub mod loader;
pub mod time;
pub mod title_slot;
pub mod writer;

pub use context::{
	BranchSummaryMessage, CompactionSummaryMessage, ContextMessage, CustomMessage, SessionContext,
	build_session_context,
};
pub use entries::{
	CURRENT_SESSION_VERSION, CompactionEntry, KnownEntry, SESSION_TITLE_SLOT_BYTES, SessionEntry,
	SessionHeader, message_from_json,
};
pub use loader::{
	LoadedSession, load_entries_from_file, load_session_context, load_session_messages,
	parse_session_content,
};
pub use title_slot::{TitleSlot, parse_title_slot_line, serialize_title_slot};
pub use writer::SessionWriter;
