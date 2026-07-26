//! N-API shell over the pure `pi-diff` engine.
//!
//! Line, line-array, and word diffs plus a unified-patch hunk builder, all
//! producing byte-identical output to the `diff` npm package (jsdiff v9) under
//! its default options. All diff logic lives in the [`pi_diff`] crate; this
//! module owns only the N-API surface: the `#[napi(object)]` DTOs, the
//! [`Utf16String`] boundary, and the conversions to and from [`pi_diff`].
//!
//! Everything operates on UTF-16 code units end to end — [`Utf16String`] at the
//! N-API boundary, `&[u16]` inside the engine — which is the exact value space
//! of JS strings. Ill-formed input (unpaired surrogates) is legal content that
//! diffs code-unit-for-code-unit like jsdiff, so callers never need a JS
//! fallback, and no UTF-8 conversion happens in either direction.
//!
//! # Example
//! ```ignore
//! // JS: native.diffLines("a\nb\n", "a\nc\n")
//! //   -> [{ value: "a\n", count: 1, added: false, removed: false },
//! //       { value: "b\n", count: 1, added: false, removed: true },
//! //       { value: "c\n", count: 1, added: true, removed: false }]
//! ```

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// One jsdiff change object: a run of added, removed, or common tokens.
#[napi(object)]
pub struct DiffChange {
	/// Joined token text for this run (lines keep their `\n` terminators).
	pub value:   Utf16String,
	/// Number of tokens in this run.
	pub count:   u32,
	/// True when this run exists only in the new text.
	pub added:   bool,
	/// True when this run exists only in the old text.
	pub removed: bool,
}

impl From<pi_diff::DiffChange> for DiffChange {
	fn from(change: pi_diff::DiffChange) -> Self {
		Self {
			value:   change.value.into(),
			count:   change.count,
			added:   change.added,
			removed: change.removed,
		}
	}
}

/// A change run without its token text, for callers that only need counts.
#[napi(object)]
pub struct DiffRun {
	/// Number of tokens in this run.
	pub count:   u32,
	/// True when this run exists only in the new text.
	pub added:   bool,
	/// True when this run exists only in the old text.
	pub removed: bool,
}

impl From<pi_diff::DiffRun> for DiffRun {
	fn from(run: pi_diff::DiffRun) -> Self {
		Self { count: run.count, added: run.added, removed: run.removed }
	}
}

/// One hunk of a unified diff, matching jsdiff `structuredPatch` hunks.
#[napi(object)]
pub struct PatchHunk {
	/// 1-based first line of the hunk in the old text.
	pub old_start: u32,
	/// Number of old-text lines covered by the hunk.
	pub old_lines: u32,
	/// 1-based first line of the hunk in the new text.
	pub new_start: u32,
	/// Number of new-text lines covered by the hunk.
	pub new_lines: u32,
	/// Hunk body: `+`/`-`/` `-prefixed lines without trailing newlines, plus
	/// `\ No newline at end of file` markers where applicable.
	pub lines:     Vec<Utf16String>,
}

impl From<pi_diff::PatchHunk> for PatchHunk {
	fn from(hunk: pi_diff::PatchHunk) -> Self {
		Self {
			old_start: hunk.old_start,
			old_lines: hunk.old_lines,
			new_start: hunk.new_start,
			new_lines: hunk.new_lines,
			lines:     hunk.lines.into_iter().map(Utf16String::from).collect(),
		}
	}
}

/// Line diff with jsdiff `diffLines(oldText, newText)` semantics (default
/// options). Change values keep line terminators, and common runs are joined
/// from the new text.
#[napi]
pub fn diff_lines(old_text: Utf16String, new_text: Utf16String) -> Vec<DiffChange> {
	pi_diff::diff_lines(&old_text, &new_text)
		.into_iter()
		.map(DiffChange::from)
		.collect()
}

/// Diff `oldText.split("\n")` against `newText.split("\n")` with jsdiff
/// `diffArrays` semantics (exact code-unit equality, empty lines preserved),
/// returning only run lengths.
///
/// Callers that map line numbers — like hashline recovery — need the counts,
/// not another copy of the text.
#[napi]
pub fn diff_line_runs(old_text: Utf16String, new_text: Utf16String) -> Vec<DiffRun> {
	pi_diff::diff_line_runs(&old_text, &new_text)
		.into_iter()
		.map(DiffRun::from)
		.collect()
}

/// Unified-diff hunks with jsdiff
/// `structuredPatch(_, _, oldText, newText, _, _, { context }).hunks`
/// semantics. `context` defaults to 4 like jsdiff.
#[napi]
pub fn structured_patch_hunks(
	old_text: Utf16String,
	new_text: Utf16String,
	context: Option<u32>,
) -> Vec<PatchHunk> {
	pi_diff::structured_patch_hunks(&old_text, &new_text, context)
		.into_iter()
		.map(PatchHunk::from)
		.collect()
}

/// Word diff with jsdiff `diffWords(oldText, newText)` semantics (default
/// options).
///
/// Tokens carry surrounding whitespace, equality ignores it, and the
/// post-pass dedupes whitespace across change boundaries.
#[napi]
pub fn diff_words(old_text: Utf16String, new_text: Utf16String) -> Vec<DiffChange> {
	pi_diff::diff_words(&old_text, &new_text)
		.into_iter()
		.map(DiffChange::from)
		.collect()
}
