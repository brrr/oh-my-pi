//! `edit` tool — replace mode with exact matching only.
//!
//! Ported from `packages/coding-agent/src/edit/modes/replace.ts`
//! (`executeReplaceSingle`), `packages/coding-agent/src/edit/diff.ts`
//! (`replaceText`, `formatOccurrenceMatchError`), and
//! `packages/coding-agent/src/edit/index.ts` (`executeSinglePathEntries`).
//!
//! Deferred vs. TS: the 9-strategy fuzzy fallback chain (`PI_EDIT_FUZZY`
//! pinned off), `adjustIndentation` (exact matches never shift indentation),
//! the closest-match preview in `EditMatchError` (fuzzy scoring is not
//! ported, so the no-closest message is always used — goldens pin cases where
//! the TS closest is also absent), `hashline`/`patch`/`apply_patch` modes, LSP
//! writethrough/diagnostics, plan-mode guard, ACP bridge, notebook files,
//! diff generation (`details.diff` is UI-only), and snapshot recording.

use std::path::PathBuf;

use serde_json::{Value, json};

use super::paths::resolve_to_cwd;
use crate::tool::{Tool, ToolError, ToolResult};

/// `OCCURRENCE_PREVIEW_CONTEXT` (replace.ts:168).
const OCCURRENCE_PREVIEW_CONTEXT: usize = 5;
/// `OCCURRENCE_PREVIEW_MAX_LEN` (replace.ts:171).
const OCCURRENCE_PREVIEW_MAX_LEN: usize = 80;
/// `MAX_RECORDED_MATCHES` (replace.ts:174) == `MAX_OCCURRENCE_PREVIEWS`
/// (diff.ts:436).
const MAX_RECORDED_MATCHES: usize = 5;

fn normalize_to_lf(text: &str) -> String {
	text.replace("\r\n", "\n").replace('\r', "\n")
}

/// `stripBom` (hashline normalize.ts:36).
fn strip_bom(content: &str) -> (&'static str, &str) {
	content
		.strip_prefix('\u{FEFF}')
		.map_or(("", content), |text| ("\u{FEFF}", text))
}

/// `detectLineEnding` (hashline normalize.ts:10).
fn detect_line_ending(content: &str) -> &'static str {
	let crlf_idx = content.find("\r\n");
	let lf_idx = content.find('\n');
	match (lf_idx, crlf_idx) {
		(None, _) | (_, None) => "\n",
		(Some(lf), Some(crlf)) if crlf < lf => "\r\n",
		_ => "\n",
	}
}

/// `restoreLineEndings` (hashline normalize.ts:24).
fn restore_line_endings(text: &str, ending: &str) -> String {
	if ending == "\r\n" {
		text.replace('\n', "\r\n")
	} else {
		text.to_owned()
	}
}

/// JS `.slice(0, n)` by UTF-16 units with `…` suffix
/// (`formatPreviewWindow`, replace.ts:255).
fn preview_truncate(line: &str, max_len: usize) -> String {
	let units: usize = line.chars().map(char::len_utf16).sum();
	if units <= max_len {
		return line.to_owned();
	}
	let mut taken = 0usize;
	let mut end = 0usize;
	for (byte_idx, ch) in line.char_indices() {
		if taken + ch.len_utf16() > max_len - 1 {
			end = byte_idx;
			break;
		}
		taken += ch.len_utf16();
		end = byte_idx + ch.len_utf8();
	}
	format!("{}…", &line[..end])
}

/// `formatPreviewWindow` (replace.ts:248): `  ${num} | ${text}` window.
fn format_preview_window(lines: &[&str], center_index: usize) -> String {
	let start = center_index.saturating_sub(OCCURRENCE_PREVIEW_CONTEXT);
	let end = (center_index + OCCURRENCE_PREVIEW_CONTEXT + 1).min(lines.len());
	lines[start..end]
		.iter()
		.enumerate()
		.map(|(offset, line)| {
			format!(
				"  {} | {}",
				start + offset + 1,
				preview_truncate(line, OCCURRENCE_PREVIEW_MAX_LEN)
			)
		})
		.collect::<Vec<_>>()
		.join("\n")
}

struct ExactOutcome {
	/// Byte index of the unique match.
	match_index: Option<usize>,
	occurrences: usize,
	previews:    Vec<String>,
}

/// `findExactMatchOutcome` (replace.ts:261), previews included.
fn find_exact_match_outcome(content: &str, target: &str) -> Option<ExactOutcome> {
	let first = content.find(target)?;
	let occurrences = content.split(target).count() - 1;
	if occurrences > 1 {
		let content_lines: Vec<&str> = content.split('\n').collect();
		let mut previews = Vec::new();
		let mut search_start = 0usize;
		for _ in 0..MAX_RECORDED_MATCHES {
			let Some(rel) = content[search_start..].find(target) else {
				break;
			};
			let idx = search_start + rel;
			let line_number = content[..idx].split('\n').count();
			previews.push(format_preview_window(&content_lines, line_number - 1));
			search_start = idx + 1;
		}
		return Some(ExactOutcome { match_index: None, occurrences, previews });
	}
	Some(ExactOutcome { match_index: Some(first), occurrences: 1, previews: Vec::new() })
}

/// `formatOccurrenceMatchError` (diff.ts:458) — the variant `replaceText`
/// throws (no path suffix).
fn format_occurrence_match_error(occurrences: usize, previews: &[String]) -> String {
	let previews = previews.join("\n\n");
	let more = if occurrences > MAX_RECORDED_MATCHES {
		format!(" (showing first {MAX_RECORDED_MATCHES} of {occurrences})")
	} else {
		String::new()
	};
	format!(
		"Found {occurrences} occurrences{more}:\n\n{previews}\n\nAdd more context lines to \
		 disambiguate."
	)
}

struct ReplaceOutcome {
	content: String,
	count:   usize,
}

/// `replaceText` (diff.ts:850), exact matching only (fuzzy pinned off).
fn replace_text_exact(
	content: &str,
	old_text: &str,
	new_text: &str,
	all: bool,
) -> Result<ReplaceOutcome, ToolError> {
	if old_text.is_empty() {
		return Err(ToolError::new("oldText must not be empty."));
	}
	if all {
		let exact_count = content.split(old_text).count() - 1;
		if exact_count > 0 {
			return Ok(ReplaceOutcome {
				content: content.split(old_text).collect::<Vec<_>>().join(new_text),
				count:   exact_count,
			});
		}
		// Fuzzy-off: the iterative fuzzy loop never accepts a match.
		return Ok(ReplaceOutcome { content: content.to_owned(), count: 0 });
	}

	match find_exact_match_outcome(content, old_text) {
		Some(outcome) if outcome.occurrences > 1 => {
			Err(ToolError::new(format_occurrence_match_error(outcome.occurrences, &outcome.previews)))
		},
		Some(ExactOutcome { match_index: Some(idx), .. }) => {
			// Exact match: `adjustIndentation` is the identity (old == actual).
			let mut replaced = String::with_capacity(content.len());
			replaced.push_str(&content[..idx]);
			replaced.push_str(new_text);
			replaced.push_str(&content[idx + old_text.len()..]);
			Ok(ReplaceOutcome { content: replaced, count: 1 })
		},
		_ => Ok(ReplaceOutcome { content: content.to_owned(), count: 0 }),
	}
}

/// `EditMatchError.formatMessage` (replace.ts:87), no-closest branch with
/// fuzzy disabled. The closest-match preview requires the fuzzy scorer, which
/// is deferred.
fn edit_match_error(path: &str) -> ToolError {
	ToolError::new(format!(
		"Could not find the exact text in {path}. The old text must match exactly including all \
		 whitespace and newlines."
	))
}

/// One `{old_text, new_text, all?}` entry.
struct ReplaceEntry {
	old_text: String,
	new_text: String,
	all:      bool,
}

/// `executeReplaceSingle` (replace.ts:1041), exact-only.
fn execute_replace_single(
	cwd: &std::path::Path,
	path: &str,
	entry: &ReplaceEntry,
) -> Result<String, ToolError> {
	if entry.old_text.is_empty() {
		return Err(ToolError::new("old_text must not be empty."));
	}

	let absolute_path = resolve_to_cwd(path, cwd);
	let raw_content = match std::fs::read_to_string(&absolute_path) {
		Ok(text) => text,
		Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
			return Err(ToolError::new(format!("File not found: {path}")));
		},
		Err(err) => return Err(ToolError::new(format!("edit: cannot read '{path}': {err}"))),
	};
	let (bom, content) = strip_bom(&raw_content);
	let original_ending = detect_line_ending(content);
	let normalized_content = normalize_to_lf(content);
	let normalized_old = normalize_to_lf(&entry.old_text);
	let normalized_new = normalize_to_lf(&entry.new_text);

	let result =
		replace_text_exact(&normalized_content, &normalized_old, &normalized_new, entry.all)?;

	if result.count == 0 {
		// Re-run the match for diagnostics (replace.ts:1078). Exact-only: the
		// ambiguity case already threw inside `replace_text_exact`.
		return Err(edit_match_error(path));
	}

	if normalized_content == result.content {
		return Err(ToolError::new(format!("Edits to {path} resulted in no changes being made.")));
	}

	let final_content = format!("{bom}{}", restore_line_endings(&result.content, original_ending));
	std::fs::write(&absolute_path, &final_content)
		.map_err(|err| ToolError::new(format!("edit: cannot write '{path}': {err}")))?;

	Ok(if result.count > 1 {
		format!("Successfully replaced {} occurrences in {path}.", result.count)
	} else {
		format!("Successfully replaced text in {path}.")
	})
}

/// The `edit` tool (replace mode).
pub struct EditTool {
	cwd: PathBuf,
}

impl EditTool {
	#[must_use]
	pub fn new(cwd: impl Into<PathBuf>) -> Self {
		Self { cwd: cwd.into() }
	}
}

impl Tool for EditTool {
	fn name(&self) -> &'static str {
		"edit"
	}

	fn description(&self) -> &str {
		crate::prompts::REPLACE
	}

	/// `edit` is exclusive — it rewrites a file and must not overlap other calls
	/// (`packages/coding-agent/src/edit/index.ts:374`).
	fn concurrency(&self) -> crate::Concurrency {
		crate::Concurrency::Exclusive
	}

	fn input_schema(&self) -> Value {
		json!({
			"type": "object",
			"properties": {
				"path": { "type": "string" },
				"edits": {
					"type": "array",
					"items": {
						"type": "object",
						"properties": {
							"old_text": { "type": "string" },
							"new_text": { "type": "string" },
							"all": { "type": "boolean" },
						},
						"required": ["old_text", "new_text"],
						"additionalProperties": false,
					},
				},
			},
			"required": ["path", "edits"],
			"additionalProperties": false,
		})
	}

	async fn execute(
		&self,
		_tool_call_id: &str,
		args: Value,
		_ct: &pi_shell::cancel::CancelToken,
	) -> Result<ToolResult, ToolError> {
		let path = args
			.get("path")
			.and_then(Value::as_str)
			.ok_or_else(|| ToolError::new("edit: missing required parameter `path`"))?;
		let edits_value = args
			.get("edits")
			.and_then(Value::as_array)
			.ok_or_else(|| ToolError::new("edit: missing required parameter `edits`"))?;
		let entries: Vec<ReplaceEntry> = edits_value
			.iter()
			.map(|entry| {
				let old_text = entry
					.get("old_text")
					.and_then(Value::as_str)
					.ok_or_else(|| ToolError::new("edit: entry missing `old_text`"))?;
				let new_text = entry
					.get("new_text")
					.and_then(Value::as_str)
					.ok_or_else(|| ToolError::new("edit: entry missing `new_text`"))?;
				Ok(ReplaceEntry {
					old_text: old_text.to_owned(),
					new_text: new_text.to_owned(),
					all:      entry.get("all").and_then(Value::as_bool).unwrap_or(false),
				})
			})
			.collect::<Result<_, ToolError>>()?;

		// `executeSinglePathEntries` (edit/index.ts:239).
		if entries.len() == 1 {
			let text = execute_replace_single(&self.cwd, path, &entries[0])?;
			return Ok(ToolResult::text(text));
		}

		let total = entries.len();
		let mut content_texts: Vec<String> = Vec::new();
		let mut has_error = false;
		for (i, entry) in entries.iter().enumerate() {
			match execute_replace_single(&self.cwd, path, entry) {
				Ok(text) => {
					if !text.is_empty() {
						content_texts.push(text);
					}
				},
				Err(err) => {
					content_texts
						.push(format!("Error editing {path} (entry {} of {total}): {err}", i + 1));
					if i > 0 {
						content_texts.push(if i == 1 {
							"Entry 1 was already applied.".to_owned()
						} else {
							format!("Entries 1-{i} were already applied.")
						});
					}
					if i + 1 < total {
						let head = if i + 2 == total {
							format!("Entry {total} was NOT applied")
						} else {
							format!("Entries {}-{total} were NOT applied", i + 2)
						};
						content_texts.push(format!(
							"{head}; re-read the file and re-issue only the failed and unapplied entries."
						));
					}
					has_error = true;
					break;
				},
			}
		}

		let mut result = ToolResult::text(content_texts.join("\n"));
		if has_error {
			result = result.error();
		}
		Ok(result)
	}
}
