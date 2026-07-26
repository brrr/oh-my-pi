//! `read` tool — minimal face, ported from
//! `packages/coding-agent/src/tools/read.ts` (plain text files, hashline
//! display mode, single `:N` / `:N-M` selectors).
//!
//! Deferred vs. TS: images/PDF/SQLite/archives/notebooks/URLs/internal URIs,
//! directory listings, multi-range selectors, `:raw` / `:conflicts`, code
//! structural summaries (`read.summarize.*`), binary sniffing, suffix-match
//! recovery, ACP bridge reads, conflict-marker scanning, delimited multi-path
//! reads, tree-sitter block context (lexical bracket scan only), and
//! display-mode switching (always hashline; `readLineNumbers` ignored).

use std::path::PathBuf;

use serde_json::{Value, json};

use super::{
	block_context::{LineEntry, LineSpan, build_line_entries_with_block_context},
	paths::{format_path_relative_to_cwd, resolve_to_cwd, shorten_path},
	selector::{ParsedSelector, parse_sel, sel_to_offset_limit, split_path_and_sel},
};
use crate::{
	hashline::{compute_file_hash, format_hashline_header, format_numbered_line},
	tool::{Tool, ToolError, ToolResult},
	truncate::DEFAULT_MAX_LINES,
};

/// `read.defaultLimit` settings default (settings-schema.ts:3079).
pub const READ_DEFAULT_LIMIT: usize = 300;
/// `tools.outputMaxColumns` settings default (settings-schema.ts:745).
pub const OUTPUT_MAX_COLUMNS: usize = 768;
/// `RANGE_LEADING_CONTEXT_LINES` (read.ts:410).
const RANGE_LEADING_CONTEXT_LINES: usize = 1;
/// `RANGE_TRAILING_CONTEXT_LINES` (read.ts:411).
const RANGE_TRAILING_CONTEXT_LINES: usize = 3;
/// `DEFAULT_MAX_BYTES` is the floor for the per-read byte budget
/// (read.ts:2558).
const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// `formatBytes` (packages/utils/src/format.ts:54) — only reached by the
/// giant-single-line paths, ported for completeness.
fn format_bytes(bytes: usize) -> String {
	const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
	if bytes == 0 {
		return "0B".to_owned();
	}
	let mut value = bytes as f64;
	let mut unit = 0usize;
	while value >= 1024.0 && unit + 1 < UNITS.len() {
		value /= 1024.0;
		unit += 1;
	}
	if unit == 0 {
		format!("{bytes}B")
	} else {
		format!("{:.1}{}", value, UNITS[unit])
	}
}

/// `truncateLine` (streaming-output.ts:261): cap by UTF-16 code units,
/// appending `…` (JS `slice` operates on UTF-16 units).
fn truncate_line_utf16(line: &str, max_chars: usize) -> (String, bool) {
	let mut units = 0usize;
	for (byte_idx, ch) in line.char_indices() {
		let ch_units = ch.len_utf16();
		if units + ch_units > max_chars {
			// JS slice would cut inside a surrogate pair only for astral chars
			// at the boundary; cutting before the char matches for BMP text.
			return (format!("{}…", &line[..byte_idx]), true);
		}
		units += ch_units;
	}
	(line.to_owned(), false)
}

/// `normalizeToLF` (hashline normalize.ts:19).
fn normalize_to_lf(text: &str) -> String {
	text.replace("\r\n", "\n").replace('\r', "\n")
}

/// `formatReadHashlineHeader` (read.ts:191): basename for workspace-relative
/// display paths, `~`-shortened for absolute ones.
fn format_read_hashline_header(display_path: &str, tag: &str) -> String {
	let anchor = if display_path.starts_with('/') {
		shorten_path(display_path)
	} else {
		display_path
			.rsplit('/')
			.next()
			.unwrap_or(display_path)
			.to_owned()
	};
	format_hashline_header(&anchor, tag)
}

/// The `read` tool.
pub struct ReadTool {
	cwd: PathBuf,
}

impl ReadTool {
	#[must_use]
	pub fn new(cwd: impl Into<PathBuf>) -> Self {
		Self { cwd: cwd.into() }
	}
}

impl Tool for ReadTool {
	fn name(&self) -> &'static str {
		"read"
	}

	fn description(&self) -> &str {
		crate::prompts::READ
	}

	fn input_schema(&self) -> Value {
		json!({
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "Local path, internal URI (e.g. memory://, skill://), or URL. Inline selectors are supported.",
				},
			},
			"required": ["path"],
			"additionalProperties": false,
		})
	}

	async fn execute(
		&self,
		_tool_call_id: &str,
		args: Value,
		_ct: &pi_shell::cancel::CancelToken,
	) -> Result<ToolResult, ToolError> {
		let read_path = args
			.get("path")
			.and_then(Value::as_str)
			.ok_or_else(|| ToolError::new("read: missing required parameter `path`"))?;

		// `splitPathAndSelPreferringLiteral`: a literal on-disk match wins over
		// selector interpretation.
		let (split_path, split_sel) = split_path_and_sel(read_path);
		let (local_read_path, sel) = if split_sel.is_some() {
			let literal = resolve_to_cwd(read_path, &self.cwd);
			if std::fs::symlink_metadata(&literal).is_ok() {
				(read_path.to_owned(), None)
			} else {
				(split_path, split_sel)
			}
		} else {
			(split_path, split_sel)
		};

		let parsed = parse_sel(sel.as_deref())?;
		match &parsed {
			ParsedSelector::Raw => {
				return Err(ToolError::new("read: ':raw' selectors are deferred in the Rust core"));
			},
			ParsedSelector::Conflicts => {
				return Err(ToolError::new(
					"read: ':conflicts' selectors are deferred in the Rust core",
				));
			},
			ParsedSelector::Lines { ranges, raw } => {
				if *raw {
					return Err(ToolError::new("read: ':raw' selectors are deferred in the Rust core"));
				}
				if ranges.len() > 1 {
					return Err(ToolError::new(
						"read: multi-range selectors are deferred in the Rust core",
					));
				}
			},
			ParsedSelector::None => {},
		}

		let absolute_path = resolve_to_cwd(&local_read_path, &self.cwd);
		let metadata = std::fs::metadata(&absolute_path)
			.map_err(|_| ToolError::new(format!("Path '{local_read_path}' not found")))?;
		if metadata.is_dir() {
			return Err(ToolError::new("read: directory listings are deferred in the Rust core"));
		}

		let full_text = std::fs::read_to_string(&absolute_path)
			.map_err(|err| ToolError::new(format!("read: cannot read '{local_read_path}': {err}")))?;

		let (offset, limit) = sel_to_offset_limit(&parsed);

		// ── Disk single-range path (read.ts:2538-2816) ─────────────────────
		let all_lines: Vec<&str> = full_text.split('\n').collect();
		let total_file_lines = all_lines.len();

		let requested_start = offset.map_or(0, |o| o.saturating_sub(1));
		let expand_start = offset.is_some_and(|o| o > 1);
		let expand_end = limit.is_some();
		let leading_context = if expand_start {
			requested_start.min(RANGE_LEADING_CONTEXT_LINES)
		} else {
			0
		};
		let trailing_context = if expand_end {
			RANGE_TRAILING_CONTEXT_LINES
		} else {
			0
		};
		let start_line = requested_start - leading_context;
		let start_line_display = start_line + 1;

		let effective_limit = limit.unwrap_or(READ_DEFAULT_LIMIT);
		let max_lines_to_collect =
			(effective_limit + leading_context + trailing_context).min(DEFAULT_MAX_LINES);
		let max_bytes_for_read = DEFAULT_MAX_BYTES.max(max_lines_to_collect * 512);

		// Out-of-bounds check (read.ts:2581).
		if requested_start >= total_file_lines {
			let suggestion = if total_file_lines == 0 {
				"The file is empty.".to_owned()
			} else {
				format!("Use :1 to read from the start, or :{total_file_lines} to read the last line.")
			};
			return Ok(ToolResult::text(format!(
				"Line {} is beyond end of file ({total_file_lines} lines total). {suggestion}",
				requested_start + 1
			)));
		}

		// `streamLinesFromFile` equivalent on in-memory lines: collect up to
		// `max_lines_to_collect` lines subject to the byte budget.
		let mut collected: Vec<&str> = Vec::new();
		let mut collected_bytes = 0usize;
		let mut stopped_by_byte_limit = false;
		let mut first_line_byte_length: Option<usize> = None;
		for line in all_lines.iter().skip(start_line) {
			let line_bytes = line.len();
			if collected.len() >= max_lines_to_collect {
				break;
			}
			let separator = usize::from(!collected.is_empty());
			if collected.is_empty() && line_bytes > max_bytes_for_read {
				stopped_by_byte_limit = true;
				first_line_byte_length = Some(line_bytes);
				break;
			}
			if !collected.is_empty() && collected_bytes + separator + line_bytes > max_bytes_for_read {
				stopped_by_byte_limit = true;
				break;
			}
			collected.push(line);
			collected_bytes += separator + line_bytes;
			first_line_byte_length.get_or_insert(line_bytes);
			if collected_bytes > max_bytes_for_read {
				stopped_by_byte_limit = true;
				break;
			}
		}

		let first_line_exceeds_limit =
			first_line_byte_length.is_some_and(|len| len > max_bytes_for_read) && collected.is_empty();

		// Per-line column cap (display only; read.ts:2597).
		let display_lines: Vec<String> = collected
			.iter()
			.map(|line| truncate_line_utf16(line, OUTPUT_MAX_COLUMNS).0)
			.collect();

		let total_selected_lines = total_file_lines - start_line;
		let was_truncated = collected.len() < total_selected_lines || stopped_by_byte_limit;

		// Hashline tag: content hash of the whole (LF-normalized) file
		// (read.ts:2649; recordFileSnapshot / whole-file record agree on it).
		let hash_context = if !collected.is_empty() && !first_line_exceeds_limit {
			let tag = compute_file_hash(&normalize_to_lf(&full_text));
			let display_path =
				format_path_relative_to_cwd(&absolute_path.to_string_lossy(), &self.cwd, false);
			Some(format_read_hashline_header(&display_path, &tag))
		} else {
			None
		};

		if first_line_exceeds_limit {
			let first_line_bytes = first_line_byte_length.unwrap_or(0);
			// Hashline mode: no editable preview for a truncated line (read.ts:2720).
			let text = format!(
				"[Line {start_line_display} is {}, exceeds {} limit. Hashline output requires full \
				 lines; cannot emit an editable numbered preview for a truncated line.]",
				format_bytes(first_line_bytes),
				format_bytes(max_bytes_for_read)
			);
			return Ok(ToolResult::text(text));
		}

		// Bracket-aware entries (read.ts:2682): normalized full lines feed the
		// context scan; visible window text comes from the (column-capped)
		// collected lines.
		let normalized_full: Vec<String> = normalize_to_lf(&full_text)
			.split('\n')
			.map(str::to_owned)
			.collect();
		let displayed_end_line = start_line_display + display_lines.len().saturating_sub(1);
		let entries = build_line_entries_with_block_context(
			&normalized_full,
			&[LineSpan { start_line: start_line_display, end_line: displayed_end_line }],
			|line_number, source_text, _context| {
				if line_number >= start_line_display && line_number <= displayed_end_line {
					display_lines[line_number - start_line_display].clone()
				} else {
					truncate_line_utf16(source_text, OUTPUT_MAX_COLUMNS).0
				}
			},
		);

		let formatted = entries
			.iter()
			.map(|entry| match entry {
				LineEntry::Ellipsis => "…".to_owned(),
				LineEntry::Line { line_number, text, .. } => format_numbered_line(*line_number, text),
			})
			.collect::<Vec<_>>()
			.join("\n");
		let mut output_text = match &hash_context {
			Some(header) => format!("{header}\n{formatted}"),
			None => formatted,
		};

		// Branch order mirrors read.ts:2715/2741/2753/2762: a truncated window
		// emits bare lines (truncation is meta-only); the `[N more lines…]`
		// notice fires only on the non-truncated partial path.
		let user_limited_lines = collected.len();
		if !was_truncated && start_line + user_limited_lines < total_file_lines {
			let next_offset = start_line + user_limited_lines + 1;
			let remaining = total_file_lines - (start_line + user_limited_lines);
			use std::fmt::Write as _;
			let _ = write!(
				output_text,
				"\n\n[{remaining} more lines in file. Use :{next_offset} to continue]"
			);
		}

		Ok(ToolResult::text(output_text))
	}
}
