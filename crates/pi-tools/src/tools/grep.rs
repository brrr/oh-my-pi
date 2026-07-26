//! `grep` tool — minimal face, ported from
//! `packages/coding-agent/src/tools/grep.ts` on top of [`pi_grep::grep`]
//! (the same engine the TS tool calls through pi-natives).
//!
//! Deferred vs. TS: path-embedded line-range selectors (`file:50-100`),
//! semicolon-delimited multi-path scopes / fan-out targets, internal URLs +
//! archives + external URL materialization, the 30s native timeout, oversized
//! explicit-target notes, seen-line snapshot recording (tags are still
//! minted for output parity), and display (TUI) line rendering.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::{
	path_tree::{GroupedFileSection, format_grouped_files},
	paths::{format_path_relative_to_cwd, parse_search_path, resolve_to_cwd},
};
use crate::{
	hashline::{compute_file_hash, format_hashline_header},
	tool::{Tool, ToolError, ToolResult},
	truncate::{TruncateOptions, truncate_head},
};

/// `DEFAULT_FILE_LIMIT` (grep.ts:92).
const DEFAULT_FILE_LIMIT: usize = 20;
/// `MULTI_FILE_PER_FILE_MATCHES` (grep.ts:95).
const MULTI_FILE_PER_FILE_MATCHES: usize = 20;
/// `SINGLE_FILE_MATCHES` (grep.ts:98).
const SINGLE_FILE_MATCHES: usize = 200;
/// `INTERNAL_TOTAL_CAP` (grep.ts:103).
const INTERNAL_TOTAL_CAP: u32 = 2000;
/// `DEFAULT_MAX_COLUMN` (streaming-output.ts:12).
const DEFAULT_MAX_COLUMN: u32 = 512;
/// `grep.contextBefore` settings default (settings-schema.ts:3555).
const CONTEXT_BEFORE: u32 = 1;
/// `grep.contextAfter` settings default (settings-schema.ts:3573).
const CONTEXT_AFTER: u32 = 3;
/// `SNAPSHOT_MAX_BYTES` (file-snapshot-store.ts:22) — tag mint size cap.
const SNAPSHOT_MAX_BYTES: u64 = 4 * 1024 * 1024;

fn normalize_to_lf(text: &str) -> String {
	text.replace("\r\n", "\n").replace('\r', "\n")
}

/// `formatMatchLine` (match-line-format.ts:9), hashline mode.
fn format_match_line(line_number: u32, line: &str, is_match: bool) -> String {
	let marker = if is_match { "*" } else { " " };
	format!("{marker}{line_number}:{line}")
}

/// `formatResultPath` (file-recorder.ts).
fn format_result_path(file_path: &str, is_directory: bool, base_path: &Path, cwd: &Path) -> String {
	let clean = file_path.strip_prefix('/').unwrap_or(file_path);
	if is_directory {
		let resolved = base_path.join(clean);
		format_path_relative_to_cwd(&resolved.to_string_lossy(), cwd, false)
	} else {
		format_path_relative_to_cwd(&base_path.to_string_lossy(), cwd, false)
	}
}

/// Mint a whole-file content tag (`recordFileSnapshot`): `None` for missing /
/// oversized files.
fn mint_file_tag(absolute: &Path) -> Option<String> {
	let metadata = std::fs::metadata(absolute).ok()?;
	if metadata.len() > SNAPSHOT_MAX_BYTES {
		return None;
	}
	let text = std::fs::read_to_string(absolute).ok()?;
	Some(compute_file_hash(&normalize_to_lf(&text)))
}

/// The `grep` tool.
pub struct GrepTool {
	cwd: PathBuf,
}

impl GrepTool {
	#[must_use]
	pub fn new(cwd: impl Into<PathBuf>) -> Self {
		Self { cwd: cwd.into() }
	}
}

impl Tool for GrepTool {
	fn name(&self) -> &'static str {
		"grep"
	}

	fn description(&self) -> &str {
		crate::prompts::GREP
	}

	fn input_schema(&self) -> Value {
		json!({
			"type": "object",
			"properties": {
				"pattern": { "type": "string", "description": "regex pattern" },
				"path": {
					"type": "string",
					"description": "file, directory, glob, internal URL, or \"<file>:<lines>\" selector to search; pass several as a semicolon-delimited list (\"src; tests\"). Omitted -> searches the workspace root (\".\")",
				},
				"case": { "type": "boolean", "description": "case-sensitive search" },
				"gitignore": { "type": "boolean", "description": "respect gitignore" },
				"skip": {
					"anyOf": [
						{
							"type": "number",
							"description": "files to skip before collecting results — use to paginate when the prior call hit the file limit",
						},
						{
							"type": "null",
							"description": "files to skip before collecting results — use to paginate when the prior call hit the file limit",
						},
					],
					"description": "files to skip before collecting results — use to paginate when the prior call hit the file limit",
				},
			},
			"required": ["pattern"],
			"additionalProperties": false,
		})
	}

	#[allow(clippy::too_many_lines, reason = "1:1 port of the TS execute pipeline")]
	async fn execute(
		&self,
		_tool_call_id: &str,
		args: Value,
		ct: &pi_shell::cancel::CancelToken,
	) -> Result<ToolResult, ToolError> {
		let pattern = args
			.get("pattern")
			.and_then(Value::as_str)
			.ok_or_else(|| ToolError::new("grep: missing required parameter `pattern`"))?;
		if pattern.trim().is_empty() {
			return Err(ToolError::new("Pattern must not be empty"));
		}
		let case_sensitive = args.get("case").and_then(Value::as_bool);
		let use_gitignore = args
			.get("gitignore")
			.and_then(Value::as_bool)
			.unwrap_or(true);
		let skip_raw = args.get("skip").and_then(Value::as_f64);
		let normalized_skip = match skip_raw {
			None => 0usize,
			Some(v) if v.is_finite() && v >= 0.0 => v.floor() as usize,
			Some(_) => return Err(ToolError::new("Skip must be a non-negative number")),
		};

		let raw_path = args.get("path").and_then(Value::as_str).unwrap_or(".");
		let raw_path = super::paths::normalize_path_like_input(raw_path);
		let raw_path = if raw_path.is_empty() {
			".".to_owned()
		} else {
			raw_path
		};

		// Single-scope resolution (`resolveToolSearchScope`, single-entry arm):
		// a literal on-disk path wins over glob interpretation.
		let literal = resolve_to_cwd(&raw_path, &self.cwd);
		let parsed =
			if super::paths::has_glob_path_chars(&raw_path) && std::fs::metadata(&literal).is_err() {
				parse_search_path(&raw_path)
			} else {
				super::paths::ParsedSearchPath { base_path: raw_path, glob: None }
			};
		let search_path = resolve_to_cwd(&parsed.base_path, &self.cwd);
		let scope_path =
			format_path_relative_to_cwd(&search_path.to_string_lossy(), &self.cwd, false);
		let metadata = std::fs::metadata(&search_path)
			.map_err(|_| ToolError::new(format!("Path not found: {scope_path}")))?;
		let is_directory = metadata.is_dir();

		let ignore_case = !case_sensitive.unwrap_or(true);
		let effective_multiline = pattern.contains('\n') || pattern.contains("\\n");

		let is_multi_scope = is_directory;
		let per_file_match_cap = if is_multi_scope {
			MULTI_FILE_PER_FILE_MATCHES
		} else {
			SINGLE_FILE_MATCHES
		};
		let native_max_count_per_file = (per_file_match_cap + 1) as u32;

		let result = pi_grep::grep(
			pi_grep::GrepConfig {
				pattern:            pattern.to_owned(),
				path:               search_path.to_string_lossy().into_owned(),
				glob:               parsed.glob,
				type_filter:        None,
				ignore_case:        Some(ignore_case),
				multiline:          Some(effective_multiline),
				hidden:             Some(true),
				gitignore:          Some(use_gitignore),
				max_count:          Some(INTERNAL_TOTAL_CAP),
				offset:             None,
				context_before:     Some(CONTEXT_BEFORE),
				context_after:      Some(CONTEXT_AFTER),
				context:            None,
				max_columns:        Some(DEFAULT_MAX_COLUMN),
				mode:               None,
				max_count_per_file: Some(native_max_count_per_file),
			},
			None,
			ct,
		)
		.map_err(|err| {
			// `Invalid regex: ` rewrite (grep.ts:1199).
			static REGEX_ERR_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
				regex::Regex::new(r"(?i)^regex(?: parse)? error:?\s*").expect("REGEX_ERR_RE")
			});
			let message = err.to_string();
			if REGEX_ERR_RE.is_match(&message) {
				ToolError::new(
					REGEX_ERR_RE
						.replace(&message, "Invalid regex: ")
						.into_owned(),
				)
			} else {
				ToolError::new(message)
			}
		})?;

		// Group matches by path in encounter order (grep.ts:1276).
		let mut file_order: Vec<String> = Vec::new();
		let mut matches_by_path: std::collections::HashMap<String, Vec<&pi_grep::GrepMatch>> =
			std::collections::HashMap::new();
		for m in &result.matches {
			if !matches_by_path.contains_key(&m.path) {
				file_order.push(m.path.clone());
				matches_by_path.insert(m.path.clone(), Vec::new());
			}
			matches_by_path.get_mut(&m.path).expect("entry").push(m);
		}
		for list in matches_by_path.values_mut() {
			list.truncate(per_file_match_cap);
		}
		let total_files = file_order.len();
		let total_files_label = if result.limit_reached.unwrap_or(false) {
			format!("{total_files}+")
		} else {
			total_files.to_string()
		};
		let can_paginate = is_multi_scope;
		let skip_files = if can_paginate {
			normalized_skip.min(total_files)
		} else {
			0
		};
		let window_files: Vec<String> = if can_paginate {
			file_order
				.iter()
				.skip(skip_files)
				.take(DEFAULT_FILE_LIMIT)
				.cloned()
				.collect()
		} else {
			file_order.clone()
		};
		let file_limit_reached = can_paginate && total_files > skip_files + DEFAULT_FILE_LIMIT;
		let next_skip = skip_files + window_files.len();
		let limit_message = if file_limit_reached {
			format!(
				"Showing files {}-{next_skip} of {total_files_label}. Use skip={next_skip} for the \
				 next page, or narrow paths/pattern.",
				skip_files + 1
			)
		} else {
			String::new()
		};

		let selected_count: usize = window_files
			.iter()
			.map(|file| matches_by_path.get(file).map_or(0, Vec::len))
			.sum();
		if selected_count == 0 {
			let skip_past_end =
				can_paginate && normalized_skip > 0 && total_files > 0 && skip_files >= total_files;
			let text = if skip_past_end {
				format!(
					"No more results ({total_files_label} files total; skip={normalized_skip} is past \
					 the end)"
				)
			} else {
				"No matches found".to_owned()
			};
			return Ok(ToolResult::text(text).useless());
		}

		// Relative display paths in window order (`createFileRecorder`).
		let mut file_list: Vec<String> = Vec::new();
		let mut matches_by_display: std::collections::HashMap<String, Vec<&pi_grep::GrepMatch>> =
			std::collections::HashMap::new();
		for file in &window_files {
			let relative = format_result_path(file, is_directory, &search_path, &self.cwd);
			if !matches_by_display.contains_key(&relative) {
				file_list.push(relative.clone());
				matches_by_display.insert(relative.clone(), Vec::new());
			}
			let list = matches_by_display.get_mut(&relative).expect("entry");
			list.extend(matches_by_path.get(file).into_iter().flatten().copied());
		}

		// Hashline tags per file (grep.ts:1405).
		let mut hash_tags: std::collections::HashMap<String, String> =
			std::collections::HashMap::new();
		for relative in &file_list {
			let absolute = self.cwd.join(relative);
			if let Some(tag) = mint_file_tag(&absolute) {
				hash_tags.insert(relative.clone(), tag);
			}
		}

		let render_matches_for_file = |relative: &str| -> Vec<String> {
			let mut out: Vec<String> = Vec::new();
			let file_matches = matches_by_display
				.get(relative)
				.map_or(&[][..], Vec::as_slice);
			let mut last_emitted: Option<u32> = None;
			for m in file_matches {
				let mut push_line =
					|line_number: u32, line: &str, is_match: bool, out: &mut Vec<String>| {
						if let Some(last) = last_emitted
							&& line_number > last + 1
						{
							out.push("...".to_owned());
						}
						out.push(format_match_line(line_number, line, is_match));
						last_emitted = Some(line_number);
					};
				if let Some(before) = &m.context_before {
					for ctx in before {
						push_line(ctx.line_number, &ctx.line, false, &mut out);
					}
				}
				push_line(m.line_number, &m.line, true, &mut out);
				if let Some(after) = &m.context_after {
					for ctx in after {
						push_line(ctx.line_number, &ctx.line, false, &mut out);
					}
				}
			}
			out
		};

		let mut output_lines: Vec<String> = Vec::new();
		let use_grouped_output = is_directory || is_multi_scope;
		if use_grouped_output {
			let grouped = format_grouped_files(&file_list, |relative| {
				let model_lines = render_matches_for_file(relative);
				GroupedFileSection {
					header_suffix: hash_tags
						.get(relative)
						.map_or(String::new(), |tag| format!("#{tag}")),
					skip: model_lines.is_empty(),
					model_lines,
				}
			});
			output_lines.extend(grouped);
		} else {
			for relative in &file_list {
				let rendered = render_matches_for_file(relative);
				if rendered.is_empty() {
					continue;
				}
				if !output_lines.is_empty() {
					output_lines.push(String::new());
				}
				if let Some(tag) = hash_tags.get(relative) {
					output_lines.push(format_hashline_header(relative, tag));
				}
				output_lines.extend(rendered);
			}
		}
		if !limit_message.is_empty() {
			output_lines.push(String::new());
			output_lines.push(limit_message);
		}

		let raw_output = output_lines.join("\n");
		let truncation = truncate_head(&raw_output, &TruncateOptions {
			max_lines: Some(usize::MAX),
			max_bytes: None,
		});
		Ok(ToolResult::text(truncation.content))
	}
}
