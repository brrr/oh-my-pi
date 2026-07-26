//! `glob` tool — minimal face.
//!
//! Ported from `packages/coding-agent/src/tools/glob.ts` with the native walk
//! logic from `crates/pi-natives/src/glob.rs` re-implemented on `pi-walker` +
//! `pi-grep` directly (no N-API dependency).
//!
//! Deferred vs. TS: semicolon-delimited multi-path targets, internal URLs +
//! `ssh://` rejection paths, missing-path partitioning across multiple
//! entries, the 5s walk timeout + partial-result notices, custom operations
//! (SSH delegation), and the scan cache.

use std::path::PathBuf;

use serde_json::{Value, json};

use super::{
	path_tree::format_grouped_paths,
	paths::{
		format_path_relative_to_cwd, normalize_path_like_input, parse_find_pattern, resolve_to_cwd,
	},
};
use crate::{
	tool::{Tool, ToolError, ToolResult},
	truncate::{TruncateOptions, truncate_head},
};

/// `DEFAULT_LIMIT` / `MAX_LIMIT` (glob.ts:52).
const DEFAULT_LIMIT: usize = 200;
const MAX_LIMIT: usize = 200;

struct RankedMatch {
	/// Relative path from the search root (forward slashes).
	path:   String,
	is_dir: bool,
	mtime:  f64,
}

/// `run_glob` (pi-natives glob.rs:169), ranked mtime path only (the glob tool
/// always passes `sortByMtime: true`, `recursive: false`, no type filter).
fn run_walker_glob(
	root: &std::path::Path,
	pattern: &str,
	include_hidden: bool,
	use_gitignore: bool,
	max_results: usize,
) -> Result<Vec<RankedMatch>, ToolError> {
	let pattern = pattern.trim();
	let pattern = if pattern.is_empty() { "*" } else { pattern };
	let walk_glob_pattern = pi_grep::glob::build_glob_pattern(pattern, false);
	let walk_depth_limit = pi_grep::glob::walk_depth_bound(&walk_glob_pattern);
	let walk_glob = pi_walker::CompiledWalkGlob::new([walk_glob_pattern])
		.map_err(|err| ToolError::new(format!("Invalid glob pattern: {err}")))?;
	if max_results == 0 {
		return Ok(Vec::new());
	}
	let mentions_node_modules = pattern.contains("node_modules");

	let request = pi_walker::WalkRequest::new(root.to_path_buf())
		.hidden(include_hidden)
		.gitignore(use_gitignore)
		.skip_git(true)
		.skip_node_modules(!mentions_node_modules)
		.follow_links(pi_walker::FollowLinks::Never)
		.detail(pi_walker::WalkDetail::Full)
		.order(pi_walker::WalkOrder::Path)
		.emit_root(false)
		.depth(1, walk_depth_limit)
		.directory_errors(pi_walker::DirectoryErrorMode::SkipSkippable)
		.cache(false)
		.empty_recheck(pi_walker::EmptyRecheck::Configured)
		.filter(
			pi_walker::WalkFilter::all()
				.glob(walk_glob)
				.node_modules_unless_mentioned(mentions_node_modules),
		);

	let outcome = request
		.collect_ranked_with_heartbeat(pi_walker::WalkRank::MtimeDescPathAsc, max_results, || {
			Ok::<(), std::convert::Infallible>(())
		})
		.map_err(|err| ToolError::new(format!("{err}")))?;

	let mut matches: Vec<RankedMatch> = outcome
		.entries
		.into_iter()
		.map(|entry| RankedMatch {
			is_dir: matches!(entry.file_type, pi_walker::FileType::Dir),
			mtime:  entry.mtime.unwrap_or(0.0),
			path:   entry.path,
		})
		.collect();
	// Rank by mtime desc, path asc; then cap (glob.rs:219).
	matches.sort_by(|a, b| {
		b.mtime
			.total_cmp(&a.mtime)
			.then_with(|| a.path.cmp(&b.path))
	});
	matches.truncate(max_results);
	Ok(matches)
}

/// The `glob` tool.
pub struct GlobTool {
	cwd: PathBuf,
}

impl GlobTool {
	#[must_use]
	pub fn new(cwd: impl Into<PathBuf>) -> Self {
		Self { cwd: cwd.into() }
	}
}

impl Tool for GlobTool {
	fn name(&self) -> &'static str {
		"glob"
	}

	fn description(&self) -> &str {
		crate::prompts::GLOB
	}

	fn input_schema(&self) -> Value {
		json!({
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "glob, file, or directory to search — a single path or a semicolon-delimited list (\"src/**/*.ts; test/**/*.ts\"). Omitted -> searches the workspace root (\".\")",
				},
				"hidden": { "type": "boolean", "description": "include hidden files" },
				"gitignore": { "type": "boolean", "description": "respect gitignore" },
				"limit": { "type": "number", "description": "max results" },
			},
			"additionalProperties": false,
		})
	}

	async fn execute(
		&self,
		_tool_call_id: &str,
		args: Value,
		_ct: &pi_shell::cancel::CancelToken,
	) -> Result<ToolResult, ToolError> {
		let path_input = args.get("path").and_then(Value::as_str);
		let limit = args.get("limit").and_then(Value::as_f64);
		let hidden = args.get("hidden").and_then(Value::as_bool);
		let gitignore = args.get("gitignore").and_then(Value::as_bool);

		let raw_pattern = normalize_path_like_input(path_input.unwrap_or(".")).replace('\\', "/");
		if !raw_pattern.is_empty() && raw_pattern.chars().all(|c| c == '/') {
			return Err(ToolError::new("Searching from root directory '/' is not allowed"));
		}
		if raw_pattern.is_empty() {
			return Err(ToolError::new("`path` must contain non-empty globs or paths"));
		}

		let parsed = parse_find_pattern(&raw_pattern);
		let search_path = resolve_to_cwd(&parsed.base_path, &self.cwd);
		let scope_path =
			format_path_relative_to_cwd(&search_path.to_string_lossy(), &self.cwd, false);
		if search_path == std::path::Path::new("/") {
			return Err(ToolError::new("Searching from root directory '/' is not allowed"));
		}

		let requested_limit = limit.unwrap_or(DEFAULT_LIMIT as f64);
		if !requested_limit.is_finite() || requested_limit <= 0.0 {
			return Err(ToolError::new("Limit must be a positive number"));
		}
		let effective_limit = (requested_limit.floor() as usize).clamp(1, MAX_LIMIT);
		let include_hidden = hidden.unwrap_or(true);
		let use_gitignore = gitignore.unwrap_or(true);

		let format_match_path = |match_path: &str, is_dir: bool| -> String {
			let had_trailing_slash = match_path.ends_with('/') || match_path.ends_with('\\');
			let absolute = if std::path::Path::new(match_path).is_absolute() {
				PathBuf::from(match_path)
			} else {
				search_path.join(match_path)
			};
			format_path_relative_to_cwd(
				&absolute.to_string_lossy(),
				&self.cwd,
				is_dir || had_trailing_slash,
			)
		};

		let files: Vec<String> = {
			let metadata = std::fs::metadata(&search_path)
				.map_err(|_| ToolError::new(format!("Path not found: {scope_path}")))?;
			if !parsed.has_glob && metadata.is_file() {
				vec![format_path_relative_to_cwd(&search_path.to_string_lossy(), &self.cwd, false)]
			} else if !metadata.is_dir() {
				return Err(ToolError::new(format!(
					"Path is not a directory: {}",
					search_path.display()
				)));
			} else {
				run_walker_glob(
					&search_path,
					&parsed.glob_pattern,
					include_hidden,
					use_gitignore,
					effective_limit,
				)?
				.into_iter()
				.map(|m| format_match_path(&m.path, m.is_dir))
				.collect()
			}
		};

		// `buildResult` (glob.ts:260).
		if files.is_empty() {
			return Ok(ToolResult::text("No files found matching pattern").useless());
		}
		// `applyListLimit`: slice at the limit (meta-only side effects elided).
		let limited: Vec<String> = if files.len() >= effective_limit {
			files[..effective_limit].to_vec()
		} else {
			files
		};
		let base_output = format_grouped_paths(&limited);
		let truncation = truncate_head(&base_output, &TruncateOptions {
			max_lines: Some(usize::MAX),
			max_bytes: None,
		});
		Ok(ToolResult::text(truncation.content))
	}
}
