//! `write` tool — minimal face, ported from
//! `packages/coding-agent/src/tools/write.ts` (plain filesystem writes in
//! replace edit-mode, i.e. no hashline snapshot header).
//!
//! Deferred vs. TS: hashline display-prefix stripping + snapshot header
//! (hashline mode), internal URLs (`xd://`, `local://`, `conflict://`),
//! archive/SQLite writes, plan-mode guard, ACP bridge routing, LSP
//! writethrough/diagnostics, and the auto-generated-file guard
//! (`assertEditableFile`).

use std::path::PathBuf;

use serde_json::{Value, json};

use super::paths::{format_path_relative_to_cwd, resolve_to_cwd};
use crate::tool::{Tool, ToolError, ToolResult};

/// `EXECUTABLE_NOTICE` (write.ts:87).
const EXECUTABLE_NOTICE: &str = "[Notice: Made executable via chmod +x]";

/// JS `String.prototype.length` (UTF-16 code units) — the "bytes" the TS
/// success line reports (write.ts:1208 uses `cleanContent.length`).
fn utf16_length(text: &str) -> usize {
	text.chars().map(char::len_utf16).sum()
}

/// `maybeMarkExecutableForShebang` (write.ts:321): `chmod a+x` when content
/// starts with `#!`; errors swallowed.
#[cfg(unix)]
fn maybe_mark_executable(absolute_path: &std::path::Path, content: &str) -> bool {
	use std::os::unix::fs::PermissionsExt;
	if !content.starts_with("#!") {
		return false;
	}
	let Ok(metadata) = std::fs::metadata(absolute_path) else {
		return false;
	};
	let current = metadata.permissions().mode() & 0o7777;
	let new_mode = current | 0o111;
	if new_mode == current {
		return false;
	}
	std::fs::set_permissions(absolute_path, std::fs::Permissions::from_mode(new_mode)).is_ok()
}

#[cfg(not(unix))]
fn maybe_mark_executable(_absolute_path: &std::path::Path, _content: &str) -> bool {
	false
}

/// The `write` tool.
pub struct WriteTool {
	cwd: PathBuf,
}

impl WriteTool {
	#[must_use]
	pub fn new(cwd: impl Into<PathBuf>) -> Self {
		Self { cwd: cwd.into() }
	}
}

impl Tool for WriteTool {
	fn name(&self) -> &'static str {
		"write"
	}

	fn description(&self) -> &str {
		crate::prompts::WRITE
	}

	fn input_schema(&self) -> Value {
		json!({
			"type": "object",
			"properties": {
				"path": { "type": "string", "description": "file path" },
				"content": { "type": "string", "description": "file content" },
			},
			"required": ["path", "content"],
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
			.ok_or_else(|| ToolError::new("write: missing required parameter `path`"))?;
		let content = args
			.get("content")
			.and_then(Value::as_str)
			.ok_or_else(|| ToolError::new("write: missing required parameter `content`"))?;

		let absolute_path = resolve_to_cwd(path, &self.cwd);
		if let Some(parent) = absolute_path.parent() {
			std::fs::create_dir_all(parent).map_err(|err| {
				ToolError::new(format!("write: cannot create parent directory: {err}"))
			})?;
		}
		// `Bun.write(dst, cleanContent)` via `writethroughNoop` — content is
		// written verbatim (no LF normalization; write.ts normalizes only for
		// the hashline snapshot hash).
		std::fs::write(&absolute_path, content)
			.map_err(|err| ToolError::new(format!("write: cannot write '{path}': {err}")))?;

		let made_executable = maybe_mark_executable(&absolute_path, content);

		let display_path =
			format_path_relative_to_cwd(&absolute_path.to_string_lossy(), &self.cwd, false);
		let mut result_text =
			format!("Successfully wrote {} bytes to {display_path}", utf16_length(content));
		if made_executable {
			result_text.push('\n');
			result_text.push_str(EXECUTABLE_NOTICE);
		}
		Ok(ToolResult::text(result_text))
	}
}
