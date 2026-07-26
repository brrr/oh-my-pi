//! `bash` tool — one-shot execution.
//!
//! Ported from `packages/coding-agent/src/tools/bash.ts` (foreground path),
//! `packages/coding-agent/src/exec/bash-executor.ts` (`executeBash`), and the
//! output-sink composition in
//! `packages/coding-agent/src/session/streaming-output.ts` (`OutputSink`,
//! `enforceInlineByteCap`).
//!
//! Deferred vs. TS: persistent shell sessions (each call is a one-shot
//! `pi_shell::execute_shell`), the output minimizer, PTY/interactive mode
//! (`pty` accepted but ignored), async/auto-background jobs, ACP bridge
//! terminals, artifact capture (`artifact://` footers never appear), shell
//! snapshots/user-shell wrapping, bash-interceptor rules, internal-URL
//! expansion in commands, ANSI/control sanitization (plain passthrough), and
//! github cache invalidation.

use std::{collections::HashMap, path::PathBuf, sync::LazyLock, time::Instant};

use regex::Regex;
use serde_json::{Value, json};

use super::paths::resolve_to_cwd;
use crate::{
	tool::{Tool, ToolError, ToolResult},
	truncate::truncate_head_bytes,
};

/// `DEFAULT_MAX_BYTES` (streaming-output.ts:11).
const DEFAULT_MAX_BYTES: usize = 50 * 1024;
/// `tools.artifactHeadBytes` default (20 KiB) — the sink's head-retention
/// budget (`resolveOutputSinkHeadBytes`).
const HEAD_RETENTION_BYTES: usize = 20 * 1024;
/// `tools.outputMaxColumns` default (settings-schema.ts:745).
const OUTPUT_MAX_COLUMNS: usize = 768;
/// `TOOL_TIMEOUTS.bash` (tool-timeouts.ts).
const BASH_TIMEOUT_DEFAULT: u32 = 300;
const BASH_TIMEOUT_MIN: u32 = 1;
const BASH_TIMEOUT_MAX: u32 = 3600;

/// `BASH_ENV_NAME_PATTERN` (bash.ts:52).
static ENV_NAME_RE: LazyLock<Regex> =
	LazyLock::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").expect("ENV_NAME_RE"));
/// Leading `cd <path> && …` extraction (bash.ts:760).
static CD_PREFIX_RE: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(r"^cd[ \t]+((?:[^&\\\n\r]|\\.)+?)[ \t]*&&[ \t]*").expect("CD_PREFIX_RE")
});

/// `truncateTailBytes`: keep the trailing `max_bytes` on a UTF-8 boundary.
fn truncate_tail_bytes(data: &str, max_bytes: usize) -> String {
	if data.len() <= max_bytes {
		return data.to_owned();
	}
	let mut start = data.len() - max_bytes;
	while start < data.len() && !data.is_char_boundary(start) {
		start += 1;
	}
	data[start..].to_owned()
}

/// `formatMiddleElisionMarker` (streaming-output.ts:500).
fn format_middle_elision_marker(elided_lines: usize, elided_bytes: usize) -> String {
	if elided_lines <= 1 {
		format!("[…{elided_bytes}B elided…]")
	} else {
		format!("[…{elided_lines}ln elided…]")
	}
}

fn count_newlines(text: &str) -> usize {
	text.bytes().filter(|&b| b == b'\n').count()
}

/// Line-column cap state (`OutputSink#applyColumnCap`).
struct ColumnCap {
	current_line_units: usize,
	ellipsis_added:     bool,
	dropped_bytes:      usize,
}

/// Minimal port of `OutputSink`: CR normalization, per-line column cap, head
/// retention + rolling tail, and middle-elision `dump()`.
struct OutputSinkLite {
	head:         String,
	head_bytes:   usize,
	head_lines:   usize,
	buffer:       String,
	buffer_bytes: usize,
	total_bytes:  usize,
	total_lines:  usize,
	saw_data:     bool,
	truncated:    bool,
	pending_cr:   bool,
	column:       ColumnCap,
}

impl OutputSinkLite {
	const fn new() -> Self {
		Self {
			head:         String::new(),
			head_bytes:   0,
			head_lines:   0,
			buffer:       String::new(),
			buffer_bytes: 0,
			total_bytes:  0,
			total_lines:  0,
			saw_data:     false,
			truncated:    false,
			pending_cr:   false,
			column:       ColumnCap {
				current_line_units: 0,
				ellipsis_added:     false,
				dropped_bytes:      0,
			},
		}
	}

	/// `#normalizeCarriageReturns`: CR → NL, CRLF → one NL, trailing CR held.
	fn normalize_carriage_returns(&mut self, text: &str) -> String {
		if text.is_empty() || (!self.pending_cr && !text.contains('\r')) {
			return text.to_owned();
		}
		let mut normalized = String::new();
		let mut chars = text.char_indices().peekable();
		if self.pending_cr {
			self.pending_cr = false;
			normalized.push('\n');
			if text.starts_with('\n') {
				chars.next();
			}
		}
		while let Some((_, ch)) = chars.next() {
			if ch != '\r' {
				normalized.push(ch);
				continue;
			}
			match chars.peek() {
				None => {
					self.pending_cr = true;
					break;
				},
				Some((_, '\n')) => {
					normalized.push('\n');
					chars.next();
				},
				Some(_) => normalized.push('\n'),
			}
		}
		normalized
	}

	/// `#applyColumnCap`: per-line UTF-16-unit cap with a single `…` marker.
	fn apply_column_cap(&mut self, chunk: &str) -> String {
		if chunk.is_empty() {
			return String::new();
		}
		let mut out = String::new();
		for (i, segment) in chunk.split('\n').enumerate() {
			if i > 0 {
				out.push('\n');
				self.column.current_line_units = 0;
				self.column.ellipsis_added = false;
			}
			if segment.is_empty() {
				continue;
			}
			let mut kept = String::new();
			for ch in segment.chars() {
				let units = ch.len_utf16();
				if self.column.current_line_units + units > OUTPUT_MAX_COLUMNS {
					if !self.column.ellipsis_added {
						kept.push('…');
						self.column.ellipsis_added = true;
					}
					self.column.dropped_bytes += ch.len_utf8();
					continue;
				}
				self.column.current_line_units += units;
				kept.push(ch);
			}
			out.push_str(&kept);
		}
		out
	}

	fn push(&mut self, chunk: &str) {
		let chunk = self.normalize_carriage_returns(chunk);
		let raw_bytes = chunk.len();
		self.total_bytes += raw_bytes;
		if !chunk.is_empty() {
			self.saw_data = true;
			self.total_lines += count_newlines(&chunk);
		}
		let capped = self.apply_column_cap(&chunk);
		if capped.is_empty() {
			return;
		}
		let capped_bytes = capped.len();

		// Head retention.
		let mut tail_chunk = capped.as_str();
		let mut tail_bytes = capped_bytes;
		if self.head_bytes < HEAD_RETENTION_BYTES {
			let room = HEAD_RETENTION_BYTES - self.head_bytes;
			if capped_bytes <= room {
				self.head.push_str(&capped);
				self.head_bytes += capped_bytes;
				self.head_lines += count_newlines(&capped);
				return;
			}
			let head_slice = truncate_head_bytes(&capped, room);
			if head_slice.bytes > 0 {
				self.head.push_str(&head_slice.text);
				self.head_bytes += head_slice.bytes;
				self.head_lines += count_newlines(&head_slice.text);
				tail_chunk = &capped[head_slice.text.len()..];
				tail_bytes = capped_bytes - head_slice.bytes;
			}
		}

		// Rolling tail (`#pushTail`).
		if tail_bytes == 0 {
			return;
		}
		if self.buffer_bytes + tail_bytes > DEFAULT_MAX_BYTES {
			self.truncated = true;
			if tail_bytes >= DEFAULT_MAX_BYTES {
				self.buffer = truncate_tail_bytes(tail_chunk, DEFAULT_MAX_BYTES);
			} else {
				self.buffer.push_str(tail_chunk);
				self.buffer = truncate_tail_bytes(&self.buffer, DEFAULT_MAX_BYTES);
			}
			self.buffer_bytes = self.buffer.len();
		} else {
			self.buffer.push_str(tail_chunk);
			self.buffer_bytes += tail_bytes;
		}
	}

	/// `dump(notice?)`: compose `[notice]\n` + head (+ elision marker) + tail.
	fn dump(&mut self, notice: Option<&str>) -> String {
		if self.pending_cr {
			self.pending_cr = false;
			self.push("\n");
		}
		let notice_line = notice.map_or(String::new(), |n| format!("[{n}]\n"));
		let total_lines = if self.saw_data {
			self.total_lines + 1
		} else {
			0
		};
		let head_lines =
			self.head_lines + usize::from(self.head_bytes > 0 && !self.head.ends_with('\n'));
		let tail_lines = if self.buffer.is_empty() {
			0
		} else {
			count_newlines(&self.buffer) + 1
		};
		let effective_total_bytes = self.total_bytes.saturating_sub(self.column.dropped_bytes);

		let body =
			if self.head_bytes > 0 && effective_total_bytes > self.head_bytes + self.buffer_bytes {
				let elided_bytes = effective_total_bytes - self.head_bytes - self.buffer_bytes;
				let elided_lines = total_lines.saturating_sub(head_lines + tail_lines);
				let marker = format_middle_elision_marker(elided_lines, elided_bytes);
				let head_sep = if self.head.ends_with('\n') { "" } else { "\n" };
				let tail_sep = if self.buffer.starts_with('\n') {
					""
				} else {
					"\n"
				};
				self.truncated = true;
				format!("{}{head_sep}{marker}{tail_sep}{}", self.head, self.buffer)
			} else if self.head_bytes > 0 {
				format!("{}{}", self.head, self.buffer)
			} else {
				self.buffer.clone()
			};

		format!("{notice_line}{body}")
	}
}

/// `trimHeadToLineBoundary` (streaming-output.ts:595).
fn trim_head_to_line_boundary(text: &str) -> &str {
	match text.rfind('\n') {
		Some(idx) if idx > 0 => &text[..idx],
		_ => text,
	}
}

/// `trimTailToLineBoundary` (streaming-output.ts:601).
fn trim_tail_to_line_boundary(text: &str) -> &str {
	match text.find('\n') {
		Some(idx) if idx != text.len() - 1 => &text[idx + 1..],
		_ => text,
	}
}

/// `enforceInlineByteCap` (streaming-output.ts:616), no artifact footer.
fn enforce_inline_byte_cap(text: &str) -> String {
	let total_bytes = text.len();
	if total_bytes <= DEFAULT_MAX_BYTES {
		return text.to_owned();
	}
	let head_budget = DEFAULT_MAX_BYTES * 6 / 10;
	let tail_budget = DEFAULT_MAX_BYTES / 4;
	let head_slice = truncate_head_bytes(text, head_budget).text;
	let head = trim_head_to_line_boundary(&head_slice);
	let tail_slice = truncate_tail_bytes(text, tail_budget);
	let tail = trim_tail_to_line_boundary(&tail_slice);
	let elided_bytes = total_bytes.saturating_sub(head.len() + tail.len());
	format!("{head}\n[…{elided_bytes}B elided…]\n{tail}")
}

/// `formatWallTimeNotice` (bash.ts:332).
fn format_wall_time_notice(wall_time_ms: f64) -> String {
	format!("Wall time: {:.2} seconds", wall_time_ms / 1000.0)
}

/// `clampTimeout("bash", …)` with `tools.maxTimeout` default 0 (no cap).
const fn clamp_bash_timeout(raw: u32) -> u32 {
	let capped = raw;
	if capped < BASH_TIMEOUT_MIN {
		BASH_TIMEOUT_MIN
	} else if capped > BASH_TIMEOUT_MAX {
		BASH_TIMEOUT_MAX
	} else {
		capped
	}
}

/// The `bash` tool.
pub struct BashTool {
	cwd: PathBuf,
}

impl BashTool {
	#[must_use]
	pub fn new(cwd: impl Into<PathBuf>) -> Self {
		Self { cwd: cwd.into() }
	}
}

impl Tool for BashTool {
	fn name(&self) -> &'static str {
		"bash"
	}

	// `concurrency` uses the trait default (`Shared`). The TS `bash` tool declares
	// a dynamic `concurrency = (args) => args.pty ? "exclusive" : "shared"`
	// (`packages/coding-agent/src/tools/bash.ts:422`); the non-pty default is
	// `shared`, which the trait default matches. The args-driven pty→exclusive
	// upgrade is deferred to WP-1.4b (`Concurrency` doc registers the deferral).

	fn description(&self) -> &str {
		crate::prompts::BASH
	}

	fn input_schema(&self) -> Value {
		json!({
			"type": "object",
			"properties": {
				"command": { "type": "string", "description": "command to execute" },
				"env": {
					"type": "object",
					"description": "extra env vars",
					"additionalProperties": { "type": "string" },
				},
				"timeout": {
					"type": "number",
					"description": "timeout in seconds; 0 disables the command deadline; nonzero values are clamped to 1-3600",
				},
				"cwd": { "type": "string", "description": "working directory" },
				"pty": { "type": "boolean", "description": "run in pty mode" },
			},
			"required": ["command"],
			"additionalProperties": false,
		})
	}

	async fn execute(
		&self,
		_tool_call_id: &str,
		args: Value,
		ct: &pi_shell::cancel::CancelToken,
	) -> Result<ToolResult, ToolError> {
		let raw_command = args
			.get("command")
			.and_then(Value::as_str)
			.ok_or_else(|| ToolError::new("bash: missing required parameter `command`"))?;
		let mut cwd_arg: Option<String> = args.get("cwd").and_then(Value::as_str).map(str::to_owned);
		let raw_timeout = args
			.get("timeout")
			.and_then(Value::as_f64)
			.map_or(BASH_TIMEOUT_DEFAULT, |t| t as u32);

		// `normalizeBashEnv` (bash.ts:213).
		let env: Option<HashMap<String, String>> = match args.get("env").and_then(Value::as_object) {
			Some(env_obj) if !env_obj.is_empty() => {
				let mut normalized = HashMap::new();
				for (key, value) in env_obj {
					if !ENV_NAME_RE.is_match(key) {
						return Err(ToolError::new(format!("Invalid bash env name: {key}")));
					}
					let value = value.as_str().ok_or_else(|| {
						ToolError::new(format!("Invalid bash env value for {key}: expected string"))
					})?;
					normalized.insert(key.clone(), value.to_owned());
				}
				Some(normalized)
			},
			_ => None,
		};

		// Leading `cd <path> && …` extraction (bash.ts:759).
		let mut command = raw_command.to_owned();
		if cwd_arg.is_none()
			&& let Some(caps) = CD_PREFIX_RE.captures(&command)
		{
			let target = caps.get(1).expect("cd capture").as_str();
			if !target.contains(['$', '`', '(']) {
				let cleaned = target
					.trim()
					.trim_matches(|c| c == '"' || c == '\'')
					.to_owned();
				let full = caps.get(0).expect("cd match").as_str().len();
				command = command[full..].to_owned();
				cwd_arg = Some(cleaned);
			}
		}

		let command_cwd = cwd_arg
			.as_deref()
			.map_or_else(|| self.cwd.clone(), |c| resolve_to_cwd(c, &self.cwd));
		let cwd_meta = std::fs::metadata(&command_cwd).map_err(|_| {
			ToolError::new(format!("Working directory does not exist: {}", command_cwd.display()))
		})?;
		if !cwd_meta.is_dir() {
			return Err(ToolError::new(format!(
				"Working directory is not a directory: {}",
				command_cwd.display()
			)));
		}

		let timeout_disabled = raw_timeout == 0;
		let timeout_sec = if timeout_disabled {
			None
		} else {
			Some(clamp_bash_timeout(raw_timeout))
		};
		let timeout_ms = timeout_sec.map(|sec| sec * 1000);

		// One-shot `executeBash` equivalent.
		let sink = std::sync::Arc::new(std::sync::Mutex::new(OutputSinkLite::new()));
		let (tx, rx) = flume::unbounded::<String>();
		let drain_sink = sink.clone();
		let drain = tokio::spawn(async move {
			while let Ok(chunk) = rx.recv_async().await {
				drain_sink.lock().expect("sink lock").push(&chunk);
			}
		});

		// The one-shot runner takes its deadline from the CancelToken (the
		// `ShellExecuteOptions.timeout_ms` field is not consumed on this
		// path); bridge the caller's token into a timeout-carrying run token.
		let mut run_ct = pi_shell::cancel::CancelToken::new(timeout_ms);
		let run_abort = run_ct.emplace_abort_token();
		let parent_ct = ct.clone();
		let cancel_bridge = tokio::spawn(async move {
			let reason = parent_ct.wait().await;
			run_abort.abort(reason);
		});

		let wall_start = Instant::now();
		let exec_result = pi_shell::execute_shell(
			pi_shell::ShellExecuteOptions {
				command: command.clone(),
				cwd: Some(command_cwd.to_string_lossy().into_owned()),
				env,
				timeout_ms,
				..Default::default()
			},
			Some(tx),
			run_ct,
		)
		.await
		.map_err(|err| ToolError::new(format!("Shell execution failed: {err}")))?;
		cancel_bridge.abort();
		drain.await.ok();
		let wall_time_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

		let mut sink = sink.lock().expect("sink lock");
		let timed_out = exec_result.timed_out;
		let cancelled = exec_result.cancelled;

		if cancelled && !timed_out {
			// `executeBash` prepends `[Command cancelled]` via the sink notice.
			let out = sink.dump(Some("Command cancelled"));
			return Err(ToolError::new(out));
		}

		let output = if timed_out {
			let annotation = timeout_sec.map_or_else(
				|| "Command timed out".to_owned(),
				|sec| format!("Command timed out after {sec} seconds"),
			);
			sink.dump(Some(&annotation))
		} else {
			sink.dump(None)
		};

		// `#buildCompletedResult` (bash.ts:481).
		let exit_code = exec_result.exit_code;
		let failed_exit = matches!(exit_code, Some(code) if code != 0);
		let formatted_output = if output.is_empty() {
			"(no output)".to_owned()
		} else {
			output.clone()
		};
		let mut output_lines: Vec<String> = vec![formatted_output];
		output_lines.push(String::new());
		output_lines.push(format_wall_time_notice(wall_time_ms));
		if failed_exit {
			output_lines.push(String::new());
			output_lines
				.push(format!("Command exited with code {}", exit_code.expect("failed exit code")));
		}

		if timed_out {
			let message = timeout_sec.map_or_else(
				|| "Command timed out".to_owned(),
				|sec| format!("Command timed out after {sec} seconds"),
			);
			if !output.starts_with(&format!("[{message}]\n")) {
				output_lines.push(String::new());
				output_lines.push(format!("[{message}]"));
			}
			let text = enforce_inline_byte_cap(&output_lines.join("\n"));
			return Ok(ToolResult::text(text).error());
		}

		if exit_code.is_none() {
			return Err(ToolError::new(format!(
				"{}\n\nCommand failed: missing exit status",
				output_lines.join("\n")
			)));
		}

		let text = enforce_inline_byte_cap(&output_lines.join("\n"));
		let mut result = ToolResult::text(text);
		if failed_exit {
			result = result.error();
		}
		Ok(result)
	}
}
