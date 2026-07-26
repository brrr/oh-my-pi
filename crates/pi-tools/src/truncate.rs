//! Output truncation.
//!
//! Ported from `packages/coding-agent/src/session/streaming-output.ts`.
//! Covers the head-keeping variants used by tool output rendering:
//! [`truncate_head`] (line + byte aware, never returns a partial line; TS
//! `truncateHead` :288) and [`truncate_head_bytes`] (raw byte cap on a UTF-8
//! boundary; TS `truncateHeadBytes` :250 via `truncateBytesWindowed`). The
//! Rust port operates directly on UTF-8 byte indices of `&str`, which collapses
//! the TS "UTF-16 code-unit fast reject vs. exact UTF-8 byte" two-step into a
//! single byte comparison with identical results.

/// Default line cap (`streaming-output.ts:10`).
pub const DEFAULT_MAX_LINES: usize = 3000;
/// Default byte cap, 50 KiB (`streaming-output.ts:11`).
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// Which limit triggered truncation (`TruncationResult.truncatedBy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncatedBy {
	Lines,
	Bytes,
	Middle,
}

/// Result of content-level truncation (`interface TruncationResult` :90).
/// Fields the head path never populates (`elided*`) are omitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncationResult {
	pub content:                  String,
	pub truncated:                bool,
	pub truncated_by:             Option<TruncatedBy>,
	pub total_lines:              usize,
	pub total_bytes:              usize,
	pub output_lines:             Option<usize>,
	pub output_bytes:             Option<usize>,
	pub last_line_partial:        bool,
	pub first_line_exceeds_limit: bool,
}

impl TruncationResult {
	/// `noTruncResult` (:274): whole content, no truncation flags.
	fn no_trunc(content: &str, total_lines: usize, total_bytes: usize) -> Self {
		Self {
			content: content.to_owned(),
			truncated: false,
			truncated_by: None,
			total_lines,
			total_bytes,
			output_lines: None,
			output_bytes: None,
			last_line_partial: false,
			first_line_exceeds_limit: false,
		}
	}

	/// The `firstLineExceedsLimit` early return in `truncateHead` (:344).
	const fn first_line_exceeds(total_lines: usize, total_bytes: usize) -> Self {
		Self {
			content: String::new(),
			truncated: true,
			truncated_by: Some(TruncatedBy::Bytes),
			total_lines,
			total_bytes,
			output_lines: Some(0),
			output_bytes: Some(0),
			last_line_partial: false,
			first_line_exceeds_limit: true,
		}
	}
}

/// Line/byte caps (`interface TruncationOptions` :106). `None` falls back to
/// the `DEFAULT_MAX_*` constants.
#[derive(Debug, Clone, Copy, Default)]
pub struct TruncateOptions {
	pub max_lines: Option<usize>,
	pub max_bytes: Option<usize>,
}

/// Byte-level truncation result (`interface ByteTruncationResult` :124).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteTruncationResult {
	pub text:  String,
	pub bytes: usize,
}

/// `countNewlines` (:145).
fn count_newlines(text: &str) -> usize {
	text.bytes().filter(|&b| b == b'\n').count()
}

/// Truncate content from the head, keeping the first N lines/bytes.
///
/// Never returns a partial line: if the first line already exceeds the byte cap
/// the result is empty with `first_line_exceeds_limit = true` (`truncateHead`
/// :288).
#[must_use]
pub fn truncate_head(content: &str, options: &TruncateOptions) -> TruncationResult {
	let max_lines = options.max_lines.unwrap_or(DEFAULT_MAX_LINES);
	let max_bytes = options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);

	let total_bytes = content.len();
	let total_lines = count_newlines(content) + 1;

	if total_lines <= max_lines && total_bytes <= max_bytes {
		return TruncationResult::no_trunc(content, total_lines, total_bytes);
	}

	let bytes = content.as_bytes();
	let mut included_lines = 0usize;
	let mut bytes_used = 0usize;
	let mut cut_index = 0usize; // byte index where we cut (exclusive)
	let mut cursor = 0usize;
	let mut truncated_by = TruncatedBy::Lines;

	while included_lines < max_lines {
		// Newline byte position at or after `cursor`; `None` → last line.
		let nl = bytes[cursor..]
			.iter()
			.position(|&b| b == b'\n')
			.map(|off| cursor + off);
		let line_end = nl.unwrap_or(content.len());

		let sep_bytes = usize::from(included_lines > 0);
		// `remaining` may go negative in the TS source; model with i64.
		let remaining = max_bytes as i64 - bytes_used as i64 - sep_bytes as i64;
		if remaining < 0 {
			truncated_by = TruncatedBy::Bytes;
			break;
		}

		// Line UTF-8 byte length equals the byte-index span in Rust.
		let line_bytes = line_end - cursor;
		if line_bytes as i64 > remaining {
			truncated_by = TruncatedBy::Bytes;
			if included_lines == 0 {
				return TruncationResult::first_line_exceeds(total_lines, total_bytes);
			}
			break;
		}

		bytes_used += sep_bytes + line_bytes;
		included_lines += 1;

		cut_index = line_end; // exclude the newline after the last included line
		match nl {
			None => break,
			Some(pos) => cursor = pos + 1,
		}
	}

	if included_lines >= max_lines && bytes_used <= max_bytes {
		truncated_by = TruncatedBy::Lines;
	}

	TruncationResult {
		content: content[..cut_index].to_owned(),
		truncated: true,
		truncated_by: Some(truncated_by),
		total_lines,
		total_bytes,
		output_lines: Some(included_lines),
		output_bytes: Some(bytes_used),
		last_line_partial: false,
		first_line_exceeds_limit: false,
	}
}

/// Truncate a string to fit within a byte limit, keeping the head and landing
/// on a UTF-8 char boundary (`truncateHeadBytes` :250 / `truncateBytesWindowed`
/// head branch :180).
#[must_use]
pub fn truncate_head_bytes(data: &str, max_bytes: usize) -> ByteTruncationResult {
	if max_bytes == 0 {
		return ByteTruncationResult { text: String::new(), bytes: 0 };
	}
	if data.len() <= max_bytes {
		return ByteTruncationResult { text: data.to_owned(), bytes: data.len() };
	}

	// `findUtf8BoundaryBackward`: floor `max_bytes` to a char boundary.
	let mut end = max_bytes.min(data.len());
	while end > 0 && !data.is_char_boundary(end) {
		end -= 1;
	}
	if end == 0 {
		return ByteTruncationResult { text: String::new(), bytes: 0 };
	}
	ByteTruncationResult { text: data[..end].to_owned(), bytes: end }
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn no_truncation_returns_whole_content() {
		let content = "alpha\nbeta\ngamma";
		let out = truncate_head(content, &TruncateOptions::default());
		assert!(!out.truncated);
		assert_eq!(out.truncated_by, None);
		assert_eq!(out.content, content);
		assert_eq!(out.total_lines, 3);
		assert_eq!(out.total_bytes, content.len());
		assert_eq!(out.output_lines, None);
	}

	#[test]
	fn line_truncation_keeps_head_lines_without_trailing_newline() {
		let content = "a\nb\nc\nd\ne";
		let out = truncate_head(content, &TruncateOptions { max_lines: Some(3), max_bytes: None });
		assert!(out.truncated);
		assert_eq!(out.truncated_by, Some(TruncatedBy::Lines));
		assert_eq!(out.content, "a\nb\nc"); // no trailing newline after last kept line
		assert_eq!(out.output_lines, Some(3));
		assert_eq!(out.total_lines, 5);
		assert!(!out.first_line_exceeds_limit);
	}

	#[test]
	fn byte_truncation_stops_before_overrunning_limit() {
		// Five 4-byte lines ("aaaa".."eeee"); cap at 9 bytes: "aaaa" (4) + sep
		// (1) + "bbbb" (4) = 9 fits, next line would overrun.
		let content = "aaaa\nbbbb\ncccc\ndddd\neeee";
		let out = truncate_head(content, &TruncateOptions { max_lines: None, max_bytes: Some(9) });
		assert!(out.truncated);
		assert_eq!(out.truncated_by, Some(TruncatedBy::Bytes));
		assert_eq!(out.content, "aaaa\nbbbb");
		assert_eq!(out.output_bytes, Some(9));
		assert_eq!(out.output_lines, Some(2));
	}

	#[test]
	fn first_line_exceeding_byte_cap_yields_empty_flagged_result() {
		let content = "this-single-line-is-too-long\nsecond";
		let out = truncate_head(content, &TruncateOptions { max_lines: None, max_bytes: Some(5) });
		assert!(out.truncated);
		assert!(out.first_line_exceeds_limit);
		assert_eq!(out.content, "");
		assert_eq!(out.output_lines, Some(0));
		assert_eq!(out.truncated_by, Some(TruncatedBy::Bytes));
	}

	#[test]
	fn head_bytes_no_truncation() {
		let out = truncate_head_bytes("hello", 50);
		assert_eq!(out.text, "hello");
		assert_eq!(out.bytes, 5);
	}

	#[test]
	fn head_bytes_lands_on_utf8_boundary() {
		// "a中b" = 0x61, 0xE4B8AD (3 bytes), 0x62. Cap at 3 bytes must not split
		// the 3-byte char; floor lands at byte 1 ("a").
		let out = truncate_head_bytes("a中b", 3);
		assert_eq!(out.text, "a");
		assert_eq!(out.bytes, 1);
	}

	#[test]
	fn head_bytes_zero_cap_is_empty() {
		let out = truncate_head_bytes("anything", 0);
		assert_eq!(out.text, "");
		assert_eq!(out.bytes, 0);
	}
}
