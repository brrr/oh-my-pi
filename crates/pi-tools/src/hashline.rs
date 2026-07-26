//! Hashline format primitives, ported from `packages/hashline/src/format.ts`
//! (the *format* face only — no patch applier) plus the strict prefix stripper
//! from `packages/hashline/src/prefixes.ts`.
//!
//! The content-hash tag ([`compute_file_hash`]) is XXH32 (seed 0) of the
//! whitespace-normalized text, low 16 bits, rendered as 4 uppercase hex digits
//! — byte-for-byte parity with the TS `Bun.hash.xxHash32` path
//! (`format.ts:112`), pinned by cross-validation tests below.

use std::sync::LazyLock;

use regex::Regex;
use twox_hash::XxHash32;

// ─── Sigils & separators (format.ts:10-88) ──────────────────────────────────

/// File-section header opening delimiter.
pub const HL_FILE_PREFIX: &str = "[";
/// File-section header closing delimiter.
pub const HL_FILE_SUFFIX: &str = "]";
/// Separator between a hashline file path and its snapshot tag.
pub const HL_FILE_HASH_SEP: &str = "#";
/// Separator between a line number and displayed line content.
pub const HL_LINE_BODY_SEP: &str = ":";
/// Number of hex characters in a content-derived file-hash tag.
pub const HL_FILE_HASH_LENGTH: usize = 4;

// ─── Display formatting (format.ts) ─────────────────────────────────────────

/// `formatNumberedLine` (format.ts): `LINE:TEXT`.
#[must_use]
pub fn format_numbered_line(line_number: usize, line: &str) -> String {
	format!("{line_number}{HL_LINE_BODY_SEP}{line}")
}

/// `formatNumberedLines` (format.ts): number every line starting at
/// `start_line` (1-based), joined by `\n`.
#[must_use]
pub fn format_numbered_lines(text: &str, start_line: usize) -> String {
	text
		.split('\n')
		.enumerate()
		.map(|(i, line)| format_numbered_line(start_line + i, line))
		.collect::<Vec<_>>()
		.join("\n")
}

/// `formatHashlineHeader` (format.ts): `[path#HASH]`.
#[must_use]
pub fn format_hashline_header(file_path: &str, file_hash: &str) -> String {
	format!("{HL_FILE_PREFIX}{file_path}{HL_FILE_HASH_SEP}{file_hash}{HL_FILE_SUFFIX}")
}

// ─── Content hash (format.ts:96-115) ────────────────────────────────────────

/// `normalizeFileHashText` (format.ts:100): trim trailing `[ \t\r]` from every
/// line (and the final line). Implemented by splitting on `\n` and trimming
/// each segment — equivalent to the TS `replace(/[ \t\r]+(?=\n|$)/g, "")`
/// (Rust `regex` has no lookahead, but the semantics are identical).
fn normalize_file_hash_text(text: &str) -> String {
	text
		.split('\n')
		.map(|segment| segment.trim_end_matches([' ', '\t', '\r']))
		.collect::<Vec<_>>()
		.join("\n")
}

/// `computeFileHash` (format.ts:112): XXH32(seed 0) of the normalized text, low
/// 16 bits, as 4 uppercase hex digits.
#[must_use]
pub fn compute_file_hash(text: &str) -> String {
	let normalized = normalize_file_hash_text(text);
	let low16 = XxHash32::oneshot(0, normalized.as_bytes()) & 0xffff;
	let width = HL_FILE_HASH_LENGTH;
	format!("{low16:0width$X}")
}

// ─── Strict prefix stripping (prefixes.ts) ──────────────────────────────────

/// `HL_PREFIX_RE` (prefixes.ts:19).
static HL_PREFIX_RE: LazyLock<Regex> =
	LazyLock::new(|| Regex::new(r"^\s*(?:>>>|>>)?\s*(?:[+*-]\s*)?\d+:").expect("HL_PREFIX_RE"));
/// `HL_HEADER_RE` (prefixes.ts:21).
static HL_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(&format!(r"^\s*\[[^#\r\n]+#[0-9a-fA-F]{{{HL_FILE_HASH_LENGTH}}}\]\s*$"))
		.expect("HL_HEADER_RE")
});
/// `READ_TRUNCATION_NOTICE_RE` (prefixes.ts:23).
static READ_TRUNCATION_NOTICE_RE: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(
		r"^\[(?:Showing lines \d+-\d+ of \d+|\d+ more lines? in (?:file|\S+))\b.*\bUse :L?\d+",
	)
	.expect("READ_TRUNCATION_NOTICE_RE")
});

/// `stripLeadingHashlinePrefixes` (prefixes.ts:25): strip leading hashline
/// prefixes repeatedly until stable.
fn strip_leading_hashline_prefixes(line: &str) -> String {
	let mut result = line.to_owned();
	loop {
		let replaced = HL_PREFIX_RE.replace(&result, "");
		if replaced == result {
			return result;
		}
		result = replaced.into_owned();
	}
}

/// `stripHashlinePrefixes` (prefixes.ts:121, strict variant): strip hashline
/// line-number prefixes only when *every* content line is hashline-prefixed;
/// returns the input unchanged otherwise.
#[must_use]
pub fn strip_hashline_prefixes(lines: &[String]) -> Vec<String> {
	let mut non_empty = 0usize;
	let mut header_count = 0usize;
	let mut hash_prefix_count = 0usize;

	for line in lines {
		if line.is_empty() {
			continue;
		}
		if READ_TRUNCATION_NOTICE_RE.is_match(line) {
			continue; // truncation notice: not counted as content
		}
		if HL_HEADER_RE.is_match(line) {
			non_empty += 1;
			header_count += 1;
			continue;
		}
		non_empty += 1;
		if HL_PREFIX_RE.is_match(line) {
			hash_prefix_count += 1;
		}
	}

	if non_empty == 0 {
		return lines.to_vec();
	}
	let content_line_count = non_empty - header_count;
	if content_line_count == 0 || hash_prefix_count != content_line_count {
		return lines.to_vec();
	}

	lines
		.iter()
		.filter(|line| !READ_TRUNCATION_NOTICE_RE.is_match(line) && !HL_HEADER_RE.is_match(line))
		.map(|line| strip_leading_hashline_prefixes(line))
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn numbered_line_and_lines() {
		assert_eq!(format_numbered_line(42, "hello"), "42:hello");
		assert_eq!(format_numbered_lines("a\nb\nc", 1), "1:a\n2:b\n3:c");
		assert_eq!(format_numbered_lines("x\ny", 10), "10:x\n11:y");
	}

	#[test]
	fn hashline_header() {
		assert_eq!(format_hashline_header("src/foo.ts", "1A2B"), "[src/foo.ts#1A2B]");
	}

	// Cross-validated against the TS `computeFileHash` (Bun.hash.xxHash32) —
	// expected values captured via `bun -e` over `packages/hashline`:
	//   "第一行\n\n  第二行有中文  \n第三行\t"        => E77D  (raw32 976545661)
	//   "line one   \nline two\t\t\nline three\n"      => 2172  (raw32 719987058)
	//   "hello world\nfoo bar\n"                        => 469A  (raw32 2333492890)
	#[test]
	fn compute_file_hash_matches_bun_xxhash32() {
		assert_eq!(compute_file_hash("第一行\n\n  第二行有中文  \n第三行\t"), "E77D");
		assert_eq!(compute_file_hash("line one   \nline two\t\t\nline three\n"), "2172");
		assert_eq!(compute_file_hash("hello world\nfoo bar\n"), "469A");
	}

	#[test]
	fn hash_is_stable_under_trailing_whitespace_normalization() {
		// Trailing spaces/tabs/CR before newlines must not change the tag.
		assert_eq!(compute_file_hash("a\nb\nc"), compute_file_hash("a  \nb\t\nc"));
		assert_eq!(compute_file_hash("a\nb"), compute_file_hash("a\r\nb"));
	}

	#[test]
	fn strip_prefixes_when_all_content_lines_prefixed() {
		let lines = vec!["1:foo".to_owned(), "2:bar".to_owned(), "3:baz".to_owned()];
		assert_eq!(strip_hashline_prefixes(&lines), vec!["foo", "bar", "baz"]);
	}

	#[test]
	fn strip_leaves_unprefixed_input_untouched() {
		let lines = vec!["1:foo".to_owned(), "plain".to_owned()];
		// Not every content line is prefixed → unchanged.
		assert_eq!(strip_hashline_prefixes(&lines), lines);
	}

	#[test]
	fn strip_drops_header_and_truncation_notice_lines() {
		let lines = vec!["[src/foo.ts#1A2B]".to_owned(), "1:foo".to_owned(), "2:bar".to_owned()];
		// Header is not content; both remaining content lines are prefixed.
		assert_eq!(strip_hashline_prefixes(&lines), vec!["foo", "bar"]);
	}
}
