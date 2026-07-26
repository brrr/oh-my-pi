//! Line entries with bracket block context, ported from
//! `packages/coding-agent/src/utils/block-context.ts`.
//!
//! Deferred vs. TS: the tree-sitter `enclosingBlockBoundaries` fast path
//! (`nativeBlockContext`) is not ported — the Rust core always uses the
//! lexical bracket scan, which is byte-identical to the TS behavior for
//! sources tree-sitter cannot parse (e.g. `.txt` fixtures). Golden fixtures
//! for `read` therefore stick to extensions without a tree-sitter grammar.

use std::collections::{BTreeMap, BTreeSet};

/// `LineEntry` (block-context.ts:27).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineEntry {
	Line { line_number: usize, text: String, context: bool },
	Ellipsis,
}

/// Inclusive 1-based line span.
#[derive(Debug, Clone, Copy)]
pub struct LineSpan {
	pub start_line: usize,
	pub end_line:   usize,
}

fn normalize_line_spans(spans: &[LineSpan], total_lines: usize) -> Vec<LineSpan> {
	if total_lines == 0 {
		return Vec::new();
	}
	let mut normalized: Vec<LineSpan> = spans
		.iter()
		.filter_map(|span| {
			let start = span.start_line.max(1);
			let end = span.end_line.min(total_lines);
			(end >= start).then_some(LineSpan { start_line: start, end_line: end })
		})
		.collect();
	if normalized.len() <= 1 {
		return normalized;
	}
	normalized.sort_by(|a, b| {
		a.start_line
			.cmp(&b.start_line)
			.then(a.end_line.cmp(&b.end_line))
	});
	let mut merged: Vec<LineSpan> = Vec::new();
	for span in normalized {
		if let Some(previous) = merged.last_mut()
			&& span.start_line <= previous.end_line + 1
		{
			previous.end_line = previous.end_line.max(span.end_line);
			continue;
		}
		merged.push(span);
	}
	merged
}

const OPENERS: &[(char, char)] = &[('(', ')'), ('[', ']'), ('{', '}')];

fn close_to_open(ch: char) -> Option<char> {
	OPENERS
		.iter()
		.find(|(_, close)| *close == ch)
		.map(|(open, _)| *open)
}

fn is_opener(ch: char) -> bool {
	OPENERS.iter().any(|(open, _)| *open == ch)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScannerMode {
	Code,
	Single,
	Double,
	Template,
	BlockComment,
}

struct StackEntry {
	opener:      char,
	line_number: usize,
	visible:     bool,
}

fn is_hash_comment_start(line: &[char], index: usize) -> bool {
	if line.get(index) != Some(&'#') {
		return false;
	}
	line[..index].iter().all(|&ch| ch == ' ' || ch == '\t')
}

/// `lexicalBracketContext` (block-context.ts:148): pair `()[]{}` skipping
/// strings and comments; report the matching line when exactly one endpoint is
/// visible.
fn lexical_bracket_context(
	full_lines: &[String],
	visible: &BTreeSet<usize>,
) -> BTreeMap<usize, String> {
	let mut context: BTreeMap<usize, String> = BTreeMap::new();
	let mut stack: Vec<StackEntry> = Vec::new();
	let mut mode = ScannerMode::Code;
	let mut escaped = false;

	for (line_index, line_str) in full_lines.iter().enumerate() {
		let line_number = line_index + 1;
		let line: Vec<char> = line_str.chars().collect();
		let line_visible = visible.contains(&line_number);
		let mut index = 0usize;
		while index < line.len() {
			let ch = line[index];
			let next = line.get(index + 1).copied().unwrap_or('\0');

			if mode == ScannerMode::BlockComment {
				if ch == '*' && next == '/' {
					mode = ScannerMode::Code;
					index += 2;
					continue;
				}
				index += 1;
				continue;
			}

			if matches!(mode, ScannerMode::Single | ScannerMode::Double | ScannerMode::Template) {
				if escaped {
					escaped = false;
					index += 1;
					continue;
				}
				if ch == '\\' {
					escaped = true;
					index += 1;
					continue;
				}
				if (mode == ScannerMode::Single && ch == '\'')
					|| (mode == ScannerMode::Double && ch == '"')
					|| (mode == ScannerMode::Template && ch == '`')
				{
					mode = ScannerMode::Code;
				}
				index += 1;
				continue;
			}

			if ch == '/' && next == '/' {
				break;
			}
			if ch == '/' && next == '*' {
				mode = ScannerMode::BlockComment;
				index += 2;
				continue;
			}
			if is_hash_comment_start(&line, index) {
				break;
			}
			if ch == '\'' {
				mode = ScannerMode::Single;
				escaped = false;
				index += 1;
				continue;
			}
			if ch == '"' {
				mode = ScannerMode::Double;
				escaped = false;
				index += 1;
				continue;
			}
			if ch == '`' {
				mode = ScannerMode::Template;
				escaped = false;
				index += 1;
				continue;
			}

			if is_opener(ch) {
				stack.push(StackEntry { opener: ch, line_number, visible: line_visible });
				index += 1;
				continue;
			}

			if let Some(opener) = close_to_open(ch)
				&& let Some(match_index) = stack.iter().rposition(|entry| entry.opener == opener)
			{
				let matched = stack.remove(match_index);
				if line_visible && !matched.visible {
					context.insert(matched.line_number, full_lines[matched.line_number - 1].clone());
				}
				if matched.visible && !line_visible {
					context.insert(line_number, line_str.clone());
				}
			}

			index += 1;
		}

		if matches!(mode, ScannerMode::Single | ScannerMode::Double) {
			mode = ScannerMode::Code;
			escaped = false;
		}
	}

	for line_number in visible {
		context.remove(line_number);
	}
	context
}

/// `buildLineEntriesWithBlockContext` (block-context.ts:275) with the lexical
/// fallback only. `line_text` substitutes display text for a line number
/// (visible column-truncated lines vs. raw context lines).
#[must_use]
pub fn build_line_entries_with_block_context(
	full_lines: &[String],
	visible_spans: &[LineSpan],
	mut line_text: impl FnMut(usize, &str, bool) -> String,
) -> Vec<LineEntry> {
	let spans = normalize_line_spans(visible_spans, full_lines.len());
	let mut visible: BTreeSet<usize> = BTreeSet::new();
	for span in &spans {
		for line in span.start_line..=span.end_line {
			visible.insert(line);
		}
	}
	let context = if visible.is_empty() || visible.len() >= full_lines.len() {
		BTreeMap::new()
	} else {
		lexical_bracket_context(full_lines, &visible)
	};

	let mut all_lines: BTreeSet<usize> = visible.clone();
	all_lines.extend(context.keys().copied());

	let mut entries = Vec::new();
	let mut previous: Option<usize> = None;
	for line_number in all_lines {
		if let Some(prev) = previous
			&& line_number > prev + 1
		{
			entries.push(LineEntry::Ellipsis);
		}
		let source_text = full_lines.get(line_number - 1).map_or("", String::as_str);
		let is_context = context.contains_key(&line_number);
		entries.push(LineEntry::Line {
			line_number,
			text: line_text(line_number, source_text, is_context),
			context: is_context,
		});
		previous = Some(line_number);
	}
	entries
}

#[cfg(test)]
mod tests {
	use super::*;

	fn lines(text: &str) -> Vec<String> {
		text.split('\n').map(str::to_owned).collect()
	}

	#[test]
	fn full_visibility_yields_plain_lines() {
		let full = lines("a\nb\nc");
		let entries = build_line_entries_with_block_context(
			&full,
			&[LineSpan { start_line: 1, end_line: 3 }],
			|_, text, _| text.to_owned(),
		);
		assert_eq!(entries.len(), 3);
		assert!(
			entries
				.iter()
				.all(|e| matches!(e, LineEntry::Line { context: false, .. }))
		);
	}

	#[test]
	fn bracket_context_surfaces_off_window_opener() {
		// Context is recorded only when exactly one endpoint of a bracket pair
		// is visible (block-context.ts:233-234): the visible closer on line 5
		// pairs the off-window opener on line 3.
		let full = lines("fn outer {\n  a\n  inner {\n  b\n  }\n}\ntail");
		let entries = build_line_entries_with_block_context(
			&full,
			&[LineSpan { start_line: 4, end_line: 5 }],
			|_, text, _| text.to_owned(),
		);
		let rendered: Vec<String> = entries
			.iter()
			.map(|e| match e {
				LineEntry::Line { line_number, context, .. } => format!("{line_number}:{context}"),
				LineEntry::Ellipsis => "…".to_owned(),
			})
			.collect();
		assert_eq!(rendered, vec!["3:true", "4:false", "5:false"]);
	}

	#[test]
	fn comments_and_strings_are_skipped() {
		// Brackets inside strings (line 1) and line comments (line 2) never
		// enter the stack; the visible closer on line 5 pairs the real opener
		// on line 3.
		let full = lines("x = \"{\"\n// {\nreal {\n  y\n}\nz");
		let entries = build_line_entries_with_block_context(
			&full,
			&[LineSpan { start_line: 4, end_line: 5 }],
			|_, text, _| text.to_owned(),
		);
		let context_lines: Vec<usize> = entries
			.iter()
			.filter_map(|e| match e {
				LineEntry::Line { line_number, context: true, .. } => Some(*line_number),
				_ => None,
			})
			.collect();
		assert_eq!(context_lines, vec![3]);
	}
}
