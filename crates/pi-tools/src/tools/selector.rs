//! Path-embedded selector parsing, ported from
//! `packages/coding-agent/src/tools/path-utils.ts` (`parseLineRanges`,
//! `splitPathAndSel`) and `packages/coding-agent/src/tools/read.ts`
//! (`parseSel`, `selToOffsetLimit`).

use std::sync::LazyLock;

use regex::Regex;

use crate::tool::ToolError;

/// `LineRange` (path-utils.ts:199): inclusive; `end_line == None` is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRange {
	pub start_line: usize,
	pub end_line:   Option<usize>,
}

/// `RANGE_CHUNK_SRC` / `RANGE_LIST_SRC` (path-utils.ts:15).
const RANGE_CHUNK_SRC: &str = r"L?\d+(?:(?:[-+]|\.\.)L?\d+|-|\.\.)?";

/// `FILE_LINE_RANGE_RE` (path-utils.ts:17): full trailing-selector candidate.
static FILE_LINE_RANGE_RE: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(&format!(r"(?i)^(?:{RANGE_CHUNK_SRC}(?:,{RANGE_CHUNK_SRC})*|raw|conflicts)$"))
		.expect("FILE_LINE_RANGE_RE")
});
/// `FILE_LINE_RANGE_ONLY_RE` (path-utils.ts:18).
static FILE_LINE_RANGE_ONLY_RE: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(&format!(r"(?i)^{RANGE_CHUNK_SRC}(?:,{RANGE_CHUNK_SRC})*$"))
		.expect("FILE_LINE_RANGE_ONLY_RE")
});
/// `LINE_RANGE_CHUNK_RE` (path-utils.ts:204).
static LINE_RANGE_CHUNK_RE: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(r"(?i)^L?(\d+)(?:(\.\.|[-+])L?(\d+)?)?$").expect("LINE_RANGE_CHUNK_RE")
});

/// `parseLineRangeChunk` (path-utils.ts:207). `Ok(None)` = not range-shaped.
pub fn parse_line_range_chunk(sel: &str) -> Result<Option<LineRange>, ToolError> {
	let Some(caps) = LINE_RANGE_CHUNK_RE.captures(sel) else {
		return Ok(None);
	};
	let raw_start: i64 = caps[1].parse().unwrap_or(i64::MAX);
	if raw_start < 1 {
		return Err(ToolError::new("Line selector 0 is invalid; lines are 1-indexed. Use :1."));
	}
	let sep = caps
		.get(2)
		.map(|m| if m.as_str() == ".." { "-" } else { m.as_str() });
	let rhs: Option<i64> = caps.get(3).map(|m| m.as_str().parse().unwrap_or(i64::MAX));
	let mut raw_end: Option<i64> = None;
	match sep {
		Some("+") => match rhs {
			Some(count) if count >= 1 => raw_end = Some(raw_start + count - 1),
			other => {
				return Err(ToolError::new(format!(
					"Invalid range {raw_start}+{}: count must be >= 1.",
					other.unwrap_or(0)
				)));
			},
		},
		Some("-") => {
			if let Some(end) = rhs {
				if end < raw_start {
					return Err(ToolError::new(format!(
						"Invalid range {raw_start}-{end}: end must be >= start."
					)));
				}
				raw_end = Some(end);
			}
		},
		_ => {},
	}
	Ok(Some(LineRange {
		start_line: usize::try_from(raw_start).unwrap_or(usize::MAX),
		end_line:   raw_end.map(|end| usize::try_from(end).unwrap_or(usize::MAX)),
	}))
}

/// `parseLineRanges` (path-utils.ts): comma list, sorted + merged.
pub fn parse_line_ranges(sel: &str) -> Result<Option<Vec<LineRange>>, ToolError> {
	let mut parsed = Vec::new();
	for chunk in sel.split(',') {
		match parse_line_range_chunk(chunk)? {
			Some(range) => parsed.push(range),
			None => return Ok(None),
		}
	}
	if parsed.is_empty() {
		return Ok(None);
	}
	parsed.sort_by_key(|range| range.start_line);
	let mut merged: Vec<LineRange> = vec![parsed[0]];
	for current in &parsed[1..] {
		let last = merged.last_mut().expect("non-empty");
		let Some(last_end) = last.end_line else {
			continue; // open-ended absorbs everything after it
		};
		if current.start_line <= last_end + 1 {
			if current.end_line.is_none_or(|end| end > last_end) {
				last.end_line = current.end_line;
			}
			continue;
		}
		merged.push(*current);
	}
	Ok(Some(merged))
}

/// `splitPathAndSel` (path-utils.ts): peel a trailing `:sel` chunk (plus the
/// compound `:range:raw` / `:raw:range` forms).
#[must_use]
pub fn split_path_and_sel(raw_path: &str) -> (String, Option<String>) {
	let Some(colon) = raw_path.rfind(':') else {
		return (raw_path.to_owned(), None);
	};
	if colon == 0 {
		return (raw_path.to_owned(), None);
	}
	let candidate = &raw_path[colon + 1..];
	if !FILE_LINE_RANGE_RE.is_match(candidate) {
		return (raw_path.to_owned(), None);
	}
	let mut base_path = &raw_path[..colon];
	let mut sel = candidate.to_owned();
	if let Some(inner_colon) = base_path.rfind(':')
		&& inner_colon > 0
	{
		let inner = &base_path[inner_colon + 1..];
		let inner_is_raw = inner.eq_ignore_ascii_case("raw");
		let outer_is_raw = candidate.eq_ignore_ascii_case("raw");
		let inner_is_range = FILE_LINE_RANGE_ONLY_RE.is_match(inner);
		let outer_is_range = FILE_LINE_RANGE_ONLY_RE.is_match(candidate);
		if (inner_is_raw && outer_is_range) || (inner_is_range && outer_is_raw) {
			sel = format!("{inner}:{candidate}");
			base_path = &base_path[..inner_colon];
		}
	}
	(base_path.to_owned(), Some(sel))
}

/// `ParsedSelector` (read.ts:786).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedSelector {
	None,
	Raw,
	Conflicts,
	Lines { ranges: Vec<LineRange>, raw: bool },
}

fn selector_chunk_looks_read_like(chunk: &str) -> Result<bool, ToolError> {
	let lower = chunk.to_lowercase();
	if lower == "raw" || lower == "conflicts" {
		return Ok(true);
	}
	static NEGATIVE_RE: LazyLock<Regex> =
		LazyLock::new(|| Regex::new(r"^-\d+(?:[-+]\d+)?$").expect("NEGATIVE_RE"));
	if NEGATIVE_RE.is_match(chunk) {
		return Ok(true);
	}
	Ok(parse_line_ranges(chunk)?.is_some())
}

fn invalid_selector(sel: &str) -> ToolError {
	ToolError::new(format!(
		"Invalid selector ':{sel}'. Use :N, :N-M, :N+K, :N- (open-ended), a comma-separated list of \
		 ranges, :raw, or a range combined with raw (e.g. :raw:50-100)."
	))
}

/// `parseSel` (read.ts:815).
pub fn parse_sel(sel: Option<&str>) -> Result<ParsedSelector, ToolError> {
	let Some(sel) = sel else {
		return Ok(ParsedSelector::None);
	};
	if sel.is_empty() {
		return Ok(ParsedSelector::None);
	}
	if sel.contains(':') {
		let chunks: Vec<&str> = sel.split(':').collect();
		if chunks.len() == 2 {
			let (a, b) = (chunks[0], chunks[1]);
			let a_is_raw = a.eq_ignore_ascii_case("raw");
			let b_is_raw = b.eq_ignore_ascii_case("raw");
			let range_chunk = if a_is_raw {
				Some(b)
			} else if b_is_raw {
				Some(a)
			} else {
				None
			};
			if range_chunk.is_some_and(|_| a_is_raw || b_is_raw)
				&& let Some(chunk) = range_chunk
				&& let Some(ranges) = parse_line_ranges(chunk)?
			{
				return Ok(ParsedSelector::Lines { ranges, raw: true });
			}
		}
		let mut all_read_like = true;
		for chunk in &chunks {
			if !selector_chunk_looks_read_like(chunk)? {
				all_read_like = false;
				break;
			}
		}
		if all_read_like {
			return Err(invalid_selector(sel));
		}
		return Ok(ParsedSelector::None);
	}
	if sel.eq_ignore_ascii_case("raw") {
		return Ok(ParsedSelector::Raw);
	}
	if sel.eq_ignore_ascii_case("conflicts") {
		return Ok(ParsedSelector::Conflicts);
	}
	if let Some(ranges) = parse_line_ranges(sel)? {
		return Ok(ParsedSelector::Lines { ranges, raw: false });
	}
	Ok(ParsedSelector::None)
}

/// `selToOffsetLimit` (read.ts:858): first range → 1-based offset + limit.
#[must_use]
pub fn sel_to_offset_limit(parsed: &ParsedSelector) -> (Option<usize>, Option<usize>) {
	if let ParsedSelector::Lines { ranges, .. } = parsed
		&& let Some(first) = ranges.first()
	{
		let limit = first.end_line.map(|end| end - first.start_line + 1);
		return (Some(first.start_line), limit);
	}
	(None, None)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn split_peels_trailing_ranges_only() {
		assert_eq!(
			split_path_and_sel("a/b.txt:5-10"),
			("a/b.txt".to_owned(), Some("5-10".to_owned()))
		);
		assert_eq!(split_path_and_sel("a/b.txt:5"), ("a/b.txt".to_owned(), Some("5".to_owned())));
		assert_eq!(split_path_and_sel("a/b.txt"), ("a/b.txt".to_owned(), None));
		assert_eq!(split_path_and_sel("a/b:xyz"), ("a/b:xyz".to_owned(), None));
	}

	#[test]
	fn parse_sel_basic_forms() {
		assert_eq!(parse_sel(None).unwrap(), ParsedSelector::None);
		assert_eq!(parse_sel(Some("raw")).unwrap(), ParsedSelector::Raw);
		let ranges = parse_sel(Some("5-10")).unwrap();
		assert_eq!(ranges, ParsedSelector::Lines {
			ranges: vec![LineRange { start_line: 5, end_line: Some(10) }],
			raw:    false,
		});
		let open = parse_sel(Some("7")).unwrap();
		assert_eq!(open, ParsedSelector::Lines {
			ranges: vec![LineRange { start_line: 7, end_line: None }],
			raw:    false,
		});
	}

	#[test]
	fn ranges_merge_and_sort() {
		let merged = parse_line_ranges("10-12,1-3,2-5").unwrap().unwrap();
		assert_eq!(merged, vec![LineRange { start_line: 1, end_line: Some(5) }, LineRange {
			start_line: 10,
			end_line:   Some(12),
		},]);
	}

	#[test]
	fn invalid_bounds_error_text() {
		let err = parse_line_ranges("5-2").unwrap_err();
		assert_eq!(err.to_string(), "Invalid range 5-2: end must be >= start.");
		let err = parse_line_ranges("0").unwrap_err();
		assert_eq!(err.to_string(), "Line selector 0 is invalid; lines are 1-indexed. Use :1.");
	}
}
