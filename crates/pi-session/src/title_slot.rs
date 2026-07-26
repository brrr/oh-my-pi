//! Fixed-width first-line title slot. Rust mirror of
//! `packages/coding-agent/src/session/session-title-slot.ts`.
//!
//! The optional first physical line of a session file is a 256-byte (including
//! the trailing newline) JSON object carrying the mutable session title, padded
//! with spaces to the fixed width so the title can be overwritten in place
//! without rewriting the whole file. A first line that is not a valid slot is a
//! real entry and is left for the entry parser (loader tolerance).

use serde::Serialize;
use serde_json::Value;

use crate::entries::SESSION_TITLE_SLOT_BYTES;

/// Parsed title slot after the fixed-width padding is folded away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TitleSlot {
	pub title:      String,
	pub source:     Option<String>,
	pub updated_at: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SlotWire<'a> {
	#[serde(rename = "type")]
	kind:       &'a str,
	v:          u8,
	title:      &'a str,
	#[serde(skip_serializing_if = "Option::is_none")]
	source:     Option<&'a str>,
	updated_at: &'a str,
	pad:        &'a str,
}

/// One physical slot line, `JSON.stringify(slot) + "\n"`, byte-identical to the
/// TS `titleSlotLine`.
fn slot_line(title: &str, source: Option<&str>, updated_at: &str, pad: &str) -> String {
	let wire = SlotWire { kind: "title", v: 1, title, source, updated_at, pad };
	// serde_json compact output matches JSON.stringify for these scalar fields.
	let mut line = serde_json::to_string(&wire).expect("title slot serializes");
	line.push('\n');
	line
}

/// Largest code-point prefix of `title` whose unpadded slot line still fits the
/// fixed width — mirrors the TS `truncateTitleForSlot` binary search.
fn truncate_title_for_slot(title: &str, source: Option<&str>, updated_at: &str) -> String {
	let code_points: Vec<char> = title.chars().collect();
	let mut low = 0usize;
	let mut high = code_points.len();
	let mut best = String::new();
	while low <= high {
		let mid = low.midpoint(high);
		let candidate: String = code_points[..mid].iter().collect();
		if slot_line(&candidate, source, updated_at, "").len() <= SESSION_TITLE_SLOT_BYTES {
			best = candidate;
			low = mid + 1;
		} else {
			if mid == 0 {
				break;
			}
			high = mid - 1;
		}
	}
	best
}

/// Serialize the fixed-width title slot: exactly [`SESSION_TITLE_SLOT_BYTES`]
/// bytes including the trailing newline.
pub fn serialize_title_slot(title: &str, source: Option<&str>, updated_at: &str) -> String {
	let title = truncate_title_for_slot(title, source, updated_at);
	let unpadded = slot_line(&title, source, updated_at, "");
	let pad_bytes = SESSION_TITLE_SLOT_BYTES.saturating_sub(unpadded.len());
	let pad = " ".repeat(pad_bytes);
	let line = slot_line(&title, source, updated_at, &pad);
	debug_assert_eq!(line.len(), SESSION_TITLE_SLOT_BYTES, "title slot must be fixed width");
	line
}

fn is_title_source(value: &str) -> bool {
	value == "auto" || value == "user"
}

/// Parse a physical title slot line. Returns `None` for a legacy header line
/// (i.e. a real entry occupying the first line).
pub fn parse_title_slot_line(line: &str) -> Option<TitleSlot> {
	let value: Value = serde_json::from_str(line).ok()?;
	let obj = value.as_object()?;
	if obj.get("type").and_then(Value::as_str) != Some("title") {
		return None;
	}
	if obj.get("v").and_then(Value::as_u64) != Some(1) {
		return None;
	}
	let title = obj.get("title").and_then(Value::as_str)?;
	let updated_at = obj.get("updatedAt").and_then(Value::as_str)?;
	obj.get("pad").and_then(Value::as_str)?; // pad must be a string, value ignored
	let source = match obj.get("source") {
		None | Some(Value::Null) => None,
		Some(Value::String(s)) if is_title_source(s) => Some(s.clone()),
		Some(_) => return None,
	};
	Some(TitleSlot { title: title.to_string(), source, updated_at: updated_at.to_string() })
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn empty_title_slot_is_fixed_width() {
		let line = serialize_title_slot("", None, "2026-01-01T00:00:00.000Z");
		assert_eq!(line.len(), SESSION_TITLE_SLOT_BYTES);
		assert!(line.ends_with('\n'));
		let parsed = parse_title_slot_line(line.trim_end()).unwrap();
		assert_eq!(parsed.title, "");
		assert_eq!(parsed.source, None);
	}

	#[test]
	fn title_and_source_roundtrip() {
		let line = serialize_title_slot("My Session Title", Some("user"), "2026-01-01T00:00:00.000Z");
		assert_eq!(line.len(), SESSION_TITLE_SLOT_BYTES);
		let parsed = parse_title_slot_line(line.trim_end()).unwrap();
		assert_eq!(parsed.title, "My Session Title");
		assert_eq!(parsed.source.as_deref(), Some("user"));
	}

	#[test]
	fn non_slot_first_line_is_none() {
		assert!(parse_title_slot_line(r#"{"type":"session","id":"x"}"#).is_none());
		assert!(parse_title_slot_line("not json").is_none());
	}
}
