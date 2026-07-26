//! N-API shell over the pure `pi_term::keys` engine.
//!
//! Parses Kitty keyboard protocol sequences and matches codepoints plus
//! modifiers. All parsing/matching logic lives in [`pi_term::keys`]; this
//! module owns only the N-API surface: the `#[napi]` DTOs ([`KeyEventType`] /
//! [`ParsedKittyResult`]) and the string-boundary wrappers over the engine.
//!
//! # Example
//! ```ignore
//! // JS: native.matchesKittySequence("\x1b[65;5u", 65, 4) -> true
//! // JS: native.parseKey("\x1b[65;5u", false) -> "ctrl+a"
//! ```

use napi_derive::napi;

/// Event types from Kitty keyboard protocol (flag 2).
#[napi]
pub enum KeyEventType {
	/// Key press event.
	Press   = 1,
	/// Key repeat event.
	Repeat  = 2,
	/// Key release event.
	Release = 3,
}

#[inline]
fn optional_kitty_event_type(event: Option<u32>) -> Option<KeyEventType> {
	event.and_then(|ev| match ev {
		1 => Some(KeyEventType::Press),
		2 => Some(KeyEventType::Repeat),
		3 => Some(KeyEventType::Release),
		_ => None,
	})
}

/// Parsed Kitty keyboard protocol sequence result for a Kitty input sequence.
#[napi(object)]
pub struct ParsedKittyResult {
	/// Primary codepoint associated with the key.
	pub codepoint:       i32,
	/// Optional shifted key codepoint from the sequence.
	pub shifted_key:     Option<i32>,
	/// Optional base layout key codepoint from the sequence.
	pub base_layout_key: Option<i32>,
	/// Modifier bitmask (shift/alt/ctrl), excluding lock bits.
	pub modifier:        u32,
	/// Optional event type (1 = press, 2 = repeat, 3 = release).
	pub event_type:      Option<KeyEventType>,
}

/// Match Kitty protocol input against a codepoint and modifier mask.
///
/// Returns true when the parsed sequence matches the expected codepoint (or
/// base layout key) and modifier bits.
#[napi]
pub fn matches_kitty_sequence(
	data: String,
	expected_codepoint: i32,
	expected_modifier: u32,
) -> bool {
	pi_term::keys::matches_kitty_sequence(data.as_bytes(), expected_codepoint, expected_modifier)
}

/// Parse terminal input and return a normalized key identifier.
///
/// Returns a key id like "escape" or "ctrl+c", or None if unrecognized.
#[napi]
pub fn parse_key(data: String, kitty_protocol_active: bool) -> Option<String> {
	pi_term::keys::parse_key_inner(data.as_bytes(), kitty_protocol_active).map(|s| s.into_owned())
}

/// Check if input matches a legacy escape sequence for the given key name.
///
/// Returns true only when the byte sequence maps to the exact key identifier.
#[napi]
pub fn matches_legacy_sequence(data: String, key_name: String) -> bool {
	pi_term::keys::matches_legacy_sequence(data.as_bytes(), &key_name)
}

/// Match input data against a key identifier string.
///
/// Returns true when the bytes represent the specified key with modifiers.
#[napi]
pub fn matches_key(data: String, key_id: String, kitty_protocol_active: bool) -> bool {
	pi_term::keys::matches_key_inner(data.as_bytes(), &key_id, kitty_protocol_active)
}

/// Parse a Kitty keyboard protocol sequence.
///
/// Returns a structured parse result when the input is a valid Kitty sequence.
#[napi]
pub fn parse_kitty_sequence(data: String) -> Option<ParsedKittyResult> {
	pi_term::keys::parse_kitty_sequence_bytes(data.as_bytes()).map(|p| ParsedKittyResult {
		codepoint:       p.codepoint,
		shifted_key:     p.shifted_key,
		base_layout_key: p.base_layout_key,
		modifier:        p.modifier,
		event_type:      optional_kitty_event_type(p.event_type),
	})
}
