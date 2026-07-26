//! Forgiving JSON recovery for streaming tool-call arguments.
//!
//! Line-for-line port of the strict-mode path of `parseJsonWithRepair`
//! (`packages/utils/src/json-parse.ts:553` → the `RelaxedJson` class,
//! constructed with `partial = false`). The provider tries a strict
//! `serde_json` parse first (fast path, exact JSON semantics); on failure it
//! runs the relaxed recursive-descent parser, which accepts and normalizes the
//! malformations LLM tool-call bodies leak in practice:
//!
//! - single-quoted strings and unquoted object keys (JSON5);
//! - trailing / stray / doubled commas, and `//` + `/* */` comments;
//! - Python literals `True` / `False` / `None`;
//! - raw control characters and invalid `\x` escapes inside strings (kept
//!   literally);
//! - unescaped quotes inside strings — a quote only closes a string when
//!   followed by a value terminator, recovering apostrophes such as `'it's'`;
//! - unquoted string values in value position (`{"paths": packages/foo/*}`).
//!
//! The streaming *partial* mode (`parseStreamingJson`, which auto-closes a
//! truncated buffer) is deliberately NOT ported: the builder keeps `arguments`
//! at `{}` until the block closes and never runs a mid-stream throttled parse
//! (contract appendix B, F2), so only the final strict recovery is needed. A
//! truncated buffer therefore fails here and lands in the `{__parseError,
//! __rawJson}` fallback — exactly as the TS strict path throws.

use serde_json::{Map, Number, Value};

/// Bareword literals: standard JSON plus Python `True`/`False`/`None`
/// (`json-parse.ts:43`).
const KEYWORDS: &[(&str, KeywordValue)] = &[
	("true", KeywordValue::True),
	("false", KeywordValue::False),
	("null", KeywordValue::Null),
	("True", KeywordValue::True),
	("False", KeywordValue::False),
	("None", KeywordValue::Null),
];

#[derive(Clone, Copy)]
enum KeywordValue {
	True,
	False,
	Null,
}

impl KeywordValue {
	const fn value(self) -> Value {
		match self {
			Self::True => Value::Bool(true),
			Self::False => Value::Bool(false),
			Self::Null => Value::Null,
		}
	}
}

/// JS-only atoms never recovered as bareword strings — a tool must not execute
/// with a non-finite or undefined argument masquerading as a string
/// (`json-parse.ts:56`).
const NON_RECOVERABLE_BAREWORDS: &[&str] =
	&["NaN", "Infinity", "-Infinity", "+Infinity", "undefined"];

/// Final-parse a JSON value, repairing the common LLM malformations. Tries
/// strict `serde_json` first (exact JSON semantics), then the relaxed parser.
///
/// # Errors
///
/// Returns the relaxed parser's diagnostic string when the input is
/// unrepairable, truncated, or carries trailing garbage — so the caller can
/// fall back to the `{__parseError, __rawJson}` shape instead of executing a
/// half-formed tool call (`parseJsonWithRepair`, `json-parse.ts:553`).
pub fn parse_json_with_repair(json: &str) -> Result<Value, String> {
	serde_json::from_str::<Value>(json).map_or_else(|_| RelaxedJson::new(json).parse(), Ok)
}

const fn is_whitespace(ch: char) -> bool {
	matches!(ch, ' ' | '\t' | '\n' | '\r')
}

const fn is_ident_char(ch: char) -> bool {
	ch.is_ascii_alphanumeric() || ch == '_' || ch == '$'
}

/// Recursive-descent parser for a forgiving superset of JSON, strict mode
/// (`RelaxedJson` with `partial = false`). Operates over Unicode scalar values
/// so multibyte tool arguments slice on character boundaries (the TS parser
/// indexes UTF-16 code units; all structural comparisons are ASCII, so the two
/// agree on well-formed content and never split a codepoint).
struct RelaxedJson {
	chars: Vec<char>,
	n:     usize,
	i:     usize,
}

impl RelaxedJson {
	fn new(source: &str) -> Self {
		let chars: Vec<char> = source.chars().collect();
		let n = chars.len();
		Self { chars, n, i: 0 }
	}

	fn parse(&mut self) -> Result<Value, String> {
		self.skip_ws();
		if self.i >= self.n {
			return Err("Unexpected end of JSON input".to_string());
		}
		let value = self.value(false)?;
		self.skip_ws();
		if self.i < self.n {
			return Err(format!("Unexpected trailing characters at position {}", self.i));
		}
		Ok(value)
	}

	/// Skip whitespace plus `//` line and `/* */` block comments
	/// (`json-parse.ts:209`).
	fn skip_ws(&mut self) {
		loop {
			while self.i < self.n && is_whitespace(self.chars[self.i]) {
				self.i += 1;
			}
			if self.i + 1 < self.n && self.chars[self.i] == '/' {
				let next = self.chars[self.i + 1];
				if next == '/' {
					self.i += 2;
					while self.i < self.n && self.chars[self.i] != '\n' {
						self.i += 1;
					}
					continue;
				}
				if next == '*' {
					self.i += 2;
					while self.i + 1 < self.n
						&& !(self.chars[self.i] == '*' && self.chars[self.i + 1] == '/')
					{
						self.i += 1;
					}
					self.i = (self.i + 2).min(self.n);
					continue;
				}
			}
			break;
		}
	}

	fn value(&mut self, allow_bareword: bool) -> Result<Value, String> {
		let c = self.chars[self.i];
		match c {
			'{' => self.object(),
			'[' => self.array(),
			'"' | '\'' => self.string(c).map(Value::String),
			_ if c == '-' || c == '+' || c == '.' || c.is_ascii_digit() => self.number(),
			_ => self.keyword(allow_bareword),
		}
	}

	fn object(&mut self) -> Result<Value, String> {
		self.i += 1; // consume {
		let mut out = Map::new();
		loop {
			self.skip_ws();
			if self.i >= self.n {
				return Err("Unterminated object".to_string());
			}
			let c = self.chars[self.i];
			if c == '}' {
				self.i += 1;
				return Ok(Value::Object(out));
			}
			if c == ',' {
				// Tolerate leading / doubled / trailing commas.
				self.i += 1;
				continue;
			}
			let key = self.key()?;
			self.skip_ws();
			if self.i < self.n && self.chars[self.i] == ':' {
				self.i += 1;
			} else {
				return Err("Expected ':' in object".to_string());
			}
			self.skip_ws();
			if self.i >= self.n {
				return Err("Expected value after ':'".to_string());
			}
			let value = self.value(true)?;
			out.insert(key, value);
			self.skip_ws();
			let d = if self.i < self.n {
				self.chars[self.i]
			} else {
				'\0'
			};
			if d == ',' {
				self.i += 1;
				continue;
			}
			if d == '}' {
				self.i += 1;
				return Ok(Value::Object(out));
			}
			return Err("Expected ',' or '}' in object".to_string());
		}
	}

	fn array(&mut self) -> Result<Value, String> {
		self.i += 1; // consume [
		let mut out = Vec::new();
		loop {
			self.skip_ws();
			if self.i >= self.n {
				return Err("Unterminated array".to_string());
			}
			let c = self.chars[self.i];
			if c == ']' {
				self.i += 1;
				return Ok(Value::Array(out));
			}
			if c == ',' {
				self.i += 1;
				continue;
			}
			let value = self.value(true)?;
			out.push(value);
			self.skip_ws();
			let d = if self.i < self.n {
				self.chars[self.i]
			} else {
				'\0'
			};
			if d == ',' {
				self.i += 1;
				continue;
			}
			if d == ']' {
				self.i += 1;
				return Ok(Value::Array(out));
			}
			return Err("Expected ',' or ']' in array".to_string());
		}
	}

	fn key(&mut self) -> Result<String, String> {
		let c = self.chars[self.i];
		if c == '"' || c == '\'' {
			return self.string(c);
		}
		// Unquoted identifier key: read until a structural delimiter / whitespace.
		let start = self.i;
		while self.i < self.n {
			let ch = self.chars[self.i];
			if ch == ':' || ch == ',' || ch == '}' || is_whitespace(ch) {
				break;
			}
			self.i += 1;
		}
		if self.i == start {
			return Err("Expected object key".to_string());
		}
		Ok(self.chars[start..self.i].iter().collect())
	}

	fn string(&mut self, quote: char) -> Result<String, String> {
		let mut i = self.i + 1; // skip opening quote
		let mut out = String::new();
		while i < self.n {
			let cc = self.chars[i];
			if cc != '\\' && cc != quote {
				out.push(cc);
				i += 1;
				continue;
			}
			if cc == quote {
				// Apostrophe / inner-quote recovery is safe for single quotes; for
				// double quotes close on the first unescaped quote like standard JSON
				// so malformed structure fails loudly instead of swallowing
				// commas/colons into one string. (`partial` is always false here.)
				let lenient = quote == '\'';
				if !lenient || self.closes_string(i + 1) {
					self.i = i + 1;
					return Ok(out);
				}
				// Unescaped inner quote (e.g. apostrophe in `'it's'`) — keep literal.
				out.push(cc);
				i += 1;
				continue;
			}
			// Backslash escape.
			i += 1;
			if i >= self.n {
				out.push('\\');
				break;
			}
			let esc = self.chars[i];
			match esc {
				'"' => out.push('"'),
				'\'' => out.push('\''),
				'\\' => out.push('\\'),
				'/' => out.push('/'),
				'b' => out.push('\u{08}'),
				'f' => out.push('\u{0c}'),
				'n' => out.push('\n'),
				'r' => out.push('\r'),
				't' => out.push('\t'),
				'u' => {
					if let Some(ch) = self.read_hex4(i + 1) {
						out.push(ch);
						i += 4;
					} else {
						out.push_str("\\u"); // invalid \u — keep literal
					}
				},
				other => {
					out.push('\\'); // invalid escape — keep backslash literal
					out.push(other);
				},
			}
			i += 1;
		}
		// End-of-input before the closing quote: strict mode rejects.
		Err("Unterminated string".to_string())
	}

	/// A `\uXXXX` escape: four hex digits mapping to a scalar value, else `None`
	/// so the caller keeps the `\u` literal (lone surrogates are dropped — a
	/// rare edge vs the TS `String.fromCharCode` code-unit path).
	fn read_hex4(&self, start: usize) -> Option<char> {
		if start + 4 > self.n {
			return None;
		}
		let hex: String = self.chars[start..start + 4].iter().collect();
		if hex.chars().all(|c| c.is_ascii_hexdigit()) {
			u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
		} else {
			None
		}
	}

	/// A quote closes a string only when the next non-space char ends a value
	/// (`json-parse.ts:446`).
	fn closes_string(&self, from: usize) -> bool {
		let mut k = from;
		while k < self.n && is_whitespace(self.chars[k]) {
			k += 1;
		}
		if k >= self.n {
			return true;
		}
		matches!(self.chars[k], ',' | '}' | ']' | ':')
	}

	fn number(&mut self) -> Result<Value, String> {
		let start = self.i;
		while self.i < self.n {
			let ch = self.chars[self.i];
			if ch.is_ascii_hexdigit() || matches!(ch, '-' | '+' | '.' | 'x' | 'X') {
				self.i += 1;
			} else {
				break;
			}
		}
		let token: String = self.chars[start..self.i].iter().collect();
		js_number(&token)
			.map(Value::Number)
			.ok_or_else(|| format!("Invalid number: {token}"))
	}

	fn keyword(&mut self, allow_bareword: bool) -> Result<Value, String> {
		for (word, value) in KEYWORDS {
			// Require a non-identifier boundary so `Truex` / `nullish` are not
			// misread as the keyword followed by junk.
			let len = word.chars().count();
			if self.starts_with_at(self.i, word) && !self.is_ident_char_at(self.i + len) {
				self.i += len;
				return Ok(value.value());
			}
		}
		if allow_bareword {
			return self.bareword().map(Value::String);
		}
		Err(format!("Unexpected token at position {}", self.i))
	}

	fn starts_with_at(&self, at: usize, word: &str) -> bool {
		for (idx, wc) in (at..).zip(word.chars()) {
			if idx >= self.n || self.chars[idx] != wc {
				return false;
			}
		}
		true
	}

	fn is_ident_char_at(&self, at: usize) -> bool {
		at < self.n && is_ident_char(self.chars[at])
	}

	/// Strict-mode recovery of an unquoted string value, e.g.
	/// `{"paths": packages/foo/*}`: consume until `,` / `}` / `]` / newline and
	/// trim trailing whitespace. Still throws — so a final parse never accepts a
	/// half-formed or non-finite argument — on truncation, an embedded
	/// `"`/`{`/`[` or key-like `:`, or a non-finite atom (`json-parse.ts:519`).
	fn bareword(&mut self) -> Result<String, String> {
		let start = self.i;
		let mut i = start;
		while i < self.n {
			let cc = self.chars[i];
			if matches!(cc, ',' | '}' | ']' | '\n' | '\r') {
				break;
			}
			let colon_field = cc == ':'
				&& self.chars.get(i + 1) != Some(&'/')
				&& self.chars.get(i + 1) != Some(&'\\');
			if cc == '"' || cc == '{' || cc == '[' || colon_field {
				return Err(format!("Unexpected token at position {start}"));
			}
			i += 1;
		}
		if i >= self.n {
			return Err(format!("Unexpected token at position {start}"));
		}
		let mut end = i;
		while end > start && is_whitespace(self.chars[end - 1]) {
			end -= 1;
		}
		let word: String = self.chars[start..end].iter().collect();
		if NON_RECOVERABLE_BAREWORDS.contains(&word.as_str()) {
			return Err(format!("Unexpected token at position {start}"));
		}
		self.i = i;
		Ok(word)
	}
}

/// Parse a recovered number token with JS `Number(token)` semantics: hex
/// integer literals (`0x1F` → 31), else integer or float. Non-finite / invalid
/// tokens return `None` so the caller rejects them (`json-parse.ts:455`).
fn js_number(token: &str) -> Option<Number> {
	if let Some(hex) = token
		.strip_prefix("0x")
		.or_else(|| token.strip_prefix("0X"))
	{
		return u64::from_str_radix(hex, 16).ok().map(Number::from);
	}
	if let Ok(int) = token.parse::<i64>() {
		return Some(Number::from(int));
	}
	if let Ok(uint) = token.parse::<u64>() {
		return Some(Number::from(uint));
	}
	let float = token.parse::<f64>().ok()?;
	if float.is_finite() {
		Number::from_f64(float)
	} else {
		None
	}
}

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::parse_json_with_repair;

	// ---- fast path: valid JSON needs no repair --------------------------------

	#[test]
	fn strict_json_passes_through_unchanged() {
		let value = parse_json_with_repair(r#"{"a":1,"b":["x",true,null],"c":{"d":2.5}}"#).unwrap();
		assert_eq!(value, json!({"a":1,"b":["x",true,null],"c":{"d":2.5}}));
	}

	#[test]
	fn integers_stay_integers_on_fast_path() {
		let value = parse_json_with_repair("42").unwrap();
		assert_eq!(value, json!(42));
		assert!(value.as_i64().is_some(), "must not float-widen a strict integer");
	}

	// ---- repair succeeds: TS malformation cases -------------------------------

	#[test]
	fn trailing_comma_is_repaired() {
		let value = parse_json_with_repair(r#"{"a":1,"b":2,}"#).unwrap();
		assert_eq!(value, json!({"a":1,"b":2}));
	}

	#[test]
	fn single_quotes_and_unquoted_keys_are_repaired() {
		let value = parse_json_with_repair("{path: 'src/main.rs', ok: True}").unwrap();
		assert_eq!(value, json!({"path":"src/main.rs","ok":true}));
	}

	#[test]
	fn python_literals_are_repaired() {
		let value = parse_json_with_repair(r#"{"a":True,"b":False,"c":None}"#).unwrap();
		assert_eq!(value, json!({"a":true,"b":false,"c":null}));
	}

	#[test]
	fn apostrophe_in_single_quoted_string_recovers() {
		let value = parse_json_with_repair("{'msg': 'it's fine'}").unwrap();
		assert_eq!(value, json!({"msg":"it's fine"}));
	}

	#[test]
	fn comments_are_stripped() {
		let value =
			parse_json_with_repair("{\n// leading\n\"a\": 1 /* inline */, \"b\": 2\n}").unwrap();
		assert_eq!(value, json!({"a":1,"b":2}));
	}

	#[test]
	fn unquoted_bareword_value_recovers_as_string() {
		let value = parse_json_with_repair("{\"paths\": packages/foo/*\n}").unwrap();
		assert_eq!(value, json!({"paths":"packages/foo/*"}));
	}

	#[test]
	fn raw_control_char_inside_string_is_kept_literal() {
		// A raw newline inside a double-quoted string is invalid strict JSON but
		// the relaxed parser keeps it (`partial`-independent string scan).
		let value = parse_json_with_repair("{\"a\": \"line1\nline2\"}").unwrap();
		assert_eq!(value, json!({"a":"line1\nline2"}));
	}

	// ---- repair fails: truncation / non-finite → Err (→ fallback shape) -------

	#[test]
	fn truncated_object_is_unrepairable() {
		// Strict mode does not auto-close a truncated buffer (partial mode is not
		// ported); the caller lands in the __parseError fallback.
		assert!(parse_json_with_repair(r#"{"a": "untermin"#).is_err());
	}

	#[test]
	fn non_finite_bareword_is_rejected() {
		assert!(parse_json_with_repair(r#"{"n": NaN}"#).is_err());
		assert!(parse_json_with_repair(r#"{"n": Infinity}"#).is_err());
	}

	#[test]
	fn trailing_garbage_after_value_is_rejected() {
		assert!(parse_json_with_repair(r#"{"a":1} extra"#).is_err());
	}

	#[test]
	fn missing_comma_between_fields_is_rejected() {
		// A missed comma would otherwise silently swallow the next field via the
		// unquoted-value recovery; the bareword `:` guard rejects it instead.
		assert!(parse_json_with_repair(r#"{"a": foo "b": 1}"#).is_err());
	}
}
