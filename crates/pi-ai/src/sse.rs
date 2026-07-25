//! Incremental SSE frame parser.
//!
//! Byte-chunk in, complete frames out — chunks may split lines, UTF-8
//! sequences, or frames arbitrarily. Follows the same subset of the SSE spec
//! the TS `readSseEvents` helper relies on: `event:` / `data:` fields (multiple
//! `data:` lines join with `\n`), frames end on a blank line, `\r\n` accepted,
//! comment (`:`) and unknown fields ignored. A trailing frame not terminated by
//! a blank line is discarded, matching eager-frame semantics.

use crate::wire::{ErrorEnvelope, RawMessageStreamEvent};

/// Parse a data frame into a raw stream event. Ping frames, unknown event
/// types, and malformed payloads all yield `None` (skip — TS parity,
/// anthropic.ts:1420-1449).
#[must_use]
pub fn parse_message_event(frame: &SseFrame) -> Option<RawMessageStreamEvent> {
	const MESSAGE_EVENTS: [&str; 6] = [
		"message_start",
		"message_delta",
		"message_stop",
		"content_block_start",
		"content_block_delta",
		"content_block_stop",
	];
	let value: serde_json::Value = serde_json::from_str(&frame.data).ok()?;
	let event_type = value.get("type")?.as_str()?;
	if !MESSAGE_EVENTS.contains(&event_type) {
		return None;
	}
	serde_json::from_value(value).ok()
}

/// In-stream `error` frame → message (`createAnthropicSseStreamError`,
/// anthropic.ts:1385-1400).
#[must_use]
pub fn stream_error_message(data: &str) -> String {
	if let Ok(envelope) = serde_json::from_str::<ErrorEnvelope>(data)
		&& !envelope.error.message.is_empty()
	{
		return if envelope.error.error_type.is_empty() {
			format!("Anthropic stream error: {}", envelope.error.message)
		} else {
			format!(
				"Anthropic stream error ({}): {}",
				envelope.error.error_type, envelope.error.message
			)
		};
	}
	data.to_string()
}

/// One complete SSE frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseFrame {
	/// `event:` field, when present (e.g. `message_start`, `ping`, `error`).
	pub event: Option<String>,
	/// Joined `data:` payload.
	pub data:  String,
}

#[derive(Debug, Default)]
pub struct SseParser {
	buffer:  Vec<u8>,
	event:   Option<String>,
	data:    Vec<String>,
	started: bool,
}

impl SseParser {
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Feed a byte chunk; returns every frame completed by it.
	pub fn push(&mut self, chunk: &[u8]) -> Vec<SseFrame> {
		self.buffer.extend_from_slice(chunk);
		let mut frames = Vec::new();
		while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
			let raw: Vec<u8> = self.buffer.drain(..=newline).collect();
			let line = String::from_utf8_lossy(&raw);
			let line = line.trim_end_matches(['\n', '\r']);
			if line.is_empty() {
				if let Some(frame) = self.take_frame() {
					frames.push(frame);
				}
				continue;
			}
			self.consume_field(line);
		}
		frames
	}

	fn consume_field(&mut self, line: &str) {
		let (field, value) = line.split_once(':').unwrap_or((line, ""));
		let value = value.strip_prefix(' ').unwrap_or(value);
		match field {
			"event" => {
				self.started = true;
				self.event = Some(value.to_string());
			},
			"data" => {
				self.started = true;
				self.data.push(value.to_string());
			},
			// Comments (empty field name) and unknown fields (id, retry, …).
			_ => {},
		}
	}

	fn take_frame(&mut self) -> Option<SseFrame> {
		if !self.started {
			return None;
		}
		let frame = SseFrame { event: self.event.take(), data: self.data.join("\n") };
		self.data.clear();
		self.started = false;
		Some(frame)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_frames_split_across_chunks() {
		let mut parser = SseParser::new();
		assert!(parser.push(b"event: message_st").is_empty());
		assert!(parser.push(b"art\ndata: {\"a\"").is_empty());
		let frames = parser.push(b":1}\n\nevent: ping\ndata: {}\n\n");
		assert_eq!(frames.len(), 2);
		assert_eq!(frames[0].event.as_deref(), Some("message_start"));
		assert_eq!(frames[0].data, "{\"a\":1}");
		assert_eq!(frames[1].event.as_deref(), Some("ping"));
	}

	#[test]
	fn joins_multiple_data_lines_and_handles_crlf() {
		let mut parser = SseParser::new();
		let frames = parser.push(b"data: line1\r\ndata: line2\r\n\r\n");
		assert_eq!(frames.len(), 1);
		assert_eq!(frames[0].event, None);
		assert_eq!(frames[0].data, "line1\nline2");
	}

	#[test]
	fn ignores_comments_and_unknown_fields() {
		let mut parser = SseParser::new();
		let frames = parser.push(b": keepalive\nid: 7\nretry: 100\n\n");
		assert!(frames.is_empty(), "comment/unknown-only block must not yield a frame");
		let frames = parser.push(b": note\ndata: x\n\n");
		assert_eq!(frames.len(), 1);
		assert_eq!(frames[0].data, "x");
	}

	#[test]
	fn utf8_split_across_chunks_survives() {
		let mut parser = SseParser::new();
		let bytes = "data: héllo\n\n".as_bytes();
		let (a, b) = bytes.split_at(8); // splits inside the two-byte é
		assert!(parser.push(a).is_empty());
		let frames = parser.push(b);
		assert_eq!(frames[0].data, "héllo");
	}

	#[test]
	fn trailing_unterminated_frame_is_discarded() {
		let mut parser = SseParser::new();
		let frames = parser.push(b"event: message_stop\ndata: {}");
		assert!(frames.is_empty());
		drop(parser);
	}
}
