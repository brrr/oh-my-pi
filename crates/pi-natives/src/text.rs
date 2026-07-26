//! N-API shell over the pure `pi_term::text` engine.
//!
//! ANSI-aware text measurement and slicing, optimized for JS string interop
//! (UTF-16). All measurement/slicing logic lives in [`pi_term::text`]; this
//! module owns only the N-API surface: the `JsString`/[`Utf16String`] boundary,
//! the `#[napi]` DTOs ([`Ellipsis`] / [`SliceResult`] /
//! [`ExtractSegmentsResult`]), and the zero-copy fast path where
//! `truncateToWidth` returns the original `JsString` handle unchanged.

use napi::{JsString, bindgen_prelude::*};
use napi_derive::napi;

pub const DEFAULT_TAB_WIDTH: usize = pi_term::text::DEFAULT_TAB_WIDTH;

/// Build a [`Utf16String`] from engine output: strip trailing NUL code units
/// (so a JS string never carries a spurious trailing `\0`) and hand the `Vec`
/// straight to napi's zero-cost `From<Vec<u16>>`.
fn build_utf16_string(mut data: Vec<u16>) -> Utf16String {
	while data.last() == Some(&0) {
		data.pop();
	}
	Utf16String::from(data)
}

/// Ellipsis strategy for [`truncate_to_width`].
#[napi]
pub enum Ellipsis {
	/// Use a single Unicode ellipsis character ("…").
	Unicode = 0,
	/// Use three ASCII dots ("...").
	Ascii   = 1,
	/// Omit ellipsis entirely.
	Omit    = 2,
}

impl From<Ellipsis> for pi_term::text::Ellipsis {
	fn from(kind: Ellipsis) -> Self {
		match kind {
			Ellipsis::Unicode => Self::Unicode,
			Ellipsis::Ascii => Self::Ascii,
			Ellipsis::Omit => Self::Omit,
		}
	}
}

// ============================================================================
// Results
// ============================================================================

/// Visible slice of a line after ANSI-aware column selection
/// (`sliceWithWidth`).
#[napi(object)]
pub struct SliceResult {
	/// UTF-16 slice containing the selected text.
	pub text:  Utf16String,
	/// Visible width of the slice in terminal cells.
	pub width: u32,
}

/// Before/after UTF-16 segments around an overlay region, with measured widths.
#[napi(object)]
pub struct ExtractSegmentsResult {
	/// UTF-16 content before the overlay region.
	pub before:       Utf16String,
	/// Visible width of the `before` segment.
	pub before_width: u32,
	/// UTF-16 content after the overlay region.
	pub after:        Utf16String,
	/// Visible width of the `after` segment.
	pub after_width:  u32,
}

#[napi]
pub fn set_hangul_compat_jamo_width_override(value: u8) {
	pi_term::text::set_hangul_compat_jamo_width_override(value);
}

/// Wrap text to a visible width, preserving ANSI escape codes across line
/// breaks.
///
/// Returns UTF-16 lines with active SGR codes carried across line boundaries.
#[napi]
pub fn wrap_text_with_ansi(text: JsString, width: u32, tab_width: u32) -> Result<Vec<Utf16String>> {
	let text_u16 = text.into_utf16()?;
	let tab_width = pi_term::text::clamp_tab_width_for_ops(tab_width);
	let lines =
		pi_term::text::wrap_text_with_ansi_impl(text_u16.as_slice(), width as usize, tab_width);
	Ok(lines.into_iter().map(build_utf16_string).collect())
}

/// Truncate text to a visible width, preserving ANSI codes.
///
/// Pads with spaces when requested.
#[napi]
pub fn truncate_to_width(
	text: JsString<'_>,
	max_width: u32,
	ellipsis_kind: Option<Ellipsis>,
	pad: Option<bool>,
	tab_width: u32,
) -> Result<Either<JsString<'_>, Utf16String>> {
	let ellipsis_kind = ellipsis_kind.map_or(pi_term::text::Ellipsis::Unicode, Into::into);
	let pad = pad.unwrap_or(false);
	let tab_width = pi_term::text::clamp_tab_width_for_ops(tab_width);

	// Keep original handle so we can return it without allocating.
	let original = text;

	let text_u16 = text.into_utf16()?;

	match pi_term::text::truncate_to_width(
		text_u16.as_slice(),
		max_width as usize,
		ellipsis_kind,
		pad,
		tab_width,
	) {
		// `None` means "unchanged": return the original JsString, zero allocation.
		None => Ok(Either::A(original)),
		Some(out) => Ok(Either::B(build_utf16_string(out))),
	}
}

/// Slice a range of visible columns from a line.
///
/// Counts terminal cells, skipping ANSI escapes, and optionally enforces strict
/// width.
#[napi]
pub fn slice_with_width(
	line: JsString,
	start_col: u32,
	length: u32,
	strict: Option<bool>,
	tab_width: u32,
) -> Result<SliceResult> {
	let line_u16 = line.into_utf16()?;
	let line = line_u16.as_slice();
	let strict = strict.unwrap_or(false);

	if length == 0 {
		return Ok(SliceResult { text: build_utf16_string(vec![]), width: 0 });
	}

	let tab_width = pi_term::text::clamp_tab_width_for_ops(tab_width);
	let (out, w) = pi_term::text::slice_with_width_impl(
		line,
		start_col as usize,
		length as usize,
		strict,
		tab_width,
	);

	Ok(SliceResult { text: build_utf16_string(out), width: crate::utils::clamp_u32(w as u64) })
}

/// Extract the before/after slices around an overlay region.
///
/// Preserves ANSI state so the `after` segment renders correctly after
/// truncation.
#[napi]
pub fn extract_segments(
	line: JsString,
	before_end: u32,
	after_start: u32,
	after_len: u32,
	strict_after: bool,
	tab_width: u32,
) -> Result<ExtractSegmentsResult> {
	let line_u16 = line.into_utf16()?;
	let line = line_u16.as_slice();

	let tab_width = pi_term::text::clamp_tab_width_for_ops(tab_width);
	let (before, bw, after, aw) = pi_term::text::extract_segments_impl(
		line,
		before_end as usize,
		after_start as usize,
		after_len as usize,
		strict_after,
		tab_width,
	);

	Ok(ExtractSegmentsResult {
		before:       build_utf16_string(before),
		before_width: crate::utils::clamp_u32(bw as u64),
		after:        build_utf16_string(after),
		after_width:  crate::utils::clamp_u32(aw as u64),
	})
}

/// Calculate visible width of text, excluding ANSI escape sequences.
///
/// Tabs count as a fixed-width cell.
#[napi]
pub fn visible_width(text: JsString, tab_width: u32) -> Result<u32> {
	let text_u16 = text.into_utf16()?;
	let tab_width = pi_term::text::clamp_tab_width_for_ops(tab_width);
	Ok(crate::utils::clamp_u32(
		pi_term::text::visible_width_u16(text_u16.as_slice(), tab_width) as u64
	))
}
