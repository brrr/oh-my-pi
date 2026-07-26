//! N-API shell over the pure `pi_term::snapcompact` engine.
//!
//! Rasterizes pre-normalized conversation text onto a bitmap and encodes it as
//! PNG. All rendering/encoding logic lives in [`pi_term::snapcompact`]; this
//! module owns only the N-API surface: the `#[napi(object)]`
//! [`SnapcompactRenderOptions`] DTO, and the base64 + [`Latin1String`] boundary
//! that hands the PNG back to JS as a one-byte (Latin-1) string with no
//! `Uint8Array` hop.
//!
//! Text normalization, frame chunking, provider shape selection, and archive
//! management live in `packages/snapcompact/src/snapcompact.ts`.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::task;

/// Shape options for one snapcompact frame.
#[napi(object)]
#[derive(Default)]
pub struct SnapcompactRenderOptions {
	/// Frame width in pixels; also bounds the grid rows
	/// (`floor(size/cellHeight/lineRepeat)`). Output height hugs the rows the
	/// text actually uses instead of padding to a square.
	pub size:        u32,
	/// Bundled font: `"5x8"`, `"6x12"`, `"8x13"` (X.org BDF), `"8x8"`
	/// (unscii-8), or `"silver"` (embedded TrueType). Default `"5x8"`.
	pub font:        Option<String>,
	/// Target cell advance in pixels. Differing from the font's natural cell
	/// triggers the Lanczos stretch path. Default: font natural width.
	pub cell_width:  Option<u32>,
	/// Target cell pitch in pixels. Default: font natural height.
	pub cell_height: Option<u32>,
	/// Ink variant: `"sent"` (six-hue sentence cycling) or `"bw"` (black).
	/// Default `"sent"`.
	pub variant:     Option<String>,
	/// Print each text line this many times; copies after the first sit on a
	/// pale highlight band. Default 1.
	pub line_repeat: Option<u32>,
	/// Stretch behavior. Unset: auto — Lanczos-stretch whenever the target
	/// cell differs from the font's natural cell. `false`: never stretch —
	/// render indexed with glyphs at natural size on the requested cell box
	/// (e.g. 8x13 glyphs on an 8x16 pitch, the "8on16" shapes). `true`: force
	/// the stretch path (identical to auto; natural cells render indexed).
	pub stretch:     Option<bool>,
	/// Layout columns: `1` (default) row-major grid; `2` two newspaper "doc"
	/// columns of pre-wrapped newline-separated lines.
	pub columns:     Option<u32>,
}

impl From<SnapcompactRenderOptions> for pi_term::snapcompact::RenderOptions {
	fn from(options: SnapcompactRenderOptions) -> Self {
		Self {
			size:        options.size,
			font:        options.font,
			cell_width:  options.cell_width,
			cell_height: options.cell_height,
			variant:     options.variant,
			line_repeat: options.line_repeat,
			stretch:     options.stretch,
			columns:     options.columns,
		}
	}
}

/// Return the subset of `chars` that the named snapcompact font can render.
///
/// The TypeScript normalizer uses this to keep Unicode text intact only when
/// the selected native font has a glyph for it; renderer control codes are
/// considered renderable because they are interpreted outside font lookup.
#[napi]
pub fn snapcompact_supported_chars(font: String, chars: String) -> Result<String> {
	pi_term::snapcompact::supported_chars(&font, &chars)
		.map_err(|err| Error::from_reason(err.to_string()))
}

/// Render one snapcompact frame on a libuv worker: print pre-normalized text
/// onto a `size`-wide bitmap and encode it as PNG.
///
/// The bitmap height hugs the rows the text actually occupies
/// (`usedRows * lineRepeat * cellHeight`), so a partially filled frame never
/// pays for blank padding rows. The glyph grid holds `floor(size/cellWidth) *
/// floor(size/cellHeight/lineRepeat)` characters; input beyond that is ignored.
/// Native-cell bitmap-font shapes encode as indexed PNG; stretched bitmap-font
/// shapes (target cell != font cell) encode as RGB. TrueType shapes encode RGB
/// directly from grayscale coverage.
/// `stretch: false` pins bitmap fonts to the indexed path, printing
/// natural-size glyphs on the requested cell box; `columns: 2` flows
/// pre-wrapped newline-separated lines down two newspaper columns.
/// `U+000E`/`U+000F` in `text` toggle dim-gray ink spans without occupying a
/// cell.
/// Returns a promise for the PNG encoded as base64, created as a one-byte
/// (Latin-1) JS string straight from native code — no `Uint8Array` hop or
/// JS-side re-encode.
#[napi]
pub fn render_snapcompact_png(
	text: String,
	options: SnapcompactRenderOptions,
) -> task::Promise<Latin1String> {
	task::blocking("render_snapcompact_png", (), move |_| {
		let png = pi_term::snapcompact::render_snapcompact_png_sync(&text, options.into())
			.map_err(|err| Error::from_reason(err.to_string()))?;
		Ok(STANDARD.encode(png).into())
	})
}
