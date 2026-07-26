//! Raw tool description text, copied verbatim from
//! `packages/coding-agent/src/prompts/tools/*.md` and embedded via
//! `include_str!`.
//!
//! No template rendering happens here: files that carry Handlebars-style
//! placeholders (`{{#if ...}}` — currently [`READ`] and [`BASH`]) keep them
//! literally. Rendering is a WP-1.4 concern; a `Tool::description` returns one
//! of these strings as-is.

/// `read` tool description. Contains Handlebars placeholders
/// (`{{#if IS_HL_MODE}}`, `{{#if INSPECT_IMAGE_ENABLED}}`).
pub const READ: &str = include_str!("../prompts/read.md");
/// `write` tool description.
pub const WRITE: &str = include_str!("../prompts/write.md");
/// `replace` (edit) tool description.
pub const REPLACE: &str = include_str!("../prompts/replace.md");
/// `bash` tool description. Contains Handlebars placeholders
/// (`{{#if hasLaunch}}`, `{{#if hasEval}}`, `{{#if asyncEnabled}}`, etc.).
pub const BASH: &str = include_str!("../prompts/bash.md");
/// `grep` tool description.
pub const GREP: &str = include_str!("../prompts/grep.md");
/// `glob` tool description.
pub const GLOB: &str = include_str!("../prompts/glob.md");

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn descriptions_are_non_empty() {
		for text in [READ, WRITE, REPLACE, BASH, GREP, GLOB] {
			assert!(!text.trim().is_empty());
		}
	}
}
