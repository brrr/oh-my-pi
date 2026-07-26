//! Path resolution/formatting helpers ported from
//! `packages/coding-agent/src/tools/path-utils.ts` (the minimal face the six
//! tools need).
//!
//! Deferred vs. the TS source: internal-URL schemes, Windows drive aliases,
//! shell-escaped/NFD/curly-quote read-path variants, `@`/`:` stray-prefix
//! stripping, unicode-space normalization, and delimited multi-path expansion.

use std::path::{Component, Path, PathBuf};

/// `stripOuterDoubleQuotes` + `trim` (`normalizePathLikeInput`,
/// path-utils.ts:559).
#[must_use]
pub fn normalize_path_like_input(input: &str) -> String {
	let trimmed = input.trim();
	if trimmed.len() > 1 && trimmed.starts_with('"') && trimmed.ends_with('"') {
		trimmed[1..trimmed.len() - 1].to_owned()
	} else {
		trimmed.to_owned()
	}
}

/// `expandTilde` subset of `expandPath` (path-utils.ts:150): `~` and `~/...`.
#[must_use]
pub fn expand_path(file_path: &str) -> String {
	if file_path == "~" {
		return std::env::var("HOME").unwrap_or_else(|_| "~".to_owned());
	}
	if let Some(rest) = file_path.strip_prefix("~/")
		&& let Ok(home) = std::env::var("HOME")
	{
		return format!("{}/{rest}", home.trim_end_matches('/'));
	}
	file_path.to_owned()
}

/// Lexical normalization mirroring Node's `path.resolve` on an absolute input:
/// collapse `.`/`..` without touching the filesystem.
fn lexical_normalize(path: &Path) -> PathBuf {
	let mut out = PathBuf::new();
	for component in path.components() {
		match component {
			Component::CurDir => {},
			Component::ParentDir => {
				if !out.pop() {
					out.push(Component::RootDir);
				}
			},
			other => out.push(other),
		}
	}
	if out.as_os_str().is_empty() {
		return PathBuf::from("/");
	}
	out
}

/// `resolveToCwd` (path-utils.ts:506): `~` expansion, bare-root alias to cwd,
/// absolute passthrough, relative joined onto `cwd`.
#[must_use]
pub fn resolve_to_cwd(file_path: &str, cwd: &Path) -> PathBuf {
	let expanded = expand_path(file_path);
	if !expanded.is_empty() && expanded.chars().all(|c| c == '/') {
		return lexical_normalize(cwd);
	}
	let candidate = Path::new(&expanded);
	if candidate.is_absolute() {
		lexical_normalize(candidate)
	} else {
		lexical_normalize(&cwd.join(candidate))
	}
}

/// Node `path.relative(from, to)` on already-normalized absolute paths.
fn relative_path(from: &Path, to: &Path) -> PathBuf {
	let from: Vec<Component<'_>> = from.components().collect();
	let to: Vec<Component<'_>> = to.components().collect();
	let common = from
		.iter()
		.zip(to.iter())
		.take_while(|(a, b)| a == b)
		.count();
	let mut out = PathBuf::new();
	for _ in common..from.len() {
		out.push("..");
	}
	for component in &to[common..] {
		out.push(component);
	}
	out
}

/// `formatPathRelativeToCwd` (path-utils.ts:522).
#[must_use]
pub fn format_path_relative_to_cwd(file_path: &str, cwd: &Path, trailing_slash: bool) -> String {
	let resolved_cwd = lexical_normalize(cwd);
	let expanded = expand_path(file_path);
	let candidate = Path::new(&expanded);
	let resolved = if candidate.is_absolute() {
		lexical_normalize(candidate)
	} else {
		lexical_normalize(&cwd.join(candidate))
	};
	let relative = relative_path(&resolved_cwd, &resolved);
	let relative_str = relative.to_string_lossy().replace('\\', "/");
	let within =
		relative_str.is_empty() || (!relative_str.starts_with("..") && !relative.is_absolute());
	let mut display = if within {
		if relative_str.is_empty() {
			".".to_owned()
		} else {
			relative_str
		}
	} else {
		resolved.to_string_lossy().replace('\\', "/")
	};
	if trailing_slash && display != "." && !display.ends_with('/') {
		display.push('/');
	}
	display
}

/// `hasGlobPathChars` (path-utils.ts:598).
#[must_use]
pub fn has_glob_path_chars(file_path: &str) -> bool {
	file_path.contains(['*', '?', '[', '{'])
}

/// `parseFindPattern` (path-utils.ts:842): split a find pattern into base
/// directory + glob, auto-prefixing `**/` for leading-glob patterns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedFindPattern {
	pub base_path:    String,
	pub glob_pattern: String,
	pub has_glob:     bool,
}

#[must_use]
pub fn parse_find_pattern(pattern: &str) -> ParsedFindPattern {
	let normalized = pattern.replace('\\', "/");
	let segments: Vec<&str> = normalized.split('/').collect();
	let first_glob = segments.iter().position(|seg| has_glob_path_chars(seg));

	match first_glob {
		None => ParsedFindPattern {
			base_path:    normalized,
			glob_pattern: "**/*".to_owned(),
			has_glob:     false,
		},
		Some(0) => {
			let needs_recursive = !normalized.starts_with("**/");
			ParsedFindPattern {
				base_path:    ".".to_owned(),
				glob_pattern: if needs_recursive {
					format!("**/{normalized}")
				} else {
					normalized
				},
				has_glob:     true,
			}
		},
		Some(idx) => ParsedFindPattern {
			base_path:    segments[..idx].join("/"),
			glob_pattern: segments[idx..].join("/"),
			has_glob:     true,
		},
	}
}

/// `parseSearchPath` (path-utils.ts, grep flavor): like [`parse_find_pattern`]
/// but the glob keeps its literal shape (no `**/` prefixing) and is optional.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSearchPath {
	pub base_path: String,
	pub glob:      Option<String>,
}

#[must_use]
pub fn parse_search_path(file_path: &str) -> ParsedSearchPath {
	let normalized = file_path.replace('\\', "/");
	let segments: Vec<&str> = normalized.split('/').collect();
	let first_glob = segments.iter().position(|seg| has_glob_path_chars(seg));

	match first_glob {
		None => ParsedSearchPath { base_path: normalized, glob: None },
		Some(0) => ParsedSearchPath { base_path: ".".to_owned(), glob: Some(normalized) },
		Some(idx) => ParsedSearchPath {
			base_path: segments[..idx].join("/"),
			glob:      Some(segments[idx..].join("/")),
		},
	}
}

/// `shortenPath` subset: collapse a home-dir prefix to `~`.
#[must_use]
pub fn shorten_path(path: &str) -> String {
	if let Ok(home) = std::env::var("HOME") {
		let home = home.trim_end_matches('/');
		if path == home {
			return "~".to_owned();
		}
		if let Some(rest) = path.strip_prefix(home)
			&& rest.starts_with('/')
		{
			return format!("~{rest}");
		}
	}
	path.to_owned()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn resolve_and_format_round_trip() {
		let cwd = Path::new("/tmp/ws");
		assert_eq!(resolve_to_cwd("a/b.txt", cwd), PathBuf::from("/tmp/ws/a/b.txt"));
		assert_eq!(resolve_to_cwd("/", cwd), PathBuf::from("/tmp/ws"));
		assert_eq!(resolve_to_cwd("/abs/x", cwd), PathBuf::from("/abs/x"));
		assert_eq!(format_path_relative_to_cwd("/tmp/ws/a/b.txt", cwd, false), "a/b.txt");
		assert_eq!(format_path_relative_to_cwd("/tmp/ws", cwd, false), ".");
		assert_eq!(format_path_relative_to_cwd("/other/x", cwd, false), "/other/x");
		assert_eq!(format_path_relative_to_cwd("/tmp/ws/d", cwd, true), "d/");
	}

	#[test]
	fn find_pattern_split_matches_ts_examples() {
		assert_eq!(parse_find_pattern("src/app/**/*.tsx"), ParsedFindPattern {
			base_path:    "src/app".to_owned(),
			glob_pattern: "**/*.tsx".to_owned(),
			has_glob:     true,
		});
		assert_eq!(parse_find_pattern("*.ts"), ParsedFindPattern {
			base_path:    ".".to_owned(),
			glob_pattern: "**/*.ts".to_owned(),
			has_glob:     true,
		});
		assert_eq!(parse_find_pattern("src/app"), ParsedFindPattern {
			base_path:    "src/app".to_owned(),
			glob_pattern: "**/*".to_owned(),
			has_glob:     false,
		});
	}

	#[test]
	fn search_path_split() {
		assert_eq!(parse_search_path("src/*.rs"), ParsedSearchPath {
			base_path: "src".to_owned(),
			glob:      Some("*.rs".to_owned()),
		});
		assert_eq!(parse_search_path("src"), ParsedSearchPath {
			base_path: "src".to_owned(),
			glob:      None,
		});
		assert_eq!(parse_search_path("*.md"), ParsedSearchPath {
			base_path: ".".to_owned(),
			glob:      Some("*.md".to_owned()),
		});
	}
}
