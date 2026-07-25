//! Minimal auth/config surface.
//!
//! Mirrors `packages/ai/src/utils/anthropic-auth.ts` (`AnthropicAuthConfig`,
//! env resolution, OAuth detection) and `normalizeAnthropicBaseUrl`
//! (anthropic.ts:107). Credential storage is out of scope for WP-1.1a; besides
//! an explicit key / `ANTHROPIC_API_KEY`, the resolver can read an entry from
//! the opencode auth store (`~/.local/share/opencode/auth.json`) for dev use.
//!
//! Deviations from TS, by design: no FOUNDRY branch (headless has no Foundry
//! mode) and no `?beta=true` URL suffix by default — compatible endpoints
//! (`DeepSeek`) have undefined behavior on unknown query params; official-API
//! callers can opt back in via [`AnthropicAuthConfig::messages_url`].

use crate::AiError;

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// `normalizeAnthropicBaseUrl` (anthropic.ts:107): trim, strip trailing
/// slashes, strip a trailing `/v1` path segment. Deeper path prefixes (e.g.
/// `DeepSeek`'s `/anthropic`) survive.
#[must_use]
pub fn normalize_base_url(base_url: &str) -> String {
	let trimmed = base_url.trim().trim_end_matches('/');
	trimmed.strip_suffix("/v1").unwrap_or(trimmed).to_string()
}

/// `isOAuthToken` (anthropic-auth.ts:45).
#[must_use]
pub fn is_oauth_token(api_key: &str) -> bool {
	api_key.contains("sk-ant-oat")
}

/// `AnthropicAuthConfig` (anthropic-auth.ts:20).
#[derive(Debug, Clone)]
pub struct AnthropicAuthConfig {
	pub api_key:  String,
	pub base_url: String,
	pub is_oauth: bool,
}

impl AnthropicAuthConfig {
	/// `buildAnthropicAuthConfig` (anthropic-auth.ts:60): explicit `base_url`
	/// beats `ANTHROPIC_BASE_URL` beats the official default; `is_oauth` is
	/// derived from the token prefix.
	#[must_use]
	pub fn new(api_key: String, base_url: Option<&str>) -> Self {
		let resolved = base_url
			.map(ToOwned::to_owned)
			.or_else(|| std::env::var("ANTHROPIC_BASE_URL").ok())
			.filter(|url| !url.trim().is_empty())
			.map_or_else(|| DEFAULT_BASE_URL.to_string(), |url| normalize_base_url(&url));
		let is_oauth = is_oauth_token(&api_key);
		Self { api_key, base_url: resolved, is_oauth }
	}

	/// `{base}/v1/messages`, optionally with the official-endpoint
	/// `?beta=true` suffix (`buildAnthropicUrl`, anthropic-auth.ts:89).
	#[must_use]
	pub fn messages_url(&self, beta_query: bool) -> String {
		let suffix = if beta_query { "?beta=true" } else { "" };
		format!("{}/v1/messages{suffix}", self.base_url)
	}
}

/// Resolve an API key: explicit value → `ANTHROPIC_API_KEY` env → the named
/// entry in the opencode auth store (`.{entry}.key`), when given.
///
/// # Errors
///
/// [`AiError::Auth`] when every source comes up empty.
pub fn resolve_api_key(
	explicit: Option<&str>,
	opencode_entry: Option<&str>,
) -> Result<String, AiError> {
	if let Some(key) = explicit.map(str::trim).filter(|key| !key.is_empty()) {
		return Ok(key.to_string());
	}
	if let Ok(key) = std::env::var("ANTHROPIC_API_KEY")
		&& !key.trim().is_empty()
	{
		return Ok(key.trim().to_string());
	}
	if let Some(entry) = opencode_entry
		&& let Some(key) = opencode_auth_key(entry)
	{
		return Ok(key);
	}
	Err(AiError::Auth(format!(
		"no API key: pass one explicitly, set ANTHROPIC_API_KEY, or provide opencode auth entry \
		 {opencode_entry:?}"
	)))
}

/// Read `.{entry}.key` from the opencode auth store
/// (`$XDG_DATA_HOME|~/.local/share` + `/opencode/auth.json`).
fn opencode_auth_key(entry: &str) -> Option<String> {
	let data_dir = std::env::var("XDG_DATA_HOME")
		.ok()
		.filter(|dir| !dir.is_empty())
		.map_or_else(
			|| {
				let home = std::env::var("HOME").ok()?;
				Some(format!("{home}/.local/share"))
			},
			Some,
		)?;
	let raw = std::fs::read_to_string(format!("{data_dir}/opencode/auth.json")).ok()?;
	let store: serde_json::Value = serde_json::from_str(&raw).ok()?;
	store
		.get(entry)?
		.get("key")?
		.as_str()
		.map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn base_url_normalization() {
		assert_eq!(normalize_base_url("https://api.anthropic.com/"), "https://api.anthropic.com");
		assert_eq!(normalize_base_url("https://api.anthropic.com/v1"), "https://api.anthropic.com");
		assert_eq!(
			normalize_base_url("https://api.deepseek.com/anthropic"),
			"https://api.deepseek.com/anthropic"
		);
		assert_eq!(
			normalize_base_url("https://api.deepseek.com/anthropic/v1/"),
			"https://api.deepseek.com/anthropic"
		);
	}

	#[test]
	fn messages_url_shapes() {
		let auth =
			AnthropicAuthConfig::new("sk-test".into(), Some("https://api.deepseek.com/anthropic"));
		assert!(!auth.is_oauth);
		assert_eq!(auth.messages_url(false), "https://api.deepseek.com/anthropic/v1/messages");
		assert_eq!(
			auth.messages_url(true),
			"https://api.deepseek.com/anthropic/v1/messages?beta=true"
		);
	}

	#[test]
	fn oauth_detection() {
		assert!(is_oauth_token("sk-ant-oat01-abc"));
		assert!(!is_oauth_token("sk-ant-api03-abc"));
	}
}
