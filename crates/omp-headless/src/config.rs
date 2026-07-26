//! 配置面（非 ACP 帧）：system prompt + LLM（DeepSeek）解析。
//!
//! - system prompt：`--system-prompt <file|text>` flag >
//!   `OMP_HEADLESS_SYSTEM_PROMPT` env > 内置最小 coding prompt（flag 优先）。
//! - LLM：DeepSeek 照 pi-ai `examples/stream.rs` 样板——opencode `auth.json` 读
//!   key（`OMP_HEADLESS_AUTH_ENTRY`，缺省 `deepseek`），`OMP_HEADLESS_MODEL`
//!   可覆盖模型（缺省 `deepseek-v4-flash`）。

use std::path::Path;

use anyhow::{Context, Result};
use pi_ai::{
	Client,
	auth::{AnthropicAuthConfig, resolve_api_key},
};

/// 缺省模型（`OMP_HEADLESS_MODEL` 覆盖）。
pub const DEFAULT_MODEL: &str = "deepseek-v4-flash";
/// 缺省 base url（DeepSeek anthropic 兼容端点，`OMP_HEADLESS_BASE_URL` 覆盖）。
pub const DEFAULT_BASE_URL: &str = "https://api.deepseek.com/anthropic";
/// 缺省 opencode auth 条目名（`OMP_HEADLESS_AUTH_ENTRY` 覆盖）。
pub const DEFAULT_AUTH_ENTRY: &str = "deepseek";
/// 单轮最大输出 token。
pub const DEFAULT_MAX_TOKENS: u64 = 8192;

/// 内置最小 coding system prompt。
///
/// **非** TS `coding-agent` 的 system prompt 移植——仅供 headless 冒烟自洽；
/// 正式 prompt 由宿主经 flag/env 注入。
pub const DEFAULT_SYSTEM_PROMPT: &str =
	"You are omp-headless, a headless coding agent. You have file and shell tools (read, write, \
	 edit, bash, grep, glob) scoped to the session working directory. Use them to complete the \
	 user's task, then give a short confirmation. Prefer concrete actions over asking questions.";

/// 解析后的 LLM 连接配置。
pub struct LlmConfig {
	/// 模型 id。
	pub model:      String,
	/// 上游 base url。
	pub base_url:   String,
	/// provider id（记录在消息上；亦即 opencode auth 条目名）。
	pub auth_entry: String,
	/// 解析出的 api key。
	pub api_key:    String,
	/// 单轮最大输出 token。
	pub max_tokens: u64,
}

impl LlmConfig {
	/// 从 env + opencode auth 解析（key 缺失即 fail-fast）。
	///
	/// # Errors
	///
	/// api key 三源（显式/`ANTHROPIC_API_KEY`/opencode 条目）皆空时返错。
	pub fn resolve() -> Result<Self> {
		let model = env_or("OMP_HEADLESS_MODEL", DEFAULT_MODEL);
		let base_url = env_or("OMP_HEADLESS_BASE_URL", DEFAULT_BASE_URL);
		let auth_entry = env_or("OMP_HEADLESS_AUTH_ENTRY", DEFAULT_AUTH_ENTRY);
		let api_key = resolve_api_key(None, Some(&auth_entry))
			.with_context(|| format!("解析 api key 失败（auth entry {auth_entry}）"))?;
		Ok(Self { model, base_url, auth_entry, api_key, max_tokens: DEFAULT_MAX_TOKENS })
	}

	/// 构造一个 pi-ai [`Client`]（每轮 prompt 用；`Client` 内部 clone 开销小）。
	#[must_use]
	pub fn client(&self) -> Client {
		Client::new(
			AnthropicAuthConfig::new(self.api_key.clone(), Some(&self.base_url)),
			self.auth_entry.clone(),
		)
	}
}

/// 读环境变量，空/缺省时回退 `default`。
fn env_or(name: &str, default: &str) -> String {
	std::env::var(name)
		.ok()
		.filter(|value| !value.is_empty())
		.unwrap_or_else(|| default.to_owned())
}

/// 解析 system prompt（flag > env > 内置默认）。
///
/// flag/env 的值若指向一个存在的文件则读其内容，否则原样当文本。
///
/// # Errors
///
/// 指向的文件读失败时返错。
pub fn resolve_system_prompt(flag: Option<&str>) -> Result<String> {
	if let Some(value) = flag {
		return load_prompt_value(value);
	}
	if let Ok(value) = std::env::var("OMP_HEADLESS_SYSTEM_PROMPT")
		&& !value.is_empty()
	{
		return load_prompt_value(&value);
	}
	Ok(DEFAULT_SYSTEM_PROMPT.to_owned())
}

/// `value` 若是存在的文件路径则读其内容，否则原样返回。
fn load_prompt_value(value: &str) -> Result<String> {
	let path = Path::new(value);
	if path.is_file() {
		std::fs::read_to_string(path).with_context(|| format!("读 system prompt 文件失败: {value}"))
	} else {
		Ok(value.to_owned())
	}
}
