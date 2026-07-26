//! WP-1.5 L2 step 4：读回一份 omp-headless 写的 session journal，验证格式可被
//! pi-session loader（`load_session_messages`）重建为 LLM 消息视图。
//!
//! ```sh
//! cargo run -p omp-headless --example readback -- <cwd>/.omp-headless/sessions/<file>.jsonl
//! ```

use anyhow::{Context, Result};
use pi_ai::message::Message;
use pi_session::{ContextMessage, load_session_messages};

const fn role_of(message: &ContextMessage) -> &'static str {
	match message {
		ContextMessage::Standard(Message::User(_)) => "user",
		ContextMessage::Standard(Message::Developer(_)) => "developer",
		ContextMessage::Standard(Message::Assistant(_)) => "assistant",
		ContextMessage::Standard(Message::ToolResult(_)) => "toolResult",
		ContextMessage::CompactionSummary(_) => "compactionSummary",
		ContextMessage::Custom(_) => "custom",
		ContextMessage::BranchSummary(_) => "branchSummary",
	}
}

fn main() -> Result<()> {
	let path = std::env::args()
		.nth(1)
		.context("用法: readback <session.jsonl>")?;
	let messages =
		load_session_messages(&path).with_context(|| format!("读回 session 失败: {path}"))?;
	println!("readback ok: {} 条消息 <- {path}", messages.len());
	for (index, message) in messages.iter().enumerate() {
		println!("  [{index}] {}", role_of(message));
	}
	Ok(())
}
