//! Real-provider glue: encode an [`LlmContext`] into Anthropic wire params and
//! wrap a pi-ai [`Client`] as a [`StreamFn`].
//!
//! pi-ai has no harness-message → wire encoder (its non-streaming converter
//! goes the other way, and its `stream` example builds `MessageParam` by hand),
//! so the multi-turn loop needs this encoder to replay
//! assistant/`tool_use`/`tool_result` history on each request.
//!
//! The block/message-assembly rules mirror the TS reference encoder
//! `convertAnthropicMessages` (`packages/ai/src/providers/anthropic.ts:3546`)
//! for the barm driver's single target: the official Anthropic Messages API,
//! same-model replay of the driver's own history. Concretely (WP-1.6 C1-C4):
//!
//! * **Signed thinking** (`thinkingSignature` present) → wire `thinking` block
//!   carrying the signature (anthropic.ts:3627-3633). **Unsigned thinking** is
//!   NOT dropped: TS demotes it to a bare assistant text block for the
//!   Anthropic dialect (`renderDemotedThinking` → verbatim text,
//!   demotion.ts:35; anthropic.ts:3621-3626), which we mirror (whitespace-only
//!   thinking is skipped). The `replayUnsignedThinking` compat path (emit
//!   `signature: ""`) is a non-official-endpoint quirk barm never targets, so
//!   it is not implemented.
//! * **`redactedThinking`** → wire `redacted_thinking` block
//!   (anthropic.ts:3634-3639; empty `data` skipped). **`fallback`** boundary
//!   markers are dropped: the wire type exists, but TS only replays them when
//!   the request opts into the `server-side-fallback` beta AND the target is
//!   official Anthropic (anthropic.ts:3640-3652); barm sends no such beta, so
//!   the block is always dropped here.
//! * **Images**: user/`tool_result` images encode to base64 `image` blocks with
//!   media-type normalization and the shared `convertContentBlocks` semantics
//!   (anthropic.ts:978-1043). Assistant images are dropped — no provider
//!   accepts them in an assistant replay turn (transform-messages.ts:720-725).
//! * **Message assembly**: consecutive `tool_result` turns batch into one user
//!   message (anthropic.ts:3706-3737); adjacent assistant params and a trailing
//!   assistant param are repaired with a `Continue.` user turn
//!   (anthropic.ts:3765-3772).
//!
//! Deliberately deferred (documented, not needed for same-model well-formed
//! driver replay): the full `transformMessages` normalizer (tool-call-id
//! dedup/normalization, orphan `tool_result` handling, malformed-call
//! sanitization, synthetic aborted/`No result provided` results, cross-model
//! signature stripping — transform-messages.ts), prompt-cache `cache_control`
//! breakpoints (barm has no prompt-cache need), and the
//! developer→mid-conversation `system` upgrade (anthropic.ts:3748-3761;
//! developer maps to `user`).

use pi_ai::{
	Client,
	message::{
		AssistantContent, AssistantMessage, Message, ToolResultMessage, UserContent, UserContentBlock,
	},
	wire::{
		ContentBlockParam, ImageSource, MessageContent, MessageCreateParams, MessageParam, Role,
		SystemPrompt, Tool as WireToolParam, ToolResultContent,
	},
};
use pi_shell::cancel::CancelToken;
use tokio_util::sync::CancellationToken;

use crate::loop_::{LlmContext, StreamFn};

/// Placeholder for an image sent to a non-vision model
/// (`NON_VISION_IMAGE_PLACEHOLDER`, vision-guard.ts:6).
const NON_VISION_IMAGE_PLACEHOLDER: &str = "[image omitted: model does not support vision]";

/// Substituted for an empty `is_error` tool-result body
/// (`EMPTY_ERROR_TOOL_RESULT_TEXT`, anthropic.ts:3450).
const EMPTY_ERROR_TOOL_RESULT_TEXT: &str = "Tool failed with no output.";

/// Whether the driver's target model accepts image blocks. barm targets Claude
/// (vision-capable), so this is always true; the non-vision branch of
/// [`convert_content_blocks`] is kept for parity with the TS encoder and is
/// covered by a unit test, but is unreachable on the live path.
const TARGET_SUPPORTS_IMAGES: bool = true;

/// Wrap a pi-ai [`Client`] as a [`StreamFn`]. Each call encodes the loop's
/// [`LlmContext`] into `MessageCreateParams` and streams with mid-flight
/// cancellation bridged from the loop's [`CancelToken`].
#[must_use]
pub fn client_stream_fn(
	client: Client,
	model: String,
	max_tokens: u64,
	temperature: Option<f64>,
) -> StreamFn {
	Box::new(move |context: LlmContext, ct: CancelToken| {
		let params = encode_params(&context, &model, max_tokens, temperature);
		let token = bridge_cancel(&ct);
		client.stream_with_cancel(&params, token)
	})
}

/// Bridge a pi-shell [`CancelToken`] to a `tokio_util` [`CancellationToken`]
/// the pi-ai client understands: a detached task cancels the child when the
/// loop's token aborts. If the run never aborts the task idles until the
/// process ends (a per-request cost acceptable for the dev/example path).
fn bridge_cancel(ct: &CancelToken) -> CancellationToken {
	let token = CancellationToken::new();
	let child = token.clone();
	let ct = ct.clone();
	tokio::spawn(async move {
		tokio::select! {
			_ = ct.wait() => child.cancel(),
			() = child.cancelled() => {},
		}
	});
	token
}

/// Encode the loop context into Anthropic Messages request params.
#[must_use]
pub fn encode_params(
	context: &LlmContext,
	model: &str,
	max_tokens: u64,
	temperature: Option<f64>,
) -> MessageCreateParams {
	let system = if context.system_prompt.is_empty() {
		None
	} else {
		Some(SystemPrompt::Text(context.system_prompt.join("\n\n")))
	};

	let tools = if context.tools.is_empty() {
		None
	} else {
		Some(
			context
				.tools
				.iter()
				.map(|tool| WireToolParam {
					name:                  tool.name.clone(),
					description:           Some(tool.description.clone()),
					input_schema:          tool.input_schema.clone(),
					cache_control:         None,
					strict:                None,
					eager_input_streaming: None,
				})
				.collect(),
		)
	};

	MessageCreateParams {
		model: model.to_owned(),
		messages: encode_messages(&context.messages, TARGET_SUPPORTS_IMAGES),
		max_tokens,
		system,
		temperature,
		top_p: None,
		top_k: None,
		stop_sequences: None,
		stream: None,
		tools,
		tool_choice: None,
		metadata: None,
		thinking: None,
		output_config: None,
		speed: None,
		context_management: None,
		fallbacks: None,
	}
}

/// Encode harness messages into wire `MessageParam`s.
///
/// Mirrors the assembly loop of `convertAnthropicMessages`
/// (anthropic.ts:3560-3739) minus the deferred `transformMessages` pre-pass:
/// user/developer turns pass through (developer → user role), assistant turns
/// encode per [`encode_assistant_message`], and consecutive `tool_result`
/// messages batch into a single user message. A closing pass repairs adjacent /
/// trailing assistant params with a `Continue.` user turn.
fn encode_messages(messages: &[Message], supports_images: bool) -> Vec<MessageParam> {
	let mut params: Vec<MessageParam> = Vec::new();
	let mut index = 0;
	while index < messages.len() {
		match &messages[index] {
			// Developer maps to the user role (the mid-conversation `system`
			// upgrade, anthropic.ts:3748, is deferred).
			Message::User(user) => push_user_like(&mut params, &user.content, supports_images),
			Message::Developer(dev) => push_user_like(&mut params, &dev.content, supports_images),
			Message::Assistant(assistant) => {
				if let Some(blocks) = encode_assistant_message(assistant) {
					params.push(MessageParam {
						role:    Role::Assistant,
						content: MessageContent::Blocks(blocks),
					});
				}
			},
			Message::ToolResult(_) => {
				// Batch this and every immediately-following tool-result message
				// into one user turn (anthropic.ts:3706-3737). Images illegal in
				// error results are hoisted after the run.
				let mut blocks: Vec<ContentBlockParam> = Vec::new();
				let mut hoisted: Vec<ContentBlockParam> = Vec::new();
				let mut lookahead = index;
				while let Some(Message::ToolResult(result)) = messages.get(lookahead) {
					blocks.push(build_tool_result_block(result, supports_images, &mut hoisted));
					lookahead += 1;
				}
				index = lookahead - 1;
				if !hoisted.is_empty() {
					blocks.push(ContentBlockParam::Text {
						text:          "Attached image(s) from the tool result(s) above:".to_owned(),
						cache_control: None,
					});
					blocks.extend(hoisted);
				}
				params
					.push(MessageParam { role: Role::User, content: MessageContent::Blocks(blocks) });
			},
		}
		index += 1;
	}
	repair_adjacent_assistants(&mut params);
	params
}

/// Push a user/developer turn, dropping whitespace-only string content and
/// empty block content (anthropic.ts:3563-3581).
fn push_user_like(params: &mut Vec<MessageParam>, content: &UserContent, supports_images: bool) {
	match content {
		UserContent::Text(text) => {
			if text.trim().is_empty() {
				return;
			}
			params.push(MessageParam {
				role:    Role::User,
				content: MessageContent::Text(text.clone()),
			});
		},
		UserContent::Blocks(blocks) => {
			let converted = convert_content_blocks(blocks, supports_images);
			if is_empty_wire_content(&converted) {
				return;
			}
			params
				.push(MessageParam { role: Role::User, content: MessageContent::Blocks(converted) });
		},
	}
}

/// Whether converted content carries nothing sendable: no blocks, or a single
/// whitespace-only text block (the non-vision join can yield the latter).
/// Mirror of `isEmptyToolResultWireContent` (anthropic.ts:3452) generalized to
/// blocks.
fn is_empty_wire_content(blocks: &[ContentBlockParam]) -> bool {
	match blocks {
		[] => true,
		[ContentBlockParam::Text { text, .. }] => text.trim().is_empty(),
		_ => false,
	}
}

/// Rust port of `convertContentBlocks` (anthropic.ts:978-1043): encode text +
/// image blocks, skipping whitespace-only text, normalizing image media types,
/// and — when the target has vision — prepending a `(see attached image)` text
/// block to image-only content. When the target lacks vision, all surviving
/// text is joined into a single block (mirroring the TS string return).
fn convert_content_blocks(
	blocks: &[UserContentBlock],
	supports_images: bool,
) -> Vec<ContentBlockParam> {
	let mut out: Vec<ContentBlockParam> = Vec::new();
	let mut saw_text = false;
	let mut saw_image = false;
	for block in blocks {
		match block {
			UserContentBlock::Text(text) => {
				if text.text.trim().is_empty() {
					continue;
				}
				saw_text = true;
				out.push(ContentBlockParam::Text {
					text:          text.text.clone(),
					cache_control: None,
				});
			},
			UserContentBlock::Image(image) => {
				if !supports_images {
					out.push(ContentBlockParam::Text {
						text:          NON_VISION_IMAGE_PLACEHOLDER.to_owned(),
						cache_control: None,
					});
					continue;
				}
				match normalize_image_media_type(&image.mime_type) {
					None => out.push(ContentBlockParam::Text {
						text:          format!("[unsupported image: {}]", image.mime_type),
						cache_control: None,
					}),
					Some(media_type) => {
						saw_image = true;
						out.push(ContentBlockParam::Image {
							source:        ImageSource::Base64 { media_type, data: image.data.clone() },
							cache_control: None,
						});
					},
				}
			},
		}
	}
	if !supports_images {
		let joined = out
			.iter()
			.filter_map(|block| match block {
				ContentBlockParam::Text { text, .. } => Some(text.as_str()),
				_ => None,
			})
			.collect::<Vec<_>>()
			.join("\n");
		return vec![ContentBlockParam::Text { text: joined, cache_control: None }];
	}
	if saw_image && !saw_text {
		out.insert(0, ContentBlockParam::Text {
			text:          "(see attached image)".to_owned(),
			cache_control: None,
		});
	}
	out
}

/// Normalize an image mime type to an Anthropic-accepted media type
/// (`normalizeAnthropicImageMediaType`, anthropic.ts:325). `image/jpg` folds to
/// `image/jpeg`; unsupported types return `None`.
fn normalize_image_media_type(mime: &str) -> Option<String> {
	let normalized = mime.trim().to_ascii_lowercase();
	match normalized.as_str() {
		"image/jpg" => Some("image/jpeg".to_owned()),
		"image/jpeg" | "image/png" | "image/gif" | "image/webp" => Some(normalized),
		_ => None,
	}
}

/// Build one `tool_result` block (`buildToolResultBlock`, anthropic.ts:3471):
/// convert the body, hoist images out of error results (Anthropic rejects
/// them), and substitute a placeholder for an empty error body.
fn build_tool_result_block(
	result: &ToolResultMessage,
	supports_images: bool,
	hoisted: &mut Vec<ContentBlockParam>,
) -> ContentBlockParam {
	let mut content = convert_content_blocks(&result.content, supports_images);
	if result.is_error
		&& content
			.iter()
			.any(|b| matches!(b, ContentBlockParam::Image { .. }))
	{
		for block in &content {
			if matches!(block, ContentBlockParam::Image { .. }) {
				hoisted.push(block.clone());
			}
		}
		content.retain(|block| matches!(block, ContentBlockParam::Text { .. }));
	}
	if result.is_error && is_empty_wire_content(&content) {
		content = vec![ContentBlockParam::Text {
			text:          EMPTY_ERROR_TOOL_RESULT_TEXT.to_owned(),
			cache_control: None,
		}];
	}
	ContentBlockParam::ToolResult {
		tool_use_id:   result.tool_call_id.clone(),
		content:       Some(ToolResultContent::Blocks(content)),
		is_error:      Some(result.is_error),
		cache_control: None,
	}
}

/// Encode one assistant turn's content blocks, returning `None` when nothing
/// survives (anthropic.ts:3582-3705). After encoding, a stable partition moves
/// any non-`tool_use` block that trailed a `tool_use` back ahead of the tool
/// calls (anthropic.ts:3681-3700).
fn encode_assistant_message(assistant: &AssistantMessage) -> Option<Vec<ContentBlockParam>> {
	let mut blocks: Vec<ContentBlockParam> = Vec::new();
	for block in &assistant.content {
		match block {
			AssistantContent::Text(text) => {
				if text.text.trim().is_empty() {
					continue;
				}
				blocks.push(ContentBlockParam::Text {
					text:          text.text.clone(),
					cache_control: None,
				});
			},
			AssistantContent::Thinking(thinking) => encode_thinking_block(thinking, &mut blocks),
			AssistantContent::RedactedThinking(redacted) => {
				if redacted.data.trim().is_empty() {
					continue;
				}
				blocks.push(ContentBlockParam::RedactedThinking { data: redacted.data.clone() });
			},
			// Dropped: fallback boundary markers (barm sends no server-side-fallback
			// beta, anthropic.ts:3647) and assistant images (no provider accepts them
			// in an assistant replay turn, transform-messages.ts:720).
			AssistantContent::Fallback(_) | AssistantContent::Image(_) => {},
			AssistantContent::ToolCall(call) => blocks.push(ContentBlockParam::ToolUse {
				id:            call.id.clone(),
				name:          call.name.clone(),
				input:         if call.arguments.is_null() {
					serde_json::json!({})
				} else {
					call.arguments.clone()
				},
				cache_control: None,
			}),
		}
	}
	partition_tool_use_last(&mut blocks);
	if blocks.is_empty() {
		None
	} else {
		Some(blocks)
	}
}

/// Encode a thinking block (anthropic.ts:3596-3633, official-Anthropic path). A
/// signed block replays natively as a `thinking` block; an unsigned block is
/// demoted to a bare assistant text block (`renderDemotedThinking` for the
/// Anthropic dialect is verbatim text, demotion.ts:35). Whitespace-only
/// unsigned thinking is dropped (nothing to replay).
fn encode_thinking_block(
	thinking: &pi_ai::message::ThinkingContent,
	blocks: &mut Vec<ContentBlockParam>,
) {
	let signed = thinking
		.thinking_signature
		.as_ref()
		.is_some_and(|signature| !signature.trim().is_empty());
	if signed {
		blocks.push(ContentBlockParam::Thinking {
			thinking:  thinking.thinking.clone(),
			signature: thinking.thinking_signature.clone().unwrap_or_default(),
		});
		return;
	}
	if thinking.thinking.trim().is_empty() {
		return;
	}
	blocks.push(ContentBlockParam::Text {
		text:          thinking.thinking.clone(),
		cache_control: None,
	});
}

/// Stable-partition assistant blocks into `[..non-tool_use, ..tool_use]` when a
/// non-`tool_use` block trails a `tool_use` (Anthropic rejects that ordering).
/// Fast-path leaves already-ordered content untouched (anthropic.ts:3681-3700).
fn partition_tool_use_last(blocks: &mut Vec<ContentBlockParam>) {
	let mut saw_tool_use = false;
	let mut needs_partition = false;
	for block in blocks.iter() {
		if matches!(block, ContentBlockParam::ToolUse { .. }) {
			saw_tool_use = true;
		} else if saw_tool_use {
			needs_partition = true;
			break;
		}
	}
	if !needs_partition {
		return;
	}
	let mut non_tool_use: Vec<ContentBlockParam> = Vec::new();
	let mut tool_use: Vec<ContentBlockParam> = Vec::new();
	for block in blocks.drain(..) {
		if matches!(block, ContentBlockParam::ToolUse { .. }) {
			tool_use.push(block);
		} else {
			non_tool_use.push(block);
		}
	}
	non_tool_use.append(&mut tool_use);
	*blocks = non_tool_use;
}

/// Insert a `Continue.` user turn between adjacent assistant params and after a
/// trailing assistant param — Anthropic rejects consecutive / trailing
/// assistant messages (anthropic.ts:3765-3772).
fn repair_adjacent_assistants(params: &mut Vec<MessageParam>) {
	let mut index = params.len();
	while index > 1 {
		index -= 1;
		if params[index].role == Role::Assistant && params[index - 1].role == Role::Assistant {
			params.insert(index, MessageParam {
				role:    Role::User,
				content: MessageContent::Text("Continue.".to_owned()),
			});
		}
	}
	if params
		.last()
		.is_some_and(|last| last.role == Role::Assistant)
	{
		params.push(MessageParam {
			role:    Role::User,
			content: MessageContent::Text("Continue.".to_owned()),
		});
	}
}

#[cfg(test)]
mod tests {
	//! Round-trip encoding shape assertions for the harness → Anthropic wire
	//! encoder (WP-1.6 C1-C4). Each block kind gets a case, plus a multi-turn
	//! replay sequence (assistant thinking+signature+toolCall → toolResult →
	//! assistant).

	use pi_ai::{
		message::{
			AssistantContent, AssistantMessage, FallbackContent, ImageContent, Message,
			RedactedThinkingContent, StopReason, TextContent, ThinkingContent, ToolCall,
			ToolResultMessage, Usage, UserContent, UserContentBlock, UserMessage,
		},
		wire::{ContentBlockParam, ImageSource, MessageContent, MessageParam, ModelRef, Role},
	};
	use serde_json::json;

	use super::encode_messages;

	// ─── fixtures ───────────────────────────────────────────────────────────

	fn assistant(content: Vec<AssistantContent>, stop: StopReason) -> Message {
		Message::Assistant(Box::new(AssistantMessage {
			content,
			api: "anthropic-messages".into(),
			provider: "anthropic".into(),
			model: "claude".into(),
			context_snapshot: None,
			retry_recovery: None,
			response_id: None,
			upstream_provider: None,
			usage: Usage::default(),
			stop_reason: stop,
			stop_details: None,
			error_message: None,
			tool_call_abort_messages: None,
			error_status: None,
			error_id: None,
			disabled_features: None,
			provider_payload: None,
			timestamp: 0,
			duration: None,
			ttft: None,
		}))
	}

	fn user_text(text: &str) -> Message {
		Message::User(UserMessage {
			content:          UserContent::Text(text.into()),
			synthetic:        None,
			steering:         None,
			attribution:      None,
			provider_payload: None,
			timestamp:        0,
		})
	}

	fn user_blocks(blocks: Vec<UserContentBlock>) -> Message {
		Message::User(UserMessage {
			content:          UserContent::Blocks(blocks),
			synthetic:        None,
			steering:         None,
			attribution:      None,
			provider_payload: None,
			timestamp:        0,
		})
	}

	fn tool_result(id: &str, name: &str, content: Vec<UserContentBlock>, is_error: bool) -> Message {
		Message::ToolResult(ToolResultMessage {
			tool_call_id: id.into(),
			tool_name: name.into(),
			content,
			details: None,
			is_error,
			attribution: None,
			pruned_at: None,
			useless: None,
			timestamp: 0,
		})
	}

	fn text_block(text: &str) -> AssistantContent {
		AssistantContent::Text(TextContent { text: text.into(), text_signature: None })
	}

	fn thinking(text: &str, signature: Option<&str>) -> AssistantContent {
		AssistantContent::Thinking(ThinkingContent {
			thinking:           text.into(),
			thinking_signature: signature.map(Into::into),
			item_id:            None,
		})
	}

	fn tool_call(id: &str, name: &str, args: serde_json::Value) -> AssistantContent {
		AssistantContent::ToolCall(ToolCall {
			id:                id.into(),
			name:              name.into(),
			arguments:         args,
			thought_signature: None,
			intent:            None,
			raw_block:         None,
			custom_wire_name:  None,
		})
	}

	fn u_text(text: &str) -> UserContentBlock {
		UserContentBlock::Text(TextContent { text: text.into(), text_signature: None })
	}

	fn u_image(data: &str, mime: &str) -> UserContentBlock {
		UserContentBlock::Image(ImageContent {
			data:      data.into(),
			mime_type: mime.into(),
			detail:    None,
		})
	}

	fn assistant_blocks(params: &[MessageParam], index: usize) -> &[ContentBlockParam] {
		match &params[index].content {
			MessageContent::Blocks(blocks) => blocks,
			MessageContent::Text(_) => panic!("expected block content at param {index}"),
		}
	}

	// ─── C1: thinking ───────────────────────────────────────────────────────

	#[test]
	fn signed_thinking_round_trips_as_thinking_block() {
		let msgs = vec![
			user_text("hi"),
			assistant(
				vec![thinking("reason", Some("sig-abc")), text_block("answer")],
				StopReason::Stop,
			),
		];
		let params = encode_messages(&msgs, true);
		// user, assistant, trailing "Continue." (assistant is last).
		assert_eq!(params.len(), 3);
		assert_eq!(params[1].role, Role::Assistant);
		assert_eq!(assistant_blocks(&params, 1), &[
			ContentBlockParam::Thinking { thinking: "reason".into(), signature: "sig-abc".into() },
			ContentBlockParam::Text { text: "answer".into(), cache_control: None },
		]);
	}

	#[test]
	fn unsigned_thinking_demotes_to_bare_text() {
		// TS renderDemotedThinking for the Anthropic dialect emits verbatim text,
		// NOT a dropped block (demotion.ts:35). Whitespace-only thinking IS dropped.
		let msgs = vec![assistant(
			vec![thinking("bare reasoning", None), thinking("   ", None), text_block("done")],
			StopReason::Stop,
		)];
		let params = encode_messages(&msgs, true);
		assert_eq!(assistant_blocks(&params, 0), &[
			ContentBlockParam::Text { text: "bare reasoning".into(), cache_control: None },
			ContentBlockParam::Text { text: "done".into(), cache_control: None },
		]);
	}

	// ─── C2: redactedThinking / fallback ──────────────────────────────────────

	#[test]
	fn redacted_thinking_encodes_and_skips_empty() {
		let msgs = vec![assistant(
			vec![
				AssistantContent::RedactedThinking(RedactedThinkingContent { data: "enc==".into() }),
				AssistantContent::RedactedThinking(RedactedThinkingContent { data: "  ".into() }),
				text_block("visible"),
			],
			StopReason::Stop,
		)];
		let params = encode_messages(&msgs, true);
		assert_eq!(assistant_blocks(&params, 0), &[
			ContentBlockParam::RedactedThinking { data: "enc==".into() },
			ContentBlockParam::Text { text: "visible".into(), cache_control: None },
		]);
	}

	#[test]
	fn fallback_marker_is_dropped() {
		// barm sends no server-side-fallback beta, so persisted fallback markers
		// are dropped (anthropic.ts:3647).
		let msgs = vec![assistant(
			vec![
				AssistantContent::Fallback(FallbackContent {
					from: ModelRef { model: "fable".into() },
					to:   ModelRef { model: "opus".into() },
				}),
				text_block("after fallback"),
			],
			StopReason::Stop,
		)];
		let params = encode_messages(&msgs, true);
		assert_eq!(assistant_blocks(&params, 0), &[ContentBlockParam::Text {
			text:          "after fallback".into(),
			cache_control: None,
		}]);
	}

	// ─── C3: images ───────────────────────────────────────────────────────────

	#[test]
	fn assistant_image_is_dropped() {
		let msgs = vec![assistant(
			vec![
				AssistantContent::Image(ImageContent {
					data:      "AAAA".into(),
					mime_type: "image/png".into(),
					detail:    None,
				}),
				text_block("caption"),
			],
			StopReason::Stop,
		)];
		let params = encode_messages(&msgs, true);
		assert_eq!(assistant_blocks(&params, 0), &[ContentBlockParam::Text {
			text:          "caption".into(),
			cache_control: None,
		}]);
	}

	#[test]
	fn user_image_encodes_base64_with_media_normalization_and_synthetic_text() {
		// Image-only user content: jpg normalizes to jpeg, and a synthetic
		// "(see attached image)" text block is prepended (anthropic.ts:1035).
		let msgs = vec![user_blocks(vec![u_image("DATA", "IMAGE/JPG")])];
		let params = encode_messages(&msgs, true);
		assert_eq!(assistant_blocks(&params, 0), &[
			ContentBlockParam::Text {
				text:          "(see attached image)".into(),
				cache_control: None,
			},
			ContentBlockParam::Image {
				source:        ImageSource::Base64 {
					media_type: "image/jpeg".into(),
					data:       "DATA".into(),
				},
				cache_control: None,
			},
		]);
	}

	#[test]
	fn unsupported_image_becomes_text_placeholder_no_synthetic() {
		// Unsupported mime → text placeholder; since sawImage stays false, no
		// synthetic "(see attached image)" is added.
		let msgs = vec![user_blocks(vec![u_image("DATA", "image/tiff")])];
		let params = encode_messages(&msgs, true);
		assert_eq!(assistant_blocks(&params, 0), &[ContentBlockParam::Text {
			text:          "[unsupported image: image/tiff]".into(),
			cache_control: None,
		}]);
	}

	#[test]
	fn non_vision_target_joins_text_and_placeholders() {
		let msgs = vec![user_blocks(vec![u_text("look"), u_image("DATA", "image/png")])];
		let params = encode_messages(&msgs, false);
		assert_eq!(assistant_blocks(&params, 0), &[ContentBlockParam::Text {
			text:          format!("look\n{}", super::NON_VISION_IMAGE_PLACEHOLDER),
			cache_control: None,
		}]);
	}

	// ─── C4: message assembly ─────────────────────────────────────────────────

	#[test]
	fn consecutive_tool_results_batch_into_one_user_message() {
		let msgs = vec![
			assistant(
				vec![tool_call("t1", "read", json!({})), tool_call("t2", "grep", json!({}))],
				StopReason::ToolUse,
			),
			tool_result("t1", "read", vec![u_text("file body")], false),
			tool_result("t2", "grep", vec![u_text("match")], false),
			user_text("thanks"),
		];
		let params = encode_messages(&msgs, true);
		// assistant, ONE user (both tool results), user "thanks".
		assert_eq!(params.len(), 3);
		assert_eq!(params[1].role, Role::User);
		let batched = assistant_blocks(&params, 1);
		assert_eq!(batched.len(), 2);
		assert!(
			matches!(&batched[0], ContentBlockParam::ToolResult { tool_use_id, .. } if tool_use_id == "t1")
		);
		assert!(
			matches!(&batched[1], ContentBlockParam::ToolResult { tool_use_id, .. } if tool_use_id == "t2")
		);
		assert_eq!(params[2].content, MessageContent::Text("thanks".into()));
	}

	#[test]
	fn empty_error_tool_result_gets_placeholder() {
		let msgs = vec![
			assistant(vec![tool_call("t1", "bash", json!({}))], StopReason::ToolUse),
			tool_result("t1", "bash", vec![], true),
		];
		let params = encode_messages(&msgs, true);
		let block = &assistant_blocks(&params, 1)[0];
		match block {
			ContentBlockParam::ToolResult { content, is_error, .. } => {
				assert_eq!(*is_error, Some(true));
				assert_eq!(
					*content,
					Some(pi_ai::wire::ToolResultContent::Blocks(vec![ContentBlockParam::Text {
						text:          super::EMPTY_ERROR_TOOL_RESULT_TEXT.into(),
						cache_control: None,
					}]))
				);
			},
			other => panic!("expected tool_result, got {other:?}"),
		}
	}

	#[test]
	fn error_tool_result_hoists_images_out() {
		// Images are illegal in an error tool_result; text stays, images hoist
		// after the run behind a caption (anthropic.ts:3480/3726).
		let msgs = vec![
			assistant(vec![tool_call("t1", "screenshot", json!({}))], StopReason::ToolUse),
			tool_result("t1", "screenshot", vec![u_text("boom"), u_image("IMG", "image/png")], true),
		];
		let params = encode_messages(&msgs, true);
		let blocks = assistant_blocks(&params, 1);
		// tool_result (text only) + caption text + hoisted image.
		assert_eq!(blocks.len(), 3);
		match &blocks[0] {
			ContentBlockParam::ToolResult { content, .. } => assert_eq!(
				*content,
				Some(pi_ai::wire::ToolResultContent::Blocks(vec![ContentBlockParam::Text {
					text:          "boom".into(),
					cache_control: None,
				}]))
			),
			other => panic!("expected tool_result, got {other:?}"),
		}
		assert!(
			matches!(&blocks[1], ContentBlockParam::Text { text, .. } if text.starts_with("Attached image"))
		);
		assert!(matches!(&blocks[2], ContentBlockParam::Image { .. }));
	}

	#[test]
	fn trailing_block_after_tool_use_is_partitioned_after() {
		let msgs = vec![assistant(
			vec![text_block("before"), tool_call("t1", "read", json!({})), text_block("after")],
			StopReason::ToolUse,
		)];
		let params = encode_messages(&msgs, true);
		let blocks = assistant_blocks(&params, 0);
		assert_eq!(blocks, &[
			ContentBlockParam::Text { text: "before".into(), cache_control: None },
			ContentBlockParam::Text { text: "after".into(), cache_control: None },
			ContentBlockParam::ToolUse {
				id:            "t1".into(),
				name:          "read".into(),
				input:         json!({}),
				cache_control: None,
			},
		]);
	}

	#[test]
	fn trailing_assistant_gets_continue_turn() {
		let msgs = vec![user_text("hi"), assistant(vec![text_block("bye")], StopReason::Stop)];
		let params = encode_messages(&msgs, true);
		assert_eq!(params.len(), 3);
		assert_eq!(params[2].role, Role::User);
		assert_eq!(params[2].content, MessageContent::Text("Continue.".into()));
	}

	#[test]
	fn adjacent_assistants_get_continue_between() {
		// Two assistant turns with no user turn between (e.g. an empty user turn
		// was dropped) must be separated by a Continue. user turn.
		let msgs = vec![
			assistant(vec![text_block("one")], StopReason::Stop),
			user_text("   "), // whitespace-only → dropped, leaving adjacent assistants
			assistant(vec![text_block("two")], StopReason::Stop),
		];
		let params = encode_messages(&msgs, true);
		// assistant, Continue., assistant, trailing Continue.
		assert_eq!(params.len(), 4);
		assert_eq!(params[0].role, Role::Assistant);
		assert_eq!(params[1].content, MessageContent::Text("Continue.".into()));
		assert_eq!(params[2].role, Role::Assistant);
		assert_eq!(params[3].content, MessageContent::Text("Continue.".into()));
	}

	#[test]
	fn whitespace_text_and_empty_user_are_skipped() {
		let msgs = vec![
			user_text("   "),
			user_blocks(vec![u_text("  ")]),
			assistant(vec![text_block("  "), text_block("kept")], StopReason::Stop),
		];
		let params = encode_messages(&msgs, true);
		// Both user turns dropped; assistant keeps only "kept"; trailing Continue.
		assert_eq!(params.len(), 2);
		assert_eq!(assistant_blocks(&params, 0), &[ContentBlockParam::Text {
			text:          "kept".into(),
			cache_control: None,
		}]);
		assert_eq!(params[1].content, MessageContent::Text("Continue.".into()));
	}

	// ─── multi-turn replay sequence ──────────────────────────────────────────

	#[test]
	fn multi_turn_thinking_toolcall_result_replay() {
		// assistant(thinking+signature, toolCall) → toolResult → assistant(text).
		let msgs = vec![
			user_text("fix the bug"),
			assistant(
				vec![
					thinking("let me look", Some("sig1")),
					tool_call("t1", "read", json!({"path": "a.py"})),
				],
				StopReason::ToolUse,
			),
			tool_result("t1", "read", vec![u_text("def add(): ...")], false),
			assistant(vec![text_block("fixed it")], StopReason::Stop),
		];
		let params = encode_messages(&msgs, true);
		// user, assistant(think+tooluse), user(toolresult), assistant(text), Continue.
		assert_eq!(params.len(), 5);
		assert_eq!(params[0].content, MessageContent::Text("fix the bug".into()));

		assert_eq!(params[1].role, Role::Assistant);
		assert_eq!(assistant_blocks(&params, 1), &[
			ContentBlockParam::Thinking { thinking: "let me look".into(), signature: "sig1".into() },
			ContentBlockParam::ToolUse {
				id:            "t1".into(),
				name:          "read".into(),
				input:         json!({"path": "a.py"}),
				cache_control: None,
			},
		]);

		assert_eq!(params[2].role, Role::User);
		assert!(
			matches!(&assistant_blocks(&params, 2)[0], ContentBlockParam::ToolResult { tool_use_id, .. } if tool_use_id == "t1")
		);

		assert_eq!(params[3].role, Role::Assistant);
		assert_eq!(assistant_blocks(&params, 3), &[ContentBlockParam::Text {
			text:          "fixed it".into(),
			cache_control: None,
		}]);

		assert_eq!(params[4].content, MessageContent::Text("Continue.".into()));
	}
}
