//! Real-provider glue: encode an [`LlmContext`] into Anthropic wire params and
//! wrap a pi-ai [`Client`] as a [`StreamFn`].
//!
//! pi-ai has no harness-message → wire encoder (its non-streaming converter
//! goes the other way, and its `stream` example builds `MessageParam` by hand),
//! so the multi-turn loop needs this minimal encoder to replay
//! assistant/`tool_use`/`tool_result` history on each request. It is
//! deliberately small — a full `transformMessages` (thinking-signature
//! handling, cache breakpoints, provider quirks) is a host/provider-layer
//! concern deferred beyond WP-1.4a. Scope: text + `tool_use` + `tool_result` +
//! images; `thinking` blocks without a signature and
//! `redactedThinking`/`fallback` are dropped, and consecutive same-role turns
//! are coalesced so tool-result batches stay a single wire message.

use pi_ai::{
	Client,
	message::{AssistantContent, Message, UserContent, UserContentBlock},
	wire::{
		ContentBlockParam, ImageSource, MessageContent, MessageCreateParams, MessageParam, Role,
		SystemPrompt, Tool as WireToolParam, ToolResultContent,
	},
};
use pi_shell::cancel::CancelToken;
use tokio_util::sync::CancellationToken;

use crate::loop_::{LlmContext, StreamFn};

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
		messages: encode_messages(&context.messages),
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

/// Encode harness messages into wire `MessageParam`s, coalescing consecutive
/// same-role turns (Anthropic wants a tool-result batch as one user message).
fn encode_messages(messages: &[Message]) -> Vec<MessageParam> {
	let mut out: Vec<MessageParam> = Vec::new();
	for message in messages {
		let (role, blocks) = encode_message(message);
		if blocks.is_empty() {
			continue;
		}
		if let Some(last) = out.last_mut()
			&& last.role == role
			&& let MessageContent::Blocks(existing) = &mut last.content
		{
			existing.extend(blocks);
			continue;
		}
		out.push(MessageParam { role, content: MessageContent::Blocks(blocks) });
	}
	out
}

fn encode_message(message: &Message) -> (Role, Vec<ContentBlockParam>) {
	match message {
		// Developer maps to the user role (Anthropic has no developer role in
		// the base API); host-layer system-role handling is deferred.
		Message::User(user) => (Role::User, encode_user_content(&user.content)),
		Message::Developer(dev) => (Role::User, encode_user_content(&dev.content)),
		Message::Assistant(assistant) => {
			let blocks = assistant
				.content
				.iter()
				.filter_map(encode_assistant_block)
				.collect();
			(Role::Assistant, blocks)
		},
		Message::ToolResult(result) => {
			let block = ContentBlockParam::ToolResult {
				tool_use_id:   result.tool_call_id.clone(),
				content:       Some(ToolResultContent::Blocks(encode_blocks(&result.content))),
				is_error:      Some(result.is_error),
				cache_control: None,
			};
			(Role::User, vec![block])
		},
	}
}

fn encode_user_content(content: &UserContent) -> Vec<ContentBlockParam> {
	match content {
		UserContent::Text(text) => {
			vec![ContentBlockParam::Text { text: text.clone(), cache_control: None }]
		},
		UserContent::Blocks(blocks) => encode_blocks(blocks),
	}
}

fn encode_blocks(blocks: &[UserContentBlock]) -> Vec<ContentBlockParam> {
	blocks
		.iter()
		.map(|block| match block {
			UserContentBlock::Text(text) => {
				ContentBlockParam::Text { text: text.text.clone(), cache_control: None }
			},
			UserContentBlock::Image(image) => ContentBlockParam::Image {
				source:        ImageSource::Base64 {
					media_type: image.mime_type.clone(),
					data:       image.data.clone(),
				},
				cache_control: None,
			},
		})
		.collect()
}

fn encode_assistant_block(block: &AssistantContent) -> Option<ContentBlockParam> {
	match block {
		AssistantContent::Text(text) => {
			Some(ContentBlockParam::Text { text: text.text.clone(), cache_control: None })
		},
		AssistantContent::ToolCall(call) => Some(ContentBlockParam::ToolUse {
			id:            call.id.clone(),
			name:          call.name.clone(),
			input:         call.arguments.clone(),
			cache_control: None,
		}),
		AssistantContent::Thinking(thinking) => {
			thinking
				.thinking_signature
				.as_ref()
				.map(|signature| ContentBlockParam::Thinking {
					thinking:  thinking.thinking.clone(),
					signature: signature.clone(),
				})
		},
		// Dropped on the wire: unsigned thinking, redacted thinking, images,
		// and fallback boundary markers (not needed to replay a tool loop).
		AssistantContent::RedactedThinking(_)
		| AssistantContent::Image(_)
		| AssistantContent::Fallback(_) => None,
	}
}
