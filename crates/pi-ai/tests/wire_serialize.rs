//! Request-serialization golden snapshots.
//!
//! Fixtures are blessed from the serializer itself (`BLESS=1 cargo test -p
//! pi-ai`), reviewed, and committed; the test then locks the byte-exact JSON so
//! any serde-attribute drift (field order, missing `skip_serializing_if`, tag
//! renames) shows up as a diff against git.

use pi_ai::wire::{
	CacheControl, CacheTtl, ContentBlockParam, ContextManagement, ContextManagementEdit,
	ImageSource, MessageContent, MessageCreateParams, MessageParam, Metadata, ModelRef, Role,
	SystemPrompt, TaskBudget, ThinkingConfig, ThinkingDisplay, Tool, ToolChoice, ToolResultContent,
};

fn assert_snapshot(name: &str, actual: &str) {
	let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
	if std::env::var("BLESS").is_ok() {
		std::fs::write(&path, format!("{actual}\n")).expect("write snapshot");
		return;
	}
	let expected = std::fs::read_to_string(&path).expect("read snapshot; run with BLESS=1 first");
	assert_eq!(actual, expected.trim_end(), "snapshot mismatch for {name}");
}

fn minimal_request() -> MessageCreateParams {
	MessageCreateParams {
		model:              "deepseek-v4-flash".into(),
		messages:           vec![MessageParam {
			role:    Role::User,
			content: MessageContent::Text("Reply with exactly: pong".into()),
		}],
		max_tokens:         64,
		system:             None,
		temperature:        None,
		top_p:              None,
		top_k:              None,
		stop_sequences:     None,
		stream:             Some(false),
		tools:              None,
		tool_choice:        None,
		metadata:           None,
		thinking:           None,
		output_config:      None,
		speed:              None,
		context_management: None,
		fallbacks:          None,
	}
}

fn full_request() -> MessageCreateParams {
	MessageCreateParams {
		model:              "claude-sonnet-5".into(),
		messages:           vec![
			MessageParam {
				role:    Role::User,
				content: MessageContent::Blocks(vec![
					ContentBlockParam::Text {
						text:          "What is in this image?".into(),
						cache_control: Some(CacheControl::Ephemeral {
							ttl:   Some(CacheTtl::FiveMinutes),
							scope: None,
						}),
					},
					ContentBlockParam::Image {
						source:        ImageSource::Base64 {
							media_type: "image/png".into(),
							data:       "iVBORw0KGgo=".into(),
						},
						cache_control: None,
					},
				]),
			},
			MessageParam {
				role:    Role::Assistant,
				content: MessageContent::Blocks(vec![
					ContentBlockParam::Thinking {
						thinking:  "Considering the image…".into(),
						signature: "sig_abc".into(),
					},
					ContentBlockParam::RedactedThinking { data: "opaque".into() },
					ContentBlockParam::ToolUse {
						id:            "toolu_01".into(),
						name:          "get_weather".into(),
						input:         serde_json::json!({"location": "Paris"}),
						cache_control: None,
					},
					ContentBlockParam::Fallback {
						from: ModelRef { model: "claude-sonnet-5".into() },
						to:   ModelRef { model: "claude-haiku-4-5".into() },
					},
				]),
			},
			MessageParam {
				role:    Role::User,
				content: MessageContent::Blocks(vec![ContentBlockParam::ToolResult {
					tool_use_id:   "toolu_01".into(),
					content:       Some(ToolResultContent::Text("Sunny, 21°C".into())),
					is_error:      Some(false),
					cache_control: None,
				}]),
			},
		],
		max_tokens:         4096,
		system:             Some(SystemPrompt::Blocks(vec![ContentBlockParam::Text {
			text:          "You are a helpful assistant.".into(),
			cache_control: Some(CacheControl::Ephemeral { ttl: None, scope: None }),
		}])),
		temperature:        Some(0.7),
		top_p:              Some(0.9),
		top_k:              Some(40),
		stop_sequences:     Some(vec!["<END>".into()]),
		stream:             Some(false),
		tools:              Some(vec![Tool {
			name:                  "get_weather".into(),
			description:           Some("Get the current weather for a location.".into()),
			input_schema:          serde_json::json!({
				"type": "object",
				"properties": {"location": {"type": "string"}},
				"required": ["location"]
			}),
			cache_control:         None,
			strict:                Some(true),
			eager_input_streaming: None,
		}]),
		tool_choice:        Some(ToolChoice::Auto { disable_parallel_tool_use: Some(false) }),
		metadata:           Some(Metadata { user_id: Some("user-123".into()) }),
		thinking:           Some(ThinkingConfig::Enabled {
			budget_tokens: 2048,
			display:       Some(ThinkingDisplay::Summarized),
		}),
		output_config:      Some(pi_ai::wire::OutputConfig {
			effort:      Some("high".into()),
			task_budget: Some(TaskBudget::Tokens { total: 100_000, remaining: Some(60_000) }),
		}),
		speed:              None,
		context_management: Some(ContextManagement {
			edits: vec![ContextManagementEdit {
				edit_type: "clear_thinking_20251015".into(),
				keep:      "all".into(),
			}],
		}),
		fallbacks:          None,
	}
}

#[test]
fn request_minimal_snapshot() {
	let json = serde_json::to_string_pretty(&minimal_request()).unwrap();
	assert_snapshot("request_minimal.json", &json);
}

#[test]
fn request_full_snapshot() {
	let json = serde_json::to_string_pretty(&full_request()).unwrap();
	assert_snapshot("request_full.json", &json);
}

#[test]
fn absent_options_are_omitted() {
	let json = serde_json::to_string(&minimal_request()).unwrap();
	for key in
		["temperature", "top_p", "top_k", "system", "tools", "tool_choice", "thinking", "metadata"]
	{
		assert!(!json.contains(&format!("\"{key}\"")), "unexpected key {key} in {json}");
	}
}

#[test]
fn full_request_round_trips() {
	let request = full_request();
	let json = serde_json::to_string(&request).unwrap();
	let back: MessageCreateParams = serde_json::from_str(&json).unwrap();
	assert_eq!(request, back);
}
