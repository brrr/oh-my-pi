//! `Message` untagged 联合的 role 分派回归（I-omp-27）。
//!
//! struct 级 `#[serde(tag = "role")]` 在 untagged 探测下不校验 tag，曾使
//! `toolResult` JSON 误配进字段更宽松的 `UserMessage`。自定义 Deserialize
//! 按 `role` 显式分派后，四角色各归其位，未知 role 报错（pi-session 依赖
//! 该错误把非标准角色整条落 Unknown 透传）。

use pi_ai::message::Message;

#[test]
fn tool_result_role_dispatches_to_tool_result() {
	let json = r#"{"role":"toolResult","toolCallId":"c1","toolName":"bash","content":[{"type":"text","text":"hi"}],"isError":false,"timestamp":1}"#;
	let message: Message = serde_json::from_str(json).expect("parse");
	assert!(matches!(message, Message::ToolResult(_)), "got {message:?}");
}

#[test]
fn four_roles_round_trip_to_matching_variants() {
	let cases = [
		(r#"{"role":"user","content":"hi","timestamp":1}"#, "user"),
		(r#"{"role":"developer","content":"hi","timestamp":1}"#, "developer"),
		(
			r#"{"role":"assistant","content":[],"api":"anthropic-messages","provider":"anthropic","model":"m","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"stop","timestamp":1}"#,
			"assistant",
		),
		(
			r#"{"role":"toolResult","toolCallId":"c","toolName":"t","content":[],"isError":false,"timestamp":1}"#,
			"toolResult",
		),
	];
	for (json, role) in cases {
		let message: Message = serde_json::from_str(json).expect(role);
		let variant_ok = matches!(
			(&message, role),
			(Message::User(_), "user")
				| (Message::Developer(_), "developer")
				| (Message::Assistant(_), "assistant")
				| (Message::ToolResult(_), "toolResult")
		);
		assert!(variant_ok, "role {role} mapped to {message:?}");
		// 序列化回程仍带同一 role tag（Serialize 路径未动）。
		let back = serde_json::to_value(&message).expect("serialize");
		assert_eq!(back.get("role").and_then(|v| v.as_str()), Some(role));
	}
}

#[test]
fn unknown_role_is_an_error() {
	let json = r#"{"role":"bashExecution","content":"x","timestamp":1}"#;
	assert!(serde_json::from_str::<Message>(json).is_err());
}

#[test]
fn missing_role_is_an_error() {
	let json = r#"{"content":"x","timestamp":1}"#;
	assert!(serde_json::from_str::<Message>(json).is_err());
}
