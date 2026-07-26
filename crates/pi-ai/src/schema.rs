//! Anthropic tool `input_schema` normalization.
//!
//! 1:1 Rust port of `normalizeAnthropicToolSchema` from
//! `packages/ai/src/providers/anthropic.ts` (`:3906` node logic, `:4005` entry)
//! plus the `spillToDescription` helper from
//! `packages/ai/src/utils/schema/spill.ts:18` (default `"spill"` format).
//!
//! Tracks the Anthropic Python SDK's
//! `lib/_parse/_transform.py::transform_schema`: only structural/metadata
//! keywords Anthropic's Messages validator honors are kept; everything else is
//! demoted ("spilled") into the node's `description` as `\n\n{key: value, ...}`
//! so the model still sees the constraint as a natural-language hint.
//!
//! Deviation from the TS source (documented, behavior-preserving): the TS
//! implementation threads a `WeakMap` cache keyed by object identity to dedup
//! shared sub-schemas and guard against cyclic references. The Rust entry point
//! takes a `&serde_json::Value`, which is always a finite tree (JSON has no
//! shared references or cycles), so the cache is a no-op here and is omitted.

use serde_json::{Map, Value};

/// Keys preserved on every node (`anthropic.ts:3790`).
const UNIVERSAL_KEEP: &[&str] = &[
	"$ref",
	"$defs",
	"$schema",
	"definitions",
	"type",
	"anyOf",
	"allOf",
	"enum",
	"const",
	"description",
	"title",
	"default",
	"nullable",
];

/// Keys preserved additively on `type: "object"` nodes (`anthropic.ts:3806`).
const OBJECT_KEEP: &[&str] = &["properties", "required", "additionalProperties"];

/// Keys preserved additively on `type: "array"` nodes; `minItems` only when its
/// value is 0 or 1 (`anthropic.ts:3808`).
const ARRAY_KEEP: &[&str] = &["items", "prefixItems", "minItems"];

/// Keys preserved additively on `type: "string"` nodes; `format` only when in
/// [`STRING_FORMATS`] (`anthropic.ts:3810`).
const STRING_KEEP: &[&str] = &["format"];

/// String `format` values Anthropic accepts (`anthropic.ts:3816`).
const STRING_FORMATS: &[&str] =
	&["date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"];

/// Combinator keys (`packages/ai/src/utils/schema/fields.ts:195`). Root
/// `anyOf`/`allOf` are spilled but kept when nested; `oneOf` is spilled at
/// every position.
const COMBINATOR_KEYS: &[&str] = &["anyOf", "allOf", "oneOf"];

/// Pick the principal non-null scalar type from a `type` keyword
/// (`anthropic.ts:3855`). `type` may be a single string or an array (e.g.
/// `["number", "null"]`); `"null"` is ignored so nullable variants normalize as
/// their underlying type.
fn pick_scalar_type(ty: Option<&Value>) -> Option<&str> {
	match ty {
		Some(Value::String(s)) => Some(s.as_str()),
		Some(Value::Array(entries)) => entries.iter().find_map(|entry| match entry {
			Value::String(s) if s != "null" => Some(s.as_str()),
			_ => None,
		}),
		_ => None,
	}
}

/// `pickAnthropicEffectiveScalarType` (`anthropic.ts:3865`).
fn pick_effective_scalar_type(schema: &Map<String, Value>) -> Option<&str> {
	if let Some(explicit) = pick_scalar_type(schema.get("type")) {
		return Some(explicit);
	}
	if schema.get("properties").is_some_and(Value::is_object) {
		return Some("object");
	}
	if schema.contains_key("items") || schema.get("prefixItems").is_some_and(Value::is_array) {
		return Some("array");
	}
	None
}

/// `anthropicPerTypeKeep` (`anthropic.ts:3873`).
fn per_type_keep(scalar_type: Option<&str>) -> Option<&'static [&'static str]> {
	match scalar_type {
		Some("object") => Some(OBJECT_KEEP),
		Some("array") => Some(ARRAY_KEEP),
		Some("string") => Some(STRING_KEEP),
		_ => None,
	}
}

/// Demote stripped keywords into `node.description` as `\n\n{key: value, ...}`
/// (`spill.ts:18`, default `"spill"` format). Values are `JSON.stringify`d,
/// which `serde_json::to_string` reproduces byte-for-byte for tree JSON.
fn spill_to_description(node: &mut Map<String, Value>, spill: &[(String, Value)]) {
	// The TS filter drops `undefined` values; serde_json values are never
	// `undefined` (JSON `null` is kept and stringifies as `"null"`), so every
	// entry survives.
	if spill.is_empty() {
		return;
	}

	let existing = match node.get("description") {
		Some(Value::String(s)) => s.clone(),
		_ => String::new(),
	};

	let body = spill
		.iter()
		.map(|(key, value)| {
			let json = serde_json::to_string(value).unwrap_or_else(|_| "null".to_owned());
			format!("{key}: {json}")
		})
		.collect::<Vec<_>>()
		.join(", ");
	let formatted = format!("{{{body}}}");

	let description = if existing.is_empty() {
		formatted
	} else {
		format!("{existing}\n\n{formatted}")
	};
	node.insert("description".to_owned(), Value::String(description));
}

/// `normalizeAnthropicToolSchemaNode` (`anthropic.ts:3906`).
fn normalize_node(schema: &Value, is_root: bool) -> Value {
	match schema {
		Value::Array(entries) => {
			// TS recurses with the default `isRoot = false`.
			Value::Array(
				entries
					.iter()
					.map(|entry| normalize_node(entry, false))
					.collect(),
			)
		},
		Value::Object(schema) => Value::Object(normalize_object(schema, is_root)),
		other => other.clone(),
	}
}

fn normalize_object(schema: &Map<String, Value>, is_root: bool) -> Map<String, Value> {
	let mut result = Map::new();
	let scalar_type = pick_effective_scalar_type(schema);
	let keep = per_type_keep(scalar_type);
	let mut spill: Vec<(String, Value)> = Vec::new();

	// Insertion order is preserved (serde_json `preserve_order` → IndexMap),
	// mirroring JS `for..in` over string keys.
	for (key, value) in schema {
		let is_root_combinator = is_root && COMBINATOR_KEYS.contains(&key.as_str());
		let kept = !is_root_combinator
			&& (UNIVERSAL_KEEP.contains(&key.as_str())
				|| keep.is_some_and(|set| set.contains(&key.as_str())));
		if kept {
			result.insert(key.clone(), value.clone());
		} else {
			spill.push((key.clone(), value.clone()));
		}
	}

	// Per-type conditional prune within the kept set.
	if scalar_type == Some("string")
		&& let Some(Value::String(format)) = result.get("format")
		&& !STRING_FORMATS.contains(&format.as_str())
	{
		let format = result.remove("format").expect("format present");
		spill.push(("format".to_owned(), format));
	}
	if scalar_type == Some("array")
		&& let Some(min_items) = result.get("minItems")
		&& !is_zero_or_one(min_items)
	{
		let min_items = result.remove("minItems").expect("minItems present");
		spill.push(("minItems".to_owned(), min_items));
	}
	if scalar_type == Some("object") && !result.contains_key("additionalProperties") {
		result.insert("additionalProperties".to_owned(), Value::Bool(false));
	}

	// Recurse on structural keys (mutating in place to preserve key positions).
	if let Some(Value::Object(props)) = result.get("properties") {
		let normalized = props
			.iter()
			.map(|(name, node)| (name.clone(), normalize_node(node, false)))
			.collect();
		result.insert("properties".to_owned(), Value::Object(normalized));
	}
	if let Some(additional @ Value::Object(_)) = result.get("additionalProperties") {
		let normalized = normalize_node(additional, false);
		let collapsed = matches!(&normalized, Value::Object(map) if map.is_empty());
		result.insert(
			"additionalProperties".to_owned(),
			if collapsed {
				Value::Bool(true)
			} else {
				normalized
			},
		);
	}
	match result.get("items") {
		Some(Value::Array(items)) => {
			let normalized = items
				.iter()
				.map(|item| normalize_node(item, false))
				.collect();
			result.insert("items".to_owned(), Value::Array(normalized));
		},
		Some(items @ Value::Object(_)) => {
			let normalized = normalize_node(items, false);
			result.insert("items".to_owned(), normalized);
		},
		_ => {},
	}
	if let Some(Value::Array(prefix_items)) = result.get("prefixItems") {
		let normalized = prefix_items
			.iter()
			.map(|item| normalize_node(item, false))
			.collect();
		result.insert("prefixItems".to_owned(), Value::Array(normalized));
	}
	for combinator in COMBINATOR_KEYS {
		if let Some(Value::Array(variants)) = result.get(*combinator) {
			let normalized = variants
				.iter()
				.map(|variant| normalize_node(variant, false))
				.collect();
			result.insert((*combinator).to_owned(), Value::Array(normalized));
		}
	}
	for defs_key in ["$defs", "definitions"] {
		if let Some(Value::Object(defs)) = result.get(defs_key) {
			let normalized = defs
				.iter()
				.map(|(name, node)| (name.clone(), normalize_node(node, false)))
				.collect();
			result.insert(defs_key.to_owned(), Value::Object(normalized));
		}
	}

	spill_to_description(&mut result, &spill);
	result
}

/// JS `typeof v === "number" && (v === 0 || v === 1)`.
fn is_zero_or_one(value: &Value) -> bool {
	value.as_f64().is_some_and(|n| n == 0.0 || n == 1.0)
}

/// Normalize a JSON Schema for use as an Anthropic tool `input_schema`
/// (`anthropic.ts:4005`, `isRoot = true`).
#[must_use]
pub fn normalize_anthropic_tool_schema(schema: &Value) -> Value {
	normalize_node(schema, true)
}

#[cfg(test)]
mod tests {
	//! Ported 1:1 from `packages/ai/test/anthropic-tool-schema.test.ts`. Test
	//! names carry the TS `describe`/`it` intent. Two TS cases are intentionally
	//! omitted because they exercise JS object-identity semantics that a
	//! `serde_json::Value` tree cannot express: `resolves self-referential
	//! schemas` (:409, cyclic references — impossible in tree JSON) and the
	//! memoization-slot half of `does not mutate` (:388, which we cover by
	//! taking `&Value` and cloning). Beyond the ports, extra edge cases cover
	//! root-combinator / open-map / format-downgrade / minItems=2 / recursive
	//! `$defs` / spill-description append formatting.

	use serde_json::json;

	use super::normalize_anthropic_tool_schema as norm;

	// ─── SDK whitelist: number / integer ────────────────────────────────────

	#[test]
	fn demotes_range_and_multiple_of_on_number_nodes() {
		let out = norm(&json!({
			"type": "object",
			"properties": { "temperature": {
				"type": "number", "minimum": 0, "maximum": 1,
				"exclusiveMinimum": 0, "exclusiveMaximum": 1, "multipleOf": 0.1,
			} },
		}));
		assert_eq!(
			out["properties"]["temperature"],
			json!({
				"type": "number",
				"description": "{minimum: 0, maximum: 1, exclusiveMinimum: 0, exclusiveMaximum: 1, multipleOf: 0.1}",
			})
		);
	}

	#[test]
	fn demotes_range_on_integer_nodes() {
		let out = norm(&json!({
			"type": "object",
			"properties": { "count": { "type": "integer", "minimum": 0, "maximum": 100, "multipleOf": 1 } },
		}));
		assert_eq!(
			out["properties"]["count"],
			json!({ "type": "integer", "description": "{minimum: 0, maximum: 100, multipleOf: 1}" })
		);
	}

	#[test]
	fn demotes_numeric_range_on_union_type_including_number() {
		let out = norm(&json!({
			"type": "object",
			"properties": { "value": { "type": ["number", "null"], "minimum": 0, "maximum": 10 } },
		}));
		assert_eq!(
			out["properties"]["value"],
			json!({ "type": ["number", "null"], "description": "{minimum: 0, maximum: 10}" })
		);
	}

	// ─── SDK whitelist: string ──────────────────────────────────────────────

	#[test]
	fn demotes_pattern_min_max_length_into_description() {
		let out = norm(&json!({
			"type": "object",
			"properties": { "name": { "type": "string", "pattern": "^[a-z]+$", "minLength": 1, "maxLength": 32 } },
		}));
		assert_eq!(
			out["properties"]["name"],
			json!({ "type": "string", "description": r#"{pattern: "^[a-z]+$", minLength: 1, maxLength: 32}"# })
		);
	}

	#[test]
	fn keeps_format_only_when_in_supported_set() {
		let out = norm(&json!({
			"type": "object",
			"properties": {
				"email": { "type": "string", "format": "email" },
				"weird": { "type": "string", "format": "color-hex" },
			},
		}));
		assert_eq!(out["properties"]["email"], json!({ "type": "string", "format": "email" }));
		assert_eq!(
			out["properties"]["weird"],
			json!({ "type": "string", "description": r#"{format: "color-hex"}"# })
		);
	}

	// ─── SDK whitelist: array ───────────────────────────────────────────────

	#[test]
	fn keeps_min_items_only_when_zero_or_one() {
		let out01 = norm(&json!({ "type": "array", "items": { "type": "string" }, "minItems": 1 }));
		assert_eq!(out01["minItems"], json!(1));
		assert!(out01.get("description").is_none());

		let out5 = norm(&json!({
			"type": "array", "items": { "type": "string" },
			"minItems": 5, "maxItems": 10, "uniqueItems": true,
		}));
		assert!(out5.get("minItems").is_none());
		assert!(out5.get("maxItems").is_none());
		assert!(out5.get("uniqueItems").is_none());
		assert_eq!(out5["description"], json!("{maxItems: 10, uniqueItems: true, minItems: 5}"));
	}

	#[test]
	fn recurses_into_items_and_prefix_items() {
		let out = norm(&json!({
			"type": "array",
			"items": { "type": "number", "minimum": 0 },
			"prefixItems": [{ "type": "string", "minLength": 1 }],
		}));
		assert_eq!(out["items"], json!({ "type": "number", "description": "{minimum: 0}" }));
		assert_eq!(
			out["prefixItems"],
			json!([{ "type": "string", "description": "{minLength: 1}" }])
		);
	}

	// ─── SDK whitelist: object ──────────────────────────────────────────────

	#[test]
	fn defaults_additional_properties_false_on_closed_objects() {
		let out = norm(&json!({ "type": "object", "properties": { "a": { "type": "string" } } }));
		assert_eq!(out["additionalProperties"], json!(false));
	}

	#[test]
	fn preserves_explicit_open_map_true() {
		let out = norm(&json!({
			"type": "object", "additionalProperties": true, "properties": { "a": { "type": "string" } },
		}));
		assert_eq!(out["additionalProperties"], json!(true));
	}

	#[test]
	fn preserves_and_recurses_additional_properties_schema() {
		let out = norm(
			&json!({ "type": "object", "additionalProperties": { "type": "number", "minimum": 0 } }),
		);
		assert_eq!(
			out["additionalProperties"],
			json!({ "type": "number", "description": "{minimum: 0}" })
		);
	}

	#[test]
	fn open_map_empty_schema_collapses_to_true() {
		// Zod's `z.record(z.string(), z.unknown())` produces `{}`.
		let out = norm(&json!({ "type": "object", "additionalProperties": {} }));
		assert_eq!(out["additionalProperties"], json!(true));
	}

	#[test]
	fn demotes_pattern_properties_property_names_min_items_on_objects() {
		let out = norm(&json!({
			"type": "object",
			"properties": { "tag": { "type": "string" } },
			"patternProperties": { "^x-": { "type": "string" } },
			"propertyNames": { "pattern": "^[a-z]+$" },
			"minItems": 1,
		}));
		assert!(out.get("patternProperties").is_none());
		assert!(out.get("propertyNames").is_none());
		assert!(out.get("minItems").is_none());
		let desc = out["description"].as_str().expect("description string");
		assert!(desc.contains("patternProperties"));
		assert!(desc.contains("propertyNames"));
		assert!(desc.contains("minItems"));
	}

	// ─── Universal preservation ─────────────────────────────────────────────

	#[test]
	fn appends_spilled_keywords_to_existing_description_with_blank_line() {
		let out = norm(&json!({
			"type": "object",
			"properties": { "ratio": { "type": "number", "description": "A ratio", "minimum": 0, "maximum": 1 } },
		}));
		assert_eq!(
			out["properties"]["ratio"],
			json!({ "type": "number", "description": "A ratio\n\n{minimum: 0, maximum: 1}" })
		);
	}

	#[test]
	fn preserves_universal_keys() {
		let out = norm(&json!({
			"$defs": { "Color": { "type": "string", "enum": ["r", "g", "b"] } },
			"type": "object",
			"title": "Sample",
			"properties": {
				"ref": { "$ref": "#/$defs/Color" },
				"union": { "anyOf": [{ "type": "string" }, { "type": "number" }] },
				"choice": { "const": "x" },
				"hint": { "type": "string", "default": "anon" },
			},
		}));
		assert_eq!(out["title"], json!("Sample"));
		assert_eq!(out["$defs"], json!({ "Color": { "type": "string", "enum": ["r", "g", "b"] } }));
		assert_eq!(out["properties"]["ref"], json!({ "$ref": "#/$defs/Color" }));
		assert_eq!(
			out["properties"]["union"]["anyOf"],
			json!([{ "type": "string" }, { "type": "number" }])
		);
		assert_eq!(out["properties"]["choice"], json!({ "const": "x" }));
		assert_eq!(out["properties"]["hint"], json!({ "type": "string", "default": "anon" }));
	}

	// ─── Parity with anthropic-sdk-python transform_schema ──────────────────

	#[test]
	fn preserves_lone_ref_node() {
		assert_eq!(
			norm(&json!({ "$ref": "#/components/schemas/SomeSchema" })),
			json!({ "$ref": "#/components/schemas/SomeSchema" })
		);
	}

	#[test]
	fn recurses_nested_anyof_variants_and_spills_per_variant() {
		let out = norm(&json!({
			"type": "object",
			"properties": { "value": { "anyOf": [{ "type": "string" }, { "type": "integer", "minimum": 1 }] } },
		}));
		assert_eq!(
			out,
			json!({
				"type": "object",
				"properties": {
					"value": { "anyOf": [{ "type": "string" }, { "type": "integer", "description": "{minimum: 1}" }] },
				},
				"additionalProperties": false,
			})
		);
	}

	#[test]
	fn keeps_enum_on_string_verbatim() {
		let out = norm(&json!({ "type": "string", "enum": ["foo", "bar"] }));
		assert_eq!(out, json!({ "type": "string", "enum": ["foo", "bar"] }));
	}

	#[test]
	fn spills_top_level_anyof_into_description() {
		let out = norm(&json!({
			"type": "object",
			"properties": { "id": { "type": "string" } },
			"anyOf": [{ "required": ["id"] }, { "required": ["name"] }],
		}));
		assert_eq!(
			out,
			json!({
				"type": "object",
				"properties": { "id": { "type": "string" } },
				"additionalProperties": false,
				"description": r#"{anyOf: [{"required":["id"]},{"required":["name"]}]}"#,
			})
		);
	}

	#[test]
	fn spills_top_level_allof_into_description() {
		let out = norm(&json!({
			"type": "object",
			"properties": { "id": { "type": "string" } },
			"allOf": [
				{ "type": "object", "properties": { "name": { "type": "string" } } },
				{ "type": "object", "properties": { "age": { "type": "integer", "minimum": 0 } } },
			],
		}));
		assert_eq!(
			out,
			json!({
				"type": "object",
				"properties": { "id": { "type": "string" } },
				"additionalProperties": false,
				"description":
					r#"{allOf: [{"type":"object","properties":{"name":{"type":"string"}}},{"type":"object","properties":{"age":{"type":"integer","minimum":0}}}]}"#,
			})
		);
	}

	#[test]
	fn spills_oneof_at_root_and_nested() {
		let root =
			norm(&json!({ "oneOf": [{ "type": "string" }, { "type": "integer", "minimum": 1 }] }));
		assert_eq!(
			root,
			json!({ "description": r#"{oneOf: [{"type":"string"},{"type":"integer","minimum":1}]}"# })
		);

		let nested = norm(&json!({
			"type": "object",
			"properties": { "value": { "oneOf": [{ "type": "string" }, { "type": "integer", "minimum": 1 }] } },
		}));
		assert_eq!(
			nested,
			json!({
				"type": "object",
				"properties": {
					"value": { "description": r#"{oneOf: [{"type":"string"},{"type":"integer","minimum":1}]}"# },
				},
				"additionalProperties": false,
			})
		);
	}

	#[test]
	fn preserves_object_required_and_spills_per_property() {
		let out = norm(&json!({
			"type": "object",
			"properties": {
				"name": { "type": "string", "default": "John" },
				"age": { "type": "integer", "minimum": 0 },
			},
			"required": ["name"],
			"description": "Person object",
		}));
		assert_eq!(
			out,
			json!({
				"type": "object",
				"description": "Person object",
				"properties": {
					"name": { "type": "string", "default": "John" },
					"age": { "type": "integer", "description": "{minimum: 0}" },
				},
				"additionalProperties": false,
				"required": ["name"],
			})
		);
	}

	#[test]
	fn spills_min_items_gt_one_with_two_newline_preamble() {
		let out = norm(&json!({
			"type": "array", "items": { "type": "string" }, "minItems": 2, "description": "A list of strings",
		}));
		assert_eq!(
			out,
			json!({
				"type": "array",
				"description": "A list of strings\n\n{minItems: 2}",
				"items": { "type": "string" },
			})
		);
	}

	#[test]
	fn keeps_allowlisted_format_alongside_preserved_default() {
		let out = norm(&json!({
			"type": "string", "format": "email", "default": "user@example.com", "description": "User email",
		}));
		assert_eq!(
			out,
			json!({
				"type": "string",
				"description": "User email",
				"format": "email",
				"default": "user@example.com",
			})
		);
	}

	#[test]
	fn passes_bare_string_node_through_unchanged() {
		assert_eq!(norm(&json!({ "type": "string" })), json!({ "type": "string" }));
	}

	#[test]
	fn spills_integer_keywords_in_source_order() {
		let out = norm(&json!({
			"type": "integer", "minimum": 1, "maximum": 10,
			"exclusiveMinimum": 0, "exclusiveMaximum": 20, "description": "A number",
		}));
		assert_eq!(
			out,
			json!({
				"type": "integer",
				"description": "A number\n\n{minimum: 1, maximum: 10, exclusiveMinimum: 0, exclusiveMaximum: 20}",
			})
		);
	}

	#[test]
	fn passes_boolean_with_description_through_unchanged() {
		let out = norm(&json!({ "type": "boolean", "description": "A flag" }));
		assert_eq!(out, json!({ "type": "boolean", "description": "A flag" }));
	}

	#[test]
	fn passes_null_type_node_through_unchanged() {
		assert_eq!(norm(&json!({ "type": "null" })), json!({ "type": "null" }));
	}

	#[test]
	fn does_not_mutate_input_schema() {
		let original = json!({
			"type": "object",
			"properties": {
				"name": { "type": "string", "default": "John" },
				"age": { "type": "integer", "minimum": 0 },
			},
			"required": ["name"],
			"description": "Person object",
			"additionalProperties": true,
		});
		let snapshot = original.clone();
		let _ = norm(&original);
		assert_eq!(original, snapshot);
	}

	// ─── Extra edge coverage ────────────────────────────────────────────────

	#[test]
	fn recurses_into_defs_and_definitions() {
		let out = norm(&json!({
			"type": "object",
			"$defs": { "Age": { "type": "integer", "minimum": 0 } },
			"definitions": { "Name": { "type": "string", "minLength": 1 } },
			"properties": { "who": { "$ref": "#/$defs/Age" } },
		}));
		assert_eq!(out["$defs"]["Age"], json!({ "type": "integer", "description": "{minimum: 0}" }));
		assert_eq!(
			out["definitions"]["Name"],
			json!({ "type": "string", "description": "{minLength: 1}" })
		);
	}
}
