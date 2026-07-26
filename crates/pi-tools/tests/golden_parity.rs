//! Golden parity: run the Rust six-tool minimal face with the exact params the
//! TS tools were driven with (`scripts/gen-tool-goldens.ts`) and compare
//! content blocks byte-for-byte against `tests/goldens/*.json`.
//!
//! Determinism mirrors the generator: the fixture tree is copied to a scratch
//! dir per case group, every path gets a fixed mtime (base 1700000000 + 60s
//! per sorted relative path), scratch roots normalize to `«WS»`, and
//! wall-time notices to `Wall time: «WT» seconds`.

use std::{
	path::{Path, PathBuf},
	time::{Duration, SystemTime},
};

use pi_shell::cancel::CancelToken;
use pi_tools::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, Tool, WriteTool};
use serde_json::Value;

const MTIME_BASE_SEC: u64 = 1_700_000_000;
const MTIME_STEP_SEC: u64 = 60;

fn goldens_dir() -> PathBuf {
	Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens")
}

fn fixture_src() -> PathBuf {
	Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixture-ws")
}

fn copy_tree(src: &Path, dst: &Path) {
	std::fs::create_dir_all(dst).expect("create dst dir");
	for entry in std::fs::read_dir(src).expect("read src dir") {
		let entry = entry.expect("dir entry");
		let target = dst.join(entry.file_name());
		if entry.file_type().expect("file type").is_dir() {
			copy_tree(&entry.path(), &target);
		} else {
			std::fs::copy(entry.path(), &target).expect("copy file");
		}
	}
}

fn collect_rel_paths(root: &Path) -> Vec<String> {
	fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
		for entry in std::fs::read_dir(dir).expect("read dir") {
			let entry = entry.expect("dir entry");
			let path = entry.path();
			let rel = path
				.strip_prefix(root)
				.expect("under root")
				.to_string_lossy()
				.replace('\\', "/");
			out.push(rel);
			if entry.file_type().expect("file type").is_dir() {
				walk(root, &path, out);
			}
		}
	}
	let mut out = Vec::new();
	walk(root, root, &mut out);
	out.sort();
	out
}

/// Same fixed-mtime rule as `applyFixedMtimes` in the generator.
fn apply_fixed_mtimes(root: &Path) {
	for (index, rel) in collect_rel_paths(root).iter().enumerate() {
		let t = SystemTime::UNIX_EPOCH
			+ Duration::from_secs(MTIME_BASE_SEC + index as u64 * MTIME_STEP_SEC);
		let file = std::fs::File::open(root.join(rel)).expect("open for utimes");
		file.set_modified(t).expect("set mtime");
	}
}

struct ScratchWs {
	root: PathBuf,
}

impl ScratchWs {
	fn new(tag: &str) -> Self {
		let base = std::env::temp_dir()
			.canonicalize()
			.expect("canonicalize temp dir");
		let root = base.join(format!("pi-tools-parity-{tag}-{}", std::process::id()));
		if root.exists() {
			std::fs::remove_dir_all(&root).expect("clear stale scratch");
		}
		copy_tree(&fixture_src(), &root);
		apply_fixed_mtimes(&root);
		Self { root }
	}
}

impl Drop for ScratchWs {
	fn drop(&mut self) {
		let _ = std::fs::remove_dir_all(&self.root);
	}
}

fn normalize(text: &str, ws: &Path) -> String {
	static WALL_TIME_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
		regex::Regex::new(r"Wall time: \d+\.\d+ seconds").expect("WALL_TIME_RE")
	});
	let replaced = text.replace(&ws.to_string_lossy().into_owned(), "«WS»");
	WALL_TIME_RE
		.replace_all(&replaced, "Wall time: «WT» seconds")
		.into_owned()
}

/// Substitute `«WS»` placeholders back into params (none are used today, but
/// keep the goldens future-proof).
fn denormalize_params(params: &Value, ws: &Path) -> Value {
	match params {
		Value::String(s) => Value::String(s.replace("«WS»", &ws.to_string_lossy())),
		Value::Array(items) => {
			Value::Array(items.iter().map(|v| denormalize_params(v, ws)).collect())
		},
		Value::Object(map) => Value::Object(
			map.iter()
				.map(|(k, v)| (k.clone(), denormalize_params(v, ws)))
				.collect(),
		),
		other => other.clone(),
	}
}

struct CaseOutcome {
	content:  Option<Vec<String>>,
	is_error: bool,
	useless:  bool,
	threw:    Option<String>,
}

async fn run_tool(tool: &(impl Tool + ?Sized), case_name: &str, params: Value) -> CaseOutcome {
	let ct = CancelToken::default();
	match tool
		.execute(&format!("golden-{case_name}"), params, &ct)
		.await
	{
		Ok(result) => CaseOutcome {
			content:  Some(
				result
					.content
					.iter()
					.map(|block| match block {
						pi_ai::UserContentBlock::Text(text) => text.text.clone(),
						other => panic!("unexpected non-text block: {other:?}"),
					})
					.collect(),
			),
			is_error: result.is_error,
			useless:  result.useless,
			threw:    None,
		},
		Err(err) => CaseOutcome {
			content:  None,
			is_error: false,
			useless:  false,
			threw:    Some(err.to_string()),
		},
	}
}

fn assert_case(golden: &Value, outcome: &CaseOutcome, ws: &Path, name: &str) {
	let golden_threw = golden.get("threw").and_then(Value::as_str);
	let actual_threw = outcome.threw.as_ref().map(|t| normalize(t, ws));
	assert_eq!(golden_threw, actual_threw.as_deref(), "{name}: threw mismatch (golden vs rust)");

	if golden_threw.is_none() {
		let golden_content: Vec<&str> = golden
			.get("content")
			.and_then(Value::as_array)
			.expect("golden content")
			.iter()
			.map(|block| {
				block
					.get("text")
					.and_then(Value::as_str)
					.expect("text block")
			})
			.collect();
		let actual_content: Vec<String> = outcome
			.content
			.as_ref()
			.expect("rust content")
			.iter()
			.map(|text| normalize(text, ws))
			.collect();
		assert_eq!(
			golden_content,
			actual_content
				.iter()
				.map(String::as_str)
				.collect::<Vec<_>>(),
			"{name}: content mismatch"
		);
		assert_eq!(
			golden
				.get("isError")
				.and_then(Value::as_bool)
				.unwrap_or(false),
			outcome.is_error,
			"{name}: isError mismatch"
		);
		assert_eq!(
			golden
				.get("useless")
				.and_then(Value::as_bool)
				.unwrap_or(false),
			outcome.useless,
			"{name}: useless mismatch"
		);
	}

	if let Some(post_files) = golden.get("postFiles").and_then(Value::as_object) {
		for (rel, expected) in post_files {
			let expected = expected.as_str().expect("postFile string");
			let actual =
				std::fs::read_to_string(ws.join(rel)).unwrap_or_else(|_| "<missing>".to_owned());
			assert_eq!(expected, actual, "{name}: postFile {rel} mismatch");
		}
	}
}

fn load_golden(name: &str) -> Value {
	let path = goldens_dir().join(format!("{name}.json"));
	let text = std::fs::read_to_string(&path)
		.unwrap_or_else(|err| panic!("read golden {}: {err}", path.display()));
	serde_json::from_str(&text).expect("parse golden")
}

fn case_goldens() -> Vec<Value> {
	let mut cases = Vec::new();
	for entry in std::fs::read_dir(goldens_dir()).expect("read goldens dir") {
		let entry = entry.expect("golden entry");
		let name = entry.file_name().to_string_lossy().into_owned();
		#[allow(
			clippy::case_sensitive_file_extension_comparisons,
			reason = "goldens are generated with a lowercase .json suffix"
		)]
		if !name.ends_with(".json") || name.starts_with("schema-") {
			continue;
		}
		let golden: Value =
			serde_json::from_str(&std::fs::read_to_string(entry.path()).expect("read golden"))
				.expect("parse golden");
		cases.push(golden);
	}
	cases
}

enum AnyTool {
	Read(ReadTool),
	Write(WriteTool),
	Edit(EditTool),
	Bash(BashTool),
	Grep(GrepTool),
	Glob(GlobTool),
}

impl AnyTool {
	fn for_name(tool: &str, ws: &Path) -> Self {
		match tool {
			"read" => Self::Read(ReadTool::new(ws)),
			"write" => Self::Write(WriteTool::new(ws)),
			"edit" => Self::Edit(EditTool::new(ws)),
			"bash" => Self::Bash(BashTool::new(ws)),
			"grep" => Self::Grep(GrepTool::new(ws)),
			"glob" => Self::Glob(GlobTool::new(ws)),
			other => panic!("unknown tool {other}"),
		}
	}

	async fn run(&self, case_name: &str, params: Value) -> CaseOutcome {
		match self {
			Self::Read(tool) => run_tool(tool, case_name, params).await,
			Self::Write(tool) => run_tool(tool, case_name, params).await,
			Self::Edit(tool) => run_tool(tool, case_name, params).await,
			Self::Bash(tool) => run_tool(tool, case_name, params).await,
			Self::Grep(tool) => run_tool(tool, case_name, params).await,
			Self::Glob(tool) => run_tool(tool, case_name, params).await,
		}
	}

	fn as_tool(&self) -> &dyn ToolFacts {
		match self {
			Self::Read(tool) => tool,
			Self::Write(tool) => tool,
			Self::Edit(tool) => tool,
			Self::Bash(tool) => tool,
			Self::Grep(tool) => tool,
			Self::Glob(tool) => tool,
		}
	}
}

/// Object-safe subset of [`Tool`] for schema checks.
trait ToolFacts {
	fn schema(&self) -> Value;
}

impl<T: Tool> ToolFacts for T {
	fn schema(&self) -> Value {
		self.input_schema()
	}
}

#[tokio::test]
async fn golden_parity_all_cases() {
	let cases = case_goldens();
	assert!(cases.len() >= 24, "expected >= 24 goldens, found {}", cases.len());

	// Mutating tools get a fresh scratch per case; read-only tools share one.
	let shared = ScratchWs::new("shared");
	let mut ran = 0usize;
	for golden in &cases {
		let tool_name = golden
			.get("tool")
			.and_then(Value::as_str)
			.expect("tool name");
		let case_name = golden
			.get("case")
			.and_then(Value::as_str)
			.expect("case name");
		let full_name = format!("{tool_name}-{case_name}");
		let fresh;
		let ws: &Path = if matches!(tool_name, "edit" | "write") {
			fresh = ScratchWs::new(&full_name);
			&fresh.root
		} else {
			&shared.root
		};
		let params = denormalize_params(golden.get("params").expect("params"), ws);
		let tool = AnyTool::for_name(tool_name, ws);
		let outcome = tool.run(case_name, params).await;
		assert_case(golden, &outcome, ws, &full_name);
		ran += 1;
	}
	println!("golden parity: {ran} cases passed");
}

#[tokio::test]
async fn schema_parity_all_tools() {
	let shared = ScratchWs::new("schema");
	for name in ["read", "write", "edit", "bash", "grep", "glob"] {
		let golden = load_golden(&format!("schema-{name}"));
		let tool = AnyTool::for_name(name, &shared.root);
		let wire = tool.as_tool().schema();
		assert_eq!(
			golden.get("wireSchema").expect("wireSchema"),
			&wire,
			"schema-{name}: wireSchema mismatch"
		);
		let normalized = pi_ai::normalize_anthropic_tool_schema(&wire);
		assert_eq!(
			golden.get("normalized").expect("normalized"),
			&normalized,
			"schema-{name}: normalized mismatch"
		);
	}
}
