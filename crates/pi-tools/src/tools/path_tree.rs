//! Grouped path/file output, ported from `packages/utils/src/path-tree.ts`
//! (`buildPathTree` / `walkPathTree` / `formatGroupedPaths`) and
//! `packages/coding-agent/src/tools/grouped-file-output.ts`
//! (`formatGroupedFiles`, model lines only — the display-lines twin is a TUI
//! concern and is not ported).

use std::collections::HashMap;

/// `URL_LIKE_PATH_RE` (path-tree.ts:13).
fn is_url_like_path(file_path: &str) -> bool {
	let Some(idx) = file_path.find("://") else {
		return false;
	};
	let scheme = &file_path[..idx];
	let mut chars = scheme.chars();
	match chars.next() {
		Some(c) if c.is_ascii_alphabetic() => {},
		_ => return false,
	}
	chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '.' | '-'))
}

#[derive(Default)]
struct PathTreeNode {
	/// (name, key) leaves in first-seen order.
	files:      Vec<(String, String)>,
	file_names: std::collections::HashSet<String>,
	subdirs:    Vec<(String, usize)>,
	dir_index:  HashMap<String, usize>,
}

struct PathTree {
	nodes: Vec<PathTreeNode>,
}

/// One tree-walk event (`GroupedTreeEvent`).
pub enum GroupedTreeEvent {
	Dir { depth: usize, name: String },
	File { depth: usize, name: String, key: String },
}

/// `buildPathTree` (path-tree.ts:67).
fn build_path_tree(entries: &[(String, bool)]) -> PathTree {
	let mut tree = PathTree { nodes: vec![PathTreeNode::default()] };
	for (raw_path, is_dir) in entries {
		let normalized = raw_path.replace('\\', "/");
		let file_key = raw_path.clone();
		if is_url_like_path(&normalized) {
			let root = &mut tree.nodes[0];
			if root.file_names.insert(normalized.clone()) {
				root.files.push((normalized, file_key));
			}
			continue;
		}
		let trimmed = normalized.strip_suffix('/').unwrap_or(&normalized);
		if trimmed.is_empty() {
			continue;
		}
		let segments: Vec<&str> = trimmed.split('/').collect();
		let dir_count = if *is_dir {
			segments.len()
		} else {
			segments.len() - 1
		};
		let mut node = 0usize;
		for segment in &segments[..dir_count] {
			if let Some(&child) = tree.nodes[node].dir_index.get(*segment) {
				node = child;
			} else {
				let child = tree.nodes.len();
				tree.nodes.push(PathTreeNode::default());
				tree.nodes[node]
					.dir_index
					.insert((*segment).to_owned(), child);
				tree.nodes[node]
					.subdirs
					.push(((*segment).to_owned(), child));
				node = child;
			}
		}
		if !*is_dir {
			let name = (*segments.last().expect("non-empty segments")).to_owned();
			let target = &mut tree.nodes[node];
			if target.file_names.insert(name.clone()) {
				target.files.push((name, file_key));
			}
		}
	}
	tree
}

/// `walkPathTree` (path-tree.ts:104): DFS with single-child chain folding.
fn walk_path_tree(tree: &PathTree, node: usize, depth: usize, out: &mut Vec<GroupedTreeEvent>) {
	for (name, key) in &tree.nodes[node].files {
		out.push(GroupedTreeEvent::File { depth, name: name.clone(), key: key.clone() });
	}
	for (name, child) in &tree.nodes[node].subdirs {
		let mut dir_node = *child;
		let mut parts = vec![name.clone()];
		while tree.nodes[dir_node].files.is_empty() && tree.nodes[dir_node].subdirs.len() == 1 {
			let (only_name, only_child) = &tree.nodes[dir_node].subdirs[0];
			parts.push(only_name.clone());
			dir_node = *only_child;
		}
		out.push(GroupedTreeEvent::Dir { depth, name: parts.join("/") });
		walk_path_tree(tree, dir_node, depth + 1, out);
	}
}

fn tree_events(entries: &[(String, bool)]) -> Vec<GroupedTreeEvent> {
	let tree = build_path_tree(entries);
	let mut out = Vec::new();
	walk_path_tree(&tree, 0, 0, &mut out);
	out
}

/// `formatGroupedPaths` (path-tree.ts:135), no annotator.
#[must_use]
pub fn format_grouped_paths(paths: &[String]) -> String {
	if paths.is_empty() {
		return String::new();
	}
	let entries: Vec<(String, bool)> = paths
		.iter()
		.map(|entry| (entry.clone(), entry.ends_with('/')))
		.collect();
	let mut lines = Vec::new();
	for event in tree_events(&entries) {
		match event {
			GroupedTreeEvent::Dir { depth, name } => {
				lines.push(format!("{} {name}/", "#".repeat(depth + 1)));
			},
			GroupedTreeEvent::File { name, .. } => lines.push(name),
		}
	}
	lines.join("\n")
}

/// One file's contribution to grouped output (`GroupedFileSection`, model
/// side).
pub struct GroupedFileSection {
	pub header_suffix: String,
	pub model_lines:   Vec<String>,
	pub skip:          bool,
}

/// `formatGroupedFiles` (grouped-file-output.ts): multi-level prefix-folded
/// tree with per-file body lines; blank line before every depth-0 event and
/// every directory header (after the first emit).
#[must_use]
pub fn format_grouped_files(
	files: &[String],
	mut render_file: impl FnMut(&str) -> GroupedFileSection,
) -> Vec<String> {
	let mut sections: HashMap<String, GroupedFileSection> = HashMap::new();
	let mut inputs: Vec<(String, bool)> = Vec::new();
	for file_path in files {
		if sections.contains_key(file_path) {
			continue;
		}
		let section = render_file(file_path);
		if section.skip {
			continue;
		}
		sections.insert(file_path.clone(), section);
		inputs.push((file_path.clone(), false));
	}

	let mut model = Vec::new();
	let mut emitted = false;
	for event in tree_events(&inputs) {
		let (depth, is_dir) = match &event {
			GroupedTreeEvent::Dir { depth, .. } => (*depth, true),
			GroupedTreeEvent::File { depth, .. } => (*depth, false),
		};
		if emitted && (depth == 0 || is_dir) {
			model.push(String::new());
		}
		emitted = true;
		let hashes = "#".repeat(depth + 1);
		match event {
			GroupedTreeEvent::Dir { name, .. } => model.push(format!("{hashes} {name}/")),
			GroupedTreeEvent::File { name, key, .. } => {
				let section = sections.get(&key).expect("section recorded for key");
				model.push(format!("{hashes} {name}{}", section.header_suffix));
				model.extend(section.model_lines.iter().cloned());
			},
		}
	}
	model
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn grouped_paths_fold_single_child_chains() {
		let paths = vec![
			"src/util/helper.txt".to_owned(),
			"src/util/other.txt".to_owned(),
			"src/dir/".to_owned(),
		];
		assert_eq!(format_grouped_paths(&paths), "# src/\n## util/\nhelper.txt\nother.txt\n## dir/");
	}

	#[test]
	fn grouped_paths_root_files_are_bare() {
		let paths = vec!["a.txt".to_owned(), "b.txt".to_owned()];
		assert_eq!(format_grouped_paths(&paths), "a.txt\nb.txt");
	}

	#[test]
	fn grouped_files_emit_headers_and_bodies() {
		let files = vec!["src/a.txt".to_owned(), "src/b.txt".to_owned()];
		let model = format_grouped_files(&files, |path| GroupedFileSection {
			header_suffix: "#TAG".to_owned(),
			model_lines:   vec![format!("*1:{path}")],
			skip:          false,
		});
		// Depth>0 file headers get no blank-line separator (only depth-0 events
		// and dir headers do — grouped-file-output.ts `needsSeparator`).
		assert_eq!(model, vec![
			"# src/".to_owned(),
			"## a.txt#TAG".to_owned(),
			"*1:src/a.txt".to_owned(),
			"## b.txt#TAG".to_owned(),
			"*1:src/b.txt".to_owned(),
		]);
	}
}
