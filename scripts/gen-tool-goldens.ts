#!/usr/bin/env bun
/**
 * WP-1.2 C2 golden generator: drives the REAL TypeScript tools
 * (read/write/edit/bash/grep/glob) against the checked-in fixture workspace
 * (`crates/pi-tools/tests/fixture-ws/`) and dumps
 * `{tool, case, params, content, isError, useless, threw, postFile}` JSON to
 * `crates/pi-tools/tests/goldens/`, plus per-tool
 * `{wireSchema, normalized}` schema goldens.
 *
 * Pinned context (U-omp-27):
 * - read/grep/glob/bash sessions: `Settings.isolated` defaults, i.e.
 *   `edit.mode = hashline` → hashline display; plus
 *   `read.summarize.enabled = false` (summaries deferred in the Rust core),
 *   `async.enabled = false` (bash base schema, no async param),
 *   `shellMinimizer.enabled = false` (minimizer deferred in the Rust core).
 * - edit/write sessions: `edit.mode = replace` (replace-mode edit tool; write
 *   emits no hashline snapshot header) + `PI_EDIT_FUZZY=false` (exact-only).
 * - Determinism: the fixture tree is copied to a scratch dir per case group,
 *   every path gets a fixed mtime (base 1700000000 + 60s per sorted path),
 *   absolute scratch roots are normalized to `«WS»`, and wall-time notices to
 *   `Wall time: «WT» seconds`.
 */

process.env.PI_EDIT_FUZZY = "false";

import * as fs from "node:fs";
import * as fsp from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { normalizeAnthropicToolSchema } from "@oh-my-pi/pi-ai/providers/anthropic";
import { toolWireSchema } from "@oh-my-pi/pi-ai/utils/schema/wire";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { EditTool } from "@oh-my-pi/pi-coding-agent/edit";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { BashTool } from "@oh-my-pi/pi-coding-agent/tools/bash";
import { GlobTool } from "@oh-my-pi/pi-coding-agent/tools/glob";
import { GrepTool } from "@oh-my-pi/pi-coding-agent/tools/grep";
import { ReadTool } from "@oh-my-pi/pi-coding-agent/tools/read";
import { WriteTool } from "@oh-my-pi/pi-coding-agent/tools/write";

const REPO_ROOT = path.resolve(import.meta.dir, "..");
const FIXTURE_SRC = path.join(REPO_ROOT, "crates/pi-tools/tests/fixture-ws");
const GOLDENS_DIR = path.join(REPO_ROOT, "crates/pi-tools/tests/goldens");

const MTIME_BASE_SEC = 1_700_000_000;
const MTIME_STEP_SEC = 60;

/** Collect every path under `root` (files + dirs), root-relative, sorted. */
function collectRelPaths(root: string): string[] {
	const out: string[] = [];
	const walk = (dir: string) => {
		for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
			const abs = path.join(dir, entry.name);
			out.push(path.relative(root, abs).replaceAll("\\", "/"));
			if (entry.isDirectory()) walk(abs);
		}
	};
	walk(root);
	out.sort();
	return out;
}

/** Fixed mtimes: base + 60s per sorted relative path (mirrored in Rust). */
function applyFixedMtimes(root: string): void {
	const rel = collectRelPaths(root);
	rel.forEach((entry, index) => {
		const t = MTIME_BASE_SEC + index * MTIME_STEP_SEC;
		fs.utimesSync(path.join(root, entry), t, t);
	});
}

async function makeScratchWs(): Promise<string> {
	const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "pi-tools-golden-"));
	const ws = await fsp.realpath(dir);
	await fsp.cp(FIXTURE_SRC, ws, { recursive: true });
	applyFixedMtimes(ws);
	return ws;
}

function makeSession(cwd: string, overrides: Record<string, unknown>): ToolSession {
	return {
		cwd,
		hasUI: false,
		getSessionFile: () => null,
		getSessionSpawns: () => null,
		settings: Settings.isolated(overrides as never),
		enableLsp: false,
	};
}

const HASH_SESSION_OVERRIDES = {
	"read.summarize.enabled": false,
	"async.enabled": false,
	"shellMinimizer.enabled": false,
};
const REPLACE_SESSION_OVERRIDES = {
	...HASH_SESSION_OVERRIDES,
	"edit.mode": "replace",
};

const WALL_TIME_RE = /Wall time: \d+\.\d+ seconds/g;

function normalizeText(text: string, ws: string): string {
	return text.replaceAll(ws, "«WS»").replace(WALL_TIME_RE, "Wall time: «WT» seconds");
}

interface GoldenCase {
	tool: string;
	case: string;
	params: unknown;
	content: Array<{ type: string; text: string }> | null;
	isError: boolean;
	useless: boolean;
	threw: string | null;
	/** Post-execution file state for mutating cases (path → content). */
	postFiles?: Record<string, string>;
}

function writeGolden(name: string, data: unknown): void {
	const file = path.join(GOLDENS_DIR, `${name}.json`);
	fs.writeFileSync(file, `${JSON.stringify(data, null, "\t")}\n`);
}

async function runCase(
	tool: { execute: (id: string, params: never, signal?: AbortSignal) => Promise<unknown> },
	toolName: string,
	caseName: string,
	params: unknown,
	ws: string,
	postFilePaths?: string[],
): Promise<void> {
	const golden: GoldenCase = {
		tool: toolName,
		case: caseName,
		params,
		content: null,
		isError: false,
		useless: false,
		threw: null,
	};
	try {
		const result = (await tool.execute(`golden-${caseName}`, params as never)) as {
			content: Array<{ type: string; text?: string }>;
			isError?: boolean;
			useless?: boolean;
		};
		golden.content = result.content
			.filter((block): block is { type: string; text: string } => typeof block.text === "string")
			.map(block => ({ type: block.type, text: normalizeText(block.text, ws) }));
		golden.isError = result.isError === true;
		golden.useless = result.useless === true;
	} catch (error) {
		golden.threw = normalizeText(error instanceof Error ? error.message : String(error), ws);
	}
	if (postFilePaths) {
		golden.postFiles = {};
		for (const rel of postFilePaths) {
			const abs = path.join(ws, rel);
			golden.postFiles[rel] = fs.existsSync(abs) ? fs.readFileSync(abs, "utf8") : "<missing>";
		}
	}
	writeGolden(`${toolName}-${caseName}`, golden);
	console.log(`golden: ${toolName}-${caseName}${golden.threw !== null ? " (threw)" : ""}`);
}

async function main(): Promise<void> {
	await Settings.init({ inMemory: true, overrides: HASH_SESSION_OVERRIDES as never });
	fs.mkdirSync(GOLDENS_DIR, { recursive: true });
	// Idempotent regeneration: drop stale goldens first.
	for (const entry of fs.readdirSync(GOLDENS_DIR)) {
		if (entry.endsWith(".json")) fs.rmSync(path.join(GOLDENS_DIR, entry));
	}

	// ── read / grep / glob / bash: one shared read-only scratch ws ─────────
	const ws = await makeScratchWs();
	const hashSession = makeSession(ws, HASH_SESSION_OVERRIDES);
	const readTool = new ReadTool(hashSession);
	const grepTool = new GrepTool(hashSession);
	const globTool = new GlobTool(hashSession);
	const bashTool = new BashTool(hashSession);

	// read
	await runCase(readTool, "read", "whole-small", { path: "README.txt" }, ws);
	await runCase(readTool, "read", "selector-open", { path: "README.txt:5" }, ws);
	await runCase(readTool, "read", "selector-range-brackets", { path: "src/app.txt:4-5" }, ws);
	await runCase(readTool, "read", "out-of-bounds", { path: "README.txt:999" }, ws);
	await runCase(readTool, "read", "truncated-big", { path: "big.txt" }, ws);
	await runCase(readTool, "read", "empty", { path: "empty.txt" }, ws);
	await runCase(readTool, "read", "not-found", { path: "nope/missing.txt" }, ws);
	await runCase(readTool, "read", "chinese", { path: "notes/中文.txt" }, ws);

	// grep
	await runCase(grepTool, "grep", "dir-default", { pattern: "needle" }, ws);
	await runCase(grepTool, "grep", "single-file", { pattern: "needle", path: "src/lib.txt" }, ws);
	await runCase(grepTool, "grep", "zero", { pattern: "zzz_no_such_token" }, ws);
	await runCase(grepTool, "grep", "invalid-regex", { pattern: "(" }, ws);
	await runCase(grepTool, "grep", "case-insensitive", { pattern: "NEEDLE", case: false, path: "src" }, ws);
	await runCase(grepTool, "grep", "glob-path", { pattern: "alpha", path: "src/*.txt" }, ws);

	// glob
	await runCase(globTool, "glob", "all-txt", { path: "**/*.txt" }, ws);
	await runCase(globTool, "glob", "dir", { path: "src" }, ws);
	await runCase(globTool, "glob", "zero", { path: "*.nomatch" }, ws);
	await runCase(globTool, "glob", "root", { path: "/" }, ws);
	await runCase(globTool, "glob", "limit", { path: "**/*.txt", limit: 3 }, ws);
	await runCase(globTool, "glob", "gitignore-off", { path: "ignored/*", gitignore: false }, ws);

	// bash
	await runCase(bashTool, "bash", "echo", { command: "echo hello from fixture" }, ws);
	await runCase(bashTool, "bash", "exit-code", { command: "printf 'partial output\\n'; exit 3" }, ws);
	await runCase(bashTool, "bash", "no-output", { command: "true" }, ws);
	await runCase(bashTool, "bash", "cwd-missing", { command: "echo hi", cwd: "does-not-exist" }, ws);
	await runCase(bashTool, "bash", "truncated", { command: "seq 1 20000" }, ws);
	await runCase(bashTool, "bash", "env", { command: 'printf \'%s\\n\' "$GREETING"', env: { GREETING: "你好 fixture" } }, ws);
	await runCase(bashTool, "bash", "timeout", { command: "printf start; sleep 2", timeout: 1 }, ws);

	// ── edit / write: fresh scratch ws per case ────────────────────────────
	const NOT_FOUND_OLD = Array.from({ length: 12 }, (_, i) => `phantom line ${i + 1}`).join("\n");
	const editCases: Array<{ name: string; params: unknown; postFiles?: string[] }> = [
		{
			name: "single",
			params: {
				path: "edit-target.txt",
				edits: [{ old_text: "target line to replace", new_text: "replaced line" }],
			},
			postFiles: ["edit-target.txt"],
		},
		{
			name: "all",
			params: { path: "edit-target.txt", edits: [{ old_text: "dup line", new_text: "dupe line", all: true }] },
			postFiles: ["edit-target.txt"],
		},
		{
			name: "ambiguous",
			params: { path: "edit-target.txt", edits: [{ old_text: "dup line", new_text: "dupe line" }] },
		},
		{
			name: "not-found-long",
			params: { path: "edit-target.txt", edits: [{ old_text: NOT_FOUND_OLD, new_text: "x" }] },
		},
		{
			name: "empty-old",
			params: { path: "edit-target.txt", edits: [{ old_text: "", new_text: "x" }] },
		},
		{
			name: "no-change",
			params: { path: "edit-target.txt", edits: [{ old_text: "alpha line", new_text: "alpha line" }] },
		},
		{
			name: "missing-file",
			params: { path: "missing.txt", edits: [{ old_text: "a", new_text: "b" }] },
		},
		{
			name: "multi-fail",
			params: {
				path: "edit-target.txt",
				edits: [
					{ old_text: "alpha line", new_text: "ALPHA line" },
					{ old_text: "dup line", new_text: "dupe line" },
					{ old_text: "omega line", new_text: "OMEGA line" },
				],
			},
			postFiles: ["edit-target.txt"],
		},
		{
			name: "crlf",
			params: { path: "crlf.txt", edits: [{ old_text: "crlf two", new_text: "crlf TWO" }] },
			postFiles: ["crlf.txt"],
		},
	];
	for (const editCase of editCases) {
		const caseWs = await makeScratchWs();
		const session = makeSession(caseWs, REPLACE_SESSION_OVERRIDES);
		const editTool = new EditTool(session);
		await runCase(editTool, "edit", editCase.name, editCase.params, caseWs, editCase.postFiles);
	}

	const writeCases: Array<{ name: string; params: unknown; postFiles?: string[] }> = [
		{
			name: "ascii-nested",
			params: { path: "out/plain.txt", content: "hello fixture\n" },
			postFiles: ["out/plain.txt"],
		},
		{
			name: "chinese",
			params: { path: "chinese.txt", content: "你好，世界\n" },
			postFiles: ["chinese.txt"],
		},
		{
			name: "shebang",
			params: { path: "run.sh", content: "#!/bin/sh\necho hi\n" },
			postFiles: ["run.sh"],
		},
		{
			name: "overwrite",
			params: { path: "README.txt", content: "replaced readme\n" },
			postFiles: ["README.txt"],
		},
	];
	for (const writeCase of writeCases) {
		const caseWs = await makeScratchWs();
		const session = makeSession(caseWs, REPLACE_SESSION_OVERRIDES);
		const writeTool = new WriteTool(session);
		await runCase(writeTool, "write", writeCase.name, writeCase.params, caseWs, writeCase.postFiles);
	}

	// ── schema goldens ─────────────────────────────────────────────────────
	const schemaWs = await makeScratchWs();
	const schemaHash = makeSession(schemaWs, HASH_SESSION_OVERRIDES);
	const schemaReplace = makeSession(schemaWs, REPLACE_SESSION_OVERRIDES);
	const schemaTools: Array<[string, unknown]> = [
		["read", new ReadTool(schemaHash)],
		["write", new WriteTool(schemaReplace)],
		["edit", new EditTool(schemaReplace)],
		["bash", new BashTool(schemaHash)],
		["grep", new GrepTool(schemaHash)],
		["glob", new GlobTool(schemaHash)],
	];
	for (const [name, tool] of schemaTools) {
		const wireSchema = toolWireSchema(tool as never);
		writeGolden(`schema-${name}`, {
			tool: name,
			wireSchema,
			normalized: normalizeAnthropicToolSchema(wireSchema),
		});
		console.log(`golden: schema-${name}`);
	}

	console.log("done");
	process.exit(0);
}

await main();
