#!/usr/bin/env bun
/**
 * WP-1.3 golden generator (TS→Rust arm): drives the REAL TypeScript
 * `SessionManager` (`packages/coding-agent/src/session/session-manager.ts`) to
 * write three v3 session JSONL fixtures into
 * `crates/pi-session/tests/goldens/*.jsonl`, then dumps the read-only message
 * view produced by `loadSessionMessagesReadOnly` for each into
 * `crates/pi-session/tests/goldens/*.messages.json`.
 *
 * The Rust `pi-session` loader replays the same file → `build_session_context`
 * and asserts its messages equal the `.messages.json` expectation
 * (`tests/golden_parity.rs`).
 *
 * Determinism (U-omp-30): entry ids, the session id, and every wall-clock
 * timestamp are injected with fixed values by stubbing `crypto.randomUUID`,
 * `Bun.randomUUIDv7`, and the no-arg `Date` constructor / `Date.now()` BEFORE
 * the SessionManager module is imported. The stubs leave `new Date(<iso>)`
 * parsing intact so `createCompactionSummaryMessage` etc. still derive their
 * unix-ms `timestamp` from the (now fixed) ISO envelope. Re-running the
 * generator is a byte-for-byte no-op (`git diff` clean).
 */

// ── Deterministic clock + id stubs (install before importing SessionManager) ──
const FIXED_ISO = "2026-01-01T00:00:00.000Z";
const FIXED_WALL_MS = Date.parse(FIXED_ISO);
/** Fixed unix-ms stamped on the message payloads the fixtures append. */
const FIXED_MSG_MS = 1_704_067_200_000; // 2024-01-01T00:00:00.000Z

const RealDate = Date;
// Replace the global Date so `new Date()` (no args) and `Date.now()` return a
// fixed instant; every other form (parsing an ISO string, ms number) delegates
// to the real implementation so downstream `getTime()` math is unchanged.
class FixedDate extends RealDate {
	constructor(...args: ConstructorParameters<typeof Date>) {
		if (args.length === 0) {
			super(FIXED_WALL_MS);
		} else {
			// @ts-expect-error variadic forward to the real Date constructor
			super(...args);
		}
	}

	static now(): number {
		return FIXED_WALL_MS;
	}
}
// @ts-expect-error swap the global Date binding for the generator run
globalThis.Date = FixedDate;

let uuidCounter = 0;
// `generateId` takes `crypto.randomUUID().slice(-8)`; a monotonic hex counter in
// the low 12 nibbles yields deterministic, collision-free 8-hex entry ids.
const realRandomUUID = crypto.randomUUID.bind(crypto);
void realRandomUUID;
// @ts-expect-error override for deterministic entry ids
crypto.randomUUID = (): `${string}-${string}-${string}-${string}-${string}` => {
	const n = (uuidCounter++).toString(16).padStart(12, "0");
	return `00000000-0000-0000-0000-${n}` as `${string}-${string}-${string}-${string}-${string}`;
};

let sessionCounter = 0;
// `mintSessionId` uses `Bun.randomUUIDv7()`; a fixed per-session uuid keeps the
// header id (and thus filename) deterministic.
Bun.randomUUIDv7 = ((): string => {
	const n = (sessionCounter++).toString(16).padStart(12, "0");
	return `01900000-0000-7000-8000-${n}`;
}) as typeof Bun.randomUUIDv7;

import * as fs from "node:fs";
import * as fsp from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { loadSessionMessagesReadOnly } from "@oh-my-pi/pi-coding-agent/session/session-loader";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";

const REPO_ROOT = path.resolve(import.meta.dir, "..");
const GOLDENS_DIR = path.join(REPO_ROOT, "crates/pi-session/tests/goldens");
const FIXED_CWD = "/omp/fixture";

// ── Message payload builders (unix-ms timestamps, pinned) ─────────────────────
type Json = Record<string, unknown>;

function userMsg(text: string): Json {
	return { role: "user", content: text, timestamp: FIXED_MSG_MS };
}

function assistantMsg(text: string, toolCall?: { id: string; name: string; arguments: unknown }): Json {
	const content: Json[] = [{ type: "text", text }];
	if (toolCall) {
		content.push({ type: "toolCall", id: toolCall.id, name: toolCall.name, arguments: toolCall.arguments });
	}
	return {
		role: "assistant",
		content,
		api: "anthropic",
		provider: "anthropic",
		model: "claude-sonnet-4",
		usage: {
			input: 10,
			output: 5,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 15,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: toolCall ? "toolUse" : "stop",
		timestamp: FIXED_MSG_MS,
	};
}

function toolResultMsg(id: string, name: string, text: string): Json {
	return {
		role: "toolResult",
		toolCallId: id,
		toolName: name,
		content: [{ type: "text", text }],
		isError: false,
		timestamp: FIXED_MSG_MS,
	};
}

interface Fixture {
	name: string;
	build: (sm: SessionManager) => Promise<void>;
}

const FIXTURES: Fixture[] = [
	{
		// ① Basic conversation: user → assistant(toolCall) → toolResult → assistant.
		name: "basic",
		build: async sm => {
			sm.appendMessage(userMsg("hello, read /x for me"));
			sm.appendMessage(
				assistantMsg("let me check", { id: "call_1", name: "read", arguments: { path: "/x" } }),
			);
			sm.appendMessage(toolResultMsg("call_1", "read", "file contents of /x"));
			sm.appendMessage(assistantMsg("done — /x contains the contents above"));
		},
	},
	{
		// ② Settings folding + compaction: model_change / thinking_level_change on
		//    the path, a compaction that keeps the second user turn onward, plus
		//    post-compaction messages.
		name: "compaction",
		build: async sm => {
			sm.appendMessage(userMsg("first question"));
			sm.appendMessage(assistantMsg("first answer"));
			sm.appendModelChange("anthropic/claude-opus-4");
			sm.appendThinkingLevelChange("high", "high");
			const keptId = sm.appendMessage(userMsg("second question"));
			sm.appendMessage(assistantMsg("second answer"));
			sm.appendCompaction("This is the compaction summary of the earlier turns.", "short summary", keptId, 4321);
			sm.appendMessage(userMsg("third question"));
			sm.appendMessage(assistantMsg("third answer"));
		},
	},
	{
		// ③ Unknown-type passthrough (custom / label / title_change) + a fork
		//    branch off the first turn that must be excluded by leaf→root walk.
		name: "unknown-and-fork",
		build: async sm => {
			const idA = sm.appendMessage(userMsg("hi"));
			sm.appendMessage(assistantMsg("hello"));
			sm.appendCustomMessageEntry("note", "injected context", true);
			// Fork: a message hanging off the first turn on a NON-active branch. The
			// leaf stays on the custom_message, so this turn must not appear.
			sm.appendMessageToBranch(userMsg("side branch that must be excluded"), idA);
			// Unknown passthrough entries on the active path (skipped by context):
			sm.appendCustomEntry("mymarker", { foo: "bar" });
			sm.appendLabelChange(idA, "important");
			await sm.setSessionName("My Session Title", "user");
		},
	},
];

async function main(): Promise<void> {
	fs.mkdirSync(GOLDENS_DIR, { recursive: true });
	for (const fixture of FIXTURES) {
		// Reset the deterministic id/clock counters per fixture so each file is a
		// self-contained, stable golden.
		uuidCounter = 0;
		sessionCounter = 0;
		const scratch = fs.mkdtempSync(path.join(os.tmpdir(), `omp-session-${fixture.name}-`));
		try {
			const sm = SessionManager.create(FIXED_CWD, scratch);
			await fixture.build(sm);
			await sm.ensureOnDisk();
			await sm.flush();
			const file = sm.getSessionFile();
			if (!file) throw new Error(`fixture ${fixture.name}: no session file produced`);
			const jsonl = fs.readFileSync(file, "utf-8");

			// Dump the read-only message view (the parity expectation).
			const messages = await loadSessionMessagesReadOnly(file);

			const jsonlPath = path.join(GOLDENS_DIR, `${fixture.name}.jsonl`);
			const messagesPath = path.join(GOLDENS_DIR, `${fixture.name}.messages.json`);
			fs.writeFileSync(jsonlPath, jsonl);
			fs.writeFileSync(messagesPath, `${JSON.stringify(messages, null, 2)}\n`);
			console.log(`✓ ${fixture.name}: ${jsonl.split("\n").filter(Boolean).length} lines, ${messages.length} messages`);
		} finally {
			await fsp.rm(scratch, { recursive: true, force: true });
		}
	}
	console.log(`goldens written to ${path.relative(REPO_ROOT, GOLDENS_DIR)}`);
}

await main();
