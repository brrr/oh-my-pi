#!/usr/bin/env bun
/**
 * Rust→TS session parity check (WP-1.3, U-omp-30). Builds and runs the Rust
 * `write_rust_session` example (which writes a v3 session JSONL with the
 * pure-Rust `SessionWriter` plus a Rust-side expected message dump), loads the
 * same file through the TypeScript `loadSessionMessagesReadOnly`, and asserts
 * the two message arrays are equal.
 *
 * Numbers compare by value (`0` == `0.0`): Rust's `f64` cost fields render
 * `0.0` where the JS number renders `0`; JSON draws no such distinction.
 *
 * Run: PATH="$HOME/.nvm/versions/node/v22.13.1/bin:$PATH" bun scripts/verify-session-rust-to-ts.ts
 */

import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { loadSessionMessagesReadOnly } from "@oh-my-pi/pi-coding-agent/session/session-loader";

const REPO_ROOT = path.resolve(import.meta.dir, "..");

/** Structural deep-equality with numbers compared by value. */
function jsonEq(a: unknown, b: unknown): boolean {
	if (typeof a === "number" && typeof b === "number") return a === b;
	if (Array.isArray(a) && Array.isArray(b)) {
		return a.length === b.length && a.every((x, i) => jsonEq(x, b[i]));
	}
	if (a && b && typeof a === "object" && typeof b === "object") {
		const ao = a as Record<string, unknown>;
		const bo = b as Record<string, unknown>;
		const ak = Object.keys(ao);
		const bk = Object.keys(bo);
		return ak.length === bk.length && ak.every(k => k in bo && jsonEq(ao[k], bo[k]));
	}
	return a === b;
}

async function main(): Promise<void> {
	const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "omp-rust-to-ts-"));
	try {
		// Build + run the Rust writer example; it prints {jsonl, expected} as JSON.
		const stdout = execFileSync(
			"cargo",
			["run", "-q", "-p", "pi-session", "--example", "write_rust_session", "--", scratch],
			{ cwd: REPO_ROOT, encoding: "utf-8" },
		);
		const { jsonl, expected } = JSON.parse(stdout.trim()) as { jsonl: string; expected: string };
		console.log(`rust wrote:    ${path.relative(REPO_ROOT, jsonl)}`);

		const rustExpected = JSON.parse(fs.readFileSync(expected, "utf-8")) as unknown[];
		// Round-trip through JSON so `undefined`-valued keys the live TS objects
		// carry (e.g. `providerPayload` on a compaction summary) are dropped, matching
		// the Rust dump which omits absent optionals.
		const tsMessages = JSON.parse(JSON.stringify(await loadSessionMessagesReadOnly(jsonl))) as unknown[];

		console.log(`rust expected: ${rustExpected.length} messages`);
		console.log(`ts  read back: ${tsMessages.length} messages`);

		if (!jsonEq(tsMessages, rustExpected)) {
			console.error("✗ MISMATCH — Rust-written session did not read back identically in TS");
			console.error("--- ts (loadSessionMessagesReadOnly) ---");
			console.error(JSON.stringify(tsMessages, null, 2));
			console.error("--- rust (load_session_messages) ---");
			console.error(JSON.stringify(rustExpected, null, 2));
			process.exit(1);
		}

		console.log(`✓ OK — Rust-written session reads back byte-identically in TS (${tsMessages.length} messages)`);
		console.log(JSON.stringify(tsMessages, null, 2));
	} finally {
		fs.rmSync(scratch, { recursive: true, force: true });
	}
}

await main();
