#!/usr/bin/env bun
/**
 * WP-1.6 (B4) migration golden generator. Loads the real v1 session fixtures
 * through the TypeScript `loadSessionMessagesReadOnly` (which migrates
 * v1→v2→v3, resolves blob refs, and rebuilds the read-only message view) and
 * dumps the resulting message array next to the copied fixture in
 * `crates/pi-session/tests/fixtures/*.messages.json`.
 *
 * The Rust `pi-session` loader replays the same v1 file → in-memory migration →
 * `build_session_context` and asserts its messages equal this expectation
 * (`tests/migration_parity.rs`). Migration entry ids differ per run (random in
 * TS, counter in Rust) but never appear in the message output, so the two
 * message arrays are id-independent and compare equal.
 *
 * Run: PATH="$HOME/.nvm/versions/node/v22.13.1/bin:$PATH" bun scripts/gen-migration-goldens.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { loadSessionMessagesReadOnly } from "@oh-my-pi/pi-coding-agent/session/session-loader";

const REPO_ROOT = path.resolve(import.meta.dir, "..");
const FIXTURES_DIR = path.join(REPO_ROOT, "crates/pi-session/tests/fixtures");

const FIXTURES = ["v1-before-compaction", "v1-large-session"];

async function main(): Promise<void> {
	for (const name of FIXTURES) {
		const jsonl = path.join(FIXTURES_DIR, `${name}.jsonl`);
		const messages = await loadSessionMessagesReadOnly(jsonl);
		const out = path.join(FIXTURES_DIR, `${name}.messages.json`);
		fs.writeFileSync(out, `${JSON.stringify(messages, null, 2)}\n`);
		console.log(`✓ ${name}: ${messages.length} messages → ${path.relative(REPO_ROOT, out)}`);
	}
}

await main();
