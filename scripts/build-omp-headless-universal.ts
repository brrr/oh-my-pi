#!/usr/bin/env bun

// Builds omp-headless as a macOS universal binary (arm64 + x86_64).
//
// Why this exists: a plain `cargo build --release -p omp-headless` on an Apple
// Silicon box produces an arm64-only binary that also links the *host's*
// Homebrew PCRE2 (`/opt/homebrew/opt/pcre2/lib/libpcre2-8.0.dylib`), because
// pcre2-sys prefers a pkg-config hit over its vendored source. Such a binary
// cannot be shipped: it fails to launch on Intel Macs and on any machine
// without Homebrew. Distribution needs both slices and zero non-system dylib
// dependencies, so this script pins PCRE2_SYS_STATIC=1 (shared with the native
// addon build) and lipos the two target builds together.

import * as path from "node:path";
import { $ } from "bun";
import { withPortableNativeBuildEnv } from "./ci-build-native";

const repoRoot = path.join(import.meta.dir, "..");
const isDryRun = process.argv.includes("--dry-run");

export const DARWIN_TARGETS = ["aarch64-apple-darwin", "x86_64-apple-darwin"] as const;
export const UNIVERSAL_OUTFILE = "target/universal-apple-darwin/release/omp-headless";

/** Oldest macOS the x86_64 slice is expected to run on. */
const DEFAULT_DEPLOYMENT_TARGET = "11.0";

/** Dylib prefixes that ship with macOS and are therefore safe to depend on. */
const SYSTEM_DYLIB_PREFIXES = ["/usr/lib/", "/System/"];

/** Build env for a portable, self-contained omp-headless. */
export function universalBuildEnv(env: Record<string, string | undefined>): Record<string, string | undefined> {
	return {
		...withPortableNativeBuildEnv(env),
		MACOSX_DEPLOYMENT_TARGET: env.MACOSX_DEPLOYMENT_TARGET || DEFAULT_DEPLOYMENT_TARGET,
	};
}

/**
 * Extracts the load-command dependencies that are NOT part of macOS itself,
 * given `otool -L` output. A non-empty result means the binary is tied to the
 * build machine (Homebrew, MacPorts, a Nix store path, ...) and is not
 * distributable.
 */
export function nonSystemDylibs(otoolOutput: string): string[] {
	return otoolOutput
		.split("\n")
		.slice(1) // first line is the inspected binary's own path
		.map(line => line.trim().split(/\s+/)[0] ?? "")
		.filter(dep => dep.startsWith("/") && !SYSTEM_DYLIB_PREFIXES.some(prefix => dep.startsWith(prefix)));
}

async function buildTarget(target: string, env: Record<string, string | undefined>): Promise<void> {
	if (isDryRun) {
		console.log(
			`DRY RUN cargo build --release -p omp-headless --target ${target} PCRE2_SYS_STATIC=${env.PCRE2_SYS_STATIC}`,
		);
		return;
	}
	console.log(`Building omp-headless [${target}]...`);
	await $`cargo build --release -p omp-headless --target ${target}`.cwd(repoRoot).env(env);
}

async function lipoTargets(outfile: string): Promise<void> {
	const slices = DARWIN_TARGETS.map(target => `target/${target}/release/omp-headless`);
	if (isDryRun) {
		console.log(`DRY RUN lipo -create -output ${outfile} ${slices.join(" ")}`);
		return;
	}
	await $`mkdir -p ${path.dirname(outfile)}`.cwd(repoRoot);
	await $`lipo -create -output ${outfile} ${slices}`.cwd(repoRoot);
}

/** Fails the build if either slice would drag in a build-machine dylib. */
async function assertPortable(outfile: string): Promise<void> {
	for (const arch of ["arm64", "x86_64"]) {
		const otool = await $`otool -arch ${arch} -L ${outfile}`.cwd(repoRoot).quiet().text();
		const offenders = nonSystemDylibs(otool);
		if (offenders.length > 0) {
			throw new Error(
				`${outfile} [${arch}] links non-system dylibs and is not distributable: ${offenders.join(", ")}`,
			);
		}
	}
}

async function main(): Promise<void> {
	if (!isDryRun && process.platform !== "darwin") {
		throw new Error("build-omp-headless-universal only runs on macOS");
	}

	const env = universalBuildEnv(Bun.env);
	for (const target of DARWIN_TARGETS) {
		await buildTarget(target, env);
	}
	await lipoTargets(UNIVERSAL_OUTFILE);
	if (isDryRun) return;

	await assertPortable(UNIVERSAL_OUTFILE);
	const archs = (await $`lipo -archs ${UNIVERSAL_OUTFILE}`.cwd(repoRoot).quiet().text()).trim();
	console.log(`Built ${UNIVERSAL_OUTFILE} (${archs})`);
}

if (import.meta.main) await main();
