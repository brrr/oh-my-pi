import { describe, expect, it } from "bun:test";
import * as path from "node:path";
import { $ } from "bun";
import { DARWIN_TARGETS, nonSystemDylibs, UNIVERSAL_OUTFILE, universalBuildEnv } from "./build-omp-headless-universal";

const repoRoot = path.join(import.meta.dir, "..");

async function runUniversalDryRun(env: Record<string, string | undefined> = {}): Promise<string> {
	const result = await $`bun scripts/build-omp-headless-universal.ts --dry-run`
		.cwd(repoRoot)
		.quiet()
		.env({
			...process.env,
			PCRE2_SYS_STATIC: "0",
			MACOSX_DEPLOYMENT_TARGET: "",
			...env,
		})
		.nothrow();
	expect(result.exitCode).toBe(0);
	return result.text();
}

describe("omp-headless universal build environment", () => {
	it("forces static PCRE2 so the binary does not link the host's Homebrew copy", () => {
		expect(universalBuildEnv({ PCRE2_SYS_STATIC: "0" }).PCRE2_SYS_STATIC).toBe("1");
	});

	it("pins a deployment target for the x86_64 slice but lets the caller override it", () => {
		expect(universalBuildEnv({}).MACOSX_DEPLOYMENT_TARGET).toBe("11.0");
		expect(universalBuildEnv({ MACOSX_DEPLOYMENT_TARGET: "12.0" }).MACOSX_DEPLOYMENT_TARGET).toBe("12.0");
	});

	it("builds both darwin slices before lipoing them together", async () => {
		const output = await runUniversalDryRun();
		for (const target of DARWIN_TARGETS) {
			expect(output).toContain(`cargo build --release -p omp-headless --target ${target} PCRE2_SYS_STATIC=1`);
		}
		expect(output).toContain(`lipo -create -output ${UNIVERSAL_OUTFILE}`);
	});
});

describe("nonSystemDylibs", () => {
	const header = "target/universal-apple-darwin/release/omp-headless:\n";

	it("accepts a binary that only depends on macOS itself", () => {
		const otool = `${header}\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)\n\t/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 4109.1.255)\n\t/usr/lib/libiconv.2.dylib (compatibility version 7.0.0, current version 7.0.0)\n`;
		expect(nonSystemDylibs(otool)).toEqual([]);
	});

	it("flags a Homebrew dependency — the exact breakage this script exists to prevent", () => {
		const otool = `${header}\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)\n\t/opt/homebrew/opt/pcre2/lib/libpcre2-8.0.dylib (compatibility version 16.0.0, current version 16.0.0)\n`;
		expect(nonSystemDylibs(otool)).toEqual(["/opt/homebrew/opt/pcre2/lib/libpcre2-8.0.dylib"]);
	});
});
