// Bundles src/ssr-check.tsx with esbuild (shipped with Vite) and runs it in Node.
import { build } from "esbuild";
import { execFileSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
const out = join(mkdtempSync(join(tmpdir(), "flag-ssr-")), "check.mjs");
await build({ entryPoints: ["src/ssr-check.tsx"], bundle: true, platform: "node", format: "esm", outfile: out,
  jsx: "automatic", loader: { ".json": "json" }, logLevel: "error",
  banner: { js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);" } });
execFileSync(process.execPath, [out], { stdio: "inherit" });
