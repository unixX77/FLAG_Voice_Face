// Copies the canonical registry into the app so every displayed number comes from it.
import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../public_results/experiments.json");
const dst = resolve(here, "../src/data/experiments.json");
if (!existsSync(src)) { console.error("missing", src); process.exit(1); }
mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
console.log("synced", src, "->", dst);
