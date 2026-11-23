"""Internal links must resolve: relative links in Markdown docs, and repository paths linked from the website."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"node_modules", "dist", ".git"}


class LinkTests(unittest.TestCase):
    def test_markdown_relative_links(self):
        bad = []
        for md in ROOT.rglob("*.md"):
            if any(p in SKIP for p in md.parts) or "docs/research" in md.as_posix():
                continue  # verbatim research reports cite the private research tree by design
            for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", md.read_text()):
                if re.match(r"^[a-z]+://|^mailto:", target):
                    continue
                if not (md.parent / target).resolve().exists():
                    bad.append(f"{md.relative_to(ROOT)} -> {target}")
        self.assertEqual(bad, [])

    def test_site_repository_links(self):
        bad = []
        for src in (ROOT / "site/src").rglob("*.ts*"):
            text = src.read_text()
            for path in re.findall(r"\$\{REPO_URL\}/blob/main/([^`\"'#?]+)", text):
                if not path.startswith("${") and not (ROOT / path).exists():  # skip template placeholders
                    bad.append(f"{src.relative_to(ROOT)} -> {path}")
        self.assertEqual(bad, [])

    def test_registry_source_paths_exist(self):
        import json
        reg = json.loads((ROOT / "public_results/experiments.json").read_text())
        missing = [s["path"] for e in reg["experiments"] for s in e["sources"] if not (ROOT / s["path"]).exists()]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
