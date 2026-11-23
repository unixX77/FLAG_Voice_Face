"""The public registry must agree with its copied source files (no hand-typed numbers)."""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = json.loads((ROOT / "public_results/experiments.json").read_text())
PCT = 100.0


class RegistryTests(unittest.TestCase):
    def test_every_source_exists(self):
        for e in REG["experiments"]:
            self.assertTrue(e["sources"], e["id"])
            for s in e["sources"]:
                self.assertTrue((ROOT / s["path"]).exists(), f"{e['id']}: missing {s['path']}")

    def test_compare_entries_match_their_json(self):
        n = 0
        for e in REG["experiments"]:
            src = [s for s in e["sources"] if s.get("extract", "").startswith("compare_fold_runs")]
            if not src:
                continue
            d = json.loads((ROOT / src[0]["path"]).read_text())
            if "comparison" not in e:  # reference entry: stores only the A side of the cited comparison
                (name,) = e["systems"]
                for t in ("standard", "gender", "mean"):
                    self.assertAlmostEqual(e["systems"][name][t], round(d[t]["a"] * PCT, 2), places=6, msg=e["id"])
                n += 1
                continue
            a, b = e["comparison"]["a"], e["comparison"]["b"]
            for t in ("standard", "gender", "mean"):
                self.assertAlmostEqual(e["systems"][a][t], round(d[t]["a"] * PCT, 2), places=6, msg=f"{e['id']} {t} A")
                self.assertAlmostEqual(e["systems"][b][t], round(d[t]["b"] * PCT, 2), places=6, msg=f"{e['id']} {t} B")
                self.assertAlmostEqual(e["comparison"]["delta"][t]["value"], round(d[t]["b_minus_a"] * PCT, 2), places=6)
                self.assertEqual(e["comparison"]["delta"][t]["ci95"], [round(x * PCT, 2) for x in d[t]["ci95"]])
            n += 1
        self.assertGreater(n, 30)

    def test_terms_entries_use_unrounded_means(self):
        for e in REG["experiments"]:
            src = [s for s in e["sources"] if "cv_mean_eer" in s.get("extract", "")]
            if not src:
                continue
            w = json.loads((ROOT / src[0]["path"]).read_text())["weights"]
            name = next(k for k in e["systems"] if k.endswith("+ terms"))
            mean = (w["no_gender"]["cv_mean_eer"] + w["gender"]["cv_mean_eer"]) / 2 * PCT
            self.assertAlmostEqual(e["systems"][name]["mean"], round(mean, 2), places=6, msg=e["id"])

    def test_log_sources_point_at_real_lines_with_the_numbers(self):
        log = (ROOT / "docs/research/progress_log.md").read_text().splitlines()
        for e in REG["experiments"]:
            for s in e["sources"]:
                if s["type"] != "progress_log":
                    continue
                a, b = (int(x) for x in s["lines"].split("-"))
                self.assertLessEqual(b, len(log), e["id"])
                text = "\n".join(log[a - 1:b])
                nums = set(re.findall(r"\d+\.\d+", text))
                tables = [e.get(k) for k in ("systems", "cells", "points") if e.get(k)]
                vals = []
                for t in tables:
                    rows = t.values() if isinstance(t, dict) else t
                    for r in rows:
                        vals += [f"{v:.2f}" for v in r.values() if isinstance(v, float)] + [f"{v:.1f}" for v in r.values() if isinstance(v, float)]
                if vals:
                    self.assertTrue(any(v in nums for v in vals), f"{e['id']}: none of its numbers occur in log lines {s['lines']}")

    def test_no_unverified_submission_claims(self):
        sub = next(e for e in REG["experiments"] if e["id"] == "codabench_eval_submissions")
        self.assertEqual(set(sub["systems"]), {"FOP baseline (organisers)", "Eval #1: V2", "Eval #2: V2 + age/gender terms",
                                               "Eval #3: V2 + external data + terms"})
        final = next(e for e in REG["experiments"] if e["id"] == "final_c1x3_terms")
        self.assertEqual(final["outcome"], "candidate")
        self.assertIn("No CodaBench score", final["interpretation"])

    def test_protocols_are_declared(self):
        for e in REG["experiments"]:
            self.assertIn(e["protocol"], REG["protocols"], e["id"])


if __name__ == "__main__":
    unittest.main()
