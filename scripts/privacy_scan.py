#!/usr/bin/env python3
"""Fail if the release contains machine paths, account names, e-mail addresses, secrets or media files."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"node_modules", "dist", ".git", "__pycache__"}
PATTERNS = {
    "absolute home/NFS path": re.compile(r"/(ipb\d+|home_domuser\d*|home)/[A-Za-z0-9_.-]+/"),
    "e-mail address": re.compile(r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com|anthropic\.com)[A-Za-z0-9.-]+\.[a-z]{2,}"),
    "token-like secret": re.compile(r"(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})"),
}
MEDIA = {".wav", ".mp4", ".jpg", ".jpeg", ".m4a", ".flac", ".pth", ".tar", ".pt", ".ckpt", ".npy"}
ALLOWED_MEDIA_DIRS = {"validation"}  # fold index arrays (.npy) only: row numbers, no media
problems = []
for p in ROOT.rglob("*"):
    if any(part in SKIP_DIRS for part in p.parts) or not p.is_file():
        continue
    rel = p.relative_to(ROOT)
    if p.suffix.lower() in MEDIA and rel.parts[0] not in ALLOWED_MEDIA_DIRS:
        problems.append(f"{rel}: media/checkpoint file must not be published")
        continue
    if p.suffix.lower() in {".png", ".svg", ".ico", ".woff2", ".npy"} or p.stat().st_size > 5_000_000:
        continue
    try:
        text = p.read_text(errors="ignore")
    except Exception:
        continue
    for name, rx in PATTERNS.items():
        m = rx.search(text)
        if m and rel != Path("scripts/privacy_scan.py"):
            problems.append(f"{rel}: {name}: {m.group(0)[:60]}")
for x in problems:
    print("PRIVACY:", x)
print(f"privacy scan: {len(problems)} problem(s)")
sys.exit(1 if problems else 0)
