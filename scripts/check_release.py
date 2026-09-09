#!/usr/bin/env python3
"""Audit only release files: syntax, accidental raw data, local paths, secrets."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "__pycache__", ".venv", "generated"}
RAW_SUFFIXES = {".dcd", ".xtc", ".trr", ".chk", ".cpt"}
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_" + r"[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_" + r"[A-Za-z0-9_]{40,}"),
    re.compile(r"-----BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def release_files():
    return sorted(p for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & EXCLUDED))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    failures = []
    files = release_files()
    for path in files:
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            failures.append(f"Symlink: {rel}")
        if path.suffix in RAW_SUFFIXES or path.stat().st_size >= 90_000_000:
            failures.append(f"Raw or oversized file: {rel}")
        if path.suffix in {".py", ".sh", ".pml", ".md", ".json", ".csv", ".tsv", ".yml", ".txt", ".toml"}:
            text = path.read_text()
            if re.search(r"/(?:home|Users|lustre|scratch)/[^\s\"']+", text):
                failures.append(f"Absolute machine-specific path: {rel}")
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                failures.append(f"Possible credential: {rel}")
            if path.suffix == ".py":
                try:
                    ast.parse(text, filename=str(rel))
                except SyntaxError:
                    failures.append(f"Syntax error: {rel}")
    if failures:
        raise SystemExit("\n".join(failures))
    if args.write_manifest:
        provenance = ROOT / "docs/source_provenance.json"
        rows = json.loads(provenance.read_text())
        for row in rows:
            path = ROOT / row["destination"]
            row["export_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance.write_text(json.dumps(rows, indent=2) + "\n")
        manifest = ROOT / "SHA256SUMS"
        entries = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}" for p in files if p != manifest]
        manifest.write_text("\n".join(entries) + "\n")
    else:
        manifest = ROOT / "SHA256SUMS"
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                expected, name = line.split("  ", 1)
                path = ROOT / name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise SystemExit(f"Checksum mismatch: {name}")
    print(json.dumps({"files_audited": len(files), "total_megabytes": round(sum(p.stat().st_size for p in files) / 1e6, 2),
                      "syntax_and_basic_secret_scan": "passed", "raw_trajectories_in_release": False}))


if __name__ == "__main__":
    main()
