#!/usr/bin/env python3
"""Check released file hashes and the complete restored-input manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_inputs import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restored", action="store_true", help="Also verify every restored PDB/XML/DCD")
    args = parser.parse_args()
    files = 0
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Release checksum mismatch: {relative}")
        files += 1
    manifest = json.loads((ROOT / "simulation/input_manifest.json").read_text())
    objects = {row["path"]: row for row in manifest["objects"]}
    for record in manifest["files"]:
        if record["object"] not in objects:
            raise RuntimeError(f"Unlisted input object: {record['object']}")
        if args.restored:
            path = ROOT / record["path"]
            if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise RuntimeError(f"Restored input mismatch: {record['path']}")
    references = list((ROOT / "reference_figures").rglob("*.pdf"))
    if len(references) != 13:
        raise RuntimeError("Expected all 13 FIG-10-10 reference PDFs")
    print(json.dumps({"release_files_verified": files, "input_objects": len(objects),
                      "input_files": len(manifest["files"]), "reference_pdfs": len(references),
                      "restored_inputs_verified": bool(args.restored)}))


if __name__ == "__main__":
    main()
