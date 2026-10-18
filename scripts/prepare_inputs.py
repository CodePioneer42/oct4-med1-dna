#!/usr/bin/env python3
"""Restore byte-identical structures, systems and figure snapshots from gzip objects."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore(root: Path = ROOT) -> dict[str, int]:
    root = root.resolve()
    manifest = json.loads((root / "simulation/input_manifest.json").read_text())
    if manifest["schema"] != "oct4-med1-gzip-inputs-v1":
        raise ValueError("Unsupported input manifest")
    restored = existing = 0
    for record in manifest["files"]:
        destination = (root / record["path"]).resolve()
        source = (root / record["object"]).resolve()
        destination.relative_to(root)
        source.relative_to(root)
        if destination.exists():
            if destination.stat().st_size != record["bytes"] or sha256(destination) != record["sha256"]:
                raise RuntimeError(f"Existing input differs from this release: {destination}")
            existing += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                with gzip.open(source, "rb") as compressed:
                    shutil.copyfileobj(compressed, output)
                output.flush()
                if temporary.stat().st_size != record["bytes"] or sha256(temporary) != record["sha256"]:
                    raise RuntimeError(f"Input checksum mismatch: {source}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        restored += 1
    return {"restored": restored, "already_verified": existing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(restore(args.root)))
