#!/usr/bin/env python3
"""Inventory existing raw files for a later data archive; never upload or copy."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dna-only-root", type=Path, required=True)
    parser.add_argument("--sha256", action="store_true", help="Hash all files; may take a long time for full trajectories")
    parser.add_argument("--output", type=Path, default=ROOT / "data/trajectory_inventory.csv")
    args = parser.parse_args()
    with (ROOT / "simulation/task_manifest.tsv").open() as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))
    sources = []
    for task in tasks:
        for name in ("start.pdb", "output.dcd", "system.xml", "molecule_map.tsv", "run_metadata.json", "DONE"):
            sources.append((f"simulation/tasks/{task['task_id']}/{name}", args.data_root / task["task_id"] / name))
    for replicate in ("work", "work-1", "work-2"):
        for name in ("start.pdb", "output.dcd", "system.xml"):
            sources.append((f"data/dna_only/{replicate}/D1/{name}", args.dna_only_root / replicate / "D1" / name))
    rows = []
    for archive_path, source in sources:
        sha256 = ""
        if args.sha256 and source.is_file():
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            sha256 = digest.hexdigest()
        rows.append({"archive_path": archive_path, "exists_locally": source.is_file(),
                     "size_bytes": source.stat().st_size if source.is_file() else "",
                     "sha256": sha256, "upload_status": "not_started"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    trajectories = [r for r in rows if r["archive_path"].endswith(".dcd")]
    print(f"Inventoried {len(rows)} files, {len(trajectories)} trajectories; "
          f"{sum(int(r['size_bytes'] or 0) for r in trajectories) / 1e9:.2f} GB of trajectories. No upload started.")


if __name__ == "__main__":
    main()
