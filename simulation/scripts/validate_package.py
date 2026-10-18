#!/usr/bin/env python3
"""Static validation for the balanced MED1/OCT4/DNA design package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LENGTHS = {"MED1": 1581, "OCT4": 360, "DNA": 400}
BASE_BOX_NM = 75.0
EXPECTED_CORE_TASKS = 45
EXPECTED_OPTIONAL_TASKS = 6


def fail(message: str) -> None:
    raise SystemExit(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atom_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return [
            line
            for line in handle
            if line.startswith(("ATOM  ", "HETATM"))
        ]


def fixed_residue_box_nm(n_med1: int) -> float:
    residues = LENGTHS["OCT4"] * (100 - n_med1) + LENGTHS["MED1"] * n_med1
    return BASE_BOX_NM * (residues / 36_000) ** (1 / 3)


with (ROOT / "task_manifest.tsv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected_total = EXPECTED_CORE_TASKS + EXPECTED_OPTIONAL_TASKS
if len(rows) != expected_total:
    fail(f"Expected {expected_total} tasks, found {len(rows)}")
if len({row["task_id"] for row in rows}) != len(rows):
    fail("Duplicate task_id in task_manifest.tsv")
if sum(row["tier"] == "core" for row in rows) != EXPECTED_CORE_TASKS:
    fail("Core task count is not 45")
if sum(row["tier"] == "optional" for row in rows) != EXPECTED_OPTIONAL_TASKS:
    fail("Optional task count is not 6")
if any(row["tier"] != "core" for row in rows[:EXPECTED_CORE_TASKS]):
    fail("Core tasks must occupy array indices 0-44")
if any(row["tier"] != "optional" for row in rows[EXPECTED_CORE_TASKS:]):
    fail("Optional tasks must occupy array indices 45-50")

by_task = {row["task_id"]: row for row in rows}
starts: dict[str, Path] = {}

for row in rows:
    task_id = row["task_id"]
    start = ROOT / "tasks" / task_id / "start.pdb"
    wrapper = start.with_name("run.sh")
    molecule_map = start.with_name("molecule_map.tsv")
    if not start.is_file() or not wrapper.is_file() or not molecule_map.is_file():
        fail(f"Missing task input/wrapper for {task_id}")
    atoms = atom_lines(start)
    expected_particles = (
        int(row["n_med1"]) * LENGTHS["MED1"]
        + int(row["n_oct4"]) * LENGTHS["OCT4"]
        + int(row["n_dna"]) * LENGTHS["DNA"]
    )
    if len(atoms) != expected_particles:
        fail(
            f"{task_id}: start.pdb particles={len(atoms)}, "
            f"expected={expected_particles}"
        )
    fragments = int(row["med1_fragments"])
    if fragments not in (1, 2, 4):
        fail(f"{task_id}: invalid med1_fragments={fragments}")
    if int(row["n_med1"]) == 0 and fragments != 1:
        fail(f"{task_id}: MED1-free system cannot have split fragments")
    with molecule_map.open(newline="", encoding="utf-8") as handle:
        molecule_rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_molecules = (
        int(row["n_med1"]) * fragments
        + int(row["n_oct4"])
        + int(row["n_dna"])
    )
    if len(molecule_rows) != expected_molecules:
        fail(
            f"{task_id}: molecule map has {len(molecule_rows)} rows; "
            f"expected {expected_molecules}"
        )
    cursor = 0
    for molecule in molecule_rows:
        start_index = int(molecule["particle_start"])
        stop_index = int(molecule["particle_stop_exclusive"])
        if start_index != cursor or stop_index <= start_index:
            fail(f"{task_id}: non-contiguous molecule map at {molecule['molecule_id']}")
        cursor = stop_index
    if cursor != expected_particles:
        fail(
            f"{task_id}: molecule map ends at particle {cursor}; "
            f"expected {expected_particles}"
        )
    if "composition_" in row["series_tags"]:
        if int(row["n_med1"]) + int(row["n_oct4"]) != 100:
            fail(f"{task_id}: composition series does not contain 100 proteins")
        expected_box = fixed_residue_box_nm(int(row["n_med1"]))
        if abs(float(row["box_nm"]) - expected_box) > 1e-6:
            fail(
                f"{task_id}: box={row['box_nm']}, expected={expected_box:.6f}"
            )
    starts[task_id] = start

for row in rows:
    task_id = row["task_id"]
    source = row["coordinate_source"]
    parent_task = row["parent_task"]
    if source.startswith("added_DNA_to:"):
        if parent_task not in by_task:
            fail(f"{task_id}: missing DNA-addition parent {parent_task}")
        parent = by_task[parent_task]
        for field in ("n_med1", "n_oct4", "med1_fragments", "box_nm", "seed"):
            if row[field] != parent[field]:
                fail(f"{task_id}: DNA addition changed matched field {field}")
        if int(parent["n_dna"]) != 0 or int(row["n_dna"]) != 1:
            fail(f"{task_id}: expected D0 -> D1 comparison")
        protein_particles = (
            int(parent["n_med1"]) * LENGTHS["MED1"]
            + int(parent["n_oct4"]) * LENGTHS["OCT4"]
        )
        child_atoms = atom_lines(starts[task_id])
        parent_atoms = atom_lines(starts[parent_task])
        if child_atoms[:protein_particles] != parent_atoms:
            fail(f"{task_id}: protein coordinates differ from D0 parent")
    elif source.startswith("exact_copy_of:"):
        if parent_task not in by_task:
            fail(f"{task_id}: missing split-control parent {parent_task}")
        parent = by_task[parent_task]
        for field in ("n_med1", "n_oct4", "n_dna", "box_nm", "coord_seed", "seed"):
            if row[field] != parent[field]:
                fail(f"{task_id}: split control changed matched field {field}")
        if sha256(starts[task_id]) != sha256(starts[parent_task]):
            fail(f"{task_id}: split-control start coordinates differ from Full")

with (ROOT / "design_matrix.tsv").open(newline="", encoding="utf-8") as handle:
    matrix = list(csv.DictReader(handle, delimiter="\t"))
if len(matrix) != 22:
    fail(f"Expected 22 design-matrix rows, found {len(matrix)}")
manifest_systems = {row["system"] for row in rows}
missing_systems = sorted({row["system"] for row in matrix} - manifest_systems)
if missing_systems:
    fail(f"Design matrix refers to missing systems: {missing_systems}")

checksum_file = ROOT / "INPUT_SHA256SUMS"
if not checksum_file.is_file():
    fail("Missing INPUT_SHA256SUMS; run scripts/update_checksums.sh")
for line in checksum_file.read_text(encoding="utf-8").splitlines():
    digest, relative = line.split(maxsplit=1)
    path = ROOT / relative.lstrip("*")
    if not path.is_file():
        fail(f"Checksummed file is missing: {relative}")
    actual = sha256(path)
    if actual != digest:
        fail(f"Checksum mismatch: {relative}")

print(
    "Static validation passed: 45 core tasks, 6 optional tasks, "
    "matched DNA-addition coordinates, and matched Full/Split coordinates."
)
