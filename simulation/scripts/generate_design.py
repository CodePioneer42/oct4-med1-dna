#!/usr/bin/env python3
"""Generate the reorganized MED1/OCT4/DNA task matrix and start structures."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "OpenABC"))

import numpy as np
import pandas as pd
from openabc.utils import (
    atomistic_pdb_to_ca_pdb,
    atomistic_pdb_to_nucleotide_pdb,
    parse_pdb,
    write_pdb,
)
from openabc.utils.insert import insert_molecules_dataframe


MED1_LENGTH = 1581
OCT4_LENGTH = 360
DNA_LENGTH = 400
BASE_BOX_NM = 75.0
REPLICATES = ("work", "work-1", "work-2")
TOTAL_STEPS = 500_000_000
OUTPUT_INTERVAL = 100_000
MAX_INSERTION_ATTEMPTS = 250_000
MED1_CUTS_AFTER_RESIDUE = {1: (), 2: (807,), 4: (354, 807, 1171)}


def fixed_residue_box_nm(n_med1: int) -> float:
    residues = OCT4_LENGTH * (100 - n_med1) + MED1_LENGTH * n_med1
    return BASE_BOX_NM * (residues / (100 * OCT4_LENGTH)) ** (1 / 3)


@dataclass(frozen=True)
class BaseSystem:
    name: str
    n_med1: int
    n_oct4: int
    box_nm: float
    tier: str
    tags: str


BASE_SYSTEMS = (
    BaseSystem("M0O100", 0, 100, fixed_residue_box_nm(0), "core", "composition_core,dna_addition"),
    BaseSystem("M5O95", 5, 95, fixed_residue_box_nm(5), "core", "composition_core"),
    BaseSystem("M10O90", 10, 90, fixed_residue_box_nm(10), "core", "composition_core,dna_addition,chain_length"),
    BaseSystem("M15O85", 15, 85, fixed_residue_box_nm(15), "core", "composition_core"),
    BaseSystem("M20O80", 20, 80, fixed_residue_box_nm(20), "core", "composition_core,dna_addition"),
    BaseSystem("M30O70", 30, 70, fixed_residue_box_nm(30), "core", "composition_core"),
    BaseSystem("M50O0", 50, 0, BASE_BOX_NM, "core", "pure_med1,dna_addition"),
    BaseSystem("M25O75", 25, 75, fixed_residue_box_nm(25), "optional", "composition_optional"),
    BaseSystem("M50O50", 50, 50, fixed_residue_box_nm(50), "optional", "composition_optional"),
)
BASE_BY_NAME = {system.name: system for system in BASE_SYSTEMS}

CORE_ORDER = (
    "M0O100",
    "M5O95",
    "M10O90",
    "M15O85",
    "M20O80",
    "M30O70",
    "M0O100D1",
    "M10O90D1",
    "M20O80D1",
    "M50O0",
    "M50O0D1",
    "M10O90-S2",
    "M10O90D1-S2",
    "M10O90-S4",
    "M10O90D1-S4",
)
OPTIONAL_ORDER = ("M25O75", "M50O50")

MANIFEST_FIELDS = (
    "task_id",
    "system",
    "replicate",
    "tier",
    "series_tags",
    "n_med1",
    "n_oct4",
    "n_dna",
    "med1_fragments",
    "box_nm",
    "coord_seed",
    "seed",
    "total_steps",
    "output_interval",
    "parent_task",
    "coordinate_source",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_pdb_atoms(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(
            line.startswith(("ATOM  ", "HETATM")) for line in handle
        )


def atomic_write_pdb(atoms: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Stale temporary file exists; inspect it before retrying: {temporary}"
        )
    write_pdb(atoms, str(temporary))
    os.replace(temporary, destination)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def insert_species(
    template: pd.DataFrame,
    n_molecules: int,
    box_nm: float,
    existing: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if n_molecules == 0:
        return existing
    before = 0 if existing is None else len(existing.index)
    atoms = insert_molecules_dataframe(
        template,
        n_mol=n_molecules,
        radius=0.5,
        existing_atoms=existing,
        max_n_attempts=MAX_INSERTION_ATTEMPTS,
        box=[box_nm, box_nm, box_nm],
        reset_serial=True,
    )
    expected = before + n_molecules * len(template.index)
    if len(atoms.index) != expected:
        raise RuntimeError(
            f"Insertion produced {len(atoms.index)} particles; expected {expected}"
        )
    return atoms


def generate_protein_start(
    system: BaseSystem,
    med1_atoms: pd.DataFrame,
    oct4_atoms: pd.DataFrame,
    coord_seed: int,
) -> pd.DataFrame:
    np.random.seed(coord_seed)
    atoms = insert_species(med1_atoms, system.n_med1, system.box_nm, None)
    atoms = insert_species(oct4_atoms, system.n_oct4, system.box_nm, atoms)
    if atoms is None:
        raise RuntimeError(f"No particles generated for {system.name}")
    expected = system.n_med1 * MED1_LENGTH + system.n_oct4 * OCT4_LENGTH
    if len(atoms.index) != expected:
        raise RuntimeError(
            f"{system.name}: generated {len(atoms.index)} particles; expected {expected}"
        )
    return atoms


def install_wrapper(task_dir: Path) -> None:
    wrapper = task_dir / "run.sh"
    if not wrapper.exists():
        shutil.copyfile(ROOT / "common" / "task_run.sh", wrapper)
        wrapper.chmod(0o755)


def ensure_start(
    task_id: str,
    expected_atoms: int,
    producer: Any,
) -> Path:
    task_dir = ROOT / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    start = task_dir / "start.pdb"
    if start.exists():
        actual = count_pdb_atoms(start)
        if actual != expected_atoms:
            raise RuntimeError(
                f"Existing {start} has {actual} particles; expected {expected_atoms}"
            )
        print(f"Reuse existing {task_id}/start.pdb", flush=True)
    else:
        atoms = producer()
        if len(atoms.index) != expected_atoms:
            raise RuntimeError(
                f"{task_id}: producer returned {len(atoms.index)} particles; "
                f"expected {expected_atoms}"
            )
        atomic_write_pdb(atoms, start)
        print(f"Generated {task_id}/start.pdb", flush=True)
    install_wrapper(task_dir)
    return start


def copy_matched_start(task_id: str, parent: Path, expected_atoms: int) -> Path:
    task_dir = ROOT / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    start = task_dir / "start.pdb"
    if start.exists():
        if count_pdb_atoms(start) != expected_atoms or sha256(start) != sha256(parent):
            raise RuntimeError(f"Existing matched start differs from parent: {start}")
    else:
        shutil.copyfile(parent, start)
        print(f"Copied matched coordinates to {task_id}/start.pdb", flush=True)
    install_wrapper(task_dir)
    return start


def row_for(
    replicate: str,
    system: str,
    n_med1: int,
    n_oct4: int,
    n_dna: int,
    med1_fragments: int,
    box_nm: float,
    tier: str,
    tags: str,
    coord_seed: int,
    dynamics_seed: int,
    parent_task: str = "",
    coordinate_source: str = "random_insertion",
) -> dict[str, Any]:
    return {
        "task_id": f"{replicate}/{system}",
        "system": system,
        "replicate": replicate,
        "tier": tier,
        "series_tags": tags,
        "n_med1": n_med1,
        "n_oct4": n_oct4,
        "n_dna": n_dna,
        "med1_fragments": med1_fragments,
        "box_nm": f"{box_nm:.6f}",
        "coord_seed": coord_seed,
        "seed": dynamics_seed,
        "total_steps": TOTAL_STEPS,
        "output_interval": OUTPUT_INTERVAL,
        "parent_task": parent_task,
        "coordinate_source": coordinate_source,
    }


def write_molecule_map(row: dict[str, Any]) -> None:
    """Write zero-based particle ranges for physical molecules/fragments."""
    mapping: list[dict[str, Any]] = []
    particle_offset = 0
    fragments = int(row["med1_fragments"])
    boundaries = (0,) + MED1_CUTS_AFTER_RESIDUE[fragments] + (MED1_LENGTH,)
    for parent_index in range(1, int(row["n_med1"]) + 1):
        for fragment_index, (local_start, local_stop) in enumerate(
            zip(boundaries[:-1], boundaries[1:]), start=1
        ):
            mapping.append(
                {
                    "molecule_id": f"MED1_{parent_index:03d}_F{fragment_index}",
                    "molecule_type": "MED1",
                    "parent_med1": parent_index,
                    "fragment": fragment_index,
                    "particle_start": particle_offset + local_start,
                    "particle_stop_exclusive": particle_offset + local_stop,
                    "residue_start": local_start + 1,
                    "residue_stop_inclusive": local_stop,
                }
            )
        particle_offset += MED1_LENGTH
    for molecule_index in range(1, int(row["n_oct4"]) + 1):
        mapping.append(
            {
                "molecule_id": f"OCT4_{molecule_index:03d}",
                "molecule_type": "OCT4",
                "parent_med1": "",
                "fragment": 1,
                "particle_start": particle_offset,
                "particle_stop_exclusive": particle_offset + OCT4_LENGTH,
                "residue_start": 1,
                "residue_stop_inclusive": OCT4_LENGTH,
            }
        )
        particle_offset += OCT4_LENGTH
    for molecule_index in range(1, int(row["n_dna"]) + 1):
        mapping.append(
            {
                "molecule_id": f"DNA_{molecule_index:03d}",
                "molecule_type": "DNA",
                "parent_med1": "",
                "fragment": 1,
                "particle_start": particle_offset,
                "particle_stop_exclusive": particle_offset + DNA_LENGTH,
                "residue_start": 1,
                "residue_stop_inclusive": DNA_LENGTH,
            }
        )
        particle_offset += DNA_LENGTH
    fields = (
        "molecule_id",
        "molecule_type",
        "parent_med1",
        "fragment",
        "particle_start",
        "particle_stop_exclusive",
        "residue_start",
        "residue_stop_inclusive",
    )
    write_tsv(ROOT / "tasks" / row["task_id"] / "molecule_map.tsv", mapping, fields)


def build_design_rows_and_structures(
    med1_atoms: pd.DataFrame,
    oct4_atoms: pd.DataFrame,
    dna_atoms: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows_by_task: dict[str, dict[str, Any]] = {}

    for replicate_index, replicate in enumerate(REPLICATES, start=1):
        base_starts: dict[str, Path] = {}
        base_rows: dict[str, dict[str, Any]] = {}
        for system_index, system in enumerate(BASE_SYSTEMS, start=1):
            coord_seed = 2_026_080_100 + replicate_index * 100 + system_index
            dynamics_seed = 2_026_081_000 + replicate_index * 100 + system_index
            expected = system.n_med1 * MED1_LENGTH + system.n_oct4 * OCT4_LENGTH
            task_id = f"{replicate}/{system.name}"
            start = ensure_start(
                task_id,
                expected,
                lambda system=system, coord_seed=coord_seed: generate_protein_start(
                    system, med1_atoms, oct4_atoms, coord_seed
                ),
            )
            row = row_for(
                replicate,
                system.name,
                system.n_med1,
                system.n_oct4,
                0,
                1,
                system.box_nm,
                system.tier,
                system.tags,
                coord_seed,
                dynamics_seed,
            )
            base_starts[system.name] = start
            base_rows[system.name] = row
            rows_by_task[task_id] = row

        for base_name in ("M0O100", "M10O90", "M20O80", "M50O0"):
            base = BASE_BY_NAME[base_name]
            parent_row = base_rows[base_name]
            d1_name = f"{base_name}D1"
            task_id = f"{replicate}/{d1_name}"
            protein_particles = base.n_med1 * MED1_LENGTH + base.n_oct4 * OCT4_LENGTH
            expected = protein_particles + DNA_LENGTH
            dna_coord_seed = int(parent_row["coord_seed"]) + 50_000

            def add_dna(
                parent=base_starts[base_name],
                box_nm=base.box_nm,
                dna_coord_seed=dna_coord_seed,
            ) -> pd.DataFrame:
                np.random.seed(dna_coord_seed)
                proteins = parse_pdb(str(parent))
                result = insert_species(dna_atoms, 1, box_nm, proteins)
                assert result is not None
                return result

            d1_start = ensure_start(task_id, expected, add_dna)
            tags = "dna_addition"
            if base_name == "M10O90":
                tags += ",chain_length"
            elif base_name == "M50O0":
                tags += ",pure_med1"
            row = row_for(
                replicate,
                d1_name,
                base.n_med1,
                base.n_oct4,
                1,
                1,
                base.box_nm,
                "core",
                tags,
                dna_coord_seed,
                int(parent_row["seed"]),
                parent_task=f"{replicate}/{base_name}",
                coordinate_source=f"added_DNA_to:{replicate}/{base_name}",
            )
            rows_by_task[task_id] = row
            if base_name == "M10O90":
                base_starts[d1_name] = d1_start
                base_rows[d1_name] = row

        for fragments in (2, 4):
            for parent_name in ("M10O90", "M10O90D1"):
                split_name = f"{parent_name}-S{fragments}"
                parent_row = base_rows[parent_name]
                parent_start = base_starts[parent_name]
                expected = count_pdb_atoms(parent_start)
                task_id = f"{replicate}/{split_name}"
                copy_matched_start(task_id, parent_start, expected)
                rows_by_task[task_id] = row_for(
                    replicate,
                    split_name,
                    10,
                    90,
                    int(parent_row["n_dna"]),
                    fragments,
                    float(parent_row["box_nm"]),
                    "core",
                    "chain_length",
                    int(parent_row["coord_seed"]),
                    int(parent_row["seed"]),
                    parent_task=f"{replicate}/{parent_name}",
                    coordinate_source=f"exact_copy_of:{replicate}/{parent_name}",
                )

    ordered: list[dict[str, Any]] = []
    for replicate in REPLICATES:
        ordered.extend(rows_by_task[f"{replicate}/{name}"] for name in CORE_ORDER)
    for replicate in REPLICATES:
        ordered.extend(rows_by_task[f"{replicate}/{name}"] for name in OPTIONAL_ORDER)
    for row in ordered:
        write_molecule_map(row)
    return ordered


def design_matrix_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in ("M0O100", "M5O95", "M10O90", "M15O85", "M20O80", "M30O70"):
        rows.append({"series": "fixed_residue_composition", "comparison": "core", "condition": name, "system": name, "tier": "core"})
    for name in OPTIONAL_ORDER:
        rows.append({"series": "fixed_residue_composition", "comparison": "optional", "condition": name, "system": name, "tier": "optional"})
    for base_name in ("M0O100", "M10O90", "M20O80", "M50O0"):
        comparison = f"{base_name}_DNA_addition"
        rows.append({"series": "dna_addition", "comparison": comparison, "condition": "D0", "system": base_name, "tier": "core"})
        rows.append({"series": "dna_addition", "comparison": comparison, "condition": "+D1", "system": f"{base_name}D1", "tier": "core"})
    for dna_suffix, comparison in (("", "M10O90_D0_chain_length"), ("D1", "M10O90_D1_chain_length")):
        for condition, split_suffix in (("Full", ""), ("Split-2", "-S2"), ("Split-4", "-S4")):
            rows.append({"series": "med1_chain_length", "comparison": comparison, "condition": condition, "system": f"M10O90{dna_suffix}{split_suffix}", "tier": "core"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Rewrite manifests from already generated structures without inserting molecules.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="balanced_design_") as scratch_name:
        scratch = Path(scratch_name)
        med1_cg = scratch / "MED1_CA.pdb"
        oct4_cg = scratch / "OCT4_CA.pdb"
        dna_cg = scratch / "DNA_CG.pdb"
        atomistic_pdb_to_ca_pdb(str(ROOT / "inputs" / "MED1-alphafold.pdb"), str(med1_cg))
        atomistic_pdb_to_ca_pdb(str(ROOT / "inputs" / "OCT4.pdb"), str(oct4_cg))
        atomistic_pdb_to_nucleotide_pdb(str(ROOT / "inputs" / "all_atom_200bpDNA.pdb"), str(dna_cg))
        med1_atoms = parse_pdb(str(med1_cg))
        oct4_atoms = parse_pdb(str(oct4_cg))
        dna_atoms = parse_pdb(str(dna_cg))
        if len(med1_atoms.index) != MED1_LENGTH or len(oct4_atoms.index) != OCT4_LENGTH or len(dna_atoms.index) != DNA_LENGTH:
            raise RuntimeError(
                "Unexpected template lengths: "
                f"MED1={len(med1_atoms.index)}, OCT4={len(oct4_atoms.index)}, "
                f"DNA={len(dna_atoms.index)}"
            )
        if args.manifest_only:
            existing = list((ROOT / "tasks").glob("*/*/start.pdb"))
            if len(existing) != 51:
                raise RuntimeError(
                    f"--manifest-only requires all 51 start structures; "
                    f"found {len(existing)}"
                )
        rows = build_design_rows_and_structures(med1_atoms, oct4_atoms, dna_atoms)

    write_tsv(ROOT / "task_manifest.tsv", rows, MANIFEST_FIELDS)
    matrix_fields = ("series", "comparison", "condition", "system", "tier")
    write_tsv(ROOT / "design_matrix.tsv", design_matrix_rows(), matrix_fields)
    n_core = sum(row["tier"] == "core" for row in rows)
    n_optional = sum(row["tier"] == "optional" for row in rows)
    print(
        f"Generated design: {n_core} core tasks and {n_optional} optional tasks.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
