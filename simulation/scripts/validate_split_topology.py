#!/usr/bin/env python3
"""Validate that MED1 splitting changes connectivity but not composition."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.run_sim import build_med1_parser


parsers = {}
audits = {}
expected_exclusions = {1: 3651, 2: 3650, 4: 3404}
for fragments in (1, 2, 4):
    with tempfile.TemporaryDirectory(prefix=f"split{fragments}_") as scratch:
        parser, audit = build_med1_parser(Path(scratch), fragments)
    parsers[fragments] = parser
    audits[fragments] = audit
    if audit["after_split"] != expected_exclusions[fragments]:
        raise SystemExit(
            f"Split-{fragments} exclusions={audit['after_split']}, "
            f"expected={expected_exclusions[fragments]}"
        )

full_atoms = parsers[1].atoms
for fragments in (2, 4):
    split_atoms = parsers[fragments].atoms
    cuts = tuple(audits[fragments]["cuts_after_residue"])
    for column in ("name", "resname", "mass", "charge"):
        if not split_atoms[column].equals(full_atoms[column]):
            raise SystemExit(
                f"Split-{fragments} changes MED1 atom column {column}"
            )
    if len(split_atoms.index) != 1581:
        raise SystemExit(
            f"Split-{fragments} has {len(split_atoms.index)} residues, expected 1581"
        )
    removed = audits[fragments]["interactions_removed_by_split"]
    if removed["protein_bonds"] != fragments - 1:
        raise SystemExit(
            f"Split-{fragments} removed {removed['protein_bonds']} boundary bonds; "
            f"expected {fragments - 1}"
        )
    if fragments == 4 and removed["native_pairs"] != 244:
        raise SystemExit(
            f"Split-4 removed {removed['native_pairs']} cross-fragment native "
            "pairs; expected 244"
        )
    actual_bonds = {
        (int(row.a1), int(row.a2))
        for row in parsers[fragments].protein_bonds.itertuples()
    }
    expected_bonds = {
        (particle, particle + 1)
        for particle in range(1580)
        if particle + 1 not in cuts
    }
    if actual_bonds != expected_bonds:
        raise SystemExit(f"Split-{fragments} backbone components are incorrect")

    def fragment_of(particle: int) -> int:
        return sum(particle >= cut for cut in cuts)

    for attr_name in (
        "protein_bonds",
        "protein_angles",
        "protein_dihedrals",
        "native_pairs",
    ):
        table = getattr(parsers[fragments], attr_name)
        columns = [
            column for column in ("a1", "a2", "a3", "a4")
            if column in table.columns
        ]
        for row in table.itertuples(index=False):
            if len({fragment_of(int(getattr(row, column))) for column in columns}) != 1:
                raise SystemExit(
                    f"Split-{fragments} retains cross-fragment {attr_name}"
                )

print(json.dumps(audits, indent=2, sort_keys=True))
print("Split topology validation passed: sequence, mass, and charge are unchanged.")
