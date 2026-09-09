#!/usr/bin/env python3
"""Audit membership, graph-mask, and split-topology effects.

This analysis is deliberately separate from the publication cache.  It tests
whether Full/Split radial trends survive fixed membership definitions, applies
all three covalent block masks to every coordinate ensemble, and records every
interaction/exclusion removed by the MED1 split construction.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import warnings

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import run_balanced_analysis as base
import run_mechanism_extensions as mechanism


AUDIT_VERSION = "membership-mask-exclusion-audit-v1"
CHAIN_SYSTEMS = [
    "M10O90",
    "M10O90-S2",
    "M10O90-S4",
    "M10O90D1",
    "M10O90D1-S2",
    "M10O90D1-S4",
]
DNA_CHAIN_SYSTEMS = ["M10O90D1", "M10O90D1-S2", "M10O90D1-S4"]
COORDINATE_LABEL = {
    "M10O90": "Full",
    "M10O90-S2": "Split-2",
    "M10O90-S4": "Split-4",
    "M10O90D1": "Full",
    "M10O90D1-S2": "Split-2",
    "M10O90D1-S4": "Split-4",
}
DNA_LABEL = {
    "M10O90": "−DNA",
    "M10O90-S2": "−DNA",
    "M10O90-S4": "−DNA",
    "M10O90D1": "+DNA",
    "M10O90D1-S2": "+DNA",
    "M10O90D1-S4": "+DNA",
}
MASK_LABELS = ["Full", "Split-2", "Split-4"]
RADIAL_METRICS = [
    "physical_linked_bead_q90_nm",
    "physical_linked_block_com_q90_nm",
    "fixed_parent_linked_bead_q90_nm",
    "fixed_parent_linked_block_com_q90_nm",
    "external_block_linked_bead_q90_nm",
    "external_block_linked_block_com_q90_nm",
    "all_protein_bead_q90_nm",
    "all_protein_block_com_q90_nm",
]


def audit_paths(cache_dir: Path, task: base.Task) -> dict[str, Path]:
    stem = f"{task.replicate}__{task.system}"
    return {
        "meta": cache_dir / f"{stem}.json",
        "frame": cache_dir / f"{stem}.frames.csv",
    }


def audit_signature(task: base.Task) -> dict[str, object]:
    signature = base.task_signature(task)
    signature.update(
        {
            "audit_version": AUDIT_VERSION,
            "molecule_map_sha256": base.sha256(task.path / "molecule_map.tsv"),
            "fixed_blocks_after_residue": [354, 807, 1171],
            "membership_definitions": [
                "physical-chain augmented component",
                "fixed-parent augmented component with all parent residues",
                "fixed-block external-contact augmented component",
                "all protein residues",
            ],
            "covalent_masks": MASK_LABELS,
        }
    )
    return signature


def cache_valid(paths: dict[str, Path], signature: dict[str, object]) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        return json.loads(paths["meta"].read_text()) == signature
    except (OSError, json.JSONDecodeError):
        return False


def block_parent_nodes(task: base.Task) -> np.ndarray:
    med1 = np.repeat(np.arange(task.n_med1, dtype=np.int32), 4)
    oct4 = task.n_med1 + np.arange(task.n_oct4, dtype=np.int32)
    return np.concatenate([med1, oct4])


def covalent_edges_for_mask(task: base.Task, mask_label: str) -> np.ndarray:
    retained_boundaries = {
        "Full": (0, 1, 2),
        "Split-2": (0, 2),
        "Split-4": (),
    }[mask_label]
    edges = [
        (4 * parent + boundary, 4 * parent + boundary + 1)
        for parent in range(task.n_med1)
        for boundary in retained_boundaries
    ]
    return np.asarray(edges, dtype=np.int32).reshape(-1, 2)


def residue_weighted_lcc(
    n_nodes: int,
    external_edges: np.ndarray,
    covalent_edges: np.ndarray,
    residue_weights: np.ndarray,
) -> float:
    if len(covalent_edges):
        edges = np.unique(np.vstack([external_edges, covalent_edges]), axis=0)
    else:
        edges = external_edges
    labels, sizes = base.components(n_nodes, edges)
    residue_sizes = np.bincount(
        labels,
        weights=residue_weights,
        minlength=len(sizes),
    )
    return float(residue_sizes.max() / residue_weights.sum())


def component_with_anchor(
    n_nodes: int,
    edges: np.ndarray,
    direct_nodes: np.ndarray,
) -> np.ndarray:
    anchor = n_nodes
    dna_edges = np.column_stack(
        [direct_nodes, np.full(len(direct_nodes), anchor, dtype=np.int32)]
    ).astype(np.int32)
    augmented = (
        np.unique(np.vstack([edges, dna_edges]), axis=0)
        if len(edges) or len(dna_edges)
        else np.empty((0, 2), dtype=np.int32)
    )
    labels, _ = base.components(n_nodes + 1, augmented)
    return np.flatnonzero(labels[:n_nodes] == labels[anchor]).astype(np.int32)


def q90(values: np.ndarray) -> float:
    if not len(values):
        return float("nan")
    return mechanism.weighted_quantile(
        np.asarray(values, dtype=float),
        np.ones(len(values), dtype=float),
        0.90,
    )


def distance_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_count": float(len(values)),
        f"{prefix}_mean_nm": float(np.mean(values)) if len(values) else float("nan"),
        f"{prefix}_q90_nm": q90(values),
    }


def fixed_block_com_positions(
    positions_a: np.ndarray,
    task: base.Task,
    maps: dict[str, object],
) -> np.ndarray:
    positions: list[np.ndarray] = []
    for indices in maps["block_atom_indices"]:
        whole = base.unwrap_segment(positions_a[indices] / 10.0, task.box_nm)
        positions.append(np.mod(whole.mean(axis=0), task.box_nm))
    return np.vstack(positions)


def metrics_for_positions(
    positions_a: np.ndarray,
    task: base.Task,
    molecules: base.Molecules,
    maps: dict[str, object],
) -> dict[str, float]:
    atom_pairs = mechanism.atom_pairs_at_primary_cutoff(positions_a, task.box_nm)
    first = atom_pairs[:, 0]
    second = atom_pairs[:, 1]
    first_parent = maps["atom_parent"][first]
    second_parent = maps["atom_parent"][second]
    protein_pair = (first_parent >= 0) & (second_parent >= 0)
    cross_parent = protein_pair & (first_parent != second_parent)
    parent_edges = mechanism.unique_node_pairs(
        first_parent[cross_parent], second_parent[cross_parent]
    )
    block_edges = mechanism.unique_node_pairs(
        maps["atom_block"][first[cross_parent]],
        maps["atom_block"][second[cross_parent]],
    )

    row: dict[str, float] = {}
    for mask_label in MASK_LABELS:
        row[f"block_residue_LCC_mask_{mask_label.replace('-', '_')}"] = (
            residue_weighted_lcc(
                int(maps["n_block_nodes"]),
                block_edges,
                covalent_edges_for_mask(task, mask_label),
                maps["block_residue_weight"],
            )
        )

    if not task.n_dna:
        return row

    atom_type = molecules.molecule_type[molecules.atom_molecule]
    dna_nodes = molecules.dna_nodes
    if len(dna_nodes) != 1:
        raise RuntimeError(f"{task.replicate}/{task.system}: expected one DNA node")
    dna_node = int(dna_nodes[0])
    molecule_pairs = base.unique_molecule_pairs(atom_pairs, molecules.atom_molecule)
    physical_labels, _ = base.components(len(molecules.molecule_type), molecule_pairs)
    physical_linked = np.flatnonzero(
        (physical_labels == physical_labels[dna_node])
        & (molecules.molecule_type != base.TYPE_DNA)
    ).astype(np.int32)

    first_is_dna = atom_type[first] == base.TYPE_DNA
    second_is_dna = atom_type[second] == base.TYPE_DNA
    protein_atoms_touching_dna = np.concatenate(
        [second[first_is_dna & ~second_is_dna], first[second_is_dna & ~first_is_dna]]
    ).astype(np.int32)
    direct_parents = np.unique(maps["atom_parent"][protein_atoms_touching_dna])
    direct_parents = direct_parents[direct_parents >= 0]
    direct_blocks = np.unique(maps["atom_block"][protein_atoms_touching_dna])
    direct_blocks = direct_blocks[direct_blocks >= 0]

    fixed_parent_linked = component_with_anchor(
        int(maps["n_parent_nodes"]), parent_edges, direct_parents
    )
    external_block_linked = component_with_anchor(
        int(maps["n_block_nodes"]), block_edges, direct_blocks
    )

    wrapped_nm = np.mod(positions_a / 10.0, task.box_nm)
    dna_start = int(molecules.starts[dna_node])
    dna_stop = int(molecules.stops[dna_node])
    dna_tree = cKDTree(wrapped_nm[dna_start:dna_stop], boxsize=task.box_nm)
    protein_atom_indices = np.flatnonzero(maps["atom_parent"] >= 0)
    all_protein_distances = mechanism.nearest_dna_distances(
        wrapped_nm, dna_tree, protein_atom_indices
    )
    all_block_com = fixed_block_com_positions(positions_a, task, maps)
    all_block_distances = dna_tree.query(all_block_com, k=1, workers=1)[0]

    physical_atoms = np.flatnonzero(
        (maps["atom_parent"] >= 0)
        & np.isin(molecules.atom_molecule, physical_linked)
    )
    fixed_parent_atoms = np.flatnonzero(
        (maps["atom_parent"] >= 0)
        & np.isin(maps["atom_parent"], fixed_parent_linked)
    )
    external_block_atoms = np.flatnonzero(
        (maps["atom_block"] >= 0)
        & np.isin(maps["atom_block"], external_block_linked)
    )
    physical_blocks = np.flatnonzero(
        np.isin(maps["block_physical_molecule"], physical_linked)
    )
    block_parent = block_parent_nodes(task)
    fixed_parent_blocks = np.flatnonzero(np.isin(block_parent, fixed_parent_linked))

    row.update(
        distance_summary(
            "physical_linked_bead",
            mechanism.nearest_dna_distances(
                wrapped_nm, dna_tree, physical_atoms
            ),
        )
    )
    row.update(
        distance_summary(
            "physical_linked_block_com", all_block_distances[physical_blocks]
        )
    )
    row.update(
        distance_summary(
            "fixed_parent_linked_bead",
            mechanism.nearest_dna_distances(
                wrapped_nm, dna_tree, fixed_parent_atoms
            ),
        )
    )
    row.update(
        distance_summary(
            "fixed_parent_linked_block_com",
            all_block_distances[fixed_parent_blocks],
        )
    )
    row.update(
        distance_summary(
            "external_block_linked_bead",
            mechanism.nearest_dna_distances(
                wrapped_nm, dna_tree, external_block_atoms
            ),
        )
    )
    row.update(
        distance_summary(
            "external_block_linked_block_com",
            all_block_distances[external_block_linked],
        )
    )
    row.update(distance_summary("all_protein_bead", all_protein_distances))
    row.update(distance_summary("all_protein_block_com", all_block_distances))
    protein_residues = float(len(protein_atom_indices))
    row.update(
        {
            "physical_linked_residue_fraction": len(physical_atoms) / protein_residues,
            "fixed_parent_linked_parent_fraction": len(fixed_parent_linked)
            / int(maps["n_parent_nodes"]),
            "fixed_parent_linked_residue_fraction": len(fixed_parent_atoms)
            / protein_residues,
            "external_block_linked_residue_fraction": len(external_block_atoms)
            / protein_residues,
        }
    )
    return row


def analyze_task(
    task: base.Task,
    cache_dir: Path,
    *,
    recompute: bool,
) -> pd.DataFrame:
    paths = audit_paths(cache_dir, task)
    signature = audit_signature(task)
    if not recompute and cache_valid(paths, signature):
        return pd.read_csv(paths["frame"])

    print(f"Definition audit {task.replicate}/{task.system}", flush=True)
    molecules = base.read_molecule_map(task)
    maps = mechanism.fixed_resolution_maps(task, molecules)
    rows: list[dict[str, object]] = []

    start_universe = mda.Universe(str(task.path / "start.pdb"))
    initial = {
        "system": task.system,
        "replicate": task.replicate,
        "frame_scope": "initial",
        "frame": 0,
        "time_us": 0.0,
    }
    initial.update(
        metrics_for_positions(
            start_universe.atoms.positions.astype(np.float64),
            task,
            molecules,
            maps,
        )
    )
    rows.append(initial)

    universe = mda.Universe(
        str(task.path / "start.pdb"), str(task.path / "output.dcd")
    )
    for frame in range(base.PRODUCTION_START, base.PRODUCTION_END + 1, base.STRIDE):
        universe.trajectory[frame - 1]
        row = {
            "system": task.system,
            "replicate": task.replicate,
            "frame_scope": "production",
            "frame": frame,
            "time_us": frame * 0.001,
        }
        row.update(
            metrics_for_positions(
                universe.atoms.positions.astype(np.float64),
                task,
                molecules,
                maps,
            )
        )
        rows.append(row)

    table = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["frame"], index=False)
    paths["meta"].write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")
    return table


def summarize(framewise: pd.DataFrame) -> dict[str, pd.DataFrame]:
    production = framewise[framewise["frame_scope"] == "production"].copy()
    identifiers = {"system", "replicate", "frame_scope", "frame", "time_us"}
    metrics = [column for column in production.columns if column not in identifiers]
    repeat = (
        production.groupby(["system", "replicate"], observed=True)[metrics]
        .mean()
        .reset_index()
    )
    group = base.mean_sd(repeat, ["system"], metrics)
    initial = framewise[framewise["frame_scope"] == "initial"].copy()

    mask_rows: list[dict[str, object]] = []
    for row in repeat.itertuples(index=False):
        for mask_label in MASK_LABELS:
            mask_rows.append(
                {
                    "DNA_condition": DNA_LABEL[row.system],
                    "coordinate_ensemble": COORDINATE_LABEL[row.system],
                    "covalent_mask": mask_label,
                    "replicate": row.replicate,
                    "residue_LCC_fraction": getattr(
                        row, f"block_residue_LCC_mask_{mask_label.replace('-', '_')}"
                    ),
                }
            )
    mask_repeat = pd.DataFrame(mask_rows)
    mask_group = base.mean_sd(
        mask_repeat,
        ["DNA_condition", "coordinate_ensemble", "covalent_mask"],
        ["residue_LCC_fraction"],
    )

    mask_index = mask_repeat.set_index(
        ["DNA_condition", "replicate", "coordinate_ensemble", "covalent_mask"]
    )["residue_LCC_fraction"]
    decomposition_rows: list[dict[str, object]] = []
    for dna_condition in ["−DNA", "+DNA"]:
        for replicate in base.REPEATS:
            for split_label in ["Split-2", "Split-4"]:
                full_full = float(
                    mask_index.loc[(dna_condition, replicate, "Full", "Full")]
                )
                full_split_mask = float(
                    mask_index.loc[(dna_condition, replicate, "Full", split_label)]
                )
                split_split = float(
                    mask_index.loc[
                        (dna_condition, replicate, split_label, split_label)
                    ]
                )
                total = full_full - split_split
                direct_mask = full_full - full_split_mask
                coordinate = full_split_mask - split_split
                decomposition_rows.append(
                    {
                        "DNA_condition": dna_condition,
                        "replicate": replicate,
                        "contrast": f"Full−{split_label}",
                        "total_diagonal_effect": total,
                        "direct_mask_effect_on_Full_coordinates": direct_mask,
                        "coordinate_ensemble_effect_under_fragment_mask": coordinate,
                        "additivity_error": total - direct_mask - coordinate,
                    }
                )
    decomposition_repeat = pd.DataFrame(decomposition_rows)
    decomposition_group = base.mean_sd(
        decomposition_repeat,
        ["DNA_condition", "contrast"],
        [
            "total_diagonal_effect",
            "direct_mask_effect_on_Full_coordinates",
            "coordinate_ensemble_effect_under_fragment_mask",
            "additivity_error",
        ],
    )

    radial_index = repeat[repeat["system"].isin(DNA_CHAIN_SYSTEMS)].set_index(
        ["system", "replicate"]
    )
    radial_rows: list[dict[str, object]] = []
    for replicate in base.REPEATS:
        for metric in RADIAL_METRICS:
            values = [
                float(radial_index.loc[(system, replicate), metric])
                for system in DNA_CHAIN_SYSTEMS
            ]
            radial_rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "Full": values[0],
                    "Split_2": values[1],
                    "Split_4": values[2],
                    "Full_minus_Split_2": values[0] - values[1],
                    "Full_minus_Split_4": values[0] - values[2],
                }
            )
    radial_repeat = pd.DataFrame(radial_rows)
    radial_group = base.mean_sd(
        radial_repeat,
        ["metric"],
        [
            "Full",
            "Split_2",
            "Split_4",
            "Full_minus_Split_2",
            "Full_minus_Split_4",
        ],
    )

    initial_dna = initial[initial["system"].isin(DNA_CHAIN_SYSTEMS)].copy()
    initial_index = initial_dna.set_index(["system", "replicate"])
    initial_rows: list[dict[str, object]] = []
    for replicate in base.REPEATS:
        for metric in RADIAL_METRICS:
            values = [
                float(initial_index.loc[(system, replicate), metric])
                for system in DNA_CHAIN_SYSTEMS
            ]
            initial_rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "Full": values[0],
                    "Split_2": values[1],
                    "Split_4": values[2],
                    "range_across_definitions": max(values) - min(values),
                }
            )
    initial_comparison = pd.DataFrame(initial_rows)
    return {
        "framewise": framewise,
        "repeat": repeat,
        "group": group,
        "initial": initial,
        "initial_radial_comparison": initial_comparison,
        "mask_transfer_repeat": mask_repeat,
        "mask_transfer_group": mask_group,
        "mask_decomposition_repeat": decomposition_repeat,
        "mask_decomposition_group": decomposition_group,
        "radial_paired_repeat": radial_repeat,
        "radial_paired_group": radial_group,
    }


def retained_region_label(residue_one_based: int) -> str:
    for label, start, stop in [
        ("M1", 77, 232),
        ("M2", 252, 274),
        ("M3", 287, 349),
        ("M4", 361, 518),
    ]:
        if start <= residue_one_based <= stop:
            return label
    return "unrestrained"


def interaction_endpoints(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("a1", "a2", "a3", "a4")
        if column in table.columns
    ]


def exclusion_sources(parser: object) -> dict[tuple[int, int], set[str]]:
    sources: dict[tuple[int, int], set[str]] = {}
    endpoint_columns = {
        "protein_bonds": ("a1", "a2"),
        "protein_angles": ("a1", "a3"),
        "protein_dihedrals": ("a1", "a4"),
        "native_pairs": ("a1", "a2"),
    }
    for interaction_type, columns in endpoint_columns.items():
        table = getattr(parser, interaction_type)
        for row in table.itertuples(index=False):
            pair = tuple(sorted((int(getattr(row, columns[0])), int(getattr(row, columns[1])))))
            sources.setdefault(pair, set()).add(interaction_type)
    return sources


def audit_split_topology(package_root: Path) -> dict[str, pd.DataFrame]:
    sys.path.insert(0, str(package_root))
    from common import run_sim

    with tempfile.TemporaryDirectory(prefix="med1_definition_audit_") as scratch:
        full_parser, full_audit = run_sim.build_med1_parser(Path(scratch), 1)
    source_map = exclusion_sources(full_parser)
    full_exclusions = {
        tuple(sorted((int(row.a1), int(row.a2))))
        for row in full_parser.exclusions.itertuples(index=False)
    }
    detail_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for fragments, variant in [(2, "Split-2"), (4, "Split-4")]:
        cuts = tuple(run_sim.MED1_SPLIT_CUTS_AFTER_RESIDUE[fragments])
        parser = copy.deepcopy(full_parser)
        removed_counts = run_sim.split_med1_topology(parser, cuts)
        for interaction_type in (
            "protein_bonds",
            "protein_angles",
            "protein_dihedrals",
            "native_pairs",
        ):
            full_table = getattr(full_parser, interaction_type)
            columns = interaction_endpoints(full_table)
            count = 0
            for row_number, row in enumerate(full_table.itertuples(index=False)):
                atoms = [int(getattr(row, column)) for column in columns]
                fragment_ids = [sum(atom >= cut for cut in cuts) for atom in atoms]
                if len(set(fragment_ids)) == 1:
                    continue
                count += 1
                residues = [atom + 1 for atom in atoms]
                crossed = [
                    cut
                    for cut in cuts
                    if min(atoms) < cut <= max(atoms)
                ]
                detail = {
                    "variant": variant,
                    "interaction_type": interaction_type,
                    "source_row": row_number,
                    "crossed_cuts_after_residue": ";".join(map(str, crossed)),
                    "minimum_residue": min(residues),
                    "maximum_residue": max(residues),
                    "span_residues": max(residues) - min(residues),
                }
                for index, residue in enumerate(residues, start=1):
                    detail[f"residue_{index}"] = residue
                    detail[f"region_{index}"] = retained_region_label(residue)
                detail_rows.append(detail)
            if count != removed_counts[interaction_type]:
                raise RuntimeError(
                    f"{variant}/{interaction_type}: enumerated {count}, "
                    f"builder removed {removed_counts[interaction_type]}"
                )
            summary_rows.append(
                {
                    "variant": variant,
                    "interaction_type": interaction_type,
                    "removed_count": count,
                }
            )

        split_exclusions = {
            tuple(sorted((int(row.a1), int(row.a2))))
            for row in parser.exclusions.itertuples(index=False)
        }
        for first, second in sorted(full_exclusions - split_exclusions):
            crossed = [cut for cut in cuts if first < cut <= second]
            exclusion_rows.append(
                {
                    "variant": variant,
                    "residue_1": first + 1,
                    "residue_2": second + 1,
                    "region_1": retained_region_label(first + 1),
                    "region_2": retained_region_label(second + 1),
                    "crossed_cuts_after_residue": ";".join(map(str, crossed)),
                    "source_interaction_types": ";".join(
                        sorted(source_map.get((first, second), {"unknown"}))
                    ),
                }
            )
        expected = full_audit["after_split"] - len(full_exclusions - split_exclusions)
        if len(split_exclusions) != expected:
            raise RuntimeError(
                f"{variant}: exclusion reconstruction mismatch "
                f"{len(split_exclusions)} != {expected}"
            )

    detail = pd.DataFrame(detail_rows)
    exclusions = pd.DataFrame(exclusion_rows)
    summary = pd.DataFrame(summary_rows)
    cut_summary = (
        detail.groupby(
            ["variant", "interaction_type", "crossed_cuts_after_residue"],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("removed_count")
        .reset_index()
    )
    native_detail = detail[detail["interaction_type"] == "native_pairs"].copy()
    return {
        "split_interaction_summary": summary,
        "split_interaction_cut_summary": cut_summary,
        "split_interaction_removed_detail": detail,
        "split_native_pair_removed_detail": native_detail,
        "split_exclusion_removed_detail": exclusions,
    }


def validate(
    summaries: dict[str, pd.DataFrame],
    exclusion_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def record(check: str, value: object, required: str, passed: bool) -> None:
        rows.append(
            {"check": check, "value": value, "required": required, "passed": passed}
        )

    production_counts = (
        summaries["framewise"]
        .query("frame_scope == 'production'")
        .groupby(["system", "replicate"], observed=True)
        .size()
    )
    record(
        "each trajectory contributes 40 production frames",
        sorted(production_counts.unique().tolist()),
        "[40]",
        set(production_counts) == {40},
    )
    additivity = summaries["mask_decomposition_repeat"]["additivity_error"].abs().max()
    record(
        "mask decomposition is exactly additive",
        f"{additivity:.3g}",
        "<=1e-12",
        bool(additivity <= 1e-12),
    )
    initial = summaries["initial_radial_comparison"]
    fixed_metrics = [
        "fixed_parent_linked_bead_q90_nm",
        "fixed_parent_linked_block_com_q90_nm",
        "external_block_linked_bead_q90_nm",
        "external_block_linked_block_com_q90_nm",
        "all_protein_bead_q90_nm",
        "all_protein_block_com_q90_nm",
    ]
    initial_fixed_range = initial[initial["metric"].isin(fixed_metrics)][
        "range_across_definitions"
    ].max()
    record(
        "fixed-member initial metrics are invariant across Full/Split maps",
        f"max range={initial_fixed_range:.3g} nm",
        "<=1e-10 nm",
        bool(initial_fixed_range <= 1e-10),
    )
    cut_summary = exclusion_tables["split_interaction_cut_summary"]
    split4_native = int(
        cut_summary[
            (cut_summary["variant"] == "Split-4")
            & (cut_summary["interaction_type"] == "native_pairs")
        ]["removed_count"].sum()
    )
    split2_native = int(
        cut_summary[
            (cut_summary["variant"] == "Split-2")
            & (cut_summary["interaction_type"] == "native_pairs")
        ]["removed_count"].sum()
    )
    record(
        "native-pair removal matches task metadata",
        {"Split-2": split2_native, "Split-4": split4_native},
        "{'Split-2': 0, 'Split-4': 244}",
        split2_native == 0 and split4_native == 244,
    )
    table = pd.DataFrame(rows)
    if not table["passed"].all():
        raise RuntimeError("Definition audit validation failed:\n" + table.to_string(index=False))
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=base.PACKAGE_ROOT)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=base.DEFAULT_OUTPUT)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")
    warnings.filterwarnings("ignore", message="Unknown element .*", category=UserWarning)
    output = args.output.resolve()
    tables_dir = output / "tables"
    cache_dir = output / "cache" / AUDIT_VERSION
    tables_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = base.read_manifest(args.package_root.resolve(), args.data_root.resolve())
    selected = manifest[
        manifest["analysis_included"] & manifest["system"].isin(CHAIN_SYSTEMS)
    ]
    frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        task = base.task_from_row(pd.Series(row._asdict()))
        paths = audit_paths(cache_dir, task)
        signature = audit_signature(task)
        if args.aggregate_only:
            if not cache_valid(paths, signature):
                raise RuntimeError(f"Missing audit cache for {task.replicate}/{task.system}")
            frames.append(pd.read_csv(paths["frame"]))
        else:
            frames.append(analyze_task(task, cache_dir, recompute=args.recompute))
    framewise = pd.concat(frames, ignore_index=True)
    summaries = summarize(framewise)
    exclusion_tables = audit_split_topology(args.package_root.resolve())
    validation = validate(summaries, exclusion_tables)
    for name, table in {**summaries, **exclusion_tables}.items():
        table.to_csv(tables_dir / f"definition_audit_{name}.csv", index=False)
    validation.to_csv(tables_dir / "definition_audit_validation.csv", index=False)
    print(
        f"Wrote definition audits for {len(selected)} tasks; "
        f"validation checks={len(validation)}"
    )


if __name__ == "__main__":
    main()
