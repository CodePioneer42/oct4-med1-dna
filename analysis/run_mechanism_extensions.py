#!/usr/bin/env python3
"""Fixed-resolution network and DNA-reach analyses for the reframed paper.

This extension deliberately leaves the validated balanced-analysis cache
unchanged.  It revisits only the systems needed to compare Full, Split-2, and
Split-4 at a common graph resolution and to define DNA-associated material in
the same way across formulations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.lib.nsgrid import FastNS
from scipy.spatial import cKDTree

import run_balanced_analysis as base


EXTENSION_VERSION = "fixed-parent-block-dna-component-v5"
MED1_BLOCK_CUTS_ZERO_BASED = np.asarray([354, 807, 1171], dtype=np.int32)
DISTANCE_TYPES = ["all", "OCT4", "MED1"]
DISTANCE_SCOPES = [
    "bead",
    "parent_weighted_bead",
    "physical_chain_com",
    "block_com",
]
CHAIN_CONTROL_SYSTEMS = [
    "M10O90",
    "M10O90-S2",
    "M10O90-S4",
    "M10O90D1",
    "M10O90D1-S2",
    "M10O90D1-S4",
]
RADIAL_SYSTEMS = base.DNA_SYSTEMS + ["M10O90D1-S2", "M10O90D1-S4"]
EXTENSION_SYSTEMS = list(dict.fromkeys(CHAIN_CONTROL_SYSTEMS + RADIAL_SYSTEMS))


def extension_paths(cache_dir: Path, task: base.Task) -> dict[str, Path]:
    stem = f"{task.replicate}__{task.system}"
    return {
        "meta": cache_dir / f"{stem}.json",
        "frame": cache_dir / f"{stem}.frames.csv",
        "arrays": cache_dir / f"{stem}.arrays.npz",
    }


def extension_signature(task: base.Task) -> dict[str, object]:
    signature = base.task_signature(task)
    signature["extension_version"] = EXTENSION_VERSION
    signature["primary_cutoff_nm"] = base.PRIMARY_CUTOFF_NM
    signature["molecule_map_sha256"] = base.sha256(
        task.path / "molecule_map.tsv"
    )
    signature["med1_block_cuts_after_residue"] = [354, 807, 1171]
    signature["dna_association_definition"] = (
        "direct DNA contacts, exact second neighbors, and remaining third-plus "
        "DNA-linked proteins; augmented-graph component retained for reach analyses"
    )
    return signature


def extension_cache_valid(paths: dict[str, Path], signature: dict[str, object]) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        return json.loads(paths["meta"].read_text()) == signature
    except (OSError, json.JSONDecodeError):
        return False


def atom_pairs_at_primary_cutoff(
    positions_a: np.ndarray,
    box_nm: float,
) -> np.ndarray:
    box_a = np.asarray([box_nm * 10] * 3 + [90, 90, 90], dtype=np.float32)
    wrapped = np.mod(positions_a, box_nm * 10).astype(np.float32, copy=False)
    result = FastNS(
        base.PRIMARY_CUTOFF_NM * 10,
        wrapped,
        box=box_a,
        pbc=True,
    ).self_search()
    pairs = result.get_pairs().astype(np.int32, copy=False)
    distances = result.get_pair_distances().astype(np.float32, copy=False) / 10
    return pairs[distances < base.PRIMARY_CUTOFF_NM]


def unique_node_pairs(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(first):
        return np.empty((0, 2), dtype=np.int32)
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    keep = low != high
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.int32)
    return np.unique(
        np.column_stack([low[keep], high[keep]]).astype(np.int32),
        axis=0,
    )


def fixed_resolution_maps(
    task: base.Task,
    molecules: base.Molecules,
) -> dict[str, object]:
    """Map physical fragments to fixed parent and four-block graph nodes."""
    molecule_parent = np.full(len(molecules.molecule_type), -1, dtype=np.int32)
    molecule_block_type = np.full(len(molecules.molecule_type), -1, dtype=np.int8)
    med1_nodes = np.flatnonzero(molecules.molecule_type == base.TYPE_MED1)
    oct4_nodes = np.flatnonzero(molecules.molecule_type == base.TYPE_OCT4)
    molecule_parent[med1_nodes] = molecules.parent_med1[med1_nodes]
    molecule_parent[oct4_nodes] = task.n_med1 + np.arange(
        len(oct4_nodes), dtype=np.int32
    )
    molecule_block_type[med1_nodes] = base.TYPE_MED1
    molecule_block_type[oct4_nodes] = base.TYPE_OCT4

    atom_parent = molecule_parent[molecules.atom_molecule]
    atom_block = np.full(len(molecules.atom_molecule), -1, dtype=np.int32)
    med1_atom_mask = (
        molecules.molecule_type[molecules.atom_molecule] == base.TYPE_MED1
    )
    oct4_atom_mask = (
        molecules.molecule_type[molecules.atom_molecule] == base.TYPE_OCT4
    )
    med1_block_index = np.searchsorted(
        MED1_BLOCK_CUTS_ZERO_BASED,
        molecules.atom_residue[med1_atom_mask],
        side="right",
    )
    atom_block[med1_atom_mask] = (
        atom_parent[med1_atom_mask] * 4 + med1_block_index
    )
    oct4_parent_index = atom_parent[oct4_atom_mask] - task.n_med1
    atom_block[oct4_atom_mask] = 4 * task.n_med1 + oct4_parent_index

    n_parent_nodes = task.n_med1 + task.n_oct4
    n_block_nodes = 4 * task.n_med1 + task.n_oct4
    block_atom_indices = [
        np.flatnonzero(atom_block == node) for node in range(n_block_nodes)
    ]
    block_type = np.full(n_block_nodes, base.TYPE_OCT4, dtype=np.int8)
    block_type[: 4 * task.n_med1] = base.TYPE_MED1
    block_physical_molecule = np.asarray(
        [
            int(molecules.atom_molecule[indices[0]]) if len(indices) else -1
            for indices in block_atom_indices
        ],
        dtype=np.int32,
    )
    if np.any(block_physical_molecule < 0):
        raise RuntimeError(f"{task.replicate}/{task.system}: empty fixed block")
    block_residue_weight = np.full(n_block_nodes, base.OCT4_LENGTH, dtype=np.int32)
    med1_block_lengths = np.diff(np.asarray([0, 354, 807, 1171, 1581]))
    for parent_node in range(task.n_med1):
        observed = np.asarray(
            [len(block_atom_indices[parent_node * 4 + block]) for block in range(4)]
        )
        if not np.array_equal(observed, med1_block_lengths):
            raise RuntimeError(
                f"{task.replicate}/{task.system}: MED1 fixed-block lengths "
                f"{observed.tolist()} != {med1_block_lengths.tolist()}"
            )
        block_residue_weight[parent_node * 4 : parent_node * 4 + 4] = (
            med1_block_lengths
        )
    covalent_block_edges: list[tuple[int, int]] = []
    for parent_node in range(task.n_med1):
        for first_block in range(3):
            first_node = parent_node * 4 + first_block
            second_node = first_node + 1
            if (
                block_physical_molecule[first_node]
                == block_physical_molecule[second_node]
            ):
                covalent_block_edges.append((first_node, second_node))
    return {
        "molecule_parent": molecule_parent,
        "atom_parent": atom_parent,
        "atom_block": atom_block,
        "block_atom_indices": block_atom_indices,
        "block_type": block_type,
        "block_physical_molecule": block_physical_molecule,
        "block_residue_weight": block_residue_weight,
        "covalent_block_edges": np.asarray(
            covalent_block_edges, dtype=np.int32
        ).reshape(-1, 2),
        "n_parent_nodes": n_parent_nodes,
        "n_block_nodes": n_block_nodes,
        "med1_parent_nodes": np.arange(task.n_med1, dtype=np.int32),
        "oct4_parent_nodes": np.arange(
            task.n_med1,
            n_parent_nodes,
            dtype=np.int32,
        ),
    }


def graph_metrics(n_nodes: int, edges: np.ndarray) -> dict[str, float]:
    degrees = np.zeros(n_nodes, dtype=np.int32)
    if len(edges):
        np.add.at(degrees, edges.ravel(), 1)
    _, sizes = base.components(n_nodes, edges)
    return {
        "n_edges": float(len(edges)),
        "edge_density": base.safe_probability(
            len(edges), n_nodes * (n_nodes - 1) / 2
        ),
        "mean_partner_number": float(degrees.mean()),
        "largest_component_fraction": float(sizes.max() / n_nodes),
        "degrees": degrees,
    }


def nearest_dna_distances(
    wrapped_positions_nm: np.ndarray,
    dna_tree: cKDTree,
    atom_indices: np.ndarray,
) -> np.ndarray:
    if not len(atom_indices):
        return np.empty(0, dtype=float)
    return dna_tree.query(
        wrapped_positions_nm[atom_indices],
        k=1,
        workers=1,
    )[0]


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if not len(values):
        return float("nan")
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    target = quantile * cumulative[-1]
    return float(ordered_values[min(np.searchsorted(cumulative, target), len(values) - 1)])


def analyze_extension_task(
    task: base.Task,
    cache_dir: Path,
    *,
    recompute: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    paths = extension_paths(cache_dir, task)
    signature = extension_signature(task)
    if not recompute and extension_cache_valid(paths, signature):
        arrays_file = np.load(paths["arrays"])
        arrays = {key: arrays_file[key] for key in arrays_file.files}
        return pd.read_csv(paths["frame"]), arrays

    print(f"Extension analysis {task.replicate}/{task.system}", flush=True)
    molecules = base.read_molecule_map(task)
    maps = fixed_resolution_maps(task, molecules)
    universe = mda.Universe(
        str(task.path / "start.pdb"),
        str(task.path / "output.dcd"),
    )
    frames = list(
        range(base.PRODUCTION_START, base.PRODUCTION_END + 1, base.STRIDE)
    )
    histograms = {
        "radial_histogram": np.zeros(
            (
                len(frames),
                len(DISTANCE_SCOPES),
                len(DISTANCE_TYPES),
                len(base.PUBLICATION_DISTANCE_EDGES_NM) - 1,
            ),
            dtype=np.float64,
        ),
        "sampled_frames": np.asarray([len(frames)], dtype=np.int32),
    }
    frame_rows: list[dict[str, object]] = []
    protein_atom_mask = maps["atom_parent"] >= 0
    protein_total_residues = int(np.sum(protein_atom_mask))
    atom_type = molecules.molecule_type[molecules.atom_molecule]
    n_physical_proteins = len(molecules.protein_nodes)

    for frame_index, frame in enumerate(frames):
        universe.trajectory[frame - 1]
        positions_a = universe.atoms.positions.astype(np.float64)
        atom_pairs = atom_pairs_at_primary_cutoff(positions_a, task.box_nm)
        first = atom_pairs[:, 0]
        second = atom_pairs[:, 1]
        first_parent = maps["atom_parent"][first]
        second_parent = maps["atom_parent"][second]
        first_molecule = molecules.atom_molecule[first]
        second_molecule = molecules.atom_molecule[second]
        protein_pair_mask = (first_parent >= 0) & (second_parent >= 0)
        sibling_mask = (
            protein_pair_mask
            & (first_parent == second_parent)
            & (first_molecule != second_molecule)
            & (atom_type[first] == base.TYPE_MED1)
            & (atom_type[second] == base.TYPE_MED1)
        )
        cross_parent_mask = (
            protein_pair_mask
            & (first_parent != second_parent)
        )
        cross_first_parent = first_parent[cross_parent_mask]
        cross_second_parent = second_parent[cross_parent_mask]
        parent_edges = unique_node_pairs(
            cross_first_parent,
            cross_second_parent,
        )
        block_edges = unique_node_pairs(
            maps["atom_block"][first[cross_parent_mask]],
            maps["atom_block"][second[cross_parent_mask]],
        )
        parent = graph_metrics(int(maps["n_parent_nodes"]), parent_edges)
        block = graph_metrics(int(maps["n_block_nodes"]), block_edges)
        eligible_block_pairs = (
            int(maps["n_block_nodes"])
            * (int(maps["n_block_nodes"]) - 1)
            / 2
            - task.n_med1 * 6
        )
        block["edge_density"] = base.safe_probability(
            len(block_edges), eligible_block_pairs
        )
        continuity_block_edges = np.unique(
            np.vstack([block_edges, maps["covalent_block_edges"]]),
            axis=0,
        )
        continuity_labels, continuity_sizes = base.components(
            int(maps["n_block_nodes"]), continuity_block_edges
        )
        continuity_residue_sizes = np.bincount(
            continuity_labels,
            weights=maps["block_residue_weight"],
            minlength=len(continuity_sizes),
        )
        parent_degrees = np.asarray(parent.pop("degrees"))
        block.pop("degrees")
        med1_parent_nodes = np.asarray(maps["med1_parent_nodes"])
        oct4_parent_nodes = np.asarray(maps["oct4_parent_nodes"])
        cross_endpoint_types = np.concatenate(
            [atom_type[first[cross_parent_mask]], atom_type[second[cross_parent_mask]]]
        )
        sibling_molecule_edges = unique_node_pairs(
            first_molecule[sibling_mask], second_molecule[sibling_mask]
        )

        row: dict[str, object] = {
            "system": task.system,
            "replicate": task.replicate,
            "frame": frame,
            "time_us": frame * 0.001,
            "parent_n_nodes": int(maps["n_parent_nodes"]),
            "parent_n_edges": parent["n_edges"],
            "parent_edge_density": parent["edge_density"],
            "parent_mean_partner_number": parent["mean_partner_number"],
            "parent_largest_component_fraction": parent[
                "largest_component_fraction"
            ],
            "parent_free_chain_fraction": float(np.mean(parent_degrees == 0)),
            "MED1_parent_mean_partner_number": (
                float(parent_degrees[med1_parent_nodes].mean())
                if len(med1_parent_nodes)
                else float("nan")
            ),
            "OCT4_parent_mean_partner_number": (
                float(parent_degrees[oct4_parent_nodes].mean())
                if len(oct4_parent_nodes)
                else float("nan")
            ),
            "block_n_nodes": int(maps["n_block_nodes"]),
            "block_n_eligible_external_pairs": eligible_block_pairs,
            "block_n_edges": block["n_edges"],
            "block_edge_density": block["edge_density"],
            "block_mean_partner_number": block["mean_partner_number"],
            "block_largest_component_fraction": block[
                "largest_component_fraction"
            ],
            "continuity_block_n_edges": len(continuity_block_edges),
            "continuity_block_n_covalent_edges": len(
                maps["covalent_block_edges"]
            ),
            "continuity_block_largest_component_fraction": (
                continuity_sizes.max() / int(maps["n_block_nodes"])
            ),
            "continuity_block_residue_LCC_fraction": (
                continuity_residue_sizes.max()
                / np.sum(maps["block_residue_weight"])
            ),
            "cross_parent_residue_contacts_per_100_residues": (
                int(np.sum(cross_parent_mask)) * 100.0 / protein_total_residues
            ),
            "MED1_cross_parent_contact_incidences_per_100_residues": (
                np.sum(cross_endpoint_types == base.TYPE_MED1)
                * 100.0
                / (task.n_med1 * base.MED1_LENGTH)
                if task.n_med1
                else float("nan")
            ),
            "OCT4_cross_parent_contact_incidences_per_100_residues": (
                np.sum(cross_endpoint_types == base.TYPE_OCT4)
                * 100.0
                / (task.n_oct4 * base.OCT4_LENGTH)
                if task.n_oct4
                else float("nan")
            ),
            "n_sibling_MED1_fragment_edges_excluded": len(
                sibling_molecule_edges
            ),
            "n_sibling_MED1_residue_pairs_excluded": int(
                np.sum(sibling_mask)
            ),
        }

        molecule_pairs = base.unique_molecule_pairs(
            atom_pairs,
            molecules.atom_molecule,
        )
        dna_nodes = molecules.dna_nodes
        if len(dna_nodes):
            augmented_labels, _ = base.components(
                len(molecules.molecule_type), molecule_pairs
            )
            dna_node = int(dna_nodes[0])
            linked_molecules = np.flatnonzero(
                augmented_labels == augmented_labels[dna_node]
            )
            linked_proteins = linked_molecules[
                molecules.molecule_type[linked_molecules] != base.TYPE_DNA
            ]
            direct_neighbors: set[int] = set()
            for first_molecule, second_molecule in molecule_pairs:
                if int(first_molecule) == dna_node:
                    direct_neighbors.add(int(second_molecule))
                elif int(second_molecule) == dna_node:
                    direct_neighbors.add(int(first_molecule))
            direct_proteins = np.asarray(
                sorted(
                    node
                    for node in direct_neighbors
                    if molecules.molecule_type[node] != base.TYPE_DNA
                ),
                dtype=np.int32,
            )
            direct_set = set(direct_proteins.tolist())
            second_neighbors: set[int] = set()
            for first_molecule, second_molecule in molecule_pairs:
                first_node = int(first_molecule)
                second_node = int(second_molecule)
                first_is_protein = (
                    molecules.molecule_type[first_node] != base.TYPE_DNA
                )
                second_is_protein = (
                    molecules.molecule_type[second_node] != base.TYPE_DNA
                )
                if not (first_is_protein and second_is_protein):
                    continue
                if first_node in direct_set and second_node not in direct_set:
                    second_neighbors.add(second_node)
                elif second_node in direct_set and first_node not in direct_set:
                    second_neighbors.add(first_node)
            second_proteins = np.asarray(sorted(second_neighbors), dtype=np.int32)
            third_plus_neighbors = (
                set(linked_proteins.tolist()) - direct_set - second_neighbors
            )
            third_plus_proteins = np.asarray(
                sorted(third_plus_neighbors), dtype=np.int32
            )
            linked_residues = int(molecules.molecule_length[linked_proteins].sum())
            direct_residues = int(molecules.molecule_length[direct_proteins].sum())
            second_residues = int(molecules.molecule_length[second_proteins].sum())
            third_plus_residues = int(
                molecules.molecule_length[third_plus_proteins].sum()
            )
            linked_parents = np.unique(maps["molecule_parent"][linked_proteins])
            direct_parents = np.unique(maps["molecule_parent"][direct_proteins])
            second_parents = np.unique(maps["molecule_parent"][second_proteins])
            third_plus_parents = np.unique(
                maps["molecule_parent"][third_plus_proteins]
            )
            row.update(
                {
                    "DNA_direct_MED1_parent_count": int(
                        np.sum(direct_parents < task.n_med1)
                    ),
                    "DNA_direct_OCT4_parent_count": int(
                        np.sum(direct_parents >= task.n_med1)
                    ),
                    "DNA_linked_physical_fraction": len(linked_proteins)
                    / n_physical_proteins,
                    "DNA_direct_physical_fraction": len(direct_proteins)
                    / n_physical_proteins,
                    "DNA_second_physical_fraction": len(second_proteins)
                    / n_physical_proteins,
                    "DNA_third_plus_physical_fraction": len(third_plus_proteins)
                    / n_physical_proteins,
                    "DNA_linked_residue_fraction": linked_residues
                    / protein_total_residues,
                    "DNA_direct_residue_fraction": direct_residues
                    / protein_total_residues,
                    "DNA_second_residue_fraction": second_residues
                    / protein_total_residues,
                    "DNA_third_plus_residue_fraction": third_plus_residues
                    / protein_total_residues,
                    "DNA_linked_parent_fraction": len(linked_parents)
                    / int(maps["n_parent_nodes"]),
                    "DNA_direct_parent_fraction": len(direct_parents)
                    / int(maps["n_parent_nodes"]),
                    "DNA_second_parent_fraction": len(second_parents)
                    / int(maps["n_parent_nodes"]),
                    "DNA_third_plus_parent_fraction": len(third_plus_parents)
                    / int(maps["n_parent_nodes"]),
                }
            )

            wrapped_positions_nm = np.mod(positions_a / 10.0, task.box_nm)
            dna_start = int(molecules.starts[dna_node])
            dna_stop = int(molecules.stops[dna_node])
            dna_tree = cKDTree(
                wrapped_positions_nm[dna_start:dna_stop],
                boxsize=task.box_nm,
            )
            linked_atom_mask = protein_atom_mask & np.isin(
                molecules.atom_molecule,
                linked_proteins,
            )
            linked_atom_indices = np.flatnonzero(linked_atom_mask)
            bead_distances = nearest_dna_distances(
                wrapped_positions_nm,
                dna_tree,
                linked_atom_indices,
            )
            bead_types = atom_type[linked_atom_indices]
            bead_parents = maps["atom_parent"][linked_atom_indices]
            parent_weights = np.zeros(len(linked_atom_indices), dtype=float)
            for parent_node in np.unique(bead_parents):
                parent_mask = bead_parents == parent_node
                parent_weights[parent_mask] = 1.0 / np.sum(parent_mask)

            chain_com = []
            chain_types = []
            for molecule_node in linked_proteins:
                start = int(molecules.starts[molecule_node])
                stop = int(molecules.stops[molecule_node])
                whole = base.unwrap_segment(
                    positions_a[start:stop] / 10.0,
                    task.box_nm,
                )
                chain_com.append(np.mod(whole.mean(axis=0), task.box_nm))
                chain_types.append(int(molecules.molecule_type[molecule_node]))
            chain_com_array = (
                np.vstack(chain_com)
                if chain_com
                else np.empty((0, 3), dtype=float)
            )
            chain_distances = (
                dna_tree.query(chain_com_array, k=1, workers=1)[0]
                if len(chain_com_array)
                else np.empty(0, dtype=float)
            )
            chain_types_array = np.asarray(chain_types, dtype=np.int8)

            linked_block_nodes = np.flatnonzero(
                np.isin(maps["block_physical_molecule"], linked_proteins)
            )
            block_com = []
            for block_node in linked_block_nodes:
                indices = maps["block_atom_indices"][int(block_node)]
                whole = base.unwrap_segment(
                    positions_a[indices] / 10.0,
                    task.box_nm,
                )
                block_com.append(np.mod(whole.mean(axis=0), task.box_nm))
            block_com_array = (
                np.vstack(block_com)
                if block_com
                else np.empty((0, 3), dtype=float)
            )
            block_distances = (
                dna_tree.query(block_com_array, k=1, workers=1)[0]
                if len(block_com_array)
                else np.empty(0, dtype=float)
            )
            block_types = maps["block_type"][linked_block_nodes]

            scope_values = {
                "bead": (
                    bead_distances,
                    bead_types,
                    np.ones(len(bead_distances), dtype=float),
                ),
                "parent_weighted_bead": (
                    bead_distances,
                    bead_types,
                    parent_weights,
                ),
                "physical_chain_com": (
                    chain_distances,
                    chain_types_array,
                    np.ones(len(chain_distances), dtype=float),
                ),
                "block_com": (
                    block_distances,
                    block_types,
                    np.ones(len(block_distances), dtype=float),
                ),
            }
            for scope_index, scope in enumerate(DISTANCE_SCOPES):
                values_all, types_all, weights_all = scope_values[scope]
                for type_index, type_label in enumerate(DISTANCE_TYPES):
                    if type_label == "all":
                        mask = np.ones(len(values_all), dtype=bool)
                    else:
                        mask = types_all == base.TYPE_BY_NAME[type_label]
                    values = values_all[mask]
                    weights = weights_all[mask]
                    prefix = f"DNA_linked_{scope}_{type_label}"
                    row[f"{prefix}_count"] = len(values)
                    row[f"{prefix}_total_weight"] = float(weights.sum())
                    row[f"{prefix}_mean_nm"] = (
                        float(np.average(values, weights=weights))
                        if len(values)
                        else float("nan")
                    )
                    row[f"{prefix}_q90_nm"] = weighted_quantile(
                        values,
                        weights,
                        0.90,
                    )
                    histograms["radial_histogram"][
                        frame_index, scope_index, type_index
                    ] = np.histogram(
                        values,
                        bins=base.PUBLICATION_DISTANCE_EDGES_NM,
                        weights=weights,
                    )[0]
        frame_rows.append(row)

    frame_table = pd.DataFrame(frame_rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_table.to_csv(paths["frame"], index=False)
    np.savez_compressed(paths["arrays"], **histograms)
    paths["meta"].write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")
    return frame_table, histograms


def summarize_extension(
    framewise: pd.DataFrame,
    arrays_by_task: dict[tuple[str, str], dict[str, np.ndarray]],
) -> dict[str, pd.DataFrame]:
    identifier_columns = {"system", "replicate", "frame", "time_us"}
    metrics = [
        column
        for column in framewise.columns
        if column not in identifier_columns
    ]
    repeat = (
        framewise.groupby(["system", "replicate"], observed=True)[metrics]
        .mean()
        .reset_index()
    )
    group = base.mean_sd(repeat, ["system"], metrics)

    histogram_rows: list[dict[str, object]] = []
    for (system, replicate), arrays in arrays_by_task.items():
        histogram = arrays["radial_histogram"].astype(float)
        for scope_index, scope in enumerate(DISTANCE_SCOPES):
            for type_index, type_label in enumerate(DISTANCE_TYPES):
                frame_counts = histogram[:, scope_index, type_index]
                totals = frame_counts.sum(axis=1)
                valid = totals > 0
                if np.any(valid):
                    frame_probabilities = (
                        frame_counts[valid] / totals[valid, np.newaxis]
                    )
                    probability = frame_probabilities.mean(axis=0)
                else:
                    probability = np.full(frame_counts.shape[1], np.nan)
                mean_count = frame_counts.mean(axis=0)
                cumulative = np.cumsum(probability)
                for bin_index in range(len(probability)):
                    histogram_rows.append(
                        {
                            "system": system,
                            "replicate": replicate,
                            "distance_scope": scope,
                            "protein_type": type_label,
                            "bin_start_nm": base.PUBLICATION_DISTANCE_EDGES_NM[
                                bin_index
                            ],
                            "bin_end_nm": base.PUBLICATION_DISTANCE_EDGES_NM[
                                bin_index + 1
                            ],
                            "frame_mean_count": mean_count[bin_index],
                            "frame_mean_probability": probability[bin_index],
                            "cumulative_probability": cumulative[bin_index],
                            "valid_frames": int(np.sum(valid)),
                        }
                    )
    histogram_repeat = pd.DataFrame(histogram_rows)
    histogram_group = base.mean_sd(
        histogram_repeat,
        [
            "system",
            "distance_scope",
            "protein_type",
            "bin_start_nm",
            "bin_end_nm",
        ],
        ["frame_mean_count", "frame_mean_probability", "cumulative_probability"],
    )

    effect_rows: list[dict[str, object]] = []
    effect_metrics = [
        "continuity_block_residue_LCC_fraction",
        "parent_mean_partner_number",
        "parent_edge_density",
        "block_mean_partner_number",
        "block_edge_density",
        "cross_parent_residue_contacts_per_100_residues",
        "MED1_cross_parent_contact_incidences_per_100_residues",
        "OCT4_cross_parent_contact_incidences_per_100_residues",
        "n_sibling_MED1_fragment_edges_excluded",
        "n_sibling_MED1_residue_pairs_excluded",
    ]
    for dna_label, systems in [
        ("−DNA", ["M10O90", "M10O90-S2", "M10O90-S4"]),
        ("+DNA", ["M10O90D1", "M10O90D1-S2", "M10O90D1-S4"]),
    ]:
        indexed = repeat[repeat["system"].isin(systems)].set_index(
            ["system", "replicate"]
        )
        for replicate in base.REPEATS:
            for metric in effect_metrics:
                full = float(indexed.loc[(systems[0], replicate), metric])
                split2 = float(indexed.loc[(systems[1], replicate), metric])
                split4 = float(indexed.loc[(systems[2], replicate), metric])
                effect_rows.append(
                    {
                        "DNA_condition": dna_label,
                        "replicate": replicate,
                        "metric": metric,
                        "Full": full,
                        "Split_2": split2,
                        "Split_4": split4,
                        "Full_minus_Split_2": full - split2,
                        "Full_minus_Split_4": full - split4,
                    }
                )
    effects = pd.DataFrame(effect_rows)
    effect_group = base.mean_sd(
        effects,
        ["DNA_condition", "metric"],
        ["Full", "Split_2", "Split_4", "Full_minus_Split_2", "Full_minus_Split_4"],
    )
    did_rows: list[dict[str, object]] = []
    for metric in effect_metrics:
        sub = effects[effects["metric"] == metric].set_index(
            ["DNA_condition", "replicate"]
        )
        for replicate in base.REPEATS:
            without = float(sub.loc[("−DNA", replicate), "Full_minus_Split_4"])
            with_dna = float(sub.loc[("+DNA", replicate), "Full_minus_Split_4"])
            did_rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "Full_minus_Split_4_without_DNA": without,
                    "Full_minus_Split_4_with_DNA": with_dna,
                    "difference_in_differences": with_dna - without,
                }
            )
    did = pd.DataFrame(did_rows)
    did_group = base.mean_sd(
        did,
        ["metric"],
        [
            "Full_minus_Split_4_without_DNA",
            "Full_minus_Split_4_with_DNA",
            "difference_in_differences",
        ],
    )
    radial_pair_rows: list[dict[str, object]] = []
    radial_metrics = [
        f"DNA_linked_{scope}_{protein_type}_q90_nm"
        for scope in DISTANCE_SCOPES
        for protein_type in DISTANCE_TYPES
    ]
    radial_systems = ["M10O90D1", "M10O90D1-S2", "M10O90D1-S4"]
    radial_index = repeat[repeat["system"].isin(radial_systems)].set_index(
        ["system", "replicate"]
    )
    for replicate in base.REPEATS:
        for metric in radial_metrics:
            full = float(radial_index.loc[(radial_systems[0], replicate), metric])
            split2 = float(radial_index.loc[(radial_systems[1], replicate), metric])
            split4 = float(radial_index.loc[(radial_systems[2], replicate), metric])
            radial_pair_rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "Full": full,
                    "Split_2": split2,
                    "Split_4": split4,
                    "Split_2_minus_Full": split2 - full,
                    "Split_4_minus_Full": split4 - full,
                }
            )
    radial_paired = pd.DataFrame(radial_pair_rows)
    radial_paired_group = base.mean_sd(
        radial_paired,
        ["metric"],
        ["Full", "Split_2", "Split_4", "Split_2_minus_Full", "Split_4_minus_Full"],
    )
    return {
        "framewise": framewise,
        "repeat": repeat,
        "group": group,
        "radial_histogram_repeat": histogram_repeat,
        "radial_histogram_group": histogram_group,
        "split_effects_repeat": effects,
        "split_effects_group": effect_group,
        "split_did_repeat": did,
        "split_did_group": did_group,
        "chain_continuity_radial_paired": radial_paired,
        "chain_continuity_radial_paired_group": radial_paired_group,
    }


def load_cached_tasks(
    manifest: pd.DataFrame,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, np.ndarray]]]:
    frames: list[pd.DataFrame] = []
    arrays_by_task: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for row in manifest[
        manifest["analysis_included"] & manifest["system"].isin(EXTENSION_SYSTEMS)
    ].itertuples(index=False):
        task = base.task_from_row(pd.Series(row._asdict()))
        paths = extension_paths(cache_dir, task)
        signature = extension_signature(task)
        if not extension_cache_valid(paths, signature):
            raise RuntimeError(
                f"Missing or stale extension cache for {task.replicate}/{task.system}"
            )
        frames.append(pd.read_csv(paths["frame"]))
        arrays_file = np.load(paths["arrays"])
        arrays_by_task[(task.system, task.replicate)] = {
            key: arrays_file[key] for key in arrays_file.files
        }
    return pd.concat(frames, ignore_index=True), arrays_by_task


def validate_extension(
    framewise: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    arrays_by_task: dict[tuple[str, str], dict[str, np.ndarray]],
    balanced_framewise_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def record(check: str, value: object, required: str, passed: bool) -> None:
        rows.append(
            {
                "check": check,
                "value": value,
                "required": required,
                "passed": bool(passed),
            }
        )

    task_counts = framewise.groupby(["system", "replicate"], observed=True).size()
    record(
        "all mechanism tasks have 40 sampled frames",
        f"min={int(task_counts.min())}, max={int(task_counts.max())}",
        "40",
        bool((task_counts == 40).all()),
    )
    chain = framewise[framewise["system"].isin(CHAIN_CONTROL_SYSTEMS)]
    record(
        "Full/Split parent graph has fixed 100 nodes",
        sorted(chain["parent_n_nodes"].unique().tolist()),
        "[100]",
        set(chain["parent_n_nodes"]) == {100},
    )
    parent_free = chain["parent_free_chain_fraction"].to_numpy(float)
    record(
        "Full/Split free-parent fractions are bounded probabilities",
        f"min={parent_free.min():.4f}, max={parent_free.max():.4f}",
        "0 <= value <= 1",
        bool(np.all((parent_free >= 0) & (parent_free <= 1))),
    )
    dna_chain = chain[chain["DNA_direct_MED1_parent_count"].notna()]
    med1_bound = dna_chain["DNA_direct_MED1_parent_count"].to_numpy(float)
    oct4_bound = dna_chain["DNA_direct_OCT4_parent_count"].to_numpy(float)
    occupancy_valid = (
        np.all((med1_bound >= 0) & (med1_bound <= 10))
        and np.all((oct4_bound >= 0) & (oct4_bound <= 90))
        and np.allclose(med1_bound, np.round(med1_bound))
        and np.allclose(oct4_bound, np.round(oct4_bound))
    )
    record(
        "direct DNA occupancy respects the 10-MED1-parent/90-OCT4 design",
        (
            f"MED1 range={med1_bound.min():.0f}-{med1_bound.max():.0f}; "
            f"OCT4 range={oct4_bound.min():.0f}-{oct4_bound.max():.0f}"
        ),
        "integer counts within 0-10 and 0-90",
        bool(occupancy_valid),
    )
    record(
        "Full/Split block graph has fixed 130 nodes",
        sorted(chain["block_n_nodes"].unique().tolist()),
        "[130]",
        set(chain["block_n_nodes"]) == {130},
    )
    record(
        "Full/Split block external graph has fixed 8,325 eligible pairs",
        sorted(chain["block_n_eligible_external_pairs"].unique().tolist()),
        "[8325]",
        set(chain["block_n_eligible_external_pairs"]) == {8325.0},
    )
    covalent_expected = {
        "M10O90": 30,
        "M10O90D1": 30,
        "M10O90-S2": 20,
        "M10O90D1-S2": 20,
        "M10O90-S4": 0,
        "M10O90D1-S4": 0,
    }
    covalent_observed = (
        chain.groupby("system", observed=True)[
            "continuity_block_n_covalent_edges"
        ]
        .first()
        .to_dict()
    )
    record(
        "continuity-aware block graph contains only retained covalent links",
        covalent_observed,
        str(covalent_expected),
        covalent_observed == covalent_expected,
    )
    additivity_error = 0.0
    radial_valid_frames: list[int] = []
    for (system, _), arrays in arrays_by_task.items():
        histogram = arrays["radial_histogram"].astype(float)
        additivity_error = max(
            additivity_error,
            float(
                np.max(
                    np.abs(
                        histogram[:, :, 0]
                        - histogram[:, :, 1]
                        - histogram[:, :, 2]
                    )
                )
            ),
        )
        if system in RADIAL_SYSTEMS:
            totals = histogram[:, DISTANCE_SCOPES.index("bead"), 0].sum(axis=1)
            radial_valid_frames.append(int(np.sum(totals > 0)))
    record(
        "all-protein radial histograms equal OCT4 plus MED1",
        f"max absolute error={additivity_error:.3g}",
        "0",
        np.isclose(additivity_error, 0.0),
    )
    record(
        "every radial trajectory has 40 valid DNA-linked frames",
        sorted(set(radial_valid_frames)),
        "[40]",
        bool(radial_valid_frames) and set(radial_valid_frames) == {40},
    )
    histogram_repeat = summaries["radial_histogram_repeat"]
    final_bins = histogram_repeat[np.isinf(histogram_repeat["bin_end_nm"])]
    finite_final = final_bins["cumulative_probability"].dropna().to_numpy(float)
    cdf_error = float(np.max(np.abs(finite_final - 1.0)))
    record(
        "frame-normalized trajectory CDFs end at one",
        f"max absolute error={cdf_error:.3g}",
        "<=1e-12",
        cdf_error <= 1e-12,
    )
    bead_count_totals = (
        histogram_repeat[
            (histogram_repeat["distance_scope"] == "bead")
            & (histogram_repeat["protein_type"] == "all")
        ]
        .groupby(["system", "replicate"], as_index=False, observed=True)[
            "frame_mean_count"
        ]
        .sum()
    )
    bead_count_expected = summaries["repeat"][
        [
            "system",
            "replicate",
            "DNA_linked_bead_all_total_weight",
        ]
    ]
    bead_count_check = bead_count_totals.merge(
        bead_count_expected,
        on=["system", "replicate"],
        validate="one_to_one",
    )
    bead_count_error = float(
        np.max(
            np.abs(
                bead_count_check["frame_mean_count"]
                - bead_count_check["DNA_linked_bead_all_total_weight"]
            )
        )
    )
    record(
        "absolute radial bead-count profiles reproduce linked-bead totals",
        f"max absolute error={bead_count_error:.3g}",
        "<=1e-12",
        bead_count_error <= 1e-12,
    )
    group = summaries["group"].set_index("system")
    radial_n = group.loc[
        RADIAL_SYSTEMS, "DNA_linked_bead_all_q90_nm_n"
    ].to_numpy(int)
    record(
        "radial group statistics use three trajectories",
        sorted(set(radial_n.tolist())),
        "[3]",
        set(radial_n) == {3},
    )
    dna_rows = framewise[framewise["DNA_third_plus_physical_fraction"].notna()]
    neighbor_partition_error = float(
        np.max(
            np.abs(
                dna_rows["DNA_direct_physical_fraction"]
                + dna_rows["DNA_second_physical_fraction"]
                + dna_rows["DNA_third_plus_physical_fraction"]
                - dna_rows["DNA_linked_physical_fraction"]
            )
        )
    )
    record(
        "direct, second, and third-plus neighbors partition the DNA-linked component",
        f"maximum partition error={neighbor_partition_error:.3g}",
        "<=1e-12",
        neighbor_partition_error <= 1e-12,
    )
    if balanced_framewise_path.exists():
        balanced = pd.read_csv(balanced_framewise_path)
        merged = framewise[framewise["system"].isin(RADIAL_SYSTEMS)].merge(
            balanced[
                [
                    "system",
                    "replicate",
                    "frame",
                    "DNA_component_protein_fraction",
                ]
            ],
            on=["system", "replicate", "frame"],
            validate="one_to_one",
        )
        component_error = float(
            np.max(
                np.abs(
                    merged["DNA_linked_physical_fraction"]
                    - merged["DNA_component_protein_fraction"]
                )
            )
        )
        record(
            "DNA-linked physical fractions reproduce balanced component counts",
            f"max absolute error={component_error:.3g}",
            "<=1e-12",
            component_error <= 1e-12,
        )
        unsplit_dna_systems = [
            system
            for system in RADIAL_SYSTEMS
            if "-S" not in system and system != "D1"
        ]
        occupancy_check = framewise[
            framewise["system"].isin(unsplit_dna_systems)
        ].merge(
            balanced[
                [
                    "system",
                    "replicate",
                    "frame",
                    "N_MED1_molecules_contacting_DNA",
                    "N_OCT4_molecules_contacting_DNA",
                ]
            ],
            on=["system", "replicate", "frame"],
            validate="one_to_one",
        )
        occupancy_error = float(
            np.max(
                np.abs(
                    occupancy_check[
                        [
                            "DNA_direct_MED1_parent_count",
                            "DNA_direct_OCT4_parent_count",
                        ]
                    ].to_numpy(float)
                    - occupancy_check[
                        [
                            "N_MED1_molecules_contacting_DNA",
                            "N_OCT4_molecules_contacting_DNA",
                        ]
                    ].to_numpy(float)
                )
            )
        )
        record(
            "unsplit parent-level DNA occupancy reproduces physical-chain counts",
            f"max absolute error={occupancy_error:.3g}",
            "0",
            np.isclose(occupancy_error, 0.0),
        )
    validation = pd.DataFrame(rows)
    if not validation["passed"].all():
        raise RuntimeError(
            "Mechanism-extension validation failed:\n"
            + validation.to_string(index=False)
        )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=base.PACKAGE_ROOT)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=base.DEFAULT_OUTPUT)
    parser.add_argument("--replicate", choices=base.REPEATS)
    parser.add_argument("--system", choices=EXTENSION_SYSTEMS)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only and args.aggregate_only:
        parser.error("--analyze-only and --aggregate-only are mutually exclusive")

    output = args.output.resolve()
    cache_dir = output / "cache" / EXTENSION_VERSION
    tables_dir = output / "tables"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest = base.read_manifest(
        args.package_root.resolve(),
        args.data_root.resolve(),
    )
    selected = manifest[
        manifest["analysis_included"] & manifest["system"].isin(EXTENSION_SYSTEMS)
    ].copy()
    if args.replicate:
        selected = selected[selected["replicate"] == args.replicate]
    if args.system:
        selected = selected[selected["system"] == args.system]

    if not args.aggregate_only:
        for row in selected.itertuples(index=False):
            task = base.task_from_row(pd.Series(row._asdict()))
            analyze_extension_task(task, cache_dir, recompute=args.recompute)
    if args.analyze_only:
        return

    framewise, arrays_by_task = load_cached_tasks(manifest, cache_dir)
    summaries = summarize_extension(framewise, arrays_by_task)
    validation = validate_extension(
        framewise,
        summaries,
        arrays_by_task,
        tables_dir / "balanced_framewise_network_contacts_DNA.csv",
    )
    for name, table in summaries.items():
        table.to_csv(tables_dir / f"mechanism_{name}.csv", index=False)
    validation.to_csv(tables_dir / "mechanism_validation.csv", index=False)
    print(
        f"Wrote {len(framewise)} frame rows for {len(arrays_by_task)} mechanism tasks"
    )


if __name__ == "__main__":
    main()
