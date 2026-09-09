#!/usr/bin/env python3
"""Test whether shared MED1 chains give OCT4 bridges a DNA-related function.

The analysis is deliberately OCT4 centered.  It measures whether one physical
MED1 chain simultaneously connects otherwise non-contacting OCT4 molecules at
different DNA contour positions, how persistent those bridges are, and whether
bridge participation predicts OCT4 retention at or return to DNA.  Full,
Split-2, and Split-4 are analyzed as matched trajectories; the formal paired
contrast remains Full versus Split-2.  No figures are produced.

Frames 1001--5000 are sampled every 10 DCD frames.  The DCD interval is 1 ns,
so the dynamic sampling interval is 10 ns and each trajectory contributes 400
observations spanning the 1--5 microsecond production interval.  Lifetimes are
therefore reported as sampled persistence, not sub-10-ns kinetic lifetimes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.lib.nsgrid import FastNS
import numpy as np
import pandas as pd

import run_balanced_analysis as base


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs"
ANALYSIS_VERSION = "med1-functional-bridge-v1"

SYSTEMS = ("M10O90D1", "M10O90D1-S2", "M10O90D1-S4")
FRAME_START = base.PRODUCTION_START
FRAME_END = base.PRODUCTION_END
FRAME_STRIDE = 10
FRAME_NUMBERS = np.arange(FRAME_START, FRAME_END + 1, FRAME_STRIDE, dtype=int)
SAMPLE_INTERVAL_NS = 10.0
CONTACT_CUTOFF_NM = base.PRIMARY_CUTOFF_NM
DNA_BP_COUNT = base.DNA_LENGTH // 2
WINDOW_BP = 20
SEPARATION_THRESHOLDS_BP = (10, 20, 40)
LAG_NS = (10, 20, 50, 100, 200, 500)
HORIZON_NS = (20, 50, 100, 200, 500)

BRIDGE_STATE_NAMES = (
    "physical_pair_all",
    "physical_pair_both_DNA",
    "physical_pair_crosssite20",
    "physical_triple_all",
    "physical_triple_both_DNA",
    "physical_triple_crosssite20",
    "parent_pair_all",
    "parent_pair_both_DNA",
    "parent_pair_crosssite20",
    "parent_triple_all",
    "parent_triple_both_DNA",
    "parent_triple_crosssite20",
)


def cache_paths(cache_dir: Path, task: base.Task) -> dict[str, Path]:
    stem = f"{task.replicate}__{task.system}"
    return {
        "meta": cache_dir / f"{stem}.json",
        "frame": cache_dir / f"{stem}.frames.csv",
        "occurrence": cache_dir / f"{stem}.occurrences.csv.gz",
        "run": cache_dir / f"{stem}.runs.csv.gz",
        "survival": cache_dir / f"{stem}.survival.csv",
        "retention": cache_dir / f"{stem}.retention.csv",
        "rebinding": cache_dir / f"{stem}.rebinding.csv",
        "loss_rebinding": cache_dir / f"{stem}.loss-rebinding.csv",
    }


def signature(task: base.Task) -> dict[str, object]:
    dcd = task.path / "output.dcd"
    return {
        "analysis_version": ANALYSIS_VERSION,
        "system": task.system,
        "replicate": task.replicate,
        "box_nm": task.box_nm,
        "dcd_size": dcd.stat().st_size,
        "dcd_mtime_ns": dcd.stat().st_mtime_ns,
        "start_pdb_sha256": base.sha256(task.path / "start.pdb"),
        "molecule_map_sha256": base.sha256(task.path / "molecule_map.tsv"),
        "frame_numbers": FRAME_NUMBERS.tolist(),
        "sample_interval_ns": SAMPLE_INTERVAL_NS,
        "contact_cutoff_nm": CONTACT_CUTOFF_NM,
        "dna_physical_bp_mapping": "min(local_index, 399-local_index)",
        "window_bp": WINDOW_BP,
        "separation_thresholds_bp": list(SEPARATION_THRESHOLDS_BP),
        "lag_ns": list(LAG_NS),
        "horizon_ns": list(HORIZON_NS),
    }


def cache_valid(paths: dict[str, Path], expected: dict[str, object]) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        return json.loads(paths["meta"].read_text()) == expected
    except (OSError, json.JSONDecodeError):
        return False


def safe_fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def atom_groups(
    molecules: base.Molecules,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    atom_type = molecules.molecule_type[molecules.atom_molecule]
    oct4_atoms = np.flatnonzero(atom_type == base.TYPE_OCT4)
    med1_atoms = np.flatnonzero(atom_type == base.TYPE_MED1)
    dna_atoms = np.flatnonzero(atom_type == base.TYPE_DNA)

    oct4_nodes = np.flatnonzero(molecules.molecule_type == base.TYPE_OCT4)
    med1_nodes = np.flatnonzero(molecules.molecule_type == base.TYPE_MED1)
    oct4_local = np.full(len(molecules.molecule_type), -1, dtype=np.int32)
    med1_local = np.full(len(molecules.molecule_type), -1, dtype=np.int32)
    oct4_local[oct4_nodes] = np.arange(len(oct4_nodes), dtype=np.int32)
    med1_local[med1_nodes] = np.arange(len(med1_nodes), dtype=np.int32)
    oct4_atom_chain = oct4_local[molecules.atom_molecule[oct4_atoms]]
    med1_atom_chain = med1_local[molecules.atom_molecule[med1_atoms]]
    dna_physical_bp = np.minimum(
        molecules.atom_residue[dna_atoms],
        base.DNA_LENGTH - 1 - molecules.atom_residue[dna_atoms],
    ).astype(np.int16)
    return (
        oct4_atoms,
        med1_atoms,
        dna_atoms,
        oct4_atom_chain,
        med1_atom_chain,
        dna_physical_bp,
    )


def fastns_box(box_nm: float) -> np.ndarray:
    return np.asarray([box_nm * 10.0] * 3 + [90.0, 90.0, 90.0], dtype=np.float32)


def cross_pairs(
    query_positions_a: np.ndarray,
    target_positions_a: np.ndarray,
    box_a: np.ndarray,
) -> np.ndarray:
    search = FastNS(
        CONTACT_CUTOFF_NM * 10.0,
        target_positions_a,
        box=box_a,
        pbc=True,
    ).search(query_positions_a)
    pairs = search.get_pairs().astype(np.int32, copy=False)
    distances_nm = search.get_pair_distances().astype(np.float32, copy=False) / 10.0
    return pairs[distances_nm < CONTACT_CUTOFF_NM]


def self_pairs(positions_a: np.ndarray, box_a: np.ndarray) -> np.ndarray:
    search = FastNS(
        CONTACT_CUTOFF_NM * 10.0,
        positions_a,
        box=box_a,
        pbc=True,
    ).self_search()
    pairs = search.get_pairs().astype(np.int32, copy=False)
    distances_nm = search.get_pair_distances().astype(np.float32, copy=False) / 10.0
    return pairs[distances_nm < CONTACT_CUTOFF_NM]


def unique_chain_pairs(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    keep = low != high
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.int32)
    return np.unique(np.column_stack([low[keep], high[keep]]), axis=0)


def frame_contacts(
    positions_a: np.ndarray,
    task: base.Task,
    molecules: base.Molecules,
    groups: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[set[tuple[int, int]], dict[int, set[int]], list[set[int]]]:
    (
        oct4_atoms,
        med1_atoms,
        dna_atoms,
        oct4_atom_chain,
        med1_atom_chain,
        dna_physical_bp,
    ) = groups
    wrapped_a = np.mod(positions_a, task.box_nm * 10.0).astype(np.float32, copy=False)
    box_a = fastns_box(task.box_nm)

    oo_atom_pairs = self_pairs(wrapped_a[oct4_atoms], box_a)
    oo_chain_pairs = unique_chain_pairs(
        oct4_atom_chain[oo_atom_pairs[:, 0]],
        oct4_atom_chain[oo_atom_pairs[:, 1]],
    )
    direct_oo = {tuple(map(int, pair)) for pair in oo_chain_pairs}

    om_atom_pairs = cross_pairs(wrapped_a[oct4_atoms], wrapped_a[med1_atoms], box_a)
    om_chain_pairs = np.unique(
        np.column_stack(
            [
                oct4_atom_chain[om_atom_pairs[:, 0]],
                med1_atom_chain[om_atom_pairs[:, 1]],
            ]
        ),
        axis=0,
    )
    med1_neighbors: dict[int, set[int]] = defaultdict(set)
    for oct4, med1 in om_chain_pairs:
        med1_neighbors[int(med1)].add(int(oct4))

    od_atom_pairs = cross_pairs(wrapped_a[oct4_atoms], wrapped_a[dna_atoms], box_a)
    od_pairs = np.unique(
        np.column_stack(
            [
                oct4_atom_chain[od_atom_pairs[:, 0]],
                dna_physical_bp[od_atom_pairs[:, 1]],
            ]
        ),
        axis=0,
    )
    dna_contacts = [set() for _ in range(task.n_oct4)]
    for oct4, bp in od_pairs:
        dna_contacts[int(oct4)].add(int(bp))
    return direct_oo, med1_neighbors, dna_contacts


def dna_pair_features(first_bp: set[int], second_bp: set[int]) -> dict[str, object]:
    if not first_bp or not second_bp:
        return {
            "both_DNA_bound": False,
            "minimum_contour_separation_bp": float("nan"),
            "center_contour_separation_bp": float("nan"),
            "shared_20bp_window": False,
        }
    first = np.asarray(sorted(first_bp), dtype=int)
    second = np.asarray(sorted(second_bp), dtype=int)
    minimum = int(np.min(np.abs(first[:, None] - second[None, :])))
    center = float(abs(np.median(first) - np.median(second)))
    first_windows = set((first // WINDOW_BP).tolist())
    second_windows = set((second // WINDOW_BP).tolist())
    return {
        "both_DNA_bound": True,
        "minimum_contour_separation_bp": float(minimum),
        "center_contour_separation_bp": center,
        "shared_20bp_window": bool(first_windows & second_windows),
    }


def bridge_occurrences(
    neighbors: dict[int, set[int]],
    direct_oo: set[tuple[int, int]],
    dna_contacts: list[set[int]],
    med1_parent: np.ndarray,
    *,
    scope: str,
) -> list[dict[str, object]]:
    if scope == "physical":
        scoped_neighbors = neighbors
    elif scope == "parent":
        scoped_neighbors: dict[int, set[int]] = defaultdict(set)
        for physical, oct4_neighbors in neighbors.items():
            scoped_neighbors[int(med1_parent[physical])].update(oct4_neighbors)
    else:
        raise ValueError(scope)

    rows = []
    for med1, oct4_neighbors in scoped_neighbors.items():
        for first, second in itertools.combinations(sorted(oct4_neighbors), 2):
            pair = (int(first), int(second))
            if pair in direct_oo:
                continue
            features = dna_pair_features(dna_contacts[first], dna_contacts[second])
            row: dict[str, object] = {
                "scope": scope,
                "med1_id": int(med1),
                "OCT4_first": first,
                "OCT4_second": second,
            }
            row.update(features)
            rows.append(row)
    return rows


def occurrence_sets(
    rows: list[dict[str, object]],
    scope: str,
) -> dict[str, set[tuple[int, ...]]]:
    selected = [row for row in rows if row["scope"] == scope]
    pair_all: set[tuple[int, ...]] = set()
    pair_dna: set[tuple[int, ...]] = set()
    pair_cross: set[tuple[int, ...]] = set()
    triple_all: set[tuple[int, ...]] = set()
    triple_dna: set[tuple[int, ...]] = set()
    triple_cross: set[tuple[int, ...]] = set()
    for row in selected:
        pair = (int(row["OCT4_first"]), int(row["OCT4_second"]))
        triple = (int(row["med1_id"]), *pair)
        pair_all.add(pair)
        triple_all.add(triple)
        if bool(row["both_DNA_bound"]):
            pair_dna.add(pair)
            triple_dna.add(triple)
            if float(row["minimum_contour_separation_bp"]) >= 20.0:
                pair_cross.add(pair)
                triple_cross.add(triple)
    return {
        f"{scope}_pair_all": pair_all,
        f"{scope}_pair_both_DNA": pair_dna,
        f"{scope}_pair_crosssite20": pair_cross,
        f"{scope}_triple_all": triple_all,
        f"{scope}_triple_both_DNA": triple_dna,
        f"{scope}_triple_crosssite20": triple_cross,
    }


def pair_participants(pairs: set[tuple[int, ...]], n_oct4: int) -> np.ndarray:
    result = np.zeros(n_oct4, dtype=bool)
    for first, second in pairs:
        result[int(first)] = True
        result[int(second)] = True
    return result


def anchored_unbound_participants(
    pairs: set[tuple[int, ...]],
    dna_bound: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(dna_bound), dtype=bool)
    for first, second in pairs:
        if dna_bound[first] and not dna_bound[second]:
            result[second] = True
        elif dna_bound[second] and not dna_bound[first]:
            result[first] = True
    return result


def summarize_frame(
    task: base.Task,
    direct_oo: set[tuple[int, int]],
    dna_contacts: list[set[int]],
    physical_rows: list[dict[str, object]],
    parent_rows: list[dict[str, object]],
    state_sets: dict[str, set[tuple[int, ...]]],
) -> dict[str, float]:
    dna_bound = np.asarray([bool(values) for values in dna_contacts], dtype=bool)
    n_bound = int(dna_bound.sum())
    possible_bound_pairs = n_bound * (n_bound - 1) // 2
    row: dict[str, float] = {
        "direct_DNA_OCT4_n": float(n_bound),
        "direct_DNA_OCT4_fraction": n_bound / task.n_oct4,
        "direct_OO_pair_count": float(len(direct_oo)),
    }
    for scope, occurrences in (("physical", physical_rows), ("parent", parent_rows)):
        pair_all = state_sets[f"{scope}_pair_all"]
        pair_dna = state_sets[f"{scope}_pair_both_DNA"]
        triple_all = state_sets[f"{scope}_triple_all"]
        triple_dna = state_sets[f"{scope}_triple_both_DNA"]
        row[f"{scope}_strict_OMO_pair_count"] = float(len(pair_all))
        row[f"{scope}_strict_OMO_triple_count"] = float(len(triple_all))
        row[f"{scope}_DNA_bound_pair_count"] = float(len(pair_dna))
        row[f"{scope}_DNA_bound_triple_count"] = float(len(triple_dna))
        row[f"{scope}_DNA_bound_pair_fraction_of_bridges"] = safe_fraction(
            len(pair_dna), len(pair_all)
        )
        row[f"{scope}_DNA_bound_pair_fraction_of_bound_O_pairs"] = safe_fraction(
            len(pair_dna), possible_bound_pairs
        )
        row[f"{scope}_bridged_DNA_bound_OCT4_n"] = float(
            pair_participants(pair_dna, task.n_oct4).sum()
        )
        dna_occurrences = [item for item in occurrences if bool(item["both_DNA_bound"])]
        unique_dna_pair_rows: dict[tuple[int, int], dict[str, object]] = {}
        for item in dna_occurrences:
            pair = (int(item["OCT4_first"]), int(item["OCT4_second"]))
            unique_dna_pair_rows[pair] = item
        separations = np.asarray(
            [
                float(item["minimum_contour_separation_bp"])
                for item in unique_dna_pair_rows.values()
            ],
            dtype=float,
        )
        center_separations = np.asarray(
            [
                float(item["center_contour_separation_bp"])
                for item in unique_dna_pair_rows.values()
            ],
            dtype=float,
        )
        row[f"{scope}_DNA_bound_pair_minsep_mean_bp"] = (
            float(separations.mean()) if len(separations) else float("nan")
        )
        row[f"{scope}_DNA_bound_pair_minsep_median_bp"] = (
            float(np.median(separations)) if len(separations) else float("nan")
        )
        row[f"{scope}_DNA_bound_pair_centersep_mean_bp"] = (
            float(center_separations.mean())
            if len(center_separations)
            else float("nan")
        )
        different_window_pairs = {
            (int(item["OCT4_first"]), int(item["OCT4_second"]))
            for item in dna_occurrences
            if not bool(item["shared_20bp_window"])
        }
        row[f"{scope}_DNA_bound_different_20bp_window_pair_count"] = float(
            len(different_window_pairs)
        )
        for threshold in SEPARATION_THRESHOLDS_BP:
            cross_pairs_at_threshold = {
                (int(item["OCT4_first"]), int(item["OCT4_second"]))
                for item in dna_occurrences
                if float(item["minimum_contour_separation_bp"]) >= threshold
            }
            row[f"{scope}_DNA_bound_minsep_ge_{threshold}bp_pair_count"] = float(
                len(cross_pairs_at_threshold)
            )
            row[
                f"{scope}_DNA_bound_minsep_ge_{threshold}bp_pair_fraction"
            ] = safe_fraction(len(cross_pairs_at_threshold), len(pair_dna))
    return row


def run_records(
    states: dict[str, list[set[tuple[int, ...]]]],
    task: base.Task,
) -> pd.DataFrame:
    rows = []
    for state_name, sequence in states.items():
        active: dict[tuple[int, ...], tuple[int, int]] = {}
        for frame_index, current in enumerate(sequence):
            ended = set(active) - current
            for key in ended:
                start, observations = active.pop(key)
                rows.append(
                    {
                        "system": task.system,
                        "replicate": task.replicate,
                        "bridge_state": state_name,
                        "key": ":".join(map(str, key)),
                        "start_frame_index": start,
                        "end_frame_index": frame_index - 1,
                        "consecutive_observations": observations,
                        "sampled_span_ns": (observations - 1) * SAMPLE_INTERVAL_NS,
                    }
                )
            for key in current:
                if key in active:
                    start, observations = active[key]
                    active[key] = (start, observations + 1)
                else:
                    active[key] = (frame_index, 1)
        for key, (start, observations) in active.items():
            rows.append(
                {
                    "system": task.system,
                    "replicate": task.replicate,
                    "bridge_state": state_name,
                    "key": ":".join(map(str, key)),
                    "start_frame_index": start,
                    "end_frame_index": len(sequence) - 1,
                    "consecutive_observations": observations,
                    "sampled_span_ns": (observations - 1) * SAMPLE_INTERVAL_NS,
                }
            )
    return pd.DataFrame(rows)


def survival_summary(
    states: dict[str, list[set[tuple[int, ...]]]],
    task: base.Task,
) -> pd.DataFrame:
    rows = []
    for state_name, sequence in states.items():
        for lag_ns in LAG_NS:
            lag_steps = int(round(lag_ns / SAMPLE_INTERVAL_NS))
            denominator = 0
            intermittent_numerator = 0
            continuous_numerator = 0
            for start in range(len(sequence) - lag_steps):
                initial = sequence[start]
                denominator += len(initial)
                intermittent_numerator += len(initial & sequence[start + lag_steps])
                continuous = set(initial)
                for offset in range(1, lag_steps + 1):
                    continuous.intersection_update(sequence[start + offset])
                    if not continuous:
                        break
                continuous_numerator += len(continuous)
            rows.append(
                {
                    "system": task.system,
                    "replicate": task.replicate,
                    "bridge_state": state_name,
                    "lag_ns": float(lag_ns),
                    "state_observations_at_risk": float(denominator),
                    "intermittent_persistence_probability": safe_fraction(
                        intermittent_numerator, denominator
                    ),
                    "continuous_persistence_probability": safe_fraction(
                        continuous_numerator, denominator
                    ),
                }
            )
    return pd.DataFrame(rows)


def retention_summary(
    dna_history: np.ndarray,
    participant_history: dict[str, np.ndarray],
    task: base.Task,
) -> pd.DataFrame:
    rows = []
    categories = (
        "physical_any_bridge",
        "physical_DNA_bridge",
        "physical_crosssite20_bridge",
        "parent_any_bridge",
        "parent_DNA_bridge",
        "parent_crosssite20_bridge",
    )
    for category in categories:
        bridge_history = participant_history[category]
        for lag_ns in LAG_NS:
            steps = int(round(lag_ns / SAMPLE_INTERVAL_NS))
            bound_now = dna_history[:-steps]
            bound_later = dna_history[steps:]
            bridge_now = bridge_history[:-steps]
            eligible_bridge = bound_now & bridge_now
            eligible_reference = bound_now & ~bridge_now
            bridge_denominator = int(eligible_bridge.sum())
            reference_denominator = int(eligible_reference.sum())
            bridge_probability = safe_fraction(
                int((eligible_bridge & bound_later).sum()), bridge_denominator
            )
            reference_probability = safe_fraction(
                int((eligible_reference & bound_later).sum()), reference_denominator
            )
            rows.append(
                {
                    "system": task.system,
                    "replicate": task.replicate,
                    "bridge_category": category,
                    "lag_ns": float(lag_ns),
                    "bridged_bound_observations": float(bridge_denominator),
                    "unbridged_bound_observations": float(reference_denominator),
                    "P_DNA_retained_if_bridged": bridge_probability,
                    "P_DNA_retained_if_unbridged": reference_probability,
                    "bridged_minus_unbridged_retention": bridge_probability
                    - reference_probability,
                }
            )
    return pd.DataFrame(rows)


def rebinding_summary(
    dna_history: np.ndarray,
    participant_history: dict[str, np.ndarray],
    task: base.Task,
) -> pd.DataFrame:
    rows = []
    categories = ("physical_DNA_anchored_unbound", "parent_DNA_anchored_unbound")
    for category in categories:
        anchored_history = participant_history[category]
        for horizon_ns in HORIZON_NS:
            steps = int(round(horizon_ns / SAMPLE_INTERVAL_NS))
            outcome = np.zeros_like(dna_history[:-steps], dtype=bool)
            for offset in range(1, steps + 1):
                outcome |= dna_history[offset : len(dna_history) - steps + offset]
            unbound_now = ~dna_history[:-steps]
            anchored_now = anchored_history[:-steps]
            eligible_anchored = unbound_now & anchored_now
            eligible_reference = unbound_now & ~anchored_now
            anchored_denominator = int(eligible_anchored.sum())
            reference_denominator = int(eligible_reference.sum())
            anchored_probability = safe_fraction(
                int((eligible_anchored & outcome).sum()), anchored_denominator
            )
            reference_probability = safe_fraction(
                int((eligible_reference & outcome).sum()), reference_denominator
            )
            rows.append(
                {
                    "system": task.system,
                    "replicate": task.replicate,
                    "bridge_category": category,
                    "horizon_ns": float(horizon_ns),
                    "anchored_unbound_observations": float(anchored_denominator),
                    "unanchored_unbound_observations": float(reference_denominator),
                    "P_rebind_if_DNA_anchored_bridge": anchored_probability,
                    "P_rebind_if_no_DNA_anchored_bridge": reference_probability,
                    "anchored_minus_unanchored_rebinding": anchored_probability
                    - reference_probability,
                }
            )
    return pd.DataFrame(rows)


def loss_rebinding_summary(
    dna_history: np.ndarray,
    participant_history: dict[str, np.ndarray],
    task: base.Task,
) -> pd.DataFrame:
    rows = []
    categories = ("physical_DNA_bridge", "parent_DNA_bridge")
    for category in categories:
        bridge_history = participant_history[category]
        for horizon_ns in HORIZON_NS:
            steps = int(round(horizon_ns / SAMPLE_INTERVAL_NS))
            # loss_index is the first sampled unbound observation.  The bridge
            # label is taken from the immediately preceding bound observation.
            loss = dna_history[:-steps][:-1] & ~dna_history[1:-steps]
            bridge_before_loss = bridge_history[:-steps][:-1]
            rebound = np.zeros_like(loss, dtype=bool)
            for offset in range(1, steps + 1):
                rebound |= dna_history[1 + offset : len(dna_history) - steps + offset]
            eligible_bridge = loss & bridge_before_loss
            eligible_reference = loss & ~bridge_before_loss
            bridge_denominator = int(eligible_bridge.sum())
            reference_denominator = int(eligible_reference.sum())
            bridge_probability = safe_fraction(
                int((eligible_bridge & rebound).sum()), bridge_denominator
            )
            reference_probability = safe_fraction(
                int((eligible_reference & rebound).sum()), reference_denominator
            )
            rows.append(
                {
                    "system": task.system,
                    "replicate": task.replicate,
                    "bridge_category": category,
                    "horizon_ns": float(horizon_ns),
                    "loss_events_bridged_before_loss": float(bridge_denominator),
                    "loss_events_unbridged_before_loss": float(reference_denominator),
                    "P_rebind_after_loss_if_previously_bridged": bridge_probability,
                    "P_rebind_after_loss_if_previously_unbridged": reference_probability,
                    "bridged_minus_unbridged_loss_rebinding": bridge_probability
                    - reference_probability,
                }
            )
    return pd.DataFrame(rows)


def analyze_task(
    task: base.Task,
    cache_dir: Path,
    *,
    recompute: bool,
) -> dict[str, pd.DataFrame]:
    paths = cache_paths(cache_dir, task)
    expected = signature(task)
    if not recompute and cache_valid(paths, expected):
        return {
            "frame": pd.read_csv(paths["frame"]),
            "occurrence": pd.read_csv(paths["occurrence"]),
            "run": pd.read_csv(paths["run"]),
            "survival": pd.read_csv(paths["survival"]),
            "retention": pd.read_csv(paths["retention"]),
            "rebinding": pd.read_csv(paths["rebinding"]),
            "loss_rebinding": pd.read_csv(paths["loss_rebinding"]),
        }

    print(f"Functional MED1 bridge analysis {task.replicate}/{task.system}", flush=True)
    molecules = base.read_molecule_map(task)
    groups = atom_groups(molecules)
    med1_nodes = np.flatnonzero(molecules.molecule_type == base.TYPE_MED1)
    med1_parent = molecules.parent_med1[med1_nodes]
    universe = mda.Universe(str(task.path / "start.pdb"), str(task.path / "output.dcd"))
    if len(universe.trajectory) < FRAME_END:
        raise RuntimeError(f"{task.replicate}/{task.system}: incomplete DCD")

    frame_rows = []
    occurrence_rows = []
    state_history: dict[str, list[set[tuple[int, ...]]]] = {
        name: [] for name in BRIDGE_STATE_NAMES
    }
    dna_history = []
    participant_lists: dict[str, list[np.ndarray]] = defaultdict(list)

    for sample_index, frame in enumerate(FRAME_NUMBERS):
        universe.trajectory[int(frame) - 1]
        direct_oo, med1_neighbors, dna_contacts = frame_contacts(
            universe.atoms.positions.astype(np.float64), task, molecules, groups
        )
        physical_rows = bridge_occurrences(
            med1_neighbors,
            direct_oo,
            dna_contacts,
            med1_parent,
            scope="physical",
        )
        parent_rows = bridge_occurrences(
            med1_neighbors,
            direct_oo,
            dna_contacts,
            med1_parent,
            scope="parent",
        )
        states = {}
        states.update(occurrence_sets(physical_rows, "physical"))
        states.update(occurrence_sets(parent_rows, "parent"))
        for name in BRIDGE_STATE_NAMES:
            state_history[name].append(states[name])

        dna_bound = np.asarray([bool(values) for values in dna_contacts], dtype=bool)
        dna_history.append(dna_bound)
        participant_lists["physical_any_bridge"].append(
            pair_participants(states["physical_pair_all"], task.n_oct4)
        )
        participant_lists["physical_DNA_bridge"].append(
            pair_participants(states["physical_pair_both_DNA"], task.n_oct4)
        )
        participant_lists["physical_crosssite20_bridge"].append(
            pair_participants(states["physical_pair_crosssite20"], task.n_oct4)
        )
        participant_lists["parent_any_bridge"].append(
            pair_participants(states["parent_pair_all"], task.n_oct4)
        )
        participant_lists["parent_DNA_bridge"].append(
            pair_participants(states["parent_pair_both_DNA"], task.n_oct4)
        )
        participant_lists["parent_crosssite20_bridge"].append(
            pair_participants(states["parent_pair_crosssite20"], task.n_oct4)
        )
        participant_lists["physical_DNA_anchored_unbound"].append(
            anchored_unbound_participants(states["physical_pair_all"], dna_bound)
        )
        participant_lists["parent_DNA_anchored_unbound"].append(
            anchored_unbound_participants(states["parent_pair_all"], dna_bound)
        )

        frame_row: dict[str, object] = {
            "system": task.system,
            "replicate": task.replicate,
            "sample_index": sample_index,
            "frame": int(frame),
            "time_us": float(universe.trajectory.ts.time / 1.0e6),
        }
        frame_row.update(
            summarize_frame(
                task,
                direct_oo,
                dna_contacts,
                physical_rows,
                parent_rows,
                states,
            )
        )
        frame_rows.append(frame_row)
        for item in physical_rows + parent_rows:
            occurrence_rows.append(
                {
                    "system": task.system,
                    "replicate": task.replicate,
                    "sample_index": sample_index,
                    "frame": int(frame),
                    "time_us": float(universe.trajectory.ts.time / 1.0e6),
                    **item,
                }
            )

    frame_table = pd.DataFrame(frame_rows)
    occurrence_table = pd.DataFrame(occurrence_rows)
    run_table = run_records(state_history, task)
    survival_table = survival_summary(state_history, task)
    dna_array = np.stack(dna_history)
    participant_history = {
        name: np.stack(values) for name, values in participant_lists.items()
    }
    retention_table = retention_summary(dna_array, participant_history, task)
    rebinding_table = rebinding_summary(dna_array, participant_history, task)
    loss_rebinding_table = loss_rebinding_summary(
        dna_array, participant_history, task
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_table.to_csv(paths["frame"], index=False)
    occurrence_table.to_csv(paths["occurrence"], index=False, compression="gzip")
    run_table.to_csv(paths["run"], index=False, compression="gzip")
    survival_table.to_csv(paths["survival"], index=False)
    retention_table.to_csv(paths["retention"], index=False)
    rebinding_table.to_csv(paths["rebinding"], index=False)
    loss_rebinding_table.to_csv(paths["loss_rebinding"], index=False)
    paths["meta"].write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return {
        "frame": frame_table,
        "occurrence": occurrence_table,
        "run": run_table,
        "survival": survival_table,
        "retention": retention_table,
        "rebinding": rebinding_table,
        "loss_rebinding": loss_rebinding_table,
    }


def mean_sd_table(
    repeat: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    return base.mean_sd(repeat, keys, metrics)


def paired_long_table(
    repeat: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_columns = ["system", "replicate", *keys]
    index = repeat.set_index(index_columns)
    full_rows = repeat[repeat["system"] == SYSTEMS[0]][["replicate", *keys]].drop_duplicates()
    rows = []
    for _, identifier in full_rows.iterrows():
        replicate = str(identifier["replicate"])
        key_values = tuple(identifier[key] for key in keys)
        for metric in metrics:
            full_key = (SYSTEMS[0], replicate, *key_values)
            split_key = (SYSTEMS[1], replicate, *key_values)
            full = float(index.loc[full_key, metric])
            split = float(index.loc[split_key, metric])
            if not (np.isfinite(full) and np.isfinite(split)):
                continue
            row: dict[str, object] = {
                "replicate": replicate,
                "metric": metric,
                "Full": full,
                "Split_2": split,
                "Split_2_minus_Full": split - full,
            }
            row.update({key: value for key, value in zip(keys, key_values)})
            rows.append(row)
    paired = pd.DataFrame(rows)
    group_keys = [*keys, "metric"]
    group = base.mean_sd(
        paired,
        group_keys,
        ["Full", "Split_2", "Split_2_minus_Full"],
    )
    return paired, group


def summarize(all_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    frame = all_tables["frame"]
    frame_metrics = [
        column
        for column in frame.columns
        if column not in {"system", "replicate", "sample_index", "frame", "time_us"}
    ]
    frame_repeat = (
        frame.groupby(["system", "replicate"], observed=True)[frame_metrics]
        .mean()
        .reset_index()
    )
    frame_group = mean_sd_table(frame_repeat, ["system"], frame_metrics)
    frame_paired, frame_paired_group = paired_long_table(
        frame_repeat, [], frame_metrics
    )

    runs = all_tables["run"]
    lifetime_rows = []
    for (system, replicate, bridge_state), group in runs.groupby(
        ["system", "replicate", "bridge_state"], observed=True
    ):
        observations = group["consecutive_observations"].to_numpy(float)
        spans = group["sampled_span_ns"].to_numpy(float)
        lifetime_rows.append(
            {
                "system": system,
                "replicate": replicate,
                "bridge_state": bridge_state,
                "n_runs": float(len(group)),
                "n_unique_keys": float(group["key"].nunique()),
                "total_state_observations": float(observations.sum()),
                "mean_consecutive_observations": float(observations.mean()),
                "median_consecutive_observations": float(np.median(observations)),
                "mean_sampled_span_ns": float(spans.mean()),
                "median_sampled_span_ns": float(np.median(spans)),
                "maximum_sampled_span_ns": float(spans.max()),
                "fraction_single_observation_runs": float(np.mean(observations == 1)),
                "fraction_runs_span_ge_20ns": float(np.mean(spans >= 20)),
                "fraction_runs_span_ge_50ns": float(np.mean(spans >= 50)),
                "fraction_runs_span_ge_100ns": float(np.mean(spans >= 100)),
            }
        )
    lifetime_repeat = pd.DataFrame(lifetime_rows)
    lifetime_metrics = [
        column
        for column in lifetime_repeat.columns
        if column not in {"system", "replicate", "bridge_state"}
    ]
    lifetime_group = mean_sd_table(
        lifetime_repeat, ["system", "bridge_state"], lifetime_metrics
    )
    lifetime_paired, lifetime_paired_group = paired_long_table(
        lifetime_repeat, ["bridge_state"], lifetime_metrics
    )

    result = {
        **all_tables,
        "frame_repeat": frame_repeat,
        "frame_group": frame_group,
        "frame_full_split_repeat": frame_paired,
        "frame_full_split_group": frame_paired_group,
        "lifetime_repeat": lifetime_repeat,
        "lifetime_group": lifetime_group,
        "lifetime_full_split_repeat": lifetime_paired,
        "lifetime_full_split_group": lifetime_paired_group,
    }

    specifications = {
        "survival": (
            ["bridge_state", "lag_ns"],
            [
                "state_observations_at_risk",
                "intermittent_persistence_probability",
                "continuous_persistence_probability",
            ],
        ),
        "retention": (
            ["bridge_category", "lag_ns"],
            [
                "bridged_bound_observations",
                "unbridged_bound_observations",
                "P_DNA_retained_if_bridged",
                "P_DNA_retained_if_unbridged",
                "bridged_minus_unbridged_retention",
            ],
        ),
        "rebinding": (
            ["bridge_category", "horizon_ns"],
            [
                "anchored_unbound_observations",
                "unanchored_unbound_observations",
                "P_rebind_if_DNA_anchored_bridge",
                "P_rebind_if_no_DNA_anchored_bridge",
                "anchored_minus_unanchored_rebinding",
            ],
        ),
        "loss_rebinding": (
            ["bridge_category", "horizon_ns"],
            [
                "loss_events_bridged_before_loss",
                "loss_events_unbridged_before_loss",
                "P_rebind_after_loss_if_previously_bridged",
                "P_rebind_after_loss_if_previously_unbridged",
                "bridged_minus_unbridged_loss_rebinding",
            ],
        ),
    }
    for name, (keys, metrics) in specifications.items():
        repeat = all_tables[name]
        group = mean_sd_table(repeat, ["system", *keys], metrics)
        paired, paired_group = paired_long_table(repeat, keys, metrics)
        result[f"{name}_group"] = group
        result[f"{name}_full_split_repeat"] = paired
        result[f"{name}_full_split_group"] = paired_group
    return result


def validate(
    summaries: dict[str, pd.DataFrame],
    output: Path,
) -> pd.DataFrame:
    checks = []
    frame = summaries["frame"]
    counts = frame.groupby(["system", "replicate"], observed=True).size()
    checks.append(
        {
            "check": "each trajectory contributes 400 observations",
            "value": sorted(counts.unique().tolist()),
            "required": [400],
            "passed": bool((counts == 400).all()),
        }
    )

    time_steps_ns = (
        frame.sort_values(["system", "replicate", "sample_index"])
        .groupby(["system", "replicate"], observed=True)["time_us"]
        .diff()
        .dropna()
        * 1000.0
    )
    spacing_error = float(np.max(np.abs(time_steps_ns - SAMPLE_INTERVAL_NS)))
    checks.append(
        {
            "check": "dynamic observations are spaced by 10 ns",
            "value": f"max error={spacing_error:.3g} ns",
            "required": "<=1e-6 ns",
            "passed": spacing_error <= 1e-6,
        }
    )

    previous = pd.read_csv(output / "tables" / "oct4_centric_framewise.csv")
    previous = previous[previous["system"].isin(SYSTEMS)][
        [
            "system",
            "replicate",
            "frame",
            "OMO_noncontact_pair_count",
            "OCT4_direct_DNA_fraction",
        ]
    ]
    overlap = frame.merge(
        previous,
        on=["system", "replicate", "frame"],
        how="inner",
        validate="many_to_one",
    )
    bridge_error = float(
        np.max(
            np.abs(
                overlap["physical_strict_OMO_pair_count"]
                - overlap["OMO_noncontact_pair_count"]
            )
        )
    )
    dna_error = float(
        np.max(
            np.abs(
                overlap["direct_DNA_OCT4_fraction"]
                - overlap["OCT4_direct_DNA_fraction"]
            )
        )
    )
    checks.append(
        {
            "check": "10-ns analysis reproduces validated 100-ns bridge and DNA metrics",
            "value": f"bridge error={bridge_error:.3g}; DNA error={dna_error:.3g}",
            "required": "both <=1e-12",
            "passed": bridge_error <= 1e-12 and dna_error <= 1e-12,
        }
    )

    full = frame[frame["system"] == SYSTEMS[0]]
    full_parent_error = float(
        np.max(
            np.abs(
                full["physical_strict_OMO_pair_count"]
                - full["parent_strict_OMO_pair_count"]
            )
        )
    )
    checks.append(
        {
            "check": "physical and parent bridge counts coincide for Full MED1",
            "value": f"max error={full_parent_error:.3g}",
            "required": "<=1e-12",
            "passed": full_parent_error <= 1e-12,
        }
    )

    occurrence = summaries["occurrence"]
    finite_separation = occurrence[
        occurrence["both_DNA_bound"].astype(bool)
    ]["minimum_contour_separation_bp"]
    contour_ok = bool(
        len(finite_separation)
        and finite_separation.between(0, DNA_BP_COUNT - 1).all()
    )
    checks.append(
        {
            "check": "DNA contour separations lie within the 200-bp model",
            "value": [float(finite_separation.min()), float(finite_separation.max())],
            "required": [0, DNA_BP_COUNT - 1],
            "passed": contour_ok,
        }
    )

    probability_columns = [
        column
        for name in ("survival", "retention", "rebinding", "loss_rebinding")
        for column in summaries[name].columns
        if column.startswith("P_") or "probability" in column
    ]
    probability_ok = True
    for name in ("survival", "retention", "rebinding", "loss_rebinding"):
        for column in summaries[name].columns:
            if column.startswith("P_") or "probability" in column:
                values = summaries[name][column].dropna()
                probability_ok &= bool(values.between(0, 1).all())
    checks.append(
        {
            "check": "all conditional probabilities are bounded",
            "value": len(probability_columns),
            "required": "all within [0,1]",
            "passed": probability_ok,
        }
    )

    paired = summaries["frame_full_split_repeat"]
    paired_n = paired.groupby("metric", observed=True).size()
    checks.append(
        {
            "check": "Full-Split-2 frame metrics use three matched trajectories",
            "value": sorted(paired_n.unique().tolist()),
            "required": [3],
            "passed": bool((paired_n == 3).all()),
        }
    )
    validation = pd.DataFrame(checks)
    if not validation["passed"].all():
        raise RuntimeError("Functional bridge validation failed:\n" + validation.to_string(index=False))
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    tables_dir = output / "tables"
    cache_dir = output / "cache" / ANALYSIS_VERSION
    manifest = pd.read_csv(tables_dir / "trajectory_manifest.csv")
    selected = manifest[
        manifest["available"].astype(bool)
        & manifest["analysis_included"].astype(bool)
        & manifest["system"].isin(SYSTEMS)
        & manifest["replicate"].isin(base.REPEATS)
    ].copy()
    repeat_order = {value: index for index, value in enumerate(base.REPEATS)}
    system_order = {value: index for index, value in enumerate(SYSTEMS)}
    selected["repeat_order"] = selected["replicate"].map(repeat_order)
    selected["system_order"] = selected["system"].map(system_order)
    selected = selected.sort_values(["repeat_order", "system_order"])
    tasks = [base.task_from_row(row) for _, row in selected.iterrows()]
    if len(tasks) != 9:
        raise RuntimeError(
            f"Expected nine Full/Split-2/Split-4 +DNA tasks, found {len(tasks)}"
        )

    collected: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for task in tasks:
        result = analyze_task(task, cache_dir, recompute=args.recompute)
        for name, table in result.items():
            collected[name].append(table)
    combined = {
        name: pd.concat(tables, ignore_index=True) for name, tables in collected.items()
    }
    summaries = summarize(combined)
    validation = validate(summaries, output)
    summaries["validation"] = validation
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in summaries.items():
        table.to_csv(tables_dir / f"med1_function_{name}.csv", index=False)
    print(
        f"Wrote functional MED1 bridge analysis for {len(tasks)} trajectories; "
        f"validation checks={len(validation)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
