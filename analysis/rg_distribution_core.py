"""Existing dense-Rg extraction and trajectory summaries, without hypothesis tests."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import MDAnalysis as mda
import run_balanced_analysis as base
CHAIN_SYSTEMS = ("M10O90D1", "M10O90D1-S2")

REPEATS = tuple(base.REPEATS)


RG_SYSTEMS = (
    "D1",
    "M0O100D1",
    "M10O90D1",
    "M20O80D1",
    "M50O0D1",
    "M10O90D1-S2",
)


LABELS = {
    "D1": "DNA only",
    "M0O100D1": "OCT4-only +DNA",
    "M10O90D1": "10% MED1 +DNA / Full",
    "M20O80D1": "20% MED1 +DNA",
    "M50O0D1": "MED1-only +DNA",
    "M10O90D1-S2": "Split-2 +DNA",
    "M10O90D1-S4": "Split-4 +DNA*",
}


RG_FRAMES = np.arange(1001, 5001, dtype=int)


def task_lookup() -> tuple[pd.DataFrame, dict[tuple[str, str], base.Task]]:
    manifest = base.read_manifest(base.PACKAGE_ROOT, base.DEFAULT_DATA_ROOT)
    rows = manifest[manifest["analysis_included"] & manifest["n_dna"].eq(1)]
    lookup: dict[tuple[str, str], base.Task] = {}
    for row in rows.itertuples(index=False):
        task = base.task_from_row(pd.Series(row._asdict()))
        lookup[(task.system, task.replicate)] = task
    return manifest, lookup


def dna_source(
    system: str,
    replicate: str,
    lookup: dict[tuple[str, str], base.Task],
) -> tuple[Path, float, slice]:
    if system == "D1":
        path = base.PROJECT_ROOT / "data/dna_only" / replicate / "D1"
        universe = mda.Universe(str(path / "start.pdb"))
        return path, 75.0, slice(len(universe.atoms) - base.DNA_LENGTH, len(universe.atoms))
    task = lookup[(system, replicate)]
    molecules = base.read_molecule_map(task)
    dna_node = int(molecules.dna_nodes[0])
    return (
        task.path,
        task.box_nm,
        slice(int(molecules.starts[dna_node]), int(molecules.stops[dna_node])),
    )


def dense_rg_table(
    lookup: dict[tuple[str, str], base.Task],
    tables_dir: Path,
    *,
    recompute: bool,
) -> pd.DataFrame:
    output_path = tables_dir / "dna_rg_distribution_hypothesis_validation_framewise.csv"
    expected_pairs = {(system, rep) for system in RG_SYSTEMS for rep in REPEATS}
    if output_path.exists() and not recompute:
        cached = pd.read_csv(output_path)
        counts = cached.groupby(["system", "replicate"]).size()
        if set(counts.index) == expected_pairs and set(counts) == {4000}:
            cached["system_label"] = cached["system"].map(LABELS)
            return cached

    rows: list[dict[str, object]] = []
    reusable_path = tables_dir / "split2_dna_compaction_framewise.csv"
    reusable_systems: set[str] = set()
    if reusable_path.exists() and not recompute:
        reusable = pd.read_csv(
            reusable_path,
            usecols=["system", "replicate", "frame", "time_us", "DNA_Rg_nm"],
        )
        for system in ("D1", *CHAIN_SYSTEMS):
            subset = reusable[reusable["system"].eq(system)]
            counts = subset.groupby("replicate").size()
            if set(counts.index) == set(REPEATS) and set(counts) == {4000}:
                reusable_systems.add(system)
                for row in subset.itertuples(index=False):
                    rows.append(
                        {
                            "system": row.system,
                            "system_label": LABELS[row.system],
                            "replicate": row.replicate,
                            "frame": int(row.frame),
                            "time_us": float(row.time_us),
                            "DNA_Rg_nm": float(row.DNA_Rg_nm),
                            "source": "validated dense Split-2 geometry cache",
                            "sampling_interval_ns": 1.0,
                        }
                    )

    for system in RG_SYSTEMS:
        if system in reusable_systems:
            continue
        for replicate in REPEATS:
            path, box_nm, dna_slice = dna_source(system, replicate, lookup)
            universe = mda.Universe(str(path / "start.pdb"), str(path / "output.dcd"))
            if len(universe.trajectory) < 5000:
                raise RuntimeError(f"{replicate}/{system}: fewer than 5000 frames")
            for frame in RG_FRAMES:
                universe.trajectory[int(frame) - 1]
                dna = universe.atoms.positions[dna_slice].astype(np.float64) / 10.0
                dna = base.unwrap_dna(dna, box_nm)
                centered = dna - dna.mean(axis=0)
                rg = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
                rows.append(
                    {
                        "system": system,
                        "system_label": LABELS[system],
                        "replicate": replicate,
                        "frame": int(frame),
                        "time_us": float(frame / 1000.0),
                        "DNA_Rg_nm": rg,
                        "source": "raw DCD recomputation",
                        "sampling_interval_ns": 1.0,
                    }
                )
            print(f"Dense Rg: {replicate}/{system} (4000 frames)", flush=True)
    result = pd.DataFrame(rows).sort_values(["system", "replicate", "frame"])
    counts = result.groupby(["system", "replicate"]).size()
    if set(counts.index) != expected_pairs or set(counts) != {4000}:
        raise RuntimeError(f"Dense Rg frame-count failure: {counts.to_dict()}")
    return result.reset_index(drop=True)


def summarize_rg(framewise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat = (
        framewise.groupby(["system", "system_label", "replicate"], observed=True)[
            "DNA_Rg_nm"
        ]
        .agg(
            trajectory_mean_nm="mean",
            trajectory_sd_nm="std",
            trajectory_median_nm="median",
            q05_nm=lambda x: x.quantile(0.05),
            q25_nm=lambda x: x.quantile(0.25),
            q75_nm=lambda x: x.quantile(0.75),
            q95_nm=lambda x: x.quantile(0.95),
            n_frames="count",
        )
        .reset_index()
    )
    group = (
        repeat.groupby(["system", "system_label"], observed=True)["trajectory_mean_nm"]
        .agg(group_mean_nm="mean", between_trajectory_sd_nm="std", n_trajectories="count")
        .reset_index()
    )
    return repeat, group
