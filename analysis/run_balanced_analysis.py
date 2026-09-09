#!/usr/bin/env python3
"""Paper analysis for the corrected, balanced MED1/OCT4/DNA simulations.

The original paper workflow assumed one 75-nm box and DNA formulations made
by replacing a protein chain.  The corrected design instead keeps protein
residue concentration fixed along the composition series, uses a different
box length for each composition, and appends DNA without removing protein.

This driver deliberately reads the task manifest and per-task molecule map.
It samples 40 evenly spaced production frames (1.001--4.901 microseconds),
uses explicit periodic boundary conditions, and caches every trajectory
separately.  Adding a newly completed trajectory therefore invalidates only
that one cache entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox
from MDAnalysis.lib.nsgrid import FastNS
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PACKAGE_ROOT = (
    PROJECT_ROOT / "simulation"
)
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "tasks"
DEFAULT_OUTPUT = HERE / "outputs"

ANALYSIS_VERSION = "balanced-paper-v2-publication-panels"
DATASET_POLICY = "m15-m50-three-repeats-20260820"
REPEATS = ["work", "work-1", "work-2"]
PRODUCTION_START = 1001
PRODUCTION_END = 5000
STRIDE = 100
CUTOFFS_NM = [1.0, 1.2, 1.4]
PRIMARY_CUTOFF_NM = 1.2

TYPE_MED1 = 0
TYPE_OCT4 = 1
TYPE_DNA = 2
TYPE_BY_NAME = {"MED1": TYPE_MED1, "OCT4": TYPE_OCT4, "DNA": TYPE_DNA}
TYPE_NAME = {TYPE_MED1: "MED1", TYPE_OCT4: "OCT4", TYPE_DNA: "DNA"}

MED1_LENGTH = 1581
OCT4_LENGTH = 360
DNA_LENGTH = 400

COMPOSITION_SYSTEMS = [
    "M0O100",
    "M5O95",
    "M10O90",
    "M15O85",
    "M20O80",
    "M25O75",
    "M30O70",
    "M50O50",
]
DNA_PAIRS = [
    ("M0O100", "M0O100D1", "0% MED1"),
    ("M10O90", "M10O90D1", "10% MED1"),
    ("M20O80", "M20O80D1", "20% MED1"),
    ("M50O0", "M50O0D1", "pure MED1"),
]
DNA_SYSTEMS = [pair[1] for pair in DNA_PAIRS]
PUBLICATION_DNA_PAIRS = [DNA_PAIRS[0], DNA_PAIRS[1], DNA_PAIRS[3]]
PUBLICATION_DNA_SYSTEMS = [pair[1] for pair in PUBLICATION_DNA_PAIRS]
FULL_SYSTEMS = COMPOSITION_SYSTEMS + ["M50O0"] + DNA_SYSTEMS
SPLIT_SYSTEMS = [
    "M10O90-S2",
    "M10O90-S4",
    "M10O90D1-S2",
    "M10O90D1-S4",
]
ANALYSIS_SYSTEMS = FULL_SYSTEMS + SPLIT_SYSTEMS
EXPLICITLY_INCLUDED_TASKS = {
    ("M15O85", "work-1"),
    ("M50O50", "work-2"),
}

COLORS = {
    "OCT4": "#4C78A8",
    "MED1": "#E45756",
    "DNA": "#54A24B",
    "mixed": "#B279A2",
    "neutral": "#6E6E6E",
    "light": "#C9C9C9",
}

AA_CHARGE = {
    "ARG": 1.0,
    "LYS": 1.0,
    "HIS": 0.5,
    "ASP": -1.0,
    "GLU": -1.0,
}

PUBLICATION_DISTANCE_BIN_WIDTH_NM = 0.25
PUBLICATION_DISTANCE_EDGES_NM = np.concatenate(
    [
        np.arange(
            0.0,
            60.0 + PUBLICATION_DISTANCE_BIN_WIDTH_NM,
            PUBLICATION_DISTANCE_BIN_WIDTH_NM,
        ),
        np.asarray([np.inf]),
    ]
)


def configure_style() -> None:
    """Retain the typography, line weights, and palette of the old figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.5,
            "errorbar.capsize": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(title, pad=5)
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )


def set_panel_title(
    ax: plt.Axes,
    label: str,
    title: str,
    *,
    title_size: float = 9.5,
    label_size: float = 11,
    title_pad: float = 5,
    label_x: float = -0.10,
    label_y: float = 1.02,
) -> None:
    """Original paper-figure panel-title placement."""
    ax.set_title(title, fontsize=title_size, pad=title_pad)
    ax.text(
        label_x,
        label_y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=label_size,
        fontweight="bold",
        clip_on=False,
    )


def save_figure(
    fig: plt.Figure,
    path: Path,
    target_pixels: tuple[int, int] | None = None,
    target_pdf_points: tuple[float, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox_inches: str | Bbox = "tight"
    pad_inches = 0.03
    if target_pixels is not None:
        # New system labels alter a tight bounding box by a few pixels even
        # when figsize/GridSpec are unchanged. Center the original artifact
        # dimensions on the current tight box so structural figures retain
        # exactly the pre-migration exported size and aspect ratio.
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.03)
        target_width = target_pixels[0] / 600.0
        target_height = target_pixels[1] / 600.0
        bbox_inches = Bbox.from_bounds(
            tight.x0 + (tight.width - target_width) / 2,
            tight.y0 + (tight.height - target_height) / 2,
            target_width,
            target_height,
        )
        pad_inches = 0
    fig.savefig(
        path.with_suffix(".png"),
        dpi=600,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    pdf_bbox_inches = bbox_inches
    pdf_pad_inches = pad_inches
    if target_pdf_points is not None:
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.03)
        target_width = target_pdf_points[0] / 72.0
        target_height = target_pdf_points[1] / 72.0
        pdf_bbox_inches = Bbox.from_bounds(
            tight.x0 + (tight.width - target_width) / 2,
            tight.y0 + (tight.height - target_height) / 2,
            target_width,
            target_height,
        )
        pdf_pad_inches = 0
    fig.savefig(
        path.with_suffix(".pdf"),
        dpi=600,
        bbox_inches=pdf_bbox_inches,
        pad_inches=pdf_pad_inches,
    )
    plt.close(fig)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Task:
    system: str
    replicate: str
    n_med1: int
    n_oct4: int
    n_dna: int
    med1_fragments: int
    box_nm: float
    path: Path

    @property
    def expected_atoms(self) -> int:
        return self.n_med1 * MED1_LENGTH + self.n_oct4 * OCT4_LENGTH + self.n_dna * DNA_LENGTH

    @property
    def med1_fraction(self) -> float:
        n = self.n_med1 + self.n_oct4
        return self.n_med1 / n if n else float("nan")


@dataclass
class Molecules:
    atom_molecule: np.ndarray
    atom_residue: np.ndarray
    molecule_type: np.ndarray
    molecule_length: np.ndarray
    parent_med1: np.ndarray
    starts: np.ndarray
    stops: np.ndarray

    @property
    def protein_nodes(self) -> np.ndarray:
        return np.flatnonzero(self.molecule_type != TYPE_DNA)

    @property
    def dna_nodes(self) -> np.ndarray:
        return np.flatnonzero(self.molecule_type == TYPE_DNA)


def read_manifest(package_root: Path, data_root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(package_root / "task_manifest.tsv", sep="\t")
    manifest = manifest[manifest["system"].isin(ANALYSIS_SYSTEMS)].copy()
    manifest["path"] = manifest.apply(
        lambda row: data_root / str(row["replicate"]) / str(row["system"]),
        axis=1,
    )
    manifest["available"] = manifest["path"].map(
        lambda path: (path / "DONE").exists()
        and (path / "start.pdb").exists()
        and (path / "output.dcd").exists()
        and (path / "molecule_map.tsv").exists()
    )
    def active_run_parameters(path: Path) -> dict[str, float]:
        metadata_path = path / "run_metadata.json"
        if not metadata_path.exists():
            return {}
        metadata = json.loads(metadata_path.read_text())
        # Early 10-fs production metadata predates run_parameters; the package
        # runner default and task protocol are 10 fs. Every 5-fs rerun records
        # run_parameters explicitly.
        return metadata.get("run_parameters", {})

    active_parameters = manifest["path"].map(active_run_parameters)
    manifest["timestep_fs"] = active_parameters.map(
        lambda values: float(values.get("timestep_fs", 10.0))
    )
    manifest["total_steps"] = [
        int(values.get("total_steps", planned))
        for values, planned in zip(active_parameters, manifest["total_steps"])
    ]
    manifest["output_interval"] = [
        int(values.get("output_interval", planned))
        for values, planned in zip(active_parameters, manifest["output_interval"])
    ]
    manifest["output_spacing_ns"] = (
        manifest["timestep_fs"] * manifest["output_interval"] / 1.0e6
    )
    manifest["duration_us"] = (
        manifest["timestep_fs"] * manifest["total_steps"] / 1.0e9
    )
    explicit_inclusion = pd.Series(
        [
            (str(system), str(replicate)) in EXPLICITLY_INCLUDED_TASKS
            for system, replicate in zip(manifest["system"], manifest["replicate"])
        ],
        index=manifest.index,
    )
    manifest["analysis_included"] = manifest["available"] & (
        manifest["timestep_fs"].eq(10.0) | explicit_inclusion
    )
    manifest["inclusion_basis"] = np.select(
        [
            ~manifest["available"],
            explicit_inclusion,
            manifest["analysis_included"],
        ],
        ["unavailable", "explicit_third_repeat", "standard"],
        default="excluded",
    )
    return manifest


def task_from_row(row: pd.Series) -> Task:
    return Task(
        system=str(row["system"]),
        replicate=str(row["replicate"]),
        n_med1=int(row["n_med1"]),
        n_oct4=int(row["n_oct4"]),
        n_dna=int(row["n_dna"]),
        med1_fragments=int(row["med1_fragments"]),
        box_nm=float(row["box_nm"]),
        path=Path(row["path"]),
    )


def read_molecule_map(task: Task) -> Molecules:
    table = pd.read_csv(task.path / "molecule_map.tsv", sep="\t")
    n_atoms = int(table["particle_stop_exclusive"].max())
    atom_molecule = np.empty(n_atoms, dtype=np.int32)
    atom_residue = np.empty(n_atoms, dtype=np.int32)
    molecule_type = np.empty(len(table), dtype=np.int8)
    molecule_length = np.empty(len(table), dtype=np.int32)
    parent_med1 = np.full(len(table), -1, dtype=np.int32)
    starts = table["particle_start"].to_numpy(dtype=np.int32)
    stops = table["particle_stop_exclusive"].to_numpy(dtype=np.int32)
    for molecule, row in table.iterrows():
        start = int(row["particle_start"])
        stop = int(row["particle_stop_exclusive"])
        residue_start = int(row["residue_start"])
        atom_molecule[start:stop] = molecule
        atom_residue[start:stop] = np.arange(residue_start - 1, residue_start - 1 + stop - start)
        molecule_type[molecule] = TYPE_BY_NAME[str(row["molecule_type"])]
        molecule_length[molecule] = stop - start
        if molecule_type[molecule] == TYPE_MED1:
            parent_med1[molecule] = int(row["parent_med1"]) - 1
    if n_atoms != task.expected_atoms:
        raise RuntimeError(f"{task.replicate}/{task.system}: molecule map has {n_atoms} atoms, expected {task.expected_atoms}")
    return Molecules(
        atom_molecule=atom_molecule,
        atom_residue=atom_residue,
        molecule_type=molecule_type,
        molecule_length=molecule_length,
        parent_med1=parent_med1,
        starts=starts,
        stops=stops,
    )


def atom_pairs_with_distances(positions_a: np.ndarray, box_nm: float) -> tuple[np.ndarray, np.ndarray]:
    box_a = np.asarray([box_nm * 10] * 3 + [90, 90, 90], dtype=np.float32)
    wrapped = np.mod(positions_a, box_nm * 10).astype(np.float32, copy=False)
    result = FastNS(max(CUTOFFS_NM) * 10, wrapped, box=box_a, pbc=True).self_search()
    return result.get_pairs().astype(np.int32, copy=False), result.get_pair_distances().astype(np.float32, copy=False) / 10


def unique_molecule_pairs(atom_pairs: np.ndarray, atom_molecule: np.ndarray) -> np.ndarray:
    if not len(atom_pairs):
        return np.empty((0, 2), dtype=np.int32)
    first = atom_molecule[atom_pairs[:, 0]]
    second = atom_molecule[atom_pairs[:, 1]]
    keep = first != second
    low = np.minimum(first[keep], second[keep])
    high = np.maximum(first[keep], second[keep])
    if not len(low):
        return np.empty((0, 2), dtype=np.int32)
    return np.unique(np.column_stack([low, high]).astype(np.int32), axis=0)


def components(n_nodes: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parent = np.arange(n_nodes, dtype=np.int32)
    rank = np.zeros(n_nodes, dtype=np.int8)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            nxt = int(parent[node])
            parent[node] = root
            node = nxt
        return root

    for first, second in edges:
        a, b = find(int(first)), find(int(second))
        if a == b:
            continue
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
    roots = np.asarray([find(i) for i in range(n_nodes)], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels, np.bincount(labels)


def protein_components(molecules: Molecules, molecule_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    protein_nodes = molecules.protein_nodes
    node_to_local = np.full(len(molecules.molecule_type), -1, dtype=np.int32)
    node_to_local[protein_nodes] = np.arange(len(protein_nodes), dtype=np.int32)
    if len(molecule_pairs):
        keep = (
            (molecules.molecule_type[molecule_pairs[:, 0]] != TYPE_DNA)
            & (molecules.molecule_type[molecule_pairs[:, 1]] != TYPE_DNA)
        )
        edges = node_to_local[molecule_pairs[keep]]
    else:
        edges = np.empty((0, 2), dtype=np.int32)
    labels, sizes = components(len(protein_nodes), edges)
    return labels, sizes, edges


def unwrap_segment(coords_nm: np.ndarray, box_nm: float) -> np.ndarray:
    wrapped = np.mod(coords_nm, box_nm)
    if len(wrapped) < 2:
        return wrapped.copy()
    delta = np.diff(wrapped, axis=0)
    delta -= box_nm * np.round(delta / box_nm)
    whole = np.empty_like(wrapped)
    whole[0] = wrapped[0]
    whole[1:] = whole[0] + np.cumsum(delta, axis=0)
    return whole


def unwrap_dna(coords_nm: np.ndarray, box_nm: float) -> np.ndarray:
    half = len(coords_nm) // 2
    first = unwrap_segment(coords_nm[:half], box_nm)
    second = unwrap_segment(coords_nm[half:], box_nm)
    shift = first[-1] + (second[0] - first[-1] - box_nm * np.round((second[0] - first[-1]) / box_nm)) - second[0]
    return np.vstack([first, second + shift])


def dna_shape(coords_nm: np.ndarray, box_nm: float) -> tuple[float, float, float]:
    coords = unwrap_dna(coords_nm, box_nm)
    centered = coords - coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    half = len(coords) // 2
    end_a = np.vstack([coords[:5], coords[-5:]]).mean(axis=0)
    end_b = np.vstack([coords[half - 5 : half], coords[half : half + 5]]).mean(axis=0)
    ree = float(np.linalg.norm(end_b - end_a))
    evals = np.sort(np.linalg.eigvalsh(centered.T @ centered / len(coords)))[::-1]
    denominator = float(evals.sum() ** 2)
    asphericity = float(
        ((evals[0] - evals[1]) ** 2 + (evals[1] - evals[2]) ** 2 + (evals[2] - evals[0]) ** 2)
        / (2 * denominator)
    )
    return rg, ree, asphericity


def task_signature(task: Task) -> dict[str, object]:
    dcd = task.path / "output.dcd"
    metadata = json.loads((task.path / "run_metadata.json").read_text())
    return {
        "analysis_version": ANALYSIS_VERSION,
        "system": task.system,
        "replicate": task.replicate,
        "box_nm": task.box_nm,
        "output_dcd_size": dcd.stat().st_size,
        "output_dcd_mtime_ns": dcd.stat().st_mtime_ns,
        "start_pdb_sha256": metadata["start_pdb_sha256"] if "start_pdb_sha256" in metadata else sha256(task.path / "start.pdb"),
        "system_xml_sha256": metadata["system_xml_sha256"] if "system_xml_sha256" in metadata else sha256(task.path / "system.xml"),
        "frames": list(range(PRODUCTION_START, PRODUCTION_END + 1, STRIDE)),
        "cutoffs_nm": CUTOFFS_NM,
    }


def cache_paths(cache_dir: Path, task: Task) -> dict[str, Path]:
    stem = f"{task.replicate}__{task.system}"
    return {
        "meta": cache_dir / f"{stem}.json",
        "frame": cache_dir / f"{stem}.frames.csv",
        "cluster": cache_dir / f"{stem}.clusters.csv",
        "degree": cache_dir / f"{stem}.degrees.csv",
        "arrays": cache_dir / f"{stem}.arrays.npz",
    }


def cache_valid(paths: dict[str, Path], signature: dict[str, object]) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        return json.loads(paths["meta"].read_text()) == signature
    except (OSError, json.JSONDecodeError):
        return False


def safe_probability(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def analyze_task(task: Task, cache_dir: Path, recompute: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    paths = cache_paths(cache_dir, task)
    signature = task_signature(task)
    if not recompute and cache_valid(paths, signature):
        arrays_file = np.load(paths["arrays"])
        arrays = {key: arrays_file[key] for key in arrays_file.files}
        return pd.read_csv(paths["frame"]), pd.read_csv(paths["cluster"]), pd.read_csv(paths["degree"]), arrays

    print(f"Analyzing {task.replicate}/{task.system}", flush=True)
    molecules = read_molecule_map(task)
    universe = mda.Universe(str(task.path / "start.pdb"), str(task.path / "output.dcd"))
    if len(universe.atoms) != task.expected_atoms:
        raise RuntimeError(f"{task.replicate}/{task.system}: atom count mismatch")
    if len(universe.trajectory) < PRODUCTION_END:
        raise RuntimeError(f"{task.replicate}/{task.system}: only {len(universe.trajectory)} frames")

    frame_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    degree_counts: dict[int, np.ndarray] = {}
    frames = list(range(PRODUCTION_START, PRODUCTION_END + 1, STRIDE))
    site_arrays = {
        "oct4_dna": np.zeros(OCT4_LENGTH, dtype=np.float64),
        "med1_dna": np.zeros(MED1_LENGTH, dtype=np.float64),
        "med1_oct4_med1": np.zeros(MED1_LENGTH, dtype=np.float64),
        "med1_oct4_oct4": np.zeros(OCT4_LENGTH, dtype=np.float64),
        "om_map": np.zeros((math.ceil(MED1_LENGTH / 50), math.ceil(OCT4_LENGTH / 20)), dtype=np.float64),
        "od_map": np.zeros((math.ceil(OCT4_LENGTH / 20), math.ceil((DNA_LENGTH // 2) / 10)), dtype=np.float64),
        "md_map": np.zeros((math.ceil(MED1_LENGTH / 50), math.ceil((DNA_LENGTH // 2) / 10)), dtype=np.float64),
        "publication_local_edges": np.zeros(4, dtype=np.int64),
        "publication_local_opportunities": np.zeros(4, dtype=np.int64),
        "publication_distance_histogram": np.zeros(
            len(PUBLICATION_DISTANCE_EDGES_NM) - 1,
            dtype=np.int64,
        ),
        "publication_distance_q90_nm": np.full(len(frames), np.nan),
        "publication_distance_within_10nm": np.full(len(frames), np.nan),
    }

    protein_nodes = molecules.protein_nodes
    n_physical_proteins = len(protein_nodes)
    protein_node_to_local = np.full(
        len(molecules.molecule_type), -1, dtype=np.int32
    )
    protein_node_to_local[protein_nodes] = np.arange(
        n_physical_proteins, dtype=np.int32
    )
    possible_pair_first, possible_pair_second = np.triu_indices(
        n_physical_proteins, k=1
    )
    protein_total_residues = int(molecules.molecule_length[protein_nodes].sum())
    atom_type = molecules.molecule_type[molecules.atom_molecule]
    for frame_index, frame in enumerate(frames):
        universe.trajectory[frame - 1]
        atom_pairs14, distances_nm = atom_pairs_with_distances(universe.atoms.positions, task.box_nm)
        cutoff_pairs: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for cutoff in CUTOFFS_NM:
            atoms = atom_pairs14[distances_nm < cutoff]
            cutoff_pairs[cutoff] = (atoms, unique_molecule_pairs(atoms, molecules.atom_molecule))
            labels, sizes, _ = protein_components(molecules, cutoff_pairs[cutoff][1])
            largest_label = int(np.argmax(sizes))
            largest_size = int(sizes[largest_label])
            largest_nodes = protein_nodes[labels == largest_label]
            largest_residues = int(molecules.molecule_length[largest_nodes].sum())
            ordered_sizes = np.sort(sizes)[::-1]
            finite_sizes = ordered_sizes[1:]
            finite_cluster_mean_size = (
                float(np.sum(finite_sizes.astype(float) ** 2) / finite_sizes.sum())
                if len(finite_sizes) and finite_sizes.sum() > 0
                else float("nan")
            )
            cluster_rows.append(
                {
                    "replicate": task.replicate,
                    "system": task.system,
                    "frame": frame,
                    "time_us": frame * 0.001,
                    "cutoff_nm": cutoff,
                    "largest_cluster_size": largest_size,
                    "largest_cluster_fraction": largest_size / n_physical_proteins,
                    "largest_cluster_residue_fraction": largest_residues / protein_total_residues,
                    "second_cluster_size": int(np.sort(sizes)[-2]) if len(sizes) > 1 else 0,
                    "second_cluster_fraction": (int(np.sort(sizes)[-2]) / n_physical_proteins) if len(sizes) > 1 else 0.0,
                    "finite_cluster_mean_size": finite_cluster_mean_size,
                    "N_clusters_size_ge_2": int(np.sum(sizes >= 2)),
                    "monomer_fraction": float(np.sum(sizes == 1) / n_physical_proteins),
                }
            )

        atom_pairs, molecule_pairs = cutoff_pairs[PRIMARY_CUTOFF_NM]
        labels, sizes, protein_edges = protein_components(molecules, molecule_pairs)
        degrees = np.zeros(len(molecules.molecule_type), dtype=np.int32)
        if len(molecule_pairs):
            np.add.at(degrees, molecule_pairs.ravel(), 1)
        protein_degree = degrees[protein_nodes]
        histogram = np.bincount(protein_degree, minlength=n_physical_proteins)
        degree_counts[frame] = histogram

        pair_counts = {"OO": 0, "MM": 0, "OM": 0, "OD": 0, "MD": 0}
        for first, second in molecule_pairs:
            pair = tuple(sorted((int(molecules.molecule_type[first]), int(molecules.molecule_type[second]))))
            if pair == (TYPE_OCT4, TYPE_OCT4):
                pair_counts["OO"] += 1
            elif pair == (TYPE_MED1, TYPE_MED1):
                pair_counts["MM"] += 1
            elif pair == (TYPE_MED1, TYPE_OCT4):
                pair_counts["OM"] += 1
            elif pair == (TYPE_OCT4, TYPE_DNA):
                pair_counts["OD"] += 1
            elif pair == (TYPE_MED1, TYPE_DNA):
                pair_counts["MD"] += 1
        med1_nodes = np.flatnonzero(molecules.molecule_type == TYPE_MED1)
        oct4_nodes = np.flatnonzero(molecules.molecule_type == TYPE_OCT4)
        dna_nodes = molecules.dna_nodes
        possible = {
            "OO": len(oct4_nodes) * (len(oct4_nodes) - 1) / 2,
            "MM": len(med1_nodes) * (len(med1_nodes) - 1) / 2,
            "OM": len(oct4_nodes) * len(med1_nodes),
            "OD": len(oct4_nodes) * len(dna_nodes),
            "MD": len(med1_nodes) * len(dna_nodes),
        }
        dna_neighbors = set()
        for pair in molecule_pairs:
            if any(node in dna_nodes for node in pair):
                dna_neighbors.update(int(node) for node in pair if node not in dna_nodes)
        oct4_dna_neighbors = {node for node in dna_neighbors if molecules.molecule_type[node] == TYPE_OCT4}
        med1_dna_neighbors = {node for node in dna_neighbors if molecules.molecule_type[node] == TYPE_MED1}

        largest_label = int(np.argmax(sizes))
        direct_lcf = float(sizes[largest_label] / n_physical_proteins)
        augmented_labels, augmented_sizes = components(len(molecules.molecule_type), molecule_pairs)
        augmented_protein_counts = np.bincount(
            augmented_labels,
            weights=(molecules.molecule_type != TYPE_DNA).astype(float),
            minlength=len(augmented_sizes),
        )
        augmented_lcf = float(augmented_protein_counts.max() / n_physical_proteins)
        dna_component_protein_fraction = float("nan")
        dna_component_med1 = float("nan")
        dna_component_oct4 = float("nan")
        if len(dna_nodes):
            component_nodes = np.flatnonzero(augmented_labels == augmented_labels[dna_nodes[0]])
            dna_component_med1 = float(np.sum(molecules.molecule_type[component_nodes] == TYPE_MED1))
            dna_component_oct4 = float(np.sum(molecules.molecule_type[component_nodes] == TYPE_OCT4))
            dna_component_protein_fraction = (dna_component_med1 + dna_component_oct4) / n_physical_proteins

        row: dict[str, object] = {
            "replicate": task.replicate,
            "system": task.system,
            "frame": frame,
            "time_us": frame * 0.001,
            "box_nm": task.box_nm,
            "n_MED1_sequence_equivalents": task.n_med1,
            "n_MED1_physical_chains": len(med1_nodes),
            "n_OCT4": len(oct4_nodes),
            "n_DNA": len(dna_nodes),
            "n_physical_proteins": n_physical_proteins,
            "direct_largest_protein_fraction": direct_lcf,
            "augmented_largest_protein_fraction": augmented_lcf,
            "DNA_component_protein_fraction": dna_component_protein_fraction,
            "DNA_component_MED1": dna_component_med1,
            "DNA_component_OCT4": dna_component_oct4,
            "n_PP_edges": len(protein_edges),
            "PP_edge_density": safe_probability(len(protein_edges), n_physical_proteins * (n_physical_proteins - 1) / 2),
            "OO_contact_probability": safe_probability(pair_counts["OO"], possible["OO"]),
            "MM_contact_probability": safe_probability(pair_counts["MM"], possible["MM"]),
            "OM_contact_probability": safe_probability(pair_counts["OM"], possible["OM"]),
            "OD_contact_probability": safe_probability(pair_counts["OD"], possible["OD"]),
            "MD_contact_probability": safe_probability(pair_counts["MD"], possible["MD"]),
            "OCT4_mean_degree": float(degrees[oct4_nodes].mean()) if len(oct4_nodes) else float("nan"),
            "MED1_mean_degree": float(degrees[med1_nodes].mean()) if len(med1_nodes) else float("nan"),
            "DNA_mean_degree": float(degrees[dna_nodes].mean()) if len(dna_nodes) else float("nan"),
            "OCT4_fraction_contacting_DNA": safe_probability(len(oct4_dna_neighbors), len(oct4_nodes)),
            "MED1_fraction_contacting_DNA": safe_probability(len(med1_dna_neighbors), len(med1_nodes)),
        }

        first = atom_pairs[:, 0]
        second = atom_pairs[:, 1]
        first_type = atom_type[first]
        second_type = atom_type[second]

        if task.system in PUBLICATION_DNA_SYSTEMS and len(dna_nodes):
            edge_flags = np.zeros(
                n_physical_proteins * n_physical_proteins,
                dtype=bool,
            )
            if len(protein_edges):
                edge_flags[
                    protein_edges[:, 0] * n_physical_proteins
                    + protein_edges[:, 1]
                ] = True
            possible_edge_flags = edge_flags[
                possible_pair_first * n_physical_proteins
                + possible_pair_second
            ]

            dna_node = int(dna_nodes[0])
            first_is_dna = molecules.atom_molecule[first] == dna_node
            second_is_dna = molecules.atom_molecule[second] == dna_node
            protein_dna_mask = first_is_dna ^ second_is_dna
            protein_atoms = np.where(
                first_is_dna[protein_dna_mask],
                second[protein_dna_mask],
                first[protein_dna_mask],
            )
            dna_atoms = np.where(
                first_is_dna[protein_dna_mask],
                first[protein_dna_mask],
                second[protein_dna_mask],
            )
            bound = np.zeros(n_physical_proteins, dtype=bool)
            if len(protein_atoms):
                contact_nodes = np.unique(molecules.atom_molecule[protein_atoms])
                bound[protein_node_to_local[contact_nodes]] = True
            window_contacts = np.zeros(
                (n_physical_proteins, math.ceil((DNA_LENGTH // 2) / 20)),
                dtype=bool,
            )
            if len(protein_atoms):
                dna_local = dna_atoms - int(molecules.starts[dna_node])
                physical_dna = np.minimum(
                    dna_local,
                    DNA_LENGTH - 1 - dna_local,
                )
                unique_windows = np.unique(
                    np.column_stack(
                        [
                            protein_node_to_local[
                                molecules.atom_molecule[protein_atoms]
                            ],
                            physical_dna // 20,
                        ]
                    ),
                    axis=0,
                )
                window_contacts[
                    unique_windows[:, 0].astype(int),
                    unique_windows[:, 1].astype(int),
                ] = True
            shared_window = np.any(
                window_contacts[possible_pair_first]
                & window_contacts[possible_pair_second],
                axis=1,
            )
            both_bound = (
                bound[possible_pair_first] & bound[possible_pair_second]
            )
            local_category = np.zeros(
                len(possible_pair_first), dtype=np.int8
            )
            local_category[
                bound[possible_pair_first] ^ bound[possible_pair_second]
            ] = 1
            local_category[both_bound & ~shared_window] = 2
            local_category[shared_window] = 3
            site_arrays["publication_local_opportunities"] += np.bincount(
                local_category, minlength=4
            )
            site_arrays["publication_local_edges"] += np.bincount(
                local_category,
                weights=possible_edge_flags.astype(np.int64),
                minlength=4,
            ).astype(np.int64)

            direct_oct4 = set(map(int, oct4_dna_neighbors))
            direct_med1 = set(map(int, med1_dna_neighbors))
            indirect_med1: set[int] = set()
            for pair_first_node, pair_second_node in protein_edges:
                first_node = int(protein_nodes[int(pair_first_node)])
                second_node = int(protein_nodes[int(pair_second_node)])
                if (
                    first_node in direct_oct4
                    and molecules.molecule_type[second_node] == TYPE_MED1
                    and second_node not in direct_med1
                ):
                    indirect_med1.add(second_node)
                if (
                    second_node in direct_oct4
                    and molecules.molecule_type[first_node] == TYPE_MED1
                    and first_node not in direct_med1
                ):
                    indirect_med1.add(first_node)
            associated_nodes = direct_oct4 | direct_med1 | indirect_med1
            if associated_nodes:
                wrapped_positions_nm = np.mod(
                    universe.atoms.positions.astype(np.float64) / 10.0,
                    task.box_nm,
                )
                dna_start = int(molecules.starts[dna_node])
                dna_stop = int(molecules.stops[dna_node])
                dna_tree = cKDTree(
                    wrapped_positions_nm[dna_start:dna_stop],
                    boxsize=task.box_nm,
                )
                protein_atom_indices = np.flatnonzero(
                    atom_type != TYPE_DNA
                )
                bead_distances = dna_tree.query(
                    wrapped_positions_nm[protein_atom_indices],
                    k=1,
                    workers=1,
                )[0]
                associated_mask = np.isin(
                    molecules.atom_molecule[protein_atom_indices],
                    np.asarray(sorted(associated_nodes), dtype=np.int32),
                )
                associated_distances = bead_distances[associated_mask]
                if len(associated_distances):
                    site_arrays["publication_distance_histogram"] += np.histogram(
                        associated_distances,
                        bins=PUBLICATION_DISTANCE_EDGES_NM,
                    )[0]
                    site_arrays["publication_distance_q90_nm"][frame_index] = np.quantile(
                        associated_distances, 0.90
                    )
                    site_arrays["publication_distance_within_10nm"][frame_index] = np.mean(
                        associated_distances <= 10.0
                    )

        dna_contact_residues: dict[int, set[int]] = {TYPE_OCT4: set(), TYPE_MED1: set()}
        for protein_type, prefix, target in [
            (TYPE_OCT4, "oct4", site_arrays["oct4_dna"]),
            (TYPE_MED1, "med1", site_arrays["med1_dna"]),
        ]:
            mask = ((first_type == protein_type) & (second_type == TYPE_DNA)) | ((first_type == TYPE_DNA) & (second_type == protein_type))
            if np.any(mask):
                protein_atoms = np.where(first_type[mask] == protein_type, first[mask], second[mask])
                dna_atoms = np.where(first_type[mask] == TYPE_DNA, first[mask], second[mask])
                unique_protein_res = np.unique(np.column_stack([molecules.atom_molecule[protein_atoms], molecules.atom_residue[protein_atoms]]), axis=0)
                np.add.at(target, unique_protein_res[:, 1], 1)
                dna_contact_residues[protein_type].update(map(int, molecules.atom_residue[dna_atoms]))
                physical = np.minimum(molecules.atom_residue[dna_atoms], DNA_LENGTH - 1 - molecules.atom_residue[dna_atoms])
                x_res = molecules.atom_residue[protein_atoms]
                x_bin = x_res // (20 if protein_type == TYPE_OCT4 else 50)
                y_bin = physical // 10
                unique_bins = np.unique(np.column_stack([molecules.atom_molecule[protein_atoms], x_bin, y_bin]), axis=0)
                map_key = "od_map" if protein_type == TYPE_OCT4 else "md_map"
                np.add.at(site_arrays[map_key], (unique_bins[:, 1], unique_bins[:, 2]), 1)

        om_mask = ((first_type == TYPE_MED1) & (second_type == TYPE_OCT4)) | ((first_type == TYPE_OCT4) & (second_type == TYPE_MED1))
        if np.any(om_mask):
            med1_atoms = np.where(first_type[om_mask] == TYPE_MED1, first[om_mask], second[om_mask])
            oct4_atoms = np.where(first_type[om_mask] == TYPE_OCT4, first[om_mask], second[om_mask])
            unique_med1 = np.unique(np.column_stack([molecules.atom_molecule[med1_atoms], molecules.atom_residue[med1_atoms]]), axis=0)
            unique_oct4 = np.unique(np.column_stack([molecules.atom_molecule[oct4_atoms], molecules.atom_residue[oct4_atoms]]), axis=0)
            np.add.at(site_arrays["med1_oct4_med1"], unique_med1[:, 1], 1)
            np.add.at(site_arrays["med1_oct4_oct4"], unique_oct4[:, 1], 1)
            unique_bins = np.unique(np.column_stack([molecules.atom_molecule[med1_atoms], molecules.atom_molecule[oct4_atoms], molecules.atom_residue[med1_atoms] // 50, molecules.atom_residue[oct4_atoms] // 20]), axis=0)
            np.add.at(site_arrays["om_map"], (unique_bins[:, 2], unique_bins[:, 3]), 1)

        if len(dna_nodes):
            dna_start, dna_stop = molecules.starts[dna_nodes[0]], molecules.stops[dna_nodes[0]]
            rg, ree, asphericity = dna_shape(universe.atoms.positions[dna_start:dna_stop].astype(float) / 10, task.box_nm)
            row.update(
                {
                    "DNA_Rg_nm": rg,
                    "DNA_Ree_nm": ree,
                    "DNA_asphericity": asphericity,
                    "DNA_fraction_beads_contacting_OCT4": len(dna_contact_residues[TYPE_OCT4]) / DNA_LENGTH,
                    "DNA_fraction_beads_contacting_MED1": len(dna_contact_residues[TYPE_MED1]) / DNA_LENGTH,
                    "DNA_fraction_beads_contacting_any_protein": len(dna_contact_residues[TYPE_OCT4] | dna_contact_residues[TYPE_MED1]) / DNA_LENGTH,
                    "N_OCT4_molecules_contacting_DNA": len(oct4_dna_neighbors),
                    "N_MED1_molecules_contacting_DNA": len(med1_dna_neighbors),
                }
            )
        frame_rows.append(row)

    degree_rows = []
    for frame, histogram in degree_counts.items():
        for degree, count in enumerate(histogram):
            if count:
                degree_rows.append(
                    {
                        "replicate": task.replicate,
                        "system": task.system,
                        "frame": frame,
                        "degree": degree,
                        "count": int(count),
                        "probability": count / n_physical_proteins,
                    }
                )
    arrays = {**site_arrays, "sampled_frames": np.asarray([len(frames)], dtype=np.int32)}
    frame_table = pd.DataFrame(frame_rows)
    cluster_table = pd.DataFrame(cluster_rows)
    degree_table = pd.DataFrame(degree_rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_table.to_csv(paths["frame"], index=False)
    cluster_table.to_csv(paths["cluster"], index=False)
    degree_table.to_csv(paths["degree"], index=False)
    np.savez_compressed(paths["arrays"], **arrays)
    paths["meta"].write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")
    return frame_table, cluster_table, degree_table, arrays


def mean_sd(table: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    grouped = table.groupby(groups, observed=True)[metrics]
    return pd.concat(
        [
            grouped.mean().add_suffix("_mean"),
            grouped.std(ddof=1).add_suffix("_sd"),
            grouped.count().add_suffix("_n"),
        ],
        axis=1,
    ).reset_index()


def summarize_tables(
    framewise: pd.DataFrame,
    clusters: pd.DataFrame,
    degrees: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frame_metrics = [
        column
        for column in [
            "direct_largest_protein_fraction",
            "augmented_largest_protein_fraction",
            "DNA_component_protein_fraction",
            "DNA_component_MED1",
            "DNA_component_OCT4",
            "n_PP_edges",
            "PP_edge_density",
            "OO_contact_probability",
            "MM_contact_probability",
            "OM_contact_probability",
            "OD_contact_probability",
            "MD_contact_probability",
            "OCT4_mean_degree",
            "MED1_mean_degree",
            "DNA_mean_degree",
            "OCT4_fraction_contacting_DNA",
            "MED1_fraction_contacting_DNA",
            "DNA_Rg_nm",
            "DNA_Ree_nm",
            "DNA_asphericity",
            "DNA_fraction_beads_contacting_OCT4",
            "DNA_fraction_beads_contacting_MED1",
            "DNA_fraction_beads_contacting_any_protein",
            "N_OCT4_molecules_contacting_DNA",
            "N_MED1_molecules_contacting_DNA",
        ]
        if column in framewise.columns
    ]
    frame_repeat = (
        framewise.groupby(["system", "replicate"], observed=True)[frame_metrics]
        .mean()
        .reset_index()
    )
    frame_group = mean_sd(frame_repeat, ["system"], frame_metrics)

    cluster_metrics = [
        "largest_cluster_size",
        "largest_cluster_fraction",
        "largest_cluster_residue_fraction",
        "second_cluster_size",
        "second_cluster_fraction",
        "finite_cluster_mean_size",
        "N_clusters_size_ge_2",
        "monomer_fraction",
    ]
    cluster_repeat = (
        clusters.groupby(["system", "replicate", "cutoff_nm"], observed=True)[cluster_metrics]
        .mean()
        .reset_index()
    )
    cluster_group = mean_sd(cluster_repeat, ["system", "cutoff_nm"], cluster_metrics)
    clusters = clusters.copy()
    clusters["production_block"] = ((clusters["frame"] - 1) // 1000).astype(int)
    block_repeat = (
        clusters[np.isclose(clusters["cutoff_nm"], PRIMARY_CUTOFF_NM)]
        .groupby(["system", "replicate", "production_block"], observed=True)[cluster_metrics]
        .mean()
        .reset_index()
    )
    block_group = mean_sd(block_repeat, ["system", "production_block"], cluster_metrics)

    degree_repeat = (
        degrees.groupby(["system", "replicate", "degree"], observed=True)["probability"]
        .mean()
        .reset_index()
    )
    degree_group = mean_sd(degree_repeat, ["system", "degree"], ["probability"])

    paired_rows: list[dict[str, object]] = []
    cluster_index = cluster_repeat.set_index(["system", "replicate", "cutoff_nm"])
    frame_index = frame_repeat.set_index(["system", "replicate"])
    for without, with_dna, label in DNA_PAIRS:
        common = sorted(
            set(frame_repeat.loc[frame_repeat["system"] == without, "replicate"])
            & set(frame_repeat.loc[frame_repeat["system"] == with_dna, "replicate"])
        )
        for repeat in common:
            for cutoff in CUTOFFS_NM:
                without_cluster = cluster_index.loc[(without, repeat, cutoff)]
                with_cluster = cluster_index.loc[(with_dna, repeat, cutoff)]
                paired_rows.append(
                    {
                        "comparison": label,
                        "without_DNA_system": without,
                        "with_DNA_system": with_dna,
                        "replicate": repeat,
                        "cutoff_nm": cutoff,
                        "metric": "largest_cluster_fraction",
                        "without_DNA": without_cluster["largest_cluster_fraction"],
                        "with_DNA": with_cluster["largest_cluster_fraction"],
                        "delta_with_minus_without": with_cluster["largest_cluster_fraction"] - without_cluster["largest_cluster_fraction"],
                    }
                )
            for metric in ["PP_edge_density", "n_PP_edges"]:
                a = float(frame_index.loc[(without, repeat), metric])
                b = float(frame_index.loc[(with_dna, repeat), metric])
                paired_rows.append(
                    {
                        "comparison": label,
                        "without_DNA_system": without,
                        "with_DNA_system": with_dna,
                        "replicate": repeat,
                        "cutoff_nm": PRIMARY_CUTOFF_NM,
                        "metric": metric,
                        "without_DNA": a,
                        "with_DNA": b,
                        "delta_with_minus_without": b - a,
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired_group = mean_sd(
        paired,
        ["comparison", "cutoff_nm", "metric"],
        ["without_DNA", "with_DNA", "delta_with_minus_without"],
    )
    return {
        "frame_repeat": frame_repeat,
        "frame_group": frame_group,
        "cluster_repeat": cluster_repeat,
        "cluster_group": cluster_group,
        "block_repeat": block_repeat,
        "block_group": block_group,
        "degree_repeat": degree_repeat,
        "degree_group": degree_group,
        "paired": paired,
        "paired_group": paired_group,
    }


def summarize_site_arrays(
    manifest: pd.DataFrame,
    task_arrays: dict[tuple[str, str], dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    manifest_index = manifest.set_index(["system", "replicate"])
    profile_spec = [
        ("oct4_dna", "OCT4-DNA", "OCT4", OCT4_LENGTH),
        ("med1_dna", "MED1-DNA", "MED1", MED1_LENGTH),
        ("med1_oct4_med1", "MED1-OCT4", "MED1", MED1_LENGTH),
        ("med1_oct4_oct4", "MED1-OCT4", "OCT4", OCT4_LENGTH),
    ]
    for (system, replicate), arrays in task_arrays.items():
        row = manifest_index.loc[(system, replicate)]
        frames = int(arrays["sampled_frames"][0])
        n_med1_physical = int(row["n_med1"]) * int(row["med1_fragments"])
        denominators = {
            "oct4_dna": frames * int(row["n_oct4"]),
            "med1_dna": frames * n_med1_physical,
            "med1_oct4_med1": frames * n_med1_physical,
            "med1_oct4_oct4": frames * int(row["n_oct4"]),
        }
        for key, interaction, side, length in profile_spec:
            denominator = denominators[key]
            if denominator == 0:
                continue
            probability = arrays[key] / denominator
            for residue in range(length):
                profile_rows.append(
                    {
                        "system": system,
                        "replicate": replicate,
                        "interaction": interaction,
                        "side": side,
                        "residue_index": residue + 1,
                        "contact_probability": probability[residue],
                        "sampled_frames": frames,
                    }
                )
        map_specs = [
            ("om_map", "MED1-OCT4", frames * n_med1_physical * int(row["n_oct4"])),
            ("od_map", "OCT4-DNA", frames * int(row["n_oct4"])),
            ("md_map", "MED1-DNA", frames * n_med1_physical),
        ]
        for key, interaction, denominator in map_specs:
            if denominator == 0:
                continue
            matrix = arrays[key] / denominator
            for x_bin, y_bin in np.ndindex(matrix.shape):
                map_rows.append(
                    {
                        "system": system,
                        "replicate": replicate,
                        "interaction": interaction,
                        "x_bin": x_bin,
                        "y_bin": y_bin,
                        "contact_probability": matrix[x_bin, y_bin],
                        "sampled_frames": frames,
                    }
                )
    profiles = pd.DataFrame(profile_rows)
    maps = pd.DataFrame(map_rows)
    if profiles.empty or maps.empty:
        return profiles, maps
    profile_group = mean_sd(
        profiles,
        ["system", "interaction", "side", "residue_index"],
        ["contact_probability"],
    )
    map_group = mean_sd(
        maps,
        ["system", "interaction", "x_bin", "y_bin"],
        ["contact_probability"],
    )
    return profile_group, map_group


def publication_panel_tables(
    task_arrays: dict[tuple[str, str], dict[str, np.ndarray]],
) -> dict[str, pd.DataFrame]:
    """Summarize corrected-data metrics required by the final manuscript panels."""
    category_labels = [
        "neither_bound",
        "only_one_bound",
        "both_bound_different_windows",
        "shared_window",
    ]
    local_rows: list[dict[str, object]] = []
    odds_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    histogram_rows: list[dict[str, object]] = []
    for system in PUBLICATION_DNA_SYSTEMS:
        for replicate in REPEATS:
            key = (system, replicate)
            if key not in task_arrays:
                continue
            arrays = task_arrays[key]
            edges = arrays["publication_local_edges"].astype(float)
            opportunities = arrays[
                "publication_local_opportunities"
            ].astype(float)
            for index, category in enumerate(category_labels):
                local_rows.append(
                    {
                        "system": system,
                        "replicate": replicate,
                        "binding_category": category,
                        "PP_edges": edges[index],
                        "pair_frame_opportunities": opportunities[index],
                        "conditional_PP_probability": safe_probability(
                            edges[index], opportunities[index]
                        ),
                    }
                )
            cells = np.asarray(
                [
                    edges[3],
                    opportunities[3] - edges[3],
                    edges[2],
                    opportunities[2] - edges[2],
                ],
                dtype=float,
            )
            if np.any(cells == 0):
                cells += 0.5
            odds_ratio = (cells[0] / cells[1]) / (cells[2] / cells[3])
            odds_rows.append(
                {
                    "system": system,
                    "replicate": replicate,
                    "odds_ratio": odds_ratio,
                    "log2_odds_ratio": math.log2(odds_ratio),
                }
            )

            q90 = arrays["publication_distance_q90_nm"].astype(float)
            within_10 = arrays[
                "publication_distance_within_10nm"
            ].astype(float)
            valid = np.isfinite(q90)
            if np.any(valid):
                distance_rows.append(
                    {
                        "system": system,
                        "replicate": replicate,
                        "q90_distance_nm": float(np.mean(q90[valid])),
                        "fraction_within_10nm": float(
                            np.mean(within_10[valid])
                        ),
                        "sampled_frames": int(np.sum(valid)),
                    }
                )
            counts = arrays["publication_distance_histogram"].astype(float)
            total = float(counts.sum())
            cumulative = np.cumsum(counts)
            for bin_index, count in enumerate(counts):
                histogram_rows.append(
                    {
                        "system": system,
                        "replicate": replicate,
                        "bin_start_nm": PUBLICATION_DISTANCE_EDGES_NM[
                            bin_index
                        ],
                        "bin_end_nm": PUBLICATION_DISTANCE_EDGES_NM[
                            bin_index + 1
                        ],
                        "probability": safe_probability(count, total),
                        "cumulative_probability": safe_probability(
                            cumulative[bin_index], total
                        ),
                    }
                )

    local_repeat = pd.DataFrame(local_rows)
    local_group = mean_sd(
        local_repeat,
        ["system", "binding_category"],
        ["conditional_PP_probability"],
    )
    odds_repeat = pd.DataFrame(odds_rows)
    odds_group = mean_sd(
        odds_repeat,
        ["system"],
        ["log2_odds_ratio"],
    )
    odds_group["geometric_mean_odds_ratio"] = np.power(
        2.0, odds_group["log2_odds_ratio_mean"]
    )
    distance_repeat = pd.DataFrame(distance_rows)
    distance_group = mean_sd(
        distance_repeat,
        ["system"],
        ["q90_distance_nm", "fraction_within_10nm"],
    )
    histogram_repeat = pd.DataFrame(histogram_rows)
    histogram_group = mean_sd(
        histogram_repeat,
        ["system", "bin_start_nm", "bin_end_nm"],
        ["probability", "cumulative_probability"],
    )
    return {
        "local_repeat": local_repeat,
        "local_group": local_group,
        "odds_repeat": odds_repeat,
        "odds_group": odds_group,
        "distance_repeat": distance_repeat,
        "distance_group": distance_group,
        "histogram_repeat": histogram_repeat,
        "histogram_group": histogram_group,
    }


def analyze_dna_only_control(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for repeat in REPEATS:
        path = project_root / "data/dna_only" / repeat / "D1"
        if not (path / "output.dcd").exists():
            continue
        universe = mda.Universe(str(path / "start.pdb"), str(path / "output.dcd"))
        for frame in range(PRODUCTION_START, min(PRODUCTION_END, len(universe.trajectory)) + 1, STRIDE):
            universe.trajectory[frame - 1]
            rg, ree, asphericity = dna_shape(universe.atoms.positions.astype(float) / 10, 75.0)
            rows.append(
                {
                    "system": "D1",
                    "replicate": repeat,
                    "frame": frame,
                    "time_us": frame * 0.001,
                    "DNA_Rg_nm": rg,
                    "DNA_Ree_nm": ree,
                    "DNA_asphericity": asphericity,
                }
            )
    return pd.DataFrame(rows)


def design_table(manifest: pd.DataFrame) -> pd.DataFrame:
    table = (
        manifest.sort_values(["system", "replicate"])
        .drop_duplicates("system")
        [["system", "n_med1", "n_oct4", "n_dna", "med1_fragments", "box_nm", "tier", "series_tags"]]
        .copy()
    )
    avogadro = 6.02214076e23
    volume_l = table["box_nm"] ** 3 * 1e-24
    table["n_protein_sequence_equivalents"] = table["n_med1"] + table["n_oct4"]
    table["n_physical_protein_chains"] = table["n_med1"] * table["med1_fragments"] + table["n_oct4"]
    table["protein_residues"] = table["n_med1"] * MED1_LENGTH + table["n_oct4"] * OCT4_LENGTH
    table["MED1_fraction_sequence_equivalents"] = table["n_med1"] / table["n_protein_sequence_equivalents"]
    table["protein_residue_concentration_mM"] = table["protein_residues"] / avogadro / volume_l * 1e3
    table["protein_chain_concentration_mM"] = table["n_physical_protein_chains"] / avogadro / volume_l * 1e3
    table["available_repeats"] = table["system"].map(manifest.groupby("system")["available"].sum())
    table["analysis_repeats"] = table["system"].map(manifest.groupby("system")["analysis_included"].sum())
    table["planned_repeats"] = table["system"].map(manifest.groupby("system").size())
    return table


def validation_table(
    manifest: pd.DataFrame,
    framewise: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    available = manifest[manifest["analysis_included"]]
    counts = framewise.groupby(["system", "replicate"]).size()
    rows.append(
        {
            "check": "40 sampled production frames per available trajectory",
            "value": f"min={counts.min()}, max={counts.max()}",
            "required": "40",
            "passed": bool(counts.eq(40).all()),
        }
    )
    cluster_counts = clusters.groupby(["system", "replicate", "cutoff_nm"]).size()
    rows.append(
        {
            "check": "40 cluster frames at every cutoff",
            "value": f"min={cluster_counts.min()}, max={cluster_counts.max()}",
            "required": "40",
            "passed": bool(cluster_counts.eq(40).all()),
        }
    )
    observed = set(zip(framewise["system"], framewise["replicate"]))
    expected = set(zip(available["system"], available["replicate"]))
    rows.append(
        {
            "check": "all and only designated balanced-design tasks analyzed",
            "value": f"observed={len(observed)}, expected={len(expected)}",
            "required": "equal task sets",
            "passed": observed == expected,
        }
    )
    for system in ["M15O85", "M50O50"]:
        repeats = sorted(framewise.loc[framewise["system"] == system, "replicate"].unique())
        rows.append(
            {
                "check": f"{system} includes all three completed repeats",
                "value": f"analyzed={','.join(repeats)}",
                "required": "analyzed=work,work-1,work-2",
                "passed": repeats == ["work", "work-1", "work-2"],
            }
        )
    explicit_observed = set(
        zip(
            manifest.loc[manifest["inclusion_basis"] == "explicit_third_repeat", "system"],
            manifest.loc[manifest["inclusion_basis"] == "explicit_third_repeat", "replicate"],
        )
    )
    rows.append(
        {
            "check": "only the designated third repeats use explicit inclusion",
            "value": ",".join(f"{system}/{replicate}" for system, replicate in sorted(explicit_observed)),
            "required": "M15O85/work-1,M50O50/work-2",
            "passed": explicit_observed == EXPLICITLY_INCLUDED_TASKS,
        }
    )
    rows.append(
        {
            "check": "trajectory box length read from balanced-design manifest",
            "value": f"{available['box_nm'].nunique()} distinct values",
            "required": ">1 (not a forced 75-nm box)",
            "passed": bool(available["box_nm"].nunique() > 1),
        }
    )
    validation = pd.DataFrame(rows)
    if not validation["passed"].all():
        raise RuntimeError("Balanced-analysis validation failed:\n" + validation.to_string(index=False))
    return validation


def system_color(system: str) -> str:
    if system.startswith("M0"):
        return COLORS["OCT4"]
    if system.startswith("M50O0"):
        return COLORS["MED1"]
    return COLORS["mixed"]


def composition_x(design: pd.DataFrame) -> dict[str, float]:
    return design.set_index("system")["MED1_fraction_sequence_equivalents"].mul(100).to_dict()


def error_series(
    ax: plt.Axes,
    group: pd.DataFrame,
    systems: list[str],
    x_map: dict[str, float],
    metric: str,
    *,
    color: str = COLORS["mixed"],
    label: str | None = None,
    marker: str = "o",
) -> None:
    indexed = group.set_index("system")
    available = [system for system in systems if system in indexed.index]
    ax.errorbar(
        [x_map[system] for system in available],
        [indexed.loc[system, f"{metric}_mean"] for system in available],
        yerr=[indexed.loc[system, f"{metric}_sd"] for system in available],
        color=color,
        marker=marker,
        label=label,
    )


def select_snapshot(framewise: pd.DataFrame, system: str) -> pd.Series:
    candidates = framewise[framewise["system"] == system].copy()
    metrics = ["direct_largest_protein_fraction", "PP_edge_density"]
    score = np.zeros(len(candidates), dtype=float)
    for metric in metrics:
        center = candidates[metric].median()
        scale = max(float(candidates[metric].std(ddof=1)), 1e-8)
        score += np.abs(candidates[metric] - center) / scale
    score += 0.05 * np.abs(candidates["time_us"] - 3.0)
    return candidates.iloc[int(np.argmin(score))]


def snapshot_data(task: Task, frame: int) -> dict[str, object]:
    """Reconstruct actual chain coordinates across PBC for one snapshot.

    Every molecule is first made whole from its bonded bead order. Molecules
    within a contact component are then placed in mutually consistent periodic
    images using the closest contacting bead pair on each graph edge. This is
    intentionally different from a molecular-centroid network diagram: the
    returned objects contain every simulated bead and preserve chain shape.
    """
    molecules = read_molecule_map(task)
    universe = mda.Universe(str(task.path / "start.pdb"), str(task.path / "output.dcd"))
    universe.trajectory[frame - 1]
    positions_nm = universe.atoms.positions.astype(float) / 10
    wrapped_nm = np.mod(positions_nm, task.box_nm)
    atom_pairs14, distances = atom_pairs_with_distances(universe.atoms.positions, task.box_nm)
    keep = distances < PRIMARY_CUTOFF_NM
    atom_pairs = atom_pairs14[keep]
    distances = distances[keep]
    molecule_pairs = unique_molecule_pairs(atom_pairs, molecules.atom_molecule)

    traces: list[np.ndarray] = []
    raw_centers = np.zeros((len(molecules.molecule_type), 3), dtype=float)
    for molecule, (start, stop) in enumerate(zip(molecules.starts, molecules.stops)):
        coords = positions_nm[start:stop]
        if molecules.molecule_type[molecule] == TYPE_DNA:
            whole = unwrap_dna(coords, task.box_nm)
        else:
            whole = unwrap_segment(coords, task.box_nm)
        traces.append(whole)
        raw_centers[molecule] = np.mod(whole.mean(axis=0), task.box_nm)

    # Keep the closest bead pair for each molecular graph edge. These exact
    # bead pairs define how a neighbor's periodic image is joined in the BFS.
    closest_contact: dict[tuple[int, int], tuple[int, int]] = {}
    for index in np.argsort(distances):
        atom_a, atom_b = map(int, atom_pairs[index])
        molecule_a = int(molecules.atom_molecule[atom_a])
        molecule_b = int(molecules.atom_molecule[atom_b])
        if molecule_a == molecule_b:
            continue
        if molecule_a < molecule_b:
            edge = (molecule_a, molecule_b)
            contact = (atom_a, atom_b)
        else:
            edge = (molecule_b, molecule_a)
            contact = (atom_b, atom_a)
        closest_contact.setdefault(edge, contact)

    n_molecules = len(molecules.molecule_type)
    labels, component_sizes = components(n_molecules, molecule_pairs)
    if len(molecules.dna_nodes):
        anchor = int(molecules.dna_nodes[0])
        highlight_label = int(labels[anchor])
        selection_mode = "DNA component"
    else:
        protein_nodes = molecules.protein_nodes
        protein_component_counts = np.bincount(labels[protein_nodes], minlength=len(component_sizes))
        highlight_label = int(np.argmax(protein_component_counts))
        candidates = protein_nodes[labels[protein_nodes] == highlight_label]
        anchor = int(candidates[0])
        selection_mode = "direct LCC"
    highlight = labels == highlight_label

    adjacency: list[list[int]] = [[] for _ in range(n_molecules)]
    for first, second in molecule_pairs:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))

    aligned: list[np.ndarray | None] = [None] * n_molecules
    component_order = [highlight_label] + [
        label for label in range(len(component_sizes)) if label != highlight_label
    ]
    anchor_raw_center = raw_centers[anchor]
    for component_label in component_order:
        nodes = np.flatnonzero(labels == component_label)
        root = anchor if component_label == highlight_label else int(nodes[0])
        root_trace = traces[root].copy()
        if component_label == highlight_label:
            root_trace -= root_trace.mean(axis=0)
        else:
            relative_center = raw_centers[root] - anchor_raw_center
            relative_center -= task.box_nm * np.round(relative_center / task.box_nm)
            root_trace += relative_center - root_trace.mean(axis=0)
        aligned[root] = root_trace
        queue = [root]
        seen = {root}
        while queue:
            parent = queue.pop(0)
            parent_trace = aligned[parent]
            assert parent_trace is not None
            for child in adjacency[parent]:
                if child in seen or labels[child] != component_label:
                    continue
                edge = (min(parent, child), max(parent, child))
                contact_low, contact_high = closest_contact[edge]
                if parent < child:
                    parent_atom, child_atom = contact_low, contact_high
                else:
                    parent_atom, child_atom = contact_high, contact_low
                parent_local = parent_atom - int(molecules.starts[parent])
                child_local = child_atom - int(molecules.starts[child])
                raw_delta = wrapped_nm[child_atom] - wrapped_nm[parent_atom]
                raw_delta -= task.box_nm * np.round(raw_delta / task.box_nm)
                desired_child_contact = parent_trace[parent_local] + raw_delta
                child_trace = traces[child].copy()
                child_trace += desired_child_contact - child_trace[child_local]
                aligned[child] = child_trace
                seen.add(child)
                queue.append(child)

    if any(trace is None for trace in aligned):
        raise RuntimeError(f"Failed to PBC-reconstruct all molecules in {task.replicate}/{task.system}")
    aligned_traces = [np.asarray(trace) for trace in aligned]

    # Center on the highlighted assembly and use its principal axes for a
    # reproducible projection that exposes chain shapes instead of box axes.
    selected_coordinates = np.vstack(
        [trace for molecule, trace in enumerate(aligned_traces) if highlight[molecule]]
    )
    selected_center = selected_coordinates.mean(axis=0)
    aligned_traces = [trace - selected_center for trace in aligned_traces]
    selected_coordinates -= selected_center
    covariance = selected_coordinates.T @ selected_coordinates / len(selected_coordinates)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:2]]
    projected_traces = [trace @ basis for trace in aligned_traces]

    contact_segments = []
    for (first, second), (atom_first, atom_second) in closest_contact.items():
        if not (highlight[first] and highlight[second]):
            continue
        first_local = atom_first - int(molecules.starts[first])
        second_local = atom_second - int(molecules.starts[second])
        contact_segments.append(
            np.vstack(
                [projected_traces[first][first_local], projected_traces[second][second_local]]
            )
        )
    return {
        "traces": projected_traces,
        "contact_segments": contact_segments,
        "types": molecules.molecule_type,
        "highlight": highlight,
        "highlight_count": int(np.sum(highlight & (molecules.molecule_type != TYPE_DNA))),
        "selection_mode": selection_mode,
        "task": task,
        "frame": frame,
    }


def draw_snapshot(ax: plt.Axes, snapshot: dict[str, object], title: str) -> None:
    traces = snapshot["traces"]
    types = np.asarray(snapshot["types"])
    highlight = np.asarray(snapshot["highlight"], dtype=bool)

    background_lines: list[np.ndarray] = []
    colored_lines: dict[int, list[np.ndarray]] = {
        TYPE_OCT4: [],
        TYPE_MED1: [],
        TYPE_DNA: [],
    }
    for molecule, trace in enumerate(traces):
        trace = np.asarray(trace)
        segments = (
            [trace[: DNA_LENGTH // 2], trace[DNA_LENGTH // 2 :]]
            if types[molecule] == TYPE_DNA
            else [trace]
        )
        if highlight[molecule]:
            colored_lines[int(types[molecule])].extend(segments)
        else:
            background_lines.extend(segments)

    if background_lines:
        ax.add_collection(
            LineCollection(
                background_lines,
                colors="#BDBDBD",
                linewidths=0.28,
                alpha=0.20,
                zorder=1,
                rasterized=True,
            )
        )
    if snapshot["contact_segments"]:
        ax.add_collection(
            LineCollection(
                snapshot["contact_segments"],
                colors="#707070",
                linewidths=0.32,
                alpha=0.24,
                zorder=2,
                rasterized=True,
            )
        )
    for type_id, color, width, alpha in [
        (TYPE_OCT4, COLORS["OCT4"], 0.62, 0.82),
        (TYPE_MED1, COLORS["MED1"], 0.72, 0.82),
        (TYPE_DNA, COLORS["DNA"], 1.35, 0.95),
    ]:
        if colored_lines[type_id]:
            ax.add_collection(
                LineCollection(
                    colored_lines[type_id],
                    colors=color,
                    linewidths=width,
                    alpha=alpha,
                    zorder=3 if type_id != TYPE_DNA else 4,
                    rasterized=True,
                )
            )

    selected = np.vstack(
        [np.asarray(trace) for molecule, trace in enumerate(traces) if highlight[molecule]]
    )
    x_min, y_min = selected.min(axis=0)
    x_max, y_max = selected.max(axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1.0)
    pad = 0.06 * span
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    ax.set_xlim(x_center - span / 2 - pad, x_center + span / 2 + pad)
    ax.set_ylim(y_center - span / 2 - pad, y_center + span / 2 + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{title}\n{snapshot['selection_mode']}: {snapshot['highlight_count']} proteins",
        fontsize=7.2,
    )


def minimum_image_nm(delta: np.ndarray, box_nm: float) -> np.ndarray:
    """Return the minimum-image displacement for a task-specific cubic box."""
    return delta - box_nm * np.round(delta / box_nm)


def build_spatial_snapshot(
    task: Task,
    frame: int,
    mode: str = "largest_component",
) -> dict[str, object]:
    """Original conformation + molecular-COM contact renderer data path.

    This is the old paper renderer adapted only at its input boundary: task
    paths, molecule ranges, and box lengths now come from the corrected task
    manifest.  Selection, PBC graph placement, projection, and drawable
    contact-edge construction follow the original implementation.
    """
    valid_modes = {
        "largest_component",
        "full_system",
        "DNA_neighborhood",
        "DNA_associated_assembly",
        "DNA_linked_component",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unknown spatial-snapshot mode: {mode}")

    molecules = read_molecule_map(task)
    universe = mda.Universe(
        str(task.path / "start.pdb"),
        str(task.path / "output.dcd"),
    )
    universe.trajectory[frame - 1]
    coordinates_nm = universe.atoms.positions.astype(np.float64) / 10.0
    wrapped_atoms = np.mod(coordinates_nm, task.box_nm)
    all_atom_pairs, all_distances = atom_pairs_with_distances(
        universe.atoms.positions,
        task.box_nm,
    )
    keep = all_distances < PRIMARY_CUTOFF_NM
    atom_pairs = all_atom_pairs[keep]
    atom_pair_distances = all_distances[keep]
    pairs = unique_molecule_pairs(atom_pairs, molecules.atom_molecule)

    protein_nodes_all = molecules.protein_nodes
    protein_node_to_local = np.full(
        len(molecules.molecule_type), -1, dtype=np.int32
    )
    protein_node_to_local[protein_nodes_all] = np.arange(
        len(protein_nodes_all), dtype=np.int32
    )
    protein_mask = (
        (molecules.molecule_type[pairs[:, 0]] != TYPE_DNA)
        & (molecules.molecule_type[pairs[:, 1]] != TYPE_DNA)
        if len(pairs)
        else np.zeros(0, dtype=bool)
    )
    protein_pairs = pairs[protein_mask]
    protein_edges_local = (
        protein_node_to_local[protein_pairs]
        if len(protein_pairs)
        else np.empty((0, 2), dtype=np.int32)
    )
    labels, sizes = components(len(protein_nodes_all), protein_edges_local)
    largest_size = int(sizes.max())
    candidate_labels = np.flatnonzero(sizes == largest_size)
    if len(candidate_labels) > 1:
        edge_counts: list[int] = []
        for label in candidate_labels:
            local_nodes = set(np.flatnonzero(labels == label))
            edge_counts.append(
                sum(
                    int(int(first) in local_nodes and int(second) in local_nodes)
                    for first, second in protein_edges_local
                )
            )
        largest_label = int(candidate_labels[int(np.argmax(edge_counts))])
    else:
        largest_label = int(candidate_labels[0])
    protein_nodes = protein_nodes_all[labels == largest_label]
    largest_protein_nodes = set(map(int, protein_nodes))

    direct_dna_nodes: set[int] = set()
    direct_oct4_nodes: set[int] = set()
    direct_med1_nodes: set[int] = set()
    indirect_med1_nodes: set[int] = set()
    dna_associated_nodes: set[int] = set()
    dna_linked_nodes: set[int] = set()
    dna_node = int(molecules.dna_nodes[0]) if len(molecules.dna_nodes) else -1
    if mode in {"DNA_associated_assembly", "DNA_linked_component"}:
        if dna_node < 0:
            raise ValueError(f"{task.system} has no DNA-linked assembly")
        for first, second in pairs:
            first_int, second_int = int(first), int(second)
            if first_int == dna_node and second_int != dna_node:
                direct_dna_nodes.add(second_int)
            elif second_int == dna_node and first_int != dna_node:
                direct_dna_nodes.add(first_int)
        direct_oct4_nodes = {
            node
            for node in direct_dna_nodes
            if molecules.molecule_type[node] == TYPE_OCT4
        }
        direct_med1_nodes = {
            node
            for node in direct_dna_nodes
            if molecules.molecule_type[node] == TYPE_MED1
        }
        for first, second in protein_pairs:
            first_int, second_int = int(first), int(second)
            if (
                first_int in direct_oct4_nodes
                and molecules.molecule_type[second_int] == TYPE_MED1
                and second_int not in direct_med1_nodes
            ):
                indirect_med1_nodes.add(second_int)
            if (
                second_int in direct_oct4_nodes
                and molecules.molecule_type[first_int] == TYPE_MED1
                and first_int not in direct_med1_nodes
            ):
                indirect_med1_nodes.add(first_int)
        dna_associated_nodes = (
            direct_oct4_nodes | direct_med1_nodes | indirect_med1_nodes
        )
        augmented_labels, _ = components(len(molecules.molecule_type), pairs)
        dna_linked_nodes = {
            int(node)
            for node in np.flatnonzero(
                augmented_labels == augmented_labels[dna_node]
            )
            if int(node) != dna_node
        }

    if mode in {
        "full_system",
        "DNA_associated_assembly",
        "DNA_linked_component",
    }:
        selected_nodes = set(range(len(molecules.molecule_type)))
    elif mode == "DNA_neighborhood":
        if dna_node < 0:
            raise ValueError(f"{task.system} has no DNA neighborhood")
        selected_nodes = {dna_node}
        for first, second in pairs:
            if int(first) == dna_node:
                selected_nodes.add(int(second))
            elif int(second) == dna_node:
                selected_nodes.add(int(first))
    else:
        selected_nodes = largest_protein_nodes.copy()
        if dna_node >= 0 and any(
            (int(first) == dna_node and int(second) in selected_nodes)
            or (int(second) == dna_node and int(first) in selected_nodes)
            for first, second in pairs
        ):
            selected_nodes.add(dna_node)

    selected_nodes_array = np.asarray(sorted(selected_nodes), dtype=np.int32)
    selected_pairs = np.asarray(
        [
            (int(first), int(second))
            for first, second in pairs
            if int(first) in selected_nodes and int(second) in selected_nodes
        ],
        dtype=np.int32,
    )
    if not len(selected_pairs):
        selected_pairs = np.empty((0, 2), dtype=np.int32)

    whole_coordinates: dict[int, np.ndarray] = {}
    wrapped_com: dict[int, np.ndarray] = {}
    for node in selected_nodes_array:
        node_int = int(node)
        start = int(molecules.starts[node_int])
        stop = int(molecules.stops[node_int])
        coords = coordinates_nm[start:stop]
        if molecules.molecule_type[node_int] == TYPE_DNA:
            whole = unwrap_dna(coords, task.box_nm)
        else:
            whole = unwrap_segment(coords, task.box_nm)
        whole_coordinates[node_int] = whole
        wrapped_com[node_int] = np.mod(whole.mean(axis=0), task.box_nm)

    selected_edge_set = {
        (int(first), int(second)) for first, second in selected_pairs
    }
    edge_contacts: dict[tuple[int, int], tuple[int, int, float]] = {}
    for (first_atom_raw, second_atom_raw), distance_raw in zip(
        atom_pairs, atom_pair_distances
    ):
        first_atom, second_atom = int(first_atom_raw), int(second_atom_raw)
        first_molecule = int(molecules.atom_molecule[first_atom])
        second_molecule = int(molecules.atom_molecule[second_atom])
        if first_molecule == second_molecule:
            continue
        if first_molecule > second_molecule:
            first_molecule, second_molecule = second_molecule, first_molecule
            first_atom, second_atom = second_atom, first_atom
        key = (first_molecule, second_molecule)
        if key not in selected_edge_set:
            continue
        distance = float(distance_raw)
        if key not in edge_contacts or distance < edge_contacts[key][2]:
            edge_contacts[key] = (first_atom, second_atom, distance)
    if set(edge_contacts) != selected_edge_set:
        raise RuntimeError(
            f"Missing bead anchors for {task.replicate}/{task.system} snapshot edges"
        )

    adjacency: dict[int, list[int]] = {
        int(node): [] for node in selected_nodes_array
    }
    for first, second in selected_pairs:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    placed_com: dict[int, np.ndarray] = {}
    components_placed: list[list[int]] = []
    root_order = list(map(int, selected_nodes_array))
    if mode == "DNA_neighborhood":
        root_order = [dna_node]
    elif mode in {"DNA_associated_assembly", "DNA_linked_component"}:
        root_order = [dna_node] + [
            int(node) for node in selected_nodes_array if int(node) != dna_node
        ]
    elif mode != "full_system":
        root_order = [int(selected_nodes_array[0])]
    for root_node in root_order:
        if root_node in placed_com:
            continue
        placed_com[root_node] = wrapped_com[root_node].copy()
        component = [root_node]
        queue = [root_node]
        while queue:
            first = queue.pop(0)
            for second in adjacency[first]:
                if second in placed_com:
                    continue
                low, high = sorted((first, second))
                low_atom, high_atom, _ = edge_contacts[(low, high)]
                if first == low:
                    first_atom, second_atom = low_atom, high_atom
                else:
                    first_atom, second_atom = high_atom, low_atom
                first_local = first_atom - int(molecules.starts[first])
                second_local = second_atom - int(molecules.starts[second])
                placed_first_atom = whole_coordinates[first][first_local] + (
                    placed_com[first] - whole_coordinates[first].mean(axis=0)
                )
                target_second_atom = placed_first_atom + minimum_image_nm(
                    wrapped_atoms[second_atom] - wrapped_atoms[first_atom],
                    task.box_nm,
                )
                second_shift = (
                    target_second_atom - whole_coordinates[second][second_local]
                )
                placed_com[second] = (
                    whole_coordinates[second].mean(axis=0) + second_shift
                )
                component.append(second)
                queue.append(second)
        components_placed.append(component)

    lcc_center: np.ndarray | None = None
    if mode == "full_system":
        lcc_center = np.vstack(
            [placed_com[node] for node in largest_protein_nodes]
        ).mean(axis=0)
        for component in components_placed:
            component_center = np.vstack(
                [placed_com[node] for node in component]
            ).mean(axis=0)
            desired_center = lcc_center + minimum_image_nm(
                component_center - lcc_center, task.box_nm
            )
            shift = desired_center - component_center
            for node in component:
                placed_com[node] += shift
    elif mode in {"DNA_associated_assembly", "DNA_linked_component"}:
        dna_center = placed_com[dna_node]
        for component in components_placed:
            if dna_node in component:
                continue
            component_center = np.vstack(
                [placed_com[node] for node in component]
            ).mean(axis=0)
            desired_center = dna_center + minimum_image_nm(
                component_center - dna_center, task.box_nm
            )
            shift = desired_center - component_center
            for node in component:
                placed_com[node] += shift

    for node in selected_nodes_array:
        node_int = int(node)
        placed_com.setdefault(node_int, wrapped_com[node_int])
        whole_coordinates[node_int] += (
            placed_com[node_int] - whole_coordinates[node_int].mean(axis=0)
        )
    broken_edges: set[tuple[int, int]] = set()
    for (first, second), (first_atom, second_atom, _) in edge_contacts.items():
        first_local = first_atom - int(molecules.starts[first])
        second_local = second_atom - int(molecules.starts[second])
        placed_distance = float(
            np.linalg.norm(
                whole_coordinates[first][first_local]
                - whole_coordinates[second][second_local]
            )
        )
        if placed_distance > PRIMARY_CUTOFF_NM + 1e-4:
            broken_edges.add((first, second))
    drawable_pairs = np.asarray(
        [
            (int(first), int(second))
            for first, second in selected_pairs
            if (int(first), int(second)) not in broken_edges
        ],
        dtype=np.int32,
    )
    if not len(drawable_pairs):
        drawable_pairs = np.empty((0, 2), dtype=np.int32)

    com_array = np.vstack(
        [placed_com[int(node)] for node in selected_nodes_array]
    )
    if mode == "full_system":
        assert lcc_center is not None
        center = lcc_center
        centered_com = com_array - center
        basis = np.eye(3, 2)
    elif mode in {
        "DNA_neighborhood",
        "DNA_associated_assembly",
        "DNA_linked_component",
    }:
        center = whole_coordinates[dna_node].mean(axis=0)
        centered_com = com_array - center
        dna_centered = whole_coordinates[dna_node] - center
        _, _, vh = np.linalg.svd(dna_centered, full_matrices=False)
        basis = vh[:2].T
        for column in range(2):
            dominant = int(np.argmax(np.abs(basis[:, column])))
            if basis[dominant, column] < 0:
                basis[:, column] *= -1
    else:
        center = com_array.mean(axis=0)
        centered_com = com_array - center
    if mode not in {"full_system", "DNA_neighborhood"} and len(com_array) >= 3:
        _, _, vh = np.linalg.svd(centered_com, full_matrices=False)
        basis = vh[:2].T
    elif mode not in {"full_system", "DNA_neighborhood"}:
        basis = np.eye(3, 2)

    projected_com = centered_com @ basis
    projected_traces = {
        int(node): (whole_coordinates[int(node)] - center) @ basis
        for node in selected_nodes_array
    }
    node_to_index = {
        int(node): index for index, node in enumerate(selected_nodes_array)
    }
    return {
        "system": task.system,
        "replicate": task.replicate,
        "frame": frame,
        "time_us": frame * 0.001,
        "box_nm": task.box_nm,
        "lcf": len(protein_nodes) / len(protein_nodes_all),
        "mode": mode,
        "nodes": selected_nodes_array,
        "types": molecules.molecule_type[selected_nodes_array],
        "pairs": drawable_pairs,
        "periodic_cycle_edges_omitted": len(broken_edges),
        "projected_com": projected_com,
        "projected_traces": projected_traces,
        "node_to_index": node_to_index,
        "n_proteins": len(protein_nodes),
        "largest_protein_nodes": largest_protein_nodes,
        "dna_associated_nodes": dna_associated_nodes,
        "dna_linked_nodes": dna_linked_nodes,
        "direct_oct4_nodes": direct_oct4_nodes,
        "direct_med1_nodes": direct_med1_nodes,
        "indirect_med1_nodes": indirect_med1_nodes,
        "selected_protein_count": int(
            np.sum(molecules.molecule_type[selected_nodes_array] != TYPE_DNA)
        ),
    }


def draw_spatial_snapshot(
    ax: plt.Axes,
    snapshot: dict[str, object],
    half_span: float | tuple[float, float],
    panel_label: str | None = None,
    scale_bar_inset_nm: float = 2.0,
) -> None:
    """Draw the original conformation + COM-line spatial representation."""
    if isinstance(half_span, tuple):
        x_half_span, y_half_span = half_span
    else:
        x_half_span = y_half_span = half_span
    color_by_type = {
        TYPE_MED1: COLORS["MED1"],
        TYPE_OCT4: COLORS["OCT4"],
        TYPE_DNA: COLORS["DNA"],
    }
    marker_by_type = {
        TYPE_MED1: "s",
        TYPE_OCT4: "o",
        TYPE_DNA: "D",
    }
    nodes = np.asarray(snapshot["nodes"])
    types = np.asarray(snapshot["types"])
    projected_com = np.asarray(snapshot["projected_com"])
    traces = snapshot["projected_traces"]
    node_to_index = snapshot["node_to_index"]
    mode = str(snapshot.get("mode", "largest_component"))
    largest_protein_nodes = set(snapshot["largest_protein_nodes"])
    dna_associated_nodes = set(snapshot.get("dna_associated_nodes", set()))
    dna_linked_nodes = set(snapshot.get("dna_linked_nodes", set()))

    for node, type_id_raw in zip(nodes, types):
        node_int, type_id = int(node), int(type_id_raw)
        trace = np.asarray(traces[node_int])
        if mode == "full_system":
            foreground = node_int in largest_protein_nodes or type_id == TYPE_DNA
        elif mode == "DNA_associated_assembly":
            foreground = node_int in dna_associated_nodes or type_id == TYPE_DNA
        elif mode == "DNA_linked_component":
            foreground = node_int in dna_linked_nodes or type_id == TYPE_DNA
        else:
            foreground = True
        if type_id == TYPE_DNA:
            half = len(trace) // 2
            ax.plot(
                trace[:half, 0], trace[:half, 1],
                color=COLORS["DNA"], lw=1.35, alpha=0.9, zorder=4,
            )
            ax.plot(
                trace[half:, 0], trace[half:, 1],
                color=COLORS["DNA"], lw=1.35, alpha=0.9, zorder=4,
            )
        else:
            stride = 4 if type_id == TYPE_MED1 else 2
            ax.plot(
                trace[::stride, 0],
                trace[::stride, 1],
                color=color_by_type[type_id] if foreground else "#B9BEC5",
                lw=0.45 if foreground else 0.32,
                alpha=0.24 if foreground else 0.12,
            )

    for first_raw, second_raw in snapshot["pairs"]:
        first, second = int(first_raw), int(second_raw)
        first_index = int(node_to_index[first])
        second_index = int(node_to_index[second])
        first_type, second_type = int(types[first_index]), int(types[second_index])
        if TYPE_DNA in (first_type, second_type):
            continue
        is_largest_edge = first in largest_protein_nodes and second in largest_protein_nodes
        is_associated_edge = first in dna_associated_nodes and second in dna_associated_nodes
        is_linked_edge = first in dna_linked_nodes and second in dna_linked_nodes
        foreground = (
            is_largest_edge
            if mode == "full_system"
            else is_associated_edge
            if mode == "DNA_associated_assembly"
            else is_linked_edge
            if mode == "DNA_linked_component"
            else True
        )
        if not foreground:
            continue
        ax.plot(
            [projected_com[first_index, 0], projected_com[second_index, 0]],
            [projected_com[first_index, 1], projected_com[second_index, 1]],
            color="#777777",
            lw=0.45,
            ls="-",
            alpha=0.35,
            zorder=2,
        )

    for type_id in [TYPE_OCT4, TYPE_MED1, TYPE_DNA]:
        mask = types == type_id
        if not np.any(mask):
            continue
        if mode in {
            "full_system",
            "DNA_associated_assembly",
            "DNA_linked_component",
        } and type_id != TYPE_DNA:
            foreground_nodes = (
                largest_protein_nodes
                if mode == "full_system"
                else dna_linked_nodes
                if mode == "DNA_linked_component"
                else dna_associated_nodes
            )
            foreground_mask = np.asarray(
                [int(node) in foreground_nodes for node in nodes[mask]]
            )
            background_positions = projected_com[mask][~foreground_mask]
            if len(background_positions):
                ax.scatter(
                    background_positions[:, 0],
                    background_positions[:, 1],
                    s=9,
                    marker=marker_by_type[type_id],
                    color="#B9BEC5",
                    edgecolor="white",
                    linewidth=0.25,
                    alpha=0.55,
                    zorder=2,
                )
            positions = projected_com[mask][foreground_mask]
        else:
            positions = projected_com[mask]
        if not len(positions):
            continue
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=18 if type_id != TYPE_DNA else 32,
            marker=marker_by_type[type_id],
            color=color_by_type[type_id],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    ax.plot(
        [
            -x_half_span + scale_bar_inset_nm,
            -x_half_span + scale_bar_inset_nm + 10,
        ],
        [
            -y_half_span + scale_bar_inset_nm,
            -y_half_span + scale_bar_inset_nm,
        ],
        color="black",
        lw=1.5,
    )
    ax.text(
        -x_half_span + scale_bar_inset_nm + 5,
        -y_half_span + scale_bar_inset_nm + 1,
        "10 nm",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    ax.set(
        xlim=(-x_half_span, x_half_span),
        ylim=(-y_half_span, y_half_span),
        aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
        spine.set_linewidth(0.6)

    prefix = f"{panel_label}  " if panel_label else ""
    if mode == "full_system":
        detail = (
            f"all {snapshot['selected_protein_count']} proteins; "
            f"LCC={snapshot['n_proteins']} ({snapshot['lcf']:.2f})"
        )
    elif mode == "DNA_associated_assembly":
        detail = (
            f"{len(dna_associated_nodes)} DNA-associated proteins: "
            f"{len(snapshot['direct_oct4_nodes'])} direct OCT4, "
            f"{len(snapshot['direct_med1_nodes'])} direct MED1, "
            f"{len(snapshot['indirect_med1_nodes'])} indirect MED1"
        )
    elif mode == "DNA_linked_component":
        detail = f"DNA-linked component: {len(dna_linked_nodes)} proteins"
    elif mode == "DNA_neighborhood":
        detail = (
            f"DNA + {snapshot['selected_protein_count']} direct protein neighbors"
        )
    else:
        detail = (
            f"direct LCC={snapshot['n_proteins']}, "
            f"fraction={snapshot['lcf']:.2f}"
        )
    ax.set_title(
        f"{prefix}{snapshot['system']}  ({snapshot['replicate']}, "
        f"{snapshot['time_us']:.3f} μs)\n{detail}",
        fontsize=7.5,
    )


def get_task(manifest: pd.DataFrame, system: str, replicate: str) -> Task:
    row = manifest[(manifest["system"] == system) & (manifest["replicate"] == replicate) & manifest["available"]]
    if row.empty:
        raise KeyError(f"No available task for {replicate}/{system}")
    return task_from_row(row.iloc[0])


def plot_figure1(
    summaries: dict[str, pd.DataFrame],
    design: pd.DataFrame,
    framewise: pd.DataFrame,
    manifest: pd.DataFrame,
    out: Path,
) -> None:
    cluster = summaries["cluster_group"]
    cluster = cluster[np.isclose(cluster["cutoff_nm"], PRIMARY_CUTOFF_NM)]
    cluster_repeat = summaries["cluster_repeat"]
    cluster_repeat = cluster_repeat[
        np.isclose(cluster_repeat["cutoff_nm"], PRIMARY_CUTOFF_NM)
    ]
    frame = summaries["frame_group"]
    frame_repeat = summaries["frame_repeat"]
    x_map = composition_x(design)
    fig = plt.figure(figsize=(7.20, 8.05), constrained_layout=False)
    outer_grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.82, 1.0, 1.0],
        left=0.08,
        right=0.99,
        bottom=0.055,
        top=0.975,
        hspace=0.43,
    )
    top_grid = outer_grid[0, 0].subgridspec(1, 3, wspace=0.34)
    middle_grid = outer_grid[1, 0].subgridspec(1, 2, wspace=0.30)
    structure_grid = outer_grid[2, 0].subgridspec(1, 3, wspace=0.07)

    axes_top = [fig.add_subplot(top_grid[0, index]) for index in range(3)]
    cluster_index = cluster.set_index("system")
    ax = axes_top[0]
    x = np.asarray([x_map[system] for system in COMPOSITION_SYSTEMS])
    ax.plot(
        x,
        cluster_index.loc[
            COMPOSITION_SYSTEMS, "largest_cluster_fraction_mean"
        ],
        color=COLORS["MED1"],
        lw=1.8,
        zorder=2,
        label="100-chain series",
    )
    ax.errorbar(
        x,
        cluster_index.loc[
            COMPOSITION_SYSTEMS, "largest_cluster_fraction_mean"
        ],
        yerr=cluster_index.loc[
            COMPOSITION_SYSTEMS, "largest_cluster_fraction_sd"
        ],
        fmt="o",
        color=COLORS["MED1"],
        mec="white",
        mew=0.7,
        capsize=3,
        zorder=3,
    )
    for index, system in enumerate(COMPOSITION_SYSTEMS):
        values = cluster_repeat[
            cluster_repeat["system"] == system
        ]["largest_cluster_fraction"].to_numpy(float)
        ax.scatter(
            x[index] + np.linspace(-0.35, 0.35, len(values)),
            values,
            s=17,
            facecolor="white",
            edgecolor=COLORS["MED1"],
            linewidth=0.7,
            zorder=4,
        )
    control = cluster_index.loc["M50O0"]
    control_values = cluster_repeat[
        cluster_repeat["system"] == "M50O0"
    ]["largest_cluster_fraction"].to_numpy(float)
    control_x = 64.0
    ax.errorbar(
        control_x,
        control["largest_cluster_fraction_mean"],
        yerr=control["largest_cluster_fraction_sd"],
        fmt="D",
        color=COLORS["neutral"],
        mec="white",
        mew=0.7,
        capsize=3,
        zorder=3,
        label="50-chain MED1",
    )
    ax.scatter(
        control_x + np.linspace(-0.35, 0.35, len(control_values)),
        control_values,
        s=18,
        facecolor="white",
        edgecolor=COLORS["neutral"],
        linewidth=0.7,
        zorder=4,
    )
    ax.set(
        xlabel="MED1 molecule fraction (%)",
        ylabel="LCC fraction",
        xlim=(-3, 68),
        ylim=(0, 1.06),
    )
    ax.set_xticks([0, 25, 50, control_x], ["0", "25", "50", "100"])
    for break_x in [56.0, 58.0]:
        ax.plot(
            [break_x - 0.6, break_x + 0.6],
            [-0.018, 0.018],
            transform=ax.get_xaxis_transform(),
            color="black",
            lw=0.9,
            clip_on=False,
        )
    set_panel_title(ax, "A", "Largest component", label_x=-0.12)
    ax.legend(loc="lower right", fontsize=7.0, handlelength=1.7)

    for ax, metric, ylabel, label, title in [
        (
            axes_top[1],
            "second_cluster_fraction",
            "Second-largest component fraction",
            "B",
            "Subleading component",
        ),
        (
            axes_top[2],
            "finite_cluster_mean_size",
            "Finite-cluster mean size (chains)",
            "C",
            "Finite-cluster susceptibility",
        ),
    ]:
        ax.errorbar(
            x,
            cluster_index.loc[COMPOSITION_SYSTEMS, f"{metric}_mean"],
            yerr=cluster_index.loc[COMPOSITION_SYSTEMS, f"{metric}_sd"],
            marker="o",
            color=COLORS["OCT4"],
            capsize=3,
            lw=1.5,
            zorder=2,
        )
        for index, system in enumerate(COMPOSITION_SYSTEMS):
            values = cluster_repeat[
                cluster_repeat["system"] == system
            ][metric].to_numpy(float)
            ax.scatter(
                x[index] + np.linspace(-0.35, 0.35, len(values)),
                values,
                s=18,
                facecolor="white",
                edgecolor=COLORS["OCT4"],
                linewidth=0.7,
                zorder=3,
            )
        ax.set(
            xlabel="MED1 molecule fraction (%)",
            ylabel=ylabel,
            xlim=(-2, 52),
        )
        set_panel_title(
            ax,
            label,
            title,
            title_size=9.1 if label == "C" else 9.3,
            label_x=-0.11 if label == "C" else -0.12,
        )

    frame_indexed = frame.set_index("system")
    for panel, normalized in enumerate([False, True]):
        ax = fig.add_subplot(middle_grid[0, panel])
        for metric, molecule, color, length in [
            ("OCT4_mean_degree", "OCT4", COLORS["OCT4"], OCT4_LENGTH),
            ("MED1_mean_degree", "MED1", COLORS["MED1"], MED1_LENGTH),
        ]:
            systems = [
                system
                for system in COMPOSITION_SYSTEMS
                if np.isfinite(frame_indexed.loc[system, f"{metric}_mean"])
            ]
            scale = 100.0 / length if normalized else 1.0
            ax.errorbar(
                [x_map[system] for system in systems],
                [frame_indexed.loc[system, f"{metric}_mean"] * scale for system in systems],
                yerr=[frame_indexed.loc[system, f"{metric}_sd"] * scale for system in systems],
                marker="o",
                color=color,
                label=molecule,
            )
            for index, system in enumerate(systems):
                values = frame_repeat[
                    frame_repeat["system"] == system
                ][metric].dropna().to_numpy(float) * scale
                ax.scatter(
                    x_map[system] + np.linspace(-0.30, 0.30, len(values)),
                    values,
                    s=15,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.7,
                    zorder=3,
                )
            dna_values = frame_repeat[
                frame_repeat["system"] == "M10O90D1"
            ][metric].dropna().to_numpy(float) * scale
            dna_mean = frame_indexed.loc["M10O90D1", f"{metric}_mean"] * scale
            dna_sd = frame_indexed.loc["M10O90D1", f"{metric}_sd"] * scale
            dna_x = 11.6
            ax.errorbar(
                dna_x,
                dna_mean,
                yerr=dna_sd,
                marker="D",
                color=color,
                markeredgecolor=COLORS["DNA"],
                markeredgewidth=1.5,
                capsize=3,
                zorder=4,
            )
            ax.scatter(
                dna_x + np.linspace(-0.25, 0.25, len(dna_values)),
                dna_values,
                s=14,
                facecolor="white",
                edgecolor=COLORS["DNA"],
                linewidth=0.8,
                zorder=5,
            )
        ax.set(
            xlabel="MED1 molecule fraction (%)",
            ylabel=(
                "Distinct PP partners per 100 residues"
                if normalized
                else "Distinct PP partners per chain"
            ),
        )
        if not normalized:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="none",
                    markerfacecolor="white",
                    markeredgecolor=COLORS["DNA"],
                    markeredgewidth=1.5,
                    label="10% MED1 +DNA",
                )
            )
            labels.append("10% MED1 +DNA")
            ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.69, 0.99),
                ncols=3,
                fontsize=7.0,
            )
        ax.set_xlim(-2, 53)
        set_panel_title(
            ax,
            chr(ord("D") + panel),
            "Length-normalized connectivity" if normalized else "Chain-level connectivity",
        )

    axes_middle = fig.axes[3:5]
    axes_middle[0].text(
        0.98,
        0.05,
        "MED1/OCT4: 2.2–2.8×",
        transform=axes_middle[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
    )
    axes_middle[1].text(
        0.98,
        0.05,
        "MED1/OCT4: 0.50–0.64×",
        transform=axes_middle[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
    )

    for panel, system in enumerate(["M0O100", "M20O80", "M50O50"]):
        selection = select_snapshot(framewise, system)
        task = get_task(manifest, system, str(selection["replicate"]))
        ax = fig.add_subplot(structure_grid[0, panel])
        snapshot = build_spatial_snapshot(
            task,
            int(selection["frame"]),
            mode="full_system",
        )
        draw_spatial_snapshot(ax, snapshot, task.box_nm / 2)
        set_panel_title(
            ax,
            chr(ord("F") + panel),
            f"{int(x_map[system])}% MED1",
            title_size=9.2,
            label_size=11,
            title_pad=4,
            label_x=-0.07,
        )
    save_figure(
        fig,
        out / "fig1_MED1_composition_and_enrichment",
        target_pixels=(4227, 4587),
        target_pdf_points=(507.204625, 550.6027795585),
    )


def paired_bar(ax: plt.Axes, paired_group: pd.DataFrame, metric: str, ylabel: str) -> None:
    sub = paired_group[
        (paired_group["metric"] == metric)
        & np.isclose(paired_group["cutoff_nm"], PRIMARY_CUTOFF_NM)
    ].set_index("comparison").reindex([pair[2] for pair in DNA_PAIRS])
    x = np.arange(len(sub))
    ax.bar(x, sub["delta_with_minus_without_mean"], color=[system_color(pair[1]) for pair in DNA_PAIRS], alpha=0.82)
    ax.errorbar(x, sub["delta_with_minus_without_mean"], yerr=sub["delta_with_minus_without_sd"], fmt="none", color="#333333")
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(x, sub.index, rotation=22, ha="right")
    ax.set_ylabel(ylabel)


def plot_figure2(
    summaries: dict[str, pd.DataFrame],
    publication: dict[str, pd.DataFrame],
    framewise: pd.DataFrame,
    manifest: pd.DataFrame,
    out: Path,
) -> None:
    paired = summaries["paired"]
    fig = plt.figure(figsize=(7.05, 8.8), constrained_layout=False)
    outer_grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 0.90, 1.65],
        left=0.105,
        right=0.965,
        bottom=0.085,
        top=0.975,
        hspace=0.34,
    )
    metric_grid = outer_grid[0, 0].subgridspec(1, 2, wspace=0.33)
    distance_grid = outer_grid[1, 0].subgridspec(1, 2, wspace=0.24)
    structure_grid = outer_grid[2, 0].subgridspec(1, 2, wspace=0.0)

    def formulation_metric(
        ax: plt.Axes,
        metric: str,
        ylabel: str,
        ylim: tuple[float, float],
        show_legend: bool,
    ) -> None:
        table = paired[
            (paired["metric"] == metric)
            & np.isclose(paired["cutoff_nm"], PRIMARY_CUTOFF_NM)
        ]
        x = np.arange(len(PUBLICATION_DNA_PAIRS))
        for index, (_, _, label) in enumerate(PUBLICATION_DNA_PAIRS):
            sub = table[table["comparison"] == label]
            for offset, column, color, marker, legend_label in [
                (-0.15, "without_DNA", COLORS["neutral"], "o", "−DNA formulation"),
                (0.15, "with_DNA", COLORS["DNA"], "D", "+DNA formulation"),
            ]:
                values = sub[column].dropna().to_numpy(float)
                ax.scatter(
                    index + offset + np.linspace(-0.025, 0.025, len(values)),
                    values,
                    s=25,
                    color=color,
                    marker=marker,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                    label=legend_label if index == 0 else None,
                )
                ax.errorbar(
                    index + offset,
                    values.mean(),
                    yerr=values.std(ddof=1),
                    fmt="none",
                    ecolor=color,
                    capsize=3,
                    lw=1.0,
                    zorder=2,
                )
                ax.plot(
                    [index + offset - 0.06, index + offset + 0.06],
                    [values.mean(), values.mean()],
                    color="black",
                    lw=1.5,
                    zorder=4,
                )
        ax.set_xticks(x, ["OCT4-\nrich", "10%\nMED1", "MED1-\nrich"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        if show_legend:
            ax.legend(loc="best", fontsize=7.0)

    ax_a = fig.add_subplot(metric_grid[0, 0])
    formulation_metric(
        ax_a,
        "largest_cluster_fraction",
        "LCC fraction",
        (0, 1.07),
        True,
    )
    set_panel_title(ax_a, "A", "Network expansion")
    ax_b = fig.add_subplot(metric_grid[0, 1])
    formulation_metric(
        ax_b,
        "PP_edge_density",
        "Protein-network edge density",
        (0, 0.12),
        False,
    )
    set_panel_title(ax_b, "B", "Network densification")

    histogram = publication["histogram_group"]
    distance_group = publication["distance_group"].set_index("system")
    cdf_ax = fig.add_subplot(distance_grid[0, 0])
    cdf_specs = [
        ("M0O100D1", "OCT4-rich +DNA", COLORS["OCT4"]),
        ("M10O90D1", "10% MED1 +DNA", COLORS["mixed"]),
    ]
    for system, label, color in cdf_specs:
        sub = histogram[
            (histogram["system"] == system)
            & (histogram["bin_end_nm"] <= 25.0)
        ].sort_values("bin_end_nm")
        cdf_ax.plot(
            np.r_[0.0, sub["bin_end_nm"].to_numpy(float)],
            np.r_[0.0, sub["cumulative_probability_mean"].to_numpy(float)],
            color=color,
            lw=1.7,
            label=label,
        )
    cdf_ax.axvline(10.0, color="#777777", lw=0.9, ls="--", zorder=0)
    cdf_ax.axhline(0.9, color="#777777", lw=0.9, ls="--", zorder=0)
    for system, color, xytext in [
        ("M0O100D1", COLORS["OCT4"], (2.0, 0.78)),
        ("M10O90D1", COLORS["mixed"], (13.0, 0.72)),
    ]:
        q90 = float(distance_group.loc[system, "q90_distance_nm_mean"])
        within_10 = float(
            distance_group.loc[system, "fraction_within_10nm_mean"]
        )
        cdf_ax.scatter([q90, 10.0], [0.9, within_10], s=14, color=color)
        cdf_ax.annotate(
            rf"$d_{{90}}={q90:.2f}$ nm",
            xy=(q90, 0.9),
            xytext=xytext,
            color=color,
            fontsize=6.7,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65},
        )
    cdf_ax.set(
        xlabel="Distance to nearest DNA bead (nm)",
        ylabel="Cumulative fraction of beads\nin DNA-associated chains",
        xlim=(0, 25),
        ylim=(0, 1.02),
    )
    cdf_ax.legend(loc="lower right", fontsize=6.5, handlelength=1.9)
    set_panel_title(cdf_ax, "C", "DNA-distance CDF")

    distance_repeat = publication["distance_repeat"]
    d90_ax = fig.add_subplot(distance_grid[0, 1])
    for index, (system, label, color) in enumerate(cdf_specs):
        values = distance_repeat[
            distance_repeat["system"] == system
        ]["q90_distance_nm"].to_numpy(float)
        d90_ax.scatter(
            index + np.linspace(-0.04, 0.04, len(values)),
            values,
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        d90_ax.plot(
            [index - 0.12, index + 0.12],
            [values.mean(), values.mean()],
            color="black",
            lw=1.7,
            zorder=4,
        )
    d90_ax.set_xticks([0, 1], ["OCT4-rich\n+DNA", "10% MED1\n+DNA"])
    d90_ax.set_ylabel(r"Trajectory-level $d_{90}$ (nm)")
    set_panel_title(d90_ax, "D", r"Replicate-level $d_{90}$")

    snapshot_axes = [fig.add_subplot(structure_grid[0, column]) for column in range(2)]
    for panel, system in enumerate(["M0O100D1", "M10O90D1"]):
        selection = select_snapshot(framewise, system)
        task = get_task(manifest, system, str(selection["replicate"]))
        snapshot = build_spatial_snapshot(
            task,
            int(selection["frame"]),
            mode="DNA_associated_assembly",
        )
        draw_spatial_snapshot(snapshot_axes[panel], snapshot, task.box_nm / 2)
        snapshot_axes[panel].set_anchor("C")
        snapshot_axes[panel].set_title("")
        snapshot_axes[panel].text(
            -0.055,
            1.015,
            chr(ord("E") + panel),
            transform=snapshot_axes[panel].transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
        snapshot_axes[panel].text(
            0.5,
            1.015,
            "OCT4-rich +DNA" if panel == 0 else "10% MED1 +DNA",
            transform=snapshot_axes[panel].transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            clip_on=False,
        )
    save_figure(
        fig,
        out / "fig2_DNA_effects_and_spatial_snapshots",
        target_pixels=(3996, 5029),
        target_pdf_points=(479.506, 603.5985086497),
    )


def plot_figure3(
    summaries: dict[str, pd.DataFrame],
    publication: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    frame_group = summaries["frame_group"].set_index("system")
    frame_repeat = summaries["frame_repeat"]
    fig = plt.figure(figsize=(7.05, 6.25), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.08, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.10,
        right=0.985,
        bottom=0.09,
        top=0.965,
        wspace=0.43,
        hspace=0.34,
    )

    selected = [
        "M0O100",
        "M0O100D1",
        "M10O90",
        "M10O90D1",
        "M50O0",
        "M50O0D1",
    ]
    metrics = [
        "OO_contact_probability_mean",
        "OM_contact_probability_mean",
        "MM_contact_probability_mean",
        "OD_contact_probability_mean",
        "MD_contact_probability_mean",
    ]
    labels = ["O–O", "O–M", "M–M", "O–DNA", "M–DNA"]
    table = frame_group.loc[selected, metrics] * 100
    ax = fig.add_subplot(grid[0, 0])
    masked = np.ma.masked_invalid(table.to_numpy())
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad("#EEEEEE")
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    image.set_rasterized(True)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(selected)), selected)
    for row in range(len(selected)):
        for column in range(len(labels)):
            value = table.iloc[row, column]
            if np.isfinite(value):
                rgba = cmap(image.norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                label = f"{value:.1f}"
                color = "white" if luminance < 0.48 else "black"
            else:
                label, color = "NA", "#666666"
            ax.text(column, row, label, ha="center", va="center", fontsize=7.1, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Probability (%)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    set_panel_title(ax, "A", "Molecular contact network")

    ax = fig.add_subplot(grid[0, 1])
    systems = PUBLICATION_DNA_SYSTEMS
    x = np.arange(len(systems))
    oct4_mean = frame_group.loc[systems, "DNA_component_OCT4_mean"]
    med1_mean = frame_group.loc[systems, "DNA_component_MED1_mean"]
    ax.bar(x, oct4_mean, color=COLORS["OCT4"], label="OCT4")
    ax.bar(x, med1_mean, bottom=oct4_mean, color=COLORS["MED1"], label="MED1")
    for index, system in enumerate(systems):
        sub = frame_repeat[frame_repeat["system"] == system]
        component_total = sub["DNA_component_OCT4"] + sub["DNA_component_MED1"]
        direct_total = sub["N_OCT4_molecules_contacting_DNA"] + sub["N_MED1_molecules_contacting_DNA"]
        ax.scatter(
            np.full(len(sub), index + 0.05),
            component_total,
            s=20,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.scatter(
            np.full(len(sub), index - 0.05),
            direct_total,
            marker="D",
            s=17,
            facecolor="none",
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )
    ax.scatter([], [], marker="o", facecolor="white", edgecolor="black", label="component total")
    ax.scatter([], [], marker="D", facecolor="none", edgecolor="black", label="direct DNA neighbors")
    ax.set_xticks(x, ["OCT4-\nrich", "10%\nMED1", "MED1-\nrich"])
    ax.set(ylabel="Proteins linked to DNA", ylim=(0, 100))
    set_panel_title(ax, "B", "DNA-linked component", title_size=9.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncols=2,
        fontsize=7.0,
        borderaxespad=0,
        columnspacing=0.8,
        handletextpad=0.35,
        labelspacing=0.25,
    )

    probability_ax = fig.add_subplot(grid[1, 0])
    categories = [
        "neither_bound",
        "only_one_bound",
        "both_bound_different_windows",
        "shared_window",
    ]
    category_labels = ["Neither", "One", "Both,\ndifferent", "Shared\nwindow"]
    local = publication["local_repeat"]
    local = local[local["system"] == "M10O90D1"]
    for index, category in enumerate(categories):
        values = local[local["binding_category"] == category][
            "conditional_PP_probability"
        ].to_numpy(float)
        probability_ax.scatter(
            index + np.linspace(-0.04, 0.04, len(values)),
            values,
            s=21,
            facecolor="white",
            edgecolor="#333333",
            linewidth=0.7,
            zorder=3,
        )
        probability_ax.errorbar(
            index,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="o",
            color=COLORS["DNA"],
            capsize=3,
            zorder=4,
        )
    probability_ax.set_xticks(np.arange(4), category_labels)
    probability_ax.set_ylabel("P(PP edge | occupancy state)")
    set_panel_title(
        probability_ax,
        "C",
        "20-bp occupancy state | 10% MED1",
        title_size=8.8,
        label_x=-0.12,
    )

    odds_ax = fig.add_subplot(grid[1, 1])
    odds_repeat = publication["odds_repeat"]
    odds_group = publication["odds_group"].set_index("system")
    colors = [COLORS["OCT4"], COLORS["mixed"], COLORS["MED1"]]
    display = ["OCT4-rich", "10% MED1", "MED1-rich"]
    upper_values: list[float] = []
    for index, (system, color) in enumerate(zip(systems, colors)):
        values = odds_repeat[odds_repeat["system"] == system]["odds_ratio"].to_numpy(float)
        mean_log2 = odds_group.loc[system, "log2_odds_ratio_mean"]
        sd_log2 = odds_group.loc[system, "log2_odds_ratio_sd"]
        geometric = odds_group.loc[system, "geometric_mean_odds_ratio"]
        lower = geometric - 2 ** (mean_log2 - sd_log2)
        upper = 2 ** (mean_log2 + sd_log2) - geometric
        upper_values.append(geometric + upper)
        odds_ax.scatter(
            index + np.linspace(-0.05, 0.05, len(values)),
            values,
            s=21,
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
            zorder=3,
        )
        odds_ax.errorbar(
            index,
            geometric,
            yerr=np.asarray([[lower], [upper]]),
            fmt="o",
            color=color,
            capsize=3,
            zorder=4,
        )
        odds_ax.text(index, geometric + upper + 0.04 * max(upper_values), f"{geometric:.1f}", ha="center", fontsize=8)
    odds_ax.axhline(1, color="#888888", lw=0.8, ls="--")
    odds_ax.set_xticks(np.arange(3), display, rotation=15, ha="right")
    odds_ax.set_ylabel("Odds ratio: shared vs distinct windows")
    odds_ax.set_ylim(0, max(upper_values) * 1.18)
    set_panel_title(odds_ax, "D", "Shared-window enrichment", title_size=8.8, label_x=-0.12)
    save_figure(
        fig,
        out / "fig3_contact_network_mechanism",
        target_pixels=(4187, 3637),
        target_pdf_points=(502.530375, 436.6530726777),
    )


def select_matched_split_snapshot(
    framewise: pd.DataFrame,
    clusters: pd.DataFrame,
    systems: tuple[str, ...] = (
        "M10O90D1",
        "M10O90D1-S2",
        "M10O90D1-S4",
    ),
) -> pd.DataFrame:
    """Select one matched +DNA repeat/frame for requested continuity states.

    The same repeat and physical time are used for all requested systems.
    Selection minimizes the pooled standardized distance from the median
    residue-weighted LCC and edge density of each system, with only a weak
    preference for the middle of the production window.
    """
    systems = list(systems)
    primary = clusters[
        clusters["system"].isin(systems)
        & np.isclose(clusters["cutoff_nm"], PRIMARY_CUTOFF_NM)
    ][
        [
            "system",
            "replicate",
            "frame",
            "largest_cluster_residue_fraction",
        ]
    ]
    candidates = framewise[framewise["system"].isin(systems)][
        ["system", "replicate", "frame", "time_us", "PP_edge_density"]
    ].merge(primary, on=["system", "replicate", "frame"], validate="one_to_one")
    metrics = ["largest_cluster_residue_fraction", "PP_edge_density"]
    centers = candidates.groupby("system", observed=True)[metrics].median()
    scales = (
        candidates.groupby("system", observed=True)[metrics]
        .std(ddof=1)
        .clip(lower=1e-8)
    )
    ranked: list[tuple[float, str, int]] = []
    for (replicate, frame), group in candidates.groupby(
        ["replicate", "frame"], observed=True
    ):
        if set(group["system"]) != set(systems):
            continue
        score = 0.0
        for row in group.itertuples(index=False):
            for metric in metrics:
                score += abs(
                    (float(getattr(row, metric)) - centers.loc[row.system, metric])
                    / scales.loc[row.system, metric]
                )
        score += 0.05 * abs(float(group["time_us"].iloc[0]) - 3.0)
        ranked.append((score, str(replicate), int(frame)))
    if not ranked:
        raise RuntimeError(f"No common +DNA snapshot exists for {systems}")
    score, replicate, frame = min(ranked)
    selected = candidates[
        (candidates["replicate"] == replicate) & (candidates["frame"] == frame)
    ].copy()
    selected["condition"] = selected["system"].map(
        {
            "M10O90D1": "Full",
            "M10O90D1-S2": "Split-2",
            "M10O90D1-S4": "Split-4",
        }
    )
    selected["selection_score"] = score
    selected["selection_rule"] = (
        "one matched +DNA repeat/frame minimizing pooled standardized distance "
        "from system medians of residue-weighted LCC and edge density; weak "
        "3-us preference"
    )
    condition_order = {
        condition: index
        for index, condition in enumerate(["Full", "Split-2", "Split-4"])
    }
    return selected.sort_values(
        "condition",
        key=lambda values: values.map(condition_order),
    ).reset_index(drop=True)


def draw_chain_continuity_schematic(ax: plt.Axes) -> None:
    """Draw the matched Full/Split-2/Split-4 perturbation used in Fig. 3."""
    total_length = 1581
    configurations = [
        ("Full", [(1, 1581)], "10"),
        ("Split-2", [(1, 807), (808, 1581)], "20"),
        (
            "Split-4",
            [(1, 354), (355, 807), (808, 1171), (1172, 1581)],
            "40",
        ),
    ]
    y_positions = [2.15, 1.20, 0.25]
    x_start = 0.0
    x_span = 1581.0
    segment_color = COLORS["MED1"]
    for (label, segments, chain_label), y in zip(configurations, y_positions):
        ax.text(
            -255,
            y,
            label,
            ha="left",
            va="center",
            fontsize=9.0,
            fontweight="bold",
        )
        for start, stop in segments:
            left = x_start + (start - 1) / total_length * x_span
            right = x_start + stop / total_length * x_span
            gap = 7.0
            if start > 1:
                left += gap
            if stop < total_length:
                right -= gap
            patch = FancyBboxPatch(
                (left, y - 0.16),
                right - left,
                0.32,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=0.7,
                edgecolor="#A83F3E",
                facecolor=segment_color,
                alpha=0.88,
            )
            ax.add_patch(patch)
        ax.text(
            1660,
            y,
            chain_label,
            ha="left",
            va="center",
            fontsize=7.8,
            color="#4A4A4A",
        )

    for cut in [354, 807, 1171]:
        x = x_start + cut / total_length * x_span
        ax.plot([x, x], [-0.02, 0.52], color="#7A2E2D", lw=0.65, ls=":")
        ax.text(x, -0.12, str(cut), ha="center", va="top", fontsize=6.7)
    ax.text(
        790,
        -0.48,
        "Cuts: Split-2 after 807; Split-4 after 354, 807, and 1171",
        ha="center",
        va="top",
        fontsize=7.1,
        color="#555555",
    )
    ax.text(
        790,
        2.63,
        "1,581-residue MED1 sequence",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#555555",
    )
    ax.text(
        1660,
        2.63,
        "physical MED1 chains",
        ha="left",
        va="bottom",
        fontsize=7.3,
        color="#555555",
    )
    ax.text(
        2030,
        1.20,
        "Fixed:\n10 MED1 sequence equivalents\n90 OCT4 molecules\nresidues, charge, initial coordinates, box\n\nChanged:\nphysical chain continuity at the cuts\n(topology and exclusions rebuilt)",
        ha="left",
        va="center",
        fontsize=7.5,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#F5F5F5",
            "edgecolor": "#C6C6C6",
            "linewidth": 0.7,
        },
    )
    ax.set_xlim(-280, 2860)
    ax.set_ylim(-0.68, 2.92)
    ax.axis("off")
    set_panel_title(
        ax,
        "A",
        "Matched chain-continuity perturbation",
        title_size=9.5,
        label_x=-0.025,
        label_y=1.01,
    )


def plot_chain_continuity_figure(
    summaries: dict[str, pd.DataFrame],
    framewise: pd.DataFrame,
    clusters: pd.DataFrame,
    manifest: pd.DataFrame,
    out: Path,
) -> pd.DataFrame:
    """Build the proposed main Fig. 3 centered on MED1 chain continuity."""
    fig = plt.figure(figsize=(7.05, 8.15), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.90, 1.18, 1.70],
        left=0.09,
        right=0.975,
        bottom=0.075,
        top=0.975,
        hspace=0.31,
    )
    schematic_ax = fig.add_subplot(outer[0, 0])
    draw_chain_continuity_schematic(schematic_ax)

    metric_grid = outer[1, 0].subgridspec(1, 2, wspace=0.32)
    metric_axes = [fig.add_subplot(metric_grid[0, index]) for index in range(2)]
    cluster_repeat = summaries["cluster_repeat"]
    cluster_repeat = cluster_repeat[
        np.isclose(cluster_repeat["cutoff_nm"], PRIMARY_CUTOFF_NM)
    ]
    frame_repeat = summaries["frame_repeat"]
    labels = ["Full", "Split-2", "Split-4"]
    condition_specs = [
        (
            ["M10O90", "M10O90-S2", "M10O90-S4"],
            "−DNA",
            COLORS["neutral"],
            -0.09,
        ),
        (
            ["M10O90D1", "M10O90D1-S2", "M10O90D1-S4"],
            "+DNA",
            COLORS["DNA"],
            0.09,
        ),
    ]

    def paired_metric(
        ax: plt.Axes,
        table: pd.DataFrame,
        metric: str,
        ylabel: str,
        ylim: tuple[float, float],
        panel: str,
        title: str,
    ) -> None:
        x = np.arange(3, dtype=float)
        reductions: list[tuple[str, float]] = []
        for systems, condition, color, offset in condition_specs:
            subset = table[table["system"].isin(systems)][
                ["system", "replicate", metric]
            ]
            pivot = subset.pivot(index="replicate", columns="system", values=metric)
            pivot = pivot.reindex(index=REPEATS, columns=systems)
            if pivot.isna().any().any():
                raise RuntimeError(f"Incomplete paired split-control values for {metric}")
            values = pivot.to_numpy(float)
            for repeat_index, repeat_values in enumerate(values):
                jitter = (repeat_index - 1) * 0.018
                ax.plot(
                    x + offset + jitter,
                    repeat_values,
                    color=color,
                    lw=0.75,
                    alpha=0.42,
                    marker="o",
                    ms=3.2,
                    markerfacecolor="white",
                    markeredgewidth=0.65,
                    zorder=2,
                )
            means = values.mean(axis=0)
            sds = values.std(axis=0, ddof=1)
            ax.errorbar(
                x + offset,
                means,
                yerr=sds,
                fmt="D-",
                color=color,
                lw=1.55,
                ms=4.8,
                markeredgecolor="white",
                markeredgewidth=0.55,
                capsize=2.8,
                zorder=4,
                label=condition,
            )
            reductions.append((condition, 100.0 * (means[-1] / means[0] - 1.0)))
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D8D8D8", lw=0.45, alpha=0.7, zorder=0)
        ax.text(
            0.02,
            0.965,
            "Full → Split-4: "
            + "; ".join(f"{condition} {change:.0f}%" for condition, change in reductions),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color="#4A4A4A",
        )
        set_panel_title(ax, panel, title, label_x=-0.12)

    paired_metric(
        metric_axes[0],
        cluster_repeat,
        "largest_cluster_residue_fraction",
        "Residue-weighted LCC fraction",
        (0.0, 0.66),
        "B",
        "Residue-level network integration",
    )
    paired_metric(
        metric_axes[1],
        frame_repeat,
        "PP_edge_density",
        "Protein edge density",
        (0.0, 0.036),
        "C",
        "Molecular contact density",
    )
    metric_axes[0].legend(loc="center right", fontsize=7.1)
    metric_axes[1].text(
        0.98,
        0.04,
        "thin lines: matched trajectories\ndiamond: mean ± SD; n = 3",
        transform=metric_axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )

    selected = select_matched_split_snapshot(framewise, clusters)
    structure_grid = outer[2, 0].subgridspec(1, 3, wspace=0.035)
    snapshot_axes = [fig.add_subplot(structure_grid[0, index]) for index in range(3)]
    for index, row in enumerate(selected.itertuples(index=False)):
        task = get_task(manifest, row.system, row.replicate)
        snapshot = build_spatial_snapshot(task, int(row.frame), mode="full_system")
        draw_spatial_snapshot(snapshot_axes[index], snapshot, task.box_nm / 2)
        snapshot_axes[index].set_anchor("C")
        snapshot_axes[index].set_title("")
        snapshot_axes[index].text(
            -0.02,
            1.02,
            chr(ord("D") + index),
            transform=snapshot_axes[index].transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
        snapshot_axes[index].text(
            0.5,
            1.02,
            f"{row.condition} +DNA\nresidue-weighted LCC = "
            f"{row.largest_cluster_residue_fraction:.3f}",
            transform=snapshot_axes[index].transAxes,
            ha="center",
            va="bottom",
            fontsize=7.7,
            clip_on=False,
        )
    fig.text(
        0.5,
        0.405,
        f"Matched structural comparison: {selected.loc[0, 'replicate']}, "
        f"{selected.loc[0, 'time_us']:.3f} μs; common 82.7-nm view",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="#4A4A4A",
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["OCT4"], markeredgecolor="white", label="OCT4 in LCC"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["MED1"], markeredgecolor="white", label="MED1 in LCC"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["DNA"], markeredgecolor="white", label="DNA"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B9BEC5", markeredgecolor="white", label="protein outside LCC"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncols=4,
        fontsize=7.0,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    save_figure(fig, out / "fig3_chain_continuity")
    return selected


def map_matrix(maps: pd.DataFrame, system: str, interaction: str) -> np.ndarray:
    sub = maps[(maps["system"] == system) & (maps["interaction"] == interaction)]
    if sub.empty:
        raise KeyError(f"No map for {system} {interaction}")
    return sub.pivot(index="x_bin", columns="y_bin", values="contact_probability_mean").sort_index().sort_index(axis=1).to_numpy()


def read_residue_names(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[tuple[str, str]] = set()
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            key = (line[21], line[22:26])
            if key in seen:
                continue
            seen.add(key)
            names.append(line[17:20].strip())
    return names


def rolling_charge(names: list[str], window: int) -> np.ndarray:
    charge = np.asarray([AA_CHARGE.get(name, 0.0) for name in names])
    return np.convolve(charge, np.ones(window), mode="same")


def plot_figure4(root: Path, profiles: pd.DataFrame, out: Path) -> None:
    med1_charge = rolling_charge(
        read_residue_names(root / "simulation/inputs/MED1-alphafold.pdb"),
        25,
    )
    oct4_charge = rolling_charge(
        read_residue_names(root / "simulation/inputs/OCT4.pdb"),
        15,
    )

    def add_profile(
        ax: plt.Axes,
        interaction: str,
        side: str,
        system: str,
        color: str,
        label: str,
        linestyle: str,
    ) -> None:
        sub = profiles[
            (profiles["interaction"] == interaction)
            & (profiles["side"] == side)
            & (profiles["system"] == system)
        ].sort_values("residue_index")
        x = sub["residue_index"].to_numpy(float)
        mean = sub["contact_probability_mean"].to_numpy(float)
        sd = sub["contact_probability_sd"].to_numpy(float)
        ax.plot(x, mean, color=color, lw=1.35, ls=linestyle, label=label)
        ax.fill_between(
            x,
            np.maximum(mean - sd, 0),
            mean + sd,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    fig = plt.figure(figsize=(7.05, 5.55), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.94,
        bottom=0.10,
        top=0.965,
        wspace=0.42,
        hspace=0.42,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    ax = axes[0]
    for system, color, label, linestyle in [
        ("M0O100D1", COLORS["OCT4"], "OCT4-rich +DNA", "-"),
        ("M10O90D1", COLORS["mixed"], "10% MED1 +DNA", "--"),
    ]:
        add_profile(ax, "OCT4-DNA", "OCT4", system, color, label, linestyle)
    ax2 = ax.twinx()
    charge_line, = ax2.plot(
        np.arange(1, len(oct4_charge) + 1),
        oct4_charge,
        color="#999999",
        lw=0.8,
        alpha=0.7,
        label="rolling charge",
    )
    ax.axvspan(230, 234, color=COLORS["DNA"], alpha=0.12)
    ax.text(232, 0.04, "230–234", transform=ax.get_xaxis_transform(), ha="center", va="bottom", rotation=90, fontsize=7.0, color="#477A43")
    ax.set(xlabel="OCT4 residue", ylabel="DNA-contact probability")
    ax2.set_ylabel("15-residue charge sum (e)", color="#777777")
    ax2.tick_params(axis="y", labelsize=8, colors="#777777")
    set_panel_title(ax, "A", "OCT4–DNA contacts")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [charge_line], labels + [charge_line.get_label()], loc="upper left", fontsize=7.0)

    ax = axes[1]
    for system, color, label, linestyle in [
        ("M10O90D1", COLORS["mixed"], "10% MED1 +DNA", "--"),
        ("M50O0D1", COLORS["MED1"], "MED1-rich +DNA", "-"),
    ]:
        add_profile(ax, "MED1-DNA", "MED1", system, color, label, linestyle)
    ax2 = ax.twinx()
    charge_line, = ax2.plot(
        np.arange(1, len(med1_charge) + 1),
        med1_charge,
        color="#999999",
        lw=0.7,
        alpha=0.7,
        label="rolling charge",
    )
    ax.axvspan(1499, 1504, color=COLORS["DNA"], alpha=0.12)
    ax.text(1501.5, 0.04, "1499–1504", transform=ax.get_xaxis_transform(), ha="center", va="bottom", rotation=90, fontsize=7.0, color="#477A43")
    ax.set(xlabel="MED1 residue", ylabel="DNA-contact probability")
    ax2.set_ylabel("25-residue charge sum (e)", color="#777777")
    ax2.tick_params(axis="y", labelsize=8, colors="#777777")
    set_panel_title(ax, "B", "MED1–DNA contacts")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [charge_line], labels + [charge_line.get_label()], loc="upper left", fontsize=7.0)

    profile_specs = [
        (axes[2], "MED1", "MED1 side of O–M contacts"),
        (axes[3], "OCT4", "OCT4 side of O–M contacts"),
    ]
    for index, (ax, side, title) in enumerate(profile_specs):
        for system, color, label, linestyle in [
            ("M10O90", "#9C755F", "10% MED1 −DNA", "-"),
            ("M10O90D1", COLORS["mixed"], "10% MED1 +DNA", "--"),
            ("M50O50", COLORS["MED1"], "50% MED1 −DNA", ":"),
        ]:
            add_profile(ax, "MED1-OCT4", side, system, color, label, linestyle)
        if side == "MED1":
            ax.axvspan(1566, 1574, color=COLORS["MED1"], alpha=0.12)
            ax.text(1570, 0.04, "1566–1574", transform=ax.get_xaxis_transform(), ha="center", va="bottom", rotation=90, fontsize=7.0, color="#A43F3E")
        else:
            ax.axvspan(158, 186, color=COLORS["OCT4"], alpha=0.09)
            ax.axvspan(227, 229, color=COLORS["OCT4"], alpha=0.09)
            for center, label in [(172, "158–186"), (228, "227–229")]:
                ax.text(center, 0.04, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", rotation=90, fontsize=7.0, color="#365F8A")
        ax.set(
            xlabel=f"{side} residue",
            ylabel="Summed O–M contact probability",
        )
        set_panel_title(ax, chr(ord("C") + index), title, label_x=-0.13 if side == "OCT4" else -0.10)
        ax.legend(loc="upper right" if side == "OCT4" else "upper left", fontsize=7.0)

    save_figure(
        fig,
        out / "fig4_PBC_correct_binding_sites",
        target_pixels=(4253, 3248),
        target_pdf_points=(510.3425, 389.9556446281),
    )


def plot_figure5(
    framewise: pd.DataFrame,
    dna_only: pd.DataFrame,
    manifest: pd.DataFrame,
    out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dna_frame = pd.concat(
        [
            dna_only,
            framewise[framewise["system"].isin(DNA_SYSTEMS)][
                ["system", "replicate", "frame", "time_us", "DNA_Rg_nm", "DNA_Ree_nm", "DNA_asphericity"]
            ],
        ],
        ignore_index=True,
    )
    metrics = ["DNA_Rg_nm", "DNA_Ree_nm", "DNA_asphericity"]
    repeat = dna_frame.groupby(["system", "replicate"], observed=True)[metrics].mean().reset_index()
    group = mean_sd(repeat, ["system"], metrics).set_index("system")
    # Preserve the four-condition publication panel.  The corrected balanced
    # design contains an additional 20% MED1 +DNA trajectory, but adding it to
    # this figure changes both the scientific comparison and the established
    # manuscript layout.  It remains available in the exported tables.
    systems = ["D1"] + PUBLICATION_DNA_SYSTEMS
    labels = ["DNA\nonly", "OCT4-\nrich", "10%\nMED1", "MED1-\nrich"]
    colors = [
        COLORS["neutral"],
        COLORS["OCT4"],
        COLORS["mixed"],
        COLORS["MED1"],
    ]
    fig = plt.figure(figsize=(6.93, 5.15), constrained_layout=False)
    outer_grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.56, 1.30],
        left=0.055,
        right=0.99,
        bottom=0.09,
        top=0.965,
        hspace=0.04,
    )
    metric_grid = outer_grid[0, 0].subgridspec(1, 3, wspace=0.42)
    structure_grid = outer_grid[1, 0].subgridspec(1, 3, wspace=0.035)
    for panel, (metric, ylabel, title) in enumerate(
        [
            ("DNA_Rg_nm", "DNA $R_g$ (nm)", "Global DNA size"),
            (
                "DNA_Ree_nm",
                "DNA $R_{ee}$ (nm)",
                "DNA end-to-end distance",
            ),
            (
                "DNA_asphericity",
                "DNA asphericity",
                "DNA shape anisotropy",
            ),
        ]
    ):
        ax = fig.add_subplot(metric_grid[0, panel])
        x = np.arange(len(systems))
        means = group.loc[systems, f"{metric}_mean"].to_numpy(float)
        sds = group.loc[systems, f"{metric}_sd"].to_numpy(float)
        ax.errorbar(
            x,
            means,
            yerr=sds,
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=1,
            zorder=2,
        )
        ax.scatter(
            x,
            means,
            c=colors,
            s=48,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for index, system in enumerate(systems):
            values = repeat[repeat["system"] == system][metric].to_numpy(float)
            offsets = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(
                index + offsets,
                values,
                s=18,
                facecolor="white",
                edgecolor=colors[index],
                linewidth=0.8,
                zorder=4,
            )
        ax.set_xticks(x, labels, rotation=0, ha="center")
        ax.tick_params(axis="x", labelsize=7.0)
        ax.set_ylabel(ylabel)
        ax.yaxis.labelpad = 2
        reference = float(group.loc["D1", f"{metric}_mean"])
        ax.axhline(reference, color="#BBBBBB", lw=0.8, ls="--", zorder=0)
        ax.text(
            0.98,
            reference,
            "DNA-only mean",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.0,
            color="#888888",
        )
        set_panel_title(
            ax,
            chr(ord("A") + panel),
            title,
            title_size=9.2 if panel == 1 else 9.5,
            label_x=[-0.12, -0.18, -0.14][panel],
            label_y=1.04,
        )

    structure_systems = ["M0O100D1", "M10O90D1", "M50O0D1"]
    snapshots: list[dict[str, object]] = []
    for system in structure_systems:
        selection = select_snapshot(framewise, system)
        task = get_task(manifest, system, str(selection["replicate"]))
        snapshots.append(
            build_spatial_snapshot(
                task,
                int(selection["frame"]),
                mode="DNA_neighborhood",
            )
        )
    projected = [
        np.vstack(list(snapshot["projected_traces"].values()))
        for snapshot in snapshots
    ]
    shared_span = (
        max(float(np.max(np.abs(points[:, 0]))) for points in projected) + 0.8,
        max(float(np.max(np.abs(points[:, 1]))) for points in projected) + 0.8,
    )
    structure_titles = [
        "OCT4-rich + DNA",
        "10% MED1 + DNA",
        "MED1-rich + DNA",
    ]
    for panel, (snapshot, title) in enumerate(zip(snapshots, structure_titles)):
        ax = fig.add_subplot(structure_grid[0, panel])
        draw_spatial_snapshot(ax, snapshot, shared_span)
        ax.set_anchor("C")
        set_panel_title(
            ax,
            chr(ord("D") + panel),
            f"{title}\n{snapshot['selected_protein_count']} direct neighbors (≤1.2 nm)",
            title_size=8.3,
            label_size=11,
            title_pad=3,
            label_x=-0.10,
            label_y=1.07,
        )
    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["MED1"], markeredgecolor="white", label="MED1"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["OCT4"], markeredgecolor="white", label="OCT4"),
            plt.Line2D([0], [0], color=COLORS["DNA"], lw=2.2, label="DNA"),
            plt.Line2D([0], [0], color="#777777", lw=1.0, alpha=0.7, label="Direct PP edge (≤1.2 nm)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.185),
        ncols=4,
        fontsize=7.0,
        columnspacing=1.1,
        handletextpad=0.45,
        frameon=False,
    )
    save_figure(
        fig,
        out / "fig5_DNA_conformation_with_DNA_only_control",
        target_pixels=(4242, 2536),
        target_pdf_points=(509.6385078323, 304.515365038),
    )
    return repeat, group.reset_index()


def plot_system_size_confound(
    design: pd.DataFrame,
    out: Path,
) -> None:
    x_map = composition_x(design)
    comp_design = design.set_index("system").loc[COMPOSITION_SYSTEMS]
    x = [x_map[system] for system in COMPOSITION_SYSTEMS]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.2), constrained_layout=True)
    axes[0].plot(x, comp_design["box_nm"], marker="o", color=COLORS["mixed"])
    axes[0].set(xlabel="MED1 (% protein chains)", ylabel="Box length (nm)")
    panel_title(axes[0], "A", "Balanced box volume")
    residue_concentration = comp_design["protein_residue_concentration_mM"]
    concentration_center = float(residue_concentration.mean())
    axes[1].plot(x, residue_concentration, marker="o", color=COLORS["neutral"])
    axes[1].set(
        xlabel="MED1 (% protein chains)",
        ylabel="Protein-residue concentration (mM)",
        ylim=(concentration_center - 0.05, concentration_center + 0.05),
    )
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].axhline(
        concentration_center,
        color="#999999",
        lw=0.8,
        ls="--",
        zorder=0,
    )
    panel_title(axes[1], "B", "Fixed protein-residue concentration")
    save_figure(fig, out / "figS3_system_size_confound")


def plot_protein_dna_binned_contact_maps(
    maps: pd.DataFrame,
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.25), constrained_layout=True)
    panel_specs = [
        ("M0O100D1", "OCT4-DNA", "OCT4 residue", OCT4_LENGTH),
        ("M10O90D1", "OCT4-DNA", "OCT4 residue", OCT4_LENGTH),
        ("M10O90D1", "MED1-DNA", "MED1 residue", MED1_LENGTH),
        ("M20O80D1", "MED1-DNA", "MED1 residue", MED1_LENGTH),
    ]
    for ax, (system, interaction, protein_axis, protein_length) in zip(
        axes.ravel(), panel_specs
    ):
        image = ax.imshow(
            map_matrix(maps, system, interaction),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            extent=(1, 200, 1, protein_length),
        )
        ax.set(
            title=f"{system}: {interaction}",
            xlabel="DNA position (bp)",
            ylabel=protein_axis,
        )
        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.03,
            label="Contact probability",
        )
    save_figure(fig, out / "figS4_protein_DNA_binned_contact_maps")


def plot_spatial_network_timecourse(
    manifest: pd.DataFrame,
    out: Path,
) -> None:
    timecourse_systems = ["M0O100", "M0O100D1", "M10O90", "M10O90D1"]
    timecourse_frames = [1000, 3000, 5000]
    snapshots_by_system: list[list[dict[str, object]]] = []
    for system in timecourse_systems:
        task = get_task(manifest, system, "work-1")
        snapshots_by_system.append(
            [build_spatial_snapshot(task, frame) for frame in timecourse_frames]
        )
    row_spans: list[tuple[float, float]] = []
    for snapshots in snapshots_by_system:
        row_points = np.vstack(
            [
                np.vstack(list(snapshot["projected_traces"].values()))
                for snapshot in snapshots
            ]
        )
        row_spans.append(
            (
                float(np.max(np.abs(row_points[:, 0]))) + 2,
                float(np.max(np.abs(row_points[:, 1]))) + 2,
            )
        )
    fig, axes = plt.subplots(
        len(timecourse_systems),
        len(timecourse_frames),
        figsize=(7.50, 10.00),
        constrained_layout=False,
        gridspec_kw={
            "height_ratios": [
                y_half_span / x_half_span
                for x_half_span, y_half_span in row_spans
            ]
        },
    )
    fig.subplots_adjust(
        left=0.018,
        right=0.995,
        bottom=0.018,
        top=0.900,
        wspace=0.04,
        hspace=0.13,
    )
    for row, (snapshots, span) in enumerate(
        zip(snapshots_by_system, row_spans)
    ):
        for column, snapshot in enumerate(snapshots):
            draw_spatial_snapshot(axes[row, column], snapshot, span)
    for column, frame in enumerate(timecourse_frames):
        position = axes[0, column].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            0.952,
            f"{frame * 0.001:g} μs",
            ha="center",
            va="bottom",
            fontsize=8.8,
        )
    fig.suptitle(
        "PBC-reconstructed direct-component timecourse (work-1 trajectory)",
        fontsize=11,
        y=0.990,
    )
    save_figure(fig, out / "figS6_spatial_network_timecourse")


def plot_med1_rich_spatial_snapshots(
    framewise: pd.DataFrame,
    manifest: pd.DataFrame,
    out: Path,
) -> None:
    systems = ["M50O0", "M50O0D1", "M50O50"]
    snapshots: list[dict[str, object]] = []
    for system in systems:
        selection = select_snapshot(framewise, system)
        task = get_task(manifest, system, str(selection["replicate"]))
        snapshots.append(build_spatial_snapshot(task, int(selection["frame"])))
    half_span = max(
        np.max(
            np.abs(np.vstack(list(snapshot["projected_traces"].values())))
        )
        for snapshot in snapshots
    ) + 2
    fig, axes = plt.subplots(
        1,
        len(systems),
        figsize=(10.8, 3.8),
        constrained_layout=True,
    )
    for ax, snapshot in zip(axes, snapshots):
        draw_spatial_snapshot(
            ax,
            snapshot,
            float(half_span),
            scale_bar_inset_nm=5.0,
        )
    fig.suptitle(
        "MED1-rich ceiling controls and the 100-chain mixed endpoint",
        fontsize=12,
    )
    save_figure(
        fig,
        out / "figS5_MED1_rich_spatial_snapshots",
        target_pixels=(6282, 2265),
    )


def plot_absolute_om_binned_contact_maps(
    maps: pd.DataFrame,
    out: Path,
) -> None:
    """Render absolute and difference O--M maps with explicit axes and scales."""
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 3.10), constrained_layout=True)
    without_dna = map_matrix(maps, "M10O90", "MED1-OCT4")
    with_dna = map_matrix(maps, "M10O90D1", "MED1-OCT4")
    vmax = max(float(without_dna.max()), float(with_dna.max()))
    absolute_image = None
    for panel_index, (ax, matrix, title) in enumerate(
        [
            (axes[0], without_dna, "M10O90"),
            (axes[1], with_dna, "M10O90D1"),
        ]
    ):
        absolute_image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=vmax,
        )
        ax.set_xlabel("OCT4 bin (20 residues)")
        if panel_index == 0:
            ax.set_ylabel("MED1 bin (50 residues)")
        panel_title(ax, chr(ord("A") + panel_index), title)

    difference = with_dna - without_dna
    limit = max(float(np.abs(difference).max()), 1e-8)
    difference_image = axes[2].imshow(
        difference,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[2].set_xlabel("OCT4 bin (20 residues)")
    panel_title(axes[2], "C", "+DNA − no DNA")
    fig.colorbar(
        absolute_image,
        ax=axes[:2],
        fraction=0.026,
        pad=0.025,
        label="Contact probability",
    )
    fig.colorbar(
        difference_image,
        ax=axes[2],
        fraction=0.052,
        pad=0.035,
        label="Δ contact probability",
    )
    save_figure(fig, out / "figS7_absolute_OM_binned_contact_maps")


def plot_dna_network_contact_details(
    framewise: pd.DataFrame,
    design: pd.DataFrame,
    out: Path,
) -> None:
    """Separate total DNA-neighborhood size from species-normalized occupancy."""
    frame_repeat = (
        framewise[framewise["system"].isin(DNA_SYSTEMS)]
        .groupby(["system", "replicate"], as_index=False)
        .mean(numeric_only=True)
    )
    frame_group = frame_repeat.groupby("system").mean(numeric_only=True)
    design_index = design.set_index("system")
    x = np.arange(len(DNA_SYSTEMS))
    n_proteins = np.asarray(
        [
            design_index.loc[system, "n_physical_protein_chains"]
            for system in DNA_SYSTEMS
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.50, 3.15), constrained_layout=True)
    panel_specs = [
        (
            axes[0],
            "N_OCT4_molecules_contacting_DNA",
            "N_MED1_molecules_contacting_DNA",
            "Direct DNA contacts",
        ),
        (
            axes[1],
            "DNA_component_OCT4",
            "DNA_component_MED1",
            "Complete DNA-linked component",
        ),
    ]
    for panel_index, (ax, oct4_metric, med1_metric, title) in enumerate(panel_specs):
        oct4 = np.asarray(
            [frame_group.loc[system, oct4_metric] for system in DNA_SYSTEMS]
        ) / n_proteins
        med1 = np.asarray(
            [frame_group.loc[system, med1_metric] for system in DNA_SYSTEMS]
        ) / n_proteins
        ax.bar(x, oct4, color=COLORS["OCT4"], label="OCT4")
        ax.bar(x, med1, bottom=oct4, color=COLORS["MED1"], label="MED1")
        for system_index, system in enumerate(DNA_SYSTEMS):
            denominator = n_proteins[system_index]
            repeat_values = frame_repeat[frame_repeat["system"] == system]
            totals = (
                repeat_values[oct4_metric].to_numpy(float)
                + repeat_values[med1_metric].to_numpy(float)
            ) / denominator
            ax.scatter(
                system_index + np.linspace(-0.055, 0.055, len(totals)),
                totals,
                s=18,
                facecolor="white",
                edgecolor="#333333",
                linewidth=0.65,
                zorder=3,
            )
        ax.set_xticks(x, [pair[2] for pair in DNA_PAIRS], rotation=27, ha="right")
        ax.set(ylabel="Fraction of all protein chains", ylim=(0, 1.04))
        panel_title(ax, chr(ord("A") + panel_index), title)
    axes[0].legend(loc="upper left")

    ax = axes[2]
    species_specs = [
        ("OCT4_fraction_contacting_DNA", "OCT4", COLORS["OCT4"], "o"),
        ("MED1_fraction_contacting_DNA", "MED1", COLORS["MED1"], "s"),
    ]
    for metric, label, color, marker in species_specs:
        means: list[float] = []
        sds: list[float] = []
        for system_index, system in enumerate(DNA_SYSTEMS):
            values = frame_repeat.loc[
                frame_repeat["system"] == system, metric
            ].dropna().to_numpy(float)
            means.append(float(np.mean(values)) if len(values) else np.nan)
            sds.append(float(np.std(values, ddof=1)) if len(values) > 1 else np.nan)
            if len(values):
                ax.scatter(
                    system_index + np.linspace(-0.045, 0.045, len(values)),
                    values,
                    s=20,
                    facecolor="white",
                    edgecolor=color,
                    marker=marker,
                    linewidth=0.75,
                    zorder=3,
                )
        ax.errorbar(
            x,
            means,
            yerr=sds,
            color=color,
            marker=marker,
            label=label,
            capsize=2.5,
        )
    ax.set_xticks(x, [pair[2] for pair in DNA_PAIRS], rotation=27, ha="right")
    ax.set(
        ylabel="Within-species direct-DNA fraction",
        ylim=(0, 0.60),
    )
    ax.legend(loc="upper right")
    panel_title(ax, "C", "Species-normalized direct occupancy")
    save_figure(fig, out / "figS9_DNA_network_contact_details")


def plot_degree_distribution_and_med1_partners(
    framewise: pd.DataFrame,
    degree_group: pd.DataFrame,
    design: pd.DataFrame,
    out: Path,
) -> None:
    """Show the species identity of the partners contacted by MED1."""
    fig, ax = plt.subplots(figsize=(4.8, 3.35), constrained_layout=True)

    composition = framewise[
        framewise["system"].isin(COMPOSITION_SYSTEMS[1:])
    ].copy()
    composition["MED1_OCT4_partners"] = (
        composition["OM_contact_probability"] * composition["n_OCT4"]
    )
    composition["MED1_MED1_partners"] = composition[
        "MM_contact_probability"
    ] * np.maximum(composition["n_MED1_sequence_equivalents"] - 1, 0)
    repeat = (
        composition.groupby(["system", "replicate"], as_index=False)[
            ["MED1_OCT4_partners", "MED1_MED1_partners"]
        ]
        .mean()
    )
    repeat["MED1_total_partners"] = (
        repeat["MED1_OCT4_partners"] + repeat["MED1_MED1_partners"]
    )
    group = repeat.groupby("system").agg(
        OCT4_mean=("MED1_OCT4_partners", "mean"),
        MED1_mean=("MED1_MED1_partners", "mean"),
        total_mean=("MED1_total_partners", "mean"),
        total_sd=("MED1_total_partners", "std"),
    )
    x_map = composition_x(design)
    systems = COMPOSITION_SYSTEMS[1:]
    x_values = np.asarray([x_map[system] for system in systems], dtype=float)
    oct4 = np.asarray([group.loc[system, "OCT4_mean"] for system in systems])
    med1 = np.asarray([group.loc[system, "MED1_mean"] for system in systems])
    totals = np.asarray([group.loc[system, "total_mean"] for system in systems])
    total_sd = np.asarray([group.loc[system, "total_sd"] for system in systems])
    ax.bar(x_values, oct4, width=3.6, color=COLORS["OCT4"], label="OCT4 partners")
    ax.bar(
        x_values,
        med1,
        width=3.6,
        bottom=oct4,
        color=COLORS["MED1"],
        label="MED1 partners",
    )
    ax.errorbar(
        x_values,
        totals,
        yerr=total_sd,
        color="#333333",
        fmt="none",
        capsize=2.5,
        linewidth=0.9,
    )
    for system, x_value in zip(systems, x_values):
        values = repeat.loc[
            repeat["system"] == system, "MED1_total_partners"
        ].to_numpy(float)
        ax.scatter(
            x_value + np.linspace(-0.35, 0.35, len(values)),
            values,
            s=17,
            facecolor="white",
            edgecolor="#333333",
            linewidth=0.6,
            zorder=3,
        )
    ax.set(
        xlabel="MED1 molecule fraction (%)",
        ylabel="Partners per MED1 chain",
        xlim=(2, 53),
        ylim=(0, 3.7),
    )
    ax.set_xticks(x_values)
    ax.legend(loc="upper right", fontsize=6.5)
    panel_title(ax, "A", "Species composition of MED1 partners")
    save_figure(fig, out / "figS12_degree_distribution")


def plot_supplementary(
    summaries: dict[str, pd.DataFrame],
    design: pd.DataFrame,
    framewise: pd.DataFrame,
    clusters: pd.DataFrame,
    profiles: pd.DataFrame,
    maps: pd.DataFrame,
    manifest: pd.DataFrame,
    out: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    x_map = composition_x(design)
    cluster_group = summaries["cluster_group"]

    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    for cutoff, color, marker in [(1.0, COLORS["OCT4"], "o"), (1.2, COLORS["mixed"], "s"), (1.4, COLORS["MED1"], "^")]:
        sub = cluster_group[np.isclose(cluster_group["cutoff_nm"], cutoff)]
        error_series(ax, sub, COMPOSITION_SYSTEMS, x_map, "largest_cluster_fraction", color=color, label=f"{cutoff:.1f} nm", marker=marker)
    ax.set(xlabel="MED1 (% protein chains)", ylabel="Largest-component fraction", ylim=(0, 1.03))
    ax.legend()
    panel_title(ax, "A", "Contact-cutoff sensitivity")
    save_figure(fig, out / "figS1_cutoff_sensitivity")

    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    paired = summaries["paired_group"]
    for cutoff, color, marker in [(1.0, COLORS["OCT4"], "o"), (1.2, COLORS["mixed"], "s"), (1.4, COLORS["MED1"], "^")]:
        sub = paired[(paired["metric"] == "largest_cluster_fraction") & np.isclose(paired["cutoff_nm"], cutoff)].set_index("comparison").reindex([pair[2] for pair in DNA_PAIRS])
        x = np.arange(len(sub))
        ax.errorbar(x, sub["delta_with_minus_without_mean"], yerr=sub["delta_with_minus_without_sd"], marker=marker, color=color, label=f"{cutoff:.1f} nm")
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(x, sub.index, rotation=22, ha="right")
    ax.set_ylabel("Δ largest-component fraction")
    ax.legend()
    panel_title(ax, "A", "DNA-effect cutoff sensitivity")
    save_figure(fig, out / "figS1b_DNA_effect_cutoff_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.2), constrained_layout=True)
    block = summaries["block_group"]
    for system in ["M0O100", "M10O90", "M20O80", "M30O70", "M50O50"]:
        sub = block[block["system"] == system].sort_values("production_block")
        axes[0].errorbar(sub["production_block"], sub["largest_cluster_fraction_mean"], yerr=sub["largest_cluster_fraction_sd"], marker="o", label=system)
    axes[0].set(xlabel="Production block", ylabel="Largest-component fraction")
    axes[0].legend(fontsize=6, ncol=2)
    panel_title(axes[0], "A", "Composition trajectories")
    for system in DNA_SYSTEMS:
        sub = block[block["system"] == system].sort_values("production_block")
        axes[1].errorbar(sub["production_block"], sub["largest_cluster_fraction_mean"], yerr=sub["largest_cluster_fraction_sd"], marker="o", label=system)
    axes[1].set(xlabel="Production block", ylabel="Largest-component fraction")
    axes[1].legend(fontsize=6)
    panel_title(axes[1], "B", "DNA-addition trajectories")
    save_figure(fig, out / "figS2_production_block_stability")

    plot_system_size_confound(design, out)

    plot_protein_dna_binned_contact_maps(maps, out)

    plot_med1_rich_spatial_snapshots(framewise, manifest, out)

    plot_spatial_network_timecourse(manifest, out)

    plot_absolute_om_binned_contact_maps(maps, out)

    fig = plt.figure(figsize=(12.8, 7.0), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 0.82],
        left=0.07,
        right=0.985,
        bottom=0.10,
        top=0.93,
        wspace=0.22,
        hspace=0.38,
    )
    ax = fig.add_subplot(grid[0, :])
    frame_group = summaries["frame_group"].set_index("system")
    exposure_systems = [
        system for system in COMPOSITION_SYSTEMS if system in frame_group.index
    ]
    exposure_x = [x_map[system] for system in exposure_systems]
    ax.errorbar(
        exposure_x,
        [frame_group.loc[system, "direct_largest_protein_fraction_mean"] for system in exposure_systems],
        yerr=[frame_group.loc[system, "direct_largest_protein_fraction_sd"] for system in exposure_systems],
        marker="o",
        color=COLORS["mixed"],
        capsize=3,
        label="Protein LCC",
    )
    ax.set(
        xlabel="MED1 molecule fraction (%)",
        ylabel="Largest-component fraction",
        ylim=(0, 1.05),
    )
    set_panel_title(
        ax,
        "A",
        "Component exposure across the concentration-balanced composition series",
        title_size=11,
    )
    ax.legend()
    for panel, system in enumerate(["M0O100", "M10O90", "M20O80", "M50O50"]):
        selection = select_snapshot(framewise, system)
        task = get_task(manifest, system, str(selection["replicate"]))
        snapshot_ax = fig.add_subplot(grid[1, panel])
        snapshot = build_spatial_snapshot(
            task,
            int(selection["frame"]),
            mode="full_system",
        )
        draw_spatial_snapshot(snapshot_ax, snapshot, task.box_nm / 2)
        set_panel_title(
            snapshot_ax,
            chr(ord("B") + panel),
            f"{x_map[system]:.0f}% MED1",
            title_size=9.5,
            label_size=12.5,
            label_x=-0.07,
        )
    save_figure(
        fig,
        out / "figS8_component_exposure_and_composition_snapshots",
        target_pixels=(7765, 3642),
    )

    plot_dna_network_contact_details(framewise, design, out)

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.7), constrained_layout=True)

    def profile_panel(
        ax: plt.Axes,
        interaction: str,
        side: str,
        systems: list[tuple[str, str, str]],
        panel: str,
        title: str,
    ) -> None:
        for system, label, color in systems:
            sub = profiles[
                (profiles["system"] == system)
                & (profiles["interaction"] == interaction)
                & (profiles["side"] == side)
            ].sort_values("residue_index")
            x_values = sub["residue_index"].to_numpy(float)
            mean = sub["contact_probability_mean"].to_numpy(float)
            sd = sub["contact_probability_sd"].to_numpy(float)
            ax.plot(x_values, mean, color=color, lw=1.05, label=label)
            ax.fill_between(x_values, np.maximum(0, mean - sd), mean + sd, color=color, alpha=0.14, linewidth=0)
        ax.set(xlabel=f"{side} residue", ylabel="Per-chain contact probability")
        ax.legend(fontsize=6.2)
        panel_title(ax, panel, title)

    profile_panel(
        axes[0, 0],
        "OCT4-DNA",
        "OCT4",
        [("M0O100D1", "OCT4-only", COLORS["OCT4"]), ("M10O90D1", "10% MED1", COLORS["mixed"]), ("M20O80D1", "20% MED1", "#9C755F")],
        "A",
        "OCT4–DNA profile",
    )
    profile_panel(
        axes[0, 1],
        "MED1-DNA",
        "MED1",
        [("M10O90D1", "10% MED1", COLORS["mixed"]), ("M20O80D1", "20% MED1", "#9C755F"), ("M50O0D1", "pure MED1 ref.", COLORS["MED1"])],
        "B",
        "MED1–DNA profile",
    )
    om_systems = [("M10O90", "10% −DNA", COLORS["neutral"]), ("M10O90D1", "10% +DNA", COLORS["mixed"]), ("M50O50", "50% −DNA", COLORS["MED1"])]
    profile_panel(axes[1, 0], "MED1-OCT4", "MED1", om_systems, "C", "O–M profile | MED1 side")
    profile_panel(axes[1, 1], "MED1-OCT4", "OCT4", om_systems, "D", "O–M profile | OCT4 side")
    save_figure(fig, out / "figS10_DNA_contact_state")

    plot_degree_distribution_and_med1_partners(
        framewise,
        summaries["degree_group"],
        design,
        out,
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.2), constrained_layout=True)
    cluster_primary = clusters[np.isclose(clusters["cutoff_nm"], PRIMARY_CUTOFF_NM)].copy()
    cluster_primary["production_block"] = ((cluster_primary["frame"] - 1) // 1000).astype(int)
    for ax, (without, with_dna, label) in zip(axes, DNA_PAIRS[:2]):
        repeat_block = cluster_primary[cluster_primary["system"].isin([without, with_dna])].groupby(["system", "replicate", "production_block"])["largest_cluster_fraction"].mean().unstack("system")
        repeat_block["delta"] = repeat_block[with_dna] - repeat_block[without]
        group = repeat_block.reset_index().groupby("production_block")["delta"].agg(["mean", "std"]).reset_index()
        ax.errorbar(group["production_block"], group["mean"], yerr=group["std"], marker="o", color=system_color(with_dna))
        ax.axhline(0, color="#777777", ls="--", lw=0.8)
        ax.set(xlabel="Production block", ylabel="Δ LCC fraction", title=label)
    save_figure(fig, out / "figS13_network_time_stability")


def forcefield_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in manifest[manifest["available"]].itertuples():
        metadata = json.loads((Path(row.path) / "run_metadata.json").read_text())
        audit = metadata["exclusion_audit"]
        med1 = audit["med1_exclusions_per_chain"]
        parameters = metadata.get("run_parameters", {})
        rows.append(
            {
                "system": row.system,
                "replicate": row.replicate,
                "analysis_included": bool(row.analysis_included),
                "timestep_fs": parameters.get("timestep_fs", 10.0),
                "box_nm": row.box_nm,
                "particles": audit["particles"],
                "med1_sequence_equivalents": audit["med1_sequence_equivalents"],
                "med1_physical_chains": audit["med1_physical_chains"],
                "med1_exclusions_before_filter_refresh": med1["before_filter_refresh"] if med1 is not None else np.nan,
                "med1_exclusions_after_filter_refresh": med1["after_filter_refresh"] if med1 is not None else np.nan,
                "med1_exclusions_after_split": med1["after_split"] if med1 is not None else np.nan,
                "stale_MED1_exclusions_removed": med1["stale_removed"] if med1 is not None else 0,
                "med1_fragments": med1["fragments"] if med1 is not None else 0,
                "assembled_exclusions": audit["assembled_exclusions"],
                "custom_nonbonded_force_1_exclusions": audit["custom_nonbonded_force_exclusions"][0],
                "custom_nonbonded_force_2_exclusions": audit["custom_nonbonded_force_exclusions"][1],
                "system_xml_sha256": metadata["system_xml_sha256"],
            }
        )
    table = pd.DataFrame(rows)
    if not (
        table["assembled_exclusions"].eq(table["custom_nonbonded_force_1_exclusions"]).all()
        and table["assembled_exclusions"].eq(table["custom_nonbonded_force_2_exclusions"]).all()
    ):
        raise RuntimeError("A corrected trajectory has inconsistent CustomNonbondedForce exclusions")
    if table.loc[table["med1_sequence_equivalents"] > 0, "stale_MED1_exclusions_removed"].ne(2736).any():
        raise RuntimeError("A MED1-containing corrected trajectory lacks the expected stale-exclusion removal audit")
    expected_after_split = table["med1_fragments"].map({0: np.nan, 1: 3651, 2: 3650, 4: 3404})
    med1_mask = table["med1_sequence_equivalents"] > 0
    if not table.loc[med1_mask, "med1_exclusions_after_split"].eq(expected_after_split[med1_mask]).all():
        raise RuntimeError("Unexpected final MED1 exclusion count after chain splitting")
    return table


def write_results_report(
    reports: Path,
    summaries: dict[str, pd.DataFrame],
    design: pd.DataFrame,
    validation: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    cluster = summaries["cluster_group"]
    cluster = cluster[np.isclose(cluster["cutoff_nm"], PRIMARY_CUTOFF_NM)].set_index("system")
    paired = summaries["paired_group"]
    paired = paired[
        (paired["metric"] == "largest_cluster_fraction")
        & np.isclose(paired["cutoff_nm"], PRIMARY_CUTOFF_NM)
    ].set_index("comparison")
    frame_group = summaries["frame_group"].set_index("system")
    lines = [
        "# Corrected balanced-design analysis",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`.",
        f"Dataset policy: `{DATASET_POLICY}`.",
        "",
        "## Dataset boundary",
        "",
        "Only trajectories under `Supplementary_Simulation/hpc_med1_balanced_design_20260731/tasks` are used for protein-containing analyses. The old MED1-containing trajectories are excluded. DNA-only D1 is retained because it contains no MED1 and therefore cannot carry the MED1 nonbonded-exclusion error.",
        "",
        "M15O85 and M50O50 are each summarized from all three completed work, work-1, and work-2 trajectories. The separate M15O85 numerical-stability run is not pooled into the composition series.",
        "",
        "## Composition series (1.2 nm molecular contact cutoff)",
        "",
    ]
    for system in COMPOSITION_SYSTEMS:
        row = cluster.loc[system]
        lines.append(
            f"- {system}: largest-component fraction {row['largest_cluster_fraction_mean']:.3f} ± {row['largest_cluster_fraction_sd']:.3f} (n={int(row['largest_cluster_fraction_n'])})."
        )
    lines += ["", "## True DNA-addition contrasts", ""]
    for _, _, label in DNA_PAIRS:
        row = paired.loc[label]
        lines.append(
            f"- {label}: Δ largest-component fraction (+DNA − D0) {row['delta_with_minus_without_mean']:+.3f} ± {row['delta_with_minus_without_sd']:.3f} (n={int(row['delta_with_minus_without_n'])} matched trajectories)."
        )
    lines += [
        "",
        "## MED1 chain-continuity controls",
        "",
        "- Full, Split-2, and Split-4 contain different numbers of physical chains, so their physical-chain edge densities and legacy physical-node residue-weighted LCC values are not used as chain-continuity evidence.",
        "- Run `run_mechanism_extensions.py` for fixed-100-parent and fixed-130-block metrics, sibling-fragment exclusion, matched absolute differences, difference-in-differences, and unified DNA-linked radial analyses.",
    ]
    lines += ["", "## DNA conformation in protein-containing systems", ""]
    for system in DNA_SYSTEMS:
        lines.append(
            f"- {system}: Rg {frame_group.loc[system, 'DNA_Rg_nm_mean']:.2f} ± {frame_group.loc[system, 'DNA_Rg_nm_sd']:.2f} nm; Ree {frame_group.loc[system, 'DNA_Ree_nm_mean']:.2f} ± {frame_group.loc[system, 'DNA_Ree_nm_sd']:.2f} nm; asphericity {frame_group.loc[system, 'DNA_asphericity_mean']:.3f} ± {frame_group.loc[system, 'DNA_asphericity_sd']:.3f}."
        )
    concentration = design[design["system"].isin(COMPOSITION_SYSTEMS)]["protein_residue_concentration_mM"]
    lines += [
        "",
        "## Force-field provenance",
        "",
        f"- All {int((audit['med1_sequence_equivalents'] > 0).sum())} available MED1-containing task records report removal of exactly 2,736 stale intramolecular exclusions per MED1 sequence.",
        "- Final per-sequence MED1 exclusion counts are 3,651 (Full), 3,650 (Split-2), and 3,404 (Split-4). In every task, both CustomNonbondedForce exclusion counts equal the assembled exclusion table.",
        "",
        "## Design checks",
        "",
        f"- Protein-residue concentration across the 100-chain composition series: {concentration.mean():.4f} mM (range {concentration.min():.4f}–{concentration.max():.4f} mM).",
        "- M50O0 is retained only as a separate 50-chain pure-MED1 control; it is not plotted as the 100% endpoint of the 100-chain composition series.",
        "- Split-2 and Split-4 are reported as dedicated chain-length controls and are not inserted into the composition trend.",
        "- Individual trajectories are the statistical units. Sampled frames estimate each trajectory mean and do not increase n.",
        "",
        "## Validation",
        "",
    ]
    for row in validation.itertuples():
        lines.append(f"- {'PASS' if row.passed else 'FAIL'} — {row.check}: {row.value} (required: {row.required}).")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "These finite periodic-box simulations quantify composition-dependent association and network connectivity in the specified coarse-grained model. The composition response is interpreted as a gradual finite-network trend; phase coexistence, a binodal, and liquid material properties are not tested.",
    ]
    (reports / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recompute", action="store_true", help="Rebuild every per-trajectory cache")
    parser.add_argument("--skip-figures", action="store_true", help="Build and validate tables only")
    parser.add_argument("--max-tasks", type=int, help="Developer smoke-test limit; do not use for final analysis")
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message="DCDReader currently makes independent timesteps",
    )
    warnings.filterwarnings(
        "ignore",
        message="Unknown element .*",
    )
    warnings.filterwarnings(
        "ignore",
        message="Element information is missing.*",
    )

    package_root = args.package_root.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    cache_dir = output / "cache" / ANALYSIS_VERSION
    tables_dir = output / "tables"
    reports_dir = output / "reports"
    main_figures = output / "figures" / "main"
    revision_candidates = output / "figures" / "revision_candidates"
    supplementary = output / "figures" / "supplementary"
    for path in [
        cache_dir,
        tables_dir,
        reports_dir,
        main_figures,
        revision_candidates,
        supplementary,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(package_root, data_root)
    available = manifest[manifest["analysis_included"]].copy()
    available["system_order"] = available["system"].map({system: index for index, system in enumerate(ANALYSIS_SYSTEMS)})
    available["repeat_order"] = available["replicate"].map({repeat: index for index, repeat in enumerate(REPEATS)})
    available = available.sort_values(["system_order", "repeat_order"])
    if args.max_tasks is not None:
        available = available.head(args.max_tasks)
        manifest = manifest[manifest.set_index(["system", "replicate"]).index.isin(available.set_index(["system", "replicate"]).index)].copy()

    frame_tables: list[pd.DataFrame] = []
    cluster_tables: list[pd.DataFrame] = []
    degree_tables: list[pd.DataFrame] = []
    task_arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for row in available.itertuples(index=False):
        task = task_from_row(pd.Series(row._asdict()))
        frame, cluster, degree, arrays = analyze_task(task, cache_dir, recompute=args.recompute)
        frame_tables.append(frame)
        cluster_tables.append(cluster)
        degree_tables.append(degree)
        task_arrays[(task.system, task.replicate)] = arrays
    framewise = pd.concat(frame_tables, ignore_index=True)
    clusters = pd.concat(cluster_tables, ignore_index=True)
    degrees = pd.concat(degree_tables, ignore_index=True)
    summaries = summarize_tables(framewise, clusters, degrees)
    profiles, maps = summarize_site_arrays(manifest, task_arrays)
    publication = publication_panel_tables(task_arrays)
    dna_only = analyze_dna_only_control(PROJECT_ROOT)
    design = design_table(manifest)
    validation = validation_table(manifest, framewise, clusters)
    audit = forcefield_audit(manifest)

    framewise.to_csv(tables_dir / "balanced_framewise_network_contacts_DNA.csv", index=False)
    clusters.to_csv(tables_dir / "balanced_framewise_clusters.csv", index=False)
    degrees.to_csv(tables_dir / "balanced_framewise_degree_distributions.csv", index=False)
    profiles.to_csv(tables_dir / "balanced_site_profiles_group_summary.csv", index=False)
    maps.to_csv(tables_dir / "balanced_binned_contact_maps_group_summary.csv", index=False)
    dna_only.to_csv(tables_dir / "D1_unaffected_control_sampled.csv", index=False)
    design.to_csv(tables_dir / "system_design.csv", index=False)
    validation.to_csv(tables_dir / "balanced_analysis_validation.csv", index=False)
    audit.to_csv(tables_dir / "balanced_forcefield_exclusion_audit.csv", index=False)
    manifest.drop(columns=["system_order", "repeat_order"], errors="ignore").to_csv(tables_dir / "trajectory_manifest.csv", index=False)
    for name, table in summaries.items():
        table.to_csv(tables_dir / f"balanced_{name}.csv", index=False)
    for name, table in publication.items():
        table.to_csv(
            tables_dir / f"balanced_publication_{name}.csv",
            index=False,
        )

    # Also refresh the long-established paper table names so downstream figure
    # and manuscript tooling cannot silently continue to read stale old-force-
    # field summaries.
    summaries["cluster_repeat"].to_csv(tables_dir / "cluster_by_replicate.csv", index=False)
    summaries["cluster_group"].to_csv(tables_dir / "cluster_group_summary.csv", index=False)
    summaries["block_repeat"].to_csv(tables_dir / "cluster_by_production_block.csv", index=False)
    summaries["block_group"].to_csv(tables_dir / "cluster_block_summary.csv", index=False)
    summaries["frame_repeat"].to_csv(tables_dir / "contacts_by_replicate.csv", index=False)
    summaries["frame_group"].to_csv(tables_dir / "contacts_group_summary.csv", index=False)
    summaries["paired"].to_csv(tables_dir / "DNA_condition_values_by_replicate.csv", index=False)
    summaries["paired_group"].to_csv(tables_dir / "DNA_group_contrasts.csv", index=False)
    profiles.to_csv(tables_dir / "sampled_site_profiles_group_summary.csv", index=False)
    maps.to_csv(tables_dir / "binned_contact_maps_group_summary.csv", index=False)
    snapshot_rows = []
    for system in COMPOSITION_SYSTEMS + DNA_SYSTEMS + ["M50O0D1"]:
        selected = select_snapshot(framewise, system)
        snapshot_rows.append(
            {
                "system": system,
                "replicate": selected["replicate"],
                "frame": int(selected["frame"]),
                "time_us": selected["time_us"],
                "selection_rule": "sampled frame nearest pooled medians of direct LCC and edge density; weak 3-us preference",
            }
        )
    pd.DataFrame(snapshot_rows).to_csv(tables_dir / "balanced_snapshot_selection.csv", index=False)
    contact_prone_rows = []
    for keys, sub in profiles.groupby(["system", "interaction", "side"], observed=True):
        for rank, row in enumerate(sub.nlargest(10, "contact_probability_mean").itertuples(), start=1):
            contact_prone_rows.append(
                {
                    "system": keys[0],
                    "interaction": keys[1],
                    "side": keys[2],
                    "rank": rank,
                    "residue_index": row.residue_index,
                    "contact_probability_mean": row.contact_probability_mean,
                    "contact_probability_sd": row.contact_probability_sd,
                    "n_repeats": row.contact_probability_n,
                }
            )
    pd.DataFrame(contact_prone_rows).to_csv(
        tables_dir / "balanced_contact_prone_regions.csv", index=False
    )

    configure_style()
    dna_repeat = pd.DataFrame()
    dna_group = pd.DataFrame()
    if not args.skip_figures:
        # These historical panel layouts are retained only as revision
        # diagnostics. The current four main figures are rendered by
        # plot_reframed_figures.py.
        plot_figure1(summaries, design, framewise, manifest, revision_candidates)
        plot_figure2(
            summaries,
            publication,
            framewise,
            manifest,
            revision_candidates,
        )
        plot_figure3(summaries, publication, revision_candidates)
        plot_figure4(PROJECT_ROOT, profiles, revision_candidates)
        dna_repeat, dna_group = plot_figure5(
            framewise, dna_only, manifest, revision_candidates
        )
        split_snapshot = plot_chain_continuity_figure(
            summaries,
            framewise,
            clusters,
            manifest,
            revision_candidates,
        )
        plot_supplementary(summaries, design, framewise, clusters, profiles, maps, manifest, supplementary)
        dna_repeat.to_csv(tables_dir / "DNA_conformation_with_D1_by_replicate.csv", index=False)
        dna_group.to_csv(tables_dir / "DNA_conformation_with_D1_group_summary.csv", index=False)
        split_snapshot.to_csv(
            tables_dir / "fig3_chain_continuity_snapshot_selection.csv",
            index=False,
        )
    write_results_report(reports_dir, summaries, design, validation, audit)
    dataset_record = {
        "analysis_version": ANALYSIS_VERSION,
        "dataset_policy": DATASET_POLICY,
        "package_root": str(package_root),
        "data_root": str(data_root),
        "production_frames": list(range(PRODUCTION_START, PRODUCTION_END + 1, STRIDE)),
        "cutoffs_nm": CUTOFFS_NM,
        "primary_cutoff_nm": PRIMARY_CUTOFF_NM,
        "analyzed_tasks": len(available),
        "missing_tasks": manifest.loc[~manifest["available"], ["replicate", "system"]].to_dict("records"),
        "explicitly_included_tasks": manifest.loc[manifest["inclusion_basis"] == "explicit_third_repeat", ["replicate", "system"]].to_dict("records"),
        "excluded_tasks": manifest.loc[manifest["available"] & ~manifest["analysis_included"], ["replicate", "system"]].to_dict("records"),
        "excluded_stability_run": str(package_root / "stability_runs/control_work_M15O85_5fs"),
    }
    (reports_dir / "ANALYSIS_DATASET.json").write_text(json.dumps(dataset_record, indent=2) + "\n")
    (output / "CURRENT_DATASET.md").write_text(
        "# Current paper-analysis dataset\n\n"
        "The canonical base tables were generated by "
        f"`run_balanced_analysis.py` ({ANALYSIS_VERSION}) from the corrected "
        "balanced-design trajectories. Fixed-resolution mechanism tables are "
        "generated by `run_mechanism_extensions.py`; current figures are rendered "
        "by `plot_reframed_figures.py` and `plot_reframed_supplementary.py`. Use "
        "`tables/balanced_*.csv` and `tables/mechanism_*.csv` for new analysis "
        "code. Pre-correction MED1 trajectory tables are not part of the current "
        "dataset.\n"
    )
    print(f"Wrote corrected balanced-design paper analysis to {output}")
    print(f"Analyzed {len(available)} completed tasks; missing: {dataset_record['missing_tasks']}")


if __name__ == "__main__":
    main()
