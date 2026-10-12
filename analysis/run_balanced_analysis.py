#!/usr/bin/env python3
"""Paper analysis for the corrected, balanced MED1/OCT4/DNA simulations.

The original paper workflow assumed one 75-nm box and DNA formulations made
by replacing a protein chain.  The corrected design instead keeps protein
residue concentration fixed along the composition series, uses a different
box length for each composition, and appends DNA without removing protein.

This driver deliberately reads the task manifest and per-task molecule map.
It samples 1,000 evenly spaced production frames (1.001--4.997 microseconds),
uses explicit periodic boundary conditions, and caches every trajectory
separately.  Adding a newly completed trajectory therefore invalidates only
that one cache entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
from MDAnalysis.lib.nsgrid import FastNS
from scipy.spatial import cKDTree

from publication_style import COLORS, FONT, AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_SIZE


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PACKAGE_ROOT = Path(os.environ.get("OCT4_MED1_PACKAGE_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
DEFAULT_DATA_ROOT = Path(os.environ.get("OCT4_MED1_DATA_ROOT", PACKAGE_ROOT / "tasks")).expanduser().resolve()
DNA_ONLY_ROOT = Path(os.environ.get("OCT4_MED1_DNA_ONLY_ROOT", PACKAGE_ROOT / "dna_only")).expanduser().resolve()
DEFAULT_OUTPUT = Path(os.environ.get("OCT4_MED1_OUTPUT", HERE / "outputs")).expanduser().resolve()

ANALYSIS_VERSION = "balanced-paper-v3-unified-1000frames"
REPEATS = ["work", "work-1", "work-2"]
PRODUCTION_START = 1001
PRODUCTION_END = 5000
STRIDE = 4
SAMPLED_FRAMES = tuple(range(PRODUCTION_START, PRODUCTION_END + 1, STRIDE))
N_SAMPLED_FRAMES = len(SAMPLED_FRAMES)


def validate_sampling(table: pd.DataFrame, groups: list[str] | None = None) -> None:
    """Reject missing, duplicate, or differently sampled production frames."""
    groups = groups or ["system", "replicate"]
    for key, group in table.groupby(groups, observed=True):
        if tuple(sorted(group["frame"].astype(int))) != SAMPLED_FRAMES:
            raise RuntimeError(f"{key}: expected exactly {N_SAMPLED_FRAMES} frames at {STRIDE}-ns spacing")
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
    """Use the shared publication typography and visible line weights."""
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 8,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": 9.5,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.5,
            "errorbar.capsize": 2.5,
            "lines.markeredgecolor": "black",
            "lines.markeredgewidth": 0.8,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT,
            "mathtext.it": FONT + ":italic",
            "mathtext.bf": FONT + ":bold",
            "mathtext.cal": FONT,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    fig: plt.Figure,
    path: Path,
    target_pixels: tuple[int, int] | None = None,
    target_pdf_points: tuple[float, float] | None = None,
    formats: tuple[str, ...] = ("pdf",),
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
    if "png" in formats:
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
    if "pdf" in formats:
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
    universe = mda.Universe(str(task.path / "start.pdb"), str(task.path / "output.dcd"), topology_format="PDB", format="DCD")
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
            "OCT4_protein_mean_degree",
            "MED1_protein_mean_degree",
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


def analyze_dna_only_control(dna_only_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for repeat in REPEATS:
        path = dna_only_root / repeat / "D1"
        if not (path / "output.dcd").exists():
            raise FileNotFoundError(path / "output.dcd")
        universe = mda.Universe(str(path / "start.pdb"), str(path / "output.dcd"), topology_format="PDB", format="DCD")
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
    table = pd.DataFrame(rows)
    validate_sampling(table)
    return table


def dna_conformation_tables(
    framewise: pd.DataFrame, dna_only: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize DNA shape on the common production-frame grid."""
    metrics = ["DNA_Rg_nm", "DNA_Ree_nm", "DNA_asphericity"]
    columns = ["system", "replicate", "frame", "time_us", *metrics]
    dna_frames = pd.concat(
        [dna_only[columns], framewise.loc[framewise["DNA_Rg_nm"].notna(), columns]],
        ignore_index=True,
    )
    validate_sampling(dna_frames)
    repeat = dna_frames.groupby(["system", "replicate"], observed=True)[metrics].mean().reset_index()
    repeat["n_frames"] = N_SAMPLED_FRAMES
    repeat["first_frame"] = SAMPLED_FRAMES[0]
    repeat["last_frame"] = SAMPLED_FRAMES[-1]
    repeat["sample_interval_ns"] = STRIDE
    return dna_frames, repeat, mean_sd(repeat, ["system"], metrics)


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
            "check": f"{N_SAMPLED_FRAMES} sampled production frames per available trajectory",
            "value": f"min={counts.min()}, max={counts.max()}",
            "required": str(N_SAMPLED_FRAMES),
            "passed": bool(counts.eq(N_SAMPLED_FRAMES).all()),
        }
    )
    cluster_counts = clusters.groupby(["system", "replicate", "cutoff_nm"]).size()
    rows.append(
        {
            "check": f"{N_SAMPLED_FRAMES} cluster frames at every cutoff",
            "value": f"min={cluster_counts.min()}, max={cluster_counts.max()}",
            "required": str(N_SAMPLED_FRAMES),
            "passed": bool(cluster_counts.eq(N_SAMPLED_FRAMES).all()),
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
    validate_sampling(framewise)
    validate_sampling(clusters, ["system", "replicate", "cutoff_nm"])
    validation = pd.DataFrame(rows)
    if not validation["passed"].all():
        raise RuntimeError("Balanced-analysis validation failed:\n" + validation.to_string(index=False))
    return validation


def composition_x(design: pd.DataFrame) -> dict[str, float]:
    return design.set_index("system")["MED1_fraction_sequence_equivalents"].mul(100).to_dict()


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


def minimum_image_nm(delta: np.ndarray, box_nm: float) -> np.ndarray:
    """Return the minimum-image displacement for a task-specific cubic box."""
    return delta - box_nm * np.round(delta / box_nm)


def build_spatial_snapshot(
    task: Task,
    frame: int,
    mode: str = "largest_component",
) -> dict[str, object]:
    """Reconstruct periodic molecular coordinates and the original camera.

    Contact graphs determine PBC placement and foreground membership. The
    returned 3D camera traces extend the established 2D projection without
    changing the selected frame, coordinates, placement, or graph analysis.
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
        topology_format="PDB",
        format="DCD",
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
    # Complete the original 2D camera to a rigid 3D basis. The x/y coordinates
    # are identical to the existing projections used for the d90 overlay.
    camera_basis = np.column_stack([basis, np.cross(basis[:, 0], basis[:, 1])])
    if not np.allclose(camera_basis.T @ camera_basis, np.eye(3), atol=1e-10):
        raise RuntimeError("Snapshot camera must be a rigid orthonormal transform")
    camera_traces = {
        int(node): (whole_coordinates[int(node)] - center) @ camera_basis
        for node in selected_nodes_array
    }
    for node in selected_nodes_array:
        if not np.allclose(camera_traces[int(node)][:, :2], projected_traces[int(node)]):
            raise RuntimeError("3D snapshot camera disagrees with distance-contour projection")
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
        "camera_traces": camera_traces,
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
    show_scale_bar: bool = True,
) -> None:
    """Render all trajectory beads and backbone bonds using headless PyMOL."""
    from publication_snapshots import draw_snapshot

    draw_snapshot(ax, snapshot, half_span,
                  scale_bar_inset_nm=scale_bar_inset_nm,
                  show_scale_bar=show_scale_bar)


def get_task(manifest: pd.DataFrame, system: str, replicate: str) -> Task:
    row = manifest[(manifest["system"] == system) & (manifest["replicate"] == replicate) & manifest["available"]]
    if row.empty:
        raise KeyError(f"No available task for {replicate}/{system}")
    return task_from_row(row.iloc[0])


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dna-only-root", type=Path, default=DNA_ONLY_ROOT)
    parser.add_argument("--recompute", action="store_true", help="Rebuild every per-trajectory cache")
    parser.add_argument("--aggregate-only", action="store_true", help="Require existing per-trajectory caches")
    parser.add_argument("--max-tasks", type=int, help="Developer smoke-test limit; do not use for final analysis")
    args = parser.parse_args()
    if args.aggregate_only and args.recompute:
        parser.error("--aggregate-only and --recompute are mutually exclusive")

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
    for path in [cache_dir, tables_dir]:
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
        if args.aggregate_only and not cache_valid(cache_paths(cache_dir, task), task_signature(task)):
            raise RuntimeError(f"Missing or stale balanced cache for {task.replicate}/{task.system}")
        frame, cluster, degree, arrays = analyze_task(task, cache_dir, recompute=args.recompute)
        frame_tables.append(frame)
        cluster_tables.append(cluster)
        degree_tables.append(degree)
        task_arrays[(task.system, task.replicate)] = arrays
    if not frame_tables:
        raise RuntimeError("No completed trajectories found; check --package-root and --data-root")
    framewise = pd.concat(frame_tables, ignore_index=True)
    # Legacy mean_degree includes the DNA graph node. Explicit protein-only
    # means implement the manuscript definition without changing cached edges.
    n_oct4 = framewise["n_OCT4"]
    n_med1 = framewise["n_MED1_physical_chains"]
    framewise["OCT4_protein_mean_degree"] = (
        framewise["OO_contact_probability"].fillna(0) * (n_oct4 - 1)
        + framewise["OM_contact_probability"].fillna(0) * n_med1
    ).where(n_oct4 > 0)
    framewise["MED1_protein_mean_degree"] = (
        framewise["MM_contact_probability"].fillna(0) * (n_med1 - 1)
        + framewise["OM_contact_probability"].fillna(0) * n_oct4
    ).where(n_med1 > 0)
    degree_sum = (
        framewise["OCT4_protein_mean_degree"].fillna(0) * n_oct4
        + framewise["MED1_protein_mean_degree"].fillna(0) * n_med1
    )
    if not np.allclose(degree_sum, 2 * framewise["n_PP_edges"], atol=1e-8, rtol=0):
        raise RuntimeError("Protein-only partner counts do not reproduce protein edges")
    clusters = pd.concat(cluster_tables, ignore_index=True)
    degrees = pd.concat(degree_tables, ignore_index=True)
    summaries = summarize_tables(framewise, clusters, degrees)
    profiles, maps = summarize_site_arrays(manifest, task_arrays)
    publication = publication_panel_tables(task_arrays)
    dna_only = analyze_dna_only_control(args.dna_only_root.resolve())
    dna_frames, dna_repeat, dna_group = dna_conformation_tables(framewise, dna_only)
    design = design_table(manifest)
    validation = validation_table(manifest, framewise, clusters)
    audit = forcefield_audit(manifest)

    framewise.to_csv(tables_dir / "balanced_framewise_network_contacts_DNA.csv", index=False)
    clusters.to_csv(tables_dir / "balanced_framewise_clusters.csv", index=False)
    degrees.to_csv(tables_dir / "balanced_framewise_degree_distributions.csv", index=False)
    profiles.to_csv(tables_dir / "balanced_site_profiles_group_summary.csv", index=False)
    maps.to_csv(tables_dir / "balanced_binned_contact_maps_group_summary.csv", index=False)
    dna_only.to_csv(tables_dir / "D1_unaffected_control_sampled.csv", index=False)
    dna_frames.to_csv(tables_dir / "DNA_conformation_framewise.csv", index=False)
    dna_repeat.to_csv(tables_dir / "DNA_conformation_with_D1_by_replicate.csv", index=False)
    dna_group.to_csv(tables_dir / "DNA_conformation_with_D1_group_summary.csv", index=False)
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

    print(f"Wrote validated analysis tables for {len(available)} trajectories to {tables_dir}")


if __name__ == "__main__":
    main()
