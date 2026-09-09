"""Graphics-only helpers extracted from the historical driver."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MED1_NATIVE_REGIONS = [
    (77, 232),
    (252, 274),
    (287, 349),
    (361, 518),
]


OCT4_NATIVE_REGIONS = [
    (136, 217),
    (234, 286),
]


AA_CHARGE = {
    "ALA": 0.0,
    "ARG": 1.0,
    "ASN": 0.0,
    "ASP": -1.0,
    "CYS": 0.0,
    "GLN": 0.0,
    "GLU": -1.0,
    "GLY": 0.0,
    "HIS": 0.25,
    "ILE": 0.0,
    "LEU": 0.0,
    "LYS": 1.0,
    "MET": 0.0,
    "PHE": 0.0,
    "PRO": 0.0,
    "SER": 0.0,
    "THR": 0.0,
    "TRP": 0.0,
    "TYR": 0.0,
    "VAL": 0.0,
}


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


def read_pdb_bead_coordinates(
    path: Path,
    atom_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read selected PDB bead coordinates and chain identifiers."""
    coordinates: list[list[float]] = []
    chains: list[str] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip() != atom_name:
                continue
            coordinates.append(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
            )
            chains.append(line[21].strip() or "_")
    if not coordinates:
        raise RuntimeError(f"No {atom_name} beads found in {path}")
    return np.asarray(coordinates, dtype=float) / 10.0, np.asarray(chains)


def project_model_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Use a deterministic principal-plane projection for a schematic view."""
    centered = coordinates - coordinates.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:2].T
    if projected[-1, 0] < projected[0, 0]:
        projected[:, 0] *= -1
    dominant = int(np.argmax(np.abs(projected[:, 1])))
    if projected[dominant, 1] < 0:
        projected[:, 1] *= -1
    return projected


def draw_sequence_model(
    ax: plt.Axes,
    name: str,
    length: int,
    charge: str,
    regions: list[tuple[int, int]],
    region_colors: list[str],
    region_labels: list[str] | None = None,
) -> None:
    """Draw the simulation-defined flexible/native-restraint sequence map."""
    if region_labels is None:
        region_labels = [f"S{i}" for i in range(1, len(regions) + 1)]
    left, right, baseline = 0.045, 0.965, 0.43
    ax.plot(
        [left, right],
        [baseline, baseline],
        transform=ax.transAxes,
        color="#B8BEC7",
        lw=6,
        solid_capstyle="butt",
        zorder=1,
    )
    if len(regions) == 4:
        label_positions = [0.60, 0.75, 0.60, 0.75]
        label_x_offsets = [0.0, -0.012, 0.012, 0.0]
    elif len(regions) == 2:
        # The OCT4 domains are close along the sequence; stagger their
        # annotations so they remain distinct at final two-column width.
        label_positions = [0.61, 0.78]
        label_x_offsets = [-0.008, 0.008]
    else:
        label_positions = [0.64] * len(regions)
        label_x_offsets = [0.0] * len(regions)
    for index, ((start, end), color) in enumerate(
        zip(regions, region_colors),
        start=1,
    ):
        x_start = left + (right - left) * (start - 1) / (length - 1)
        x_end = left + (right - left) * (end - 1) / (length - 1)
        ax.add_patch(
            plt.Rectangle(
                (x_start, baseline - 0.105),
                max(x_end - x_start, 0.008),
                0.21,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
        )
        midpoint = (x_start + x_end) / 2
        ax.annotate(
            region_labels[index - 1],
            xy=(midpoint, baseline + 0.11),
            xycoords=ax.transAxes,
            xytext=(
                midpoint + label_x_offsets[index - 1],
                label_positions[index - 1],
            ),
            textcoords=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color=color,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 0.7,
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )
    ax.text(
        0.0,
        0.94,
        f"{name} ({charge})  ·  {length} aa",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
    )
    ax.text(
        left,
        0.22,
        "1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
    )
    ax.text(
        right,
        0.22,
        str(length),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
    )
    interval_text = "   ·   ".join(
        f"{label}: {start}–{end}"
        for label, (start, end) in zip(region_labels, regions)
    )
    ax.text(
        0.5,
        0.03,
        interval_text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#444444",
    )
    ax.set_axis_off()
