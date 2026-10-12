"""Graphics-only helpers extracted from the historical driver."""
from __future__ import annotations
from pathlib import Path
import numpy as np

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
