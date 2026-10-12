"""Cache true 3D, coordinate-preserving PyMOL CG bead-and-bond renders."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import numpy as np

from publication_style import COLORS, DNA_SNAPSHOT_COLOR

HERE = Path(__file__).resolve().parent
RENDER_DIR = Path(os.environ.get("OCT4_MED1_OUTPUT", HERE / "outputs")) / "cache" / "publication_renders"
RENDER_VERSION = "bead-and-bond-v1"
DNA_LAYER_VERSION = "dna-foreground-v1"


def _render_scene(scene, arrays):
    """Reuse the coordinate/camera cache for both opaque and transparent layers."""
    digest = hashlib.sha256(json.dumps(scene, sort_keys=True).encode())
    for xyz in arrays.values():
        digest.update(xyz.tobytes())
    key = "{}_{}_f{}_{}_{}".format(scene["system"], scene["replicate"],
                                     scene["frame"], scene["mode"], digest.hexdigest()[:14])
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    path = RENDER_DIR / (key + ".png")
    if not path.exists():
        scene["output"] = str(path)
        scene_path = RENDER_DIR / (key + ".json")
        scene_path.write_text(json.dumps(scene, indent=2) + "\n")
        np.savez_compressed(scene_path.with_suffix(".npz"), **arrays)
        scene_path.with_suffix(".bin").write_bytes(
            b"".join(xyz.astype("<f4").tobytes() for xyz in arrays.values()))
        pymol = shutil.which("pymol")
        if not pymol:
            raise RuntimeError("PyMOL is required for publication snapshot rendering")
        env = dict(os.environ, PUBLICATION_PYMOL_SCENE=str(scene_path))
        print("Ray rendering", scene["system"], scene["replicate"], scene["frame"],
              scene["version"], flush=True)
        result = subprocess.run([pymol, "-cq", str(HERE / "render_snapshot_pymol.py")],
                                env=env, capture_output=True, text=True, timeout=600)
        if result.returncode or not path.exists():
            raise RuntimeError("PyMOL rendering failed:\n" + result.stdout[-3000:] + result.stderr[-3000:])
    return path


def draw_snapshot(ax, snapshot, half_span, *, scale_bar_inset_nm=2., show_scale_bar=True):
    half_x, half_y = half_span if isinstance(half_span, tuple) else (half_span, half_span)
    mode = snapshot["mode"]
    foreground_nodes = set(snapshot[
        "largest_protein_nodes" if mode == "full_system" else
        "dna_linked_nodes" if mode == "DNA_linked_component" else
        "dna_associated_nodes" if mode == "DNA_associated_assembly" else "nodes"
    ])
    # TYPE_MED1=0, TYPE_OCT4=1, TYPE_DNA=2 in the analysis code.
    # The cached base keeps a neutral DNA underlay for depth and shadows;
    # the separate foreground layer below sets the visible DNA color.
    type_colors = {0: COLORS["MED1"], 1: COLORS["OCT4"], 2: COLORS["DNA"]}
    chains, arrays = [], {}
    for node_raw, type_raw in zip(snapshot["nodes"], snapshot["types"]):
        node, type_id = int(node_raw), int(type_raw)
        dna = type_id == 2
        foreground = node in foreground_nodes or dna
        key = "chain_" + str(node)
        arrays[key] = np.asarray(snapshot["camera_traces"][node], dtype=np.float32)
        chains.append({"key": key, "dna": dna,
                       "n_beads": len(arrays[key]),
                       "color": type_colors[type_id] if foreground else "#E0E0E0",
                       "radius": 0.42 if dna else 0.24,
                       "bucket": ("foreground_" + str(type_id)) if foreground else "context"})
    pixels_y = 1000
    scene = {key: snapshot[key] for key in ("system", "replicate", "frame", "mode")}
    scene.update({"version": RENDER_VERSION, "half_span": [half_x, half_y],
                  "pixels": [int(round(pixels_y * half_x / half_y)), pixels_y], "chains": chains})
    path = _render_scene(scene, arrays)
    ax.imshow(plt.imread(path), extent=(-half_x, half_x, -half_y, half_y),
              origin="upper", interpolation="lanczos", zorder=0)
    dna_chains = [{**chain, "color": DNA_SNAPSHOT_COLOR} for chain in chains if chain["dna"]]
    if dna_chains:
        # Render the same DNA coordinates with the same orthographic camera,
        # then composite above proteins and projected contours. This changes
        # drawing order without moving DNA or altering either strand's topology.
        dna_scene = {key: scene[key] for key in
                     ("system", "replicate", "frame", "mode", "half_span", "pixels")}
        dna_scene.update(version=DNA_LAYER_VERSION, transparent_background=True,
                         chains=dna_chains)
        dna_arrays = {chain["key"]: arrays[chain["key"]] for chain in dna_chains}
        dna_path = _render_scene(dna_scene, dna_arrays)
        # A narrow white outline separates DNA from densely packed beads.
        outline = PatchCollection(
            [Circle((x, y), radius=0.52)
             for xyz in dna_arrays.values() for x, y in xyz[:, :2]],
            facecolor="white", edgecolor="none", zorder=5.4)
        outline.set_gid("snapshot-DNA-outline")
        ax.add_collection(outline)
        dna_image = ax.imshow(plt.imread(dna_path), extent=(-half_x, half_x, -half_y, half_y),
                              origin="upper", interpolation="lanczos", zorder=5.5)
        dna_image.set_gid("snapshot-DNA-foreground")
    ax.set(xlim=(-half_x, half_x), ylim=(-half_y, half_y), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B0B0B0")
        spine.set_linewidth(0.65)
    if show_scale_bar:
        x0, y0 = -half_x + scale_bar_inset_nm, -half_y + scale_bar_inset_nm
        ax.plot([x0, x0 + 10], [y0, y0], color="black", lw=1.5, zorder=7)
        ax.text(x0 + 5, y0 + 1, "10 nm", ha="center", va="bottom", fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5}, zorder=7)
