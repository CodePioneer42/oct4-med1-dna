#!/usr/bin/env python3
"""Render supplementary figures for the reframed MED1/OCT4 manuscript.

The script is intentionally read-only with respect to analysis tables and
caches.  It writes only publication figures under ``outputs/figures``.
"""

from __future__ import annotations

import argparse
from functools import wraps
from io import BytesIO
import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.patches import Patch

import run_balanced_analysis as base
from publication_style import CONDITIONS, OCT4_REGIONS, FIG1010_ORIGINAL_SI_STYLE
from run_mechanism_extensions import EXTENSION_VERSION


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = base.DEFAULT_OUTPUT

SPLIT_D0 = ["M10O90", "M10O90-S2", "M10O90-S4"]
SPLIT_D1 = ["M10O90D1", "M10O90D1-S2", "M10O90D1-S4"]
SPLIT_LABELS = ["Full", "Split-2", "Split-4"]
SPLIT_COLORS = ["#1B4965", "#C05A24", "#505050"]

TABLE_S1_SNAPSHOTS = [
    ("D1", "1 DNA"),
    ("M0O100", "OCT4-only"),
    ("M0O100D1", "OCT4-only + DNA"),
    ("M5O95", "5% MED1"),
    ("M10O90", "10% MED1"),
    ("M10O90-S2", "10% MED1 (Split-2)"),
    ("M10O90D1", "10% MED1 + DNA\n(Full)"),
    ("M10O90D1-S2", "10% MED1 + DNA\n(Split-2)"),
    ("M15O85", "15% MED1"),
    ("M20O80", "20% MED1"),
    ("M20O80D1", "20% MED1 + DNA"),
    ("M25O75", "25% MED1"),
    ("M30O70", "30% MED1"),
    ("M50O50", "50% MED1"),
    ("M50O0", "MED1-only"),
    ("M50O0D1", "MED1-only + DNA"),
]

MED1_PLDDT_REGIONS = [
    (77, 232, "#8C360F", "M1"),
    (252, 274, "#B34D18", "M2"),
    (287, 349, "#CD6C2C", "M3"),
    (361, 518, "#E29450", "M4"),
]
MED1_ALPHAFOLD_PDB = base.PACKAGE_ROOT / "inputs" / "MED1-alphafold.pdb"


def original_fig1010_style(function):
    """Keep the typography of the four earlier PDFs in the final figure set."""
    @wraps(function)
    def render(*args, **kwargs):
        from matplotlib.font_manager import FontProperties, findfont
        findfont(FontProperties(family="Arial"), fallback_to_default=False)
        with matplotlib.rc_context(FIG1010_ORIGINAL_SI_STYLE):
            return function(*args, **kwargs)
    return render


def read_table(tables_dir: Path, name: str) -> pd.DataFrame:
    path = tables_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required table is missing: {path}")
    return pd.read_csv(path)


def load_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "balanced_cluster_repeat",
        "balanced_frame_repeat",
        "balanced_framewise_clusters",
        "balanced_snapshot_selection",
        "balanced_site_profiles_group_summary",
        "system_design",
        "trajectory_manifest",
        "mechanism_repeat",
        "mechanism_split_effects_repeat",
        "mechanism_split_did_repeat",
        "mechanism_radial_histogram_group",
        "definition_audit_repeat",
        "definition_audit_mask_transfer_group",
        "definition_audit_mask_decomposition_repeat",
        "definition_audit_initial_radial_comparison",
        "DNA_conformation_with_D1_by_replicate",
        "balanced_framewise_network_contacts_DNA",
        "balanced_degree_group",
        "balanced_binned_contact_maps_group_summary",
        "mechanism_framewise",
        "D1_unaffected_control_sampled",
    ]
    tables = {name: read_table(tables_dir, name) for name in names}
    base.validate_sampling(tables["balanced_framewise_network_contacts_DNA"])
    base.validate_sampling(tables["mechanism_framewise"])
    base.validate_sampling(tables["D1_unaffected_control_sampled"])
    if not tables["DNA_conformation_with_D1_by_replicate"]["n_frames"].eq(base.N_SAMPLED_FRAMES).all():
        raise RuntimeError("DNA conformation tables have inconsistent sampling")
    return tables


def validate_extension_cache(output: Path) -> None:
    cache_dir = output / "cache" / EXTENSION_VERSION
    expected_systems = [
        "M0O100D1",
        "M10O90",
        "M10O90-S2",
        "M10O90-S4",
        "M10O90D1",
        "M10O90D1-S2",
        "M10O90D1-S4",
        "M20O80D1",
        "M50O0D1",
    ]
    missing: list[str] = []
    for replicate in base.REPEATS:
        for system in expected_systems:
            stem = f"{replicate}__{system}"
            meta_path = cache_dir / f"{stem}.json"
            frame_path = cache_dir / f"{stem}.frames.csv"
            arrays_path = cache_dir / f"{stem}.arrays.npz"
            if not (meta_path.exists() and frame_path.exists() and arrays_path.exists()):
                missing.append(stem)
                continue
            metadata = json.loads(meta_path.read_text())
            if metadata.get("extension_version") != EXTENSION_VERSION:
                raise RuntimeError(f"Stale extension cache: {meta_path}")
    if missing:
        raise RuntimeError(
            f"Missing {EXTENSION_VERSION} cache entries: " + ", ".join(missing)
        )


def replicate_offsets(n: int, width: float = 0.035) -> np.ndarray:
    return np.linspace(-width, width, n) if n > 1 else np.zeros(1)


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.11,
    y: float = 1.02,
) -> None:
    """Add only a panel letter; supporting panels deliberately have no titles."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
        zorder=10,
    )


def annotate_structure_snapshot(
    ax: plt.Axes,
    group_label: str,
    d90_nm: float | None = None,
) -> None:
    """Match the upper-left condition annotation used by main Figs. 3 and 4."""
    text = group_label
    if d90_nm is not None:
        text += rf" | $d_{{90}}$={d90_nm:.1f} nm"
    ax.text(
        0.035,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
        zorder=6,
    )






def save_supporting(fig: plt.Figure, path: Path) -> None:
    """Enforce the title-free supporting-figure style before export."""
    titled_axes = [ax.get_title() for ax in fig.axes if ax.get_title()]
    if titled_axes:
        raise RuntimeError(f"Supporting subplot titles are not allowed: {titled_axes}")
    if path.stem in {"figS2_cutoff_sensitivity", "figS3_MED1_partner_composition"}:
        # These final PDFs followed a PNG export. Repeat its layout pass in
        # memory: constrained_layout otherwise stops at a slightly different
        # position on the first PDF-only save.
        with BytesIO() as intermediate:
            fig.savefig(intermediate, format="png", dpi=600,
                        bbox_inches="tight", pad_inches=0.03)
    base.save_figure(fig, path)


def pack_supporting_row(fig: plt.Figure, axes: list[plt.Axes], *, twins=None) -> None:
    """Arrange supporting panels in one row with equal plotting widths."""
    from plot_reframed_figures import (
        _axes_content_bbox, apply_reference_typography, pack_figure_columns, pack_figure_rows,
    )
    from matplotlib.transforms import Bbox
    twins = twins or {}
    groups = [[ax, *twins.get(index, [])] for index, ax in enumerate(axes)]
    for ax in axes:
        for text in ax.texts:
            if text.get_text() in {"A", "B", "C"}:
                text.set_gid("panel-letter")
                text.set_y(1.0)
                text.set_verticalalignment("top")
    apply_reference_typography(fig)
    pack_figure_columns(fig, groups, [1] * len(axes), gap_pt=12)
    # Put each panel letter beside the top of its plotting area. Use the
    # visible label extent so the letters cannot overlap the top y tick.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for group in groups:
        box = Bbox.union([_axes_content_bbox(ax, renderer) for ax in group])
        left = box.transformed(fig.transFigure.inverted()).x0 - 12 / (72 * fig.get_figwidth())
        position = group[0].get_position()
        for text in group[0].texts:
            if text.get_gid() == "panel-letter":
                text.set_x((left - position.x0) / position.width)
    pack_figure_rows(fig, [[ax for group in groups for ax in group]], resize=True)


@original_fig1010_style
def plot_med1_plddt(out: Path) -> None:
    """S1: AlphaFold pLDDT profile and the four MED1 restraint regions."""
    if not MED1_ALPHAFOLD_PDB.exists():
        raise FileNotFoundError(
            f"Original MED1 AlphaFold PDB is missing: {MED1_ALPHAFOLD_PDB}"
        )

    residue_plddt: dict[int, float] = {}
    for line in MED1_ALPHAFOLD_PDB.read_text().splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        residue = int(line[22:26])
        residue_plddt.setdefault(residue, float(line[60:66]))
    positions = np.arange(1, base.MED1_LENGTH + 1)
    scores = np.array(
        [residue_plddt.get(int(position), np.nan) for position in positions]
    )
    if len(residue_plddt) != base.MED1_LENGTH or not np.isfinite(scores).all():
        raise RuntimeError(
            f"Expected pLDDT for all {base.MED1_LENGTH} MED1 residues; "
            f"found {len(residue_plddt)} CA records"
        )

    fig, ax = plt.subplots(figsize=(7.15, 3.0), constrained_layout=True)
    for start, stop, color, _ in MED1_PLDDT_REGIONS:
        ax.axvspan(start - 0.5, stop + 0.5, color=color, alpha=0.20, lw=0, zorder=0)
    ax.plot(positions, scores, color="#333333", lw=0.62, zorder=2)
    ax.axhline(70, color="#666666", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set(
        xlabel="MED1 residue",
        ylabel="pLDDT",
        xlim=(0, base.MED1_LENGTH + 1),
        ylim=(0, 100),
        xticks=np.arange(0, base.MED1_LENGTH + 1, 200),
    )
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75, zorder=0)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#333333", lw=0.8, label="AlphaFold pLDDT"),
            Line2D([0], [0], color="#666666", lw=0.8, ls=(0, (3, 2)), label="pLDDT = 70"),
            *[
                Patch(
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.35,
                    label=f"{name} ({start}–{stop})",
                )
                for start, stop, color, name in MED1_PLDDT_REGIONS
            ],
        ],
        loc="upper right",
        ncols=1,
        fontsize=6.5,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        handlelength=1.45,
        columnspacing=1.0,
        handletextpad=0.55,
    )
    save_supporting(fig, out / "figS1_MED1_pLDDT")


def read_protein_charges() -> dict[str, np.ndarray]:
    """Read S8 charges from the production system and verify its input sequences."""
    task = base.DEFAULT_DATA_ROOT / "work" / "M10O90D1"
    system = ET.parse(task / "system.xml").getroot()
    forces = [
        force
        for force in system.find("Forces")
        if force.get("type") == "CustomNonbondedForce"
        and [p.get("name") for p in force.find("PerParticleParameters")] == ["q"]
    ]
    if len(forces) != 1:
        raise RuntimeError("Expected one production nonbonded charge array")
    all_charges = np.array([float(p.get("param1")) for p in forces[0].find("Particles")])
    molecules = pd.read_csv(task / "molecule_map.tsv", sep="\t")
    charge_rules = {"ARG": 1.0, "LYS": 1.0, "ASP": -1.0, "GLU": -1.0, "HIS": 0.25}
    proteins = {}
    for name, filename, length, net in [
        ("OCT4", "OCT4.pdb", base.OCT4_LENGTH, -3.75),
        ("MED1", "MED1-alphafold.pdb", base.MED1_LENGTH, 31.0),
    ]:
        pdb = base.PACKAGE_ROOT / "inputs" / filename
        residues = [
            line[17:20].strip()
            for line in pdb.read_text().splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"
        ]
        if len(residues) != length:
            raise RuntimeError(f"Expected {length} input residues for {name}")
        expected = np.array([charge_rules.get(residue, 0.0) for residue in residues])
        copies = molecules[molecules["molecule_type"] == name]
        if copies.empty:
            raise RuntimeError(f"No {name} molecules in {task}")
        for row in copies.itertuples(index=False):
            charges = all_charges[int(row.particle_start):int(row.particle_stop_exclusive)]
            if not np.array_equal(charges, expected):
                raise RuntimeError(f"Production and sequence charges differ for {row.molecule_id}")
        if not np.isclose(charges.sum(), net):
            raise RuntimeError(f"Unexpected net charge for {name}: {charges.sum()}")
        proteins[name] = charges.copy()
    return proteins


@original_fig1010_style
def plot_protein_charge_distribution(out: Path) -> None:
    """S8: per-residue charges and identically smoothed sequence profiles."""
    profiles = read_protein_charges()
    positive, negative, neutral, histidine = "#397BB5", "#C85A55", "#E9E9E9", "#ADCDE3"
    cmap = ListedColormap([negative, neutral, histidine, positive])
    norm = BoundaryNorm([-1.5, -0.5, 0.125, 0.625, 1.5], cmap.N)
    regions = {
        "OCT4": [
            (136, 217, OCT4_REGIONS[0], "POU-specific"),
            (234, 286, OCT4_REGIONS[1], "POU-homeodomain"),
        ],
        "MED1": MED1_PLDDT_REGIONS,
    }
    fig = plt.figure(figsize=(7.15, 4.8))
    grid = fig.add_gridspec(2, 1, left=0.115, right=0.985, top=0.91, bottom=0.14, hspace=0.68)
    for index, (protein, charges) in enumerate(profiles.items()):
        subgrid = grid[index].subgridspec(2, 1, height_ratios=[1, 0.105], hspace=0.09)
        ax = fig.add_subplot(subgrid[0])
        strip = fig.add_subplot(subgrid[1], sharex=ax)
        positions = np.arange(1, len(charges) + 1)
        kernel = np.ones(21)
        window_mean = np.convolve(charges, kernel, mode="same") / np.convolve(
            np.ones(len(charges)), kernel, mode="same"
        )
        for start, stop, color, name in regions[protein]:
            ax.axvspan(start - 0.5, stop + 0.5, color=color, alpha=0.20, lw=0, zorder=0)
            ax.text((start + stop) / 2, 0.455, name, ha="center", va="center", fontsize=6.5, color="#555555")
        ax.axhline(0, color="#777777", lw=0.55, zorder=1)
        ax.fill_between(positions, 0, window_mean, where=window_mean >= 0, interpolate=True, color=positive, alpha=0.30, lw=0)
        ax.fill_between(positions, 0, window_mean, where=window_mean <= 0, interpolate=True, color=negative, alpha=0.30, lw=0)
        ax.plot(positions, window_mean, color="#333333", lw=0.9, zorder=3)
        ax.set(ylabel="Mean charge\n(e / residue)", ylim=(-0.5, 0.52), yticks=[-0.4, -0.2, 0, 0.2, 0.4])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)
        ax.text(0, 1.11, f"{protein}  ·  {len(charges):,} aa  ·  net charge {charges.sum():+g} e".replace("-", "−"), transform=ax.transAxes, fontsize=8.5, fontweight="bold")
        panel_label(ax, "AB"[index], x=-0.115, y=1.07)
        strip.imshow(charges[np.newaxis, :], cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", extent=(0.5, len(charges) + 0.5, 0, 1))
        ticks = [1, 60, 120, 180, 240, 300, 360] if protein == "OCT4" else [1, 300, 600, 900, 1200, 1581]
        strip.set(xlim=(0.5, len(charges) + 0.5), yticks=[], xticks=ticks, xlabel=f"{protein} residue")
        strip.set_ylabel("Residue\ncharge", rotation=0, ha="right", va="center", labelpad=8, fontsize=6.5)
        for spine in strip.spines.values():
            spine.set_visible(False)
        strip.tick_params(axis="x", length=2.5, pad=2)
    fig.legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label) for color, label in [
            (negative, "Asp/Glu: −1 e"), (neutral, "Uncharged: 0"),
            (histidine, "His: +0.25 e"), (positive, "Lys/Arg: +1 e"),
        ]],
        loc="lower center", bbox_to_anchor=(0.55, 0.005), ncols=4,
        fontsize=7, handlelength=1.2, columnspacing=1.6,
    )
    save_supporting(fig, out / "figS8_protein_charge_distribution")




def draw_fig1c_axis_break(ax: plt.Axes, x: float = 56.0) -> None:
    """Use the separated MED1-only endpoint mark from main Figure 1C."""
    transform = ax.get_xaxis_transform()
    ax.plot(
        [x - 1.45, x + 1.45],
        [0, 0],
        transform=transform,
        color="white",
        linewidth=2.8,
        solid_capstyle="butt",
        clip_on=False,
        zorder=9,
    )
    for offset in (-0.65, 0.65):
        ax.plot(
            [x + offset - 0.28, x + offset + 0.28],
            [-0.018, 0.018],
            transform=transform,
            color="black",
            linewidth=1.0,
            clip_on=False,
            zorder=10,
        )


def draw_y_axis_break_between(top_ax: plt.Axes, bottom_ax: plt.Axes) -> None:
    """Mark the omitted interval between two vertically stacked y ranges."""
    for ax, y in ((top_ax, 0.0), (bottom_ax, 1.0)):
        transform = ax.transAxes
        ax.plot(
            [-0.014, 0.014],
            [y - 0.012, y + 0.012],
            transform=transform,
            color="black",
            linewidth=1.0,
            clip_on=False,
            zorder=10,
        )


def set_fig1c_composition_axis(ax: plt.Axes) -> None:
    """Match the composition ticks, labels, limits, and break from Figure 1C."""
    ax.set(
        xlabel="MED1 molecule fraction (%)",
        xlim=(-2, 68),
    )
    ax.set_xticks(
        [0, 10, 20, 30, 40, 50, 62],
        ["OCT4-\nonly", "10", "20", "30", "40", "50", "MED1-\nonly"],
    )
    ax.tick_params(axis="x", labelsize=7.0, pad=2)
    ax.xaxis.label.set_size(8.8)
    draw_fig1c_axis_break(ax)




def draw_dna_only_snapshot(ax: plt.Axes, selected: pd.Series) -> None:
    """Draw the standalone D1 control, which has no protein task manifest row."""
    box_nm = 75.0
    task_path = base.DNA_ONLY_ROOT / str(selected["replicate"]) / "D1"
    universe = base.mda.Universe(
        str(task_path / "start.pdb"),
        str(task_path / "output.dcd"),
        topology_format="PDB",
        format="DCD",
    )
    stored_frame = base.trajectory_frame_indices(
        task_path, len(universe.trajectory), [int(selected["frame"])]
    )[0]
    universe.trajectory[stored_frame]
    coordinates_nm = universe.atoms.positions.astype(np.float64) / 10.0
    whole = base.unwrap_dna(coordinates_nm, box_nm)
    centered = whole - whole.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ basis[:2].T
    from publication_snapshots import draw_snapshot

    camera_basis = np.column_stack([basis[:2].T, np.cross(basis[0], basis[1])])
    draw_snapshot(ax, {
        "system": "D1", "replicate": str(selected["replicate"]),
        "frame": int(selected["frame"]), "mode": "full_system",
        "nodes": [0], "types": [base.TYPE_DNA], "largest_protein_nodes": set(),
        "camera_traces": {0: centered @ camera_basis},
    }, box_nm / 2)


def plot_composition_diagnostics(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S1: show a representative snapshot for every Table S1 system."""
    fig = plt.figure(figsize=(7.20, 7.20), constrained_layout=False)
    outer = fig.add_gridspec(
        4,
        4,
        left=0.025,
        right=0.992,
        bottom=0.025,
        top=0.975,
        wspace=0.08,
        hspace=0.14,
    )
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    manifest = tables["trajectory_manifest"]
    dna_only_rows = tables["D1_unaffected_control_sampled"]
    dna_only_rows = dna_only_rows[dna_only_rows["system"] == "D1"].copy()
    dna_only_target = float(dna_only_rows["DNA_Rg_nm"].median())
    dna_only_rows["selection_score"] = (
        np.abs(dna_only_rows["DNA_Rg_nm"] - dna_only_target)
        + 0.01 * np.abs(dna_only_rows["time_us"] - 3.0)
    )
    dna_only_selection = dna_only_rows.sort_values("selection_score").iloc[0]

    for index, (system, label) in enumerate(TABLE_S1_SNAPSHOTS):
        row, column = divmod(index, 4)
        ax = fig.add_subplot(outer[row, column])
        if system == "D1":
            draw_dna_only_snapshot(ax, dna_only_selection)
        else:
            selected = base.select_snapshot(framewise, system)
            task = base.get_task(manifest, system, str(selected["replicate"]))
            snapshot = base.build_spatial_snapshot(
                task,
                int(selected["frame"]),
                mode="full_system",
            )
            base.draw_spatial_snapshot(ax, snapshot, task.box_nm / 2)
        ax.set_title("")
        annotate_structure_snapshot(ax, label)
        panel_label(ax, chr(ord("A") + index), x=-0.035, y=1.02)
    save_supporting(fig, out / "figS4_composition_snapshots")


@original_fig1010_style
def plot_cutoff_sensitivity(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S2: composition trend at three contact cutoffs."""
    repeat = tables["balanced_cluster_repeat"]
    design = tables["system_design"]
    x_map = base.composition_x(design)
    x_map["M50O0"] = 62.0
    systems = list(base.COMPOSITION_SYSTEMS)
    x = np.asarray([x_map[system] for system in systems])
    fig = plt.figure(figsize=(4.8, 3.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(0.42, 1.0), hspace=0.04)
    upper_ax = fig.add_subplot(grid[0])
    lower_ax = fig.add_subplot(grid[1], sharex=upper_ax)
    for cutoff, color, marker in [
        (1.0, base.COLORS["OCT4"], "o"),
        (1.2, base.COLORS["mixed"], "s"),
        (1.4, base.COLORS["MED1"], "^"),
    ]:
        subset = repeat[
            np.isclose(repeat["cutoff_nm"], cutoff)
            & repeat["system"].isin([*systems, "M50O0"])
        ]
        means: list[float] = []
        sds: list[float] = []
        for index, system in enumerate(systems):
            values = (
                subset[subset["system"] == system]
                .set_index("replicate")
                .reindex(base.REPEATS)["largest_cluster_fraction"]
                .to_numpy(float)
            )
            if not np.isfinite(values).all():
                raise RuntimeError(f"Incomplete cutoff values for {system}/{cutoff}")
            means.append(float(values.mean()))
            sds.append(float(values.std(ddof=1)))
            lower_ax.scatter(
                x[index] + replicate_offsets(3, 0.24),
                values,
                s=17,
                marker=marker,
                facecolor="white",
                edgecolor=color,
                linewidth=0.7,
                zorder=6,
            )
        lower_ax.plot(x, means, color=color, lw=1.15, zorder=1)
        lower_ax.errorbar(
            x,
            means,
            yerr=sds,
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.4,
            zorder=2,
        )
        lower_ax.scatter(
            x,
            means,
            s=26,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )

        med_only = (
            subset[subset["system"] == "M50O0"]
            .set_index("replicate")
            .reindex(base.REPEATS)["largest_cluster_fraction"]
            .to_numpy(float)
        )
        if not np.isfinite(med_only).all():
            raise RuntimeError(f"Incomplete cutoff values for M50O0/{cutoff}")
        upper_ax.scatter(
            x_map["M50O0"] + replicate_offsets(3, 0.24),
            med_only,
            s=17,
            marker=marker,
            facecolor="white",
            edgecolor=color,
            linewidth=0.7,
            zorder=6,
        )
        upper_mean = float(med_only.mean())
        upper_sd = float(med_only.std(ddof=1))
        upper_ax.errorbar(
            [x_map["M50O0"]],
            [upper_mean],
            yerr=[upper_sd],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.4,
            zorder=2,
        )
        upper_ax.scatter(
            [x_map["M50O0"]],
            [upper_mean],
            s=26,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=f"{cutoff:.1f} nm",
            zorder=5,
        )

    lower_ax.set(
        ylabel="Direct-protein LCC fraction",
        ylim=(0.0, 0.42),
        yticks=[0.0, 0.1, 0.2, 0.3, 0.4],
    )
    upper_ax.set(ylim=(0.95, 1.01), yticks=[0.95, 1.0])
    upper_ax.tick_params(axis="x", labelbottom=False, bottom=False)
    upper_ax.spines["bottom"].set_visible(False)
    lower_ax.spines["top"].set_visible(False)
    upper_ax.tick_params(axis="y", labelsize=7.5, pad=2)
    lower_ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75, zorder=0)
    set_fig1c_composition_axis(lower_ax)
    draw_y_axis_break_between(upper_ax, lower_ax)
    upper_ax.legend(loc="upper left", frameon=False)
    save_supporting(fig, out / "figS2_cutoff_sensitivity")


@original_fig1010_style
def plot_med1_partner_composition(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S3: species-specific partner counts per MED1 and OCT4 chain."""
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    design = tables["system_design"]
    med1_systems = [*base.COMPOSITION_SYSTEMS[1:], "M50O0"]
    oct4_systems = list(base.COMPOSITION_SYSTEMS)
    systems = list(dict.fromkeys([*med1_systems, *oct4_systems]))
    composition = framewise[framewise["system"].isin(systems)].copy()
    # The MED1-only control has no OCT4 partners by definition; its O--M
    # contact probability is undefined because there are no possible pairs.
    composition["MED1_OCT4_partners"] = np.where(
        composition["n_OCT4"].eq(0),
        0.0,
        composition["OM_contact_probability"] * composition["n_OCT4"],
    )
    composition["MED1_MED1_partners"] = composition[
        "MM_contact_probability"
    ] * np.maximum(composition["n_MED1_sequence_equivalents"] - 1, 0)
    composition["OCT4_OCT4_partners"] = composition[
        "OO_contact_probability"
    ] * np.maximum(composition["n_OCT4"] - 1, 0)
    composition["OCT4_MED1_partners"] = np.where(
        composition["n_MED1_sequence_equivalents"].eq(0),
        0.0,
        composition["OM_contact_probability"]
        * composition["n_MED1_sequence_equivalents"],
    )
    repeat = (
        composition.groupby(["system", "replicate"], as_index=False)[
            [
                "MED1_OCT4_partners",
                "MED1_MED1_partners",
                "OCT4_OCT4_partners",
                "OCT4_MED1_partners",
            ]
        ]
        .mean()
    )
    x_map = base.composition_x(design)
    x_map["M50O0"] = 62.0
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.5), constrained_layout=True
    )
    panel_specs = [
        (
            axes[0],
            med1_systems,
            "Partners per MED1 chain",
            [
                ("MED1_OCT4_partners", "OCT4 partners", base.COLORS["OCT4"], "o"),
                ("MED1_MED1_partners", "MED1 partners", base.COLORS["MED1"], "s"),
            ],
            "A",
        ),
        (
            axes[1],
            oct4_systems,
            "Partners per OCT4 chain",
            [
                ("OCT4_OCT4_partners", "OCT4 partners", base.COLORS["OCT4"], "o"),
                ("OCT4_MED1_partners", "MED1 partners", base.COLORS["MED1"], "s"),
            ],
            "B",
        ),
    ]
    for ax, panel_systems, ylabel, metric_specs, panel in panel_specs:
        x = np.asarray([x_map[system] for system in panel_systems])
        for metric, label, color, marker in metric_specs:
            means: list[float] = []
            sds: list[float] = []
            replicate_values: list[np.ndarray] = []
            for system in panel_systems:
                values = (
                    repeat[repeat["system"] == system]
                    .set_index("replicate")
                    .reindex(base.REPEATS)[metric]
                    .to_numpy(float)
                )
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        f"Incomplete partner values for {system}/{metric}"
                    )
                replicate_values.append(values)
                means.append(float(values.mean()))
                sds.append(float(values.std(ddof=1)))
            means_array = np.asarray(means)
            sds_array = np.asarray(sds)
            ax.plot(x, means_array, color=color, lw=1.25, zorder=1)
            ax.errorbar(
                x,
                means_array,
                yerr=sds_array,
                fmt="none",
                ecolor=color,
                elinewidth=0.8,
                capsize=2.4,
                zorder=2,
            )
            ax.scatter(
                x,
                means_array,
                s=24,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.55,
                label=label,
                zorder=4,
            )
            for system_index, values in enumerate(replicate_values):
                finite = np.isfinite(values)
                if finite.any():
                    ax.scatter(
                        x[system_index] + replicate_offsets(int(finite.sum()), 0.24),
                        values[finite],
                        s=14,
                        marker=marker,
                        facecolor="white",
                        edgecolor=color,
                        linewidth=0.65,
                        zorder=5,
                    )
        ax.set(ylabel=ylabel, ylim=(0.0, 4.15), yticks=[0, 1, 2, 3, 4])
        ax.yaxis.label.set_size(8.8)
        set_fig1c_composition_axis(ax)
        ax.legend(loc="upper left", fontsize=6.7, frameon=False)
        panel_label(ax, panel, x=-0.15)
    save_supporting(fig, out / "figS3_MED1_partner_composition")


def plot_dna_neighborhood_composition(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S6: species-normalized DNA occupancy and DNA-linked composition."""
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    design = tables["system_design"]
    frame_repeat = (
        framewise[framewise["system"].isin(base.DNA_SYSTEMS)]
        .groupby(["system", "replicate"], as_index=False)
        .mean(numeric_only=True)
    )
    frame_group = frame_repeat.groupby("system").mean(numeric_only=True)
    design_index = design.set_index("system")
    x = np.arange(len(base.DNA_SYSTEMS), dtype=float)
    labels = [
        "OCT4-\nonly",
        "10%\nMED1",
        "20%\nMED1",
        "MED1-\nonly",
    ]
    n_proteins = np.asarray(
        [
            design_index.loc[system, "n_physical_protein_chains"]
            for system in base.DNA_SYSTEMS
        ],
        dtype=float,
    )
    fig = plt.figure(figsize=(5.20, 2.65), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, left=0.08, right=0.975, bottom=0.21, top=0.93,
                           wspace=0.55)
    axes = [fig.add_subplot(grid[0, column]) for column in range(2)]

    # Retain B's complete DNA-linked component, including indirect contacts.
    ax = axes[1]
    oct4_metric, med1_metric = "DNA_component_OCT4", "DNA_component_MED1"
    oct4 = np.asarray(
        [frame_group.loc[system, oct4_metric] for system in base.DNA_SYSTEMS]
    ) / n_proteins
    med1 = np.asarray(
        [frame_group.loc[system, med1_metric] for system in base.DNA_SYSTEMS]
    ) / n_proteins
    ax.bar(
        x, oct4, color=base.COLORS["OCT4"], edgecolor="black",
        linewidth=0.65, label="OCT4", zorder=1,
    )
    ax.bar(
        x, med1, bottom=oct4, color=base.COLORS["MED1"], edgecolor="black",
        linewidth=0.65, label="MED1", zorder=1,
    )
    for system_index, system in enumerate(base.DNA_SYSTEMS):
        values = frame_repeat[frame_repeat["system"] == system]
        totals = (
            values[oct4_metric].to_numpy(float)
            + values[med1_metric].to_numpy(float)
        ) / n_proteins[system_index]
        ax.scatter(
            x[system_index] + replicate_offsets(len(totals), 0.055),
            totals, s=19, facecolor="white", edgecolor="#222222",
            linewidth=0.65, zorder=6,
        )
    ax.set(ylabel="DNA-linked chains /\nall protein chains", ylim=(0, 1.04))
    ax.set(xlim=(-0.55, len(x) - 0.45), xlabel="")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=5.5, pad=2)
    panel_label(ax, "B", x=-0.14, y=1.04)
    legend_options = {
        "loc": "upper left",
        "bbox_to_anchor": (0.01, 0.99),
        "borderaxespad": 0.15,
        "fontsize": 7.0,
        "labelspacing": 0.7,
        "handlelength": 1.1,
        "handletextpad": 0.6,
        "frameon": False,
    }
    axes[1].legend(
        handles=[
            Patch(
                facecolor=base.COLORS["OCT4"],
                label="OCT4",
            ),
            Patch(
                facecolor=base.COLORS["MED1"],
                label="MED1",
            ),
        ],
        **legend_options,
    )

    # Former C replaces A: direct contacts normalized by each species total.
    ax = axes[0]
    for offset, metric, label, color, marker in [
        (
            -0.19,
            "OCT4_fraction_contacting_DNA",
            "OCT4",
            base.COLORS["OCT4"],
            "o",
        ),
        (
            0.19,
            "MED1_fraction_contacting_DNA",
            "MED1",
            base.COLORS["MED1"],
            "s",
        ),
    ]:
        means: list[float] = []
        sds: list[float] = []
        per_system: list[np.ndarray] = []
        for system in base.DNA_SYSTEMS:
            values = frame_repeat.loc[
                frame_repeat["system"] == system, metric
            ].dropna().to_numpy(float)
            per_system.append(values)
            means.append(float(values.mean()) if len(values) else np.nan)
            sds.append(float(values.std(ddof=1)) if len(values) > 1 else np.nan)
        finite = np.isfinite(means)
        positions = x + offset
        # Fractions for an absent protein species remain undefined: leave
        # that bar slot empty instead of turning an undefined fraction into 0.
        ax.bar(positions[finite], np.asarray(means)[finite], width=0.34,
               color=color, edgecolor="black", linewidth=0.65, label=label, zorder=1)
        ax.errorbar(
            positions[finite],
            np.asarray(means)[finite],
            yerr=np.asarray(sds)[finite],
            fmt="none",
            ecolor="black",
            elinewidth=0.78,
            capsize=2.2,
            zorder=2,
        )
        for system_index, values in enumerate(per_system):
            if len(values):
                ax.scatter(
                    positions[system_index] + replicate_offsets(len(values), 0.045),
                    values,
                    s=20,
                    facecolor="white",
                    edgecolor="black",
                    marker=marker,
                    linewidth=0.75,
                    zorder=6,
                )
    ax.set(ylabel="Fraction contacting DNA", ylim=(0, 0.60))
    ax.set(xlim=(-0.55, len(x) - 0.45), xlabel="")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=5.5, pad=2)
    ax.legend(**legend_options)
    panel_label(ax, "A", x=-0.14, y=1.04)
    pack_supporting_row(fig, axes)
    save_supporting(fig, out / "figS6_DNA_neighborhood_composition")


def plot_contact_type_and_residue_profiles(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S5: contact heatmaps, O--M profiles, and protein--DNA profiles."""
    frame_repeat = tables["balanced_frame_repeat"].set_index(
        ["system", "replicate"]
    )
    profiles = tables["balanced_site_profiles_group_summary"].copy()
    system_design = tables["system_design"].set_index("system")
    med1_rows = profiles["side"].eq("MED1")
    fragment_counts = profiles.loc[med1_rows, "system"].map(
        system_design["med1_fragments"]
    )
    if fragment_counts.isna().any():
        raise RuntimeError("Missing MED1 fragment counts for S5 residue profiles")
    # The cached profiles use all physical chains as the denominator. Each
    # original MED1 residue occurs in exactly one fragment per sequence copy,
    # so multiplying by the fragment count gives the actual residue probability.
    for column in ["contact_probability_mean", "contact_probability_sd"]:
        profiles.loc[med1_rows, column] *= fragment_counts.to_numpy(float)
    contact_metrics = [
        "OO_contact_probability",
        "OM_contact_probability",
        "MM_contact_probability",
    ]
    contact_labels = ["O–O", "O–M", "M–M"]

    composition_contacts = [
        ("M5O95", "5% MED1"),
        ("M10O90", "10% MED1"),
        ("M15O85", "15% MED1"),
        ("M20O80", "20% MED1"),
        ("M25O75", "25% MED1"),
        ("M30O70", "30% MED1"),
        ("M50O50", "50% MED1"),
    ]
    dna_contacts = [
        ("M10O90", "10% MED1"),
        ("M10O90D1", "10% MED1\n+DNA"),
        ("M20O80", "20% MED1"),
        ("M20O80D1", "20% MED1\n+DNA"),
    ]
    split2_contacts = [
        ("M10O90", "10% MED1\n−DNA"),
        ("M10O90-S2", "Split-2\n−DNA"),
        ("M10O90D1", "10% MED1\n+DNA"),
        ("M10O90D1-S2", "Split-2\n+DNA"),
    ]
    contact_groups = [composition_contacts, dna_contacts, split2_contacts]

    def contact_matrix(systems: list[tuple[str, str]]) -> np.ndarray:
        matrix = np.full((len(systems), len(contact_metrics)), np.nan)
        for row, (system, _) in enumerate(systems):
            for column, metric in enumerate(contact_metrics):
                values = (
                    frame_repeat.xs(system, level="system")
                    .reindex(base.REPEATS)[metric]
                    .to_numpy(float)
                )
                if np.isfinite(values).any():
                    matrix[row, column] = float(np.nanmean(values))
        return matrix

    contact_matrices = [contact_matrix(systems) for systems in contact_groups]
    maximum = max(float(np.nanmax(matrix)) for matrix in contact_matrices)
    contact_cmap = plt.get_cmap("YlGnBu").copy()
    contact_cmap.set_bad("#E6E6E6")

    original_height = 6.45
    dna_profile_height = 2.05
    total_height = original_height + dna_profile_height
    fig = plt.figure(figsize=(7.2, total_height), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.00, 1.75],
        left=0.075,
        right=0.985,
        bottom=(0.075 * original_height + dna_profile_height) / total_height,
        top=(0.975 * original_height + dna_profile_height) / total_height,
        hspace=0.17,
    )
    contact_grid = outer[0, 0].subgridspec(
        2,
        3,
        height_ratios=[1.0, 0.075],
        wspace=0.46,
        hspace=0.20,
    )
    contact_axes = [
        fig.add_subplot(contact_grid[0, column]) for column in range(3)
    ]
    image: plt.AxesImage | None = None
    for column, (ax, matrix, systems) in enumerate(
        zip(contact_axes, contact_matrices, contact_groups)
    ):
        image = ax.imshow(
            matrix,
            cmap=contact_cmap,
            vmin=0,
            vmax=maximum,
            aspect="auto",
            interpolation="nearest",
            zorder=1,
        )
        image.set_rasterized(True)
        ax.set_xticks(range(len(contact_labels)), contact_labels)
        ax.set_yticks(
            range(len(systems)),
            [label for _, label in systems],
        )
        ax.tick_params(axis="x", labelsize=6.6, pad=2)
        ax.tick_params(axis="y", labelsize=5.8, pad=2)
        for row, metric_column in np.ndindex(matrix.shape):
            value = matrix[row, metric_column]
            if not np.isfinite(value):
                label, color = "N/A", "#666666"
            else:
                rgba = contact_cmap(image.norm(value))
                luminance = (
                    0.2126 * rgba[0]
                    + 0.7152 * rgba[1]
                    + 0.0722 * rgba[2]
                )
                label = f"{value:.3f}"
                color = "white" if luminance < 0.47 else "black"
            ax.text(
                metric_column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
                clip_on=True,
                zorder=3,
            )
        panel_label(ax, chr(ord("A") + column), x=-0.27, y=1.01)

    if image is None:
        raise RuntimeError("No contact heatmap was generated")
    colorbar_ax = fig.add_subplot(contact_grid[1, 1])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=5.8, pad=1)
    colorbar.ax.text(
        -0.08,
        0.5,
        "Mean pair-contact\nprobability",
        transform=colorbar.ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.7,
        clip_on=False,
    )

    def profile_panel(
        ax: plt.Axes,
        interaction: str,
        side: str,
        systems: list[tuple[str, str, object, str]],
        panel: str,
        *,
        band_alpha: float,
        show_ylabel: bool,
        legend_columns: int | None = None,
        panel_x: float = -0.18,
    ) -> None:
        for system, label, color, linestyle in systems:
            subset = profiles[
                (profiles["system"] == system)
                & (profiles["interaction"] == interaction)
                & (profiles["side"] == side)
            ].sort_values("residue_index")
            if subset.empty:
                raise RuntimeError(
                    f"Missing residue profile for {system}/{interaction}/{side}"
                )
            residues = subset["residue_index"].to_numpy(float)
            mean = subset["contact_probability_mean"].to_numpy(float)
            sd = subset["contact_probability_sd"].to_numpy(float)
            ax.fill_between(
                residues,
                np.maximum(0, mean - sd),
                mean + sd,
                color=color,
                alpha=band_alpha,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                residues,
                mean,
                color=color,
                lw=1.05,
                ls=linestyle,
                label=label,
                zorder=2,
            )
        ax.set_xlabel(f"{side} residue")
        if show_ylabel:
            ax.set_ylabel("Residue-contact probability")
        ax.tick_params(axis="both", labelsize=6.0)
        ax.xaxis.label.set_size(6.7)
        ax.yaxis.label.set_size(6.7)
        if legend_columns is not None:
            ax.legend(
                loc="upper left",
                fontsize=6.0,
                ncols=legend_columns,
                frameon=False,
                columnspacing=0.85,
                handlelength=1.65,
                handletextpad=0.40,
                borderaxespad=0.35,
                labelspacing=0.32,
            )
        panel_label(ax, panel, x=panel_x)

    composition_colors = plt.get_cmap("viridis")(
        np.linspace(0.10, 0.88, len(composition_contacts))
    )
    composition_profiles = [
        (system, label, color, "-")
        for (system, label), color in zip(
            composition_contacts,
            composition_colors,
        )
    ]
    dna_addition_profiles = [
        ("M10O90", "10% MED1 −DNA", composition_colors[1], "-"),
        ("M10O90D1", "10% MED1 +DNA", composition_colors[1], "--"),
        ("M20O80", "20% MED1 −DNA", composition_colors[3], "-"),
        ("M20O80D1", "20% MED1 +DNA", composition_colors[3], "--"),
    ]
    split2_profiles = [
        ("M10O90", "10% MED1 −DNA", base.COLORS["mixed"], "-"),
        ("M10O90D1", "10% MED1 +DNA", base.COLORS["mixed"], "--"),
        ("M10O90-S2", "Split-2 −DNA", "#555555", "-"),
        ("M10O90D1-S2", "Split-2 +DNA", "#555555", "--"),
    ]

    om_grid = outer[1, 0].subgridspec(2, 3, wspace=0.20, hspace=0.30)
    om_groups = [
        (composition_profiles, "D", "G", 0.05),
        (dna_addition_profiles, "E", "H", 0.08),
        (split2_profiles, "F", "I", 0.08),
    ]
    for column, (systems, med1_panel, oct4_panel, alpha) in enumerate(om_groups):
        med1_ax = fig.add_subplot(om_grid[0, column])
        profile_panel(
            med1_ax,
            "MED1-OCT4",
            "MED1",
            systems,
            med1_panel,
            band_alpha=alpha,
            show_ylabel=column == 0,
            legend_columns=2 if column == 0 else 1,
        )
        med1_ax.set_ylim(-0.02, 0.68)
        profile_panel(
            fig.add_subplot(om_grid[1, column]),
            "MED1-OCT4",
            "OCT4",
            systems,
            oct4_panel,
            band_alpha=alpha,
            show_ylabel=column == 0,
        )
    dna_grid = fig.add_gridspec(
        1,
        2,
        left=0.075,
        right=0.985,
        bottom=0.075 * original_height / total_height,
        top=(dna_profile_height - 0.18) / total_height,
        wspace=0.22,
    )
    dna_profile_systems = {
        "MED1": [
            ("M10O90D1", "10% MED1 +DNA", composition_colors[1], "-"),
            ("M20O80D1", "20% MED1 +DNA", composition_colors[3], "-"),
            ("M50O0D1", "MED1-only +DNA", base.COLORS["MED1"], "-"),
            ("M10O90D1-S2", "Split-2 +DNA", "#555555", "--"),
        ],
        "OCT4": [
            ("M0O100D1", "OCT4-only +DNA", base.COLORS["OCT4"], "-"),
            ("M10O90D1", "10% MED1 +DNA", composition_colors[1], "-"),
            ("M20O80D1", "20% MED1 +DNA", composition_colors[3], "-"),
            ("M10O90D1-S2", "Split-2 +DNA", "#555555", "--"),
        ],
    }
    for column, (side, panel) in enumerate([("MED1", "J"), ("OCT4", "K")]):
        ax = fig.add_subplot(dna_grid[0, column])
        profile_panel(
            ax,
            f"{side}-DNA",
            side,
            dna_profile_systems[side],
            panel,
            band_alpha=0.08,
            show_ylabel=True,
            legend_columns=1,
            panel_x=-0.12,
        )
        ax.set_ylabel("DNA-contact probability", fontsize=6.7)
        ax.set_ylim(-0.01, 0.50 if side == "MED1" else 0.43)

    save_supporting(fig, out / "figS5_contact_type_and_residue_profiles")










def plot_chain_continuity_diagnostics(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S7: corona connectivity and DNA occupancy for Full versus Split-2."""
    split_d0 = SPLIT_D0[:2]
    split_d1 = SPLIT_D1[:2]
    split_labels = SPLIT_LABELS[:2]
    mechanism_repeat = tables["mechanism_repeat"]
    cluster_repeat = tables["balanced_cluster_repeat"]
    cluster_repeat = cluster_repeat[
        np.isclose(cluster_repeat["cutoff_nm"], base.PRIMARY_CUTOFF_NM)
    ]
    fig = plt.figure(figsize=(7.20, 2.55), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        3,
        left=0.080,
        right=0.975,
        bottom=0.15,
        top=0.94,
        wspace=0.55,
    )
    axes = [fig.add_subplot(outer[0, column]) for column in range(3)]
    x = np.arange(2, dtype=float)
    ac_x = np.array([0.30, 0.70])

    def metric_values(table: pd.DataFrame, metric: str) -> np.ndarray:
        values = (
            table[table["system"].isin(split_d1)]
            .pivot(index="replicate", columns="system", values=metric)
            .reindex(index=base.REPEATS, columns=split_d1)
            .to_numpy(float)
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete +DNA Full/Split values for {metric}")
        return values

    def point_summary(
        ax: plt.Axes,
        values: np.ndarray,
        positions: np.ndarray,
        *,
        color: str,
        marker: str,
    ) -> None:
        means = values.mean(axis=0)
        sds = values.std(axis=0, ddof=1)
        ax.errorbar(
            positions,
            means,
            yerr=sds,
            fmt="none",
            ecolor=color,
            elinewidth=0.78,
            capsize=2.3,
            zorder=2,
        )
        ax.scatter(
            positions,
            means,
            s=25,
            marker=marker,
            color=color if marker == "o" else "white",
            edgecolor="white" if marker == "o" else color,
            linewidth=0.65,
            zorder=4,
        )
        for system_index in range(values.shape[1]):
            ax.scatter(
                positions[system_index] + replicate_offsets(values.shape[0], 0.045),
                values[:, system_index],
                s=15,
                marker=marker,
                facecolor="white",
                edgecolor=color,
                linewidth=0.65,
                zorder=6,
            )

    lcc_values = metric_values(cluster_repeat, "largest_cluster_fraction")
    free_values = metric_values(cluster_repeat, "monomer_fraction")
    point_summary(axes[0], lcc_values, ac_x, color="black", marker="o")
    free_ax = axes[0].twinx()
    point_summary(
        free_ax,
        free_values,
        ac_x,
        color="black",
        marker="s",
    )
    axes[0].set(ylabel="LCC fraction", ylim=(0.0, 0.60))
    free_ax.set(ylabel="Free-chain fraction", ylim=(0.0, 0.60))
    free_ax.yaxis.label.set_size(7.4)
    axes[0].spines["right"].set_visible(False)
    free_ax.spines["right"].set_color("black")
    free_ax.spines["right"].set_visible(True)
    free_ax.tick_params(axis="y", colors="black")
    free_ax.yaxis.label.set_color("black")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=4.2, label="LCC"),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="None",
                marker="s",
                markerfacecolor="white",
                markersize=4.2,
                label="free chains",
            ),
        ],
        loc="lower left",
        fontsize=6.5,
        frameon=False,
    )
    axes[0].set(xlim=(0.0, 1.0), xticks=ac_x, xticklabels=split_labels)
    panel_label(axes[0], "A", x=-0.17)

    # Restore the original connected-component comparison at a fixed graph
    # resolution. Bars remain below the independent-trajectory points.
    for condition_index, (systems, condition, color) in enumerate(
        [
            (split_d0, "−DNA", CONDITIONS["without_DNA"]),
            (split_d1, "+DNA", CONDITIONS["with_DNA"]),
        ]
    ):
        values = (
            mechanism_repeat[mechanism_repeat["system"].isin(systems)]
            .pivot(
                index="replicate",
                columns="system",
                values="continuity_block_residue_LCC_fraction",
            )
            .reindex(index=base.REPEATS, columns=systems)
            .to_numpy(float)
        )
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Incomplete fixed-block connected-component values for {condition}"
            )
        offset = -0.17 if condition_index == 0 else 0.17
        for system_index in range(values.shape[1]):
            xpos = x[system_index] + offset
            axes[1].bar(
                xpos,
                values[:, system_index].mean(),
                width=0.30,
                facecolor=to_rgba(color, 0.90),
                edgecolor="black",
                linewidth=1.05,
                zorder=1,
            )
            axes[1].errorbar(
                xpos,
                values[:, system_index].mean(),
                yerr=values[:, system_index].std(ddof=1),
                fmt="none",
                color="black",
                elinewidth=1.05,
                capsize=2.2,
                zorder=3,
            )
            axes[1].scatter(
                xpos + replicate_offsets(values.shape[0], 0.035),
                values[:, system_index],
                s=15,
                facecolor="white",
                edgecolor="black",
                linewidth=0.65,
                zorder=6,
            )
    axes[1].set(ylabel="Fixed-block LCC fraction", ylim=(0.0, 0.60))
    axes[1].yaxis.label.set_size(7.4)
    axes[1].legend(
        handles=[
            Patch(facecolor=to_rgba(CONDITIONS["without_DNA"], 0.90),
                  edgecolor="black", linewidth=1.05, label="−DNA"),
            Patch(facecolor=to_rgba(CONDITIONS["with_DNA"], 0.90),
                  edgecolor="black", linewidth=1.05, label="+DNA"),
        ],
        loc="upper right",
        fontsize=6.4,
        frameon=False,
        handlelength=1.25,
    )
    panel_label(axes[1], "B", x=-0.18)

    # Match S6A's physical-chain occupancy. Split MED1 fragments are counted
    # separately and normalized by their actual total (20 for Split-2).
    contact_repeat = (
        tables["balanced_framewise_network_contacts_DNA"]
        .loc[lambda rows: rows["system"].isin(split_d1)]
        .groupby(["system", "replicate"], as_index=False)
        .mean(numeric_only=True)
    )
    med1_bound = metric_values(contact_repeat, "N_MED1_molecules_contacting_DNA")
    oct4_bound = metric_values(contact_repeat, "N_OCT4_molecules_contacting_DNA")
    design = tables["system_design"].set_index("system").loc[split_d1]
    med1_physical_total = (design["n_med1"] * design["med1_fragments"]).to_numpy(float)
    for offset, values, total, color, marker, label in [
        (-0.19, oct4_bound, design["n_oct4"].to_numpy(float),
         base.COLORS["OCT4"], "o", "OCT4"),
        (0.19, med1_bound, med1_physical_total,
         base.COLORS["MED1"], "s", "MED1"),
    ]:
        fractions = values / total
        positions = x + offset
        axes[2].bar(positions, fractions.mean(axis=0), width=0.34,
                    color=color, edgecolor="black", linewidth=0.65,
                    label=label, zorder=1)
        axes[2].errorbar(positions, fractions.mean(axis=0),
                         yerr=fractions.std(axis=0, ddof=1), fmt="none",
                         ecolor="black", elinewidth=0.78, capsize=2.2, zorder=2)
        for index in range(fractions.shape[1]):
            axes[2].scatter(positions[index] + replicate_offsets(fractions.shape[0], 0.045),
                            fractions[:, index], s=20, marker=marker,
                            facecolor="white", edgecolor="black", linewidth=0.75, zorder=6)
    axes[2].set(ylabel="Fraction contacting DNA", ylim=(0, 0.60),
                xlim=(-0.55, len(x) - 0.45), xticks=x, xticklabels=split_labels)
    axes[2].legend(loc="upper left", frameon=False, handlelength=1.1, handletextpad=0.6)
    panel_label(axes[2], "C", x=-0.13)

    axes[1].set_xticks(x, split_labels)

    pack_supporting_row(fig, axes, twins={0: [free_ax]})
    save_supporting(fig, out / "figS7_chain_continuity_controls")








def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--charge-only", action="store_true", help="Render only the sequence-charge figure S8")
    args = parser.parse_args()
    output = args.output.resolve()
    tables_dir = output / "tables"
    out = output / "figures" / "supplementary"
    out.mkdir(parents=True, exist_ok=True)

    if args.charge_only:
        base.configure_style()
        plot_protein_charge_distribution(out)
        print(f"Wrote protein charge distribution (S8) to {out}")
        return

    warnings.filterwarnings(
        "ignore",
        message="DCDReader currently makes independent timesteps",
    )
    warnings.filterwarnings(
        "ignore",
        message="Unknown element .*",
        category=UserWarning,
    )
    validate_extension_cache(output)
    tables = load_tables(tables_dir)
    base.configure_style()

    # Number supplementary figures by their first citation in the main text.
    plot_med1_plddt(out)
    plot_cutoff_sensitivity(tables, out)
    plot_med1_partner_composition(tables, out)
    plot_composition_diagnostics(tables, out)
    plot_contact_type_and_residue_profiles(tables, out)
    plot_dna_neighborhood_composition(tables, out)
    plot_chain_continuity_diagnostics(tables, out)
    plot_protein_charge_distribution(out)
    print(f"Wrote eight standardized supplementary figures to {out}")


if __name__ == "__main__":
    main()
