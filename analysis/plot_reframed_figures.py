#!/usr/bin/env python3
"""Render the four main figures for the chain-continuity reframing."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.spatial import cKDTree

import run_balanced_analysis as base
import model_rendering as legacy


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
TABLES = OUTPUT / "tables"
MAIN_FIGURES = OUTPUT / "figures" / "main"
PRIMARY_DNA_PAIRS = list(base.DNA_PAIRS)
PRIMARY_DNA_SYSTEMS = [pair[1] for pair in PRIMARY_DNA_PAIRS]
ALL_DNA_SYSTEMS = [pair[1] for pair in base.DNA_PAIRS]
ALL_DNA_LABELS = [
    "OCT4-only",
    "10%\nMED1",
    "20%\nMED1",
    "MED1-only",
]
ALL_DNA_COLORS = [
    base.COLORS["OCT4"],
    base.COLORS["mixed"],
    "#9C755F",
    base.COLORS["MED1"],
]
DNA_CONFORMATION_SYSTEMS = ["D1", *ALL_DNA_SYSTEMS]
DNA_CONFORMATION_LABELS = ["DNA\nonly", "OCT4\nonly", "10%\nMED1", "20%\nMED1", "MED1-\nonly"]
DNA_CONFORMATION_COLORS = [base.COLORS["DNA"], *ALL_DNA_COLORS]
FIG2_DNA_LABELS = ["OCT4\nonly", "10%\nMED1", "20%\nMED1", "MED1-\nonly"]
SPLIT_D0 = ["M10O90", "M10O90-S2", "M10O90-S4"]
SPLIT_D1 = ["M10O90D1", "M10O90D1-S2", "M10O90D1-S4"]
SPLIT_LABELS = ["Full", "Split-2", "Split-4"]
LEGEND_FONT = FontProperties(family="DejaVu Sans", size=8.7, weight="normal")
D90_CONTOUR_COLOR = "#6F4C9B"


def load_tables() -> dict[str, pd.DataFrame]:
    names = [
        "balanced_cluster_repeat",
        "balanced_cluster_group",
        "balanced_frame_repeat",
        "balanced_frame_group",
        "balanced_framewise_network_contacts_DNA",
        "balanced_framewise_clusters",
        "balanced_paired",
        "system_design",
        "trajectory_manifest",
        "mechanism_framewise",
        "mechanism_repeat",
        "mechanism_group",
        "mechanism_radial_histogram_group",
        "mechanism_radial_histogram_repeat",
        "mechanism_split_effects_repeat",
        "mechanism_split_did_group",
        "definition_audit_mask_transfer_group",
        "definition_audit_mask_transfer_repeat",
        "definition_audit_mask_decomposition_repeat",
        "definition_audit_radial_paired_repeat",
        "med1_function_frame_full_split_repeat",
        "DNA_conformation_with_D1_by_replicate",
    ]
    return {name: pd.read_csv(TABLES / f"{name}.csv") for name in names}


def replicate_offsets(n: int, width: float = 0.035) -> np.ndarray:
    return np.linspace(-width, width, n) if n > 1 else np.zeros(1)


def draw_mean_sd(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    color: str,
    *,
    marker: str = "D",
    zorder: int = 4,
) -> None:
    ax.errorbar(
        x,
        np.mean(values),
        yerr=np.std(values, ddof=1),
        fmt=marker,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.55,
        capsize=2.8,
        zorder=zorder,
    )


def styled_legend(ax: plt.Axes, *args, **kwargs):
    """Use title-matched black typography and keep placement explicit."""
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("prop", LEGEND_FONT)
    kwargs.setdefault("labelcolor", "#111111")
    return ax.legend(*args, **kwargs)


def draw_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.10,
    y: float = 1.02,
) -> None:
    """Add a panel letter without creating a subplot title."""
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
    )


def draw_projected_d90_contour(
    ax: plt.Axes,
    snapshot: dict[str, object],
    d90_nm: float,
    *,
    show_key: bool = False,
) -> None:
    """Overlay the projected envelope of the nearest-DNA d90 shell."""
    nodes = np.asarray(snapshot["nodes"])
    types = np.asarray(snapshot["types"])
    traces = snapshot["projected_traces"]
    dna_traces = [
        np.asarray(traces[int(node)])[:, :2]
        for node, type_id in zip(nodes, types)
        if int(type_id) == base.TYPE_DNA
    ]
    if not dna_traces:
        raise ValueError("A d90 contour requires a DNA-containing snapshot")

    dna_points = np.vstack(dna_traces)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x = np.linspace(x_min, x_max, 360)
    y = np.linspace(y_min, y_max, 360)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    projected_distance = cKDTree(dna_points).query(grid_points)[0].reshape(xx.shape)
    if not (float(projected_distance.min()) < d90_nm < float(projected_distance.max())):
        raise ValueError(
            f"Projected d90 contour ({d90_nm:.2f} nm) falls outside the snapshot view"
        )

    # The white underlay separates the shell from protein traces and contact
    # edges; the dashed purple line is the top structural layer.
    ax.contour(
        xx,
        yy,
        projected_distance,
        levels=[d90_nm],
        colors=["white"],
        linewidths=2.25,
        alpha=0.90,
        zorder=5.1,
    )
    ax.contour(
        xx,
        yy,
        projected_distance,
        levels=[d90_nm],
        colors=[D90_CONTOUR_COLOR],
        linewidths=1.15,
        linestyles=[(0, (4.0, 2.2))],
        alpha=0.98,
        zorder=5.2,
    )
    if show_key:
        key = ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=D90_CONTOUR_COLOR,
                    lw=1.15,
                    ls=(0, (4.0, 2.2)),
                    label=r"projected $d_{90}$ contour",
                )
            ],
            loc="lower right",
            bbox_to_anchor=(0.985, 0.02),
            borderaxespad=0,
            handlelength=1.7,
            handletextpad=0.35,
            prop=FontProperties(family="DejaVu Sans", size=7.0),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.84,
        )
        key.set_zorder(7)


def select_matched_snapshot(
    framewise: pd.DataFrame,
    systems: list[str],
) -> pd.DataFrame:
    """Choose one common repeat/frame without looking at rendered structures."""
    metrics = ["direct_largest_protein_fraction", "PP_edge_density"]
    candidates = framewise[framewise["system"].isin(systems)].copy()
    centers = candidates.groupby("system", observed=True)[metrics].median()
    scales = candidates.groupby("system", observed=True)[metrics].std(ddof=1).clip(lower=1e-8)
    ranked: list[tuple[float, str, int]] = []
    for (replicate, frame), group in candidates.groupby(["replicate", "frame"], observed=True):
        if set(group["system"]) != set(systems):
            continue
        score = 0.0
        for row in group.itertuples(index=False):
            score += sum(
                abs((float(getattr(row, metric)) - centers.loc[row.system, metric]) / scales.loc[row.system, metric])
                for metric in metrics
            )
        score += 0.05 * abs(float(group["time_us"].iloc[0]) - 3.0)
        ranked.append((score, str(replicate), int(frame)))
    if not ranked:
        raise RuntimeError(f"No matched snapshot for {systems}")
    _, replicate, frame = min(ranked)
    return candidates[
        (candidates["replicate"] == replicate) & (candidates["frame"] == frame)
    ].copy()


def select_radial_snapshot(
    extension_framewise: pd.DataFrame,
    system: str,
) -> pd.Series:
    candidates = extension_framewise[extension_framewise["system"] == system].copy()
    metrics = ["DNA_linked_bead_all_q90_nm", "DNA_linked_physical_fraction"]
    score = np.zeros(len(candidates), dtype=float)
    for metric in metrics:
        center = candidates[metric].median()
        scale = max(float(candidates[metric].std(ddof=1)), 1e-8)
        score += np.abs(candidates[metric] - center) / scale
    score += 0.05 * np.abs(candidates["time_us"] - 3.0)
    return candidates.iloc[int(np.argmin(score))]


def select_matched_mechanism_snapshot(
    extension_framewise: pd.DataFrame,
) -> pd.DataFrame:
    """Select one common Full/Split-2 frame using direct-contact metrics."""
    metrics = [
        "MED1_parent_mean_partner_number",
        "DNA_linked_physical_fraction",
    ]
    candidates = extension_framewise[
        extension_framewise["system"].isin(SPLIT_D1[:2])
    ].copy()
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
        if set(group["system"]) != set(SPLIT_D1[:2]):
            continue
        score = 0.0
        for row in group.itertuples(index=False):
            score += sum(
                abs(
                    (float(getattr(row, metric)) - centers.loc[row.system, metric])
                    / scales.loc[row.system, metric]
                )
                for metric in metrics
            )
        score += 0.05 * abs(float(group["time_us"].iloc[0]) - 3.0)
        ranked.append((score, str(replicate), int(frame)))
    if not ranked:
        raise RuntimeError("No common corrected Full/Split mechanism snapshot exists")
    score, replicate, frame = min(ranked)
    selected = candidates[
        (candidates["replicate"] == replicate)
        & (candidates["frame"] == frame)
    ].copy()
    selected["condition"] = selected["system"].map(
        dict(zip(SPLIT_D1, SPLIT_LABELS))
    )
    selected["selection_score"] = score
    selected["selection_rule"] = (
        "one matched +DNA repeat/frame minimizing pooled standardized distance "
        "from system medians of external MED1 partner number and DNA-linked "
        "protein fraction; weak 3-us preference"
    )
    order = {label: index for index, label in enumerate(SPLIT_LABELS)}
    return selected.sort_values(
        "condition", key=lambda values: values.map(order)
    ).reset_index(drop=True)


def select_displayed_mechanism_snapshots(
    extension_framewise: pd.DataFrame,
) -> pd.DataFrame:
    """Choose a representative Full view distinct from the displayed Split-2 view."""
    matched = select_matched_mechanism_snapshot(extension_framewise)
    split_row = matched[matched["condition"] == "Split-2"].iloc[[0]].copy()
    split_pair = (str(split_row.iloc[0]["replicate"]), int(split_row.iloc[0]["frame"]))

    metrics = [
        "continuity_block_residue_LCC_fraction",
        "DNA_linked_bead_all_q90_nm",
    ]
    full_candidates = extension_framewise[
        extension_framewise["system"] == SPLIT_D1[0]
    ].copy()
    full_candidates = full_candidates[
        ~(
            (full_candidates["replicate"].astype(str) == split_pair[0])
            & (full_candidates["frame"].astype(int) == split_pair[1])
        )
    ].copy()
    score = np.zeros(len(full_candidates), dtype=float)
    for metric in metrics:
        center = full_candidates[metric].median()
        scale = max(float(full_candidates[metric].std(ddof=1)), 1e-8)
        score += np.abs(full_candidates[metric] - center) / scale
    score += 0.05 * np.abs(full_candidates["time_us"] - 3.0)
    full_row = full_candidates.iloc[[int(np.argmin(score))]].copy()
    full_row["condition"] = "Full"
    full_row["selection_score"] = float(np.min(score))
    full_row["selection_rule"] = (
        "Full snapshot nearest the Full medians of corrected fixed-block residue "
        "LCC and unified DNA-linked bead d90, excluding the displayed Split-2 "
        "repeat/frame; weak 3-us preference"
    )
    split_row["selection_rule"] = (
        "Split-2 snapshot from the matched mechanism screen; displayed Full "
        "snapshot selected separately"
    )
    return pd.concat([full_row, split_row], ignore_index=True, sort=False)


def _draw_pdb_model_fallback(
    ax: plt.Axes,
    coordinates: np.ndarray,
    regions: list[tuple[int, int]],
    colors: list[str],
    y_offset: float,
) -> np.ndarray:
    """PCA fallback when the documented PyMOL renderer is unavailable."""
    projected = legacy.project_model_coordinates(coordinates)
    projected -= projected.mean(axis=0)
    projected[:, 1] += y_offset
    segment_colors = np.full(len(projected) - 1, "#B8BEC7", dtype=object)
    segment_widths = np.full(len(projected) - 1, 0.85)
    for (start, end), color in zip(regions, colors):
        segment_colors[start - 1 : end - 1] = color
        segment_widths[start - 1 : end - 1] = 1.8
    for index, (start, stop) in enumerate(zip(projected[:-1], projected[1:])):
        ax.plot(
            [start[0], stop[0]],
            [start[1], stop[1]],
            color=segment_colors[index],
            lw=segment_widths[index],
            solid_capstyle="round",
            zorder=1,
        )
    sample = np.unique(np.linspace(0, len(projected) - 1, min(120, len(projected))).astype(int))
    bead_colors = np.full(len(projected), "#B8BEC7", dtype=object)
    for (start, end), color in zip(regions, colors):
        bead_colors[start - 1 : end] = color
    ax.scatter(
        projected[sample, 0],
        projected[sample, 1],
        s=7,
        color=bead_colors[sample],
        edgecolor="white",
        linewidth=0.15,
        zorder=2,
    )
    return projected


def draw_input_conformations(
    ax: plt.Axes,
    med1_coordinates: np.ndarray,
    oct4_coordinates: np.ndarray,
    med1_colors: list[str],
    oct4_colors: list[str],
) -> None:
    """Display PyMOL renders, or a reproducible fallback if unavailable."""
    pymol_images = [
        MAIN_FIGURES / "fig1_panel_b_oct4_pymol.png",
        MAIN_FIGURES / "fig1_panel_b_med1_pymol.png",
        MAIN_FIGURES / "fig1_panel_b_dna_pymol.png",
    ]
    if all(path.exists() for path in pymol_images):
        for bounds, path in zip(
            [(0.0, 0.58, 1.0, 0.42), (0.0, 0.18, 1.0, 0.40), (0.0, 0.0, 1.0, 0.18)],
            pymol_images,
        ):
            image_ax = ax.inset_axes(bounds)
            image = plt.imread(path)
            foreground = np.any(image[..., :3] < 0.985, axis=2)
            rows, columns = np.where(foreground)
            if len(rows):
                row_pad = max(3, int((rows.max() - rows.min() + 1) * 0.08))
                column_pad = max(3, int((columns.max() - columns.min() + 1) * 0.08))
                image = image[
                    max(0, rows.min() - row_pad) : min(image.shape[0], rows.max() + row_pad + 1),
                    max(0, columns.min() - column_pad) : min(image.shape[1], columns.max() + column_pad + 1),
                ]
            image_ax.imshow(image)
            image_ax.set_axis_off()
        ax.set_axis_off()
        return

    warnings.warn(
        "PyMOL renders missing; using the deterministic PCA fallback. "
        "Run render_fig1_panel_b.pml with PyMOL to replace it.",
        stacklevel=2,
    )
    oct4 = _draw_pdb_model_fallback(
        ax,
        oct4_coordinates,
        legacy.OCT4_NATIVE_REGIONS,
        oct4_colors,
        y_offset=15.0,
    )
    med1 = _draw_pdb_model_fallback(
        ax,
        med1_coordinates,
        legacy.MED1_NATIVE_REGIONS,
        med1_colors,
        y_offset=-15.0,
    )
    lower = np.min(np.vstack([oct4, med1]), axis=0)
    upper = np.max(np.vstack([oct4, med1]), axis=0)
    padding = np.maximum((upper - lower) * 0.08, 1.0)
    ax.set_xlim(float(lower[0] - padding[0]), float(upper[0] + padding[0]))
    ax.set_ylim(float(lower[1] - padding[1]), float(upper[1] + padding[1]))
    ax.set_aspect("equal", adjustable="box")
    scale_x = float(lower[0] + 0.04 * (upper[0] - lower[0]))
    scale_y = float(lower[1] + 0.07 * (upper[1] - lower[1]))
    ax.plot([scale_x, scale_x + 10.0], [scale_y, scale_y], color="#111111", lw=1.5, zorder=3)
    ax.text(scale_x + 5.0, scale_y + 1.0, "10 nm", ha="center", va="bottom", fontsize=6.8)
    ax.set_axis_off()


def draw_original_object_panels(fig: plt.Figure, spec) -> None:
    """Draw the two protein maps and their corresponding input conformations."""
    root = HERE.parent
    med1_colors = ["#2A9D8F", "#457B9D", "#8067B7", "#D08C60"]
    oct4_colors = ["#3C78B5", "#A06AB4"]
    med1_coordinates, _ = legacy.read_pdb_bead_coordinates(
        root / "simulation/inputs/MED1-alphafold.pdb", "CA"
    )
    oct4_coordinates, _ = legacy.read_pdb_bead_coordinates(
        root / "simulation/inputs/OCT4.pdb", "CA"
    )
    med1_charge = sum(
        legacy.AA_CHARGE.get(name, 0.0)
        for name in legacy.read_residue_names(
            root / "simulation/inputs/MED1-alphafold.pdb"
        )
    )
    oct4_charge = sum(
        legacy.AA_CHARGE.get(name, 0.0)
        for name in legacy.read_residue_names(root / "simulation/inputs/OCT4.pdb")
    )

    grid = spec.subgridspec(
        2,
        2,
        width_ratios=[1.15, 1.22],
        height_ratios=[1.0, 1.0],
        wspace=0.035,
        hspace=0.18,
    )
    sequence_axes = [fig.add_subplot(grid[row, 0]) for row in range(2)]
    conformation_ax = fig.add_subplot(grid[:, 1])
    legacy.draw_sequence_model(
        sequence_axes[0],
        "OCT4",
        base.OCT4_LENGTH,
        f"{oct4_charge:+g} e".replace("-", "−"),
        legacy.OCT4_NATIVE_REGIONS,
        oct4_colors,
        ["POU-specific", "POU-homeodomain"],
    )
    legacy.draw_sequence_model(
        sequence_axes[1],
        "MED1",
        base.MED1_LENGTH,
        f"{med1_charge:+g} e",
        legacy.MED1_NATIVE_REGIONS,
        med1_colors,
        ["M1", "M2", "M3", "M4"],
    )
    draw_input_conformations(
        conformation_ax,
        med1_coordinates,
        oct4_coordinates,
        med1_colors,
        oct4_colors,
    )
    draw_panel_label(sequence_axes[0], "A", x=-0.085, y=1.10)
    draw_panel_label(conformation_ax, "B", x=-0.10, y=1.02)


def draw_object_models(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x = np.linspace(0.20, 0.92, 500)
    med1_y = 0.78 + 0.025 * np.sin(11 * np.pi * x) + 0.012 * np.sin(31 * np.pi * x)
    ax.plot(x, med1_y, color=base.COLORS["MED1"], lw=2.0)
    for start, stop in [(77, 232), (252, 274), (287, 349), (361, 518)]:
        left = 0.20 + (start - 1) / base.MED1_LENGTH * 0.72
        right = 0.20 + stop / base.MED1_LENGTH * 0.72
        ax.plot([left, right], [0.78, 0.78], color="#8E3340", lw=5.0, solid_capstyle="round")
    ax.text(0.02, 0.79, "MED1", fontweight="bold", va="center")
    ax.text(0.20, 0.69, "1,581 residues; long multiregion chain", fontsize=7.1, color="#555555")

    ax.plot([0.20, 0.92], [0.47, 0.47], color=base.COLORS["OCT4"], lw=2.0)
    pou_specific_left = 0.20 + (136 - 1) / base.OCT4_LENGTH * 0.72
    pou_specific_width = (217 - 136 + 1) / base.OCT4_LENGTH * 0.72
    pou_homeo_left = 0.20 + (234 - 1) / base.OCT4_LENGTH * 0.72
    pou_homeo_width = (286 - 234 + 1) / base.OCT4_LENGTH * 0.72
    ax.add_patch(FancyBboxPatch((pou_specific_left, 0.425), pou_specific_width, 0.09, boxstyle="round,pad=0.01", facecolor="#5C88B7", edgecolor="white"))
    ax.add_patch(FancyBboxPatch((pou_homeo_left, 0.425), pou_homeo_width, 0.09, boxstyle="round,pad=0.01", facecolor="#8067B7", edgecolor="white"))
    ax.text(0.02, 0.47, "OCT4", fontweight="bold", va="center")
    ax.text(0.20, 0.35, "360 residues; POU-specific and homeodomains", fontsize=7.1, color="#555555")

    dna_x = np.linspace(0.20, 0.92, 200)
    ax.plot(dna_x, 0.17 + 0.018 * np.sin(12 * np.pi * dna_x), color=base.COLORS["DNA"], lw=1.7)
    ax.plot(dna_x, 0.13 + 0.018 * np.sin(12 * np.pi * dna_x + np.pi), color=base.COLORS["DNA"], lw=1.7)
    ax.text(0.02, 0.15, "dsDNA", fontweight="bold", va="center")
    ax.text(0.20, 0.04, "200 bp; 400 coarse-grained beads", fontsize=7.1, color="#555555")
    base.set_panel_title(ax, "A", "Coarse-grained simulation objects", label_x=-0.02)


def draw_design_matrix(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.69, 0.96, 0.25, "Composition", "100 protein chains\n0–50% MED1\nfixed residue concentration"),
        (0.02, 0.37, 0.96, 0.25, "DNA addition", "matched protein coordinates and seed\n−DNA  →  +1 dsDNA\n0%, 10%, 20%, and pure-MED1 backgrounds"),
        (0.02, 0.05, 0.96, 0.25, "Chain continuity", "10 MED1 parents + 90 OCT4\nFull  →  Split-2  →  Split-4\nfixed 100-parent / 130-block views"),
    ]
    colors = ["#FBEAEC", "#EAF5E8", "#F3ECF7"]
    for (x, y, width, height, heading, body), color in zip(boxes, colors):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012", facecolor=color, edgecolor="#B8B8B8", linewidth=0.7))
        ax.text(x + 0.025, y + height - 0.055, heading, fontweight="bold", fontsize=8.3, va="top")
        ax.text(x + 0.46, y + height / 2, body, fontsize=6.25, va="center", linespacing=1.22)
    base.set_panel_title(ax, "B", "Matched simulation designs", label_x=-0.02)


def plot_figure1(tables: dict[str, pd.DataFrame], out: Path) -> None:
    cluster_group = tables["balanced_cluster_group"]
    cluster_group = cluster_group[np.isclose(cluster_group["cutoff_nm"], base.PRIMARY_CUTOFF_NM)].set_index("system")
    cluster_repeat = tables["balanced_cluster_repeat"]
    cluster_repeat = cluster_repeat[np.isclose(cluster_repeat["cutoff_nm"], base.PRIMARY_CUTOFF_NM)]
    frame_group = tables["balanced_frame_group"].set_index("system")
    frame_repeat = tables["balanced_frame_repeat"]
    design = tables["system_design"]
    x_map = base.composition_x(design)
    x = np.asarray([x_map[system] for system in base.COMPOSITION_SYSTEMS])

    fig = plt.figure(figsize=(7.20, 8.30), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.55, 0.82, 1.05],
        left=0.075,
        right=0.985,
        bottom=0.045,
        top=0.982,
        hspace=0.24,
    )
    draw_original_object_panels(fig, outer[0, 0])
    metric_grid = outer[1, 0].subgridspec(1, 2, wspace=0.47)
    axes = [fig.add_subplot(metric_grid[0, col]) for col in range(2)]

    ax = axes[0]
    values = cluster_group.loc[base.COMPOSITION_SYSTEMS, "largest_cluster_fraction_mean"].to_numpy(float)
    errors = cluster_group.loc[base.COMPOSITION_SYSTEMS, "largest_cluster_fraction_sd"].to_numpy(float)
    free_values = cluster_group.loc[base.COMPOSITION_SYSTEMS, "monomer_fraction_mean"].to_numpy(float)
    free_errors = cluster_group.loc[base.COMPOSITION_SYSTEMS, "monomer_fraction_sd"].to_numpy(float)
    ax.plot(x, values, color="black", lw=1.45, zorder=1)
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="black", elinewidth=0.8, capsize=2.4, zorder=2)
    ax.scatter(x, values, s=24, color="black", edgecolor="white", linewidth=0.55, zorder=4)
    for index, system in enumerate(base.COMPOSITION_SYSTEMS):
        repeat_values = cluster_repeat[cluster_repeat["system"] == system]["largest_cluster_fraction"].to_numpy(float)
        ax.scatter(x[index] + replicate_offsets(len(repeat_values), 0.28), repeat_values, s=14, facecolor="white", edgecolor="black", linewidth=0.65, zorder=5)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set(xlabel="MED1 molecule fraction (%)", ylabel="LCC fraction", xlim=(-2, 52), ylim=(0, 0.38))
    free_ax = ax.twinx()
    free_ax.plot(x, free_values, color="black", lw=1.1, ls="--", zorder=1)
    free_ax.errorbar(x, free_values, yerr=free_errors, fmt="none", ecolor="black", elinewidth=0.75, capsize=2.2, zorder=2)
    free_ax.scatter(x, free_values, s=25, facecolor="white", edgecolor="black", marker="s", linewidth=0.65, zorder=4)
    for index, system in enumerate(base.COMPOSITION_SYSTEMS):
        repeat_values = cluster_repeat[cluster_repeat["system"] == system]["monomer_fraction"].to_numpy(float)
        free_ax.scatter(x[index] + replicate_offsets(len(repeat_values), 0.28), repeat_values, s=14, facecolor="white", edgecolor="black", marker="s", linewidth=0.65, zorder=5)
    free_ax.set(ylabel="Free-chain fraction", ylim=(0, 0.38))
    free_ax.yaxis.labelpad = 5.5
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", marker="o", markersize=4.5, label="LCC"),
            Line2D([0], [0], color="black", ls="--", marker="s", markerfacecolor="white", markersize=4.5, label="free chains"),
        ],
        loc="lower right",
        prop=FontProperties(family="DejaVu Sans", size=7.1),
        frameon=False,
    )
    draw_panel_label(ax, "C", x=-0.11)

    ax = axes[1]
    for metric, molecule, color in [
        ("OCT4_mean_degree", "OCT4", base.COLORS["OCT4"]),
        ("MED1_mean_degree", "MED1", base.COLORS["MED1"]),
    ]:
        systems = [system for system in base.COMPOSITION_SYSTEMS if np.isfinite(frame_group.loc[system, f"{metric}_mean"])]
        means = [frame_group.loc[system, f"{metric}_mean"] for system in systems]
        sds = [frame_group.loc[system, f"{metric}_sd"] for system in systems]
        positions = [x_map[system] for system in systems]
        ax.plot(positions, means, color=color, lw=1.25, zorder=1)
        ax.errorbar(positions, means, yerr=sds, fmt="none", ecolor=color, elinewidth=0.8, capsize=2.4, zorder=2)
        ax.scatter(positions, means, s=24, color=color, edgecolor="white", linewidth=0.55, label=molecule, zorder=4)
        for system in systems:
            repeat_values = frame_repeat[frame_repeat["system"] == system][metric].dropna().to_numpy(float)
            ax.scatter(x_map[system] + replicate_offsets(len(repeat_values), 0.24), repeat_values, s=13, facecolor="white", edgecolor=color, linewidth=0.6, zorder=5)
    ax.set(xlabel="MED1 molecule fraction (%)", ylabel="Distinct partners per chain", xlim=(-2, 52), ylim=(0.78, 3.62))
    styled_legend(ax, loc="upper right", ncols=1)
    draw_panel_label(ax, "D", x=-0.11)

    for metric_ax in axes:
        metric_ax.tick_params(axis="both", labelsize=8.2)
        metric_ax.xaxis.label.set_size(8.8)
        metric_ax.yaxis.label.set_size(8.8)
    free_ax.tick_params(axis="y", labelsize=8.2)
    free_ax.yaxis.label.set_size(8.8)

    structure_grid = outer[2, 0].subgridspec(1, 3, wspace=0.018)
    manifest = tables["trajectory_manifest"]
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    structure_systems = [
        ("M0O100", "0% MED1"),
        ("M10O90", "10% MED1"),
        ("M50O50", "50% MED1"),
    ]
    for index, (system, label) in enumerate(structure_systems):
        selection = base.select_snapshot(framewise, system)
        task = base.get_task(manifest, system, str(selection["replicate"]))
        snapshot = base.build_spatial_snapshot(task, int(selection["frame"]), mode="full_system")
        ax = fig.add_subplot(structure_grid[0, index])
        base.draw_spatial_snapshot(ax, snapshot, task.box_nm / 2)
        ax.set_title("")
        ax.text(
            0.035,
            0.965,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
            zorder=6,
        )
        if index == 0:
            draw_panel_label(ax, "E", x=-0.035, y=1.01)
    base.save_figure(fig, out / "fig1_model_design_composition")


def draw_bar_summary(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    color: str,
    *,
    width: float = 0.28,
    facecolor: str | None = None,
    hatch: str | None = None,
    marker: str = "o",
) -> None:
    """Draw a mean bar under its SD and independently initialized trajectories."""
    bar_facecolor = color if facecolor is None else facecolor
    ax.bar(
        x,
        float(np.mean(values)),
        width=width,
        color=bar_facecolor,
        alpha=0.90,
        edgecolor=color,
        linewidth=0.65,
        hatch=hatch,
        zorder=1,
    )
    ax.errorbar(
        x,
        float(np.mean(values)),
        yerr=float(np.std(values, ddof=1)),
        fmt="none",
        color=color,
        elinewidth=0.85,
        capsize=2.4,
        zorder=3,
    )
    ax.scatter(
        x + replicate_offsets(len(values), 0.075),
        values,
        s=19,
        facecolor="white",
        edgecolor=color,
        marker=marker,
        linewidth=0.75,
        zorder=5,
    )


def plot_paired_dna_bars(
    ax: plt.Axes,
    paired: pd.DataFrame,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    panel: str,
) -> None:
    """Compare matched −DNA and +DNA formulations without trajectory connectors."""
    table = paired[
        (paired["metric"] == metric)
        & np.isclose(paired["cutoff_nm"], base.PRIMARY_CUTOFF_NM)
    ]
    for index, (_, _, label) in enumerate(PRIMARY_DNA_PAIRS):
        sub = table[table["comparison"] == label].set_index("replicate").reindex(base.REPEATS)
        draw_bar_summary(ax, index - 0.16, sub["without_DNA"].to_numpy(float), base.COLORS["neutral"])
        draw_bar_summary(ax, index + 0.16, sub["with_DNA"].to_numpy(float), base.COLORS["DNA"])
    ax.set_xticks(np.arange(len(PRIMARY_DNA_PAIRS)), FIG2_DNA_LABELS)
    ax.tick_params(axis="x", labelsize=6.2, pad=1.0)
    ax.set(ylabel=ylabel, ylim=ylim)
    draw_panel_label(ax, panel, x=-0.14)


def plot_dna_association_bars(
    ax: plt.Axes,
    repeat: pd.DataFrame,
) -> None:
    for index, system in enumerate(PRIMARY_DNA_SYSTEMS):
        sub = repeat[repeat["system"] == system].set_index("replicate").reindex(base.REPEATS)
        draw_bar_summary(
            ax,
            index - 0.27,
            sub["DNA_direct_physical_fraction"].to_numpy(float),
            base.COLORS["DNA"],
            width=0.22,
        )
        draw_bar_summary(
            ax,
            index,
            sub["DNA_second_physical_fraction"].to_numpy(float),
            base.COLORS["DNA"],
            width=0.22,
            facecolor="white",
            hatch="///",
            marker="s",
        )
        draw_bar_summary(
            ax,
            index + 0.27,
            sub["DNA_third_plus_physical_fraction"].to_numpy(float),
            base.COLORS["DNA"],
            width=0.22,
            facecolor="white",
            hatch="...",
            marker="^",
        )
    ax.set_xticks(range(len(PRIMARY_DNA_SYSTEMS)), FIG2_DNA_LABELS)
    ax.tick_params(axis="x", labelsize=6.2, pad=1.0)
    ax.set(ylabel="Fraction of protein chains", ylim=(0, 1.05))
    styled_legend(
        ax,
        handles=[
            Patch(facecolor=base.COLORS["DNA"], edgecolor=base.COLORS["DNA"], alpha=0.90, label="direct neighbors"),
            Patch(facecolor="white", edgecolor=base.COLORS["DNA"], hatch="///", label="2nd neighbors"),
            Patch(facecolor="white", edgecolor=base.COLORS["DNA"], hatch="...", label="3rd+ neighbors"),
        ],
        loc="upper left",
        prop=FontProperties(family="DejaVu Sans", size=6.2, weight="normal"),
        handlelength=1.2,
        handletextpad=0.35,
    )
    draw_panel_label(ax, "C", x=-0.17)


def plot_dna_rg_line(
    ax: plt.Axes,
    table: pd.DataFrame,
) -> None:
    values_by_system = []
    for system in DNA_CONFORMATION_SYSTEMS:
        values = (
            table[table["system"] == system]
            .set_index("replicate")
            .reindex(base.REPEATS)["DNA_Rg_nm"]
            .to_numpy(float)
        )
        values_by_system.append(values)
    values_array = np.asarray(values_by_system, dtype=float)
    x = np.arange(len(DNA_CONFORMATION_SYSTEMS), dtype=float)
    color = "black"
    means = values_array.mean(axis=1)
    sds = values_array.std(axis=1, ddof=1)
    ax.plot(x, means, color=color, lw=1.55, zorder=1)
    ax.errorbar(x, means, yerr=sds, fmt="none", color=color, elinewidth=0.85, capsize=2.5, zorder=2)
    ax.scatter(x, means, s=28, facecolor=color, edgecolor="white", linewidth=0.55, zorder=4)
    for index, values in enumerate(values_array):
        ax.scatter(
            index + replicate_offsets(len(values), 0.075),
            values,
            s=20,
            facecolor="white",
            edgecolor=color,
            linewidth=0.75,
            zorder=5,
        )
    ax.set_xticks(range(len(DNA_CONFORMATION_SYSTEMS)), DNA_CONFORMATION_LABELS)
    ax.tick_params(axis="x", labelsize=6.6, pad=1.0)
    ax.set(ylabel=r"DNA $R_g$ (nm)", ylim=(16.4, 18.15))
    draw_panel_label(ax, "D", x=-0.17)


def plot_figure2(tables: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    fig = plt.figure(figsize=(7.20, 7.35), constrained_layout=False)
    outer = fig.add_gridspec(3, 1, height_ratios=[0.84, 0.84, 0.98], left=0.085, right=0.985, bottom=0.055, top=0.97, hspace=0.22)
    upper_metrics = outer[0, 0].subgridspec(1, 2, wspace=0.35)
    lower_metrics = outer[1, 0].subgridspec(1, 2, wspace=0.35)
    axes = [
        fig.add_subplot(upper_metrics[0, 0]),
        fig.add_subplot(upper_metrics[0, 1]),
        fig.add_subplot(lower_metrics[0, 0]),
        fig.add_subplot(lower_metrics[0, 1]),
    ]
    paired = tables["balanced_paired"]
    plot_paired_dna_bars(axes[0], paired, "largest_cluster_fraction", "LCC fraction", (0, 1.06), "A")
    plot_paired_dna_bars(axes[1], paired, "PP_edge_density", "Protein edge density", (0, 0.13), "B")
    styled_legend(
        axes[0],
        handles=[
            Patch(facecolor=base.COLORS["neutral"], edgecolor=base.COLORS["neutral"], alpha=0.90, label="−DNA"),
            Patch(facecolor=base.COLORS["DNA"], edgecolor=base.COLORS["DNA"], alpha=0.90, label="+DNA"),
        ],
        loc="upper left",
        ncols=2,
        prop=FontProperties(family="DejaVu Sans", size=7.0, weight="normal"),
        handlelength=1.2,
        handletextpad=0.35,
    )
    plot_dna_association_bars(axes[2], tables["mechanism_repeat"])
    plot_dna_rg_line(axes[3], tables["DNA_conformation_with_D1_by_replicate"])

    structure = outer[2, 0].subgridspec(1, 4, wspace=0.015)
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    manifest = tables["trajectory_manifest"]
    selection_rows = []
    panels = [("M0O100", "OCT4-only −DNA"), ("M0O100D1", "OCT4-only +DNA"), ("M20O80", "20% MED1 −DNA"), ("M20O80D1", "20% MED1 +DNA")]
    for pair_systems in [["M0O100", "M0O100D1"], ["M20O80", "M20O80D1"]]:
        selected = select_matched_snapshot(framewise, pair_systems)
        selection_rows.extend(selected.to_dict("records"))
    selection = pd.DataFrame(selection_rows).set_index("system")
    for index, (system, title) in enumerate(panels):
        row = selection.loc[system]
        task = base.get_task(manifest, system, str(row["replicate"]))
        snapshot = base.build_spatial_snapshot(task, int(row["frame"]), mode="full_system")
        ax = fig.add_subplot(structure[0, index])
        base.draw_spatial_snapshot(ax, snapshot, task.box_nm * 0.43)
        ax.set_title("")
        ax.text(
            0.035,
            0.965,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
            zorder=6,
        )
        if index == 0:
            draw_panel_label(ax, "E", x=-0.115, y=1.01)
    base.save_figure(fig, out / "fig2_DNA_addition_network_response")
    return pd.DataFrame(selection_rows)


def plot_figure3(tables: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    fig = plt.figure(figsize=(7.20, 5.55), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.94, 1.20],
        left=0.085,
        right=0.985,
        bottom=0.06,
        top=0.975,
        hspace=0.22,
    )
    top = outer[0, 0].subgridspec(
        1,
        3,
        width_ratios=[1.50, 1.0, 1.0],
        wspace=0.42,
    )
    density_ax = fig.add_subplot(top[0, 0])
    profile_specs = [
        ("M0O100D1", ALL_DNA_COLORS[0], "OCT4-only"),
        ("M10O90D1", ALL_DNA_COLORS[1], "10% MED1"),
        ("M20O80D1", ALL_DNA_COLORS[2], "20% MED1"),
        ("M50O0D1", ALL_DNA_COLORS[3], "MED1-only"),
    ]
    hist_repeat = tables["mechanism_radial_histogram_repeat"]
    distribution = hist_repeat[
        (hist_repeat["distance_scope"] == "bead")
        & (hist_repeat["protein_type"] == "all")
        & hist_repeat["system"].isin(ALL_DNA_SYSTEMS)
        & (hist_repeat["bin_end_nm"] <= 40)
    ].copy()
    distribution["one_nm_bin_start"] = np.floor(
        distribution["bin_start_nm"]
    ).astype(int)
    distribution = (
        distribution.groupby(
            ["system", "replicate", "one_nm_bin_start"],
            as_index=False,
            observed=True,
        )["frame_mean_count"]
        .sum()
        .groupby(["system", "one_nm_bin_start"], as_index=False, observed=True)[
            "frame_mean_count"
        ]
        .mean()
    )
    for system, color, label in profile_specs:
        sub = distribution[distribution["system"] == system].sort_values(
            "one_nm_bin_start"
        )
        density_ax.plot(
            sub["one_nm_bin_start"] + 0.5,
            sub["frame_mean_count"],
            color=color,
            lw=1.6,
            label=label,
        )
    density_ax.set(
        xlabel="Nearest-DNA distance (nm)",
        ylabel="Mean linked protein beads\nper 1 nm bin",
        xlim=(0, 40),
        ylim=(0, 3000),
    )
    density_handles, density_labels = density_ax.get_legend_handles_labels()
    density_legend_order = [0, 1, 3, 2]
    styled_legend(
        density_ax,
        handles=[density_handles[index] for index in density_legend_order],
        labels=[density_labels[index] for index in density_legend_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncols=2,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.3,
        labelspacing=0.15,
        borderaxespad=0.15,
        prop=FontProperties(family="DejaVu Sans", size=6.5, weight="normal"),
    )
    draw_panel_label(density_ax, "A", x=-0.17)

    q90_ax = fig.add_subplot(top[0, 1])
    extension_repeat = tables["mechanism_repeat"]
    for index, system in enumerate(ALL_DNA_SYSTEMS):
        sub = extension_repeat[extension_repeat["system"] == system].set_index("replicate").reindex(base.REPEATS)
        values = sub["DNA_linked_bead_all_q90_nm"].to_numpy(float)
        q90_ax.scatter(index + replicate_offsets(3, 0.035), values, s=21, facecolor="white", edgecolor="#000000", marker="o", linewidth=0.75, zorder=6)
        draw_mean_sd(q90_ax, index, values, "#000000", marker="o")
    q90_ax.set_xticks(range(len(ALL_DNA_LABELS)), ALL_DNA_LABELS)
    q90_ax.tick_params(axis="x", labelsize=6.8)
    q90_ax.set(ylabel=r"Mean framewise $d_{90}$ (nm)", ylim=(10, 39.5))
    draw_panel_label(q90_ax, "B", x=-0.17)

    species_q90_ax = fig.add_subplot(top[0, 2])
    mixed_systems = ["M10O90D1", "M20O80D1"]
    mixed_labels = ["10% MED1", "20% MED1"]
    for index, system in enumerate(mixed_systems):
        sub = extension_repeat[extension_repeat["system"] == system].set_index("replicate").reindex(base.REPEATS)
        for offset, protein_type, color, marker in [
            (-0.10, "OCT4", base.COLORS["OCT4"], "o"),
            (0.10, "MED1", base.COLORS["MED1"], "D"),
        ]:
            values = sub[f"DNA_linked_bead_{protein_type}_q90_nm"].to_numpy(float)
            species_q90_ax.scatter(index + offset + replicate_offsets(3, 0.02), values, s=20, facecolor="white", edgecolor=color, marker=marker, linewidth=0.75, zorder=6)
            draw_mean_sd(species_q90_ax, index + offset, values, color, marker=marker)
    species_q90_ax.set_xticks(range(len(mixed_labels)), mixed_labels)
    species_q90_ax.set(ylabel=r"Species bead $d_{90}$ (nm)", ylim=(17, 41))
    styled_legend(
        species_q90_ax,
        handles=[
            Line2D([0], [0], marker="o", color="none", markeredgecolor=base.COLORS["OCT4"], markerfacecolor="white", label="OCT4"),
            Line2D([0], [0], marker="D", color="none", markeredgecolor=base.COLORS["MED1"], markerfacecolor="white", label="MED1"),
        ],
        loc="upper left",
    )
    draw_panel_label(species_q90_ax, "C", x=-0.17)

    structure = outer[1, 0].subgridspec(1, 3, wspace=0.018)
    extension_frame = tables["mechanism_framewise"]
    manifest = tables["trajectory_manifest"]
    selected_rows = []
    snapshot_systems = ["M0O100D1", "M10O90D1", "M20O80D1"]
    snapshot_labels = ["OCT4-only +DNA", "10% MED1 +DNA", "20% MED1 +DNA"]
    common_half_span = max(
        float(
            manifest.loc[
                manifest["system"].isin(snapshot_systems), "box_nm"
            ].max()
        )
        / 2,
        41.5,
    )
    for index, (system, label) in enumerate(zip(snapshot_systems, snapshot_labels)):
        row = select_radial_snapshot(extension_frame, system)
        selected_rows.append(row.to_dict())
        task = base.get_task(manifest, system, str(row["replicate"]))
        snapshot = base.build_spatial_snapshot(task, int(row["frame"]), mode="DNA_linked_component")
        ax = fig.add_subplot(structure[0, index])
        base.draw_spatial_snapshot(ax, snapshot, common_half_span)
        draw_projected_d90_contour(
            ax,
            snapshot,
            float(row["DNA_linked_bead_all_q90_nm"]),
            show_key=index == 0,
        )
        ax.set_title("")
        ax.text(
            0.03,
            0.965,
            f"{label} | $d_{{90}}$={row['DNA_linked_bead_all_q90_nm']:.1f} nm",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
            zorder=6,
        )
        if index == 0:
            draw_panel_label(ax, "D", x=-0.115, y=1.01)
    base.save_figure(fig, out / "fig3_DNA_centered_spatial_reach")
    return pd.DataFrame(selected_rows)


def draw_split_med1_sequence_model(ax: plt.Axes) -> None:
    """Show the Split-2 perturbation on the same MED1 map used in Fig. 1A."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    colors = ["#2A9D8F", "#457B9D", "#8067B7", "#D08C60"]
    left, right = 0.26, 0.96
    cut_after = 807
    cut_x = left + (right - left) * cut_after / base.MED1_LENGTH
    gap = 0.030
    full_y, split_y = 0.66, 0.25

    ax.text(left, 0.98, "MED1 (+31 e)  ·  1,581 aa", ha="left", va="top", fontsize=8.2, fontweight="bold")
    ax.text(0.01, full_y, "10% MED1", ha="left", va="center", fontsize=7.4, fontweight="bold")
    ax.text(0.01, split_y, "Split-2", ha="left", va="center", fontsize=7.4, fontweight="bold")
    ax.plot([left, right], [full_y, full_y], color="#B8BEC7", lw=6, solid_capstyle="butt", zorder=1)
    ax.plot([left, cut_x - gap], [split_y, split_y], color="#B8BEC7", lw=6, solid_capstyle="butt", zorder=1)
    ax.plot([cut_x + gap, right], [split_y, split_y], color="#B8BEC7", lw=6, solid_capstyle="butt", zorder=1)

    for (start, stop), color in zip(legacy.MED1_NATIVE_REGIONS, colors):
        x_start = left + (right - left) * (start - 1) / (base.MED1_LENGTH - 1)
        x_stop = left + (right - left) * (stop - 1) / (base.MED1_LENGTH - 1)
        width = max(x_stop - x_start, 0.008)
        for y in [full_y, split_y]:
            ax.add_patch(
                plt.Rectangle(
                    (x_start, y - 0.055),
                    width,
                    0.11,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=2,
                )
            )
    ax.annotate(
        "cut after 807",
        xy=(cut_x, split_y + 0.025),
        xytext=(cut_x, 0.46),
        ha="center",
        va="center",
        fontsize=7.0,
        color="#333333",
        arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 0.75, "shrinkA": 3, "shrinkB": 4},
    )
    for offset in [-0.008, 0.008]:
        ax.plot(
            [cut_x + offset - 0.010, cut_x + offset + 0.010],
            [split_y - 0.075, split_y + 0.075],
            color="#333333",
            lw=1.0,
            zorder=4,
        )
    ax.text(left, 0.10, "1", ha="center", va="top", fontsize=6.6)
    ax.text(cut_x - gap, 0.10, "807", ha="right", va="top", fontsize=6.6)
    ax.text(cut_x + gap, 0.10, "808", ha="left", va="top", fontsize=6.6)
    ax.text(right, 0.10, "1581", ha="center", va="top", fontsize=6.6)

    draw_panel_label(ax, "A", x=-0.06, y=1.01)


def draw_split_med1_structures(ax: plt.Axes) -> None:
    """Show equally sized PyMOL views corresponding to the two sequence maps."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    structure_images = [
        MAIN_FIGURES / "fig4_panel_a_med1_full_pymol.png",
        MAIN_FIGURES / "fig4_panel_a_med1_split2_pymol.png",
    ]
    if all(path.exists() for path in structure_images):
        for y, path in zip([0.45, 0.04], structure_images):
            image_ax = ax.inset_axes([0.02, y, 0.96, 0.42])
            image = plt.imread(path)
            foreground = np.any(image[..., :3] < 0.985, axis=2)
            rows, columns = np.where(foreground)
            if len(rows):
                row_pad = max(3, int((rows.max() - rows.min() + 1) * 0.05))
                column_pad = max(3, int((columns.max() - columns.min() + 1) * 0.05))
                image = image[
                    max(0, rows.min() - row_pad) : min(image.shape[0], rows.max() + row_pad + 1),
                    max(0, columns.min() - column_pad) : min(image.shape[1], columns.max() + column_pad + 1),
                ]
            image_ax.imshow(image)
            image_ax.set_axis_off()
    else:
        warnings.warn(
            "Figure 4 PyMOL renders are missing; run render_fig4_panel_a.pml.",
            stacklevel=2,
        )
    draw_panel_label(ax, "B", x=-0.04, y=1.01)


def plot_split_radial_profiles(ax: plt.Axes, histogram: pd.DataFrame) -> None:
    """Reuse the Fig. 3A absolute bead-count profile for Full and Split-2."""
    systems = ["M10O90D1", "M10O90D1-S2"]
    distribution = histogram[
        (histogram["distance_scope"] == "bead")
        & (histogram["protein_type"] == "all")
        & histogram["system"].isin(systems)
        & (histogram["bin_end_nm"] <= 40)
    ].copy()
    distribution["one_nm_bin_start"] = np.floor(
        distribution["bin_start_nm"]
    ).astype(int)
    distribution = (
        distribution.groupby(
            ["system", "replicate", "one_nm_bin_start"],
            as_index=False,
            observed=True,
        )["frame_mean_count"]
        .sum()
        .groupby(["system", "one_nm_bin_start"], as_index=False, observed=True)[
            "frame_mean_count"
        ]
        .mean()
    )
    for system, label, linestyle in [
        ("M10O90D1", "10% MED1", "-"),
        ("M10O90D1-S2", "Split-2", "--"),
    ]:
        sub = distribution[distribution["system"] == system].sort_values(
            "one_nm_bin_start"
        )
        ax.plot(
            sub["one_nm_bin_start"] + 0.5,
            sub["frame_mean_count"],
            color="black",
            linestyle=linestyle,
            lw=1.6,
            label=label,
        )
    ax.set(
        xlabel="Nearest-DNA distance (nm)",
        ylabel="Mean linked protein beads\nper 1 nm bin",
        xlim=(0, 40),
        ylim=(0, 2300),
    )
    styled_legend(
        ax,
        loc="upper right",
        prop=FontProperties(family="DejaVu Sans", size=7.0, weight="normal"),
        handlelength=1.7,
        handletextpad=0.35,
    )
    draw_panel_label(ax, "C", x=-0.17)


def plot_split_neighbor_counts(
    ax: plt.Axes,
    repeat: pd.DataFrame,
    design: pd.DataFrame,
) -> None:
    """Compare absolute direct, second, and third-plus neighbor counts."""
    systems = ["M10O90D1", "M10O90D1-S2"]
    chain_counts = design.set_index("system")["n_physical_protein_chains"]
    metrics = [
        "DNA_direct_physical_fraction",
        "DNA_second_physical_fraction",
        "DNA_third_plus_physical_fraction",
    ]
    color = base.COLORS["DNA"]
    for index, metric in enumerate(metrics):
        values_by_system = []
        for system in systems:
            values = (
                repeat[repeat["system"] == system]
                .set_index("replicate")
                .reindex(base.REPEATS)[metric]
                .to_numpy(float)
                * float(chain_counts.loc[system])
            )
            values_by_system.append(values)
        draw_bar_summary(
            ax,
            index - 0.18,
            values_by_system[0],
            color,
            width=0.30,
            marker="o",
        )
        draw_bar_summary(
            ax,
            index + 0.18,
            values_by_system[1],
            color,
            width=0.30,
            facecolor="white",
            hatch="///",
            marker="D",
        )
    ax.set_xticks(range(3), ["direct", "2nd", "3rd+"])
    ax.set(
        ylabel="Protein chains per frame",
        ylim=(0, 40),
        xlim=(-0.55, 2.55),
    )
    ax.tick_params(axis="x", labelsize=7.2)
    styled_legend(
        ax,
        handles=[
            Patch(facecolor=color, edgecolor=color, alpha=0.90, label="10% MED1"),
            Patch(facecolor="white", edgecolor=color, hatch="///", label="Split-2"),
        ],
        loc="upper right",
        ncols=1,
        prop=FontProperties(family="DejaVu Sans", size=6.8, weight="normal"),
        handlelength=1.2,
        handletextpad=0.35,
    )
    draw_panel_label(ax, "D", x=-0.17)


def plot_split_d90(ax: plt.Axes, repeat: pd.DataFrame) -> None:
    """Reuse the Fig. 3B framewise d90 summary for the cleavage contrast."""
    systems = ["M10O90D1", "M10O90D1-S2"]
    for index, system in enumerate(systems):
        values = (
            repeat[repeat["system"] == system]
            .set_index("replicate")
            .reindex(base.REPEATS)["DNA_linked_bead_all_q90_nm"]
            .to_numpy(float)
        )
        ax.scatter(
            index + replicate_offsets(len(values), 0.035),
            values,
            s=21,
            facecolor="white",
            edgecolor="black",
            marker="o",
            linewidth=0.75,
            zorder=6,
        )
        draw_mean_sd(ax, index, values, "black", marker="o")
    ax.set_xticks(range(2), ["10% MED1", "Split-2"])
    ax.set(ylabel=r"Mean framewise $d_{90}$ (nm)", ylim=(18, 32))
    ax.tick_params(axis="x", labelsize=7.2)
    draw_panel_label(ax, "E", x=-0.20)


def plot_figure4(tables: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    fig = plt.figure(figsize=(7.20, 7.45), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.80, 0.88, 1.14],
        left=0.085,
        right=0.985,
        bottom=0.055,
        top=0.975,
        hspace=0.25,
    )
    top = outer[0, 0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.10)
    draw_split_med1_sequence_model(fig.add_subplot(top[0, 0]))
    draw_split_med1_structures(fig.add_subplot(top[0, 1]))

    data_grid = outer[1, 0].subgridspec(
        1,
        3,
        width_ratios=[1.20, 1.0, 0.88],
        wspace=0.48,
    )
    repeat = tables["mechanism_repeat"]
    plot_split_radial_profiles(
        fig.add_subplot(data_grid[0, 0]),
        tables["mechanism_radial_histogram_repeat"],
    )
    plot_split_neighbor_counts(
        fig.add_subplot(data_grid[0, 1]),
        repeat,
        tables["system_design"],
    )
    plot_split_d90(fig.add_subplot(data_grid[0, 2]), repeat)

    structure_row = outer[2, 0].subgridspec(
        2,
        1,
        height_ratios=[0.07, 0.93],
        hspace=0,
    )
    structure = structure_row[1, 0].subgridspec(1, 2, wspace=0.025)
    selected = select_matched_mechanism_snapshot(tables["mechanism_framewise"])
    manifest = tables["trajectory_manifest"]
    common_half_span = max(
        float(manifest.loc[manifest["system"].isin(SPLIT_D1[:2]), "box_nm"].max())
        / 2,
        41.5,
    )
    for index, row in enumerate(selected.itertuples(index=False)):
        task = base.get_task(manifest, row.system, row.replicate)
        snapshot = base.build_spatial_snapshot(
            task,
            int(row.frame),
            mode="DNA_linked_component",
        )
        ax = fig.add_subplot(structure[0, index])
        base.draw_spatial_snapshot(ax, snapshot, common_half_span)
        draw_projected_d90_contour(
            ax,
            snapshot,
            float(row.DNA_linked_bead_all_q90_nm),
            show_key=index == 0,
        )
        ax.set_title("")
        ax.text(
            0.035,
            0.965,
            f"{'10% MED1' if row.condition == 'Full' else row.condition} +DNA | "
            f"$d_{{90}}$={row.DNA_linked_bead_all_q90_nm:.1f} nm",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
            zorder=6,
        )
        if index == 0:
            draw_panel_label(ax, "F", x=-0.035, y=1.01)

    base.save_figure(fig, out / "fig4_MED1_chain_continuity")
    return selected


def main() -> None:
    warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")
    warnings.filterwarnings("ignore", message="Unknown element .*", category=UserWarning)
    tables = load_tables()
    out = MAIN_FIGURES
    out.mkdir(parents=True, exist_ok=True)
    base.configure_style()
    plot_figure1(tables, out)
    fig2_selection = plot_figure2(tables, out)
    fig3_selection = plot_figure3(tables, out)
    fig4_selection = plot_figure4(tables, out)
    selections = []
    for figure, table in [("Fig2", fig2_selection), ("Fig3", fig3_selection), ("Fig4", fig4_selection)]:
        copy = table.copy()
        copy.insert(0, "figure", figure)
        selections.append(copy)
    selection_table = pd.concat(selections, ignore_index=True, sort=False)
    if "frame" in selection_table:
        selection_table["frame"] = pd.to_numeric(
            selection_table["frame"], errors="coerce"
        ).astype("Int64")
    selection_table.to_csv(TABLES / "reframed_main_snapshot_selection.csv", index=False)
    print(f"Wrote four reframed main figures to {out}")


if __name__ == "__main__":
    main()
