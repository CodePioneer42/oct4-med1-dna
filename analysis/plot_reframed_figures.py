#!/usr/bin/env python3
"""Render the four main figures for the chain-continuity reframing."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from scipy.spatial import cKDTree

import run_balanced_analysis as base
import model_rendering as legacy
from publication_style import FONT, OCT4_REGIONS, MED1_REGIONS, CONDITIONS, LAYERS, D90_CONTOUR, DNA_METRIC_COLOR
from publication_style import AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_SIZE, PANEL_LABEL_SIZE


HERE = Path(__file__).resolve().parent
OUTPUT = base.DEFAULT_OUTPUT
TABLES = OUTPUT / "tables"
MAIN_FIGURES = OUTPUT / "figures" / "main"
PRIMARY_DNA_PAIRS = list(base.DNA_PAIRS)
PRIMARY_DNA_SYSTEMS = [pair[1] for pair in PRIMARY_DNA_PAIRS]
ALL_DNA_SYSTEMS = [pair[1] for pair in base.DNA_PAIRS]
ALL_DNA_LABELS = [
    "OCT4-\nonly",
    "10%\nMED1",
    "20%\nMED1",
    "MED1-\nonly",
]
ALL_DNA_COLORS = [
    base.COLORS["OCT4"],
    base.COLORS["mixed"],
    "#9B6D00",
    base.COLORS["MED1"],
]
DNA_CONFORMATION_SYSTEMS = ["D1", *ALL_DNA_SYSTEMS]
DNA_CONFORMATION_LABELS = ["DNA\nonly", "OCT4-\nonly", "10%\nMED1", "20%\nMED1", "MED1-\nonly"]
DNA_CONFORMATION_COLORS = [base.COLORS["DNA"], *ALL_DNA_COLORS]
FIG2_DNA_LABELS = ["OCT4-\nonly", "10%\nMED1", "20%\nMED1", "MED1-\nonly"]
# DNA conditions use unfilled/forest-green bars, separate from protein colors.
FIG2_CONDITION_COLORS = CONDITIONS
SPLIT_D0 = ["M10O90", "M10O90-S2", "M10O90-S4"]
SPLIT_D1 = ["M10O90D1", "M10O90D1-S2", "M10O90D1-S4"]
SPLIT_LABELS = ["Full", "Split-2", "Split-4"]
LEGEND_FONT = FontProperties(family=FONT, size=LEGEND_SIZE, weight="normal")
D90_CONTOUR_COLOR = D90_CONTOUR
OCT4_REGION_COLORS = OCT4_REGIONS
MED1_REGION_COLORS = MED1_REGIONS
# Physical clear space between row contents, including ticks and axis labels.
# Panel letters sit in the left gutter and do not add vertical padding.
# Reuse pack_figure_rows for subsequent figures; change this value to tune spacing.
FIGURE_ROW_GAP_PT = 6.0
FIG1_STRUCTURE_GAP_PT = 6.0
# Helvetica-compatible regular/bold fonts reproduce the reference figure's
# typography while allowing Unicode labels and embedded, portable PDF text.
FIG1_FONT = FONT


def apply_reference_typography(fig: plt.Figure) -> None:
    """Keep the approved Figure 1 font and size hierarchy across main figures."""
    for text in fig.findobj(Text):
        text.set_fontfamily(FONT)
        if text.get_gid() == "panel-letter":
            text.set_fontsize(PANEL_LABEL_SIZE)
            text.set_fontweight("bold")
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
        ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=0.65, length=3.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
    for legend in fig.findobj(Legend):
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_SIZE)
            text.set_fontweight("normal")


def pack_figure_columns(
    fig: plt.Figure,
    groups: list[list[plt.Axes]],
    width_ratios: list[float],
    *,
    left: float = 0.015,
    right: float = 0.985,
    margin_pt: float = 18.0,
    gap_pt: float = 12.0,
) -> None:
    """Place columns using the space actually occupied by their visible labels."""
    point_width = 72.0 * fig.get_figwidth()
    row_left = left + margin_pt / point_width
    row_right = right - margin_pt / point_width
    gap = gap_pt / point_width
    ratios = np.asarray(width_ratios, dtype=float)
    # A shorter plotting width can make an x label protrude; remeasure after
    # each placement so that the final gaps include these labels as well.
    for _ in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        extents = []
        for group in groups:
            box = Bbox.union([_axes_content_bbox(ax, renderer) for ax in group])
            box = box.transformed(fig.transFigure.inverted())
            position = group[0].get_position()
            extents.append((position.x0 - box.x0, box.x1 - position.x1))
        available = row_right - row_left - gap * (len(groups) - 1) - sum(sum(e) for e in extents)
        widths = available * ratios / ratios.sum()
        if np.any(widths <= 0):
            raise ValueError("Figure columns do not fit their typography")
        content_left = row_left
        for group, width, (left_extent, right_extent) in zip(groups, widths, extents):
            for ax in group:
                position = ax.get_position()
                ax.set_position([content_left + left_extent, position.y0, width, position.height])
            content_left += left_extent + width + right_extent + gap
    # Keep the first panel letter in the same gutter as the snapshot row.
    first = groups[0][0]
    for text in first.texts:
        if text.get_gid() == "panel-letter":
            position = first.get_position()
            text.set_x((left - position.x0) / position.width)


def _axes_content_bbox(ax: plt.Axes, renderer) -> Bbox:
    """Measure visible content without counting empty schematic axes as content."""
    if ax.axison:
        extra_artists = [
            artist for artist in ax.get_default_bbox_extra_artists()
            if artist.get_gid() != "panel-letter"
        ]
        return ax.get_tightbbox(renderer, bbox_extra_artists=extra_artists)
    boxes = []
    for artist in [*ax.texts, *ax.lines, *ax.patches, *ax.collections, *ax.images]:
        if artist.get_visible() and artist.get_gid() != "panel-letter":
            box = artist.get_tightbbox(renderer)
            if box is not None and np.isfinite(box.extents).all():
                boxes.append(box)
    boxes.extend(_axes_content_bbox(child, renderer) for child in ax.child_axes)
    return Bbox.union(boxes) if boxes else ax.get_window_extent(renderer)


def pack_figure_rows(
    fig: plt.Figure,
    rows: list[list[plt.Axes]],
    *,
    gap_pt: float | None = None,
    resize: bool = False,
) -> None:
    """Pack completed rows with a fixed gap in points, preserving panel sizes.

    Call after drawing labels and inset axes. Supply all axes in each row,
    including twins. With resize=True, also trim unused vertical canvas space.
    """
    gap_inches = (FIGURE_ROW_GAP_PT if gap_pt is None else gap_pt) / 72.0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        Bbox.union([_axes_content_bbox(ax, renderer) for ax in row]).transformed(
            fig.dpi_scale_trans.inverted()
        )
        for row in rows
    ]
    positions = [[(ax, ax.get_position().frozen()) for ax in row] for row in rows]
    old_height = fig.get_figheight()
    new_height = old_height
    row_top = boxes[0].y1
    if resize:
        margin_inches = 0.03
        new_height = (
            sum(box.height for box in boxes)
            + gap_inches * (len(rows) - 1)
            + 2 * margin_inches
        )
        row_top = new_height - margin_inches
        fig.set_size_inches(fig.get_figwidth(), new_height, forward=True)
    for box, row_positions in zip(boxes, positions):
        row_bottom = row_top - box.height
        shift = row_bottom - box.y0
        for ax, position in row_positions:
            ax.set_position(
                [
                    position.x0,
                    (position.y0 * old_height + shift) / new_height,
                    position.width,
                    position.height * old_height / new_height,
                ]
            )
        row_top = row_bottom - gap_inches
    fig.canvas.draw()


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
        "DNA_conformation_with_D1_by_replicate",
    ]
    tables = {name: pd.read_csv(TABLES / f"{name}.csv") for name in names}
    base.validate_sampling(tables["balanced_framewise_network_contacts_DNA"])
    base.validate_sampling(tables["mechanism_framewise"])
    if not tables["DNA_conformation_with_D1_by_replicate"]["n_frames"].eq(base.N_SAMPLED_FRAMES).all():
        raise RuntimeError("DNA conformation tables have inconsistent sampling")
    return tables


def replicate_offsets(n: int, width: float = 0.035) -> np.ndarray:
    return np.linspace(-width, width, n) if n > 1 else np.zeros(1)


def draw_x_axis_break(ax: plt.Axes, x: float, *, linewidth: float = 1.0) -> None:
    """Mark the compressed interval before the separate MED1-only reference."""
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
            linewidth=linewidth,
            clip_on=False,
            zorder=10,
        )


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
        markeredgecolor="black",
        markeredgewidth=0.8,
        elinewidth=0.75,
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
    align_x_fig: float | None = None,
) -> None:
    """Add a panel letter without creating a subplot title."""
    if align_x_fig is not None:
        position = ax.get_position()
        x = (align_x_fig - position.x0) / position.width
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        clip_on=False,
        gid="panel-letter",
    )


def draw_projected_d90_contour(
    ax: plt.Axes,
    snapshot: dict[str, object],
    d90_nm: float,
    *,
    show_key: bool = False,
    key_label: str = r"projected $d_{90}$ contour",
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

    # A thin white underlay separates the black geometric contour from beads.
    ax.contour(
        xx,
        yy,
        projected_distance,
        levels=[d90_nm],
        colors=["white"],
        linewidths=1.65,
        alpha=0.90,
        zorder=5.1,
    )
    ax.contour(
        xx,
        yy,
        projected_distance,
        levels=[d90_nm],
        colors=[D90_CONTOUR_COLOR],
        linewidths=0.85,
        linestyles=[(0, (3.0, 2.0))],
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
                    lw=0.85,
                    ls=(0, (3.0, 2.0)),
                    label=key_label,
                )
            ],
            loc="lower right",
            bbox_to_anchor=(0.985, 0.02),
            borderaxespad=0,
            handlelength=1.7,
            handletextpad=0.35,
            prop=FontProperties(family=FONT, size=7.0),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.84,
        )
        key.set_zorder(7)


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
        edgecolor="black",
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
    """Display OCT4 and MED1 side by side at a common spatial magnification."""
    pymol_images = [
        MAIN_FIGURES / "fig1_panel_b_oct4_pymol.png",
        MAIN_FIGURES / "fig1_panel_b_med1_pymol.png",
    ]
    scale_metadata = MAIN_FIGURES / "fig1_panel_b_scale.json"
    if all(path.exists() for path in pymol_images) and scale_metadata.exists():
        metadata = json.loads(scale_metadata.read_text())
        views = np.asarray([metadata["views"][protein] for protein in ["oct4", "med1"]])
        if not metadata["orthoscopic"] or not np.allclose(
            views[:, [9, 10, 11, 17]], views[0, [9, 10, 11, 17]]
        ):
            raise ValueError("Fig.1B protein renders must use the same orthographic camera scale")
        images = [plt.imread(path) for path in pymol_images]
        width, height = metadata["image_size_px"]
        if any(image.shape[:2] != (height, width) for image in images):
            raise ValueError("Fig.1B protein renders must have the same pixel dimensions")

        # Crop background independently, then use ONE inches-per-source-pixel
        # factor for both panels. Equal-sized axes with separate crops would
        # incorrectly enlarge OCT4; proportional axes preserve the camera scale.
        cropped_images = []
        padding_px = 8
        for image in images:
            rows, columns = np.where(np.any(image[..., :3] < 0.985, axis=2))
            cropped_images.append(
                image[
                    max(0, rows.min() - padding_px):min(height, rows.max() + padding_px + 1),
                    max(0, columns.min() - padding_px):min(width, columns.max() + padding_px + 1),
                ]
            )
        position = ax.get_position()
        panel_width = position.width * ax.figure.get_figwidth()
        panel_height = position.height * ax.figure.get_figheight()
        image_height = panel_height - 13.0 / 72.0  # protein-name strip
        gap_inches = FIG1_STRUCTURE_GAP_PT / 72.0
        inches_per_pixel = min(
            (panel_width - gap_inches) / sum(image.shape[1] for image in cropped_images),
            image_height / max(image.shape[0] for image in cropped_images),
        )
        group_width = (
            sum(image.shape[1] for image in cropped_images) * inches_per_pixel + gap_inches
        )
        left = (panel_width - group_width) / 2
        for name, image in zip(["OCT4", "MED1"], cropped_images):
            display_width = image.shape[1] * inches_per_pixel
            display_height = image.shape[0] * inches_per_pixel
            bounds = [
                left / panel_width,
                (image_height - display_height) / (2 * panel_height),
                display_width / panel_width,
                display_height / panel_height,
            ]
            image_ax = ax.inset_axes(bounds)
            image_ax.imshow(image)
            image_ax.set_axis_off()
            ax.text(
                (left + display_width / 2) / panel_width,
                0.97,
                name,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.4,
            )
            left += display_width + gap_inches
        ax.set_axis_off()
        return

    warnings.warn(
        "Shared-scale PyMOL renders missing; using the deterministic PCA fallback. "
        "Run render_fig1_panel_b.pml with PyMOL to replace it.",
        stacklevel=2,
    )
    image_bounds = [(0.0, 0.0, 0.5, 0.92), (0.5, 0.0, 0.5, 0.92)]
    for x, name in [(0.25, "OCT4"), (0.75, "MED1")]:
        ax.text(x, 0.97, name, transform=ax.transAxes, ha="center", va="top", fontsize=8.4)
    image_axes = []
    projected_models = []
    for bounds, coordinates, regions, colors in zip(
        image_bounds,
        [oct4_coordinates, med1_coordinates],
        [legacy.OCT4_NATIVE_REGIONS, legacy.MED1_NATIVE_REGIONS],
        [oct4_colors, med1_colors],
    ):
        image_ax = ax.inset_axes(bounds)
        projected_models.append(
            _draw_pdb_model_fallback(image_ax, coordinates, regions, colors, y_offset=0.0)
        )
        image_axes.append(image_ax)
    lower = np.min(np.vstack(projected_models), axis=0)
    upper = np.max(np.vstack(projected_models), axis=0)
    padding = np.maximum((upper - lower) * 0.08, 1.0)
    for image_ax in image_axes:
        image_ax.set_xlim(float(lower[0] - padding[0]), float(upper[0] + padding[0]))
        image_ax.set_ylim(float(lower[1] - padding[1]), float(upper[1] + padding[1]))
        image_ax.set_aspect("equal", adjustable="box")
        image_ax.set_axis_off()
    scale_x = float(lower[0] + 0.04 * (upper[0] - lower[0]))
    scale_y = float(lower[1] + 0.07 * (upper[1] - lower[1]))
    image_axes[0].plot([scale_x, scale_x + 10.0], [scale_y, scale_y], color="#111111", lw=1.5, zorder=3)
    image_axes[0].text(scale_x + 5.0, scale_y + 1.0, "10 nm", ha="center", va="bottom", fontsize=7.0)
    ax.set_axis_off()


def draw_sequence_region_map(
    ax: plt.Axes,
    name: str,
    length: int,
    charge: str,
    regions: list[tuple[int, int]],
    region_colors: list[str],
    residue_ticks: list[int],
    tick_alignments: dict[int, str] | None = None,
) -> None:
    """Draw a thin sequence bar with residue boundary ticks on one baseline."""
    # Layout reference: Boija et al., Cell (2018), Figure 4A.
    # https://doi.org/10.1016/j.cell.2018.10.042
    left, right = 0.045, 0.965
    bottom, height = 0.46, 0.18
    span = right - left
    ax.add_patch(
        plt.Rectangle(
            (left, bottom),
            span,
            height,
            transform=ax.transAxes,
            facecolor="#F0F0F0",
            edgecolor="none",
        )
    )
    for (start, end), color in zip(regions, region_colors):
        # Inclusive, one-based residue intervals occupy their exact fraction
        # of the sequence; every partition has the same height.
        x_start = left + span * (start - 1) / length
        width = span * (end - start + 1) / length
        ax.add_patch(
            plt.Rectangle(
                (x_start, bottom),
                width,
                height,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="black",
                linewidth=0.45,
            )
        )
    ax.add_patch(
        plt.Rectangle(
            (left, bottom),
            span,
            height,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="#666666",
            linewidth=0.65,
        )
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
    tick_alignments = tick_alignments or {}
    region_ends = {end for _, end in regions}
    for residue in residue_ticks:
        # Put ticks on partition edges, including the full sequence termini.
        edge = residue if residue in region_ends or residue == length else residue - 1
        x = left + span * edge / length
        ax.plot(
            [x, x],
            [bottom, bottom - 0.035],
            transform=ax.transAxes,
            color="#666666",
            lw=0.55,
        )
        ax.text(
            x,
            bottom - 0.055,
            str(residue),
            transform=ax.transAxes,
            ha=tick_alignments.get(residue, "center"),
            va="top",
            fontsize=7.0,
        )
    ax.set_axis_off()


def draw_original_object_panels(fig: plt.Figure, spec) -> tuple[float, list[plt.Axes]]:
    """Draw the two protein maps and their corresponding input conformations."""
    med1_colors = MED1_REGION_COLORS
    oct4_colors = OCT4_REGION_COLORS
    med1_coordinates, _ = legacy.read_pdb_bead_coordinates(
        base.PACKAGE_ROOT / "inputs/MED1-alphafold.pdb", "CA"
    )
    oct4_coordinates, _ = legacy.read_pdb_bead_coordinates(
        base.PACKAGE_ROOT / "inputs/OCT4.pdb", "CA"
    )
    med1_charge = sum(
        legacy.AA_CHARGE.get(name, 0.0)
        for name in legacy.read_residue_names(
            base.PACKAGE_ROOT / "inputs/MED1-alphafold.pdb"
        )
    )
    oct4_charge = sum(
        legacy.AA_CHARGE.get(name, 0.0)
        for name in legacy.read_residue_names(base.PACKAGE_ROOT / "inputs/OCT4.pdb")
    )

    # One compact sequence-map column followed by two equal structure columns.
    grid = spec.subgridspec(1, 3, width_ratios=[1.08, 1.0, 1.0], wspace=0.025)
    # Give both protein maps equal halves of the row, avoiding empty space
    # below a tightly packed schematic block while the structures fill the row.
    sequence_grid = grid[0, 0].subgridspec(2, 1, hspace=0.0)
    sequence_axes = [fig.add_subplot(sequence_grid[row, 0]) for row in range(2)]
    conformation_ax = fig.add_subplot(grid[0, 1:])
    draw_sequence_region_map(
        sequence_axes[0],
        "OCT4",
        base.OCT4_LENGTH,
        f"{oct4_charge:+g} e".replace("-", "−"),
        legacy.OCT4_NATIVE_REGIONS,
        oct4_colors,
        [1, 136, 217, 234, 286, 360],
        tick_alignments={217: "right", 234: "left", 286: "left"},
    )
    draw_sequence_region_map(
        sequence_axes[1],
        "MED1",
        base.MED1_LENGTH,
        f"{med1_charge:+g} e",
        legacy.MED1_NATIVE_REGIONS,
        med1_colors,
        [1, 77, 518, 1581],
        tick_alignments={1: "right", 77: "left"},
    )
    draw_input_conformations(
        conformation_ax,
        med1_coordinates,
        oct4_coordinates,
        med1_colors,
        oct4_colors,
    )
    a_label_x_fig = (
        sequence_axes[0].get_position().x0
        - 0.12 * sequence_axes[0].get_position().width
    )
    label_offset = (
        0.02 * conformation_ax.get_position().height / sequence_axes[0].get_position().height
    )
    draw_panel_label(sequence_axes[0], "A", x=-0.12, y=1.0 + label_offset)
    draw_panel_label(conformation_ax, "B", x=0.14, y=1.02)
    return a_label_x_fig, [*sequence_axes, conformation_ax]


def plot_figure1(tables: dict[str, pd.DataFrame], out: Path) -> None:
    """Use the reference's Helvetica style without changing other figures."""
    with plt.rc_context({
        "font.family": FIG1_FONT,
        "mathtext.rm": FIG1_FONT,
        "mathtext.it": FIG1_FONT + ":italic",
        "mathtext.bf": FIG1_FONT + ":bold",
        "mathtext.cal": FIG1_FONT,
        "pdf.use14corefonts": False,
        "pdf.fonttype": 42,
    }):
        _plot_figure1(tables, out)


def _plot_figure1(tables: dict[str, pd.DataFrame], out: Path) -> None:
    cluster_group = tables["balanced_cluster_group"]
    cluster_group = cluster_group[np.isclose(cluster_group["cutoff_nm"], base.PRIMARY_CUTOFF_NM)].set_index("system")
    cluster_repeat = tables["balanced_cluster_repeat"]
    cluster_repeat = cluster_repeat[np.isclose(cluster_repeat["cutoff_nm"], base.PRIMARY_CUTOFF_NM)]
    frame_group = tables["balanced_frame_group"].set_index("system")
    frame_repeat = tables["balanced_frame_repeat"]
    design = tables["system_design"]
    x_map = base.composition_x(design)
    figure1_systems = [*base.COMPOSITION_SYSTEMS, "M50O0"]
    med1_only_x = 62.0
    x_map["M50O0"] = med1_only_x
    x = np.asarray([x_map[system] for system in figure1_systems])

    fig = plt.figure(figsize=(7.20, 6.11), constrained_layout=False)
    # Stage panel sizes here; pack_figure_rows sets the final physical gaps.
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.85, 1.87],
        left=0.055,
        right=0.99,
        bottom=0.045,
        top=0.982,
        hspace=0.12,
    )
    aligned_panel_x, object_axes = draw_original_object_panels(fig, outer[0, 0])
    results_grid = outer[1, 0].subgridspec(
        2,
        1,
        height_ratios=[0.78, 1.10],
        hspace=0.22,
    )
    metric_grid = results_grid[0, 0].subgridspec(1, 2, wspace=0.47)
    axes = [fig.add_subplot(metric_grid[0, col]) for col in range(2)]

    ax = axes[0]
    values = cluster_group.loc[figure1_systems, "largest_cluster_fraction_mean"].to_numpy(float)
    errors = cluster_group.loc[figure1_systems, "largest_cluster_fraction_sd"].to_numpy(float)
    free_values = cluster_group.loc[figure1_systems, "monomer_fraction_mean"].to_numpy(float)
    free_errors = cluster_group.loc[figure1_systems, "monomer_fraction_sd"].to_numpy(float)
    ax.plot(x, values, color="black", lw=1.05, zorder=1)
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="black", elinewidth=0.75, capsize=2.8, zorder=2)
    ax.scatter(x, values, s=24, color="black", edgecolor="black", linewidth=0.55, zorder=4)
    for index, system in enumerate(figure1_systems):
        repeat_values = cluster_repeat[cluster_repeat["system"] == system]["largest_cluster_fraction"].to_numpy(float)
        ax.scatter(x[index] + replicate_offsets(len(repeat_values), 0.28), repeat_values, s=14, facecolor="white", edgecolor="black", linewidth=0.65, zorder=5)
    tick_positions = [0, 10, 20, 30, 40, 50, med1_only_x]
    tick_labels = ["OCT4-\nonly", "10", "20", "30", "40", "50", "MED1-\nonly"]
    ax.set_xticks(tick_positions, tick_labels)
    ax.set(xlabel="MED1 molecule fraction (%)", ylabel="LCC fraction", xlim=(-2, 68), ylim=(0, 1.04))
    ax.tick_params(axis="x", labelsize=7.0, pad=2)
    free_ax = ax.twinx()
    free_ax.plot(x, free_values, color="black", lw=1.0, ls="--", zorder=1)
    free_ax.errorbar(x, free_values, yerr=free_errors, fmt="none", ecolor="black", elinewidth=0.75, capsize=2.8, zorder=2)
    free_ax.scatter(x, free_values, s=25, facecolor="white", edgecolor="black", marker="s", linewidth=0.65, zorder=4)
    for index, system in enumerate(figure1_systems):
        repeat_values = cluster_repeat[cluster_repeat["system"] == system]["monomer_fraction"].to_numpy(float)
        free_ax.scatter(x[index] + replicate_offsets(len(repeat_values), 0.28), repeat_values, s=14, facecolor="white", edgecolor="black", marker="s", linewidth=0.65, zorder=5)
    free_ax.set(ylabel="Free-chain fraction", ylim=(0, 0.38))
    free_ax.yaxis.labelpad = 5.5
    free_ax.spines["right"].set_visible(True)
    draw_x_axis_break(free_ax, 56.0, linewidth=0.65)
    draw_x_axis_break(ax, 56.0, linewidth=0.65)
    curve_legend_style = {
        "prop": FontProperties(family=FIG1_FONT, size=8.4, weight="normal"),
        "frameon": False,
        "borderaxespad": 0,
        "handlelength": 1.4,
        "handletextpad": 0.4,
    }
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.05, marker="o", markersize=4.5, label="LCC"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.08, 0.33),
        **curve_legend_style,
    )
    free_ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.0, ls="--", marker="s", markerfacecolor="white", markersize=4.5, label="Free chain"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.08, 0.89),
        **curve_legend_style,
    )
    ax = axes[1]
    for metric, molecule, color in [
        ("OCT4_mean_degree", "OCT4", base.COLORS["OCT4"]),
        ("MED1_mean_degree", "MED1", base.COLORS["MED1"]),
    ]:
        systems = [system for system in figure1_systems if np.isfinite(frame_group.loc[system, f"{metric}_mean"])]
        means = [frame_group.loc[system, f"{metric}_mean"] for system in systems]
        sds = [frame_group.loc[system, f"{metric}_sd"] for system in systems]
        positions = [x_map[system] for system in systems]
        ax.plot(positions, means, color=color, lw=1.05, zorder=1)
        ax.errorbar(positions, means, yerr=sds, fmt="none", ecolor=color, elinewidth=0.75, capsize=2.8, zorder=2)
        ax.scatter(
            positions,
            means,
            s=24,
            color=color,
            edgecolor="black",
            linewidth=0.55,
            label=f"Partners per {molecule} chain",
            zorder=4,
        )
        for system in systems:
            repeat_values = frame_repeat[frame_repeat["system"] == system][metric].dropna().to_numpy(float)
            ax.scatter(x_map[system] + replicate_offsets(len(repeat_values), 0.24), repeat_values, s=13, facecolor="white", edgecolor=color, linewidth=0.6, zorder=5)
    ax.set_xticks(tick_positions, tick_labels)
    ax.set(xlabel="MED1 molecule fraction (%)", ylabel="Partners per chain", xlim=(-2, 68), ylim=(0.78, 4.15))
    ax.tick_params(axis="x", labelsize=7.0, pad=2)
    draw_x_axis_break(ax, 56.0, linewidth=0.65)
    oct4_legend = ax.legend(
        handles=[
            Line2D([0], [0], color=base.COLORS["OCT4"], lw=1.05, marker="o", markersize=4.5, label="Partners per OCT4 chain"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.08, 0.09),
        **curve_legend_style,
    )
    ax.add_artist(oct4_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color=base.COLORS["MED1"], lw=1.05, marker="o", markersize=4.5, label="Partners per MED1 chain"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.08, 0.84),
        **curve_legend_style,
    )

    # Follow the reference's light axes and larger, consistent sans-serif labels.
    for metric_ax in [*axes, free_ax]:
        metric_ax.tick_params(axis="both", labelsize=8.5, width=0.65, length=3.0)
        metric_ax.xaxis.label.set_size(10.0)
        metric_ax.yaxis.label.set_size(10.0)
        for spine in metric_ax.spines.values():
            spine.set_linewidth(0.65)

    # Center the complete C/D content, including the two y axes in C. Equal
    # plotting widths, 10 pt between labels, and 18 pt at both outer edges avoid
    # crowding the canvas or reserving excess space between these panels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    row_groups = [[axes[0], free_ax], [axes[1]]]
    side_extents = []
    for metric_ax, group in zip(axes, row_groups):
        content = Bbox.union([_axes_content_bbox(item, renderer) for item in group])
        content = content.transformed(fig.transFigure.inverted())
        position = metric_ax.get_position()
        side_extents.append((position.x0 - content.x0, content.x1 - position.x1))
    margin = 18.0 / (72.0 * fig.get_figwidth())
    gap = 10.0 / (72.0 * fig.get_figwidth())
    row_left, row_right = aligned_panel_x + margin, outer.right - margin
    plot_width = (row_right - row_left - gap - sum(sum(extent) for extent in side_extents)) / 2
    content_left = row_left
    for metric_ax, group, (left_extent, right_extent) in zip(axes, row_groups, side_extents):
        position = metric_ax.get_position()
        for item in group:
            item.set_position([content_left + left_extent, position.y0, plot_width, position.height])
        content_left += left_extent + plot_width + right_extent + gap
    draw_panel_label(axes[0], "C", align_x_fig=aligned_panel_x)
    draw_panel_label(axes[1], "D", x=-0.11)

    structure_grid = results_grid[1, 0].subgridspec(1, 4, wspace=0.0)
    manifest = tables["trajectory_manifest"]
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    structure_systems = [
        ("M0O100", "OCT4-only"),
        ("M10O90", "10% MED1"),
        ("M50O50", "50% MED1"),
        ("M50O0", "MED1-only"),
    ]
    snapshot_axes = []
    for index, (system, label) in enumerate(structure_systems):
        selection = base.select_snapshot(framewise, system)
        task = base.get_task(manifest, system, str(selection["replicate"]))
        snapshot = base.build_spatial_snapshot(task, int(selection["frame"]), mode="full_system")
        ax = fig.add_subplot(structure_grid[0, index])
        snapshot_axes.append(ax)
        base.draw_spatial_snapshot(
            ax,
            snapshot,
            task.box_nm / 2,
            scale_bar_inset_nm=0.09 * task.box_nm,
            show_scale_bar=index == 0,
        )
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
            draw_panel_label(ax, "E", y=1.01, align_x_fig=aligned_panel_x)
    pack_figure_rows(fig, [object_axes, [*axes, free_ax], snapshot_axes], resize=True)
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
        facecolor=to_rgba(bar_facecolor, 0.90),
        edgecolor="black",
        linewidth=1.05,
        hatch=hatch,
        zorder=1,
    )
    ax.errorbar(
        x,
        float(np.mean(values)),
        yerr=float(np.std(values, ddof=1)),
        fmt="none",
        color="black",
        elinewidth=1.05,
        capsize=2.4,
        zorder=3,
    )
    ax.scatter(
        x + replicate_offsets(len(values), 0.075),
        values,
        s=19,
        facecolor="white",
        edgecolor="#333333",
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
        draw_bar_summary(ax, index - 0.16, sub["without_DNA"].to_numpy(float), FIG2_CONDITION_COLORS["without_DNA"])
        draw_bar_summary(ax, index + 0.16, sub["with_DNA"].to_numpy(float), FIG2_CONDITION_COLORS["with_DNA"])
    ax.set_xticks(np.arange(len(PRIMARY_DNA_PAIRS)), FIG2_DNA_LABELS)
    ax.tick_params(axis="x", labelsize=7.0, pad=1.0)
    ax.set(ylabel=ylabel, ylim=ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    draw_panel_label(ax, panel, x=-0.14)


def plot_partner_species_rows(
    med1_ax: plt.Axes,
    oct4_ax: plt.Axes,
    repeat: pd.DataFrame,
) -> None:
    """Show DNA effects on each chain species in aligned upper and lower rows."""
    partner_ylim = (0, 7.0)
    partner_yticks = np.arange(0, 7, 2)
    for ax, metric, ylabel in [
        (med1_ax, "MED1_protein_mean_degree", "Partners per MED1 chain"),
        (oct4_ax, "OCT4_protein_mean_degree", "Partners per OCT4 chain"),
    ]:
        for index, (without, with_dna, _) in enumerate(PRIMARY_DNA_PAIRS):
            without_values = (
                repeat[repeat["system"] == without]
                .set_index("replicate")
                .reindex(base.REPEATS)[metric]
                .to_numpy(float)
            )
            with_values = (
                repeat[repeat["system"] == with_dna]
                .set_index("replicate")
                .reindex(base.REPEATS)[metric]
                .to_numpy(float)
            )
            if not np.isfinite(without_values).all() and not np.isfinite(with_values).all():
                continue
            if not np.isfinite(without_values).all() or not np.isfinite(with_values).all():
                raise RuntimeError(
                    f"Incomplete paired partner values for {without}/{with_dna} {metric}"
                )

            without_x, with_x = index - 0.14, index + 0.14
            for x_value, values, color in [
                (without_x, without_values, FIG2_CONDITION_COLORS["without_DNA"]),
                (with_x, with_values, FIG2_CONDITION_COLORS["with_DNA"]),
            ]:
                mean = float(values.mean())
                sd = float(values.std(ddof=1))
                ax.bar(
                    x_value,
                    mean,
                    width=0.21,
                    facecolor=to_rgba(color, 0.90),
                    edgecolor="black",
                    linewidth=1.05,
                    zorder=2,
                )
                ax.errorbar(
                    x_value,
                    mean,
                    yerr=sd,
                    fmt="none",
                    color="black",
                    elinewidth=1.05,
                    capsize=2.0,
                    zorder=3,
                )
                ax.scatter(
                    x_value + replicate_offsets(len(values), 0.045),
                    values,
                    s=12,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=4,
                )

        ax.set(
            ylim=partner_ylim,
            xlim=(-0.55, len(PRIMARY_DNA_PAIRS) - 0.45),
            yticks=partner_yticks,
            ylabel=ylabel.replace(" per ", " per\n"),
        )
        ax.yaxis.label.set_color("black")
        ax.yaxis.label.set_size(7.4)
        ax.yaxis.label.set_rotation(90)
        ax.yaxis.label.set_horizontalalignment("center")
        ax.yaxis.label.set_verticalalignment("center")
        ax.yaxis.label.set_linespacing(1.05)
        ax.yaxis.set_label_coords(-0.135, 0.5)
        ax.tick_params(axis="y", colors="black", labelsize=7.0, length=2.0, pad=2.0)
        ax.set_xticks(np.arange(len(PRIMARY_DNA_PAIRS)), FIG2_DNA_LABELS)
        ax.tick_params(axis="x", colors="black", labelsize=7.0, length=2.0, pad=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    med1_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    draw_panel_label(med1_ax, "B", x=-0.20)

def plot_dna_association_bars(ax: plt.Axes, repeat: pd.DataFrame) -> None:
    for index, system in enumerate(PRIMARY_DNA_SYSTEMS):
        sub = repeat[repeat["system"] == system].set_index("replicate").reindex(base.REPEATS)
        for offset, metric, color, marker in zip(
            [-0.27, 0, 0.27],
            ["DNA_direct_physical_fraction", "DNA_second_physical_fraction", "DNA_third_plus_physical_fraction"],
            LAYERS, ["o", "s", "^"]):
            draw_bar_summary(ax, index + offset, sub[metric].to_numpy(float),
                             color, width=0.22, marker=marker)
    ax.set_xticks(range(len(PRIMARY_DNA_SYSTEMS)), FIG2_DNA_LABELS)
    ax.tick_params(axis="x", labelsize=7.0, pad=1.0)
    ax.set(ylabel="Fraction of protein chains", ylim=(0, 0.8), yticks=np.linspace(0, 0.8, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    styled_legend(ax, handles=[Patch(facecolor=to_rgba(c, 0.90), edgecolor="black", linewidth=1.05, label=label)
        for c, label in zip(LAYERS, ["Direct", "2nd", "3rd+"])],
        loc="upper left", prop=FontProperties(family=FONT, size=7.0),
        handlelength=1.2, handletextpad=0.35)
    draw_panel_label(ax, "C", x=-0.17)


def plot_dna_rg_line(
    ax: plt.Axes,
    table: pd.DataFrame,
) -> None:
    x = np.arange(len(DNA_CONFORMATION_SYSTEMS), dtype=float)
    axes = [
        (
            ax, "DNA_Rg_nm", "black", "o", r"DNA $R_g$ (nm)", (16.4, 18.15),
            "-", [16.5, 17.0, 17.5, 18.0], r"DNA $R_g$", "upper left", (0.07, 0.985),
        ),
        (
            ax.twinx(),
            "DNA_Ree_nm",
            "black",
            "s",
            "DNA end-to-end distance (nm)",
            (49.0, 66.5),
            "--",
            [50.0, 55.0, 60.0, 65.0],
            "DNA end-to-end distance",
            "lower right",
            (0.98, 0.045),
        ),
    ]
    for axis, metric, color, marker, ylabel, ylim, linestyle, yticks, legend_label, legend_loc, legend_anchor in axes:
        values_array = np.asarray(
            [
                table[table["system"] == system]
                .set_index("replicate")
                .reindex(base.REPEATS)[metric]
                .to_numpy(float)
                for system in DNA_CONFORMATION_SYSTEMS
            ],
            dtype=float,
        )
        means = values_array.mean(axis=1)
        sds = values_array.std(axis=1, ddof=1)
        axis.plot(x, means, color=color, lw=1.45, ls=linestyle, zorder=1)
        axis.errorbar(x, means, yerr=sds, fmt="none", color=color, elinewidth=1.05, capsize=2.3, zorder=2)
        mean_facecolor = DNA_METRIC_COLOR if marker == "o" else "white"
        axis.scatter(x, means, s=26, marker=marker, facecolor=mean_facecolor, edgecolor="black", linewidth=0.75, zorder=4)
        for index, values in enumerate(values_array):
            axis.scatter(
                index + replicate_offsets(len(values), 0.075),
                values,
                s=18,
                marker=marker,
                facecolor="white",
                edgecolor=DNA_METRIC_COLOR,
                linewidth=0.7,
                zorder=5,
            )
        axis.set(ylabel=ylabel, ylim=ylim, yticks=yticks)
        axis.yaxis.label.set_color(color)
        axis.tick_params(axis="y", colors=color, labelsize=7.0)
        # Opposite clear corners keep both legends away from the measured curves.
        axis.legend(
            handles=[
                Line2D(
                    [0], [0], color=color, ls=linestyle, marker=marker,
                    markerfacecolor=mean_facecolor, markeredgecolor="black",
                    markersize=4.5, label=legend_label,
                ),
            ],
            loc=legend_loc,
            bbox_to_anchor=legend_anchor,
            prop=FontProperties(family=FONT, size=7.1),
            frameon=False,
            borderaxespad=0,
            handlelength=1.4,
            handletextpad=0.4,
        )
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax_right = axes[1][0]
    ax_right.spines["right"].set_color("black")
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["top"].set_visible(False)
    ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_xticks(range(len(DNA_CONFORMATION_SYSTEMS)), DNA_CONFORMATION_LABELS)
    ax.tick_params(axis="x", labelsize=7.0, pad=1.0)
    draw_panel_label(ax, "D", x=-0.17)


def plot_figure2(tables: dict[str, pd.DataFrame], out: Path) -> None:
    fig = plt.figure(figsize=(7.20, 4.80), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.15, 1],
        left=0.085,
        right=0.93,
        bottom=0.13,
        top=0.97,
        hspace=0.35,
    )
    upper_metrics = outer[0, 0].subgridspec(1, 2, wspace=0.35)
    lower_metrics = outer[1, 0].subgridspec(1, 2, wspace=0.35)
    axes = [
        fig.add_subplot(upper_metrics[0, 0]),
        fig.add_subplot(lower_metrics[0, 0]),
        fig.add_subplot(lower_metrics[0, 1]),
    ]
    partner_rows = upper_metrics[0, 1].subgridspec(2, 1, hspace=0.14)
    med1_partner_ax = fig.add_subplot(partner_rows[0, 0])
    oct4_partner_ax = fig.add_subplot(partner_rows[1, 0], sharex=med1_partner_ax)
    paired = tables["balanced_paired"].copy()
    frame_repeat = tables["balanced_frame_repeat"].set_index(["system", "replicate"])
    protein_counts = tables["system_design"].set_index("system")[
        "n_physical_protein_chains"
    ]
    partner_rows = []
    for without, with_dna, label in PRIMARY_DNA_PAIRS:
        for replicate in base.REPEATS:
            without_row = frame_repeat.loc[(without, replicate)]
            with_dna_row = frame_repeat.loc[(with_dna, replicate)]
            without_partners = (
                2.0 * float(without_row["n_PP_edges"]) / protein_counts.loc[without]
            )
            with_dna_partners = (
                2.0 * float(with_dna_row["n_PP_edges"]) / protein_counts.loc[with_dna]
            )
            partner_rows.append(
                {
                    "comparison": label,
                    "replicate": replicate,
                    "cutoff_nm": base.PRIMARY_CUTOFF_NM,
                    "metric": "partners_per_chain",
                    "without_DNA": without_partners,
                    "with_DNA": with_dna_partners,
                }
            )
    paired = pd.concat([paired, pd.DataFrame(partner_rows)], ignore_index=True)
    plot_paired_dna_bars(axes[0], paired, "largest_cluster_fraction", "LCC fraction", (0, 1.06), "A")
    axes[0].set_yticks(np.linspace(0, 1, 5))
    plot_partner_species_rows(
        med1_partner_ax,
        oct4_partner_ax,
        tables["balanced_frame_repeat"],
    )
    styled_legend(
        axes[0],
        handles=[
            Patch(facecolor=to_rgba(FIG2_CONDITION_COLORS["without_DNA"], 0.90), edgecolor="black", linewidth=1.05, label="−DNA"),
            Patch(facecolor=to_rgba(FIG2_CONDITION_COLORS["with_DNA"], 0.90), edgecolor="black", linewidth=1.05, label="+DNA"),
        ],
        loc="upper left",
        ncols=2,
        prop=FontProperties(family=FONT, size=7.0, weight="normal"),
        handlelength=1.2,
        handletextpad=0.35,
    )
    plot_dna_association_bars(axes[1], tables["mechanism_repeat"])
    plot_dna_rg_line(axes[2], tables["DNA_conformation_with_D1_by_replicate"])
    dna_right_ax = fig.axes[-1]
    dna_right_ax.set_ylabel("DNA end-to-end\ndistance (nm)")
    apply_reference_typography(fig)
    pack_figure_columns(fig, [[axes[0]], [med1_partner_ax, oct4_partner_ax]], [1, 1], gap_pt=18)
    pack_figure_columns(fig, [[axes[1]], [axes[2], dna_right_ax]], [1, 1], gap_pt=18)
    pack_figure_rows(
        fig,
        [[axes[0], med1_partner_ax, oct4_partner_ax], [axes[1], axes[2], dna_right_ax]],
        gap_pt=14,
        resize=True,
    )
    base.save_figure(fig, out / "fig2_DNA_addition_network_response")


def plot_figure3(tables: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    fig = plt.figure(figsize=(7.20, 5.55), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.90, 1.24],
        left=0.055,
        right=0.995,
        bottom=0.06,
        top=0.975,
        hspace=0.08,
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
            ls="-",
            label=label,
        )
    density_ax.set(
        xlabel="Nearest-DNA distance (nm)",
        ylabel="DNA-linked amino acids",
        xlim=(0, 40),
        ylim=(0, 3000),
    )
    density_handles, density_labels = density_ax.get_legend_handles_labels()
    density_legend_order = [0, 1, 2, 3]
    styled_legend(
        density_ax,
        handles=[density_handles[index] for index in density_legend_order],
        labels=[density_labels[index] for index in density_legend_order],
        loc="center",
        bbox_to_anchor=(0.60, 0.44),
        ncols=1,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.3,
        labelspacing=0.15,
        borderaxespad=0.15,
        prop=FontProperties(family=FONT, size=LEGEND_SIZE, weight="normal"),
    )
    draw_panel_label(density_ax, "A", x=-0.17)

    q90_ax = fig.add_subplot(top[0, 1])
    extension_repeat = tables["mechanism_repeat"]
    q90_means = []
    for index, (system, color) in enumerate(zip(ALL_DNA_SYSTEMS, ALL_DNA_COLORS)):
        sub = extension_repeat[extension_repeat["system"] == system].set_index("replicate").reindex(base.REPEATS)
        values = sub["DNA_linked_bead_all_q90_nm"].to_numpy(float)
        q90_means.append(float(values.mean()))
        q90_ax.scatter(index + replicate_offsets(3, 0.035), values, s=21, facecolor="white", edgecolor=color, marker="o", linewidth=0.75, zorder=6)
        draw_mean_sd(q90_ax, index, values, color, marker="o")
    q90_ax.plot(range(len(ALL_DNA_SYSTEMS)), q90_means, color="#333333", lw=1.05, zorder=1)
    q90_ax.set_xticks(range(len(ALL_DNA_LABELS)), ALL_DNA_LABELS)
    q90_ax.tick_params(axis="x", labelsize=7.0)
    q90_ax.set(ylabel=r"$d_{90}$ from DNA (nm)", ylim=(10, 39.5))
    draw_panel_label(q90_ax, "B", x=-0.17)

    species_q90_ax = fig.add_subplot(top[0, 2])
    mixed_systems = ["M10O90D1", "M20O80D1"]
    mixed_labels = ["10%\nMED1", "20%\nMED1"]
    for index, (system, color) in enumerate(zip(mixed_systems, ALL_DNA_COLORS[1:3])):
        sub = extension_repeat[extension_repeat["system"] == system].set_index("replicate").reindex(base.REPEATS)
        for offset, protein_type, marker in [
            (-0.10, "OCT4", "^"),
            (0.10, "MED1", "s"),
        ]:
            values = sub[f"DNA_linked_bead_{protein_type}_q90_nm"].to_numpy(float)
            species_q90_ax.scatter(index + offset + replicate_offsets(3, 0.02), values, s=20, facecolor="white", edgecolor=color, marker=marker, linewidth=0.75, zorder=6)
            draw_mean_sd(species_q90_ax, index + offset, values, color, marker=marker)
    species_q90_ax.set_xticks(range(len(mixed_labels)), mixed_labels)
    species_q90_ax.set(ylabel=r"$d_{90}$ from DNA (nm)", ylim=(17, 41))
    styled_legend(
        species_q90_ax,
        handles=[
            Line2D([0], [0], marker="^", color="none", markeredgecolor="black", markerfacecolor="white", label="OCT4 beads"),
            Line2D([0], [0], marker="s", color="none", markeredgecolor="black", markerfacecolor="white", label="MED1 beads"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.995),
        handlelength=1.0,
        handletextpad=0.35,
    )
    draw_panel_label(species_q90_ax, "C", x=-0.17)

    structure = outer[1, 0].subgridspec(1, 4, wspace=0.045)
    extension_frame = tables["mechanism_framewise"]
    manifest = tables["trajectory_manifest"]
    selected_rows = []
    snapshot_axes = []
    snapshot_systems = ["M0O100D1", "M10O90D1", "M20O80D1", "M50O0D1"]
    snapshot_labels = [
        "OCT4-only +DNA",
        "10% MED1 +DNA",
        "20% MED1 +DNA",
        "MED1-only +DNA",
    ]
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
        snapshot_axes.append(ax)
        base.draw_spatial_snapshot(
            ax,
            snapshot,
            common_half_span,
            show_scale_bar=index == 0,
        )
        draw_projected_d90_contour(
            ax,
            snapshot,
            float(row["DNA_linked_bead_all_q90_nm"]),
            show_key=index == 0,
            key_label="Projected contour",
        )
        ax.set_title("")
        ax.text(
            0.03,
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
            draw_panel_label(ax, "D", y=1.01, align_x_fig=0.015)
    apply_reference_typography(fig)
    pack_figure_columns(fig, [[density_ax], [q90_ax], [species_q90_ax]], [1.50, 1.0, 1.0])
    pack_figure_rows(fig, [[density_ax, q90_ax, species_q90_ax], snapshot_axes], gap_pt=14, resize=True)
    base.save_figure(fig, out / "fig3_DNA_centered_spatial_reach")
    return pd.DataFrame(selected_rows)


def draw_split2_med1_inset(ax: plt.Axes) -> None:
    """Show the Split-2 MED1 sequence schematic without residue annotations."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B8BEC7")
        spine.set_linewidth(0.55)

    ax.text(
        0.035,
        0.76,
        "Split-2",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#333333",
    )
    colors = MED1_REGION_COLORS
    left, right = 0.035, 0.965
    split_x = left + (right - left) * 807 / base.MED1_LENGTH
    gap = 0.018
    y = 0.38
    ax.plot([left, split_x - gap], [y, y], color="#B8BEC7", lw=4.2, solid_capstyle="butt", zorder=1)
    ax.plot([split_x + gap, right], [y, y], color="#B8BEC7", lw=4.2, solid_capstyle="butt", zorder=1)
    for (start, stop), color in zip(legacy.MED1_NATIVE_REGIONS, colors):
        x_start = left + (right - left) * (start - 1) / (base.MED1_LENGTH - 1)
        x_stop = left + (right - left) * (stop - 1) / (base.MED1_LENGTH - 1)
        ax.add_patch(
            plt.Rectangle(
                (x_start, y - 0.08),
                max(x_stop - x_start, 0.008),
                0.16,
                facecolor=color,
                edgecolor="black",
                linewidth=0.35,
                zorder=2,
            )
        )


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
        ("M10O90D1", "Full", "-"),
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
        ylabel="DNA-linked amino acids",
        xlim=(0, 40),
        ylim=(0, 2300),
    )
    styled_legend(
        ax,
        loc="upper right",
        prop=FontProperties(family=FONT, size=7.0, weight="normal"),
        handlelength=1.7,
        handletextpad=0.35,
    )
    draw_panel_label(ax, "B", x=-0.17)


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
            Patch(facecolor=to_rgba(color, 0.90), edgecolor="black", linewidth=1.05, label="Full"),
            Patch(facecolor="white", edgecolor="black", linewidth=1.05, hatch="///", label="Split-2"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.55, 0.59),
        ncols=1,
        prop=FontProperties(family=FONT, size=7.0, weight="normal"),
        handlelength=1.2,
        handletextpad=0.35,
    )
    inset = ax.inset_axes([0.37, 0.70, 0.60, 0.25], zorder=8)
    draw_split2_med1_inset(inset)
    draw_panel_label(ax, "A", x=-0.17)


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
    ax.set_xticks(range(2), ["Full", "Split-2"])
    ax.set(ylabel=r"$d_{90}$ from DNA (nm)", ylim=(18, 32))
    ax.tick_params(axis="x", labelsize=7.2)
    draw_panel_label(ax, "C", x=-0.20)


def plot_figure4(tables: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    fig = plt.figure(figsize=(7.20, 6.40), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.90, 1.50],
        left=0.055,
        right=0.985,
        bottom=0.055,
        top=0.975,
        hspace=0.18,
    )
    data_grid = outer[0, 0].subgridspec(
        1,
        3,
        width_ratios=[1.20, 1.0, 0.88],
        wspace=0.42,
    )
    repeat = tables["mechanism_repeat"]
    data_axes = [fig.add_subplot(data_grid[0, col]) for col in range(3)]
    plot_split_neighbor_counts(
        data_axes[0],
        repeat,
        tables["system_design"],
    )
    plot_split_radial_profiles(
        data_axes[1],
        tables["mechanism_radial_histogram_repeat"],
    )
    plot_split_d90(data_axes[2], repeat)

    structure = outer[1, 0].subgridspec(1, 2, wspace=0.025)
    selected = select_matched_mechanism_snapshot(tables["mechanism_framewise"])
    manifest = tables["trajectory_manifest"]
    snapshot_axes = []
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
        snapshot_axes.append(ax)
        base.draw_spatial_snapshot(
            ax,
            snapshot,
            common_half_span,
            show_scale_bar=index == 0,
        )
        draw_projected_d90_contour(
            ax,
            snapshot,
            float(row.DNA_linked_bead_all_q90_nm),
            show_key=index == 0,
            key_label="Projected contour",
        )
        ax.set_title("")
        ax.text(
            0.035,
            0.965,
            row.condition,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
            zorder=6,
        )
        if index == 0:
            draw_panel_label(ax, "D", y=1.01, align_x_fig=0.015)

    apply_reference_typography(fig)
    pack_figure_columns(fig, [[ax] for ax in data_axes], [1.20, 1.15, 0.85])
    pack_figure_rows(fig, [data_axes, snapshot_axes], gap_pt=14, resize=True)
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
    plot_figure2(tables, out)
    fig3_selection = plot_figure3(tables, out)
    fig4_selection = plot_figure4(tables, out)
    selections = []
    for figure, table in [("Fig3", fig3_selection), ("Fig4", fig4_selection)]:
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
