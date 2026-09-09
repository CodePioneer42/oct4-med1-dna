#!/usr/bin/env python3
"""Render supplementary figures for the reframed MED1/OCT4 manuscript.

The script is intentionally read-only with respect to analysis tables and
caches.  It writes only publication figures under ``outputs/figures``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

import run_balanced_analysis as base


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs"
DNA_ONLY_ROOT = HERE.parent / "data/dna_only"
EXTENSION_VERSION = "fixed-parent-block-dna-component-v5"

SPLIT_D0 = ["M10O90", "M10O90-S2", "M10O90-S4"]
SPLIT_D1 = ["M10O90D1", "M10O90D1-S2", "M10O90D1-S4"]
SPLIT_LABELS = ["Full", "Split-2", "Split-4"]
SPLIT_COLORS = ["#3B6C8E", "#9B78B4", "#D65F4A"]

COMPOSITION_SNAPSHOTS = [
    "M5O95",
    "M10O90",
    "M15O85",
    "M20O80",
    "M25O75",
    "M30O70",
    "M50O0",
]
COMPOSITION_SNAPSHOT_LABELS = [
    "5% MED1",
    "10% MED1",
    "15% MED1",
    "20% MED1",
    "25% MED1",
    "30% MED1",
    "MED1-only",
]

DNA_CONFORMATION_SYSTEMS = ["D1", "M0O100D1", "M10O90D1", "M20O80D1", "M50O0D1"]
DNA_CONFORMATION_LABELS = [
    "DNA-only",
    "OCT4-only",
    "10% MED1",
    "20% MED1",
    "pure MED1\n(50-chain ref.)",
]
DNA_CONFORMATION_COLORS = [
    "#666666",
    base.COLORS["OCT4"],
    base.COLORS["mixed"],
    "#C07AA1",
    base.COLORS["MED1"],
]


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
        "definition_audit_split_interaction_summary",
        "DNA_conformation_with_D1_by_replicate",
        "balanced_framewise_network_contacts_DNA",
        "balanced_degree_group",
        "balanced_binned_contact_maps_group_summary",
        "mechanism_framewise",
        "D1_unaffected_control_sampled",
        "dna_rg_distribution_hypothesis_validation_repeat",
    ]
    return {name: read_table(tables_dir, name) for name in names}


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
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
        zorder=6,
    )


def draw_projected_d90_contour(
    ax: plt.Axes,
    snapshot: dict[str, object],
    d90_nm: float,
    *,
    show_key: bool = False,
) -> None:
    """Overlay the projected nearest-DNA d90 shell above a snapshot."""
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
    distances = cKDTree(dna_points).query(
        np.column_stack([xx.ravel(), yy.ravel()])
    )[0].reshape(xx.shape)
    if not (float(distances.min()) < d90_nm < float(distances.max())):
        raise ValueError(
            f"Projected d90 contour ({d90_nm:.2f} nm) falls outside the snapshot view"
        )
    ax.contour(
        xx,
        yy,
        distances,
        levels=[d90_nm],
        colors=["white"],
        linewidths=2.25,
        alpha=0.90,
        zorder=5.1,
    )
    contour_color = "#6F4C9B"
    ax.contour(
        xx,
        yy,
        distances,
        levels=[d90_nm],
        colors=[contour_color],
        linewidths=1.15,
        linestyles=[(0, (4.0, 2.2))],
        alpha=0.98,
        zorder=5.2,
    )
    if show_key:
        legend = ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=contour_color,
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
            fontsize=6.7,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.84,
        )
        legend.set_zorder(7)


def selected_frame_d90(
    mechanism_framewise: pd.DataFrame,
    system: str,
    replicate: str,
    frame: int,
) -> float:
    """Return bead-weighted DNA-linked d90 for an already selected frame."""
    rows = mechanism_framewise[
        (mechanism_framewise["system"] == system)
        & (mechanism_framewise["replicate"] == replicate)
        & (mechanism_framewise["frame"] == frame)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one d90 row for {system}/{replicate}/frame {frame}, "
            f"found {len(rows)}"
        )
    value = float(rows.iloc[0]["DNA_linked_bead_all_q90_nm"])
    if not np.isfinite(value):
        raise RuntimeError(f"Non-finite d90 for {system}/{replicate}/frame {frame}")
    return value


def save_supporting(fig: plt.Figure, path: Path) -> None:
    """Enforce the title-free supporting-figure style before export."""
    titled_axes = [ax.get_title() for ax in fig.axes if ax.get_title()]
    if titled_axes:
        raise RuntimeError(f"Supporting subplot titles are not allowed: {titled_axes}")
    base.save_figure(fig, path)


def draw_mean_sd(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    color: str,
    *,
    marker: str = "D",
    zorder: int = 5,
) -> None:
    values = np.asarray(values, dtype=float)
    if len(values) != 3 or not np.isfinite(values).all():
        raise RuntimeError(f"Expected three finite replicate values, got {values}")
    ax.errorbar(
        x,
        values.mean(),
        yerr=values.std(ddof=1),
        fmt=marker,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.55,
        capsize=2.6,
        zorder=zorder,
    )


def composition_metric(
    ax: plt.Axes,
    repeat: pd.DataFrame,
    design: pd.DataFrame,
    metric: str,
    ylabel: str,
    panel: str,
    title: str,
) -> None:
    primary = repeat[
        np.isclose(repeat["cutoff_nm"], base.PRIMARY_CUTOFF_NM)
        & repeat["system"].isin(base.COMPOSITION_SYSTEMS)
    ]
    x_map = base.composition_x(design)
    x = np.asarray([x_map[system] for system in base.COMPOSITION_SYSTEMS], dtype=float)
    means: list[float] = []
    sds: list[float] = []
    for index, system in enumerate(base.COMPOSITION_SYSTEMS):
        values = (
            primary[primary["system"] == system]
            .set_index("replicate")
            .reindex(base.REPEATS)[metric]
            .to_numpy(float)
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete composition values for {system}/{metric}")
        means.append(float(values.mean()))
        sds.append(float(values.std(ddof=1)))
        ax.scatter(
            x[index] + replicate_offsets(3, 0.28),
            values,
            s=16,
            facecolor="white",
            edgecolor=base.COLORS["mixed"],
            linewidth=0.65,
            zorder=4,
        )
    ax.plot(x, means, color=base.COLORS["mixed"], lw=1.55, zorder=2)
    ax.errorbar(
        x,
        means,
        yerr=sds,
        fmt="D",
        color=base.COLORS["mixed"],
        markeredgecolor="white",
        markeredgewidth=0.55,
        capsize=2.5,
        zorder=5,
    )
    ax.set(xlabel="MED1 molecule fraction (%)", ylabel=ylabel, xlim=(-2, 52))
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75)
    base.set_panel_title(ax, panel, title, label_x=-0.12)


def plot_composition_diagnostics(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S1: show every 5--30% composition plus the MED1-only reference."""
    fig = plt.figure(figsize=(7.20, 4.05), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        8,
        left=0.025,
        right=0.992,
        bottom=0.025,
        top=0.975,
        wspace=0.12,
        hspace=0.085,
    )
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    manifest = tables["trajectory_manifest"]
    panel_positions = [
        (0, slice(0, 2)),
        (0, slice(2, 4)),
        (0, slice(4, 6)),
        (0, slice(6, 8)),
        (1, slice(1, 3)),
        (1, slice(3, 5)),
        (1, slice(5, 7)),
    ]
    for index, (system, (row, columns)) in enumerate(
        zip(COMPOSITION_SNAPSHOTS, panel_positions)
    ):
        selected = base.select_snapshot(framewise, system)
        task = base.get_task(manifest, system, str(selected["replicate"]))
        snapshot = base.build_spatial_snapshot(
            task,
            int(selected["frame"]),
            mode="full_system",
        )
        ax = fig.add_subplot(outer[row, columns])
        base.draw_spatial_snapshot(ax, snapshot, task.box_nm / 2)
        ax.set_title("")
        annotate_structure_snapshot(ax, COMPOSITION_SNAPSHOT_LABELS[index])
        panel_label(ax, chr(ord("A") + index), x=-0.035, y=1.005)
    save_supporting(fig, out / "figS1_composition_snapshots")


def plot_cutoff_sensitivity(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S2: composition trend at three contact cutoffs."""
    repeat = tables["balanced_cluster_repeat"]
    design = tables["system_design"]
    x_map = base.composition_x(design)
    x = np.asarray([x_map[system] for system in base.COMPOSITION_SYSTEMS])
    fig, ax = plt.subplots(figsize=(4.8, 3.35), constrained_layout=True)
    for cutoff, color, marker in [
        (1.0, base.COLORS["OCT4"], "o"),
        (1.2, base.COLORS["mixed"], "s"),
        (1.4, base.COLORS["MED1"], "^"),
    ]:
        subset = repeat[
            np.isclose(repeat["cutoff_nm"], cutoff)
            & repeat["system"].isin(base.COMPOSITION_SYSTEMS)
        ]
        means: list[float] = []
        sds: list[float] = []
        for index, system in enumerate(base.COMPOSITION_SYSTEMS):
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
            ax.scatter(
                x[index] + replicate_offsets(3, 0.24),
                values,
                s=17,
                marker=marker,
                facecolor="white",
                edgecolor=color,
                linewidth=0.7,
                zorder=6,
            )
        ax.plot(x, means, color=color, lw=1.15, zorder=1)
        ax.errorbar(
            x,
            means,
            yerr=sds,
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.4,
            zorder=2,
        )
        ax.scatter(
            x,
            means,
            s=26,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=f"{cutoff:.1f} nm",
            zorder=5,
        )
    ax.set(
        xlabel="MED1 molecule fraction (%)",
        ylabel="Direct-protein LCC fraction",
        xlim=(-2, 52),
        ylim=(0.04, 0.42),
    )
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75, zorder=0)
    ax.legend(loc="upper left")
    panel_label(ax, "A", x=-0.13)
    save_supporting(fig, out / "figS2_cutoff_sensitivity")


def plot_med1_partner_composition(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S3: species identity of partners contacted by MED1."""
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    design = tables["system_design"]
    composition = framewise[
        framewise["system"].isin(base.COMPOSITION_SYSTEMS[1:])
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
    systems = base.COMPOSITION_SYSTEMS[1:]
    x_map = base.composition_x(design)
    x = np.asarray([x_map[system] for system in systems])
    oct4 = np.asarray([group.loc[system, "OCT4_mean"] for system in systems])
    med1 = np.asarray([group.loc[system, "MED1_mean"] for system in systems])
    totals = np.asarray([group.loc[system, "total_mean"] for system in systems])
    total_sd = np.asarray([group.loc[system, "total_sd"] for system in systems])

    fig, ax = plt.subplots(figsize=(4.8, 3.35), constrained_layout=True)
    ax.bar(
        x,
        oct4,
        width=3.6,
        color=base.COLORS["OCT4"],
        label="OCT4 partners",
        zorder=1,
    )
    ax.bar(
        x,
        med1,
        width=3.6,
        bottom=oct4,
        color=base.COLORS["MED1"],
        label="MED1 partners",
        zorder=1,
    )
    ax.errorbar(
        x,
        totals,
        yerr=total_sd,
        color="#333333",
        fmt="none",
        capsize=2.5,
        linewidth=0.9,
        zorder=3,
    )
    for system, x_value in zip(systems, x):
        values = repeat.loc[
            repeat["system"] == system, "MED1_total_partners"
        ].to_numpy(float)
        ax.scatter(
            x_value + replicate_offsets(len(values), 0.35),
            values,
            s=18,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.65,
            zorder=6,
        )
    ax.set(
        xlabel="MED1 molecule fraction (%)",
        ylabel="Partners per MED1 chain",
        xlim=(2, 53),
        ylim=(0, 3.7),
    )
    ax.set_xticks(x)
    ax.legend(loc="upper right", fontsize=6.8)
    panel_label(ax, "A", x=-0.13)
    save_supporting(fig, out / "figS3_MED1_partner_composition")


def plot_med1_reference_snapshots(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S4: MED1-only controls and mixed-series endpoint at a common scale."""
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    mechanism_framewise = tables["mechanism_framewise"]
    manifest = tables["trajectory_manifest"]
    systems = ["M50O0", "M50O0D1", "M50O50"]
    labels = ["MED1-only", "MED1-only +DNA", "50% MED1"]
    snapshots: list[dict[str, object]] = []
    selections: list[pd.Series] = []
    for system in systems:
        selected = base.select_snapshot(framewise, system)
        task = base.get_task(manifest, system, str(selected["replicate"]))
        selections.append(selected)
        snapshots.append(
            base.build_spatial_snapshot(task, int(selected["frame"]), mode="full_system")
        )
    half_span = max(
        np.max(np.abs(np.vstack(list(snapshot["projected_traces"].values()))))
        for snapshot in snapshots
    ) + 2
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    for index, (ax, snapshot, system, label, selected) in enumerate(
        zip(axes, snapshots, systems, labels, selections)
    ):
        base.draw_spatial_snapshot(
            ax,
            snapshot,
            float(half_span),
            scale_bar_inset_nm=5.0,
        )
        ax.set_title("")
        d90_nm = None
        if system.endswith("D1"):
            d90_nm = selected_frame_d90(
                mechanism_framewise,
                system,
                str(selected["replicate"]),
                int(selected["frame"]),
            )
        annotate_structure_snapshot(ax, label, d90_nm)
        panel_label(ax, chr(ord("A") + index), x=-0.035, y=1.005)
    save_supporting(fig, out / "figS4_MED1_reference_snapshots")


def plot_dna_neighborhood_composition(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S5: direct occupancy and complete DNA-linked component composition."""
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    design = tables["system_design"]
    frame_repeat = (
        framewise[framewise["system"].isin(base.DNA_SYSTEMS)]
        .groupby(["system", "replicate"], as_index=False)
        .mean(numeric_only=True)
    )
    frame_group = frame_repeat.groupby("system").mean(numeric_only=True)
    design_index = design.set_index("system")
    x = np.arange(len(base.DNA_SYSTEMS))
    n_proteins = np.asarray(
        [
            design_index.loc[system, "n_physical_protein_chains"]
            for system in base.DNA_SYSTEMS
        ],
        dtype=float,
    )
    labels = [pair[2] for pair in base.DNA_PAIRS]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.15), constrained_layout=True)
    panel_specs = [
        ("N_OCT4_molecules_contacting_DNA", "N_MED1_molecules_contacting_DNA"),
        ("DNA_component_OCT4", "DNA_component_MED1"),
    ]
    for panel_index, (ax, (oct4_metric, med1_metric)) in enumerate(
        zip(axes[:2], panel_specs)
    ):
        oct4 = np.asarray(
            [frame_group.loc[system, oct4_metric] for system in base.DNA_SYSTEMS]
        ) / n_proteins
        med1 = np.asarray(
            [frame_group.loc[system, med1_metric] for system in base.DNA_SYSTEMS]
        ) / n_proteins
        ax.bar(
            x,
            oct4,
            color=base.COLORS["OCT4"],
            label="OCT4",
            zorder=1,
        )
        ax.bar(
            x,
            med1,
            bottom=oct4,
            color=base.COLORS["MED1"],
            label="MED1",
            zorder=1,
        )
        for system_index, system in enumerate(base.DNA_SYSTEMS):
            values = frame_repeat[frame_repeat["system"] == system]
            totals = (
                values[oct4_metric].to_numpy(float)
                + values[med1_metric].to_numpy(float)
            ) / n_proteins[system_index]
            ax.scatter(
                system_index + replicate_offsets(len(totals), 0.055),
                totals,
                s=19,
                facecolor="white",
                edgecolor="#222222",
                linewidth=0.65,
                zorder=6,
            )
        ax.set_xticks(x, labels, rotation=27, ha="right")
        ax.set(ylabel="Fraction of all protein chains", ylim=(0, 1.04))
        panel_label(ax, chr(ord("A") + panel_index), x=-0.14)
    axes[0].legend(loc="upper left")

    ax = axes[2]
    for metric, label, color, marker in [
        ("OCT4_fraction_contacting_DNA", "OCT4", base.COLORS["OCT4"], "o"),
        ("MED1_fraction_contacting_DNA", "MED1", base.COLORS["MED1"], "s"),
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
        ax.plot(x[finite], np.asarray(means)[finite], color=color, lw=1.0, zorder=1)
        ax.errorbar(
            x,
            means,
            yerr=sds,
            fmt="none",
            ecolor=color,
            capsize=2.5,
            zorder=2,
        )
        ax.scatter(
            x[finite],
            np.asarray(means)[finite],
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=5,
        )
        for system_index, values in enumerate(per_system):
            if len(values):
                ax.scatter(
                    system_index + replicate_offsets(len(values), 0.045),
                    values,
                    s=20,
                    facecolor="white",
                    edgecolor=color,
                    marker=marker,
                    linewidth=0.75,
                    zorder=6,
                )
    ax.set_xticks(x, labels, rotation=27, ha="right")
    ax.set(ylabel="Within-species direct-DNA fraction", ylim=(0, 0.60))
    ax.legend(loc="upper right")
    panel_label(ax, "C", x=-0.14)
    save_supporting(fig, out / "figS5_DNA_neighborhood_composition")


def plot_contact_type_and_residue_profiles(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S6: three contact heatmaps aligned with three residue-profile groups."""
    frame_repeat = tables["balanced_frame_repeat"].set_index(
        ["system", "replicate"]
    )
    profiles = tables["balanced_site_profiles_group_summary"]
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
        ("M10O90", "10% MED1\n−DNA"),
        ("M10O90D1", "10% MED1\n+DNA"),
        ("M20O80", "20% MED1\n−DNA"),
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

    fig = plt.figure(figsize=(7.2, 6.45), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.00, 1.75],
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.975,
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
        if column == 0:
            panel_label(ax, "A", x=-0.27, y=1.01)

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
            ax.set_ylabel("Per-chain contact probability")
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
        panel_label(ax, panel, x=-0.18)

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
    save_supporting(fig, out / "figS6_contact_type_and_residue_profiles")


def paired_split_metric(
    ax: plt.Axes,
    repeat: pd.DataFrame,
    metric: str,
    ylabel: str,
    panel: str,
    title: str,
    *,
    scale: float = 1.0,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(3, dtype=float)
    for systems, condition, color, offset in [
        (SPLIT_D0, "−DNA", base.COLORS["neutral"], -0.075),
        (SPLIT_D1, "+DNA", base.COLORS["DNA"], 0.075),
    ]:
        pivot = (
            repeat[repeat["system"].isin(systems)]
            .pivot(index="replicate", columns="system", values=metric)
            .reindex(index=base.REPEATS, columns=systems)
        )
        values = pivot.to_numpy(float) * scale
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete Full/Split values for {metric}")
        for repeat_index, row in enumerate(values):
            jitter = (repeat_index - 1) * 0.014
            ax.plot(
                x + offset + jitter,
                row,
                color=color,
                lw=0.7,
                alpha=0.44,
                marker="o",
                ms=3.0,
                markerfacecolor="white",
                markeredgewidth=0.6,
                zorder=2,
            )
        ax.errorbar(
            x + offset,
            values.mean(axis=0),
            yerr=values.std(axis=0, ddof=1),
            fmt="D-",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            capsize=2.5,
            label=condition,
            zorder=5,
        )
    ax.set_xticks(x, SPLIT_LABELS)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75)
    base.set_panel_title(ax, panel, title, label_x=-0.13)


def plot_absolute_split_losses(
    ax: plt.Axes,
    effects: pd.DataFrame,
) -> None:
    subset = effects[
        effects["metric"] == "continuity_block_residue_LCC_fraction"
    ]
    x = np.arange(2, dtype=float)
    for offset, column, color, marker, label in [
        (-0.07, "Full_minus_Split_2", SPLIT_COLORS[1], "o", "Full − Split-2"),
        (0.07, "Full_minus_Split_4", SPLIT_COLORS[2], "s", "Full − Split-4"),
    ]:
        pivot = (
            subset.pivot(index="replicate", columns="DNA_condition", values=column)
            .reindex(index=base.REPEATS, columns=["−DNA", "+DNA"])
        )
        values = pivot.to_numpy(float)
        for repeat_index, row in enumerate(values):
            ax.plot(
                x + offset + (repeat_index - 1) * 0.013,
                row,
                color=color,
                lw=0.75,
                alpha=0.45,
                marker=marker,
                ms=3.0,
                markerfacecolor="white",
                markeredgewidth=0.55,
            )
        ax.errorbar(
            x + offset,
            values.mean(axis=0),
            yerr=values.std(axis=0, ddof=1),
            fmt=marker + "-",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            capsize=2.5,
            label=label,
            zorder=5,
        )
    ax.set_xticks(x, ["−DNA", "+DNA"])
    ax.set(ylabel="Absolute Full − Split LCC", ylim=(0, 0.17))
    ax.legend(loc="upper left", fontsize=6.6)
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75)
    base.set_panel_title(ax, "B", "Absolute LCC losses", label_x=-0.13)


def plot_lcc_difference_in_differences(
    ax: plt.Axes,
    did: pd.DataFrame,
) -> None:
    subset = (
        did[did["metric"] == "continuity_block_residue_LCC_fraction"]
        .set_index("replicate")
        .reindex(base.REPEATS)
    )
    values = subset["difference_in_differences"].to_numpy(float)
    for offset, value in zip(replicate_offsets(3, 0.085), values):
        ax.plot([offset, offset], [0, value], color="#A9A9A9", lw=0.8, zorder=1)
    ax.scatter(
        replicate_offsets(3, 0.085),
        values,
        s=24,
        facecolor="white",
        edgecolor=base.COLORS["mixed"],
        linewidth=0.75,
        zorder=4,
    )
    draw_mean_sd(ax, 0, values, base.COLORS["mixed"])
    ax.axhline(0, color="#777777", ls="--", lw=0.8)
    ax.set_xticks([0], ["(+DNA loss) − (−DNA loss)"])
    ax.set(ylabel="LCC difference-in-differences", xlim=(-0.42, 0.42), ylim=(-0.006, 0.040))
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75)
    ax.text(
        0.97,
        0.94,
        f"{values.mean():+.3f} ± {values.std(ddof=1):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#555555",
    )
    base.set_panel_title(ax, "C", "Full − Split-4 interaction", label_x=-0.13)


def plot_parent_density_and_partners(
    ax: plt.Axes,
    repeat: pd.DataFrame,
) -> None:
    twin = ax.twinx()
    x = np.arange(3, dtype=float)
    for systems, condition, color, offset in [
        (SPLIT_D0, "−DNA", base.COLORS["neutral"], -0.075),
        (SPLIT_D1, "+DNA", base.COLORS["DNA"], 0.075),
    ]:
        subset = repeat[repeat["system"].isin(systems)]
        density = (
            subset.pivot(index="replicate", columns="system", values="parent_edge_density")
            .reindex(index=base.REPEATS, columns=systems)
            .to_numpy(float)
            * 100.0
        )
        partners = (
            subset.pivot(
                index="replicate",
                columns="system",
                values="MED1_parent_mean_partner_number",
            )
            .reindex(index=base.REPEATS, columns=systems)
            .to_numpy(float)
        )
        for repeat_index in range(3):
            jitter = (repeat_index - 1) * 0.012
            ax.plot(
                x + offset + jitter,
                density[repeat_index],
                color=color,
                ls="--",
                lw=0.65,
                alpha=0.36,
                marker="o",
                ms=2.8,
                markerfacecolor="white",
            )
            twin.plot(
                x + offset + jitter,
                partners[repeat_index],
                color=color,
                ls="-",
                lw=0.65,
                alpha=0.36,
                marker="s",
                ms=2.8,
                markerfacecolor="white",
            )
        ax.errorbar(
            x + offset,
            density.mean(axis=0),
            yerr=density.std(axis=0, ddof=1),
            fmt="o--",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            capsize=2.3,
            zorder=5,
        )
        twin.errorbar(
            x + offset,
            partners.mean(axis=0),
            yerr=partners.std(axis=0, ddof=1),
            fmt="s-",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            capsize=2.3,
            zorder=5,
        )
    ax.set_xticks(x, SPLIT_LABELS)
    ax.set(ylabel="Fixed-parent edge density (%)", ylim=(0, 5.0))
    twin.set(ylabel="External partners per MED1 parent", ylim=(0, 10.5))
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color("#777777")
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75)
    condition_handles = [
        Line2D([0], [0], color=base.COLORS["neutral"], lw=1.3, label="−DNA"),
        Line2D([0], [0], color=base.COLORS["DNA"], lw=1.3, label="+DNA"),
    ]
    metric_handles = [
        Line2D([0], [0], color="#555555", marker="o", ls="--", label="parent density"),
        Line2D([0], [0], color="#555555", marker="s", ls="-", label="MED1 partners"),
    ]
    condition_legend = ax.legend(
        handles=condition_handles,
        loc="upper left",
        ncols=1,
        fontsize=6.4,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
    )
    ax.add_artist(condition_legend)
    ax.legend(
        handles=metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.005),
        ncols=1,
        fontsize=6.2,
        labelspacing=0.25,
        handletextpad=0.55,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
    )
    base.set_panel_title(ax, "D", "Fixed-parent density and MED1 partners", label_x=-0.09)


def plot_chain_continuity_diagnostics(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S8: corona connectivity, DNA occupancy, and matched structures."""
    s8_split_d0 = SPLIT_D0[:2]
    s8_split_d1 = SPLIT_D1[:2]
    s8_split_labels = SPLIT_LABELS[:2]
    mechanism_repeat = tables["mechanism_repeat"]
    cluster_repeat = tables["balanced_cluster_repeat"]
    cluster_repeat = cluster_repeat[
        np.isclose(cluster_repeat["cutoff_nm"], base.PRIMARY_CUTOFF_NM)
    ]
    fig = plt.figure(figsize=(7.20, 6.45), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.70, 0.70, 1.18],
        left=0.080,
        right=0.975,
        bottom=0.025,
        top=0.980,
        hspace=0.38,
    )
    first_metric_row = outer[0, 0].subgridspec(1, 2, wspace=0.42)
    second_metric_row = outer[1, 0].subgridspec(1, 2, wspace=0.68)
    axes = [
        fig.add_subplot(first_metric_row[0, 0]),
        fig.add_subplot(first_metric_row[0, 1]),
        fig.add_subplot(second_metric_row[0, 0]),
        fig.add_subplot(second_metric_row[0, 1]),
    ]
    x = np.arange(2, dtype=float)

    def metric_values(table: pd.DataFrame, metric: str) -> np.ndarray:
        values = (
            table[table["system"].isin(s8_split_d1)]
            .pivot(index="replicate", columns="system", values=metric)
            .reindex(index=base.REPEATS, columns=s8_split_d1)
            .to_numpy(float)
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete +DNA Full/Split values for {metric}")
        return values

    def line_summary(
        ax: plt.Axes,
        values: np.ndarray,
        *,
        color: str,
        marker: str,
        linestyle: str = "-",
    ) -> None:
        means = values.mean(axis=0)
        sds = values.std(axis=0, ddof=1)
        ax.plot(x, means, color=color, lw=1.25, ls=linestyle, zorder=1)
        ax.errorbar(
            x,
            means,
            yerr=sds,
            fmt="none",
            ecolor=color,
            elinewidth=0.78,
            capsize=2.3,
            zorder=2,
        )
        ax.scatter(
            x,
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
                x[system_index] + replicate_offsets(values.shape[0], 0.045),
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
    line_summary(axes[0], lcc_values, color="black", marker="o")
    free_ax = axes[0].twinx()
    line_summary(
        free_ax,
        free_values,
        color="#777777",
        marker="s",
        linestyle="--",
    )
    axes[0].set(ylabel="LCC fraction", ylim=(0.0, 0.60))
    free_ax.set(ylabel="Free-chain fraction", ylim=(0.0, 0.60))
    free_ax.yaxis.label.set_size(7.4)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="black", marker="o", markersize=4.2, label="LCC"),
            Line2D(
                [0],
                [0],
                color="#777777",
                ls="--",
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
    axes[0].grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75, zorder=0)
    panel_label(axes[0], "A", x=-0.17)

    # Restore the original connected-component comparison at a fixed graph
    # resolution. Bars remain below the independent-trajectory points.
    for condition_index, (systems, condition, color) in enumerate(
        [
            (s8_split_d0, "−DNA", base.COLORS["neutral"]),
            (s8_split_d1, "+DNA", base.COLORS["DNA"]),
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
                color=color,
                alpha=0.90,
                edgecolor=color,
                linewidth=0.65,
                zorder=1,
            )
            axes[1].errorbar(
                xpos,
                values[:, system_index].mean(),
                yerr=values[:, system_index].std(ddof=1),
                fmt="none",
                color="#444444",
                elinewidth=0.78,
                capsize=2.2,
                zorder=3,
            )
            axes[1].scatter(
                xpos + replicate_offsets(values.shape[0], 0.035),
                values[:, system_index],
                s=15,
                facecolor="white",
                edgecolor=color,
                linewidth=0.65,
                zorder=6,
            )
    axes[1].set(ylabel="Fixed-block LCC fraction", ylim=(0.0, 0.60))
    axes[1].yaxis.label.set_size(7.4)
    axes[1].legend(
        handles=[
            Line2D([0], [0], color=base.COLORS["neutral"], lw=5.5, label="−DNA"),
            Line2D([0], [0], color=base.COLORS["DNA"], lw=5.5, label="+DNA"),
        ],
        loc="upper right",
        fontsize=6.4,
        frameon=False,
        handlelength=1.25,
    )
    axes[1].grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75, zorder=0)
    panel_label(axes[1], "B", x=-0.18)

    # Combine the two DNA-occupancy counts in one dual-axis line panel so that
    # the much smaller MED1-parent count is not compressed by the OCT4 scale.
    med1_bound = metric_values(mechanism_repeat, "DNA_direct_MED1_parent_count")
    oct4_bound = metric_values(mechanism_repeat, "DNA_direct_OCT4_parent_count")
    line_summary(
        axes[2],
        oct4_bound,
        color=base.COLORS["OCT4"],
        marker="o",
    )
    med1_ax = axes[2].twinx()
    line_summary(
        med1_ax,
        med1_bound,
        color=base.COLORS["MED1"],
        marker="s",
        linestyle="--",
    )
    axes[2].set(ylabel="DNA-bound OCT4", ylim=(26.0, 35.5))
    med1_ax.set(ylabel="DNA-bound MED1 parents", ylim=(0, 6.5))
    axes[2].yaxis.label.set_color(base.COLORS["OCT4"])
    axes[2].tick_params(axis="y", colors=base.COLORS["OCT4"])
    axes[2].spines["left"].set_color(base.COLORS["OCT4"])
    med1_ax.yaxis.label.set_color(base.COLORS["MED1"])
    med1_ax.tick_params(axis="y", colors=base.COLORS["MED1"])
    med1_ax.spines["right"].set_color(base.COLORS["MED1"])
    axes[2].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=base.COLORS["OCT4"],
                marker="o",
                markersize=4.2,
                label="OCT4",
            ),
            Line2D(
                [0],
                [0],
                color=base.COLORS["MED1"],
                ls="--",
                marker="s",
                markerfacecolor="white",
                markersize=4.2,
                label="MED1",
            ),
        ],
        loc="lower left",
        fontsize=6.5,
        frameon=False,
        ncols=2,
        columnspacing=0.9,
        handlelength=1.2,
    )
    axes[2].grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75, zorder=0)
    panel_label(axes[2], "C", x=-0.13)

    # Dense-frame trajectory means, matching the summary panel of the
    # exploratory Rg-distribution figure. Split-4 is intentionally omitted.
    rg_repeat = tables["dna_rg_distribution_hypothesis_validation_repeat"]
    rg_systems = [
        "D1",
        "M0O100D1",
        "M10O90D1",
        "M20O80D1",
        "M50O0D1",
        "M10O90D1-S2",
    ]
    rg_labels = [
        "DNA only",
        "OCT4-only +DNA",
        "10% MED1 +DNA / Full",
        "20% MED1 +DNA",
        "MED1-only +DNA",
        "Split-2 +DNA",
    ]
    replicate_colors = {
        "work": "#4C78A8",
        "work-1": "#E45756",
        "work-2": "#54A24B",
    }
    y = np.arange(len(rg_systems), dtype=float)[::-1]
    y_offsets = replicate_offsets(len(base.REPEATS), 0.12)
    for position, system in zip(y, rg_systems):
        values = (
            rg_repeat[rg_repeat["system"].eq(system)]
            .set_index("replicate")
            .reindex(base.REPEATS)["trajectory_mean_nm"]
        )
        if not np.isfinite(values.to_numpy(float)).all():
            raise RuntimeError(f"Incomplete dense-frame DNA Rg summary for {system}")
        for offset, replicate in zip(y_offsets, base.REPEATS):
            axes[3].scatter(
                values.loc[replicate],
                position + offset,
                s=19,
                facecolor="white",
                edgecolor=replicate_colors[replicate],
                linewidth=0.85,
                zorder=5,
            )
        axes[3].errorbar(
            values.mean(),
            position,
            xerr=values.std(ddof=1),
            fmt="D",
            color="black",
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.0,
            elinewidth=0.85,
            capsize=2.2,
            zorder=3,
        )
    axes[3].set_yticks(y, rg_labels)
    axes[3].set(
        xlabel=r"Trajectory-mean DNA $R_g$ (nm)",
        xlim=(16.8, 18.0),
        ylim=(-0.55, len(y) - 0.45),
    )
    axes[3].tick_params(axis="y", labelsize=6.5)
    axes[3].grid(axis="x", color="#DDDDDD", lw=0.45, alpha=0.75, zorder=0)
    panel_label(axes[3], "D", x=-0.39)

    for ax in axes:
        if ax is not axes[3]:
            ax.set_xticks(x, s8_split_labels)

    # E--F: matched Full/Split-2 structural views. Split-4 is omitted.
    selected = base.select_matched_split_snapshot(
        tables["balanced_framewise_network_contacts_DNA"],
        tables["balanced_framewise_clusters"],
        systems=tuple(s8_split_d1),
    )
    manifest = tables["trajectory_manifest"]
    common_half_span = max(
        float(manifest.loc[manifest["system"].isin(s8_split_d1), "box_nm"].max()) / 2,
        41.5,
    )
    structure_grid = outer[2, 0].subgridspec(1, 2, wspace=0.055)
    structure_labels = ["10% MED1 +DNA", "Split-2 +DNA"]
    for index, (row, label) in enumerate(
        zip(selected.itertuples(index=False), structure_labels)
    ):
        task = base.get_task(manifest, row.system, row.replicate)
        snapshot = base.build_spatial_snapshot(
            task,
            int(row.frame),
            mode="DNA_linked_component",
        )
        d90_nm = selected_frame_d90(
            tables["mechanism_framewise"],
            row.system,
            row.replicate,
            int(row.frame),
        )
        ax = fig.add_subplot(structure_grid[0, index])
        base.draw_spatial_snapshot(ax, snapshot, common_half_span)
        draw_projected_d90_contour(ax, snapshot, d90_nm, show_key=index == 0)
        ax.set_title("")
        ax.set_anchor("C")
        annotate_structure_snapshot(ax, label, d90_nm)
        panel_label(ax, chr(ord("E") + index), x=-0.035, y=1.005)
    save_supporting(fig, out / "figS8_chain_continuity_controls")


def conformation_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    panel: str,
    title: str,
    ylim: tuple[float, float],
) -> None:
    for index, (system, color) in enumerate(
        zip(DNA_CONFORMATION_SYSTEMS, DNA_CONFORMATION_COLORS)
    ):
        values = (
            table[table["system"] == system]
            .set_index("replicate")
            .reindex(base.REPEATS)[metric]
            .to_numpy(float)
        )
        ax.scatter(
            index + replicate_offsets(3, 0.07),
            values,
            s=22,
            facecolor="white",
            edgecolor=color,
            linewidth=0.75,
            zorder=4,
        )
        draw_mean_sd(ax, index, values, color)
    ax.set_xticks(range(5), DNA_CONFORMATION_LABELS, rotation=16, ha="right")
    ax.set(ylabel=ylabel, ylim=ylim)
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75)
    base.set_panel_title(ax, panel, title, label_x=-0.13)


def select_representative_conformation(
    framewise: pd.DataFrame,
    system: str,
) -> pd.Series:
    candidates = framewise[framewise["system"] == system].copy()
    if candidates.empty:
        raise RuntimeError(f"No conformation frames for {system}")
    metrics = ["DNA_Rg_nm", "DNA_Ree_nm", "DNA_asphericity"]
    score = np.zeros(len(candidates), dtype=float)
    for metric in metrics:
        center = float(candidates[metric].median())
        scale = max(float(candidates[metric].std(ddof=1)), 1e-8)
        score += np.abs(candidates[metric].to_numpy(float) - center) / scale
    score += 0.04 * np.abs(candidates["time_us"].to_numpy(float) - 3.0)
    return candidates.iloc[int(np.argmin(score))]


def dna_only_projected_trace(replicate: str, frame: int) -> np.ndarray:
    path = DNA_ONLY_ROOT / replicate / "D1"
    universe = base.mda.Universe(str(path / "start.pdb"), str(path / "output.dcd"))
    universe.trajectory[frame - 1]
    whole = base.unwrap_dna(
        universe.atoms.positions.astype(np.float64) / 10.0,
        75.0,
    )
    centered = whole - whole.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    for column in range(2):
        dominant = int(np.argmax(np.abs(basis[:, column])))
        if basis[dominant, column] < 0:
            basis[:, column] *= -1
    return centered @ basis


def draw_dna_only_snapshot(
    ax: plt.Axes,
    projected: np.ndarray,
    half_span: float,
) -> None:
    half = len(projected) // 2
    ax.plot(
        projected[:half, 0],
        projected[:half, 1],
        color=base.COLORS["DNA"],
        lw=1.45,
    )
    ax.plot(
        projected[half:, 0],
        projected[half:, 1],
        color=base.COLORS["DNA"],
        lw=1.45,
    )
    ax.scatter(
        [0],
        [0],
        s=28,
        marker="D",
        color=base.COLORS["DNA"],
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.plot(
        [-half_span + 2, -half_span + 12],
        [-half_span + 2, -half_span + 2],
        color="black",
        lw=1.5,
    )
    ax.text(
        -half_span + 7,
        -half_span + 3,
        "10 nm",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    ax.set(
        xlim=(-half_span, half_span),
        ylim=(-half_span, half_span),
        aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
        spine.set_linewidth(0.6)


def plot_dna_conformation(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S7: restore former S14B and D--F using the corrected data set."""
    table = tables["DNA_conformation_with_D1_by_replicate"]
    framewise = tables["balanced_framewise_network_contacts_DNA"]
    mechanism_framewise = tables["mechanism_framewise"]
    manifest = tables["trajectory_manifest"]
    fig = plt.figure(figsize=(7.2, 4.95), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.72, 1.45],
        left=0.035,
        right=0.99,
        bottom=0.10,
        top=0.97,
        hspace=0.10,
    )
    metric_grid = outer[0, 0].subgridspec(1, 3, width_ratios=[0.72, 1.5, 0.72])
    ax = fig.add_subplot(metric_grid[0, 1])
    values_by_system: list[np.ndarray] = []
    for system in DNA_CONFORMATION_SYSTEMS:
        values = (
            table[table["system"] == system]
            .set_index("replicate")
            .reindex(base.REPEATS)["DNA_Ree_nm"]
            .to_numpy(float)
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete DNA end-to-end values for {system}")
        values_by_system.append(values)
    values_array = np.asarray(values_by_system, dtype=float)
    x = np.arange(len(DNA_CONFORMATION_SYSTEMS), dtype=float)
    color = "black"
    means = values_array.mean(axis=1)
    sds = values_array.std(axis=1, ddof=1)
    # Match Fig. 1: line and uncertainty below filled means, with open
    # independent-trajectory points rendered on the top layer.
    ax.plot(x, means, color=color, lw=1.55, zorder=1)
    ax.errorbar(
        x,
        means,
        yerr=sds,
        fmt="none",
        color=color,
        elinewidth=0.85,
        capsize=2.5,
        zorder=2,
    )
    ax.scatter(
        x,
        means,
        s=28,
        facecolor=color,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
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
    ax.set_xticks(
        x,
        ["DNA\nonly", "OCT4\nonly", "10%\nMED1", "20%\nMED1", "MED1-\nonly"],
    )
    ax.tick_params(axis="x", labelsize=6.6, pad=1.0)
    ax.set(ylabel=r"DNA $R_{ee}$ (nm)", ylim=(50.5, 59.5))
    panel_label(ax, "A", x=-0.12)

    structure_systems = ["M0O100D1", "M10O90D1", "M50O0D1"]
    structure_labels = ["OCT4-only +DNA", "10% MED1 +DNA", "MED1-only +DNA"]
    snapshots: list[dict[str, object]] = []
    structure_d90: list[float] = []
    for system in structure_systems:
        # Match the former S14D--F representative-frame rule.
        selected = base.select_snapshot(framewise, system)
        task = base.get_task(manifest, system, str(selected["replicate"]))
        structure_d90.append(
            selected_frame_d90(
                mechanism_framewise,
                system,
                str(selected["replicate"]),
                int(selected["frame"]),
            )
        )
        snapshots.append(
            base.build_spatial_snapshot(
                task,
                int(selected["frame"]),
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
    structure_grid = outer[1, 0].subgridspec(1, 3, wspace=0.035)
    for index, (snapshot, label, d90_nm) in enumerate(
        zip(snapshots, structure_labels, structure_d90)
    ):
        snapshot_ax = fig.add_subplot(structure_grid[0, index])
        base.draw_spatial_snapshot(snapshot_ax, snapshot, shared_span)
        snapshot_ax.set_title("")
        snapshot_ax.set_anchor("C")
        annotate_structure_snapshot(snapshot_ax, label, d90_nm)
        panel_label(snapshot_ax, chr(ord("B") + index), x=-0.035, y=1.005)
    save_supporting(fig, out / "figS7_DNA_end_to_end_and_neighborhoods")


def radial_series_panel(
    ax: plt.Axes,
    repeat: pd.DataFrame,
    specs: list[tuple[str, str, str, str]],
    panel: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    x = np.arange(3, dtype=float)
    offsets = np.linspace(-0.12, 0.12, len(specs))
    for offset, (metric, color, marker, label) in zip(offsets, specs):
        pivot = (
            repeat[repeat["system"].isin(SPLIT_D1)]
            .pivot(index="replicate", columns="system", values=metric)
            .reindex(index=base.REPEATS, columns=SPLIT_D1)
        )
        values = pivot.to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete radial values for {metric}")
        for replicate_index, row in enumerate(values):
            ax.plot(
                x + offset + (replicate_index - 1) * 0.010,
                row,
                color=color,
                lw=0.65,
                alpha=0.34,
                marker=marker,
                ms=2.7,
                markerfacecolor="white",
                markeredgewidth=0.5,
            )
        ax.errorbar(
            x + offset,
            values.mean(axis=0),
            yerr=values.std(axis=0, ddof=1),
            fmt=marker + "-",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            capsize=2.3,
            label=label,
            zorder=5,
        )
    ax.set_xticks(x, SPLIT_LABELS)
    ax.set(ylabel=ylabel, ylim=ylim)
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.75)
    ax.legend(loc="upper right", fontsize=6.1, ncols=2 if len(specs) > 2 else 1)
    base.set_panel_title(ax, panel, title, label_x=-0.13)


def cdf_subset(
    histogram: pd.DataFrame,
    system: str,
    scope: str,
    protein_type: str,
    max_nm: float = 45.0,
) -> pd.DataFrame:
    subset = histogram[
        (histogram["system"] == system)
        & (histogram["distance_scope"] == scope)
        & (histogram["protein_type"] == protein_type)
        & np.isfinite(histogram["bin_end_nm"])
        & (histogram["bin_end_nm"] <= max_nm)
        & np.isfinite(histogram["cumulative_probability_mean"])
    ].sort_values("bin_end_nm")
    if subset.empty:
        raise RuntimeError(f"No radial CDF for {system}/{scope}/{protein_type}")
    return subset


def plot_split2_dna_compaction_mechanism(
    tables: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """S9: distinguish global DNA bending from contour contraction."""
    validation = tables["split2_dna_compaction_validation"]
    if not validation["passed"].astype(bool).all():
        raise RuntimeError("S9 compaction analysis did not pass validation")

    repeat = tables["split2_dna_compaction_repeat"]
    paired = tables["split2_dna_compaction_paired"]
    internal = tables["split2_dna_internal_distance_paired"]
    mediator = tables["split2_dna_mediator_correlation_repeat"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.20, 5.30),
        gridspec_kw={"hspace": 0.43, "wspace": 0.48},
    )

    # A: dense trajectory means place Split-2 relative to the DNA-only ensemble.
    systems = ["D1", "M10O90D1", "M10O90D1-S2", "M10O90D1-S4"]
    labels = ["DNA\nonly", "Full", "Split-2", "Split-4*"]
    colors = ["#666666", *SPLIT_COLORS]
    rg = (
        repeat.pivot(index="replicate", columns="system", values="DNA_Rg_nm")
        .reindex(index=base.REPEATS, columns=systems)
        .to_numpy(float)
    )
    if not np.isfinite(rg).all():
        raise RuntimeError("Incomplete S9 DNA Rg table")
    x = np.arange(len(systems), dtype=float)
    for row in rg:
        axes[0, 0].plot(x[1:], row[1:], color="#B8B8B8", lw=0.65, zorder=1)
    axes[0, 0].plot(
        x[1:], rg[:, 1:].mean(axis=0), color="#555555", lw=1.0, zorder=2
    )
    for index, (color, values) in enumerate(zip(colors, rg.T)):
        axes[0, 0].scatter(
            index + replicate_offsets(3, 0.055),
            values,
            s=21,
            facecolor="white",
            edgecolor=color,
            linewidth=0.75,
            zorder=5,
        )
        draw_mean_sd(axes[0, 0], index, values, color, marker="o", zorder=4)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set(
        ylabel=r"DNA $R_g$ (nm)",
        ylim=(16.82, 17.98),
    )
    axes[0, 0].tick_params(axis="x", labelsize=7.0)
    panel_label(axes[0, 0], "A", x=-0.14)

    # B: the Rg decrease is axial loss plus transverse spread, not a shorter contour.
    effect_specs = [
        ("DNA_axial_Rg_nm", "Axial\nspread"),
        ("DNA_transverse_Rg_nm", "Transverse\nspread"),
        ("DNA_centerline_contour_nm", "Contour\nlength"),
    ]
    effect_x = np.arange(len(effect_specs), dtype=float)
    for index, (metric, _) in enumerate(effect_specs):
        values = (
            paired[
                (paired["comparison"] == "Split-2_minus_Full")
                & (paired["metric"] == metric)
            ]
            .set_index("replicate")
            .reindex(base.REPEATS)["difference"]
            .to_numpy(float)
        )
        axes[0, 1].scatter(
            index + replicate_offsets(3, 0.055),
            values,
            s=21,
            facecolor="white",
            edgecolor=SPLIT_COLORS[1],
            linewidth=0.75,
            zorder=5,
        )
        draw_mean_sd(
            axes[0, 1], index, values, SPLIT_COLORS[1], marker="D", zorder=4
        )
    axes[0, 1].axhline(0, color="#777777", lw=0.75, ls="--", zorder=0)
    axes[0, 1].set_xticks(effect_x, [label for _, label in effect_specs])
    axes[0, 1].set(
        ylabel="Split-2 − Full (nm)",
        ylim=(-1.25, 1.18),
    )
    axes[0, 1].tick_params(axis="x", labelsize=7.0)
    panel_label(axes[0, 1], "B", x=-0.14)

    # C: shorter distances appear only over long contour separations.
    for replicate, color in zip(base.REPEATS, ["#B8A7C8", "#8E6AA9", "#65447E"]):
        subset = internal[internal["replicate"] == replicate].sort_values(
            "contour_separation_bp"
        )
        axes[1, 0].plot(
            subset["contour_separation_bp"],
            subset["Split_2_minus_Full_nm"],
            color=color,
            lw=0.70,
            alpha=0.82,
            zorder=1,
        )
    group_internal = (
        internal[
            [
                "contour_separation_bp",
                "Split_2_minus_Full_nm_mean",
                "Split_2_minus_Full_nm_sd",
            ]
        ]
        .drop_duplicates()
        .sort_values("contour_separation_bp")
    )
    lag = group_internal["contour_separation_bp"].to_numpy(float)
    mean = group_internal["Split_2_minus_Full_nm_mean"].to_numpy(float)
    sd = group_internal["Split_2_minus_Full_nm_sd"].to_numpy(float)
    axes[1, 0].fill_between(
        lag, mean - sd, mean + sd, color=SPLIT_COLORS[1], alpha=0.16, lw=0
    )
    axes[1, 0].plot(lag, mean, color=SPLIT_COLORS[1], lw=1.55, zorder=3)
    axes[1, 0].axhline(0, color="#777777", lw=0.75, ls="--", zorder=0)
    axes[1, 0].set(
        xlabel="Contour separation (bp)",
        ylabel="Split-2 − Full\ninternal distance (nm)",
        xlim=(0, 200),
        ylim=(-5.3, 0.9),
    )
    axes[1, 0].legend(
        handles=[
            Line2D([0], [0], color=SPLIT_COLORS[1], lw=1.55, label="mean ± SD"),
            Line2D([0], [0], color="#8E6AA9", lw=0.7, label="trajectory"),
        ],
        frameon=False,
        loc="lower left",
        fontsize=6.7,
        handlelength=1.6,
    )
    panel_label(axes[1, 0], "C", x=-0.14)

    # D: measured occupancy/network covariates do not track Split-2 Rg.
    mediator_specs = [
        ("direct_DNA_OCT4_n", "DNA-bound OCT4"),
        ("direct_OO_pair_count", "Direct O–O pairs"),
        ("physical_strict_OMO_pair_count", "Physical O–M–O pairs"),
        (
            "physical_DNA_bound_minsep_ge_20bp_pair_count",
            "Physical O–M–O ≥20 bp",
        ),
        (
            "parent_DNA_bound_minsep_ge_20bp_pair_count",
            "Parent O–M–O ≥20 bp",
        ),
    ]
    med = mediator[mediator["system"] == "M10O90D1-S2"]
    y = np.arange(len(mediator_specs))[::-1]
    for ypos, (metric, _) in zip(y, mediator_specs):
        values = (
            med[med["metric"] == metric]
            .set_index("replicate")
            .reindex(base.REPEATS)["pearson_r_with_DNA_Rg"]
            .to_numpy(float)
        )
        axes[1, 1].scatter(
            values,
            ypos + replicate_offsets(3, 0.10),
            s=18,
            facecolor="white",
            edgecolor="#555555",
            linewidth=0.70,
            zorder=4,
        )
        axes[1, 1].errorbar(
            values.mean(),
            ypos,
            xerr=values.std(ddof=1),
            fmt="D",
            color=SPLIT_COLORS[1],
            markeredgecolor="white",
            markeredgewidth=0.5,
            capsize=2.4,
            zorder=3,
        )
    axes[1, 1].axvline(0, color="#777777", lw=0.75, ls="--", zorder=0)
    axes[1, 1].set_yticks(y, [label for _, label in mediator_specs])
    axes[1, 1].set(
        xlabel=r"Within-trajectory Pearson $r$ with DNA $R_g$",
        xlim=(-0.38, 0.38),
        ylim=(-0.65, len(y) - 0.35),
    )
    axes[1, 1].tick_params(axis="y", labelsize=6.5)
    panel_label(axes[1, 1], "D", x=-0.24)

    save_supporting(fig, out / "figS9_split2_DNA_compaction_mechanism")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    tables_dir = output / "tables"
    out = output / "figures" / "supplementary"
    out.mkdir(parents=True, exist_ok=True)

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

    plot_composition_diagnostics(tables, out)
    plot_cutoff_sensitivity(tables, out)
    plot_med1_partner_composition(tables, out)
    plot_med1_reference_snapshots(tables, out)
    plot_dna_neighborhood_composition(tables, out)
    plot_contact_type_and_residue_profiles(tables, out)
    plot_dna_conformation(tables, out)
    plot_chain_continuity_diagnostics(tables, out)
    obsolete_stems = [
        "figS1_cutoff_sensitivity",
        "figS1b_DNA_effect_cutoff_sensitivity",
        "figS2_composition_diagnostics",
        "figS4_chain_continuity_diagnostics",
        "figS5_MED1_rich_spatial_snapshots",
        "figS9_DNA_network_contact_details",
        "figS12_degree_distribution",
        "figS13_network_time_stability",
        "figS15_DNA_conformation",
        "figS9_split2_DNA_compaction_mechanism",
    ]
    for stem in obsolete_stems:
        for suffix in [".pdf", ".png"]:
            (out / f"{stem}{suffix}").unlink(missing_ok=True)
    print(f"Wrote eight standardized supplementary figures to {out}")


if __name__ == "__main__":
    main()
