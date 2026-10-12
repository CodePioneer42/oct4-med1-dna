"""A compact graphical abstract made from actual selected trajectory frames."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import plot_reframed_figures as mainfig
from publication_style import COLORS, MED1_REGIONS, OCT4_REGIONS, DNA_SNAPSHOT_COLOR


def sequence_bar(ax, protein, split=False, *, label_size=5.5):
    length = 1581 if protein == "MED1" else 360
    regions = mainfig.legacy.MED1_NATIVE_REGIONS if protein == "MED1" else mainfig.legacy.OCT4_NATIVE_REGIONS
    colors = MED1_REGIONS if protein == "MED1" else OCT4_REGIONS
    ax.set(xlim=(0, length), ylim=(0, 1))
    ax.axis("off")
    ax.add_patch(Rectangle((0, .38), length, .25, facecolor="#E6E6E6", edgecolor="#666666", lw=.5))
    for (start, stop), color in zip(regions, colors):
        ax.add_patch(Rectangle((start - 1, .38), stop - start + 1, .25, facecolor=color, edgecolor="none"))
    if split:
        ax.add_patch(Rectangle((788, .32), 38, .38, facecolor="white", edgecolor="none", zorder=4))
        ax.annotate("", (807, .67), (807, .82), arrowprops={"arrowstyle": "-|>", "lw": .6, "mutation_scale": 5, "shrinkA": 0, "shrinkB": 0, "color": COLORS["MED1"]})
        ax.text(length / 2, .16, "1–807 + 808–1581", ha="center", va="top", fontsize=label_size)
    else:
        ax.text(length / 2, .16, "1–" + str(length) + " aa", ha="center", va="top", fontsize=label_size)


def plot_toc(tables, out):
    fig = plt.figure(figsize=(3.25, 1.75))
    framewise = tables["mechanism_framewise"]
    oct4 = mainfig.select_radial_snapshot(framewise, "M0O100D1")
    matched = mainfig.select_matched_mechanism_snapshot(framewise)
    selected = [oct4, matched.iloc[0], matched.iloc[1]]
    titles = ["OCT4 + DNA", "Intact MED1", "Split-2 MED1"]
    outcomes = ["DNA-proximal OCT4", "Extended protein layer", "Reduced peripheral reach"]
    half_span = max(float(tables["trajectory_manifest"].loc[
        tables["trajectory_manifest"].system.isin([row["system"] for row in selected]), "box_nm"].max()) / 2, 41.5)
    for i, (row, title, outcome) in enumerate(zip(selected, titles, outcomes)):
        left = .018 + i * .328
        fig.text(left + .15, .955, title, ha="center", va="top", fontsize=7.2, fontweight="bold")
        sequence_bar(fig.add_axes([left + .025, .785, .25, .135]), "OCT4" if i == 0 else "MED1", split=i == 2)
        ax = fig.add_axes([left, .225, .300, .525])
        task = mainfig.base.get_task(tables["trajectory_manifest"], row["system"], str(row["replicate"]))
        snapshot = mainfig.base.build_spatial_snapshot(task, int(row["frame"]), mode="DNA_linked_component")
        mainfig.base.draw_spatial_snapshot(ax, snapshot, half_span, show_scale_bar=False)
        mainfig.draw_projected_d90_contour(ax, snapshot, float(row["DNA_linked_bead_all_q90_nm"]))
        for collection, width in zip(ax.collections[-2:], [.9, .45]):
            collection.set_linewidths([width])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(left + .15, .18, outcome, ha="center", va="center", fontsize=5.7)
        if i < 2:
            arrow_ax = fig.add_axes([left + .299, .80, .025, .12])
            arrow_ax.axis("off")
            arrow_ax.annotate("", (1, .5), (0, .5), arrowprops={"arrowstyle": "-|>", "lw": .7, "color": "#555555"})
    fig.legend(handles=[Line2D([0], [0], color=DNA_SNAPSHOT_COLOR if name == "DNA" else COLORS[name], lw=2.2, label=name)
                        for name in ["OCT4", "MED1", "DNA"]],
               loc="lower center", bbox_to_anchor=(.5, .015), ncol=3, fontsize=6.0,
               handlelength=1.0, columnspacing=1.4, borderaxespad=0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Preserve the ACS TOC dimensions and vector text in the PDF export.
    for suffix in ["pdf"]:
        fig.savefig(out / ("TOC." + suffix), dpi=600, facecolor="white")
    plt.close(fig)
