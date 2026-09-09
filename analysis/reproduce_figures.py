#!/usr/bin/env python3
"""Render current figures to a separate directory using published table inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import run_balanced_analysis as base
import plot_reframed_figures as main_figures
import plot_reframed_supplementary as supporting


ROOT = Path(__file__).resolve().parents[1]
TABLE_FUNCTIONS = (
    supporting.plot_cutoff_sensitivity,
    supporting.plot_med1_partner_composition,
    supporting.plot_dna_neighborhood_composition,
    supporting.plot_contact_type_and_residue_profiles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("tables", "main", "supporting", "all"), default="tables")
    parser.add_argument("--output", type=Path, default=ROOT / "generated/figures")
    parser.add_argument("--data-root", type=Path, default=ROOT / "simulation/tasks")
    parser.add_argument("--dna-only-root", type=Path, default=ROOT / "data/dna_only")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "analysis/outputs/figures":
        parser.error("Use a separate output directory; do not overwrite archived assets")
    data_root = args.data_root.resolve()
    dna_root = args.dna_only_root.resolve()
    warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")
    base.configure_style()

    def remap(tables):
        manifest = tables["trajectory_manifest"].copy()
        manifest["path"] = manifest.apply(lambda r: str(data_root / r["replicate"] / r["system"]), axis=1)
        tables["trajectory_manifest"] = manifest
        return tables

    # Keep coordinate reconstruction unchanged while relocating DNA-only data.
    supporting.DNA_ONLY_ROOT = dna_root

    if args.set != "tables":
        needed = [data_root / rep / system / name
                  for rep in base.REPEATS for system in base.FULL_SYSTEMS + ["M10O90-S2", "M10O90D1-S2"]
                  for name in ("start.pdb", "output.dcd", "molecule_map.tsv")]
        if args.set in ("supporting", "all"):
            needed += [dna_root / rep / "D1" / name for rep in base.REPEATS for name in ("start.pdb", "output.dcd")]
        missing = [p for p in needed if not p.is_file()]
        if missing:
            parser.error(f"Trajectory archive required ({len(missing)} missing files; first: {missing[0]}). "
                         "Use --set tables for the four trajectory-free figures, or specify existing data roots.")

    if args.set in ("main", "all"):
        tables = remap(main_figures.load_tables())
        folder = output / "main"
        folder.mkdir(parents=True, exist_ok=True)
        for function in (main_figures.plot_figure1, main_figures.plot_figure2, main_figures.plot_figure3, main_figures.plot_figure4):
            function(tables, folder)
    if args.set in ("tables", "supporting", "all"):
        tables = remap(supporting.load_tables(ROOT / "analysis/outputs/tables"))
        folder = output / "supplementary"
        folder.mkdir(parents=True, exist_ok=True)
        functions = TABLE_FUNCTIONS if args.set == "tables" else (
            supporting.plot_composition_diagnostics, supporting.plot_cutoff_sensitivity,
            supporting.plot_med1_partner_composition, supporting.plot_med1_reference_snapshots,
            supporting.plot_dna_neighborhood_composition, supporting.plot_contact_type_and_residue_profiles,
            supporting.plot_dna_conformation, supporting.plot_chain_continuity_diagnostics,
        )
        for function in functions:
            function(tables, folder)
    print(f"Rendered {args.set} figures to {output}")


if __name__ == "__main__":
    main()
