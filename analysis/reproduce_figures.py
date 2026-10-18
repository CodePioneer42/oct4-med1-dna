#!/usr/bin/env python3
"""Render the FIG-10-10 PDF figures from the included source data.

Example: python analysis/reproduce_figures.py --set all --package-root /path/to/data
--analysis-output /path/to/results --output /path/to/figures

The default 'tables' set renders S2, S3, S5, S6, S7 without trajectories.
The 'main' set renders Figures 1-4 and TOC; 'supporting' renders S1-S8.
Run scripts/prepare_inputs.py first. Molecular snapshots use the included
selected coordinate frames and require a PyMOL executable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import warnings

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=["tables", "main", "supporting", "all"], default="tables")
    parser.add_argument("--package-root", type=Path, default=HERE.parent / "simulation")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dna-only-root", type=Path)
    parser.add_argument("--analysis-output", type=Path, default=HERE / "outputs")
    parser.add_argument("--output", type=Path, default=HERE.parent / "generated")
    parser.add_argument("--rerender-structures", action="store_true",
                        help="Regenerate the two Figure 1B molecular renders from input PDBs")
    args = parser.parse_args()
    package = args.package_root.expanduser().resolve()
    data = (args.data_root or package / "tasks").expanduser().resolve()
    dna = (args.dna_only_root or package / "dna_only").expanduser().resolve()
    analysis_output = args.analysis_output.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (analysis_output / "tables").is_dir():
        parser.error(f"Missing analysis tables: {analysis_output / 'tables'}")
    os.environ.update({
        "OCT4_MED1_PACKAGE_ROOT": str(package), "OCT4_MED1_DATA_ROOT": str(data),
        "OCT4_MED1_DNA_ONLY_ROOT": str(dna), "OCT4_MED1_OUTPUT": str(analysis_output),
    })

    import pandas as pd
    from matplotlib.font_manager import FontProperties, findfont
    import run_balanced_analysis as base
    import plot_reframed_figures as mainfig
    import plot_reframed_supplementary as supporting
    import publication_snapshots as snapshots
    from plot_academic_toc import plot_toc
    from publication_style import FONT

    findfont(FontProperties(family=FONT), fallback_to_default=False)
    if args.set in {"tables", "supporting", "all"}:
        findfont(FontProperties(family="Arial"), fallback_to_default=False)
    base.configure_style()
    warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")
    warnings.filterwarnings("ignore", message="Unknown element .*")
    warnings.filterwarnings("ignore", message="Element information is missing.*")
    snapshots.RENDER_DIR = output / "cache" / "snapshots"
    mainfig.MAIN_FIGURES = (output / "cache" / "structures" if args.rerender_structures
                           else HERE / "figure_assets")

    def relocate(tables):
        manifest = tables["trajectory_manifest"].copy()
        manifest["path"] = [str(data / row.replicate / row.system)
                            for row in manifest.itertuples(index=False)]
        tables["trajectory_manifest"] = manifest
        return tables

    def export(function, tables, folder, stem, name):
        folder.mkdir(parents=True, exist_ok=True)
        result = function(folder) if tables is None else function(tables, folder)
        (folder / f"{stem}.pdf").replace(folder / name)
        return result

    if args.set in {"main", "all"}:
        pymol = shutil.which("pymol")
        if pymol is None:
            parser.error("Molecular graphics require a PyMOL executable on PATH")
        structures = mainfig.MAIN_FIGURES
        assets = [structures / f"fig1_panel_b_{name}_pymol.png" for name in ("oct4", "med1")]
        assets.append(structures / "fig1_panel_b_scale.json")
        if not all(path.is_file() for path in assets):
            env = os.environ.copy()
            env.update(PUBLICATION_INPUT_ROOT=str(package / "inputs"),
                       PUBLICATION_STRUCTURE_OUTPUT=str(structures))
            subprocess.run([pymol, "-cq", str(HERE / "render_fig1_panel_b.pml")], env=env, check=True)
        tables = relocate(mainfig.load_tables())
        selections = []
        for index, (function, stem) in enumerate((
            (mainfig.plot_figure1, "fig1_model_design_composition"),
            (mainfig.plot_figure2, "fig2_DNA_addition_network_response"),
            (mainfig.plot_figure3, "fig3_DNA_centered_spatial_reach"),
            (mainfig.plot_figure4, "fig4_MED1_chain_continuity"),
        ), 1):
            selected = export(function, tables, output / "figures", stem, f"Fig{index}.pdf")
            if selected is not None:
                selections.append(selected.assign(figure=f"Fig{index}"))
        if selections:
            (output / "cache").mkdir(parents=True, exist_ok=True)
            pd.concat(selections, ignore_index=True).to_csv(output / "cache" / "snapshot_selection.csv", index=False)
        plot_toc(tables, output / "TOC")

    if args.set in {"tables", "supporting", "all"}:
        tables = relocate(supporting.load_tables(analysis_output / "tables"))
        for index, function, stem in (
            (1, supporting.plot_med1_plddt, "figS1_MED1_pLDDT"),
            (2, supporting.plot_cutoff_sensitivity, "figS2_cutoff_sensitivity"),
            (3, supporting.plot_med1_partner_composition, "figS3_MED1_partner_composition"),
            (4, supporting.plot_composition_diagnostics, "figS4_composition_snapshots"),
            (5, supporting.plot_contact_type_and_residue_profiles, "figS5_contact_type_and_residue_profiles"),
            (6, supporting.plot_dna_neighborhood_composition, "figS6_DNA_neighborhood_composition"),
            (7, supporting.plot_chain_continuity_diagnostics, "figS7_chain_continuity_controls"),
            (8, supporting.plot_protein_charge_distribution, "figS8_protein_charge_distribution"),
        ):
            if args.set == "tables" and index not in {2, 3, 5, 6, 7}:
                continue
            export(function, None if index in {1, 8} else tables,
                   output / "supporting", stem, f"S{index}_Fig.pdf")
    print(f"PDF figures written to {output}")


if __name__ == "__main__":
    main()
