# Data-to-figure map for FIG-10-10

The reference is `reference_figures/`, copied without modification from the
final `FIG-10-10` export. Table paths below are relative to
`analysis/outputs/tables/`. Plot functions live in
`plot_reframed_figures.py` (main figures),
`plot_reframed_supplementary.py` (supporting figures), and
`plot_academic_toc.py` (TOC).

The final S1/S2/S3/S8 PDFs use their original Arial typography and line widths;
the other figures use Nimbus Sans. S2/S3 also retain the layout pass performed
by their original PNG-then-PDF export (the preliminary PNG is now kept in
memory). These settings preserve the supplied final figure set.

| Figure | Plot function | Principal numeric or molecular inputs |
| --- | --- | --- |
| 1 | `plot_figure1` | `balanced_cluster_repeat.csv`, `balanced_frame_repeat.csv`, `system_design.csv`; OCT4/MED1 PDBs; Figure 1B render assets; four selected original trajectory frames |
| 2 | `plot_figure2` | `balanced_frame_repeat.csv`, `balanced_paired.csv`, `DNA_conformation_with_D1_by_replicate.csv`, `system_design.csv` |
| 3 | `plot_figure3` | `mechanism_repeat.csv`, `mechanism_radial_histogram_group.csv`, `mechanism_radial_histogram_repeat.csv`, `mechanism_framewise.csv`; four selected DNA-linked snapshots |
| 4 | `plot_figure4` | `mechanism_repeat.csv`, `mechanism_split_effects_repeat.csv`, `mechanism_framewise.csv`; the matched Full/Split-2 frames at work-1/frame 2389 |
| S1 | `plot_med1_plddt` | AlphaFold confidence values from `simulation/inputs/MED1-alphafold.pdb` |
| S2 | `plot_cutoff_sensitivity` | `balanced_cluster_repeat.csv`, `system_design.csv` |
| S3 | `plot_med1_partner_composition` | `balanced_framewise_network_contacts_DNA.csv`, `system_design.csv` |
| S4 | `plot_composition_diagnostics` | `balanced_framewise_network_contacts_DNA.csv`, `D1_unaffected_control_sampled.csv`, `trajectory_manifest.csv`; one selected frame for each of the 16 displayed systems |
| S5 | `plot_contact_type_and_residue_profiles` | `balanced_frame_repeat.csv`, `balanced_site_profiles_group_summary.csv`, `system_design.csv` |
| S6 | `plot_dna_neighborhood_composition` | `balanced_framewise_network_contacts_DNA.csv`, `system_design.csv` |
| S7 | `plot_chain_continuity_diagnostics` | `mechanism_repeat.csv`, `balanced_cluster_repeat.csv`, `balanced_framewise_network_contacts_DNA.csv`, `system_design.csv` |
| S8 | `plot_protein_charge_distribution` | Charges extracted from `simulation/tasks/work/M10O90D1/system.xml`, `molecule_map.tsv`, and the input protein PDB sequences |
| TOC | `plot_toc` | `mechanism_framewise.csv`, `trajectory_manifest.csv`, the OCT4-only DNA-linked frame and the same matched Full/Split-2 snapshots as Figure 4 |

Each loader's full input list is declared in `load_tables()`. The repository
contains their union (29 CSVs) plus seven sampling-validation and selection
tables. `data/figure_snapshot_selection.csv` specifies every displayed
snapshot and its original replicate/frame ID; the per-trajectory sidecars
map those IDs to the stored sparse DCD indices.

## Definitions retained from the final figures

- Frame-level production statistics use exactly 1,000 frames per trajectory,
  with three independent trajectory means per system. Error bars summarize
  those trajectory means, rather than treating frames as independent repeats.
- Figure 2B counts protein partners only, excluding DNA neighbors.
- The protein contact graph excludes DNA as a bridging graph node.
- S6 has two panels. A uses species-specific physical-chain denominators;
  B reports fractions in the complete DNA-linked component.
- S7 has three panels. B uses the largest component by residue weight in
  the fixed-block graph, including retained covalent links; DNA is not a node.
- S7C counts physical MED1 fragments separately. Its Full/Split-2 denominators
  are 10 and 20; the Split-2 mean is 0.23813333333333334. The alternative
  parental-MED1 statistic is a different quantity and is not substituted here.
- Split-4 systems remain in the task and source tables because the complete
  analysis workflow includes them. The final comparison panels select the
  Full/Split-2 systems as in `FIG-10-10`.
