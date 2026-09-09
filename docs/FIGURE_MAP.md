# Current figure-to-code map

All numerical/statistical summaries use trajectories as the independent unit
(three per condition), not individual frames. Rendering preserves the current
figure definitions; it does not revive the removed S9 mechanism analysis.

| Figure | Rendering function | Required input |
| --- | --- | --- |
| Fig. 1 | `plot_reframed_figures.plot_figure1` | Derived tables, molecular input PDBs/PyMOL assets, protein snapshots |
| Fig. 2 | `plot_reframed_figures.plot_figure2` | Network/DNA tables and matched −DNA/+DNA snapshots |
| Fig. 3 | `plot_reframed_figures.plot_figure3` | DNA-linked radial profiles and selected +DNA frames |
| Fig. 4 | `plot_reframed_figures.plot_figure4` | Full/Split-2 comparisons, model renders, selected +DNA frames |
| S1 | `plot_reframed_supplementary.plot_composition_diagnostics` | Seven composition/reference snapshot conditions |
| S2 | `plot_cutoff_sensitivity` | Derived tables only |
| S3 | `plot_med1_partner_composition` | Derived tables only |
| S4 | `plot_med1_reference_snapshots` | MED1-reference tables and frames |
| S5 | `plot_dna_neighborhood_composition` | Derived tables only |
| S6 | `plot_contact_type_and_residue_profiles` | Contact tables and residue profiles only |
| S7 | `plot_dna_conformation` | DNA conformation tables and selected frames, including DNA-only |
| S8 | `plot_chain_continuity_diagnostics` | Full/Split-2 connectivity/occupancy, dense-Rg trajectory means, snapshots |

S8D uses the trajectory means of 4,000 stored frames (1.001–5.000 μs),
calculated by the existing functions extracted into `rg_distribution_core.py`
and exposed by `recompute_rg_summary.py`. Split-4 is excluded from S8.
S8C retains blue left-axis and red right-axis labels/ticks. S9 is not rendered.

The four-block MED1 graph in S8B is a common-resolution representation of
**Full and Split-2**, with the appropriate covalent links restored. Four graph
blocks do not imply that the Split-4 simulation is included in that figure.

