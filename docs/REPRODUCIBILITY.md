# Reproducing figures and trajectory analyses

The quick-start command in the README reproduces Figures 1–4, S1–S8 and the
TOC from the data included in this repository. It restores actual initial
structures and OpenMM systems, reads the released numerical source tables,
reconstructs contacts and cameras for the selected snapshots, and renders
the molecular views with PyMOL. It does not copy the reference PDFs to its
output directory.

`reference_figures/` supplies the unmodified target PDFs. PDF timestamps and
renderer/library versions may change binary output hashes even when the
rendered content is the same. The release validation records the comparison
method and its actual results in `docs/validation_results.json`.

## Input archive

`simulation/input_manifest.json` maps each original `start.pdb` and
`system.xml` to a SHA-256-named gzip object. All 51 protein tasks and three
DNA-only controls have their actual starting structures and serialized
systems. Identical objects are deduplicated; decompression recovers the
original bytes, not regenerated approximations. Molecular identity and
fragment-to-parent relationships are in each task's `molecule_map.tsv`.

The input archive additionally contains 22 original frames used for the
figures. Snapshot DCDs have explicit frame-ID sidecars and are not complete
analysis trajectories. `run_publication_analysis.py` rejects this snapshot-only
dataset as an input to the 1,000-frame analysis.

## Recompute numerical results from trajectories

Supply the original 5,000-frame DCDs or the exact 1,000 analysis frames with
their valid `trajectory_frames.json` maps. A full protein-data directory has
`<replicate>/<system>/start.pdb`, `output.dcd`, `system.xml`,
`molecule_map.tsv`, `run_metadata.json`, and the original completion marker
`DONE`. The DNA-only directory has `<replicate>/D1/start.pdb` and `output.dcd`.

```bash
python analysis/run_publication_analysis.py \
  --package-root simulation \
  --data-root /path/to/full-analysis-data/tasks \
  --dna-only-root /path/to/full-analysis-data/dna_only \
  --output generated/recomputed-analysis --workers 4
python analysis/reproduce_figures.py --set all \
  --package-root simulation \
  --data-root /path/to/full-analysis-data/tasks \
  --dna-only-root /path/to/full-analysis-data/dna_only \
  --analysis-output generated/recomputed-analysis \
  --output generated/recomputed-figures
```

Those complete trajectories are not distributed in this Git repository.
The framewise and trajectory-level source tables are included, so the plotted
statistics and figures can be reconstructed without rerunning dynamics.
They do not replace the coordinates needed for an independent recomputation
of every contact metric.

## Simulation provenance

The original construction and force-field code is in `simulation/common/`,
`simulation/scripts/`, and the vendored OpenABC 1.0.7 snapshot. Source molecular
templates are in `simulation/inputs/`. The actual serialized systems provide
the production bonded/nonbonded parameters and exclusions.

`simulation/production_runtime.tsv` separates the original design seed from
the seed actually used for production. The work-1/M15O85 and work-2/M50O50
production runs use replacement seeds 2026090401 and 2026090402. Both use
10 fs, 500,000,000 steps, and 100,000-step trajectory output intervals.
`simulation/run_recorded.py` reads those actual metadata values and the
archived XML. The historical `common/run_sim.py` uses the design-manifest
seed; retain this distinction when using the historical runner directly.

The DNA-only runners originally selected initial placements and velocities
without recording explicit random seeds. Their real initial coordinates and
systems are included; the missing historical seeds are not inferred.
