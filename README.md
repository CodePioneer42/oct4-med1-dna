# OCT4–MED1–DNA: FIG-10-10 reproducibility materials

Materials for **Polymer Connectivity of MED1 Shapes the Spatial Organization of
DNA-Associated Protein Assemblies**. The reference figures are the 13 PDFs from
the authors' final `FIG-10-10` export: Figures 1–4, Figures S1–S8, and the TOC.
Their original bytes are preserved in [`reference_figures/`](reference_figures/).

| Material | Location |
| --- | --- |
| OCT4, MED1, and 200-bp DNA input structures | `simulation/inputs/` |
| Actual initial coordinates and serialized OpenMM systems for 51 protein tasks and 3 DNA-only controls | `simulation/input_manifest.json` and `simulation/input_objects/` |
| System construction, simulation code, force-field parameters | `simulation/common/`, `simulation/scripts/`, `simulation/OpenABC/` |
| Per-task molecular topology maps, recorded runtime metadata and seeds | `simulation/tasks/`, `simulation/production_runtime.tsv` |
| Trajectory analysis and plotting code | `analysis/` |
| Figure source tables, including individual-trajectory and framewise values | `analysis/outputs/tables/` |
| The 22 original coordinate frames used in the figures | Compressed input objects; frame/panel map in `data/figure_snapshot_selection.csv` |
| Data-to-figure map and metric definitions | [`docs/FIGURE_MAP.md`](docs/FIGURE_MAP.md) |
| Provenance, hashes and validation | `docs/source_provenance.json`, `SHA256SUMS`, [`docs/VALIDATION.md`](docs/VALIDATION.md) |

## Reproduce the figures

Create the environment, then run from the repository root:

```bash
conda env create -f environment.yml
conda activate oct4-med1-reproduce
python scripts/prepare_inputs.py
python analysis/reproduce_figures.py --set all
```

PyMOL must be on `PATH`. The main figures and later supplementary exports use
**Nimbus Sans**; on Debian/Ubuntu, install `fonts-urw-base35` if it is unavailable.
The final S1/S2/S3/S8 PDFs retain **Arial** from their earlier export, so a
locally installed regular Arial font is also required. Fonts are not bundled.
The plotting code checks the fonts instead of silently substituting them.
The version pins in `requirements-analysis.txt` describe the validated
analysis environment. Simulation production versions are recorded separately
in each task's metadata and in `simulation/env/`.

`prepare_inputs.py` restores the compressed inputs with SHA-256 verification.
Identical inputs are stored once; restoration uses about 1.62 GB of disk space.
Existing files with different contents are rejected. It is safe to rerun after
an interruption. Figure outputs and render caches are written to `generated/`.
For the five table-only supplementary figures, use `--set tables`.
Figure 1B uses the included original PyMOL renders; add `--rerender-structures`
to regenerate those two assets from their PDB files.

## Scope of the coordinates and source data

The repository includes all tabular and molecular inputs used by the final
plotting workflow. It contains **22 selected coordinate frames from 19
trajectories**, reused across 29 displayed panels. Their float32 coordinate
records are copied byte-for-byte from the original DCDs, with original frame
IDs in `trajectory_frames.json`. The sparse DCD header uses storage indices;
physical times must be taken from the sidecar map.

These coordinates reproduce the published snapshots. They are **not the
complete production trajectories or the 1,000-frame analysis dataset**.
The source tables retain the original 1,000-frame sampling (frames 1001,
1005, …, 4997; 1.001–4.997 µs, 4-ns spacing) and three independent trajectory
means per system. Recomputing all contact metrics from coordinates requires
the corresponding analysis trajectories; see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

S7C follows the final `FIG-10-10` plot: MED1 contacts are normalized by the
number of **physical chains/fragments** (10 for Full, 20 for Split-2).
This is distinct from counting each parental MED1 molecule once.

## Inspect or rerun a simulation

The archived `system.xml` contains the actual production topology, parameters
and exclusions. The construction code and the used OpenABC source/parameter
snapshot are also included. Check a restored system without running dynamics:

```bash
python simulation/run_recorded.py --task-id work/M10O90D1 --validate-only
```

To run a short new simulation from its archived initial structure and system:

```bash
python simulation/run_recorded.py --task-id work/M10O90D1 \
  --steps 1000 --platform CUDA --output-dir generated/simulation-smoke
```

Omit `--steps` only when a full 5-µs production run is intended. The runner uses
the actual recorded production seed, including the two replacement-seed runs.
The historical construction runner remains in `simulation/common/run_sim.py`.
DNA-only initial random seeds were not recorded; a new DNA-only run requires
an explicit `--seed`. New stochastic simulations need not reproduce the exact
original trajectory coordinates.

## Validation

```bash
python -m unittest discover -s tests -v
python analysis/validate_publication_sampling.py
python scripts/check_release.py
```

With Poppler's `pdftotext` and `pdftoppm` installed (the `poppler-utils`
package on Debian/Ubuntu), compare the regenerated PDFs with the references:

```bash
python scripts/compare_figures.py
```

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for structure provenance
and the preserved OpenABC license. Cite the manuscript title and the exact
Git commit used for a reproduction.
