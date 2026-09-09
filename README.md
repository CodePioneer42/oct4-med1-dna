# OCT4–MED1–DNA simulation and figure code

Code accompanying **MED1 chain continuity supports an extended DNA-linked
protein layer** (Jiahu Tang and Xiakun Chu).

Public code repository: https://github.com/CodePioneer42/oct4-med1-dna

This release separates the corrected coarse-grained simulation workflow from
historical analysis and manuscript drafting files. It contains the current
Fig. 1–4 and Supporting Fig. S1–S8 rendering code, the small derived tables
those scripts consume, molecular input structures, and the simulation and
topology-construction code. No manuscript PDF, credentials, full trajectory,
or checkpoint is included.

## Data status

Raw simulation trajectories are being prepared for deposition and will be
added through a linked data archive in a subsequent update. **Transfer has not
started and an accession/DOI is not yet available.** The initial GitHub code
release and the future trajectory archive are separate deliverables. This
repository does not yet constitute a complete public raw-data deposition.
See [data/README.md](data/README.md) for the expected files and layout.

## Contents

- `simulation/common/run_sim.py`: the production simulation runner, including
  MOFF/MRG model construction, native-restraint filtering, corrected
  nonbonded exclusions, and physical MED1 cleavage.
- `simulation/scripts/`: composition/start-structure generation and topology
  checks; `simulation/inputs/`: the OCT4, MED1, and 200-bp DNA input structures.
- `simulation/OpenABC/`: the vendored OpenABC 1.0.7 source used by the runner,
  including MOFF/MRG parameters and its original MIT license.
- `simulation/task_manifest.tsv` and `production_runtime.tsv`: planned task
  design and **actual** production settings, respectively.
- `analysis/`: current numerical analysis, plotting, PyMOL rendering, and
  graphics-only utilities; `analysis/outputs/tables/`: derived plot inputs.
- `docs/`: figure mapping, provenance hashes, and reproducibility notes.

The numerical formulae in the copied analysis scripts are unchanged. Path
adaptations and extraction of the existing graphics/Rg helpers are recorded
in [docs/source_provenance.json](docs/source_provenance.json). The obsolete
pre-correction analysis driver is **not** included. Split-4 remains in the
historical production inventory and topology checks, but is not plotted in
the current manuscript; its extra native-pair deletion is documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Installation

For analysis and plotting only:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

For simulation, use the reconstruction specification in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate oct4-med1-dna
```

The production metadata records **OpenMM 8.3.1.dev-6e13f13 and OpenABC 1.0.7**.
The current local analysis environment is newer. Installing a stable OpenMM
8.3.1 package does not reproduce the original development binary bit for bit.
The scripts prepend the vendored OpenABC path, so a globally installed newer
OpenABC is not substituted. PyMOL is optional: the five original input-model
renders are supplied, along with the `.pml` scripts needed to regenerate them.

## Reproduce figures

Four table-only Supporting figures can be regenerated without trajectories:

```bash
python analysis/reproduce_figures.py --set tables
```

This produces **S2, S3, S5, and S6**, with current labeling and layout. All
other figures contain coordinate snapshots and require trajectory files.
After the trajectory archive becomes available:

```bash
python analysis/reproduce_figures.py --set all
```

Existing data may instead be kept outside the repository:

```bash
python analysis/reproduce_figures.py --set all \
  --data-root /path/to/corrected-package/tasks \
  --dna-only-root /path/to/dna-only-replicates
```

The corrected root contains `work/SYSTEM`, `work-1/SYSTEM`, and
`work-2/SYSTEM`; the DNA-only root contains `work/D1`, `work-1/D1`, and
`work-2/D1`. New plots go to `generated/figures/`; the original manuscript and
derived plot tables are not overwritten. See [docs/FIGURE_MAP.md](docs/FIGURE_MAP.md).

## Recompute analysis or rerun simulations

```bash
# Requires the full input/trajectory archive, not just this code release.
bash scripts/recompute_analysis.sh

# Inspect the production runner without starting a simulation.
python simulation/common/run_sim.py --help

# Example: uses supplied or regenerated start coordinates in simulation/tasks.
# WARNING: this command starts a full production simulation.
bash simulation/scripts/run_one.sh work/M10O90 --platform CUDA
```

Do not use the last example to overwrite a completed production directory.
The two 5-fs reruns require the overrides in `production_runtime.tsv`; see
the reproducibility notes. The DNA-only script is a separate historical
runner and must be executed in a new, empty run directory.

## Checks and licensing

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
```

The checks do not start MD simulations. Original project code has no
redistribution license selected yet; **do not assume that OpenABC's license
applies to the entire repository**. Its vendored source retains
[the upstream MIT license](simulation/OpenABC/LICENSE). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
