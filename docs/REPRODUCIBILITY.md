# Reproducibility and scope

## Simulation provenance

The corrected production package used OpenABC 1.0.7 and recorded OpenMM
8.3.1.dev-6e13f13 for all 51 protein-containing runs. The supplied source and
MOFF/MRG parameters are copied from that package, not from the newer global
OpenABC installation. `production_runtime.tsv` records the actual run settings
and SHA-256 hashes of the original starting coordinates and serialized systems.

The planned `task_manifest.tsv` has the standard 10-fs schedule. Two completed
trajectories instead used 5 fs: `work-1/M15O85` and `work-2/M50O50`. These used
1,000,000,000 integration steps and an output interval of 200,000 steps,
retaining a 5-μs duration and 1-ns coordinate interval. To rerun either, pass
`--timestep-fs 5 --total-steps 1000000000 --output-interval 200000` to the runner.
Do not silently treat planned settings as the settings of those actual runs.

The DNA-only controls use the separate historical script supplied under
`simulation/dna_only/`. This script is identical across the original three
replicates, does not explicitly set/store its stochastic seeds, and initializes
new positions on each invocation. It can regenerate the protocol, not an
identical stochastic trajectory. The original coordinates and systems belong
in the future data archive.

## Model and analysis definitions

- MOFF: one protein bead per residue; MRG: one bead per DNA nucleotide.
- MED1 restrained intervals: 77–232, 252–274, 287–349, 361–518; OCT4:
  136–217 and 234–286. These are force-field restraint intervals.
- Fixed protein-residue concentration for the 100-chain composition series;
  50-chain MED1-only references remain separate.
- DNA-addition pairs retain their protein coordinates/composition/box;
  physical Split-2 cuts MED1 after residue 807 and rebuilds exclusions.
- Three trajectories per condition; most contact analyses sample 40 frames
  at 100-ns intervals; primary cutoff 1.2 nm, with 1.0/1.4-nm checks.
- DNA-linked membership follows physical chains in the DNA-containing graph
  component. The `d90` contrast depends on membership definition and must not
  be treated as a universal contraction of all proteins.

Split-4 is retained in source/inventory for provenance and topology validation,
not as a current plotted result. Its three cuts also delete 244 retained native
pairs per MED1 sequence, so it is not a clean chain-length-only control.

## Release adaptations

Source hashes and destination hashes are recorded in `source_provenance.json`.
The production runner and force-field package are copied without modification.
Analysis changes are limited to portable input paths and extracting existing
graphics and dense-Rg functions. The excluded historical numerical driver is
not needed to draw the molecular sequence diagrams.

`reproduce_figures.py` reads the archived derived tables directly. It calls the
current plotting functions without the old standalone script's blanket cache
existence check, because those plotting functions do not consume those caches.
Recomputing numerical results from trajectories remains a separate workflow
and retains its cache/signature checks. Figure regeneration from saved tables
is not an independent reproduction of the underlying simulation results.

The environment YAML is a reconstruction specification, not a historical
lockfile. See `VALIDATION.md` for checks actually performed for this release;
no fresh production simulation should be inferred from a plotting smoke test.

