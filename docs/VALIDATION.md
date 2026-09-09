# Checks performed for this code release

Date: 2026-09-09. No new production MD simulation was started.

| Check | Result |
| --- | --- |
| Python syntax and basic credential/local-path scan | Passed for the curated release |
| Unit tests | 6 passed: portable paths, runtime metadata, residue counts, Rg group/frame scope, PBC translation invariance, exclusion of legacy numerical driver |
| Shell syntax | Passed for the analysis launcher and simulation wrappers |
| Production simulation runner import and `--help` | Passed using the vendored OpenABC 1.0.7 package |
| MED1 topology validation | Passed: sequence/mass/charge preserved; Full/Split-2/Split-4 exclusions 3651/3650/3404; Split-2 removes one bond and no native pair |
| Full current figure rendering with existing local production data | Fig. 1–4 and S1–S8 generated successfully |
| Comparison to current archived PNG figures | All 12 regenerated images have identical dimensions and identical decoded pixels |
| Missing-trajectory guard | Full rendering stops with an explicit archive-required message when raw inputs are absent |
| Table-only rendering without any trajectory path | S2, S3, S5, and S6 generated successfully |
| Dense-Rg helper extraction | All 18 current-condition trajectory means match the previously derived summaries exactly |

The plotting test used the existing local trajectories only as read-only
inputs, with new figures written to `generated/`. Original figures and
manuscripts were not overwritten. The production trajectory archive is not
included in this code release and its upload has not started.

The testing environment had OpenMM 8.5.1, MDAnalysis 2.10.0, MDTraj 1.11.1,
NumPy 2.4.6, pandas 2.3.3, SciPy 1.17.1, and Matplotlib 3.10.9. Simulation
imports loaded vendored OpenABC 1.0.7, not the globally installed 1.0.9.
These are **validation-time versions**, not a replacement for the original
production versions recorded in `simulation/production_runtime.tsv`.

Not performed: a fresh GPU production rerun, full recomputation of all
trajectory-derived numerical tables, solving/installing the supplied Conda
environment from scratch, regeneration of PyMOL raster assets, or independent
validation of a remote trajectory archive. The figure comparison validates
packaging/rendering consistency, not independent scientific replication.
