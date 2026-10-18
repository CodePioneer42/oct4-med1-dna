# Validation of the FIG-10-10 release

The machine-readable record is [`validation_results.json`](validation_results.json).
Validation uses the supplied `FIG-10-10` PDFs as the figure authority.

- All 13 reference PDFs are byte-identical copies of the final export.
- All 51 protein task inputs pass the production package's static checks,
  including atom counts, molecular maps, matched DNA-addition coordinates,
  and matched Full/Split initial coordinates. The package's 232 included
  input/source checksums pass.
- The Split-2/Split-4 construction checks preserve sequence, mass and charge
  while removing the intended cross-boundary interactions and exclusions.
- All 22 released snapshot coordinate records are byte-identical to their
  original DCD frames. Independent MDAnalysis reads give identical float32
  coordinate arrays. Original frame identities are retained explicitly.
- Seven unit tests and 21 sampling/numerical consistency checks pass.
- Representative serialized systems load in OpenMM. The two replacement
  production seeds are read correctly. A 20-step DNA-only CPU smoke run
  finishes using the archived system and initial coordinates.
- An isolated checkout restores its own inputs, passes the sampling checks,
  and regenerates all five table-only figures using only checkout contents.

The full plotting entry point regenerates all 13 PDFs from the released
tables and molecular coordinates. `scripts/compare_figures.py` compares
Poppler's extracted text layout and 144-dpi RGB raster output with each
reference PDF. The JSON report records dimensions, exact pixel equality,
the fraction of differing pixels, and maximum/mean color-channel error.
Small molecular-render differences are reported quantitatively; equality
of PDF file hashes is not required because PDF timestamps can differ.

These checks establish input integrity, consistency of the released source
tables, and figure reproduction. They do not rerun the 54 complete production
trajectories or independently recompute every 1,000-frame contact statistic.
The repository's selected snapshot frames are explicitly insufficient for
that latter task.
