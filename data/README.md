# Trajectory archive status and expected layout

Status: **preparation for deposition; transfer not started**.

The code release contains derived plot tables and input molecular models, not
the raw trajectories. The trajectory archive will be linked here when the
upload is complete and a persistent accession/DOI is assigned. No repository
URL or DOI has been invented. Files must not be labeled as deposited before
their remote transfer and checksum verification are complete.

For corrected protein-containing runs, place these files under
`simulation/tasks/{work,work-1,work-2}/SYSTEM/`:

- `start.pdb`: original, matched starting coordinates, not the final snapshot;
- `output.dcd`: full production trajectory;
- `system.xml`: serialized production system;
- `molecule_map.tsv` and `run_metadata.json`: already supplied with this release;
- `DONE`: original completion record, required by the numerical analysis driver.

DNA-only controls go under `data/dna_only/REPLICATE/D1/` with at least
`start.pdb`, `output.dcd`, and the original `system.xml`. These older controls
do not have the same task manifest or explicit stored random seeds as the
corrected protein-containing runs.

Keep large trajectory files out of Git. The eventual archive should include
per-file byte sizes, SHA-256 digests, the production runtime table, original
input coordinates, topologies, parameters, and enough metadata to match each
trajectory to its figure/analysis use. Validate the archive against the
original source files before marking this status complete.

`trajectory_inventory.csv` records the local file inventory and byte sizes,
with `upload_status=not_started`. Blank SHA-256 values have not yet been
computed and are not evidence of verification. To refresh the inventory and
hash the actual large files before a later upload:

```bash
python scripts/prepare_trajectory_inventory.py \
  --data-root /path/to/corrected-package/tasks \
  --dna-only-root /path/to/dna-only-replicates --sha256
```
