#!/usr/bin/env python3
"""Analyze external trajectories and export validated numerical tables.

Example: python analysis/run_publication_analysis.py --package-root /path/to/data
--dna-only-root /path/to/dna_only --output /path/to/results --workers 4

The package contains task_manifest.tsv, inputs/*.pdb, and tasks/<replicate>/<system>/
with start.pdb, output.dcd, molecule_map.tsv, run_metadata.json, system.xml, and DONE.
DNA-only controls contain <replicate>/D1/start.pdb and output.dcd. The repository
includes production inputs, figure source tables, and the selected figure
snapshots. Recomputing the complete analysis needs the full analysis-frame
trajectories; the small figure-snapshot subset is insufficient for that step.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import warnings

HERE = Path(__file__).resolve().parent


def cache_job(kind, task, cache_dir, recompute):
    import run_balanced_analysis as base
    import run_definition_audits as audit
    import run_mechanism_extensions as mechanism

    warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")
    warnings.filterwarnings("ignore", message="Unknown element .*")
    warnings.filterwarnings("ignore", message="Element information is missing.*")
    function = {
        "balanced": base.analyze_task,
        "mechanism": mechanism.analyze_extension_task,
        "audit": audit.analyze_task,
    }[kind]
    function(task, cache_dir, recompute=recompute)
    return f"{kind}: {task.replicate}/{task.system}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-root", type=Path, default=HERE.parent / "simulation")
    parser.add_argument("--data-root", type=Path, help="Trajectory directory; default: PACKAGE_ROOT/tasks")
    parser.add_argument("--dna-only-root", type=Path, help="DNA-only controls; default: PACKAGE_ROOT/dna_only")
    parser.add_argument("--output", type=Path, default=HERE.parent / "generated/recomputed-analysis")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recompute", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--aggregate-only", action="store_true", help="Reuse existing trajectory caches")
    mode.add_argument("--cache-only", action="store_true", help="Compute trajectory caches without aggregation")
    args = parser.parse_args()
    if args.aggregate_only and args.recompute:
        parser.error("--aggregate-only and --recompute are mutually exclusive")
    package = args.package_root.expanduser().resolve()
    data = (args.data_root or package / "tasks").expanduser().resolve()
    dna = (args.dna_only_root or package / "dna_only").expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not (package / "task_manifest.tsv").is_file():
        parser.error(f"Missing input manifest: {package / 'task_manifest.tsv'}")
    if not args.aggregate_only:
        for sidecar in list(data.glob("*/*/trajectory_frames.json")) + list(dna.glob("*/D1/trajectory_frames.json")):
            if json.loads(sidecar.read_text()).get("purpose") == "figure_snapshots_only":
                parser.error("The supplied coordinates contain figure snapshots only. "
                             "Use reproduce_figures.py to redraw the figures, or supply full "
                             "analysis trajectories with --data-root and --dna-only-root.")
    os.environ.update({
        "OCT4_MED1_PACKAGE_ROOT": str(package), "OCT4_MED1_DATA_ROOT": str(data),
        "OCT4_MED1_DNA_ONLY_ROOT": str(dna), "OCT4_MED1_OUTPUT": str(output),
    })
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")

    import pandas as pd
    import run_balanced_analysis as base
    import run_definition_audits as audit
    import run_mechanism_extensions as mechanism

    if not args.aggregate_only:
        manifest = base.read_manifest(package, data)
        tasks = [base.task_from_row(pd.Series(row._asdict()))
                 for row in manifest[manifest.analysis_included].itertuples(index=False)]
        if not tasks:
            parser.error("No completed trajectories found in the specified inputs")
        jobs = []
        for task in sorted(tasks, key=lambda task: task.expected_atoms, reverse=True):
            for kind, version, included in (
                ("balanced", base.ANALYSIS_VERSION, True),
                ("mechanism", mechanism.EXTENSION_VERSION, task.system in mechanism.EXTENSION_SYSTEMS),
                ("audit", audit.AUDIT_VERSION, task.system in audit.CHAIN_SYSTEMS),
            ):
                if included:
                    cache_dir = output / "cache" / version
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    jobs.append((kind, task, cache_dir, args.recompute))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(cache_job, *job) for job in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)
    if args.cache_only:
        return

    for script, arguments in (
        ("run_balanced_analysis.py", ["--aggregate-only"]),
        ("run_mechanism_extensions.py", ["--aggregate-only"]),
        ("run_definition_audits.py", ["--aggregate-only"]),
        ("validate_publication_sampling.py", []),
    ):
        subprocess.run([sys.executable, str(HERE / script), *arguments], check=True)
    print(f"Analysis complete: {output / 'tables'}")


if __name__ == "__main__":
    main()
