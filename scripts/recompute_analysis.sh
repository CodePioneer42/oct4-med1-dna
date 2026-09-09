#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_root"
analysis_python="${ANALYSIS_PYTHON:-python}"

# Deliberately opt-in: these commands rebuild the saved analysis tables.
# Requires full trajectories and the original per-task completion records.
"$analysis_python" analysis/run_balanced_analysis.py --skip-figures
"$analysis_python" analysis/run_mechanism_extensions.py
"$analysis_python" analysis/run_definition_audits.py
"$analysis_python" analysis/run_med1_functional_bridge_analysis.py
"$analysis_python" analysis/recompute_rg_summary.py --recompute

