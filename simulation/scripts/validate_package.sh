#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$package_root/OpenABC${PYTHONPATH:+:$PYTHONPATH}"

python "$package_root/scripts/validate_package.py"
python "$package_root/scripts/validate_split_topology.py"
python "$package_root/scripts/validate_replicate_groups.py"
python "$package_root/scripts/check_environment.py"
bash -n \
  "$package_root/common/task_run.sh" \
  "$package_root/scripts/create_environment.sh" \
  "$package_root/scripts/run_one.sh" \
  "$package_root/scripts/update_checksums.sh" \
  "$package_root/slurm/submit_array.slurm" \
  "$package_root/slurm/run_work_task.slurm" \
  "$package_root/slurm/run_stability_task.slurm" \
  "$package_root/slurm/submit_work_individual_jobs.sh" \
  "$package_root/slurm/run_replicate_group.slurm" \
  "$package_root/slurm/submit_replicate_groups.sh"
