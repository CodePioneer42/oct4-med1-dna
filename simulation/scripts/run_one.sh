#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 REPLICATE/SYSTEM [extra run_sim.py arguments]" >&2
  exit 2
fi

task_id="$1"
shift
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$package_root/OpenABC${PYTHONPATH:+:$PYTHONPATH}"

exec python "$package_root/common/run_sim.py" --task-id "$task_id" "$@"
