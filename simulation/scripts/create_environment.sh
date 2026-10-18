#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_exe="${CONDA_EXE:-conda}"
environment_file="${ENV_FILE:-$package_root/env/environment.yml}"
environment_name="${CONDA_ENV_NAME:-$(
  awk -F ': *' '$1 == "name" {print $2; exit}' "$environment_file"
)}"

"$conda_exe" env create -f "$environment_file"
set +u
eval "$("$conda_exe" shell.bash hook)"
conda activate "$environment_name"
set -u
python -m pip install --no-deps "$package_root/OpenABC"

PYTHONPATH="$package_root/OpenABC${PYTHONPATH:+:$PYTHONPATH}" \
  python "$package_root/scripts/check_environment.py"
