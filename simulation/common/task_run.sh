#!/usr/bin/env bash
set -euo pipefail

task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "$task_dir/../../.." && pwd)"
task_id="$(basename "$(dirname "$task_dir")")/$(basename "$task_dir")"

exec "$package_root/scripts/run_one.sh" "$task_id" "$@"
