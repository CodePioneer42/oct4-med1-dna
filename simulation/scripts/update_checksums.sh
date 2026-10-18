#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$package_root"

checksum_tmp="INPUT_SHA256SUMS.tmp"
{
  find OpenABC common docs env inputs scripts slurm -type f \
    ! -name '*.pyc' ! -path '*/__pycache__/*' -print0
  find tasks -type f \( -name start.pdb -o -name run.sh -o -name molecule_map.tsv \) -print0
  printf '%s\0' README.md task_manifest.tsv design_matrix.tsv
} | LC_ALL=C sort -z | xargs -0 sha256sum > "$checksum_tmp"
mv "$checksum_tmp" INPUT_SHA256SUMS
