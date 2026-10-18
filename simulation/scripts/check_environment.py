#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import mdtraj
import numpy
import openabc
import openmm
import pandas


parser = argparse.ArgumentParser()
parser.add_argument("--require-cuda", action="store_true")
args = parser.parse_args()

print(f"Python:  {sys.version.split()[0]}")
print(f"OpenMM:  {openmm.version.full_version}")
print(f"OpenABC: {openabc.__version__} ({openabc.__file__})")
print(f"MDTraj:  {mdtraj.__version__}")
print(f"NumPy:   {numpy.__version__}")
print(f"pandas:  {pandas.__version__}")
platforms = [
    openmm.Platform.getPlatform(index).getName()
    for index in range(openmm.Platform.getNumPlatforms())
]
print(f"OpenMM platforms: {', '.join(platforms)}")
if args.require_cuda and "CUDA" not in platforms:
    raise SystemExit("ERROR: OpenMM CUDA platform is unavailable.")
if "CUDA" not in platforms:
    print(
        "WARNING: CUDA platform is not visible on this node. This is normal "
        "on some login nodes; repeat this check on a GPU compute node."
    )
