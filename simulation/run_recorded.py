#!/usr/bin/env python3
"""Run an archived OpenMM system using its recorded production parameters."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import openmm as mm
from openmm import app, unit

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, help="e.g. work/M10O90D1 or work/D1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU", "Reference"], default="CUDA")
    parser.add_argument("--steps", type=int, help="Override the recorded production length for a short run")
    parser.add_argument("--seed", type=int, help="Required for a new DNA-only run; its original seed was not recorded")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    task = Path(args.task_id)
    if task.is_absolute() or len(task.parts) != 2 or ".." in task.parts:
        parser.error("Use a relative REPLICATE/SYSTEM task identifier")
    dna_only = task.name == "D1"
    source = ROOT / ("dna_only" if dna_only else "tasks") / task
    pdb = app.PDBFile(str(source / "start.pdb"))
    xml = (source / "system.xml").read_text()
    system = mm.XmlSerializer.deserialize(xml)
    if system.getNumParticles() != pdb.topology.getNumAtoms():
        raise RuntimeError("System and input structure particle counts differ")
    metadata = json.loads((source / "run_metadata.json").read_text()) if not dna_only else {}
    parameters = metadata.get("run_parameters", {})
    task_metadata = metadata.get("task", {})
    seed = args.seed if args.seed is not None else parameters.get("random_seed", task_metadata.get("seed"))
    timestep = parameters.get("timestep_fs", 10.0)
    friction = parameters.get("friction_per_ps", 0.01)
    steps = args.steps if args.steps is not None else parameters.get("total_steps", task_metadata.get("total_steps", 500000000))
    interval = parameters.get("output_interval", task_metadata.get("output_interval", 100000))
    record = {"task_id": args.task_id, "particles": system.getNumParticles(),
              "forces": system.getNumForces(), "random_seed": seed,
              "timestep_fs": timestep, "friction_per_ps": friction,
              "total_steps": steps, "output_interval": interval,
              "system_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
              "openmm_version": mm.__version__, "platform": args.platform}
    if args.validate_only:
        print(json.dumps(record, indent=2))
        return
    if steps < 1 or seed is None or seed < 1:
        parser.error("A positive number of steps and a positive seed are required; supply --seed for DNA-only runs")
    if args.output_dir is None:
        parser.error("Use --output-dir for new simulation results")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("The output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    temperature = 300 * unit.kelvin
    integrator = mm.LangevinMiddleIntegrator(temperature, friction / unit.picosecond, timestep * unit.femtosecond)
    integrator.setRandomNumberSeed(seed)
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {"Precision": "mixed"} if args.platform in {"CUDA", "OpenCL"} else {}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature, seed)
    for name in ("start.pdb", "system.xml", "molecule_map.tsv"):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    (output / "reproduction_parameters.json").write_text(json.dumps(record, indent=2) + "\n")
    if metadata:
        new_metadata = dict(metadata)
        new_metadata.update(platform=args.platform, openmm_version=mm.__version__)
        new_metadata["task"] = {**task_metadata, "seed": seed, "total_steps": steps,
                                "output_interval": interval}
        new_metadata["run_parameters"] = {**parameters, "random_seed": seed,
                                           "timestep_fs": timestep, "friction_per_ps": friction,
                                           "total_steps": steps, "output_interval": interval,
                                           "target_simulation_time_ns": steps * timestep / 1e6}
        (output / "run_metadata.json").write_text(json.dumps(new_metadata, indent=2) + "\n")
    simulation.reporters.append(app.DCDReporter(str(output / "output.dcd"), interval, enforcePeriodicBox=True))
    simulation.reporters.append(app.StateDataReporter(str(output / "state.csv"), min(interval, steps),
                                                     step=True, time=True, potentialEnergy=True, temperature=True))
    simulation.step(steps)
    simulation.saveCheckpoint(str(output / "final.chk"))
    (output / "DONE").write_text(f"steps={steps}\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
