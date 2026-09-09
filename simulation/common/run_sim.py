#!/usr/bin/env python3
"""Run one balanced-design MED1/OCT4/DNA OpenMM task.

In addition to refreshing MED1 exclusions after native-pair/local-geometry
filtering, this runner supports physical Split-2 and Split-4 controls.  A
split removes every bond, angle, dihedral, native pair, and corresponding
nonbonded exclusion that crosses a designated fragment boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "OpenABC"))

import numpy as np
import pandas as pd
from openabc.forcefields import MOFFMRGModel
from openabc.forcefields.parsers import MOFFParser, MRGdsDNAParser

import openmm as mm
import openmm.app as app
import openmm.unit as unit


EXPECTED_MED1_EXCLUSIONS_BEFORE = 6387
EXPECTED_MED1_EXCLUSIONS_AFTER = 3651
EXPECTED_STALE_MED1_EXCLUSIONS = 2736
EXPECTED_MED1_EXCLUSIONS_BY_FRAGMENTS = {1: 3651, 2: 3650, 4: 3404}
MED1_RETAINED_REGIONS = (
    np.arange(76, 232),
    np.arange(251, 274),
    np.arange(286, 349),
    np.arange(360, 518),
)
OCT4_RETAINED_REGIONS = (np.arange(135, 217), np.arange(233, 286))
PARTICLE_LENGTHS = {"MED1": 1581, "OCT4": 360, "DNA": 400}
MED1_SPLIT_CUTS_AFTER_RESIDUE = {
    1: (),
    2: (807,),
    4: (354, 807, 1171),
}

STOP_REQUESTED = False


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        f"Received signal {signum}; a checkpoint will be written after the "
        "current reporting block.",
        flush=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_checkpoint_atomic(simulation: app.Simulation, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    simulation.saveCheckpoint(str(tmp))
    os.replace(tmp, path)


def save_state_atomic(simulation: app.Simulation, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    simulation.saveState(str(tmp))
    os.replace(tmp, path)


def load_task(task_id: str) -> dict[str, Any]:
    manifest = PACKAGE_ROOT / "task_manifest.tsv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [row for row in rows if row["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"Task {task_id!r} occurs {len(matches)} times in {manifest}")
    row = matches[0]
    return {
        "task_id": row["task_id"],
        "system": row["system"],
        "replicate": row["replicate"],
        "tier": row["tier"],
        "series_tags": row["series_tags"],
        "n_med1": int(row["n_med1"]),
        "n_oct4": int(row["n_oct4"]),
        "n_dna": int(row["n_dna"]),
        "med1_fragments": int(row["med1_fragments"]),
        "box_nm": float(row["box_nm"]),
        "coord_seed": int(row["coord_seed"]),
        "seed": int(row["seed"]),
        "total_steps": int(row["total_steps"]),
        "output_interval": int(row["output_interval"]),
        "parent_task": row["parent_task"],
    }


def filter_native_pairs(
    parser: MOFFParser,
    retained_regions: tuple[np.ndarray, ...],
    allow_between_regions: bool = False,
) -> None:
    old_pairs = parser.native_pairs.copy()
    retained = pd.DataFrame(columns=old_pairs.columns)
    retained_sets = [set(int(x) for x in region) for region in retained_regions]
    retained_union = set().union(*retained_sets)
    for _, row in old_pairs.iterrows():
        a1, a2 = sorted((int(row["a1"]), int(row["a2"])))
        if allow_between_regions:
            keep = a1 in retained_union and a2 in retained_union
        else:
            keep = any(a1 in region and a2 in region for region in retained_sets)
        if keep:
            retained.loc[len(retained.index)] = row
    parser.native_pairs = retained


def filter_local_geometry(
    parser: MOFFParser, retained_regions: tuple[np.ndarray, ...]
) -> None:
    angle_mask = np.zeros(len(parser.protein_angles.index), dtype=bool)
    dihedral_mask = np.zeros(len(parser.protein_dihedrals.index), dtype=bool)
    for region in retained_regions:
        angle_mask |= (
            parser.protein_angles["a1"].isin(region)
            & parser.protein_angles["a3"].isin(region)
        ).to_numpy()
        dihedral_mask |= (
            parser.protein_dihedrals["a1"].isin(region)
            & parser.protein_dihedrals["a4"].isin(region)
        ).to_numpy()
    parser.protein_angles = parser.protein_angles.loc[angle_mask].copy()
    parser.protein_dihedrals = parser.protein_dihedrals.loc[dihedral_mask].copy()


def split_med1_topology(
    parser: MOFFParser, cuts_after_residue: tuple[int, ...]
) -> dict[str, int]:
    """Remove every interaction whose atoms span two MED1 fragments.

    Cut positions are one-based residue counts.  For example, a cut after
    residue 807 separates zero-based particles 806 and 807.
    """
    removed: dict[str, int] = {}
    cuts = np.asarray(cuts_after_residue, dtype=int)
    for attr_name in (
        "protein_bonds",
        "protein_angles",
        "protein_dihedrals",
        "native_pairs",
    ):
        table = getattr(parser, attr_name)
        atom_columns = [
            column for column in ("a1", "a2", "a3", "a4")
            if column in table.columns
        ]
        fragment_ids = [
            np.searchsorted(cuts, table[column].to_numpy(dtype=int), side="right")
            for column in atom_columns
        ]
        keep = np.ones(len(table.index), dtype=bool)
        for ids in fragment_ids[1:]:
            keep &= ids == fragment_ids[0]
        removed[attr_name] = int((~keep).sum())
        setattr(parser, attr_name, table.loc[keep].copy().reset_index(drop=True))
    parser.parse_exclusions()
    return removed


def build_med1_parser(
    work_dir: Path, med1_fragments: int
) -> tuple[MOFFParser, dict[str, Any]]:
    if med1_fragments not in MED1_SPLIT_CUTS_AFTER_RESIDUE:
        raise ValueError(
            f"Unsupported med1_fragments={med1_fragments}; expected 1, 2, or 4"
        )
    parser = MOFFParser.from_atomistic_pdb(
        str(PACKAGE_ROOT / "inputs" / "MED1-alphafold.pdb"),
        str(work_dir / "MED1_CA.pdb"),
    )
    before = len(parser.exclusions.index)
    # The original MED1 selection used the union of all retained residue
    # ranges, so native pairs connecting two different retained ranges remain.
    filter_native_pairs(
        parser, MED1_RETAINED_REGIONS, allow_between_regions=True
    )
    filter_local_geometry(parser, MED1_RETAINED_REGIONS)

    # THE MED1 FIX: rebuild exclusions after all bonded/native-pair filtering.
    parser.parse_exclusions()
    after_filter_refresh = len(parser.exclusions.index)
    stale_removed = before - after_filter_refresh
    if (
        before != EXPECTED_MED1_EXCLUSIONS_BEFORE
        or after_filter_refresh != EXPECTED_MED1_EXCLUSIONS_AFTER
        or stale_removed != EXPECTED_STALE_MED1_EXCLUSIONS
    ):
        raise RuntimeError(
            "MED1 exclusion guard failed: "
            f"before={before}, after={after_filter_refresh}, "
            f"removed={stale_removed}; expected "
            f"{EXPECTED_MED1_EXCLUSIONS_BEFORE}, "
            f"{EXPECTED_MED1_EXCLUSIONS_AFTER}, "
            f"{EXPECTED_STALE_MED1_EXCLUSIONS}"
        )

    cuts = MED1_SPLIT_CUTS_AFTER_RESIDUE[med1_fragments]
    removed_interactions = (
        split_med1_topology(parser, cuts) if cuts else {
            "protein_bonds": 0,
            "protein_angles": 0,
            "protein_dihedrals": 0,
            "native_pairs": 0,
        }
    )
    after_split = len(parser.exclusions.index)
    expected_after_split = EXPECTED_MED1_EXCLUSIONS_BY_FRAGMENTS[med1_fragments]
    if after_split != expected_after_split:
        raise RuntimeError(
            "MED1 split exclusion guard failed: "
            f"Split-{med1_fragments} has {after_split}, "
            f"expected {expected_after_split}"
        )
    audit: dict[str, Any] = {
        "before_filter_refresh": before,
        "after_filter_refresh": after_filter_refresh,
        "stale_removed": stale_removed,
        "fragments": med1_fragments,
        "cuts_after_residue": list(cuts),
        "interactions_removed_by_split": removed_interactions,
        "after_split": after_split,
        "exclusions_removed_by_split": after_filter_refresh - after_split,
    }
    return parser, audit


def build_oct4_parser(work_dir: Path) -> MOFFParser:
    parser = MOFFParser.from_atomistic_pdb(
        str(PACKAGE_ROOT / "inputs" / "OCT4.pdb"),
        str(work_dir / "OCT4_CA.pdb"),
    )
    filter_native_pairs(parser, OCT4_RETAINED_REGIONS)
    filter_local_geometry(parser, OCT4_RETAINED_REGIONS)
    parser.parse_exclusions()
    return parser


def build_dna_parser(work_dir: Path) -> MRGdsDNAParser:
    return MRGdsDNAParser.from_atomistic_pdb(
        str(PACKAGE_ROOT / "inputs" / "all_atom_200bpDNA.pdb"),
        str(work_dir / "dsDNA_CG.pdb"),
    )


def validate_topology(model: MOFFMRGModel, topology: app.Topology) -> None:
    topology_records = [
        (atom.name, atom.residue.name) for atom in topology.atoms()
    ]
    model_records = list(zip(model.atoms["name"], model.atoms["resname"]))
    if topology_records != model_records:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(topology_records, model_records)
                )
                if actual != expected
            ),
            min(len(topology_records), len(model_records)),
        )
        raise RuntimeError(
            f"start.pdb/model topology mismatch at particle {mismatch}; "
            f"PDB particles={len(topology_records)}, "
            f"model particles={len(model_records)}"
        )


def build_system(
    config: dict[str, Any], start_pdb: Path, scratch: Path
) -> tuple[MOFFMRGModel, app.PDBFile, dict[str, Any]]:
    med1_parser = None
    med1_audit = None
    if config["n_med1"]:
        med1_parser, med1_audit = build_med1_parser(
            scratch, config["med1_fragments"]
        )
    oct4_parser = build_oct4_parser(scratch) if config["n_oct4"] else None
    dna_parser = build_dna_parser(scratch) if config["n_dna"] else None

    expected_particles = (
        config["n_med1"] * PARTICLE_LENGTHS["MED1"]
        + config["n_oct4"] * PARTICLE_LENGTHS["OCT4"]
        + config["n_dna"] * PARTICLE_LENGTHS["DNA"]
    )
    pdb = app.PDBFile(str(start_pdb))
    if pdb.topology.getNumAtoms() != expected_particles:
        raise RuntimeError(
            f"{config['task_id']}: start.pdb has {pdb.topology.getNumAtoms()} "
            f"particles; expected {expected_particles}"
        )

    model = MOFFMRGModel()
    if med1_parser is not None:
        for _ in range(config["n_med1"]):
            model.append_mol(med1_parser)
    if oct4_parser is not None:
        for _ in range(config["n_oct4"]):
            model.append_mol(oct4_parser)
    if dna_parser is not None:
        for _ in range(config["n_dna"]):
            model.append_mol(dna_parser)

    validate_topology(model, pdb.topology)
    model.native_pairs.loc[:, "epsilon"] = 6.0
    model.create_system(
        pdb.topology,
        box_a=config["box_nm"],
        box_b=config["box_nm"],
        box_c=config["box_nm"],
    )
    model.add_protein_bonds(force_group=1)
    model.add_protein_angles(force_group=2, verbose=True)
    model.add_protein_dihedrals(force_group=3)
    model.add_native_pairs(force_group=4)
    model.add_dna_bonds(force_group=5)
    model.add_dna_angles(force_group=6)
    model.add_dna_fan_bonds(force_group=7)
    model.add_contacts(force_group=8)
    model.add_elec_switch(
        82 * unit.millimolar,
        300 * unit.kelvin,
        force_group=9,
    )

    expected_exclusions = len(model.exclusions.index)
    nonbonded_exclusion_counts = [
        force.getNumExclusions()
        for force in model.system.getForces()
        if isinstance(force, mm.CustomNonbondedForce)
    ]
    if len(nonbonded_exclusion_counts) != 2:
        raise RuntimeError(
            "Expected exactly two CustomNonbondedForce objects "
            f"(contact and electrostatics), found {len(nonbonded_exclusion_counts)}"
        )
    if any(count != expected_exclusions for count in nonbonded_exclusion_counts):
        raise RuntimeError(
            f"Nonbonded-force exclusion mismatch: model={expected_exclusions}, "
            f"forces={nonbonded_exclusion_counts}"
        )

    audit = {
        "med1_exclusions_per_chain": med1_audit,
        "assembled_exclusions": expected_exclusions,
        "custom_nonbonded_force_exclusions": nonbonded_exclusion_counts,
        "particles": expected_particles,
        "med1_sequence_equivalents": config["n_med1"],
        "med1_physical_chains": (
            config["n_med1"] * config["med1_fragments"]
        ),
    }
    return model, pdb, audit


def write_metadata(
    output_dir: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    audit: dict[str, Any],
    start_pdb: Path,
    system_sha256: str,
    run_parameters: dict[str, Any],
) -> None:
    import mdtraj
    import openabc
    import pandas

    metadata = {
        "task": config,
        "platform": args.platform,
        "precision": args.precision,
        "openmm_version": mm.version.full_version,
        "openabc_version": openabc.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pandas.__version__,
        "mdtraj_version": mdtraj.__version__,
        "start_pdb_sha256": sha256_file(start_pdb),
        "system_xml_sha256": system_sha256,
        "exclusion_audit": audit,
        "run_parameters": run_parameters,
    }
    atomic_write_text(
        output_dir / "run_metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )


def instantaneous_temperature_kelvin(
    system: mm.System, state: mm.State
) -> float:
    degrees_of_freedom = 0
    for particle_index in range(system.getNumParticles()):
        mass = system.getParticleMass(particle_index)
        if mass.value_in_unit(unit.dalton) > 0:
            degrees_of_freedom += 3
    degrees_of_freedom -= system.getNumConstraints()
    if any(isinstance(force, mm.CMMotionRemover) for force in system.getForces()):
        degrees_of_freedom -= 3
    if degrees_of_freedom <= 0:
        raise RuntimeError(f"Invalid degrees of freedom: {degrees_of_freedom}")
    return (
        2 * state.getKineticEnergy()
        / (degrees_of_freedom * unit.MOLAR_GAS_CONSTANT_R)
    ).value_in_unit(unit.kelvin)


def run(args: argparse.Namespace) -> int:
    config = load_task(args.task_id)
    task_dir = PACKAGE_ROOT / "tasks" / config["task_id"]
    start_pdb = task_dir / "start.pdb"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else task_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not start_pdb.is_file():
        raise FileNotFoundError(start_pdb)
    if (output_dir / "DONE").exists() and not args.validate_only:
        print(f"{config['task_id']} is already complete; nothing to do.")
        return 0
    instability_path = output_dir / "NUMERICAL_INSTABILITY"
    if instability_path.exists() and not args.validate_only:
        raise RuntimeError(
            f"{instability_path} exists. Archive the unstable outputs before "
            "starting or resuming this task."
        )

    with tempfile.TemporaryDirectory(
        prefix=f"openabc_{config['system']}_", dir=os.environ.get("TMPDIR")
    ) as scratch_name:
        model, pdb, audit = build_system(config, start_pdb, Path(scratch_name))

    med1_audit = audit["med1_exclusions_per_chain"]
    if med1_audit is not None:
        print(
            "MED1 exclusion audit passed: "
            f"{med1_audit['before_filter_refresh']} -> "
            f"{med1_audit['after_filter_refresh']} after filtering; "
            f"{med1_audit['after_split']} after Split-"
            f"{med1_audit['fragments']} topology.",
            flush=True,
        )
    print(
        "CustomNonbondedForce exclusions: "
        f"{audit['custom_nonbonded_force_exclusions']}",
        flush=True,
    )
    if args.validate_only:
        print(f"Validation-only build passed for {config['task_id']}.", flush=True)
        return 0

    total_steps = args.total_steps or config["total_steps"]
    output_interval = args.output_interval or config["output_interval"]
    safety_interval = args.safety_interval or output_interval
    portable_state_interval = max(
        output_interval,
        args.portable_state_interval or 10 * output_interval,
    )
    if total_steps <= 0 or output_interval <= 0 or safety_interval <= 0:
        raise ValueError("Step counts and intervals must all be positive")
    if output_interval % safety_interval != 0:
        raise ValueError(
            "output_interval must be an integer multiple of safety_interval"
        )
    if args.timestep_fs <= 0 or args.friction_per_ps <= 0:
        raise ValueError("Timestep and Langevin friction must be positive")
    if args.max_temperature_k <= 300:
        raise ValueError("max_temperature_k must be greater than 300 K")

    run_parameters = {
        "timestep_fs": args.timestep_fs,
        "friction_per_ps": args.friction_per_ps,
        "total_steps": total_steps,
        "output_interval": output_interval,
        "safety_interval": safety_interval,
        "portable_state_interval": portable_state_interval,
        "max_temperature_k": args.max_temperature_k,
        "target_simulation_time_ns": total_steps * args.timestep_fs / 1.0e6,
    }

    system_text = mm.XmlSerializer.serialize(model.system)
    system_sha256 = hashlib.sha256(system_text.encode("utf-8")).hexdigest()
    system_path = output_dir / "system.xml"
    if system_path.exists():
        previous_sha256 = sha256_file(system_path)
        if previous_sha256 != system_sha256:
            raise RuntimeError(
                f"Existing {system_path} differs from the rebuilt corrected "
                "system. Move the old output directory aside before restarting."
            )
    else:
        atomic_write_text(system_path, system_text)
    write_metadata(
        output_dir,
        config,
        args,
        audit,
        start_pdb,
        system_sha256,
        run_parameters,
    )

    temperature = 300 * unit.kelvin
    integrator = mm.LangevinMiddleIntegrator(
        temperature,
        args.friction_per_ps / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(config["seed"])
    properties = {"Precision": args.precision}
    model.set_simulation(
        integrator,
        args.platform,
        properties=properties,
        init_coord=pdb.getPositions(),
    )
    simulation = model.simulation

    checkpoint_path = output_dir / "checkpoint.chk"
    state_path = output_dir / "restart_state.xml"
    trajectory_path = output_dir / "output.dcd"
    resumed_from: str | None = None
    if checkpoint_path.exists():
        try:
            simulation.loadCheckpoint(str(checkpoint_path))
            resumed_from = "checkpoint.chk"
        except Exception as checkpoint_error:
            if not state_path.exists():
                raise RuntimeError(
                    f"Could not load {checkpoint_path}, and no portable state "
                    "is available."
                ) from checkpoint_error
            print(
                f"Checkpoint load failed ({checkpoint_error!r}); falling back "
                f"to {state_path.name}.",
                flush=True,
            )
            simulation.loadState(str(state_path))
            resumed_from = "restart_state.xml"
    elif state_path.exists():
        simulation.loadState(str(state_path))
        resumed_from = "restart_state.xml"

    if resumed_from is None:
        if trajectory_path.exists() and trajectory_path.stat().st_size:
            raise RuntimeError(
                f"{trajectory_path} exists without a restart checkpoint/state. "
                "Move old outputs aside before starting a new trajectory."
            )
        print("Minimizing initial coordinates.", flush=True)
        simulation.minimizeEnergy(maxIterations=args.minimize_max_iterations)
        simulation.context.setVelocitiesToTemperature(
            temperature, config["seed"]
        )
    else:
        if not trajectory_path.exists():
            raise RuntimeError(
                f"Loaded {resumed_from}, but {trajectory_path} is missing."
            )
        print(
            f"Resumed {config['task_id']} from {resumed_from} at step "
            f"{simulation.currentStep}.",
            flush=True,
        )

    if simulation.currentStep > total_steps:
        raise RuntimeError(
            f"Restart step {simulation.currentStep} exceeds requested total "
            f"{total_steps}."
        )

    append_dcd = resumed_from is not None and trajectory_path.stat().st_size > 0
    dcd_reporter = app.DCDReporter(
        str(trajectory_path),
        output_interval,
        append=append_dcd,
        enforcePeriodicBox=True,
    )
    state_reporter = app.StateDataReporter(
        sys.stdout,
        output_interval,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=total_steps,
        separator="\t",
    )
    simulation.reporters.extend([dcd_reporter, state_reporter])

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    last_portable_state = simulation.currentStep
    while simulation.currentStep < total_steps:
        remaining = total_steps - simulation.currentStep
        block_steps = min(safety_interval, remaining)
        simulation.step(block_steps)
        if hasattr(dcd_reporter, "_out"):
            dcd_reporter._out.flush()

        safety_state = simulation.context.getState(getEnergy=True)
        potential_kj_mol = safety_state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        kinetic_kj_mol = safety_state.getKineticEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        instantaneous_temperature = instantaneous_temperature_kelvin(
            model.system, safety_state
        )
        numerical_values = np.asarray(
            [potential_kj_mol, kinetic_kj_mol, instantaneous_temperature]
        )
        if (
            not np.isfinite(numerical_values).all()
            or instantaneous_temperature > args.max_temperature_k
        ):
            message = (
                f"task_id={config['task_id']}\n"
                f"step={simulation.currentStep}\n"
                f"potential_kj_mol={potential_kj_mol}\n"
                f"kinetic_kj_mol={kinetic_kj_mol}\n"
                f"temperature_k={instantaneous_temperature}\n"
                f"threshold_k={args.max_temperature_k}\n"
                "checkpoint_status=last_good_checkpoint_was_not_overwritten\n"
            )
            atomic_write_text(instability_path, message)
            print(
                "Numerical instability detected; the last good checkpoint "
                f"was preserved. Details: {instability_path}",
                flush=True,
            )
            return 98

        checkpoint_due = (
            simulation.currentStep % output_interval == 0
            or simulation.currentStep == total_steps
            or STOP_REQUESTED
        )
        if checkpoint_due:
            save_checkpoint_atomic(simulation, checkpoint_path)
        if (
            simulation.currentStep - last_portable_state
            >= portable_state_interval
            or simulation.currentStep == total_steps
            or STOP_REQUESTED
        ):
            save_state_atomic(simulation, state_path)
            last_portable_state = simulation.currentStep
        if STOP_REQUESTED:
            print(
                f"Stopped safely at step {simulation.currentStep}; resubmit "
                "the same task to continue.",
                flush=True,
            )
            return 99

    final_state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
        enforcePeriodicBox=True,
    )
    with (output_dir / "final.pdb").open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(
            simulation.topology,
            final_state.getPositions(),
            handle,
            keepIds=True,
        )
    save_checkpoint_atomic(simulation, checkpoint_path)
    save_state_atomic(simulation, state_path)
    atomic_write_text(
        output_dir / "DONE",
        (
            f"task_id={config['task_id']}\n"
            f"steps={simulation.currentStep}\n"
            f"timestep_fs={args.timestep_fs}\n"
            f"simulation_time_ns="
            f"{simulation.currentStep * args.timestep_fs / 1.0e6}\n"
        ),
    )
    print(
        f"Completed {config['task_id']} at step {simulation.currentStep}.",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True, help="For example: work/M10O90")
    parser.add_argument("--platform", default="CUDA", choices=("CUDA", "OpenCL", "CPU"))
    parser.add_argument(
        "--precision", default="mixed", choices=("single", "mixed", "double")
    )
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--output-interval", type=int, default=0)
    parser.add_argument("--safety-interval", type=int, default=0)
    parser.add_argument("--portable-state-interval", type=int, default=0)
    parser.add_argument("--timestep-fs", type=float, default=10.0)
    parser.add_argument("--friction-per-ps", type=float, default=0.01)
    parser.add_argument("--max-temperature-k", type=float, default=1000.0)
    parser.add_argument("--minimize-max-iterations", type=int, default=0)
    parser.add_argument("--output-dir")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
