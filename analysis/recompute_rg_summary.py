#!/usr/bin/env python3
"""Recompute only the six current S8D DNA-Rg distributions and summaries."""
from pathlib import Path
import argparse
import rg_distribution_core as rg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs/tables")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _, lookup = rg.task_lookup()
    required = {(system, rep) for system in rg.RG_SYSTEMS if system != "D1" for rep in rg.REPEATS}
    missing = required - set(lookup)
    if missing:
        parser.error(f"Missing completed production trajectories: {sorted(missing)}")
    frames = rg.dense_rg_table(lookup, args.output, recompute=args.recompute)
    repeat, group = rg.summarize_rg(frames)
    for suffix, table in (("framewise", frames), ("repeat", repeat), ("group", group)):
        table.to_csv(args.output / f"dna_rg_distribution_hypothesis_validation_{suffix}.csv", index=False)
    print("Recomputed six DNA groups × three trajectories; no mechanistic hypothesis analysis was run.")


if __name__ == "__main__":
    main()
