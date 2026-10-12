#!/usr/bin/env python3
"""Validate the common sampling grid, protein degrees, and DNA-contact fractions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_balanced_analysis as base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=base.DEFAULT_OUTPUT / "tables")
    args = parser.parse_args()
    tables = args.tables.expanduser().resolve()
    checks = []

    def read(name):
        return pd.read_csv(tables / f"{name}.csv")

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise RuntimeError(f"{name}: {detail}")

    frames = read("balanced_framewise_network_contacts_DNA")
    clusters = read("balanced_framewise_clusters")
    mechanism_frames = read("mechanism_framewise")
    audit_frames = read("definition_audit_framewise").query("frame_scope == 'production'")
    dna_frames = read("DNA_conformation_framewise")
    dna = read("DNA_conformation_with_D1_by_replicate")
    mechanisms = read("mechanism_repeat")
    audits = read("definition_audit_repeat")
    for name, table, keys in [
        ("protein contacts", frames, ["system", "replicate"]),
        ("clusters at all cutoffs", clusters, ["system", "replicate", "cutoff_nm"]),
        ("mechanism", mechanism_frames, ["system", "replicate"]),
        ("radial membership", audit_frames, ["system", "replicate"]),
        ("DNA shape", dna_frames, ["system", "replicate"]),
    ]:
        base.validate_sampling(table, keys)
        check(f"{name}: exact common grid", True,
              f"{table.groupby(keys).ngroups} groups; {base.N_SAMPLED_FRAMES} frames each")
        repeats = table.groupby("system")["replicate"].nunique()
        check(f"{name}: three trajectory repeats", repeats.eq(3).all(), str(repeats.to_dict()))
    counts = frames.assign(block=(frames["frame"] - 1) // 1000).groupby(["system", "replicate", "block"]).size()
    check("250 frames per 1-us block", counts.eq(250).all(), f"min={counts.min()}, max={counts.max()}")
    check("DNA shape sample counts", dna["n_frames"].eq(base.N_SAMPLED_FRAMES).all(), f"{len(dna)} trajectory means")
    recalculated = dna_frames.groupby(["system", "replicate"])["DNA_Rg_nm"].mean()
    stored = dna.set_index(["system", "replicate"])["DNA_Rg_nm"].reindex(recalculated.index)
    check("DNA Rg summaries agree with sampled frames", np.allclose(stored, recalculated, atol=1e-12, rtol=0),
          f"max error={(stored-recalculated).abs().max():.3g} nm")
    merged = audits.merge(mechanisms, on=["system", "replicate"])
    finite = merged["physical_linked_bead_q90_nm"].notna()
    difference = (merged.loc[finite, "physical_linked_bead_q90_nm"] - merged.loc[finite, "DNA_linked_bead_all_q90_nm"]).abs().max()
    check("physical membership agrees across analyses", difference < 1e-6, f"max error={difference:.3g} nm")
    degree_sum = (frames["OCT4_protein_mean_degree"].fillna(0) * frames["n_OCT4"]
                  + frames["MED1_protein_mean_degree"].fillna(0) * frames["n_MED1_physical_chains"])
    check("protein degrees count only protein edges", np.allclose(degree_sum, 2 * frames["n_PP_edges"], atol=1e-8, rtol=0),
          f"max error={(degree_sum-2*frames['n_PP_edges']).abs().max():.3g}")
    for species, denominator in [("OCT4", "n_OCT4"), ("MED1", "n_MED1_physical_chains")]:
        present = (frames[denominator] > 0) & (frames["n_DNA"] > 0)
        calculated = frames.loc[present, f"N_{species}_molecules_contacting_DNA"] / frames.loc[present, denominator]
        stored_fraction = frames.loc[present, f"{species}_fraction_contacting_DNA"]
        check(f"{species}: DNA contacts normalized by actual chain count",
              np.allclose(calculated, stored_fraction, atol=1e-12, rtol=0),
              f"max error={(calculated-stored_fraction).abs().max():.3g}")
    selection = tables / "reframed_main_snapshot_selection.csv"
    if selection.exists():
        selected = pd.read_csv(selection)
        check("main snapshots use the common grid", selected["frame"].isin(base.SAMPLED_FRAMES).all(), f"{len(selected)} snapshots")
    for name in ["balanced_analysis_validation", "mechanism_validation", "definition_audit_validation"]:
        table = read(name)
        check(name, table["passed"].eq(True).all(), f"{len(table)} checks")
    pd.DataFrame(checks).to_csv(tables / "publication_1000frame_validation.csv", index=False)
    print(f"Passed {len(checks)} sampling and numerical consistency checks")


if __name__ == "__main__":
    main()
