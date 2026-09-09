from pathlib import Path
import csv
import importlib.util
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import run_balanced_analysis as base
import model_rendering
import rg_distribution_core as rg


class ReleaseTests(unittest.TestCase):
    def test_portable_package_root(self):
        self.assertEqual(base.PACKAGE_ROOT, ROOT / "simulation")

    def test_production_runtime(self):
        with (ROOT / "simulation/production_runtime.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 51)
        self.assertEqual({r["openabc_version"] for r in rows}, {"1.0.7"})
        self.assertEqual({r["task_id"] for r in rows if float(r["timestep_fs"]) == 5},
                         {"work-1/M15O85", "work-2/M50O50"})
        for row in rows:
            self.assertAlmostEqual(float(row["timestep_fs"]) * int(row["total_steps"]) / 1e9, 5)
            self.assertAlmostEqual(float(row["timestep_fs"]) * int(row["output_interval"]) / 1e6, 1)

    def test_input_residue_counts(self):
        for filename, count in (("MED1-alphafold.pdb", 1581), ("OCT4.pdb", 360)):
            self.assertEqual(len(model_rendering.read_residue_names(ROOT / "simulation/inputs" / filename)), count)

    def test_dense_rg_scope(self):
        self.assertEqual(len(rg.RG_SYSTEMS), 6)
        self.assertNotIn("M10O90D1-S4", rg.RG_SYSTEMS)
        self.assertEqual(len(rg.RG_FRAMES), 4000)

    def test_dna_unwrapping_is_translation_invariant(self):
        strand = np.column_stack((np.linspace(73, 139, 200), np.zeros(200), np.zeros(200)))
        dna = np.vstack((strand, strand[::-1] + [0, 2, 0]))
        values = []
        for shift in (np.zeros(3), np.array([13, 6, 1])):
            unwrapped = base.unwrap_dna((dna + shift) % 75, 75)
            centered = unwrapped - unwrapped.mean(axis=0)
            values.append(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        self.assertAlmostEqual(values[0], values[1], places=10)

    def test_no_legacy_analysis_driver(self):
        self.assertFalse((ROOT / "analysis/run_paper_analysis.py").exists())
        self.assertFalse(hasattr(model_rendering, "main"))


if __name__ == "__main__":
    unittest.main()
