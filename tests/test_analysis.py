"""Numerical tests for sampling, protein networks, and split-chain definitions."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import run_balanced_analysis as base
import run_mechanism_extensions as mechanism


class AnalysisTests(unittest.TestCase):
    def test_exact_production_grid(self):
        frames = pd.DataFrame({"system": "test", "replicate": "work", "frame": base.SAMPLED_FRAMES})
        self.assertEqual(len(frames), 1000)
        base.validate_sampling(frames.sample(frac=1, random_state=0))
        for invalid in (frames.iloc[:-1], pd.concat([frames, frames.iloc[[0]]])):
            with self.assertRaises(RuntimeError):
                base.validate_sampling(invalid)

    def test_dna_cannot_bridge_protein_components(self):
        molecules = base.Molecules(
            atom_molecule=np.arange(4), atom_residue=np.zeros(4, dtype=int),
            molecule_type=np.array([base.TYPE_OCT4, base.TYPE_MED1, base.TYPE_OCT4, base.TYPE_DNA]),
            molecule_length=np.ones(4, dtype=int), parent_med1=np.array([-1, 0, -1, -1]),
            starts=np.arange(4), stops=np.arange(1, 5),
        )
        pairs = np.array([[0, 1], [1, 3], [2, 3]])
        labels, sizes, edges = base.protein_components(molecules, pairs)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])
        np.testing.assert_array_equal(sorted(sizes), [1, 2])
        np.testing.assert_array_equal(edges, [[0, 1]])

    def test_dna_shape_is_invariant_to_periodic_translation(self):
        z = np.linspace(0, 65, 200)
        first = np.column_stack([np.zeros(200), np.ones(200), z])
        second = np.column_stack([np.zeros(200), -np.ones(200), z[::-1]])
        dna = np.vstack([first, second])
        reference = base.dna_shape(dna, 75)
        translated = base.dna_shape(np.mod(dna + [53, 74.5, 60], 75), 75)
        np.testing.assert_allclose(reference, translated, atol=1e-12, rtol=0)

    def test_split2_removes_only_the_broken_covalent_link(self):
        def maps(split):
            lengths = [807, 774, 360] if split else [1581, 360]
            types = [base.TYPE_MED1] * (len(lengths) - 1) + [base.TYPE_OCT4]
            stops = np.cumsum(lengths)
            molecules = base.Molecules(
                atom_molecule=np.repeat(np.arange(len(lengths)), lengths),
                atom_residue=np.r_[np.arange(1581), np.arange(360)],
                molecule_type=np.array(types), molecule_length=np.array(lengths),
                parent_med1=np.array([0] * (len(lengths) - 1) + [-1]),
                starts=np.r_[0, stops[:-1]], stops=stops,
            )
            task = base.Task("test", "work", 1, 1, 0, 2 if split else 1, 75, Path("unused"))
            return mechanism.fixed_resolution_maps(task, molecules)

        full, split = maps(False), maps(True)
        np.testing.assert_array_equal(full["block_residue_weight"], [354, 453, 364, 410, 360])
        np.testing.assert_array_equal(full["atom_block"], split["atom_block"])
        np.testing.assert_array_equal(full["atom_parent"], split["atom_parent"])
        np.testing.assert_array_equal(full["covalent_block_edges"], [[0, 1], [1, 2], [2, 3]])
        np.testing.assert_array_equal(split["covalent_block_edges"], [[0, 1], [2, 3]])


if __name__ == "__main__":
    unittest.main()
