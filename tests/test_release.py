"""Guard against corrupted archives and accidental replacement of scientific inputs."""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_inputs import restore


class ReleaseTests(unittest.TestCase):
    def fixture(self, directory, expected=b"recorded input\n"):
        root = Path(directory)
        (root / "simulation").mkdir()
        obj = root / "simulation/input.gz"
        obj.write_bytes(gzip.compress(b"recorded input\n"))
        manifest = {"schema": "oct4-med1-gzip-inputs-v1", "files": [
            {"path": "simulation/tasks/example/start.pdb", "object": "simulation/input.gz",
             "bytes": len(expected), "sha256": hashlib.sha256(expected).hexdigest()}]}
        (root / "simulation/input_manifest.json").write_text(json.dumps(manifest))
        return root, root / manifest["files"][0]["path"]

    def test_restore_is_idempotent_and_preserves_different_existing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self.fixture(directory)
            self.assertEqual(restore(root), {"restored": 1, "already_verified": 0})
            self.assertEqual(output.read_bytes(), b"recorded input\n")
            self.assertEqual(restore(root), {"restored": 0, "already_verified": 1})
            output.write_bytes(b"another simulation input")
            with self.assertRaises(RuntimeError):
                restore(root)
            self.assertEqual(output.read_bytes(), b"another simulation input")

    def test_wrong_archive_digest_cannot_create_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self.fixture(directory, expected=b"wrong expected input")
            with self.assertRaises(RuntimeError):
                restore(root)
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
