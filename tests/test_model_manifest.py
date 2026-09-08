from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from model_manifest import sha256_file, verify_model_entry  # noqa: E402


class ModelManifestTest(unittest.TestCase):
    def write_case(self, root: Path) -> tuple[Path, Path]:
        artifact = root / "model.pt"
        artifact.write_bytes(b"trusted-model")
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "models": {
                        "object": {
                            "artifact_path": "model.pt",
                            "sha256": sha256_file(artifact),
                            "size_bytes": artifact.stat().st_size,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest, artifact

    def test_accepts_exact_pinned_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, artifact = self.write_case(Path(directory))
            entry, resolved = verify_model_entry(manifest, "object", artifact)
            self.assertEqual(resolved, artifact.resolve())
            self.assertEqual(entry["sha256"], sha256_file(artifact))

    def test_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, artifact = self.write_case(Path(directory))
            artifact.write_bytes(b"changed-model")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_model_entry(manifest, "object", artifact)

    def test_rejects_unpinned_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _artifact = self.write_case(root)
            other = root / "other.pt"
            other.write_bytes(b"trusted-model")
            with self.assertRaisesRegex(ValueError, "not pinned"):
                verify_model_entry(manifest, "object", other)


if __name__ == "__main__":
    unittest.main()
