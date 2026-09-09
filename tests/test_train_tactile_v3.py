from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_tactile_v3 import build_training_config  # noqa: E402


class TrainTactileV3Test(unittest.TestCase):
    def test_builds_pinned_non_overwriting_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.yaml"
            checkpoint = root / "models" / "guidetwsi" / "yolo11n_tactile.pt"
            dataset.write_text("names:\n  0: tactile_paving\n", encoding="utf-8")
            (root / "export_manifest.json").write_text(
                json.dumps({"dataset_version": "tactile-v3-public4-station1"}),
                encoding="utf-8",
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            config = build_training_config(
                dataset, checkpoint, root / "runs", "tactile-v3-public4-station1"
            )
            self.assertEqual(config["seed"], 42)
            self.assertEqual(config["imgsz"], 640)
            self.assertEqual(config["patience"], 15)
            self.assertEqual(config["init_checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(config["dataset_version"], "tactile-v3-public4-station1")

    def test_rejects_v2_checkpoint_and_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.yaml"
            checkpoint = root / "artifacts" / "candidates" / "tactile-one-class-v2" / "best.pt"
            dataset.write_text("x", encoding="utf-8")
            (root / "export_manifest.json").write_text(
                json.dumps({"dataset_version": "tactile-v3-public4-station1"}),
                encoding="utf-8",
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"v2")
            with self.assertRaisesRegex(ValueError, "GuideTWSI"):
                build_training_config(dataset, checkpoint, root / "runs", "run")

            guide = root / "models" / "guidetwsi" / "yolo11n_tactile.pt"
            guide.parent.mkdir(parents=True)
            guide.write_bytes(b"guide")
            (root / "runs" / "run").mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "run output"):
                build_training_config(dataset, guide, root / "runs", "run")


if __name__ == "__main__":
    unittest.main()
