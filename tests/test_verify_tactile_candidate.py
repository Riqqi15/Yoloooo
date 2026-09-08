from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from model_manifest import sha256_file  # noqa: E402
from verify_tactile_candidate import validate_candidate  # noqa: E402


class VerifyTactileCandidateTest(unittest.TestCase):
    def make_candidate(self, root: Path) -> Path:
        candidate = root / "candidate"
        candidate.mkdir()
        checkpoint = candidate / "best.pt"
        checkpoint.write_bytes(b"candidate-checkpoint")
        (candidate / "metrics.json").write_text(
            json.dumps({"metrics/mask50(B)": 0.5}), encoding="utf-8"
        )
        (candidate / "training_config.json").write_text(
            json.dumps(
                {
                    "dataset_version": "station-tactile-v1",
                    "seed": 42,
                    "imgsz": 640,
                    "epochs": 80,
                    "ultralytics_version": "8.4.138",
                }
            ),
            encoding="utf-8",
        )
        (candidate / "environment.txt").write_text(
            "ultralytics==8.4.138\n", encoding="utf-8"
        )
        (candidate / "sha256.txt").write_text(
            f"{sha256_file(checkpoint)}  best.pt\n", encoding="utf-8"
        )
        (candidate / "MODEL_CARD.md").write_text(
            "# Candidate only\nNot approved for deployment.\n", encoding="utf-8"
        )
        return candidate

    def valid_model(self) -> SimpleNamespace:
        return SimpleNamespace(task="segment", names={0: "tactile_paving"})

    def test_accepts_complete_one_class_segment_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = self.make_candidate(Path(directory))
            with patch("verify_tactile_candidate.YOLO", return_value=self.valid_model()):
                result = validate_candidate(candidate)
            self.assertEqual(result["status"], "candidate_valid")
            self.assertEqual(result["labels"], {0: "tactile_paving"})

    def test_rejects_wrong_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = self.make_candidate(Path(directory))
            (candidate / "sha256.txt").write_text("0" * 64 + "  best.pt\n", encoding="utf-8")
            with patch("verify_tactile_candidate.YOLO", return_value=self.valid_model()):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    validate_candidate(candidate)

    def test_rejects_wrong_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = self.make_candidate(Path(directory))
            model = SimpleNamespace(task="detect", names={0: "tactile_paving"})
            with patch("verify_tactile_candidate.YOLO", return_value=model):
                with self.assertRaisesRegex(ValueError, "task must be segment"):
                    validate_candidate(candidate)

    def test_rejects_wrong_class_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = self.make_candidate(Path(directory))
            model = SimpleNamespace(task="segment", names={0: "braille"})
            with patch("verify_tactile_candidate.YOLO", return_value=model):
                with self.assertRaisesRegex(ValueError, "labels must be"):
                    validate_candidate(candidate)

    def test_rejects_missing_required_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = self.make_candidate(Path(directory))
            (candidate / "metrics.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing required file: metrics.json"):
                validate_candidate(candidate)

    def test_rejects_missing_training_config_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = self.make_candidate(Path(directory))
            config_path = candidate / "training_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            del config["seed"]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training config missing keys: seed"):
                validate_candidate(candidate)


if __name__ == "__main__":
    unittest.main()
