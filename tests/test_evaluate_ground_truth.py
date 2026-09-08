from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_ground_truth import (  # noqa: E402
    binary_iou,
    evaluate,
    presence_metrics,
    require_complete_manifest,
)
from baseline_tactile_inference import sha256_file  # noqa: E402
from ground_truth_manifest import build_rows, write_manifest  # noqa: E402
from baseline_object_inference import STOP_UNCERTAIN  # noqa: E402


class EvaluateGroundTruthTest(unittest.TestCase):
    def evaluation_case(self, root: Path, status: str = "provisional") -> tuple[Path, Path]:
        source = root / "station.jpg"
        self.assertTrue(cv2.imwrite(str(source), np.zeros((4, 6, 3), dtype=np.uint8)))
        manifest = root / "manifest.csv"
        rows = build_rows(
            {
                "dataset_role": "test_only",
                "images": [{"source": str(source), "tactile_instance_count": 0}],
            },
            manifest,
        )
        ground_truth_mask = root / "ground-truth.png"
        prediction_mask = root / "prediction.png"
        self.assertTrue(cv2.imwrite(str(ground_truth_mask), np.zeros((4, 6), dtype=np.uint8)))
        self.assertTrue(cv2.imwrite(str(prediction_mask), np.zeros((4, 6), dtype=np.uint8)))
        rows[0].update(
            {
                "ground_truth_tactile_present": "0",
                "ground_truth_mask_path": str(ground_truth_mask),
                "ground_truth_mask_sha256": sha256_file(ground_truth_mask),
                "annotation_origin": "ai_assisted",
                "annotation_id": "annotation-sha256",
                "review_status": status,
                "reviewer": "human" if status == "human_reviewed" else "",
                "reviewed_at": "2026-09-05T10:00:00+07:00" if status == "human_reviewed" else "",
            }
        )
        write_manifest(manifest, rows)
        scope = root / "scope.json"
        scope.write_text("{}", encoding="utf-8")
        object_model = root / "object.pt"
        tactile_model = root / "tactile.pt"
        object_model.write_bytes(b"object")
        tactile_model.write_bytes(b"tactile")
        model_manifest = root / "model-manifest.json"
        model_manifest.write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "models": {
                        "object": {
                            "artifact_path": object_model.name,
                            "sha256": sha256_file(object_model),
                            "size_bytes": object_model.stat().st_size,
                        },
                        "tactile": {
                            "artifact_path": tactile_model.name,
                            "sha256": sha256_file(tactile_model),
                            "size_bytes": tactile_model.stat().st_size,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        report = {
            "dataset_role": "test_only",
            "dataset_scope": "station_environment",
            "scope_file": str(scope),
            "scope_sha256": sha256_file(scope),
            "model_manifest": str(model_manifest),
            "model_manifest_sha256": sha256_file(model_manifest),
            "models": {
                "object": {"path": str(object_model), "sha256": sha256_file(object_model)},
                "tactile": {"path": str(tactile_model), "sha256": sha256_file(tactile_model)},
            },
            "images": [
                {
                    "source": str(source),
                    "source_sha256": sha256_file(source),
                    "combined_status": STOP_UNCERTAIN,
                    "tactile_status": STOP_UNCERTAIN,
                    "tactile_instance_count": 0,
                    "tactile_prediction_mask": str(prediction_mask),
                    "tactile_prediction_mask_sha256": sha256_file(prediction_mask),
                }
            ],
        }
        predictions = root / "report.json"
        predictions.write_text(json.dumps(report), encoding="utf-8")
        return manifest, predictions

    def test_presence_metrics_counts_all_outcomes(self) -> None:
        metrics = presence_metrics([(1, 1), (0, 1), (1, 0), (0, 0)])
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_binary_iou(self) -> None:
        ground_truth = np.array([[255, 255], [0, 0]], dtype=np.uint8)
        prediction = np.array([[255, 0], [255, 0]], dtype=np.uint8)
        self.assertAlmostEqual(binary_iou(ground_truth, prediction), 1 / 3)

    def test_incomplete_manifest_is_rejected(self) -> None:
        summary = {"invalid_rows": 0, "incomplete_rows": 1}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            require_complete_manifest(summary)

    def test_provisional_manifest_requires_explicit_diagnostic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, predictions = self.evaluation_case(Path(directory))
            with self.assertRaisesRegex(ValueError, "provisional annotations"):
                evaluate(manifest, predictions)
            payload = evaluate(manifest, predictions, allow_provisional=True)
            self.assertEqual(payload["evaluation_status"], "provisional")

    def test_failed_prediction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, predictions = self.evaluation_case(root, "human_reviewed")
            report = json.loads(predictions.read_text(encoding="utf-8"))
            report["images"][0]["combined_status"] = "STOP_FAILURE"
            predictions.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction failed"):
                evaluate(manifest, predictions)


if __name__ == "__main__":
    unittest.main()
