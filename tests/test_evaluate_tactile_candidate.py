from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_tactile_candidate import (  # noqa: E402
    binary_iou,
    rasterize_polygons,
    select_best_threshold,
    summarize_presence,
    validate_evaluation_manifest,
)


class EvaluateTactileCandidateTest(unittest.TestCase):
    def test_binary_iou_for_partial_overlap(self) -> None:
        ground_truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        self.assertAlmostEqual(binary_iou(ground_truth, prediction), 1 / 3)

    def test_binary_iou_for_two_empty_masks(self) -> None:
        empty = np.zeros((2, 2), dtype=np.uint8)
        self.assertEqual(binary_iou(empty, empty), 1.0)

    def test_presence_precision_recall_and_false_positive_rate(self) -> None:
        summary = summarize_presence([(True, True), (True, False), (False, True), (False, False)])
        self.assertEqual(summary["true_positive"], 1)
        self.assertEqual(summary["false_negative"], 1)
        self.assertEqual(summary["false_positive"], 1)
        self.assertEqual(summary["true_negative"], 1)
        self.assertEqual(summary["precision"], 0.5)
        self.assertEqual(summary["recall"], 0.5)
        self.assertEqual(summary["false_positive_rate"], 0.5)

    def test_rasterize_polygons_filters_by_confidence(self) -> None:
        polygons = [
            np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
            np.array([[3, 3], [5, 3], [5, 5], [3, 5]]),
        ]
        mask = rasterize_polygons(polygons, [0.9, 0.2], 0.5, (6, 6))
        self.assertEqual(mask[1, 1], 255)
        self.assertEqual(mask[4, 4], 0)

    def test_threshold_selection_prioritizes_recall_then_iou(self) -> None:
        reports = [
            {"confidence": 0.25, "recall": 0.8, "mean_iou": 0.7, "false_positive_rate": 0.0},
            {"confidence": 0.10, "recall": 1.0, "mean_iou": 0.5, "false_positive_rate": 0.5},
            {"confidence": 0.05, "recall": 1.0, "mean_iou": 0.6, "false_positive_rate": 0.5},
        ]
        self.assertEqual(select_best_threshold(reports)["confidence"], 0.05)

    def test_v3_candidate_can_use_declared_station_validation_manifest(self) -> None:
        validate_evaluation_manifest(
            {
                "dataset_version": "tactile-v3-public4-station1",
                "validation_manifest_version": "station-tactile-v2",
            },
            {"dataset_version": "station-tactile-v2"},
        )
        with self.assertRaisesRegex(ValueError, "differ"):
            validate_evaluation_manifest(
                {"dataset_version": "tactile-v3-public4-station1"},
                {"dataset_version": "station-tactile-v2"},
            )


if __name__ == "__main__":
    unittest.main()
