from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_tactile_candidate import binary_iou, summarize_presence  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
