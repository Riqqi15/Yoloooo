from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_reviewed_training_sets import check_no_cross_split_leakage  # noqa: E402


class MergeReviewedTrainingSetsTest(unittest.TestCase):
    def test_near_duplicate_cannot_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            for name, value, split in (
                ("dark.png", 30, "train"),
                ("light.png", 220, "validation"),
            ):
                path = root / name
                self.assertTrue(cv2.imwrite(str(path), np.full((20, 20, 3), value, np.uint8)))
                samples.append({"sample_id": name, "source_path": str(path), "split": split})

            with self.assertRaisesRegex(ValueError, "near-duplicate crosses splits"):
                check_no_cross_split_leakage(samples, threshold=5)

    def test_near_duplicate_in_same_split_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            for name, value in (("dark.png", 30), ("light.png", 220)):
                path = root / name
                self.assertTrue(cv2.imwrite(str(path), np.full((20, 20, 3), value, np.uint8)))
                samples.append({"sample_id": name, "source_path": str(path), "split": "train"})

            check_no_cross_split_leakage(samples, threshold=5)

            self.assertEqual(samples[1]["near_duplicate_of"], "dark.png")
            self.assertEqual(samples[1]["near_duplicate_distance"], 0)


if __name__ == "__main__":
    unittest.main()
