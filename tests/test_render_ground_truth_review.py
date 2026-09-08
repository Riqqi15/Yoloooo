from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_tactile_inference import sha256_file  # noqa: E402
from ground_truth_manifest import build_rows, write_manifest  # noqa: E402
from render_ground_truth_review import render_review  # noqa: E402


class RenderGroundTruthReviewTest(unittest.TestCase):
    def test_overlay_preserves_dimensions_and_marks_mask_red(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "station.jpg"
            self.assertTrue(cv2.imwrite(str(source), np.zeros((60, 30, 3), dtype=np.uint8)))
            manifest = root / "manifest.csv"
            rows = build_rows(
                {
                    "dataset_role": "test_only",
                    "images": [{"source": str(source), "tactile_instance_count": 1}],
                },
                manifest,
            )
            mask = root / "mask.png"
            values = np.zeros((60, 30), dtype=np.uint8)
            values[40:50, 10:20] = 255
            self.assertTrue(cv2.imwrite(str(mask), values))
            rows[0].update(
                {
                    "ground_truth_tactile_present": "1",
                    "ground_truth_mask_path": str(mask),
                    "ground_truth_mask_sha256": sha256_file(mask),
                    "annotation_origin": "ai_assisted",
                    "annotation_id": "annotation-sha256",
                    "review_status": "provisional",
                }
            )
            write_manifest(manifest, rows)
            output_dir = root / "review"
            payload = render_review(manifest, output_dir)
            overlay = cv2.imread(payload["items"][0]["overlay"])
            self.assertEqual(overlay.shape, (60, 30, 3))
            self.assertGreater(int(overlay[45, 15, 2]), int(overlay[45, 15, 0]))
            self.assertTrue((output_dir / "index.json").is_file())


if __name__ == "__main__":
    unittest.main()
