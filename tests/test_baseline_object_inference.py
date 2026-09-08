from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_object_inference import (  # noqa: E402
    STOP_OBSTACLE,
    STOP_UNCERTAIN,
    box_in_corridor,
    corridor_polygon,
    frame_status,
    percentile,
)


class BaselineObjectInferenceTest(unittest.TestCase):
    def test_corridor_includes_center_and_excludes_sides(self) -> None:
        polygon = corridor_polygon(1000, 1000)
        self.assertTrue(box_in_corridor((450, 600, 550, 900), polygon))
        self.assertFalse(box_in_corridor((0, 600, 100, 900), polygon))
        self.assertFalse(box_in_corridor((900, 600, 999, 900), polygon))

    def test_frame_status_is_always_stop(self) -> None:
        self.assertEqual(frame_status(1), STOP_OBSTACLE)
        self.assertEqual(frame_status(0), STOP_UNCERTAIN)

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)


if __name__ == "__main__":
    unittest.main()
