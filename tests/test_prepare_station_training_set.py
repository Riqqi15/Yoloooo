from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from model_manifest import sha256_file  # noqa: E402
from prepare_station_training_set import prepare_dataset  # noqa: E402


class PrepareStationTrainingSetTest(unittest.TestCase):
    def test_converts_labelme_and_keeps_missing_json_unreviewed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for name, value in (("001_a.jpg", 60), ("002_b.jpg", 120)):
                image = np.full((20, 30, 3), value, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(source / name), image))
            (source / "sources.json").write_text(
                json.dumps(
                    [
                        {"set_index": 1, "filename": "001_a.jpg", "source": "https://example/a", "license": "CC0", "artist": "A"},
                        {"set_index": 2, "filename": "002_b.jpg", "source": "https://example/b", "license": "CC0", "artist": "B"},
                    ]
                ),
                encoding="utf-8",
            )
            (source / "001_a.json").write_text(
                json.dumps(
                    {
                        "imagePath": "001_a.jpg",
                        "imageWidth": 30,
                        "imageHeight": 20,
                        "shapes": [
                            {
                                "label": "tactile_paving",
                                "shape_type": "polygon",
                                "points": [[1, 1], [10, 1], [10, 8]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            metadata = root / "intake.csv"
            annotations = root / "annotations"
            report = prepare_dataset(
                source,
                metadata,
                annotations,
                groups={
                    "001_a.jpg": ("station_a", "session_a"),
                    "002_b.jpg": ("station_b", "session_b"),
                },
            )

            self.assertEqual(report["samples"], 2)
            self.assertEqual(report["positive_labelme"], 1)
            self.assertEqual(report["empty_unreviewed"], 1)
            with metadata.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["source_path"] for row in rows], ["001_a.jpg", "002_b.jpg"])
            positive = json.loads(
                (annotations / f"{sha256_file(source / '001_a.jpg')}.json").read_text(encoding="utf-8")
            )
            negative = json.loads(
                (annotations / f"{sha256_file(source / '002_b.jpg')}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(positive["tactile"]), 1)
            self.assertEqual(negative["tactile"], [])


if __name__ == "__main__":
    unittest.main()
