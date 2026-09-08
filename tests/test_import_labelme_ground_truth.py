from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ground_truth_manifest import build_rows, write_manifest  # noqa: E402
from import_labelme_ground_truth import import_annotations  # noqa: E402


class ImportLabelmeGroundTruthTest(unittest.TestCase):
    def setup_case(self, root: Path) -> tuple[Path, Path, Path]:
        source = root / "sample.jpg"
        self.assertTrue(cv2.imwrite(str(source), np.zeros((6, 8, 3), dtype=np.uint8)))
        manifest = root / "manifest.csv"
        rows = build_rows(
            {
                "dataset_role": "test_only",
                "images": [
                    {
                        "source": str(source),
                        "tactile_instance_count": 0,
                        "tactile_mask_coverage": 0.0,
                    }
                ],
            },
            manifest,
        )
        write_manifest(manifest, rows)
        annotations = root / "labelme"
        annotations.mkdir()
        return source, manifest, annotations

    def write_annotation(self, path: Path, source: Path, shapes: list[dict]) -> None:
        path.write_text(
            json.dumps(
                {
                    "imagePath": source.name,
                    "imageHeight": 6,
                    "imageWidth": 8,
                    "shapes": shapes,
                }
            ),
            encoding="utf-8",
        )

    def read_row(self, manifest: Path) -> dict[str, str]:
        with manifest.open(encoding="utf-8", newline="") as handle:
            return next(csv.DictReader(handle))

    def test_imports_positive_polygon_as_binary_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest, annotations = self.setup_case(root)
            self.write_annotation(
                annotations / "sample.json",
                source,
                [
                    {
                        "label": "tactile_paving",
                        "shape_type": "polygon",
                        "points": [[1, 1], [6, 1], [6, 4], [1, 4]],
                    }
                ],
            )
            summary = import_annotations(annotations, manifest, batch_id="test-batch")
            row = self.read_row(manifest)
            mask = cv2.imread(row["ground_truth_mask_path"], cv2.IMREAD_GRAYSCALE)
            self.assertEqual(summary["imported_rows"], 1)
            self.assertEqual(row["ground_truth_tactile_present"], "1")
            self.assertEqual(row["review_status"], "provisional")
            self.assertEqual(row["annotation_origin"], "ai_assisted")
            self.assertEqual(row["reviewer"], "")
            self.assertIn("masks\\batches\\test-batch", row["ground_truth_mask_path"])
            self.assertEqual(mask.shape, (6, 8))
            self.assertEqual(set(np.unique(mask).tolist()), {0, 255})

    def test_imports_empty_shapes_as_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest, annotations = self.setup_case(root)
            self.write_annotation(annotations / "sample.json", source, [])
            import_annotations(annotations, manifest, batch_id="negative-batch")
            row = self.read_row(manifest)
            mask = cv2.imread(row["ground_truth_mask_path"], cv2.IMREAD_GRAYSCALE)
            self.assertEqual(row["ground_truth_tactile_present"], "0")
            self.assertFalse(np.any(mask))

    def test_accepts_labelme_polygon_on_image_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest, annotations = self.setup_case(root)
            self.write_annotation(
                annotations / "sample.json",
                source,
                [
                    {
                        "label": "tactile_paving",
                        "shape_type": "polygon",
                        "points": [[6, 4], [8, 4], [8, 6], [6, 6]],
                    }
                ],
            )
            import_annotations(annotations, manifest, batch_id="boundary-batch")
            row = self.read_row(manifest)
            mask = cv2.imread(row["ground_truth_mask_path"], cv2.IMREAD_GRAYSCALE)
            self.assertEqual(mask[5, 7], 255)

    def test_unknown_label_rejects_without_manifest_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest, annotations = self.setup_case(root)
            before = manifest.read_bytes()
            self.write_annotation(
                annotations / "sample.json",
                source,
                [
                    {
                        "label": "braille",
                        "shape_type": "polygon",
                        "points": [[1, 1], [6, 1], [6, 4]],
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "unknown label"):
                import_annotations(annotations, manifest)
            self.assertEqual(manifest.read_bytes(), before)
            self.assertFalse(Path(self.read_row(manifest)["ground_truth_mask_path"]).exists())

    def test_manifest_stays_unchanged_when_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest, annotations = self.setup_case(root)
            self.write_annotation(annotations / "sample.json", source, [])
            before = manifest.read_bytes()
            with patch(
                "import_labelme_ground_truth.replace_manifest",
                side_effect=OSError("commit failed"),
            ):
                with self.assertRaisesRegex(OSError, "commit failed"):
                    import_annotations(annotations, manifest, batch_id="failed-batch")
            self.assertEqual(manifest.read_bytes(), before)
            self.assertFalse(
                (root / "masks" / "batches" / "failed-batch").exists()
            )


if __name__ == "__main__":
    unittest.main()
