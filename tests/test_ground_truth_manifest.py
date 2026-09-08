from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_tactile_inference import sha256_file  # noqa: E402
from ground_truth_manifest import (  # noqa: E402
    build_rows,
    mark_ambiguous,
    mark_human_reviewed,
    read_manifest,
    validate_rows,
    write_manifest,
)


class GroundTruthManifestTest(unittest.TestCase):
    def write_source(self, path: Path) -> None:
        self.assertTrue(cv2.imwrite(str(path), np.zeros((4, 6, 3), dtype=np.uint8)))

    def report_for(self, source: Path) -> dict:
        return {
            "dataset_role": "test_only",
            "images": [
                {
                    "source": str(source),
                    "tactile_instance_count": 1,
                    "tactile_mask_coverage": 0.25,
                }
            ],
        }

    def test_build_rows_keeps_predictions_separate_from_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "station.webp"
            self.write_source(source)
            row = build_rows(self.report_for(source), root / "manifest.csv")[0]
            self.assertEqual(row["prediction_tactile_present"], "1")
            self.assertEqual(row["ground_truth_tactile_present"], "")
            self.assertEqual(row["review_status"], "unreviewed")
            self.assertEqual(row["source_sha256"], sha256_file(source))
            self.assertEqual(Path(row["ground_truth_mask_path"]).name, "station_webp.png")

    def test_write_manifest_refuses_to_overwrite_review_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text("reviewed data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_manifest(path, [])
            self.assertEqual(path.read_text(encoding="utf-8"), "reviewed data")

    def test_unreviewed_row_is_incomplete_not_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.jpg"
            self.write_source(source)
            rows = build_rows(self.report_for(source), root / "manifest.csv")
            summary = validate_rows(rows)
            self.assertEqual(summary["incomplete_rows"], 1)
            self.assertEqual(summary["invalid_rows"], 0)

    def test_valid_positive_binary_mask_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.jpg"
            self.write_source(source)
            rows = build_rows(self.report_for(source), root / "manifest.csv")
            mask_path = Path(rows[0]["ground_truth_mask_path"])
            mask_path.parent.mkdir(parents=True)
            mask = np.zeros((4, 6), dtype=np.uint8)
            mask[1:3, 2:4] = 255
            self.assertTrue(cv2.imwrite(str(mask_path), mask))
            rows[0]["ground_truth_tactile_present"] = "1"
            rows[0]["ground_truth_mask_sha256"] = sha256_file(mask_path)
            rows[0]["annotation_origin"] = "human"
            rows[0]["annotation_id"] = "labelme-sha256"
            rows[0]["review_status"] = "human_reviewed"
            rows[0]["reviewer"] = "reviewer@example"
            rows[0]["reviewed_at"] = "2026-09-05T10:00:00+07:00"
            summary = validate_rows(rows)
            self.assertEqual(summary["valid_reviewed_rows"], 1)
            self.assertEqual(summary["invalid_rows"], 0)
            self.assertEqual(summary["positive_rows"], 1)

    def test_positive_mask_with_wrong_dimensions_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.jpg"
            self.write_source(source)
            rows = build_rows(self.report_for(source), root / "manifest.csv")
            mask_path = Path(rows[0]["ground_truth_mask_path"])
            mask_path.parent.mkdir(parents=True)
            self.assertTrue(
                cv2.imwrite(str(mask_path), np.full((2, 3), 255, dtype=np.uint8))
            )
            rows[0]["ground_truth_tactile_present"] = "1"
            rows[0]["ground_truth_mask_sha256"] = sha256_file(mask_path)
            rows[0]["annotation_origin"] = "ai_assisted"
            rows[0]["annotation_id"] = "labelme-sha256"
            rows[0]["review_status"] = "provisional"
            summary = validate_rows(rows)
            self.assertEqual(summary["invalid_rows"], 1)
            self.assertTrue(any("dimensions" in error for error in summary["errors"]))

    def test_provisional_annotation_requires_explicit_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.jpg"
            self.write_source(source)
            manifest = root / "manifest.csv"
            rows = build_rows(self.report_for(source), manifest)
            mask_path = Path(rows[0]["ground_truth_mask_path"])
            mask_path.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(mask_path), np.zeros((4, 6), dtype=np.uint8)))
            rows[0].update(
                {
                    "ground_truth_tactile_present": "0",
                    "ground_truth_mask_sha256": sha256_file(mask_path),
                    "annotation_origin": "ai_assisted",
                    "annotation_id": "labelme-sha256",
                    "review_status": "provisional",
                }
            )
            write_manifest(manifest, rows)
            before = validate_rows(read_manifest(manifest))
            self.assertFalse(before["official_ready"])
            self.assertEqual(before["provisional_rows"], 1)
            mark_human_reviewed(manifest, [source.name], "human-reviewer")
            after = validate_rows(read_manifest(manifest))
            self.assertTrue(after["official_ready"])
            self.assertEqual(after["human_reviewed_rows"], 1)

    def test_ambiguous_annotation_remains_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.jpg"
            self.write_source(source)
            manifest = root / "manifest.csv"
            rows = build_rows(self.report_for(source), manifest)
            mask_path = Path(rows[0]["ground_truth_mask_path"])
            mask_path.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(mask_path), np.zeros((4, 6), dtype=np.uint8)))
            rows[0].update(
                {
                    "ground_truth_tactile_present": "0",
                    "ground_truth_mask_sha256": sha256_file(mask_path),
                    "annotation_origin": "ai_assisted",
                    "annotation_id": "labelme-sha256",
                    "review_status": "provisional",
                }
            )
            write_manifest(manifest, rows)
            mark_ambiguous(manifest, [source.name], "surface is occluded")
            summary = validate_rows(read_manifest(manifest))
            self.assertEqual(summary["ambiguous_rows"], 1)
            self.assertEqual(summary["incomplete_rows"], 1)
            self.assertFalse(summary["official_ready"])


if __name__ == "__main__":
    unittest.main()
