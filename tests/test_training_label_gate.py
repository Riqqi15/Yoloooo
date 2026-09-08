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
from ultralytics.data.utils import check_det_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from model_manifest import sha256_file  # noqa: E402
from training_label_gate import (  # noqa: E402
    REVIEW_FIELDS,
    export_yolo,
    mark_review,
    render_qa,
    validate_training_dataset,
    yaml_text,
)


class TrainingLabelGateTest(unittest.TestCase):
    def test_repository_tactile_taxonomy_is_approved_one_class(self) -> None:
        taxonomy = json.loads(
            Path("data/training/taxonomy_tactile_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(taxonomy["status"], "approved")
        self.assertEqual(taxonomy["tactile_segmentation"], ["tactile_paving"])
        self.assertTrue(taxonomy["object_detection"])

    def test_exported_yaml_resolves_from_its_own_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tactile"
            for split in ("train", "val", "test"):
                (dataset / "images" / split).mkdir(parents=True)
            yaml_path = dataset / "data.yaml"
            yaml_path.write_text(yaml_text(["tactile_paving"]), encoding="utf-8")
            with patch("ultralytics.data.utils.check_font"):
                resolved = check_det_dataset(str(yaml_path), autodownload=False)
            self.assertEqual(Path(resolved["path"]), dataset.resolve())

    def make_case(self, root: Path, taxonomy_status: str = "approved") -> dict[str, Path | str]:
        source = root / "station.png"
        pixels = np.zeros((20, 20, 3), dtype=np.uint8)
        pixels[2:10, 2:10] = (40, 180, 230)
        self.assertTrue(cv2.imwrite(str(source), pixels))
        digest = sha256_file(source)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "dataset_version": "station-v1",
                    "dataset_scope": "station_environment",
                    "samples": [
                        {
                            "sample_id": digest,
                            "source_path": str(source),
                            "source_sha256": digest,
                            "split": "train",
                            "split_group": "station-a::session-1",
                            "location_type": "station_platform",
                            "lighting": "normal",
                            "motion": "static",
                            "surface": "matte",
                            "dataset_task": "both",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        taxonomy = root / "taxonomy.json"
        taxonomy.write_text(
            json.dumps(
                {
                    "taxonomy_version": "station-v1",
                    "status": taxonomy_status,
                    "object_detection": ["person"],
                    "tactile_segmentation": ["tactile_guiding_path"],
                }
            ),
            encoding="utf-8",
        )
        annotations = root / "annotations"
        annotations.mkdir()
        annotation = annotations / f"{digest}.json"
        annotation.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sample_id": digest,
                    "source_sha256": digest,
                    "objects": [{"label": "person", "bbox": [2, 2, 10, 10]}],
                    "tactile": [
                        {
                            "label": "tactile_guiding_path",
                            "points": [[1, 1], [5, 1], [5, 5]],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        reviews = root / "reviews.csv"
        with reviews.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
        return {
            "source": source,
            "sample_id": digest,
            "manifest": manifest,
            "taxonomy": taxonomy,
            "annotations": annotations,
            "annotation": annotation,
            "reviews": reviews,
        }

    def validate(self, case: dict[str, Path | str], protected: list[Path] | None = None):
        return validate_training_dataset(
            Path(case["manifest"]),
            Path(case["annotations"]),
            Path(case["reviews"]),
            Path(case["taxonomy"]),
            protected or [],
            require_review=True,
        )

    def review(self, case: dict[str, Path | str]) -> None:
        mark_review(
            Path(case["manifest"]),
            Path(case["annotations"]),
            Path(case["reviews"]),
            [str(case["sample_id"])],
            "human-reviewer",
            "human_reviewed",
            "visual overlay checked",
        )

    def test_unreviewed_annotation_cannot_be_training_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = self.make_case(Path(directory))
            result = self.validate(case)
            self.assertFalse(result["training_ready"])
            self.assertEqual(result["summary"]["unreviewed_samples"], 1)

    def test_review_binds_annotation_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = self.make_case(Path(directory))
            self.review(case)
            self.assertTrue(self.validate(case)["training_ready"])
            annotation = Path(case["annotation"])
            data = json.loads(annotation.read_text(encoding="utf-8"))
            data["objects"][0]["bbox"] = [3, 3, 10, 10]
            annotation.write_text(json.dumps(data), encoding="utf-8")
            result = self.validate(case)
            self.assertFalse(result["training_ready"])
            self.assertTrue(any("review hash mismatch" in item["error"] for item in result["errors"]))

    def test_unknown_label_and_invalid_geometry_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = self.make_case(Path(directory))
            annotation = Path(case["annotation"])
            data = json.loads(annotation.read_text(encoding="utf-8"))
            data["objects"][0] = {"label": "hole", "bbox": [-1, 2, 10, 10]}
            annotation.write_text(json.dumps(data), encoding="utf-8")
            result = self.validate(case)
            self.assertTrue(any("unknown object label" in item["error"] for item in result["errors"]))

    def test_invalid_tactile_polygon_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = self.make_case(Path(directory))
            annotation = Path(case["annotation"])
            data = json.loads(annotation.read_text(encoding="utf-8"))
            data["tactile"][0]["points"] = [[1, 1], [25, 1], [5, 5]]
            annotation.write_text(json.dumps(data), encoding="utf-8")
            result = self.validate(case)
            self.assertTrue(any("polygon outside image" in item["error"] for item in result["errors"]))

    def test_protected_test_content_is_rejected_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = self.make_case(Path(directory))
            self.review(case)
            result = self.validate(case, [Path(case["source"])])
            self.assertFalse(result["training_ready"])
            self.assertTrue(any("protected test content" in item["error"] for item in result["errors"]))

    def test_render_overlay_keeps_source_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.make_case(root)
            output = root / "qa"
            report = render_qa(
                Path(case["manifest"]), Path(case["annotations"]), output,
                Path(case["taxonomy"]), Path(case["reviews"])
            )
            overlay = cv2.imread(report["items"][0]["overlay"])
            self.assertEqual(overlay.shape[:2], (20, 20))
            self.assertFalse(np.array_equal(overlay, cv2.imread(str(case["source"]))))

    def test_export_writes_normalized_yolo_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.make_case(root)
            self.review(case)
            output = root / "export"
            export_yolo(
                Path(case["manifest"]), Path(case["annotations"]),
                Path(case["reviews"]), Path(case["taxonomy"]), [], output
            )
            sample_id = str(case["sample_id"])
            object_label = output / "object" / "labels" / "train" / f"{sample_id}.txt"
            tactile_label = output / "tactile" / "labels" / "train" / f"{sample_id}.txt"
            self.assertEqual(
                object_label.read_text(encoding="utf-8"),
                "0 0.300000 0.300000 0.400000 0.400000\n",
            )
            self.assertEqual(
                tactile_label.read_text(encoding="utf-8"),
                "0 0.050000 0.050000 0.250000 0.050000 0.250000 0.250000\n",
            )

    def test_export_refuses_draft_taxonomy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.make_case(root, taxonomy_status="draft_requires_human_approval")
            self.review(case)
            with self.assertRaisesRegex(ValueError, "dataset is not training-ready"):
                export_yolo(
                    Path(case["manifest"]), Path(case["annotations"]),
                    Path(case["reviews"]), Path(case["taxonomy"]), [], root / "export"
                )
            self.assertFalse((root / "export").exists())

    def test_export_failure_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.make_case(root)
            self.review(case)
            output = root / "export"
            with patch("training_label_gate.shutil.copy2", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    export_yolo(
                        Path(case["manifest"]), Path(case["annotations"]),
                        Path(case["reviews"]), Path(case["taxonomy"]), [], output
                    )
            self.assertFalse(output.exists())
            self.assertFalse(any(root.glob(".export.*")))


if __name__ == "__main__":
    unittest.main()
