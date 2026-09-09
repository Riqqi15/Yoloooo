from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from model_manifest import sha256_file  # noqa: E402
from prepare_guidetwsi_subset import (  # noqa: E402
    parse_yolo_polygons,
    prepare_subset,
)


class PrepareGuideTWSISubsetTest(unittest.TestCase):
    def write_image(self, path: Path, seed: int = 1) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), pixels))

    def test_parses_valid_one_class_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "a.txt"
            label.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")
            self.assertEqual(len(parse_yolo_polygons(label)), 1)

    def test_rejects_wrong_class_and_out_of_range_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "a.txt"
            label.write_text("1 0.1 0.1 0.9 0.1 0.9 0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class 0"):
                parse_yolo_polygons(label)
            label.write_text("0 0.1 0.1 1.1 0.1 0.9 0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "normalized"):
                parse_yolo_polygons(label)

    def test_prepares_verified_pair_and_rejects_protected_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            image = cache / "images" / "a.jpg"
            label = cache / "labels" / "a.txt"
            self.write_image(image)
            label.parent.mkdir(parents=True)
            label.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9\n", encoding="utf-8")
            provenance = root / "provenance.json"
            provenance.write_text(
                json.dumps(
                    {
                        "dataset_handle": "guidedogrobot/guidetwsi",
                        "dataset_version": 1,
                        "license": "CC0: Public Domain",
                        "files": [
                            {
                                "pair_key": "train/a",
                                "role": "image",
                                "upstream_path": "RBar/images/train/a.jpg",
                                "bytes": image.stat().st_size,
                                "sha256": sha256_file(image),
                                "local_cache_key": "images/a.jpg",
                            },
                            {
                                "pair_key": "train/a",
                                "role": "label",
                                "upstream_path": "RBar/labels/train/a.txt",
                                "bytes": label.stat().st_size,
                                "sha256": sha256_file(label),
                                "local_cache_key": "labels/a.txt",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = prepare_subset(provenance, cache, [], [], limit=10)
            self.assertEqual(manifest["summary"]["retained_samples"], 1)
            self.assertEqual(manifest["samples"][0]["polygon_count"], 1)
            rejected = prepare_subset(provenance, cache, [image], [], limit=10)
            self.assertEqual(rejected["summary"]["retained_samples"], 0)
            self.assertEqual(rejected["summary"]["rejected"]["protected_exact"], 1)

    def test_aborts_on_provenance_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "cache" / "a.jpg"
            label = root / "cache" / "a.txt"
            self.write_image(image)
            label.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9\n", encoding="utf-8")
            provenance = root / "provenance.json"
            provenance.write_text(
                json.dumps(
                    {
                        "dataset_handle": "guidedogrobot/guidetwsi",
                        "dataset_version": 1,
                        "license": "CC0: Public Domain",
                        "files": [
                            {
                                "pair_key": "train/a",
                                "role": "image",
                                "bytes": image.stat().st_size,
                                "sha256": "0" * 64,
                                "local_cache_key": "a.jpg",
                            },
                            {
                                "pair_key": "train/a",
                                "role": "label",
                                "bytes": label.stat().st_size,
                                "sha256": sha256_file(label),
                                "local_cache_key": "a.txt",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare_subset(provenance, root / "cache", [], [], limit=10)


if __name__ == "__main__":
    unittest.main()
