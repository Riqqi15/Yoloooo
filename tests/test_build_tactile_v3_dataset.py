from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_tactile_v3_dataset import build_dataset  # noqa: E402
from model_manifest import sha256_file  # noqa: E402


class BuildTactileV3DatasetTest(unittest.TestCase):
    def write_pair(self, image: Path, label: Path, content: bytes) -> None:
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(content)
        label.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9\n", encoding="utf-8")

    def test_builds_exact_ratio_and_keeps_validation_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            public_samples = []
            for index in range(10):
                image = cache / "images" / f"p{index}.jpg"
                label = cache / "labels" / f"p{index}.txt"
                self.write_pair(image, label, f"public-{index}".encode())
                public_samples.append(
                    {
                        "sample_id": sha256_file(image),
                        "image_cache_key": f"images/p{index}.jpg",
                        "image_sha256": sha256_file(image),
                        "label_cache_key": f"labels/p{index}.txt",
                        "label_sha256": sha256_file(label),
                    }
                )
            public_manifest = root / "public.json"
            public_manifest.write_text(
                json.dumps({"dataset_version": "public-v1", "samples": public_samples}),
                encoding="utf-8",
            )
            station = root / "station"
            for index in range(2):
                self.write_pair(
                    station / "images" / "train" / f"s{index}.jpg",
                    station / "labels" / "train" / f"s{index}.txt",
                    f"station-{index}".encode(),
                )
            self.write_pair(
                station / "images" / "val" / "v.jpg",
                station / "labels" / "val" / "v.txt",
                b"validation",
            )

            report = build_dataset(
                public_manifest,
                cache,
                station,
                root / "output",
                public_to_station=4,
                public_validation_count=2,
            )
            self.assertEqual(report["summary"]["train_public"], 8)
            self.assertEqual(report["summary"]["train_station"], 2)
            self.assertEqual(report["summary"]["validation_public"], 2)
            self.assertEqual(report["summary"]["validation_station"], 1)
            self.assertFalse((root / "output" / "images" / "test").exists())
            self.assertTrue((root / "output" / "dataset.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
