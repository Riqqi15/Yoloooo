from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_training_manifest import (  # noqa: E402
    METADATA_FIELDS,
    assigned_split,
    build_manifest,
)


class BuildTrainingManifestTest(unittest.TestCase):
    def write_image(self, path: Path, seed: int) -> None:
        pixels = np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), pixels))

    def write_metadata(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, source: str, session: str = "s1") -> dict[str, str]:
        return {
            "source_path": source,
            "session_id": session,
            "location_id": "station-a-platform-1",
            "location_type": "station_platform",
            "source_name": "local-capture",
            "usage_permission": "consent-record-1",
            "device_model": "test-phone",
            "camera_position": "chest",
            "lighting": "normal",
            "motion": "walking",
            "surface": "matte",
            "dataset_task": "both",
            "notes": "",
        }

    def test_same_location_session_stays_in_same_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "inbox"
            source.mkdir()
            self.write_image(source / "a.png", 1)
            self.write_image(source / "b.png", 2)
            metadata = root / "intake.csv"
            self.write_metadata(metadata, [self.row("a.png"), self.row("b.png")])
            payload = build_manifest(metadata, source, [], "station-v1")
            self.assertEqual(payload["summary"]["total_samples"], 2)
            self.assertEqual(len({row["split"] for row in payload["samples"]}), 1)
            self.assertEqual(
                len({row["split_group"] for row in payload["samples"]}), 1
            )

    def test_rejects_protected_test_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "inbox"
            source.mkdir()
            image = source / "copied-test.png"
            self.write_image(image, 3)
            protected = root / "test.png"
            protected.write_bytes(image.read_bytes())
            metadata = root / "intake.csv"
            self.write_metadata(metadata, [self.row(image.name)])
            with self.assertRaisesRegex(ValueError, "protected test content"):
                build_manifest(metadata, source, [protected], "station-v1")

    def test_rejects_nonstation_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "inbox"
            source.mkdir()
            self.write_image(source / "bus.png", 4)
            row = self.row("bus.png")
            row["location_type"] = "bus_only"
            metadata = root / "intake.csv"
            self.write_metadata(metadata, [row])
            with self.assertRaisesRegex(ValueError, "unapproved station context"):
                build_manifest(metadata, source, [], "station-v1")

    def test_rejects_exact_duplicates_inside_intake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "inbox"
            source.mkdir()
            self.write_image(source / "a.png", 5)
            (source / "b.png").write_bytes((source / "a.png").read_bytes())
            metadata = root / "intake.csv"
            self.write_metadata(metadata, [self.row("a.png"), self.row("b.png")])
            with self.assertRaisesRegex(ValueError, "exact duplicate content"):
                build_manifest(metadata, source, [], "station-v1")

    def test_rejects_near_duplicate_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "inbox"
            source.mkdir()
            pixels = np.random.default_rng(6).integers(
                0, 256, (32, 32, 3), dtype=np.uint8
            )
            self.assertTrue(cv2.imwrite(str(source / "a.png"), pixels))
            pixels[0, 0, 0] ^= 1
            self.assertTrue(cv2.imwrite(str(source / "b.png"), pixels))
            first_session = "s0"
            first_split = assigned_split("station-a-platform-1", first_session, 42)
            second_session = next(
                f"s{index}"
                for index in range(1, 1000)
                if assigned_split("station-a-platform-1", f"s{index}", 42)
                != first_split
            )
            metadata = root / "intake.csv"
            self.write_metadata(
                metadata,
                [self.row("a.png", first_session), self.row("b.png", second_session)],
            )
            with self.assertRaisesRegex(ValueError, "near-duplicate crosses splits"):
                build_manifest(metadata, source, [], "station-v1")


if __name__ == "__main__":
    unittest.main()
