from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from acquire_guidetwsi_subset import (  # noqa: E402
    build_pair_inventory,
    download_with_retries,
    materialize_selection,
    select_pairs,
    validate_source,
)


class AcquireGuideTWSISubsetTest(unittest.TestCase):
    def test_accepts_expected_public_source(self) -> None:
        validate_source("guidedogrobot/guidetwsi", "CC0: Public Domain")

    def test_rejects_wrong_source_or_unknown_license(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset handle"):
            validate_source("someone/other", "CC0: Public Domain")
        with self.assertRaisesRegex(ValueError, "license"):
            validate_source("guidedogrobot/guidetwsi", "unknown")

    def test_pairs_only_rbar_images_and_labels(self) -> None:
        files = [
            {"path": "RBar-22K/images/train/a.jpg", "size": 100},
            {"path": "RBar-22K/labels/train/a.txt", "size": 20},
            {"path": "RBar-22K/images/train/unpaired.jpg", "size": 90},
            {"path": "RDot-13K/images/train/b.jpg", "size": 80},
            {"path": "../RBar-22K/labels/train/a.txt", "size": 20},
        ]
        pairs, rejected = build_pair_inventory(files)
        self.assertEqual([pair["key"] for pair in pairs], ["train/a"])
        self.assertEqual(pairs[0]["total_bytes"], 120)
        self.assertEqual(rejected["unpaired"], 1)
        self.assertEqual(rejected["outside_rbar"], 1)
        self.assertEqual(rejected["unsafe_path"], 1)

    def test_rejects_duplicate_inventory_path(self) -> None:
        files = [
            {"path": "RBar/images/train/a.jpg", "size": 1},
            {"path": "RBar/images/train/a.jpg", "size": 1},
            {"path": "RBar/labels/train/a.txt", "size": 1},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate inventory path"):
            build_pair_inventory(files)

    def test_selection_is_stable_and_never_exceeds_limits(self) -> None:
        pairs = [
            {
                "key": f"train/{letter}",
                "image_path": f"RBar/images/train/{letter}.jpg",
                "label_path": f"RBar/labels/train/{letter}.txt",
                "total_bytes": size,
            }
            for letter, size in (("a", 60), ("b", 60), ("c", 40))
        ]
        first = select_pairs(pairs, max_bytes=100, max_images=2, seed=42)
        second = select_pairs(list(reversed(pairs)), max_bytes=100, max_images=2, seed=42)
        self.assertEqual(first, second)
        self.assertLessEqual(sum(pair["total_bytes"] for pair in first), 100)
        self.assertLessEqual(len(first), 2)

    def test_materializes_verified_files_and_hashes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = {
                "key": "train/a",
                "image_path": "GuideTWSI/RBar/images/train/a.jpg",
                "image_bytes": 3,
                "label_path": "GuideTWSI/RBar/labels/train/a.txt",
                "label_bytes": 2,
                "total_bytes": 5,
            }

            def fake_download(remote_path: str, staging: Path) -> Path:
                output = staging / Path(*PurePath(remote_path).parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"img" if remote_path.endswith(".jpg") else b"0\n")
                return output

            payload = materialize_selection(
                [pair], root / "cache", fake_download, workers=1
            )
            self.assertEqual(payload["summary"]["downloaded_files"], 2)
            self.assertEqual(payload["summary"]["downloaded_bytes"], 5)
            self.assertEqual(len(payload["files"][0]["sha256"]), 64)

    def test_retries_transient_download_failure(self) -> None:
        calls = 0

        def flaky_download(remote_path: str, staging: Path) -> Path:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TimeoutError(remote_path)
            return staging / "done"

        result = download_with_retries(
            flaky_download, "RBar/images/train/a.jpg", Path("stage"), delays=(0, 0)
        )
        self.assertEqual(result, Path("stage/done"))
        self.assertEqual(calls, 3)


def PurePath(value: str) -> Path:
    return Path(*value.split("/"))


if __name__ == "__main__":
    unittest.main()
