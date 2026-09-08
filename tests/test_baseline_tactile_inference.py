from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_tactile_inference import (  # noqa: E402
    EXPECTED_LABELS,
    MODEL_URL,
    ensure_model,
    mask_summary,
    normalize_labels,
    rasterize_masks,
    validate_model_contract,
)


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class TactileInferenceTest(unittest.TestCase):
    def test_ensure_model_downloads_official_github_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yolo11n_tactile.pt"

            def fake_download(url: str, target: Path) -> None:
                self.assertEqual(url, MODEL_URL)
                Path(target).write_bytes(b"weight")

            with patch("baseline_tactile_inference.urlretrieve", side_effect=fake_download):
                self.assertEqual(ensure_model(path), path)
            self.assertEqual(path.read_bytes(), b"weight")

    def test_normalize_labels_accepts_dict_or_list(self) -> None:
        self.assertEqual(normalize_labels({0: "braille"}), EXPECTED_LABELS)
        self.assertEqual(normalize_labels(["braille"]), EXPECTED_LABELS)

    def test_contract_rejects_wrong_task(self) -> None:
        model = SimpleNamespace(task="detect", names=EXPECTED_LABELS)
        with self.assertRaisesRegex(ValueError, "expected segment task"):
            validate_model_contract(model)

    def test_contract_rejects_wrong_labels(self) -> None:
        model = SimpleNamespace(task="segment", names={0: "guiding", 1: "warning"})
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            validate_model_contract(model)

    def test_mask_summary_handles_no_mask(self) -> None:
        self.assertEqual(mask_summary(SimpleNamespace(masks=None)), (0, 0.0))

    def test_mask_summary_unions_instances(self) -> None:
        values = np.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        result = SimpleNamespace(masks=SimpleNamespace(data=FakeTensor(values)))
        instances, coverage = mask_summary(result)
        self.assertEqual(instances, 2)
        self.assertEqual(coverage, 0.75)

    def test_rasterize_masks_handles_no_masks(self) -> None:
        mask = rasterize_masks(SimpleNamespace(masks=None), height=4, width=6)
        self.assertEqual(mask.shape, (4, 6))
        self.assertEqual(set(np.unique(mask).tolist()), {0})

    def test_rasterize_masks_uses_source_coordinates(self) -> None:
        polygon = np.array(
            [[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]],
            dtype=np.float32,
        )
        result = SimpleNamespace(masks=SimpleNamespace(xy=[polygon]))
        mask = rasterize_masks(result, height=5, width=6)
        self.assertEqual(mask.shape, (5, 6))
        self.assertEqual(set(np.unique(mask).tolist()), {0, 255})
        self.assertEqual(mask[2, 2], 255)
        self.assertEqual(mask[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
