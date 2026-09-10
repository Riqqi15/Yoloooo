from __future__ import annotations

import json
import unittest
from pathlib import Path


NOTEBOOK = Path("notebooks/train_tactile_one_class_colab.ipynb")
V3_NOTEBOOK = Path("notebooks/train_tactile_v3_public_colab.ipynb")


class TactileTrainingNotebookTest(unittest.TestCase):
    def test_notebook_contains_reproducible_one_class_contract(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        for required in (
            "ultralytics==8.4.138",
            "tactile_paving",
            'task="segment"',
            "seed=42",
            "epochs=80",
            "imgsz=640",
            "station-tactile-v2",
            "--branch codex/model-first-mobile-ready --single-branch",
            'RUN_NAME = "tactile-one-class-v2"',
            "export_manifest.json",
            "best.pt",
            "sha256.txt",
        ):
            self.assertIn(required, source)
        self.assertNotIn("data/samples", source)
        self.assertNotIn("station-photo-set-50-v1", source)
        self.assertNotIn('RUN_NAME = "tactile-one-class-v1"', source)

    def test_notebook_has_no_saved_execution_outputs(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_notebook_replaces_stale_candidate_and_downloads_archive(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("shutil.rmtree(CANDIDATE_DIR)", source)
        self.assertIn("shutil.make_archive", source)
        self.assertIn("files.download", source)

    def test_python_cells_compile(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.lstrip().startswith("!"):
                continue
            compile(source, f"notebook-cell-{index}", "exec")

    def test_v3_notebook_is_self_contained_and_trains_only_ratio_two(self) -> None:
        notebook = json.loads(V3_NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 6)
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        for required in (
            "GIT_LFS_SKIP_SMUDGE",
            "--include",
            "data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip",
            "data/public/guidetwsi-rbar-v1/sha256.txt",
            "models/guidetwsi/yolo11n_tactile.pt",
            "artifacts/datasets/station-tactile-v2",
            "artifacts/candidates/tactile-one-class-v3-public4-station1",
            "hashlib.sha256",
            "EXPECTED_CACHE_FILES = 3960",
            "TRAIN_RATIOS = (2,)",
            "build_tactile_v3_dataset.py",
            "train_tactile_v3.py",
            "torch.cuda.is_available",
            "REQUIRED_OUTPUTS",
            "files.download",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "files.upload",
            "Upload guidetwsi-rbar-v1.zip",
            "acquire_guidetwsi_subset.py",
            "prepare_guidetwsi_subset.py",
            "evaluate_tactile_candidate.py",
            "data/ground_truth",
            "data/samples",
        ):
            self.assertNotIn(forbidden, source)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])


if __name__ == "__main__":
    unittest.main()
