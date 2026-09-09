from __future__ import annotations

import json
import unittest
from pathlib import Path


NOTEBOOK = Path("notebooks/train_tactile_one_class_colab.ipynb")


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

    def test_python_cells_compile(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.lstrip().startswith("!"):
                continue
            compile(source, f"notebook-cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
