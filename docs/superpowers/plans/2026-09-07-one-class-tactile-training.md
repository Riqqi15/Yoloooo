# One-Class Tactile Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Colab fine-tuning path and a fail-closed local verifier for a one-class `tactile_paving` segmentation candidate.

**Architecture:** The existing intake and label gate remain the only route into training. A dedicated approved taxonomy exports one tactile class, a Colab notebook validates the export contract and trains YOLO segmentation, and a small local CLI verifies the returned candidate bundle and checkpoint metadata.

**Tech Stack:** Python 3.11, standard library, Ultralytics 8.4.138, unittest, Google Colab, YOLO segmentation.

---

Workspace note: this directory is not a Git repository, so commit steps are intentionally omitted. Changes are made directly and verified locally.

## File map

- Create `data/training/taxonomy_tactile_v1.json`: approved one-class tactile taxonomy used by export.
- Create `notebooks/train_tactile_one_class_colab.ipynb`: Colab training and artifact packaging.
- Create `scripts/verify_tactile_candidate.py`: local candidate integrity and model-contract CLI.
- Create `tests/test_verify_tactile_candidate.py`: verifier behavior tests.
- Create `tests/test_tactile_training_notebook.py`: static notebook contract tests.
- Modify `data/training/README.md`: current 17-test protection and one-class workflow.
- Modify `training/02-colab-training-and-export.md`: point to the runnable notebook and candidate verifier.
- Modify `NEW_CHAT_CONTEXT.md`: record the new training scaffold and remaining data blocker.

### Task 1: Approve one-class tactile taxonomy

**Files:**
- Create: `data/training/taxonomy_tactile_v1.json`
- Test: `tests/test_training_label_gate.py`

- [x] **Step 1: Write the failing taxonomy contract test**

Add a test that loads the repository taxonomy and asserts the selected contract:

```python
def test_repository_tactile_taxonomy_is_approved_one_class(self):
    taxonomy = json.loads(Path("data/training/taxonomy_tactile_v1.json").read_text(encoding="utf-8"))
    self.assertEqual(taxonomy["status"], "approved")
    self.assertEqual(taxonomy["tactile_segmentation"], ["tactile_paving"])
    self.assertTrue(taxonomy["object_detection"])
```

- [x] **Step 2: Verify the test fails because the taxonomy does not exist**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest tests.test_training_label_gate.TrainingLabelGateTest.test_repository_tactile_taxonomy_is_approved_one_class -v
```

Expected: `FileNotFoundError` for `taxonomy_tactile_v1.json`.

- [x] **Step 3: Create the approved taxonomy**

Create JSON with `taxonomy_version: station-tactile-v1`, `status: approved`, the existing object label list retained for gate compatibility, and exactly:

```json
"tactile_segmentation": ["tactile_paving"]
```

Record that Riyadh approved the one-class direction on 7 September 2026 and that guiding/warning separation is deferred.

- [x] **Step 4: Run the targeted test**

Expected: one test passes.

### Task 2: Add fail-closed candidate verification

**Files:**
- Create: `scripts/verify_tactile_candidate.py`
- Create: `tests/test_verify_tactile_candidate.py`

- [x] **Step 1: Write failing tests for a valid bundle and corrupt states**

The test fixture creates nonempty `best.pt`, `metrics.json`, `training_config.json`, `environment.txt`, `sha256.txt`, and `MODEL_CARD.md`. Patch `verify_tactile_candidate.YOLO` so the valid model exposes:

```python
model.task = "segment"
model.names = {0: "tactile_paving"}
```

Add tests for:

```python
self.assertEqual(validate_candidate(candidate)["status"], "candidate_valid")
```

and `ValueError` for a wrong checksum, wrong task, wrong class name, missing required file, or missing training configuration key.

- [x] **Step 2: Run the verifier tests and confirm import failure**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest tests.test_verify_tactile_candidate -v
```

Expected: failure because `verify_tactile_candidate.py` does not exist.

- [x] **Step 3: Implement the smallest verifier**

Implement:

```python
REQUIRED_FILES = (
    "best.pt", "metrics.json", "training_config.json",
    "environment.txt", "sha256.txt", "MODEL_CARD.md",
)

def validate_candidate(candidate_dir: Path) -> dict[str, object]:
    # Reject missing/empty files.
    # Compare sha256.txt with sha256_file(best.pt).
    # Require dataset_version, seed, imgsz, epochs, and ultralytics_version.
    # Load YOLO(best.pt); require task == "segment" and names == {0: "tactile_paving"}.
    # Return a JSON-serializable candidate_valid report.
```

CLI contract:

```powershell
rtk .\.venv\Scripts\python.exe scripts\verify_tactile_candidate.py artifacts\candidates\tactile-20260907-run1
```

Success prints JSON and exits 0. Failure prints `STOP_FAILURE: <reason>` to stderr and exits 1.

- [x] **Step 4: Run verifier tests**

Expected: valid bundle passes; all corrupt-state tests pass by observing rejection.

### Task 3: Add the Colab notebook contract

**Files:**
- Create: `tests/test_tactile_training_notebook.py`
- Create: `notebooks/train_tactile_one_class_colab.ipynb`

- [x] **Step 1: Write a failing static notebook test**

Use standard-library `json` to load the notebook. Concatenate code-cell sources and assert it contains these operational contracts:

```python
self.assertIn('ultralytics==8.4.138', source)
self.assertIn('tactile_paving', source)
self.assertIn('task="segment"', source)
self.assertIn('seed=42', source)
self.assertIn('epochs=80', source)
self.assertIn('imgsz=640', source)
self.assertIn('export_manifest.json', source)
self.assertIn('best.pt', source)
self.assertIn('sha256.txt', source)
```

Also assert the notebook contains no training reference to `data/samples`.

- [x] **Step 2: Run the notebook test and confirm missing-file failure**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest tests.test_tactile_training_notebook -v
```

Expected: `FileNotFoundError` for the notebook.

- [x] **Step 3: Create the minimal notebook**

Cells must perform:

1. Runtime warning and GPU check.
2. `pip install ultralytics==8.4.138`.
3. User-editable `DATASET_ROOT`, `CHECKPOINT`, `RUN_NAME`, and `CANDIDATE_DIR` paths.
4. Validation that `tactile/data.yaml`, `export_manifest.json`, train images/labels, and val images/labels exist and are nonempty.
5. Validation that YAML names normalize to `{0: "tactile_paving"}` and export validation says `training_ready: true`.
6. Checkpoint load with `YOLO(CHECKPOINT, task="segment")` and one-image smoke prediction.
7. Fine-tuning with `epochs=80`, `imgsz=640`, `batch=-1`, `device=0`, `seed=42`, and conservative augmentation values.
8. Validation of `best.pt`.
9. Packaging of required candidate files and SHA-256.

The packaging cell writes a `training_config.json` containing:

```json
{
  "dataset_version": "from export_manifest.json",
  "seed": 42,
  "imgsz": 640,
  "epochs": 80,
  "ultralytics_version": "8.4.138"
}
```

- [x] **Step 4: Run notebook static tests**

Expected: notebook parses as valid JSON and all contract assertions pass.

### Task 4: Synchronize active documentation

**Files:**
- Modify: `data/training/README.md`
- Modify: `training/02-colab-training-and-export.md`
- Modify: `NEW_CHAT_CONTEXT.md`

- [x] **Step 1: Update the training intake guide**

Change protected test count from 10 to 17. Document `taxonomy_tactile_v1.json`, `--taxonomy data/training/taxonomy_tactile_v1.json`, the notebook path, and the candidate verification command.

- [x] **Step 2: Update the Colab guide**

State that the runnable one-class baseline is `notebooks/train_tactile_one_class_colab.ipynb`, pinned to Ultralytics 8.4.138. Keep the two-class guiding/warning target explicitly deferred.

- [x] **Step 3: Update handoff status**

Record that preprocessing/export and the training scaffold exist, but full training remains blocked until new human-reviewed training data provides nonempty train and validation splits.

### Task 5: Full verification

**Files:**
- Verify all modified and created files.

- [x] **Step 1: Run the complete test suite**

```powershell
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Expected: all tests pass with zero failures.

- [x] **Step 2: Compile Python sources and tests**

```powershell
rtk .\.venv\Scripts\python.exe -m compileall -q scripts tests
```

Expected: exit code 0 and no output.

- [x] **Step 3: Validate active ground truth remains unchanged**

```powershell
rtk .\.venv\Scripts\python.exe scripts\ground_truth_manifest.py validate
```

Expected: 17 reviewed rows, zero invalid rows, and `official_ready: true`.

- [x] **Step 4: Verify notebook JSON and documentation state**

```powershell
rtk .\.venv\Scripts\python.exe -m json.tool notebooks\train_tactile_one_class_colab.ipynb > NUL
rtk rg -n "taxonomy_tactile_v1|train_tactile_one_class_colab|verify_tactile_candidate|17 foto" data\training\README.md training\02-colab-training-and-export.md NEW_CHAT_CONTEXT.md
```

Expected: JSON command exits 0 and each new workflow component appears in active documentation.

## Execution note

During notebook integration, a regression test showed that exporter output `path: .` resolved against the process working directory under Ultralytics 8.4.138. `scripts/training_label_gate.py` now omits that field so paths resolve from the YAML directory. The focused regression test and full suite pass.
