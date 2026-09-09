# Model-First Mobile-Ready Tactile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reviewed tactile dataset v2, a measured segmentation candidate, and a verified LiteRT/TFLite package for later Android camera integration.

**Architecture:** Existing LabelMe, manifest, training gate, and candidate verifier remain the only path from images to a model. New code is limited to a Commons batch adapter, model evaluation/export verification, and a mobile-package verifier; training remains in the pinned Colab notebook. Safety-critical people, obstacle, hole, platform-edge, and train-door detection stay separate from the one-class tactile model.

**Tech Stack:** Python 3.12 locally, OpenCV, NumPy, Ultralytics 8.4.138 in Colab, LabelMe JSON, YOLO11 segmentation, LiteRT/TFLite, unittest.

---

## File map

- Modify `scripts/prepare_station_training_set.py`: accept either the existing `sources.json` or Commons `sources.csv`, while preserving the current output contract.
- Modify `tests/test_prepare_station_training_set.py`: cover Commons CSV ingestion and LabelMe geometry.
- Create `data/training/manifests/commons-tactile-v1.json`: split and provenance manifest after human review.
- Create `data/training/annotations/commons-tactile-v1/`: converted internal annotations keyed by content hash.
- Modify `notebooks/train_tactile_one_class_colab.ipynb`: train candidate v2 and export reference/mobile models without overwriting v1.
- Create `scripts/evaluate_tactile_candidate.py`: per-image mask IoU, recall, false-positive, latency, and failure-case report.
- Create `tests/test_evaluate_tactile_candidate.py`: deterministic metric tests.
- Create `scripts/export_tactile_mobile.py`: FP32/INT8 export plus manifest/checksum assembly.
- Create `scripts/verify_tactile_mobile.py`: fail-closed package and parity verification.
- Create `tests/test_verify_tactile_mobile.py`: corrupt-package and parity-gate tests.
- Create `artifacts/mobile/tactile-one-class-v2/`: final package only after every gate passes.

### Task 1: Make the Commons LabelMe batch reproducible

**Files:**
- Modify: `scripts/prepare_station_training_set.py`
- Modify: `tests/test_prepare_station_training_set.py`
- Input: `data/training/inbox/commons_tactile_station_20260908/`

- [x] **Step 1: Write the failing Commons ingestion test**

Add a test fixture containing `sources.csv`, one JPEG, and one matching LabelMe JSON. Assert:

```python
report = prepare_dataset(source, metadata, annotations, groups={"station.jpg": ("station", "session")})
self.assertEqual(report["samples"], 1)
self.assertEqual(report["positive_labelme"], 1)
self.assertEqual(rows[0]["usage_permission"], "CC BY-SA 4.0; https://commons.wikimedia.org/wiki/File:Station.jpg")
```

- [x] **Step 2: Run the focused test and confirm it fails**

```powershell
rtk .\.venv\Scripts\python.exe -m unittest tests.test_prepare_station_training_set -v
```

Expected: the CSV fixture fails because the loader currently requires `sources.json`.

- [x] **Step 3: Add the minimal source-table loader**

Implement one helper:

```python
def read_source_rows(source_dir: Path) -> list[dict[str, Any]]:
    json_path = source_dir / "sources.json"
    csv_path = source_dir / "sources.csv"
    if json_path.is_file():
        return json.loads(json_path.read_text(encoding="utf-8"))
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            return [
                {
                    "set_index": index,
                    "filename": row["filename"],
                    "title": row["title"],
                    "source": row["source_url"],
                    "license": row["license"],
                    "artist": row["artist"],
                }
                for index, row in enumerate(csv.DictReader(handle), start=1)
            ]
    raise FileNotFoundError("sources.json or sources.csv is required")
```

- [x] **Step 4: Convert and structurally validate the current 19-image batch**

```powershell
rtk .\.venv\Scripts\python.exe scripts\prepare_station_training_set.py --source-dir data\training\inbox\commons_tactile_station_20260908 --metadata data\training\intake_commons_tactile_v1.csv --annotations-dir data\training\annotations\commons-tactile-v1
rtk .\.venv\Scripts\python.exe scripts\build_training_manifest.py --metadata data\training\intake_commons_tactile_v1.csv --source-dir data\training\inbox\commons_tactile_station_20260908 --dataset-version commons-tactile-v1 --output data\training\manifests\commons-tactile-v1.json
rtk .\.venv\Scripts\python.exe scripts\training_label_gate.py --manifest data\training\manifests\commons-tactile-v1.json --annotations-dir data\training\annotations\commons-tactile-v1 --taxonomy data\training\taxonomy_tactile_v1.json render --output-dir runs\commons-tactile-v1-review
```

Expected: 19 valid samples and 19 review overlays; `training_ready` remains false until review hashes are recorded.

- [x] **Step 5: Human review gate**

Riyadh has opened the batch in LabelMe, corrected visible-pixel polygons, excluded the trolley photo, and saved 19 accepted annotations. Re-run Step 4 after every later revision. Only then record review hashes:

```powershell
rtk .\.venv\Scripts\python.exe scripts\training_label_gate.py --manifest data\training\manifests\commons-tactile-v1.json --annotations-dir data\training\annotations\commons-tactile-v1 --review-manifest data\training\review_commons_tactile_v1.csv --taxonomy data\training\taxonomy_tactile_v1.json review --reviewer Riyadh --samples <all 64-character sample IDs from the manifest>
```

Expected: review rows bind the exact source and annotation hashes; later edits invalidate the review.

### Task 2: Build dataset v2 without split leakage

**Files:**
- Create: `data/training/manifests/station-tactile-v2.json`
- Create: `data/training/review_station_tactile_v2.csv`
- Create: `artifacts/datasets/station-tactile-v2/`

- [x] **Step 1: Merge the reviewed local 50-image set and reviewed Commons batch by content hash and source group.**
- [x] **Step 2: Run exact-hash and dHash leakage checks; reject a derivative crossing train/validation/test.**
- [x] **Step 3: Keep the existing 17-image local ground-truth set protected and outside training.**
- [x] **Step 4: Export YOLO segmentation only after `training_ready: true`.**
- [x] **Step 5: Commit the immutable dataset manifest and checksums.**

Verification:

```powershell
rtk .\.venv\Scripts\python.exe scripts\training_label_gate.py --manifest data\training\manifests\station-tactile-v2.json --annotations-dir data\training\annotations\station-tactile-v2 --review-manifest data\training\review_station_tactile_v2.csv --taxonomy data\training\taxonomy_tactile_v1.json validate
```

Expected: zero invalid/unreviewed rows and nonempty train, validation, and test splits.

### Task 3: Train candidate v2 in Colab

**Files:**
- Modify: `notebooks/train_tactile_one_class_colab.ipynb`
- Create: `artifacts/candidates/tactile-one-class-v2/`

- [x] **Step 1: Add a static notebook test requiring dataset version `station-tactile-v2`, run name `tactile-one-class-v2`, `imgsz=640`, and unique output paths.**
- [ ] **Step 2: Clone the Riqqi15 repository with Git LFS and copy the v2 export in Colab.**
- [ ] **Step 3: Train with the pinned seed/dependency/config and preserve every metric/config/environment file.**
- [ ] **Step 4: Validate the v2 candidate locally.**

```powershell
rtk .\.venv\Scripts\python.exe scripts\verify_tactile_candidate.py artifacts\candidates\tactile-one-class-v2
```

Expected: `candidate_valid`; this does not yet mean mobile-ready.

### Task 4: Evaluate and lock the model threshold

**Files:**
- Create: `scripts/evaluate_tactile_candidate.py`
- Create: `tests/test_evaluate_tactile_candidate.py`
- Create: `artifacts/candidates/tactile-one-class-v2/evaluation_report.json`

- [ ] **Step 1: Write deterministic mask-IoU, precision, recall, empty-mask, and false-positive tests.**
- [ ] **Step 2: Implement evaluation using the existing ground-truth manifest and visible-pixel masks.**
- [ ] **Step 3: Tune confidence/mask thresholds on validation only.**
- [ ] **Step 4: Lock configuration, then run the 17-image protected local test once.**
- [ ] **Step 5: Reject mobile export unless mean IoU is at least 0.75 and mask recall at least 0.90.**

### Task 5: Export and verify the mobile package

**Files:**
- Create: `scripts/export_tactile_mobile.py`
- Create: `scripts/verify_tactile_mobile.py`
- Create: `tests/test_verify_tactile_mobile.py`
- Create: `artifacts/mobile/tactile-one-class-v2/`

- [ ] **Step 1: Write tests rejecting missing files, wrong class, wrong tensor metadata, checksum mismatch, NaN/Inf parity, and metrics below gate.**
- [ ] **Step 2: Export FP32 LiteRT/TFLite at 320x320 and reload it for smoke inference.**
- [ ] **Step 3: Export INT8 using only train-split representative images and reload it.**
- [ ] **Step 4: Run `.pt`/FP32/INT8 parity on identical preprocessing and samples.**
- [ ] **Step 5: Require median mask IoU at least 0.95 for FP32, 0.90 for INT8, and INT8 recall drop no more than 0.03.**
- [ ] **Step 6: Assemble labels, thresholds, model manifest, checksums, model card, evaluation, parity report, and sample input/output.**

### Task 6: Final verification and handoff

**Files:**
- Verify: all scripts, tests, candidate files, and mobile artifacts.

- [ ] **Step 1: Run the full local suite and compile Python files.**

```powershell
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -q
rtk .\.venv\Scripts\python.exe -m compileall -q scripts tests
```

- [ ] **Step 2: Run candidate and mobile package verifiers.**
- [ ] **Step 3: Confirm local `main`, GitHub `Riqqi15/Yoloooo`, and all LFS objects match.**
- [ ] **Step 4: Mark the model package `mobile_integration_ready` only if every model and parity gate passes.**

The separate Android application plan begins only after Task 6. It will combine tactile segmentation with separately verified obstacle/hazard perception, depth when supported, stable-frame direction logic, TTS/haptics, and fail-closed `STOP`/`Uncertain` behavior.
