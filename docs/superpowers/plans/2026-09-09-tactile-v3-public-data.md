# Tactile v3 Public-Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Acquire and curate a reproducible GuideTWSI RBar subset under a hard 5 GB local cap, train tactile candidate v3 without losing the upstream model's generalization, and evaluate it against the existing offline quality gates.

**Architecture:** Raw public files live only in an ignored cache. A fail-closed acquisition manifest binds every retained file to its upstream path, license, byte size, and SHA-256. A preparation stage validates YOLO polygons, removes exact/perceptual duplicates and protected-set leakage, then exports source-aware training mixtures. Candidate selection uses validation data; the protected set is touched only once after thresholds are locked. Mobile export remains out of scope.

**Tech Stack:** Python 3.12, `kagglehub==1.0.2`, OpenCV, PyYAML, Ultralytics 8.4.138, YOLO11 segmentation, `unittest`.

---

## File map

- Modify `.gitignore`: exclude the raw public-data cache and local acquisition environment.
- Create `requirements-public-data.txt`: pin the optional Kaggle acquisition dependency.
- Create `scripts/acquire_guidetwsi_subset.py`: inventory and download only selected RBar files while enforcing source, license, path, and byte-budget rules.
- Create `tests/test_acquire_guidetwsi_subset.py`: deterministic acquisition policy and fail-closed download tests.
- Create `scripts/prepare_guidetwsi_subset.py`: pair images/YOLO polygons, validate content, sample source groups, and reject duplicates/leakage.
- Create `tests/test_prepare_guidetwsi_subset.py`: polygon, grouping, sampling, duplicate, and protected-leakage tests.
- Create `scripts/build_tactile_v3_dataset.py`: combine curated public train samples with reviewed station samples at fixed ratios.
- Create `tests/test_build_tactile_v3_dataset.py`: mixture and split-isolation tests.
- Create `scripts/train_tactile_v3.py`: reproducible local/Colab-compatible candidate training entry point.
- Create `tests/test_train_tactile_v3.py`: static configuration and command-construction tests.
- Create `data/public/guidetwsi-rbar-v1/provenance.json`: compact retained-file provenance only.
- Create `data/training/manifests/tactile-v3-*.json`: immutable dataset and experiment manifests.
- Create `artifacts/candidates/tactile-one-class-v3-*/`: candidate checkpoints, configurations, metrics, checksums, and evaluation reports.

### Task 1: Build a budgeted, auditable GuideTWSI acquisition path

**Files:**
- Modify: `.gitignore`
- Create: `requirements-public-data.txt`
- Create: `scripts/acquire_guidetwsi_subset.py`
- Create: `tests/test_acquire_guidetwsi_subset.py`

- [ ] **Step 1: Write failing policy tests**

Cover an allowed `guidedogrobot/guidetwsi` RBar path, rejection of unrelated paths, missing/unknown license, path traversal, duplicate inventory entries, and a selection that would exceed `5_000_000_000` bytes.

- [ ] **Step 2: Run the focused tests and confirm failure**

```powershell
rtk .\.venv-preprocess\Scripts\python.exe -m unittest tests.test_acquire_guidetwsi_subset -v
```

- [ ] **Step 3: Implement the minimal inventory/selection core**

Keep network access behind an injected downloader so tests remain offline. Sort eligible image/label pairs by a stable seeded hash, retain complete pairs only, and stop before the configured byte limit.

- [ ] **Step 4: Add atomic download and provenance writing**

Download to `runs/public-data-cache/guidetwsi-rbar-v1/.staging`, verify byte size and SHA-256, then move into the cache. Write only compact provenance to `data/public/guidetwsi-rbar-v1/provenance.json`. An interrupted or mismatched file must never be marked retained.

- [ ] **Step 5: Install the pinned optional dependency in a dedicated ignored environment and query the public inventory**

```powershell
rtk py -3.12 -m venv .venv-public-data
rtk .\.venv-public-data\Scripts\python.exe -m pip install -r requirements-public-data.txt
rtk .\.venv-public-data\Scripts\python.exe scripts\acquire_guidetwsi_subset.py inventory --output runs\public-data-cache\guidetwsi-inventory.json
```

Expected: the inventory identifies the RBar train images and labels without downloading the complete 45 GB release.

### Task 2: Curate at most 2,000 valid public training images

**Files:**
- Create: `scripts/prepare_guidetwsi_subset.py`
- Create: `tests/test_prepare_guidetwsi_subset.py`
- Create: `data/training/manifests/guidetwsi-rbar-2k-v1.json`

- [ ] **Step 1: Write failing annotation and deduplication tests**

Cover normalized YOLO segmentation parsing, invalid class/coordinates, missing image-label pairs, corrupt images, exact duplicates, dHash near-duplicates, and matches against `data/samples` protected hashes.

- [ ] **Step 2: Implement validation and source-aware deterministic sampling**

Use upstream source/group metadata when present; otherwise derive a conservative group from the upstream directory. Preserve only class `0 = tactile_paving`, nondegenerate polygons, and decoded walking-scene images.

- [ ] **Step 3: Produce the capped local subset and manifest**

```powershell
rtk .\.venv-public-data\Scripts\python.exe scripts\acquire_guidetwsi_subset.py download --inventory runs\public-data-cache\guidetwsi-inventory.json --max-bytes 5000000000 --max-images 2000
rtk .\.venv-preprocess\Scripts\python.exe scripts\prepare_guidetwsi_subset.py --cache runs\public-data-cache\guidetwsi-rbar-v1 --protected-dir data\samples --limit 2000 --output data\training\manifests\guidetwsi-rbar-2k-v1.json
```

Expected: no invalid samples, no protected leakage, no exact duplicates, and a recorded near-duplicate report.

### Task 3: Build source-safe v3 training mixtures

**Files:**
- Create: `scripts/build_tactile_v3_dataset.py`
- Create: `tests/test_build_tactile_v3_dataset.py`
- Create: `data/training/manifests/tactile-v3-public4-station1.json`
- Create: `data/training/manifests/tactile-v3-public2-station1.json`

- [ ] **Step 1: Write failing ratio and leakage tests**

Assert that public samples remain train-only, reviewed station train/validation assignments are preserved, protected samples never appear, public-to-station ratios are exact within one sample, and every exported label has a matching image.

- [ ] **Step 2: Implement immutable YOLO dataset exports**

Create independent `4:1` and `2:1` exports under ignored `artifacts/datasets/`. Use deterministic oversampling/list generation instead of copying large images repeatedly when Ultralytics input permits it.

- [ ] **Step 3: Validate both manifests and exports**

Expected: hashes, source groups, ratios, class names, and split counts match the manifest; station validation is unchanged.

### Task 4: Train and rank candidate v3 experiments

**Files:**
- Create: `scripts/train_tactile_v3.py`
- Create: `tests/test_train_tactile_v3.py`
- Create: `artifacts/candidates/tactile-one-class-v3-public4-station1/`
- Create: `artifacts/candidates/tactile-one-class-v3-public2-station1/`

- [ ] **Step 1: Write failing static configuration tests**

Require the GuideTWSI checkpoint as initialization, seed `42`, `imgsz=640`, early stopping, unique non-overwriting run directories, and complete environment/config/checksum capture.

- [ ] **Step 2: Implement the training entry point**

Support resumable background execution and a `--dry-run` mode. Refuse the regressed v2 checkpoint as initialization.

- [ ] **Step 3: Train the `4:1` and `2:1` mixtures**

Run sequentially to stay within disk/GPU limits. If both show validation forgetting versus the GuideTWSI baseline, add one frozen-backbone warm-up experiment; otherwise do not expand the sweep.

- [ ] **Step 4: Verify and rank candidates by validation recall, then mean IoU, then false-positive rate**

No protected-set thresholds or results are used to choose among experiments.

### Task 5: Lock threshold and run the protected gate once

**Files:**
- Modify: `artifacts/candidates/tactile-one-class-v3-*/evaluation_report.json`
- Verify: `scripts/evaluate_tactile_candidate.py`
- Verify: `scripts/verify_tactile_candidate.py`

- [ ] **Step 1: Tune confidence only on validation and write the locked value into the selected candidate manifest.**
- [ ] **Step 2: Verify the candidate bundle and checksum before protected evaluation.**
- [ ] **Step 3: Evaluate the selected candidate once on the existing 17-image protected set.**
- [ ] **Step 4: Accept only if recall is `>= 0.90`, positive-mask mean IoU is `>= 0.75`, and false-positive rate is `<= 0.50`.**
- [ ] **Step 5: If the gate fails, record failure categories and the next data slice; do not tune against protected images.**

### Task 6: Final verification and GitHub handoff

- [ ] **Step 1: Run focused tests after each task, then the full suite.**

```powershell
rtk .\.venv-preprocess\Scripts\python.exe -m unittest discover -s tests -q
rtk .\.venv-preprocess\Scripts\python.exe -m compileall -q scripts tests
```

- [ ] **Step 2: Confirm raw public images, virtual environments, and training runs are ignored and absent from the staged diff.**
- [ ] **Step 3: Commit only code, manifests, provenance, compact reports, and approved candidate artifacts; push to `Riqqi15/Yoloooo`.**
- [ ] **Step 4: Report measured gates plainly. Do not claim mobile readiness or walking safety from this phase.**

