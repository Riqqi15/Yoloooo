# Colab Training Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a reproducible tactile-v3 training handoff whose required public data lives in GitHub LFS and whose cleaned Colab notebook trains only the unfinished 2:1 candidate from a fresh GPU runtime.

**Architecture:** GitHub is the source of truth for code, manifests, the GuideTWSI checkpoint, the station dataset, the completed 4:1 candidate, and one checksummed public-data ZIP. The notebook performs a targeted LFS pull, verifies and extracts the ZIP, builds the deterministic 2:1 dataset, trains on CUDA, validates the candidate bundle, and downloads it. `AGENTS.md` gives Codex immediate context while `docs/HANDOFF_TACTILE_V3.md` carries the detailed progress, scores, targets, safety boundaries, and continuation checklist.

**Tech Stack:** Python 3, Google Colab, Ultralytics `8.4.138`, PyTorch/CUDA, Git LFS, Jupyter Notebook JSON, Python `unittest`.

---

### Task 1: Lock the handoff and notebook contract with tests

**Files:**
- Create: `tests/test_tactile_handoff.py`
- Modify: `tests/test_tactile_training_notebook.py`

- [ ] **Step 1: Write the failing handoff test**

Create `tests/test_tactile_handoff.py` with assertions that the handoff files exist and expose the required facts:

```python
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ZIP = ROOT / "data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip"
CHECKSUM = ROOT / "data/public/guidetwsi-rbar-v1/sha256.txt"


class TactileHandoffTest(unittest.TestCase):
    def test_codex_entrypoint_and_handoff_cover_progress_and_targets(self) -> None:
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        handoff = (ROOT / "docs/HANDOFF_TACTILE_V3.md").read_text(encoding="utf-8")
        self.assertIn("docs/HANDOFF_TACTILE_V3.md", agents)
        for required in (
            "82.50%",
            "56.41%",
            "100.00%",
            "tactile-one-class-v3-public2-station1",
            "not mobile-ready",
            "5 meter",
            "Commuter",
            "trolley",
        ):
            self.assertIn(required, handoff)

    def test_public_archive_matches_committed_checksum(self) -> None:
        expected = CHECKSUM.read_text(encoding="utf-8").split()[0].lower()
        digest = hashlib.sha256()
        with PUBLIC_ZIP.open("rb") as archive:
            for chunk in iter(lambda: archive.read(1024 * 1024), b""):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Strengthen the notebook contract**

Replace the v3 notebook assertions in `tests/test_tactile_training_notebook.py` so the test requires:

```python
self.assertEqual(len(notebook["cells"]), 6)
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
```

Keep the existing checks that every Python cell compiles and every code cell has `execution_count: null` and `outputs: []`.

- [ ] **Step 3: Run the tests and confirm the new contract fails**

Run:

```powershell
rtk python -m unittest tests.test_tactile_handoff tests.test_tactile_training_notebook -v
```

Expected: failure because `AGENTS.md`, `docs/HANDOFF_TACTILE_V3.md`, and the tracked public ZIP/checksum do not exist yet and the notebook still uses manual upload.

- [ ] **Step 4: Commit the failing tests**

```powershell
rtk git add -- tests/test_tactile_handoff.py tests/test_tactile_training_notebook.py
rtk git commit -m "test: define tactile Colab handoff contract"
```

### Task 2: Publish the immutable public cache through Git LFS

**Files:**
- Modify: `.gitattributes`
- Create: `data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip`
- Create: `data/public/guidetwsi-rbar-v1/sha256.txt`

- [ ] **Step 1: Add a path-scoped LFS rule**

Append this exact rule to `.gitattributes`:

```gitattributes
data/public/guidetwsi-rbar-v1/*.zip filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 2: Copy the already verified cache to its tracked location**

Run:

```powershell
rtk powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path 'data/public/guidetwsi-rbar-v1' | Out-Null; Copy-Item -LiteralPath 'runs/public-data-cache/guidetwsi-rbar-v1.zip' -Destination 'data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip'"
```

Expected: the tracked copy is exactly `345332596` bytes; the original ignored cache remains untouched.

- [ ] **Step 3: Record the SHA-256 checksum**

Create `data/public/guidetwsi-rbar-v1/sha256.txt` with the checksum already verified from the source ZIP:

```text
4796d2eea993f62b2db0934b3fe639c4d33b7bf481281509d33e3901c8dbd99e  guidetwsi-rbar-v1.zip
```

- [ ] **Step 4: Verify LFS and archive integrity**

Run:

```powershell
rtk git check-attr filter -- data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip
rtk python -m unittest tests.test_tactile_handoff.TactileHandoffTest.test_public_archive_matches_committed_checksum -v
```

Expected: `filter: lfs`; the checksum test passes.

- [ ] **Step 5: Commit the immutable cache**

```powershell
rtk git add -- .gitattributes data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip data/public/guidetwsi-rbar-v1/sha256.txt
rtk git commit -m "data: publish GuideTWSI cache for Colab"
```

### Task 3: Add the Codex entrypoint and collaborator handoff

**Files:**
- Create: `AGENTS.md`
- Create: `docs/HANDOFF_TACTILE_V3.md`

- [ ] **Step 1: Create the concise Codex entrypoint**

Create `AGENTS.md` with these exact sections and rules:

```markdown
# Tactile Navigation Model Context

Read `docs/HANDOFF_TACTILE_V3.md` before changing training data, notebooks, models, evaluation, or mobile integration.

## Non-negotiable rules

- The current model is single-class `tactile_paving` segmentation, not a complete navigation safety system.
- Never train on or tune from protected ground-truth data.
- Never describe a candidate as mobile-ready until every documented gate passes.
- Preserve dataset provenance, manifests, checksums, and the excluded trolley source.
- Keep large binary artifacts in Git LFS.
```

- [ ] **Step 2: Write the full collaborator handoff**

Create `docs/HANDOFF_TACTILE_V3.md` using the approved design as the source of truth. It must contain:

1. product goal and current one-class model boundary;
2. completed 4:1 training metrics and protected-test table;
3. overall status `rejected / not mobile-ready`;
4. incomplete 2:1 status and the exact next Colab task;
5. repository artifact inventory and public-data license/provenance;
6. the P0/P1/P2 improvement roadmap, including the excluded trolley source;
7. the model gate and video/safety gate;
8. a numbered collaborator checklist ending with returning the candidate ZIP for protected local evaluation;
9. exact notebook path and branch;
10. explicit warning that 5 meter is an observation target, not a monocular safety guarantee.

- [ ] **Step 3: Run the handoff content test**

```powershell
rtk python -m unittest tests.test_tactile_handoff.TactileHandoffTest.test_codex_entrypoint_and_handoff_cover_progress_and_targets -v
```

Expected: PASS.

- [ ] **Step 4: Commit the documentation**

```powershell
rtk git add -- AGENTS.md docs/HANDOFF_TACTILE_V3.md
rtk git commit -m "docs: add tactile training handoff"
```

### Task 4: Replace the Colab notebook with a clean six-cell workflow

**Files:**
- Modify: `notebooks/train_tactile_v3_public_colab.ipynb`

- [ ] **Step 1: Replace all existing notebook cells**

Keep exactly six purposeful cells in this order:

1. Markdown: current 4:1 score/status, 2:1 goal, safety boundary, GPU instruction, and run order.
2. Code: install pinned Ultralytics and Git LFS.
3. Code: clone with `GIT_LFS_SKIP_SMUDGE=1`, pull only the checkpoint, station dataset, completed 4:1 candidate, and public ZIP, then define `run_script`.
4. Code: verify ZIP SHA-256, reject unsafe member paths, extract it, verify exactly 3,960 files, and build only `tactile-v3-public2-station1`.
5. Code: require CUDA and train only `tactile-one-class-v3-public2-station1` for 80 epochs with batch 8.
6. Code: require `best.pt`, `metrics.json`, `training_config.json`, `MODEL_CARD.md`, and `sha256.txt`; archive and download the candidate.

The targeted clone/pull code must use this include list:

```python
LFS_INCLUDE = ",".join((
    "models/guidetwsi/yolo11n_tactile.pt",
    "artifacts/datasets/station-tactile-v2/**",
    "artifacts/candidates/tactile-one-class-v3-public4-station1/**",
    "data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip",
))
env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
subprocess.run([
    "git", "clone", "--branch", BRANCH, "--single-branch",
    "https://github.com/Riqqi15/Yoloooo.git", str(REPO),
], env=env, check=True)
subprocess.run([
    "git", "-C", str(REPO), "lfs", "pull", "--include", LFS_INCLUDE,
], check=True)
```

The checksum comparison must read the committed checksum and fail before extraction on mismatch:

```python
PUBLIC_DIR = REPO / "data/public/guidetwsi-rbar-v1"
ARCHIVE = PUBLIC_DIR / "guidetwsi-rbar-v1.zip"
expected_sha256 = (PUBLIC_DIR / "sha256.txt").read_text(encoding="utf-8").split()[0]
actual_sha256 = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
if actual_sha256 != expected_sha256:
    raise ValueError(f"SHA-256 dataset publik tidak cocok: {actual_sha256}")
```

- [ ] **Step 2: Remove all stale and manual-upload content**

Ensure the notebook contains no saved output, execution count, trial cells, ad-hoc diagnostics, manual file upload, public downloader, protected evaluation, or protected-data path. The notebook must never train ratio 4 by default.

- [ ] **Step 3: Run notebook contract and compilation tests**

```powershell
rtk python -m unittest tests.test_tactile_training_notebook -v
```

Expected: all notebook tests PASS, including exact six-cell count, targeted LFS paths, SHA-256 verification, GPU requirement, output validation, and empty outputs.

- [ ] **Step 4: Commit the cleaned notebook**

```powershell
rtk git add -- notebooks/train_tactile_v3_public_colab.ipynb
rtk git commit -m "feat: make tactile Colab training self-contained"
```

### Task 5: Verify the full handoff and publish it

**Files:**
- Verify: all files changed in Tasks 1–4

- [ ] **Step 1: Run the focused suite**

```powershell
rtk python -m unittest tests.test_tactile_handoff tests.test_tactile_training_notebook tests.test_build_tactile_v3_dataset tests.test_train_tactile_v3 -v
```

Expected: all tests PASS.

- [ ] **Step 2: Validate notebook JSON and repository whitespace**

```powershell
rtk python -m json.tool notebooks/train_tactile_v3_public_colab.ipynb
rtk git diff --check
```

Expected: valid JSON and no whitespace errors.

- [ ] **Step 3: Confirm every required large object is in LFS**

```powershell
rtk git lfs ls-files
rtk git lfs fsck
```

Expected: the public ZIP, GuideTWSI checkpoint, dataset images, and candidate model are listed; `git lfs fsck` reports `Git LFS fsck OK`.

- [ ] **Step 4: Perform a clean-clone smoke check**

Use a fixed, validated temporary path and clone with LFS smudging disabled:

```powershell
rtk powershell -NoProfile -Command '$smokePath = "C:\Users\riyadh\Downloads\AI Camera\.handoff-smoke"; if (Test-Path -LiteralPath $smokePath) { throw "Refusing to overwrite existing smoke directory" }; $env:GIT_LFS_SKIP_SMUDGE = "1"; rtk git clone --branch codex/model-first-mobile-ready --single-branch https://github.com/Riqqi15/Yoloooo.git $smokePath'
rtk git -C "C:\Users\riyadh\Downloads\AI Camera\.handoff-smoke" lfs pull --include "models/guidetwsi/yolo11n_tactile.pt,artifacts/datasets/station-tactile-v2/**,artifacts/candidates/tactile-one-class-v3-public4-station1/**,data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip"
rtk powershell -NoProfile -Command '$smokePath = "C:\Users\riyadh\Downloads\AI Camera\.handoff-smoke"; if ((Get-Item -LiteralPath "$smokePath\data\public\guidetwsi-rbar-v1\guidetwsi-rbar-v1.zip").Length -ne 345332596) { throw "Public ZIP size mismatch" }; @("models\guidetwsi\yolo11n_tactile.pt", "artifacts\datasets\station-tactile-v2\tactile\images", "artifacts\candidates\tactile-one-class-v3-public4-station1\best.pt") | ForEach-Object { if (-not (Test-Path -LiteralPath "$smokePath\$_")) { throw "Missing smoke artifact: $_" } }'
```

Expected verification:

```text
data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip = 345332596 bytes
models/guidetwsi/yolo11n_tactile.pt exists
artifacts/datasets/station-tactile-v2/tactile/images exists
artifacts/candidates/tactile-one-class-v3-public4-station1/best.pt exists
```

Resolve and verify the exact cleanup target, then remove only that temporary clone:

```powershell
rtk powershell -NoProfile -Command '$smokePath = "C:\Users\riyadh\Downloads\AI Camera\.handoff-smoke"; $resolvedSmoke = (Resolve-Path -LiteralPath $smokePath).Path; if ($resolvedSmoke -ne "C:\Users\riyadh\Downloads\AI Camera\.handoff-smoke") { throw "Unsafe cleanup target: $resolvedSmoke" }; Remove-Item -LiteralPath $resolvedSmoke -Recurse -Force'
```

- [ ] **Step 5: Push commits and LFS objects**

```powershell
rtk git push origin codex/model-first-mobile-ready
rtk git lfs push origin codex/model-first-mobile-ready
```

Expected: Git and LFS uploads complete without error.

- [ ] **Step 6: Verify remote branch equality**

```powershell
rtk git fetch origin codex/model-first-mobile-ready
rtk git rev-parse HEAD
rtk git rev-parse origin/codex/model-first-mobile-ready
rtk git status --short
```

Expected: the two commit hashes match; only the unrelated pre-existing untracked `-C` entry may remain.
