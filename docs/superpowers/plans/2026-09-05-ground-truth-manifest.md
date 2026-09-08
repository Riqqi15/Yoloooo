# Ground-Truth Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare and validate human-authored tactile-paving ground truth for all test images without contaminating it with model predictions.

**Architecture:** One standard-library CLI owns CSV preparation and validation. It reuses `output_name` for stable mask names and OpenCV only for image/mask decoding and dimension checks.

**Tech Stack:** Python 3.11, `argparse`, `csv`, `json`, OpenCV, NumPy, `unittest`.

---

### Task 1: Contract tests

**Files:**

- Create: `tests/test_ground_truth_manifest.py`

- [x] Add tests for `build_rows(report, manifest_path)` using a one-image report and assert prediction fields remain separate from blank ground-truth fields.
- [x] Add validation tests for unreviewed rows, a valid positive binary mask, and a positive mask with wrong dimensions.
- [x] Run focused tests and confirm failure because `ground_truth_manifest` does not exist.

### Task 2: CLI implementation

**Files:**

- Create: `scripts/ground_truth_manifest.py`

- [x] Define fixed CSV fields and `build_rows` using `output_name(source)` for `masks/<stable-name>.jpg` converted to `.png`.
- [x] Implement CSV read/write; refuse to overwrite an existing manifest.
- [x] Implement validation: source must decode, role must be `test_only`, review must be `unreviewed` or `reviewed`, reviewed presence must be `0` or `1`, positive mask must exist and be nonempty binary 0/255 with matching dimensions, negative mask must be absent or empty.
- [x] Implement `prepare` and `validate` subcommands with exit codes `0`, `2`, and `1` as specified.

### Task 3: Human instructions and generated manifest

**Files:**

- Create: `data/ground_truth/README.md`
- Generate: `data/ground_truth/manifest.csv`

- [x] Document exact editing workflow and mask format without recommending model masks as truth.
- [x] Run `prepare` against current batch report and confirm 11 unreviewed `test_only` rows.
- [x] Run `validate`; expect exit `2`, zero invalid rows, and 11 incomplete rows.

### Task 4: Verification

- [x] Run full tests and byte-compilation.
- [x] Run environment check and `pip check`.
- [x] Confirm manifest sources cover exactly the 11 batch-report images and no ground-truth mask was auto-generated.

No Git step applies because workspace is not a Git repository.

## Execution Notes

- Prepared 11 `test_only` rows from the batch report; all remain `unreviewed` with blank ground-truth labels.
- Created no masks. Prediction columns remain reference hints only.
- Validator reports 11 incomplete rows, zero invalid rows, and child exit code `2`. RTK normalizes nonzero child exit codes in its own command result, so the child code was verified through PowerShell.
- Final checks: 18 tests passed; byte-compilation, environment check, dependency check, and manifest coverage assertions passed.
