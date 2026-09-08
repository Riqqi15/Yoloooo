# Tactile Accuracy Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save raw tactile prediction masks and expose accuracy metrics only when human ground truth is complete and valid.

**Architecture:** Reuse existing inference and manifest validators. Rasterize original-coordinate polygons to avoid letterbox distortion; keep prediction artifacts separate from ground truth.

**Tech Stack:** Python 3.11, OpenCV, NumPy, standard-library JSON/argparse, `unittest`.

---

### Task 1: Add failing mask and metric tests

**Files:**

- Modify: `tests/test_baseline_tactile_inference.py`
- Create: `tests/test_evaluate_ground_truth.py`

- [x] Test polygon rasterization returns source-sized binary 0/255 output and handles no masks.
- [x] Test TP/FP/TN/FN, precision, recall, F1, accuracy, binary IoU, and incomplete-manifest rejection.
- [x] Run focused tests and confirm missing functions/module failures.

### Task 2: Save prediction masks

**Files:**

- Modify: `scripts/baseline_tactile_inference.py`
- Modify: `scripts/evaluate_samples.py`

- [x] Implement `rasterize_masks(result, height, width)` using `result.masks.xy` and `cv2.fillPoly`.
- [x] Add `tactile_prediction_mask` to CSV/report rows and save one binary PNG per successful image.
- [x] Rerun sample evaluation and assert 11 prediction masks are binary with source dimensions.

### Task 3: Accuracy evaluator

**Files:**

- Create: `scripts/evaluate_ground_truth.py`

- [x] Implement `presence_metrics`, `binary_iou`, and `require_complete_manifest`.
- [x] Validate all manifest rows before matching report predictions by `source_path`.
- [x] Compute per-image presence outcome and IoU for positive ground-truth rows; write JSON only after every gate passes.
- [x] Run against current manifest and confirm fail-safe rejection because 11 rows remain unreviewed.

### Task 4: Final verification

- [x] Run all tests, byte-compilation, environment check, and `pip check`.
- [x] Confirm report still covers 11 test images and all generated prediction masks pass format checks.
- [x] Record current annotation blocker without reporting accuracy values.

No Git step applies because workspace is not a Git repository.

## Execution Notes

- Generated 11 source-resolution binary prediction masks under `runs/sample-evaluation/tactile_masks/`; 8 are nonempty.
- Accuracy evaluator computes presence confusion metrics and positive-sample IoU only after ground-truth validation passes.
- Current run fails safe with `STOP_FAILURE: manifest incomplete: 11 rows` and writes no accuracy output.
- Final checks: 26 tests passed; byte-compilation, environment check, dependency check, and prediction-mask assertions passed.
