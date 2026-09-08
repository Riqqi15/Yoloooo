# Sample Batch Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate all local sample images with both existing baselines and generate fail-safe overlays plus JSON/CSV evidence.

**Architecture:** A single CLI reuses model loading, inference, annotation, and status helpers from both baseline scripts. Pure discovery/status helpers remain unit-testable without loading neural networks. One model pair is reused across the batch.

**Tech Stack:** Python 3.11 standard library, OpenCV, Ultralytics, existing baseline modules, `unittest`.

---

### Task 1: Add failing helper tests

**Files:**

- Create: `tests/test_evaluate_samples.py`

- [x] Write tests proving discovery includes JPG/WebP, ignores unrelated files, produces stable `.jpg` output names, and combined status never grants movement.
- [x] Run `rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_evaluate_samples.py -v` and confirm missing-module failure.

### Task 2: Implement batch evaluator

**Files:**

- Create: `scripts/evaluate_samples.py`

- [x] Add `discover_images`, `output_name`, and `combined_status` pure helpers.
- [x] Add CLI options for source/output directories, model paths, image size, confidence, and warmup.
- [x] Load both existing models once; warm them using first decodable image.
- [x] For every image, run one measured pass per model, write two overlays, and collect one report row.
- [x] Catch per-image errors as `STOP_FAILURE`; keep global model-contract errors fatal.
- [x] Write `report.json` with summary and rows, then write flattened `report.csv` using `csv.DictWriter`.

### Task 3: Verify code and evaluate samples

**Files:**

- Generate: `runs/sample-evaluation/object/*.jpg`
- Generate: `runs/sample-evaluation/tactile/*.jpg`
- Generate: `runs/sample-evaluation/report.json`
- Generate: `runs/sample-evaluation/report.csv`

- [x] Run focused tests, full tests, and CLI help.
- [x] Run evaluator against `data/samples` at image size 320 and confidence 0.25.
- [x] Assert report count equals discovered image count, every row is `test_only`, every combined status starts with `STOP_`, output files exist for successful rows, and timings are finite positive numbers.
- [x] Inspect representative JPG and WebP overlays.
- [x] Run byte-compilation, environment check, and `pip check`.

No Git step applies because workspace is not a Git repository.

## Execution Notes

- Evaluated 11 images: 7 JPG and 4 WebP; all decoded successfully.
- Generated 22 overlays plus JSON and CSV reports.
- Object status was `STOP_OBSTACLE` for all 11 images. Tactile model produced 8 instances across 8 images. These are unverified predictions until ground-truth masks exist.
- Final checks: 13 tests passed; byte-compilation, environment check, dependency check, and report assertions passed.
