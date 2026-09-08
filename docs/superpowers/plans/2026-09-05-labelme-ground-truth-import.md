# Labelme Ground-Truth Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import human-drawn Labelme polygons into binary masks and the existing ground-truth manifest.

**Architecture:** A focused importer reuses manifest CSV helpers and validates every annotation before writing. Mask and manifest writes use temporary files followed by atomic replacement.

**Tech Stack:** Python 3.11, JSON/CSV standard library, OpenCV, NumPy, `unittest`.

---

### Task 1: Failing importer tests

- [x] Create `tests/test_import_labelme_ground_truth.py` covering positive polygons, negative empty shapes, and unknown-label rejection without manifest mutation.
- [x] Run focused tests and confirm missing-module failure.

### Task 2: Import implementation

- [x] Add atomic `replace_manifest` helper to `scripts/ground_truth_manifest.py`.
- [x] Create `scripts/import_labelme_ground_truth.py` with strict basename matching, dimension checks, label checks, polygon checks, reviewed-row protection, atomic mask writes, and incremental manifest updates.
- [x] Add CLI options `--annotations`, `--manifest`, and `--overwrite`.
- [x] Run focused and full tests.

### Task 3: Workflow documentation and verification

- [x] Update `data/ground_truth/README.md` with Labelme export/import commands and exact label name.
- [x] Create `data/ground_truth/labelme/README.md` without generating fake annotations.
- [x] Run importer against the empty annotation directory and confirm fail-safe rejection.
- [x] Run byte-compilation, environment check, dependency check, and complete test suite.

No Git step applies because workspace is not a Git repository.

## Execution Notes

- Import accepts only polygon shapes labeled exactly `tactile_paving`; empty shapes represent a reviewed negative image.
- All annotation files are validated before the manifest is replaced, and reviewed rows require explicit `--overwrite`.
- Empty annotation input fails safe without changing the manifest; SHA-256 remained unchanged.
- Final checks: 26 tests passed; byte-compilation, environment check, dependency check, and mask assertions passed.
