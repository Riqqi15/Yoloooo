# Station Dataset Scope Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject dataset drift outside the approved station environment before inference.

**Architecture:** Keep one JSON allowlist beside the data. Add one pure validator to the existing batch evaluator and include approved context metadata in reports.

**Tech Stack:** Python 3.11 standard-library JSON, `unittest`.

---

### Task 1: Scope registry and failing tests

- [x] Create `data/dataset_scope.json` with ten active station images and approved contexts, including `station_access_sidewalk`.
- [x] Add tests for a valid registry, unregistered files, missing files, wrong scope, and invalid context.
- [x] Run focused tests and confirm the validator is missing.

### Task 2: Fail-safe evaluator gate

- [x] Add `load_station_scope` to `scripts/evaluate_samples.py`.
- [x] Add `--scope-file` and validate before model loading.
- [x] Record `dataset_scope` and `station_context` in JSON and CSV outputs.
- [x] Run focused tests.

### Task 3: Documentation and verification

- [x] Document inclusion and exclusion rules in `README.md`.
- [x] Run all tests and byte-compilation.
- [x] Rerun the real batch and ground-truth evaluation.
- [x] Confirm ten active station images and zero bus image records in active reports.

No Git step applies because workspace is not a Git repository.

## Execution evidence

- Focused test command uses unittest discovery because `tests` is not a Python package. Confirmed missing-validator ImportError before implementation.
- 33 tests passed; byte-compilation passed. Coverage includes rejection before either model loads and refusal to expand allowed contexts to bus-only scenes.
- Real batch: 10/10 successful; report JSON/CSV includes station context, including the station-access sidewalk. Bus source remains archived.
- Existing ground-truth format validation and metric computation passed; semantic annotation quality remains provisional and requires visual QA.
- Kept implementation inside the existing evaluator, with standard-library validation and no new dependency. Training pipeline is not part of this completed scope gate.
