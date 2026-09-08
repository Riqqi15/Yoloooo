# Training Label Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build fail-closed validation, visual QA, review attestation, and YOLO export for station training labels.

**Architecture:** Add one focused Python CLI using existing JSON/CSV/OpenCV helpers. Keep human review separate from annotation content and bind approval to source plus annotation SHA-256. Export through a temporary directory only after full validation.

**Tech Stack:** Python 3.11 standard library, OpenCV, NumPy, unittest.

---

### Task 1: Annotation and review contract

**Files:**
- Create: `scripts/training_label_gate.py`
- Create: `tests/test_training_label_gate.py`

- [x] Add fixtures for one intake image, approved taxonomy, annotation JSON, and empty review CSV.
- [x] Test rejection of missing review, stale annotation hash, unknown class, invalid box, invalid polygon, and protected test content.
- [x] Implement manifest/taxonomy/annotation parsing and fail-closed validation.
- [x] Run focused unittest discovery; all gate tests pass.

### Task 2: Visual QA and review attestation

**Files:**
- Modify: `scripts/training_label_gate.py`
- Modify: `tests/test_training_label_gate.py`

- [x] Test overlay keeps source dimensions and draws object/polygon boundaries.
- [x] Implement `render`, `review`, and `ambiguous`; review rows bind current source and annotation hashes.
- [x] Test annotation modification invalidates prior review.
- [x] Run focused tests; all pass.

### Task 3: Atomic YOLO export and report

**Files:**
- Modify: `scripts/training_label_gate.py`
- Modify: `tests/test_training_label_gate.py`

- [x] Test exact normalized detection/segmentation label output.
- [x] Test export refuses draft taxonomy and leaves no final/temporary directory on injected failure.
- [x] Implement per-task split folders, YAML metadata, QA report, and staging-directory commit.
- [x] Run focused tests; all pass.

### Task 4: Documentation and real-state verification

**Files:**
- Create: `data/training/annotations/README.md`
- Create: `data/training/review_manifest.csv`
- Modify: `data/training/README.md`
- Modify: `training/01-dataset-and-labeling.md`
- Modify: `docs/AUDIT_MISMATCH_DAN_ROADMAP.md`
- Modify: `NEW_CHAT_CONTEXT.md`

- [x] Document annotation schema and ordered human-review workflow.
- [x] Preserve empty training intake and explicitly avoid claiming a training dataset.
- [x] Run compileall, full unittest suite, pip check, and existing 10-image test batch: 56 tests pass, dependencies healthy, batch 10/10 succeeds.
- [x] Mark only implemented roadmap items complete. No Git commit: workspace is not a Git repository.
