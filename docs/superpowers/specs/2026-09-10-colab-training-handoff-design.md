# Colab Training Handoff Design

## Goal

Make the tactile-paving training work reproducible for a collaborator from a fresh Google Colab runtime, with every required input hosted in `Riqqi15/Yoloooo` and an accurate project handoff that Codex can discover immediately.

## Current truth

- The `public4-station1` candidate completed 80 epochs on a Colab T4.
- Its mask metrics are mAP50 `0.8250` and mAP50-95 `0.6979`.
- Protected evaluation rejected it: recall `1.0`, precision `0.6471`, mean IoU `0.5641`, and false positives on all 6 negative images.
- The local `public2-station1` run was stopped during epoch 5 and is not a completed candidate. A collaborator must train this candidate from the GuideTWSI checkpoint.
- The public cache contains 3,960 files in a 329.33 MiB ZIP and is licensed CC0/Public Domain.

## Repository layout

- `AGENTS.md`: short, automatically discovered Codex context and a pointer to the full handoff.
- `docs/HANDOFF_TACTILE_V3.md`: project overview, verified progress, artifact inventory, exact continuation checklist, and safety constraints.
- `data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip`: immutable public cache bundle stored with Git LFS.
- `data/public/guidetwsi-rbar-v1/sha256.txt`: checksum for the cache bundle.
- `notebooks/train_tactile_v3_public_colab.ipynb`: clean, output-free notebook that trains only the unfinished 2:1 candidate by default.

## Colab data flow

1. Install the pinned Ultralytics version and Git LFS when absent.
2. Clone `codex/model-first-mobile-ready` with LFS smudging disabled.
3. Pull only the initial checkpoint, station dataset, completed 4:1 candidate metadata/model, and public cache ZIP required for this workflow.
4. Verify the ZIP SHA-256 and its 3,960-file payload before extraction.
5. Build `tactile-v3-public2-station1` from the committed public manifest and station dataset.
6. Require an active CUDA GPU and train `tactile-one-class-v3-public2-station1` for up to 80 epochs with existing early stopping.
7. Validate the candidate bundle and download it as a ZIP.

No protected ground-truth image or label is read during training. Protected evaluation remains a separate local gate after the collaborator returns the candidate ZIP.

## Consistency rules

- The public ZIP is treated as immutable. Replacing it requires updating `sha256.txt`, provenance, manifest, and the documented file count together.
- The notebook contains no saved outputs or ad-hoc diagnostic cells.
- The notebook never retrains the completed 4:1 candidate unless a maintainer deliberately changes `TRAIN_RATIOS`.
- An interrupted run directory is not represented as a completed candidate.
- A candidate is never described as mobile-ready until protected evaluation passes.

## Verification

- Static notebook contract tests cover the Git LFS include list, checksum verification, file count, selected ratio, GPU requirement, packaging, and absence of protected-data paths.
- Existing dataset-builder and training-configuration tests continue to pass.
- A clean-clone smoke test confirms every referenced tracked file and Git LFS object exists.
- The pushed branch and LFS upload are verified against `origin/codex/model-first-mobile-ready`.

## Out of scope

- Mobile export and application integration.
- Training people, obstacle, hole, platform-edge, or train-door detectors.
- Declaring the one-class tactile model safe based only on training metrics.
