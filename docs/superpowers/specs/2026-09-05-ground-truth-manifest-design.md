# Ground-Truth Manifest Design

## Goal

Create a human-review boundary between model predictions and evaluation truth. Seed one CSV row per test image from the existing batch report, then validate manually supplied presence labels and binary PNG masks before any accuracy metric is allowed.

## Commands

- `prepare`: reads `runs/sample-evaluation/report.json`, writes `data/ground_truth/manifest.csv`, and creates `data/ground_truth/masks/`. Existing manifest is never overwritten.
- `validate`: checks every source image, review state, presence label, and optional/required mask. Exit `0` only when all rows are valid and reviewed, `2` when rows remain unreviewed, and `1` for invalid data.

## Manifest Contract

Columns: `sample_id`, `source_path`, `dataset_role`, `prediction_tactile_present`, `prediction_mask_coverage`, `ground_truth_tactile_present`, `ground_truth_mask_path`, `review_status`, and `notes`.

All rows remain `test_only`. Prediction columns are reference hints, never ground truth. Reviewer sets `ground_truth_tactile_present` to `0` or `1` and `review_status` to `reviewed`. Positive rows require a non-empty binary PNG mask with source-image dimensions. Negative rows may omit a mask; any supplied mask must be empty.

## Safety

No predicted mask is copied into the ground-truth directory. Validation blocks evaluation on incomplete or malformed annotations. This phase computes no precision, recall, or IoU.

## Testing

Unit tests cover manifest seeding, incomplete review, valid positive masks, and invalid mask dimensions. Full regression and environment checks remain required.
