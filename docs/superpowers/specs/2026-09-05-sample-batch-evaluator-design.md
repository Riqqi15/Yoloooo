# Sample Batch Evaluator Design

## Goal

Evaluate every decodable image in `data/samples` with the existing object and GuideTWSI tactile baselines. Produce per-model overlays plus machine-readable JSON and CSV reports without treating the images as training data.

## Scope

- Accept BMP, JPEG, JPG, PNG, and WebP files from one directory.
- Load object and tactile models once, validate the tactile checkpoint contract, and warm both models on the first valid image.
- Run one measured object pass and one measured tactile pass per image.
- Save object overlays under `runs/sample-evaluation/object/` and tactile overlays under `runs/sample-evaluation/tactile/`.
- Save `report.json` and `report.csv` with detections, mask coverage, latency, output paths, and fail-safe status.
- Continue after a per-image decode or write failure and record `STOP_FAILURE` for that image.
- Abort on global failures such as missing models or invalid checkpoint metadata.

## Safety Contract

`combined_status` may only be `STOP_OBSTACLE`, `STOP_UNCERTAIN`, or `STOP_FAILURE`. Tactile output remains a binary candidate; checkpoint label `braille` is preserved and never interpreted as guiding-path versus warning-block. Report declares `dataset_role: test_only` and never grants movement permission.

## Testing

Standard-library unit tests cover deterministic image discovery, WebP inclusion, stable output names, and combined status behavior. Final verification runs all tests, byte-compilation, environment checks, and validates that the generated report covers all discovered samples with no non-STOP status.
