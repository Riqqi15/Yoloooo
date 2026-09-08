# Tactile Accuracy Gate Design

## Goal

Preserve source-resolution binary prediction masks during sample evaluation and compute tactile presence/segmentation metrics only after human ground truth passes validation.

## Components

- `baseline_tactile_inference.rasterize_masks`: converts Ultralytics original-coordinate mask polygons into a binary 0/255 mask matching the source image.
- `evaluate_samples.py`: saves one prediction mask per successful image under `runs/sample-evaluation/tactile_masks/` and records its path.
- `evaluate_ground_truth.py`: validates the manifest, matches rows by source path, then calculates TP, FP, TN, FN, accuracy, precision, recall, F1, and mean IoU for positive ground-truth rows.

## Gates

Accuracy evaluation refuses incomplete or invalid manifests, missing prediction rows, malformed prediction masks, or source mismatches. Current unreviewed manifest must produce `STOP_FAILURE` and no accuracy claim.

## Testing

Tests cover source-resolution polygon rasterization, presence confusion metrics, binary IoU, and incomplete-manifest rejection. Generated prediction masks must be binary and match each source image.
