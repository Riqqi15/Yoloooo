# Labelme Ground-Truth Import Design

## Goal

Turn independently drawn Labelme polygon annotations into validated tactile ground-truth masks and manifest reviews without using model predictions as labels.

## Contract

Importer reads JSON files from `data/ground_truth/labelme/`. Each JSON matches a manifest row by source-image basename. Accepted shapes use label `tactile_paving` and polygon geometry only. One or more polygons mean positive presence; zero shapes mean reviewed negative.

All JSON files are validated before any manifest update. Masks are source-sized binary PNG files. Manifest replacement is atomic. Reviewed rows are protected unless `--overwrite` is explicitly passed. Partial annotation batches are allowed; remaining rows stay unreviewed and keep accuracy gate closed.

## Safety

Unknown labels, invalid polygons, mismatched dimensions, duplicate source annotations, and unrecognized images abort import. Prediction overlays and prediction masks are never read by importer.

## Testing

Tests cover positive polygon import, negative empty-shape import, unknown-label rejection, mask shape/binariness, and preservation of manifest data after rejected imports.
