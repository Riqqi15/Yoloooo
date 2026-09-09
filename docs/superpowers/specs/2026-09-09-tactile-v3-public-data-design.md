# Tactile v3 Public-Data Design

## Objective

Raise tactile-paving segmentation recall to at least `0.90` and positive-mask mean IoU to at least `0.75` before any mobile work resumes. The design addresses the observed regression from the GuideTWSI baseline (`0.8182` protected recall) to the 45-image station fine-tune (`0.7273`) by preventing catastrophic forgetting and increasing real-world visual diversity.

## Scope

This phase covers public-data acquisition, provenance, curation, split isolation, candidate-v3 training, and offline evaluation. Mobile export, Android integration, walking directions, obstacle fusion, and safety authorization are explicitly out of scope.

## Public sources

The primary source is `guidedogrobot/guidetwsi`, specifically its real directional-bar subset `RBar-22K`. GuideTWSI documents 19,925 curated mask-annotated real images and supplies normalized YOLO polygon annotations with class `0 = tactile_paving`. Kaggle currently lists the complete 45 GB release as CC0; the upstream repository and every retained source record remain cited because RBar is compiled from several prior datasets.

Roboflow sets such as Station Detect, `b_bv3`, and `tactile` are not downloaded separately in the first pass because GuideTWSI already incorporates those community sources. This avoids duplicate images and split leakage.

## Acquisition and storage

- Cap the first acquisition at 5 GB.
- Download only files needed from the original GuideTWSI `RBar-22K` train split.
- Store raw public files under an ignored local cache; never commit or push them to GitHub.
- Commit only a compact provenance manifest containing source dataset, upstream path, license, content hash, selected role, and local relative cache key.
- Abort before exceeding the cap, on checksum failure, or when source licensing/provenance is absent.

## Sampling and curation

Build an initial public subset of up to 2,000 distinct training images. Sample across upstream source groups rather than taking adjacent filenames, then remove exact duplicates and perceptual near-duplicates against both the public subset and the existing 69-image station dataset.

Retain tactile instances that represent directional bars, warning dots, intersections, distant/thin paths, partial occlusion, and varied colors or lighting. Exclude corrupt images, missing/invalid polygons, unrelated tactile-sensor imagery, and non-walking surfaces. Existing protected ground truth remains outside all training and threshold selection.

## Dataset composition

Candidate v3 trains on two components:

1. GuideTWSI public train samples for broad tactile appearance retention.
2. Existing reviewed station train samples, oversampled during training so station imagery is not overwhelmed by the larger public component.

The first experiment uses one source-group-safe validation set and no protected-set access during tuning. Augmentation may simulate blur, lower exposure, mild perspective changes, compression, and partial occlusion, but may not change geometry labels or manufacture new test samples.

## Training strategy

Initialize from the existing GuideTWSI YOLO11n segmentation checkpoint, not from the regressed v2 checkpoint. Compare a minimal set of mixtures rather than sweeping many hyperparameters:

- public-to-station sampling ratio `4:1`;
- public-to-station sampling ratio `2:1`;
- one frozen-backbone warm-up followed by full fine-tuning if the unfrozen run still forgets the baseline.

Select by validation recall first, then mean IoU, then false-positive rate. Epoch count is controlled by early stopping; extra epochs are not treated as a substitute for data diversity.

## Evaluation gates

- The current 17-image protected set is not used for threshold tuning.
- Lock confidence on validation before evaluating a candidate on protected data.
- Candidate acceptance requires protected recall `>= 0.90`, positive-mask mean IoU `>= 0.75`, and no increase in protected false-positive rate over the current `0.50` baseline.
- A passing result remains an offline dataset result, not evidence that walking guidance is safe.
- If v3 fails, use validation failure categories to choose the next data slice; do not repeatedly tune against protected images.

## Verification and outputs

The phase produces a public-data provenance manifest, duplicate/leakage report, immutable training manifest, candidate-v3 bundle, and evaluation report. Every retained image and annotation is hash-bound. Candidate packaging retains the pinned training configuration, environment, metrics, model card, and checkpoint checksum.

Training or evaluation failures stop closed and preserve the prior validated artifacts. The v2 candidate and its reports are never overwritten.
