# Tactile Inference Baseline Design

## Scope

Build a second local CPU CLI for GuideTWSI `yolo11n_tactile.pt`. Keep tactile segmentation independent from the existing COCO object detector so each model can be benchmarked and rejected separately. Download only the 5.71 MB checkpoint; do not download the 39.5K-image GuideTWSI dataset.

This phase measures segmentation output. It does not fuse models, infer walking direction, activate tactile guidance, or grant permission to proceed.

## Upstream Contract and License Boundary

The current GuideTWSI repository identifies `yolo11n_tactile.pt` as a YOLOv11-Seg-N model trained on GuideTWSI and licenses the repository under MIT. Its checked-in `yolov11_seg_n.yaml` defines one class only:

```yaml
num_classes: 1
class_names: ["tactile_paving"]
```

This differs from the local future training taxonomy of `tactile_guiding_path` and `tactile_warning_block`. The baseline must preserve the checkpoint's actual `tactile_paving` label and must not invent directional-bar versus warning-dome classification. A later two-class fine-tune requires its own dataset and evaluation.

GuideTWSI's MIT repository license does not erase Ultralytics YOLO model/runtime obligations. Keep the existing AGPL-3.0/Enterprise warning. This local evaluation makes no distribution or commercial-use claim.

## Files

- Create `scripts/baseline_tactile_inference.py` for checkpoint download, validation, segmentation inference, annotation, and benchmark output.
- Create `tests/test_baseline_tactile_inference.py` for deterministic checkpoint-contract and mask-summary checks.
- Download `models/guidetwsi/yolo11n_tactile.pt` through `huggingface_hub.hf_hub_download` from `guidedogrobot-tactile/GuideTWSI-weights`.
- Generate outputs under `runs/baseline/tactile/`.

Reuse `parse_source`, `predict_frame`, `timing_summary`, and `write_metrics` from `scripts/baseline_object_inference.py`. Do not add a framework, shared abstraction layer, or dependency.

## CLI

`scripts/baseline_tactile_inference.py` accepts:

- `--source`: one image or video path.
- `--model`: checkpoint destination, default `models/guidetwsi/yolo11n_tactile.pt`.
- `--imgsz`: default `320`.
- `--confidence`: default `0.25`.
- `--warmup`: default `15`.
- `--runs`: default `30` for static images.
- `--output`: annotated image/video destination.
- `--metrics`: JSON destination.

Webcam support is not added in this phase. The object CLI already has unverified webcam support, while the tactile milestone needs reproducible image/video evidence first.

## Checkpoint Validation

After loading with `YOLO`, require:

- `model.task == "segment"`.
- Normalized class mapping equals `{0: "tactile_paving"}`.
- Model file exists and has a recorded SHA-256 checksum.

Any mismatch exits nonzero with `STOP_FAILURE`. Do not continue with guessed labels or a detect-only checkpoint.

## Processing and Output

Process one frame at a time with `device="cpu"`. For each result:

1. Keep Ultralytics mask overlay from `result.plot()`.
2. Count instances and classes.
3. Combine instance masks into one binary mask for measurement only.
4. Report predicted-mask coverage as a fraction of model-output pixels.
5. Overlay `STOP_UNCERTAIN` and `TACTILE CANDIDATE ONLY - NOT WALKING GUIDANCE`.

No mask also produces `STOP_UNCERTAIN`; absence of tactile detection is not proof that no tactile path exists. Model load, source decode, inference, or output failure produces `STOP_FAILURE`.

Static images use 15 excluded warm-ups and at least 30 measured calls. Videos process sequentially; initial warm-up frames are excluded from metrics. No background queue is introduced.

## Metrics

JSON contains:

- Model repository, path, SHA-256, task, and exact labels.
- Device, input size, confidence, source kind, and warm-up count.
- Mean, p50, p95, maximum wall-clock `model.predict` latency, effective FPS, and sampled peak process RSS.
- Ultralytics preprocess/inference/postprocess timing when available.
- Instance count, class counts, frames with masks, and mean/max mask coverage.
- A warning that output is a single-model baseline and cannot authorize movement.

No image bytes, identity metadata, credentials, or precise location enter metrics.

## Verification

- Red/green unit tests reject wrong task and wrong label mapping.
- Mask-summary test handles both no-mask and synthetic-mask results.
- Environment check and `pip check` remain green.
- CLI help exits zero.
- Checkpoint downloads once, loads under installed Ultralytics 8.4.138, and passes the exact metadata contract.
- Static image smoke run produces annotated output and 30 finite timing samples.
- User-provided files in `data/samples/` remain test-only; after implementation, run them as an evaluation batch without moving or relabeling them for training.

## Non-Goals

- No full GuideTWSI dataset download.
- No training, two-class remapping, or Colab notebook.
- No object/tactile fusion, centerline extraction, multi-frame stability, direction decision, watchdog, depth, TTS, haptics, or Android integration.
- No claim that predicted `tactile_paving` distinguishes guiding bars from warning domes or is reliable in Indonesian stations.
