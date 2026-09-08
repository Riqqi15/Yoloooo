# Object Inference Baseline Design

## Scope

Build one local CLI for pretrained YOLO11n object detection on CPU. It accepts an image, video, or webcam, identifies detections whose bottom-center footpoint falls inside a configurable walking corridor, produces visual output, and reports latency and memory metrics. It downloads only the small pretrained model when missing; no dataset is downloaded.

This baseline measures perception only. It is not a navigation system, does not activate walking guidance, and never reports that a route is safe.

## Interface

Create `scripts/baseline_object_inference.py` with these inputs:

- `--source`: image path, video path, or webcam index.
- `--model`: model path/name, default `models/yolo11n.pt`.
- `--imgsz`: inference size, default `320`.
- `--confidence`: COCO detection threshold, default `0.25`.
- `--warmup`: warm-up iterations, default `15`.
- `--runs`: measured iterations for static-image benchmarking, default `30`.
- `--output`: annotated media destination under `runs/baseline/`.
- `--metrics`: JSON metrics destination under `runs/baseline/`.

The CLI creates parent output directories when required. Invalid or unreadable sources exit nonzero after printing `STOP_FAILURE`.

## Processing Flow

1. Load YOLO11n once on CPU.
2. Decode one image or stream frames sequentially.
3. Keep one frame in flight; no frame queue or background capture is introduced.
4. Run inference at 320 by default.
5. For each detection, calculate bottom-center footpoint.
6. Test footpoint against a normalized trapezoid representing the forward walking corridor.
7. Draw every detection, but visually emphasize only corridor detections.
8. Derive a fail-safe baseline status.
9. Write annotated output and final benchmark JSON.

For static images, warm-up runs are excluded and at least 30 measured iterations use the same decoded image. For video/webcam, initial frames are warm-up and subsequent processed frames are measured until EOF or user stop.

## Corridor Geometry

Use a centered trapezoid expressed as normalized coordinates so behavior is resolution-independent:

- Top-left: `(0.40, 0.45)`
- Top-right: `(0.60, 0.45)`
- Bottom-right: `(0.90, 1.00)`
- Bottom-left: `(0.10, 1.00)`

OpenCV polygon testing determines inclusion. This is an initial demo calibration knob, not a physical clearance guarantee. Side detections remain visible but do not become primary obstacle alerts.

## Fail-Safe Status

- `STOP_OBSTACLE`: at least one confident detection has a footpoint inside the corridor.
- `STOP_UNCERTAIN`: processing succeeded but no corridor obstacle was detected; absence of detection is not evidence of safety.
- `STOP_FAILURE`: model load, source decode, inference, or output writing failed.

Status affects only baseline logs and overlay. The script provides no TTS, haptic feedback, or permission to proceed.

## Measurements

Measure wall-clock latency around the complete `model.predict` call using `time.perf_counter_ns()`. Report warm-up count, measured sample count, mean, p50, p95, maximum latency, effective FPS (`1000 / mean_ms`), and peak process RSS sampled after each inference. Also preserve Ultralytics-reported preprocess, inference, and postprocess means when available.

Metrics JSON records model name, device, input size, source kind, thresholds, corridor coordinates, status counts, detection counts, and timing summary. It stores no image bytes, precise location, credentials, or identity metadata.

## Verification

- Geometry self-check: center footpoint is inside; far-left and far-right footpoints are outside.
- Environment check and `pip check` remain green.
- CLI help exits successfully.
- Static-image smoke run loads YOLO11n, completes warm-up plus 30 measured runs, writes annotated output and JSON, and reports all required statistics.
- JSON values are checked for finite, positive timing and correct sample count.
- Side-object filtering is verified with deterministic synthetic boxes; no dangerous physical scene is created.

## Non-Goals

- No dataset download or model training.
- No tactile segmentation, depth, decision engine, watchdog, TTS, haptics, or Flutter integration.
- No claim that COCO detections cover holes, drop-offs, glass, cables, stairs, or other critical hazards.
- No parallel inference, frame buffering, new dependency, or custom model abstraction.
