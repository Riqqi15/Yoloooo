# Object Inference Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a fail-safe local YOLO11n CPU inference CLI for images, videos, and webcams, including corridor filtering and benchmark metrics.

**Architecture:** One Python CLI owns model loading, sequential inference, corridor classification, annotation, output writing, and metrics. Pure geometry/statistics helpers stay importable for one small standard-library test module. No queue, background worker, dataset, training code, or safety-guidance claim is added.

**Tech Stack:** Python 3.11, Ultralytics 8.4.138, PyTorch CPU 2.14.0, OpenCV 5.0.0, psutil 7.2.2, `unittest`.

---

## File Map

- Create `scripts/baseline_object_inference.py`: CLI, model download/load, inference, corridor filter, annotation, output, metrics, fail-safe exit behavior.
- Create `tests/test_baseline_object_inference.py`: deterministic tests for corridor geometry, status selection, and percentile calculation.
- Download `models/yolo11n.pt`: small official pretrained checkpoint; ignored as a large binary artifact.
- Download `data/samples/bus.jpg`: small official smoke-test image; ignored as local data.
- Generate `runs/baseline/bus_annotated.jpg`: smoke-test visualization.
- Generate `runs/baseline/bus_metrics.json`: benchmark evidence.

This workspace has no `.git` directory. Do not create a repository, worktree, branch, or commit.

### Task 1: Add deterministic corridor and timing tests

**Files:**

- Create: `tests/test_baseline_object_inference.py`
- Test: `tests/test_baseline_object_inference.py`

- [ ] **Step 1: Write failing tests for pure behavior**

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_object_inference import (  # noqa: E402
    STOP_OBSTACLE,
    STOP_UNCERTAIN,
    box_in_corridor,
    corridor_polygon,
    frame_status,
    percentile,
)


class BaselineObjectInferenceTest(unittest.TestCase):
    def test_corridor_includes_center_and_excludes_sides(self) -> None:
        polygon = corridor_polygon(1000, 1000)
        self.assertTrue(box_in_corridor((450, 600, 550, 900), polygon))
        self.assertFalse(box_in_corridor((0, 600, 100, 900), polygon))
        self.assertFalse(box_in_corridor((900, 600, 999, 900), polygon))

    def test_frame_status_is_always_stop(self) -> None:
        self.assertEqual(frame_status(1), STOP_OBSTACLE)
        self.assertEqual(frame_status(0), STOP_UNCERTAIN)

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test and confirm missing module failure**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_baseline_object_inference.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'baseline_object_inference'`.

### Task 2: Implement pure fail-safe helpers

**Files:**

- Create: `scripts/baseline_object_inference.py`
- Test: `tests/test_baseline_object_inference.py`

- [ ] **Step 1: Add imports, constants, corridor geometry, status, and percentile**

```python
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import psutil
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

STOP_OBSTACLE = "STOP_OBSTACLE"
STOP_UNCERTAIN = "STOP_UNCERTAIN"
STOP_FAILURE = "STOP_FAILURE"
CORRIDOR_NORMALIZED = (
    (0.40, 0.45),
    (0.60, 0.45),
    (0.90, 1.00),
    (0.10, 1.00),
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def corridor_polygon(width: int, height: int) -> Any:
    import numpy as np

    return np.array(
        [[round(x * (width - 1)), round(y * (height - 1))] for x, y in CORRIDOR_NORMALIZED],
        dtype=np.int32,
    )


def box_in_corridor(box: Sequence[float], polygon: Any) -> bool:
    x1, _y1, x2, y2 = box
    footpoint = ((float(x1) + float(x2)) / 2.0, float(y2))
    return cv2.pointPolygonTest(polygon, footpoint, False) >= 0


def frame_status(corridor_detection_count: int) -> str:
    return STOP_OBSTACLE if corridor_detection_count else STOP_UNCERTAIN


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
```

- [ ] **Step 2: Run pure tests**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_baseline_object_inference.py -v
```

Expected: 3 tests PASS.

### Task 3: Complete single-file inference CLI

**Files:**

- Modify: `scripts/baseline_object_inference.py`

- [ ] **Step 1: Add argument parsing and source classification**

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-safe YOLO11n CPU baseline. Output never grants permission to proceed."
    )
    parser.add_argument("--source", required=True, help="Image/video path or webcam index")
    parser.add_argument("--model", default="models/yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--runs", type=int, default=30, help="Measured runs for a static image")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics", type=Path, default=Path("runs/baseline/metrics.json"))
    args = parser.parse_args()
    if args.imgsz <= 0 or args.warmup < 0 or args.runs < 1 or not 0.0 <= args.confidence <= 1.0:
        parser.error("imgsz/runs must be positive, warmup nonnegative, confidence within [0, 1]")
    return args


def parse_source(value: str) -> tuple[str, str | int]:
    if value.isdecimal():
        return "webcam", int(value)
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"source not found: {path}")
    return ("image" if path.suffix.lower() in IMAGE_SUFFIXES else "video"), str(path)


def default_output(source_kind: str) -> Path:
    suffix = ".jpg" if source_kind == "image" else ".mp4"
    return Path("runs/baseline") / f"annotated{suffix}"
```

- [ ] **Step 2: Add model loading and one-frame prediction**

```python
def load_model(model_path: str) -> YOLO:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = Path(attempt_download_asset(path))
    if not resolved.is_file():
        raise FileNotFoundError(f"model unavailable after download attempt: {path}")
    return YOLO(str(resolved), task="detect")


def predict_frame(model: YOLO, frame: Any, imgsz: int, confidence: float) -> tuple[Any, float, int]:
    started_ns = time.perf_counter_ns()
    result = model.predict(frame, imgsz=imgsz, conf=confidence, device="cpu", verbose=False)[0]
    latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return result, latency_ms, psutil.Process().memory_info().rss


def analyze_result(result: Any, frame: Any) -> tuple[Any, str, int, int]:
    height, width = frame.shape[:2]
    polygon = corridor_polygon(width, height)
    annotated = result.plot()
    corridor_count = 0
    boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().tolist()
    for box in boxes:
        if box_in_corridor(box, polygon):
            corridor_count += 1
            x1, y1, x2, y2 = (round(value) for value in box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
    status = frame_status(corridor_count)
    cv2.polylines(annotated, [polygon], True, (0, 255, 255), 2)
    cv2.putText(annotated, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(
        annotated,
        "BASELINE ONLY - NOT WALKING GUIDANCE",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    return annotated, status, len(boxes), corridor_count
```

- [ ] **Step 3: Add metrics summarization and JSON writing**

```python
def timing_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("no measured inference samples")
    mean_ms = statistics.fmean(values)
    return {
        "samples": len(values),
        "mean_ms": mean_ms,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": max(values),
        "effective_fps": 1000.0 / mean_ms,
    }


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 4: Add static-image processing**

```python
def process_image(
    model: YOLO,
    source: str,
    output: Path,
    imgsz: int,
    confidence: float,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    frame = cv2.imread(source)
    if frame is None:
        raise ValueError(f"unable to decode image: {source}")
    for _ in range(warmup):
        predict_frame(model, frame, imgsz, confidence)
    latencies: list[float] = []
    rss_samples: list[int] = []
    final_result = None
    for _ in range(runs):
        final_result, latency_ms, rss = predict_frame(model, frame, imgsz, confidence)
        latencies.append(latency_ms)
        rss_samples.append(rss)
    annotated, status, object_count, corridor_count = analyze_result(final_result, frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise OSError(f"unable to write image: {output}")
    return {
        "timing": timing_summary(latencies),
        "peak_rss_mb": max(rss_samples) / (1024 * 1024),
        "status_counts": {status: 1},
        "object_count": object_count,
        "corridor_object_count": corridor_count,
        "ultralytics_speed_ms": final_result.speed,
    }
```

- [ ] **Step 5: Add sequential video/webcam processing**

```python
def process_stream(
    model: YOLO,
    source: str | int,
    output: Path,
    imgsz: int,
    confidence: float,
    warmup: int,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"unable to open stream: {source}")
    writer = None
    latencies: list[float] = []
    rss_samples: list[int] = []
    statuses: Counter[str] = Counter()
    object_count = 0
    corridor_count = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result, latency_ms, rss = predict_frame(model, frame, imgsz, confidence)
            annotated, status, detected, relevant = analyze_result(result, frame)
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                height, width = annotated.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS)
                fps = fps if math.isfinite(fps) and fps > 0 else 30.0
                writer = cv2.VideoWriter(
                    str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise OSError(f"unable to create video: {output}")
            writer.write(annotated)
            if frame_index >= warmup:
                latencies.append(latency_ms)
                rss_samples.append(rss)
                statuses[status] += 1
                object_count += detected
                corridor_count += relevant
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frame_index <= warmup:
        raise ValueError("stream ended before measured frames were available")
    return {
        "timing": timing_summary(latencies),
        "peak_rss_mb": max(rss_samples) / (1024 * 1024),
        "status_counts": dict(statuses),
        "object_count": object_count,
        "corridor_object_count": corridor_count,
    }
```

- [ ] **Step 6: Add fail-safe main entrypoint**

```python
def run(args: argparse.Namespace) -> int:
    source_kind, source = parse_source(args.source)
    output = args.output or default_output(source_kind)
    model = load_model(args.model)
    if source_kind == "image":
        result = process_image(
            model, str(source), output, args.imgsz, args.confidence, args.warmup, args.runs
        )
    else:
        result = process_stream(model, source, output, args.imgsz, args.confidence, args.warmup)
    payload = {
        "warning": "Baseline only. No result grants permission to proceed.",
        "model": args.model,
        "device": "cpu",
        "source_kind": source_kind,
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "warmup": args.warmup,
        "corridor_normalized": CORRIDOR_NORMALIZED,
        "output": str(output),
        **result,
    }
    write_metrics(args.metrics, payload)
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        print(f"{STOP_FAILURE}: interrupted", file=sys.stderr)
    except Exception as error:
        print(f"{STOP_FAILURE}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run tests and CLI help**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_baseline_object_inference.py -v
rtk .\.venv\Scripts\python.exe scripts\baseline_object_inference.py --help
```

Expected: 3 tests PASS; help exits 0 and lists all CLI arguments.

### Task 4: Verify official model information and run CPU baseline

**Files:**

- Download: `models/yolo11n.pt`
- Download: `data/samples/bus.jpg`
- Generate: `runs/baseline/bus_annotated.jpg`
- Generate: `runs/baseline/bus_metrics.json`

- [ ] **Step 1: Recheck official documentation before downloading**

Read current official pages:

- `https://docs.ultralytics.com/models/yolo11/`
- `https://docs.ultralytics.com/quickstart/#ultralytics-license`
- `https://www.ultralytics.com/license`

Record in handoff that Ultralytics offers AGPL-3.0 and enterprise licensing. Do not claim this local evaluation grants distribution rights.

- [ ] **Step 2: Download one small official sample image**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -c "from ultralytics.utils.downloads import download; download('https://ultralytics.com/images/bus.jpg', dir='data/samples')"
```

Expected: `data/samples/bus.jpg` exists. No dataset is downloaded.

- [ ] **Step 3: Run environment verification**

Run:

```powershell
rtk .\.venv\Scripts\python.exe scripts\check_environment.py
rtk .\.venv\Scripts\python.exe -m pip check
```

Expected: `Environment check: OK`; `No broken requirements found.`

- [ ] **Step 4: Run 15 warm-ups and 30 measured CPU inferences**

Run:

```powershell
rtk .\.venv\Scripts\python.exe scripts\baseline_object_inference.py --source data\samples\bus.jpg --model models\yolo11n.pt --imgsz 320 --warmup 15 --runs 30 --output runs\baseline\bus_annotated.jpg --metrics runs\baseline\bus_metrics.json
```

Expected: exit 0; weight downloads once if missing; JSON reports `samples: 30`, mean, p50, p95, max, effective FPS, peak RSS, and only fail-safe STOP statuses.

- [ ] **Step 5: Validate generated evidence**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -c "import json, math; from pathlib import Path; p=Path('runs/baseline/bus_metrics.json'); d=json.loads(p.read_text()); t=d['timing']; assert p.is_file() and Path(d['output']).is_file(); assert t['samples']==30; assert all(math.isfinite(t[k]) and t[k]>0 for k in ('mean_ms','p50_ms','p95_ms','max_ms','effective_fps')); assert d['peak_rss_mb']>0; assert set(d['status_counts']) <= {'STOP_OBSTACLE','STOP_UNCERTAIN'}; print('Baseline evidence: OK')"
```

Expected: `Baseline evidence: OK`.

- [ ] **Step 6: Inspect annotated image**

Open `runs/baseline/bus_annotated.jpg`. Confirm corridor polygon is visible, every model detection remains drawn, corridor detections have stronger red boxes, and overlay never states that movement is safe.

### Task 5: Final regression and handoff

**Files:**

- Verify: `scripts/baseline_object_inference.py`
- Verify: `tests/test_baseline_object_inference.py`
- Verify: `runs/baseline/bus_metrics.json`

- [ ] **Step 1: Run complete lightweight verification again**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -v
rtk .\.venv\Scripts\python.exe scripts\check_environment.py
rtk .\.venv\Scripts\python.exe -m pip check
```

Expected: all tests PASS, environment OK, dependencies consistent.

- [ ] **Step 2: Report measured result without safety overclaim**

Handoff must include exact mean, p50, p95, max, FPS, peak RSS, model/input/device, generated file links, and whether research target `p95 < 200 ms` was observed on this laptop. State clearly that one sample image and COCO pretrained weights do not validate real walking guidance, hazard coverage, or Android performance.
