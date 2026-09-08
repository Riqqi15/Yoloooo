# Tactile Inference Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a fail-safe GuideTWSI YOLO11n segmentation CLI for local CPU image/video inference and benchmarking.

**Architecture:** One tactile-specific CLI downloads and validates the official checkpoint, reuses proven timing/source helpers from the object baseline, renders masks, and writes benchmark JSON. One standard-library test module checks the safety-critical task/label contract and mask aggregation without loading a model.

**Tech Stack:** Python 3.11, Ultralytics 8.4.138, PyTorch CPU 2.14.0, OpenCV 5.0.0, NumPy, psutil, huggingface_hub 1.29.0, `unittest`.

---

## File Map

- Create `scripts/baseline_tactile_inference.py`: checkpoint fetch/validation, SHA-256, image/video segmentation, annotation, metrics, fail-safe exits.
- Create `tests/test_baseline_tactile_inference.py`: deterministic task, label, and mask tests.
- Reuse `scripts/baseline_object_inference.py`: `parse_source`, `predict_frame`, `timing_summary`, and `write_metrics`; do not modify it.
- Download `models/guidetwsi/yolo11n_tactile.pt`: official 5.71 MB checkpoint only.
- Generate `runs/baseline/tactile/smoke_annotated.jpg` and `runs/baseline/tactile/smoke_metrics.json`.

This workspace has no `.git` directory. Do not create a repository, worktree, branch, or commit.

### Task 1: Add failing checkpoint and mask tests

**Files:**

- Create: `tests/test_baseline_tactile_inference.py`

- [x] **Step 1: Write the test module**

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_tactile_inference import (  # noqa: E402
    EXPECTED_LABELS,
    mask_summary,
    normalize_labels,
    validate_model_contract,
)


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class TactileInferenceTest(unittest.TestCase):
    def test_normalize_labels_accepts_dict_or_list(self) -> None:
        self.assertEqual(normalize_labels({0: "tactile_paving"}), EXPECTED_LABELS)
        self.assertEqual(normalize_labels(["tactile_paving"]), EXPECTED_LABELS)

    def test_contract_rejects_wrong_task(self) -> None:
        model = SimpleNamespace(task="detect", names=EXPECTED_LABELS)
        with self.assertRaisesRegex(ValueError, "expected segment task"):
            validate_model_contract(model)

    def test_contract_rejects_wrong_labels(self) -> None:
        model = SimpleNamespace(task="segment", names={0: "guiding", 1: "warning"})
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            validate_model_contract(model)

    def test_mask_summary_handles_no_mask(self) -> None:
        self.assertEqual(mask_summary(SimpleNamespace(masks=None)), (0, 0.0))

    def test_mask_summary_unions_instances(self) -> None:
        values = np.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        result = SimpleNamespace(masks=SimpleNamespace(data=FakeTensor(values)))
        instances, coverage = mask_summary(result)
        self.assertEqual(instances, 2)
        self.assertEqual(coverage, 0.75)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test and confirm missing module failure**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_baseline_tactile_inference.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'baseline_tactile_inference'`.

### Task 2: Implement contract validation and mask aggregation

**Files:**

- Create: `scripts/baseline_tactile_inference.py`
- Test: `tests/test_baseline_tactile_inference.py`

- [x] **Step 1: Add imports, constants, and pure helpers**

```python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from baseline_object_inference import (
    STOP_FAILURE,
    STOP_UNCERTAIN,
    parse_source,
    predict_frame,
    timing_summary,
    write_metrics,
)

MODEL_REPO = "guidedogrobot-tactile/GuideTWSI-weights"
MODEL_FILENAME = "yolo11n_tactile.pt"
EXPECTED_LABELS = {0: "tactile_paving"}


def normalize_labels(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    items = names.items() if isinstance(names, Mapping) else enumerate(names)
    return {int(index): str(name) for index, name in items}


def validate_model_contract(model: Any) -> dict[int, str]:
    if getattr(model, "task", None) != "segment":
        raise ValueError(f"expected segment task, got {getattr(model, 'task', None)!r}")
    labels = normalize_labels(model.names)
    if labels != EXPECTED_LABELS:
        raise ValueError(f"label mismatch: expected {EXPECTED_LABELS}, got {labels}")
    return labels


def mask_summary(result: Any) -> tuple[int, float]:
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return 0, 0.0
    values = masks.data.cpu().numpy()
    if values.size == 0:
        return 0, 0.0
    combined = np.any(values > 0.5, axis=0)
    return int(values.shape[0]), float(combined.mean())
```

- [x] **Step 2: Run the focused tests**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_baseline_tactile_inference.py -v
```

Expected: 5 tests PASS.

### Task 3: Complete checkpoint, inference, and metrics CLI

**Files:**

- Modify: `scripts/baseline_tactile_inference.py`

- [x] **Step 1: Add CLI parsing and paths**

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-safe GuideTWSI tactile baseline. Never authorizes movement."
    )
    parser.add_argument("--source", required=True, help="Image or video path")
    parser.add_argument("--model", type=Path, default=Path("models/guidetwsi/yolo11n_tactile.pt"))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--metrics", type=Path, default=Path("runs/baseline/tactile/metrics.json")
    )
    args = parser.parse_args()
    if args.imgsz <= 0 or args.warmup < 0 or args.runs < 1 or not 0.0 <= args.confidence <= 1.0:
        parser.error("imgsz/runs must be positive, warmup nonnegative, confidence within [0, 1]")
    return args


def default_output(source_kind: str) -> Path:
    suffix = ".jpg" if source_kind == "image" else ".mp4"
    return Path("runs/baseline/tactile") / f"annotated{suffix}"
```

- [x] **Step 2: Add checkpoint download, SHA-256, and load**

```python
def ensure_model(path: Path) -> Path:
    if path.is_file():
        return path
    if path.name != MODEL_FILENAME:
        raise FileNotFoundError(f"custom model not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir=path.parent)
    )
    if downloaded.resolve() != path.resolve():
        shutil.copy2(downloaded, path)
    if not path.is_file():
        raise FileNotFoundError(f"model unavailable after download: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path) -> tuple[YOLO, dict[int, str], Path]:
    resolved = ensure_model(path)
    model = YOLO(str(resolved))
    return model, validate_model_contract(model), resolved
```

- [x] **Step 3: Add class counting and annotation**

```python
def result_class_counts(result: Any, labels: Mapping[int, str]) -> dict[str, int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "cls", None) is None:
        return {}
    counts: Counter[str] = Counter()
    for raw_class_id in boxes.cls.cpu().tolist():
        class_id = int(raw_class_id)
        if class_id not in labels:
            raise ValueError(f"result class ID {class_id} absent from model labels")
        counts[labels[class_id]] += 1
    return dict(counts)


def analyze_result(
    result: Any, labels: Mapping[int, str]
) -> tuple[np.ndarray, int, float, dict[str, int]]:
    annotated = result.plot()
    instances, coverage = mask_summary(result)
    counts = result_class_counts(result, labels)
    cv2.putText(annotated, STOP_UNCERTAIN, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(
        annotated,
        "TACTILE CANDIDATE ONLY - NOT WALKING GUIDANCE",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )
    return annotated, instances, coverage, counts
```

- [x] **Step 4: Add image processing**

```python
def process_image(
    model: YOLO,
    labels: Mapping[int, str],
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
    annotated, instances, coverage, counts = analyze_result(final_result, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise OSError(f"unable to write image: {output}")
    return {
        "timing": timing_summary(latencies),
        "peak_rss_mb": max(rss_samples) / (1024 * 1024),
        "status_counts": {STOP_UNCERTAIN: 1},
        "instance_count": instances,
        "class_counts": counts,
        "frames_with_masks": int(instances > 0),
        "mean_mask_coverage": coverage,
        "max_mask_coverage": coverage,
        "ultralytics_speed_ms": final_result.speed,
    }
```

- [x] **Step 5: Add sequential video processing**

```python
def process_video(
    model: YOLO,
    labels: Mapping[int, str],
    source: str,
    output: Path,
    imgsz: int,
    confidence: float,
    warmup: int,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {source}")
    writer = None
    latencies: list[float] = []
    rss_samples: list[int] = []
    coverages: list[float] = []
    counts: Counter[str] = Counter()
    instances = 0
    frames_with_masks = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result, latency_ms, rss = predict_frame(model, frame, imgsz, confidence)
            annotated, frame_instances, coverage, frame_counts = analyze_result(result, labels)
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
                coverages.append(coverage)
                counts.update(frame_counts)
                instances += frame_instances
                frames_with_masks += int(frame_instances > 0)
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frame_index <= warmup:
        raise ValueError("video ended before measured frames were available")
    return {
        "timing": timing_summary(latencies),
        "peak_rss_mb": max(rss_samples) / (1024 * 1024),
        "status_counts": {STOP_UNCERTAIN: len(latencies)},
        "instance_count": instances,
        "class_counts": dict(counts),
        "frames_with_masks": frames_with_masks,
        "mean_mask_coverage": float(np.mean(coverages)),
        "max_mask_coverage": max(coverages),
    }
```

- [x] **Step 6: Add run and fail-safe entrypoint**

```python
def run(args: argparse.Namespace) -> int:
    source_kind, source = parse_source(args.source)
    if source_kind == "webcam":
        raise ValueError("webcam is outside this reproducible tactile baseline")
    output = args.output or default_output(source_kind)
    model, labels, model_path = load_model(args.model)
    if source_kind == "image":
        result = process_image(
            model, labels, str(source), output, args.imgsz, args.confidence, args.warmup, args.runs
        )
    else:
        result = process_video(
            model, labels, str(source), output, args.imgsz, args.confidence, args.warmup
        )
    payload = {
        "warning": "Tactile candidate baseline only. No result authorizes movement.",
        "model_repo": MODEL_REPO,
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "task": model.task,
        "labels": labels,
        "device": "cpu",
        "source_kind": source_kind,
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "warmup": args.warmup,
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

- [x] **Step 7: Run both test modules and CLI help**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -v
rtk .\.venv\Scripts\python.exe scripts\baseline_tactile_inference.py --help
```

Expected: 8 tests PASS; help exits 0 and lists tactile CLI arguments.

### Task 4: Download, validate, and benchmark the official checkpoint

**Files:**

- Download: `models/guidetwsi/yolo11n_tactile.pt`
- Generate: `runs/baseline/tactile/smoke_annotated.jpg`
- Generate: `runs/baseline/tactile/smoke_metrics.json`

- [x] **Step 1: Run a fresh official-source audit**

Verify the current GuideTWSI README, MIT license, `model_weights/yolo11n_tactile.pt`, and `configs/yolov11_seg_n.yaml`. Confirm the config still states `task: segment`, one class, and `class_names: ["tactile_paving"]`. Keep the separate Ultralytics AGPL-3.0/Enterprise warning.

- [x] **Step 2: Run the static CPU benchmark**

Run:

```powershell
rtk .\.venv\Scripts\python.exe scripts\baseline_tactile_inference.py --source data\samples\bus.jpg --imgsz 320 --warmup 15 --runs 30 --output runs\baseline\tactile\smoke_annotated.jpg --metrics runs\baseline\tactile\smoke_metrics.json
```

Expected: checkpoint downloads once, validates as segment + `{0: "tactile_paving"}`, exits 0, writes annotated output and 30 measured timing samples. Zero masks on this non-tactile image is valid and remains `STOP_UNCERTAIN`.

- [x] **Step 3: Validate benchmark evidence**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -c "import json, math; from pathlib import Path; p=Path('runs/baseline/tactile/smoke_metrics.json'); d=json.loads(p.read_text()); t=d['timing']; assert p.is_file() and Path(d['output']).is_file(); assert d['task']=='segment' and d['labels']=={'0':'tactile_paving'}; assert t['samples']==30; assert all(math.isfinite(t[k]) and t[k]>0 for k in ('mean_ms','p50_ms','p95_ms','max_ms','effective_fps')); assert d['peak_rss_mb']>0 and set(d['status_counts'])=={'STOP_UNCERTAIN'}; print('Tactile baseline evidence: OK')"
```

Expected: `Tactile baseline evidence: OK`.

- [x] **Step 4: Inspect the annotated image**

Open `runs/baseline/tactile/smoke_annotated.jpg`. Confirm mask output, if any, is visible; both fail-safe overlay lines are readable; no text implies safe movement or distinguishes guiding path from warning block.

### Task 5: Final regression and handoff

**Files:**

- Verify: `scripts/baseline_object_inference.py`
- Verify: `scripts/baseline_tactile_inference.py`
- Verify: `tests/test_baseline_object_inference.py`
- Verify: `tests/test_baseline_tactile_inference.py`
- Verify: `runs/baseline/tactile/smoke_metrics.json`

- [x] **Step 1: Run complete lightweight verification**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -v
rtk .\.venv\Scripts\python.exe -m py_compile scripts\baseline_object_inference.py scripts\baseline_tactile_inference.py tests\test_baseline_object_inference.py tests\test_baseline_tactile_inference.py
rtk .\.venv\Scripts\python.exe scripts\check_environment.py
rtk .\.venv\Scripts\python.exe -m pip check
```

Expected: all tests PASS, compilation succeeds, environment is OK, dependencies remain consistent.

- [x] **Step 2: Report without safety overclaim**

Report exact checkpoint label mapping, SHA-256, mean/p50/p95/max/FPS/peak RSS, output links, and whether tactile inference alone observed p95 below 200 ms on this laptop. State that the metric excludes capture, fusion, decision, and feedback; one checkpoint smoke test does not validate path guidance, Indonesia-domain accuracy, Android performance, or safe use.

## Execution Notes (2026-09-05)

- The documented Hugging Face weights repository returned HTTP 401. The public 5.71 MB checkpoint was downloaded from the official GuideTWSI GitHub repository instead.
- The checkpoint metadata is `{0: "braille"}`, while the upstream config declares `{0: "tactile_paving"}`. The CLI preserves and validates the checkpoint label, records the mismatch, and treats output only as a binary tactile candidate.
- Final verification: 9 tests passed; byte-compilation, environment check, and `pip check` passed. The workspace is not a Git repository, so no branch integration step applies.
