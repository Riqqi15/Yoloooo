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
import numpy as np
import psutil
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from model_manifest import (
    DEFAULT_MODEL_MANIFEST,
    normalize_labels,
    sha256_file,
    verify_model_entry,
)

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


def corridor_polygon(width: int, height: int) -> np.ndarray:
    return np.array(
        [[round(x * (width - 1)), round(y * (height - 1))] for x, y in CORRIDOR_NORMALIZED],
        dtype=np.int32,
    )


def box_in_corridor(box: Sequence[float], polygon: np.ndarray) -> bool:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-safe YOLO11n CPU baseline. Output never grants permission to proceed."
    )
    parser.add_argument("--source", required=True, help="Image/video path or webcam index")
    parser.add_argument("--model", default="models/yolo11n.pt")
    parser.add_argument(
        "--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST
    )
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


def load_model(
    model_path: str, manifest_path: Path = DEFAULT_MODEL_MANIFEST
) -> YOLO:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = Path(attempt_download_asset(path))
    if not resolved.is_file():
        raise FileNotFoundError(f"model unavailable after download attempt: {path}")
    entry, resolved = verify_model_entry(manifest_path, "object", resolved)
    model = YOLO(str(resolved), task="detect")
    if model.task != entry["task"]:
        raise ValueError(f"object task mismatch: expected {entry['task']!r}, got {model.task!r}")
    expected_labels = normalize_labels(entry["raw_labels"])
    actual_labels = normalize_labels(model.names)
    if actual_labels != expected_labels:
        raise ValueError("object model labels differ from pinned manifest")
    return model


def predict_frame(
    model: YOLO, frame: np.ndarray, imgsz: int, confidence: float
) -> tuple[Any, float, int]:
    started_ns = time.perf_counter_ns()
    result = model.predict(frame, imgsz=imgsz, conf=confidence, device="cpu", verbose=False)[0]
    latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return result, latency_ms, psutil.Process().memory_info().rss


def analyze_result(
    result: Any, frame: np.ndarray
) -> tuple[np.ndarray, str, int, int]:
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
    cv2.putText(
        annotated,
        status,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
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
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
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


def run(args: argparse.Namespace) -> int:
    source_kind, source = parse_source(args.source)
    output = args.output or default_output(source_kind)
    model = load_model(args.model, args.model_manifest)
    if source_kind == "image":
        result = process_image(
            model,
            str(source),
            output,
            args.imgsz,
            args.confidence,
            args.warmup,
            args.runs,
        )
    else:
        result = process_stream(model, source, output, args.imgsz, args.confidence, args.warmup)
    payload = {
        "warning": "Baseline only. No result grants permission to proceed.",
        "model": args.model,
        "model_sha256": sha256_file(Path(args.model)),
        "model_manifest": str(args.model_manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
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
