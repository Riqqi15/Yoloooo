from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import urlretrieve

import cv2
import numpy as np
from ultralytics import YOLO

from baseline_object_inference import (
    STOP_FAILURE,
    STOP_UNCERTAIN,
    parse_source,
    predict_frame,
    timing_summary,
    write_metrics,
)
from model_manifest import (
    DEFAULT_MODEL_MANIFEST,
    normalize_labels,
    sha256_file,
    verify_model_entry,
)

MODEL_REPO = "DARoSLab/GuideTWSI"
MODEL_FILENAME = "yolo11n_tactile.pt"
MODEL_URL = (
    "https://raw.githubusercontent.com/DARoSLab/GuideTWSI/"
    "master/model_weights/yolo11n_tactile.pt"
)
EXPECTED_LABELS = {0: "braille"}
UPSTREAM_CONFIG_LABELS = {0: "tactile_paving"}


def validate_model_contract(
    model: Any,
    expected_labels: Mapping[int, str] = EXPECTED_LABELS,
    expected_task: str = "segment",
) -> dict[int, str]:
    if getattr(model, "task", None) != expected_task:
        raise ValueError(f"expected {expected_task} task, got {getattr(model, 'task', None)!r}")
    labels = normalize_labels(model.names)
    normalized_expected = normalize_labels(expected_labels)
    if labels != normalized_expected:
        raise ValueError(f"label mismatch: expected {normalized_expected}, got {labels}")
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


def rasterize_masks(result: Any, height: int, width: int) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint8)
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return output
    for polygon in masks.xy:
        points = np.rint(np.asarray(polygon)).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(output, [points], 255)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-safe GuideTWSI tactile baseline. Never authorizes movement."
    )
    parser.add_argument("--source", required=True, help="Image or video path")
    parser.add_argument(
        "--model", type=Path, default=Path("models/guidetwsi/yolo11n_tactile.pt")
    )
    parser.add_argument(
        "--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--metrics", type=Path, default=Path("runs/baseline/tactile/metrics.json")
    )
    args = parser.parse_args()
    if (
        args.imgsz <= 0
        or args.warmup < 0
        or args.runs < 1
        or not 0.0 <= args.confidence <= 1.0
    ):
        parser.error(
            "imgsz/runs must be positive, warmup nonnegative, confidence within [0, 1]"
        )
    return args


def default_output(source_kind: str) -> Path:
    suffix = ".jpg" if source_kind == "image" else ".mp4"
    return Path("runs/baseline/tactile") / f"annotated{suffix}"


def ensure_model(path: Path) -> Path:
    if path.is_file():
        return path
    if path.name != MODEL_FILENAME:
        raise FileNotFoundError(f"custom model not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    try:
        urlretrieve(MODEL_URL, partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    if not path.is_file():
        raise FileNotFoundError(f"model unavailable after download: {path}")
    return path


def load_model(
    path: Path, manifest_path: Path = DEFAULT_MODEL_MANIFEST
) -> tuple[YOLO, dict[int, str], Path]:
    resolved = ensure_model(path)
    entry, resolved = verify_model_entry(manifest_path, "tactile", resolved)
    model = YOLO(str(resolved))
    labels = validate_model_contract(
        model,
        normalize_labels(entry["raw_labels"]),
        str(entry["task"]),
    )
    return model, labels, resolved


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
    cv2.putText(
        annotated,
        STOP_UNCERTAIN,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
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


def run(args: argparse.Namespace) -> int:
    source_kind, source = parse_source(args.source)
    if source_kind == "webcam":
        raise ValueError("webcam is outside this reproducible tactile baseline")
    output = args.output or default_output(source_kind)
    model, labels, model_path = load_model(args.model, args.model_manifest)
    if source_kind == "image":
        result = process_image(
            model,
            labels,
            str(source),
            output,
            args.imgsz,
            args.confidence,
            args.warmup,
            args.runs,
        )
    else:
        result = process_video(
            model, labels, str(source), output, args.imgsz, args.confidence, args.warmup
        )
    payload = {
        "warning": "Tactile candidate baseline only. No result authorizes movement.",
        "label_notice": (
            "Checkpoint metadata uses 'braille'; upstream config declares "
            "'tactile_paving'. Output remains a binary tactile candidate only."
        ),
        "model_repo": MODEL_REPO,
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "model_manifest": str(args.model_manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "task": model.task,
        "labels": labels,
        "upstream_config_labels": UPSTREAM_CONFIG_LABELS,
        "upstream_label_mismatch": labels != UPSTREAM_CONFIG_LABELS,
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
