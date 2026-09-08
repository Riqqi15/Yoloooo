from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from baseline_object_inference import (
    IMAGE_SUFFIXES,
    STOP_FAILURE,
    STOP_OBSTACLE,
    STOP_UNCERTAIN,
    analyze_result as analyze_object_result,
    load_model as load_object_model,
    predict_frame,
    timing_summary,
    write_metrics,
)
from baseline_tactile_inference import (
    MODEL_REPO,
    UPSTREAM_CONFIG_LABELS,
    analyze_result as analyze_tactile_result,
    load_model as load_tactile_model,
    rasterize_masks,
    sha256_file,
)
from model_manifest import DEFAULT_MODEL_MANIFEST


STATION_CONTEXTS = frozenset({
    "station_interior", "station_concourse", "station_platform",
    "station_track_area", "station_access", "station_access_sidewalk",
})


def load_station_scope(path: Path, images: list[Path]) -> dict[str, Any]:
    scope = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(scope, dict) or scope.get("scope") != "station_environment":
        raise ValueError("scope must be station_environment")
    included = scope.get("included_contexts")
    if (
        not isinstance(included, list)
        or not included
        or any(not isinstance(context, str) or context not in STATION_CONTEXTS
               for context in included)
    ):
        raise ValueError("included_contexts contains an unapproved context")
    samples = scope.get("samples")
    if not isinstance(samples, dict):
        raise ValueError("scope samples must be a filename-to-context object")
    for name, context in samples.items():
        if not name or "/" in name or "\\" in name:
            raise ValueError("scope samples must use image basenames")
        if not isinstance(context, str) or context not in included:
            raise ValueError(f"unapproved context for {name}: {context}")
    active = {image.name for image in images}
    unregistered = active - samples.keys()
    missing = samples.keys() - active
    if unregistered:
        raise ValueError(f"unregistered active images: {', '.join(sorted(unregistered))}")
    if missing:
        raise ValueError(f"registered images missing: {', '.join(sorted(missing))}")
    return scope


def discover_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"sample directory not found: {directory}")
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


def output_name(source: Path) -> str:
    return f"{source.stem}_{source.suffix.lstrip('.').lower()}.jpg"


def combined_status(object_status: str, failed: bool = False) -> str:
    if failed or object_status not in {STOP_OBSTACLE, STOP_UNCERTAIN}:
        return STOP_FAILURE
    return object_status


def batch_exit_code(rows: list[dict[str, Any]]) -> int:
    return int(any(row.get("combined_status") == STOP_FAILURE for row in rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local test images with fail-safe object and tactile baselines."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--scope-file", type=Path, default=Path("data/dataset_scope.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sample-evaluation"))
    parser.add_argument("--object-model", default="models/yolo11n.pt")
    parser.add_argument(
        "--tactile-model",
        type=Path,
        default=Path("models/guidetwsi/yolo11n_tactile.pt"),
    )
    parser.add_argument(
        "--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.imgsz <= 0 or args.warmup < 0 or not 0.0 <= args.confidence <= 1.0:
        parser.error("imgsz must be positive, warmup nonnegative, confidence within [0, 1]")
    return args


def first_decodable_image(paths: list[Path]) -> Any:
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is not None:
            return frame
    raise ValueError("no decodable sample images")


def write_image(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"unable to write image: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset_role",
        "dataset_scope",
        "station_context",
        "source",
        "source_sha256",
        "input_format",
        "combined_status",
        "object_status",
        "object_count",
        "corridor_object_count",
        "object_latency_ms",
        "tactile_status",
        "tactile_instance_count",
        "tactile_class_counts",
        "tactile_mask_coverage",
        "tactile_latency_ms",
        "peak_rss_mb",
        "object_output",
        "tactile_output",
        "tactile_prediction_mask",
        "tactile_prediction_mask_sha256",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {field: row.get(field, "") for field in fields}
            flattened["tactile_class_counts"] = json.dumps(
                row.get("tactile_class_counts", {}), ensure_ascii=False
            )
            writer.writerow(flattened)


def evaluate_image(
    source: Path,
    output_dir: Path,
    object_model: Any,
    tactile_model: Any,
    tactile_labels: dict[int, str],
    imgsz: int,
    confidence: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset_role": "test_only",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "input_format": source.suffix.lower().lstrip("."),
    }
    try:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"unable to decode image: {source}")

        object_result, object_ms, object_rss = predict_frame(
            object_model, frame, imgsz, confidence
        )
        object_overlay, object_status, object_count, corridor_count = (
            analyze_object_result(object_result, frame)
        )
        tactile_result, tactile_ms, tactile_rss = predict_frame(
            tactile_model, frame, imgsz, confidence
        )
        tactile_overlay, tactile_instances, coverage, tactile_counts = (
            analyze_tactile_result(tactile_result, tactile_labels)
        )
        tactile_mask = rasterize_masks(tactile_result, *frame.shape[:2])

        name = output_name(source)
        object_output = output_dir / "object" / name
        tactile_output = output_dir / "tactile" / name
        tactile_mask_output = output_dir / "tactile_masks" / Path(name).with_suffix(".png")
        write_image(object_output, object_overlay)
        write_image(tactile_output, tactile_overlay)
        write_image(tactile_mask_output, tactile_mask)

        row.update(
            {
                "combined_status": combined_status(object_status),
                "object_status": object_status,
                "object_count": object_count,
                "corridor_object_count": corridor_count,
                "object_latency_ms": object_ms,
                "tactile_status": STOP_UNCERTAIN,
                "tactile_instance_count": tactile_instances,
                "tactile_class_counts": tactile_counts,
                "tactile_mask_coverage": coverage,
                "tactile_latency_ms": tactile_ms,
                "peak_rss_mb": max(object_rss, tactile_rss) / (1024 * 1024),
                "object_output": str(object_output),
                "tactile_output": str(tactile_output),
                "tactile_prediction_mask": str(tactile_mask_output),
                "tactile_prediction_mask_sha256": sha256_file(tactile_mask_output),
                "error": "",
            }
        )
    except Exception as error:
        row.update(
            {
                "combined_status": STOP_FAILURE,
                "tactile_status": STOP_FAILURE,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return row


def run(args: argparse.Namespace) -> int:
    images = discover_images(args.source_dir)
    if not images:
        raise ValueError(f"no supported images in {args.source_dir}")
    scope = load_station_scope(args.scope_file, images)

    object_model = load_object_model(args.object_model, args.model_manifest)
    object_model_path = Path(args.object_model)
    if not object_model_path.is_file():
        raise FileNotFoundError(f"object model artifact not found after load: {object_model_path}")
    tactile_model, tactile_labels, tactile_path = load_tactile_model(
        args.tactile_model, args.model_manifest
    )
    warmup_frame = first_decodable_image(images)
    for _ in range(args.warmup):
        predict_frame(object_model, warmup_frame, args.imgsz, args.confidence)
        predict_frame(tactile_model, warmup_frame, args.imgsz, args.confidence)

    rows = [
        evaluate_image(
            image,
            args.output_dir,
            object_model,
            tactile_model,
            tactile_labels,
            args.imgsz,
            args.confidence,
        )
        for image in images
    ]
    for row, source in zip(rows, images):
        row["dataset_scope"] = scope["scope"]
        row["station_context"] = scope["samples"][source.name]
    successful = [row for row in rows if row["combined_status"] != STOP_FAILURE]
    object_latencies = [float(row["object_latency_ms"]) for row in successful]
    tactile_latencies = [float(row["tactile_latency_ms"]) for row in successful]
    statuses = Counter(str(row["combined_status"]) for row in rows)
    tactile_instances = sum(int(row.get("tactile_instance_count", 0)) for row in rows)

    payload = {
        "warning": "Offline test evidence only. No result authorizes movement.",
        "dataset_role": "test_only",
        "dataset_scope": scope["scope"],
        "scope_file": str(args.scope_file),
        "scope_sha256": sha256_file(args.scope_file),
        "scope_registry": scope,
        "model_manifest": str(args.model_manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "settings": {
            "imgsz": args.imgsz,
            "confidence": args.confidence,
            "warmup": args.warmup,
            "measured_runs_per_image": 1,
        },
        "models": {
            "object": {"path": args.object_model, "sha256": sha256_file(object_model_path)},
            "tactile": {
                "repo": MODEL_REPO,
                "path": str(tactile_path),
                "sha256": sha256_file(tactile_path),
                "checkpoint_labels": tactile_labels,
                "upstream_config_labels": UPSTREAM_CONFIG_LABELS,
                "upstream_label_mismatch": tactile_labels != UPSTREAM_CONFIG_LABELS,
            },
        },
        "summary": {
            "total_images": len(rows),
            "successful_images": len(successful),
            "failed_images": len(rows) - len(successful),
            "status_counts": dict(statuses),
            "tactile_instance_count": tactile_instances,
            "images_with_tactile_masks": sum(
                int(row.get("tactile_instance_count", 0) > 0) for row in rows
            ),
            "object_timing": timing_summary(object_latencies) if successful else None,
            "tactile_timing": timing_summary(tactile_latencies) if successful else None,
            "peak_rss_mb": max(
                (float(row.get("peak_rss_mb", 0.0)) for row in rows), default=0.0
            ),
        },
        "images": rows,
    }
    report_json = args.output_dir / "report.json"
    report_csv = args.output_dir / "report.csv"
    write_metrics(report_json, payload)
    write_csv(report_csv, rows)
    print(json.dumps({**payload["summary"], "report": str(report_json)}, indent=2))
    return batch_exit_code(rows)


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
