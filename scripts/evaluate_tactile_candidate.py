from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from baseline_object_inference import STOP_FAILURE
from model_manifest import sha256_file


def binary_iou(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    if ground_truth.shape != prediction.shape:
        raise ValueError("mask shapes must match")
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    union = np.logical_or(ground_truth, prediction).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(ground_truth, prediction).sum() / union)


def summarize_presence(pairs: Sequence[tuple[bool, bool]]) -> dict[str, float | int | None]:
    if not pairs:
        raise ValueError("presence pairs are required")
    true_positive = sum(truth and prediction for truth, prediction in pairs)
    false_negative = sum(truth and not prediction for truth, prediction in pairs)
    false_positive = sum(not truth and prediction for truth, prediction in pairs)
    true_negative = sum(not truth and not prediction for truth, prediction in pairs)
    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None,
        "recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None,
        "false_positive_rate": false_positive / (false_positive + true_negative)
        if false_positive + true_negative
        else None,
    }


def rasterize_polygons(
    polygons: Sequence[np.ndarray],
    scores: Sequence[float],
    confidence: float,
    shape: tuple[int, int],
) -> np.ndarray:
    if len(polygons) != len(scores):
        raise ValueError("polygon and score counts must match")
    output = np.zeros(shape, dtype=np.uint8)
    for polygon, score in zip(polygons, scores):
        points = np.rint(np.asarray(polygon)).astype(np.int32)
        if score >= confidence and len(points) >= 3:
            cv2.fillPoly(output, [points], 255)
    return output


def select_best_threshold(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("threshold reports are required")

    def value(report: dict[str, Any], name: str, fallback: float) -> float:
        raw = report.get(name)
        return float(raw) if raw is not None else fallback

    return max(
        reports,
        key=lambda report: (
            value(report, "recall", -1.0),
            value(report, "mean_iou", -1.0),
            -value(report, "false_positive_rate", 1.0),
            -float(report["confidence"]),
        ),
    )


def annotation_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = np.zeros(shape, dtype=np.uint8)
    for item in payload.get("tactile", []):
        if item.get("label") != "tactile_paving":
            raise ValueError(f"unexpected tactile label in {path}")
        points = np.rint(np.asarray(item.get("points", []))).astype(np.int32)
        if len(points) < 3:
            raise ValueError(f"invalid tactile polygon in {path}")
        cv2.fillPoly(output, [points], 255)
    return output


def read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"unable to decode mask: {path}")
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"mask must be binary 0/255: {path}")
    return mask


def validation_samples(manifest_path: Path, annotations_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = []
    for row in payload.get("samples", []):
        if row.get("split") != "validation":
            continue
        source = Path(row["source_path"])
        if not source.is_file() or sha256_file(source) != row.get("source_sha256"):
            raise ValueError(f"validation source missing or hash mismatch: {source}")
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"unable to decode validation source: {source}")
        annotation = annotations_dir / f"{row['sample_id']}.json"
        if not annotation.is_file():
            raise ValueError(f"validation annotation missing: {annotation}")
        samples.append(
            {
                "sample_id": row["sample_id"],
                "source": source,
                "ground_truth": annotation_mask(annotation, image.shape[:2]),
            }
        )
    if not samples:
        raise ValueError("training manifest has no validation samples")
    return samples


def protected_samples(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ground_truth_manifest import read_manifest, validate_rows

    rows = read_manifest(manifest_path)
    validation = validate_rows(rows)
    if not validation.get("official_ready"):
        raise ValueError("protected ground-truth manifest is not official_ready")
    samples = [
        {
            "sample_id": row["sample_id"],
            "source": Path(row["source_path"]),
            "ground_truth": read_binary_mask(Path(row["ground_truth_mask_path"])),
        }
        for row in rows
    ]
    return samples, validation


def infer_samples(
    model: Any,
    samples: Sequence[dict[str, Any]],
    imgsz: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    outputs = []
    for sample in samples:
        started = time.perf_counter()
        result = model.predict(
            source=str(sample["source"]),
            imgsz=imgsz,
            conf=min_confidence,
            device="cpu",
            retina_masks=True,
            verbose=False,
        )[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        masks = getattr(result, "masks", None)
        polygons = list(masks.xy) if masks is not None else []
        boxes = getattr(result, "boxes", None)
        scores = boxes.conf.cpu().tolist() if boxes is not None else []
        if len(polygons) != len(scores):
            raise ValueError(f"prediction polygon/score mismatch: {sample['source']}")
        outputs.append(
            {**sample, "polygons": polygons, "scores": scores, "latency_ms": latency_ms}
        )
    return outputs


def threshold_report(
    predictions: Sequence[dict[str, Any]], confidence: float
) -> dict[str, Any]:
    pairs: list[tuple[bool, bool]] = []
    positive_ious: list[float] = []
    images = []
    for item in predictions:
        ground_truth = item["ground_truth"]
        prediction = rasterize_polygons(
            item["polygons"], item["scores"], confidence, ground_truth.shape
        )
        truth_present = bool(np.any(ground_truth))
        prediction_present = bool(np.any(prediction))
        pairs.append((truth_present, prediction_present))
        iou = binary_iou(ground_truth, prediction) if truth_present else None
        if iou is not None:
            positive_ious.append(iou)
        images.append(
            {
                "sample_id": item["sample_id"],
                "source": str(item["source"]),
                "ground_truth_present": truth_present,
                "prediction_present": prediction_present,
                "iou": iou,
                "latency_ms": item["latency_ms"],
            }
        )
    presence = summarize_presence(pairs)
    return {
        "confidence": confidence,
        "mask_threshold": 0.5,
        **presence,
        "mean_iou": statistics.fmean(positive_ious) if positive_ious else None,
        "median_latency_ms": statistics.median(
            item["latency_ms"] for item in predictions
        ),
        "images": images,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(
            f"evaluation report already exists; refusing protected-set rerun: {args.output}"
        )

    from ultralytics import YOLO
    from verify_tactile_candidate import validate_candidate

    candidate = validate_candidate(args.candidate_dir)
    config = json.loads(
        (args.candidate_dir / "training_config.json").read_text(encoding="utf-8")
    )
    training_manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    if training_manifest.get("dataset_version") != config.get("dataset_version"):
        raise ValueError("candidate and training manifest dataset versions differ")

    model = YOLO(str(args.candidate_dir / "best.pt"))
    validation = validation_samples(args.training_manifest, args.annotations_dir)
    validation_predictions = infer_samples(
        model, validation, args.imgsz, min(args.confidences)
    )
    reports = [
        threshold_report(validation_predictions, confidence)
        for confidence in args.confidences
    ]
    selected = select_best_threshold(reports)
    locked_confidence = float(selected["confidence"])

    protected, protected_validation = protected_samples(args.protected_manifest)
    protected_predictions = infer_samples(
        model, protected, args.imgsz, locked_confidence
    )
    protected_report = threshold_report(protected_predictions, locked_confidence)
    recall = protected_report["recall"]
    mean_iou = protected_report["mean_iou"]
    gate_passed = bool(
        recall is not None
        and mean_iou is not None
        and recall >= args.minimum_recall
        and mean_iou >= args.minimum_mean_iou
    )
    return {
        "warning": "Offline model evaluation only. This does not authorize walking or deployment.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate,
        "training_manifest": str(args.training_manifest),
        "training_manifest_sha256": sha256_file(args.training_manifest),
        "protected_manifest": str(args.protected_manifest),
        "protected_manifest_sha256": sha256_file(args.protected_manifest),
        "protected_manifest_validation": protected_validation,
        "selection_policy": "highest validation recall, then mean IoU, then lowest false-positive rate",
        "mask_threshold_note": "Ultralytics polygon outputs are already binarized; mask threshold is locked at 0.5.",
        "imgsz": args.imgsz,
        "validation_thresholds": reports,
        "locked_threshold": {"confidence": locked_confidence, "mask_threshold": 0.5},
        "protected_test": protected_report,
        "deployment_gate": {
            "minimum_mean_iou": args.minimum_mean_iou,
            "minimum_recall": args.minimum_recall,
            "passed": gate_passed,
            "status": "mobile_export_allowed" if gate_passed else "mobile_export_rejected",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate a tactile segmentation candidate."
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("artifacts/candidates/tactile-one-class-v2"),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/training/manifests/station-tactile-v2.json"),
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/training/annotations/station-tactile-v2"),
    )
    parser.add_argument(
        "--protected-manifest",
        type=Path,
        default=Path("data/ground_truth/manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/candidates/tactile-one-class-v2/evaluation_report.json"
        ),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--confidences",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25],
    )
    parser.add_argument("--minimum-mean-iou", type=float, default=0.75)
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    args = parser.parse_args()
    if (
        args.imgsz <= 0
        or not args.confidences
        or any(not 0.0 <= value <= 1.0 for value in args.confidences)
    ):
        parser.error("imgsz must be positive and confidences must be within [0, 1]")
    return args


def run(args: argparse.Namespace) -> int:
    payload = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "deployment_gate": payload["deployment_gate"]},
            indent=2,
        )
    )
    return 0 if payload["deployment_gate"]["passed"] else 2


def main() -> int:
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        print(f"{STOP_FAILURE}: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"{STOP_FAILURE}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
