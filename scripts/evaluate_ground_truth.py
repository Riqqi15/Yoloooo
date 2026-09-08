from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from baseline_object_inference import STOP_FAILURE, write_metrics
from baseline_tactile_inference import sha256_file
from ground_truth_manifest import read_manifest, validate_rows
from model_manifest import verify_model_entry


def presence_metrics(pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("no presence labels")
    tp = sum(ground_truth == 1 and prediction == 1 for ground_truth, prediction in pairs)
    fp = sum(ground_truth == 0 and prediction == 1 for ground_truth, prediction in pairs)
    fn = sum(ground_truth == 1 and prediction == 0 for ground_truth, prediction in pairs)
    tn = sum(ground_truth == 0 and prediction == 0 for ground_truth, prediction in pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if precision == 0.0 and recall == 0.0 else None
    )
    return {
        "samples": len(pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(pairs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def binary_iou(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"prediction dimensions {prediction.shape} do not match ground truth {ground_truth.shape}"
        )
    ground_truth_pixels = ground_truth > 0
    prediction_pixels = prediction > 0
    union = np.logical_or(ground_truth_pixels, prediction_pixels).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(ground_truth_pixels, prediction_pixels).sum()
    return float(intersection / union)


def require_complete_manifest(
    summary: dict[str, Any], allow_provisional: bool = False
) -> None:
    if summary.get("invalid_rows"):
        raise ValueError(f"manifest invalid: {summary['invalid_rows']} rows")
    if summary.get("incomplete_rows"):
        raise ValueError(f"manifest incomplete: {summary['incomplete_rows']} rows")
    if summary.get("provisional_rows") and not allow_provisional:
        raise ValueError(
            f"manifest contains provisional annotations: {summary['provisional_rows']} rows"
        )


def read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"unable to decode mask: {path}")
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"mask must be binary 0/255: {path}")
    return mask


def evaluate(
    manifest_path: Path, predictions_path: Path, allow_provisional: bool = False
) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    validation = validate_rows(rows)
    require_complete_manifest(validation, allow_provisional)

    report = json.loads(predictions_path.read_text(encoding="utf-8"))
    if report.get("dataset_role") != "test_only":
        raise ValueError("prediction report must declare dataset_role=test_only")
    if report.get("dataset_scope") != "station_environment":
        raise ValueError("prediction report must declare station_environment scope")
    report_rows = report.get("images")
    if not isinstance(report_rows, list) or not report_rows:
        raise ValueError("prediction report has no image rows")
    predictions = {str(row.get("source", "")): row for row in report_rows}
    if len(predictions) != len(report_rows):
        raise ValueError("prediction report contains duplicate source paths")
    manifest_sources = {row["source_path"] for row in rows}
    prediction_sources = set(predictions)
    if prediction_sources != manifest_sources:
        missing = sorted(manifest_sources - prediction_sources)
        extra = sorted(prediction_sources - manifest_sources)
        raise ValueError(f"prediction source set mismatch: missing={missing}, extra={extra}")
    scope_path = Path(str(report.get("scope_file", "")))
    if not scope_path.is_file() or sha256_file(scope_path) != report.get("scope_sha256"):
        raise ValueError("scope registry missing or hash mismatch")
    model_manifest_path = Path(str(report.get("model_manifest", "")))
    if (
        not model_manifest_path.is_file()
        or sha256_file(model_manifest_path) != report.get("model_manifest_sha256")
    ):
        raise ValueError("model manifest missing or hash mismatch")
    for model_name in ("object", "tactile"):
        model = report.get("models", {}).get(model_name, {})
        model_path = Path(str(model.get("path", "")))
        if not model_path.is_file() or sha256_file(model_path) != model.get("sha256"):
            raise ValueError(f"{model_name} model missing or hash mismatch")
        verify_model_entry(model_manifest_path, model_name, model_path)

    pairs: list[tuple[int, int]] = []
    positive_ious: list[float] = []
    image_results = []
    for row in rows:
        source = row["source_path"]
        if source not in predictions:
            raise ValueError(f"prediction missing for source: {source}")
        prediction_row = predictions[source]
        if prediction_row.get("combined_status") == STOP_FAILURE or prediction_row.get(
            "tactile_status"
        ) == STOP_FAILURE:
            raise ValueError(f"prediction failed for source: {source}")
        if prediction_row.get("source_sha256") != sha256_file(Path(source)):
            raise ValueError(f"source hash mismatch: {source}")
        try:
            tactile_instances = int(prediction_row["tactile_instance_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid tactile_instance_count: {source}") from error
        if tactile_instances < 0:
            raise ValueError(f"invalid tactile_instance_count: {source}")
        prediction_mask_path = Path(
            str(prediction_row.get("tactile_prediction_mask", ""))
        )
        prediction_mask = read_binary_mask(prediction_mask_path)
        if prediction_row.get("tactile_prediction_mask_sha256") != sha256_file(
            prediction_mask_path
        ):
            raise ValueError(f"prediction mask hash mismatch: {source}")
        source_image = cv2.imread(source)
        if source_image is None or prediction_mask.shape != source_image.shape[:2]:
            raise ValueError(f"prediction mask dimensions mismatch: {source}")
        ground_truth_present = int(row["ground_truth_tactile_present"])
        prediction_present = int(tactile_instances > 0)
        pairs.append((ground_truth_present, prediction_present))
        outcome = {
            (1, 1): "TP",
            (0, 1): "FP",
            (1, 0): "FN",
            (0, 0): "TN",
        }[(ground_truth_present, prediction_present)]
        iou = None
        if ground_truth_present:
            ground_truth_mask = read_binary_mask(Path(row["ground_truth_mask_path"]))
            iou = binary_iou(ground_truth_mask, prediction_mask)
            positive_ious.append(iou)
        image_results.append(
            {
                "source": source,
                "ground_truth_present": ground_truth_present,
                "prediction_present": prediction_present,
                "outcome": outcome,
                "iou": iou,
            }
        )

    return {
        "warning": (
            "Offline test-set metrics only. Results do not validate safe walking or deployment."
        ),
        "evaluation_status": "provisional" if validation["provisional_rows"] else "official",
        "dataset_role": "test_only",
        "dataset_scope": "station_environment",
        "annotation_validation": validation,
        "manifest": str(manifest_path),
        "predictions": str(predictions_path),
        "presence": presence_metrics(pairs),
        "segmentation": {
            "positive_ground_truth_samples": len(positive_ious),
            "mean_iou": statistics.fmean(positive_ious) if positive_ious else None,
        },
        "images": image_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate tactile predictions against validated human ground truth."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/ground_truth/manifest.csv")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("runs/sample-evaluation/report.json"),
    )
    parser.add_argument(
        "--output", type=Path
    )
    parser.add_argument("--allow-provisional", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    payload = evaluate(args.manifest, args.predictions, args.allow_provisional)
    output = args.output or Path(
        "runs/accuracy/tactile_metrics_provisional.json"
        if args.allow_provisional
        else "runs/accuracy/tactile_metrics.json"
    )
    write_metrics(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
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
