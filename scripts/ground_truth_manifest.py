from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from baseline_object_inference import STOP_FAILURE
from baseline_tactile_inference import sha256_file
from evaluate_samples import output_name

LEGACY_FIELDS = [
    "sample_id",
    "source_path",
    "dataset_role",
    "prediction_tactile_present",
    "prediction_mask_coverage",
    "ground_truth_tactile_present",
    "ground_truth_mask_path",
    "review_status",
    "notes",
]
FIELDS = [
    "sample_id",
    "source_path",
    "source_sha256",
    "dataset_role",
    "prediction_tactile_present",
    "prediction_mask_coverage",
    "ground_truth_tactile_present",
    "ground_truth_mask_path",
    "ground_truth_mask_sha256",
    "annotation_origin",
    "annotation_id",
    "review_status",
    "reviewer",
    "reviewed_at",
    "notes",
]
ANNOTATION_ORIGINS = {"ai_assisted", "human"}
REVIEW_STATUSES = {"unreviewed", "provisional", "human_reviewed", "ambiguous"}


def build_rows(report: dict[str, Any], manifest_path: Path) -> list[dict[str, str]]:
    if report.get("dataset_role") != "test_only":
        raise ValueError("batch report must declare dataset_role=test_only")
    images = report.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("batch report has no image rows")
    mask_dir = manifest_path.parent / "masks"
    rows = []
    for image in images:
        source = Path(str(image["source"]))
        mask_name = Path(output_name(source)).with_suffix(".png").name
        rows.append(
            {
                "sample_id": source.name,
                "source_path": str(source),
                "source_sha256": str(image.get("source_sha256") or sha256_file(source)),
                "dataset_role": "test_only",
                "prediction_tactile_present": str(
                    int(int(image.get("tactile_instance_count", 0)) > 0)
                ),
                "prediction_mask_coverage": str(
                    float(image.get("tactile_mask_coverage", 0.0))
                ),
                "ground_truth_tactile_present": "",
                "ground_truth_mask_path": str(mask_dir / mask_name),
                "ground_truth_mask_sha256": "",
                "annotation_origin": "",
                "annotation_id": "",
                "review_status": "unreviewed",
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise FileExistsError(f"manifest already exists; refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def replace_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in LEGACY_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest missing columns: {missing}")
        return [{field: row.get(field, "") for field in FIELDS} for row in reader]


def valid_utc_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def mask_errors(mask_path: Path, source: np.ndarray, positive: bool) -> list[str]:
    if not mask_path.is_file():
        return ["positive row requires a mask file"] if positive else []
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ["mask cannot be decoded"]
    errors = []
    if mask.shape != source.shape[:2]:
        errors.append(
            f"mask dimensions {mask.shape} do not match source {source.shape[:2]}"
        )
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 255}):
        errors.append(f"mask must be binary 0/255, got values {sorted(values)}")
    if positive and not np.any(mask == 255):
        errors.append("positive mask contains no foreground pixels")
    if not positive and np.any(mask != 0):
        errors.append("negative row mask must be empty")
    return errors


def validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_rows": len(rows),
        "valid_annotated_rows": 0,
        "valid_reviewed_rows": 0,
        "provisional_rows": 0,
        "human_reviewed_rows": 0,
        "ambiguous_rows": 0,
        "incomplete_rows": 0,
        "invalid_rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "errors": [],
    }
    for index, row in enumerate(rows, start=2):
        row_errors: list[str] = []
        sample_id = row.get("sample_id", "")
        source = cv2.imread(row.get("source_path", ""))
        if source is None:
            row_errors.append("source image cannot be decoded")
        else:
            expected_source_hash = row.get("source_sha256", "").lower()
            if not expected_source_hash:
                row_errors.append("source_sha256 is required")
            elif expected_source_hash != sha256_file(Path(row["source_path"])):
                row_errors.append("source_sha256 mismatch")
        if row.get("dataset_role") != "test_only":
            row_errors.append("dataset_role must remain test_only")

        review_status = row.get("review_status", "")
        if review_status not in REVIEW_STATUSES:
            row_errors.append(f"review_status must be one of {sorted(REVIEW_STATUSES)}")
        elif review_status in {"unreviewed", "ambiguous"}:
            summary["incomplete_rows"] += 1
            if review_status == "ambiguous":
                summary["ambiguous_rows"] += 1
        else:
            origin = row.get("annotation_origin", "")
            if origin not in ANNOTATION_ORIGINS:
                row_errors.append(
                    f"annotation_origin must be one of {sorted(ANNOTATION_ORIGINS)}"
                )
            if not row.get("annotation_id", "").strip():
                row_errors.append("annotation_id is required")
            presence = row.get("ground_truth_tactile_present", "").strip()
            if presence not in {"0", "1"}:
                row_errors.append("annotated ground_truth_tactile_present must be 0 or 1")
            elif source is not None:
                mask_path = Path(row.get("ground_truth_mask_path", ""))
                row_errors.extend(
                    mask_errors(
                        mask_path,
                        source,
                        positive=presence == "1",
                    )
                )
                expected_mask_hash = row.get("ground_truth_mask_sha256", "").lower()
                if mask_path.is_file():
                    if not expected_mask_hash:
                        row_errors.append("ground_truth_mask_sha256 is required")
                    elif expected_mask_hash != sha256_file(mask_path):
                        row_errors.append("ground_truth_mask_sha256 mismatch")
                elif expected_mask_hash:
                    row_errors.append("ground_truth_mask_sha256 set but mask is absent")
            if review_status == "human_reviewed":
                if not row.get("reviewer", "").strip():
                    row_errors.append("human_reviewed row requires reviewer")
                if not valid_utc_timestamp(row.get("reviewed_at", "")):
                    row_errors.append("human_reviewed row requires timezone-aware reviewed_at")
            elif row.get("reviewer", "").strip() or row.get("reviewed_at", "").strip():
                row_errors.append("provisional row cannot claim reviewer or reviewed_at")

        if row_errors:
            summary["invalid_rows"] += 1
            summary["errors"].extend(
                f"row {index} ({sample_id}): {message}" for message in row_errors
            )
        elif review_status in {"provisional", "human_reviewed"}:
            summary["valid_annotated_rows"] += 1
            if review_status == "provisional":
                summary["provisional_rows"] += 1
            else:
                summary["human_reviewed_rows"] += 1
                summary["valid_reviewed_rows"] += 1
            key = "positive_rows" if row["ground_truth_tactile_present"] == "1" else "negative_rows"
            summary[key] += 1
    summary["official_ready"] = bool(
        summary["total_rows"]
        and not summary["invalid_rows"]
        and not summary["incomplete_rows"]
        and summary["human_reviewed_rows"] == summary["total_rows"]
    )
    return summary


def mark_human_reviewed(
    manifest_path: Path, samples: list[str], reviewer: str
) -> dict[str, Any]:
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer must not be empty")
    rows = read_manifest(manifest_path)
    selected = set(samples)
    available = {row["sample_id"] for row in rows}
    missing = selected - available
    if missing:
        raise ValueError(f"samples absent from manifest: {sorted(missing)}")
    reviewed_at = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if row["sample_id"] not in selected:
            continue
        if row["review_status"] != "provisional":
            raise ValueError(f"sample is not provisional: {row['sample_id']}")
        row["review_status"] = "human_reviewed"
        row["reviewer"] = reviewer
        row["reviewed_at"] = reviewed_at
    summary = validate_rows(rows)
    if summary["invalid_rows"]:
        raise ValueError(f"review update failed validation: {summary['errors']}")
    replace_manifest(manifest_path, rows)
    return {"reviewed_samples": len(selected), "reviewer": reviewer, "reviewed_at": reviewed_at}


def mark_ambiguous(
    manifest_path: Path, samples: list[str], reason: str
) -> dict[str, Any]:
    reason = reason.strip()
    if not reason:
        raise ValueError("ambiguity reason must not be empty")
    rows = read_manifest(manifest_path)
    selected = set(samples)
    available = {row["sample_id"] for row in rows}
    missing = selected - available
    if missing:
        raise ValueError(f"samples absent from manifest: {sorted(missing)}")
    for row in rows:
        if row["sample_id"] not in selected:
            continue
        if row["review_status"] != "provisional":
            raise ValueError(f"sample is not provisional: {row['sample_id']}")
        row["review_status"] = "ambiguous"
        row["reviewer"] = ""
        row["reviewed_at"] = ""
        row["notes"] = f"Ambiguous provisional annotation: {reason}"
    summary = validate_rows(rows)
    if summary["invalid_rows"]:
        raise ValueError(f"ambiguity update failed validation: {summary['errors']}")
    replace_manifest(manifest_path, rows)
    return {"ambiguous_samples": len(selected), "reason": reason}


def prepare(report_path: Path, manifest_path: Path) -> int:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = build_rows(report, manifest_path)
    write_manifest(manifest_path, rows)
    (manifest_path.parent / "masks").mkdir(parents=True, exist_ok=True)
    print(json.dumps({"manifest": str(manifest_path), "rows": len(rows)}, indent=2))
    return 0


def validate(manifest_path: Path) -> int:
    summary = validate_rows(read_manifest(manifest_path))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["invalid_rows"]:
        return 1
    return 0 if summary["official_ready"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and validate human-authored tactile ground truth."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--report", type=Path, default=Path("runs/sample-evaluation/report.json")
    )
    prepare_parser.add_argument(
        "--manifest", type=Path, default=Path("data/ground_truth/manifest.csv")
    )
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--manifest", type=Path, default=Path("data/ground_truth/manifest.csv")
    )
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument(
        "--manifest", type=Path, default=Path("data/ground_truth/manifest.csv")
    )
    review_parser.add_argument("--reviewer", required=True)
    review_parser.add_argument("--samples", nargs="+", required=True)
    ambiguous_parser = subparsers.add_parser("ambiguous")
    ambiguous_parser.add_argument(
        "--manifest", type=Path, default=Path("data/ground_truth/manifest.csv")
    )
    ambiguous_parser.add_argument("--reason", required=True)
    ambiguous_parser.add_argument("--samples", nargs="+", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.command == "prepare":
        return prepare(args.report, args.manifest)
    if args.command == "review":
        print(
            json.dumps(
                mark_human_reviewed(args.manifest, args.samples, args.reviewer),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "ambiguous":
        print(
            json.dumps(
                mark_ambiguous(args.manifest, args.samples, args.reason),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    return validate(args.manifest)


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
