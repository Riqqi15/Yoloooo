from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from baseline_object_inference import IMAGE_SUFFIXES, STOP_FAILURE
from evaluate_samples import write_image
from model_manifest import sha256_file


REVIEW_FIELDS = (
    "sample_id",
    "source_sha256",
    "annotation_sha256",
    "review_status",
    "reviewer",
    "reviewed_at",
    "notes",
)
REVIEW_STATUSES = {"human_reviewed", "ambiguous"}
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def read_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(REVIEW_FIELDS) - set(reader.fieldnames or ()):
            raise ValueError(f"invalid review manifest columns: {path}")
        rows = [{field: str(row.get(field, "")).strip() for field in REVIEW_FIELDS} for row in reader]
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate sample_id in review manifest")
    return result


def write_reviews(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(rows[key] for key in sorted(rows))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_taxonomy(path: Path) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    taxonomy = read_json(path)
    objects = taxonomy.get("object_detection")
    tactile = taxonomy.get("tactile_segmentation")
    if not isinstance(objects, list) or not objects or len(objects) != len(set(objects)):
        raise ValueError("invalid object taxonomy")
    if not isinstance(tactile, list) or not tactile or len(tactile) != len(set(tactile)):
        raise ValueError("invalid tactile taxonomy")
    if any(not isinstance(label, str) or not label for label in [*objects, *tactile]):
        raise ValueError("taxonomy labels must be nonempty strings")
    return taxonomy, {name: index for index, name in enumerate(objects)}, {
        name: index for index, name in enumerate(tactile)
    }


def annotation_path(annotations_dir: Path, sample_id: str) -> Path:
    return annotations_dir / f"{sample_id}.json"


def validate_annotation(
    annotation: dict[str, Any],
    sample: dict[str, Any],
    width: int,
    height: int,
    object_labels: dict[str, int],
    tactile_labels: dict[str, int],
) -> None:
    if annotation.get("schema_version") != 1:
        raise ValueError("annotation schema_version must be 1")
    if annotation.get("sample_id") != sample["sample_id"]:
        raise ValueError("annotation sample_id mismatch")
    if annotation.get("source_sha256") != sample["source_sha256"]:
        raise ValueError("annotation source hash mismatch")
    objects = annotation.get("objects")
    tactile = annotation.get("tactile")
    if not isinstance(objects, list) or not isinstance(tactile, list):
        raise ValueError("annotation objects and tactile must be lists")
    task = sample.get("dataset_task")
    if task not in {"object", "tactile", "both"}:
        raise ValueError(f"invalid dataset_task: {task}")
    if task == "object" and tactile:
        raise ValueError("object-only sample contains tactile labels")
    if task == "tactile" and objects:
        raise ValueError("tactile-only sample contains object labels")

    for item in objects:
        if not isinstance(item, dict) or item.get("label") not in object_labels:
            raise ValueError(f"unknown object label: {item.get('label') if isinstance(item, dict) else item}")
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("object bbox must be xyxy with four numbers")
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError) as error:
            raise ValueError("object bbox contains nonnumeric coordinates") from error
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("object bbox contains non-finite coordinates")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("object bbox outside image or has zero area")

    for item in tactile:
        if not isinstance(item, dict) or item.get("label") not in tactile_labels:
            raise ValueError(f"unknown tactile label: {item.get('label') if isinstance(item, dict) else item}")
        try:
            points = np.asarray(item.get("points"), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("tactile polygon contains nonnumeric coordinates") from error
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
            raise ValueError("tactile polygon needs at least three points")
        if not np.isfinite(points).all():
            raise ValueError("tactile polygon contains non-finite coordinates")
        if (
            (points[:, 0] < 0).any()
            or (points[:, 0] >= width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] >= height).any()
        ):
            raise ValueError("tactile polygon outside image")
        if cv2.contourArea(points.astype(np.float32)) <= 0:
            raise ValueError("tactile polygon has zero area")


def valid_review_time(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def validate_training_dataset(
    manifest_path: Path,
    annotations_dir: Path,
    review_path: Path,
    taxonomy_path: Path,
    protected_paths: list[Path],
    require_review: bool = True,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    taxonomy, object_labels, tactile_labels = load_taxonomy(taxonomy_path)
    samples = manifest.get("samples")
    if manifest.get("manifest_version") != 1 or manifest.get("dataset_scope") != "station_environment":
        raise ValueError("invalid station training manifest")
    if not isinstance(samples, list) or not samples:
        raise ValueError("training manifest has no samples")
    reviews = read_reviews(review_path)
    protected = {sha256_file(path) for path in protected_paths if path.is_file()}
    errors: list[dict[str, str]] = []
    statuses: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    tactile_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    condition_counts: dict[str, Counter[str]] = {
        "lighting": Counter(), "motion": Counter(), "surface": Counter(), "location_type": Counter()
    }
    seen_ids: set[str] = set()
    structurally_valid = 0

    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        try:
            if not sample_id or sample_id in seen_ids:
                raise ValueError("missing or duplicate sample_id")
            seen_ids.add(sample_id)
            source = Path(str(sample.get("source_path", "")))
            if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError("source image missing or unsupported")
            source_hash = sha256_file(source)
            if source_hash != sample.get("source_sha256") or source_hash != sample_id:
                raise ValueError("source hash or sample_id mismatch")
            if source_hash in protected:
                raise ValueError("source duplicates protected test content")
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError("source image cannot be decoded")
            if sample.get("split") not in SPLITS:
                raise ValueError("invalid split")
            path = annotation_path(annotations_dir, sample_id)
            annotation = read_json(path)
            validate_annotation(
                annotation, sample, image.shape[1], image.shape[0], object_labels, tactile_labels
            )
            annotation_hash = sha256_file(path)
            structurally_valid += 1
            object_counts.update(item["label"] for item in annotation["objects"])
            tactile_counts.update(item["label"] for item in annotation["tactile"])
            split_counts[sample["split"]] += 1
            for field, counts in condition_counts.items():
                counts[str(sample.get(field, "unknown"))] += 1

            review = reviews.get(sample_id)
            if review is None:
                statuses["unreviewed"] += 1
            else:
                status = review.get("review_status", "")
                if status not in REVIEW_STATUSES:
                    raise ValueError(f"invalid review status: {status}")
                if (
                    review.get("source_sha256") != source_hash
                    or review.get("annotation_sha256") != annotation_hash
                ):
                    raise ValueError("review hash mismatch; annotation needs re-review")
                if not review.get("reviewer") or not valid_review_time(review.get("reviewed_at", "")):
                    raise ValueError("reviewer or timezone-aware reviewed_at missing")
                if status == "ambiguous" and not review.get("notes"):
                    raise ValueError("ambiguous review requires notes")
                statuses[status] += 1
        except Exception as error:
            statuses["invalid"] += 1
            errors.append({"sample_id": sample_id, "error": str(error)})

    taxonomy_approved = taxonomy.get("status") == "approved"
    total = len(samples)
    human_reviewed = statuses["human_reviewed"]
    training_ready = (
        taxonomy_approved
        and not errors
        and structurally_valid == total
        and human_reviewed == total
    )
    if not require_review:
        training_ready = taxonomy_approved and not errors and structurally_valid == total
    return {
        "dataset_version": manifest.get("dataset_version"),
        "dataset_scope": manifest.get("dataset_scope"),
        "taxonomy_version": taxonomy.get("taxonomy_version"),
        "taxonomy_approved": taxonomy_approved,
        "require_review": require_review,
        "training_ready": training_ready,
        "summary": {
            "total_samples": total,
            "structurally_valid_samples": structurally_valid,
            "human_reviewed_samples": human_reviewed,
            "ambiguous_samples": statuses["ambiguous"],
            "unreviewed_samples": statuses["unreviewed"],
            "invalid_samples": statuses["invalid"],
            "split_counts": dict(split_counts),
            "object_class_counts": dict(object_counts),
            "tactile_class_counts": dict(tactile_counts),
            "condition_counts": {field: dict(counts) for field, counts in condition_counts.items()},
        },
        "errors": errors,
    }


def mark_review(
    manifest_path: Path,
    annotations_dir: Path,
    review_path: Path,
    sample_ids: list[str],
    reviewer: str,
    status: str,
    notes: str = "",
) -> None:
    if not reviewer.strip() or status not in REVIEW_STATUSES:
        raise ValueError("reviewer and valid review status are required")
    if status == "ambiguous" and not notes.strip():
        raise ValueError("ambiguous review requires notes")
    manifest = read_json(manifest_path)
    samples = {str(sample["sample_id"]): sample for sample in manifest.get("samples", [])}
    reviews = read_reviews(review_path)
    reviewed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for sample_id in sample_ids:
        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError(f"unknown sample_id: {sample_id}")
        source = Path(str(sample["source_path"]))
        annotation = annotation_path(annotations_dir, sample_id)
        if not source.is_file() or not annotation.is_file():
            raise ValueError(f"source or annotation missing: {sample_id}")
        source_hash = sha256_file(source)
        if source_hash != sample.get("source_sha256"):
            raise ValueError(f"source hash mismatch: {sample_id}")
        reviews[sample_id] = {
            "sample_id": sample_id,
            "source_sha256": source_hash,
            "annotation_sha256": sha256_file(annotation),
            "review_status": status,
            "reviewer": reviewer.strip(),
            "reviewed_at": reviewed_at,
            "notes": notes.strip(),
        }
    write_reviews(review_path, reviews)


def render_qa(
    manifest_path: Path,
    annotations_dir: Path,
    output_dir: Path,
    taxonomy_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    validation = validate_training_dataset(
        manifest_path, annotations_dir, review_path, taxonomy_path, [], require_review=False
    )
    if validation["errors"]:
        raise ValueError(f"annotation validation failed: {len(validation['errors'])} samples")
    manifest = read_json(manifest_path)
    reviews = read_reviews(review_path)
    items = []
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"tactile_guiding_path": (0, 255, 0), "tactile_warning_block": (0, 165, 255)}
    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        source = cv2.imread(sample["source_path"])
        annotation = read_json(annotation_path(annotations_dir, sample_id))
        overlay = source.copy()
        for item in annotation["objects"]:
            x1, y1, x2, y2 = (round(float(value)) for value in item["bbox"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 80, 0), 2)
            cv2.putText(overlay, item["label"], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 0), 1)
        for item in annotation["tactile"]:
            points = np.rint(np.asarray(item["points"])).astype(np.int32)
            color = colors.get(item["label"], (255, 0, 255))
            layer = overlay.copy()
            cv2.fillPoly(layer, [points], color)
            overlay = cv2.addWeighted(overlay, 0.65, layer, 0.35, 0)
            cv2.polylines(overlay, [points], True, color, 2)
        status = reviews.get(sample_id, {}).get("review_status", "unreviewed")
        cv2.rectangle(overlay, (0, 0), (min(overlay.shape[1], 560), 28), (0, 0, 0), -1)
        cv2.putText(overlay, f"{status} | {sample_id[:12]}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        destination = output_dir / f"{sample_id}.jpg"
        write_image(destination, overlay)
        items.append({"sample_id": sample_id, "source": sample["source_path"], "overlay": str(destination), "review_status": status})
    report = {"validation": validation, "items": items}
    (output_dir / "index.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def yaml_text(object_or_tactile: list[str]) -> str:
    lines = ["train: images/train", "val: images/val", "test: images/test", "names:"]
    lines.extend(f"  {index}: {json.dumps(label)}" for index, label in enumerate(object_or_tactile))
    return "\n".join(lines) + "\n"


def export_yolo(
    manifest_path: Path,
    annotations_dir: Path,
    review_path: Path,
    taxonomy_path: Path,
    protected_paths: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"export output already exists: {output_dir}")
    validation = validate_training_dataset(
        manifest_path, annotations_dir, review_path, taxonomy_path, protected_paths, True
    )
    if not validation["training_ready"]:
        raise ValueError("dataset is not training-ready")
    manifest = read_json(manifest_path)
    taxonomy, object_labels, tactile_labels = load_taxonomy(taxonomy_path)
    reviews = read_reviews(review_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    exported = []
    try:
        for task, labels in (("object", taxonomy["object_detection"]), ("tactile", taxonomy["tactile_segmentation"])):
            (staging / task).mkdir(parents=True)
            (staging / task / "data.yaml").write_text(yaml_text(labels), encoding="utf-8")
        for sample in manifest["samples"]:
            sample_id = sample["sample_id"]
            source = Path(sample["source_path"])
            annotation_file = annotation_path(annotations_dir, sample_id)
            annotation = read_json(annotation_file)
            image = cv2.imread(str(source))
            height, width = image.shape[:2]
            split = SPLITS[sample["split"]]
            tasks = ("object", "tactile") if sample["dataset_task"] == "both" else (sample["dataset_task"],)
            for task in tasks:
                image_dir = staging / task / "images" / split
                label_dir = staging / task / "labels" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, image_dir / f"{sample_id}{source.suffix.lower()}")
                lines = []
                if task == "object":
                    for item in annotation["objects"]:
                        x1, y1, x2, y2 = (float(value) for value in item["bbox"])
                        lines.append(
                            f"{object_labels[item['label']]} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}"
                        )
                else:
                    for item in annotation["tactile"]:
                        normalized = " ".join(
                            f"{coordinate:.6f}"
                            for point in item["points"]
                            for coordinate in (float(point[0]) / width, float(point[1]) / height)
                        )
                        lines.append(f"{tactile_labels[item['label']]} {normalized}")
                (label_dir / f"{sample_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            exported.append(
                {
                    "sample_id": sample_id,
                    "source_sha256": sample["source_sha256"],
                    "annotation_sha256": sha256_file(annotation_file),
                    "review": reviews[sample_id],
                    "split": sample["split"],
                    "dataset_task": sample["dataset_task"],
                }
            )
        export_manifest = {
            "dataset_version": manifest["dataset_version"],
            "source_manifest_sha256": sha256_file(manifest_path),
            "taxonomy_sha256": sha256_file(taxonomy_path),
            "review_manifest_sha256": sha256_file(review_path),
            "validation": validation,
            "samples": exported,
        }
        (staging / "export_manifest.json").write_text(json.dumps(export_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        staging.replace(output_dir)
        return export_manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def protected_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed training annotation QA and YOLO export.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/training/annotations"))
    parser.add_argument("--review-manifest", type=Path, default=Path("data/training/review_manifest.csv"))
    parser.add_argument("--taxonomy", type=Path, default=Path("data/training/taxonomy_v1.json"))
    parser.add_argument("--protected-test-dir", type=Path, default=Path("data/samples"))
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--output", type=Path)
    render = commands.add_parser("render")
    render.add_argument("--output-dir", type=Path, default=Path("runs/training-label-review"))
    review = commands.add_parser("review")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--samples", nargs="+", required=True)
    review.add_argument("--notes", default="")
    ambiguous = commands.add_parser("ambiguous")
    ambiguous.add_argument("--reviewer", required=True)
    ambiguous.add_argument("--samples", nargs="+", required=True)
    ambiguous.add_argument("--reason", required=True)
    export = commands.add_parser("export")
    export.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    protected = protected_images(args.protected_test_dir)
    if args.command == "validate":
        result = validate_training_dataset(
            args.manifest, args.annotations_dir, args.review_manifest, args.taxonomy, protected, True
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return int(not result["training_ready"])
    if args.command == "render":
        result = render_qa(args.manifest, args.annotations_dir, args.output_dir, args.taxonomy, args.review_manifest)
        print(json.dumps({"images": len(result["items"]), "output": str(args.output_dir)}, indent=2))
        return 0
    if args.command in {"review", "ambiguous"}:
        mark_review(
            args.manifest,
            args.annotations_dir,
            args.review_manifest,
            args.samples,
            args.reviewer,
            "human_reviewed" if args.command == "review" else "ambiguous",
            args.notes if args.command == "review" else args.reason,
        )
        print(f"recorded {args.command} for {len(args.samples)} sample(s)")
        return 0
    result = export_yolo(
        args.manifest, args.annotations_dir, args.review_manifest, args.taxonomy, protected, args.output_dir
    )
    print(json.dumps({"samples": len(result["samples"]), "output": str(args.output_dir)}, indent=2))
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
