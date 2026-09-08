from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from baseline_object_inference import STOP_FAILURE
from baseline_tactile_inference import sha256_file
from evaluate_samples import output_name
from ground_truth_manifest import (
    ANNOTATION_ORIGINS,
    read_manifest,
    replace_manifest,
    validate_rows,
)

GROUND_TRUTH_LABEL = "tactile_paving"


def annotation_mask(data: dict[str, Any], source: np.ndarray, annotation: Path) -> np.ndarray:
    height, width = source.shape[:2]
    if int(data.get("imageHeight", height)) != height or int(
        data.get("imageWidth", width)
    ) != width:
        raise ValueError(f"image dimensions mismatch: {annotation}")
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"shapes must be a list: {annotation}")
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        label = shape.get("label")
        if label != GROUND_TRUTH_LABEL:
            raise ValueError(f"unknown label {label!r}: {annotation}")
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"shape must be polygon: {annotation}")
        points = np.asarray(shape.get("points", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
            raise ValueError(f"polygon needs at least three points: {annotation}")
        if not np.isfinite(points).all():
            raise ValueError(f"polygon contains non-finite coordinates: {annotation}")
        if (
            (points[:, 0] < 0).any()
            or (points[:, 0] > width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] > height).any()
        ):
            raise ValueError(f"polygon coordinates outside image: {annotation}")
        raster_points = np.rint(points).astype(np.int32)
        raster_points[:, 0] = np.clip(raster_points[:, 0], 0, width - 1)
        raster_points[:, 1] = np.clip(raster_points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [raster_points], 255)
    return mask


def write_mask_atomic(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.png")
    try:
        if not cv2.imwrite(str(temporary), mask):
            raise OSError(f"unable to write mask: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def import_annotations(
    annotations_dir: Path,
    manifest_path: Path,
    overwrite: bool = False,
    annotation_origin: str = "ai_assisted",
    batch_id: str | None = None,
) -> dict[str, Any]:
    if annotation_origin not in ANNOTATION_ORIGINS:
        raise ValueError(f"annotation_origin must be one of {sorted(ANNOTATION_ORIGINS)}")
    batch_id = batch_id or f"import-{uuid.uuid4().hex}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", batch_id):
        raise ValueError("batch_id may contain only letters, numbers, dot, underscore, and dash")
    if not annotations_dir.is_dir():
        raise NotADirectoryError(f"annotation directory not found: {annotations_dir}")
    annotation_paths = sorted(annotations_dir.glob("*.json"), key=lambda path: path.name.casefold())
    if not annotation_paths:
        raise ValueError(f"no Labelme JSON files in {annotations_dir}")

    rows = read_manifest(manifest_path)
    rows_by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = Path(row["source_path"]).name
        if name in rows_by_name:
            raise ValueError(f"duplicate source basename in manifest: {name}")
        rows_by_name[name] = row

    prepared: list[tuple[dict[str, str], Path, np.ndarray]] = []
    seen: set[str] = set()
    for annotation_path in annotation_paths:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_name = Path(str(data.get("imagePath", ""))).name
        if not image_name:
            raise ValueError(f"imagePath missing: {annotation_path}")
        if image_name in seen:
            raise ValueError(f"duplicate annotation for image: {image_name}")
        seen.add(image_name)
        if image_name not in rows_by_name:
            raise ValueError(f"annotation image absent from manifest: {image_name}")
        row = rows_by_name[image_name]
        if row["review_status"] != "unreviewed" and not overwrite:
            raise ValueError(f"row already annotated; use --overwrite: {image_name}")
        source = cv2.imread(row["source_path"])
        if source is None:
            raise ValueError(f"source image cannot be decoded: {row['source_path']}")
        mask = annotation_mask(data, source, annotation_path)
        prepared.append((row, annotation_path, mask))

    batches_dir = manifest_path.parent / "masks" / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    final_batch_dir = batches_dir / batch_id
    if final_batch_dir.exists():
        raise FileExistsError(f"batch already exists: {final_batch_dir}")
    staging_dir = Path(tempfile.mkdtemp(prefix=".import-", dir=batches_dir))
    manifest_committed = False
    try:
        for row, annotation_path, mask in prepared:
            mask_name = Path(output_name(Path(row["source_path"]))).with_suffix(".png").name
            staged_mask = staging_dir / mask_name
            write_mask_atomic(staged_mask, mask)
            row["source_sha256"] = sha256_file(Path(row["source_path"]))
            row["ground_truth_tactile_present"] = str(int(np.any(mask)))
            row["ground_truth_mask_path"] = str(final_batch_dir / mask_name)
            row["ground_truth_mask_sha256"] = sha256_file(staged_mask)
            row["annotation_origin"] = annotation_origin
            row["annotation_id"] = sha256_file(annotation_path)
            row["review_status"] = "provisional"
            row["reviewer"] = ""
            row["reviewed_at"] = ""
            row["notes"] = "Imported from Labelme; pending human review"
        staging_dir.replace(final_batch_dir)
        validation = validate_rows(rows)
        if validation["invalid_rows"]:
            raise ValueError(f"imported annotations failed validation: {validation['errors']}")
        replace_manifest(manifest_path, rows)
        manifest_committed = True
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if final_batch_dir.exists() and not manifest_committed:
            shutil.rmtree(final_batch_dir)
    return {
        "imported_rows": len(prepared),
        "positive_rows": sum(int(np.any(mask)) for _row, _path, mask in prepared),
        "negative_rows": sum(int(not np.any(mask)) for _row, _path, mask in prepared),
        "remaining_unreviewed": validation["incomplete_rows"],
        "review_status": "provisional",
        "batch_id": batch_id,
        "batch_dir": str(final_batch_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import human-drawn Labelme polygons as tactile ground truth."
    )
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/ground_truth/labelme")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/ground_truth/manifest.csv")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--annotation-origin",
        choices=sorted(ANNOTATION_ORIGINS),
        default="ai_assisted",
    )
    parser.add_argument("--batch-id")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    summary = import_annotations(
        args.annotations,
        args.manifest,
        args.overwrite,
        args.annotation_origin,
        args.batch_id,
    )
    print(json.dumps(summary, indent=2))
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
