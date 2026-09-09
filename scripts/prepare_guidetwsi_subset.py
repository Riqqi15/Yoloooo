from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import cv2

from acquire_guidetwsi_subset import DATASET_HANDLE, IMAGE_SUFFIXES, validate_source
from build_training_manifest import image_dhash
from model_manifest import sha256_file


def parse_yolo_polygons(path: Path) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        tokens = raw_line.split()
        if not tokens:
            continue
        if tokens[0] != "0":
            raise ValueError(f"{path}:{line_number}: expected class 0")
        if len(tokens) < 7 or len(tokens[1:]) % 2:
            raise ValueError(f"{path}:{line_number}: invalid polygon")
        try:
            values = [float(token) for token in tokens[1:]]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: nonnumeric polygon") from error
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{path}:{line_number}: coordinates must be normalized")
        points = list(zip(values[::2], values[1::2]))
        area = abs(
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
            )
        ) / 2.0
        if area <= 1e-8:
            raise ValueError(f"{path}:{line_number}: degenerate polygon")
        polygons.append(points)
    return polygons


def _local_file(cache_root: Path, record: dict[str, Any]) -> Path:
    relative = PurePosixPath(str(record["local_cache_key"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe cache key: {relative}")
    path = cache_root.joinpath(*relative.parts)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"byte-size mismatch: {relative}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    return path


def _source_group(upstream_path: str) -> str:
    parts = PurePosixPath(upstream_path).parts
    rbar_index = next(
        index for index, part in enumerate(parts) if "rbar" in part.lower()
    )
    ignored = {"images", "labels", "yolo_labels", "train"}
    for part in parts[rbar_index + 1 : -1]:
        if part.lower() not in ignored and "label" not in part.lower():
            return part
    return "rbar_train"


def _reference_hashes(paths: list[Path]) -> tuple[set[str], list[int]]:
    images = [path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return {sha256_file(path) for path in images}, [image_dhash(path) for path in images]


def prepare_subset(
    provenance_path: Path,
    cache_root: Path,
    protected_paths: list[Path],
    existing_paths: list[Path],
    limit: int = 2000,
    seed: int = 42,
    near_duplicate_threshold: int = 5,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    validate_source(provenance["dataset_handle"], provenance["license"])
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in provenance["files"]:
        role = str(record["role"])
        if role not in {"image", "label"}:
            raise ValueError(f"invalid provenance role: {role}")
        slot = grouped.setdefault(str(record["pair_key"]), {})
        if role in slot:
            raise ValueError(f"duplicate provenance role: {record['pair_key']} {role}")
        slot[role] = record

    protected_exact, protected_near = _reference_hashes(protected_paths)
    existing_exact, existing_near = _reference_hashes(existing_paths)
    rejected: Counter[str] = Counter()
    retained: list[dict[str, Any]] = []
    retained_exact: set[str] = set()
    retained_near: list[int] = []
    ranked = sorted(
        grouped.items(),
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).digest(),
    )
    for pair_key, pair in ranked:
        if len(retained) == limit:
            break
        if set(pair) != {"image", "label"}:
            rejected["unpaired"] += 1
            continue
        try:
            image_path = _local_file(cache_root, pair["image"])
            label_path = _local_file(cache_root, pair["label"])
            if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                raise ValueError("corrupt image")
            polygons = parse_yolo_polygons(label_path)
            if not polygons:
                raise ValueError("empty annotation")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            rejected["invalid_pair"] += 1
            continue

        image_sha = pair["image"]["sha256"]
        if image_sha in protected_exact:
            rejected["protected_exact"] += 1
            continue
        if image_sha in existing_exact:
            rejected["station_exact"] += 1
            continue
        if image_sha in retained_exact:
            rejected["public_exact"] += 1
            continue
        perceptual = image_dhash(image_path)
        if any((perceptual ^ other).bit_count() <= near_duplicate_threshold for other in protected_near):
            rejected["protected_near"] += 1
            continue
        if any((perceptual ^ other).bit_count() <= near_duplicate_threshold for other in existing_near):
            rejected["station_near"] += 1
            continue
        if any((perceptual ^ other).bit_count() <= near_duplicate_threshold for other in retained_near):
            rejected["public_near"] += 1
            continue

        retained_exact.add(image_sha)
        retained_near.append(perceptual)
        retained.append(
            {
                "sample_id": image_sha,
                "source": DATASET_HANDLE,
                "source_group": _source_group(pair["image"]["upstream_path"]),
                "split": "train",
                "image_cache_key": pair["image"]["local_cache_key"],
                "image_sha256": image_sha,
                "label_cache_key": pair["label"]["local_cache_key"],
                "label_sha256": pair["label"]["sha256"],
                "polygon_count": len(polygons),
            }
        )

    return {
        "manifest_version": 1,
        "dataset_version": "guidetwsi-rbar-2k-v1",
        "source_dataset": DATASET_HANDLE,
        "source_dataset_version": provenance["dataset_version"],
        "license": provenance["license"],
        "selection_seed": seed,
        "near_duplicate_algorithm": "dHash-64",
        "near_duplicate_threshold": near_duplicate_threshold,
        "summary": {
            "candidate_pairs": len(grouped),
            "retained_samples": len(retained),
            "retained_polygons": sum(sample["polygon_count"] for sample in retained),
            "source_groups": dict(Counter(sample["source_group"] for sample in retained)),
            "rejected": dict(rejected),
        },
        "samples": retained,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and curate a GuideTWSI RBar subset.")
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/public/guidetwsi-rbar-v1/provenance.json"),
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("runs/public-data-cache/guidetwsi-rbar-v1")
    )
    parser.add_argument("--protected-dir", type=Path, default=Path("data/samples"))
    parser.add_argument(
        "--existing-images",
        type=Path,
        default=Path("artifacts/datasets/station-tactile-v2/tactile/images"),
    )
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-threshold", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/manifests/guidetwsi-rbar-2k-v1.json"),
    )
    return parser.parse_args()


def _images_under(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ] if root.is_dir() else []


def main() -> int:
    args = parse_args()
    try:
        manifest = prepare_subset(
            args.provenance,
            args.cache,
            _images_under(args.protected_dir),
            _images_under(args.existing_images),
            args.limit,
            args.seed,
            args.near_duplicate_threshold,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps(manifest["summary"], indent=2))
        return 0
    except Exception as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
