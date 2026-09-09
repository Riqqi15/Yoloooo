from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from baseline_object_inference import IMAGE_SUFFIXES, STOP_FAILURE
from model_manifest import sha256_file


METADATA_FIELDS = (
    "source_path",
    "session_id",
    "location_id",
    "location_type",
    "source_name",
    "usage_permission",
    "device_model",
    "camera_position",
    "lighting",
    "motion",
    "surface",
    "dataset_task",
    "notes",
)
DATASET_TASKS = {"object", "tactile", "both"}
STATION_CONTEXTS = frozenset(
    {
        "station_interior",
        "station_concourse",
        "station_platform",
        "station_track_area",
        "station_access",
        "station_access_sidewalk",
    }
)


def content_hashes(paths: list[Path]) -> set[str]:
    return {sha256_file(path) for path in paths if path.is_file()}


def image_dhash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"unable to decode image: {path}")
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def assigned_split(location_id: str, session_id: str, seed: int) -> str:
    key = f"{seed}:{location_id}:{session_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100
    if bucket < 70:
        return "train"
    return "validation" if bucket < 85 else "test"


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(METADATA_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"metadata columns missing: {', '.join(sorted(missing))}")
        return [
            {field: str(row.get(field, "")).strip() for field in METADATA_FIELDS}
            for row in reader
        ]


def build_manifest(
    metadata_path: Path,
    source_dir: Path,
    protected_paths: list[Path],
    dataset_version: str,
    seed: int = 42,
    near_duplicate_threshold: int = 5,
) -> dict[str, Any]:
    if not dataset_version.strip():
        raise ValueError("dataset version is required")
    if not 0 <= near_duplicate_threshold <= 64:
        raise ValueError("near-duplicate threshold must be within 0..64")
    source_root = source_dir.resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"training source directory not found: {source_dir}")
    protected = content_hashes(protected_paths)
    rows = read_metadata(metadata_path)
    if not rows:
        raise ValueError("metadata contains no samples")

    seen_hashes: dict[str, str] = {}
    hashes_and_rows: list[tuple[int, dict[str, Any]]] = []
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        missing = [field for field in METADATA_FIELDS[:-1] if not row[field]]
        if missing:
            raise ValueError(f"metadata row {index} missing: {', '.join(missing)}")
        relative = Path(row["source_path"])
        source = (source_root / relative).resolve()
        if not source.is_relative_to(source_root):
            raise ValueError(f"metadata row {index} escapes source directory")
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported or missing image: {source}")
        if row["location_type"] not in STATION_CONTEXTS:
            raise ValueError(f"unapproved station context: {row['location_type']}")
        if row["dataset_task"] not in DATASET_TASKS:
            raise ValueError(f"invalid dataset_task: {row['dataset_task']}")

        digest = sha256_file(source)
        if digest in protected:
            raise ValueError(f"training image duplicates protected test content: {relative}")
        if digest in seen_hashes:
            raise ValueError(
                f"exact duplicate content: {relative} duplicates {seen_hashes[digest]}"
            )
        seen_hashes[digest] = relative.as_posix()
        split = assigned_split(row["location_id"], row["session_id"], seed)
        output = {
            "sample_id": digest,
            "source_path": str(source),
            "source_sha256": digest,
            "split": split,
            "split_group": f"{row['location_id']}::{row['session_id']}",
            "annotation_status": "unreviewed",
            "near_duplicate_of": "",
            "near_duplicate_distance": None,
            **{field: row[field] for field in METADATA_FIELDS if field != "source_path"},
        }
        perceptual_hash = image_dhash(source)
        for prior_hash, prior in hashes_and_rows:
            distance = (perceptual_hash ^ prior_hash).bit_count()
            if distance <= near_duplicate_threshold:
                if output["split"] != prior["split"]:
                    raise ValueError(
                        "near-duplicate crosses splits: "
                        f"{relative} vs {Path(prior['source_path']).name} (distance={distance})"
                    )
                output["near_duplicate_of"] = prior["sample_id"]
                output["near_duplicate_distance"] = distance
                break
        hashes_and_rows.append((perceptual_hash, output))
        output_rows.append(output)

    split_counts = Counter(row["split"] for row in output_rows)
    context_counts = Counter(row["location_type"] for row in output_rows)
    task_counts = Counter(row["dataset_task"] for row in output_rows)
    return {
        "manifest_version": 1,
        "dataset_version": dataset_version,
        "dataset_scope": "station_environment",
        "split_strategy": "sha256(seed:location_id:session_id), 70/15/15",
        "split_seed": seed,
        "near_duplicate_algorithm": "dHash-64",
        "near_duplicate_threshold": near_duplicate_threshold,
        "protected_test_hash_count": len(protected),
        "summary": {
            "total_samples": len(output_rows),
            "split_counts": dict(split_counts),
            "context_counts": dict(context_counts),
            "task_counts": dict(task_counts),
            "near_duplicate_candidates": sum(
                bool(row["near_duplicate_of"]) for row in output_rows
            ),
        },
        "samples": output_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an auditable station-only training intake manifest."
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/training/intake.csv"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/training/inbox"))
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-threshold", type=int, default=5)
    parser.add_argument(
        "--protected-test-dir", type=Path, default=Path("data/samples")
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    protected = sorted(
        path
        for path in args.protected_test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    payload = build_manifest(
        args.metadata,
        args.source_dir,
        protected,
        args.dataset_version,
        args.seed,
        args.near_duplicate_threshold,
    )
    output = args.output or Path("data/training/manifests") / f"{args.dataset_version}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({**payload["summary"], "manifest": str(output)}, indent=2))
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
