from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from baseline_object_inference import IMAGE_SUFFIXES, STOP_FAILURE
from build_training_manifest import image_dhash
from model_manifest import sha256_file
from training_label_gate import (
    annotation_path,
    protected_images,
    read_json,
    read_reviews,
    validate_training_dataset,
    write_reviews,
)


def check_no_cross_split_leakage(
    samples: list[dict[str, Any]], threshold: int = 5
) -> None:
    seen_ids: dict[str, str] = {}
    hashes: list[tuple[int, dict[str, Any]]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(
                f"exact duplicate content: {sample_id} also appears in {seen_ids[sample_id]}"
            )
        seen_ids[sample_id] = str(sample["source_path"])
        perceptual_hash = image_dhash(Path(sample["source_path"]))
        sample["near_duplicate_of"] = ""
        sample["near_duplicate_distance"] = None
        for prior_hash, prior in hashes:
            distance = (perceptual_hash ^ prior_hash).bit_count()
            if distance <= threshold:
                if sample["split"] != prior["split"]:
                    raise ValueError(
                        "near-duplicate crosses splits: "
                        f"{Path(sample['source_path']).name} vs "
                        f"{Path(prior['source_path']).name} (distance={distance})"
                    )
                sample["near_duplicate_of"] = prior["sample_id"]
                sample["near_duplicate_distance"] = distance
                break
        hashes.append((perceptual_hash, sample))


def merge_reviewed_sets(
    inputs: list[tuple[Path, Path, Path]],
    output_manifest: Path,
    output_annotations: Path,
    output_reviews: Path,
    taxonomy: Path,
    protected_dir: Path,
    dataset_version: str,
    near_duplicate_threshold: int = 5,
) -> dict[str, Any]:
    if output_annotations.exists():
        raise FileExistsError(f"output annotations already exist: {output_annotations}")
    protected = protected_images(protected_dir)
    protected_hashes = {sha256_file(path) for path in protected}
    samples: list[dict[str, Any]] = []
    annotations: dict[str, Path] = {}
    reviews: dict[str, dict[str, str]] = {}
    source_versions: list[str] = []

    for manifest_path, annotations_dir, review_path in inputs:
        validation = validate_training_dataset(
            manifest_path,
            annotations_dir,
            review_path,
            taxonomy,
            protected,
            True,
        )
        if not validation["training_ready"]:
            raise ValueError(f"input dataset is not training-ready: {manifest_path}")
        manifest = read_json(manifest_path)
        source_versions.append(str(manifest["dataset_version"]))
        input_reviews = read_reviews(review_path)
        for original in manifest["samples"]:
            sample = dict(original)
            sample_id = str(sample["sample_id"])
            if sample_id in annotations:
                raise ValueError(f"exact duplicate content: {sample_id}")
            if sample_id in protected_hashes:
                raise ValueError(f"training image duplicates protected test content: {sample_id}")
            annotation = annotation_path(annotations_dir, sample_id)
            if not annotation.is_file() or sample_id not in input_reviews:
                raise ValueError(f"missing reviewed annotation: {sample_id}")
            sample["annotation_status"] = "human_reviewed"
            samples.append(sample)
            annotations[sample_id] = annotation
            reviews[sample_id] = input_reviews[sample_id]

    samples.sort(key=lambda sample: sample["sample_id"])
    check_no_cross_split_leakage(samples, near_duplicate_threshold)
    split_counts = Counter(sample["split"] for sample in samples)
    if set(split_counts) != {"train", "validation", "test"}:
        raise ValueError("merged dataset must contain train, validation, and test splits")
    manifest = {
        "manifest_version": 1,
        "dataset_version": dataset_version,
        "dataset_scope": "station_environment",
        "source_dataset_versions": source_versions,
        "split_strategy": "preserved reviewed source-group splits",
        "near_duplicate_algorithm": "dHash-64",
        "near_duplicate_threshold": near_duplicate_threshold,
        "protected_test_hash_count": len(protected_hashes),
        "summary": {
            "total_samples": len(samples),
            "split_counts": dict(split_counts),
            "context_counts": dict(Counter(sample["location_type"] for sample in samples)),
            "task_counts": dict(Counter(sample["dataset_task"] for sample in samples)),
            "near_duplicate_candidates": sum(
                bool(sample["near_duplicate_of"]) for sample in samples
            ),
        },
        "samples": samples,
    }

    output_annotations.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_annotations.name}.", dir=output_annotations.parent
        )
    )
    try:
        for sample_id, source in annotations.items():
            shutil.copy2(source, staging / f"{sample_id}.json")
        staging.replace(output_annotations)
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = output_manifest.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary_manifest.replace(output_manifest)
        write_reviews(output_reviews, reviews)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed station training sets.")
    parser.add_argument(
        "--input",
        action="append",
        nargs=3,
        metavar=("MANIFEST", "ANNOTATIONS", "REVIEWS"),
        required=True,
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-reviews", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument(
        "--taxonomy", type=Path, default=Path("data/training/taxonomy_tactile_v1.json")
    )
    parser.add_argument("--protected-test-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--near-duplicate-threshold", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = merge_reviewed_sets(
            [(Path(a), Path(b), Path(c)) for a, b, c in args.input],
            args.output_manifest,
            args.output_annotations,
            args.output_reviews,
            args.taxonomy,
            args.protected_test_dir,
            args.dataset_version,
            args.near_duplicate_threshold,
        )
        print(json.dumps(manifest["summary"], indent=2))
        return 0
    except Exception as error:
        print(f"{STOP_FAILURE}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
