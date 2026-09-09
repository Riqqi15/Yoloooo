from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from acquire_guidetwsi_subset import IMAGE_SUFFIXES
from model_manifest import sha256_file


def _link(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _cache_path(cache_root: Path, key: str) -> Path:
    relative = PurePosixPath(key)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe cache key: {key}")
    return cache_root.joinpath(*relative.parts)


def _station_pairs(station_root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = station_root / "images" / split
    label_dir = station_root / "labels" / split
    pairs = []
    for image in sorted(image_dir.glob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"missing station label: {label}")
        pairs.append((image, label))
    if not pairs:
        raise ValueError(f"station {split} split is empty")
    return pairs


def build_dataset(
    public_manifest_path: Path,
    public_cache: Path,
    station_root: Path,
    output_root: Path,
    public_to_station: int,
    public_validation_count: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    if public_to_station not in {2, 4}:
        raise ValueError("public_to_station must be 2 or 4")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    public = sorted(
        public_manifest["samples"],
        key=lambda sample: hashlib.sha256(
            f"{seed}:{sample['sample_id']}".encode()
        ).digest(),
    )
    if len(public) < 2:
        raise ValueError("at least two public samples are required")
    validation_count = public_validation_count
    if validation_count is None:
        validation_count = max(1, round(len(public) * 0.1))
    if not 0 < validation_count < len(public):
        raise ValueError("public validation count must leave nonempty train data")
    public_validation = public[:validation_count]
    public_train = public[validation_count:]
    station_train = _station_pairs(station_root, "train")
    station_validation = _station_pairs(station_root, "val")
    station_needed = max(1, round(len(public_train) / public_to_station))

    output_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for split, samples in (("train", public_train), ("val", public_validation)):
            for sample in samples:
                image = _cache_path(public_cache, sample["image_cache_key"])
                label = _cache_path(public_cache, sample["label_cache_key"])
                if sha256_file(image) != sample["image_sha256"]:
                    raise ValueError(f"public image hash mismatch: {sample['sample_id']}")
                if sha256_file(label) != sample["label_sha256"]:
                    raise ValueError(f"public label hash mismatch: {sample['sample_id']}")
                name = f"public-{sample['sample_id']}"
                _link(image, output_root / "images" / split / f"{name}{image.suffix.lower()}")
                _link(label, output_root / "labels" / split / f"{name}.txt")
                records.append({"split": split, "source": "public", "sample_id": sample["sample_id"]})

        ranked_station = sorted(
            station_train,
            key=lambda pair: hashlib.sha256(f"{seed}:{pair[0].stem}".encode()).digest(),
        )
        for index in range(station_needed):
            image, label = ranked_station[index % len(ranked_station)]
            name = f"station-{index:04d}-{image.stem}"
            _link(image, output_root / "images" / "train" / f"{name}{image.suffix.lower()}")
            _link(label, output_root / "labels" / "train" / f"{name}.txt")
            records.append({"split": "train", "source": "station", "sample_id": image.stem})
        for image, label in station_validation:
            name = f"station-{image.stem}"
            _link(image, output_root / "images" / "val" / f"{name}{image.suffix.lower()}")
            _link(label, output_root / "labels" / "val" / f"{name}.txt")
            records.append({"split": "val", "source": "station", "sample_id": image.stem})

        yaml_path = output_root.resolve().as_posix().replace("'", "''")
        (output_root / "dataset.yaml").write_text(
            f"path: '{yaml_path}'\ntrain: images/train\nval: images/val\nnames:\n  0: tactile_paving\n",
            encoding="utf-8",
        )
        report = {
            "manifest_version": 1,
            "dataset_version": f"tactile-v3-public{public_to_station}-station1",
            "public_dataset_version": public_manifest["dataset_version"],
            "station_dataset_version": "station-tactile-v2",
            "selection_seed": seed,
            "public_to_station": public_to_station,
            "summary": {
                "train_public": len(public_train),
                "train_station": station_needed,
                "validation_public": len(public_validation),
                "validation_station": len(station_validation),
            },
            "samples": records,
        }
        (output_root / "export_manifest.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tactile v3 YOLO dataset mixtures.")
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=Path("data/training/manifests/guidetwsi-rbar-2k-v1.json"),
    )
    parser.add_argument(
        "--public-cache",
        type=Path,
        default=Path("runs/public-data-cache/guidetwsi-rbar-v1"),
    )
    parser.add_argument(
        "--station-root",
        type=Path,
        default=Path("artifacts/datasets/station-tactile-v2/tactile"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--public-to-station", type=int, choices=(2, 4), required=True)
    parser.add_argument("--public-validation-count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_dataset(
            args.public_manifest,
            args.public_cache,
            args.station_root,
            args.output_root,
            args.public_to_station,
            args.public_validation_count,
            args.seed,
        )
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.manifest_output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.manifest_output)
        print(json.dumps(report["summary"], indent=2))
        return 0
    except Exception as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
