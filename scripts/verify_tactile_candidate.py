from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from baseline_object_inference import STOP_FAILURE
from model_manifest import sha256_file


REQUIRED_FILES = (
    "best.pt",
    "metrics.json",
    "training_config.json",
    "environment.txt",
    "sha256.txt",
    "MODEL_CARD.md",
)
CONFIG_KEYS = {"dataset_version", "seed", "imgsz", "epochs", "ultralytics_version"}
EXPECTED_LABELS = {0: "tactile_paving"}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return payload


def normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        try:
            return {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("model label indices must be integers") from error
    raise ValueError("model labels must be a list or object")


def validate_candidate(candidate_dir: Path) -> dict[str, object]:
    candidate_dir = candidate_dir.resolve()
    if not candidate_dir.is_dir():
        raise ValueError(f"candidate directory not found: {candidate_dir}")

    for name in REQUIRED_FILES:
        path = candidate_dir / name
        if not path.is_file():
            raise ValueError(f"missing required file: {name}")
        if path.stat().st_size == 0:
            raise ValueError(f"required file is empty: {name}")

    checkpoint = candidate_dir / "best.pt"
    checksum_fields = (candidate_dir / "sha256.txt").read_text(encoding="utf-8").split()
    if len(checksum_fields) != 2 or checksum_fields[1] != "best.pt":
        raise ValueError("sha256.txt must contain '<sha256>  best.pt'")
    actual_checksum = sha256_file(checkpoint)
    if checksum_fields[0].lower() != actual_checksum:
        raise ValueError("checkpoint checksum mismatch")

    metrics = read_json(candidate_dir / "metrics.json")
    if not metrics:
        raise ValueError("metrics.json must not be empty")
    config = read_json(candidate_dir / "training_config.json")
    missing = sorted(CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(f"training config missing keys: {', '.join(missing)}")
    if not isinstance(config["dataset_version"], str) or not config["dataset_version"]:
        raise ValueError("dataset_version must be a nonempty string")
    for key in ("seed", "imgsz", "epochs"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["ultralytics_version"] != "8.4.138":
        raise ValueError("ultralytics_version must be 8.4.138")

    model = YOLO(str(checkpoint))
    if model.task != "segment":
        raise ValueError("model task must be segment")
    labels = normalize_names(model.names)
    if labels != EXPECTED_LABELS:
        raise ValueError(f"model labels must be {EXPECTED_LABELS}")

    return {
        "status": "candidate_valid",
        "candidate_dir": str(candidate_dir),
        "checkpoint_sha256": actual_checksum,
        "dataset_version": config["dataset_version"],
        "ultralytics_version": config["ultralytics_version"],
        "task": model.task,
        "labels": labels,
        "warning": "Candidate integrity only; not approved for deployment.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a one-class tactile segmentation candidate bundle."
    )
    parser.add_argument("candidate_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        print(json.dumps(validate_candidate(parse_args().candidate_dir), indent=2))
        return 0
    except KeyboardInterrupt:
        print(f"{STOP_FAILURE}: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"{STOP_FAILURE}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
