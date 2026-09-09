from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from model_manifest import sha256_file


ULTRALYTICS_VERSION = "8.4.138"


def build_training_config(
    dataset_yaml: Path,
    checkpoint: Path,
    runs_root: Path,
    run_name: str,
    *,
    device: str = "cpu",
    epochs: int = 80,
    batch: int = 8,
) -> dict[str, Any]:
    if not dataset_yaml.is_file():
        raise FileNotFoundError(dataset_yaml)
    export_manifest = dataset_yaml.parent / "export_manifest.json"
    if not export_manifest.is_file():
        raise FileNotFoundError(export_manifest)
    dataset_version = json.loads(export_manifest.read_text(encoding="utf-8")).get(
        "dataset_version"
    )
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("export manifest dataset_version is required")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.name != "yolo11n_tactile.pt" or "guidetwsi" not in {
        part.lower() for part in checkpoint.parts
    }:
        raise ValueError("initial checkpoint must be the GuideTWSI yolo11n_tactile.pt")
    if not run_name.strip():
        raise ValueError("run name is required")
    run_dir = runs_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"run output already exists: {run_dir}")
    return {
        "dataset_yaml": str(dataset_yaml.resolve()),
        "dataset_version": dataset_version,
        "validation_manifest_version": "station-tactile-v2",
        "init_checkpoint": str(checkpoint.resolve()),
        "init_checkpoint_sha256": sha256_file(checkpoint),
        "run_name": run_name,
        "runs_root": str(runs_root.resolve()),
        "seed": 42,
        "imgsz": 640,
        "epochs": epochs,
        "batch": batch,
        "patience": 15,
        "device": device,
        "workers": 0,
        "ultralytics_version": ULTRALYTICS_VERSION,
    }


def train(config: dict[str, Any], candidate_root: Path) -> Path:
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise ValueError(
            f"Ultralytics must be {ULTRALYTICS_VERSION}, got {ultralytics.__version__}"
        )
    candidate_dir = candidate_root / config["run_name"]
    if candidate_dir.exists():
        raise FileExistsError(f"candidate output already exists: {candidate_dir}")
    model = YOLO(config["init_checkpoint"])
    result = model.train(
        data=config["dataset_yaml"],
        project=config["runs_root"],
        name=config["run_name"],
        exist_ok=False,
        seed=config["seed"],
        deterministic=True,
        imgsz=config["imgsz"],
        epochs=config["epochs"],
        batch=config["batch"],
        patience=config["patience"],
        device=config["device"],
        workers=config["workers"],
        task="segment",
        plots=True,
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f"training did not produce best.pt: {best}")
    metrics = {
        key: float(value)
        for key, value in getattr(result, "results_dict", {}).items()
    }
    staging = candidate_root / f".{config['run_name']}.staging"
    if staging.exists():
        raise FileExistsError(f"candidate staging already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        shutil.copy2(best, staging / "best.pt")
        (staging / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (staging / "training_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (staging / "environment.txt").write_text(
            f"Python {platform.python_version()}\n{freeze}", encoding="utf-8"
        )
        (staging / "MODEL_CARD.md").write_text(
            f"# Tactile candidate {config['run_name']}\n\n"
            "Status: candidate only; not approved for deployment.\n\n"
            f"Dataset: {config['dataset_version']}\n"
            "Task: segment\nLabels: tactile_paving\n"
            f"Initialization: GuideTWSI `{config['init_checkpoint_sha256']}`\n\n"
            "This offline model does not validate safe walking.\n",
            encoding="utf-8",
        )
        digest = sha256_file(staging / "best.pt")
        (staging / "sha256.txt").write_text(f"{digest}  best.pt\n", encoding="utf-8")
        candidate_root.mkdir(parents=True, exist_ok=True)
        staging.replace(candidate_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return candidate_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tactile v3 candidate.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/guidetwsi/yolo11n_tactile.pt"),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/segment"))
    parser.add_argument(
        "--candidate-root", type=Path, default=Path("artifacts/candidates")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = build_training_config(
            args.dataset,
            args.checkpoint,
            args.runs_root,
            args.run_name,
            device=args.device,
            epochs=args.epochs,
            batch=args.batch,
        )
        if args.dry_run:
            print(json.dumps(config, indent=2))
            return 0
        print(train(config, args.candidate_root))
        return 0
    except Exception as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
