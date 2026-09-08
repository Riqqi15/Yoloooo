from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MODEL_MANIFEST = Path("models/baseline_manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_labels(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    items = names.items() if isinstance(names, Mapping) else enumerate(names)
    return {int(index): str(name) for index, name in items}


def load_model_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != 1 or not isinstance(payload.get("models"), dict):
        raise ValueError(f"invalid model manifest: {path}")
    return payload


def verify_model_entry(
    manifest_path: Path, model_key: str, requested_path: Path
) -> tuple[dict[str, Any], Path]:
    manifest_path = manifest_path.resolve()
    payload = load_model_manifest(manifest_path)
    entry = payload["models"].get(model_key)
    if not isinstance(entry, dict):
        raise ValueError(f"model {model_key!r} absent from manifest")

    relative_path = entry.get("artifact_path")
    expected_sha256 = entry.get("sha256")
    expected_size = entry.get("size_bytes")
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        raise ValueError(f"invalid manifest entry for {model_key}")
    expected_path = (manifest_path.parent / relative_path).resolve()
    actual_path = requested_path.resolve()
    if actual_path != expected_path:
        raise ValueError(
            f"{model_key} model path is not pinned by manifest: {requested_path}"
        )
    if not actual_path.is_file():
        raise FileNotFoundError(f"model artifact not found: {actual_path}")
    if not isinstance(expected_size, int) or actual_path.stat().st_size != expected_size:
        raise ValueError(f"{model_key} model size mismatch")
    if sha256_file(actual_path) != expected_sha256:
        raise ValueError(f"{model_key} model SHA-256 mismatch")
    return entry, actual_path
