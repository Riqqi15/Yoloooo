from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from model_manifest import sha256_file


DATASET_HANDLE = "guidedogrobot/guidetwsi"
ALLOWED_LICENSES = frozenset({"CC0", "CC0-1.0", "CC0: Public Domain"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MAX_BYTES = 5_000_000_000


def validate_source(handle: str, license_name: str) -> None:
    if handle != DATASET_HANDLE:
        raise ValueError(f"unexpected dataset handle: {handle}")
    if license_name.strip() not in ALLOWED_LICENSES:
        raise ValueError(f"unapproved or missing license: {license_name!r}")


def _safe_parts(raw_path: str) -> tuple[str, ...] | None:
    if not raw_path or "\\" in raw_path:
        return None
    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.parts


def _pair_key(parts: tuple[str, ...], kind_index: int) -> str:
    rbar_index = next(
        index for index, part in enumerate(parts) if "rbar" in part.lower()
    )
    relative = [
        part
        for index, part in enumerate(parts)
        if index > rbar_index and index != kind_index
    ]
    relative[-1] = str(PurePosixPath(relative[-1]).with_suffix(""))
    return "/".join(relative)


def build_pair_inventory(
    files: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen_paths: set[str] = set()
    grouped: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()

    for record in files:
        raw_path = str(record.get("path", ""))
        if raw_path in seen_paths:
            raise ValueError(f"duplicate inventory path: {raw_path}")
        seen_paths.add(raw_path)
        parts = _safe_parts(raw_path)
        if parts is None:
            rejected["unsafe_path"] += 1
            continue
        lowered = tuple(part.lower() for part in parts)
        if not any("rbar" in part for part in lowered):
            rejected["outside_rbar"] += 1
            continue
        if "train" not in lowered:
            rejected["outside_train"] += 1
            continue
        kind = ""
        kind_index = -1
        for index, part in enumerate(lowered):
            if part == "images":
                kind, kind_index = "image", index
                break
            if "label" in part:
                kind, kind_index = "label", index
                break
        suffix = PurePosixPath(raw_path).suffix.lower()
        if not kind or (kind == "image" and suffix not in IMAGE_SUFFIXES) or (
            kind == "label" and suffix != ".txt"
        ):
            rejected["unsupported"] += 1
            continue
        size = int(record.get("size", -1))
        if size < 0:
            raise ValueError(f"invalid byte size for {raw_path}")
        key = _pair_key(parts, kind_index)
        slot = grouped.setdefault(key, {"key": key})
        if kind in slot:
            raise ValueError(f"duplicate {kind} for pair: {key}")
        slot[kind] = {"path": raw_path, "size": size}

    pairs: list[dict[str, Any]] = []
    for key, slot in grouped.items():
        if "image" not in slot or "label" not in slot:
            rejected["unpaired"] += 1
            continue
        pairs.append(
            {
                "key": key,
                "image_path": slot["image"]["path"],
                "image_bytes": slot["image"]["size"],
                "label_path": slot["label"]["path"],
                "label_bytes": slot["label"]["size"],
                "total_bytes": slot["image"]["size"] + slot["label"]["size"],
            }
        )
    pairs.sort(key=lambda pair: pair["key"])
    return pairs, dict(rejected)


def select_pairs(
    pairs: list[dict[str, Any]], max_bytes: int, max_images: int, seed: int
) -> list[dict[str, Any]]:
    if not 0 < max_bytes <= MAX_BYTES:
        raise ValueError(f"max_bytes must be within 1..{MAX_BYTES}")
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    ranked = sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"{seed}:{pair['key']}".encode("utf-8")
        ).digest(),
    )
    selected: list[dict[str, Any]] = []
    used = 0
    for pair in ranked:
        size = int(pair["total_bytes"])
        if size < 0:
            raise ValueError(f"invalid pair size: {pair['key']}")
        if used + size > max_bytes:
            continue
        selected.append(pair)
        used += size
        if len(selected) == max_images:
            break
    return selected


def _kaggle_download(remote_path: str, staging: Path) -> Path:
    import kagglehub

    return Path(
        kagglehub.dataset_download(
            DATASET_HANDLE,
            path=remote_path,
            output_dir=str(staging),
            force_download=True,
        )
    )


def download_with_retries(
    downloader: Callable[[str, Path], Path],
    remote_path: str,
    staging: Path,
    delays: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> Path:
    for attempt in range(len(delays) + 1):
        try:
            return Path(downloader(remote_path, staging))
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            if status is not None and 400 <= status < 500 and status not in {408, 425, 429}:
                raise
            if attempt == len(delays):
                raise
            time.sleep(delays[attempt])
    raise AssertionError("unreachable")


def _materialize_file(
    record: dict[str, Any],
    cache_root: Path,
    staging_root: Path,
    downloader: Callable[[str, Path], Path],
) -> dict[str, Any]:
    remote_path = str(record["remote_path"])
    parts = _safe_parts(remote_path)
    if parts is None:
        raise ValueError(f"unsafe download path: {remote_path}")
    target = cache_root.joinpath(*parts)
    expected_bytes = int(record["expected_bytes"])
    if target.is_file():
        if target.stat().st_size != expected_bytes:
            raise ValueError(f"cached size mismatch: {remote_path}")
    else:
        stage = staging_root / hashlib.sha256(remote_path.encode()).hexdigest()
        stage.mkdir(parents=True, exist_ok=False)
        try:
            downloaded = download_with_retries(downloader, remote_path, stage)
            if not downloaded.is_file():
                raise FileNotFoundError(f"download did not produce a file: {remote_path}")
            if downloaded.stat().st_size != expected_bytes:
                raise ValueError(f"downloaded size mismatch: {remote_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(target)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return {
        "pair_key": record["pair_key"],
        "role": record["role"],
        "upstream_path": remote_path,
        "bytes": expected_bytes,
        "sha256": sha256_file(target),
        "local_cache_key": target.relative_to(cache_root).as_posix(),
    }


def materialize_selection(
    selected: list[dict[str, Any]],
    cache_root: Path,
    downloader: Callable[[str, Path], Path] = _kaggle_download,
    workers: int = 8,
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be within 1..8")
    files = [
        {
            "pair_key": pair["key"],
            "role": role,
            "remote_path": pair[f"{role}_path"],
            "expected_bytes": pair[f"{role}_bytes"],
        }
        for pair in selected
        for role in ("image", "label")
    ]
    cache_root.mkdir(parents=True, exist_ok=True)
    staging_root = cache_root / ".staging"
    staging_root.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        retained = list(
            executor.map(
                lambda record: _materialize_file(
                    record, cache_root, staging_root, downloader
                ),
                files,
            )
        )
    if not any(staging_root.iterdir()):
        staging_root.rmdir()
    retained.sort(key=lambda record: record["upstream_path"])
    return {
        "summary": {
            "selected_pairs": len(selected),
            "downloaded_files": len(retained),
            "downloaded_bytes": sum(item["bytes"] for item in retained),
        },
        "files": retained,
    }


def query_kaggle_inventory(page_size: int = 1000) -> dict[str, Any]:
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import (
        ApiGetDatasetRequest,
        ApiListDatasetFilesRequest,
    )

    owner, dataset_slug = DATASET_HANDLE.split("/", maxsplit=1)
    with build_kaggle_client() as client:
        metadata_request = ApiGetDatasetRequest()
        metadata_request.owner_slug = owner
        metadata_request.dataset_slug = dataset_slug
        metadata = client.datasets.dataset_api_client.get_dataset(metadata_request)
        license_name = metadata.license_name
        validate_source(DATASET_HANDLE, license_name)

        files: list[dict[str, Any]] = []
        page_token = ""
        while True:
            request = ApiListDatasetFilesRequest()
            request.owner_slug = owner
            request.dataset_slug = dataset_slug
            request.dataset_version_number = metadata.current_version_number
            request.page_size = page_size
            if page_token:
                request.page_token = page_token
            response = client.datasets.dataset_api_client.list_dataset_files(request)
            if response.error_message:
                raise RuntimeError(response.error_message)
            files.extend(
                {"path": item.name, "size": item.total_bytes, "url": item.url}
                for item in response.dataset_files
            )
            page_token = response.next_page_token
            if not page_token:
                break

    return {
        "inventory_version": 1,
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_handle": DATASET_HANDLE,
        "dataset_version": metadata.current_version_number,
        "license": license_name,
        "files": files,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire a bounded GuideTWSI RBar training subset."
    )
    parser.add_argument("command", choices=("inventory", "download"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache", type=Path, default=Path("runs/public-data-cache/guidetwsi-rbar-v1")
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/public/guidetwsi-rbar-v1/provenance.json"),
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "inventory":
            if not args.output:
                raise ValueError("--output is required for inventory")
            payload = query_kaggle_inventory()
            pairs, rejected = build_pair_inventory(payload["files"])
            payload["summary"] = {
                "total_files": len(payload["files"]),
                "eligible_pairs": len(pairs),
                "eligible_pair_bytes": sum(pair["total_bytes"] for pair in pairs),
                "rejected": rejected,
            }
            write_json_atomic(args.output, payload)
            print(json.dumps(payload["summary"], indent=2))
            return 0
        if not args.inventory:
            raise ValueError("--inventory is required for download")
        payload = json.loads(args.inventory.read_text(encoding="utf-8"))
        validate_source(payload["dataset_handle"], payload["license"])
        pairs, rejected = build_pair_inventory(payload["files"])
        selected = select_pairs(pairs, args.max_bytes, args.max_images, args.seed)
        selected_bytes = sum(pair["total_bytes"] for pair in selected)
        if shutil.disk_usage(args.cache.parent).free < selected_bytes + 1_000_000_000:
            raise OSError("insufficient free space after 1 GB safety reserve")
        materialized = materialize_selection(
            selected, args.cache, workers=args.workers
        )
        provenance = {
            "provenance_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset_handle": payload["dataset_handle"],
            "dataset_version": payload["dataset_version"],
            "license": payload["license"],
            "selection": {
                "subset": "RBar train",
                "seed": args.seed,
                "max_bytes": args.max_bytes,
                "max_images": args.max_images,
                "rejected": rejected,
            },
            **materialized,
        }
        write_json_atomic(args.provenance, provenance)
        print(json.dumps(provenance["summary"], indent=2))
        return 0
    except Exception as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
