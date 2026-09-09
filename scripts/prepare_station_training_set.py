from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2

from baseline_object_inference import IMAGE_SUFFIXES, STOP_FAILURE
from model_manifest import sha256_file
from training_label_gate import load_taxonomy, validate_annotation


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


# Only sequences known to share a location/session are grouped. Other published
# photos stay isolated; the manifest builder still checks exact and near duplicates.
GROUP_OVERRIDES = {
    "003_tegalluar.jpg": ("tegalluar", "tegalluar_nfarras"),
    "009_137824158.jpg": ("tegalluar", "tegalluar_nfarras"),
    "010_137778724.jpg": ("tegalluar", "tegalluar_nfarras"),
    "004_cikampek_malam.jpg": ("cikampek", "cikampek_004"),
    "007_65003351.jpg": ("cikampek", "cikampek_007"),
    "005_istora_mandiri.jpg": ("istora_mandiri", "istora_005"),
    "049_77649720.jpg": ("istora_mandiri", "istora_049"),
    "018_26110885.jpg": ("surabaya_pasarturi", "pasarturi_018"),
    "021_167019772.jpg": ("surabaya_pasarturi", "pasarturi_bahnfrend_2025"),
    "022_167019789.jpg": ("surabaya_pasarturi", "pasarturi_bahnfrend_2025"),
    "019_54646875.jpg": ("semarang_tawang", "tawang_019"),
    "020_2452061.jpg": ("semarang_tawang", "tawang_020"),
    "032_166526831.jpg": ("semarang_tawang", "tawang_bahnfrend_2025"),
    "033_166526840.jpg": ("semarang_tawang", "tawang_bahnfrend_2025"),
    "034_166526836.jpg": ("semarang_tawang", "tawang_bahnfrend_2025"),
    "035_166526837.jpg": ("semarang_tawang", "tawang_bahnfrend_2025"),
    "036_166544048.jpg": ("semarang_tawang", "tawang_bahnfrend_2025"),
    "026_116115102.jpg": ("jatinegara", "jatinegara_herryz"),
    "027_116114944.jpg": ("jatinegara", "jatinegara_herryz"),
    "042_140835783.jpg": ("blok_m", "blok_m_042"),
    "044_77601320.jpg": ("blok_m", "blok_m_044"),
    "048_77563734.jpg": ("blok_m", "blok_m_048"),
}


def read_source_rows(source_dir: Path) -> list[dict[str, Any]]:
    json_path = source_dir / "sources.json"
    if json_path.is_file():
        rows = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        csv_path = source_dir / "sources.csv"
        if not csv_path.is_file():
            raise FileNotFoundError("sources.json or sources.csv is required")
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [
                {
                    "set_index": index,
                    "filename": row["filename"],
                    "title": row["title"],
                    "source": row["source_url"],
                    "license": row["license"],
                    "artist": row["artist"],
                }
                for index, row in enumerate(csv.DictReader(handle), start=1)
            ]
    if not isinstance(rows, list) or not rows:
        raise ValueError("source table must contain a nonempty list")
    return rows


def default_groups(rows: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    groups: dict[str, tuple[str, str]] = {}
    for row in rows:
        filename = str(row["filename"])
        index = int(row["set_index"])
        groups[filename] = GROUP_OVERRIDES.get(
            filename, (f"station_source_{index:03d}", f"published_{index:03d}")
        )
    return groups


def location_type(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(field, "")) for field in ("filename", "title")
    ).lower()
    if any(word in text for word in ("interior", "inside", "building", "dalam")):
        return "station_interior"
    if any(word in text for word in ("platform", "peron")):
        return "station_platform"
    return "station_track_area"


def convert_labelme(
    path: Path,
    image_name: str,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if Path(str(payload.get("imagePath", ""))).name != image_name:
        raise ValueError(f"LabelMe imagePath mismatch: {path}")
    if payload.get("imageWidth") != width or payload.get("imageHeight") != height:
        raise ValueError(f"LabelMe dimensions mismatch: {path}")
    tactile = []
    for shape in payload.get("shapes", []):
        if shape.get("label") != "tactile_paving" or shape.get("shape_type") != "polygon":
            raise ValueError(f"unsupported LabelMe shape: {path}")
        tactile.append({"label": "tactile_paving", "points": shape.get("points")})
    return tactile


def prepare_dataset(
    source_dir: Path,
    metadata_path: Path,
    annotations_dir: Path,
    groups: dict[str, tuple[str, str]] | None = None,
    taxonomy_path: Path = Path("data/training/taxonomy_tactile_v1.json"),
) -> dict[str, Any]:
    source_rows = read_source_rows(source_dir)
    source_rows = sorted(source_rows, key=lambda row: int(row["set_index"]))
    names = [str(row["filename"]) for row in source_rows]
    images = sorted(
        path.name
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(names) != len(set(names)) or sorted(names) != images:
        raise ValueError("source table and source images do not match exactly")
    groups = groups or default_groups(source_rows)
    if set(groups) != set(names):
        raise ValueError("group mapping must cover every source image exactly")

    _, object_labels, tactile_labels = load_taxonomy(taxonomy_path)
    metadata_rows: list[dict[str, str]] = []
    annotations: dict[str, dict[str, Any]] = {}
    positive = 0
    for row in source_rows:
        name = str(row["filename"])
        image_path = source_dir / name
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"unable to decode image: {image_path}")
        height, width = image.shape[:2]
        digest = sha256_file(image_path)
        tactile = convert_labelme(image_path.with_suffix(".json"), name, width, height)
        annotation = {
            "schema_version": 1,
            "sample_id": digest,
            "source_sha256": digest,
            "objects": [],
            "tactile": tactile,
        }
        validate_annotation(
            annotation,
            {"sample_id": digest, "source_sha256": digest, "dataset_task": "tactile"},
            width,
            height,
            object_labels,
            tactile_labels,
        )
        annotations[digest] = annotation
        positive += bool(tactile)
        location_id, session_id = groups[name]
        title = str(row.get("title") or name)
        source = str(row.get("source") or "unknown_source")
        license_name = str(row.get("license") or "license_unknown")
        metadata_rows.append(
            {
                "source_path": name,
                "session_id": session_id,
                "location_id": location_id,
                "location_type": location_type(row),
                "source_name": "wikimedia_commons" if "wikimedia.org" in source else "curated_source",
                "usage_permission": f"{license_name}; {source}",
                "device_model": "unknown_published_image",
                "camera_position": "varied_published_photo",
                "lighting": "low_light" if any(word in title.lower() for word in ("malam", "night")) else "normal_or_unknown",
                "motion": "static_photo",
                "surface": "mixed",
                "dataset_task": "tactile",
                "notes": f"{title}; artist={row.get('artist', 'unknown')}",
            }
        )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
    with temporary_metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(metadata_rows)
    temporary_metadata.replace(metadata_path)

    annotations_dir.mkdir(parents=True, exist_ok=True)
    for digest, annotation in annotations.items():
        destination = annotations_dir / f"{digest}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(destination)
    return {
        "samples": len(source_rows),
        "positive_labelme": positive,
        "empty_unreviewed": len(source_rows) - positive,
        "metadata": str(metadata_path),
        "annotations": str(annotations_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the curated station LabelMe set into training-gate inputs."
    )
    parser.add_argument(
        "--source-dir", type=Path, default=Path("data/training/station_photo_set_50")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/training/intake_station_photo_set_50.csv")
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/training/annotations/station-photo-set-50-v1"),
    )
    parser.add_argument(
        "--taxonomy", type=Path, default=Path("data/training/taxonomy_tactile_v1.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = prepare_dataset(
            args.source_dir, args.metadata, args.annotations_dir, taxonomy_path=args.taxonomy
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"{STOP_FAILURE}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
