from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Coordinates are normalized (x / width, y / height). These are deliberately
# provisional polygons: the owner must inspect and correct them in LabelMe.
POLYGONS: dict[str, list[list[tuple[float, float]]]] = {
    "commons_10004207_Grafton_station.jpg": [[(0.16, 0.92), (0.30, 0.92), (0.43, 0.57), (0.37, 0.55)], [(0.58, 0.91), (0.76, 0.88), (0.70, 0.53), (0.58, 0.54)], [(0.16, 0.72), (0.75, 0.72), (0.75, 0.80), (0.16, 0.80)]],
    "commons_11548638_ADL_807_at_Onehunga.jpg": [[(0.44, 0.99), (0.56, 0.99), (0.77, 0.55), (0.68, 0.55)]],
    "commons_12357929_Paris_-_Gare_de_l_Est_-_20101105_1.jpg": [[(0.57, 0.99), (0.63, 0.99), (0.83, 0.53), (0.79, 0.52)], [(0.67, 0.99), (0.73, 0.99), (0.91, 0.55), (0.87, 0.54)], [(0.80, 0.99), (0.86, 0.99), (0.98, 0.54), (0.94, 0.54)]],
    "commons_12811471_Grafton_Train_Station_Finally_Finished_I.jpg": [[(0.69, 0.97), (0.86, 0.92), (0.82, 0.63), (0.72, 0.62)], [(0.09, 0.90), (0.19, 0.90), (0.30, 0.60), (0.24, 0.59)]],
    "commons_14571980_Middlemore_Train_Station_Platforms.jpg": [[(0.72, 0.99), (0.98, 0.99), (0.90, 0.55), (0.76, 0.55)], [(0.08, 0.68), (0.24, 0.68), (0.29, 0.53), (0.18, 0.52)]],
    "commons_14572036_Middlemore_Train_Station_Shelters.jpg": [[(0.10, 0.91), (0.27, 0.91), (0.32, 0.59), (0.22, 0.58)], [(0.76, 0.88), (0.93, 0.88), (0.84, 0.58), (0.76, 0.58)]],
    "commons_14703064_Newmarket_Train_Station_Is_Finished_II.jpg": [[(0.10, 0.94), (0.26, 0.94), (0.34, 0.58), (0.27, 0.58)], [(0.67, 0.88), (0.82, 0.88), (0.79, 0.58), (0.70, 0.58)]],
    "commons_14874418_Panmure_Train_Station_Trenching.jpg": [[(0.10, 0.91), (0.25, 0.91), (0.34, 0.59), (0.27, 0.59)], [(0.71, 0.88), (0.87, 0.88), (0.78, 0.60), (0.69, 0.60)], [(0.08, 0.74), (0.84, 0.74), (0.84, 0.82), (0.08, 0.82)]],
    "commons_15282574_Cityrail_Tactile_Platform_Line.jpg": [[(0.00, 0.00), (0.30, 0.00), (0.46, 0.14), (0.00, 0.16)]],
    "commons_17952932_KiwiRail_Locomotive_At_Britomart.jpg": [[(0.04, 0.95), (0.28, 0.95), (0.31, 0.58), (0.18, 0.58)]],
    "commons_19170842_Manukau_Station.jpg": [[(0.16, 0.92), (0.32, 0.92), (0.38, 0.57), (0.29, 0.57)], [(0.69, 0.92), (0.86, 0.92), (0.73, 0.57), (0.64, 0.57)]],
    "commons_19822898_Paris_Est_caddie_sur_un_quai.jpg": [[(0.00, 0.78), (0.12, 0.83), (0.34, 0.54), (0.28, 0.50)], [(0.52, 0.99), (0.68, 0.99), (0.99, 0.56), (0.91, 0.52)]],
    "commons_20765001_Platform_end_marker.jpg": [[(0.00, 0.22), (0.98, 0.22), (0.98, 0.53), (0.00, 0.53)]],
    "commons_21583331_Est_Ing_Castello_Anden.jpg": [[(0.52, 0.98), (0.72, 0.98), (0.75, 0.58), (0.58, 0.58)]],
    "commons_23215780_Arden_-_Del_Paso_4043_02.jpg": [[(0.41, 0.99), (0.60, 0.99), (0.57, 0.55), (0.48, 0.55)]],
    "commons_2793050_HongKongStationplatform1_20070922.jpg": [[(0.72, 0.96), (0.91, 0.96), (0.99, 0.66), (0.83, 0.66)], [(0.00, 0.73), (0.20, 0.73), (0.40, 0.58), (0.25, 0.56)]],
    "commons_28917160_PieveemanueleFS_4.jpg": [[(0.00, 0.74), (0.15, 0.84), (0.50, 0.57), (0.43, 0.53)], [(0.50, 0.96), (0.66, 0.96), (0.98, 0.64), (0.89, 0.58)]],
    "commons_5554713_Kingsland_Train_Station_Photos_II.jpg": [[(0.16, 0.95), (0.29, 0.95), (0.39, 0.58), (0.31, 0.58)]],
    "commons_5554790_Kingsland_Train_Station_Photos_I.jpg": [[(0.17, 0.95), (0.30, 0.95), (0.42, 0.58), (0.34, 0.58)], [(0.70, 0.95), (0.84, 0.95), (0.75, 0.58), (0.66, 0.58)]],
    "commons_5577600_DC4444atNewLynn.jpg": [[(0.25, 0.93), (0.43, 0.93), (0.56, 0.61), (0.45, 0.61)]],
}


def jpeg_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"only JPEG input is supported by this seeder: {path}")
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        length = int.from_bytes(data[index : index + 2], "big")
        if 0xC0 <= marker <= 0xC3 or 0xC5 <= marker <= 0xC7 or 0xC9 <= marker <= 0xCB or 0xCD <= marker <= 0xCF:
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            return width, height
        index += length
    raise ValueError(f"JPEG dimensions not found: {path}")


def make_annotations(image_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, polygons in POLYGONS.items():
        image_path = image_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        width, height = jpeg_size(image_path)
        shapes = []
        for polygon in polygons:
            points = [[round(x * width, 2), round(y * height, 2)] for x, y in polygon]
            shapes.append({"label": "tactile_paving", "points": points, "group_id": None, "description": "provisional seed; inspect in LabelMe", "shape_type": "polygon", "flags": {}})
        payload = {
            "version": "5.8.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": filename,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        destination = output_dir / f"{Path(filename).stem}.json"
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(str(destination))
    sources_path = image_dir / "sources.csv"
    intake_path = output_dir / "intake.csv"
    with sources_path.open(encoding="utf-8-sig", newline="") as handle, intake_path.open("w", encoding="utf-8", newline="") as target:
        source_rows = {row["filename"]: row for row in csv.DictReader(handle)}
        fields = ["source_path", "session_id", "location_id", "location_type", "source_name", "usage_permission", "device_model", "camera_position", "lighting", "motion", "surface", "dataset_task", "notes"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for filename in POLYGONS:
            row = source_rows[filename]
            writer.writerow({
                "source_path": filename,
                "session_id": f"commons_{filename.split('_', 2)[1]}",
                "location_id": f"commons_{filename.split('_', 2)[1]}",
                "location_type": "station_platform",
                "source_name": "Wikimedia Commons",
                "usage_permission": f"{row['license']}; source and attribution recorded in sources.csv",
                "device_model": "unknown_public_camera",
                "camera_position": "unknown",
                "lighting": "unknown",
                "motion": "unknown",
                "surface": "station_platform",
                "dataset_task": "tactile",
                "notes": "Provisional LabelMe seed; human review required before training.",
            })
    return {"annotations": len(written), "output_dir": str(output_dir), "intake": str(intake_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_annotations(args.images, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
