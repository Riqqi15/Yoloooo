from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from baseline_object_inference import STOP_FAILURE, write_metrics
from evaluate_samples import output_name, write_image
from ground_truth_manifest import read_manifest


def render_review(manifest_path: Path, output_dir: Path) -> dict[str, object]:
    rows = read_manifest(manifest_path)
    items = []
    for row in rows:
        source = cv2.imread(row["source_path"])
        if source is None:
            raise ValueError(f"source image cannot be decoded: {row['source_path']}")
        overlay = source.copy()
        mask_path = Path(row["ground_truth_mask_path"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape != source.shape[:2]:
                raise ValueError(f"mask dimensions mismatch: {mask_path}")
            foreground = mask > 0
            red = np.zeros_like(source)
            red[:, :, 2] = 255
            overlay[foreground] = (
                source[foreground] * 0.45 + red[foreground] * 0.55
            ).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        label = f"{row['review_status']} | {row['annotation_origin'] or 'no-origin'}"
        cv2.rectangle(overlay, (0, 0), (min(source.shape[1], 560), 30), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        destination = output_dir / output_name(Path(row["source_path"]))
        write_image(destination, overlay)
        items.append(
            {
                "sample_id": row["sample_id"],
                "source": row["source_path"],
                "overlay": str(destination),
                "review_status": row["review_status"],
                "annotation_origin": row["annotation_origin"],
            }
        )
    payload: dict[str, object] = {
        "manifest": str(manifest_path),
        "images": len(items),
        "items": items,
    }
    write_metrics(output_dir / "index.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ground-truth masks over source images for QA.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/ground_truth/manifest.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/ground-truth-review")
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        payload = render_review(args.manifest, args.output_dir)
        print(json.dumps({"images": payload["images"]}, indent=2))
        return 0
    except KeyboardInterrupt:
        print(f"{STOP_FAILURE}: interrupted", file=sys.stderr)
    except Exception as error:
        print(f"{STOP_FAILURE}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
