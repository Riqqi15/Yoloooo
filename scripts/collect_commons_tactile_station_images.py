from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_LICENSES = {
    "CC0",
    "CC BY 1.0",
    "CC BY 2.0",
    "CC BY 2.5",
    "CC BY 3.0",
    "CC BY 4.0",
    "CC BY-SA 1.0",
    "CC BY-SA 2.0",
    "CC BY-SA 2.5",
    "CC BY-SA 3.0",
    "CC BY-SA 4.0",
    "Public domain",
}


def api_get(params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "origin": "*"})
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "Yoloooo tactile dataset collector/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def value(metadata: dict, key: str) -> str:
    item = metadata.get(key, {})
    return str(item.get("value", "")).strip() if isinstance(item, dict) else ""


def safe_stem(title: str) -> str:
    stem = title.removeprefix("File:").rsplit(".", 1)[0]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return stem[:80] or "commons_image"


def collect(output_dir: Path, count: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = api_get(
        {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": "Category:Tactile_paving_in_train_stations",
            "gcmtype": "file",
            "gcmlimit": "100",
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": "1600",
        }
    )
    pages = sorted(payload.get("query", {}).get("pages", {}).values(), key=lambda item: int(item["pageid"]))
    rows: list[dict[str, str]] = []
    used_names: set[str] = set()
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        metadata = info.get("extmetadata") or {}
        license_name = value(metadata, "LicenseShortName")
        if license_name not in ALLOWED_LICENSES:
            continue
        image_url = str(info.get("thumburl") or info.get("url") or "").strip()
        page_url = f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(str(page['title']).replace(' ', '_'))}"
        if not image_url:
            continue
        suffix = Path(urllib.parse.urlparse(image_url).path).suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        name = f"commons_{page['pageid']}_{safe_stem(str(page['title']))}{suffix}"
        if name in used_names:
            continue
        used_names.add(name)
        destination = output_dir / name
        if not destination.exists():
            request = urllib.request.Request(image_url, headers={"User-Agent": "Yoloooo tactile dataset collector/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                destination.write_bytes(response.read())
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        rows.append(
            {
                "filename": name,
                "title": str(page["title"]),
                "source_url": page_url,
                "direct_url": image_url,
                "license": license_name,
                "artist": value(metadata, "Artist"),
                "description": value(metadata, "ImageDescription"),
                "sha256": digest,
            }
        )
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise RuntimeError(f"only {len(rows)} licensed images found; requested {count}")
    with (output_dir / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"downloaded": len(rows), "output_dir": str(output_dir), "sources": str(output_dir / "sources.csv")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(collect(args.output_dir, args.count), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
