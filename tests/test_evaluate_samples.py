from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_object_inference import STOP_FAILURE, STOP_OBSTACLE, STOP_UNCERTAIN  # noqa: E402
from evaluate_samples import (  # noqa: E402
    batch_exit_code,
    combined_status,
    discover_images,
    load_station_scope,
    output_name,
    run,
)


class EvaluateSamplesTest(unittest.TestCase):
    def write_scope(
        self, root: Path, samples: dict[str, str], scope: str = "station_environment"
    ) -> Path:
        path = root / "scope.json"
        path.write_text(
            json.dumps(
                {
                    "scope": scope,
                    "included_contexts": ["station_platform", "station_access_sidewalk"],
                    "samples": samples,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_station_scope_accepts_registered_station_sidewalk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "station-sidewalk.webp"
            image.touch()
            scope = load_station_scope(
                self.write_scope(root, {image.name: "station_access_sidewalk"}),
                [image],
            )
            self.assertEqual(scope["samples"][image.name], "station_access_sidewalk")

    def test_station_scope_rejects_unregistered_active_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "bus.jpg"
            image.touch()
            with self.assertRaisesRegex(ValueError, "unregistered active images: bus.jpg"):
                load_station_scope(self.write_scope(root, {}), [image])

    def test_station_scope_rejects_missing_registered_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "registered images missing"):
                load_station_scope(
                    self.write_scope(root, {"missing.jpg": "station_platform"}), []
                )

    def test_station_scope_rejects_wrong_scope_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "scope must be station_environment"):
                load_station_scope(self.write_scope(root, {}, scope="public_transport"), [])

    def test_station_scope_rejects_unapproved_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "road.jpg"
            image.touch()
            with self.assertRaisesRegex(ValueError, "unapproved context"):
                load_station_scope(
                    self.write_scope(root, {image.name: "generic_road"}), [image]
                )

    def test_scope_cannot_allow_bus_by_expanding_included_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_scope(root, {"bus.jpg": "bus_only"})
            data = json.loads(path.read_text(encoding="utf-8"))
            data["included_contexts"].append("bus_only")
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unapproved context"):
                load_station_scope(path, [root / "bus.jpg"])

    def test_batch_rejects_unregistered_image_before_loading_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bus.jpg").touch()
            args = argparse.Namespace(
                source_dir=root, scope_file=self.write_scope(root, {}),
                object_model="unused.pt",
            )
            with patch("evaluate_samples.load_object_model") as object_loader, patch(
                "evaluate_samples.load_tactile_model"
            ) as tactile_loader:
                with self.assertRaisesRegex(ValueError, "unregistered active images"):
                    run(args)
                object_loader.assert_not_called()
                tactile_loader.assert_not_called()

    def test_discover_images_includes_webp_and_ignores_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("z.WEBP", "a.jpg", "notes.txt"):
                (root / name).touch()
            self.assertEqual(
                [path.name for path in discover_images(root)],
                ["a.jpg", "z.WEBP"],
            )

    def test_output_name_preserves_source_extension_as_suffix(self) -> None:
        self.assertEqual(output_name(Path("station.photo.webp")), "station.photo_webp.jpg")

    def test_combined_status_never_grants_movement(self) -> None:
        self.assertEqual(combined_status(STOP_OBSTACLE), STOP_OBSTACLE)
        self.assertEqual(combined_status(STOP_UNCERTAIN), STOP_UNCERTAIN)
        self.assertEqual(combined_status("unexpected"), STOP_FAILURE)
        self.assertEqual(combined_status(STOP_OBSTACLE, failed=True), STOP_FAILURE)

    def test_batch_exit_code_fails_if_any_image_failed(self) -> None:
        self.assertEqual(batch_exit_code([{"combined_status": STOP_UNCERTAIN}]), 0)
        self.assertEqual(
            batch_exit_code(
                [
                    {"combined_status": STOP_UNCERTAIN},
                    {"combined_status": STOP_FAILURE},
                ]
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
