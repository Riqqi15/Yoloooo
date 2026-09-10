from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ZIP = ROOT / "data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip"
CHECKSUM = ROOT / "data/public/guidetwsi-rbar-v1/sha256.txt"


class TactileHandoffTest(unittest.TestCase):
    def test_codex_entrypoint_and_handoff_cover_progress_and_targets(self) -> None:
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        start_here = (ROOT / "CODEX_START_HERE.md").read_text(encoding="utf-8")
        handoff = (ROOT / "docs/HANDOFF_TACTILE_V3.md").read_text(encoding="utf-8")
        self.assertIn("CODEX_START_HERE.md", agents)
        self.assertIn("docs/HANDOFF_TACTILE_V3.md", agents)
        self.assertIn("AGENTS.md", start_here)
        self.assertIn("Wajib dibaca oleh Codex", start_here)
        self.assertIn("README.md", start_here)
        for required in (
            "82.50%",
            "56.41%",
            "100.00%",
            "tactile-one-class-v3-public2-station1",
            "not mobile-ready",
            "5 meter",
            "Commuter",
            "trolley",
        ):
            self.assertIn(required, handoff)

    def test_public_archive_matches_committed_checksum(self) -> None:
        expected = CHECKSUM.read_text(encoding="utf-8").split()[0].lower()
        digest = hashlib.sha256()
        with PUBLIC_ZIP.open("rb") as archive:
            for chunk in iter(lambda: archive.read(1024 * 1024), b""):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
