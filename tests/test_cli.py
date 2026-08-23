from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from toontra.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_process_accepts_tiling_options(self) -> None:
        args = build_parser().parse_args(
            [
                "process",
                "page.png",
                "--output",
                "output",
                "--tile-height",
                "1600",
                "--tile-overlap",
                "256",
            ]
        )
        self.assertEqual(args.tile_height, 1600)
        self.assertEqual(args.tile_overlap, 256)

    def test_demo_writes_documented_artifacts_and_overwrite_is_explicit(self) -> None:
        # The bundled sample exercises the real default detector and output
        # contract; detector accuracy is evaluated separately.
        with tempfile.TemporaryDirectory(prefix="toontra-cli-") as directory:
            output = Path(directory) / "demo"
            self.assertEqual(main(["demo", "--output", str(output)]), 0)
            expected = {"source.png", "cleaned.png", "mask.png", "overlay.png", "result.json"}
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            metadata = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertGreater(metadata["bubble_count"], 0)
            self.assertEqual(len(metadata["bubbles"]), metadata["bubble_count"])
            self.assertEqual(main(["demo", "--output", str(output)]), 2)
            self.assertEqual(main(["demo", "--output", str(output), "--force"]), 0)

    def test_demo_force_does_not_write_through_existing_source_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toontra-cli-") as directory:
            root = Path(directory)
            output = root / "demo"
            external = root / "external.png"

            output.mkdir()
            external.write_bytes(b"external")

            source_target = output / "source.png"
            source_target.hardlink_to(external)

            self.assertEqual(
                main(["demo", "--output", str(output), "--force"]),
                0,
            )

            self.assertEqual(external.read_bytes(), b"external")
            self.assertNotEqual(source_target.read_bytes(), b"external")


if __name__ == "__main__":
    unittest.main()
