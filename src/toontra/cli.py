from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .errors import OutputExistsError, ToontraError
from .imaging import load_rgb, save_rgb
from .pipeline import Toontra

SAMPLE_WEBTOON = Path(__file__).resolve().parent / "assets" / "sample_webtoon.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toontra",
        description="Detect and erase speech-bubble contents from webtoon images.",
    )
    parser.add_argument("--version", action="version", version=f"Toontra {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    process = commands.add_parser("process", help="Process a PNG or JPEG page")
    process.add_argument("input", type=Path)
    _add_output_arguments(process)
    _add_pipeline_arguments(process)

    demo = commands.add_parser("demo", help="Run the offline bundled demo")
    _add_output_arguments(demo)
    return parser


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--save-crops", action="store_true", help="Save detected bubble crops")
    parser.add_argument("--force", action="store_true", help="Overwrite known output files")


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ocr", choices=("off", "easyocr"), default="off")
    parser.add_argument("--source-language", default="ko")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow an explicitly requested OCR adapter to download public weights",
    )
    parser.add_argument(
        "--masker",
        choices=("white", "manet"),
        default="white",
        help="Bubble masker: the weight-free default, or the optional MA-Net adapter",
    )
    parser.add_argument(
        "--manet-checkpoint",
        type=Path,
        default=None,
        help="Path to the MA-Net checkpoint (required with --masker manet)",
    )
    parser.add_argument("--tile-height", type=int, default=1600)
    parser.add_argument("--tile-overlap", type=int, default=256)
    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=0.5,
        help="IoU threshold for seam-duplicate NMS on long pages",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            return _run_demo(args)
        return _run_process(args)
    except (ToontraError, FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"toontra: {error}", file=sys.stderr)
        return 2


def _run_demo(args: argparse.Namespace) -> int:
    if not SAMPLE_WEBTOON.is_file():
        raise FileNotFoundError(f"Missing bundled demo image: {SAMPLE_WEBTOON}")
    source = load_rgb(SAMPLE_WEBTOON)
    toontra = Toontra()
    result = toontra.process(source)
    source_target = Path(args.output) / "source.png"
    if args.force and (source_target.exists() or source_target.is_symlink()):
        if source_target.is_dir() and not source_target.is_symlink():
            raise OutputExistsError(
                f"Output target is a directory: {source_target}"
            )
        source_target.unlink()
    destination = result.save(
        args.output,
        save_crops=args.save_crops,
        force=args.force,
    )
    save_rgb(destination / "source.png", source)
    print(f"Detected {len(result.bubbles)} bubbles")
    print(f"Saved demo to {destination.resolve()}")
    return 0


def _run_process(args: argparse.Namespace) -> int:
    recognizer = None
    if args.ocr == "easyocr":
        from .modules import EasyOcrRecognizer

        recognizer = EasyOcrRecognizer(
            gpu=_use_gpu(args.device),
            download_enabled=args.allow_model_download,
        )

    masker = None
    if args.masker == "manet":
        if args.manet_checkpoint is None:
            raise ToontraError("--manet-checkpoint is required when --masker manet is selected")
        from .modules import ManetBubbleMasker

        masker = ManetBubbleMasker(args.manet_checkpoint, device=args.device)

    from .modules import Yolo26BubbleDetector

    detector_device = None if args.device == "auto" else args.device
    detector = Yolo26BubbleDetector(device=detector_device)
    toontra = Toontra(
        detector=detector,
        masker=masker,
        recognizer=recognizer,
        tile_height=args.tile_height,
        tile_overlap=args.tile_overlap,
        duplicate_iou=args.duplicate_iou,
    )
    result = toontra.process_file(args.input, source_language=args.source_language)
    destination = result.save(
        args.output,
        save_crops=args.save_crops,
        force=args.force,
    )
    print(f"Detected {len(result.bubbles)} bubbles")
    print(f"Saved output to {destination.resolve()}")
    return 0


def _use_gpu(device: str) -> bool:
    if device == "cpu":
        return False
    if device == "cuda":
        return True
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


if __name__ == "__main__":
    raise SystemExit(main())
