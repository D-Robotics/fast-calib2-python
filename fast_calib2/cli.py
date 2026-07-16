"""Command-line interface for FAST-Calib2 Python."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config
from .pipeline import main as run_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-calib2",
        description="LiDAR-camera extrinsic calibration with an annular target",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate-config", help="validate a calibration YAML file")
    validate.add_argument("--config", required=True, type=Path)
    calibrate = subcommands.add_parser("calibrate", help="run calibration for scene_* directories")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--scenes", required=True)
    calibrate.add_argument("--out", default="output")
    calibrate.add_argument("--raw", default=None)
    calibrate.add_argument("--nf", type=int, default=30)
    calibrate.add_argument("--multi", nargs=3, default=None)
    calibrate.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        try:
            load_config(args.config)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(f"Configuration is valid: {args.config}")
        return 0

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    pipeline_args = ["--config", args.config, "--scenes", args.scenes, "--out", args.out, "--nf", str(args.nf)]
    if args.raw:
        pipeline_args.extend(["--raw", args.raw])
    if args.multi:
        pipeline_args.extend(["--multi", *args.multi])
    results = run_calibration(pipeline_args) or []
    output_directory = Path(args.out)
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "calibration_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "successful_scenes": [
                    {"name": result["name"], "rmse_m": result["rmse"]}
                    for result in results
                    if result is not None
                ],
                "failed_scene_count": sum(result is None for result in results),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    logging.getLogger(__name__).info("Wrote run summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())