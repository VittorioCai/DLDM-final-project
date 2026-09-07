#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from shortcut_learning.analysis import write_analysis_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics and draw breaking curves.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in write_analysis_outputs(args.output_root, args.destination):
        print(path.resolve())


if __name__ == "__main__":
    main()
