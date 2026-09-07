#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from shortcut_learning.experiment import evaluate_saved_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute metrics for a saved run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_saved_run(args.run_dir, device_requested=args.device)
    print(
        f"bias_conflict_accuracy={metrics['bias_conflict_accuracy']:.4f} "
        f"flip_rate={metrics['pairwise_flip_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
